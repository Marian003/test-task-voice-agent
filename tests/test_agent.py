"""Tests for the agent, using real LiveKit context objects and a stub session.

Confidence always enters through the real `stt_node` override rather than by seeding the
buffer, so these exercise the actual capture -> aggregate -> act path in one piece. No API
keys and no media are involved.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from typing import Any

import pytest
from livekit.agents import Agent, AgentSession, StopResponse, stt
from livekit.agents.llm import ChatContext, ChatMessage

import agent as agent_module
from agent import ConfidenceAwareAgent, build_session, entrypoint, main, prewarm
from confidence import ConfigError, Settings, Strategy

Final = tuple[str, float]


# --- fixtures & helpers -------------------------------------------------------------
class StubSession:
    """Captures what the agent would have spoken."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str, **_: Any) -> None:
        self.said.append(text)


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> StubSession:
    stub = StubSession()
    monkeypatch.setattr(Agent, "session", property(lambda _self: stub))
    return stub


@pytest.fixture
def provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake credentials. Plugin constructors validate presence but make no network calls."""
    for name in ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "ELEVEN_API_KEY"):
        monkeypatch.setenv(name, "test-key")


def make_agent(**env: str) -> ConfidenceAwareAgent:
    return ConfidenceAwareAgent(Settings.from_env(env))


def final(text: str, confidence: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en-US", text=text, confidence=confidence)],
    )


def interim(text: str, confidence: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en-US", text=text, confidence=confidence)],
    )


async def drain(
    target: ConfidenceAwareAgent,
    events: Sequence[stt.SpeechEvent | str],
    monkeypatch: pytest.MonkeyPatch,
) -> list[stt.SpeechEvent | str]:
    """Push `events` through the agent's real `stt_node` and return what it forwarded."""

    async def replay(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        for event in events:
            yield event

    monkeypatch.setattr(Agent.default, "stt_node", replay)
    return [event async for event in target.stt_node(iter(()), None)]  # type: ignore[arg-type]


async def speak(
    target: ConfidenceAwareAgent,
    finals: Sequence[Final],
    monkeypatch: pytest.MonkeyPatch,
    *,
    ctx: ChatContext | None = None,
    native_confidence: float | None = None,
) -> ChatContext:
    """Simulate one full user turn: STT events, then the turn-completion hook."""
    await drain(target, [final(text, confidence) for text, confidence in finals], monkeypatch)
    turn_ctx = ChatContext.empty() if ctx is None else ctx
    message = ChatMessage(
        type="message",
        role="user",
        content=[" ".join(text for text, _ in finals)],
        transcript_confidence=native_confidence,
    )
    await target.on_user_turn_completed(turn_ctx, message)
    return turn_ctx


def notes(ctx: ChatContext) -> list[str]:
    return [item.text_content or "" for item in ctx.items if item.role == "system"]


# --- stt_node ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stt_node_records_finals_and_forwards_every_event(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[stt.SpeechEvent | str] = [
        interim("what time", 0.55),
        final("what time do you", 0.91),
        final("close today", 0.87),
        # A bare string: the default node is typed to allow these, and they must not crash.
        "trailing text event",
    ]
    target = make_agent()

    forwarded = await drain(target, events, monkeypatch)

    assert forwarded == events, "stt_node must be a pass-through, not a filter"
    # Only the two finals were recorded; the interim and the string were ignored.
    result = target._buffer.consume("what time do you close today")
    assert result is not None
    assert result.samples == 2


@pytest.mark.asyncio
async def test_stt_node_ignores_empty_finals(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_agent()
    await drain(target, [final("   ", 0.1)], monkeypatch)
    assert len(target._buffer) == 0


# --- band behaviour -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_high_confidence_annotates_the_turn(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await speak(make_agent(), [("what time do you close today", 0.94)], monkeypatch)

    assert session.said == []
    assert len(notes(ctx)) == 1
    assert "band=high" in notes(ctx)[0]
    assert "score=0.94" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_medium_confidence_annotates_the_turn(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await speak(make_agent(), [("what time do you close today", 0.7)], monkeypatch)

    assert session.said == []
    assert "band=medium" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_low_confidence_short_circuits_the_turn_by_default(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_agent()
    ctx = ChatContext.empty()

    with pytest.raises(StopResponse):
        await speak(target, [("i need help with my order", 0.35)], monkeypatch, ctx=ctx)

    assert session.said == ["Sorry, I may have misunderstood you. Could you please repeat that?"]
    assert notes(ctx) == [], "the LLM must not be reached at all on a low-confidence turn"


@pytest.mark.asyncio
async def test_custom_clarification_is_spoken(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_agent(CLARIFICATION_PROMPT="Come again?")

    with pytest.raises(StopResponse):
        await speak(target, [("mumble", 0.2)], monkeypatch)

    assert session.said == ["Come again?"]


@pytest.mark.asyncio
async def test_llm_strategy_annotates_instead_of_short_circuiting(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await speak(
        make_agent(LOW_CONFIDENCE_STRATEGY=Strategy.LLM.value),
        [("i need help with my order", 0.35)],
        monkeypatch,
    )

    assert session.said == []
    assert "band=low" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_a_filler_word_does_not_trigger_a_repeat(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end version of the weighting rule, driven through the real STT path."""
    ctx = await speak(
        make_agent(), [("um", 0.31), ("what time do you close today", 0.96)], monkeypatch
    )

    assert session.said == []
    assert "band=high" in notes(ctx)[0]
    assert "weighted over 2 final transcripts" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_a_long_garbled_clause_still_forces_a_repeat(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_agent()

    with pytest.raises(StopResponse):
        await speak(
            target,
            [("yes", 0.97), ("i need help with the thing about my order please", 0.34)],
            monkeypatch,
        )

    assert session.said == ["Sorry, I may have misunderstood you. Could you please repeat that?"]


# --- context handling -------------------------------------------------------------------
@pytest.mark.asyncio
async def test_the_note_lands_immediately_before_the_user_message(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LiveKit appends `new_message` after this hook returns, so the note must come last."""
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="an earlier question")
    ctx.add_message(role="assistant", content="an earlier answer")

    await speak(make_agent(), [("hello", 0.9)], monkeypatch, ctx=ctx)

    assert [item.role for item in ctx.items] == ["user", "assistant", "system"]


@pytest.mark.asyncio
async def test_annotation_never_leaks_into_persisted_history(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`turn_ctx` is a per-turn copy; the agent's own history must stay clean."""
    target = make_agent()
    before = len(target.chat_ctx.items)

    await speak(target, [("hello", 0.9)], monkeypatch)

    assert len(target.chat_ctx.items) == before


# --- source resolution ------------------------------------------------------------------
@pytest.mark.asyncio
async def test_falls_back_to_livekits_own_measurement(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing captured in `stt_node` -> use `ChatMessage.transcript_confidence`."""
    target = make_agent()
    message = ChatMessage(
        type="message", role="user", content=["hello"], transcript_confidence=0.93
    )
    ctx = ChatContext.empty()

    await target.on_user_turn_completed(ctx, message)

    assert "source=session" in notes(ctx)[0]
    assert "score=0.93" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_captured_confidence_wins_over_the_fallback(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await speak(make_agent(), [("hello", 0.62)], monkeypatch, native_confidence=0.99)

    assert "source=stt" in notes(ctx)[0]
    assert "score=0.62" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_unusable_buffer_defers_to_the_fallback(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finals that do not reconstruct the turn must not be scored as if they did."""
    target = make_agent()
    await drain(target, [final("leftovers from a discarded turn", 0.12)], monkeypatch)
    ctx = ChatContext.empty()

    await target.on_user_turn_completed(
        ctx,
        ChatMessage(
            type="message",
            role="user",
            content=["what time do you close"],
            transcript_confidence=0.9,
        ),
    )

    assert "source=session" in notes(ctx)[0]
    assert "score=0.90" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_forced_confidence_overrides_everything_and_is_labelled(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = await speak(
        make_agent(FORCE_CONFIDENCE="0.95", LOW_CONFIDENCE_STRATEGY="llm"),
        [("hello", 0.10)],
        monkeypatch,
        native_confidence=0.10,
    )

    assert "source=forced" in notes(ctx)[0], "a simulated run must be identifiable"
    assert "band=high" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_turn_without_any_confidence_is_left_untouched(session: StubSession) -> None:
    """A typed/text-only turn: annotate nothing rather than invent a score."""
    ctx = ChatContext.empty()

    await make_agent().on_user_turn_completed(
        ctx, ChatMessage(type="message", role="user", content=["typed question"])
    )

    assert notes(ctx) == []
    assert session.said == []


@pytest.mark.asyncio
async def test_stale_samples_do_not_poison_the_following_turn(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a discarded turn's finals must not force a repeat on the next one."""
    target = make_agent()
    await drain(target, [final("garbled leftovers", 0.15)], monkeypatch)  # hook never ran

    ctx = await speak(target, [("what time do you close", 0.95)], monkeypatch)

    assert session.said == []
    assert "band=high" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_buffer_is_drained_even_when_the_turn_is_stopped(
    session: StubSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = make_agent()

    with pytest.raises(StopResponse):
        await speak(target, [("mumble", 0.2)], monkeypatch)

    assert len(target._buffer) == 0


# --- session summary --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exit_summary_counts_low_turns(
    session: StubSession, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    target = make_agent(LOW_CONFIDENCE_STRATEGY="llm")
    for text, confidence in [("aaa", 0.95), ("bbb", 0.40), ("ccc", 0.85)]:
        await speak(target, [(text, confidence)], monkeypatch)

    with caplog.at_level("INFO", logger="confidence-agent"):
        await target.on_exit()

    summary = next(r for r in caplog.records if r.getMessage() == "confidence summary")
    assert summary.turns == 3
    assert summary.low_band_turns == 1
    assert summary.lowest == pytest.approx(0.40)


@pytest.mark.asyncio
async def test_exit_summary_is_silent_when_nothing_was_measured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="confidence-agent"):
        await make_agent().on_exit()
    assert caplog.records == []


# --- wiring -------------------------------------------------------------------------------
@contextlib.asynccontextmanager
async def built(settings: Settings) -> AsyncIterator[AgentSession]:
    """Build a real session and close it, so plugin background tasks do not outlive the test."""
    from livekit.plugins import silero

    session = build_session(settings, silero.VAD.load())
    try:
        yield session
    finally:
        await session.llm.aclose()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_build_session_wires_the_required_pipeline(provider_keys: None) -> None:
    from livekit.plugins import deepgram, elevenlabs, openai, silero

    async with built(Settings.from_env({})) as session:
        assert isinstance(session.stt, deepgram.STT)
        assert isinstance(session.llm, openai.LLM)
        assert isinstance(session.tts, elevenlabs.TTS)
        assert isinstance(session.vad, silero.VAD)
        assert session.stt._opts.model == "nova-3"
        assert session.stt._opts.smart_format is False, "smart_format breaks final matching"


@pytest.mark.asyncio
async def test_build_session_warns_when_confidence_is_simulated(
    provider_keys: None, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="confidence-agent"):
        async with built(Settings.from_env({"FORCE_CONFIDENCE": "0.35"})):
            pass

    assert any("SIMULATED" in r.getMessage() for r in caplog.records)


def test_prewarm_loads_the_vad_once() -> None:
    class StubProcess:
        def __init__(self) -> None:
            self.userdata: dict[str, Any] = {}

    process = StubProcess()
    prewarm(process)  # type: ignore[arg-type]
    assert process.userdata["vad"] is not None


def test_main_exits_cleanly_on_bad_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must stop the parent process, not spawn a retry loop of failing workers."""
    monkeypatch.setenv("FORCE_CONFIDENCE", "high")
    monkeypatch.setattr(agent_module.cli, "run_app", lambda *_a, **_k: pytest.fail("spawned"))

    with pytest.raises(SystemExit, match="configuration error"):
        main()


def test_main_starts_the_worker_when_configuration_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FORCE_CONFIDENCE", raising=False)
    started: list[Any] = []
    monkeypatch.setattr(agent_module.cli, "run_app", lambda opts: started.append(opts))

    main()

    assert started[0].entrypoint_fnc is entrypoint
    assert started[0].prewarm_fnc is prewarm


@pytest.mark.asyncio
async def test_entrypoint_connects_starts_and_greets(
    provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    class StubCtx:
        room = object()
        proc = type("P", (), {"userdata": {}})()

        async def connect(self) -> None:
            calls.append("connect")

    class StubAgentSession:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def start(self, *, agent: Any, room: Any) -> None:
            calls.append("start")
            assert isinstance(agent, ConfidenceAwareAgent)
            assert room is StubCtx.room

        async def generate_reply(self, *, instructions: str) -> None:
            calls.append("greet")
            assert "Greet" in instructions

    monkeypatch.setattr(agent_module, "AgentSession", StubAgentSession)

    await entrypoint(StubCtx())  # type: ignore[arg-type]

    assert calls == ["connect", "start", "greet"]


@pytest.mark.asyncio
async def test_entrypoint_rejects_bad_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOW_CONFIDENCE_STRATEGY", "maybe")
    with pytest.raises(ConfigError):
        await entrypoint(object())  # type: ignore[arg-type]


def test_agent_session_import_is_the_real_one() -> None:
    """Guards the monkeypatch above from silently testing nothing."""
    assert agent_module.AgentSession is AgentSession
