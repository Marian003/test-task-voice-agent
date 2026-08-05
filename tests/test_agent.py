"""Tests for the agent hooks, using real LiveKit context objects and a stub session.

These exercise the two override points end-to-end in-process: what `stt_node` records,
and what `on_user_turn_completed` does with it. No API keys and no media are involved.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from livekit.agents import Agent, StopResponse, stt
from livekit.agents.llm import ChatContext, ChatMessage

from agent import ConfidenceAwareAgent
from confidence import Settings, Strategy


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


def make_agent(**env: str) -> ConfidenceAwareAgent:
    return ConfidenceAwareAgent(Settings.from_env(env))


def user_message(text: str, *, native_confidence: float | None = None) -> ChatMessage:
    return ChatMessage(
        type="message",
        role="user",
        content=[text],
        transcript_confidence=native_confidence,
    )


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


def notes(ctx: ChatContext) -> list[str]:
    return [item.text_content or "" for item in ctx.items if item.role == "system"]


# --- stt_node ------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stt_node_records_finals_and_forwards_every_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = [
        interim("what time", 0.55),
        final("what time do you", 0.91),
        final("close today", 0.87),
        # A bare string: the default node is typed to allow these, and they must not crash.
        "trailing text event",
    ]

    async def fake_default(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        for event in events:
            yield event

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)

    agent = make_agent()
    forwarded = [event async for event in agent.stt_node(iter(()), None)]  # type: ignore[arg-type]

    assert forwarded == events, "stt_node must be a pass-through, not a filter"
    # Only the two finals were recorded; the interim and the string were ignored.
    result = agent._buffer.consume("what time do you close today")
    assert result is not None
    assert result.samples == 2


@pytest.mark.asyncio
async def test_stt_node_ignores_empty_finals(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_default(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any, None]:
        yield final("   ", 0.1)

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)

    agent = make_agent()
    _ = [event async for event in agent.stt_node(iter(()), None)]  # type: ignore[arg-type]
    assert len(agent._buffer) == 0


# --- on_user_turn_completed: annotation ----------------------------------------------
@pytest.mark.asyncio
async def test_high_confidence_annotates_the_turn(session: StubSession) -> None:
    agent = make_agent()
    agent._buffer.add("what time do you close today", 0.94)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("what time do you close today"))

    assert session.said == []
    assert len(notes(ctx)) == 1
    assert "band=high" in notes(ctx)[0]
    assert "score=0.94" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_medium_confidence_annotates_the_turn(session: StubSession) -> None:
    agent = make_agent()
    agent._buffer.add("what time do you close today", 0.7)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("what time do you close today"))

    assert session.said == []
    assert "band=medium" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_the_note_lands_immediately_before_the_user_message(session: StubSession) -> None:
    """The framework appends `new_message` after this hook, so the note must be last."""
    agent = make_agent()
    agent._buffer.add("hello", 0.9)
    ctx = ChatContext.empty()
    ctx.add_message(role="user", content="an earlier question")
    ctx.add_message(role="assistant", content="an earlier answer")

    await agent.on_user_turn_completed(ctx, user_message("hello"))

    assert [item.role for item in ctx.items] == ["user", "assistant", "system"]


@pytest.mark.asyncio
async def test_annotation_never_leaks_into_persisted_history(session: StubSession) -> None:
    """`turn_ctx` is a per-turn copy; the agent's own history must stay clean."""
    agent = make_agent()
    agent._buffer.add("hello", 0.9)
    before = len(agent.chat_ctx.items)

    await agent.on_user_turn_completed(ChatContext.empty(), user_message("hello"))

    assert len(agent.chat_ctx.items) == before


# --- on_user_turn_completed: low band -------------------------------------------------
@pytest.mark.asyncio
async def test_low_confidence_short_circuits_the_turn_by_default(session: StubSession) -> None:
    agent = make_agent()
    agent._buffer.add("i need help with my order", 0.35)
    ctx = ChatContext.empty()

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(ctx, user_message("i need help with my order"))

    assert session.said == ["Sorry, I may have misunderstood you. Could you please repeat that?"]
    assert notes(ctx) == [], "the LLM must not be reached at all on a low-confidence turn"


@pytest.mark.asyncio
async def test_custom_clarification_is_spoken(session: StubSession) -> None:
    agent = make_agent(CLARIFICATION_PROMPT="Come again?")
    agent._buffer.add("mumble", 0.2)

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(ChatContext.empty(), user_message("mumble"))

    assert session.said == ["Come again?"]


@pytest.mark.asyncio
async def test_llm_strategy_annotates_instead_of_short_circuiting(session: StubSession) -> None:
    agent = make_agent(LOW_CONFIDENCE_STRATEGY=Strategy.LLM.value)
    agent._buffer.add("i need help with my order", 0.35)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("i need help with my order"))

    assert session.said == []
    assert "band=low" in notes(ctx)[0]


# --- source resolution ----------------------------------------------------------------
@pytest.mark.asyncio
async def test_falls_back_to_livekits_own_measurement(session: StubSession) -> None:
    """Nothing captured in `stt_node` -> use `ChatMessage.transcript_confidence`."""
    agent = make_agent()
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("hello", native_confidence=0.93))

    assert "source=session" in notes(ctx)[0]
    assert "score=0.93" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_captured_confidence_wins_over_the_fallback(session: StubSession) -> None:
    agent = make_agent()
    agent._buffer.add("hello", 0.62)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("hello", native_confidence=0.99))

    assert "source=stt" in notes(ctx)[0]
    assert "score=0.62" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_forced_confidence_overrides_everything_and_is_labelled(
    session: StubSession,
) -> None:
    agent = make_agent(FORCE_CONFIDENCE="0.95", LOW_CONFIDENCE_STRATEGY="llm")
    agent._buffer.add("hello", 0.10)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("hello", native_confidence=0.10))

    assert "source=forced" in notes(ctx)[0], "a simulated run must be identifiable in the log"
    assert "band=high" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_turn_without_any_confidence_is_left_untouched(session: StubSession) -> None:
    """A typed/text-only turn: annotate nothing rather than invent a score."""
    agent = make_agent()
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("typed question"))

    assert notes(ctx) == []
    assert session.said == []


@pytest.mark.asyncio
async def test_stale_samples_do_not_poison_the_following_turn(session: StubSession) -> None:
    """Regression: a discarded turn's finals must not force a repeat on the next one."""
    agent = make_agent()
    agent._buffer.add("garbled leftovers", 0.15)  # turn whose hook never ran
    agent._buffer.add("what time do you close", 0.95)
    ctx = ChatContext.empty()

    await agent.on_user_turn_completed(ctx, user_message("what time do you close"))

    assert session.said == []
    assert "band=high" in notes(ctx)[0]


@pytest.mark.asyncio
async def test_buffer_is_drained_even_when_the_turn_is_stopped(session: StubSession) -> None:
    agent = make_agent()
    agent._buffer.add("mumble", 0.2)

    with pytest.raises(StopResponse):
        await agent.on_user_turn_completed(ChatContext.empty(), user_message("mumble"))

    assert len(agent._buffer) == 0


# --- session summary --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_exit_summary_counts_low_turns(
    session: StubSession, caplog: pytest.LogCaptureFixture
) -> None:
    agent = make_agent(LOW_CONFIDENCE_STRATEGY="llm")
    for text, confidence in [("aaa", 0.95), ("bbb", 0.40), ("ccc", 0.85)]:
        agent._buffer.add(text, confidence)
        await agent.on_user_turn_completed(ChatContext.empty(), user_message(text))

    with caplog.at_level("INFO", logger="confidence-agent"):
        await agent.on_exit()

    summary = next(r for r in caplog.records if r.message == "confidence summary")
    assert summary.turns == 3
    assert summary.low_band_turns == 1


@pytest.mark.asyncio
async def test_exit_summary_is_silent_when_nothing_was_measured(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("INFO", logger="confidence-agent"):
        await make_agent().on_exit()
    assert caplog.records == []
