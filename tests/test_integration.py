"""End-to-end tests against a real AgentSession.

Everything here is the genuine framework: a real session, real turn detection, real
scheduling, and the agent's hooks invoked by LiveKit rather than by the test. Only the
three providers are swapped for fakes, so no keys and no network are needed.

This is what covers the parts a stubbed session cannot: that `on_user_turn_completed`
actually fires, that a low-confidence turn reaches TTS with the clarification and never
reaches the LLM, and that the metadata note is present in the context the LLM receives.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest
import pytest_asyncio
from livekit.agents import AgentSession
from livekit.agents.testing import fake_job_context

from agent import ConfidenceAwareAgent
from confidence import Settings
from tests.fakes import (
    CollectingAudioOutput,
    Script,
    ScriptedLLM,
    ScriptedSTT,
    ScriptedTTS,
    SilentAudioInput,
)

CLARIFICATION = "Sorry, I may have misunderstood you. Could you please repeat that?"


class Harness:
    def __init__(self, session: AgentSession, stt: ScriptedSTT, llm: ScriptedLLM, tts: ScriptedTTS):
        self.session = session
        self.stt = stt
        self.llm = llm
        self.tts = tts

    async def say(self, script: Script) -> None:
        # TODO: this is a wall-clock guess, not a real wait. It's stable locally but it's
        # the thing most likely to go flaky on a loaded CI runner, and it's why the suite
        # takes ~17s instead of ~3s. Should hook session events (agent_state_changed /
        # conversation_item_added) and wait on those instead.
        await self.stt.feed(script)
        await asyncio.sleep(0.6)

    @property
    def spoken(self) -> str:
        """Everything sent to TTS. The framework splits replies per sentence, so join it."""
        return " ".join(self.tts.spoken)

    @property
    def notes(self) -> list[str]:
        return self.llm.system_notes()


async def _run(**env: str) -> AsyncIterator[Harness]:
    stt, llm, tts = ScriptedSTT(), ScriptedLLM(), ScriptedTTS()
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        # No VAD in the fakes, so end the turn on the STT's own END_OF_SPEECH, promptly.
        turn_handling={"turn_detection": "stt", "endpointing": {"min_delay": 0.05}},
    )
    with fake_job_context():
        session.input.audio = SilentAudioInput()
        session.output.audio = CollectingAudioOutput()
        await session.start(ConfidenceAwareAgent(Settings.from_env(env)))
        try:
            yield Harness(session, stt, llm, tts)
        finally:
            await session.aclose()


@pytest_asyncio.fixture
async def agent_session() -> AsyncIterator[Harness]:
    async for harness in _run():
        yield harness


@pytest_asyncio.fixture
async def llm_strategy_session() -> AsyncIterator[Harness]:
    async for harness in _run(LOW_CONFIDENCE_STRATEGY="llm"):
        yield harness


@pytest.mark.asyncio
async def test_a_clear_turn_reaches_the_llm_with_its_confidence(agent_session: Harness) -> None:
    await agent_session.say([("what time do you close today", 0.94)])

    assert agent_session.llm.seen_contexts, "the LLM should have been called"
    note = agent_session.notes[-1]
    assert "band=high" in note
    assert "score=0.94" in note
    assert "Sure, happy to help." in agent_session.spoken


@pytest.mark.asyncio
async def test_a_garbled_turn_never_reaches_the_llm(agent_session: Harness) -> None:
    """The behaviour the whole task is about, driven by the real session."""
    await agent_session.say([("i need help with my order", 0.35)])

    assert agent_session.llm.seen_contexts == [], "the LLM must not be called on a low turn"
    assert agent_session.spoken == CLARIFICATION


@pytest.mark.asyncio
async def test_the_garbled_transcript_stays_out_of_history(agent_session: Harness) -> None:
    await agent_session.say([("i need help with my order", 0.35)])

    history = " ".join(
        getattr(item, "text_content", "") or "" for item in agent_session.session.history.items
    )
    assert "i need help with my order" not in history


@pytest.mark.asyncio
async def test_the_same_words_get_opposite_treatment(agent_session: Harness) -> None:
    """Identical text, different confidence, different behaviour end to end."""
    await agent_session.say([("i need help with my order", 0.35)])
    low_calls = len(agent_session.llm.seen_contexts)

    await agent_session.say([("i need help with my order", 0.92)])

    assert low_calls == 0
    assert len(agent_session.llm.seen_contexts) == 1
    assert "band=high" in agent_session.notes[-1]


@pytest.mark.asyncio
async def test_a_filler_word_does_not_cost_the_user_a_repeat(agent_session: Harness) -> None:
    await agent_session.say([("um", 0.31), ("what time do you close today", 0.96)])

    assert "band=high" in agent_session.notes[-1]
    assert "Sorry" not in agent_session.spoken


@pytest.mark.asyncio
async def test_confidence_is_scoped_to_its_own_turn(agent_session: Harness) -> None:
    """A low turn must not leave anything behind that taints the next one."""
    await agent_session.say([("mumble mumble", 0.30)])
    await agent_session.say([("what time do you close today", 0.95)])

    assert "band=high" in agent_session.notes[-1]
    assert len(agent_session.notes) == 1, "only the second turn should have reached the LLM"


@pytest.mark.asyncio
async def test_the_note_does_not_accumulate_across_turns(agent_session: Harness) -> None:
    await agent_session.say([("first question here", 0.95)])
    await agent_session.say([("second question here", 0.95)])

    latest = agent_session.llm.seen_contexts[-1]
    notes = [
        item
        for item in latest.items
        if getattr(item, "role", None) == "system"
        and (item.text_content or "").startswith("[stt-confidence]")
    ]
    assert len(notes) == 1, "each generation should carry exactly one note, for the current turn"


@pytest.mark.asyncio
async def test_llm_strategy_lets_the_model_answer_a_low_turn(
    llm_strategy_session: Harness,
) -> None:
    await llm_strategy_session.say([("i need help with my order", 0.35)])

    assert llm_strategy_session.llm.seen_contexts, "the LLM should be reached under this strategy"
    assert "band=low" in llm_strategy_session.notes[-1]


@pytest.mark.asyncio
async def test_the_note_sits_directly_before_the_user_message(agent_session: Harness) -> None:
    await agent_session.say([("what time do you close today", 0.94)])

    roles = [getattr(item, "role", None) for item in agent_session.llm.seen_contexts[-1].items]
    assert roles[-2:] == ["system", "user"]


@pytest.mark.asyncio
async def test_several_turns_in_a_row_stay_correct(agent_session: Harness) -> None:
    script: Sequence[tuple[Script, str]] = [
        ([("first clear question", 0.95)], "high"),
        ([("garbled thing", 0.20)], "skipped"),
        ([("another clear question", 0.88)], "high"),
        ([("half heard request", 0.65)], "medium"),
    ]
    for turn, _ in script:
        await agent_session.say(turn)

    assert [n.split("band=")[1].split()[0] for n in agent_session.notes] == [
        "high",
        "high",
        "medium",
    ]
    assert agent_session.spoken.count(CLARIFICATION) == 1
