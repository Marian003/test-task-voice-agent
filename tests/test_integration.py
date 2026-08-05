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
import contextlib
from collections.abc import AsyncIterator

import pytest
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


@contextlib.asynccontextmanager
async def running_agent(**env: str) -> AsyncIterator[Harness]:
    """Start a real AgentSession on fake providers.

    Deliberately a context manager used inside the test rather than a fixture.
    fake_job_context() sets a ContextVar, and pytest-asyncio finalises an async-generator
    fixture in a different Context from the one that set it up, so resetting the token
    there blows up with "was created in a different Context". Entering and leaving inside
    the test body keeps both halves in the same context.
    """
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


def roles_of(ctx: object) -> list[str | None]:
    return [getattr(item, "role", None) for item in ctx.items]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_a_clear_turn_reaches_the_llm_with_its_confidence() -> None:
    async with running_agent() as agent:
        await agent.say([("what time do you close today", 0.94)])

        assert agent.llm.seen_contexts, "the LLM should have been called"
        note = agent.notes[-1]
        assert "band=high" in note
        assert "score=0.94" in note
        assert "Sure, happy to help." in agent.spoken


@pytest.mark.asyncio
async def test_a_garbled_turn_never_reaches_the_llm() -> None:
    """The behaviour the whole task is about, driven by the real session."""
    async with running_agent() as agent:
        await agent.say([("i need help with my order", 0.35)])

        assert agent.llm.seen_contexts == [], "the LLM must not be called on a low turn"
        assert agent.spoken == CLARIFICATION


@pytest.mark.asyncio
async def test_the_garbled_transcript_stays_out_of_history() -> None:
    async with running_agent() as agent:
        await agent.say([("i need help with my order", 0.35)])

        history = " ".join(
            getattr(item, "text_content", "") or "" for item in agent.session.history.items
        )
        assert "i need help with my order" not in history


@pytest.mark.asyncio
async def test_the_same_words_get_opposite_treatment() -> None:
    """Identical text, different confidence, different behaviour end to end."""
    async with running_agent() as agent:
        await agent.say([("i need help with my order", 0.35)])
        low_calls = len(agent.llm.seen_contexts)

        await agent.say([("i need help with my order", 0.92)])

        assert low_calls == 0
        assert len(agent.llm.seen_contexts) == 1
        assert "band=high" in agent.notes[-1]


@pytest.mark.asyncio
async def test_a_filler_word_does_not_cost_the_user_a_repeat() -> None:
    async with running_agent() as agent:
        await agent.say([("um", 0.31), ("what time do you close today", 0.96)])

        assert "band=high" in agent.notes[-1]
        assert "Sorry" not in agent.spoken


@pytest.mark.asyncio
async def test_confidence_is_scoped_to_its_own_turn() -> None:
    """A low turn must not leave anything behind that taints the next one."""
    async with running_agent() as agent:
        await agent.say([("mumble mumble", 0.30)])
        await agent.say([("what time do you close today", 0.95)])

        assert "band=high" in agent.notes[-1]
        assert len(agent.notes) == 1, "only the second turn should have reached the LLM"


@pytest.mark.asyncio
async def test_the_note_does_not_accumulate_across_turns() -> None:
    async with running_agent() as agent:
        await agent.say([("first question here", 0.95)])
        await agent.say([("second question here", 0.95)])

        latest = agent.llm.seen_contexts[-1]
        notes = [
            item
            for item in latest.items
            if getattr(item, "role", None) == "system"
            and (item.text_content or "").startswith("[stt-confidence]")
        ]
        assert len(notes) == 1, "each generation carries one note, for the current turn"


@pytest.mark.asyncio
async def test_llm_strategy_lets_the_model_answer_a_low_turn() -> None:
    async with running_agent(LOW_CONFIDENCE_STRATEGY="llm") as agent:
        await agent.say([("i need help with my order", 0.35)])

        assert agent.llm.seen_contexts, "the LLM should be reached under this strategy"
        assert "band=low" in agent.notes[-1]


@pytest.mark.asyncio
async def test_the_note_sits_directly_before_the_user_message() -> None:
    """The note says "the next user message", so it had better come before one."""
    async with running_agent() as agent:
        await agent.say([("what time do you close today", 0.94)])

        items = agent.llm.seen_contexts[-1].items
        note_at = next(
            i
            for i, item in enumerate(items)
            if getattr(item, "role", None) == "system"
            and (item.text_content or "").startswith("[stt-confidence]")
        )
        assert roles_of(agent.llm.seen_contexts[-1])[note_at + 1] == "user"
        assert "what time do you close today" in (items[note_at + 1].text_content or "")


@pytest.mark.asyncio
async def test_several_turns_in_a_row_stay_correct() -> None:
    async with running_agent() as agent:
        for turn in (
            [("first clear question", 0.95)],
            [("garbled thing", 0.20)],
            [("another clear question", 0.88)],
            [("half heard request", 0.65)],
        ):
            await agent.say(turn)

        bands = [n.split("band=")[1].split()[0] for n in agent.notes]
        assert bands == ["high", "high", "medium"]
        assert agent.spoken.count(CLARIFICATION) == 1
