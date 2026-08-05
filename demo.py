"""Offline demo: push STT events through the real agent and print what the LLM gets.

Runs the actual stt_node and on_user_turn_completed from agent.py against synthetic
Deepgram events. No keys, no microphone, no LiveKit server. Everything it prints is
produced by the shipped code.

    python demo.py

For the real audio loop see the README. For the same paths driven by a real AgentSession,
see tests/test_integration.py.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Iterator, Sequence

from livekit import rtc
from livekit.agents import Agent, StopResponse, stt
from livekit.agents.llm import ChatContext, ChatMessage

from agent import ConfidenceAwareAgent
from confidence import Settings, build_instructions

Final = tuple[str, float]


class _Session:
    """Stands in for AgentSession, just to capture what would have been said."""

    def __init__(self, spoken: list[str]) -> None:
        self._spoken = spoken

    def say(self, text: str, **_: object) -> None:
        self._spoken.append(text)


class _Agent(ConfidenceAwareAgent):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.spoken: list[str] = []

    @property
    def session(self) -> _Session:  # type: ignore[override]
        return _Session(self.spoken)


@contextlib.contextmanager
def _canned_stt(events: Sequence[stt.SpeechEvent]) -> Iterator[None]:
    original = Agent.default.stt_node

    async def replay(*_args: object, **_kwargs: object) -> AsyncIterator[stt.SpeechEvent]:
        for event in events:
            yield event

    Agent.default.stt_node = replay  # type: ignore[assignment]
    try:
        yield
    finally:
        Agent.default.stt_node = original  # type: ignore[assignment]


def _final(text: str, confidence: float) -> stt.SpeechEvent:
    return stt.SpeechEvent(
        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
        alternatives=[stt.SpeechData(language="en-US", text=text, confidence=confidence)],
    )


async def _no_audio() -> AsyncIterator[rtc.AudioFrame]:
    return
    yield  # pragma: no cover - only here to make this an async generator


async def scenario(title: str, finals: Sequence[Final], **env: str) -> None:
    agent = _Agent(Settings.from_env(env))
    transcript = " ".join(text for text, _ in finals)

    print(f"\n=== {title} ===")
    for text, confidence in finals:
        print(f"  Deepgram final : {text!r}  confidence={confidence:.2f}")

    with _canned_stt([_final(text, confidence) for text, confidence in finals]):
        async for _ in agent.stt_node(_no_audio(), None):  # type: ignore[arg-type]
            pass

    turn_ctx = ChatContext.empty()
    message = ChatMessage(type="message", role="user", content=[transcript])

    try:
        await agent.on_user_turn_completed(turn_ctx, message)
    except StopResponse:
        print("  -> LLM SKIPPED (StopResponse); agent speaks a fixed clarification:")
        print(f'     "{agent.spoken[0]}"')
        return

    notes = [item.text_content for item in turn_ctx.items if item.role == "system"]
    if not notes:
        print("  -> no confidence available; context left untouched")
        return
    print("  -> injected into the LLM context for this turn only:")
    for line in (notes[0] or "").splitlines():
        print(f"     | {line}")
    print(f"     | (user) {transcript}")


async def main() -> None:
    print("Instructions given to the LLM")
    print("-" * 78)
    for line in build_instructions(Settings.from_env({})).splitlines():
        print(f"  {line}")

    await scenario(
        "High confidence - answer normally",
        [("What time do you close today?", 0.94)],
    )
    await scenario(
        "High confidence - a low-confidence filler must not trigger a repeat",
        [("um", 0.31), ("what time do you close today", 0.96)],
    )
    await scenario(
        "Medium confidence - answer, but confirm",
        [("I need help with my order", 0.71)],
    )
    await scenario(
        "Low confidence - deterministic clarification (default)",
        [("I need help with my order", 0.35)],
    )
    await scenario(
        "Low confidence - LLM phrases the clarification itself",
        [("I need help with my order", 0.35)],
        LOW_CONFIDENCE_STRATEGY="llm",
    )
    await scenario(
        "Low confidence - a long garbled clause does pull the turn down",
        [("yes", 0.97), ("i need help with the thing about my order please", 0.34)],
        LOW_CONFIDENCE_STRATEGY="llm",
    )


if __name__ == "__main__":
    asyncio.run(main())
