"""Voice agent that reads Deepgram's STT confidence and changes behaviour on it.

    User speech -> Deepgram STT -> LLM (+ confidence) -> ElevenLabs TTS -> User
                        |                  ^
                        +---- confidence --+

Two hooks carry the metadata across the pipeline:

  1. `stt_node`                — the only place the raw `SpeechEvent` (and therefore
                                 `alternatives[i].confidence`) is visible. We observe the
                                 final transcripts as they stream past and change nothing.
  2. `on_user_turn_completed`  — fires after the user's turn ends and before inference.
                                 We aggregate what we saw, then either inject the score
                                 into the context for this turn only, or short-circuit
                                 the turn with a fixed clarification.

All decisions live in `confidence.py`; this module is wiring and logging.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterable

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    ModelSettings,
    StopResponse,
    WorkerOptions,
    cli,
    stt,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import deepgram, elevenlabs, openai, silero

from confidence import (
    Aggregate,
    Band,
    ConfigError,
    Settings,
    Source,
    Strategy,
    TurnBuffer,
    build_annotation,
    build_instructions,
)

load_dotenv()

# No `basicConfig` here: `cli.run_app` installs LiveKit's own handler, and configuring
# the root logger ourselves would duplicate every record.
logger = logging.getLogger("confidence-agent")


class ConfidenceAwareAgent(Agent):
    """An agent that answers, hedges, or asks for a repeat based on STT confidence."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(instructions=build_instructions(settings))
        self._settings = settings
        self._buffer = TurnBuffer()
        self._scores: list[float] = []
        self._low_turns = 0

    # -- 1) Observe confidence at the source ---------------------------------------
    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent | str, None]:
        """Pass STT events through untouched, recording the confidence of each final."""
        async for event in Agent.default.stt_node(self, audio, model_settings):
            # The node may also yield plain strings (transcript-only STT paths).
            if (
                isinstance(event, stt.SpeechEvent)
                and event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
                and event.alternatives
            ):
                alternative = event.alternatives[0]
                if self._buffer.add(alternative.text, alternative.confidence):
                    logger.debug(
                        "stt final transcript",
                        extra={
                            "transcript": alternative.text,
                            "confidence": round(alternative.confidence, 4),
                        },
                    )
            yield event

    # -- 2) Act on it, just before inference ----------------------------------------
    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Resolve this turn's confidence and either annotate the context or stop the turn."""
        transcript = new_message.text_content or ""
        detail = self._buffer.consume(transcript)
        score, source = self._resolve(detail, new_message)

        if score is None:
            # A text-only turn, or STT reported nothing. Say nothing rather than guess:
            # the instructions treat an un-annotated message as an ordinary one.
            logger.debug("no confidence available for this turn", extra={"transcript": transcript})
            return

        band = self._settings.thresholds.band(score)
        self._scores.append(score)
        if band is Band.LOW:
            self._low_turns += 1

        logger.info(
            "stt confidence resolved",
            extra={
                "confidence": round(score, 4),
                "band": str(band),
                "confidence_source": source.value,
                "finals": detail.samples if detail else 0,
            },
        )

        if band is Band.LOW and self._settings.strategy is Strategy.DETERMINISTIC:
            # Guaranteed behaviour: speak the clarification and skip inference entirely,
            # so a garbled turn can never be answered even if the model ignores a prompt.
            # `StopResponse` also keeps the unreliable transcript out of the history.
            self.session.say(self._settings.clarification)
            raise StopResponse

        # `turn_ctx` is a per-turn copy, so this note reaches the LLM for this generation
        # only and never accumulates in the persisted conversation history.
        turn_ctx.add_message(role="system", content=build_annotation(band, score, source, detail))

    async def on_exit(self) -> None:
        """Log a session-level confidence summary — a cheap check for a bad mic or model."""
        if not self._scores:
            return
        logger.info(
            "confidence summary",
            extra={
                "turns": len(self._scores),
                "mean": round(sum(self._scores) / len(self._scores), 4),
                "lowest": round(min(self._scores), 4),
                "low_band_turns": self._low_turns,
            },
        )

    # -- helpers ---------------------------------------------------------------------
    def _resolve(
        self, detail: Aggregate | None, new_message: ChatMessage
    ) -> tuple[float, Source] | tuple[None, None]:
        """Pick this turn's confidence, most trustworthy source first."""
        if self._settings.forced_confidence is not None:
            return self._settings.forced_confidence, Source.FORCED
        if detail is not None:
            return detail.score, Source.STT
        # Safety net: LiveKit computes its own mean of the turn's finals. Used when our
        # buffer was empty or its contents did not describe this turn.
        native = new_message.transcript_confidence
        if native is not None:
            return native, Source.SESSION
        return None, None


def build_session(settings: Settings, vad: silero.VAD) -> AgentSession:
    """Assemble the voice pipeline. Separated from `entrypoint` so it can be asserted on."""
    if settings.forced_confidence is not None:
        logger.warning(
            "FORCE_CONFIDENCE=%.2f is set: every turn will use this SIMULATED value and "
            "Deepgram's real confidence will be ignored. Unset it for real measurements.",
            settings.forced_confidence,
        )
    return AgentSession(
        # `smart_format` is left off deliberately: it rewrites numbers and dates, which
        # would stop a final transcript from matching its slice of the assembled turn.
        stt=deepgram.STT(model="nova-3", language="en-US"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
        vad=vad,
    )


def prewarm(process: JobProcess) -> None:
    """Load slow resources once per worker process rather than once per call."""
    process.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    settings = Settings.from_env()
    await ctx.connect()

    session = build_session(settings, ctx.proc.userdata.get("vad") or silero.VAD.load())
    await session.start(agent=ConfidenceAwareAgent(settings), room=ctx.room)
    await session.generate_reply(
        instructions="Greet the user in one short sentence and offer to help."
    )


def main() -> None:
    # Validate before handing control to the worker. A `ConfigError` raised inside
    # `prewarm` would only be logged and retried by LiveKit's process pool
    # (`ipc/proc_pool.py`: "error initializing process" ... "retrying"), and one raised
    # inside `on_user_turn_completed` is swallowed outright, silently dropping the reply.
    # Failing here exits the parent process with a readable message and nothing spawned.
    try:
        Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    main()
