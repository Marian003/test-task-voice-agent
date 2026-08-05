"""Voice agent that reads Deepgram's STT confidence and changes what it does with it.

    speech -> Deepgram STT -> LLM (+ confidence) -> ElevenLabs TTS -> speech
                   |                ^
                   +-- confidence --+

Two hooks carry the number across the pipeline:

  stt_node                 the only place the raw SpeechEvent is visible, so the only
                           place alternatives[i].confidence can be read. We watch the
                           finals go past and change nothing about them.
  on_user_turn_completed   fires once the user's turn ends, before inference. We score
                           what we saw and either annotate the context or stop the turn.

The policy lives in confidence.py. This file is wiring and logging.
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

# No basicConfig: cli.run_app installs LiveKit's handler and we'd only duplicate records.
logger = logging.getLogger("confidence-agent")


class ConfidenceAwareAgent(Agent):
    def __init__(self, settings: Settings) -> None:
        super().__init__(instructions=build_instructions(settings))
        self._settings = settings
        self._buffer = TurnBuffer()
        self._scores: list[float] = []
        self._low_turns = 0

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: ModelSettings,
    ) -> AsyncGenerator[stt.SpeechEvent | str, None]:
        """Tap, not filter: note the confidence of each final and pass everything through."""
        async for event in Agent.default.stt_node(self, audio, model_settings):
            # the node is typed to yield plain strings too, on transcript-only paths
            if (
                isinstance(event, stt.SpeechEvent)
                and event.type is stt.SpeechEventType.FINAL_TRANSCRIPT
                and event.alternatives
            ):
                alt = event.alternatives[0]
                if self._buffer.add(alt.text, alt.confidence):
                    logger.debug(
                        "stt final transcript",
                        extra={"transcript": alt.text, "confidence": round(alt.confidence, 4)},
                    )
            yield event

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        transcript = new_message.text_content or ""
        detail = self._buffer.consume(transcript)
        score, source = self._resolve(detail, new_message)

        if score is None:
            # Text-only turn, or STT gave us nothing usable. Leave it alone; the
            # instructions already say an un-annotated message is an ordinary one.
            logger.debug("no confidence for this turn", extra={"transcript": transcript})
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
            # Structural, not prompted: the LLM is never called, so it cannot answer a
            # garbled request however it might have been inclined to. StopResponse also
            # keeps the untrusted transcript out of the conversation history.
            self.session.say(self._settings.clarification)
            raise StopResponse

        # turn_ctx is a per-turn copy, so this reaches the model for this generation only
        # and never piles up in the history.
        turn_ctx.add_message(role="system", content=build_annotation(band, score, source, detail))

    async def on_exit(self) -> None:
        """Session summary. Cheap way to spot a bad mic or a model mismatch after the fact."""
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

    def _resolve(
        self, detail: Aggregate | None, new_message: ChatMessage
    ) -> tuple[float, Source] | tuple[None, None]:
        """This turn's confidence, most trustworthy source first."""
        if self._settings.forced_confidence is not None:
            return self._settings.forced_confidence, Source.FORCED
        if detail is not None:
            return detail.score, Source.STT
        # Safety net. LiveKit keeps its own mean of the turn's finals; use it when our
        # buffer was empty or didn't describe this turn.
        native = new_message.transcript_confidence
        if native is not None:
            return native, Source.SESSION
        return None, None


def build_session(settings: Settings, vad: silero.VAD) -> AgentSession:
    """Assemble the pipeline. Split out from entrypoint so it can be asserted on."""
    if settings.forced_confidence is not None:
        logger.warning(
            "FORCE_CONFIDENCE=%.2f is set: every turn will use this SIMULATED value and "
            "Deepgram's real confidence will be ignored. Unset it for real measurements.",
            settings.forced_confidence,
        )
    return AgentSession(
        # smart_format stays off. It rewrites numbers and dates, which would stop a final
        # from lining up with its slice of the assembled turn.
        stt=deepgram.STT(model="nova-3", language="en-US"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
        vad=vad,
    )


def prewarm(process: JobProcess) -> None:
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
    # Validate before handing over to the worker. A ConfigError from prewarm only gets
    # logged and retried by the process pool (ipc/proc_pool.py: "error initializing
    # process" -> "retrying"), and one from on_user_turn_completed is swallowed outright,
    # which silently drops the reply. Failing here exits with a readable message and
    # nothing spawned.
    try:
        Settings.from_env()
    except ConfigError as exc:
        raise SystemExit(f"configuration error: {exc}") from exc

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))


if __name__ == "__main__":
    main()
