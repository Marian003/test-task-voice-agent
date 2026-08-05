"""Stand-in STT/LLM/TTS plugins so a real AgentSession can run without network or keys.

These implement the same base classes the Deepgram/OpenAI/ElevenLabs plugins do, so the
session, the turn lifecycle and the agent hooks are all the genuine article. Only the
provider calls are faked.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from livekit import rtc
from livekit.agents import APIConnectOptions, llm, stt, tts, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice import io

# ruff flags a call in a default arg; the base classes use this shape, so share one instance.
DEFAULT_CONN = APIConnectOptions()

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# (text, confidence) pairs making up one user turn.
Script = Sequence[tuple[str, float]]


class ScriptedSTT(stt.STT):
    """Emits a canned set of final transcripts, one turn per `feed()` call.

    The audio is ignored. What matters for these tests is the shape of the event stream
    the agent sees, not what produced it.
    """

    def __init__(self) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=False))
        self._streams: list[_ScriptedStream] = []

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_CONN,
    ) -> _ScriptedStream:
        stream = _ScriptedStream(stt=self, conn_options=conn_options)
        self._streams.append(stream)
        return stream

    async def _recognize_impl(self, *args: object, **kwargs: object) -> stt.SpeechEvent:
        raise NotImplementedError("streaming only")

    async def feed(self, script: Script) -> None:
        """Push one user turn through every open stream and wait for it to be picked up."""
        for stream in self._streams:
            stream.push_turn(script)
        await asyncio.sleep(0.2)


class _ScriptedStream(stt.SpeechStream):
    def __init__(self, *, stt: ScriptedSTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options)
        self._pending: asyncio.Queue[Script] = asyncio.Queue()

    def push_turn(self, script: Script) -> None:
        self._pending.put_nowait(script)

    async def _run(self) -> None:
        # Drain the audio channel so the session's input does not back up.
        async def swallow_audio() -> None:
            async for _ in self._input_ch:
                pass

        audio_task = asyncio.create_task(swallow_audio())
        try:
            while True:
                script = await self._pending.get()
                self._event_ch.send_nowait(
                    stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH)
                )
                for text, confidence in script:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[
                                stt.SpeechData(language="en-US", text=text, confidence=confidence)
                            ],
                        )
                    )
                self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
        finally:
            await utils.aio.cancel_and_wait(audio_task)


class ScriptedLLM(llm.LLM):
    """Always replies with the same sentence, and records the context it was given."""

    def __init__(self, reply: str = "Sure, happy to help.") -> None:
        super().__init__()
        self._reply = reply
        self.seen_contexts: list[llm.ChatContext] = []

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_CONN,
        **kwargs: object,
    ) -> _ScriptedLLMStream:
        self.seen_contexts.append(chat_ctx.copy())
        return _ScriptedLLMStream(
            self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options, reply=self._reply
        )

    def system_notes(self) -> list[str]:
        """Every [stt-confidence] note the LLM was handed, oldest first."""
        return [
            item.text_content or ""
            for ctx in self.seen_contexts
            for item in ctx.items
            if getattr(item, "role", None) == "system"
            and (item.text_content or "").startswith("[stt-confidence]")
        ]


class _ScriptedLLMStream(llm.LLMStream):
    def __init__(self, llm_: ScriptedLLM, *, reply: str, **kwargs: object) -> None:
        super().__init__(llm_, **kwargs)  # type: ignore[arg-type]
        self._reply = reply

    async def _run(self) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=utils.shortuuid(),
                delta=llm.ChoiceDelta(role="assistant", content=self._reply),
            )
        )


class ScriptedTTS(tts.TTS):
    """Returns silence, and records every line it was asked to speak."""

    def __init__(self) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        self.spoken: list[str] = []

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_CONN
    ) -> _ScriptedTTSStream:
        self.spoken.append(text)
        return _ScriptedTTSStream(tts=self, input_text=text, conn_options=conn_options)


class _ScriptedTTSStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )
        output_emitter.push(b"\x00\x00" * (SAMPLE_RATE // 10))
        output_emitter.flush()


class CollectingAudioOutput(io.AudioOutput):
    """Swallows synthesised audio. Without an audio sink the session skips TTS entirely."""

    def __init__(self) -> None:
        super().__init__(
            label="collector",
            capabilities=io.AudioOutputCapabilities(pause=False),
            sample_rate=SAMPLE_RATE,
        )
        self._captured = False

    async def capture_frame(self, frame: rtc.AudioFrame) -> None:
        self._captured = True

    def flush(self) -> None:
        # Only report a segment we were actually given. Reporting unconditionally makes the
        # framework log "playback_finished called more times than playback segments were
        # captured" and can trip its false-interruption resume, which fires a second
        # generation and makes anything asserting on the last LLM context non-deterministic.
        if not self._captured:
            return
        self._captured = False
        self.on_playback_finished(playback_position=0.1, interrupted=False)

    def clear_buffer(self) -> None:
        self._captured = False


class SilentAudioInput(io.AudioInput):
    """A never-ending stream of silence, so the session always has an audio input."""

    def __init__(self) -> None:
        super().__init__(label="silence")
        self._samples = SAMPLE_RATE // 100  # 10 ms

    async def __anext__(self) -> rtc.AudioFrame:
        await asyncio.sleep(0.01)
        return rtc.AudioFrame(
            data=b"\x00\x00" * self._samples,
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            samples_per_channel=self._samples,
        )
