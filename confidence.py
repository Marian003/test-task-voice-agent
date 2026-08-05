"""Confidence policy: thresholds, bands, turn aggregation, config.

No livekit or network imports in here on purpose. Everything the agent's behaviour
hangs off is a pure function of its inputs, so it can be tested in milliseconds
without a session, a key or a microphone. agent.py does the I/O.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "DEFAULT_CLARIFICATION",
    "Aggregate",
    "Band",
    "ConfigError",
    "Sample",
    "Settings",
    "Source",
    "Strategy",
    "Thresholds",
    "TurnBuffer",
    "aggregate",
    "build_annotation",
    "build_instructions",
    "tokenize",
]


class ConfigError(ValueError):
    """A env var is present but malformed. Raised at startup, never mid-turn."""


class Band(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:
        # so f-strings render "low" and not "Band.LOW"
        return self.value


class Strategy(str, Enum):
    # speak a fixed line and skip the LLM. guaranteed, can't be prompted away.
    DETERMINISTIC = "deterministic"
    # annotate the context and let the model phrase the clarification itself.
    LLM = "llm"


class Source(str, Enum):
    STT = "stt"  # aggregated from the finals we saw in stt_node
    SESSION = "session"  # fallback: LiveKit's ChatMessage.transcript_confidence
    FORCED = "forced"  # FORCE_CONFIDENCE, i.e. simulated. never a measurement.


DEFAULT_CLARIFICATION = "Sorry, I may have misunderstood you. Could you please repeat that?"

# Words only. Drops punctuation and case so a Deepgram final ("my order.") still lines
# up with its slice of the assembled turn ("I need help with my order").
_WORD_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Band edges. `high` is inclusive, `low` exclusive, so the three bands tile [0, 1]:
    low = [0, low), medium = [low, high), high = [high, 1].
    """

    high: float = 0.8
    low: float = 0.6

    def __post_init__(self) -> None:
        for name in ("high", "low"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{name.upper()}_CONFIDENCE must be in [0.0, 1.0], got {value!r}")
        if self.low > self.high:
            raise ConfigError(
                f"LOW_CONFIDENCE ({self.low}) must not exceed HIGH_CONFIDENCE ({self.high})"
            )

    def band(self, confidence: float) -> Band:
        if confidence >= self.high:
            return Band.HIGH
        if confidence < self.low:
            return Band.LOW
        return Band.MEDIUM


@dataclass(frozen=True, slots=True)
class Sample:
    """One final transcript and the confidence Deepgram gave it."""

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Aggregate:
    score: float  # word-weighted mean; this is what the agent acts on
    minimum: float  # surfaced in the note, so you can see the spread behind the score
    samples: int
    words: int


def aggregate(samples: Sequence[Sample]) -> Aggregate | None:
    """Roll a turn's finals into one score. None if there's nothing to score.

    Weighted by word count rather than a plain mean (what LiveKit does internally) or a
    minimum. Deepgram emits filler words as their own low-confidence finals, so under
    `min` a stray 0.30 "um" drags a perfectly clear sentence into the low band and costs
    the user a pointless "could you repeat that?". Under a flat mean that one-word final
    counts as much as a twelve-word clause. Weighting makes a short garbled fragment
    cheap and a long garbled one expensive, which is closer to what we actually mean by
    "did I understand this turn?".
    """
    if not samples:
        return None

    weights = [len(tokenize(s.text)) for s in samples]
    total_words = sum(weights)
    if total_words:
        score = sum(s.confidence * w for s, w in zip(samples, weights, strict=True)) / total_words
    else:
        # everything was punctuation; nothing to weight by
        score = sum(s.confidence for s in samples) / len(samples)

    return Aggregate(
        score=score,
        minimum=min(s.confidence for s in samples),
        samples=len(samples),
        words=total_words,
    )


@dataclass(slots=True)
class TurnBuffer:
    """Holds the final transcripts of the turn currently being spoken."""

    _samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, text: str, confidence: float) -> bool:
        """Record a final. False if it was blank and got skipped.

        Out-of-range confidence is clamped rather than dropped. The provider contract is
        [0, 1] and a value just outside it is still signal; throwing it away loses a turn.
        """
        if not text.strip():
            return False
        self._samples.append(Sample(text=text, confidence=_clamp01(confidence)))
        return True

    def consume(self, transcript: str) -> Aggregate | None:
        """Drain the buffer and score the finals that actually reconstruct `transcript`.

        The buffer is always emptied, whatever the outcome.

        LiveKit assembles a turn by appending each final in order (audio_recognition.py:
        `self._audio_transcript += f" {transcript}"`), so we replay that: walk the samples
        in order and keep each one whose tokens sit exactly at the cursor. Anything that
        doesn't fit is left over from a turn that got interrupted or discarded before the
        completion hook ran, and would otherwise mark an unrelated later turn as low.

        Requiring an exact fit at the cursor (rather than "somewhere after it") is what
        makes this a real check. A looser search accepts gaps: ["yes", "no"] against
        "yes yes no" would pass a start/end check while skipping the middle token.

        Matching is on token runs, never substrings, or "no" matches inside "i know that".

        Returns None when the buffer doesn't describe this turn, which tells the caller to
        fall back to LiveKit's own number instead of scoring on partial evidence.

        Not perfect: a leftover final that happens to be token-identical to the start of
        the next turn ("yes", then "yes, what time do you close") still fits at the cursor
        and gets kept. You'd need per-turn ids from the framework to tell those apart, and
        it only costs that final's own word weight, so I've left it.
        """
        samples, self._samples = self._samples, []
        if not samples:
            return None

        target = tokenize(transcript)
        if not target:
            # nothing to check against, so trust what we have
            return aggregate(samples)

        kept: list[Sample] = []
        cursor = 0
        for sample in samples:
            tokens = tokenize(sample.text)
            if tokens and target[cursor : cursor + len(tokens)] == tokens:
                kept.append(sample)
                cursor += len(tokens)

        if cursor != len(target):
            return None
        return aggregate(kept)


@dataclass(frozen=True, slots=True)
class Settings:
    thresholds: Thresholds = field(default_factory=Thresholds)
    strategy: Strategy = Strategy.DETERMINISTIC
    clarification: str = DEFAULT_CLARIFICATION
    forced_confidence: float | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            thresholds=Thresholds(
                high=_unit_float(source, "HIGH_CONFIDENCE", 0.8),
                low=_unit_float(source, "LOW_CONFIDENCE", 0.6),
            ),
            strategy=_strategy(source),
            clarification=source.get("CLARIFICATION_PROMPT", "").strip() or DEFAULT_CLARIFICATION,
            forced_confidence=_optional_unit_float(source, "FORCE_CONFIDENCE"),
        )


def _unit_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _optional_unit_float(env, key)
    return default if value is None else value


def _optional_unit_float(env: Mapping[str, str], key: str) -> float | None:
    raw = env.get(key, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number in [0.0, 1.0], got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{key} must be in [0.0, 1.0], got {value!r}")
    return value


def _strategy(env: Mapping[str, str]) -> Strategy:
    raw = env.get("LOW_CONFIDENCE_STRATEGY", "").strip().lower()
    if not raw:
        return Strategy.DETERMINISTIC
    try:
        return Strategy(raw)
    except ValueError as exc:
        allowed = ", ".join(s.value for s in Strategy)
        raise ConfigError(
            f"LOW_CONFIDENCE_STRATEGY must be one of: {allowed}. Got {raw!r}"
        ) from exc


def build_instructions(settings: Settings) -> str:
    """System prompt, generated from the thresholds so the two can't drift apart."""
    high, low = settings.thresholds.high, settings.thresholds.low
    return "\n".join(
        [
            "You are a friendly, concise voice assistant. Keep replies short and speakable.",
            "",
            "Some user messages are preceded by an [stt-confidence] note giving the speech "
            "recogniser's confidence in the transcript that follows, as a score in [0.0, 1.0] "
            "and one of three bands. When that note is present it applies to the very next "
            "user message, and you must follow the matching rule:",
            f"  - high   ({high:.2f} to 1.00): you heard the user clearly. Answer normally.",
            f"  - medium ({low:.2f} up to {high:.2f}): probably correct. Answer, but you may "
            "briefly confirm the part you are least sure of.",
            f"  - low    (0.00 up to {low:.2f}): you likely misheard. Do NOT attempt to answer "
            "or guess at the request. Apologise briefly and ask the user to repeat or rephrase.",
            "",
            "When no note is present, treat the message normally.",
            "Never say the confidence number, the band name, or the word 'transcript' out loud. "
            "The user must never hear that this metadata exists.",
        ]
    )


def build_annotation(band: Band, score: float, source: Source, detail: Aggregate | None) -> str:
    """The note we slip in ahead of the user's message.

    We don't repeat the transcript here. It arrives in the very next message anyway, and
    quoting it twice burns tokens on every turn and tempts the model to answer the copy.
    """
    note = f"[stt-confidence] score={score:.2f} band={band} source={source.value}"
    if detail is not None and detail.samples > 1:
        note += f" (weighted over {detail.samples} final transcripts, lowest {detail.minimum:.2f})"
    return f"{note}\nApply the {band} rule from your instructions to the next user message."
