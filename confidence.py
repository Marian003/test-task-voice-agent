"""Confidence policy for STT transcripts.

Pure logic — no LiveKit, no network, no I/O — so every rule the agent's behaviour
depends on is unit-testable without a running voice session. `agent.py` owns all
I/O and delegates every decision to this module.

Contents:
  * `Thresholds` / `Band`  — where the high/medium/low boundaries sit.
  * `TurnBuffer`           — collects a turn's final transcripts and aggregates them.
  * `Settings`             — validated configuration read from the environment.
  * `build_instructions` / `build_annotation` — the prompt surface.
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
    "normalize",
]


class ConfigError(ValueError):
    """Raised when an environment variable is present but malformed.

    Raised at worker start-up, never mid-turn: a typo must stop the process with a
    readable message rather than silently killing one reply at a time.
    """


class Band(str, Enum):
    """The behavioural bucket a confidence score falls into."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __str__(self) -> str:  # so f-strings render "low", not "Band.LOW"
        return self.value


class Strategy(str, Enum):
    """What to do when a turn lands in the LOW band."""

    DETERMINISTIC = "deterministic"
    """Speak a fixed clarification and skip the LLM entirely. Guaranteed behaviour."""

    LLM = "llm"
    """Annotate the context and let the model phrase the clarification itself."""


class Source(str, Enum):
    """Where the confidence value for a turn came from. Logged for auditability."""

    STT = "stt"
    """Aggregated from the Deepgram final transcripts we observed in `stt_node`."""

    SESSION = "session"
    """Fallback: LiveKit's own `ChatMessage.transcript_confidence` (mean of finals)."""

    FORCED = "forced"
    """Simulated via FORCE_CONFIDENCE. Never a real measurement."""


DEFAULT_CLARIFICATION = "Sorry, I may have misunderstood you. Could you please repeat that?"

# Words only: drops punctuation and casing so a Deepgram final ("my order.") still
# matches its slice of the assembled turn transcript ("I need help with my order").
_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, punctuation-free, single-spaced form of `text`."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text.lower()))


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Band boundaries.

    `high` is inclusive and `low` is exclusive, so the bands tile [0, 1] with no gaps:
    LOW = [0, low), MEDIUM = [low, high), HIGH = [high, 1].
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
    """One Deepgram final transcript and its confidence."""

    text: str
    confidence: float


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A turn's confidence, summarised over its final transcripts."""

    score: float
    """Word-count-weighted mean — the value the agent acts on."""

    minimum: float
    maximum: float
    samples: int
    words: int

    def __str__(self) -> str:
        return (
            f"score={self.score:.2f} min={self.minimum:.2f} "
            f"max={self.maximum:.2f} finals={self.samples} words={self.words}"
        )


def aggregate(samples: Sequence[Sample]) -> Aggregate | None:
    """Combine a turn's finals into a single score, or `None` if there are none.

    Weighting by word count rather than taking the mean (LiveKit's built-in) or the
    minimum matters in practice: Deepgram emits filler words as their own low-confidence
    finals, so a stray 0.30 "um" would drag an otherwise clean sentence into the LOW band
    under `min`, and would count as much as a twelve-word clause under an unweighted mean.
    Weighting makes a short garbled fragment cheap and a long garbled one expensive —
    which is what "did I understand this turn?" actually means.
    """
    if not samples:
        return None

    weights = [_word_count(sample.text) for sample in samples]
    total_words = sum(weights)
    if total_words:
        score = sum(s.confidence * w for s, w in zip(samples, weights, strict=False)) / total_words
    else:
        # Every sample was punctuation-only; fall back to an unweighted mean.
        score = sum(s.confidence for s in samples) / len(samples)

    confidences = [sample.confidence for sample in samples]
    return Aggregate(
        score=score,
        minimum=min(confidences),
        maximum=max(confidences),
        samples=len(samples),
        words=total_words,
    )


@dataclass(slots=True)
class TurnBuffer:
    """Accumulates the final transcripts belonging to the current user turn."""

    _samples: list[Sample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, text: str, confidence: float) -> bool:
        """Record one final transcript. Returns False if it was empty and skipped.

        Confidence is clamped rather than rejected: the provider contract is [0, 1],
        but a value slightly outside it is still signal, whereas dropping it loses a turn.
        """
        if not text.strip():
            return False
        self._samples.append(Sample(text=text, confidence=_clamp01(confidence)))
        return True

    def clear(self) -> None:
        self._samples.clear()

    def consume(self, transcript: str) -> Aggregate | None:
        """Drain the buffer and aggregate the samples that belong to `transcript`.

        The buffer is *always* emptied, even when the result is discarded. Samples that
        do not appear in the assembled turn transcript are dropped: they are left over
        from a turn that was interrupted or discarded before the completion hook ran, and
        letting them survive would mark an unrelated later turn as low-confidence.

        Returns `None` when nothing usable remains, which tells the caller to fall back
        to LiveKit's own measurement rather than guess.
        """
        samples, self._samples = self._samples, []
        if not samples:
            return None

        target = normalize(transcript)
        if not target:
            return aggregate(samples)

        kept = [s for s in samples if (n := normalize(s.text)) and n in target]
        return aggregate(kept)


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime configuration."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    strategy: Strategy = Strategy.DETERMINISTIC
    clarification: str = DEFAULT_CLARIFICATION
    forced_confidence: float | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the environment, raising `ConfigError` on bad input."""
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            thresholds=Thresholds(
                high=_read_unit_float(source, "HIGH_CONFIDENCE", 0.8),
                low=_read_unit_float(source, "LOW_CONFIDENCE", 0.6),
            ),
            strategy=_read_strategy(source),
            clarification=source.get("CLARIFICATION_PROMPT", "").strip() or DEFAULT_CLARIFICATION,
            forced_confidence=_read_optional_unit_float(source, "FORCE_CONFIDENCE"),
        )


def _read_unit_float(env: Mapping[str, str], key: str, default: float) -> float:
    value = _read_optional_unit_float(env, key)
    return default if value is None else value


def _read_optional_unit_float(env: Mapping[str, str], key: str) -> float | None:
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


def _read_strategy(env: Mapping[str, str]) -> Strategy:
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
    """System instructions, generated so the prompt can never drift from the thresholds."""
    lines = [
        "You are a friendly, concise voice assistant. Keep replies short and speakable.",
        "",
        "Some user messages are preceded by an [stt-confidence] note giving the speech "
        "recogniser's confidence in the transcript that follows, as a score in [0.0, 1.0] "
        "and one of three bands. When that note is present, it applies to the very next "
        "user message and you must follow the matching rule:",
        f"  - high   (>= {settings.thresholds.high:.2f}): you heard the user clearly. "
        "Answer normally.",
        f"  - medium (>= {settings.thresholds.low:.2f}): probably correct. Answer, but you may "
        "briefly confirm the part you are least sure of.",
        f"  - low    (<  {settings.thresholds.low:.2f}): you likely misheard. Do NOT attempt to "
        "answer or guess at the request. Apologise briefly and ask the user to repeat or "
        "rephrase.",
        "",
        "When no note is present, treat the message normally.",
        "Never say the confidence number, the band name, or the word 'transcript' out loud. "
        "The user must never hear that this metadata exists.",
    ]
    return "\n".join(lines)


def build_annotation(band: Band, score: float, source: Source, detail: Aggregate | None) -> str:
    """The metadata note injected ahead of the user's message.

    The transcript itself is deliberately not repeated here: it arrives in the very next
    message, and quoting it a second time both wastes tokens and invites the model to
    answer the quoted copy instead of the user.
    """
    note = f"[stt-confidence] score={score:.2f} band={band} source={source.value}"
    if detail is not None and detail.samples > 1:
        note += f" (weighted over {detail.samples} final transcripts, lowest {detail.minimum:.2f})"
    return f"{note}\nApply the {band} rule from your instructions to the next user message."
