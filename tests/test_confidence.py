"""Unit tests for the confidence policy. No network, no LiveKit session required."""

from __future__ import annotations

import pytest

from confidence import (
    DEFAULT_CLARIFICATION,
    Band,
    ConfigError,
    Sample,
    Settings,
    Source,
    Strategy,
    Thresholds,
    TurnBuffer,
    aggregate,
    build_annotation,
    build_instructions,
    tokenize,
)


# --- bands --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.0, Band.LOW),
        (0.59, Band.LOW),
        (0.5999, Band.LOW),
        (0.6, Band.MEDIUM),  # `low` is the inclusive floor of MEDIUM
        (0.79, Band.MEDIUM),
        (0.8, Band.HIGH),  # `high` is inclusive
        (1.0, Band.HIGH),
    ],
)
def test_bands_tile_the_unit_interval(score: float, expected: Band) -> None:
    assert Thresholds().band(score) is expected


def test_custom_thresholds_are_respected() -> None:
    strict = Thresholds(high=0.95, low=0.9)
    assert strict.band(0.92) is Band.MEDIUM
    assert strict.band(0.89) is Band.LOW
    assert strict.band(0.95) is Band.HIGH


def test_degenerate_thresholds_collapse_medium_without_error() -> None:
    both = Thresholds(high=0.7, low=0.7)
    assert both.band(0.7) is Band.HIGH
    assert both.band(0.69) is Band.LOW


@pytest.mark.parametrize(
    "kwargs",
    [
        {"high": 1.5},
        {"low": -0.1},
        {"high": 0.5, "low": 0.9},  # low above high
    ],
)
def test_invalid_thresholds_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        Thresholds(**kwargs)


# --- aggregation --------------------------------------------------------------------
def test_aggregate_of_nothing_is_none() -> None:
    assert aggregate([]) is None


def test_single_sample_scores_itself() -> None:
    result = aggregate([Sample("hello there", 0.91)])
    assert result is not None
    assert result.score == pytest.approx(0.91)
    assert (result.minimum, result.samples, result.words) == (0.91, 1, 2)


def test_short_filler_does_not_drag_a_clean_turn_low() -> None:
    """The regression this weighting exists for: a stray 'um' must not trigger a repeat."""
    samples = [
        Sample("um", 0.30),
        Sample("what time do you close today", 0.96),
    ]
    result = aggregate(samples)
    assert result is not None
    # min() would give 0.30 -> LOW; an unweighted mean 0.63 -> MEDIUM; weighted 0.88 -> HIGH.
    assert result.score == pytest.approx((0.30 * 1 + 0.96 * 6) / 7)
    assert Thresholds().band(result.score) is Band.HIGH
    assert result.minimum == 0.30


def test_a_long_garbled_segment_still_pulls_the_turn_low() -> None:
    samples = [
        Sample("yes", 0.97),
        Sample("i need help with the thing about my order please", 0.34),
    ]
    result = aggregate(samples)
    assert result is not None
    assert Thresholds().band(result.score) is Band.LOW


def test_punctuation_only_samples_fall_back_to_unweighted_mean() -> None:
    result = aggregate([Sample("...", 0.4), Sample("?!", 0.8)])
    assert result is not None
    assert result.words == 0
    assert result.score == pytest.approx(0.6)


# --- turn buffer --------------------------------------------------------------------
def test_buffer_skips_blank_transcripts() -> None:
    buffer = TurnBuffer()
    assert buffer.add("   ", 0.9) is False
    assert buffer.add("hello", 0.9) is True
    assert len(buffer) == 1


def test_buffer_clamps_out_of_range_confidence() -> None:
    buffer = TurnBuffer()
    buffer.add("a", 1.4)
    buffer.add("b", -0.2)
    result = buffer.consume("a b")
    assert result is not None
    assert result.minimum == 0.0
    assert result.score == pytest.approx(0.5)  # 1.4 -> 1.0 and -0.2 -> 0.0


def test_consume_matches_finals_against_the_assembled_transcript() -> None:
    buffer = TurnBuffer()
    buffer.add("I need help", 0.9)
    buffer.add("with my order.", 0.5)
    result = buffer.consume("I need help with my order")
    assert result is not None
    assert result.samples == 2


def test_matching_is_on_token_runs_not_substrings() -> None:
    """Regression: `"no"` must not match inside `"i know that"` and drag the turn down."""
    buffer = TurnBuffer()
    buffer.add("no", 0.10)  # stale leftover from a discarded turn
    buffer.add("i know that", 0.95)
    result = buffer.consume("i know that")
    assert result is not None
    assert result.samples == 1
    assert result.score == pytest.approx(0.95)


def test_a_fragment_inside_a_longer_word_is_not_a_match() -> None:
    buffer = TurnBuffer()
    buffer.add("art", 0.10)
    buffer.add("that was smart", 0.95)
    result = buffer.consume("that was smart")
    assert result is not None
    assert result.samples == 1
    assert result.score == pytest.approx(0.95)


def test_repeated_words_are_matched_positionally_not_reused() -> None:
    buffer = TurnBuffer()
    buffer.add("yes", 0.9)
    buffer.add("yes", 0.5)
    result = buffer.consume("yes yes")
    assert result is not None
    assert result.samples == 2
    assert result.score == pytest.approx(0.7)


def test_punctuation_only_finals_are_skipped_without_breaking_coverage() -> None:
    buffer = TurnBuffer()
    buffer.add("...", 0.10)
    buffer.add("hello", 0.90)
    result = buffer.consume("hello")
    assert result is not None
    assert result.samples == 1
    assert result.score == pytest.approx(0.90)


def test_partial_coverage_defers_to_the_caller() -> None:
    """Half a turn is not evidence for the whole turn — fall back rather than guess."""
    buffer = TurnBuffer()
    buffer.add("i need help", 0.4)
    assert buffer.consume("i need help with my order") is None


def test_coverage_must_start_at_the_first_token() -> None:
    buffer = TurnBuffer()
    buffer.add("with my order", 0.4)
    assert buffer.consume("i need help with my order") is None


def test_consume_always_empties_the_buffer() -> None:
    buffer = TurnBuffer()
    buffer.add("hello", 0.9)
    buffer.consume("hello")
    assert len(buffer) == 0
    assert buffer.consume("hello") is None


def test_stale_samples_from_a_discarded_turn_are_dropped() -> None:
    """Regression: a turn whose completion hook never ran must not poison the next one."""
    buffer = TurnBuffer()
    buffer.add("mumbled nonsense", 0.21)  # previous turn, never consumed
    buffer.add("what time do you close", 0.95)  # this turn
    result = buffer.consume("what time do you close")
    assert result is not None
    assert result.samples == 1
    assert result.score == pytest.approx(0.95)


def test_completely_mismatched_buffer_defers_to_the_caller() -> None:
    buffer = TurnBuffer()
    buffer.add("something else entirely", 0.2)
    assert buffer.consume("what time do you close") is None


def test_empty_transcript_keeps_every_sample() -> None:
    buffer = TurnBuffer()
    buffer.add("hello", 0.7)
    result = buffer.consume("")
    assert result is not None
    assert result.samples == 1


def test_a_gap_in_the_middle_is_not_coverage() -> None:
    """Regression: matching "somewhere after the cursor" let ["yes", "no"] skip a token
    in "yes yes no" and still pass a start/end check, scoring 0.53 instead of deferring.
    """
    buffer = TurnBuffer()
    buffer.add("yes", 0.95)
    buffer.add("no", 0.10)
    assert buffer.consume("yes yes no") is None


def test_tokenize_strips_case_and_punctuation() -> None:
    assert tokenize("  Hello,   World! ") == ["hello", "world"]
    assert tokenize("...") == []


# --- settings -----------------------------------------------------------------------
def test_defaults_when_environment_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings.thresholds == Thresholds(high=0.8, low=0.6)
    assert settings.strategy is Strategy.DETERMINISTIC
    assert settings.clarification == DEFAULT_CLARIFICATION
    assert settings.forced_confidence is None


def test_blank_values_are_treated_as_unset() -> None:
    settings = Settings.from_env(
        {"FORCE_CONFIDENCE": "   ", "LOW_CONFIDENCE_STRATEGY": "", "CLARIFICATION_PROMPT": "  "}
    )
    assert settings.forced_confidence is None
    assert settings.strategy is Strategy.DETERMINISTIC
    assert settings.clarification == DEFAULT_CLARIFICATION


def test_environment_overrides_are_applied() -> None:
    settings = Settings.from_env(
        {
            "HIGH_CONFIDENCE": "0.9",
            "LOW_CONFIDENCE": "0.7",
            "LOW_CONFIDENCE_STRATEGY": "LLM",
            "CLARIFICATION_PROMPT": "Come again?",
            "FORCE_CONFIDENCE": "0.35",
        }
    )
    assert settings.thresholds == Thresholds(high=0.9, low=0.7)
    assert settings.strategy is Strategy.LLM
    assert settings.clarification == "Come again?"
    assert settings.forced_confidence == 0.35


def test_force_confidence_of_zero_is_honoured_not_ignored() -> None:
    """`0.0` is falsy but meaningful — the classic bug this guards against."""
    assert Settings.from_env({"FORCE_CONFIDENCE": "0"}).forced_confidence == 0.0


@pytest.mark.parametrize("raw", ["high", " abc ", "0.9.9", "0,5"])
def test_unparseable_force_confidence_is_rejected_at_startup(raw: str) -> None:
    with pytest.raises(ConfigError, match="FORCE_CONFIDENCE"):
        Settings.from_env({"FORCE_CONFIDENCE": raw})


@pytest.mark.parametrize("raw", ["7.5", "-1", "1.0001"])
def test_out_of_range_force_confidence_is_rejected(raw: str) -> None:
    with pytest.raises(ConfigError, match=r"\[0.0, 1.0\]"):
        Settings.from_env({"FORCE_CONFIDENCE": raw})


def test_unknown_strategy_is_rejected_with_the_valid_options() -> None:
    with pytest.raises(ConfigError, match="deterministic"):
        Settings.from_env({"LOW_CONFIDENCE_STRATEGY": "maybe"})


def test_bad_threshold_from_env_is_rejected() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({"HIGH_CONFIDENCE": "1.2"})


# --- prompt surface -----------------------------------------------------------------
def test_instructions_quote_the_configured_thresholds() -> None:
    text = build_instructions(
        Settings.from_env({"HIGH_CONFIDENCE": "0.9", "LOW_CONFIDENCE": "0.7"})
    )
    assert "high   (0.90 to 1.00)" in text
    assert "medium (0.70 up to 0.90)" in text
    assert "low    (0.00 up to 0.70)" in text


def test_instructions_forbid_leaking_the_metadata() -> None:
    assert "Never say the confidence number" in build_instructions(Settings.from_env({}))


def test_annotation_does_not_repeat_the_transcript() -> None:
    note = build_annotation(Band.LOW, 0.42, Source.STT, None)
    assert "score=0.42" in note
    assert "band=low" in note
    assert "source=stt" in note
    assert "next user message" in note


def test_annotation_reports_the_spread_when_several_finals_were_merged() -> None:
    detail = aggregate([Sample("um", 0.3), Sample("hello there friend", 0.9)])
    note = build_annotation(Band.MEDIUM, 0.75, Source.STT, detail)
    assert "weighted over 2 final transcripts" in note
    assert "lowest 0.30" in note
