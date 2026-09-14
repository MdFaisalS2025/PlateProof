"""RED-first tests for plateproof.copilot.question_validation.sanitize_question
-- the single shared question-hygiene function reused by CopilotService, the
FastAPI request schema, and the Ollama prompt builder, so all three apply
identical rules. Rejection reasons are always static/generic and never echo
the offending raw input."""

from __future__ import annotations

from plateproof.copilot.question_validation import sanitize_question


def test_empty_question_is_rejected() -> None:
    sanitized, reason = sanitize_question("", max_length=500)
    assert sanitized is None
    assert reason is not None


def test_whitespace_only_question_is_rejected() -> None:
    sanitized, reason = sanitize_question("   \t  \n  ", max_length=500)
    assert sanitized is None
    assert reason is not None


def test_ordinary_question_passes_through_stripped() -> None:
    sanitized, reason = sanitize_question("  What violations recur?  ", max_length=500)
    assert reason is None
    assert sanitized == "What violations recur?"


def test_nul_byte_is_rejected() -> None:
    sanitized, reason = sanitize_question("What about\x00 this?", max_length=500)
    assert sanitized is None
    assert reason is not None
    assert "\x00" not in reason


def test_over_length_question_is_rejected() -> None:
    sanitized, reason = sanitize_question("a" * 501, max_length=500)
    assert sanitized is None
    assert reason is not None


def test_max_length_question_is_accepted() -> None:
    sanitized, reason = sanitize_question("a" * 500, max_length=500)
    assert reason is None
    assert sanitized == "a" * 500


def test_crlf_and_cr_line_endings_are_normalized_to_lf() -> None:
    sanitized, reason = sanitize_question("line one\r\nline two\rline three", max_length=500)
    assert reason is None
    assert sanitized == "line one\nline two\nline three"


def test_tab_is_normalized_to_a_single_space() -> None:
    sanitized, reason = sanitize_question("What about\tviolations?", max_length=500)
    assert reason is None
    assert sanitized == "What about violations?"


def test_other_control_characters_are_rejected_not_silently_stripped() -> None:
    sanitized, reason = sanitize_question("What about\x07 this?", max_length=500)
    assert sanitized is None
    assert reason is not None
    assert "\x07" not in reason


def test_rejection_reason_never_echoes_the_raw_input() -> None:
    raw = "SECRET_MARKER_YOU_SHOULD_NEVER_SEE\x00"
    _, reason = sanitize_question(raw, max_length=500)
    assert reason is not None
    assert "SECRET_MARKER_YOU_SHOULD_NEVER_SEE" not in reason
