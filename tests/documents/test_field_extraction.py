"""Tests for jurisdiction-neutral extraction primitives: label-proximity
matching, date/number parsing, and confidence combination."""

from __future__ import annotations

from datetime import date


def _block(
    text: str, *, page: int = 1, source: str = "embedded_text", ocr_confidence: float | None = None
) -> object:
    from plateproof.documents.models import TextBlock

    return TextBlock(
        page=page, text=text, bounding_box=None, source=source, ocr_confidence=ocr_confidence
    )


def test_find_label_matches_extracts_value_after_colon() -> None:
    from plateproof.documents.field_extraction import find_label_matches

    blocks = (_block("Score: 14\nGrade: A"),)
    matches = find_label_matches(blocks, ("Score",))
    assert len(matches) == 1
    assert matches[0].value_text == "14"


def test_find_label_matches_is_case_insensitive() -> None:
    from plateproof.documents.field_extraction import find_label_matches

    blocks = (_block("score: 22"),)
    matches = find_label_matches(blocks, ("Score",))
    assert len(matches) == 1


def test_find_label_matches_returns_empty_when_label_absent() -> None:
    from plateproof.documents.field_extraction import find_label_matches

    blocks = (_block("Nothing relevant here"),)
    matches = find_label_matches(blocks, ("Score",))
    assert matches == []


def test_parse_date_multi_tries_formats_in_order() -> None:
    from plateproof.documents.field_extraction import NYC_DATE_FORMATS, parse_date_multi

    assert parse_date_multi("2024-01-15", NYC_DATE_FORMATS) == date(2024, 1, 15)
    assert parse_date_multi("01/15/2024", NYC_DATE_FORMATS) == date(2024, 1, 15)
    assert parse_date_multi("not a date", NYC_DATE_FORMATS) is None


def test_florida_date_order_prefers_slashed_format() -> None:
    from plateproof.documents.field_extraction import FLORIDA_DATE_FORMATS, parse_date_multi

    assert parse_date_multi("01/15/2024", FLORIDA_DATE_FORMATS) == date(2024, 1, 15)
    assert parse_date_multi("2024-01-15", FLORIDA_DATE_FORMATS) == date(2024, 1, 15)


def test_parse_nonneg_number_rejects_negative() -> None:
    from plateproof.documents.field_extraction import parse_nonneg_number

    assert parse_nonneg_number("14") == 14.0
    assert parse_nonneg_number("-5") is None
    assert parse_nonneg_number("not a number") is None


def test_combine_confidence_averages_available_components() -> None:
    from plateproof.documents.field_extraction import combine_confidence

    result = combine_confidence(
        ocr_confidence=None, pattern_confidence=0.9, corroboration_confidence=None
    )
    assert result.overall == 0.9
    assert result.ocr_confidence is None

    result2 = combine_confidence(
        ocr_confidence=0.8, pattern_confidence=0.9, corroboration_confidence=1.0
    )
    assert 0.8 <= result2.overall <= 1.0


def test_classify_confidence_is_categorical_not_numeric() -> None:
    """Finding 8: confidence is evidence-condition-based (embedded text vs
    OCR, corroborated vs not) -- never a numeric-average threshold. See
    test_confidence_semantics.py for the full behavioral test suite."""
    from plateproof.documents.field_extraction import classify_confidence

    assert classify_confidence(source="embedded_text") == "high"
    assert classify_confidence(source="ocr") == "needs_review"
    assert classify_confidence(source="embedded_text", corroborated=False) == "needs_review"


def test_resolve_field_returns_none_when_no_matches() -> None:
    from plateproof.documents.field_extraction import resolve_field

    candidate, ambiguity = resolve_field("score", [], parse_value=float)
    assert candidate is None
    assert ambiguity is None


def test_resolve_field_builds_candidate_from_single_match() -> None:
    from plateproof.documents.field_extraction import (
        find_label_matches,
        parse_nonneg_number,
        resolve_field,
    )

    blocks = (_block("Score: 14"),)
    matches = find_label_matches(blocks, ("Score",))
    candidate, ambiguity = resolve_field("score", matches, parse_value=parse_nonneg_number)
    assert ambiguity is None
    assert candidate is not None
    assert candidate.value == 14.0
    assert candidate.evidence[0].page == 1


def test_resolve_field_reports_ambiguity_for_conflicting_values() -> None:
    from plateproof.documents.field_extraction import (
        find_label_matches,
        parse_nonneg_number,
        resolve_field,
    )

    blocks = (_block("Score: 14"), _block("Score: 28", page=2))
    matches = find_label_matches(blocks, ("Score",))
    candidate, ambiguity = resolve_field("score", matches, parse_value=parse_nonneg_number)
    assert candidate is None
    assert ambiguity is not None
    assert ambiguity.field_name == "score"
    assert set(ambiguity.candidate_values) == {"14.0", "28.0"}


def test_resolve_field_repeated_identical_value_is_not_ambiguous() -> None:
    from plateproof.documents.field_extraction import (
        find_label_matches,
        parse_nonneg_number,
        resolve_field,
    )

    blocks = (_block("Score: 14"), _block("Score: 14", page=2))
    matches = find_label_matches(blocks, ("Score",))
    candidate, ambiguity = resolve_field("score", matches, parse_value=parse_nonneg_number)
    assert ambiguity is None
    assert candidate is not None
    assert candidate.value == 14.0
