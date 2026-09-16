"""Tests for strict, mechanical user-correction validation."""

from __future__ import annotations

from datetime import UTC, datetime


def _draft(jurisdiction: str = "nyc") -> object:
    from plateproof.documents.models import ExtractionDraft, OcrEngineInfo, UploadMetadata

    return ExtractionDraft(
        draft_id="d1",
        jurisdiction_expected=jurisdiction,
        jurisdiction_detected=jurisdiction,
        jurisdiction_mismatch=False,
        restaurant_id="nyc:1",
        restaurant_identity_corroborated=True,
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=10, page_count=1),
        pages=(),
        candidates={},
        violations=(),
        ambiguities=(),
        missing_fields=("score",),
        warnings=(),
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )


def test_correction_for_known_field_is_parsed() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("score", "14", _draft())
    assert reason is None
    assert validated.parse_status == "parsed"
    assert validated.parsed_value == 14.0


def test_correction_for_unknown_field_rejected() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("michelin_star_rank", "3", _draft())
    assert validated is None
    assert reason is not None


def test_correction_exceeding_max_length_rejected() -> None:
    from plateproof.documents.corrections import MAX_CORRECTION_FIELD_LENGTH, validate_correction

    too_long = "x" * (MAX_CORRECTION_FIELD_LENGTH + 1)
    validated, reason = validate_correction("restaurant_name", too_long, _draft())
    assert validated is None
    assert reason is not None


def test_correction_with_control_character_rejected() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("restaurant_name", "Joe\x00's Pizza", _draft())
    assert validated is None
    assert reason is not None


def test_unparseable_date_correction_is_visibly_unparseable_not_dropped() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("inspection_date", "not a date", _draft())
    assert reason is None
    assert validated is not None
    assert validated.parse_status == "unparseable"
    assert validated.parsed_value is None
    assert validated.raw_display_value == "not a date"


def test_non_finite_numeric_correction_rejected() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("score", "nan", _draft())
    # "nan" does not parse as a nonneg number via float() semantics used
    # by parse_nonneg_number, matching machine extraction's own parser --
    # it must not silently become a numeric candidate.
    assert reason is None
    assert validated is not None
    assert validated.parse_status in ("unparseable", "rejected")


def test_florida_disposition_correction_uses_jurisdiction_appropriate_parser() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction(
        "disposition_status", "Warning Issued", _draft("florida")
    )
    assert reason is None
    assert validated.parse_status == "parsed"
    assert validated.parsed_value == "follow_up_required"


def test_free_text_field_never_uses_wrong_jurisdiction_parser() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("high_priority_count", "2", _draft("florida"))
    assert reason is None
    assert validated.parsed_value == 2.0
    # The same field name is not a known NYC field.
    validated_nyc, reason_nyc = validate_correction("high_priority_count", "2", _draft("nyc"))
    assert validated_nyc is None
    assert reason_nyc is not None


def test_validate_corrections_batch_enforces_total_payload_limit() -> None:
    from plateproof.documents.corrections import validate_corrections

    # Individual fields are unknown for this draft's jurisdiction, but the
    # aggregate-length check must fire before any per-field key check would.
    many_fields = {f"field_{i}": "x" * 400 for i in range(60)}
    results, reason = validate_corrections(many_fields, _draft())
    assert reason is not None
    assert results == {}


def test_validate_corrections_batch_succeeds_for_valid_small_payload() -> None:
    from plateproof.documents.corrections import validate_corrections

    results, reason = validate_corrections({"score": "14", "grade": "A"}, _draft())
    assert reason is None
    assert results["score"].parsed_value == 14.0
    assert results["grade"].parsed_value == "A"
