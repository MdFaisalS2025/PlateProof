"""Finding 1: NYC and Florida extraction must cover every field the
approved plan requires, support multiple violation rows without collapsing
them into one scalar, keep jurisdiction violation-code vocabularies
strictly separate, never infer a code from description similarity, never
calculate a deadline, and preserve unknown codes as review-needed evidence
with no normalized proposed value.
"""

from __future__ import annotations


def _block(text: str, page: int = 1) -> object:
    from plateproof.documents.models import TextBlock

    return TextBlock(
        page=page, text=text, bounding_box=None, source="embedded_text", ocr_confidence=None
    )


# --------------------------------------------------------------------------- #
# NYC additional scalar fields                                               #
# --------------------------------------------------------------------------- #


def test_nyc_extracts_camis_and_inspection_type() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (
        _block(
            "CAMIS: 41234567\n"
            "Restaurant Name: Joe's Pizza\n"
            "Inspection Date: 01/15/2024\n"
            "Inspection Type: Cycle Inspection / Initial Inspection\n"
            "Score: 14\nGrade: A"
        ),
    )
    candidates, violations, ambiguities, missing = extract_nyc_candidates(blocks)
    assert candidates["camis"].value == "41234567"
    assert candidates["inspection_type"].value == "Cycle Inspection / Initial Inspection"


def test_nyc_same_line_layout_does_not_let_camis_swallow_dba() -> None:
    """Finding 1: avoid a single greedy label match consuming a subsequent
    labeled field on the same line."""
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (_block("CAMIS: 41234567   DBA: Joe's Pizza"),)
    candidates, violations, ambiguities, missing = extract_nyc_candidates(blocks)
    assert candidates["camis"].value == "41234567"
    assert candidates["restaurant_name"].value == "Joe's Pizza"


# --------------------------------------------------------------------------- #
# NYC violation rows                                                         #
# --------------------------------------------------------------------------- #


def test_nyc_extracts_multiple_violation_rows_newline_layout() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    text = (
        "Violation Code: 10F\n"
        "Violation Description: Non-food contact surface improperly constructed\n"
        "Critical Flag: Not Critical\n"
        "Violation Code: 04L\n"
        "Violation Description: Evidence of mice\n"
        "Critical Flag: Critical\n"
    )
    candidates, violations, ambiguities, missing = extract_nyc_candidates((_block(text),))
    assert len(violations) == 2
    assert violations[0].code == "10f"
    assert violations[0].description == "Non-food contact surface improperly constructed"
    assert violations[0].critical is False
    assert violations[1].code == "04l"
    assert violations[1].critical is True


def test_nyc_extracts_violation_row_same_line_layout() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    text = (
        "Violation Code: 06C  Violation Description: Improper hand washing  Critical Flag: Critical"
    )
    candidates, violations, ambiguities, missing = extract_nyc_candidates((_block(text),))
    assert len(violations) == 1
    assert violations[0].code == "06c"
    assert violations[0].description == "Improper hand washing"
    assert violations[0].critical is True


def test_nyc_violation_rows_never_collapse_to_one_scalar_field() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    text = "Violation Code: 10F\nViolation Code: 04L\nViolation Code: 08A\n"
    candidates, violations, ambiguities, missing = extract_nyc_candidates((_block(text),))
    assert "violation_code" not in candidates  # never a single scalar
    assert len(violations) == 3


def test_nyc_violation_evidence_has_bounded_page_provenance() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (_block("Violation Code: 10F", page=1), _block("Violation Code: 04L", page=2))
    candidates, violations, ambiguities, missing = extract_nyc_candidates(blocks)
    pages = {v.evidence[0].page for v in violations}
    assert pages == {1, 2}
    for v in violations:
        assert len(v.evidence[0].excerpt) <= 300


# --------------------------------------------------------------------------- #
# Florida additional scalar fields                                           #
# --------------------------------------------------------------------------- #


def test_florida_extracts_license_visit_id_type_and_sequence() -> None:
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = (
        "License Number: HTL1234567\n"
        "Inspection Visit ID: 987654\n"
        "Inspection Type: Routine - Food\n"
        "Visit Sequence: 1\n"
        "Restaurant Name: Sunshine Cafe\n"
        "Inspection Date: 03/02/2024\n"
        "High Priority: 2\nIntermediate: 1\nBasic: 3\n"
        "Disposition: Warning Issued\n"
    )
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert candidates["license_number"].value == "HTL1234567"
    assert candidates["inspection_visit_id"].value == "987654"
    assert candidates["inspection_type"].value == "Routine - Food"
    assert candidates["visit_sequence"].value == 1.0


def test_florida_extracts_explicitly_printed_correction_deadline_only() -> None:
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = (
        "Correction Deadline: 04/15/2024\nRestaurant Name: Sunshine Cafe\n"
        "Inspection Date: 03/02/2024\n"
    )
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert candidates["correction_deadline"].value.isoformat() == "2024-04-15"


def test_florida_never_calculates_a_deadline_when_not_printed() -> None:
    """No 'Correction Deadline' text anywhere -- the field is simply
    missing, never computed from the inspection date."""
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = "Restaurant Name: Sunshine Cafe\nInspection Date: 03/02/2024\n"
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert "correction_deadline" not in candidates


# --------------------------------------------------------------------------- #
# Florida violation rows: unknown-code review-needed, cross-jurisdiction     #
# isolation, multiple rows                                                   #
# --------------------------------------------------------------------------- #


def test_florida_extracts_multiple_violation_rows() -> None:
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = (
        "Violation Code: 03\nViolation Description: Food Out of Temperature\n"
        "Violation Code: 12\nViolation Description: Hands washed and clean\n"
    )
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert len(violations) == 2
    assert violations[0].code == "03"
    assert violations[1].code == "12"


def test_florida_unknown_violation_code_is_review_needed_not_guessed() -> None:
    """A code outside 1-58 (Florida's real closed vocabulary) is preserved
    as raw, review-needed evidence with no normalized proposed value --
    never guessed from the description."""
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = "Violation Code: 99\nViolation Description: Some unrecognized category\n"
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert len(violations) == 1
    assert violations[0].code is None
    assert violations[0].raw_code_text == "99"
    assert violations[0].description == "Some unrecognized category"


def test_florida_violation_code_never_inferred_from_description_similarity() -> None:
    """Even though the description text closely matches a known category's
    real description, an unrecognized raw code must never be silently
    mapped to that category by description matching."""
    from plateproof.documents.florida_extractor import extract_florida_candidates

    text = "Violation Code: XX\nViolation Description: Food Out of Temperature\n"
    candidates, violations, ambiguities, missing = extract_florida_candidates((_block(text),))
    assert violations[0].code is None


def test_florida_and_nyc_violation_vocabularies_are_never_cross_applied() -> None:
    """A Florida-shaped code (numeric, 1-58) run through the NYC extractor
    is treated as an NYC free-text code (normalized, never validated
    against Florida's vocabulary) and vice versa -- the two normalizers
    are never substituted for each other."""
    from plateproof.documents.florida_extractor import extract_florida_candidates
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    nyc_text = "Violation Code: 99\nViolation Description: Not a real FL category\n"
    _, nyc_violations, _, _ = extract_nyc_candidates((_block(nyc_text),))
    # NYC has no fixed 1-58 vocabulary -- "99" is accepted as a normalized
    # free-text NYC code, never rejected the way Florida would reject it.
    assert nyc_violations[0].code == "99"

    fl_text = "Violation Code: 99\nViolation Description: Not a real FL category\n"
    _, fl_violations, _, _ = extract_florida_candidates((_block(fl_text),))
    assert fl_violations[0].code is None  # Florida DOES reject out-of-range codes


# --------------------------------------------------------------------------- #
# Corrections updated for every newly supported field                       #
# --------------------------------------------------------------------------- #


def _nyc_draft() -> object:
    from datetime import UTC, datetime

    from plateproof.documents.models import ExtractionDraft, OcrEngineInfo, UploadMetadata

    return ExtractionDraft(
        draft_id="d1",
        jurisdiction_expected="nyc",
        jurisdiction_detected="nyc",
        jurisdiction_mismatch=False,
        restaurant_id="nyc:1",
        restaurant_identity_corroborated=True,
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=10, page_count=1),
        pages=(),
        candidates={},
        violations=(),
        ambiguities=(),
        missing_fields=("camis",),
        warnings=(),
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )


def _florida_draft() -> object:
    from datetime import UTC, datetime

    from plateproof.documents.models import ExtractionDraft, OcrEngineInfo, UploadMetadata

    return ExtractionDraft(
        draft_id="d2",
        jurisdiction_expected="florida",
        jurisdiction_detected="florida",
        jurisdiction_mismatch=False,
        restaurant_id="florida:1",
        restaurant_identity_corroborated=True,
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=10, page_count=1),
        pages=(),
        candidates={},
        violations=(),
        ambiguities=(),
        missing_fields=("license_number",),
        warnings=(),
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )


def test_correction_accepted_for_nyc_camis() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("camis", "41234567", _nyc_draft())
    assert reason is None
    assert validated.parse_status == "parsed"
    assert validated.parsed_value == "41234567"


def test_correction_accepted_for_florida_visit_sequence() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("visit_sequence", "2", _florida_draft())
    assert reason is None
    assert validated.parsed_value == 2.0


def test_correction_accepted_for_florida_correction_deadline() -> None:
    from plateproof.documents.corrections import validate_correction

    validated, reason = validate_correction("correction_deadline", "04/15/2024", _florida_draft())
    assert reason is None
    assert validated.parsed_value.isoformat() == "2024-04-15"
