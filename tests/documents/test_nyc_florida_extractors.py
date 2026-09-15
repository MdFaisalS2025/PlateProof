"""Tests for deterministic NYC/Florida candidate extraction from validated
text blocks. No target/model/risk-score logic from Task 6 is touched here --
these extractors only build reviewable candidates from document text."""

from __future__ import annotations


def _block(text: str, page: int = 1) -> object:
    from plateproof.documents.models import TextBlock

    return TextBlock(
        page=page, text=text, bounding_box=None, source="embedded_text", ocr_confidence=None
    )


def test_nyc_extractor_finds_score_date_grade_and_name() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (
        _block("Restaurant Name: Joe's Pizza\nInspection Date: 01/15/2024\nScore: 14\nGrade: A"),
    )
    candidates, ambiguities, missing = extract_nyc_candidates(blocks)
    assert candidates["score"].value == 14.0
    assert candidates["grade"].value == "A"
    assert candidates["restaurant_name"].value == "Joe's Pizza"
    assert ambiguities == ()
    assert missing == ()


def test_nyc_extractor_reports_missing_required_fields() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (_block("Some unrelated inspection paperwork with no recognizable labels"),)
    candidates, ambiguities, missing = extract_nyc_candidates(blocks)
    assert "score" in missing
    assert "inspection_date" in missing
    assert "restaurant_name" in missing


def test_nyc_extractor_reports_ambiguous_conflicting_score() -> None:
    from plateproof.documents.nyc_extractor import extract_nyc_candidates

    blocks = (_block("Score: 14"), _block("Score: 28", page=2))
    candidates, ambiguities, missing = extract_nyc_candidates(blocks)
    assert "score" not in candidates
    assert any(a.field_name == "score" for a in ambiguities)


def test_nyc_feature_scope_never_touches_florida_fields() -> None:
    from plateproof.documents.nyc_extractor import NYC_LABELS

    for field_name in NYC_LABELS:
        assert not field_name.startswith("fl_")


def test_florida_extractor_finds_disposition_and_counts() -> None:
    from plateproof.documents.florida_extractor import extract_florida_candidates

    blocks = (
        _block(
            "Restaurant Name: Sunshine Cafe\n"
            "Inspection Date: 03/02/2024\n"
            "High Priority: 2\n"
            "Intermediate: 1\n"
            "Basic: 3\n"
            "Disposition: Warning Issued"
        ),
    )
    candidates, ambiguities, missing = extract_florida_candidates(blocks)
    assert candidates["high_priority_count"].value == 2.0
    assert candidates["intermediate_count"].value == 1.0
    assert candidates["basic_count"].value == 3.0
    assert candidates["disposition_status"].value == "follow_up_required"
    assert missing == ()


def test_florida_extractor_unrecognized_disposition_text_is_missing_not_guessed() -> None:
    from plateproof.documents.florida_extractor import extract_florida_candidates

    blocks = (_block("Disposition: Some Unrecognized Free Text Phrase"),)
    candidates, ambiguities, missing = extract_florida_candidates(blocks)
    assert "disposition_status" not in candidates
    assert "disposition_status" in missing


def test_florida_feature_scope_never_touches_nyc_fields() -> None:
    from plateproof.documents.florida_extractor import FLORIDA_LABELS

    for field_name in FLORIDA_LABELS:
        assert not field_name.startswith("nyc_")
