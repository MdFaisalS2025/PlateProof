"""Tests for jurisdiction detection and restaurant-identity corroboration."""

from __future__ import annotations


def _block(text: str, page: int = 1) -> object:
    from plateproof.documents.models import TextBlock

    return TextBlock(
        page=page, text=text, bounding_box=None, source="embedded_text", ocr_confidence=None
    )


def test_detects_nyc_from_dohmh_marker() -> None:
    from plateproof.documents.jurisdiction_detection import detect_jurisdiction

    blocks = (_block("NYC Department of Health and Mental Hygiene\nRestaurant Inspection Report"),)
    assert detect_jurisdiction(blocks) == "nyc"


def test_detects_florida_from_dbpr_marker() -> None:
    from plateproof.documents.jurisdiction_detection import detect_jurisdiction

    blocks = (_block("Florida DBPR Division of Hotels and Restaurants Inspection Report"),)
    assert detect_jurisdiction(blocks) == "florida"


def test_returns_none_when_no_marker_found() -> None:
    from plateproof.documents.jurisdiction_detection import detect_jurisdiction

    blocks = (_block("Generic Restaurant Report with no jurisdiction markers"),)
    assert detect_jurisdiction(blocks) is None


def test_returns_none_when_both_markers_present_never_guesses() -> None:
    from plateproof.documents.jurisdiction_detection import detect_jurisdiction

    blocks = (_block("DOHMH and DBPR both mentioned here"),)
    assert detect_jurisdiction(blocks) is None


def test_mismatch_true_when_detected_differs_from_expected() -> None:
    from plateproof.documents.jurisdiction_detection import check_jurisdiction_mismatch

    assert check_jurisdiction_mismatch("nyc", "florida") is True
    assert check_jurisdiction_mismatch("nyc", "nyc") is False


def test_mismatch_false_when_detection_inconclusive() -> None:
    """No positive detection means we cannot confirm a mismatch -- never
    flag one on absence of evidence alone."""
    from plateproof.documents.jurisdiction_detection import check_jurisdiction_mismatch

    assert check_jurisdiction_mismatch("nyc", None) is False


def test_corroborates_matching_restaurant_name() -> None:
    from plateproof.documents.restaurant_matching import corroborate_restaurant_identity

    assert corroborate_restaurant_identity("Joe's Pizza LLC", "Joe's Pizza") is True


def test_does_not_corroborate_different_names() -> None:
    from plateproof.documents.restaurant_matching import corroborate_restaurant_identity

    assert corroborate_restaurant_identity("Totally Different Diner", "Joe's Pizza") is False


def test_does_not_corroborate_when_candidate_name_missing() -> None:
    from plateproof.documents.restaurant_matching import corroborate_restaurant_identity

    assert corroborate_restaurant_identity(None, "Joe's Pizza") is False
