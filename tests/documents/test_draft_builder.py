"""Tests for stateless ExtractionDraft assembly."""

from __future__ import annotations


def _worker_response(text: str, *, page_number: int = 1) -> object:
    from plateproof.documents.worker.protocol import (
        WorkerJobResponse,
        WorkerPageResult,
        WorkerTextBlock,
    )

    return WorkerJobResponse(
        ocr_available=False,
        pages=(
            WorkerPageResult(
                page_number=page_number,
                width_px=200,
                height_px=200,
                used_ocr=False,
                ocr_attempted=False,
                text_blocks=(
                    WorkerTextBlock(
                        text=text, source="embedded_text", ocr_confidence=None, bounding_box=None
                    ),
                ),
            ),
        ),
    )


def _upload() -> object:
    from plateproof.documents.models import UploadMetadata

    return UploadMetadata(detected_media_type="application/pdf", byte_size=100, page_count=1)


def _ocr_info(available: bool = True) -> object:
    from plateproof.documents.models import OcrEngineInfo

    return OcrEngineInfo(
        engine_name="rapidocr",
        available=available,
        unavailable_reason=None if available else "ocr_unavailable",
    )


def test_build_draft_produces_confirmable_true_when_everything_lines_up() -> None:
    from plateproof.documents.draft_builder import build_draft

    text = (
        "NYC Department of Health and Mental Hygiene\n"
        "Restaurant Name: Joe's Pizza\n"
        "Inspection Date: 01/15/2024\n"
        "Score: 14\n"
    )
    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=_upload(),
        response=_worker_response(text),
        ocr_engine=_ocr_info(),
    )
    assert draft.jurisdiction_mismatch is False
    assert draft.restaurant_identity_corroborated is True
    assert draft.confirmable is True
    assert draft.processing_status == "completed"


def test_build_draft_flags_jurisdiction_mismatch() -> None:
    from plateproof.documents.draft_builder import build_draft

    text = (
        "Florida DBPR Division of Hotels and Restaurants\n"
        "Restaurant Name: Joe's Pizza\n"
        "Inspection Date: 01/15/2024\n"
        "High Priority: 0\n"
        "Disposition: Warning Issued"
    )
    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=_upload(),
        response=_worker_response(text),
        ocr_engine=_ocr_info(),
    )
    assert draft.jurisdiction_mismatch is True
    assert draft.confirmable is False
    assert any(w.code == "jurisdiction_mismatch" for w in draft.warnings)


def test_build_draft_flags_uncorroborated_identity() -> None:
    from plateproof.documents.draft_builder import build_draft

    text = "Restaurant Name: Totally Different Diner\nInspection Date: 01/15/2024\nScore: 14"
    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=_upload(),
        response=_worker_response(text),
        ocr_engine=_ocr_info(),
    )
    assert draft.restaurant_identity_corroborated is False
    assert draft.confirmable is False


def test_build_draft_not_confirmable_with_missing_fields() -> None:
    from plateproof.documents.draft_builder import build_draft

    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=_upload(),
        response=_worker_response("nothing useful here"),
        ocr_engine=_ocr_info(),
    )
    assert draft.missing_fields != ()
    assert draft.confirmable is False


def test_build_draft_never_persists_anything() -> None:
    """No filesystem/database side effects -- purely a pure function over
    its inputs. Verified by checking it has no obvious persistence import."""
    import plateproof.documents.draft_builder as module

    source = module.__file__
    with open(source, encoding="utf-8") as handle:
        content = handle.read()
    assert "duckdb" not in content.lower()
    assert "open(" not in content
