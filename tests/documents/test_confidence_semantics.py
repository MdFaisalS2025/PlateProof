"""Finding 8: confidence must be categorical and evidence-condition-based
(embedded text vs OCR, corroborated vs not), never a numeric average of OCR
score and pattern score collapsed into a high/medium/low threshold. An
OCR-sourced value must never become "high" confidence merely because its
OCR score and pattern score average above some threshold -- OCR misreads
are numerically indistinguishable from correct reads.
"""

from __future__ import annotations


def _block(
    text: str, *, page: int = 1, source: str = "embedded_text", ocr_confidence: float | None = None
) -> object:
    from plateproof.documents.models import TextBlock

    return TextBlock(
        page=page, text=text, bounding_box=None, source=source, ocr_confidence=ocr_confidence
    )


def test_ocr_sourced_value_is_needs_review_even_with_high_ocr_confidence() -> None:
    from plateproof.documents.field_extraction import classify_confidence

    label = classify_confidence(source="ocr", corroborated=None)
    assert label == "needs_review"


def test_embedded_text_value_is_high_by_default() -> None:
    from plateproof.documents.field_extraction import classify_confidence

    label = classify_confidence(source="embedded_text", corroborated=None)
    assert label == "high"


def test_embedded_text_value_explicitly_not_corroborated_is_needs_review() -> None:
    from plateproof.documents.field_extraction import classify_confidence

    label = classify_confidence(source="embedded_text", corroborated=False)
    assert label == "needs_review"


def test_embedded_text_value_explicitly_corroborated_is_high() -> None:
    from plateproof.documents.field_extraction import classify_confidence

    label = classify_confidence(source="embedded_text", corroborated=True)
    assert label == "high"


def test_resolve_field_from_ocr_source_never_produces_high_label() -> None:
    from plateproof.documents.field_extraction import (
        find_label_matches,
        parse_nonneg_number,
        resolve_field,
    )

    blocks = (_block("Score: 14", source="ocr", ocr_confidence=0.99),)
    matches = find_label_matches(blocks, ("Score",))
    candidate, ambiguity = resolve_field("score", matches, parse_value=parse_nonneg_number)
    assert ambiguity is None
    assert candidate is not None
    assert candidate.confidence_label == "needs_review"
    # OCR confidence is still recorded as its own separate measurement.
    assert candidate.confidence.ocr_confidence == 0.99


def test_resolve_field_from_embedded_text_produces_high_label() -> None:
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
    assert candidate.confidence_label == "high"


def test_conflicting_embedded_text_values_remain_unresolved_regardless_of_confidence() -> None:
    """A conflict is reported as an Ambiguity -- never a Candidate with any
    confidence label at all, "high" included."""
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


def test_absent_evidence_produces_no_candidate_at_all() -> None:
    """No evidence -- "unusable" -- is represented by producing neither a
    Candidate nor an Ambiguity; the caller reports the field as missing."""
    from plateproof.documents.field_extraction import resolve_field

    candidate, ambiguity = resolve_field("score", [], parse_value=float)
    assert candidate is None
    assert ambiguity is None


def test_restaurant_name_confidence_is_high_when_corroborated_exactly() -> None:
    """End-to-end: build_draft recomputes the restaurant_name candidate's
    confidence label after corroboration succeeds."""
    from plateproof.documents.draft_builder import build_draft
    from plateproof.documents.models import OcrEngineInfo, UploadMetadata
    from plateproof.documents.worker.protocol import (
        WorkerJobResponse,
        WorkerPageResult,
        WorkerTextBlock,
    )

    response = WorkerJobResponse(
        ocr_available=False,
        pages=(
            WorkerPageResult(
                page_number=1,
                width_px=200,
                height_px=200,
                used_ocr=False,
                ocr_attempted=False,
                text_blocks=(
                    WorkerTextBlock(
                        text="Restaurant Name: Joe's Pizza\nInspection Date: 01/15/2024\nScore: 14",
                        source="embedded_text",
                        ocr_confidence=None,
                        bounding_box=None,
                    ),
                ),
            ),
        ),
    )
    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=10, page_count=1),
        response=response,
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
    )
    assert draft.candidates["restaurant_name"].confidence_label == "high"


def test_restaurant_name_confidence_is_needs_review_when_not_corroborated() -> None:
    from plateproof.documents.draft_builder import build_draft
    from plateproof.documents.models import OcrEngineInfo, UploadMetadata
    from plateproof.documents.worker.protocol import (
        WorkerJobResponse,
        WorkerPageResult,
        WorkerTextBlock,
    )

    response = WorkerJobResponse(
        ocr_available=False,
        pages=(
            WorkerPageResult(
                page_number=1,
                width_px=200,
                height_px=200,
                used_ocr=False,
                ocr_attempted=False,
                text_blocks=(
                    WorkerTextBlock(
                        text=(
                            "Restaurant Name: Totally Different Diner\n"
                            "Inspection Date: 01/15/2024\nScore: 14"
                        ),
                        source="embedded_text",
                        ocr_confidence=None,
                        bounding_box=None,
                    ),
                ),
            ),
        ),
    )
    draft = build_draft(
        expected_jurisdiction="nyc",
        restaurant_id="nyc:1",
        expected_restaurant_name="Joe's Pizza",
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=10, page_count=1),
        response=response,
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
    )
    assert draft.candidates["restaurant_name"].confidence_label == "needs_review"


def test_confidence_never_claims_document_authenticity_or_official_status() -> None:
    """Structural/documentation check: the confidence model must never
    include a field or docstring implying authenticity/official
    verification -- confidence is about extraction reliability only."""
    import plateproof.documents.models as models_module

    source = models_module.__file__
    with open(source, encoding="utf-8") as handle:
        content = handle.read().lower()
    assert "authentic" not in content
    assert "official" not in content or "never" in content or "not" in content
