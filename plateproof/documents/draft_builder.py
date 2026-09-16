"""Stateless assembly of an ExtractionDraft from validated worker output.

Pure function of its inputs: no filesystem, database, or network access,
and no reference to Task 6 models, the Task 8 graph, or Copilot evidence.
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Literal

from plateproof.documents.field_extraction import classify_confidence
from plateproof.documents.florida_extractor import extract_florida_candidates
from plateproof.documents.jurisdiction_detection import (
    check_jurisdiction_mismatch,
    detect_jurisdiction,
)
from plateproof.documents.models import (
    BoundingBox,
    Candidate,
    ExtractionDraft,
    Jurisdiction,
    OcrEngineInfo,
    PageMetadata,
    TextBlock,
    UploadMetadata,
)
from plateproof.documents.models import Warning as DocumentWarning
from plateproof.documents.nyc_extractor import extract_nyc_candidates
from plateproof.documents.restaurant_matching import corroborate_restaurant_identity
from plateproof.documents.worker.protocol import WorkerJobResponse


def _text_blocks_from_response(response: WorkerJobResponse) -> tuple[TextBlock, ...]:
    blocks: list[TextBlock] = []
    for page in response.pages:
        for block in page.text_blocks:
            bounding_box = None
            if block.bounding_box is not None:
                bounding_box = BoundingBox(
                    x0=block.bounding_box.x0,
                    y0=block.bounding_box.y0,
                    x1=block.bounding_box.x1,
                    y1=block.bounding_box.y1,
                )
            blocks.append(
                TextBlock(
                    page=page.page_number,
                    text=block.text,
                    bounding_box=bounding_box,
                    source=block.source,
                    ocr_confidence=block.ocr_confidence,
                )
            )
    return tuple(blocks)


def _recompute_confidence_label(candidate: Candidate[Any], *, corroborated: bool) -> Candidate[Any]:
    source = candidate.evidence[0].source if candidate.evidence else "embedded_text"
    return replace(
        candidate, confidence_label=classify_confidence(source=source, corroborated=corroborated)
    )


def build_draft(
    *,
    expected_jurisdiction: Jurisdiction,
    restaurant_id: str,
    expected_restaurant_name: str,
    upload: UploadMetadata,
    response: WorkerJobResponse,
    ocr_engine: OcrEngineInfo,
) -> ExtractionDraft:
    blocks = _text_blocks_from_response(response)
    preview_by_page = {preview.page_number: preview.png_bytes for preview in response.previews}
    pages = tuple(
        PageMetadata(
            page_number=p.page_number,
            width_px=p.width_px,
            height_px=p.height_px,
            used_ocr=p.used_ocr,
            preview_png=preview_by_page.get(p.page_number),
        )
        for p in response.pages
    )

    detected = detect_jurisdiction(blocks)
    mismatch = check_jurisdiction_mismatch(expected_jurisdiction, detected)

    # Finding 5: a page that had no usable embedded text and needed OCR,
    # but didn't get usable text from it (engine unavailable or a
    # per-page OCR failure), means this document is not genuinely
    # "completed" -- it is reported ocr_unavailable, never silently
    # folded into "no fields found."
    ocr_needed_but_missing = any(p.ocr_attempted and not p.used_ocr for p in response.pages)

    if expected_jurisdiction == "nyc":
        candidates, violations, ambiguities, missing = extract_nyc_candidates(blocks)
    else:
        candidates, violations, ambiguities, missing = extract_florida_candidates(blocks)

    name_candidate = candidates.get("restaurant_name")
    candidate_name_text = name_candidate.display_value if name_candidate is not None else None
    corroborated = corroborate_restaurant_identity(candidate_name_text, expected_restaurant_name)

    if name_candidate is not None:
        # Finding 8: corroboration is only known *after* extraction, so the
        # restaurant_name candidate's confidence label is recomputed here
        # once the corroboration result is available -- never left at its
        # pre-corroboration default.
        candidates = {
            **candidates,
            "restaurant_name": _recompute_confidence_label(
                name_candidate, corroborated=corroborated
            ),
        }

    warnings: list[DocumentWarning] = []
    if mismatch:
        warnings.append(
            DocumentWarning(
                code="jurisdiction_mismatch",
                message="The document appears to describe a different jurisdiction than expected.",
            )
        )
    if not corroborated:
        warnings.append(
            DocumentWarning(
                code="restaurant_identity_not_corroborated",
                message=(
                    "The document's restaurant name could not be confirmed "
                    "against the selected restaurant."
                ),
            )
        )
    if ocr_needed_but_missing:
        warnings.append(
            DocumentWarning(
                code="ocr_unavailable",
                message=(
                    "This document appears to need OCR to read its text, but OCR is "
                    "not available right now. Try a document with selectable text instead."
                ),
            )
        )

    has_unresolved_violation_codes = any(v.code is None for v in violations)

    processing_status: Literal["completed", "ocr_unavailable", "failed"] = (
        "ocr_unavailable" if ocr_needed_but_missing else "completed"
    )
    confirmable = (
        not mismatch
        and corroborated
        and not ambiguities
        and not missing
        and not ocr_needed_but_missing
        and not has_unresolved_violation_codes
    )

    return ExtractionDraft(
        draft_id=str(uuid.uuid4()),
        jurisdiction_expected=expected_jurisdiction,
        jurisdiction_detected=detected,
        jurisdiction_mismatch=mismatch,
        restaurant_id=restaurant_id,
        restaurant_identity_corroborated=corroborated,
        upload=upload,
        pages=pages,
        candidates=candidates,
        violations=violations,
        ambiguities=ambiguities,
        missing_fields=missing,
        warnings=tuple(warnings),
        ocr_engine=ocr_engine,
        processing_status=processing_status,
        generated_at=datetime.now(UTC),
        confirmable=confirmable,
    )
