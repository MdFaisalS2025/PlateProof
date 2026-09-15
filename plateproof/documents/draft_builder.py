"""Stateless assembly of an ExtractionDraft from validated worker output.

Pure function of its inputs: no filesystem, database, or network access,
and no reference to Task 6 models, the Task 8 graph, or Copilot evidence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from plateproof.documents.florida_extractor import extract_florida_candidates
from plateproof.documents.jurisdiction_detection import (
    check_jurisdiction_mismatch,
    detect_jurisdiction,
)
from plateproof.documents.models import (
    BoundingBox,
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
    pages = tuple(
        PageMetadata(
            page_number=p.page_number,
            width_px=p.width_px,
            height_px=p.height_px,
            used_ocr=p.used_ocr,
        )
        for p in response.pages
    )

    detected = detect_jurisdiction(blocks)
    mismatch = check_jurisdiction_mismatch(expected_jurisdiction, detected)

    if expected_jurisdiction == "nyc":
        candidates, ambiguities, missing = extract_nyc_candidates(blocks)
    else:
        candidates, ambiguities, missing = extract_florida_candidates(blocks)

    name_candidate = candidates.get("restaurant_name")
    candidate_name_text = name_candidate.display_value if name_candidate is not None else None
    corroborated = corroborate_restaurant_identity(candidate_name_text, expected_restaurant_name)

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

    confirmable = not mismatch and corroborated and not ambiguities and not missing

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
        ambiguities=ambiguities,
        missing_fields=missing,
        warnings=tuple(warnings),
        ocr_engine=ocr_engine,
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=confirmable,
    )
