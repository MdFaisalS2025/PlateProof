"""Projects a Task 9A ``ExtractionDraft`` into the public
``DocumentExtractionResponse`` contract -- the ONE place a document
extraction result is translated for the HTTP boundary.

Never exposes: raw upload bytes, a filesystem path, a native-parser
exception message, or full OCR text (only the same short, bounded evidence
excerpt already produced inside ``plateproof.documents.field_extraction``).
A worker-generated preview is base64-encoded here purely so it can travel
inside a JSON body -- it is never re-derived from the original upload.

This module never imports ``plateproof.documents.pdf``,
``plateproof.documents.images``, or any ``plateproof.documents.ocr``
submodule -- see ``tests/documents/test_task9b_boundary.py``.
"""

from __future__ import annotations

import base64
from typing import Any

from plateproof.api.schemas import (
    AmbiguityPublic,
    BoundingBoxPublic,
    CandidatePublic,
    DocumentExtractionResponse,
    DocumentWarningPublic,
    EvidenceSpanPublic,
    OcrEngineInfoPublic,
    PageMetadataPublic,
    ViolationRowPublic,
)
from plateproof.documents.models import (
    Ambiguity,
    BoundingBox,
    Candidate,
    EvidenceSpan,
    ExtractionDraft,
    OcrEngineInfo,
    PageMetadata,
    ViolationRowCandidate,
)
from plateproof.documents.models import (
    Warning as DocumentWarning,
)
from plateproof.serving.display import USER_SUBMITTED_RECORD_DISCLAIMER


def _to_public_bounding_box(box: BoundingBox | None) -> BoundingBoxPublic | None:
    if box is None:
        return None
    return BoundingBoxPublic(x0=box.x0, y0=box.y0, x1=box.x1, y1=box.y1)


def _to_public_evidence(evidence: tuple[EvidenceSpan, ...]) -> list[EvidenceSpanPublic]:
    return [
        EvidenceSpanPublic(
            page=span.page,
            excerpt=span.excerpt,
            bounding_box=_to_public_bounding_box(span.bounding_box),
            source=span.source,
        )
        for span in evidence
    ]


def _to_public_candidate(candidate: Candidate[Any]) -> CandidatePublic:
    return CandidatePublic(
        field_name=candidate.field_name,
        value=candidate.value,
        display_value=candidate.display_value,
        evidence=_to_public_evidence(candidate.evidence),
        confidence_label=candidate.confidence_label,
    )


def _to_public_violation(violation: ViolationRowCandidate) -> ViolationRowPublic:
    return ViolationRowPublic(
        raw_code_text=violation.raw_code_text,
        code=violation.code,
        description=violation.description,
        critical=violation.critical,
        evidence=_to_public_evidence(violation.evidence),
        confidence_label=violation.confidence_label,
    )


def _to_public_ambiguity(ambiguity: Ambiguity) -> AmbiguityPublic:
    return AmbiguityPublic(
        field_name=ambiguity.field_name,
        reason=ambiguity.reason,
        candidate_values=list(ambiguity.candidate_values),
    )


def _to_public_warning(warning: DocumentWarning) -> DocumentWarningPublic:
    return DocumentWarningPublic(code=warning.code, message=warning.message)


def _to_public_page(page: PageMetadata) -> PageMetadataPublic:
    preview_b64 = base64.b64encode(page.preview_png).decode("ascii") if page.preview_png else None
    return PageMetadataPublic(
        page_number=page.page_number,
        width_px=page.width_px,
        height_px=page.height_px,
        used_ocr=page.used_ocr,
        preview_png_base64=preview_b64,
    )


def _to_public_ocr_engine(engine: OcrEngineInfo) -> OcrEngineInfoPublic:
    return OcrEngineInfoPublic(
        engine_name=engine.engine_name,
        available=engine.available,
        unavailable_reason=engine.unavailable_reason,
    )


def project_draft(draft: ExtractionDraft) -> DocumentExtractionResponse:
    return DocumentExtractionResponse(
        draft_id=draft.draft_id,
        jurisdiction_expected=draft.jurisdiction_expected,
        jurisdiction_detected=draft.jurisdiction_detected,
        jurisdiction_mismatch=draft.jurisdiction_mismatch,
        restaurant_id=draft.restaurant_id,
        restaurant_identity_corroborated=draft.restaurant_identity_corroborated,
        upload_media_type=draft.upload.detected_media_type,
        upload_byte_size=draft.upload.byte_size,
        upload_page_count=draft.upload.page_count,
        pages=[_to_public_page(p) for p in draft.pages],
        candidates={
            name: _to_public_candidate(candidate) for name, candidate in draft.candidates.items()
        },
        violations=[_to_public_violation(v) for v in draft.violations],
        ambiguities=[_to_public_ambiguity(a) for a in draft.ambiguities],
        missing_fields=list(draft.missing_fields),
        warnings=[_to_public_warning(w) for w in draft.warnings],
        ocr_engine=_to_public_ocr_engine(draft.ocr_engine),
        processing_status=draft.processing_status,
        generated_at=draft.generated_at,
        confirmable=draft.confirmable,
        disclaimer=USER_SUBMITTED_RECORD_DISCLAIMER,
    )
