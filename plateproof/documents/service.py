"""``extract_document(...)`` -- the ONE public entry point either FastAPI or
Streamlit may call (Task 9B). Every byte is validated here before it ever
reaches the worker pool, and every native parser call happens only inside
that pool's worker processes -- this module never imports ``pdf``,
``images``, or any ``ocr`` submodule itself, and never parses a document
directly. Stateless: nothing here is persisted, cached, or sent to Ollama
or any other network service.

Typed worker outcomes are preserved, never collapsed (Finding 5 of the
independent review of commit 735d4a3): a validated ``WorkerJobError``'s
closed ``error_kind`` becomes a specific, static-message draft warning; a
``WorkerTimeout``/``WorkerCrashed``/``WorkerInvalidResponse`` each get their
own specific warning code too. No warning ever includes a raw exception
message, a filename, or a filesystem path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from plateproof.documents.draft_builder import build_draft
from plateproof.documents.limits import (
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_PIXELS_PER_PAGE,
    DEFAULT_MAX_UPLOAD_BYTES,
)
from plateproof.documents.models import ExtractionDraft, Jurisdiction, OcrEngineInfo, UploadMetadata
from plateproof.documents.models import Warning as DocumentWarning
from plateproof.documents.validation import validate_upload_bytes
from plateproof.documents.worker.pool import DEFAULT_PAGE_TIMEOUT_SECONDS, WorkerPool
from plateproof.documents.worker.protocol import (
    WorkerBusy,
    WorkerCrashed,
    WorkerInvalidResponse,
    WorkerJobError,
    WorkerJobRequest,
    WorkerJobResponse,
    WorkerTimeout,
)

# Re-exported for backward-compatible import sites; the authoritative
# values now live in plateproof.documents.limits.
DEFAULT_MAX_PIXELS = DEFAULT_MAX_PIXELS_PER_PAGE


def _failed_draft(
    *,
    expected_jurisdiction: Jurisdiction,
    restaurant_id: str,
    upload: UploadMetadata,
    warning: DocumentWarning,
) -> ExtractionDraft:
    ocr_engine = OcrEngineInfo(
        engine_name="rapidocr", available=False, unavailable_reason=warning.code
    )
    return ExtractionDraft(
        draft_id=str(uuid.uuid4()),
        jurisdiction_expected=expected_jurisdiction,
        jurisdiction_detected=None,
        jurisdiction_mismatch=False,
        restaurant_id=restaurant_id,
        restaurant_identity_corroborated=False,
        upload=upload,
        pages=(),
        candidates={},
        violations=(),
        ambiguities=(),
        missing_fields=(),
        warnings=(warning,),
        ocr_engine=ocr_engine,
        processing_status="failed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )


#: Fixed, static, public-safe messages per closed error_kind (never the
#: underlying exception's own text). Anything not listed falls back to a
#: generic message -- the warning `code` itself (the closed enum value) is
#: always preserved regardless.
_ERROR_KIND_MESSAGES: dict[str, str] = {
    "encrypted_document": "This document is password-protected or encrypted and cannot be read.",
    "pdf_malformed": "This PDF could not be read. It may be corrupted or in an unsupported format.",
    "image_malformed": (
        "This image could not be read. It may be corrupted or in an unsupported format."
    ),
    "decode_failed": "This document could not be decoded.",
    "ocr_failed": "Text recognition failed for this document.",
    "ocr_unavailable": "Text recognition is not available right now.",
    "unsupported_media_type": "This file type is not supported.",
    "page_limit_exceeded": "This document has too many pages to process.",
    "pixel_limit_exceeded": "This document's page size exceeds the processing limit.",
    "text_limit_exceeded": "This document's text content exceeds the processing limit.",
    "internal_error": "This document could not be processed due to an internal error.",
}


def extract_document(
    data: bytes,
    *,
    expected_jurisdiction: Jurisdiction,
    restaurant_id: str,
    expected_restaurant_name: str,
    pool: WorkerPool,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_pixels: int = DEFAULT_MAX_PIXELS_PER_PAGE,
) -> ExtractionDraft | None:
    """Validate, submit to the worker pool, and assemble a draft.

    Returns ``None`` if the raw bytes fail basic signature/size validation
    (the caller decides how to surface that, e.g. a 413/422 at the API
    layer) -- the worker pool is never touched for input that couldn't
    possibly be a supported document. Otherwise always returns a draft
    (never raises): a completed/ocr_unavailable draft on a valid response,
    or a failed draft with a specific, sanitized warning for any worker
    timeout, crash, error, or invalid response.
    """
    validation = validate_upload_bytes(data, max_bytes=max_upload_bytes)
    if not validation.accepted or validation.detected_media_type is None:
        return None

    request = WorkerJobRequest(
        expected_jurisdiction=expected_jurisdiction,
        media_type=validation.detected_media_type,
        max_pages=max_pages,
        max_pixels=max_pixels,
        ocr_enabled=True,
        page_timeout_seconds=DEFAULT_PAGE_TIMEOUT_SECONDS,
    )
    outcome = pool.submit(request, data)

    upload = UploadMetadata(
        detected_media_type=validation.detected_media_type,
        byte_size=validation.byte_size,
        page_count=0,
    )

    if isinstance(outcome, WorkerJobResponse):
        if not outcome.pages:
            # An empty-but-structurally-valid response is not a legitimate
            # outcome (Finding 4) -- never silently become a "completed"
            # draft with zero pages.
            return _failed_draft(
                expected_jurisdiction=expected_jurisdiction,
                restaurant_id=restaurant_id,
                upload=upload,
                warning=DocumentWarning(
                    code="invalid_worker_response",
                    message="This document could not be processed due to an internal error.",
                ),
            )
        upload = UploadMetadata(
            detected_media_type=upload.detected_media_type,
            byte_size=upload.byte_size,
            page_count=len(outcome.pages),
        )
        ocr_engine = OcrEngineInfo(
            engine_name="rapidocr",
            available=outcome.ocr_available,
            unavailable_reason=None if outcome.ocr_available else "ocr_unavailable",
        )
        return build_draft(
            expected_jurisdiction=expected_jurisdiction,
            restaurant_id=restaurant_id,
            expected_restaurant_name=expected_restaurant_name,
            upload=upload,
            response=outcome,
            ocr_engine=ocr_engine,
        )

    if isinstance(outcome, WorkerJobError):
        message = _ERROR_KIND_MESSAGES.get(
            outcome.error_kind, "This document could not be processed."
        )
        warning = DocumentWarning(code=outcome.error_kind, message=message)
    elif isinstance(outcome, WorkerTimeout):
        warning = DocumentWarning(
            code="worker_timeout",
            message="Processing this document took too long. Please try again.",
        )
    elif isinstance(outcome, WorkerCrashed):
        warning = DocumentWarning(
            code="worker_crashed",
            message="This document could not be processed due to an internal error.",
        )
    elif isinstance(outcome, WorkerBusy):
        warning = DocumentWarning(
            code="worker_busy",
            message="The document processing service is busy right now. Please try again shortly.",
        )
    else:
        assert isinstance(outcome, WorkerInvalidResponse)
        warning = DocumentWarning(
            code="invalid_worker_response",
            message="This document could not be processed due to an internal error.",
        )

    return _failed_draft(
        expected_jurisdiction=expected_jurisdiction,
        restaurant_id=restaurant_id,
        upload=upload,
        warning=warning,
    )
