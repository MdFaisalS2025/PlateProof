"""``extract_document(...)`` -- the ONE public entry point either FastAPI or
Streamlit may call (Task 9B). Every byte is validated here before it ever
reaches the worker pool, and every native parser call happens only inside
that pool's worker processes -- this module never imports ``pdf``,
``images``, or any ``ocr`` submodule itself, and never parses a document
directly. Stateless: nothing here is persisted, cached, or sent to Ollama
or any other network service.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from plateproof.documents.draft_builder import build_draft
from plateproof.documents.models import ExtractionDraft, Jurisdiction, OcrEngineInfo, UploadMetadata
from plateproof.documents.models import Warning as DocumentWarning
from plateproof.documents.validation import validate_upload_bytes
from plateproof.documents.worker.pool import DEFAULT_PAGE_TIMEOUT_SECONDS, WorkerPool
from plateproof.documents.worker.protocol import WorkerJobRequest, WorkerJobResponse

DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024
DEFAULT_MAX_PAGES = 10
DEFAULT_MAX_PIXELS = 40_000_000


def _failed_draft(
    *,
    expected_jurisdiction: Jurisdiction,
    restaurant_id: str,
    upload: UploadMetadata,
    ocr_engine: OcrEngineInfo,
) -> ExtractionDraft:
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
        ambiguities=(),
        missing_fields=(),
        warnings=(
            DocumentWarning(
                code="processing_failed",
                message=(
                    "The document could not be processed. Please try again with a clearer scan."
                ),
            ),
        ),
        ocr_engine=ocr_engine,
        processing_status="failed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )


def extract_document(
    data: bytes,
    *,
    expected_jurisdiction: Jurisdiction,
    restaurant_id: str,
    expected_restaurant_name: str,
    pool: WorkerPool,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_pages: int = DEFAULT_MAX_PAGES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> ExtractionDraft | None:
    """Validate, submit to the worker pool, and assemble a draft.

    Returns ``None`` if the raw bytes fail basic signature/size validation
    (the caller decides how to surface that, e.g. a 413/422 at the API
    layer) -- the worker pool is never touched for input that couldn't
    possibly be a supported document. Returns a ``processing_status="failed"``
    draft (never raises) for a worker timeout, crash, or invalid response.
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

    # WorkerTimeout / WorkerCrashed / WorkerInvalidResponse -- a failed,
    # stateless draft, never a raised exception reaching the caller.
    ocr_engine = OcrEngineInfo(
        engine_name="rapidocr", available=False, unavailable_reason="processing_failed"
    )
    return _failed_draft(
        expected_jurisdiction=expected_jurisdiction,
        restaurant_id=restaurant_id,
        upload=upload,
        ocr_engine=ocr_engine,
    )
