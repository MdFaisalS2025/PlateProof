"""Second independent review, Finding 2: end-to-end proof that a document
whose cumulative extracted text exceeds the configured limit is never
reported as a successful, completed extraction -- through the real worker
entrypoint (embedded text and OCR text) and through the full
``extract_document`` service pipeline against a real spawned worker
process.
"""

from __future__ import annotations

import threading
from multiprocessing import Pipe
from typing import Any

from plateproof.documents.worker.protocol import (
    WorkerJobRequest,
    recv_frame,
    send_bytes_frame,
    send_frame,
    validate_worker_message,
)


def _minimal_pdf(text: str, *, page_count: int = 1) -> bytes:
    from tests.documents.test_pdf import _minimal_pdf as _multi_page_pdf

    return _multi_page_pdf(text, page_count=page_count)


def _pdf_request(**overrides: Any) -> WorkerJobRequest:
    defaults: dict[str, Any] = dict(
        expected_jurisdiction="nyc",
        media_type="application/pdf",
        max_pages=5,
        max_pixels=50_000_000,
        ocr_enabled=False,
        page_timeout_seconds=10.0,
    )
    defaults.update(overrides)
    return WorkerJobRequest(**defaults)


def _run_job_in_thread(request: WorkerJobRequest, document_bytes: bytes) -> list[Any]:
    from plateproof.documents.worker.entrypoint import worker_main

    parent_conn, child_conn = Pipe(duplex=True)
    thread = threading.Thread(target=worker_main, args=(child_conn,), daemon=True)
    thread.start()
    try:
        send_frame(
            parent_conn,
            {
                "protocol_version": 1,
                "message_type": "job_request",
                "expected_jurisdiction": request.expected_jurisdiction,
                "media_type": request.media_type,
                "max_pages": request.max_pages,
                "max_pixels": request.max_pixels,
                "ocr_enabled": request.ocr_enabled,
                "page_timeout_seconds": request.page_timeout_seconds,
            },
        )
        send_bytes_frame(parent_conn, document_bytes)
        messages = []
        while True:
            if not parent_conn.poll(15):
                raise AssertionError("worker did not respond within the test timeout")
            raw = recv_frame(parent_conn)
            validated = validate_worker_message(raw)
            messages.append(validated)
            if type(validated).__name__ in ("WorkerJobResponse", "WorkerJobError"):
                break
        return messages
    finally:
        parent_conn.close()
        thread.join(timeout=5)


def test_worker_entrypoint_reports_text_limit_exceeded_not_a_partial_response(
    monkeypatch: Any,
) -> None:
    """A multi-page PDF whose page-1 embedded text alone exceeds a tiny
    configured limit must produce a job_error with the closed
    ``text_limit_exceeded`` error kind -- never a job_response containing
    only the pages processed before the limit was hit."""
    from plateproof.documents.worker import entrypoint
    from plateproof.documents.worker.protocol import WorkerJobError, WorkerJobResponse

    monkeypatch.setattr(entrypoint, "DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT", 5)

    data = _minimal_pdf("Restaurant Name: Joe's Pizza, Score: 14", page_count=3)
    messages = _run_job_in_thread(_pdf_request(), data)
    response = messages[-1]

    assert not isinstance(response, WorkerJobResponse)
    assert isinstance(response, WorkerJobError)
    assert response.error_kind == "text_limit_exceeded"


def test_worker_entrypoint_ocr_text_also_triggers_the_limit(monkeypatch: Any) -> None:
    """A scanned page (no embedded text) whose OCR-extracted text alone
    exceeds a tiny configured limit must also produce a job_error -- proving
    OCR text cannot bypass the same cumulative budget as embedded text.
    OCR itself is mocked to a fixed, large result so the assertion is
    deterministic rather than dependent on real OCR model output."""
    from plateproof.documents.ocr.base import OcrOutcome, OcrResult, OcrTextBlock
    from plateproof.documents.worker import entrypoint
    from plateproof.documents.worker.protocol import WorkerJobError, WorkerJobResponse

    monkeypatch.setattr(entrypoint, "DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT", 5)
    monkeypatch.setattr(
        entrypoint,
        "run_ocr",
        lambda rgb_bytes, *, width, height: OcrResult(
            outcome=OcrOutcome.SUCCESS,
            blocks=(
                OcrTextBlock(text="x" * 1_000, confidence=0.9, x0=0.0, y0=0.0, x1=1.0, y1=1.0),
            ),
            unavailable_reason=None,
        ),
    )

    # Blank content stream -- no embedded text at all, forcing OCR to be
    # attempted and to be the sole source of extracted text.
    data = _minimal_pdf(" ", page_count=1)
    request = _pdf_request(ocr_enabled=True)
    messages = _run_job_in_thread(request, data)
    response = messages[-1]

    assert not isinstance(response, WorkerJobResponse)
    assert isinstance(response, WorkerJobError)
    assert response.error_kind == "text_limit_exceeded"


def _text_limit_worker(conn: Any) -> None:
    """Module-level (picklable) worker_main wrapper that patches the
    document text budget down to a tiny value *inside the spawned worker
    process itself* before running the real entrypoint loop -- proving the
    full extract_document(...) -> WorkerPool -> real worker -> real pdf.py
    pipeline never returns a completed draft for an over-budget document."""
    from plateproof.documents.worker import entrypoint

    entrypoint.DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT = 5
    entrypoint.worker_main(conn)


def test_extract_document_returns_non_confirmable_failed_draft_over_text_limit() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    data = _minimal_pdf("Restaurant Name: Joe's Pizza, Score: 14", page_count=3)
    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, page_timeout_seconds=20.0, total_timeout_seconds=40.0),
        worker_main=_text_limit_worker,
    )
    try:
        draft = extract_document(
            data,
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        # Never a partially-processed document presented as completed.
        assert draft.processing_status == "failed"
        assert draft.confirmable is False
        assert draft.pages == ()
        assert any(w.code == "text_limit_exceeded" for w in draft.warnings)
    finally:
        pool.shutdown()
