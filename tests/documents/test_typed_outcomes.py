"""Finding 5: typed processing outcomes must not be collapsed.

A valid worker job_error was being converted to a generic
WorkerInvalidResponse, losing its closed error_kind. OCR-unavailable
image/scanned-PDF results were also being reported as
processing_status="completed" merely because the worker produced a
structurally valid response with no text.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def _job_error_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_error",
                "error_kind": "encrypted_document",
            },
        )


def test_pool_preserves_worker_job_error_as_a_typed_outcome_not_generic_invalid() -> None:
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerJobError, WorkerJobRequest

    request = WorkerJobRequest(
        expected_jurisdiction="nyc",
        media_type="application/pdf",
        max_pages=5,
        max_pixels=10_000_000,
        ocr_enabled=False,
        page_timeout_seconds=10.0,
    )
    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_job_error_worker)
    try:
        outcome = pool.submit(request, b"%PDF-1.4\nfake")
        assert isinstance(outcome, WorkerJobError)
        assert outcome.error_kind == "encrypted_document"
    finally:
        pool.shutdown()


def _png_bytes(text_free: bool = True) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (100, 50), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_encrypted_document_job_error_surfaces_a_specific_draft_warning() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_job_error_worker)
    try:
        draft = extract_document(
            b"%PDF-1.4\nfake",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "failed"
        assert any(w.code == "encrypted_document" for w in draft.warnings)
        assert draft.confirmable is False
    finally:
        pool.shutdown()


def _text_pdf_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [
                            {
                                "text": "Score: 14",
                                "source": "embedded_text",
                                "ocr_confidence": None,
                                "bounding_box": None,
                            }
                        ],
                    }
                ],
            },
        )


def _scanned_needs_ocr_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": True,
                        "text_blocks": [],
                    }
                ],
            },
        )


def test_text_pdf_remains_completed_when_ocr_unavailable_but_unnecessary() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_text_pdf_worker)
    try:
        draft = extract_document(
            b"%PDF-1.4\nfake",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "completed"
    finally:
        pool.shutdown()


def test_scanned_document_needing_unavailable_ocr_is_ocr_unavailable_not_completed() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_scanned_needs_ocr_worker)
    try:
        draft = extract_document(
            _png_bytes(),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "ocr_unavailable"
        assert draft.confirmable is False
    finally:
        pool.shutdown()


def _hang_worker(conn: Any) -> None:
    import time

    time.sleep(600)


def test_worker_timeout_outcome_maps_to_failed_with_specific_warning() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, total_timeout_seconds=1.0, page_timeout_seconds=1.0),
        worker_main=_hang_worker,
    )
    try:
        draft = extract_document(
            b"%PDF-1.4\nfake",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "failed"
        assert any(w.code == "worker_timeout" for w in draft.warnings)
    finally:
        pool.shutdown()


def _crash_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import recv_bytes_frame, recv_frame

    recv_frame(conn)
    recv_bytes_frame(conn, max_length=64_000_000)
    raise RuntimeError("simulated crash")


def test_worker_crash_outcome_maps_to_failed_with_specific_warning() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_crash_worker)
    try:
        draft = extract_document(
            b"%PDF-1.4\nfake",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "failed"
        assert any(w.code == "worker_crashed" for w in draft.warnings)
    finally:
        pool.shutdown()


def _crash_with_sensitive_message_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import recv_bytes_frame, recv_frame

    recv_frame(conn)
    recv_bytes_frame(conn, max_length=64_000_000)
    raise RuntimeError("sensitive/path/leak C:\\Users\\secret\\file.pdf")


def test_no_warning_ever_contains_a_raw_exception_message() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1), worker_main=_crash_with_sensitive_message_worker
    )
    try:
        draft = extract_document(
            b"%PDF-1.4\nfake",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        for warning in draft.warnings:
            assert "sensitive" not in warning.message
            assert "secret" not in warning.message
    finally:
        pool.shutdown()
