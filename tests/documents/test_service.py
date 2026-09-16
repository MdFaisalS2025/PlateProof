"""Tests for the one shared, worker-backed extract_document(...) entry point."""

from __future__ import annotations

from typing import Any


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _echo_success_worker(conn: Any) -> None:
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
                        "width_px": 200,
                        "height_px": 200,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [
                            {
                                "text": (
                                    "Restaurant Name: Joe's Pizza\n"
                                    "Inspection Date: 01/15/2024\nScore: 14"
                                ),
                                "source": "embedded_text",
                                "ocr_confidence": None,
                                "bounding_box": None,
                            }
                        ],
                    }
                ],
            },
        )


def _timeout_worker(conn: Any) -> None:
    import time

    time.sleep(600)


def test_extract_document_returns_completed_draft_on_success() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_echo_success_worker)
    try:
        draft = extract_document(
            _minimal_pdf("Score: 14"),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "completed"
        assert draft.candidates["score"].value == 14.0
    finally:
        pool.shutdown()


def test_extract_document_rejects_invalid_bytes_before_touching_pool() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    calls: list[int] = []

    def _counting_worker(conn: Any) -> None:
        calls.append(1)
        _echo_success_worker(conn)

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_counting_worker)
    try:
        draft = extract_document(
            b"not a real document",
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is None
        assert calls == []
    finally:
        pool.shutdown()


def test_extract_document_returns_failed_draft_on_worker_timeout() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, total_timeout_seconds=1.0, page_timeout_seconds=1.0),
        worker_main=_timeout_worker,
    )
    try:
        draft = extract_document(
            _minimal_pdf("x"),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.processing_status == "failed"
        assert draft.confirmable is False
    finally:
        pool.shutdown()


def test_extract_document_never_imports_native_parsers_directly() -> None:
    """service.py must route everything through the worker pool -- never
    call pdf.py/images.py/ocr/* itself."""
    import plateproof.documents.service as module

    with open(module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "import plateproof.documents.pdf" not in source
    assert "from plateproof.documents.pdf" not in source
    assert "import plateproof.documents.images" not in source
    assert "from plateproof.documents.images" not in source
    assert "plateproof.documents.ocr" not in source.replace("rapidocr_engine", "")
