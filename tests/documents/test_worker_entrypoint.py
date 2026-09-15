"""Tests for the worker main loop (plateproof.documents.worker.entrypoint).

Run in-process against a real multiprocessing Pipe pair (no actual spawn)
for speed -- process-boundary behavior itself is covered by
test_worker_pool.py. This module is the ONLY place pdf.py/images.py/ocr/*
are ever called, so these tests exercise the real parsers end to end.
"""

from __future__ import annotations

import threading
from io import BytesIO
from multiprocessing import Pipe
from typing import Any

from PIL import Image, ImageDraw

from plateproof.documents.worker.protocol import (
    WorkerJobRequest,
    recv_frame,
    send_bytes_frame,
    send_frame,
    validate_worker_message,
)


def _minimal_pdf(text: str, *, page_count: int = 1) -> bytes:
    if page_count == 1:
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
            b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\n"
            b"startxref\n0\n%%EOF\n"
        )
    from tests.documents.test_pdf import _minimal_pdf as _multi_page_pdf

    return _multi_page_pdf(text, page_count=page_count)


def _png_with_text(text: str) -> bytes:
    image = Image.new("RGB", (300, 80), "white")
    ImageDraw.Draw(image).text((10, 25), text, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _run_job(request: WorkerJobRequest, document_bytes: bytes) -> Any:
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


def test_embedded_text_pdf_produces_job_response_without_ocr() -> None:
    from plateproof.documents.worker.protocol import WorkerJobResponse

    messages = _run_job(_pdf_request(), _minimal_pdf("Score: 14"))
    response = messages[-1]
    assert isinstance(response, WorkerJobResponse)
    assert response.pages[0].used_ocr is False
    joined = " ".join(b.text for b in response.pages[0].text_blocks)
    assert "Score: 14" in joined


def test_image_without_embedded_text_uses_ocr_when_enabled() -> None:
    from plateproof.documents.worker.protocol import WorkerJobResponse

    request = _pdf_request(media_type="image/png", ocr_enabled=True)
    messages = _run_job(request, _png_with_text("Restaurant Test Diner"))
    response = messages[-1]
    assert isinstance(response, WorkerJobResponse)
    assert response.pages[0].used_ocr is True
    joined = " ".join(b.text for b in response.pages[0].text_blocks)
    assert "Restaurant" in joined or "Diner" in joined


def test_image_without_ocr_enabled_produces_no_text_blocks() -> None:
    from plateproof.documents.worker.protocol import WorkerJobResponse

    request = _pdf_request(media_type="image/png", ocr_enabled=False)
    messages = _run_job(request, _png_with_text("Restaurant Test Diner"))
    response = messages[-1]
    assert isinstance(response, WorkerJobResponse)
    assert response.pages[0].used_ocr is False
    assert response.pages[0].text_blocks == ()


def test_malformed_pdf_produces_job_error_with_closed_error_kind() -> None:
    from plateproof.documents.worker.protocol import WorkerJobError

    messages = _run_job(_pdf_request(), b"not a real pdf")
    response = messages[-1]
    assert isinstance(response, WorkerJobError)
    assert response.error_kind == "pdf_malformed"


def test_page_limit_exceeded_produces_job_error() -> None:
    from plateproof.documents.worker.protocol import WorkerJobError

    request = _pdf_request(max_pages=1)
    messages = _run_job(request, _minimal_pdf("x", page_count=3))
    response = messages[-1]
    assert isinstance(response, WorkerJobError)
    assert response.error_kind == "page_limit_exceeded"


def test_ocr_unavailable_falls_back_gracefully_for_image_without_embedded_text() -> None:
    """When OCR construction fails, a document that needs OCR must not crash
    the worker -- it must still return a completed job_response with no text
    blocks and used_ocr=False, letting the caller report ocr_unavailable."""
    from plateproof.documents.ocr import rapidocr_engine
    from plateproof.documents.worker.protocol import WorkerJobResponse

    rapidocr_engine._reset_for_testing()
    try:
        with __import__("pytest").MonkeyPatch.context() as mp:
            mp.setattr(
                rapidocr_engine,
                "_construct_engine",
                lambda: (_ for _ in ()).throw(RuntimeError("x")),
            )
            request = _pdf_request(media_type="image/png", ocr_enabled=True)
            messages = _run_job(request, _png_with_text("Restaurant Test Diner"))
        response = messages[-1]
        assert isinstance(response, WorkerJobResponse)
        assert response.pages[0].used_ocr is False
    finally:
        rapidocr_engine._reset_for_testing()


def test_worker_handles_second_job_on_same_connection() -> None:
    """Proves the recyclable-worker design: one worker_main call serves
    multiple sequential jobs on the same connection."""
    from plateproof.documents.worker.entrypoint import worker_main
    from plateproof.documents.worker.protocol import WorkerJobResponse

    parent_conn, child_conn = Pipe(duplex=True)
    thread = threading.Thread(target=worker_main, args=(child_conn,), daemon=True)
    thread.start()
    try:
        for _ in range(2):
            request = _pdf_request()
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
            send_bytes_frame(parent_conn, _minimal_pdf("Score: 20"))
            while True:
                raw = recv_frame(parent_conn)
                validated = validate_worker_message(raw)
                if isinstance(validated, WorkerJobResponse):
                    break
    finally:
        parent_conn.close()
        thread.join(timeout=5)
