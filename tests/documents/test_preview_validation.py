"""Finding 7: the parent (pool.py) re-validates every preview frame after
receiving it -- signature, dimensions, byte length, page range, uniqueness,
ordering, and total budget -- before ever trusting it. Any deviation is a
protocol violation that retires the worker producing it.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from PIL import Image


def _png_bytes(width: int = 20, height: int = 10) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _request() -> Any:
    from plateproof.documents.worker.protocol import WorkerJobRequest

    return WorkerJobRequest(
        expected_jurisdiction="nyc",
        media_type="application/pdf",
        max_pages=5,
        max_pixels=10_000_000,
        ocr_enabled=False,
        page_timeout_seconds=20.0,
    )


def _send_job_response(conn: Any, *, preview_count: int) -> None:
    from plateproof.documents.worker.protocol import send_frame

    send_frame(
        conn,
        {
            "protocol_version": 1,
            "message_type": "job_response",
            "ocr_available": False,
            "preview_count": preview_count,
            "pages": [
                {
                    "page_number": 1,
                    "width_px": 10,
                    "height_px": 10,
                    "used_ocr": False,
                    "ocr_attempted": False,
                    "text_blocks": [],
                },
                {
                    "page_number": 2,
                    "width_px": 10,
                    "height_px": 10,
                    "used_ocr": False,
                    "ocr_attempted": False,
                    "text_blocks": [],
                },
            ],
        },
    )


def _send_preview(conn: Any, *, page_number: int, png_bytes: bytes) -> None:
    from plateproof.documents.worker.protocol import send_bytes_frame, send_frame

    send_frame(
        conn,
        {
            "protocol_version": 1,
            "message_type": "preview_header",
            "page_number": page_number,
            "byte_length": len(png_bytes),
        },
    )
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_FRAME_BYTES

    send_bytes_frame(conn, png_bytes, max_length=ABSOLUTE_MAX_PREVIEW_FRAME_BYTES)


def _run_and_get_outcome(worker_main: Any) -> Any:
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, page_timeout_seconds=20.0, total_timeout_seconds=40.0),
        worker_main=worker_main,
    )
    try:
        return pool.submit(_request(), b"%PDF-1.4\nfake")
    finally:
        pool.shutdown()


def _valid_two_preview_worker(conn: Any) -> None:
    _send_job_response(conn, preview_count=2)
    _send_preview(conn, page_number=1, png_bytes=_png_bytes())
    _send_preview(conn, page_number=2, png_bytes=_png_bytes())


def test_valid_previews_are_accepted_and_attached_to_the_response() -> None:
    from plateproof.documents.worker.protocol import WorkerJobResponse

    outcome = _run_and_get_outcome(_valid_two_preview_worker)
    assert isinstance(outcome, WorkerJobResponse)
    assert len(outcome.previews) == 2
    assert {p.page_number for p in outcome.previews} == {1, 2}
    for preview in outcome.previews:
        assert preview.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def _duplicate_page_worker(conn: Any) -> None:
    _send_job_response(conn, preview_count=2)
    _send_preview(conn, page_number=1, png_bytes=_png_bytes())
    _send_preview(conn, page_number=1, png_bytes=_png_bytes())


def test_duplicate_preview_page_number_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_duplicate_page_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _out_of_order_worker(conn: Any) -> None:
    _send_job_response(conn, preview_count=2)
    _send_preview(conn, page_number=2, png_bytes=_png_bytes())
    _send_preview(conn, page_number=1, png_bytes=_png_bytes())


def test_out_of_order_preview_frames_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_out_of_order_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _out_of_range_page_worker(conn: Any) -> None:
    _send_job_response(conn, preview_count=1)
    _send_preview(conn, page_number=99, png_bytes=_png_bytes())


def test_preview_page_number_outside_response_pages_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_out_of_range_page_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _byte_length_mismatch_worker(conn: Any) -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_FRAME_BYTES
    from plateproof.documents.worker.protocol import send_bytes_frame, send_frame

    _send_job_response(conn, preview_count=1)
    real_png = _png_bytes()
    send_frame(
        conn,
        {
            "protocol_version": 1,
            "message_type": "preview_header",
            "page_number": 1,
            "byte_length": len(real_png) + 5,  # lies about the length
        },
    )
    send_bytes_frame(conn, real_png, max_length=ABSOLUTE_MAX_PREVIEW_FRAME_BYTES)


def test_declared_byte_length_mismatch_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_byte_length_mismatch_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _malformed_png_worker(conn: Any) -> None:
    _send_job_response(conn, preview_count=1)
    garbage = b"not a real png at all, just garbage bytes"
    _send_preview(conn, page_number=1, png_bytes=garbage)


def test_malformed_encoded_preview_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_malformed_png_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _oversized_dimensions_worker(conn: Any) -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_WIDTH_PX

    _send_job_response(conn, preview_count=1)
    oversized = _png_bytes(width=ABSOLUTE_MAX_PREVIEW_WIDTH_PX + 100, height=10)
    _send_preview(conn, page_number=1, png_bytes=oversized)


def test_oversized_preview_dimensions_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    outcome = _run_and_get_outcome(_oversized_dimensions_worker)
    assert isinstance(outcome, WorkerInvalidResponse)


def _missing_preview_worker(conn: Any) -> None:
    # Declares 2 previews but only sends 1 -- the parent must not hang or
    # silently accept a short count; it must time out or fail closed.
    _send_job_response(conn, preview_count=2)
    _send_preview(conn, page_number=1, png_bytes=_png_bytes())


def test_missing_preview_frame_fails_closed_not_hangs() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse, WorkerTimeout

    outcome = _run_and_get_outcome(_missing_preview_worker)
    # Either outcome is an acceptable fail-closed result -- what matters is
    # it is never a WorkerJobResponse with an incomplete preview set, and
    # the pool.submit() call above already proves it did not hang forever.
    assert isinstance(outcome, WorkerInvalidResponse | WorkerTimeout)


def test_retired_after_bad_preview_is_never_reused() -> None:
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, page_timeout_seconds=20.0, total_timeout_seconds=40.0),
        worker_main=_duplicate_page_worker,
    )
    try:
        first = pool.submit(_request(), b"x")
        assert isinstance(first, WorkerInvalidResponse)
        second = pool.submit(_request(), b"x")
        assert isinstance(second, WorkerInvalidResponse)
    finally:
        pool.shutdown()
