"""Finding 3: JSON control frames, raw document frames, and preview frames
use three distinct, non-conflated size ceilings. Before this fix,
send_bytes_frame() used the small 8 MB MAX_WORKER_FRAME_BYTES ceiling for
every binary frame, including the document itself -- rejecting an
otherwise-accepted 8-15 MB upload before it ever reached the worker.

Large payloads are received on a background thread concurrently with the
send -- a real OS pipe's buffer is far smaller than several MB, so sending
and receiving several-MB payloads sequentially on one thread deadlocks
(the writer blocks waiting for the reader to drain the pipe, and nothing is
reading yet). This mirrors how the real parent/child processes actually
operate: concurrently, never one fully blocking on the other.
"""

from __future__ import annotations

import threading
from multiprocessing import Pipe

import pytest


def _recv_in_background(conn: object, max_length: int, result: list[object]) -> threading.Thread:
    from plateproof.documents.worker.protocol import recv_bytes_frame

    def _run() -> None:
        result.append(recv_bytes_frame(conn, max_length=max_length))  # type: ignore[arg-type]

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def test_send_bytes_frame_accepts_an_explicit_larger_max_length() -> None:
    from plateproof.documents.worker.protocol import MAX_JSON_FRAME_BYTES, send_bytes_frame

    payload = b"x" * (MAX_JSON_FRAME_BYTES + 1_000_000)  # bigger than the JSON ceiling
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        result: list[object] = []
        thread = _recv_in_background(child_conn, len(payload), result)
        send_bytes_frame(parent_conn, payload, max_length=len(payload))
        thread.join(timeout=30)
        assert result == [payload]
    finally:
        parent_conn.close()
        child_conn.close()


def test_send_bytes_frame_default_max_length_is_the_json_ceiling() -> None:
    """Backward-compatible default: callers that don't pass max_length keep
    the small JSON-frame ceiling (this is the safe default; document-byte
    callers must pass the larger ceiling explicitly). This raises before
    any bytes are ever written, so no receiver is needed."""
    from plateproof.documents.worker.protocol import (
        MAX_JSON_FRAME_BYTES,
        ProtocolViolationError,
        send_bytes_frame,
    )

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        oversized = b"x" * (MAX_JSON_FRAME_BYTES + 1)
        with pytest.raises(ProtocolViolationError):
            send_bytes_frame(parent_conn, oversized)
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.parametrize(
    "size",
    [
        8 * 1024 * 1024,  # exactly 8 MB -- at the JSON ceiling, but this is a document-frame send
        8 * 1024 * 1024 + 1,  # 8 MB + 1 byte -- must NOT be rejected by the old JSON ceiling
    ],
)
def test_document_frame_is_not_bounded_by_the_json_ceiling(size: int) -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_UPLOAD_BYTES
    from plateproof.documents.worker.protocol import send_bytes_frame

    payload = b"x" * size
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        result: list[object] = []
        thread = _recv_in_background(child_conn, ABSOLUTE_MAX_UPLOAD_BYTES, result)
        send_bytes_frame(parent_conn, payload, max_length=ABSOLUTE_MAX_UPLOAD_BYTES)
        thread.join(timeout=30)
        assert len(result) == 1
        assert len(result[0]) == size  # type: ignore[arg-type]
    finally:
        parent_conn.close()
        child_conn.close()


def test_at_operational_upload_limit_and_plus_one_byte() -> None:
    """Exactly the configured operational upload limit succeeds; the same
    limit + 1 byte is rejected -- proven directly against the receiver's
    max_length parameter (the authoritative, receiver-side ceiling)."""
    from plateproof.documents.worker.protocol import ProtocolViolationError, send_bytes_frame

    operational_limit = 1_000_000  # a representative Settings.documents_max_upload_bytes value

    at_limit = b"y" * operational_limit
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        result: list[object] = []
        thread = _recv_in_background(child_conn, operational_limit, result)
        send_bytes_frame(parent_conn, at_limit, max_length=operational_limit)
        thread.join(timeout=30)
        assert len(result) == 1 and len(result[0]) == operational_limit  # type: ignore[arg-type]
    finally:
        parent_conn.close()
        child_conn.close()

    over_limit = b"y" * (operational_limit + 1)
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        with pytest.raises(ProtocolViolationError):
            send_bytes_frame(parent_conn, over_limit, max_length=operational_limit)
    finally:
        parent_conn.close()
        child_conn.close()


def test_at_absolute_document_ceiling_and_plus_one_byte_declared_header() -> None:
    """A declared header length exactly at the absolute ceiling is
    accepted by the receiver's length check; one byte over is rejected
    before any oversized read is attempted -- exercised via a test-double
    connection so no multi-megabyte payload is actually transmitted."""
    import struct

    from plateproof.documents.limits import ABSOLUTE_MAX_UPLOAD_BYTES
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_bytes_frame

    class _HeaderOnlyConnection:
        def __init__(self, declared_length: int) -> None:
            self._header = struct.pack(">I", declared_length)
            self._first_call = True

        def recv_bytes(self, maxlength: int | None = None) -> bytes:
            if self._first_call:
                self._first_call = False
                return self._header
            raise AssertionError(f"payload recv_bytes called with maxlength={maxlength}")

    # Exactly at the ceiling: the receiver's length check must let the
    # second recv_bytes() call happen (it then fails for an unrelated
    # reason in this double, proving the ceiling check itself passed).
    conn_at_ceiling = _HeaderOnlyConnection(ABSOLUTE_MAX_UPLOAD_BYTES)
    with pytest.raises(AssertionError, match="payload recv_bytes called"):
        recv_bytes_frame(conn_at_ceiling, max_length=ABSOLUTE_MAX_UPLOAD_BYTES)  # type: ignore[arg-type]

    # One byte over: rejected before the second recv_bytes() call at all.
    conn_over_ceiling = _HeaderOnlyConnection(ABSOLUTE_MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ProtocolViolationError):
        recv_bytes_frame(conn_over_ceiling, max_length=ABSOLUTE_MAX_UPLOAD_BYTES)  # type: ignore[arg-type]


def _echo_size_worker(conn: object) -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_UPLOAD_BYTES
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)  # type: ignore[arg-type]
            document_bytes = recv_bytes_frame(conn, max_length=ABSOLUTE_MAX_UPLOAD_BYTES)  # type: ignore[arg-type]
        except ProtocolViolationError:
            return
        send_frame(
            conn,  # type: ignore[arg-type]
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": len(document_bytes) % 100000 + 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [],
                    }
                ],
            },
        )


def test_pool_sends_document_bytes_using_the_document_ceiling_not_json_ceiling() -> None:
    """End-to-end proof at the pool level, real separate OS processes (no
    same-thread deadlock risk here -- parent and child run concurrently):
    an 8MB+1-byte document, which would have been rejected outright by the
    old hardcoded 8 MB ceiling, round-trips successfully through a worker."""
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerJobRequest, WorkerJobResponse

    request = WorkerJobRequest(
        expected_jurisdiction="nyc",
        media_type="application/pdf",
        max_pages=5,
        max_pixels=10_000_000,
        ocr_enabled=False,
        page_timeout_seconds=20.0,
    )
    oversized_for_old_ceiling = b"0" * (8 * 1024 * 1024 + 1)  # 8MB + 1 byte
    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, page_timeout_seconds=30.0, total_timeout_seconds=60.0),
        worker_main=_echo_size_worker,
    )
    try:
        outcome = pool.submit(request, oversized_for_old_ceiling)
        assert isinstance(outcome, WorkerJobResponse)
    finally:
        pool.shutdown()
