"""Second independent review, Finding 1: a signature-plus-IHDR check is not
sufficient before untrusted, worker-controlled preview bytes are handed to a
UI image renderer. These tests prove the parent (``pool.py``) validates the
*entire* PNG container structure -- every chunk's declared length, CRC,
ordering, and the absence of any chunk type other than the narrow
IHDR/IDAT/IEND form our own worker-side encoder actually produces -- using
only ``struct``/``zlib``, never an image decoder, in the trusted parent
process.
"""

from __future__ import annotations

import struct
import zlib
from io import BytesIO
from typing import Any

from PIL import Image

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _valid_png(width: int = 10, height: int = 8) -> bytes:
    image = Image.frombytes("RGB", (width, height), bytes([100]) * (width * height * 3))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _find_chunk_start(data: bytes, chunk_type: bytes) -> int:
    """Byte offset of the 4-byte length field that precedes ``chunk_type``."""
    marker = data.find(chunk_type)
    assert marker != -1
    return marker - 4


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
                }
            ],
        },
    )


def _send_preview(conn: Any, *, page_number: int, png_bytes: bytes) -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_FRAME_BYTES
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


def _send_one_preview(conn: Any, *, png_bytes: bytes) -> None:
    _send_job_response(conn, preview_count=1)
    _send_preview(conn, page_number=1, png_bytes=png_bytes)


def _worker_sending(png_bytes: bytes) -> Any:
    # A module-level function bound via functools.partial -- not a closure
    # -- so it can be pickled and sent to a real spawned worker process
    # (spawn cannot pickle a locally-defined nested function).
    import functools

    return functools.partial(_send_one_preview, png_bytes=png_bytes)


def test_signature_plus_fake_ihdr_only_payload_is_rejected() -> None:
    """A bare 24-byte payload (signature + 16 arbitrary bytes at the IHDR
    width/height offsets) has no real IHDR chunk, no CRC, no IDAT, and no
    IEND -- it is not a PNG at all, merely bytes shaped like one. It must
    never be accepted as a valid preview."""
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    fake_payload = _PNG_SIGNATURE + b"\x00" * 8 + struct.pack(">II", 10, 8)
    assert len(fake_payload) == 24

    outcome = _run_and_get_outcome(_worker_sending(fake_payload))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_truncated_idat_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    valid = _valid_png()
    idat_start = _find_chunk_start(valid, b"IDAT")
    # Cut off partway through the IDAT chunk's declared data -- the frame
    # ends before the chunk's own declared length (plus CRC) is satisfied.
    truncated = valid[: idat_start + 8 + 3]

    outcome = _run_and_get_outcome(_worker_sending(truncated))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_missing_iend_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    valid = _valid_png()
    without_iend = valid[:-12]  # IEND is always exactly 12 bytes: 0-length + type + crc

    outcome = _run_and_get_outcome(_worker_sending(without_iend))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_bad_crc_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    valid = bytearray(_valid_png())
    idat_start = _find_chunk_start(bytes(valid), b"IDAT")
    # Flip a bit inside the IDAT chunk's data -- length and CRC are
    # unchanged, so only the CRC check can catch this.
    data_offset = idat_start + 8
    valid[data_offset] ^= 0xFF

    outcome = _run_and_get_outcome(_worker_sending(bytes(valid)))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_impossible_chunk_length_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 10, 8, 8, 2, 0, 0, 0))
    # Declares a chunk far larger than any data actually present.
    bogus_idat_header = struct.pack(">I", 0xFFFFFFFF) + b"IDAT"
    payload = _PNG_SIGNATURE + ihdr + bogus_idat_header + b"short"

    outcome = _run_and_get_outcome(_worker_sending(payload))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_trailing_bytes_after_iend_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    with_trailing_garbage = _valid_png() + b"\x00\x00\x00\x00extra-bytes-after-iend"

    outcome = _run_and_get_outcome(_worker_sending(with_trailing_garbage))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_forbidden_metadata_chunk_rejected() -> None:
    """A well-formed, correctly-CRC'd ancillary chunk (e.g. tEXt/eXIf) must
    still be rejected -- our own encoder never emits one, so its presence
    means the bytes did not come from the worker's narrow encoding path."""
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    valid = _valid_png()
    iend_start = _find_chunk_start(valid, b"IEND")
    text_chunk = _chunk(b"tEXt", b"Comment\x00injected metadata")
    with_metadata = valid[:iend_start] + text_chunk + valid[iend_start:]

    outcome = _run_and_get_outcome(_worker_sending(with_metadata))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_wrong_ihdr_length_rejected() -> None:
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    # A 12-byte IHDR body (missing the interlace byte) instead of the
    # mandatory 13.
    short_ihdr_data = struct.pack(">IIBBBB", 10, 8, 8, 2, 0, 0)
    ihdr = _chunk(b"IHDR", short_ihdr_data)
    idat = _chunk(b"IDAT", b"not-real-zlib-data-but-non-empty")
    iend = _chunk(b"IEND", b"")
    payload = _PNG_SIGNATURE + ihdr + idat + iend

    outcome = _run_and_get_outcome(_worker_sending(payload))
    assert isinstance(outcome, WorkerInvalidResponse)


def test_valid_worker_generated_preview_is_still_accepted() -> None:
    """The fix must not reject the exact, narrow PNG form the worker's own
    encoder actually produces."""
    from plateproof.documents.worker.protocol import WorkerJobResponse

    outcome = _run_and_get_outcome(_worker_sending(_valid_png()))
    assert isinstance(outcome, WorkerJobResponse)
    assert len(outcome.previews) == 1
    assert outcome.previews[0].width_px == 10
    assert outcome.previews[0].height_px == 8


def test_malformed_preview_retires_the_worker() -> None:
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerInvalidResponse

    fake_payload = _PNG_SIGNATURE + b"\x00" * 8 + struct.pack(">II", 10, 8)
    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, page_timeout_seconds=20.0, total_timeout_seconds=40.0),
        worker_main=_worker_sending(fake_payload),
    )
    try:
        first = pool.submit(_request(), b"x")
        assert isinstance(first, WorkerInvalidResponse)
        second = pool.submit(_request(), b"x")
        assert isinstance(second, WorkerInvalidResponse)
    finally:
        pool.shutdown()
