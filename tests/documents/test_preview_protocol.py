"""Finding 7: bounded, worker-validated previews. Previews are generated
only inside the worker, transported as separately-bounded binary frames
(never JSON/base64), and re-validated on the parent side (signature,
dimensions, byte length, page range, uniqueness, and total budget) before
ever being trusted.
"""

from __future__ import annotations

from io import BytesIO
from multiprocessing import Pipe

import pytest
from PIL import Image


def _tiny_png(width: int = 20, height: int = 10) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_preview_header_round_trips() -> None:
    from plateproof.documents.worker.protocol import (
        WorkerPreviewHeader,
        recv_frame,
        send_frame,
        validate_worker_message,
    )

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(
            parent_conn,
            {
                "protocol_version": 1,
                "message_type": "preview_header",
                "page_number": 1,
                "byte_length": 1234,
            },
        )
        raw = recv_frame(child_conn)
        validated = validate_worker_message(raw)
        assert isinstance(validated, WorkerPreviewHeader)
        assert validated.page_number == 1
        assert validated.byte_length == 1234
    finally:
        parent_conn.close()
        child_conn.close()


def test_preview_header_rejects_non_positive_page_number() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message(
            {
                "protocol_version": 1,
                "message_type": "preview_header",
                "page_number": 0,
                "byte_length": 10,
            }
        )


def test_preview_header_rejects_byte_length_above_absolute_ceiling() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_FRAME_BYTES
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message(
            {
                "protocol_version": 1,
                "message_type": "preview_header",
                "page_number": 1,
                "byte_length": ABSOLUTE_MAX_PREVIEW_FRAME_BYTES + 1,
            }
        )


def test_job_response_carries_preview_count() -> None:
    from plateproof.documents.worker.protocol import validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "ocr_available": False,
        "preview_count": 2,
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
    }
    response = validate_job_response(raw)
    assert response.preview_count == 2


def test_job_response_preview_count_above_absolute_max_pages_rejected() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PREVIEW_PAGES
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "ocr_available": False,
        "preview_count": ABSOLUTE_MAX_PREVIEW_PAGES + 1,
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
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)
