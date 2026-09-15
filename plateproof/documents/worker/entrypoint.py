"""The worker process main loop -- the ONLY place ``pdf.py``, ``images.py``,
and ``ocr/*`` are ever called. Loops, handling one job after another on the
same connection until the connection closes or a protocol violation occurs,
matching the pool's "long-lived, recyclable worker" design.

Never logs raw document bytes, OCR text, extracted values, filenames, or
paths. A processing failure is reported as a ``job_error`` with a closed
``error_kind`` only -- never the underlying exception's text.
"""

from __future__ import annotations

from multiprocessing.connection import Connection
from typing import Any

from plateproof.documents.images import ImageProcessingError, load_image_page
from plateproof.documents.ocr.rapidocr_engine import is_available as ocr_is_available
from plateproof.documents.ocr.rapidocr_engine import run_ocr
from plateproof.documents.pdf import PdfProcessingError, load_pdf_pages
from plateproof.documents.worker.protocol import (
    MAX_TEXT_BLOCK_LENGTH,
    MAX_TEXT_BLOCKS_PER_PAGE,
    PROTOCOL_VERSION,
    ProtocolViolationError,
    WorkerJobRequest,
    recv_bytes_frame,
    recv_frame,
    send_frame,
    validate_worker_message,
)

#: Defense-in-depth ceiling on the raw document frame this worker will ever
#: accept, independent of whatever operational upload limit the caller (API
#: or Streamlit) already enforced before submitting the job.
MAX_DOCUMENT_FRAME_BYTES = 25 * 1024 * 1024


def _text_block_payload(
    *, text: str, source: str, ocr_confidence: float | None, bounding_box: dict[str, float] | None
) -> dict[str, Any]:
    return {
        "text": text[:MAX_TEXT_BLOCK_LENGTH],
        "source": source,
        "ocr_confidence": ocr_confidence,
        "bounding_box": bounding_box,
    }


def _process_pdf(
    request: WorkerJobRequest, document_bytes: bytes, conn: Connection
) -> list[dict[str, Any]]:
    pdf_pages = load_pdf_pages(
        document_bytes, max_pages=request.max_pages, max_pixels=request.max_pixels
    )
    pages: list[dict[str, Any]] = []
    for page in pdf_pages:
        text_blocks: list[dict[str, Any]] = []
        used_ocr = False
        embedded = page.embedded_text.strip()
        if embedded:
            text_blocks.append(
                _text_block_payload(
                    text=page.embedded_text,
                    source="embedded_text",
                    ocr_confidence=None,
                    bounding_box=None,
                )
            )
        elif request.ocr_enabled:
            ocr_result = run_ocr(page.rgb_bytes, width=page.width_px, height=page.height_px)
            if ocr_result.outcome.value == "success":
                used_ocr = True
                for block in ocr_result.blocks:
                    text_blocks.append(
                        _text_block_payload(
                            text=block.text,
                            source="ocr",
                            ocr_confidence=block.confidence,
                            bounding_box={
                                "x0": block.x0,
                                "y0": block.y0,
                                "x1": block.x1,
                                "y1": block.y1,
                            },
                        )
                    )
        pages.append(
            {
                "page_number": page.page_number,
                "width_px": page.width_px,
                "height_px": page.height_px,
                "used_ocr": used_ocr,
                "text_blocks": text_blocks[:MAX_TEXT_BLOCKS_PER_PAGE],
            }
        )
        try:
            send_frame(
                conn,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "page_progress",
                    "page_number": page.page_number,
                },
            )
        except ProtocolViolationError:
            raise
    return pages


def _process_image(request: WorkerJobRequest, document_bytes: bytes) -> list[dict[str, Any]]:
    image_page = load_image_page(document_bytes, max_pixels=request.max_pixels)
    text_blocks: list[dict[str, Any]] = []
    used_ocr = False
    if request.ocr_enabled:
        ocr_result = run_ocr(
            image_page.rgb_bytes, width=image_page.width_px, height=image_page.height_px
        )
        if ocr_result.outcome.value == "success":
            used_ocr = True
            for block in ocr_result.blocks:
                text_blocks.append(
                    _text_block_payload(
                        text=block.text,
                        source="ocr",
                        ocr_confidence=block.confidence,
                        bounding_box={
                            "x0": block.x0,
                            "y0": block.y0,
                            "x1": block.x1,
                            "y1": block.y1,
                        },
                    )
                )
    return [
        {
            "page_number": 1,
            "width_px": image_page.width_px,
            "height_px": image_page.height_px,
            "used_ocr": used_ocr,
            "text_blocks": text_blocks[:MAX_TEXT_BLOCKS_PER_PAGE],
        }
    ]


def _handle_one_job(
    request: WorkerJobRequest, document_bytes: bytes, conn: Connection
) -> dict[str, Any]:
    try:
        if request.media_type == "application/pdf":
            pages = _process_pdf(request, document_bytes, conn)
        else:
            pages = _process_image(request, document_bytes)
    except (PdfProcessingError, ImageProcessingError) as exc:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "job_error",
            "error_kind": exc.args[0],
        }
    except Exception:  # noqa: BLE001 - any other failure fails closed with a static, generic reason
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "job_error",
            "error_kind": "internal_error",
        }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "job_response",
        "pages": pages,
        "ocr_available": ocr_is_available(),
    }


def worker_main(conn: Connection) -> None:
    """Receive one job, process it, respond, and loop back for the next job
    on the same connection -- until the connection closes or a protocol
    violation occurs, at which point this function simply returns and the
    process exits."""
    while True:
        try:
            raw = recv_frame(conn)
            request = validate_worker_message(raw)
        except ProtocolViolationError:
            return
        if not isinstance(request, WorkerJobRequest):
            return

        try:
            document_bytes = recv_bytes_frame(conn, max_length=MAX_DOCUMENT_FRAME_BYTES)
        except ProtocolViolationError:
            return

        response = _handle_one_job(request, document_bytes, conn)
        try:
            send_frame(conn, response)
        except ProtocolViolationError:
            return
