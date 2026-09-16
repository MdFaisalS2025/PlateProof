"""The worker process main loop -- the ONLY place ``pdf.py``, ``images.py``,
and ``ocr/*`` are ever called. Loops, handling one job after another on the
same connection until the connection closes or a protocol violation occurs,
matching the pool's "long-lived, recyclable worker" design.

Never logs raw document bytes, OCR text, extracted values, filenames, or
paths. A processing failure is reported as a ``job_error`` with a closed
``error_kind`` only -- never the underlying exception's text.

Bounded previews (Finding 7 of the independent review of commit 735d4a3)
are generated here, from already-decoded pixels, at a small fixed
resolution -- never a copy of the original uploaded bytes, so there is no
EXIF/metadata to strip in the first place. They are sent as separate,
smaller-bounded binary frames immediately after the job_response, one
``preview_header`` + binary pair per previewed page, in page order.
"""

from __future__ import annotations

from io import BytesIO
from multiprocessing.connection import Connection
from typing import Any

from PIL import Image

from plateproof.documents.images import ImageProcessingError, load_image_page
from plateproof.documents.limits import (
    ABSOLUTE_MAX_PREVIEW_FRAME_BYTES,
    ABSOLUTE_MAX_PREVIEW_PAGES,
    ABSOLUTE_MAX_UPLOAD_BYTES,
    DEFAULT_MAX_PREVIEW_HEIGHT_PX,
    DEFAULT_MAX_PREVIEW_WIDTH_PX,
    DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT,
)
from plateproof.documents.ocr.rapidocr_engine import is_available as ocr_is_available
from plateproof.documents.ocr.rapidocr_engine import run_ocr
from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages
from plateproof.documents.worker.protocol import (
    MAX_TEXT_BLOCK_LENGTH,
    MAX_TEXT_BLOCKS_PER_PAGE,
    PROTOCOL_VERSION,
    ProtocolViolationError,
    WorkerJobRequest,
    recv_bytes_frame,
    recv_frame,
    send_bytes_frame,
    send_frame,
    validate_worker_message,
)

#: The same absolute ceiling the parent (WorkerPool) uses when sending the
#: document frame (Finding 3: parent and child must enforce identical
#: limits -- a single shared constant, never two independently-chosen
#: numbers that could silently drift apart).
MAX_DOCUMENT_FRAME_BYTES = ABSOLUTE_MAX_UPLOAD_BYTES


def _text_block_payload(
    *, text: str, source: str, ocr_confidence: float | None, bounding_box: dict[str, float] | None
) -> dict[str, Any]:
    return {
        "text": text[:MAX_TEXT_BLOCK_LENGTH],
        "source": source,
        "ocr_confidence": ocr_confidence,
        "bounding_box": bounding_box,
    }


def _encode_preview_png(rgb_bytes: bytes, width_px: int, height_px: int) -> bytes | None:
    """Re-encodes a raw RGB buffer as PNG. Because this constructs a brand
    new image from decoded pixel data -- never copying the original
    uploaded file's bytes -- there is no EXIF or other file-level metadata
    to strip; none was ever carried over. Returns ``None`` (never raises)
    if the resulting frame would exceed the preview binary-frame ceiling,
    so a caller can simply skip that page's preview rather than fail the
    whole job over it."""
    try:
        image = Image.frombytes("RGB", (width_px, height_px), rgb_bytes)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
    except Exception:  # noqa: BLE001 - a preview is best-effort, never fatal to the job
        return None
    if len(png_bytes) > ABSOLUTE_MAX_PREVIEW_FRAME_BYTES:
        return None
    return png_bytes


def _preview_scale_for(width_px: int, height_px: int, *, base_scale: float) -> float:
    """The PDFium ``render(scale=...)`` factor that fits this page within
    the fixed preview dimensions, computed from its already-known
    full-resolution pixel size at ``base_scale``."""
    width_ratio = DEFAULT_MAX_PREVIEW_WIDTH_PX / width_px
    height_ratio = DEFAULT_MAX_PREVIEW_HEIGHT_PX / height_px
    return base_scale * min(width_ratio, height_ratio, 1.0)


def _downscale_rgb_to_preview(
    rgb_bytes: bytes, width_px: int, height_px: int
) -> tuple[bytes, int, int]:
    image = Image.frombytes("RGB", (width_px, height_px), rgb_bytes)
    image.thumbnail(
        (DEFAULT_MAX_PREVIEW_WIDTH_PX, DEFAULT_MAX_PREVIEW_HEIGHT_PX), Image.Resampling.LANCZOS
    )
    return image.tobytes(), image.width, image.height


def _process_pdf(
    request: WorkerJobRequest, document_bytes: bytes, conn: Connection
) -> tuple[list[dict[str, Any]], list[tuple[int, bytes]]]:
    pages: list[dict[str, Any]] = []
    previews: list[tuple[int, bytes]] = []

    def on_page(info: Any, render_rgb: Any) -> None:
        text_blocks: list[dict[str, Any]] = []
        used_ocr = False
        embedded = info.embedded_text.strip()
        ocr_attempted = (not embedded) and request.ocr_enabled
        if embedded:
            text_blocks.append(
                _text_block_payload(
                    text=info.embedded_text,
                    source="embedded_text",
                    ocr_confidence=None,
                    bounding_box=None,
                )
            )
        elif ocr_attempted:
            rgb_bytes, width_px, height_px = render_rgb()
            ocr_result = run_ocr(rgb_bytes, width=width_px, height=height_px)
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

        if len(previews) < ABSOLUTE_MAX_PREVIEW_PAGES:
            preview_scale = _preview_scale_for(info.width_px, info.height_px, base_scale=2.0)
            preview_rgb, preview_w, preview_h = render_rgb(scale=preview_scale)
            png_bytes = _encode_preview_png(preview_rgb, preview_w, preview_h)
            if png_bytes is not None:
                previews.append((info.page_number, png_bytes))

        pages.append(
            {
                "page_number": info.page_number,
                "width_px": info.width_px,
                "height_px": info.height_px,
                "used_ocr": used_ocr,
                "ocr_attempted": ocr_attempted,
                "text_blocks": text_blocks[:MAX_TEXT_BLOCKS_PER_PAGE],
            }
        )

    def on_page_complete(page_number: int) -> None:
        # Fired only after process_pdf_pages has already closed this
        # page's native PDFium resources -- progress is never signaled
        # while a page's resources are still open (Finding 2).
        try:
            send_frame(
                conn,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "page_progress",
                    "page_number": page_number,
                },
            )
        except ProtocolViolationError:
            raise

    process_pdf_pages(
        document_bytes,
        max_pages=request.max_pages,
        max_pixels=request.max_pixels,
        on_page=on_page,
        on_page_complete=on_page_complete,
        max_cumulative_text_bytes=DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT,
    )
    return pages, previews


def _process_image(
    request: WorkerJobRequest, document_bytes: bytes
) -> tuple[list[dict[str, Any]], list[tuple[int, bytes]]]:
    image_page = load_image_page(document_bytes, max_pixels=request.max_pixels)
    text_blocks: list[dict[str, Any]] = []
    used_ocr = False
    ocr_attempted = request.ocr_enabled  # an image has no embedded text concept
    if ocr_attempted:
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

    previews: list[tuple[int, bytes]] = []
    preview_rgb, preview_w, preview_h = _downscale_rgb_to_preview(
        image_page.rgb_bytes, image_page.width_px, image_page.height_px
    )
    png_bytes = _encode_preview_png(preview_rgb, preview_w, preview_h)
    if png_bytes is not None:
        previews.append((1, png_bytes))

    pages = [
        {
            "page_number": 1,
            "width_px": image_page.width_px,
            "height_px": image_page.height_px,
            "used_ocr": used_ocr,
            "ocr_attempted": ocr_attempted,
            "text_blocks": text_blocks[:MAX_TEXT_BLOCKS_PER_PAGE],
        }
    ]
    return pages, previews


def _handle_one_job(
    request: WorkerJobRequest, document_bytes: bytes, conn: Connection
) -> tuple[dict[str, Any], list[tuple[int, bytes]]]:
    try:
        if request.media_type == "application/pdf":
            pages, previews = _process_pdf(request, document_bytes, conn)
        else:
            pages, previews = _process_image(request, document_bytes)
    except (PdfProcessingError, ImageProcessingError) as exc:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "job_error",
            "error_kind": exc.args[0],
        }, []
    except Exception:  # noqa: BLE001 - any other failure fails closed with a static, generic reason
        return {
            "protocol_version": PROTOCOL_VERSION,
            "message_type": "job_error",
            "error_kind": "internal_error",
        }, []
    return {
        "protocol_version": PROTOCOL_VERSION,
        "message_type": "job_response",
        "pages": pages,
        "ocr_available": ocr_is_available(),
        "preview_count": len(previews),
    }, previews


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

        response, previews = _handle_one_job(request, document_bytes, conn)
        try:
            send_frame(conn, response)
            for page_number, png_bytes in previews:
                send_frame(
                    conn,
                    {
                        "protocol_version": PROTOCOL_VERSION,
                        "message_type": "preview_header",
                        "page_number": page_number,
                        "byte_length": len(png_bytes),
                    },
                )
                send_bytes_frame(conn, png_bytes, max_length=ABSOLUTE_MAX_PREVIEW_FRAME_BYTES)
        except ProtocolViolationError:
            return
