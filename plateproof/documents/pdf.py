"""pypdfium2 calls -- ONLY ever invoked from inside a worker process.

PDFium is documented as not thread-safe, even across different documents,
and executes no JavaScript. This module never runs in the calling API/UI
process; see ``plateproof.documents.worker`` for the process-isolation
boundary that enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass

import pypdfium2 as pdfium


class PdfProcessingError(Exception):
    """Raised with exactly one closed-enum reason string as its sole
    argument (matching ``plateproof.documents.worker.protocol``'s
    ``error_kind`` values): ``"pdf_malformed"``, ``"encrypted_document"``,
    ``"page_limit_exceeded"``, or ``"pixel_limit_exceeded"``. The underlying
    PDFium exception is chained for local debugging only -- it is never
    forwarded across the worker boundary."""


@dataclass(frozen=True, kw_only=True)
class PdfPageRender:
    page_number: int
    width_px: int
    height_px: int
    embedded_text: str
    rgb_bytes: bytes


def load_pdf_pages(
    data: bytes, *, max_pages: int, max_pixels: int, render_scale: float = 2.0
) -> list[PdfPageRender]:
    """Open ``data`` as a PDF and return one :class:`PdfPageRender` per page,
    with embedded text and a rendered RGB raster (for OCR fallback) bounded
    by ``max_pages``/``max_pixels``. Raises :class:`PdfProcessingError`
    for anything malformed, encrypted, or over a configured limit."""
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        message = str(exc).lower()
        if "password" in message:
            raise PdfProcessingError("encrypted_document") from exc
        raise PdfProcessingError("pdf_malformed") from exc

    try:
        page_count = len(doc)
        if page_count == 0:
            raise PdfProcessingError("pdf_malformed")
        if page_count > max_pages:
            raise PdfProcessingError("page_limit_exceeded")

        pages: list[PdfPageRender] = []
        for index in range(page_count):
            try:
                page = doc[index]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                width_pt, height_pt = page.get_size()
            except pdfium.PdfiumError as exc:
                raise PdfProcessingError("pdf_malformed") from exc

            width_px = max(1, int(width_pt * render_scale))
            height_px = max(1, int(height_pt * render_scale))
            if width_px * height_px > max_pixels:
                raise PdfProcessingError("pixel_limit_exceeded")

            try:
                bitmap = page.render(scale=render_scale)
                pil_image = bitmap.to_pil().convert("RGB")
            except pdfium.PdfiumError as exc:
                raise PdfProcessingError("pdf_malformed") from exc

            pages.append(
                PdfPageRender(
                    page_number=index + 1,
                    width_px=pil_image.width,
                    height_px=pil_image.height,
                    embedded_text=text,
                    rgb_bytes=pil_image.tobytes(),
                )
            )
        return pages
    finally:
        doc.close()
