"""pypdfium2 calls -- ONLY ever invoked from inside a worker process.

PDFium is documented as not thread-safe, even across different documents,
and executes no JavaScript. This module never runs in the calling API/UI
process; see ``plateproof.documents.worker`` for the process-isolation
boundary that enforces this.

Processing is incremental (Finding 5 of the independent review of commit
735d4a3): one page at a time, never retaining every page's raw RGB raster
simultaneously. Embedded text is extracted first; a full-resolution OCR
raster is rendered only when that page actually has no usable embedded
text and OCR is enabled. Every native PDFium/Pillow resource (page, text
page, bitmap) is closed deterministically -- via explicit ``close()`` calls
in ``finally`` blocks, never left to garbage collection -- before the next
page is opened.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw


class PdfProcessingError(Exception):
    """Raised with exactly one closed-enum reason string as its sole
    argument (matching ``plateproof.documents.worker.protocol``'s
    ``error_kind`` values): ``"pdf_malformed"``, ``"encrypted_document"``,
    ``"page_limit_exceeded"``, ``"pixel_limit_exceeded"``, or
    ``"text_limit_exceeded"``. The underlying PDFium exception is chained
    for local debugging only -- it is never forwarded across the worker
    boundary."""


@dataclass(frozen=True, kw_only=True)
class PdfPageInfo:
    page_number: int
    width_px: int
    height_px: int
    embedded_text: str


#: Renders this page to an RGB raster on demand, at ``render_scale`` unless
#: an explicit ``scale`` override is given (e.g. a much smaller scale for a
#: bounded preview render, independent of the OCR-resolution default). Must
#: be called (if at all) only from within the ``on_page`` callback for this
#: page -- the underlying PDFium page is closed as soon as that callback
#: returns, and calling this afterward would use an already-closed native
#: resource.
RenderRgb = Callable[..., tuple[bytes, int, int]]

#: May return the number of additional extracted-text bytes to count
#: toward the cumulative document text budget beyond ``embedded_text``
#: (e.g. OCR-derived text the caller extracted from a rendered raster,
#: which ``process_pdf_pages`` itself has no visibility into) -- returning
#: ``None`` counts as zero additional bytes. Without this, a scanned PDF
#: whose text comes entirely from OCR could bypass the cumulative text
#: limit, since embedded-text length alone would always read as zero.
OnPage = Callable[[PdfPageInfo, RenderRgb], int | None]
OnPageComplete = Callable[[int], None]


def _is_encrypted(doc: pdfium.PdfDocument) -> bool:
    """``FPDF_GetSecurityHandlerRevision`` returns a non-negative revision
    number for any document opened under a Standard Security Handler
    (including one with an empty user password, which pypdfium2 opens
    without ever raising a password-related error) and ``-1`` for a
    document with no security handler at all. This is the documented,
    reliable check -- unlike inspecting a raised exception's message text,
    it also catches a document that *opened successfully* but is still
    encrypted (e.g. empty user password, restricted owner permissions)."""
    return bool(pdfium_raw.FPDF_GetSecurityHandlerRevision(doc.raw) >= 0)


def process_pdf_pages(
    data: bytes,
    *,
    max_pages: int,
    max_pixels: int,
    on_page: OnPage,
    on_page_complete: OnPageComplete | None = None,
    max_cumulative_text_bytes: int | None = None,
    render_scale: float = 2.0,
) -> None:
    """Open ``data`` as a PDF and call ``on_page(info, render_rgb)`` once
    per page, in order, never holding more than one page's native
    resources or rendered raster at a time.

    ``render_rgb`` is a zero-argument callable the caller may invoke from
    within ``on_page`` to render *this* page to an RGB raster -- call it
    only if embedded text is unusable and OCR is actually needed. After
    ``on_page`` returns, this page's PDFium page/text-page/bitmap resources
    are closed immediately (before the next page is opened), and only then
    is ``on_page_complete(page_number)`` invoked, if provided -- so a
    caller using it to emit a progress signal never does so before this
    page's resources are already released.

    If ``max_cumulative_text_bytes`` is given, the running total of
    ``embedded_text`` lengths (plus whatever additional byte count
    ``on_page`` reports, e.g. OCR text) across processed pages is checked
    after every page. If including this page's text would exceed the
    limit, :class:`PdfProcessingError` (``"text_limit_exceeded"``) is
    raised immediately -- before this page's text is folded into the
    cumulative total, before ``on_page_complete`` is invoked for it, and
    before any further page is opened. This never returns normally with
    only a partial document processed: a caller must not be able to treat
    a truncated result as a completed extraction.

    Raises :class:`PdfProcessingError` for anything malformed, encrypted,
    or over a configured limit. The pixel-limit check uses the page's
    declared dimensions and happens *before* any render() call -- an
    oversized page never reaches raster allocation.
    """
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        message = str(exc).lower()
        if "password" in message:
            raise PdfProcessingError("encrypted_document") from exc
        raise PdfProcessingError("pdf_malformed") from exc

    try:
        if _is_encrypted(doc):
            raise PdfProcessingError("encrypted_document")

        page_count = len(doc)
        if page_count == 0:
            raise PdfProcessingError("pdf_malformed")
        if page_count > max_pages:
            raise PdfProcessingError("page_limit_exceeded")

        cumulative_text_bytes = 0
        for index in range(page_count):
            try:
                page = doc[index]
            except pdfium.PdfiumError as exc:
                raise PdfProcessingError("pdf_malformed") from exc

            try:
                try:
                    textpage = page.get_textpage()
                except pdfium.PdfiumError as exc:
                    raise PdfProcessingError("pdf_malformed") from exc
                try:
                    text = textpage.get_text_range()
                finally:
                    textpage.close()

                try:
                    width_pt, height_pt = page.get_size()
                except pdfium.PdfiumError as exc:
                    raise PdfProcessingError("pdf_malformed") from exc

                width_px = max(1, int(width_pt * render_scale))
                height_px = max(1, int(height_pt * render_scale))
                if width_px * height_px > max_pixels:
                    raise PdfProcessingError("pixel_limit_exceeded")

                def _render_rgb(
                    *,
                    scale: float | None = None,
                    _page: pdfium.PdfPage = page,
                    _default_scale: float = render_scale,
                ) -> tuple[bytes, int, int]:
                    effective_scale = scale if scale is not None else _default_scale
                    try:
                        bitmap = _page.render(scale=effective_scale)
                    except pdfium.PdfiumError as exc:
                        raise PdfProcessingError("pdf_malformed") from exc
                    try:
                        pil_image = bitmap.to_pil().convert("RGB")
                        return pil_image.tobytes(), pil_image.width, pil_image.height
                    finally:
                        bitmap.close()

                info = PdfPageInfo(
                    page_number=index + 1,
                    width_px=width_px,
                    height_px=height_px,
                    embedded_text=text,
                )
                extra_text_bytes = on_page(info, _render_rgb) or 0
                page_text_bytes = len(text.encode("utf-8", errors="ignore")) + max(
                    0, extra_text_bytes
                )
                if (
                    max_cumulative_text_bytes is not None
                    and cumulative_text_bytes + page_text_bytes > max_cumulative_text_bytes
                ):
                    # Fail closed immediately -- never fold this page's text
                    # into the total, never signal progress for it, and
                    # never open another page. A caller must never be able
                    # to receive a normal return with only a partial
                    # document processed (second independent review,
                    # Finding 2).
                    raise PdfProcessingError("text_limit_exceeded")
                cumulative_text_bytes += page_text_bytes
            finally:
                page.close()

            if on_page_complete is not None:
                on_page_complete(index + 1)
    finally:
        doc.close()
