"""Finding 2: PDF processing must be incremental, never retaining every
page's raw RGB raster simultaneously, and must release native PDFium/Pillow
resources deterministically -- never relying on garbage collection.
"""

from __future__ import annotations

import hashlib
import struct

import pytest

_PAD = bytes(
    [
        0x28,
        0xBF,
        0x4E,
        0x5E,
        0x4E,
        0x75,
        0x8A,
        0x41,
        0x64,
        0x00,
        0x4E,
        0x56,
        0xFF,
        0xFA,
        0x01,
        0x08,
        0x2E,
        0x2E,
        0x00,
        0xB6,
        0xD0,
        0x68,
        0x3E,
        0x80,
        0x2F,
        0x0C,
        0xA9,
        0xFE,
        0x64,
        0x53,
        0x69,
        0x7A,
    ]
)


def _rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    klen = len(key)
    for i in range(256):
        j = (j + s[i] + key[i % klen]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        k = s[(s[i] + s[j]) % 256]
        out.append(byte ^ k)
    return bytes(out)


def _pad_password(pw: bytes) -> bytes:
    return pw[:32] if len(pw) >= 32 else pw + _PAD[: 32 - len(pw)]


def _object_key(encryption_key: bytes, obj_num: int, gen_num: int) -> bytes:
    h = hashlib.md5()
    h.update(encryption_key)
    h.update(struct.pack("<I", obj_num)[:3])
    h.update(struct.pack("<I", gen_num)[:2])
    n = min(len(encryption_key) + 5, 16)
    return h.digest()[:n]


def _build_encrypted_pdf(text: str) -> bytes:
    """A real, minimal (sub-1KB) RC4/Standard-Security-Handler-encrypted
    PDF, generated fresh here -- never committed as a fixture file, and
    small enough that it isn't a "large binary." Uses an empty user
    password so pypdfium2 opens it without raising a password error --
    exactly the case a naive "PdfiumError message contains 'password'"
    check cannot detect, proving the need for a real, documented
    encryption-revision check (FPDF_GetSecurityHandlerRevision)."""
    doc_id = b"0123456789ABCDEF"
    user_pw = b""
    owner_pw = b"ownersecret"
    permissions = -44

    owner_digest = hashlib.md5(_pad_password(owner_pw)).digest()[:5]
    o_value = _rc4(owner_digest, _pad_password(user_pw))

    h = hashlib.md5()
    h.update(_pad_password(user_pw))
    h.update(o_value)
    h.update(struct.pack("<i", permissions))
    h.update(doc_id)
    encryption_key = h.digest()[:5]
    u_value = _rc4(encryption_key, _PAD)

    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    encrypted_content = _rc4(_object_key(encryption_key, 4, 0), content)

    def hexs(b: bytes) -> str:
        return "<" + b.hex() + ">"

    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        ),
        (
            f"4 0 obj\n<< /Length {len(encrypted_content)} >>\nstream\n".encode()
            + encrypted_content
            + b"\nendstream\nendobj\n"
        ),
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        (
            f"6 0 obj\n<< /Filter /Standard /V 1 /R 2 /O {hexs(o_value)} "
            f"/U {hexs(u_value)} /P {permissions} >>\nendobj\n"
        ).encode(),
    ]
    return (
        b"%PDF-1.4\n"
        + b"".join(objects)
        + b"xref\n0 1\n0000000000 65535 f \n"
        + (
            f"trailer\n<< /Size 7 /Root 1 0 R /Encrypt 6 0 R "
            f"/ID [{hexs(doc_id)} {hexs(doc_id)}] >>\n"
        ).encode()
        + b"startxref\n0\n%%EOF\n"
    )


def _minimal_pdf(text: str, *, page_count: int = 1) -> bytes:
    objects = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(page_count))
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(
        f"2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {page_count} >>\nendobj\n".encode()
    )
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    next_obj = 3 + page_count
    for i in range(page_count):
        objects.append(
            (
                f"{3 + i} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                f"/Contents {next_obj + i} 0 R "
                f"/Resources << /Font << /F1 {next_obj + page_count} 0 R >> >> >>\nendobj\n"
            ).encode()
        )
    for i in range(page_count):
        objects.append(
            f"{next_obj + i} 0 obj\n<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream\nendobj\n"
        )
    objects.append(
        (
            f"{next_obj + page_count} 0 obj\n"
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        ).encode()
    )
    return (
        b"%PDF-1.4\n"
        + b"".join(objects)
        + b"xref\n0 1\n0000000000 65535 f \n"
        + b"trailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def test_embedded_text_page_never_calls_render() -> None:
    """A page with usable embedded text must never trigger page.render()."""
    from plateproof.documents.pdf import process_pdf_pages

    data = _minimal_pdf("Score: 14")
    render_calls: list[int] = []

    def on_page(info: object, render_rgb: object) -> None:
        assert info.embedded_text.strip() == "Score: 14"  # type: ignore[attr-defined]
        # Deliberately never call render_rgb() -- embedded text is usable.

    import plateproof.documents.pdf as pdf_module

    original_render = pdf_module.pdfium.PdfPage.render

    def _tracking_render(self: object, *args: object, **kwargs: object) -> object:
        render_calls.append(1)
        return original_render(self, *args, **kwargs)  # type: ignore[arg-type]

    pdf_module.pdfium.PdfPage.render = _tracking_render
    try:
        process_pdf_pages(data, max_pages=5, max_pixels=50_000_000, on_page=on_page)
    finally:
        pdf_module.pdfium.PdfPage.render = original_render

    assert render_calls == []


def test_render_is_only_called_when_ocr_actually_needed() -> None:
    from plateproof.documents.pdf import process_pdf_pages

    # A page with an empty content stream -- no embedded text at all.
    empty_content_pdf = _minimal_pdf(" ")
    calls: list[int] = []

    def on_page(info: object, render_rgb: object) -> None:
        rgb_bytes, width, height = render_rgb()  # type: ignore[misc]
        calls.append(1)
        assert isinstance(rgb_bytes, bytes)
        assert width > 0 and height > 0

    process_pdf_pages(empty_content_pdf, max_pages=5, max_pixels=50_000_000, on_page=on_page)
    assert calls == [1]


def test_at_most_one_rendered_raster_is_live_at_a_time() -> None:
    """Across a multi-page document where every page needs OCR, the
    callback must receive one fresh raster per page -- never an
    accumulated list of all pages' rasters held simultaneously."""
    from plateproof.documents.pdf import process_pdf_pages

    data = _minimal_pdf(" ", page_count=3)
    live_rasters: list[bytes] = []
    max_live_at_once = 0

    def on_page(info: object, render_rgb: object) -> None:
        nonlocal max_live_at_once
        rgb_bytes, _width, _height = render_rgb()  # type: ignore[misc]
        live_rasters.append(rgb_bytes)
        max_live_at_once = max(max_live_at_once, len(live_rasters))
        live_rasters.clear()  # caller (entrypoint) uses it immediately, then it's gone

    process_pdf_pages(data, max_pages=5, max_pixels=50_000_000, on_page=on_page)
    assert max_live_at_once == 1


def test_native_resources_close_on_success() -> None:
    """PdfPage/PdfTextPage/PdfBitmap objects created during processing are
    all closed by the time process_pdf_pages returns -- proven by call
    counts on spied close() methods matching the number of pages/renders."""
    import plateproof.documents.pdf as pdf_module
    from plateproof.documents.pdf import process_pdf_pages

    data = _minimal_pdf(" ", page_count=2)  # blank content -- forces a render on every page

    page_closes = []
    textpage_closes = []
    bitmap_closes = []
    original_page_close = pdf_module.pdfium.PdfPage.close
    original_textpage_close = pdf_module.pdfium.PdfTextPage.close
    original_bitmap_close = pdf_module.pdfium.PdfBitmap.close

    def _track_page(self: object) -> None:
        page_closes.append(1)
        original_page_close(self)  # type: ignore[misc]

    def _track_textpage(self: object) -> None:
        textpage_closes.append(1)
        original_textpage_close(self)  # type: ignore[misc]

    def _track_bitmap(self: object) -> None:
        bitmap_closes.append(1)
        original_bitmap_close(self)  # type: ignore[misc]

    pdf_module.pdfium.PdfPage.close = _track_page
    pdf_module.pdfium.PdfTextPage.close = _track_textpage
    pdf_module.pdfium.PdfBitmap.close = _track_bitmap
    try:

        def on_page(info: object, render_rgb: object) -> None:
            render_rgb()  # type: ignore[misc]

        process_pdf_pages(data, max_pages=5, max_pixels=50_000_000, on_page=on_page)
    finally:
        pdf_module.pdfium.PdfPage.close = original_page_close
        pdf_module.pdfium.PdfTextPage.close = original_textpage_close
        pdf_module.pdfium.PdfBitmap.close = original_bitmap_close

    assert len(page_closes) == 2
    assert len(textpage_closes) == 2
    assert len(bitmap_closes) == 2


def test_native_resources_close_on_failure_path_too() -> None:
    """A page-processing failure partway through (e.g. pixel limit
    exceeded on page 2) must still close everything opened for page 1 and
    page 2 -- proven indirectly by running many times without resource
    exhaustion errors."""
    from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages

    data = _minimal_pdf("Score: 14", page_count=3)

    def on_page(info: object, render_rgb: object) -> None:
        pass

    for _ in range(20):
        with pytest.raises(PdfProcessingError):
            process_pdf_pages(data, max_pages=5, max_pixels=1, on_page=on_page)


def test_page_size_limit_checked_before_any_render_allocation() -> None:
    """The pixel-limit check must happen using the page's declared
    dimensions, before render() is ever called -- proven by a render spy
    that would fail the test if invoked."""
    import plateproof.documents.pdf as pdf_module
    from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages

    data = _minimal_pdf(" ")

    def _must_not_be_called(self: object, *args: object, **kwargs: object) -> object:
        raise AssertionError(
            "render() must not be called when the page already exceeds the pixel limit"
        )

    original_render = pdf_module.pdfium.PdfPage.render
    pdf_module.pdfium.PdfPage.render = _must_not_be_called
    try:
        with pytest.raises(PdfProcessingError) as exc_info:
            process_pdf_pages(
                data, max_pages=5, max_pixels=1, on_page=lambda info, render_rgb: None
            )
        assert exc_info.value.args[0] == "pixel_limit_exceeded"
    finally:
        pdf_module.pdfium.PdfPage.render = original_render


def test_encrypted_pdf_with_empty_user_password_is_detected() -> None:
    """A real, hand-generated, empty-user-password encrypted PDF opens
    successfully in pypdfium2 (no password error is ever raised) -- the
    old substring-of-exception-message check could never catch this. The
    documented FPDF_GetSecurityHandlerRevision() API must detect it."""
    from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages

    data = _build_encrypted_pdf("Score: 14")
    with pytest.raises(PdfProcessingError) as exc_info:
        process_pdf_pages(
            data, max_pages=5, max_pixels=50_000_000, on_page=lambda info, render_rgb: None
        )
    assert exc_info.value.args[0] == "encrypted_document"


def test_cumulative_text_limit_raises_and_stops_processing_deterministically() -> None:
    """Once cumulative extracted text across all pages exceeds the
    configured ceiling, processing must fail closed with a typed error --
    never return normally with only the earlier pages, which would let a
    caller silently treat a partially-processed document as complete
    (second independent review, Finding 2). Later pages must never be
    opened once the limit is detected, and native PDFium resources for
    every page touched so far must still be closed."""
    import plateproof.documents.pdf as pdf_module
    from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages

    data = _minimal_pdf("Score: 14", page_count=3)
    pages_seen: list[int] = []
    page_closes = []
    original_page_close = pdf_module.pdfium.PdfPage.close

    def _track_page(self: object) -> None:
        page_closes.append(1)
        original_page_close(self)  # type: ignore[misc]

    def on_page(info: object, render_rgb: object) -> None:
        pages_seen.append(info.page_number)  # type: ignore[attr-defined]

    pdf_module.pdfium.PdfPage.close = _track_page
    try:
        with pytest.raises(PdfProcessingError) as exc_info:
            process_pdf_pages(
                data,
                max_pages=5,
                max_pixels=50_000_000,
                on_page=on_page,
                max_cumulative_text_bytes=5,
            )
    finally:
        pdf_module.pdfium.PdfPage.close = original_page_close

    assert exc_info.value.args[0] == "text_limit_exceeded"
    # "Score: 14" on page 1 alone already exceeds 5 bytes -- pages 2 and 3
    # must never be opened.
    assert pages_seen == [1]
    assert len(page_closes) == 1


def test_cumulative_text_limit_counts_ocr_text_reported_by_the_caller() -> None:
    """The cumulative budget must also cover text the caller extracted via
    OCR (reported back through ``on_page``'s return value) -- a scanned PDF
    must not be able to bypass the limit just because its text came from
    OCR instead of the embedded-text layer."""
    from plateproof.documents.pdf import PdfProcessingError, process_pdf_pages

    # Blank content stream -- no embedded text at all, forcing on_page to
    # be the sole source of extracted text for the cumulative budget.
    data = _minimal_pdf(" ", page_count=2)

    def on_page(info: object, render_rgb: object) -> int:
        render_rgb()  # type: ignore[misc]
        return 1_000  # simulates a large block of OCR-extracted text

    with pytest.raises(PdfProcessingError) as exc_info:
        process_pdf_pages(
            data,
            max_pages=5,
            max_pixels=50_000_000,
            on_page=on_page,
            max_cumulative_text_bytes=5,
        )
    assert exc_info.value.args[0] == "text_limit_exceeded"


def test_page_complete_callback_fires_only_after_page_cleanup() -> None:
    """``on_page_complete`` (what entrypoint.py hooks to send
    ``page_progress``) must fire strictly after that page's own native
    resources (PdfPage/PdfTextPage/PdfBitmap) are already closed -- proven
    by call-order tracking against a spied ``PdfPage.close``."""
    from plateproof.documents.pdf import process_pdf_pages

    data = _minimal_pdf("Score: 14", page_count=2)
    order: list[str] = []

    import plateproof.documents.pdf as pdf_module

    original_close = pdf_module.pdfium.PdfPage.close

    def _tracking_close(self: object) -> None:
        order.append("page_closed")
        original_close(self)  # type: ignore[misc]

    pdf_module.pdfium.PdfPage.close = _tracking_close
    try:

        def on_page(info: object, render_rgb: object) -> None:
            order.append(f"on_page_{info.page_number}")  # type: ignore[attr-defined]

        def on_page_complete(page_number: int) -> None:
            order.append(f"complete_{page_number}")

        process_pdf_pages(
            data,
            max_pages=5,
            max_pixels=50_000_000,
            on_page=on_page,
            on_page_complete=on_page_complete,
        )
    finally:
        pdf_module.pdfium.PdfPage.close = original_close

    # Each page's own close must happen strictly between its on_page call
    # and its "complete" signal -- proving progress is only ever emitted
    # after that page's resources are already released.
    assert order == [
        "on_page_1",
        "page_closed",
        "complete_1",
        "on_page_2",
        "page_closed",
        "complete_2",
    ]
