"""Tests for pypdfium2-backed PDF page loading.

This module is normally invoked only inside a worker process; tests call it
directly for unit coverage, which is safe here (the test runner is not a
production API/UI process handling untrusted network input).
"""

from __future__ import annotations

import pytest


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
                f"/Resources << /Font << /F1 {next_obj + page_count} 0 R >> >> >>\n"
                "endobj\n"
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
    body = (
        b"%PDF-1.4\n"
        + b"".join(objects)
        + b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\n"
        + b"startxref\n0\n%%EOF\n"
    )
    return body


def test_load_single_page_embedded_text_pdf() -> None:
    from plateproof.documents.pdf import load_pdf_pages

    data = _minimal_pdf("Score: 14")
    pages = load_pdf_pages(data, max_pages=5, max_pixels=50_000_000)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Score: 14" in pages[0].embedded_text
    assert pages[0].width_px > 0
    assert pages[0].height_px > 0
    assert len(pages[0].rgb_bytes) == pages[0].width_px * pages[0].height_px * 3


def test_load_multi_page_pdf() -> None:
    from plateproof.documents.pdf import load_pdf_pages

    data = _minimal_pdf("Restaurant: Test Diner", page_count=3)
    pages = load_pdf_pages(data, max_pages=5, max_pixels=50_000_000)
    assert [p.page_number for p in pages] == [1, 2, 3]


def test_malformed_pdf_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError, load_pdf_pages

    with pytest.raises(PdfProcessingError) as exc_info:
        load_pdf_pages(b"not a real pdf at all", max_pages=5, max_pixels=50_000_000)
    assert exc_info.value.args[0] == "pdf_malformed"


def test_page_limit_exceeded_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError, load_pdf_pages

    data = _minimal_pdf("x", page_count=3)
    with pytest.raises(PdfProcessingError) as exc_info:
        load_pdf_pages(data, max_pages=2, max_pixels=50_000_000)
    assert exc_info.value.args[0] == "page_limit_exceeded"


def test_pixel_limit_exceeded_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError, load_pdf_pages

    data = _minimal_pdf("x")
    with pytest.raises(PdfProcessingError) as exc_info:
        load_pdf_pages(data, max_pages=5, max_pixels=10)
    assert exc_info.value.args[0] == "pixel_limit_exceeded"


def test_empty_pdf_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError, load_pdf_pages

    with pytest.raises(PdfProcessingError):
        load_pdf_pages(b"%PDF-1.4\n%%EOF\n", max_pages=5, max_pixels=50_000_000)
