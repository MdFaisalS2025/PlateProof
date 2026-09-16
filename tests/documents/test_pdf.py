"""Tests for pypdfium2-backed PDF page processing (incremental API --
see test_pdf_streaming.py for the memory/resource-lifecycle-specific
tests added in the independent-review correction of commit 735d4a3).

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


def _collect_pages(data: bytes, **kwargs: object) -> list[dict]:
    from plateproof.documents.pdf import process_pdf_pages

    collected: list[dict] = []

    def on_page(info: object, render_rgb: object) -> None:
        rgb_bytes, width, height = render_rgb() if not info.embedded_text.strip() else (b"", 0, 0)  # type: ignore[attr-defined,misc]
        collected.append(
            {
                "page_number": info.page_number,  # type: ignore[attr-defined]
                "embedded_text": info.embedded_text,  # type: ignore[attr-defined]
                "width_px": info.width_px,  # type: ignore[attr-defined]
                "height_px": info.height_px,  # type: ignore[attr-defined]
                "rgb_bytes": rgb_bytes,
            }
        )

    process_pdf_pages(data, on_page=on_page, **kwargs)  # type: ignore[arg-type]
    return collected


def test_load_single_page_embedded_text_pdf() -> None:
    data = _minimal_pdf("Score: 14")
    pages = _collect_pages(data, max_pages=5, max_pixels=50_000_000)
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert "Score: 14" in pages[0]["embedded_text"]
    assert pages[0]["width_px"] > 0
    assert pages[0]["height_px"] > 0


def test_load_multi_page_pdf() -> None:
    data = _minimal_pdf("Restaurant: Test Diner", page_count=3)
    pages = _collect_pages(data, max_pages=5, max_pixels=50_000_000)
    assert [p["page_number"] for p in pages] == [1, 2, 3]


def test_malformed_pdf_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError

    with pytest.raises(PdfProcessingError) as exc_info:
        _collect_pages(b"not a real pdf at all", max_pages=5, max_pixels=50_000_000)
    assert exc_info.value.args[0] == "pdf_malformed"


def test_page_limit_exceeded_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError

    data = _minimal_pdf("x", page_count=3)
    with pytest.raises(PdfProcessingError) as exc_info:
        _collect_pages(data, max_pages=2, max_pixels=50_000_000)
    assert exc_info.value.args[0] == "page_limit_exceeded"


def test_pixel_limit_exceeded_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError

    data = _minimal_pdf("x")
    with pytest.raises(PdfProcessingError) as exc_info:
        _collect_pages(data, max_pages=5, max_pixels=10)
    assert exc_info.value.args[0] == "pixel_limit_exceeded"


def test_empty_pdf_raises_typed_error() -> None:
    from plateproof.documents.pdf import PdfProcessingError

    with pytest.raises(PdfProcessingError):
        _collect_pages(b"%PDF-1.4\n%%EOF\n", max_pages=5, max_pixels=50_000_000)
