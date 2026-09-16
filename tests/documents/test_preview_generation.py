"""Finding 7: previews are generated only inside the worker, bounded to a
low resolution, re-encoded as PNG with no original metadata, and surfaced
on the resulting ExtractionDraft so a future Task 9B page can display them
without ever reopening the original upload."""

from __future__ import annotations

import struct
from io import BytesIO

from PIL import Image


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _png_dimensions(data: bytes) -> tuple[int, int]:
    return struct.unpack(">II", data[16:24])


def test_worker_generates_a_bounded_preview_for_a_pdf_page() -> None:
    """Uses the real WorkerPool (not raw protocol calls) so the full
    preview receive-and-validate path (pool.py's _receive_previews) runs,
    matching how service.py actually consumes a worker's output."""
    from plateproof.documents.limits import (
        DEFAULT_MAX_PREVIEW_HEIGHT_PX,
        DEFAULT_MAX_PREVIEW_WIDTH_PX,
    )
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerJobRequest, WorkerJobResponse

    request = WorkerJobRequest(
        expected_jurisdiction="nyc",
        media_type="application/pdf",
        max_pages=5,
        max_pixels=50_000_000,
        ocr_enabled=False,
        page_timeout_seconds=10.0,
    )
    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1))
    try:
        outcome = pool.submit(request, _minimal_pdf("Score: 14"))
    finally:
        pool.shutdown()

    assert isinstance(outcome, WorkerJobResponse)
    assert outcome.preview_count == 1
    assert len(outcome.previews) == 1
    preview = outcome.previews[0]
    assert preview.page_number == 1
    assert preview.png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = _png_dimensions(preview.png_bytes)
    assert width <= DEFAULT_MAX_PREVIEW_WIDTH_PX
    assert height <= DEFAULT_MAX_PREVIEW_HEIGHT_PX


def test_preview_png_carries_no_exif_or_original_metadata() -> None:
    """The preview is re-encoded from decoded pixels, never a copy of the
    original file's bytes -- there is no EXIF/metadata to strip because
    none was ever carried over in the first place."""
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
    from plateproof.documents.worker.protocol import WorkerJobRequest, WorkerJobResponse

    # A JPEG with real EXIF data, to prove the preview doesn't inherit it.
    buffer = BytesIO()
    image = Image.new("RGB", (100, 60), "white")
    exif = image.getexif()
    exif[0x010E] = "sensitive description"  # ImageDescription tag
    image.save(buffer, format="JPEG", exif=exif)
    jpeg_bytes = buffer.getvalue()
    assert b"sensitive description" in jpeg_bytes  # sanity: the source really has it

    request = WorkerJobRequest(
        expected_jurisdiction="nyc",
        media_type="image/jpeg",
        max_pages=5,
        max_pixels=50_000_000,
        ocr_enabled=False,
        page_timeout_seconds=10.0,
    )
    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1))
    try:
        outcome = pool.submit(request, jpeg_bytes)
    finally:
        pool.shutdown()

    assert isinstance(outcome, WorkerJobResponse)
    assert len(outcome.previews) == 1
    assert b"sensitive description" not in outcome.previews[0].png_bytes


def test_pool_never_imports_native_parsers_directly() -> None:
    """Finding 7: no parent-side parser imports. pool.py only reads a tiny
    fixed-offset PNG header via struct -- it never imports Pillow, PDFium,
    or any OCR module."""
    import plateproof.documents.worker.pool as pool_module

    with open(pool_module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert "PIL" not in source
    assert "pypdfium2" not in source
    assert "rapidocr" not in source
