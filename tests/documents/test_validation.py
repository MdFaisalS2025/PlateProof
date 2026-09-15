"""Tests for the shared, parent-process-safe byte/signature validator.

This validator never parses a document -- it only sniffs a small magic-byte
signature and checks size/media-type. Real parsing (PDFium/Pillow/OCR)
happens only inside a worker process (see test_worker_pool.py /
test_worker_protocol.py).
"""

from __future__ import annotations

import pytest

_PDF_MAGIC = b"%PDF-1.7\n%rest of a pdf..."
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
_JPEG_MAGIC = b"\xff\xd8\xff\xe0" + b"\x00" * 20


def test_pdf_signature_detected() -> None:
    from plateproof.documents.validation import sniff_media_type

    assert sniff_media_type(_PDF_MAGIC) == "application/pdf"


def test_png_signature_detected() -> None:
    from plateproof.documents.validation import sniff_media_type

    assert sniff_media_type(_PNG_MAGIC) == "image/png"


def test_jpeg_signature_detected() -> None:
    from plateproof.documents.validation import sniff_media_type

    assert sniff_media_type(_JPEG_MAGIC) == "image/jpeg"


def test_unrecognized_signature_returns_none() -> None:
    from plateproof.documents.validation import sniff_media_type

    assert sniff_media_type(b"not a real document at all") is None


def test_declared_filename_and_mime_type_are_never_trusted() -> None:
    """The validator's public entry point does not even accept a filename
    or a declared content-type parameter -- proving neither can influence
    the result even if a caller tried to pass one."""
    import inspect

    from plateproof.documents.validation import validate_upload_bytes

    params = inspect.signature(validate_upload_bytes).parameters
    assert "filename" not in params
    assert "declared_media_type" not in params
    assert "content_type" not in params


def test_valid_pdf_within_size_limit_accepted() -> None:
    from plateproof.documents.validation import validate_upload_bytes

    result = validate_upload_bytes(_PDF_MAGIC, max_bytes=10_000)
    assert result.accepted is True
    assert result.detected_media_type == "application/pdf"
    assert result.rejection_reason is None


def test_oversized_upload_rejected() -> None:
    from plateproof.documents.validation import validate_upload_bytes

    big = _PDF_MAGIC + b"\x00" * 10_000
    result = validate_upload_bytes(big, max_bytes=100)
    assert result.accepted is False
    assert result.rejection_reason is not None


def test_empty_upload_rejected() -> None:
    from plateproof.documents.validation import validate_upload_bytes

    result = validate_upload_bytes(b"", max_bytes=1_000)
    assert result.accepted is False


def test_unsupported_media_type_rejected_with_static_reason() -> None:
    from plateproof.documents.validation import validate_upload_bytes

    result = validate_upload_bytes(b"GIF89a not supported", max_bytes=1_000)
    assert result.accepted is False
    assert result.rejection_reason == "unsupported_media_type"


def test_tiff_is_not_supported() -> None:
    """No TIFF support, per the approved plan's non-goals."""
    from plateproof.documents.validation import validate_upload_bytes

    tiff_magic = b"II*\x00" + b"\x00" * 20  # little-endian TIFF signature
    result = validate_upload_bytes(tiff_magic, max_bytes=1_000)
    assert result.accepted is False
    assert result.rejection_reason == "unsupported_media_type"


def test_polyglot_pdf_png_prefix_uses_first_matching_signature_only() -> None:
    """A file crafted to look like two formats at once must still get exactly
    one deterministic classification, never both/neither."""
    from plateproof.documents.validation import sniff_media_type

    polyglot = _PDF_MAGIC + _PNG_MAGIC
    assert sniff_media_type(polyglot) == "application/pdf"


@pytest.mark.parametrize("max_bytes", [-1, 0])
def test_non_positive_max_bytes_is_rejected_outright(max_bytes: int) -> None:
    from plateproof.documents.validation import validate_upload_bytes

    result = validate_upload_bytes(_PDF_MAGIC, max_bytes=max_bytes)
    assert result.accepted is False
