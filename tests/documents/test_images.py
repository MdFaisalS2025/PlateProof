"""Tests for Pillow-backed image page loading (PNG/JPEG only)."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image


def _png_bytes(width: int = 100, height: int = 50, color: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(width: int = 100, height: int = 50) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_load_png_page() -> None:
    from plateproof.documents.images import load_image_page

    data = _png_bytes(120, 80)
    result = load_image_page(data, max_pixels=50_000_000)
    assert result.width_px == 120
    assert result.height_px == 80
    assert len(result.rgb_bytes) == 120 * 80 * 3


def test_load_jpeg_page() -> None:
    from plateproof.documents.images import load_image_page

    data = _jpeg_bytes(64, 32)
    result = load_image_page(data, max_pixels=50_000_000)
    assert result.width_px == 64
    assert result.height_px == 32


def test_malformed_image_raises_typed_error() -> None:
    from plateproof.documents.images import ImageProcessingError, load_image_page

    with pytest.raises(ImageProcessingError) as exc_info:
        load_image_page(b"not a real image", max_pixels=50_000_000)
    assert exc_info.value.args[0] == "image_malformed"


def test_pixel_limit_exceeded_raises_typed_error() -> None:
    from plateproof.documents.images import ImageProcessingError, load_image_page

    data = _png_bytes(1000, 1000)
    with pytest.raises(ImageProcessingError) as exc_info:
        load_image_page(data, max_pixels=100)
    assert exc_info.value.args[0] == "pixel_limit_exceeded"


def test_decompression_bomb_guard_is_bounded_by_configured_max_pixels() -> None:
    """Pillow's own MAX_IMAGE_PIXELS bomb guard must not be relied on with
    its default value alone -- our own configured max_pixels is the
    authoritative bound, applied before decoding fully succeeds silently."""
    from plateproof.documents.images import ImageProcessingError, load_image_page

    data = _png_bytes(5000, 5000)
    with pytest.raises(ImageProcessingError) as exc_info:
        load_image_page(data, max_pixels=1_000_000)
    assert exc_info.value.args[0] == "pixel_limit_exceeded"
