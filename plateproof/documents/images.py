"""Pillow calls -- ONLY ever invoked from inside a worker process."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


class ImageProcessingError(Exception):
    """Raised with exactly one closed-enum reason string as its sole
    argument: ``"image_malformed"`` or ``"pixel_limit_exceeded"``."""


@dataclass(frozen=True, kw_only=True)
class ImagePageRender:
    width_px: int
    height_px: int
    rgb_bytes: bytes


def load_image_page(data: bytes, *, max_pixels: int) -> ImagePageRender:
    """Decode ``data`` as a PNG/JPEG image and return a bounded RGB raster.

    The declared header dimensions are checked against ``max_pixels``
    *before* the pixel data is fully decoded, closing the same class of
    decompression-bomb risk Pillow's own (larger, global) default guard is
    meant for -- our own configured limit is authoritative here, not
    Pillow's default ``Image.MAX_IMAGE_PIXELS``.
    """
    try:
        image = Image.open(BytesIO(data))
        width, height = image.size
        if width * height > max_pixels:
            raise ImageProcessingError("pixel_limit_exceeded")
        image.load()
    except ImageProcessingError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageProcessingError("image_malformed") from exc

    rgb_image = image.convert("RGB")
    return ImagePageRender(width_px=width, height_px=height, rgb_bytes=rgb_image.tobytes())
