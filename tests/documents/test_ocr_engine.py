"""Tests for the RapidOCR engine adapter: fail-closed behavior and the
network tripwire that is the actual proof of the plan's offline claim."""

from __future__ import annotations

import http.client
import socket
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw


def _text_image_rgb_bytes(
    text: str, *, width: int = 300, height: int = 80
) -> tuple[bytes, int, int]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 25), text, fill="black")
    return image.tobytes(), width, height


def test_rapidocr_reads_real_rendered_text() -> None:
    from plateproof.documents.ocr.base import OcrOutcome
    from plateproof.documents.ocr.rapidocr_engine import run_ocr

    rgb_bytes, width, height = _text_image_rgb_bytes("Score: 14")
    result = run_ocr(rgb_bytes, width=width, height=height)
    assert result.outcome == OcrOutcome.SUCCESS
    joined = " ".join(b.text for b in result.blocks)
    assert "Score" in joined or "14" in joined


def test_rapidocr_blocks_have_confidence_and_bounding_box() -> None:
    from plateproof.documents.ocr.rapidocr_engine import run_ocr

    rgb_bytes, width, height = _text_image_rgb_bytes("Restaurant Name")
    result = run_ocr(rgb_bytes, width=width, height=height)
    for block in result.blocks:
        assert 0.0 <= block.confidence <= 1.0
        assert block.x0 < block.x1
        assert block.y0 < block.y1


def test_rapidocr_blank_image_returns_success_with_no_blocks() -> None:
    from plateproof.documents.ocr.base import OcrOutcome
    from plateproof.documents.ocr.rapidocr_engine import run_ocr

    blank = np.full((60, 200, 3), 255, dtype=np.uint8).tobytes()
    result = run_ocr(blank, width=200, height=60)
    assert result.outcome == OcrOutcome.SUCCESS
    assert result.blocks == ()


def test_network_tripwire_no_socket_calls_during_construction_and_inference() -> None:
    """The actual proof of the plan's offline claim: patch every network
    entry point to raise, then construct the engine and run real inference
    against a fixture image. Zero calls must occur."""
    from plateproof.documents.ocr import rapidocr_engine

    rapidocr_engine._reset_for_testing()

    def _blocked(*args: object, **kwargs: object) -> None:
        raise AssertionError("a network call was attempted during OCR init/inference")

    original_connect = socket.socket.connect
    original_urlopen = urllib.request.urlopen
    original_http_connect = http.client.HTTPConnection.connect
    socket.socket.connect = _blocked  # type: ignore[method-assign]
    urllib.request.urlopen = _blocked  # type: ignore[assignment]
    http.client.HTTPConnection.connect = _blocked  # type: ignore[method-assign]
    try:
        rgb_bytes, width, height = _text_image_rgb_bytes("Score: 14")
        result = rapidocr_engine.run_ocr(rgb_bytes, width=width, height=height)
        assert result.outcome.value == "success"
    finally:
        socket.socket.connect = original_connect
        urllib.request.urlopen = original_urlopen
        http.client.HTTPConnection.connect = original_http_connect
        rapidocr_engine._reset_for_testing()


def test_engine_construction_failure_fails_closed_not_crashes() -> None:
    from plateproof.documents.ocr import rapidocr_engine
    from plateproof.documents.ocr.base import OcrOutcome

    rapidocr_engine._reset_for_testing()
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                rapidocr_engine,
                "_construct_engine",
                lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            )
            assert rapidocr_engine.is_available() is False
            result = rapidocr_engine.run_ocr(b"\x00" * 300, width=10, height=10)
            assert result.outcome == OcrOutcome.UNAVAILABLE
            assert result.unavailable_reason == "ocr_unavailable"
    finally:
        rapidocr_engine._reset_for_testing()
