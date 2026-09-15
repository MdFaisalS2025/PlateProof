"""RapidOCR adapter. Lazy import, fail-closed if unavailable -- OCR is
entirely optional; embedded-text PDFs must keep working with this module
absent, broken, or its construction raising for any reason.

Called only from inside a worker process (see ``plateproof.documents.worker``).
Never reachable from the API/UI parent process.

Verified offline: constructing ``rapidocr.RapidOCR()`` and running inference
against the default bundled det/cls/rec models makes zero network calls
(proven by ``test_network_tripwire_no_socket_calls_during_construction_and_inference``,
which patches ``socket.socket.connect``, ``urllib.request.urlopen``, and
``http.client.HTTPConnection.connect`` to raise). If a future RapidOCR
release ever needs a network call this module has not anticipated, engine
construction fails and this module reports ``ocr_unavailable`` rather than
letting an unexpected network attempt occur silently.
"""

from __future__ import annotations

import logging

import numpy as np

from plateproof.documents.ocr.base import OcrOutcome, OcrResult, OcrTextBlock

_engine: object | None = None
_unavailable_reason: str | None = None
_attempted = False


def _construct_engine() -> object:
    # RapidOCR logs its own model-file paths at INFO level by default --
    # never document-derived content, but suppressed anyway as a hardening
    # measure consistent with "never log ... paths" elsewhere in Task 9.
    logging.getLogger("RapidOCR").setLevel(logging.CRITICAL)
    from rapidocr import RapidOCR

    return RapidOCR()


def _get_engine() -> object | None:
    global _engine, _unavailable_reason, _attempted
    if _attempted:
        return _engine
    _attempted = True
    try:
        _engine = _construct_engine()
    except Exception:  # noqa: BLE001 - any construction failure is a fail-closed signal, not a crash
        _engine = None
        _unavailable_reason = "ocr_unavailable"
    return _engine


def is_available() -> bool:
    _get_engine()
    return _unavailable_reason is None


def run_ocr(rgb_bytes: bytes, *, width: int, height: int) -> OcrResult:
    engine = _get_engine()
    if engine is None:
        return OcrResult(
            outcome=OcrOutcome.UNAVAILABLE, blocks=(), unavailable_reason="ocr_unavailable"
        )

    try:
        array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
        result = engine(array)  # type: ignore[operator]
    except Exception:  # noqa: BLE001 - a bad frame must never crash the worker
        return OcrResult(outcome=OcrOutcome.FAILED, blocks=(), unavailable_reason=None)

    blocks: list[OcrTextBlock] = []
    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is not None and texts is not None and scores is not None:
        for box, text, score in zip(boxes, texts, scores, strict=True):
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            blocks.append(
                OcrTextBlock(
                    text=str(text),
                    confidence=float(score),
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                )
            )
    return OcrResult(outcome=OcrOutcome.SUCCESS, blocks=tuple(blocks), unavailable_reason=None)


def _reset_for_testing() -> None:
    """Test-only hook to force re-evaluation of engine availability."""
    global _engine, _unavailable_reason, _attempted
    _engine = None
    _unavailable_reason = None
    _attempted = False
