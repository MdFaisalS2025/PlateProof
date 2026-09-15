"""Engine-neutral OCR contracts. OCR confidence is deliberately kept
separate from any downstream factual-field confidence -- see
``plateproof.documents.models.ConfidenceComponents``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class OcrOutcome(StrEnum):
    SUCCESS = "success"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True, kw_only=True)
class OcrTextBlock:
    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, kw_only=True)
class OcrResult:
    outcome: OcrOutcome
    blocks: tuple[OcrTextBlock, ...]
    unavailable_reason: str | None


class OcrEngine(Protocol):
    def run(self, rgb_bytes: bytes, *, width: int, height: int) -> OcrResult: ...
