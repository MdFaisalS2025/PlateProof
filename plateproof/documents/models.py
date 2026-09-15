"""Typed, frozen contracts for Task 9 document extraction.

Every dataclass here uses ``@dataclass(frozen=True, kw_only=True)`` --
construction is always by keyword, and no field is ever mutated after
construction (a correction produces a new value, never an in-place edit).
Nothing in this module is trusted: a ``Candidate`` is always a proposal, and
an ``ExtractionDraft`` is never "confirmed" data until a human explicitly
reviews it (see ``plateproof.documents.corrections`` and ``draft_builder``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

Jurisdiction = Literal["nyc", "florida"]
EvidenceSource = Literal["embedded_text", "ocr"]
ConfidenceLabel = Literal["high", "medium", "low"]
ProcessingStatus = Literal["completed", "ocr_unavailable", "failed"]
DetectedMediaType = Literal["application/pdf", "image/png", "image/jpeg"]


@dataclass(frozen=True, kw_only=True)
class BoundingBox:
    """Pixel-space coordinates on one rendered page. Always finite and
    ordered (``x0 < x1``, ``y0 < y1``) -- enforced by callers that construct
    one, not by this container itself."""

    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, kw_only=True)
class EvidenceSpan:
    """One bounded pointer back to the source document backing a candidate
    value. ``excerpt`` is a short quotation of the surrounding text, never
    the full page -- see ``corrections.MAX_EVIDENCE_EXCERPT_LENGTH``."""

    page: int
    excerpt: str
    bounding_box: BoundingBox | None
    source: EvidenceSource


@dataclass(frozen=True, kw_only=True)
class ConfidenceComponents:
    """The individual signals behind one candidate's confidence, kept
    separate from OCR confidence per the plan's requirement that OCR
    confidence and factual confidence never collapse into one number."""

    ocr_confidence: float | None
    pattern_confidence: float
    corroboration_confidence: float | None
    overall: float


@dataclass(frozen=True, kw_only=True)
class Ambiguity:
    """A field where extraction found more than one plausible value and
    deliberately did not guess which is correct."""

    field_name: str
    reason: str
    candidate_values: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class Warning:
    code: str
    message: str


@dataclass(frozen=True, kw_only=True)
class Candidate[T]:
    """One machine-proposed field value. ``value`` is the typed, parsed
    result; ``display_value`` is what a reviewer sees. Both may be ``None``
    when a field could not be parsed even though evidence exists."""

    field_name: str
    value: T | None
    display_value: str | None
    evidence: tuple[EvidenceSpan, ...]
    confidence: ConfidenceComponents
    confidence_label: ConfidenceLabel


@dataclass(frozen=True, kw_only=True)
class UploadMetadata:
    detected_media_type: DetectedMediaType
    byte_size: int
    page_count: int


@dataclass(frozen=True, kw_only=True)
class PageMetadata:
    page_number: int
    width_px: int
    height_px: int
    used_ocr: bool


@dataclass(frozen=True, kw_only=True)
class TextBlock:
    """One unit of text recovered from a page, tagged with where it came
    from -- embedded PDF text is never confused with OCR output."""

    page: int
    text: str
    bounding_box: BoundingBox | None
    source: EvidenceSource
    ocr_confidence: float | None


@dataclass(frozen=True, kw_only=True)
class OcrEngineInfo:
    engine_name: Literal["rapidocr"]
    available: bool
    unavailable_reason: str | None


@dataclass(frozen=True, kw_only=True)
class UserCorrection:
    field_name: str
    raw_display_value: str
    corrected_at: datetime


@dataclass(frozen=True, kw_only=True)
class ValidatedCorrection:
    """Distinguishes raw input, parsed value, and status -- these never
    collapse into one field (see ``plateproof.documents.corrections``)."""

    raw_display_value: str
    parsed_value: Any | None
    parse_status: Literal["parsed", "unparseable", "rejected"]
    rejection_reason: str | None


@dataclass(frozen=True, kw_only=True)
class ExtractionDraft:
    """The complete, stateless result of one extraction job. Never written
    anywhere server-side; a caller holds this in memory/session state only."""

    draft_id: str
    jurisdiction_expected: Jurisdiction
    jurisdiction_detected: Jurisdiction | None
    jurisdiction_mismatch: bool
    restaurant_id: str
    restaurant_identity_corroborated: bool
    upload: UploadMetadata
    pages: tuple[PageMetadata, ...]
    candidates: Mapping[str, Candidate[Any]]
    ambiguities: tuple[Ambiguity, ...]
    missing_fields: tuple[str, ...]
    warnings: tuple[Warning, ...]
    ocr_engine: OcrEngineInfo
    processing_status: ProcessingStatus
    generated_at: datetime
    confirmable: bool


@dataclass(frozen=True, kw_only=True)
class ConfirmedDocumentRecord:
    """A user-downloaded artifact only -- never persisted server-side and
    never elevated past ``record_status="user_submitted"``."""

    restaurant_id: str
    confirmed_at: datetime
    machine_candidates: Mapping[str, Candidate[Any]]
    user_corrections: Mapping[str, UserCorrection]
    disclaimer: str
    record_status: Literal["user_submitted"] = "user_submitted"
