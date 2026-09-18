"""Typed request/response contracts for the Task 7 API.

Every prediction response is a discriminated union on ``available`` --
there is no schema shape in which an "unavailable" response can also carry
a probability.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from plateproof.copilot.question_validation import ABSOLUTE_MAX_QUESTION_LENGTH, sanitize_question

Jurisdiction = Literal["nyc", "florida"]
RiskBand = Literal["low", "moderate", "high", "insufficient_history"]
ReadinessStatus = Literal[
    "ready",
    "insufficient_performance",
    "uncalibrated",
    "insufficient_uncertainty",
    "insufficient_data",
    "evaluation_only",
]


class PaginationMeta(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: int
    limit: int
    offset: int
    has_more: bool


class HealthComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    # "configured_unverified": the component is enabled in configuration,
    # but /health performs no live network probe of it (see the "local_ai"
    # component) -- truthfully distinct from "ok" (verified working).
    status: Literal["ok", "unavailable", "disabled", "empty", "configured_unverified"]
    detail: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok", "degraded", "unavailable"]
    version: str
    components: list[HealthComponent]


class MichelinDistinctionItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    guide_name: str
    guide_year: int
    distinction: str
    announced_date: date | None
    source_url: str
    source_publisher: str


class RestaurantSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    restaurant_id: str
    jurisdiction: Jurisdiction
    name: str
    address: str | None
    city: str | None
    region: str | None
    cuisine: str | None
    latest_documented_michelin_distinctions: list[str] = Field(default_factory=list)
    latest_documented_guide_year: int | None = None


class RestaurantSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: list[RestaurantSummary]
    pagination: PaginationMeta
    filters_applied: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    optional_data_status: dict[str, str] = Field(default_factory=dict)


class InspectionHistoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    inspection_id: str
    inspection_date: date
    inspection_type: str
    jurisdiction: Jurisdiction
    score: float | None = None
    grade: str | None = None
    critical_violation_count: int | None = None
    action: str | None = None
    high_priority_count: int | None = None
    intermediate_count: int | None = None
    basic_count: int | None = None
    total_violation_count: int | None = None
    disposition_status: str | None = None
    source_snapshot_date: date | None = None


class ViolationHistoryItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    inspection_id: str
    inspection_date: date
    violation_code: str
    description: str | None
    severity: str | None
    count: int
    corrected_on_site: bool | None = None


class RestaurantDetail(BaseModel):
    model_config = ConfigDict(frozen=True)

    restaurant: RestaurantSummary
    official_source_links: list[str]
    michelin_history: list[MichelinDistinctionItem]
    michelin_context_note: str | None = None
    #: Task 10: a constructed Google Maps search link, present only when
    #: ``Settings.google_integration_enabled`` is true and a link could be
    #: built (a blank name, or a query too long for Google's own URL
    #: length ceiling, yields None even when enabled). Never a Google
    #: API response and never a verified match -- see google_attribution.
    google_search_link: str | None = None
    google_attribution: str | None = None


class PredictionAvailable(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: Literal[True] = True
    restaurant_id: str
    jurisdiction: Jurisdiction
    target_name: str
    probability: float
    lower_bound: float
    upper_bound: float
    risk_band: RiskBand
    model_version: str
    generated_at: datetime
    as_of_date: date
    uncertainty_status: str
    calibration_status: str
    readiness_status: ReadinessStatus
    top_factors: list[str] | None = None
    explanation_unavailable_reason: str | None = "not yet implemented in this release"
    jurisdiction_native_context: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = "This is a PlateProof estimate, not an official inspection result."


class PredictionUnavailable(BaseModel):
    model_config = ConfigDict(frozen=True)

    available: Literal[False] = False
    restaurant_id: str
    jurisdiction: Jurisdiction
    reason: Literal["no_ready_model", "insufficient_history", "stale", "not_scored"]
    detail: str
    insufficient_history_reason: str | None = None


PredictionResponse = Annotated[
    PredictionAvailable | PredictionUnavailable, Field(discriminator="available")
]


class ModelCardResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    jurisdiction: Jurisdiction
    target_name: str
    model_version: str
    readiness_status: ReadinessStatus
    markdown: str
    registered_at: str


class ApiError(BaseModel):
    model_config = ConfigDict(frozen=True)

    error: str
    message: str


# --------------------------------------------------------------------------- #
# Task 8B: Copilot API contract.                                              #
# --------------------------------------------------------------------------- #

# Two-level question-length design: a Pydantic field_validator runs at
# parse time, before any Settings instance is available, so it can only
# ever enforce the fixed ABSOLUTE_MAX_QUESTION_LENGTH ceiling -- never the
# smaller, administrator-configured operational limit
# (Settings.copilot_max_question_length). The route
# (plateproof.api.routes.copilot) enforces that tighter operational limit
# afterward, and CopilotService enforces it again as defense in depth.
# Settings itself is validated to never allow a configured limit above
# this ceiling (see plateproof.core.config).


class CopilotQueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    restaurant_id: str = Field(min_length=1, max_length=128)
    question: str

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        sanitized, reason = sanitize_question(value, max_length=ABSOLUTE_MAX_QUESTION_LENGTH)
        if sanitized is None:
            assert reason is not None
            raise ValueError(reason)
        return sanitized


class PublicClaim(BaseModel):
    """A safe, stable public projection of a Claim -- never the internal
    ``values`` mapping. ``text`` is produced by the same deterministic
    renderer (``plateproof.copilot.rendering.render_claim``) that builds
    the full answer_text, so it is never a second, independently-written
    description of the same fact."""

    model_config = ConfigDict(frozen=True)

    claim_type: str
    jurisdiction: str
    as_of_date: date
    text: str
    evidence_ids: tuple[str, ...]
    provenance_category: str


class PublicCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    citation_id: str
    evidence_category: str
    title: str
    url: str | None
    excerpt: str
    jurisdiction: str
    as_of_date: date | None
    superseded: bool
    issuing_authority: str | None = None
    section_locator: str | None = None
    access_date: date | None = None
    effective_date: date | None = None
    revision_date: date | None = None


class PublicRefusal(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    detail: str
    supported_intents_hint: tuple[str, ...] = ()


class CopilotQueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    restaurant_id: str
    jurisdiction: Jurisdiction | None
    intent: str | None
    grounding_status: Literal["grounded", "refused"]
    answer_text: str
    claims: list[PublicClaim]
    citations: list[PublicCitation]
    refusal: PublicRefusal | None
    generator_mode: Literal["deterministic", "local_llm_assisted"]
    local_helper_status: str
    warnings: list[str]
    generated_at: datetime
    disclaimer: str


# --------------------------------------------------------------------------- #
# Task 9B: owner document extraction (POST /owners/documents/extract).       #
# Every field below is a bounded, public-safe projection of an              #
# ``ExtractionDraft`` (see plateproof.api.documents_projection) -- never the  #
# raw upload bytes, a filesystem path, or a native-parser exception         #
# message. Full OCR text is never exposed: only the same short, bounded     #
# evidence excerpt already used to build each Candidate.                    #
# --------------------------------------------------------------------------- #


class BoundingBoxPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class EvidenceSpanPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    page: int
    excerpt: str
    bounding_box: BoundingBoxPublic | None
    source: Literal["embedded_text", "ocr"]


class CandidatePublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_name: str
    value: Any
    display_value: str | None
    evidence: list[EvidenceSpanPublic]
    confidence_label: Literal["high", "needs_review"]


class ViolationRowPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    raw_code_text: str
    code: str | None
    description: str | None
    critical: bool | None
    evidence: list[EvidenceSpanPublic]
    confidence_label: Literal["high", "needs_review"]


class AmbiguityPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_name: str
    reason: str
    candidate_values: list[str]


class DocumentWarningPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str


class PageMetadataPublic(BaseModel):
    """``preview_png_base64`` is the same worker-generated, worker-validated
    PNG bytes as ``PageMetadata.preview_png`` (Task 9A, Finding 7), base64-
    encoded only so it can travel inside a JSON response body -- never a
    second, independent render of the original upload."""

    model_config = ConfigDict(frozen=True)

    page_number: int
    width_px: int
    height_px: int
    used_ocr: bool
    preview_png_base64: str | None


class OcrEngineInfoPublic(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine_name: Literal["rapidocr"]
    available: bool
    unavailable_reason: str | None


class DocumentExtractionResponse(BaseModel):
    """The complete, public-safe projection of one ``ExtractionDraft``.
    Never persisted server-side -- this response is the only place the
    result of one extraction job exists once the request completes."""

    model_config = ConfigDict(frozen=True)

    draft_id: str
    jurisdiction_expected: Jurisdiction
    jurisdiction_detected: Jurisdiction | None
    jurisdiction_mismatch: bool
    restaurant_id: str
    restaurant_identity_corroborated: bool
    upload_media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    upload_byte_size: int
    upload_page_count: int
    pages: list[PageMetadataPublic]
    candidates: dict[str, CandidatePublic]
    violations: list[ViolationRowPublic]
    ambiguities: list[AmbiguityPublic]
    missing_fields: list[str]
    warnings: list[DocumentWarningPublic]
    ocr_engine: OcrEngineInfoPublic
    processing_status: Literal["completed", "ocr_unavailable", "failed"]
    generated_at: datetime
    confirmable: bool
    disclaimer: str
