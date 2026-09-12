"""Typed request/response contracts for the Task 7 API.

Every prediction response is a discriminated union on ``available`` --
there is no schema shape in which an "unavailable" response can also carry
a probability.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    status: Literal["ok", "unavailable", "disabled", "empty"]
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
