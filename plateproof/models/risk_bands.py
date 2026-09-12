"""Absolute, jurisdiction-independent presentation risk bands.

These are PlateProof presentation bands only -- they are not official grades,
are not comparable between NYC and Florida even when the numeric thresholds
match, and later versions may replace them with validation-derived,
jurisdiction-specific thresholds.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

RiskBand = Literal["low", "moderate", "high", "insufficient_history"]

_DEFAULT_POLICY = (
    "PlateProof presentation band only: not an official grade, not a health "
    "department determination, and not comparable across jurisdictions even "
    "when numeric thresholds match."
)


class RiskBandThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    jurisdiction: Literal["nyc", "florida"]
    model_version: str
    low_upper_bound: float = 0.25
    moderate_upper_bound: float = 0.50
    policy: str = _DEFAULT_POLICY


def assign_risk_band(probability: float | None, thresholds: RiskBandThresholds) -> RiskBand:
    if probability is None:
        return "insufficient_history"
    if probability < thresholds.low_upper_bound:
        return "low"
    if probability < thresholds.moderate_upper_bound:
        return "moderate"
    return "high"


def has_sufficient_history(
    jurisdiction: str, features_row: dict[str, Any]
) -> tuple[bool, str | None]:
    """NYC: requires any prior inspection AND at least one valid NYC score.
    Florida: requires any prior inspection AND at least one valid high-priority
    count. Both parts are required -- history without a jurisdiction-native
    measure is not sufficient."""
    if features_row.get("missing_history"):
        return False, "no_prior_inspection_event"
    if jurisdiction == "nyc":
        if not (features_row.get("nyc_prior_valid_score_count") or 0) >= 1:
            return False, "no_valid_prior_score"
        return True, None
    if jurisdiction == "florida":
        if not (features_row.get("fl_prior_valid_high_priority_count") or 0) >= 1:
            return False, "no_valid_prior_high_priority_count"
        return True, None
    raise ValueError(f"unknown jurisdiction: {jurisdiction!r}")


class ScoredRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_band: RiskBand
    probability: float | None
    lower_bound: float | None
    upper_bound: float | None
    insufficient_history_reason: str | None


def score_row(
    probability: float | None,
    lower: float | None,
    upper: float | None,
    features_row: dict[str, Any],
    jurisdiction: str,
    thresholds: RiskBandThresholds,
) -> ScoredRow:
    """Never fabricates a probability or interval when history is
    insufficient -- all three are forced to ``None`` regardless of caller
    input in that case."""
    sufficient, reason = has_sufficient_history(jurisdiction, features_row)
    if not sufficient:
        return ScoredRow(
            risk_band="insufficient_history",
            probability=None,
            lower_bound=None,
            upper_bound=None,
            insufficient_history_reason=reason,
        )
    return ScoredRow(
        risk_band=assign_risk_band(probability, thresholds),
        probability=probability,
        lower_bound=lower,
        upper_bound=upper,
        insufficient_history_reason=None,
    )
