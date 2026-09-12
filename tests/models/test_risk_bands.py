"""Tests for absolute risk-band thresholds and the insufficient-history gate."""

from __future__ import annotations


def test_absolute_boundaries_at_025_and_050() -> None:
    from plateproof.models.risk_bands import RiskBandThresholds, assign_risk_band

    thresholds = RiskBandThresholds(jurisdiction="nyc", model_version="v1")
    assert assign_risk_band(0.0, thresholds) == "low"
    assert assign_risk_band(0.2499, thresholds) == "low"
    assert assign_risk_band(0.25, thresholds) == "moderate"
    assert assign_risk_band(0.4999, thresholds) == "moderate"
    assert assign_risk_band(0.5, thresholds) == "high"
    assert assign_risk_band(1.0, thresholds) == "high"


def test_insufficient_history_band_for_none_probability() -> None:
    from plateproof.models.risk_bands import RiskBandThresholds, assign_risk_band

    thresholds = RiskBandThresholds(jurisdiction="nyc", model_version="v1")
    assert assign_risk_band(None, thresholds) == "insufficient_history"


def test_thresholds_stored_independently_per_jurisdiction() -> None:
    from plateproof.models.risk_bands import RiskBandThresholds

    nyc = RiskBandThresholds(jurisdiction="nyc", model_version="v1")
    fl = RiskBandThresholds(jurisdiction="florida", model_version="v1")
    assert nyc.jurisdiction != fl.jurisdiction
    assert nyc.low_upper_bound == fl.low_upper_bound == 0.25
    assert "not" in nyc.policy.lower() or "presentation" in nyc.policy.lower()


def test_nyc_insufficient_history_requires_valid_prior_score() -> None:
    from plateproof.models.risk_bands import has_sufficient_history

    ok, reason = has_sufficient_history(
        "nyc", {"missing_history": False, "nyc_prior_valid_score_count": 0}
    )
    assert ok is False
    assert reason == "no_valid_prior_score"

    ok2, reason2 = has_sufficient_history(
        "nyc", {"missing_history": False, "nyc_prior_valid_score_count": 1}
    )
    assert ok2 is True
    assert reason2 is None

    ok3, reason3 = has_sufficient_history(
        "nyc", {"missing_history": True, "nyc_prior_valid_score_count": None}
    )
    assert ok3 is False
    assert reason3 == "no_prior_inspection_event"


def test_florida_insufficient_history_requires_valid_prior_high_priority_count() -> None:
    from plateproof.models.risk_bands import has_sufficient_history

    ok, reason = has_sufficient_history(
        "florida", {"missing_history": False, "fl_prior_valid_high_priority_count": 0}
    )
    assert ok is False
    assert reason == "no_valid_prior_high_priority_count"

    ok2, _ = has_sufficient_history(
        "florida", {"missing_history": False, "fl_prior_valid_high_priority_count": 2}
    )
    assert ok2 is True


def test_missing_probability_and_bounds_for_insufficient_history() -> None:
    from plateproof.models.risk_bands import RiskBandThresholds, score_row

    thresholds = RiskBandThresholds(jurisdiction="nyc", model_version="v1")
    result = score_row(
        probability=None,
        lower=None,
        upper=None,
        features_row={"missing_history": True, "nyc_prior_valid_score_count": None},
        jurisdiction="nyc",
        thresholds=thresholds,
    )
    assert result.risk_band == "insufficient_history"
    assert result.probability is None
    assert result.lower_bound is None
    assert result.upper_bound is None
    assert result.insufficient_history_reason is not None
