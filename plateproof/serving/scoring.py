"""The offline scoring engine: the only place a Task 6 model artifact is ever
deserialized or run. API requests and Streamlit page loads must never import
this module for anything but reading its already-written output tables.

``run_offline_scoring`` never trains, calibrates, or mutates an artifact --
it only loads one (``trusted=True, require_ready=True``), builds as-of
feature rows via Task 5's shared engine, and applies the artifact's own
stored point estimator, bootstrap members, and risk-band thresholds.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from plateproof.features.temporal import build_asof_temporal_features
from plateproof.models.risk_bands import RiskBandThresholds, has_sufficient_history, score_row
from plateproof.models.training import (
    UncertaintyConfig,
    deserialize_bootstrap_members,
    load_artifact,
    predict_with_uncertainty,
)

PREDICTIONS_SCHEMA: dict[str, Any] = {
    "prediction_id": pl.String,
    "restaurant_id": pl.String,
    "jurisdiction": pl.String,
    "target_name": pl.String,
    "model_version": pl.String,
    "artifact_schema_version": pl.String,
    "as_of_date": pl.Date,
    "generated_at": pl.Datetime("us", "UTC"),
    "probability": pl.Float64,
    "lower_bound": pl.Float64,
    "upper_bound": pl.Float64,
    "risk_band": pl.String,
    "insufficient_history_reason": pl.String,
    "calibration_status": pl.String,
    "uncertainty_status": pl.String,
    "readiness_status": pl.String,
}

MODEL_REGISTRY_SCHEMA: dict[str, Any] = {
    "jurisdiction": pl.String,
    "target_name": pl.String,
    "model_version": pl.String,
    "artifact_schema_version": pl.String,
    "deployment_status": pl.String,
    "registered_at": pl.Date,
    "source_snapshot_date": pl.Date,
    "artifact_path": pl.String,
}


def compute_prediction_id(
    *,
    restaurant_id: str,
    jurisdiction: str,
    target_name: str,
    model_version: str,
    as_of_date: date,
    artifact_schema_version: str,
) -> str:
    """Deterministic id from every field that changes the meaning of a
    prediction row -- restaurant, jurisdiction, target, model version, as-of
    date, and artifact schema version. ``target_name`` is included even
    though Task 7 currently serves only one target per jurisdiction: a
    second target must never collide with the first."""
    key = "\x1f".join(
        [
            restaurant_id,
            jurisdiction,
            target_name,
            model_version,
            as_of_date.isoformat(),
            artifact_schema_version,
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return f"pred:{digest}"


@dataclass(frozen=True)
class ScoringReport:
    restaurant_count: int
    sufficient_history_count: int
    insufficient_history_count: int


@dataclass(frozen=True)
class ScoringResult:
    predictions: list[dict[str, Any]]
    registry_entry: dict[str, Any]
    report: ScoringReport


def run_offline_scoring(
    *,
    artifact_path: Path,
    jurisdiction: str,
    target_name: str,
    events: pl.DataFrame,
    violations: pl.DataFrame,
    as_of_date: date,
) -> ScoringResult:
    """Score every restaurant in ``events`` as of ``as_of_date``.

    ``as_of_date`` is required and has no default -- callers (notably
    ``scripts/score_predictions.py``) must always supply an explicit real
    calendar date, never "today" implicitly.
    """
    loaded = load_artifact(
        artifact_path, trusted=True, expected_jurisdiction=jurisdiction, require_ready=True
    )

    loaded_target = loaded["target_definition.json"]["target_name"]
    if loaded_target != target_name:
        raise ValueError(
            f"artifact target mismatch: requested target {target_name!r}, "
            f"artifact is for target {loaded_target!r}"
        )

    feature_order = tuple(loaded["feature_order.json"])
    artifact_schema_version = loaded["artifact_schema_version.json"]
    model_version = loaded["manifest"]["model_version"]

    asof_result = build_asof_temporal_features(events, violations, as_of_date)
    feature_frame = asof_result.features
    missing = [name for name in feature_order if name not in feature_frame.columns]
    if missing:
        raise ValueError(f"as-of feature frame is missing required column(s): {missing}")

    X = feature_frame.select(list(feature_order)).to_numpy()
    restaurant_ids = feature_frame.get_column("restaurant_id").to_list()

    point_estimator = loaded["point_estimator.joblib"]
    members = deserialize_bootstrap_members(loaded["bootstrap_members.joblib"])
    uncertainty_config = UncertaintyConfig(**loaded["uncertainty_config.json"])
    thresholds = RiskBandThresholds(**loaded["risk_band_thresholds.json"])
    calibration_status = loaded["calibration_report.json"]["status"]
    readiness_status = loaded["deployment_status.json"]["status"]
    generated_at = datetime.now(UTC)

    predictions: list[dict[str, Any]] = []
    sufficient_count = 0
    insufficient_count = 0
    feature_rows = feature_frame.to_dicts()
    for i, restaurant_id in enumerate(restaurant_ids):
        features_row = feature_rows[i]
        sufficient, reason = has_sufficient_history(jurisdiction, features_row)
        if sufficient:
            point, lower, upper = predict_with_uncertainty(
                point_estimator, members, uncertainty_config, X[i : i + 1]
            )
            sufficient_count += 1
        else:
            point, lower, upper = None, None, None
            insufficient_count += 1
        scored = score_row(point, lower, upper, features_row, jurisdiction, thresholds)
        predictions.append(
            {
                "prediction_id": compute_prediction_id(
                    restaurant_id=restaurant_id,
                    jurisdiction=jurisdiction,
                    target_name=target_name,
                    model_version=model_version,
                    as_of_date=as_of_date,
                    artifact_schema_version=artifact_schema_version,
                ),
                "restaurant_id": restaurant_id,
                "jurisdiction": jurisdiction,
                "target_name": target_name,
                "model_version": model_version,
                "artifact_schema_version": artifact_schema_version,
                "as_of_date": as_of_date,
                "generated_at": generated_at,
                "probability": scored.probability,
                "lower_bound": scored.lower_bound,
                "upper_bound": scored.upper_bound,
                "risk_band": scored.risk_band,
                "insufficient_history_reason": scored.insufficient_history_reason,
                "calibration_status": calibration_status,
                "uncertainty_status": uncertainty_config.status.value,
                "readiness_status": readiness_status,
            }
        )

    registry_entry = {
        "jurisdiction": jurisdiction,
        "target_name": target_name,
        "model_version": model_version,
        "artifact_schema_version": artifact_schema_version,
        "deployment_status": readiness_status,
        "registered_at": generated_at.date(),
        "source_snapshot_date": as_of_date,
        "artifact_path": str(artifact_path),
    }
    report = ScoringReport(
        restaurant_count=len(restaurant_ids),
        sufficient_history_count=sufficient_count,
        insufficient_history_count=insufficient_count,
    )
    return ScoringResult(predictions=predictions, registry_entry=registry_entry, report=report)


def write_predictions_and_registry(
    output_dir: Path, predictions: list[dict[str, Any]], registry_entry: dict[str, Any]
) -> None:
    """Write ``predictions.parquet`` and ``model_registry.parquet``
    atomically: both are built as ``*.part`` files first, and only replace
    the previous files (via rename) once both have been written
    successfully. A failure partway through leaves the previous complete
    pair of tables untouched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_final = output_dir / "predictions.parquet"
    registry_final = output_dir / "model_registry.parquet"
    predictions_tmp = output_dir / "predictions.parquet.part"
    registry_tmp = output_dir / "model_registry.parquet.part"

    existing_registry = (
        pl.read_parquet(registry_final)
        if registry_final.exists()
        else pl.DataFrame(schema=MODEL_REGISTRY_SCHEMA)
    )
    merged_registry = pl.concat(
        [
            existing_registry.filter(pl.col("jurisdiction") != registry_entry["jurisdiction"]),
            pl.DataFrame([registry_entry], schema=MODEL_REGISTRY_SCHEMA),
        ]
    )

    try:
        pl.DataFrame(predictions, schema=PREDICTIONS_SCHEMA).write_parquet(predictions_tmp)
        merged_registry.write_parquet(registry_tmp)
    except BaseException:
        predictions_tmp.unlink(missing_ok=True)
        registry_tmp.unlink(missing_ok=True)
        raise

    predictions_tmp.replace(predictions_final)
    registry_tmp.replace(registry_final)
