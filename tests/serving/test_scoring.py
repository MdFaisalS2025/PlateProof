"""Tests for the offline scoring engine: the only place a Task 6 estimator is
ever deserialized or run. API requests and Streamlit page loads must never
reach this module -- they only read the Parquet tables it writes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest


def _nyc_events(rows: list[dict[str, Any]]) -> pl.DataFrame:
    from plateproof.features.inspection_events import INSPECTION_EVENT_SCHEMA, finalize_event_frame

    return finalize_event_frame(rows, INSPECTION_EVENT_SCHEMA)


def _event(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "inspection_id": None,
        "restaurant_id": None,
        "source_id": None,
        "jurisdiction": "nyc",
        "inspection_date": None,
        "inspection_type": "Cycle Inspection / Initial Inspection",
        "inspection_type_raw": "Cycle Inspection / Initial Inspection",
        "action": None,
        "action_conflict": False,
        "action_conflict_values": None,
        "score": 10.0,
        "score_conflict": False,
        "score_conflict_values": None,
        "grade": None,
        "grade_conflict": False,
        "grade_conflict_values": None,
        "grade_date": None,
        "violation_count": None,
        "critical_violation_count": 0,
        "high_priority_count": None,
        "intermediate_count": None,
        "basic_count": None,
        "dba": "Test Restaurant",
        "boro_raw": "Manhattan",
        "building": "1",
        "street": "MAIN ST",
        "zipcode": "10001",
        "cuisine_description": "American",
        "latitude": 40.7,
        "longitude": -73.9,
        "source_dataset": "test",
        "source_snapshot_date": date(2026, 1, 1),
        "source_retrieved_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "source_sha256": "abc",
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
        "disposition": None,
        "disposition_status": None,
        "native_inspection_group_id": None,
        "native_visit_sequence": None,
    }
    base.update(overrides)
    if base["inspection_id"] is None:
        base["inspection_id"] = f"{base['restaurant_id']}:{base['inspection_date']}:{id(base)}"
    return base


def test_as_of_date_is_required_with_no_default() -> None:
    import inspect

    from plateproof.serving.scoring import run_offline_scoring

    params = inspect.signature(run_offline_scoring).parameters
    assert params["as_of_date"].default is inspect.Parameter.empty


def test_insufficient_history_restaurant_gets_no_probability(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.scoring import run_offline_scoring

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    events = _nyc_events(
        [_event(restaurant_id="nyc:new", inspection_date=date(2025, 1, 1))]
    )  # a single (recent) event -> zero PRIOR history as of an earlier as_of_date
    result = run_offline_scoring(
        artifact_path=artifact,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        events=events,
        violations=_nyc_events([]).clear(),
        as_of_date=date(2025, 1, 1),
    )
    row = next(p for p in result.predictions if p["restaurant_id"] == "nyc:new")
    assert row["probability"] is None
    assert row["lower_bound"] is None
    assert row["upper_bound"] is None
    assert row["risk_band"] == "insufficient_history"
    assert row["insufficient_history_reason"]


def test_sufficient_history_restaurant_is_scored(tmp_path: Path, build_ready_artifact: Any) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.scoring import run_offline_scoring

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    events = _nyc_events(
        [
            _event(
                restaurant_id="nyc:est",
                inspection_id="e1",
                inspection_date=date(2024, 1, 1),
                score=5.0,
            ),
            _event(
                restaurant_id="nyc:est",
                inspection_id="e2",
                inspection_date=date(2024, 6, 1),
                score=8.0,
            ),
        ]
    )
    result = run_offline_scoring(
        artifact_path=artifact,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        events=events,
        violations=events.clear(),
        as_of_date=date(2025, 1, 1),
    )
    row = next(p for p in result.predictions if p["restaurant_id"] == "nyc:est")
    assert row["probability"] is not None
    assert 0.0 <= row["lower_bound"] <= row["probability"] <= row["upper_bound"] <= 1.0
    assert row["risk_band"] in ("low", "moderate", "high")
    assert row["as_of_date"] == date(2025, 1, 1)
    assert row["model_version"] == "v1"


def test_non_ready_artifact_is_refused(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact
    from plateproof.serving.scoring import run_offline_scoring

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "uncalibrated", "reason": "test"},
        "target_definition.json": {"target_name": "nyc_next_initial_score_ge_14"},
    }
    artifact = write_artifact(
        tmp_path / "artifacts",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=bundle,
    )
    with pytest.raises(ValueError, match="non-ready"):
        run_offline_scoring(
            artifact_path=artifact,
            jurisdiction="nyc",
            target_name="nyc_next_initial_score_ge_14",
            events=_nyc_events([]),
            violations=_nyc_events([]).clear(),
            as_of_date=date(2025, 1, 1),
        )


def test_wrong_jurisdiction_artifact_is_refused(tmp_path: Path, build_ready_artifact: Any) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.scoring import run_offline_scoring

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    with pytest.raises(ValueError):
        run_offline_scoring(
            artifact_path=artifact,
            jurisdiction="florida",
            target_name="nyc_next_initial_score_ge_14",
            events=_nyc_events([]),
            violations=_nyc_events([]).clear(),
            as_of_date=date(2025, 1, 1),
        )


def test_wrong_target_artifact_is_refused(tmp_path: Path, build_ready_artifact: Any) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.scoring import run_offline_scoring

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    with pytest.raises(ValueError, match="target"):
        run_offline_scoring(
            artifact_path=artifact,
            jurisdiction="nyc",
            target_name="nyc_next_score_ge_28",
            events=_nyc_events([]),
            violations=_nyc_events([]).clear(),
            as_of_date=date(2025, 1, 1),
        )


def test_prediction_id_includes_target_name() -> None:
    from plateproof.serving.scoring import compute_prediction_id

    common = {
        "restaurant_id": "nyc:1",
        "jurisdiction": "nyc",
        "model_version": "v1",
        "as_of_date": date(2025, 1, 1),
        "artifact_schema_version": "1.0.0",
    }
    id_a = compute_prediction_id(target_name="nyc_next_initial_score_ge_14", **common)
    id_b = compute_prediction_id(target_name="nyc_next_score_ge_28", **common)
    assert id_a != id_b


def test_prediction_id_deterministic() -> None:
    from plateproof.serving.scoring import compute_prediction_id

    kwargs = {
        "restaurant_id": "nyc:1",
        "jurisdiction": "nyc",
        "target_name": "nyc_next_initial_score_ge_14",
        "model_version": "v1",
        "as_of_date": date(2025, 1, 1),
        "artifact_schema_version": "1.0.0",
    }
    assert compute_prediction_id(**kwargs) == compute_prediction_id(**kwargs)


def test_scoring_never_invokes_model_training(
    tmp_path: Path, build_ready_artifact: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.scoring import run_offline_scoring

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("scoring must never call fit_and_select_model")

    monkeypatch.setattr(training_module, "fit_and_select_model", _forbidden)

    run_offline_scoring(
        artifact_path=artifact,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        events=_nyc_events([_event(restaurant_id="nyc:x", inspection_date=date(2025, 1, 1))]),
        violations=_nyc_events([]).clear(),
        as_of_date=date(2025, 1, 1),
    )  # must not raise


# --------------------------------------------------------------------------- #
# Atomic write of predictions + model_registry                                #
# --------------------------------------------------------------------------- #


def test_write_predictions_atomically_creates_both_tables(tmp_path: Path) -> None:
    from plateproof.serving.scoring import write_predictions_and_registry

    predictions = [
        {
            "prediction_id": "pred:1",
            "restaurant_id": "nyc:1",
            "jurisdiction": "nyc",
            "target_name": "nyc_next_initial_score_ge_14",
            "model_version": "v1",
            "artifact_schema_version": "1.0.0",
            "as_of_date": date(2025, 1, 1),
            "generated_at": datetime(2025, 1, 1, tzinfo=UTC),
            "probability": 0.3,
            "lower_bound": 0.1,
            "upper_bound": 0.5,
            "risk_band": "moderate",
            "insufficient_history_reason": None,
            "calibration_status": "calibrated_sigmoid",
            "uncertainty_status": "available",
            "readiness_status": "ready",
        }
    ]
    registry_entry = {
        "jurisdiction": "nyc",
        "target_name": "nyc_next_initial_score_ge_14",
        "model_version": "v1",
        "artifact_schema_version": "1.0.0",
        "deployment_status": "ready",
        "registered_at": date(2025, 1, 1),
        "source_snapshot_date": date(2025, 1, 1),
        "artifact_path": str(tmp_path / "artifacts"),
    }
    output_dir = tmp_path / "processed"
    write_predictions_and_registry(output_dir, predictions, registry_entry)

    assert (output_dir / "predictions.parquet").exists()
    assert (output_dir / "model_registry.parquet").exists()
    written = pl.read_parquet(output_dir / "predictions.parquet")
    assert written.height == 1


def test_write_predictions_atomically_preserves_previous_on_failure(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from plateproof.serving.scoring import write_predictions_and_registry

    output_dir = tmp_path / "processed"
    original_predictions = [
        {
            "prediction_id": "pred:orig",
            "restaurant_id": "nyc:1",
            "jurisdiction": "nyc",
            "target_name": "nyc_next_initial_score_ge_14",
            "model_version": "v1",
            "artifact_schema_version": "1.0.0",
            "as_of_date": date(2025, 1, 1),
            "generated_at": datetime(2025, 1, 1, tzinfo=UTC),
            "probability": 0.2,
            "lower_bound": 0.1,
            "upper_bound": 0.3,
            "risk_band": "low",
            "insufficient_history_reason": None,
            "calibration_status": "calibrated_sigmoid",
            "uncertainty_status": "available",
            "readiness_status": "ready",
        }
    ]
    registry_entry = {
        "jurisdiction": "nyc",
        "target_name": "nyc_next_initial_score_ge_14",
        "model_version": "v1",
        "artifact_schema_version": "1.0.0",
        "deployment_status": "ready",
        "registered_at": date(2025, 1, 1),
        "source_snapshot_date": date(2025, 1, 1),
        "artifact_path": "irrelevant",
    }
    write_predictions_and_registry(output_dir, original_predictions, registry_entry)

    real_write_parquet = pl.DataFrame.write_parquet

    def _flaky_write_parquet(self: Any, path: Any, *args: Any, **kwargs: Any) -> Any:
        if "model_registry" in str(path):
            raise RuntimeError("simulated disk failure")
        return real_write_parquet(self, path, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _flaky_write_parquet)

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        write_predictions_and_registry(
            output_dir,
            [{**original_predictions[0], "prediction_id": "pred:new", "probability": 0.9}],
            registry_entry,
        )

    monkeypatch.undo()
    reloaded = pl.read_parquet(output_dir / "predictions.parquet")
    assert reloaded.to_dicts()[0]["prediction_id"] == "pred:orig"
