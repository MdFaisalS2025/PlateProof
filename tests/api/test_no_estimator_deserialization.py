"""Proves the offline-only trust boundary: the running web application must
never deserialize or execute a Task 6 model artifact. Only
``plateproof/serving/scoring.py`` and the offline scoring CLI
(``scripts/score_predictions.py``) may call ``load_artifact``, ``joblib.load``,
or run an estimator's ``predict``/``predict_proba``.

Every test here patches the dangerous entry points to raise, then exercises
a real request path (FastAPI ``TestClient`` or Streamlit ``AppTest``) and
asserts the request still succeeds -- proving the patched function was never
actually called, not just asserting on source text.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest


def _forbidden_load_artifact(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError(
        "load_artifact must never be called by the running web application -- "
        "only by plateproof.serving.scoring / scripts/score_predictions.py"
    )


def _forbidden_joblib_load(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("joblib.load must never be called by the running web application")


@pytest.fixture
def forbid_deserialization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches every known deserialization entry point to raise. Any web
    request path that actually reaches one of these will fail loudly."""
    import joblib

    from plateproof.models import training as training_module

    monkeypatch.setattr(training_module, "load_artifact", _forbidden_load_artifact)
    monkeypatch.setattr(joblib, "load", _forbidden_joblib_load)


# --------------------------------------------------------------------------- #
# 1-3: prediction / health / model-card requests succeed without load_artifact #
# --------------------------------------------------------------------------- #


def test_prediction_request_succeeds_without_load_artifact(
    forbid_deserialization: None,
    tmp_path: Path,
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    """A REAL ready artifact is configured, and a matching precomputed
    prediction row exists -- with load_artifact patched to raise, a
    correctly-fixed prediction route must still return the AVAILABLE
    forecast (proving it never needed load_artifact), not silently degrade
    to unavailable."""
    import polars as pl

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    schema_version = json.loads(
        (artifact / "artifact_schema_version.json").read_text(encoding="utf-8")
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()

    pl.DataFrame(
        [
            {
                "prediction_id": "pred:1",
                "restaurant_id": "nyc:1",
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": manifest["model_version"],
                "artifact_schema_version": schema_version,
                "as_of_date": date.today(),
                "generated_at": date.today(),
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
    ).write_parquet(prediction_dir / "predictions.parquet")
    pl.DataFrame(
        [
            {
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": manifest["model_version"],
                "artifact_schema_version": schema_version,
                "deployment_status": "ready",
                "registered_at": date.today(),
                "source_snapshot_date": date.today(),
                "artifact_path": str(artifact),
            }
        ]
    ).write_parquet(prediction_dir / "model_registry.parquet")

    client = make_client(
        processed_data_dir=processed_dir,
        nyc_model_artifact_path=artifact,
        prediction_table_path=prediction_dir,
    )
    response = client.get("/restaurants/nyc:1/prediction")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True, body
    assert body["risk_band"] == "moderate"


def test_health_request_succeeds_without_load_artifact(
    forbid_deserialization: None,
    tmp_path: Path,
    processed_dir: Path,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    """A REAL ready artifact is configured; with load_artifact patched to
    raise, a correctly-fixed health route must still report the NYC model
    as available -- proving it never needed load_artifact."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    nyc_model = next(c for c in body["components"] if c["name"] == "nyc_model")
    assert nyc_model["status"] == "ok", body


def test_model_card_request_succeeds_without_load_artifact(
    forbid_deserialization: None,
    tmp_path: Path,
    processed_dir: Path,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/models/nyc/card")
    assert response.status_code == 200
    body = response.json()
    assert body["readiness_status"] == "ready"
    assert "PlateProof Model Card" in body["markdown"]


# --------------------------------------------------------------------------- #
# 4-5: Streamlit pages do not import/call artifact deserialization           #
# --------------------------------------------------------------------------- #


def test_inspection_history_page_succeeds_without_load_artifact(
    forbid_deserialization: None,
    tmp_path: Path,
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    build_ready_artifact: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    schema_version = json.loads(
        (artifact / "artifact_schema_version.json").read_text(encoding="utf-8")
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    import polars as pl

    pl.DataFrame(
        [
            {
                "prediction_id": "pred:1",
                "restaurant_id": "nyc:1",
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": manifest["model_version"],
                "artifact_schema_version": schema_version,
                "as_of_date": date.today(),
                "generated_at": date.today(),
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
    ).write_parquet(prediction_dir / "predictions.parquet")
    pl.DataFrame(
        [
            {
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": manifest["model_version"],
                "artifact_schema_version": schema_version,
                "deployment_status": "ready",
                "registered_at": date.today(),
                "source_snapshot_date": date.today(),
                "artifact_path": str(artifact),
            }
        ]
    ).write_parquet(prediction_dir / "model_registry.parquet")

    app_env(nyc_model_artifact_path=str(artifact), prediction_table_path=str(prediction_dir))

    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    at.text_input[0].set_value("nyc:1").run(timeout=30)
    assert not at.exception
    # a correctly-fixed page must show the real forecast, not fall back to
    # "no model configured" just because load_artifact was patched to raise
    combined = "\n".join(w.value for w in at.warning)
    assert "no ready plateproof model" not in combined.lower(), combined
    body_text = "\n".join(m.value for m in at.markdown)
    assert "moderate" in body_text.lower()


def test_model_card_page_succeeds_without_load_artifact(
    forbid_deserialization: None,
    tmp_path: Path,
    app_env: Any,
    app_path: Any,
    build_ready_artifact: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    app_env(nyc_model_artifact_path=str(artifact))

    at = AppTest.from_file(app_path("pages", "4_Model_Card.py"))
    at.run(timeout=30)
    assert not at.exception
    combined = "\n".join(w.value for w in at.warning)
    assert "no ready plateproof model" not in combined.lower(), combined
    body_text = "\n".join(m.value for m in at.markdown)
    assert "PlateProof Model Card" in body_text


# --------------------------------------------------------------------------- #
# 6: a malicious/invalid joblib file cannot be reached by the web application #
# --------------------------------------------------------------------------- #


def test_poisoned_point_estimator_with_valid_checksum_is_never_opened(
    tmp_path: Path,
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    """Replace the real point estimator with bytes that would raise if ever
    passed to joblib.load, but keep its manifest checksum consistent (so
    integrity checks pass on hash alone). Every route must still report the
    model as fully available -- proving none of them ever opens this file
    for deserialization."""
    import hashlib
    import json

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    poisoned = b"not a real joblib payload at all"
    (artifact / "point_estimator.joblib").write_bytes(poisoned)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["point_estimator.joblib"] = hashlib.sha256(poisoned).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)

    health = client.get("/health")
    assert health.status_code == 200
    nyc_model = next(c for c in health.json()["components"] if c["name"] == "nyc_model")
    assert nyc_model["status"] == "ok"

    card = client.get("/models/nyc/card")
    assert card.status_code == 200
    assert card.json()["readiness_status"] == "ready"


def test_poisoned_point_estimator_with_tampered_checksum_fails_closed(
    tmp_path: Path, processed_dir: Path, build_ready_artifact: Any, make_client: Any
) -> None:
    """A poisoned point-estimator file whose bytes do NOT match its manifest
    checksum (i.e. tampered without also forging the checksum) must be
    rejected outright -- fail-closed, never a crash, never served as ready."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    (artifact / "point_estimator.joblib").write_bytes(b"not a real joblib payload at all")

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)

    health = client.get("/health")
    assert health.status_code == 200
    nyc_model = next(c for c in health.json()["components"] if c["name"] == "nyc_model")
    assert nyc_model["status"] == "unavailable"

    card = client.get("/models/nyc/card")
    assert card.status_code == 404
