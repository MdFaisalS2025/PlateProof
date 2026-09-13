"""Malformed model-artifact metadata (corrupt manifest.json, tampered
allowlisted metadata files) must never surface as an HTTP 500 or leak a
local filesystem path / raw exception text through a public route or
Streamlit page -- it must fail closed to an honest, typed unavailable
response. See ``plateproof.serving.model_registry_service._read_metadata``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _corrupt_manifest_json(artifact: Path) -> None:
    (artifact / "manifest.json").write_text("{not valid json", encoding="utf-8")


def test_malformed_manifest_json_prediction_request_is_typed_not_500(
    tmp_path: Path,
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
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
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    _corrupt_manifest_json(artifact)

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/restaurants/nyc:1/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_ready_model"


def test_malformed_manifest_json_health_request_reports_unavailable(
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
    _corrupt_manifest_json(artifact)

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    nyc_model = next(c for c in body["components"] if c["name"] == "nyc_model")
    assert nyc_model["status"] == "unavailable"
    assert body["status"] in ("degraded", "unavailable")


def test_malformed_manifest_json_model_card_request_fails_safely(
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
    _corrupt_manifest_json(artifact)

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/models/nyc/card")

    assert response.status_code == 404
    assert str(artifact) not in response.text


def test_malformed_manifest_json_inspection_history_page_does_not_crash(
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
    _corrupt_manifest_json(artifact)

    app_env(nyc_model_artifact_path=str(artifact))

    at = AppTest.from_file(app_path("pages", "2_Inspection_History.py"))
    at.run(timeout=30)
    at.text_input[0].set_value("nyc:1").run(timeout=30)

    assert not at.exception


def test_malformed_manifest_json_model_card_page_does_not_crash(
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
    _corrupt_manifest_json(artifact)
    app_env(nyc_model_artifact_path=str(artifact))

    at = AppTest.from_file(app_path("pages", "4_Model_Card.py"))
    at.run(timeout=30)

    assert not at.exception


# --------------------------------------------------------------------------- #
# 13: public responses never contain a local path or raw exception text
# --------------------------------------------------------------------------- #


def test_health_response_never_leaks_path_or_exception_text(
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
    _corrupt_manifest_json(artifact)

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/health")

    assert str(artifact) not in response.text
    assert "Traceback" not in response.text
    assert "JSONDecodeError" not in response.text


def test_prediction_response_never_leaks_path_or_exception_text(
    tmp_path: Path,
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
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
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    _corrupt_manifest_json(artifact)

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/restaurants/nyc:1/prediction")

    assert str(artifact) not in response.text
    assert "Traceback" not in response.text
    assert "JSONDecodeError" not in response.text


def test_poisoned_absolute_manifest_entry_is_never_exposed_or_read(
    tmp_path: Path,
    processed_dir: Path,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    """A manifest that maps an allowlisted metadata name to an absolute
    filesystem path must be rejected by every public route -- never read,
    never surfaced."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("must never be read", encoding="utf-8")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][str(outside)] = manifest["files"].pop("deployment_status.json")
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)

    health = client.get("/health")
    assert health.status_code == 200
    nyc_model = next(c for c in health.json()["components"] if c["name"] == "nyc_model")
    assert nyc_model["status"] == "unavailable"
    assert str(outside) not in health.text

    card = client.get("/models/nyc/card")
    assert card.status_code == 404
    assert str(outside) not in card.text
