"""FastAPI route tests. Every fixture is tiny and fictional; the estimator is
never deserialized in the request path (there is no code path here that
could -- routes only ever call the repository and the cached model
summary)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Health                                                                       #
# --------------------------------------------------------------------------- #


def test_health_ok_with_no_optional_components(client: Any) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    names = {c["name"] for c in body["components"]}
    expected = {
        "datastore",
        "nyc_data",
        "florida_data",
        "michelin",
        "nyc_model",
        "florida_model",
        "google",
    }
    assert expected <= names


def test_health_degraded_state_does_not_fail_the_request(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants([restaurant_row()])
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    michelin = next(c for c in body["components"] if c["name"] == "michelin")
    assert michelin["status"] == "disabled"
    assert body["status"] in ("ok", "degraded")  # never "unavailable" for an optional gap


def test_health_never_leaks_a_path(client: Any) -> None:
    response = client.get("/health")
    assert str(client.app.state.settings.processed_data_dir) not in response.text


# --------------------------------------------------------------------------- #
# Search                                                                       #
# --------------------------------------------------------------------------- #


def test_search_by_name(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants(
        [
            restaurant_row(
                restaurant_id="nyc:1",
                name="Café José's Diner",
                normalized_name="cafe joses diner",
            )
        ]
    )
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants", params={"query": "cafe jose"})
    assert response.status_code == 200
    body = response.json()
    assert body["pagination"]["total"] == 1


def test_search_by_official_source_id_via_detail_route(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants([restaurant_row(restaurant_id="nyc:42", source_id="42")])
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants/nyc:42")
    assert response.status_code == 200
    assert response.json()["restaurant"]["restaurant_id"] == "nyc:42"


def test_jurisdiction_filter(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc", normalized_name="a"),
            restaurant_row(
                restaurant_id="florida:1", jurisdiction="florida", normalized_name="b", region="FL"
            ),
        ]
    )
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants", params={"jurisdiction": "florida"})
    body = response.json()
    assert body["pagination"]["total"] == 1
    assert body["results"][0]["jurisdiction"] == "florida"


def test_unsupported_jurisdiction_returns_422(client: Any) -> None:
    response = client.get("/restaurants", params={"jurisdiction": "texas"})
    assert response.status_code == 422


def test_page_size_over_limit_returns_422(client: Any) -> None:
    response = client.get("/restaurants", params={"limit": 999})
    assert response.status_code == 422


def test_negative_offset_returns_422(client: Any) -> None:
    response = client.get("/restaurants", params={"offset": -1})
    assert response.status_code == 422


def test_sql_injection_style_query_is_treated_as_text(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants", params={"query": "'; DROP TABLE restaurants; --"})
    assert response.status_code == 200
    assert response.json()["pagination"]["total"] == 0
    # table survives -- a second search still works
    assert client.get("/restaurants").json()["pagination"]["total"] == 1


def test_michelin_filter_unavailable_is_reported_not_ignored(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants([restaurant_row()])
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants", params={"michelin_category": "one_star"})
    body = response.json()
    assert body["optional_data_status"]["michelin"] == "unavailable"
    assert body["filters_applied"]["michelin_category"] is False
    assert any("michelin" in w.lower() for w in body["warnings"])


def test_michelin_accept_only_and_review_hidden(
    processed_dir: Path,
    write_restaurants: Any,
    write_michelin: Any,
    restaurant_row: Any,
    make_client: Any,
) -> None:
    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", normalized_name="accepted"),
            restaurant_row(restaurant_id="nyc:2", normalized_name="reviewed"),
        ]
    )
    write_michelin(
        restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:a",
                "name_as_published": "Accepted",
                "normalized_name": "accepted",
            }
        ],
        distinctions=[
            {
                "michelin_distinction_event_id": "michelin:e:a",
                "michelin_restaurant_id": "michelin:r:a",
                "distinction": "one_star",
                "guide_name": "MICHELIN Guide NYC",
                "guide_year": 2025,
                "announced_date": date(2025, 1, 1),
                "source_url": "https://example.org/guide",
                "source_publisher": "Example Publisher",
            }
        ],
        matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:a",
                "decision": "accept",
            },
            {
                "official_restaurant_id": "nyc:2",
                "michelin_restaurant_id": "michelin:r:a",
                "decision": "review",
            },
        ],
    )
    client = make_client(processed_data_dir=processed_dir)

    detail_accept = client.get("/restaurants/nyc:1").json()
    assert len(detail_accept["michelin_history"]) == 1
    assert "not evidence of food safety" in detail_accept["michelin_context_note"].lower()

    detail_review = client.get("/restaurants/nyc:2").json()
    assert detail_review["michelin_history"] == []
    assert detail_review["michelin_context_note"] is None


# --------------------------------------------------------------------------- #
# Restaurant detail / inspections / violations                                #
# --------------------------------------------------------------------------- #


def test_restaurant_not_found_returns_404(client: Any) -> None:
    response = client.get("/restaurants/nyc:999")
    assert response.status_code == 404
    body = response.json()
    assert "message" in body


def test_nyc_restaurant_detail_and_native_inspection_fields(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
    make_client: Any,
) -> None:
    write_restaurants([restaurant_row()])
    write_inspections([inspection_row()])
    client = make_client(processed_data_dir=processed_dir)

    detail = client.get("/restaurants/nyc:1")
    assert detail.status_code == 200
    assert "data.cityofnewyork.us" in detail.json()["official_source_links"][0]

    inspections = client.get("/restaurants/nyc:1/inspections").json()
    assert inspections[0]["score"] == 13.0
    assert inspections[0]["grade"] == "A"


def test_florida_restaurant_detail_and_no_letter_grade(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
    make_client: Any,
) -> None:
    write_restaurants(
        [restaurant_row(restaurant_id="florida:1", jurisdiction="florida", region="FL")]
    )
    write_inspections(
        [
            inspection_row(
                inspection_id="florida:v1",
                restaurant_id="florida:1",
                jurisdiction="florida",
                score=None,
                grade=None,
                critical_violation_count=None,
                high_priority_count=2,
                intermediate_count=1,
                basic_count=0,
            )
        ]
    )
    client = make_client(processed_data_dir=processed_dir)
    detail = client.get("/restaurants/florida:1")
    assert "myfloridalicense.com" in detail.json()["official_source_links"][0]

    inspections = client.get("/restaurants/florida:1/inspections").json()
    assert inspections[0]["high_priority_count"] == 2
    assert inspections[0]["grade"] is None


def test_inspection_ordering_newest_first(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
    make_client: Any,
) -> None:
    write_restaurants([restaurant_row()])
    write_inspections(
        [
            inspection_row(inspection_id="i1", inspection_date=date(2024, 1, 1)),
            inspection_row(inspection_id="i2", inspection_date=date(2025, 1, 1)),
        ]
    )
    client = make_client(processed_data_dir=processed_dir)
    inspections = client.get("/restaurants/nyc:1/inspections").json()
    assert [i["inspection_id"] for i in inspections] == ["i2", "i1"]


def test_violation_ordering_and_florida_wording(
    processed_dir: Path,
    write_restaurants: Any,
    write_violations: Any,
    restaurant_row: Any,
    violation_row: Any,
    make_client: Any,
) -> None:
    write_restaurants([restaurant_row(restaurant_id="florida:1", jurisdiction="florida")])
    write_violations(
        [
            violation_row(
                violation_event_id="v1",
                restaurant_id="florida:1",
                inspection_date=date(2024, 1, 1),
                violation_code="12",
                severity="other",
            ),
            violation_row(
                violation_event_id="v2",
                restaurant_id="florida:1",
                inspection_date=date(2025, 1, 1),
                violation_code="04L",
                severity="high_priority",
            ),
        ]
    )
    client = make_client(processed_data_dir=processed_dir)
    violations = client.get("/restaurants/florida:1/violations").json()
    assert [v["violation_code"] for v in violations] == ["04L", "12"]
    numbered = next(v for v in violations if v["violation_code"] == "12")
    assert numbered["severity"] == "classification_unavailable_at_category_level"


def test_generic_other_violation_keeps_plain_wording(
    processed_dir: Path,
    write_restaurants: Any,
    write_violations: Any,
    restaurant_row: Any,
    violation_row: Any,
    make_client: Any,
) -> None:
    write_restaurants([restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")])
    write_violations(
        [violation_row(violation_event_id="v1", restaurant_id="nyc:1", severity="other")]
    )
    client = make_client(processed_data_dir=processed_dir)
    violations = client.get("/restaurants/nyc:1/violations").json()
    assert violations[0]["severity"] == "other"


# --------------------------------------------------------------------------- #
# Predictions                                                                  #
# --------------------------------------------------------------------------- #


def test_prediction_unavailable_when_no_ready_model(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any, make_client: Any
) -> None:
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants/nyc:1/prediction")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "no_ready_model"


def test_prediction_available_from_precomputed_ready_record(
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

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    import polars as pl

    from plateproof.models.training import load_artifact

    loaded_manifest = load_artifact(artifact, trusted=True, expected_jurisdiction="nyc")
    pl.DataFrame(
        [
            {
                "prediction_id": "pred:1",
                "restaurant_id": "nyc:1",
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": loaded_manifest["manifest"]["model_version"],
                "artifact_schema_version": loaded_manifest["artifact_schema_version.json"],
                "as_of_date": date.today(),
                "generated_at": date.today(),
                "probability": 0.3,
                "lower_bound": 0.1,
                "upper_bound": 0.5,
                "risk_band": "moderate",
                "insufficient_history_reason": None,
                "calibration_status": loaded_manifest["calibration_report.json"]["status"],
                "uncertainty_status": loaded_manifest["uncertainty_config.json"]["status"],
                "readiness_status": "ready",
            }
        ]
    ).write_parquet(prediction_dir / "predictions.parquet")

    client = make_client(
        processed_data_dir=processed_dir,
        nyc_model_artifact_path=artifact,
        prediction_table_path=prediction_dir,
    )
    response = client.get("/restaurants/nyc:1/prediction")
    body = response.json()
    assert body["available"] is True
    assert body["risk_band"] == "moderate"
    assert "estimate" in body["disclaimer"].lower()


def test_prediction_unavailable_for_insufficient_history(
    tmp_path: Path,
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
    build_ready_artifact: Any,
    make_client: Any,
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.models.training import load_artifact

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    write_restaurants([restaurant_row(restaurant_id="nyc:1")])
    loaded_manifest = load_artifact(artifact, trusted=True, expected_jurisdiction="nyc")

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
                "model_version": loaded_manifest["manifest"]["model_version"],
                "artifact_schema_version": loaded_manifest["artifact_schema_version.json"],
                "as_of_date": date.today(),
                "generated_at": date.today(),
                "probability": None,
                "lower_bound": None,
                "upper_bound": None,
                "risk_band": "insufficient_history",
                "insufficient_history_reason": "no_prior_inspection_event",
                "calibration_status": loaded_manifest["calibration_report.json"]["status"],
                "uncertainty_status": loaded_manifest["uncertainty_config.json"]["status"],
                "readiness_status": "ready",
            }
        ]
    ).write_parquet(prediction_dir / "predictions.parquet")

    client = make_client(
        processed_data_dir=processed_dir,
        nyc_model_artifact_path=artifact,
        prediction_table_path=prediction_dir,
    )
    response = client.get("/restaurants/nyc:1/prediction")
    body = response.json()
    assert body["available"] is False
    assert body["reason"] == "insufficient_history"
    assert body["insufficient_history_reason"] == "no_prior_inspection_event"


def test_prediction_not_found_for_missing_restaurant(client: Any) -> None:
    response = client.get("/restaurants/nyc:missing/prediction")
    assert response.status_code == 404


# --------------------------------------------------------------------------- #
# Model card                                                                   #
# --------------------------------------------------------------------------- #


def test_model_card_not_found_when_no_artifact_configured(client: Any) -> None:
    response = client.get("/models/nyc/card")
    assert response.status_code == 404


def test_model_card_returned_for_ready_artifact(
    tmp_path: Path, processed_dir: Path, build_ready_artifact: Any, make_client: Any
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


def test_non_ready_model_card_hidden_by_default(
    tmp_path: Path, processed_dir: Path, make_client: Any
) -> None:
    from plateproof.models.training import write_artifact

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "uncalibrated", "reason": "test"},
        "target_definition.json": {"target_name": "nyc_next_initial_score_ge_14"},
        "model_card.md": "# Not ready",
    }
    artifact = write_artifact(
        tmp_path / "artifacts",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=bundle,
    )
    client = make_client(processed_data_dir=processed_dir, nyc_model_artifact_path=artifact)
    response = client.get("/models/nyc/card")
    assert response.status_code == 404


def test_non_ready_model_card_exposed_when_explicitly_configured(
    tmp_path: Path, processed_dir: Path, make_client: Any
) -> None:
    from plateproof.models.training import write_artifact

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "uncalibrated", "reason": "test"},
        "target_definition.json": {"target_name": "nyc_next_initial_score_ge_14"},
        "model_card.md": "# Not ready",
    }
    artifact = write_artifact(
        tmp_path / "artifacts",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=bundle,
    )
    client = make_client(
        processed_data_dir=processed_dir,
        nyc_model_artifact_path=artifact,
        expose_non_ready_model_cards=True,
    )
    response = client.get("/models/nyc/card")
    assert response.status_code == 200
    assert response.json()["readiness_status"] == "uncalibrated"


def test_model_card_never_exposes_artifact_path(
    tmp_path: Path, processed_dir: Path, build_ready_artifact: Any, make_client: Any
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
    assert str(artifact) not in response.text


# --------------------------------------------------------------------------- #
# Deferred routes / errors                                                     #
# --------------------------------------------------------------------------- #


def test_copilot_query_returns_501(client: Any) -> None:
    response = client.post("/copilot/query", json={})
    assert response.status_code == 501


def test_owner_document_extract_returns_501(client: Any) -> None:
    response = client.post("/owners/documents/extract", json={})
    assert response.status_code == 501


def test_error_body_has_no_traceback_or_path(processed_dir: Path, make_client: Any) -> None:
    client = make_client(processed_data_dir=processed_dir)
    response = client.get("/restaurants/not-a-valid-id")
    assert response.status_code == 422
    body = response.json()
    assert "Traceback" not in response.text
    assert str(processed_dir) not in response.text
    assert "message" in body
