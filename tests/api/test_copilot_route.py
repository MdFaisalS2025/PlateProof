"""RED-first tests for POST /copilot/query -- the real Task 8B endpoint
that replaces the deferred 501. Restaurant-scoped; jurisdiction is never
accepted from the client; grounded/refused answers are always 200; 404
only for an unknown restaurant; 422 for malformed input; 503 only for a
genuinely unavailable graph -- never merely because the local model is
disabled/unavailable."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest


def _write_graph_fixtures(
    processed_dir: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
    *,
    jurisdiction: str = "nyc",
    restaurant_id: str = "nyc:1",
) -> None:
    write_restaurants([restaurant_row(restaurant_id=restaurant_id, jurisdiction=jurisdiction)])
    write_inspections(
        [
            inspection_row(
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2024, 1, 1),
            ),
            inspection_row(
                inspection_id=f"{restaurant_id}:2",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2025, 6, 1),
            ),
        ]
    )
    write_violations(
        [
            violation_row(
                violation_event_id=f"{restaurant_id}:v:1",
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2024, 1, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
            violation_row(
                violation_event_id=f"{restaurant_id}:v:2",
                inspection_id=f"{restaurant_id}:2",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                inspection_date=date(2025, 6, 1),
                violation_code="04L",
                violation_code_norm="04L",
            ),
        ]
    )


@pytest.fixture
def nyc_client(
    make_client: Any,
    processed_dir: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> Any:
    _write_graph_fixtures(
        processed_dir,
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
        jurisdiction="nyc",
        restaurant_id="nyc:1",
    )
    return make_client(processed_data_dir=processed_dir, guidance_corpus_manifest_path=None)


@pytest.fixture
def florida_client(
    make_client: Any,
    processed_dir: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> Any:
    _write_graph_fixtures(
        processed_dir,
        write_restaurants,
        write_inspections,
        write_violations,
        restaurant_row,
        inspection_row,
        violation_row,
        jurisdiction="florida",
        restaurant_id="florida:1",
    )
    return make_client(processed_data_dir=processed_dir, guidance_corpus_manifest_path=None)


def test_grounded_nyc_response(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "What are the recurring violations?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "grounded"
    assert body["jurisdiction"] == "nyc"
    assert body["claims"]
    assert body["generator_mode"] == "deterministic"


def test_grounded_florida_response(florida_client: Any) -> None:
    response = florida_client.post(
        "/copilot/query",
        json={"restaurant_id": "florida:1", "question": "What are the recurring violations?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "grounded"
    assert body["jurisdiction"] == "florida"


def test_prohibited_request_is_a_typed_200_refusal(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "Will this make someone get sick?"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "refused"
    assert body["refusal"]["reason"] == "prohibited_request"


def test_unknown_intent_is_a_typed_200_refusal(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "asdkjfh qwoeiur"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "refused"
    assert body["refusal"]["reason"] == "unknown_intent"
    assert body["local_helper_status"] == "disabled"


def test_ambiguous_intent_is_a_typed_200_refusal(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "recurring violation history"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "refused"
    assert body["refusal"]["reason"] == "ambiguous_intent"


def test_unknown_restaurant_is_404(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:does-not-exist", "question": "What violations recur?"},
    )
    assert response.status_code == 404


def test_empty_question_is_422(nyc_client: Any) -> None:
    response = nyc_client.post("/copilot/query", json={"restaurant_id": "nyc:1", "question": "   "})
    assert response.status_code == 422


def test_oversized_question_is_422(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "a" * 10_000}
    )
    assert response.status_code == 422


def test_control_character_question_is_422(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "hello\x07world"}
    )
    assert response.status_code == 422


def test_client_supplied_jurisdiction_field_is_rejected_as_422(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={
            "restaurant_id": "nyc:1",
            "question": "What are the recurring violations?",
            "jurisdiction": "florida",
        },
    )
    assert response.status_code == 422


def test_local_model_disabled_is_never_a_503(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "asdkjfh qwoeiur"}
    )
    assert response.status_code == 200
    assert response.json()["local_helper_status"] == "disabled"


def test_graph_unavailable_is_503(
    processed_dir: Any,
    write_restaurants: Any,
    restaurant_row: Any,
) -> None:
    from fastapi.testclient import TestClient

    from plateproof.api.dependencies import get_graph_service
    from plateproof.api.main import create_app
    from plateproof.core.config import Settings
    from plateproof.graph.models import GraphScaleExceededError

    write_restaurants([restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")])

    class _AlwaysFailsGraphService:
        def get(self) -> Any:
            raise GraphScaleExceededError("simulated failure")

    settings = Settings(
        _env_file=None, processed_data_dir=processed_dir, guidance_corpus_manifest_path=None
    )
    app = create_app(settings)
    app.dependency_overrides[get_graph_service] = lambda: _AlwaysFailsGraphService()
    client = TestClient(app)
    response = client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "What violations recur?"}
    )
    assert response.status_code == 503
    # Sanitized: the exact exception type/message never reaches the client.
    assert "GraphScaleExceededError" not in response.text
    assert "simulated failure" not in response.text
    assert "Traceback" not in response.text


def test_corpus_unavailable_never_a_503_and_reports_guidance_unavailable(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={
            "restaurant_id": "nyc:1",
            "question": "What official guidance applies to these violation codes?",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["grounding_status"] == "grounded"
    assert any(c["claim_type"] == "guidance_unavailable" for c in body["claims"])


def test_response_projection_excludes_unsafe_internals(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "What are the recurring violations?"},
    )
    body = response.json()
    for claim in body["claims"]:
        assert "values" not in claim
        assert set(claim.keys()) == {
            "claim_type",
            "jurisdiction",
            "as_of_date",
            "text",
            "evidence_ids",
            "provenance_category",
        }


def test_response_never_leaks_local_paths_or_stack_traces(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "What are the recurring violations?"},
    )
    text = response.text
    assert "Traceback" not in text
    assert "C:\\" not in text
    assert "/home/" not in text
    assert "site-packages" not in text


def test_citation_jurisdiction_isolation(nyc_client: Any) -> None:
    response = nyc_client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "What are the recurring violations?"},
    )
    body = response.json()
    for citation in body["citations"]:
        assert citation["jurisdiction"] == "nyc"


def test_health_reports_local_ai_disabled_by_default(nyc_client: Any) -> None:
    response = nyc_client.get("/health")
    assert response.status_code == 200
    components = {c["name"]: c["status"] for c in response.json()["components"]}
    assert components["local_ai"] == "disabled"


def test_health_reports_local_ai_configured_unverified_when_enabled(
    make_client: Any, processed_dir: Any
) -> None:
    client = make_client(
        processed_data_dir=processed_dir,
        local_llm_enabled=True,
        local_llm_model="fictional-model",
    )
    response = client.get("/health")
    components = {c["name"]: c["status"] for c in response.json()["components"]}
    assert components["local_ai"] == "configured_unverified"


def test_health_reports_local_ai_unavailable_when_enabled_but_missing_model(
    make_client: Any, processed_dir: Any
) -> None:
    client = make_client(
        processed_data_dir=processed_dir, local_llm_enabled=True, local_llm_model=None
    )
    response = client.get("/health")
    components = {c["name"]: c["status"] for c in response.json()["components"]}
    assert components["local_ai"] == "unavailable"


def test_health_reports_local_ai_unavailable_when_enabled_with_invalid_url(
    make_client: Any, processed_dir: Any
) -> None:
    client = make_client(
        processed_data_dir=processed_dir,
        local_llm_enabled=True,
        local_llm_model="fictional-model",
        local_llm_base_url="http://evil.example.com:11434",
    )
    response = client.get("/health")
    components = {c["name"]: c["status"] for c in response.json()["components"]}
    assert components["local_ai"] == "unavailable"


def test_health_local_ai_unavailable_does_not_degrade_overall_status(
    make_client: Any, processed_dir: Any
) -> None:
    client = make_client(
        processed_data_dir=processed_dir, local_llm_enabled=True, local_llm_model=None
    )
    response = client.get("/health")
    # Local AI is optional -- its absence/misconfiguration alone must
    # never take the whole application's reported health down.
    assert response.json()["status"] != "unavailable"


# --------------------------------------------------------------------------- #
# Correction: reconcile the API's fixed absolute ceiling with the
# administrator-configured operational question-length limit.
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_nyc_client(
    make_client: Any,
    processed_dir: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
) -> Any:
    def _make(**overrides: Any) -> Any:
        _write_graph_fixtures(
            processed_dir,
            write_restaurants,
            write_inspections,
            write_violations,
            restaurant_row,
            inspection_row,
            violation_row,
            jurisdiction="nyc",
            restaurant_id="nyc:1",
        )
        return make_client(
            processed_data_dir=processed_dir, guidance_corpus_manifest_path=None, **overrides
        )

    return _make


def test_question_within_a_configured_limit_below_500_is_accepted(make_nyc_client: Any) -> None:
    client = make_nyc_client(copilot_max_question_length=50)
    response = client.post("/copilot/query", json={"restaurant_id": "nyc:1", "question": "a" * 40})
    assert response.status_code == 200


def test_question_exceeding_a_configured_limit_below_500_is_422(make_nyc_client: Any) -> None:
    client = make_nyc_client(copilot_max_question_length=50)
    response = client.post("/copilot/query", json={"restaurant_id": "nyc:1", "question": "a" * 100})
    assert response.status_code == 422


def test_question_within_a_configured_limit_above_500_is_accepted(make_nyc_client: Any) -> None:
    client = make_nyc_client(copilot_max_question_length=1500)
    response = client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "a" * 1000}
    )
    assert response.status_code == 200


def test_question_exceeding_a_configured_limit_above_500_is_422(make_nyc_client: Any) -> None:
    client = make_nyc_client(copilot_max_question_length=1500)
    response = client.post(
        "/copilot/query", json={"restaurant_id": "nyc:1", "question": "a" * 1600}
    )
    assert response.status_code == 422


def test_question_exceeding_the_absolute_ceiling_is_422_regardless_of_configured_limit(
    make_nyc_client: Any,
) -> None:
    from plateproof.copilot.question_validation import ABSOLUTE_MAX_QUESTION_LENGTH

    client = make_nyc_client(copilot_max_question_length=ABSOLUTE_MAX_QUESTION_LENGTH)
    response = client.post(
        "/copilot/query",
        json={"restaurant_id": "nyc:1", "question": "a" * (ABSOLUTE_MAX_QUESTION_LENGTH + 500)},
    )
    assert response.status_code == 422


def test_rejected_question_is_never_echoed_in_the_422_body(make_nyc_client: Any) -> None:
    client = make_nyc_client(copilot_max_question_length=50)
    marker = "SUPER_SECRET_MARKER_" + ("z" * 100)
    response = client.post("/copilot/query", json={"restaurant_id": "nyc:1", "question": marker})
    assert response.status_code == 422
    assert marker not in response.text
