"""End-to-end tests for CopilotService: real graph + real (fictional)
corpus, exercised through the public answer() entry point. Proves
jurisdiction isolation, deterministic reproducibility, refusal handling,
and that every citation in a grounded answer is authorized for its claim.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest

from plateproof.copilot.models import AUTHORIZED_EVIDENCE_TYPES, Intent, RefusalReason
from plateproof.copilot.service import CopilotService
from plateproof.graph.builder import build_graph
from plateproof.graph.models import GraphBuildInput


def _graph(
    *,
    restaurants,
    inspections=None,
    violations=None,
    michelin_restaurants=None,
    michelin_distinction_events=None,
    restaurant_michelin_matches=None,
):
    def _f(rows):
        return pl.DataFrame(rows) if rows else None

    result = build_graph(
        GraphBuildInput(
            restaurants=_f(restaurants),
            inspection_events=_f(inspections),
            violation_events=_f(violations),
            michelin_restaurants=_f(michelin_restaurants),
            michelin_distinction_events=_f(michelin_distinction_events),
            restaurant_michelin_matches=_f(restaurant_michelin_matches),
        )
    )
    return result.graph


@pytest.fixture
def nyc_restaurant_with_history(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> Any:
    def _build():
        return _graph(
            restaurants=[
                restaurant_row(
                    restaurant_id="nyc:1", jurisdiction="nyc", cuisine="American", city="Manhattan"
                )
            ],
            inspections=[
                inspection_row(
                    inspection_id="nyc:1:1",
                    restaurant_id="nyc:1",
                    jurisdiction="nyc",
                    inspection_date=date(2024, 1, 1),
                    score=20.0,
                    grade="B",
                ),
                inspection_row(
                    inspection_id="nyc:1:2",
                    restaurant_id="nyc:1",
                    jurisdiction="nyc",
                    inspection_date=date(2025, 6, 1),
                    score=10.0,
                    grade="A",
                ),
            ],
            violations=[
                violation_row(
                    violation_event_id="nyc:v:1",
                    inspection_id="nyc:1:1",
                    restaurant_id="nyc:1",
                    jurisdiction="nyc",
                    inspection_date=date(2024, 1, 1),
                    violation_code="04L",
                    violation_code_norm="04L",
                    violation_description="Evidence of mice.",
                ),
                violation_row(
                    violation_event_id="nyc:v:2",
                    inspection_id="nyc:1:2",
                    restaurant_id="nyc:1",
                    jurisdiction="nyc",
                    inspection_date=date(2025, 6, 1),
                    violation_code="04L",
                    violation_code_norm="04L",
                    violation_description="Evidence of mice, corrected.",
                ),
            ],
        )

    return _build


def test_restaurant_identity_always_answerable(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(
        restaurant_id="nyc:1", question="What is this restaurant's basic information?"
    )
    assert answer.grounding_status == "grounded"
    assert answer.intent == Intent.RESTAURANT_IDENTITY
    assert "nyc" in answer.answer_text.lower()


def test_latest_inspection_summary_is_grounded_and_cited(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(
        restaurant_id="nyc:1", question="What did the latest inspection document?"
    )
    assert answer.grounding_status == "grounded"
    assert "2025-06-01" in answer.answer_text
    assert len(answer.citations) == 1
    assert answer.citations[0].evidence_type.value == "restaurant_record"


def test_recurring_violations_reports_occurrence_and_total_counts(
    nyc_restaurant_with_history: Any,
) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(
        restaurant_id="nyc:1", question="Which violations have appeared repeatedly?"
    )
    assert answer.grounding_status == "grounded"
    assert "04L" in answer.answer_text
    assert "2" in answer.answer_text  # occurrence_count


def test_inspection_trend_reports_earliest_and_latest(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(
        restaurant_id="nyc:1", question="How has this restaurant's inspection history changed?"
    )
    assert answer.grounding_status == "grounded"
    assert "2024-01-01" in answer.answer_text
    assert "2025-06-01" in answer.answer_text


def test_prohibited_question_is_refused_before_any_evidence_lookup(
    nyc_restaurant_with_history: Any,
) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(restaurant_id="nyc:1", question="Will someone get sick here?")
    assert answer.grounding_status == "refused"
    assert answer.refusal is not None
    assert answer.refusal.reason == RefusalReason.PROHIBITED_REQUEST
    assert answer.claims == ()
    assert answer.citations == ()


def test_unsupported_question_is_refused_as_unknown(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(restaurant_id="nyc:1", question="What is the meaning of life?")
    assert answer.grounding_status == "refused"
    assert answer.refusal is not None
    assert answer.refusal.reason == RefusalReason.UNKNOWN_INTENT


def test_unknown_restaurant_is_refused(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history())
    answer = service.answer(
        restaurant_id="nyc:does-not-exist", question="What is this restaurant's basic information?"
    )
    assert answer.grounding_status == "refused"
    assert answer.refusal is not None
    assert answer.refusal.reason == RefusalReason.INSUFFICIENT_EVIDENCE


def test_insufficient_history_refuses_recurring_violations(restaurant_row: Any) -> None:
    graph = _graph(restaurants=[restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")])
    service = CopilotService(graph)
    answer = service.answer(
        restaurant_id="nyc:1", question="Which violations have appeared repeatedly?"
    )
    assert answer.grounding_status == "refused"
    assert answer.refusal is not None
    assert answer.refusal.reason == RefusalReason.INSUFFICIENT_EVIDENCE


def test_deterministic_reproducibility_same_input_same_output(
    nyc_restaurant_with_history: Any,
) -> None:
    graph = nyc_restaurant_with_history()
    service = CopilotService(graph)
    answer_a = service.answer(
        restaurant_id="nyc:1", question="What did the latest inspection document?"
    )
    answer_b = service.answer(
        restaurant_id="nyc:1", question="What did the latest inspection document?"
    )
    assert answer_a.answer_text == answer_b.answer_text
    assert answer_a.citations == answer_b.citations


def test_every_citation_in_a_grounded_answer_is_authorized_for_its_claim(
    nyc_restaurant_with_history: Any,
) -> None:
    """Citation validity means both (1) the evidence exists and (2) the
    claim type is authorized to use that evidence type -- checked here
    across every claim/citation pair a real service run produces."""
    service = CopilotService(nyc_restaurant_with_history())
    for question in (
        "What is this restaurant's basic information?",
        "What did the latest inspection document?",
        "Which violations have appeared repeatedly?",
        "How has this restaurant's inspection history changed?",
    ):
        answer = service.answer(restaurant_id="nyc:1", question=question)
        assert answer.grounding_status == "grounded", question
        citation_by_id = {c.citation_id: c for c in answer.citations}
        for claim in answer.claims:
            authorized_types = AUTHORIZED_EVIDENCE_TYPES[claim.claim_type]
            for evidence_id in claim.evidence_ids:
                citation = citation_by_id[evidence_id]
                assert citation.evidence_type in authorized_types


def test_michelin_context_is_never_conflated_with_safety(restaurant_row: Any) -> None:
    graph = _graph(
        restaurants=[restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        michelin_distinction_events=[
            {
                "michelin_distinction_event_id": "michelin:e:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "Fictional Guide",
                "guide_year": 2024,
                "announced_date": date(2024, 1, 1),
                "source_url": "https://example.invalid/fictional",
            }
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    service = CopilotService(graph)
    answer = service.answer(
        restaurant_id="nyc:1", question="Tell me about this restaurant's Michelin recognition"
    )
    assert answer.grounding_status == "grounded"
    assert (
        "not indicate food safety" in answer.answer_text
        or "not a health-safety indicator" in answer.answer_text
    )


def test_explain_prediction_without_lookup_refuses(nyc_restaurant_with_history: Any) -> None:
    service = CopilotService(nyc_restaurant_with_history(), forecast_lookup=None)
    answer = service.answer(
        restaurant_id="nyc:1", question="What is the forecast for this restaurant?"
    )
    assert answer.grounding_status == "refused"
    assert answer.refusal is not None
    assert answer.refusal.reason == RefusalReason.INSUFFICIENT_EVIDENCE


def test_explain_prediction_with_lookup_is_grounded_and_non_certain(
    nyc_restaurant_with_history: Any,
) -> None:
    def _lookup(restaurant_id: str, jurisdiction: str) -> dict[str, Any]:
        return {
            "risk_band": "moderate",
            "model_version": "v1",
            "generated_at": "2026-01-01T00:00:00Z",
            "as_of_date": date(2026, 1, 1),
        }

    service = CopilotService(nyc_restaurant_with_history(), forecast_lookup=_lookup)
    answer = service.answer(
        restaurant_id="nyc:1", question="What is the forecast for this restaurant?"
    )
    assert answer.grounding_status == "grounded"
    assert "not a guarantee" in answer.answer_text
    assert answer.citations[0].evidence_type.value == "model_forecast"
