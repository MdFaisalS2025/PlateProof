"""End-to-end tests for the guidance-dependent intents
(official_guidance_for_documented_codes, preparation_checklist_from_official_guidance)
against the REAL committed starter corpus -- proving cross-jurisdiction
isolation and honest "guidance unavailable" behavior at the full service
level, not just inside retrieval.py in isolation.

The real corpus has no exact violation-code mappings and no
preparation_action-tagged passages (see data/reference/guidance/README.md)
-- only topic-level definitions. These tests reflect that: they prove
GUIDANCE_FOR_TOPIC (never GUIDANCE_FOR_CODE) for a severity-matched
Florida violation, strict jurisdiction isolation, and an honest
GUIDANCE_UNAVAILABLE for the preparation checklist (since no reviewed
passage is curated as an actionable step).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from plateproof.copilot.corpus import load_corpus
from plateproof.copilot.models import ClaimType
from plateproof.copilot.service import CopilotService
from plateproof.graph.builder import build_graph
from plateproof.graph.models import GraphBuildInput

_GUIDANCE_MANIFEST = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "reference"
    / "guidance"
    / "manifest.json"
)


def _graph_for(
    restaurant_id: str,
    jurisdiction: str,
    severity: str,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
):
    restaurants = pl.DataFrame(
        [restaurant_row(restaurant_id=restaurant_id, jurisdiction=jurisdiction)]
    )
    inspections = pl.DataFrame(
        [
            inspection_row(
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
            )
        ]
    )
    violations = pl.DataFrame(
        [
            violation_row(
                violation_event_id=f"{restaurant_id}:v:1",
                inspection_id=f"{restaurant_id}:1",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                violation_code="99Z",
                violation_code_norm="99Z",
                severity=severity,
            )
        ]
    )
    result = build_graph(
        GraphBuildInput(
            restaurants=restaurants,
            inspection_events=inspections,
            violation_events=violations,
            michelin_restaurants=None,
            michelin_distinction_events=None,
            restaurant_michelin_matches=None,
        )
    )
    return result.graph


def test_florida_restaurant_gets_topic_guidance_never_code_guidance(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    graph = _graph_for(
        "florida:1", "florida", "high_priority", restaurant_row, inspection_row, violation_row
    )
    corpus_result = load_corpus(_GUIDANCE_MANIFEST)
    assert corpus_result.store is not None
    service = CopilotService(graph, corpus_result.store)

    answer = service.answer(
        restaurant_id="florida:1",
        question="What official guidance applies to these violation codes?",
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert ClaimType.GUIDANCE_FOR_TOPIC in claim_types
    assert ClaimType.GUIDANCE_FOR_CODE not in claim_types
    for citation in answer.citations:
        if citation.evidence_type.value == "guidance_passage":
            assert citation.jurisdiction == "florida"


def test_nyc_restaurant_never_retrieves_florida_topic_guidance(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """The Florida corpus has a passage tagged topic "high_priority" -- an
    NYC restaurant with the same severity classification must never
    receive it. NYC's own corpus document has no such topic, so this must
    resolve to an honest guidance-unavailable claim, never a cross-
    jurisdiction leak."""
    graph = _graph_for(
        "nyc:1", "nyc", "high_priority", restaurant_row, inspection_row, violation_row
    )
    corpus_result = load_corpus(_GUIDANCE_MANIFEST)
    assert corpus_result.store is not None
    service = CopilotService(graph, corpus_result.store)

    answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert claim_types == {ClaimType.GUIDANCE_UNAVAILABLE}
    for citation in answer.citations:
        assert citation.jurisdiction != "florida"


def test_guidance_unavailable_when_severity_has_no_topical_match(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    graph = _graph_for("nyc:1", "nyc", "critical", restaurant_row, inspection_row, violation_row)
    corpus_result = load_corpus(_GUIDANCE_MANIFEST)
    assert corpus_result.store is not None
    service = CopilotService(graph, corpus_result.store)
    answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert claim_types == {ClaimType.GUIDANCE_UNAVAILABLE}


def test_preparation_checklist_reports_guidance_unavailable_for_definition_only_corpus(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """The real corpus's Florida passages are curated for `definition`
    only, never `preparation_action` -- so a preparation checklist must
    still document the recurring violation, but its guidance portion must
    be an honest guidance-unavailable claim, never the topic definition
    repurposed as a checklist step."""
    graph = _graph_for(
        "florida:1", "florida", "high_priority", restaurant_row, inspection_row, violation_row
    )
    corpus_result = load_corpus(_GUIDANCE_MANIFEST)
    assert corpus_result.store is not None
    service = CopilotService(graph, corpus_result.store)
    answer = service.answer(
        restaurant_id="florida:1", question="What should I review before the next inspection?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert ClaimType.RECURRING_VIOLATION in claim_types
    assert ClaimType.GUIDANCE_UNAVAILABLE in claim_types
    assert ClaimType.GUIDANCE_FOR_TOPIC not in claim_types
    assert ClaimType.GUIDANCE_FOR_CODE not in claim_types


def test_guidance_intent_without_configured_corpus_reports_unavailable(
    restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    graph = _graph_for("nyc:1", "nyc", "critical", restaurant_row, inspection_row, violation_row)
    service = CopilotService(graph, corpus_store=None)
    answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert claim_types == {ClaimType.GUIDANCE_UNAVAILABLE}
