"""Regression tests for the corrected guidance-authorization rules
(independent-review correction items 1 and 2):

* lexical (TF-IDF) similarity alone can never authorize GUIDANCE_FOR_CODE
  or GUIDANCE_FOR_TOPIC -- only an explicit, curated
  applicable_violation_codes/topics match can;
* a classification/definition passage can never become a preparation
  checklist step -- only a passage curated with
  permitted_uses=["preparation_action"] can.

All corpora here are fictional, built via write_guidance_corpus /
make_fictional_passage.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from plateproof.copilot.corpus import load_corpus
from plateproof.copilot.models import ClaimType
from plateproof.copilot.retrieval import search_by_topic_or_text
from plateproof.copilot.service import CopilotService
from plateproof.graph.builder import build_graph
from plateproof.graph.models import GraphBuildInput


def _graph_for(
    restaurant_id: str,
    jurisdiction: str,
    severity: str,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
    *,
    violation_code: str = "99Z",
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
                violation_code=violation_code,
                violation_code_norm=violation_code,
                severity=severity,
                violation_description=(
                    "Keep hot food above the documented safe temperature at all times "
                    "during service."
                ),
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


def test_high_tfidf_score_cannot_authorize_guidance_for_code(
    write_guidance_corpus: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """A passage whose text is near-identical to the violation's own
    documented description -- an almost-guaranteed high TF-IDF score --
    but carries no curated code or topic mapping must never become
    GUIDANCE_FOR_CODE or GUIDANCE_FOR_TOPIC."""
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#overview",
                        "section_locator": "Section 1",
                        "text": (
                            "Keep hot food above the documented safe temperature at all "
                            "times during service."
                        ),
                        "sha256": __import__("hashlib")
                        .sha256(
                            b"Keep hot food above the documented safe temperature at all "
                            b"times during service."
                        )
                        .hexdigest(),
                        "applicable_violation_codes": [],
                        "topics": [],
                        "permitted_uses": ["definition"],
                    }
                ],
            }
        ]
    )
    corpus_result = load_corpus(manifest_path)
    assert corpus_result.store is not None

    # Sanity check: this passage really would score highly under TF-IDF
    # for the violation's own description text -- proving the defect this
    # test guards against is a real, reachable condition, not a strawman.
    tfidf_hits = search_by_topic_or_text(
        corpus_result.store,
        "nyc",
        "Keep hot food above the documented safe temperature at all times during service.",
    )
    assert tfidf_hits and tfidf_hits[0].relevance_score > 0.9

    graph = _graph_for("nyc:1", "nyc", "critical", restaurant_row, inspection_row, violation_row)
    service = CopilotService(graph, corpus_result.store)
    answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert claim_types == {ClaimType.GUIDANCE_UNAVAILABLE}
    assert answer.citations == ()


def test_exact_reviewed_code_mapping_produces_guidance_for_code(
    write_guidance_corpus: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#overview",
                        "section_locator": "Section 1",
                        "text": "Fictional guidance mapped directly to code 04L.",
                        "sha256": __import__("hashlib")
                        .sha256(b"Fictional guidance mapped directly to code 04L.")
                        .hexdigest(),
                        "applicable_violation_codes": ["04L"],
                        "topics": [],
                        "permitted_uses": ["definition"],
                    }
                ],
            }
        ]
    )
    corpus_result = load_corpus(manifest_path)
    assert corpus_result.store is not None

    graph = _graph_for(
        "nyc:1",
        "nyc",
        "critical",
        restaurant_row,
        inspection_row,
        violation_row,
        violation_code="04L",
    )
    service = CopilotService(graph, corpus_result.store)
    answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert claim_types == {ClaimType.GUIDANCE_FOR_CODE}
    assert "fictional-doc#overview" in answer.claims[0].evidence_ids


def test_classification_definition_cannot_become_a_preparation_step(
    write_guidance_corpus: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """A passage curated only as a `definition` -- even with a matching
    topic -- must not surface in the preparation checklist."""
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#definition",
                        "section_locator": "Section 1",
                        "text": "High Priority violations are defined as those posing direct risk.",
                        "sha256": __import__("hashlib")
                        .sha256(
                            b"High Priority violations are defined as those posing direct risk."
                        )
                        .hexdigest(),
                        "applicable_violation_codes": [],
                        "topics": ["high_priority"],
                        "permitted_uses": ["definition"],
                    }
                ],
            }
        ]
    )
    corpus_result = load_corpus(manifest_path)
    assert corpus_result.store is not None

    graph = _graph_for(
        "nyc:1", "nyc", "high_priority", restaurant_row, inspection_row, violation_row
    )
    service = CopilotService(graph, corpus_result.store)

    # Sanity check: the same corpus DOES support a general
    # official-guidance answer via the topic match (proving the passage is
    # reachable at all) -- it is specifically the checklist path that must
    # refuse to use it.
    general_answer = service.answer(
        restaurant_id="nyc:1", question="What official guidance applies to these violation codes?"
    )
    assert ClaimType.GUIDANCE_FOR_TOPIC in {c.claim_type for c in general_answer.claims}

    checklist_answer = service.answer(
        restaurant_id="nyc:1", question="What should I review before the next inspection?"
    )
    assert checklist_answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in checklist_answer.claims}
    assert ClaimType.GUIDANCE_UNAVAILABLE in claim_types
    assert ClaimType.GUIDANCE_FOR_TOPIC not in claim_types
    assert ClaimType.GUIDANCE_FOR_CODE not in claim_types


def test_preparation_action_evidence_can_produce_a_checklist_item(
    write_guidance_corpus: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """The positive counterpart: a passage explicitly curated as
    preparation_action DOES surface in the preparation checklist."""
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#action",
                        "section_locator": "Section 1",
                        "text": "Verify hot-holding units read above the required temperature "
                        "before opening.",
                        "sha256": __import__("hashlib")
                        .sha256(
                            b"Verify hot-holding units read above the required temperature "
                            b"before opening."
                        )
                        .hexdigest(),
                        "applicable_violation_codes": ["04L"],
                        "topics": [],
                        "permitted_uses": ["preparation_action"],
                    }
                ],
            }
        ]
    )
    corpus_result = load_corpus(manifest_path)
    assert corpus_result.store is not None

    graph = _graph_for(
        "nyc:1",
        "nyc",
        "critical",
        restaurant_row,
        inspection_row,
        violation_row,
        violation_code="04L",
    )
    service = CopilotService(graph, corpus_result.store)
    answer = service.answer(
        restaurant_id="nyc:1", question="What should I review before the next inspection?"
    )
    assert answer.grounding_status == "grounded"
    claim_types = {c.claim_type for c in answer.claims}
    assert ClaimType.RECURRING_VIOLATION in claim_types
    assert ClaimType.GUIDANCE_FOR_CODE in claim_types
    guidance_claim = next(c for c in answer.claims if c.claim_type == ClaimType.GUIDANCE_FOR_CODE)
    assert "fictional-doc#action" in guidance_claim.evidence_ids
