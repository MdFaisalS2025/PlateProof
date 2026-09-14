"""Evaluation harness for the deterministic Task 8A Copilot core.

Loads the committed synthetic NYC and Florida datasets under
data/reference/copilot_eval/, builds a real graph from them, runs every
case through CopilotService, and asserts the acceptance thresholds from
the Task 8 plan: citation validity and cross-jurisdiction isolation must
be 100%; other metrics have documented thresholds. This evaluates the
implementation only -- it is not a claim of real-world legal, medical, or
public-health correctness.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from plateproof.copilot.corpus import load_corpus
from plateproof.copilot.models import AUTHORIZED_EVIDENCE_TYPES, ClaimType
from plateproof.copilot.service import CopilotService
from plateproof.graph.builder import build_graph
from plateproof.graph.models import GraphBuildInput

_EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "copilot_eval"
_GUIDANCE_MANIFEST = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "reference"
    / "guidance"
    / "manifest.json"
)


def _parse_date(value: Any) -> Any:
    if isinstance(value, str) and len(value) == 10 and value[4] == "-":
        return date.fromisoformat(value)
    return value


def _parse_dates_in_rows(
    rows: list[dict[str, Any]], date_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    parsed = []
    for row in rows:
        row = dict(row)
        for field in date_fields:
            if field in row and row[field] is not None:
                row[field] = _parse_date(row[field])
        parsed.append(row)
    return parsed


@pytest.fixture(scope="module")
def eval_datasets() -> list[dict[str, Any]]:
    datasets = []
    for name in ("nyc.json", "florida.json"):
        datasets.append(json.loads((_EVAL_DIR / name).read_text(encoding="utf-8")))
    return datasets


@pytest.fixture(scope="module")
def eval_service(eval_datasets: list[dict[str, Any]]) -> CopilotService:
    restaurants: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for dataset in eval_datasets:
        restaurants.extend(
            _parse_dates_in_rows(
                dataset["restaurants"], ("latest_inspection_date", "source_snapshot_date")
            )
        )
        inspections.extend(
            _parse_dates_in_rows(
                dataset["inspections"], ("inspection_date", "source_snapshot_date")
            )
        )
        violations.extend(_parse_dates_in_rows(dataset["violations"], ("inspection_date",)))

    for row in restaurants:
        row["source_retrieved_at_utc"] = datetime.fromisoformat(
            row["source_retrieved_at_utc"].replace("Z", "+00:00")
        )

    result = build_graph(
        GraphBuildInput(
            restaurants=pl.DataFrame(restaurants),
            inspection_events=pl.DataFrame(inspections),
            violation_events=pl.DataFrame(violations),
            michelin_restaurants=None,
            michelin_distinction_events=None,
            restaurant_michelin_matches=None,
        )
    )
    corpus_result = load_corpus(_GUIDANCE_MANIFEST)
    return CopilotService(result.graph, corpus_result.store)


def _all_cases(eval_datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases = []
    for dataset in eval_datasets:
        for case in dataset["cases"]:
            cases.append({**case, "jurisdiction": dataset["jurisdiction"]})
    return cases


def test_supported_answer_rate_is_100_percent(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    cases = [c for c in _all_cases(eval_datasets) if c["expect"] == "grounded"]
    assert cases
    successes = 0
    for case in cases:
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        if answer.grounding_status == "grounded":
            successes += 1
    assert successes == len(cases), (
        f"{successes}/{len(cases)} expected-answerable cases were answered"
    )


def test_unsupported_answer_refusal_accuracy_is_100_percent(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    cases = [c for c in _all_cases(eval_datasets) if c["expect"] == "refused"]
    assert cases
    correct = 0
    for case in cases:
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        if (
            answer.grounding_status == "refused"
            and answer.refusal is not None
            and answer.refusal.reason.value == case["expect_refusal_reason"]
        ):
            correct += 1
    assert correct == len(cases), f"{correct}/{len(cases)} refusals matched the expected reason"


def test_restaurant_record_accuracy_is_100_percent(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    cases = [
        c for c in _all_cases(eval_datasets) if c["expect"] == "grounded" and "expect_contains" in c
    ]
    assert cases
    for case in cases:
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        assert case["expect_contains"].lower() in answer.answer_text.lower(), (
            case["question"],
            answer.answer_text,
        )


def test_deterministic_reproducibility_is_100_percent(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    for case in _all_cases(eval_datasets):
        first = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        second = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        assert first.answer_text == second.answer_text
        assert first.citations == second.citations
        assert first.grounding_status == second.grounding_status


def test_citation_validity_is_100_percent(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    """Every citation an answer's claims reference must (1) exist in that
    answer's citation list and (2) be of a type its claim type is
    authorized to cite."""
    for case in _all_cases(eval_datasets):
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        if answer.grounding_status != "grounded":
            continue
        citation_by_id = {c.citation_id: c for c in answer.citations}
        for claim in answer.claims:
            authorized = AUTHORIZED_EVIDENCE_TYPES[claim.claim_type]
            for evidence_id in claim.evidence_ids:
                assert evidence_id in citation_by_id, (case["question"], evidence_id)
                assert citation_by_id[evidence_id].evidence_type in authorized


def test_jurisdiction_leakage_rate_is_zero(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    for case in _all_cases(eval_datasets):
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        if answer.grounding_status != "grounded":
            continue
        for citation in answer.citations:
            if citation.evidence_type.value == "guidance_passage":
                assert citation.jurisdiction == case["jurisdiction"], (
                    case["question"],
                    citation.citation_id,
                    citation.jurisdiction,
                )


def test_violation_code_retrieval_accuracy(
    eval_service: CopilotService, eval_datasets: list[dict[str, Any]]
) -> None:
    """The real starter corpus has no exact code-level mappings -- only
    curated topic associations (see data/reference/guidance/README.md) --
    so a documented violation whose severity matches a curated topic must
    resolve to GUIDANCE_FOR_TOPIC, never a promoted GUIDANCE_FOR_CODE and
    never GUIDANCE_UNAVAILABLE when a topic match genuinely exists."""
    cases = [
        c
        for c in _all_cases(eval_datasets)
        if c.get("expect_intent") == "official_guidance_for_documented_codes"
    ]
    assert cases
    for case in cases:
        answer = eval_service.answer(restaurant_id=case["restaurant_id"], question=case["question"])
        assert answer.grounding_status == "grounded"
        claim_types = {c.claim_type for c in answer.claims}
        assert ClaimType.GUIDANCE_FOR_TOPIC in claim_types, (case["question"], answer.answer_text)
        assert ClaimType.GUIDANCE_FOR_CODE not in claim_types
