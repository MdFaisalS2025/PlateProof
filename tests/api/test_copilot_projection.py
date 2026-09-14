"""RED-first tests for plateproof.api.copilot_projection -- the response-
limit logic that must never split a claim from its own evidence. Limits
are enforced by selecting whole claim-and-citation groups, never by
independently slicing claims/citations arrays."""

from __future__ import annotations

from datetime import UTC, date, datetime

from plateproof.api.copilot_projection import project_answer
from plateproof.copilot.models import (
    Citation,
    Claim,
    ClaimType,
    CopilotAnswer,
    EvidenceType,
)


def _citation(citation_id: str) -> Citation:
    return Citation(
        citation_id=citation_id,
        evidence_type=EvidenceType.RESTAURANT_RECORD,
        title="PlateProof documented record",
        url=None,
        excerpt=f"excerpt for {citation_id}",
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
    )


def _claim(
    claim_type: ClaimType,
    evidence_ids: tuple[str, ...],
    values: dict | None = None,
    *,
    render_template_id: str = "recurring_violation_v1",
) -> Claim:
    return Claim(
        claim_type=claim_type,
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
        period_start=None,
        values=values or {},
        evidence_ids=evidence_ids,
        provenance_category="restaurant_record",
        render_template_id=render_template_id,
    )


def _answer(claims: tuple[Claim, ...], citations: tuple[Citation, ...]) -> CopilotAnswer:
    return CopilotAnswer(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        intent=None,
        answer_text="full answer text",
        claims=claims,
        citations=citations,
        generator_mode="deterministic",
        grounding_status="grounded",
        refusal=None,
        warnings=(),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_every_evidence_id_resolves_to_exactly_one_returned_citation() -> None:
    c1, c2, c3 = _citation("c1"), _citation("c2"), _citation("c3")
    claim1 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c1",),
        {"violation_code": "04L", "occurrence_count": 2, "total_cited_count": 2},
    )
    claim2 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c2", "c3"),
        {"violation_code": "05A", "occurrence_count": 2, "total_cited_count": 2},
    )
    answer = _answer((claim1, claim2), (c1, c2, c3))

    response = project_answer(answer, max_claims=10, max_citations=10, max_excerpt_length=500)

    citation_ids = {c.citation_id for c in response.citations}
    for claim in response.claims:
        for evidence_id in claim.evidence_ids:
            assert evidence_id in citation_ids


def test_no_orphan_citations_are_returned() -> None:
    """A citation not referenced by any returned claim is dropped, never
    included just because it existed on the original answer."""
    c1, c2 = _citation("c1"), _citation("c2")
    claim1 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c1",),
        {"violation_code": "04L", "occurrence_count": 2, "total_cited_count": 2},
    )
    # c2 is never referenced by any claim.
    answer = _answer((claim1,), (c1, c2))

    response = project_answer(answer, max_claims=10, max_citations=10, max_excerpt_length=500)

    citation_ids = {c.citation_id for c in response.citations}
    assert citation_ids == {"c1"}


def test_limits_never_split_a_claim_from_its_evidence() -> None:
    """A max_citations limit that would fit claim1's own citation but not
    claim2's second citation must drop claim2 ENTIRELY, never return
    claim2 with only one of its two citations."""
    c1, c2, c3 = _citation("c1"), _citation("c2"), _citation("c3")
    claim1 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c1",),
        {"violation_code": "04L", "occurrence_count": 2, "total_cited_count": 2},
    )
    claim2 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c2", "c3"),
        {"violation_code": "05A", "occurrence_count": 2, "total_cited_count": 2},
    )
    answer = _answer((claim1, claim2), (c1, c2, c3))

    # Room for claim1's 1 citation, but not claim2's 2 additional ones.
    response = project_answer(answer, max_claims=10, max_citations=2, max_excerpt_length=500)

    returned_claim_types = [(c.claim_type, c.evidence_ids) for c in response.claims]
    assert returned_claim_types == [(ClaimType.RECURRING_VIOLATION.value, ("c1",))]
    assert {c.citation_id for c in response.citations} == {"c1"}


def test_truncation_produces_a_warning_and_a_consistently_re_rendered_answer_text() -> None:
    c1, c2 = _citation("c1"), _citation("c2")
    claim1 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c1",),
        {"violation_code": "04L", "occurrence_count": 2, "total_cited_count": 2},
    )
    claim2 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c2",),
        {"violation_code": "05A", "occurrence_count": 3, "total_cited_count": 3},
    )
    answer = _answer((claim1, claim2), (c1, c2))

    response = project_answer(answer, max_claims=1, max_citations=10, max_excerpt_length=500)

    assert len(response.claims) == 1
    assert any("truncat" in w.lower() for w in response.warnings)
    # answer_text must be re-derived from exactly the returned claims, not
    # sliced independently from the original full-claim-set text.
    assert "05A" not in response.answer_text
    assert "04L" in response.answer_text


def test_no_truncation_leaves_answer_text_byte_identical() -> None:
    c1 = _citation("c1")
    claim1 = _claim(
        ClaimType.RECURRING_VIOLATION,
        ("c1",),
        {"violation_code": "04L", "occurrence_count": 2, "total_cited_count": 2},
    )
    answer = _answer((claim1,), (c1,))
    response = project_answer(answer, max_claims=10, max_citations=10, max_excerpt_length=500)
    assert response.answer_text == answer.answer_text
    assert not any("truncat" in w.lower() for w in response.warnings)


def test_deterministic_ordering_is_preserved_under_limits() -> None:
    c1, c2, c3 = _citation("c1"), _citation("c2"), _citation("c3")
    claims = tuple(
        _claim(
            ClaimType.RECURRING_VIOLATION,
            (cid,),
            {"violation_code": f"0{i}L", "occurrence_count": 2, "total_cited_count": 2},
        )
        for i, cid in enumerate(["c1", "c2", "c3"])
    )
    answer = _answer(claims, (c1, c2, c3))
    response = project_answer(answer, max_claims=2, max_citations=10, max_excerpt_length=500)
    assert [c.evidence_ids for c in response.claims] == [("c1",), ("c2",)]


def test_excerpt_length_is_capped() -> None:
    c1 = Citation(
        citation_id="c1",
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
        title="Fictional Guidance",
        url="https://www.nyc.gov/fictional",
        excerpt="x" * 5000,
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
    )
    claim1 = _claim(
        ClaimType.GUIDANCE_FOR_CODE,
        ("c1",),
        {"violation_code": "04L", "excerpts": ["x" * 5000]},
        render_template_id="guidance_for_code_v1",
    )
    answer = _answer((claim1,), (c1,))
    response = project_answer(answer, max_claims=10, max_citations=10, max_excerpt_length=100)
    assert len(response.citations[0].excerpt) <= 100


def test_a_refused_answer_with_no_claims_projects_cleanly() -> None:
    from plateproof.copilot.models import Refusal, RefusalReason

    answer = CopilotAnswer(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        intent=None,
        answer_text="refusal text",
        claims=(),
        citations=(),
        generator_mode="deterministic",
        grounding_status="refused",
        refusal=Refusal(reason=RefusalReason.UNKNOWN_INTENT, detail="refusal text"),
        warnings=(),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    response = project_answer(answer, max_claims=10, max_citations=10, max_excerpt_length=500)
    assert response.claims == []
    assert response.citations == []
    assert response.refusal is not None
