"""Tests for plateproof.copilot.claims: the authorization matrix and the
citation-jurisdiction-compatibility check are both enforced mechanically,
not merely assumed. Citation validity means (1) the evidence exists,
(2) the claim type is authorized to use that evidence type, AND (3) the
citation's jurisdiction matches the claim's own jurisdiction -- a valid
citation ID of the wrong evidence type OR the wrong jurisdiction for its
claim type must still be rejected.
"""

from __future__ import annotations

from datetime import date

import pytest

from plateproof.copilot.claims import (
    ClaimAuthorizationError,
    build_claim,
    guidance_for_code_claim,
    guidance_for_topic_claim,
    guidance_unavailable_claim,
)
from plateproof.copilot.models import Citation, ClaimType, EvidenceType, RetrievedEvidenceItem


def _restaurant_record_citation(jurisdiction: str = "nyc") -> Citation:
    return Citation(
        citation_id="record:nyc:1:latest_inspection",
        evidence_type=EvidenceType.RESTAURANT_RECORD,
        title="PlateProof documented record",
        url=None,
        excerpt="Documented inspection.",
        jurisdiction=jurisdiction,
        as_of_date=date(2025, 1, 1),
    )


def _guidance_citation(jurisdiction: str = "nyc") -> Citation:
    return Citation(
        citation_id="fictional-doc#overview",
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
        title="Fictional Guidance",
        url="https://www.nyc.gov/fictional",
        excerpt="Fictional guidance text.",
        jurisdiction=jurisdiction,
        as_of_date=None,
    )


def test_valid_claim_construction_succeeds() -> None:
    claim = build_claim(
        claim_type=ClaimType.LATEST_INSPECTION,
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
        values={"inspection_date": date(2025, 1, 1)},
        citations=(_restaurant_record_citation(),),
        provenance_category="restaurant_record",
        render_template_id="latest_inspection_v1",
    )
    assert claim.evidence_ids == ("record:nyc:1:latest_inspection",)


def test_citing_wrong_evidence_type_is_rejected_even_with_a_valid_citation_id() -> None:
    """A valid citation ID (it genuinely exists) is not sufficient: a
    LATEST_INSPECTION claim may only cite restaurant_record evidence, so
    attaching a real, valid guidance_passage citation must still be
    rejected."""
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.LATEST_INSPECTION,
            jurisdiction="nyc",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_guidance_citation(),),
            provenance_category="restaurant_record",
            render_template_id="latest_inspection_v1",
        )


def test_guidance_for_code_claim_may_only_cite_guidance_passages() -> None:
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.GUIDANCE_FOR_CODE,
            jurisdiction="nyc",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_restaurant_record_citation(),),
            provenance_category="official_guidance",
            render_template_id="guidance_for_code_v1",
        )


def test_guidance_unavailable_claim_may_not_carry_any_evidence() -> None:
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.GUIDANCE_UNAVAILABLE,
            jurisdiction="nyc",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_guidance_citation(),),
            provenance_category="official_guidance",
            render_template_id="guidance_unavailable_v1",
        )


# --------------------------------------------------------------------------- #
# Citation-jurisdiction compatibility (independent-review correction item 4)
# --------------------------------------------------------------------------- #


def test_nyc_claim_rejects_florida_citation() -> None:
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.GUIDANCE_FOR_CODE,
            jurisdiction="nyc",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_guidance_citation(jurisdiction="florida"),),
            provenance_category="official_guidance",
            render_template_id="guidance_for_code_v1",
        )


def test_florida_claim_rejects_nyc_citation() -> None:
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.GUIDANCE_FOR_CODE,
            jurisdiction="florida",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_guidance_citation(jurisdiction="nyc"),),
            provenance_category="official_guidance",
            render_template_id="guidance_for_code_v1",
        )


def test_restaurant_record_citation_must_match_restaurant_jurisdiction() -> None:
    """Even a restaurant-record citation (never corpus-derived) is
    rejected if its own jurisdiction doesn't match the claim's -- a
    same-evidence-type mismatch is not automatically safe."""
    with pytest.raises(ClaimAuthorizationError):
        build_claim(
            claim_type=ClaimType.LATEST_INSPECTION,
            jurisdiction="nyc",
            as_of_date=date(2025, 1, 1),
            values={},
            citations=(_restaurant_record_citation(jurisdiction="florida"),),
            provenance_category="restaurant_record",
            render_template_id="latest_inspection_v1",
        )


def test_federal_citation_is_never_compatible_with_an_nyc_or_florida_claim() -> None:
    """Federal guidance is never retrievable through any Task 8A intent
    (CorpusStore.passages_for_jurisdiction is always called with the
    restaurant's own nyc/florida jurisdiction), but this is the
    independent, second line of defense: even a hand-constructed federal
    citation cannot be attached to an nyc/florida claim."""
    for restaurant_jurisdiction in ("nyc", "florida"):
        with pytest.raises(ClaimAuthorizationError):
            build_claim(
                claim_type=ClaimType.GUIDANCE_FOR_CODE,
                jurisdiction=restaurant_jurisdiction,
                as_of_date=date(2025, 1, 1),
                values={},
                citations=(_guidance_citation(jurisdiction="federal"),),
                provenance_category="official_guidance",
                render_template_id="guidance_for_code_v1",
            )


# --------------------------------------------------------------------------- #
# Split guidance claim builders (independent-review correction item 1)
# --------------------------------------------------------------------------- #


def test_guidance_unavailable_claim_carries_no_evidence() -> None:
    claim = guidance_unavailable_claim(
        jurisdiction="nyc", violation_code="04L", topic=None, as_of_date=date(2025, 1, 1)
    )
    assert claim.claim_type == ClaimType.GUIDANCE_UNAVAILABLE
    assert claim.evidence_ids == ()


def test_guidance_for_code_claim_requires_at_least_one_item() -> None:
    with pytest.raises(ClaimAuthorizationError):
        guidance_for_code_claim(
            jurisdiction="nyc", violation_code="04L", items=(), as_of_date=date(2025, 1, 1)
        )


def test_guidance_for_code_claim_builds_from_exact_match_items() -> None:
    item = RetrievedEvidenceItem(
        citation=_guidance_citation(),
        relevance_score=1.0,
        matched_terms=("04L",),
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
    )
    claim = guidance_for_code_claim(
        jurisdiction="nyc", violation_code="04L", items=(item,), as_of_date=date(2025, 1, 1)
    )
    assert claim.claim_type == ClaimType.GUIDANCE_FOR_CODE
    assert claim.evidence_ids == ("fictional-doc#overview",)


def test_guidance_for_topic_claim_requires_at_least_one_item() -> None:
    with pytest.raises(ClaimAuthorizationError):
        guidance_for_topic_claim(
            jurisdiction="nyc",
            violation_code="04L",
            topic="high_priority",
            items=(),
            as_of_date=date(2025, 1, 1),
        )


def test_guidance_for_topic_claim_builds_from_exact_match_items() -> None:
    item = RetrievedEvidenceItem(
        citation=_guidance_citation(),
        relevance_score=1.0,
        matched_terms=("high_priority",),
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
    )
    claim = guidance_for_topic_claim(
        jurisdiction="nyc",
        violation_code="04L",
        topic="high_priority",
        items=(item,),
        as_of_date=date(2025, 1, 1),
    )
    assert claim.claim_type == ClaimType.GUIDANCE_FOR_TOPIC
    assert claim.evidence_ids == ("fictional-doc#overview",)
