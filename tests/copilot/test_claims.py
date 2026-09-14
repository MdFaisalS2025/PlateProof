"""Tests for plateproof.copilot.claims: the authorization matrix is
enforced mechanically, not merely assumed. Citation validity means both
(1) the evidence exists and (2) the claim type is authorized to cite that
evidence type -- a valid citation ID of the wrong evidence type for its
claim type must still be rejected.
"""

from __future__ import annotations

from datetime import date

import pytest

from plateproof.copilot.claims import ClaimAuthorizationError, build_claim, guidance_claim
from plateproof.copilot.models import Citation, ClaimType, EvidenceType, RetrievedEvidenceItem


def _restaurant_record_citation() -> Citation:
    return Citation(
        citation_id="record:nyc:1:latest_inspection",
        evidence_type=EvidenceType.RESTAURANT_RECORD,
        title="PlateProof documented record",
        url=None,
        excerpt="Documented inspection.",
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
    )


def _guidance_citation() -> Citation:
    return Citation(
        citation_id="fictional-doc#overview",
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
        title="Fictional Guidance",
        url="https://www.nyc.gov/fictional",
        excerpt="Fictional guidance text.",
        jurisdiction="nyc",
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


def test_guidance_claim_builds_guidance_unavailable_when_no_items_found() -> None:
    claim = guidance_claim(
        jurisdiction="nyc", violation_code="04L", topic=None, items=(), as_of_date=date(2025, 1, 1)
    )
    assert claim.claim_type == ClaimType.GUIDANCE_UNAVAILABLE
    assert claim.evidence_ids == ()


def test_guidance_claim_builds_guidance_for_code_when_items_found() -> None:
    item = RetrievedEvidenceItem(
        citation=_guidance_citation(),
        relevance_score=1.0,
        matched_terms=("04L",),
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
    )
    claim = guidance_claim(
        jurisdiction="nyc",
        violation_code="04L",
        topic=None,
        items=(item,),
        as_of_date=date(2025, 1, 1),
    )
    assert claim.claim_type == ClaimType.GUIDANCE_FOR_CODE
    assert claim.evidence_ids == ("fictional-doc#overview",)
