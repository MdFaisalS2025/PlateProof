"""Deterministic claim builders. Each function takes already-assembled
facts/evidence and returns a typed :class:`~plateproof.copilot.models.Claim`
-- never free text. :func:`build_claim` mechanically enforces
``AUTHORIZED_EVIDENCE_TYPES`` before constructing a claim: an attempt to
cite an evidence type the claim type is not authorized for raises
:class:`ClaimAuthorizationError` -- a defense-in-depth assertion that
should never fire given this module's own logic, proven directly by
``tests/copilot/test_claims.py`` rather than merely assumed.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from plateproof.copilot.models import (
    AUTHORIZED_EVIDENCE_TYPES,
    Citation,
    Claim,
    ClaimType,
    RestaurantFact,
    RetrievedEvidenceItem,
)


class ClaimAuthorizationError(Exception):
    """Raised when code attempts to build a Claim citing evidence of a
    type its claim_type is not authorized to cite."""


def build_claim(
    *,
    claim_type: ClaimType,
    jurisdiction: str,
    as_of_date: date,
    values: Mapping[str, object],
    citations: tuple[Citation, ...],
    provenance_category: str,
    render_template_id: str,
    period_start: date | None = None,
) -> Claim:
    authorized = AUTHORIZED_EVIDENCE_TYPES[claim_type]
    for citation in citations:
        if citation.evidence_type not in authorized:
            raise ClaimAuthorizationError(
                f"{claim_type} may not cite evidence of type {citation.evidence_type}"
            )
    if claim_type == ClaimType.GUIDANCE_UNAVAILABLE and citations:
        raise ClaimAuthorizationError("guidance_unavailable claims may not carry evidence")
    return Claim(
        claim_type=claim_type,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        period_start=period_start,
        values=dict(values),
        evidence_ids=tuple(c.citation_id for c in citations),
        provenance_category=provenance_category,
        render_template_id=render_template_id,
    )


def restaurant_identity_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.RESTAURANT_IDENTITY,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="restaurant_identity_v1",
    )


def latest_inspection_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.LATEST_INSPECTION,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="latest_inspection_v1",
    )


def recurring_violation_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.RECURRING_VIOLATION,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="recurring_violation_v1",
    )


def violation_frequency_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.VIOLATION_FREQUENCY,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="violation_frequency_v1",
    )


def history_trend_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.HISTORY_TREND,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        period_start=fact.values.get("earliest_date"),  # type: ignore[arg-type]
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="history_trend_v1",
    )


def michelin_context_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.MICHELIN_CONTEXT,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="restaurant_record",
        render_template_id="michelin_context_v1",
    )


def forecast_availability_claim(fact: RestaurantFact) -> Claim:
    return build_claim(
        claim_type=ClaimType.FORECAST_AVAILABILITY,
        jurisdiction=fact.jurisdiction,
        as_of_date=fact.as_of_date,
        values=fact.values,
        citations=(fact.citation,),
        provenance_category="model_forecast",
        render_template_id="forecast_availability_v1",
    )


def guidance_claim(
    *,
    jurisdiction: str,
    violation_code: str | None,
    topic: str | None,
    items: tuple[RetrievedEvidenceItem, ...],
    as_of_date: date,
) -> Claim:
    """Builds a ``guidance_for_code`` claim when ``items`` is non-empty, or
    an honest ``guidance_unavailable`` claim (no evidence, per the
    authorization matrix) when retrieval found nothing -- never
    improvises guidance that isn't backed by a retrieved passage."""
    if not items:
        return build_claim(
            claim_type=ClaimType.GUIDANCE_UNAVAILABLE,
            jurisdiction=jurisdiction,
            as_of_date=as_of_date,
            values={"violation_code": violation_code, "topic": topic},
            citations=(),
            provenance_category="official_guidance",
            render_template_id="guidance_unavailable_v1",
        )
    citations = tuple(item.citation for item in items)
    return build_claim(
        claim_type=ClaimType.GUIDANCE_FOR_CODE,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        values={
            "violation_code": violation_code,
            "topic": topic,
            "excerpts": [c.excerpt for c in citations],
            "titles": [c.title for c in citations],
        },
        citations=citations,
        provenance_category="official_guidance",
        render_template_id="guidance_for_code_v1",
    )
