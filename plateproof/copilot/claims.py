"""Deterministic claim builders. Each function takes already-assembled
facts/evidence and returns a typed :class:`~plateproof.copilot.models.Claim`
-- never free text. :func:`build_claim` mechanically enforces two
independent conditions before a claim is ever constructed, both proven
directly by ``tests/copilot/test_claims.py`` rather than merely assumed:

1. ``AUTHORIZED_EVIDENCE_TYPES`` -- the cited evidence must be of a type
   the claim type may use at all.
2. Citation-jurisdiction compatibility -- every cited evidence's own
   ``jurisdiction`` must exactly equal the claim's jurisdiction (the
   resolved restaurant's jurisdiction). An NYC claim may never cite
   Florida guidance or vice versa, and a federal-jurisdiction citation is
   *never* compatible with any claim (Task 8A's answer path never
   produces an nyc/florida claim citing federal evidence -- federal
   guidance is loaded into the corpus for completeness/audit but is not
   retrievable through any Task 8A intent, since
   ``CorpusStore.passages_for_jurisdiction`` is always called with the
   restaurant's own nyc/florida jurisdiction, never "federal". This
   function's check is the second, independent line of defense should
   that ever change.)

A citation ID existing is therefore not sufficient on its own for
"citation validity": it must also be the right evidence type AND the
right jurisdiction for the claim citing it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from plateproof.copilot.models import (
    AUTHORIZED_EVIDENCE_TYPES,
    Citation,
    Claim,
    ClaimType,
    Jurisdiction,
    RestaurantFact,
    RetrievedEvidenceItem,
)


class ClaimAuthorizationError(Exception):
    """Raised when code attempts to build a Claim citing evidence of a
    type its claim_type is not authorized to cite, or citing evidence from
    a jurisdiction incompatible with the claim's own jurisdiction."""


def build_claim(
    *,
    claim_type: ClaimType,
    jurisdiction: Jurisdiction,
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
        if citation.jurisdiction != jurisdiction:
            raise ClaimAuthorizationError(
                f"a {jurisdiction} claim may not cite evidence from jurisdiction "
                f"{citation.jurisdiction!r}"
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


def guidance_unavailable_claim(
    *, jurisdiction: Jurisdiction, violation_code: str | None, topic: str | None, as_of_date: date
) -> Claim:
    """An honest "PlateProof's corpus has nothing mapped here" claim --
    carries no evidence at all (enforced by build_claim), and its wording
    (see plateproof.copilot.rendering) describes PlateProof's own corpus
    state, never the state of official guidance in general."""
    return build_claim(
        claim_type=ClaimType.GUIDANCE_UNAVAILABLE,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        values={"violation_code": violation_code, "topic": topic},
        citations=(),
        provenance_category="official_guidance",
        render_template_id="guidance_unavailable_v1",
    )


def guidance_for_code_claim(
    *,
    jurisdiction: Jurisdiction,
    violation_code: str,
    items: tuple[RetrievedEvidenceItem, ...],
    as_of_date: date,
) -> Claim:
    """Requires ``items`` to already be the result of an exact, curated
    ``applicable_violation_codes`` match (see
    ``plateproof.copilot.retrieval.passages_for_codes``) -- never a
    text-similarity result. Raises via ``build_claim`` if ``items`` is
    empty, since a code-specific claim with no evidence should never be
    constructed; callers must use :func:`guidance_unavailable_claim`
    instead when no code mapping exists."""
    if not items:
        raise ClaimAuthorizationError(
            "guidance_for_code claims require at least one exact code-mapped passage"
        )
    citations = tuple(item.citation for item in items)
    return build_claim(
        claim_type=ClaimType.GUIDANCE_FOR_CODE,
        jurisdiction=jurisdiction,
        as_of_date=as_of_date,
        values={
            "violation_code": violation_code,
            "excerpts": [c.excerpt for c in citations],
            "titles": [c.title for c in citations],
        },
        citations=citations,
        provenance_category="official_guidance",
        render_template_id="guidance_for_code_v1",
    )


def guidance_for_topic_claim(
    *,
    jurisdiction: Jurisdiction,
    violation_code: str | None,
    topic: str,
    items: tuple[RetrievedEvidenceItem, ...],
    as_of_date: date,
) -> Claim:
    """Requires ``items`` to already be the result of an exact, curated
    ``topics`` match (see
    ``plateproof.copilot.retrieval.passages_for_topics``) -- never a
    text-similarity result. This claim type is deliberately worded (see
    rendering.py) as general topical guidance, never as guidance mapped
    to the specific violation code."""
    if not items:
        raise ClaimAuthorizationError(
            "guidance_for_topic claims require at least one exact topic-mapped passage"
        )
    citations = tuple(item.citation for item in items)
    return build_claim(
        claim_type=ClaimType.GUIDANCE_FOR_TOPIC,
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
        render_template_id="guidance_for_topic_v1",
    )
