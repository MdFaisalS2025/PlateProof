"""Projects a deterministic ``CopilotAnswer`` (Task 8A/8B's internal
dataclass, which carries a raw ``Mapping[str, object]`` per claim) into the
public ``CopilotQueryResponse`` Pydantic contract -- and is the ONLY place
response-size limits are applied.

Limits are enforced by selecting whole claim-and-evidence groups, never by
independently slicing ``claims``/``citations`` arrays: a claim is included
only together with every citation its own ``evidence_ids`` reference, so a
returned claim can never reference a citation the response dropped, and a
returned citation is never orphaned (referenced by nothing). If not every
claim fits, ``answer_text`` is re-rendered from exactly the claims that
were kept (via the same deterministic ``rendering.render_answer`` Task 8A
uses) -- never truncated independently of its claims -- and a warning is
added so a truncated response is never silently indistinguishable from a
complete one.
"""

from __future__ import annotations

from plateproof.api.schemas import (
    CopilotQueryResponse,
    PublicCitation,
    PublicClaim,
    PublicRefusal,
)
from plateproof.copilot.models import Citation, Claim, CopilotAnswer
from plateproof.copilot.rendering import render_answer, render_claim

_TRUNCATION_WARNING = (
    "This response was truncated to fit PlateProof's response size limits; "
    "some documented claims were omitted along with their citations."
)


def _select_claims_and_citations(
    claims: tuple[Claim, ...],
    citations: tuple[Citation, ...],
    *,
    max_claims: int,
    max_citations: int,
) -> tuple[tuple[Claim, ...], tuple[Citation, ...], bool]:
    selected_claims: list[Claim] = []
    selected_citation_ids: set[str] = set()

    for claim in claims:
        if len(selected_claims) + 1 > max_claims:
            break
        needed_ids = set(claim.evidence_ids)
        if len(selected_citation_ids | needed_ids) > max_citations:
            break
        selected_claims.append(claim)
        selected_citation_ids |= needed_ids

    # Preserve the original citation order (not selection order) for
    # deterministic, stable output.
    selected_citations = tuple(c for c in citations if c.citation_id in selected_citation_ids)
    truncated = len(selected_claims) < len(claims)
    return tuple(selected_claims), selected_citations, truncated


def _to_public_claim(claim: Claim) -> PublicClaim:
    return PublicClaim(
        claim_type=claim.claim_type.value,
        jurisdiction=claim.jurisdiction,
        as_of_date=claim.as_of_date,
        text=render_claim(claim),
        evidence_ids=claim.evidence_ids,
        provenance_category=claim.provenance_category,
    )


def _to_public_citation(citation: Citation, *, max_excerpt_length: int) -> PublicCitation:
    return PublicCitation(
        citation_id=citation.citation_id,
        evidence_category=citation.evidence_type.value,
        title=citation.title,
        url=citation.url,
        excerpt=citation.excerpt[:max_excerpt_length],
        jurisdiction=citation.jurisdiction,
        as_of_date=citation.as_of_date,
        superseded=citation.superseded,
        issuing_authority=citation.issuing_authority,
        section_locator=citation.section_locator,
        access_date=citation.access_date,
        effective_date=citation.effective_date,
        revision_date=citation.revision_date,
    )


def project_answer(
    answer: CopilotAnswer,
    *,
    max_claims: int,
    max_citations: int,
    max_excerpt_length: int,
) -> CopilotQueryResponse:
    selected_claims, selected_citations, truncated = _select_claims_and_citations(
        answer.claims, answer.citations, max_claims=max_claims, max_citations=max_citations
    )

    public_claims = [_to_public_claim(c) for c in selected_claims]
    public_citations = [
        _to_public_citation(c, max_excerpt_length=max_excerpt_length) for c in selected_citations
    ]

    answer_text = render_answer(selected_claims) if truncated else answer.answer_text
    warnings = list(answer.warnings) + ([_TRUNCATION_WARNING] if truncated else [])

    public_evidence_ids = {c.citation_id for c in public_citations}
    for public_claim in public_claims:
        missing = set(public_claim.evidence_ids) - public_evidence_ids
        if missing:
            raise RuntimeError("internal error: a public claim referenced a dropped citation")

    refusal = (
        PublicRefusal(
            reason=answer.refusal.reason.value,
            detail=answer.refusal.detail,
            supported_intents_hint=tuple(i.value for i in answer.refusal.supported_intents_hint),
        )
        if answer.refusal is not None
        else None
    )

    return CopilotQueryResponse(
        restaurant_id=answer.restaurant_id,
        jurisdiction=answer.jurisdiction,
        intent=answer.intent.value if answer.intent is not None else None,
        grounding_status=answer.grounding_status,
        answer_text=answer_text,
        claims=public_claims,
        citations=public_citations,
        refusal=refusal,
        generator_mode=answer.generator_mode,
        local_helper_status=answer.local_helper_status,
        warnings=warnings,
        generated_at=answer.generated_at,
        disclaimer=answer.disclaimer,
    )
