"""Core typed contracts for the deterministic Copilot core: the closed
intent set, evidence/citation types, the claim-authorization matrix, and
the final answer contract.

The central invariant this module encodes: **every final factual sentence
is built from a ``Claim``, and every ``Claim`` is built by code from
``RestaurantFact``/corpus data -- never from freeform generated text.**
``AUTHORIZED_EVIDENCE_TYPES`` is the mechanical proof that a citation is
valid: it is valid only when (1) the cited evidence exists, and (2) the
claim type is authorized to cite that evidence type. Citation-ID
membership alone is never sufficient (a valid id attached to the wrong
evidence type for its claim type is still rejected -- see
``plateproof.copilot.claims``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

# A closed type, not a bare `str`: "nyc"/"florida" are restaurant/claim
# jurisdictions; "federal" appears only on a Citation (a guidance passage
# may be federal model guidance) and is never a valid restaurant/claim
# jurisdiction -- see the citation-jurisdiction-compatibility policy in
# plateproof.copilot.claims.build_claim, which is the runtime enforcement
# this static type alone cannot provide.
Jurisdiction = Literal["nyc", "florida", "federal"]
RestaurantJurisdiction = Literal["nyc", "florida"]


def parse_restaurant_jurisdiction(value: object) -> RestaurantJurisdiction | None:
    """Validates an arbitrary value (typically a graph node's raw
    ``jurisdiction`` attribute) against the two jurisdictions a restaurant
    can actually have. Returns ``None`` for anything else -- including
    ``"federal"``, which is a valid :class:`Citation` jurisdiction but
    never a valid restaurant jurisdiction -- never guesses, never raises.
    This is the runtime enforcement a bare ``str`` type cannot provide."""
    if value == "nyc":
        return "nyc"
    if value == "florida":
        return "florida"
    return None


class Intent(StrEnum):
    """The complete, closed set of questions the Copilot can answer.
    Nothing outside this enum may ever produce a substantive answer --
    unknown, ambiguous, or unsupported requests are refused instead."""

    LATEST_INSPECTION_SUMMARY = "latest_inspection_summary"
    RECURRING_VIOLATIONS = "recurring_violations"
    VIOLATION_HISTORY = "violation_history"
    INSPECTION_TREND = "inspection_trend"
    OFFICIAL_GUIDANCE_FOR_DOCUMENTED_CODES = "official_guidance_for_documented_codes"
    PREPARATION_CHECKLIST_FROM_OFFICIAL_GUIDANCE = "preparation_checklist_from_official_guidance"
    EXPLAIN_PREDICTION = "explain_prediction"
    RESTAURANT_IDENTITY = "restaurant_identity"
    MICHELIN_CONTEXT = "michelin_context"


class EvidenceType(StrEnum):
    RESTAURANT_RECORD = "restaurant_record"
    GUIDANCE_PASSAGE = "guidance_passage"
    MODEL_FORECAST = "model_forecast"


class ClaimType(StrEnum):
    LATEST_INSPECTION = "latest_inspection"
    RECURRING_VIOLATION = "recurring_violation"
    VIOLATION_FREQUENCY = "violation_frequency"
    HISTORY_TREND = "history_trend"
    GUIDANCE_FOR_CODE = "guidance_for_code"
    GUIDANCE_FOR_TOPIC = "guidance_for_topic"
    GUIDANCE_UNAVAILABLE = "guidance_unavailable"
    FORECAST_AVAILABILITY = "forecast_availability"
    RESTAURANT_IDENTITY = "restaurant_identity"
    MICHELIN_CONTEXT = "michelin_context"


# The closed authorization matrix. A claim of a given type may cite only
# evidence of the listed type(s) -- checked mechanically by
# plateproof.copilot.claims before a Claim is ever constructed.
AUTHORIZED_EVIDENCE_TYPES: Mapping[ClaimType, frozenset[EvidenceType]] = {
    ClaimType.LATEST_INSPECTION: frozenset({EvidenceType.RESTAURANT_RECORD}),
    ClaimType.RECURRING_VIOLATION: frozenset({EvidenceType.RESTAURANT_RECORD}),
    ClaimType.VIOLATION_FREQUENCY: frozenset({EvidenceType.RESTAURANT_RECORD}),
    ClaimType.HISTORY_TREND: frozenset({EvidenceType.RESTAURANT_RECORD}),
    ClaimType.GUIDANCE_FOR_CODE: frozenset({EvidenceType.GUIDANCE_PASSAGE}),
    ClaimType.GUIDANCE_FOR_TOPIC: frozenset({EvidenceType.GUIDANCE_PASSAGE}),
    ClaimType.GUIDANCE_UNAVAILABLE: frozenset(),
    ClaimType.FORECAST_AVAILABILITY: frozenset({EvidenceType.MODEL_FORECAST}),
    ClaimType.RESTAURANT_IDENTITY: frozenset({EvidenceType.RESTAURANT_RECORD}),
    ClaimType.MICHELIN_CONTEXT: frozenset({EvidenceType.RESTAURANT_RECORD}),
}

# Intents that require no guidance-corpus/graph evidence at all beyond the
# restaurant existing -- used only for RESTAURANT_IDENTITY's "never
# refuses" contract.
_ALWAYS_ANSWERABLE_INTENTS = frozenset({Intent.RESTAURANT_IDENTITY})


@dataclass(frozen=True)
class Citation:
    citation_id: str
    evidence_type: EvidenceType
    title: str
    url: str | None
    excerpt: str
    jurisdiction: Jurisdiction
    as_of_date: date | None
    superseded: bool = False


@dataclass(frozen=True)
class RestaurantFact:
    """One typed, dated fact read directly from the graph (or, for
    ``model_forecast``, from the precomputed prediction row) -- the only
    input restaurant-scoped claims are ever built from."""

    fact_type: str
    restaurant_id: str
    jurisdiction: Jurisdiction
    as_of_date: date
    values: Mapping[str, object]
    source_node_ids: tuple[str, ...]
    citation: Citation


@dataclass(frozen=True)
class RetrievedEvidenceItem:
    citation: Citation
    relevance_score: float
    matched_terms: tuple[str, ...]
    evidence_type: EvidenceType


@dataclass(frozen=True)
class EvidenceBundle:
    restaurant_id: str
    jurisdiction: Jurisdiction
    intent: Intent
    restaurant_facts: tuple[RestaurantFact, ...]
    guidance_evidence: tuple[RetrievedEvidenceItem, ...]
    generated_at: datetime


@dataclass(frozen=True)
class Claim:
    claim_type: ClaimType
    jurisdiction: Jurisdiction
    as_of_date: date
    period_start: date | None
    values: Mapping[str, object]
    evidence_ids: tuple[str, ...]
    provenance_category: str
    render_template_id: str


class RefusalReason(StrEnum):
    PROHIBITED_REQUEST = "prohibited_request"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    UNKNOWN_INTENT = "unknown_intent"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class Refusal:
    reason: RefusalReason
    detail: str
    supported_intents_hint: tuple[Intent, ...] = ()


@dataclass(frozen=True)
class CopilotAnswer:
    restaurant_id: str
    # None only when the restaurant itself couldn't be resolved -- never a
    # guessed/default jurisdiction standing in for "unknown".
    jurisdiction: Jurisdiction | None
    intent: Intent | None
    answer_text: str
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    generator_mode: str  # "deterministic" | "local_llm_assisted" (Task 8B)
    grounding_status: str  # "grounded" | "refused"
    refusal: Refusal | None
    warnings: tuple[str, ...]
    generated_at: datetime
    disclaimer: str = (
        "PlateProof does not verify restaurant ownership. This reflects "
        "documented public records and official guidance only."
    )
