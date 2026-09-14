"""``CopilotService``: the deterministic Copilot core's single entry
point. Restaurant-scoped only (Task 8A/8B MVP scope -- see the Task 8
plan's YAGNI decision); jurisdiction is always derived from the resolved
restaurant, never trusted from caller input.

No optional local model is wired in here during Task 8A. Every answer is
``generator_mode="deterministic"``; Task 8B adds a bounded intent-helper
hook without changing this module's own answer-construction logic.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime

from plateproof.copilot import claims as claims_module
from plateproof.copilot import facts, retrieval
from plateproof.copilot.corpus import CorpusStore
from plateproof.copilot.intents import IntentDetectionOutcome, detect_intent
from plateproof.copilot.models import (
    Citation,
    Claim,
    CopilotAnswer,
    Intent,
    Refusal,
    RefusalReason,
    RestaurantFact,
    RetrievedEvidenceItem,
)
from plateproof.graph import queries
from plateproof.graph.store import PlateProofGraph

ForecastLookup = Callable[[str, str], Mapping[str, object] | None]

_PROHIBITED_DETAIL = (
    "PlateProof cannot answer requests about future health outcomes, legal "
    "liability, guarantees, or evading documented violations. It can share "
    "documented inspection history and official guidance instead."
)
_AMBIGUOUS_DETAIL = (
    "This question could match more than one supported topic. Please ask more specifically."
)
_UNKNOWN_DETAIL = (
    "This question isn't one PlateProof currently supports. See the supported topics below."
)
_NO_RESTAURANT_DETAIL = "This restaurant is not documented in PlateProof."
_INSUFFICIENT_EVIDENCE_DETAIL = "There isn't enough documented history to answer this question yet."


class CopilotService:
    def __init__(
        self,
        graph: PlateProofGraph,
        corpus_store: CorpusStore | None = None,
        *,
        forecast_lookup: ForecastLookup | None = None,
    ) -> None:
        self._graph = graph
        self._corpus = corpus_store
        self._forecast_lookup = forecast_lookup

    def answer(self, *, restaurant_id: str, question: str) -> CopilotAnswer:
        now = datetime.now(UTC)
        restaurant = queries.restaurant_node(self._graph, restaurant_id)
        if restaurant is None:
            return self._refuse(
                restaurant_id,
                "",
                None,
                RefusalReason.INSUFFICIENT_EVIDENCE,
                _NO_RESTAURANT_DETAIL,
                now,
            )
        jurisdiction = str(restaurant["jurisdiction"])

        detection = detect_intent(question)
        if detection.outcome == IntentDetectionOutcome.PROHIBITED:
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.PROHIBITED_REQUEST,
                _PROHIBITED_DETAIL,
                now,
            )
        if detection.outcome == IntentDetectionOutcome.AMBIGUOUS:
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.AMBIGUOUS_INTENT,
                _AMBIGUOUS_DETAIL,
                now,
                hint=detection.candidates,
            )
        if detection.outcome == IntentDetectionOutcome.UNKNOWN:
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.UNKNOWN_INTENT,
                _UNKNOWN_DETAIL,
                now,
                hint=tuple(Intent),
            )

        assert detection.intent is not None
        return self.answer_for_intent(
            restaurant_id=restaurant_id, jurisdiction=jurisdiction, intent=detection.intent, now=now
        )

    def answer_for_intent(
        self, *, restaurant_id: str, jurisdiction: str, intent: Intent, now: datetime
    ) -> CopilotAnswer:
        # Claim.values holds plain dicts/lists, so Claim is not guaranteed
        # hashable -- ordered_claims is built by object identity, never
        # dict.fromkeys()/a set. citations_by_id only ever receives the
        # exact Citation objects a claim's own evidence_ids reference --
        # never an unrelated citation merely "nearby" in the same loop.
        ordered_claims: list[Claim] = []
        citations_by_id: dict[str, Citation] = {}

        def _record(claim: Claim, evidence: tuple[Citation, ...]) -> None:
            ordered_claims.append(claim)
            for citation in evidence:
                citations_by_id[citation.citation_id] = citation

        if intent == Intent.RESTAURANT_IDENTITY:
            fact = facts.restaurant_identity_fact(self._graph, restaurant_id)
            if fact is None:
                return self._refuse(
                    restaurant_id,
                    jurisdiction,
                    intent,
                    RefusalReason.INSUFFICIENT_EVIDENCE,
                    _NO_RESTAURANT_DETAIL,
                    now,
                )
            _record(claims_module.restaurant_identity_claim(fact), (fact.citation,))

        elif intent == Intent.LATEST_INSPECTION_SUMMARY:
            fact = facts.latest_inspection_fact(self._graph, restaurant_id)
            if fact is None:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            _record(claims_module.latest_inspection_claim(fact), (fact.citation,))

        elif intent == Intent.RECURRING_VIOLATIONS:
            recurring = facts.recurring_violation_facts(
                self._graph, restaurant_id, min_occurrences=2
            )
            if not recurring:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            for f in recurring:
                _record(claims_module.recurring_violation_claim(f), (f.citation,))

        elif intent == Intent.VIOLATION_HISTORY:
            all_violations = facts.recurring_violation_facts(
                self._graph, restaurant_id, min_occurrences=1
            )
            if not all_violations:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            for f in all_violations:
                _record(claims_module.violation_frequency_claim(f), (f.citation,))

        elif intent == Intent.INSPECTION_TREND:
            fact = facts.inspection_trend_fact(self._graph, restaurant_id)
            if fact is None:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            _record(claims_module.history_trend_claim(fact), (fact.citation,))

        elif intent == Intent.MICHELIN_CONTEXT:
            distinctions = facts.michelin_context_facts(self._graph, restaurant_id, date.today())
            if not distinctions:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            for f in distinctions:
                _record(claims_module.michelin_context_claim(f), (f.citation,))

        elif intent == Intent.EXPLAIN_PREDICTION:
            if self._forecast_lookup is None:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            row = self._forecast_lookup(restaurant_id, jurisdiction)
            if row is None:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            fact = facts.forecast_fact_from_row(restaurant_id, jurisdiction, row)
            _record(claims_module.forecast_availability_claim(fact), (fact.citation,))

        elif intent == Intent.OFFICIAL_GUIDANCE_FOR_DOCUMENTED_CODES:
            documented = facts.recurring_violation_facts(
                self._graph, restaurant_id, min_occurrences=1
            )
            if not documented:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            for f in documented:
                items = self._guidance_for_code(jurisdiction, f)
                claim = claims_module.guidance_claim(
                    jurisdiction=jurisdiction,
                    violation_code=str(f.values["violation_code"]),
                    topic=None,
                    items=items,
                    as_of_date=f.as_of_date,
                )
                _record(claim, tuple(item.citation for item in items))

        elif intent == Intent.PREPARATION_CHECKLIST_FROM_OFFICIAL_GUIDANCE:
            documented = facts.recurring_violation_facts(
                self._graph, restaurant_id, min_occurrences=1
            )
            if not documented:
                return self._insufficient(restaurant_id, jurisdiction, intent, now)
            for f in documented:
                _record(claims_module.recurring_violation_claim(f), (f.citation,))
                items = self._guidance_for_code(jurisdiction, f)
                guidance_claim = claims_module.guidance_claim(
                    jurisdiction=jurisdiction,
                    violation_code=str(f.values["violation_code"]),
                    topic=None,
                    items=items,
                    as_of_date=f.as_of_date,
                )
                _record(guidance_claim, tuple(item.citation for item in items))

        else:  # pragma: no cover - exhaustive over the closed Intent enum
            return self._refuse(
                restaurant_id,
                jurisdiction,
                intent,
                RefusalReason.UNKNOWN_INTENT,
                _UNKNOWN_DETAIL,
                now,
            )

        claims_tuple = tuple(ordered_claims)
        citations = tuple(citations_by_id.values())
        answer_text = self._render(claims_tuple)
        return CopilotAnswer(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            intent=intent,
            answer_text=answer_text,
            claims=claims_tuple,
            citations=citations,
            generator_mode="deterministic",
            grounding_status="grounded",
            refusal=None,
            warnings=(),
            generated_at=now,
        )

    def _guidance_for_code(
        self, jurisdiction: str, fact: RestaurantFact
    ) -> tuple[RetrievedEvidenceItem, ...]:
        code = str(fact.values["violation_code_norm"])
        if self._corpus is None:
            return ()
        items = retrieval.passages_for_codes(self._corpus, jurisdiction, [code])
        if items:
            return items
        description = fact.values.get("description")
        query_text = str(description) if description else code
        return retrieval.search_by_topic_or_text(self._corpus, jurisdiction, query_text)

    def _render(self, claims: tuple[Claim, ...]) -> str:
        from plateproof.copilot.rendering import render_answer

        return render_answer(claims)

    def _insufficient(
        self, restaurant_id: str, jurisdiction: str, intent: Intent, now: datetime
    ) -> CopilotAnswer:
        return self._refuse(
            restaurant_id,
            jurisdiction,
            intent,
            RefusalReason.INSUFFICIENT_EVIDENCE,
            _INSUFFICIENT_EVIDENCE_DETAIL,
            now,
        )

    def _refuse(
        self,
        restaurant_id: str,
        jurisdiction: str,
        intent: Intent | None,
        reason: RefusalReason,
        detail: str,
        now: datetime,
        *,
        hint: tuple[Intent, ...] = (),
    ) -> CopilotAnswer:
        return CopilotAnswer(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            intent=intent,
            answer_text=detail,
            claims=(),
            citations=(),
            generator_mode="deterministic",
            grounding_status="refused",
            refusal=Refusal(reason=reason, detail=detail, supported_intents_hint=hint),
            warnings=(),
            generated_at=now,
        )
