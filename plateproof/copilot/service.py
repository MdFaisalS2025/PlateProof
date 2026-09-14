"""``CopilotService``: the deterministic Copilot core's single entry
point. Restaurant-scoped only (Task 8A/8B MVP scope -- see the Task 8
plan's YAGNI decision).

Jurisdiction is never accepted as a caller-supplied argument anywhere in
this class -- both :meth:`answer` and the direct-intent interface
(:meth:`answer_for_intent`) independently resolve it from the graph's own
Restaurant node for ``restaurant_id`` and fail closed
(``insufficient_evidence``) if that node is missing or its jurisdiction
attribute isn't a recognized value. A caller cannot supply a mismatched
jurisdiction and reach another jurisdiction's guidance -- there is no
parameter through which to try.

No optional local model is wired in here during Task 8A. Every answer is
``generator_mode="deterministic"``; Task 8B adds a bounded intent-helper
hook without changing this module's own answer-construction logic.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime

from plateproof.copilot import claims as claims_module
from plateproof.copilot import facts, retrieval
from plateproof.copilot.corpus import CorpusStore, PermittedUse
from plateproof.copilot.generators.base import IntentHelper, IntentHelperOutcome
from plateproof.copilot.intents import IntentDetectionOutcome, detect_intent
from plateproof.copilot.models import (
    Citation,
    Claim,
    CopilotAnswer,
    Intent,
    Refusal,
    RefusalReason,
    RestaurantFact,
    RestaurantJurisdiction,
    RetrievedEvidenceItem,
    parse_restaurant_jurisdiction,
)
from plateproof.copilot.question_validation import sanitize_question
from plateproof.graph import queries
from plateproof.graph.store import PlateProofGraph

ForecastLookup = Callable[[str, str], Mapping[str, object] | None]

_DEFAULT_MAX_QUESTION_LENGTH = 500

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
_INVALID_QUESTION_DETAIL = "This question could not be processed. Please rephrase and try again."


class CopilotService:
    def __init__(
        self,
        graph: PlateProofGraph,
        corpus_store: CorpusStore | None = None,
        *,
        forecast_lookup: ForecastLookup | None = None,
        intent_helper: IntentHelper | None = None,
        max_question_length: int = _DEFAULT_MAX_QUESTION_LENGTH,
    ) -> None:
        self._graph = graph
        self._corpus = corpus_store
        self._forecast_lookup = forecast_lookup
        self._intent_helper = intent_helper
        self._max_question_length = max_question_length

    def _resolve_jurisdiction(self, restaurant_id: str) -> RestaurantJurisdiction | None:
        """The single point every code path uses to learn a restaurant's
        jurisdiction -- always read fresh from the graph's own Restaurant
        node, never accepted as a parameter from elsewhere. Returns
        ``None`` (fail closed) if the restaurant doesn't exist or its
        jurisdiction attribute isn't a recognized value."""
        restaurant = queries.restaurant_node(self._graph, restaurant_id)
        if restaurant is None:
            return None
        return parse_restaurant_jurisdiction(restaurant.get("jurisdiction"))

    def answer(self, *, restaurant_id: str, question: str) -> CopilotAnswer:
        """Recommended flow (Task 8B): sanitize the question -> the
        prohibited-request check (never bypassable by any helper) ->
        deterministic intent detection -> if it resolves one intent
        unambiguously, use it directly, without ever consulting the
        optional local helper -> only for an ambiguous/unknown result,
        and only if a helper is configured, ask it to propose a closed
        intent -> on acceptance, call the *same* :meth:`answer_for_intent`
        used everywhere else and only override generator-mode metadata ->
        on any rejection/unavailability, fall back to the original
        deterministic refusal, exactly as if no helper were configured."""
        now = datetime.now(UTC)
        jurisdiction = self._resolve_jurisdiction(restaurant_id)
        if jurisdiction is None:
            return self._refuse(
                restaurant_id,
                None,
                None,
                RefusalReason.INSUFFICIENT_EVIDENCE,
                _NO_RESTAURANT_DETAIL,
                now,
            )

        sanitized_question, invalid_reason = sanitize_question(
            question, max_length=self._max_question_length
        )
        if sanitized_question is None:
            assert invalid_reason is not None
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.INVALID_QUESTION,
                _INVALID_QUESTION_DETAIL,
                now,
            )

        detection = detect_intent(sanitized_question)
        if detection.outcome == IntentDetectionOutcome.PROHIBITED:
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.PROHIBITED_REQUEST,
                _PROHIBITED_DETAIL,
                now,
            )
        if detection.outcome == IntentDetectionOutcome.MATCHED:
            assert detection.intent is not None
            return self.answer_for_intent(
                restaurant_id=restaurant_id, intent=detection.intent, now=now
            )

        # AMBIGUOUS or UNKNOWN: the deterministic detector could not
        # resolve one intent by itself.
        if self._intent_helper is not None:
            helper_result = self._intent_helper.propose(
                question=sanitized_question, jurisdiction=jurisdiction, now=now
            )
            if helper_result.outcome == IntentHelperOutcome.ACCEPTED:
                assert helper_result.proposal is not None
                base_answer = self.answer_for_intent(
                    restaurant_id=restaurant_id, intent=helper_result.proposal.intent, now=now
                )
                return dataclasses.replace(
                    base_answer,
                    generator_mode="local_llm_assisted",
                    local_helper_status="accepted",
                )
            local_helper_status = (
                "rejected"
                if helper_result.outcome == IntentHelperOutcome.REJECTED
                else "unavailable"
            )
        else:
            local_helper_status = "disabled"

        if detection.outcome == IntentDetectionOutcome.AMBIGUOUS:
            return self._refuse(
                restaurant_id,
                jurisdiction,
                None,
                RefusalReason.AMBIGUOUS_INTENT,
                _AMBIGUOUS_DETAIL,
                now,
                hint=detection.candidates,
                local_helper_status=local_helper_status,
            )
        return self._refuse(
            restaurant_id,
            jurisdiction,
            None,
            RefusalReason.UNKNOWN_INTENT,
            _UNKNOWN_DETAIL,
            now,
            hint=tuple(Intent),
            local_helper_status=local_helper_status,
        )

    def answer_for_intent(
        self, *, restaurant_id: str, intent: Intent, now: datetime | None = None
    ) -> CopilotAnswer:
        """The direct-intent interface (used by the UI's example-question
        buttons, and by tests exercising one intent in isolation). Takes
        no jurisdiction argument -- it is always re-resolved from the
        graph here, so a caller cannot pass a restaurant_id belonging to
        one jurisdiction while asserting another."""
        now = now or datetime.now(UTC)
        jurisdiction = self._resolve_jurisdiction(restaurant_id)
        if jurisdiction is None:
            return self._refuse(
                restaurant_id,
                None,
                intent,
                RefusalReason.INSUFFICIENT_EVIDENCE,
                _NO_RESTAURANT_DETAIL,
                now,
            )

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
                claim, items = self._resolve_guidance(
                    jurisdiction, f, require_preparation_action=False
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
                claim, items = self._resolve_guidance(
                    jurisdiction, f, require_preparation_action=True
                )
                _record(claim, tuple(item.citation for item in items))

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

    def _resolve_guidance(
        self,
        jurisdiction: RestaurantJurisdiction,
        fact: RestaurantFact,
        *,
        require_preparation_action: bool,
    ) -> tuple[Claim, tuple[RetrievedEvidenceItem, ...]]:
        """Resolves one documented violation's guidance claim, in strict
        priority order, never falling back to text similarity:

        1. An exact, curated ``applicable_violation_codes`` mapping ->
           ``GUIDANCE_FOR_CODE``.
        2. Failing that, an exact, curated ``topics`` match against the
           violation's own severity classification -> ``GUIDANCE_FOR_TOPIC``.
        3. Failing that, an honest ``GUIDANCE_UNAVAILABLE`` -- never a
           TF-IDF/text-similarity result promoted into either claim type.

        When ``require_preparation_action`` is set (the preparation-
        checklist intent), both lookups are additionally restricted to
        passages curated for ``PermittedUse.PREPARATION_ACTION`` -- a
        classification/definition passage can never become a checklist
        step, no matter how well it matches.
        """
        code = str(fact.values["violation_code"])
        code_norm = str(fact.values["violation_code_norm"])
        severity = fact.values.get("severity")
        required_use = PermittedUse.PREPARATION_ACTION if require_preparation_action else None

        if self._corpus is not None:
            code_items = retrieval.passages_for_codes(
                self._corpus, jurisdiction, [code_norm], required_use=required_use
            )
            if code_items:
                claim = claims_module.guidance_for_code_claim(
                    jurisdiction=jurisdiction,
                    violation_code=code,
                    items=code_items,
                    as_of_date=fact.as_of_date,
                )
                return claim, code_items

            if severity:
                topic_items = retrieval.passages_for_topics(
                    self._corpus, jurisdiction, [str(severity)], required_use=required_use
                )
                if topic_items:
                    claim = claims_module.guidance_for_topic_claim(
                        jurisdiction=jurisdiction,
                        violation_code=code,
                        topic=str(severity),
                        items=topic_items,
                        as_of_date=fact.as_of_date,
                    )
                    return claim, topic_items

        unavailable = claims_module.guidance_unavailable_claim(
            jurisdiction=jurisdiction, violation_code=code, topic=None, as_of_date=fact.as_of_date
        )
        return unavailable, ()

    def _render(self, claims: tuple[Claim, ...]) -> str:
        from plateproof.copilot.rendering import render_answer

        return render_answer(claims)

    def _insufficient(
        self,
        restaurant_id: str,
        jurisdiction: RestaurantJurisdiction,
        intent: Intent,
        now: datetime,
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
        jurisdiction: RestaurantJurisdiction | None,
        intent: Intent | None,
        reason: RefusalReason,
        detail: str,
        now: datetime,
        *,
        hint: tuple[Intent, ...] = (),
        local_helper_status: str = "not_consulted",
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
            local_helper_status=local_helper_status,
        )
