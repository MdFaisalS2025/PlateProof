"""PlateProof Copilot -- restaurant-scoped, evidence-grounded Q&A (Task 8B).

Every factual sentence in an answer is produced by CopilotService's
deterministic claim builders and fixed rendering templates from authorized
evidence -- this page never generates free-form AI text. An optional local
model (Ollama) may only ever choose WHICH closed, supported question a
free-text question maps to when the deterministic detector cannot resolve
it by itself; see plateproof.copilot.service.CopilotService.answer.

Uses only safe Streamlit primitives -- never Streamlit's raw-HTML
rendering option anywhere on this page. Official source links use
st.link_button, never a raw href."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import (  # noqa: E402
    configure_page,
    copilot_service,
    format_jurisdiction_measure_note,
    get_settings,
    render_independence_footer,
    repository,
)

from plateproof.copilot.models import Claim, ClaimType, CopilotAnswer, Intent  # noqa: E402
from plateproof.copilot.rendering import render_claim  # noqa: E402

configure_page("Owner Copilot")
st.title("🧑‍🍳 PlateProof Copilot")
st.write(
    "Ask questions about a restaurant's documented inspection history and official "
    "guidance. Every answer is grounded in PlateProof's own documented records and "
    "reviewed official guidance -- never a freely generated response."
)
st.info(
    "**PlateProof does not verify restaurant ownership.** Anyone can look up any "
    "documented restaurant here; this page does not confirm you own or operate it."
)

_EXAMPLE_LABELS: dict[Intent, str] = {
    Intent.LATEST_INSPECTION_SUMMARY: "Latest inspection summary",
    Intent.RECURRING_VIOLATIONS: "Which violations are recurring?",
    Intent.VIOLATION_HISTORY: "Full documented violation history",
    Intent.INSPECTION_TREND: "How has our record changed over time?",
    Intent.OFFICIAL_GUIDANCE_FOR_DOCUMENTED_CODES: "Official guidance for our violation codes",
    Intent.PREPARATION_CHECKLIST_FROM_OFFICIAL_GUIDANCE: (
        "Checklist to prepare for our next inspection"
    ),
    Intent.EXPLAIN_PREDICTION: "Explain our PlateProof forecast",
    Intent.RESTAURANT_IDENTITY: "Basic documented restaurant information",
    Intent.MICHELIN_CONTEXT: "Michelin Guide recognition context",
}

_INSPECTION_CLAIM_TYPES = frozenset(
    {
        ClaimType.LATEST_INSPECTION,
        ClaimType.RECURRING_VIOLATION,
        ClaimType.VIOLATION_FREQUENCY,
        ClaimType.HISTORY_TREND,
        ClaimType.RESTAURANT_IDENTITY,
    }
)
_FORECAST_CLAIM_TYPES = frozenset({ClaimType.FORECAST_AVAILABILITY})
_GUIDANCE_CLAIM_TYPES = frozenset(
    {ClaimType.GUIDANCE_FOR_CODE, ClaimType.GUIDANCE_FOR_TOPIC, ClaimType.GUIDANCE_UNAVAILABLE}
)
_MICHELIN_CLAIM_TYPES = frozenset({ClaimType.MICHELIN_CONTEXT})

if "copilot_answer" not in st.session_state:
    st.session_state["copilot_answer"] = None

settings = get_settings()
repo = repository()

if settings.local_llm_enabled:
    st.caption(
        "Local AI assistance: enabled (used only when your question is ambiguous or unrecognized)"
    )
else:
    st.caption("Local AI assistance: disabled (PlateProof still answers every supported question)")

# --- Restaurant search/selection --------------------------------------- #
with st.expander("Search by name instead of ID"):
    search_query = st.text_input("Restaurant name", key="copilot_search_query")
    if st.button("Search", key="copilot_search_button") and search_query:
        try:
            result = repo.search_restaurants(
                query=search_query, jurisdiction=None, limit=10, offset=0
            )
        except Exception as exc:  # noqa: BLE001 - shown as a page message, not a traceback
            st.error(f"Search could not be completed: {type(exc).__name__}")
            result = None
        if result is not None:
            if not result.results:
                st.info("No restaurants matched that search.")
            for row in result.results:
                row_any: dict[str, Any] = dict(row)
                st.write(
                    f"`{row_any['restaurant_id']}` -- {row_any.get('name', 'Unnamed restaurant')}"
                )

default_id = st.query_params.get("restaurant_id", "")
restaurant_id = st.text_input(
    "Official restaurant ID (e.g. nyc:12345678 or florida:HR1234567)",
    value=default_id,
    key="copilot_restaurant_id",
)

if not restaurant_id:
    st.info("Enter a restaurant ID above, or search by name, to start.")
    render_independence_footer()
    st.stop()

try:
    restaurant = repo.get_restaurant(restaurant_id)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not look up this restaurant: {type(exc).__name__}")
    render_independence_footer()
    st.stop()

if restaurant is None:
    st.error("No restaurant found for that ID.")
    render_independence_footer()
    st.stop()

restaurant_any: dict[str, Any] = dict(restaurant)
jurisdiction = restaurant_any["jurisdiction"]

st.header(restaurant_any.get("name") or "Unnamed restaurant")
st.caption(f"Official ID: `{restaurant_id}`  |  Jurisdiction: {jurisdiction.upper()}")
st.write(format_jurisdiction_measure_note(jurisdiction))

# --- Example questions ---------------------------------------------------- #
st.subheader("Example questions")
example_cols = st.columns(3)
for index, (intent, label) in enumerate(_EXAMPLE_LABELS.items()):
    with example_cols[index % 3]:
        if st.button(label, key=f"copilot_intent_{intent.value}"):
            st.session_state["copilot_answer"] = copilot_service().answer_for_intent(
                restaurant_id=restaurant_id, intent=intent
            )

# --- Free-text question ---------------------------------------------------- #
st.subheader("Or ask your own question")
question = st.text_area(
    "Your question",
    key="copilot_question",
    max_chars=settings.copilot_max_question_length,
    placeholder="e.g. What official guidance applies to our violation codes?",
)
if st.button("Ask PlateProof Copilot", key="copilot_ask_button"):
    if not question or not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("PlateProof Copilot is checking documented records..."):
            st.session_state["copilot_answer"] = copilot_service().answer(
                restaurant_id=restaurant_id, question=question
            )

# --- Answer display -------------------------------------------------------- #
answer: CopilotAnswer | None = st.session_state.get("copilot_answer")

if answer is not None:
    st.divider()
    if answer.grounding_status == "refused":
        assert answer.refusal is not None
        st.warning(answer.refusal.detail)
        if answer.refusal.supported_intents_hint:
            st.write("PlateProof currently supports questions like:")
            for hinted_intent in answer.refusal.supported_intents_hint:
                st.write(f"- {_EXAMPLE_LABELS.get(hinted_intent, hinted_intent.value)}")
    else:
        if answer.generator_mode == "local_llm_assisted":
            st.success(
                "Mode: Local AI assisted intent (PlateProof's own evidence still answered "
                "the question)"
            )
        else:
            st.info("Mode: Deterministic evidence mode")

        st.write(answer.answer_text)

        claims_by_category: dict[str, list[Claim]] = {
            "Documented inspection history": [],
            "PlateProof forecast": [],
            "Official guidance": [],
            "Michelin context": [],
        }
        for claim in answer.claims:
            if claim.claim_type in _INSPECTION_CLAIM_TYPES:
                claims_by_category["Documented inspection history"].append(claim)
            elif claim.claim_type in _FORECAST_CLAIM_TYPES:
                claims_by_category["PlateProof forecast"].append(claim)
            elif claim.claim_type in _GUIDANCE_CLAIM_TYPES:
                claims_by_category["Official guidance"].append(claim)
            elif claim.claim_type in _MICHELIN_CLAIM_TYPES:
                claims_by_category["Michelin context"].append(claim)

        for category, category_claims in claims_by_category.items():
            if not category_claims:
                continue
            st.subheader(category)
            if category == "PlateProof forecast":
                st.caption(
                    "This is a statistical estimate, not a guarantee of any future inspection "
                    "outcome."
                )
            if category == "Michelin context":
                st.caption(
                    "Michelin recognition is contextual culinary information -- "
                    "it does not indicate food safety or predict inspection outcomes."
                )
            for claim in category_claims:
                st.write(render_claim(claim))

        with st.expander("Why PlateProof says this"):
            if not answer.claims:
                st.write("No documented claims back this answer.")
            for claim in answer.claims:
                st.write(f"- {render_claim(claim)}")
                if claim.evidence_ids:
                    st.caption(f"Evidence: {', '.join(claim.evidence_ids)}")

        if answer.citations:
            st.subheader("Citations")
            for citation in answer.citations:
                with st.container(border=True):
                    st.write(f"**{citation.title}**")
                    if citation.issuing_authority:
                        st.caption(f"Issuing authority: {citation.issuing_authority}")
                    if citation.section_locator:
                        st.caption(f"Section: {citation.section_locator}")
                    if citation.access_date:
                        st.caption(f"Accessed: {citation.access_date}")
                    if citation.effective_date:
                        st.caption(f"Effective: {citation.effective_date}")
                    st.caption(f"Jurisdiction: {citation.jurisdiction}")
                    st.caption(citation.excerpt)
                    if citation.url:
                        st.link_button("View official source", citation.url)

    st.caption(answer.disclaimer)

render_independence_footer()
