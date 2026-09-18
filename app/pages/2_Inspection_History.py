"""Restaurant detail page: official records, inspection/violation history,
PlateProof forecast, and Michelin context for one restaurant. Answers, in
order: what do official records show, how has it changed, is a forecast
available, how uncertain is it, what date/model produced it, and where to
verify the official record."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import (  # noqa: E402
    configure_page,
    format_jurisdiction_measure_note,
    model_metadata,
    render_independence_footer,
    repository,
    risk_band_html,
)

from plateproof.serving.prediction_service import resolve_prediction  # noqa: E402

configure_page("Inspection History")
st.title("📋 Restaurant Detail & Inspection History")

default_id = st.query_params.get("restaurant_id", "")
restaurant_id = st.text_input(
    "Official restaurant ID (e.g. nyc:12345678 or florida:HR1234567)", value=default_id
)

if not restaurant_id:
    st.info("Enter a restaurant ID above, or search from the Restaurant Search page.")
    render_independence_footer()
    st.stop()

repo = repository()

try:
    restaurant = repo.get_restaurant(restaurant_id)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not look up this restaurant: {type(exc).__name__}")
    restaurant = None
    st.stop()

if restaurant is None:
    st.error("No restaurant found for that ID.")
    render_independence_footer()
    st.stop()

restaurant_any: dict[str, Any] = dict(restaurant)
jurisdiction = restaurant_any["jurisdiction"]

st.header(restaurant_any.get("name") or "Unnamed restaurant")
st.caption(f"Official ID: `{restaurant_id}`  |  Jurisdiction: {jurisdiction.upper()}")
st.caption(f"{restaurant_any.get('address') or ''}, {restaurant_any.get('city') or ''}")
st.markdown(f"*{format_jurisdiction_measure_note(jurisdiction)}*")

# 1. What do official records currently show? -------------------------------
st.subheader("1. Official inspection history (snapshot records)")
inspections = repo.list_inspections(restaurant_id)
if not inspections:
    st.info("No inspection records are available for this restaurant.")
else:
    for row in inspections:
        row_any: dict[str, Any] = dict(row)
        with st.container(border=True):
            st.markdown(f"**{row_any['inspection_date']}** -- {row_any.get('inspection_type', '')}")
            if jurisdiction == "nyc":
                st.write(
                    f"Score: {row_any.get('score')}  |  Grade: {row_any.get('grade') or 'n/a'}  |  "
                    f"Critical violations: {row_any.get('critical_violation_count')}"
                )
            else:
                st.write(
                    f"High Priority: {row_any.get('high_priority_count')}  |  "
                    f"Intermediate: {row_any.get('intermediate_count')}  |  "
                    f"Basic: {row_any.get('basic_count')}  |  "
                    f"Disposition: {row_any.get('disposition_status') or 'n/a'}"
                )
            st.caption(f"Source snapshot date: {row_any.get('source_snapshot_date')}")

# 2. How has the record changed over time? -----------------------------------
st.subheader("2. Violation history")
violations = repo.list_violations(restaurant_id)
if not violations:
    st.info("No violation records are available for this restaurant.")
else:
    for row in violations:
        row_any = dict(row)
        st.write(
            f"{row_any['inspection_date']} -- `{row_any['violation_code']}`: "
            f"{row_any.get('description') or ''} "
            f"(severity: {row_any.get('severity') or 'unknown'}, count: {row_any.get('count')})"
        )

# 3-5. Forecast, uncertainty, model/date --------------------------------------
st.subheader("3. PlateProof forecast")
availability = resolve_prediction(
    repo,
    model_metadata(),
    restaurant_id=restaurant_id,
    jurisdiction=jurisdiction,
    staleness_days=90,
)
if availability.status == "no_ready_model":
    st.warning("No ready PlateProof model is currently configured for this jurisdiction.")
elif availability.status == "stale":
    st.warning("The most recent forecast for this restaurant is out of date (stale).")
elif availability.status == "not_scored":
    st.info("No PlateProof forecast has been computed for this restaurant yet.")
elif availability.status == "insufficient_history":
    st.info(
        "This restaurant does not yet have enough recorded history for a forecast "
        f"({availability.insufficient_history_reason or 'insufficient history'})."
    )
else:
    row_any = dict(availability.row)  # type: ignore[arg-type]
    st.markdown(risk_band_html(row_any["risk_band"]), unsafe_allow_html=True)
    st.write(
        f"Probability: {row_any['probability']:.2f} "
        f"(range {row_any['lower_bound']:.2f}-{row_any['upper_bound']:.2f})"
    )
    st.caption(
        f"Model version: {row_any['model_version']}  |  "
        f"As of: {row_any['as_of_date']}  |  Generated: {row_any['generated_at']}"
    )
    st.markdown(
        '<div class="pp-disclaimer">This is a PlateProof estimate, not an official '
        "inspection result.</div>",
        unsafe_allow_html=True,
    )

# Michelin context -------------------------------------------------------------
history = repo.michelin_history(restaurant_id)
if history:
    st.subheader("Michelin recognition (culinary context, not food-safety evidence)")
    for item in history:
        item_any = dict(item)
        st.write(f"{item_any['guide_year']}: {item_any['distinction']} ({item_any['guide_name']})")

# 6. Where can the user verify the official record? ---------------------------
st.subheader("Verify the official record")
from plateproof.serving.display import DATA_CORRECTION_NOTE, official_source_link  # noqa: E402

st.markdown(f"[Official {jurisdiction.upper()} dataset]({official_source_link(jurisdiction)})")
st.caption(DATA_CORRECTION_NOTE)

render_independence_footer()
