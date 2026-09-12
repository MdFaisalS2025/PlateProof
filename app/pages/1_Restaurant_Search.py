"""Restaurant search page: a thin adapter over
``plateproof.serving.repository.Repository.search_restaurants``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import configure_page, render_independence_footer, repository  # noqa: E402

configure_page("Restaurant Search")
st.title("🔎 Restaurant Search")

with st.form("search_form"):
    query = st.text_input("Restaurant name", value="")
    jurisdiction_choice = st.selectbox("Jurisdiction", options=["Any", "NYC", "Florida"])
    submitted = st.form_submit_button("Search")

jurisdiction: str | None
if jurisdiction_choice == "NYC":
    jurisdiction = "nyc"
elif jurisdiction_choice == "Florida":
    jurisdiction = "florida"
else:
    jurisdiction = None

repo = repository()

try:
    result = repo.search_restaurants(query=query, jurisdiction=jurisdiction, limit=25, offset=0)
except Exception as exc:  # noqa: BLE001 - shown as a page message, not a traceback
    st.error(f"Search could not be completed: {type(exc).__name__}")
    result = None

if result is not None:
    if not result.results:
        st.info("No restaurants matched your search.")
    for row in result.results:
        row_any: dict[str, Any] = dict(row)
        with st.container(border=True):
            st.markdown(f"**{row_any.get('name', 'Unnamed restaurant')}**")
            st.caption(
                f"{row_any.get('address') or ''}, {row_any.get('city') or ''} "
                f"({row_any.get('jurisdiction', '').upper()})"
            )
            st.caption(f"Official ID: `{row_any.get('restaurant_id')}`")
            if row_any.get("latest_documented_michelin_distinctions"):
                st.caption(
                    "Michelin (culinary context only): "
                    + ", ".join(row_any["latest_documented_michelin_distinctions"])
                    + f" ({row_any.get('latest_documented_guide_year')})"
                )

    for warning in result.warnings:
        st.warning(warning)

render_independence_footer()
