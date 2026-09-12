"""PlateProof Home page: what the product is, its independence statement,
and a jump-off point to restaurant search. No business logic lives here."""

from __future__ import annotations

import streamlit as st
from theme import configure_page, render_independence_footer

configure_page("Home")

st.title("🍽️ PlateProof")
st.subheader("Evidence behind every plate")

st.write(
    "PlateProof brings together official health-inspection records, violation histories, "
    "and (where available) Michelin recognition for restaurants in New York City and Florida."
)

st.markdown(
    """
**What you'll find here:**
- Official inspection history, exactly as published by NYC and Florida health authorities.
- A PlateProof forecast of upcoming inspection risk, shown as **Low / Moderate / High**,
  always with an uncertainty range and a model version -- never a bare number.
- Michelin recognition, shown as culinary context -- never as evidence of food safety.
- Direct links back to the original government source for every record.
"""
)

st.info(
    "NYC and Florida use different inspection systems. PlateProof always shows each "
    "jurisdiction's own official measure alongside any PlateProof forecast -- it never "
    "converts one system into the other."
)

st.warning(
    "Every PlateProof forecast is an estimate, not an official inspection result. "
    "Inspections themselves are point-in-time snapshots, not a continuous record."
)

st.page_link("pages/1_Restaurant_Search.py", label="🔎 Search restaurants", icon="🔎")

render_independence_footer()
