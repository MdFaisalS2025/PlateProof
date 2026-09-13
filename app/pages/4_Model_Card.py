"""Model card page: renders a jurisdiction's model card Markdown safely
(never with unsafe HTML enabled) and clearly states when no ready model is
configured."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import configure_page, model_metadata, render_independence_footer  # noqa: E402

configure_page("Model Card")
st.title("📄 PlateProof Model Card")

jurisdiction_choice = st.selectbox("Jurisdiction", options=["NYC", "Florida"])
jurisdiction = "nyc" if jurisdiction_choice == "NYC" else "florida"

metadata = model_metadata().get(jurisdiction)
if metadata is None:
    st.warning(
        f"No ready PlateProof model is currently configured for {jurisdiction_choice}. "
        "A model card is only shown for a fully validated, production-ready model."
    )
else:
    st.caption(f"Readiness status: `{metadata.deployment_status}`")
    # Safe Markdown rendering only -- unsafe_allow_html is never enabled here.
    st.markdown(metadata.model_card_markdown)

render_independence_footer()
