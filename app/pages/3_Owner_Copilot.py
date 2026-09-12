"""PlateProof Copilot -- informational placeholder only. This page contains
no chat widget, no generated answers, no document upload, and no owner
workflow: those belong to Task 8/9 and are not implemented yet."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import configure_page, render_independence_footer  # noqa: E402

configure_page("Owner Copilot")
st.title("🧑‍🍳 PlateProof Copilot")

st.info(
    "**Coming in a later phase.** PlateProof Copilot will let restaurant owners ask "
    "questions about their inspection history and receive cited guidance grounded in "
    "official inspection rules and violation definitions.\n\n"
    "This page is a placeholder: there is no chat, no generated answers, and no document "
    "upload here yet."
)

render_independence_footer()
