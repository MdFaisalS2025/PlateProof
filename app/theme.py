"""Shared PlateProof visual identity and cached service-layer accessors for
every Streamlit page. Business/jurisdiction rules never live here or in a
page -- only in ``plateproof.serving.*`` -- pages are thin adapters, exactly
like the FastAPI routes.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from plateproof.core.config import Settings, get_settings
from plateproof.serving.display import INDEPENDENCE_STATEMENT
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import Repository, open_repository

PAGE_TITLE = "PlateProof"
PAGE_ICON = "🍽️"

# A restrained, high-contrast palette. No paid assets, no third-party logos.
_CSS = """
<style>
:root {
    --pp-navy: #1B2A4A;
    --pp-gold: #B8860B;
    --pp-bg: #FAFAF7;
    --pp-text: #1A1A1A;
    --pp-low: #2E7D32;
    --pp-moderate: #B8860B;
    --pp-high: #B3261E;
}
html, body, [class*="css"] {
    font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
}
.pp-risk-pill {
    display: inline-block;
    padding: 0.25em 0.75em;
    border-radius: 999px;
    font-weight: 600;
    border: 2px solid currentColor;
}
.pp-risk-low { color: var(--pp-low); }
.pp-risk-moderate { color: var(--pp-moderate); }
.pp-risk-high { color: var(--pp-high); }
.pp-risk-insufficient_history { color: #555555; }
.pp-disclaimer {
    font-size: 0.85em;
    color: #444444;
    border-left: 3px solid var(--pp-navy);
    padding-left: 0.75em;
    margin: 0.5em 0;
}
</style>
"""

_RISK_LABELS = {
    "low": "● Low PlateProof predicted risk",
    "moderate": "● Moderate PlateProof predicted risk",
    "high": "● High PlateProof predicted risk",
    "insufficient_history": "— Insufficient history for a forecast",
}


def configure_page(title: str) -> None:
    st.set_page_config(page_title=f"{PAGE_TITLE} - {title}", page_icon=PAGE_ICON, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def risk_band_html(risk_band: str) -> str:
    """Text label plus color -- never color alone (a screen reader or a
    printed page still conveys the band)."""
    label = _RISK_LABELS.get(risk_band, risk_band)
    return f'<span class="pp-risk-pill pp-risk-{risk_band}">{label}</span>'


def render_independence_footer() -> None:
    st.markdown("---")
    st.caption(INDEPENDENCE_STATEMENT)


# A plain module-level cache, not `st.cache_resource`: it survives for the
# life of this running process exactly like `cache_resource` would in a real
# Streamlit server, but behaves predictably under `AppTest` (which executes
# without a full ScriptRunContext, where Streamlit's own cache can't key
# reliably run-to-run).
_repository_cache: dict[str, Repository] = {}
_model_metadata_cache: dict[str, ModelMetadataReader] = {}


def repository() -> Repository:
    settings = get_settings()
    key = _cache_key(settings)
    if key not in _repository_cache:
        _repository_cache[key] = open_repository(settings)
    return _repository_cache[key]


def model_metadata() -> ModelMetadataReader:
    """Sanitized metadata only -- never deserializes or executes a model
    artifact. See ``plateproof.serving.model_registry_service``."""
    settings = get_settings()
    key = _cache_key(settings)
    if key not in _model_metadata_cache:
        _model_metadata_cache[key] = ModelMetadataReader(settings)
    return _model_metadata_cache[key]


def _cache_key(settings: Settings) -> str:
    return "|".join(
        str(v)
        for v in (
            settings.processed_data_dir,
            settings.nyc_model_artifact_path,
            settings.florida_model_artifact_path,
            settings.prediction_table_path,
        )
    )


def format_jurisdiction_measure_note(jurisdiction: str) -> str:
    if jurisdiction == "nyc":
        return (
            "NYC measures inspections using violation points and a letter grade (A/B/C). "
            "Lower points are better."
        )
    if jurisdiction == "florida":
        return (
            "Florida measures inspections using High Priority, Intermediate, and Basic "
            "violation counts plus a disposition -- it does not use letter grades."
        )
    return ""


def to_dict_list(rows: list[Any]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
