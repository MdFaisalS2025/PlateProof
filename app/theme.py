"""Shared PlateProof visual identity and cached service-layer accessors for
every Streamlit page. Business/jurisdiction rules never live here or in a
page -- only in ``plateproof.serving.*`` -- pages are thin adapters, exactly
like the FastAPI routes.
"""

from __future__ import annotations

import atexit
from typing import Any

import streamlit as st

from plateproof.copilot.corpus import CorpusStore
from plateproof.copilot.generators.base import IntentHelper
from plateproof.copilot.service import CopilotService
from plateproof.copilot.wiring import build_corpus_store, build_intent_helper
from plateproof.core.config import Settings, get_settings
from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
from plateproof.graph.builder import GraphService
from plateproof.serving.display import INDEPENDENCE_STATEMENT
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.prediction_service import resolve_prediction
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


# Task 8B: graph/corpus/local-helper accessors, each built once per
# settings and cached for the life of this process -- never rebuilt on
# every script rerun. GraphService itself is cheap to construct (no I/O);
# the actual graph build is lazy (on first .get()) and then cached by
# source fingerprint, exactly like plateproof.api.main's wiring.
_graph_service_cache: dict[str, GraphService] = {}
_corpus_store_cache: dict[str, CorpusStore | None] = {}
_intent_helper_cache: dict[str, IntentHelper | None] = {}


def _copilot_cache_key(settings: Settings) -> str:
    return "|".join(
        str(v)
        for v in (
            settings.processed_data_dir,
            settings.guidance_corpus_manifest_path,
            settings.local_llm_enabled,
            settings.local_llm_base_url,
            settings.local_llm_model,
        )
    )


def graph_service() -> GraphService:
    settings = get_settings()
    key = _copilot_cache_key(settings)
    if key not in _graph_service_cache:
        _graph_service_cache[key] = GraphService(settings.resolve_path(settings.processed_data_dir))
    return _graph_service_cache[key]


def corpus_store() -> CorpusStore | None:
    settings = get_settings()
    key = _copilot_cache_key(settings)
    if key not in _corpus_store_cache:
        _corpus_store_cache[key] = build_corpus_store(settings)
    return _corpus_store_cache[key]


def intent_helper() -> IntentHelper | None:
    settings = get_settings()
    key = _copilot_cache_key(settings)
    if key not in _intent_helper_cache:
        _intent_helper_cache[key] = build_intent_helper(settings)
    return _intent_helper_cache[key]


def copilot_service() -> CopilotService:
    """A fresh, cheap ``CopilotService`` wrapper per call -- the expensive
    pieces it wraps (graph, corpus, intent helper) are each cached above,
    never rebuilt per call."""
    settings = get_settings()
    repo = repository()
    meta = model_metadata()

    def _forecast_lookup(restaurant_id: str, jurisdiction: str) -> Any:
        availability = resolve_prediction(
            repo,
            meta,
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            staleness_days=settings.prediction_staleness_days,
        )
        return availability.row if availability.status == "available" else None

    return CopilotService(
        graph_service().get().graph,
        corpus_store(),
        forecast_lookup=_forecast_lookup,
        intent_helper=intent_helper(),
        max_question_length=settings.copilot_max_question_length,
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


# Task 9B: this Streamlit process owns its OWN document worker pool --
# entirely separate from any FastAPI process's pool
# (``plateproof.api.main.create_app``). A ``multiprocessing`` pool cannot be
# meaningfully shared across unrelated parent processes, so the configured
# ``documents_worker_pool_size`` is a per-process limit, not a machine-wide
# one; see README.md for the combined-worker-count deployment-sizing
# consequence. Cached exactly like every other accessor above (a plain
# module-level dict, not ``st.cache_resource``, for AppTest compatibility) --
# built once per settings and reused for the life of this process.
_document_worker_pool_cache: dict[str, WorkerPool] = {}


def _documents_cache_key(settings: Settings) -> str:
    return "|".join(
        str(v)
        for v in (
            settings.documents_worker_pool_size,
            settings.documents_worker_page_timeout_seconds,
            settings.documents_worker_total_timeout_seconds,
            settings.documents_worker_kill_grace_seconds,
        )
    )


def document_worker_pool() -> WorkerPool:
    settings = get_settings()
    key = _documents_cache_key(settings)
    if key not in _document_worker_pool_cache:
        _document_worker_pool_cache[key] = WorkerPool(
            config=WorkerPoolConfig(
                pool_size=settings.documents_worker_pool_size,
                page_timeout_seconds=settings.documents_worker_page_timeout_seconds,
                total_timeout_seconds=settings.documents_worker_total_timeout_seconds,
                kill_grace_seconds=settings.documents_worker_kill_grace_seconds,
            )
        )
    return _document_worker_pool_cache[key]


def shutdown_document_worker_pools() -> None:
    """Explicit process-level shutdown path (Task 9B): terminates/joins/
    kills every live worker process this Streamlit process has spawned and
    closes their pipe handles. Registered below via ``atexit`` so a real
    deployment gets this for free (mirroring the FastAPI app's ``lifespan``
    shutdown handler); tests may also call it directly in teardown."""
    for pool in _document_worker_pool_cache.values():
        pool.shutdown()
    _document_worker_pool_cache.clear()


# Registered once per interpreter (module import is cached) -- covers every
# real Streamlit server process regardless of which page it started on,
# since every page imports this module.
atexit.register(shutdown_document_worker_pools)
