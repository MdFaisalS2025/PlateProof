"""Shared construction helpers for the pieces every Copilot-serving
surface (the FastAPI app, the Streamlit UI) needs: a validated
``CorpusStore`` (or ``None``) and an optional local intent helper. Both
are built exactly once per process/settings by the caller (``app.state`` in
``plateproof.api.main``, a module-level cache in ``app.theme``) -- never
rebuilt per request."""

from __future__ import annotations

from plateproof.copilot.corpus import CorpusStore, load_corpus
from plateproof.copilot.generators.base import IntentHelper
from plateproof.copilot.generators.ollama import OllamaIntentHelper
from plateproof.core.config import Settings


def build_corpus_store(settings: Settings) -> CorpusStore | None:
    """``None`` both when the corpus isn't configured and when it fails
    validation -- either way, guidance-dependent intents fail closed to an
    honest ``GUIDANCE_UNAVAILABLE`` claim rather than the process
    crashing."""
    if settings.guidance_corpus_manifest_path is None:
        return None
    manifest_path = settings.resolve_path(settings.guidance_corpus_manifest_path)
    return load_corpus(manifest_path).store


def build_intent_helper(settings: Settings) -> IntentHelper | None:
    """``None`` unless local assistance is both enabled AND a model name
    is configured -- an enabled-but-modelless configuration is treated as
    not configured, never as a helper that would immediately fail every
    request."""
    if not settings.local_llm_enabled or not settings.local_llm_model:
        return None
    return OllamaIntentHelper(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        connect_timeout=settings.local_llm_connect_timeout_seconds,
        read_timeout=settings.local_llm_read_timeout_seconds,
        max_response_bytes=settings.local_llm_max_response_bytes,
        min_confidence=settings.local_llm_min_confidence,
    )
