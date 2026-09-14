"""RED-first tests for plateproof.copilot.wiring -- the shared
construction helpers FastAPI (app.state) and Streamlit (theme.py) both use
to build a cached CorpusStore and an optional intent helper from Settings,
exactly once per process, never per request."""

from __future__ import annotations

from typing import Any

from plateproof.copilot.generators.ollama import OllamaIntentHelper
from plateproof.copilot.wiring import build_corpus_store, build_intent_helper
from plateproof.core.config import Settings


def test_build_corpus_store_returns_none_when_not_configured() -> None:
    settings = Settings(_env_file=None, guidance_corpus_manifest_path=None)
    assert build_corpus_store(settings) is None


def test_build_corpus_store_loads_a_valid_fictional_corpus(write_guidance_corpus: Any) -> None:
    manifest_path = write_guidance_corpus()
    settings = Settings(_env_file=None, guidance_corpus_manifest_path=manifest_path)
    store = build_corpus_store(settings)
    assert store is not None
    assert store.passages


def test_build_corpus_store_returns_none_for_an_invalid_corpus(tmp_path: Any) -> None:
    bad_manifest = tmp_path / "manifest.json"
    bad_manifest.write_text("not json", encoding="utf-8")
    settings = Settings(_env_file=None, guidance_corpus_manifest_path=bad_manifest)
    assert build_corpus_store(settings) is None


def test_build_intent_helper_returns_none_when_disabled() -> None:
    settings = Settings(_env_file=None, local_llm_enabled=False)
    assert build_intent_helper(settings) is None


def test_build_intent_helper_returns_none_when_enabled_without_a_model() -> None:
    settings = Settings(_env_file=None, local_llm_enabled=True, local_llm_model=None)
    assert build_intent_helper(settings) is None


def test_build_intent_helper_returns_an_ollama_helper_when_fully_configured() -> None:
    settings = Settings(_env_file=None, local_llm_enabled=True, local_llm_model="fictional-model")
    helper = build_intent_helper(settings)
    assert isinstance(helper, OllamaIntentHelper)
