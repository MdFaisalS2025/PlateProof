"""RED-first tests for app.theme's Task 8B cached accessors: graph_service,
corpus_store, intent_helper, and copilot_service. Mirrors the existing
plain-dict-cache pattern used by repository()/model_metadata() -- not
st.cache_resource, for AppTest compatibility."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def test_graph_service_returns_a_graph_service_instance(app_env: Any) -> None:
    import theme

    from plateproof.graph.builder import GraphService

    assert isinstance(theme.graph_service(), GraphService)


def test_corpus_store_is_none_when_not_configured(app_env: Any) -> None:
    app_env(guidance_corpus_manifest_path="")
    import theme

    theme._corpus_store_cache.clear()
    assert theme.corpus_store() is None


def test_intent_helper_is_none_when_local_ai_disabled(app_env: Any) -> None:
    import theme

    theme._intent_helper_cache.clear()
    assert theme.intent_helper() is None


def test_intent_helper_is_configured_when_enabled_with_a_model(app_env: Any) -> None:
    app_env(local_llm_enabled="true", local_llm_model="fictional-model")
    import theme

    from plateproof.copilot.generators.ollama import OllamaIntentHelper

    theme._intent_helper_cache.clear()
    assert isinstance(theme.intent_helper(), OllamaIntentHelper)


def test_copilot_service_returns_a_copilot_service_instance(app_env: Any) -> None:
    import theme

    from plateproof.copilot.service import CopilotService

    assert isinstance(theme.copilot_service(), CopilotService)
