"""RED-first tests for the independent-review correction to
plateproof.copilot.wiring: explicit bounded validation of every Task 8B
local-AI setting, feeding both build_intent_helper() (never raises, just
declines to construct a helper) and local_ai_health_status() (truthful,
network-free health reporting)."""

from __future__ import annotations

from typing import Any

import pytest

from plateproof.copilot.generators.ollama import OllamaIntentHelper
from plateproof.copilot.wiring import (
    build_intent_helper,
    local_ai_health_status,
    validate_local_llm_settings,
)
from plateproof.core.config import Settings


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "local_llm_enabled": True,
        "local_llm_model": "fictional-model",
    }
    base.update(overrides)
    return Settings(**base)


def test_default_enabled_configuration_is_valid() -> None:
    valid, reason = validate_local_llm_settings(_settings())
    assert valid
    assert reason is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_llm_connect_timeout_seconds": 0.0},
        {"local_llm_connect_timeout_seconds": -1.0},
        {"local_llm_connect_timeout_seconds": float("inf")},
        {"local_llm_connect_timeout_seconds": float("nan")},
        {"local_llm_connect_timeout_seconds": 999.0},
        {"local_llm_read_timeout_seconds": 0.0},
        {"local_llm_read_timeout_seconds": -1.0},
        {"local_llm_read_timeout_seconds": float("inf")},
        {"local_llm_read_timeout_seconds": 999.0},
        {"local_llm_max_response_bytes": 0},
        {"local_llm_max_response_bytes": -1},
        {"local_llm_max_response_bytes": 10_000_000_000},
        {"local_llm_min_confidence": -0.1},
        {"local_llm_min_confidence": 1.1},
        {"local_llm_min_confidence": float("nan")},
        {"local_llm_base_url": "http://evil.example.com:11434"},
        {"local_llm_base_url": "not a url"},
        {"local_llm_model": ""},
        {"local_llm_model": "x" * 10_000},
        {"local_llm_model": "bad\x00model"},
        {"local_llm_model": "bad\x07model"},
    ],
)
def test_invalid_local_llm_configuration_is_rejected(overrides: dict[str, Any]) -> None:
    valid, reason = validate_local_llm_settings(_settings(**overrides))
    assert not valid
    assert reason is not None


def test_build_intent_helper_returns_none_for_invalid_configuration_never_raises() -> None:
    helper = build_intent_helper(_settings(local_llm_min_confidence=99.0))
    assert helper is None


def test_build_intent_helper_returns_a_helper_for_valid_configuration() -> None:
    helper = build_intent_helper(_settings())
    assert isinstance(helper, OllamaIntentHelper)


def test_local_ai_health_status_disabled_when_not_enabled() -> None:
    status, detail = local_ai_health_status(Settings(_env_file=None, local_llm_enabled=False))
    assert status == "disabled"


def test_local_ai_health_status_configured_unverified_when_valid() -> None:
    status, detail = local_ai_health_status(_settings())
    assert status == "configured_unverified"


def test_local_ai_health_status_unavailable_when_enabled_but_missing_model() -> None:
    status, detail = local_ai_health_status(
        Settings(_env_file=None, local_llm_enabled=True, local_llm_model=None)
    )
    assert status == "unavailable"
    assert detail is not None


def test_local_ai_health_status_unavailable_when_enabled_with_invalid_url() -> None:
    status, detail = local_ai_health_status(
        _settings(local_llm_base_url="http://evil.example.com:11434")
    )
    assert status == "unavailable"


def test_local_ai_health_status_performs_no_network_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tripwire: patch the transport's connection factory to raise if
    ever called, then prove health status construction never invokes it."""
    import plateproof.copilot.generators.ollama as ollama_module

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("local_ai_health_status must never open a connection")

    monkeypatch.setattr(ollama_module, "_default_connection_factory", _boom)
    local_ai_health_status(_settings())
