"""RED-first tests for Task 8B's new Settings fields."""

from __future__ import annotations

from pathlib import Path

from plateproof.core.config import Settings


def test_task_8b_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.guidance_corpus_manifest_path == Path("data/reference/guidance/manifest.json")
    assert settings.local_llm_connect_timeout_seconds == 2.0
    assert settings.local_llm_read_timeout_seconds == 5.0
    assert settings.local_llm_max_response_bytes == 65_536
    assert settings.local_llm_min_confidence == 0.6
    assert settings.copilot_max_question_length == 500


def test_task_8b_settings_are_overridable() -> None:
    settings = Settings(
        _env_file=None,
        guidance_corpus_manifest_path=None,
        local_llm_connect_timeout_seconds=1.0,
        local_llm_read_timeout_seconds=3.0,
        local_llm_max_response_bytes=1_000,
        local_llm_min_confidence=0.9,
        copilot_max_question_length=100,
    )
    assert settings.guidance_corpus_manifest_path is None
    assert settings.local_llm_connect_timeout_seconds == 1.0
    assert settings.local_llm_read_timeout_seconds == 3.0
    assert settings.local_llm_max_response_bytes == 1_000
    assert settings.local_llm_min_confidence == 0.9
    assert settings.copilot_max_question_length == 100
