"""RED-first tests for Task 8B settings validation (independent-review
correction): bounded numeric settings must fail loudly and early at
Settings construction for core/always-relevant parameters, but an invalid
*optional local-AI* setting must never make the whole application
unusable -- see plateproof.copilot.wiring for how that half is handled
(it disables the helper and reports a truthful status instead of raising).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plateproof.copilot.question_validation import ABSOLUTE_MAX_QUESTION_LENGTH
from plateproof.core.config import Settings


def test_absolute_question_length_ceiling_is_generous_but_bounded() -> None:
    assert 0 < ABSOLUTE_MAX_QUESTION_LENGTH <= 10_000


def test_default_settings_construct_without_error() -> None:
    Settings(_env_file=None)


def test_question_length_within_the_absolute_ceiling_is_accepted() -> None:
    settings = Settings(_env_file=None, copilot_max_question_length=200)
    assert settings.copilot_max_question_length == 200


def test_question_length_at_exactly_the_absolute_ceiling_is_accepted() -> None:
    settings = Settings(_env_file=None, copilot_max_question_length=ABSOLUTE_MAX_QUESTION_LENGTH)
    assert settings.copilot_max_question_length == ABSOLUTE_MAX_QUESTION_LENGTH


def test_question_length_above_the_absolute_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, copilot_max_question_length=ABSOLUTE_MAX_QUESTION_LENGTH + 1)


def test_zero_or_negative_question_length_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, copilot_max_question_length=0)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, copilot_max_question_length=-1)


def test_settings_construction_never_raises_for_local_llm_fields_alone() -> None:
    """A garbage local-AI value must not crash Settings() construction --
    plateproof.copilot.wiring is responsible for treating it as an
    unavailable/disabled helper, never the app failing to start."""
    Settings(
        _env_file=None,
        local_llm_enabled=True,
        local_llm_connect_timeout_seconds=-5.0,
        local_llm_read_timeout_seconds=-5.0,
        local_llm_max_response_bytes=-1,
        local_llm_min_confidence=99.0,
        local_llm_base_url="http://evil.example.com",
        local_llm_model="x" * 10_000,
    )
