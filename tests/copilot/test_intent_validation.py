"""RED-first tests for plateproof.copilot.intent_validation.validate_intent_proposal
-- the strict, mechanical acceptance boundary for a local model's raw
intent proposal. This is the real security boundary (the transport layer's
own hardening is defense in depth on top of this)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from plateproof.copilot.intent_validation import (
    INTENT_FILTER_ALLOWLIST,
    IntentProposal,
    validate_intent_proposal,
)
from plateproof.copilot.models import Intent

_MIN_CONFIDENCE = 0.6


def test_exact_allowed_intent_with_sufficient_confidence_is_accepted() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert reason is None
    assert proposal == IntentProposal(
        intent=Intent.RECURRING_VIOLATIONS, confidence=0.9, filters={}
    )


def test_unknown_intent_string_is_rejected() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "delete_all_restaurants", "confidence": 0.9},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_low_confidence_is_rejected() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.1},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_confidence_exactly_at_minimum_is_accepted() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": _MIN_CONFIDENCE},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert reason is None
    assert proposal is not None


@pytest.mark.parametrize("bad_confidence", [math.nan, math.inf, -math.inf])
def test_non_finite_confidence_is_rejected(bad_confidence: float) -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": bad_confidence},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_boolean_confidence_is_rejected_even_though_bool_is_an_int_subclass() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": True},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_string_confidence_is_rejected() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": "0.9"},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_missing_confidence_is_rejected() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations"}, min_confidence=_MIN_CONFIDENCE
    )
    assert proposal is None
    assert reason is not None


def test_non_dict_top_level_is_rejected() -> None:
    proposal, reason = validate_intent_proposal(
        ["recurring_violations", 0.9], min_confidence=_MIN_CONFIDENCE
    )
    assert proposal is None
    assert reason is not None


@pytest.mark.parametrize(
    "extra_field",
    [
        "answer",
        "answer_text",
        "claims",
        "citations",
        "instructions",
        "tool_calls",
        "sql",
        "path",
        "url",
    ],
)
def test_unexpected_top_level_fields_are_rejected(extra_field: str) -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, extra_field: "anything"},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_unexpected_filter_key_is_rejected_since_no_intent_allows_any_filter_today() -> None:
    for intent in Intent:
        assert INTENT_FILTER_ALLOWLIST[intent] == frozenset()

    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": {"violation_code": "04L"}},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_empty_filters_object_is_accepted() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": {}},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert reason is None
    assert proposal is not None
    assert proposal.filters == {}


def test_filters_must_be_an_object() -> None:
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": ["04L"]},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None


def test_invalid_filter_value_is_rejected_when_a_filter_key_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowlist is empty for every intent today (YAGNI), but the
    value-validation logic itself is generic and reusable -- exercised
    here by monkeypatching one intent's allowlist to prove the value
    checks (type, length, non-empty) work correctly in isolation."""
    import plateproof.copilot.intent_validation as intent_validation_module

    monkeypatch.setitem(
        intent_validation_module.INTENT_FILTER_ALLOWLIST,
        Intent.RECURRING_VIOLATIONS,
        frozenset({"violation_code"}),
    )

    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": {"violation_code": ""}},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None

    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": {"violation_code": 123}},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None

    proposal, reason = validate_intent_proposal(
        {
            "intent": "recurring_violations",
            "confidence": 0.9,
            "filters": {"violation_code": "04L"},
        },
        min_confidence=_MIN_CONFIDENCE,
    )
    assert reason is None
    assert proposal is not None
    assert proposal.filters == {"violation_code": "04L"}


def test_oversized_filters_dict_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    import plateproof.copilot.intent_validation as intent_validation_module

    monkeypatch.setitem(
        intent_validation_module.INTENT_FILTER_ALLOWLIST,
        Intent.RECURRING_VIOLATIONS,
        frozenset({f"key{i}" for i in range(10)}),
    )
    filters: dict[str, Any] = {f"key{i}": "v" for i in range(10)}
    proposal, reason = validate_intent_proposal(
        {"intent": "recurring_violations", "confidence": 0.9, "filters": filters},
        min_confidence=_MIN_CONFIDENCE,
    )
    assert proposal is None
    assert reason is not None
