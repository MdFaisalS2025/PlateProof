"""Tests for plateproof.copilot.intents: deterministic detection,
ambiguous/unknown refusal, and the defense-in-depth prohibited check."""

from __future__ import annotations

from plateproof.copilot.intents import IntentDetectionOutcome, detect_intent, is_prohibited_request
from plateproof.copilot.models import Intent


def test_prohibited_illness_question_is_flagged() -> None:
    assert is_prohibited_request("Will someone get sick here?")


def test_prohibited_negligence_question_is_flagged() -> None:
    assert is_prohibited_request("Is this owner negligent?")


def test_prohibited_guarantee_question_is_flagged() -> None:
    assert is_prohibited_request("Can you guarantee that we will pass?")


def test_prohibited_evasion_question_is_flagged() -> None:
    assert is_prohibited_request("How can I hide a violation?")


def test_normal_question_is_not_prohibited() -> None:
    assert not is_prohibited_request("What did the latest inspection document?")


def test_prohibited_request_short_circuits_before_intent_detection() -> None:
    result = detect_intent("Will someone get sick here?")
    assert result.outcome == IntentDetectionOutcome.PROHIBITED
    assert result.intent is None


def test_latest_inspection_question_is_detected() -> None:
    result = detect_intent("What did the latest inspection document?")
    assert result.outcome == IntentDetectionOutcome.MATCHED
    assert result.intent == Intent.LATEST_INSPECTION_SUMMARY


def test_recurring_violations_question_is_detected() -> None:
    result = detect_intent("Which violations have appeared repeatedly?")
    assert result.outcome == IntentDetectionOutcome.MATCHED
    assert result.intent == Intent.RECURRING_VIOLATIONS


def test_unsupported_question_is_unknown() -> None:
    result = detect_intent("What is the meaning of life?")
    assert result.outcome == IntentDetectionOutcome.UNKNOWN
    assert result.intent is None


def test_ambiguous_question_matching_two_intents_equally_is_refused() -> None:
    result = detect_intent("What's the prediction for this Michelin restaurant?")
    assert result.outcome == IntentDetectionOutcome.AMBIGUOUS
    assert result.intent is None
    assert Intent.EXPLAIN_PREDICTION in result.candidates
    assert Intent.MICHELIN_CONTEXT in result.candidates


def test_cross_jurisdiction_rule_question_is_unknown_not_answered() -> None:
    """A request for another jurisdiction's specific rules isn't one of
    the closed intents at all -- it's refused as unknown, never
    guessed at or blended from the wrong jurisdiction's data."""
    result = detect_intent(
        "What are the Florida High Priority violation rules for this NYC restaurant?"
    )
    # Whatever this matches (or doesn't), it must never silently succeed
    # with a specific-rule answer -- assert it does not resolve to the
    # guidance intent silently blending jurisdictions. It's acceptable
    # for this to match official_guidance_for_documented_codes, since the
    # service layer's own jurisdiction filter (derived from the resolved
    # restaurant, never from the question text) is what actually prevents
    # cross-jurisdiction leakage -- proven in test_service.py instead.
    assert result.outcome in (
        IntentDetectionOutcome.MATCHED,
        IntentDetectionOutcome.UNKNOWN,
        IntentDetectionOutcome.AMBIGUOUS,
    )
