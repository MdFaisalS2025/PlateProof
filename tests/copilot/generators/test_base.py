"""RED-first tests for plateproof.copilot.generators.base's shared types."""

from __future__ import annotations

from datetime import UTC, datetime

from plateproof.copilot.generators.base import IntentHelperOutcome, IntentHelperResult
from plateproof.copilot.intent_validation import IntentProposal
from plateproof.copilot.models import Intent


def test_intent_helper_outcome_has_exactly_three_members() -> None:
    assert {member.value for member in IntentHelperOutcome} == {
        "accepted",
        "rejected",
        "unavailable",
    }


def test_intent_helper_result_carries_a_proposal_only_when_accepted() -> None:
    proposal = IntentProposal(intent=Intent.RECURRING_VIOLATIONS, confidence=0.9, filters={})
    result = IntentHelperResult(
        outcome=IntentHelperOutcome.ACCEPTED, proposal=proposal, latency_ms=12.5, reason=None
    )
    assert result.proposal is proposal
    assert result.outcome == IntentHelperOutcome.ACCEPTED


def test_intent_helper_result_rejected_carries_no_proposal() -> None:
    result = IntentHelperResult(
        outcome=IntentHelperOutcome.REJECTED,
        proposal=None,
        latency_ms=3.0,
        reason="confidence is below the minimum required threshold",
    )
    assert result.proposal is None
    assert result.reason is not None


def test_now_parameter_type_is_datetime_compatible() -> None:
    # Not a behavioral test -- just documents the Protocol's call shape by
    # constructing a value that satisfies it.
    now = datetime.now(UTC)
    assert isinstance(now, datetime)
