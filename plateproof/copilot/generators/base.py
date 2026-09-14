"""Shared types every intent-helper implementation returns.
``CopilotService.answer`` depends only on these -- never on
``plateproof.copilot.generators.ollama`` directly -- so a future
alternative local-model backend can be substituted without touching the
service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from plateproof.copilot.intent_validation import IntentProposal
from plateproof.copilot.models import RestaurantJurisdiction


class IntentHelperOutcome(StrEnum):
    #: The helper returned a response that survived every check in
    #: ``validate_intent_proposal`` -- ``proposal`` is populated.
    ACCEPTED = "accepted"
    #: The helper returned a structurally-parseable response, but it
    #: failed strict validation (unknown intent, low/invalid confidence,
    #: an unexpected field, an invalid filter, ...).
    REJECTED = "rejected"
    #: No usable response was obtained at all: disabled/misconfigured
    #: transport, connection failure, timeout, oversized body, malformed
    #: JSON, or invalid UTF-8.
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class IntentHelperResult:
    outcome: IntentHelperOutcome
    proposal: IntentProposal | None
    latency_ms: float | None
    reason: str | None = None


class IntentHelper(Protocol):
    def propose(
        self, *, question: str, jurisdiction: RestaurantJurisdiction, now: datetime
    ) -> IntentHelperResult: ...
