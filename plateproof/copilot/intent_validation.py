"""The strict, mechanical acceptance boundary for a local model's raw
intent proposal. This -- not the transport layer's SSRF/timeout/bounded-
read hardening -- is the actual security boundary for Task 8B: nothing a
local model returns reaches :class:`~plateproof.copilot.service.CopilotService`
unless it survives every check in :func:`validate_intent_proposal`.

The accepted shape is exactly ``{"intent": str, "confidence": number,
"filters"?: object}``. No other top-level field is ever accepted --
``answer``, ``answer_text``, ``claims``, ``citations``, ``instructions``,
``tool_calls``, ``sql``, ``path``, ``url``, or anything else causes
rejection outright, never a partial/best-effort acceptance. There is
deliberately no ``jurisdiction`` field in this schema: jurisdiction is
always re-derived by ``CopilotService`` from the graph, never trusted from
a model response -- "jurisdiction consistency" is enforced by never
accepting a jurisdiction claim from the model at all, not by cross-
checking one.

``INTENT_FILTER_ALLOWLIST`` is empty for every :class:`Intent` today,
matching that ``CopilotService.answer_for_intent`` accepts no filter
arguments yet (YAGNI) -- so today any non-empty ``filters`` object is
rejected as containing an unexpected key. The validator itself is written
generically so a future intent can gain allowed filter keys without
changing this logic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from plateproof.copilot.models import Intent

# Every intent may be extended with allowed filter keys in the future;
# today none are, since no intent consumes a filter argument yet.
INTENT_FILTER_ALLOWLIST: dict[Intent, frozenset[str]] = {intent: frozenset() for intent in Intent}

_ALLOWED_TOP_LEVEL_KEYS = frozenset({"intent", "confidence", "filters"})
_MAX_FILTERS = 5
_MAX_FILTER_KEY_LENGTH = 64
_MAX_FILTER_VALUE_LENGTH = 128
_INTENT_BY_VALUE: dict[str, Intent] = {intent.value: intent for intent in Intent}


@dataclass(frozen=True)
class IntentProposal:
    intent: Intent
    confidence: float
    filters: Mapping[str, str]


def _validate_filters(
    raw_filters: object, intent: Intent
) -> tuple[dict[str, str] | None, str | None]:
    if not isinstance(raw_filters, dict):
        return None, "filters must be a JSON object"
    if len(raw_filters) > _MAX_FILTERS:
        return None, "filters object has too many entries"

    allowed_keys = INTENT_FILTER_ALLOWLIST.get(intent, frozenset())
    validated: dict[str, str] = {}
    for key, value in raw_filters.items():
        if not isinstance(key, str) or key not in allowed_keys:
            return None, "filters contains an unexpected key for this intent"
        if not isinstance(value, str) or not value:
            return None, "a filter value is not a valid non-empty string"
        if len(key) > _MAX_FILTER_KEY_LENGTH or len(value) > _MAX_FILTER_VALUE_LENGTH:
            return None, "a filter key or value exceeds the maximum allowed length"
        validated[key] = value
    return validated, None


def validate_intent_proposal(
    raw: object, *, min_confidence: float
) -> tuple[IntentProposal | None, str | None]:
    """Returns ``(proposal, None)`` on acceptance or ``(None, reason)`` on
    rejection. Never raises -- ``raw`` is fully untrusted, already-parsed
    JSON (a plain Python object from ``json.loads``), never re-parsed
    here."""
    if not isinstance(raw, dict):
        return None, "response is not a JSON object"

    unexpected = set(raw.keys()) - _ALLOWED_TOP_LEVEL_KEYS
    if unexpected:
        return None, "response contains an unexpected field"

    raw_intent = raw.get("intent")
    if not isinstance(raw_intent, str) or raw_intent not in _INTENT_BY_VALUE:
        return None, "intent is not one of the supported closed values"
    intent = _INTENT_BY_VALUE[raw_intent]

    raw_confidence = raw.get("confidence")
    # bool is a subclass of int in Python -- checked first so a boolean
    # confidence is rejected rather than silently coerced to 0.0/1.0.
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, int | float):
        return None, "confidence is not a number"
    if not math.isfinite(raw_confidence):
        return None, "confidence is not a finite number"
    confidence = float(raw_confidence)
    if confidence < min_confidence:
        return None, "confidence is below the minimum required threshold"

    filters, reason = _validate_filters(raw.get("filters", {}), intent)
    if reason is not None:
        return None, reason
    assert filters is not None

    return IntentProposal(intent=intent, confidence=confidence, filters=filters), None
