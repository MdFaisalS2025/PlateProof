"""Deterministic intent detection and the defense-in-depth prohibited-
request check.

Only :class:`~plateproof.copilot.models.Intent` members may ever reach a
claim builder. Detection here is a fixed keyword-rule table, evaluated in
a defined order, with an explicit tie-break-to-ambiguous rule -- there is
no similarity threshold that alone authorizes an answer (a rule either
matches or it doesn't; ties are refused, never guessed). The optional
local-model intent *suggestion* (Task 8B) is validated against this same
closed enum before ever being trusted -- see the Task 8 plan's
``IntentValidationResult``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from plateproof.copilot.models import Intent

_WORD_PATTERN = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    """Whole-word, case-insensitive tokenization. Deliberately NOT a
    substring check: matching "is" as a substring would spuriously fire
    inside words like "this" or "history" and cause false ambiguous ties
    between unrelated intents."""
    return set(_WORD_PATTERN.findall(text.lower()))


# Defense-in-depth only: this is checked before intent detection and can
# never be bypassed by a matching intent rule, but it is not the
# authorization mechanism -- the closed Intent enum is. Categories:
# illness/harm, negligence/legal-liability, guaranteed outcomes, and
# rule-evasion requests.
_PROHIBITED_PATTERNS: tuple[str, ...] = (
    "get sick",
    "make someone sick",
    "food poisoning",
    "negligent",
    "negligence",
    "sue",
    "lawsuit",
    "legal liability",
    "guarantee",
    "guaranteed",
    "promise we",
    "hide a violation",
    "hide the violation",
    "cover up",
    "falsify",
    "bribe",
)


def is_prohibited_request(question: str) -> bool:
    lowered = question.lower()
    return any(pattern in lowered for pattern in _PROHIBITED_PATTERNS)


# One or more keyword groups per intent; a rule "matches" a question when
# at least one group is fully satisfied (every keyword in that group is
# present). The match *score* is the number of satisfied groups -- used
# only to detect a genuine ambiguous tie between two different intents,
# never to rank within one intent.
_INTENT_RULES: tuple[tuple[Intent, tuple[tuple[str, ...], ...]], ...] = (
    (
        Intent.RECURRING_VIOLATIONS,
        (("recurring",), ("repeat",), ("repeated",), ("repeatedly",)),
    ),
    (
        Intent.VIOLATION_HISTORY,
        (("violation", "history"), ("all", "violations")),
    ),
    (
        Intent.LATEST_INSPECTION_SUMMARY,
        (("latest", "inspection"), ("most", "recent", "inspection"), ("last", "inspection")),
    ),
    (
        Intent.INSPECTION_TREND,
        (
            ("trend",),
            ("changed", "over", "time"),
            ("history", "changed"),
            ("improving",),
            ("getting", "worse"),
        ),
    ),
    (
        Intent.PREPARATION_CHECKLIST_FROM_OFFICIAL_GUIDANCE,
        (("prepare", "next", "inspection"), ("checklist",), ("review", "before")),
    ),
    (
        Intent.OFFICIAL_GUIDANCE_FOR_DOCUMENTED_CODES,
        (("official", "guidance"), ("what", "guidance", "applies"), ("what", "does", "mean")),
    ),
    (
        Intent.EXPLAIN_PREDICTION,
        (("forecast",), ("prediction",), ("risk", "score")),
    ),
    (
        Intent.MICHELIN_CONTEXT,
        (("michelin",), ("star", "rating"), ("guide", "recognition")),
    ),
    (
        Intent.RESTAURANT_IDENTITY,
        (
            ("restaurant", "identity"),
            ("basic", "information"),
            ("tell", "me", "about", "restaurant"),
            ("cuisine", "type"),
        ),
    ),
)


class IntentDetectionOutcome(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class IntentDetectionResult:
    outcome: IntentDetectionOutcome
    intent: Intent | None
    candidates: tuple[Intent, ...] = ()


def _rule_score(words: set[str], groups: tuple[tuple[str, ...], ...]) -> int:
    return sum(1 for group in groups if all(keyword in words for keyword in group))


def detect_intent(question: str) -> IntentDetectionResult:
    """Deterministic, rule-based intent detection. Never guesses: a tie
    between two or more intents' top scores is reported as ambiguous, and
    zero matches is reported as unknown -- both are refusal conditions the
    caller must honor, not something this function papers over."""
    if is_prohibited_request(question):
        return IntentDetectionResult(IntentDetectionOutcome.PROHIBITED, None)

    words = _tokenize(question)
    scored: list[tuple[Intent, int]] = []
    for intent, groups in _INTENT_RULES:
        score = _rule_score(words, groups)
        if score > 0:
            scored.append((intent, score))

    if not scored:
        return IntentDetectionResult(IntentDetectionOutcome.UNKNOWN, None)

    top_score = max(score for _, score in scored)
    top_intents = tuple(sorted({intent for intent, score in scored if score == top_score}))
    if len(top_intents) > 1:
        return IntentDetectionResult(IntentDetectionOutcome.AMBIGUOUS, None, candidates=top_intents)
    return IntentDetectionResult(IntentDetectionOutcome.MATCHED, top_intents[0])


def intent_from_example_question_id(example_id: Intent) -> IntentDetectionResult:
    """The UI's example-question buttons map directly to an intent
    constant -- zero text parsing, zero ambiguity. This exists so the
    service layer has one call shape regardless of how the intent was
    selected."""
    return IntentDetectionResult(IntentDetectionOutcome.MATCHED, example_id)
