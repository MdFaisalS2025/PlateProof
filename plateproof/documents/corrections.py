"""Strict, mechanical validation for user-submitted corrections.

A correction is never accepted as free-form: the field must be one this
draft's jurisdiction actually defines, the value is length- and
control-character-bounded, and a date/numeric field is parsed with the
exact same jurisdiction-appropriate parser machine extraction itself uses.
An unparseable or rejected value is never silently dropped -- it is still
returned as a :class:`~plateproof.documents.models.ValidatedCorrection`
with a non-``"parsed"`` status, so the caller can render it distinctly and
block confirmation on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

from plateproof.documents.field_extraction import (
    FLORIDA_DATE_FORMATS,
    NYC_DATE_FORMATS,
    parse_date_multi,
    parse_nonneg_number,
)
from plateproof.documents.florida_extractor import FLORIDA_LABELS, _parse_disposition
from plateproof.documents.models import ExtractionDraft, ValidatedCorrection
from plateproof.documents.nyc_extractor import NYC_LABELS

#: Mirrors plateproof.copilot.question_validation's character-class
#: discipline: CRLF/CR -> LF, tab -> space, any other C0 control character
#: rejected outright (not silently stripped).
_ALLOWED_AFTER_NORMALIZATION = frozenset({"\n"})

MAX_CORRECTION_FIELD_LENGTH = 500
MAX_TOTAL_CORRECTION_PAYLOAD_LENGTH = 20_000


def _sanitize_text(value: str) -> str | None:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    for char in normalized:
        if ord(char) < 0x20 and char not in _ALLOWED_AFTER_NORMALIZATION:
            return None
    return normalized


def _known_fields_for(jurisdiction: str) -> frozenset[str]:
    return frozenset(NYC_LABELS) if jurisdiction == "nyc" else frozenset(FLORIDA_LABELS)


def _parser_for(jurisdiction: str, field_name: str) -> Callable[[str], Any] | None:
    """Returns the exact parser machine extraction uses for this field, or
    ``None`` for a free-text field (accepted as sanitized text)."""
    if jurisdiction == "nyc":
        if field_name == "score":
            return parse_nonneg_number
        if field_name == "inspection_date":
            return lambda text: parse_date_multi(text, NYC_DATE_FORMATS)
        return None
    if field_name in ("high_priority_count", "intermediate_count", "basic_count", "visit_sequence"):
        return parse_nonneg_number
    if field_name in ("inspection_date", "correction_deadline"):
        return lambda text: parse_date_multi(text, FLORIDA_DATE_FORMATS)
    if field_name == "disposition_status":
        return _parse_disposition
    return None


def validate_correction(
    field_name: str, raw_display_value: str, draft: ExtractionDraft
) -> tuple[ValidatedCorrection | None, str | None]:
    """Returns ``(validated, None)`` -- where ``validated.parse_status`` may
    still be ``"unparseable"``/``"rejected"`` -- or ``(None, reason)`` for a
    hard structural rejection (unknown field, too long, disallowed
    character) that never produces a correction object at all."""
    if field_name not in _known_fields_for(draft.jurisdiction_expected):
        return None, "unknown correction field for this document's jurisdiction"

    if len(raw_display_value) > MAX_CORRECTION_FIELD_LENGTH:
        return None, "correction value exceeds the maximum allowed length"

    sanitized = _sanitize_text(raw_display_value)
    if sanitized is None:
        return None, "correction value contains a disallowed control character"

    stripped = sanitized.strip()
    if not stripped:
        return (
            ValidatedCorrection(
                raw_display_value=raw_display_value,
                parsed_value=None,
                parse_status="unparseable",
                rejection_reason=None,
            ),
            None,
        )

    parser = _parser_for(draft.jurisdiction_expected, field_name)
    if parser is None:
        return (
            ValidatedCorrection(
                raw_display_value=raw_display_value,
                parsed_value=stripped[:MAX_CORRECTION_FIELD_LENGTH],
                parse_status="parsed",
                rejection_reason=None,
            ),
            None,
        )

    parsed = parser(stripped)
    if parsed is None:
        return (
            ValidatedCorrection(
                raw_display_value=raw_display_value,
                parsed_value=None,
                parse_status="unparseable",
                rejection_reason=None,
            ),
            None,
        )
    if isinstance(parsed, float) and not math.isfinite(parsed):
        return (
            ValidatedCorrection(
                raw_display_value=raw_display_value,
                parsed_value=None,
                parse_status="rejected",
                rejection_reason="value is not a finite number",
            ),
            None,
        )

    return (
        ValidatedCorrection(
            raw_display_value=raw_display_value,
            parsed_value=parsed,
            parse_status="parsed",
            rejection_reason=None,
        ),
        None,
    )


def validate_corrections(
    corrections: Mapping[str, str], draft: ExtractionDraft
) -> tuple[dict[str, ValidatedCorrection], str | None]:
    """Validate a whole batch of corrections for one document. The
    aggregate-size cap is checked before any per-field validation; a
    rejection here rejects the whole batch (nothing partially applied)."""
    total_length = sum(len(value) for value in corrections.values())
    if total_length > MAX_TOTAL_CORRECTION_PAYLOAD_LENGTH:
        return {}, "total correction payload exceeds the maximum allowed size"

    results: dict[str, ValidatedCorrection] = {}
    for field_name, raw_display_value in corrections.items():
        validated, reason = validate_correction(field_name, raw_display_value, draft)
        if reason is not None:
            return {}, reason
        assert validated is not None
        results[field_name] = validated
    return results, None
