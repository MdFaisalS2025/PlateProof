"""Pure, Streamlit-free helpers for the Document Reader page's downloadable
JSON artifact (independent-review correction of ``c60cc80``, Finding 3).

Kept separate from ``pages/5_Document_Reader.py`` so the JSON contract --
what a machine candidate, a user correction, and a violation row look like
in the downloaded record -- is directly unit-testable without a Streamlit
``ScriptRunContext``.

Never includes: raw upload bytes, a worker-generated preview PNG, or a
filesystem path. A user correction's raw text, its separately-parsed typed
value, its parse status, and any rejection reason are always kept as
distinct fields -- never collapsed into one string -- and the
machine-proposed value is never overwritten by a user's correction to it.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from typing import Any

from plateproof.documents.models import ExtractionDraft, ValidatedCorrection
from plateproof.serving.display import USER_SUBMITTED_RECORD_DISCLAIMER


def asdict_json_safe(value: Any) -> Any:
    """Recursively converts a dataclass (and anything it contains) into
    plain JSON-safe types. Bytes are dropped entirely (never base64-
    encoded here -- a preview PNG is a display aid, never part of the
    user-submitted record); dates/datetimes become ISO-8601 strings."""
    if is_dataclass(value) and not isinstance(value, type):
        return {k: asdict_json_safe(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: asdict_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [asdict_json_safe(v) for v in value]
    if isinstance(value, bytes):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def build_downloadable_record(
    draft: ExtractionDraft, validated_corrections: dict[str, ValidatedCorrection]
) -> dict[str, Any]:
    """Builds the ``record_status="user_submitted"`` JSON artifact.

    ``validated_corrections`` must already be the output of
    ``plateproof.documents.corrections.validate_corrections`` -- this
    function never re-parses or re-validates a correction itself, and
    never accepts a plain ``dict[str, str]`` of raw text, so a caller
    cannot accidentally skip validation before building a record.
    """
    machine_candidates = {
        name: {
            "value": asdict_json_safe(candidate.value),
            "display_value": candidate.display_value,
            "confidence_label": candidate.confidence_label,
            "evidence": [asdict_json_safe(e) for e in candidate.evidence],
        }
        for name, candidate in draft.candidates.items()
    }
    user_corrections = {
        field_name: {
            "raw_display_value": validated.raw_display_value,
            "parsed_value": asdict_json_safe(validated.parsed_value),
            "parse_status": validated.parse_status,
            "rejection_reason": validated.rejection_reason,
        }
        for field_name, validated in validated_corrections.items()
    }
    return {
        "restaurant_id": draft.restaurant_id,
        "jurisdiction": draft.jurisdiction_expected,
        "confirmed_at": datetime.now(UTC).isoformat(),
        "machine_candidates": machine_candidates,
        "user_corrections": user_corrections,
        "violations": [asdict_json_safe(v) for v in draft.violations],
        "record_status": "user_submitted",
        "disclaimer": USER_SUBMITTED_RECORD_DISCLAIMER,
    }
