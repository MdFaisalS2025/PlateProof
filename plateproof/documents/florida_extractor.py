"""Florida-shaped candidate extraction from validated document text only.

Reuses the exact same disposition vocabulary Task 2/3's Florida ingestion
already validated (``plateproof.ingestion.florida.FL_DISPOSITION_STATUS``)
rather than reinventing it -- an on-document phrase that doesn't match one
of those 14 known strings is reported missing, never guessed into one of
the four buckets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from plateproof.documents.field_extraction import (
    FLORIDA_DATE_FORMATS,
    find_label_matches,
    parse_date_multi,
    parse_free_text,
    parse_nonneg_number,
    resolve_field,
)
from plateproof.documents.models import Ambiguity, Candidate, TextBlock
from plateproof.ingestion.florida import FL_DISPOSITION_STATUS

FLORIDA_LABELS: dict[str, tuple[str, ...]] = {
    "restaurant_name": ("Restaurant Name", "Establishment"),
    "inspection_date": ("Inspection Date",),
    "high_priority_count": ("High Priority",),
    "intermediate_count": ("Intermediate",),
    "basic_count": ("Basic",),
    "disposition_status": ("Disposition",),
}

FLORIDA_REQUIRED_FIELDS: tuple[str, ...] = (
    "restaurant_name",
    "inspection_date",
    "disposition_status",
)

_DISPOSITION_BY_CASEFOLD: dict[str, str] = {
    raw.casefold(): bucket for raw, bucket in FL_DISPOSITION_STATUS.items()
}


def _parse_disposition(text: str) -> str | None:
    return _DISPOSITION_BY_CASEFOLD.get(text.strip().casefold())


def _parser_for(field_name: str) -> Callable[[str], Any]:
    if field_name in ("high_priority_count", "intermediate_count", "basic_count"):
        return parse_nonneg_number
    if field_name == "inspection_date":
        return lambda text: parse_date_multi(text, FLORIDA_DATE_FORMATS)
    if field_name == "disposition_status":
        return _parse_disposition
    return parse_free_text


def extract_florida_candidates(
    blocks: Sequence[TextBlock],
) -> tuple[dict[str, Candidate[Any]], tuple[Ambiguity, ...], tuple[str, ...]]:
    candidates: dict[str, Candidate[Any]] = {}
    ambiguities: list[Ambiguity] = []

    for field_name, labels in FLORIDA_LABELS.items():
        matches = find_label_matches(blocks, labels)
        candidate: Candidate[Any] | None
        candidate, ambiguity = resolve_field(
            field_name, matches, parse_value=_parser_for(field_name)
        )
        if candidate is not None:
            candidates[field_name] = candidate
        if ambiguity is not None:
            ambiguities.append(ambiguity)

    missing = tuple(
        field_name for field_name in FLORIDA_REQUIRED_FIELDS if field_name not in candidates
    )
    return candidates, tuple(ambiguities), missing
