"""NYC-shaped candidate extraction from validated document text only.

Runs entirely in the calling (parent) process on plain, already-validated
strings returned by a worker -- never touches raw document bytes and never
calls PDFium/Pillow/OCR itself. Nothing here trains, scores, or otherwise
touches a Task 6 model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from plateproof.documents.field_extraction import (
    NYC_DATE_FORMATS,
    find_label_matches,
    parse_date_multi,
    parse_free_text,
    parse_nonneg_number,
    resolve_field,
)
from plateproof.documents.models import Ambiguity, Candidate, TextBlock

#: Each field's accepted on-document labels. Deliberately narrow and
#: specific (not generic words like "Date") to avoid colliding with
#: unrelated fields on the same form.
NYC_LABELS: dict[str, tuple[str, ...]] = {
    "restaurant_name": ("Restaurant Name", "Establishment", "DBA"),
    "inspection_date": ("Inspection Date",),
    "score": ("Score",),
    "grade": ("Grade",),
}

NYC_REQUIRED_FIELDS: tuple[str, ...] = ("restaurant_name", "inspection_date", "score")


def _parser_for(field_name: str) -> Callable[[str], Any]:
    if field_name == "score":
        return parse_nonneg_number
    if field_name == "inspection_date":
        return lambda text: parse_date_multi(text, NYC_DATE_FORMATS)
    if field_name == "grade":
        return lambda text: parse_free_text(text, max_length=1)
    return parse_free_text


def extract_nyc_candidates(
    blocks: Sequence[TextBlock],
) -> tuple[dict[str, Candidate[Any]], tuple[Ambiguity, ...], tuple[str, ...]]:
    candidates: dict[str, Candidate[Any]] = {}
    ambiguities: list[Ambiguity] = []

    for field_name, labels in NYC_LABELS.items():
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
        field_name for field_name in NYC_REQUIRED_FIELDS if field_name not in candidates
    )
    return candidates, tuple(ambiguities), missing
