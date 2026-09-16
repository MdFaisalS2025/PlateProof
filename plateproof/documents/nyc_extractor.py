"""NYC-shaped candidate extraction from validated document text only.

Runs entirely in the calling (parent) process on plain, already-validated
strings returned by a worker -- never touches raw document bytes and never
calls PDFium/Pillow/OCR itself. Nothing here trains, scores, or otherwise
touches a Task 6 model.

NYC violation codes have no fixed enumerable vocabulary in this codebase
(unlike Florida's closed 1-58 category list) -- ``build_nyc_inspection_events``
accepts whatever the source extract provides, whitespace-collapsed and
casefolded (see ``plateproof.ingestion.nyc.normalize_violation_code``, reused
here rather than duplicated). A code found on a document is therefore always
accepted in its normalized form; there is no "unknown NYC code" concept to
reject against, and this module never guesses one from a description either
way.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from plateproof.documents.field_extraction import (
    NYC_DATE_FORMATS,
    classify_confidence,
    find_label_matches,
    find_violation_rows,
    parse_date_multi,
    parse_free_text,
    parse_nonneg_number,
    resolve_field,
)
from plateproof.documents.models import (
    Ambiguity,
    Candidate,
    EvidenceSpan,
    TextBlock,
    ViolationRowCandidate,
)
from plateproof.ingestion.nyc import normalize_violation_code

#: Each field's accepted on-document labels. Deliberately narrow and
#: specific (not generic words like "Date") to avoid colliding with
#: unrelated fields on the same form.
NYC_LABELS: dict[str, tuple[str, ...]] = {
    "camis": ("CAMIS",),
    "restaurant_name": ("Restaurant Name", "Establishment", "DBA"),
    "inspection_date": ("Inspection Date",),
    "inspection_type": ("Inspection Type",),
    "score": ("Score",),
    "grade": ("Grade",),
}

NYC_REQUIRED_FIELDS: tuple[str, ...] = ("restaurant_name", "inspection_date", "score")

NYC_VIOLATION_CODE_LABEL = "Violation Code"
NYC_VIOLATION_DESCRIPTION_LABELS: tuple[str, ...] = ("Violation Description", "Description")

_NOT_CRITICAL_PATTERN = re.compile(r"\bnot\s+critical\b|\bnon-?critical\b", re.IGNORECASE)
_CRITICAL_PATTERN = re.compile(r"\bcritical\b", re.IGNORECASE)


def _parser_for(field_name: str) -> Callable[[str], Any]:
    if field_name == "score":
        return parse_nonneg_number
    if field_name == "inspection_date":
        return lambda text: parse_date_multi(text, NYC_DATE_FORMATS)
    if field_name == "grade":
        return lambda text: parse_free_text(text, max_length=1)
    return parse_free_text


def _detect_critical_flag(window_text: str) -> bool | None:
    """Never guessed from the violation code or description -- only from an
    explicit "Critical"/"Not Critical" (or "Non-Critical") token found near
    this row. ``None`` when neither appears."""
    if _NOT_CRITICAL_PATTERN.search(window_text):
        return False
    if _CRITICAL_PATTERN.search(window_text):
        return True
    return None


def extract_nyc_violations(blocks: Sequence[TextBlock]) -> tuple[ViolationRowCandidate, ...]:
    """Every ``Violation Code`` occurrence becomes its own row -- multiple
    rows are never collapsed into a single scalar field."""
    matches = find_violation_rows(
        blocks,
        code_label=NYC_VIOLATION_CODE_LABEL,
        description_labels=NYC_VIOLATION_DESCRIPTION_LABELS,
        stop_labels=("Critical Flag",),
    )
    rows: list[ViolationRowCandidate] = []
    for match in matches:
        code = normalize_violation_code(match.raw_code_text)
        rows.append(
            ViolationRowCandidate(
                raw_code_text=match.raw_code_text,
                code=code,
                description=match.description_text,
                critical=_detect_critical_flag(match.window_text),
                evidence=(
                    EvidenceSpan(
                        page=match.page,
                        excerpt=match.excerpt,
                        bounding_box=None,
                        source=match.source,
                    ),
                ),
                confidence_label=classify_confidence(source=match.source),
            )
        )
    return tuple(rows)


def extract_nyc_candidates(
    blocks: Sequence[TextBlock],
) -> tuple[
    dict[str, Candidate[Any]],
    tuple[ViolationRowCandidate, ...],
    tuple[Ambiguity, ...],
    tuple[str, ...],
]:
    candidates: dict[str, Candidate[Any]] = {}
    ambiguities: list[Ambiguity] = []
    all_labels = tuple(label for labels in NYC_LABELS.values() for label in labels)

    for field_name, labels in NYC_LABELS.items():
        matches = find_label_matches(blocks, labels, stop_labels=all_labels)
        candidate: Candidate[Any] | None
        candidate, ambiguity = resolve_field(
            field_name, matches, parse_value=_parser_for(field_name)
        )
        if candidate is not None:
            candidates[field_name] = candidate
        if ambiguity is not None:
            ambiguities.append(ambiguity)

    violations = extract_nyc_violations(blocks)

    missing = tuple(
        field_name for field_name in NYC_REQUIRED_FIELDS if field_name not in candidates
    )
    return candidates, violations, tuple(ambiguities), missing
