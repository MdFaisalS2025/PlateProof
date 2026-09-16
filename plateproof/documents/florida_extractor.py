"""Florida-shaped candidate extraction from validated document text only.

Reuses the exact same disposition vocabulary Task 2/3's Florida ingestion
already validated (``plateproof.ingestion.florida.FL_DISPOSITION_STATUS``)
rather than reinventing it -- an on-document phrase that doesn't match one
of those 14 known strings is reported missing, never guessed into one of
the four buckets. Violation codes reuse the same real closed 1-58 category
vocabulary (``plateproof.ingestion.florida.normalize_violation_code``): a
raw code outside that range is preserved as review-needed evidence with no
normalized value, never inferred from its description.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from plateproof.documents.field_extraction import (
    FLORIDA_DATE_FORMATS,
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
from plateproof.ingestion.florida import FL_DISPOSITION_STATUS, normalize_violation_code

FLORIDA_LABELS: dict[str, tuple[str, ...]] = {
    "license_number": ("License Number",),
    "inspection_visit_id": ("Inspection Visit ID",),
    "restaurant_name": ("Restaurant Name", "Establishment"),
    "inspection_date": ("Inspection Date",),
    "inspection_type": ("Inspection Type",),
    "visit_sequence": ("Visit Sequence",),
    "high_priority_count": ("High Priority",),
    "intermediate_count": ("Intermediate",),
    "basic_count": ("Basic",),
    "disposition_status": ("Disposition",),
    "correction_deadline": ("Correction Deadline", "Callback Date"),
}

FLORIDA_REQUIRED_FIELDS: tuple[str, ...] = (
    "restaurant_name",
    "inspection_date",
    "disposition_status",
)

FLORIDA_VIOLATION_CODE_LABEL = "Violation Code"
FLORIDA_VIOLATION_DESCRIPTION_LABELS: tuple[str, ...] = ("Violation Description", "Description")

_DISPOSITION_BY_CASEFOLD: dict[str, str] = {
    raw.casefold(): bucket for raw, bucket in FL_DISPOSITION_STATUS.items()
}


def _parse_disposition(text: str) -> str | None:
    return _DISPOSITION_BY_CASEFOLD.get(text.strip().casefold())


def _parser_for(field_name: str) -> Callable[[str], Any]:
    if field_name in ("high_priority_count", "intermediate_count", "basic_count", "visit_sequence"):
        return parse_nonneg_number
    if field_name in ("inspection_date", "correction_deadline"):
        return lambda text: parse_date_multi(text, FLORIDA_DATE_FORMATS)
    if field_name == "disposition_status":
        return _parse_disposition
    return parse_free_text


def extract_florida_violations(blocks: Sequence[TextBlock]) -> tuple[ViolationRowCandidate, ...]:
    """Every ``Violation Code`` occurrence becomes its own row -- multiple
    rows are never collapsed into a single scalar field. A raw code outside
    Florida's real 1-58 category range is preserved with ``code=None`` as
    review-needed evidence, never guessed from its description."""
    matches = find_violation_rows(
        blocks,
        code_label=FLORIDA_VIOLATION_CODE_LABEL,
        description_labels=FLORIDA_VIOLATION_DESCRIPTION_LABELS,
    )
    rows: list[ViolationRowCandidate] = []
    for match in matches:
        code = normalize_violation_code(match.raw_code_text)
        rows.append(
            ViolationRowCandidate(
                raw_code_text=match.raw_code_text,
                code=code,
                description=match.description_text,
                critical=None,  # not a Florida concept -- high_priority_count carries this instead
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


def extract_florida_candidates(
    blocks: Sequence[TextBlock],
) -> tuple[
    dict[str, Candidate[Any]],
    tuple[ViolationRowCandidate, ...],
    tuple[Ambiguity, ...],
    tuple[str, ...],
]:
    candidates: dict[str, Candidate[Any]] = {}
    ambiguities: list[Ambiguity] = []
    all_labels = tuple(label for labels in FLORIDA_LABELS.values() for label in labels)

    for field_name, labels in FLORIDA_LABELS.items():
        matches = find_label_matches(blocks, labels, stop_labels=all_labels)
        candidate: Candidate[Any] | None
        candidate, ambiguity = resolve_field(
            field_name, matches, parse_value=_parser_for(field_name)
        )
        if candidate is not None:
            candidates[field_name] = candidate
        if ambiguity is not None:
            ambiguities.append(ambiguity)

    violations = extract_florida_violations(blocks)

    missing = tuple(
        field_name for field_name in FLORIDA_REQUIRED_FIELDS if field_name not in candidates
    )
    return candidates, violations, tuple(ambiguities), missing
