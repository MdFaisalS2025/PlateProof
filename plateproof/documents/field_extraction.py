"""Jurisdiction-neutral extraction primitives: label-proximity value
matching, date/number parsing, and confidence combination. Runs entirely on
already-validated plain text returned by a worker -- never touches raw
document bytes or calls any native parser.

Nothing here ever invents a value: a field with no matching label is simply
absent (reported via the caller's missing-field list), and a field with
multiple conflicting values is reported as an ambiguity, never guessed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from plateproof.documents.models import (
    Ambiguity,
    Candidate,
    ConfidenceComponents,
    ConfidenceLabel,
    EvidenceSource,
    EvidenceSpan,
    TextBlock,
)

#: Matches ``plateproof.documents.corrections``' evidence-excerpt limit.
MAX_EVIDENCE_EXCERPT_LENGTH = 300
_EXCERPT_RADIUS = 40

CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MEDIUM_THRESHOLD = 0.5

#: Mirrors plateproof.ingestion.nyc.load_nyc_raw's date-parsing order (ISO,
#: then m/d/Y) and plateproof.ingestion.florida._parse_date's order
#: (m/d/Y, then ISO) -- deliberately kept separate per jurisdiction.
NYC_DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y")
FLORIDA_DATE_FORMATS: tuple[str, ...] = ("%m/%d/%Y", "%Y-%m-%d")


def classify_confidence(overall: float) -> ConfidenceLabel:
    if overall >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if overall >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def combine_confidence(
    *,
    ocr_confidence: float | None,
    pattern_confidence: float,
    corroboration_confidence: float | None,
) -> ConfidenceComponents:
    """OCR confidence and factual (pattern/corroboration) confidence are
    kept as separate fields on the result -- ``overall`` is a deterministic
    average of whichever components are actually available, never a
    fabricated single number when a component is absent."""
    components = [pattern_confidence]
    if ocr_confidence is not None:
        components.append(ocr_confidence)
    if corroboration_confidence is not None:
        components.append(corroboration_confidence)
    overall = sum(components) / len(components)
    return ConfidenceComponents(
        ocr_confidence=ocr_confidence,
        pattern_confidence=pattern_confidence,
        corroboration_confidence=corroboration_confidence,
        overall=overall,
    )


def _excerpt(text: str, start: int, end: int) -> str:
    lo = max(0, start - _EXCERPT_RADIUS)
    hi = min(len(text), end + _EXCERPT_RADIUS)
    return text[lo:hi][:MAX_EVIDENCE_EXCERPT_LENGTH]


@dataclass(frozen=True, kw_only=True)
class LabelMatch:
    label: str
    value_text: str
    page: int
    source: EvidenceSource
    ocr_confidence: float | None
    excerpt: str


def find_label_matches(blocks: Sequence[TextBlock], labels: Sequence[str]) -> list[LabelMatch]:
    """Deterministic proximity match: for each label, find ``label`` followed
    by an optional ``:``/``-`` separator and a same-line value. Case
    insensitive; never uses any ML/fuzzy inference."""
    matches: list[LabelMatch] = []
    patterns = {
        label: re.compile(re.escape(label) + r"\s*[:\-]?\s*([^\n]{1,80})", re.IGNORECASE)
        for label in labels
    }
    for block in blocks:
        for label in labels:
            for match in patterns[label].finditer(block.text):
                value_text = match.group(1).strip()
                if not value_text:
                    continue
                matches.append(
                    LabelMatch(
                        label=label,
                        value_text=value_text,
                        page=block.page,
                        source=block.source,
                        ocr_confidence=block.ocr_confidence,
                        excerpt=_excerpt(block.text, match.start(), match.end()),
                    )
                )
    return matches


def parse_date_multi(text: str, formats: Sequence[str]) -> date | None:
    stripped = text.strip()
    for fmt in formats:
        try:
            return datetime.strptime(stripped, fmt).date()
        except ValueError:
            continue
    return None


def parse_nonneg_number(text: str) -> float | None:
    stripped = text.strip().rstrip(".,")
    try:
        value = float(stripped)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def parse_free_text(text: str, *, max_length: int = 200) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    return stripped[:max_length]


def resolve_field[T](
    field_name: str,
    matches: Sequence[LabelMatch],
    *,
    parse_value: Callable[[str], T | None],
    corroboration_confidence: float | None = None,
) -> tuple[Candidate[T] | None, Ambiguity | None]:
    """Resolve one field's matches into exactly one of: a single confident
    candidate, an ambiguity (multiple distinct parsed values), or nothing
    (no matches, or none parseable) -- the caller treats "nothing" as
    missing. Never guesses between conflicting values."""
    if not matches:
        return None, None

    parsed_values: dict[T, list[LabelMatch]] = {}
    for match in matches:
        value = parse_value(match.value_text)
        if value is None:
            continue
        parsed_values.setdefault(value, []).append(match)

    if not parsed_values:
        return None, None

    if len(parsed_values) > 1:
        ambiguity = Ambiguity(
            field_name=field_name,
            reason="multiple_candidate_values",
            candidate_values=tuple(str(v) for v in parsed_values),
        )
        return None, ambiguity

    ((value, occurrences),) = parsed_values.items()
    best = occurrences[0]
    # A value repeated identically across independent matches is mildly
    # more trustworthy than a single occurrence, bounded well below 1.0.
    pattern_confidence = min(0.95, 0.85 + 0.02 * (len(occurrences) - 1))
    confidence = combine_confidence(
        ocr_confidence=best.ocr_confidence,
        pattern_confidence=pattern_confidence,
        corroboration_confidence=corroboration_confidence,
    )
    candidate: Candidate[T] = Candidate(
        field_name=field_name,
        value=value,
        display_value=best.value_text,
        evidence=(
            EvidenceSpan(
                page=best.page, excerpt=best.excerpt, bounding_box=None, source=best.source
            ),
        ),
        confidence=confidence,
        confidence_label=classify_confidence(confidence.overall),
    )
    return candidate, None
