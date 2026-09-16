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

#: Mirrors plateproof.ingestion.nyc.load_nyc_raw's date-parsing order (ISO,
#: then m/d/Y) and plateproof.ingestion.florida._parse_date's order
#: (m/d/Y, then ISO) -- deliberately kept separate per jurisdiction.
NYC_DATE_FORMATS: tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y")
FLORIDA_DATE_FORMATS: tuple[str, ...] = ("%m/%d/%Y", "%Y-%m-%d")


def classify_confidence(
    *, source: EvidenceSource, corroborated: bool | None = None
) -> ConfidenceLabel:
    """Categorical, evidence-condition-based (Finding 8) -- never a numeric
    average compared against a threshold. An OCR-sourced value is always
    ``"needs_review"`` regardless of its OCR/pattern scores: an OCR misread
    is numerically indistinguishable from a correct read. A field where
    corroboration was checked and explicitly failed is also
    ``"needs_review"``; everything else backed by embedded text is
    ``"high"``. Conflicting values and absent evidence never reach this
    function at all -- see ``resolve_field``."""
    if source == "ocr":
        return "needs_review"
    if corroborated is False:
        return "needs_review"
    return "high"


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


def find_label_matches(
    blocks: Sequence[TextBlock], labels: Sequence[str], *, stop_labels: Sequence[str] | None = None
) -> list[LabelMatch]:
    """Deterministic proximity match: for each label, find ``label`` followed
    by an optional ``:``/``-`` separator and a same-line value. Case
    insensitive; never uses any ML/fuzzy inference.

    ``stop_labels`` (defaulting to ``labels`` itself) bounds the captured
    value so it never greedily swallows a *different* labeled field on the
    same line -- a government form frequently packs several ``Label:
    value`` pairs onto one line (e.g. ``"CAMIS: 12345   DBA: Joe's
    Pizza"``). The value capture stops at the first of: two or more spaces,
    a tab, the start of another known label, or the end of the line.
    """
    resolved_stop_labels = stop_labels if stop_labels is not None else labels
    stop_alternation = "|".join(re.escape(label) for label in resolved_stop_labels)
    stop_lookahead = (
        rf"(?=\s{{2,}}|\t|(?:{stop_alternation})\s*[:\-]|\n|$)" if stop_alternation else r"(?=\n|$)"
    )

    matches: list[LabelMatch] = []
    patterns = {
        label: re.compile(
            re.escape(label) + r"\s*[:\-]?\s*([^\n]{1,80}?)" + stop_lookahead, re.IGNORECASE
        )
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


@dataclass(frozen=True, kw_only=True)
class ViolationRowMatch:
    raw_code_text: str
    description_text: str | None
    window_text: str
    page: int
    source: EvidenceSource
    excerpt: str


_MAX_VIOLATION_DESCRIPTION_LENGTH = 200


#: How many lines after the code's own line are considered part of "this
#: row" for description/critical-flag detection (e.g. a 3-line row: code,
#: then description, then a critical-flag indicator).
_VIOLATION_ROW_TRAILING_LINES = 2


def find_violation_rows(
    blocks: Sequence[TextBlock],
    *,
    code_label: str,
    description_labels: Sequence[str] = (),
    stop_labels: Sequence[str] = (),
) -> list[ViolationRowMatch]:
    """One row per occurrence of ``code_label`` in the text. Supports both
    same-line (``"Violation Code: 10F  Description: ..."``) and
    newline-separated (code on one line, description on one of the next
    few lines) layouts: the description is searched for first in the
    remainder of the code's own line, then in each following line up to
    ``_VIOLATION_ROW_TRAILING_LINES``.

    ``stop_labels`` (combined with ``description_labels``) bounds the
    description capture so it never swallows a *different* labeled field
    packed onto the same line (e.g. a trailing "Critical Flag: ..."),
    mirroring :func:`find_label_matches`'s same fix.

    Never infers a description from a code or a code from a description --
    each is only ever the literal text found near its own label.
    """
    code_pattern = re.compile(re.escape(code_label) + r"\s*[:\-]?\s*([^\s,;]{1,20})", re.IGNORECASE)
    desc_pattern = None
    if description_labels:
        alternation = "|".join(re.escape(label) for label in description_labels)
        stop_alternation = "|".join(
            re.escape(label) for label in (*description_labels, *stop_labels)
        )
        stop_lookahead = rf"(?=\s{{2,}}|\t|(?:{stop_alternation})\s*[:\-]|\n|$)"
        desc_pattern = re.compile(
            r"(?:"
            + alternation
            + r")\s*[:\-]?\s*([^\n]{1,"
            + str(_MAX_VIOLATION_DESCRIPTION_LENGTH)
            + r"}?)"
            + stop_lookahead,
            re.IGNORECASE,
        )

    rows: list[ViolationRowMatch] = []
    for block in blocks:
        text = block.text
        for match in code_pattern.finditer(text):
            raw_code = match.group(1).strip().rstrip(".,;:")
            if not raw_code:
                continue

            same_line_end = text.find("\n", match.end())
            if same_line_end == -1:
                same_line_end = len(text)
            same_line_remainder = text[match.end() : same_line_end]

            trailing_lines: list[str] = []
            cursor = same_line_end + 1
            for _ in range(_VIOLATION_ROW_TRAILING_LINES):
                if cursor > len(text):
                    break
                line_end = text.find("\n", cursor)
                if line_end == -1:
                    line_end = len(text)
                trailing_lines.append(text[cursor:line_end])
                cursor = line_end + 1

            description_text = None
            if desc_pattern is not None:
                candidates = [same_line_remainder, *trailing_lines]
                for candidate_text in candidates:
                    desc_match = desc_pattern.search(candidate_text)
                    if desc_match is not None:
                        stripped = desc_match.group(1).strip()
                        description_text = stripped or None
                        break

            window_text = same_line_remainder + "\n" + "\n".join(trailing_lines)
            rows.append(
                ViolationRowMatch(
                    raw_code_text=raw_code,
                    description_text=description_text,
                    window_text=window_text,
                    page=block.page,
                    source=block.source,
                    excerpt=_excerpt(text, match.start(), match.end()),
                )
            )
    return rows


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
    # This numeric score is informative metadata only (e.g. a display
    # bar) -- it never drives confidence_label (Finding 8).
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
        confidence_label=classify_confidence(source=best.source, corroborated=None),
    )
    return candidate, None
