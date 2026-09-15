"""Detect which jurisdiction an uploaded document appears to describe, using
fixed keyword vocabularies -- never inferred from or compared silently
against the caller's declared expected jurisdiction until
:func:`check_jurisdiction_mismatch` is called explicitly. If both or neither
jurisdiction's markers are found, detection is honestly inconclusive
(``None``) rather than guessed.
"""

from __future__ import annotations

from collections.abc import Sequence

from plateproof.documents.models import Jurisdiction, TextBlock

NYC_MARKERS: tuple[str, ...] = (
    "department of health and mental hygiene",
    "dohmh",
    "city of new york",
)
FLORIDA_MARKERS: tuple[str, ...] = (
    "division of hotels and restaurants",
    "dbpr",
    "department of business and professional regulation",
)


def detect_jurisdiction(blocks: Sequence[TextBlock]) -> Jurisdiction | None:
    joined = " ".join(block.text for block in blocks).lower()
    nyc_hit = any(marker in joined for marker in NYC_MARKERS)
    florida_hit = any(marker in joined for marker in FLORIDA_MARKERS)
    if nyc_hit and not florida_hit:
        return "nyc"
    if florida_hit and not nyc_hit:
        return "florida"
    return None


def check_jurisdiction_mismatch(expected: Jurisdiction, detected: Jurisdiction | None) -> bool:
    """``False`` whenever detection is inconclusive -- absence of positive
    evidence is never treated as evidence of a mismatch."""
    if detected is None:
        return False
    return detected != expected
