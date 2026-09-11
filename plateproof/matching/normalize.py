"""Deterministic, locally testable normalization for restaurant identity matching.

No network access, no paid services, no semantic/ML inference. Every function
keeps the original value available to its caller -- nothing here discards
source text; normalization only ever produces an additional, comparable form.

Scope note: normalization here is deliberately mechanical (case, punctuation,
accents, a short legal-entity suffix list, a fixed street-vocabulary map). It
does not attempt to parse out chef names or hotel names, because doing so would
risk merging restaurants that are genuinely distinct (two different restaurants
sharing one hotel's name, for example). That disambiguation is left to the
matching layer in :mod:`plateproof.matching.entity_resolution`, which combines
name similarity with address/postal/city evidence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_BUSINESS_SUFFIXES = frozenset({"llc", "inc", "corp", "co", "ltd", "llp", "lp"})

_DIRECTIONALS = {
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
    "northeast": "ne",
    "northwest": "nw",
    "southeast": "se",
    "southwest": "sw",
}

_STREET_SUFFIXES = {
    "street": "st",
    "avenue": "ave",
    "boulevard": "blvd",
    "place": "pl",
    "drive": "dr",
    "road": "rd",
    "lane": "ln",
    "court": "ct",
    "terrace": "ter",
    "circle": "cir",
    "square": "sq",
    "parkway": "pkwy",
    "highway": "hwy",
}

#: Matches a unit/suite/floor/building token plus its following value, e.g.
#: "Suite 200", "Apt 4B", "Fl 2", "Bldg A", "#200". Extracted, never deleted.
_UNIT_PATTERN = re.compile(
    r"\b(?:ste|suite|apt|unit|fl|floor|bldg|building|rm|room)\.?\s*[\w-]+\b|#\s*[\w-]+\b",
    re.IGNORECASE,
)

#: Tokens ignored only when picking *blocking* tokens for candidate generation.
#: Never used to alter a normalized name used for scoring or storage.
BLOCKING_STOPWORDS = frozenset(
    {"the", "a", "an", "restaurant", "cafe", "bar", "kitchen", "bistro", "house", "eatery", "room"}
)


def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def normalize_text(value: str) -> str:
    """Casefold, strip accents, canonicalize '&', drop punctuation, collapse whitespace."""
    text = strip_accents(value).casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"['’]", "", text)  # apostrophes are removed, not spaced out
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return _collapse_whitespace(text)


def normalize_name(value: str) -> str:
    """Mechanical restaurant-name normalization: text normalization plus removal
    of a short legal-entity-suffix list only (LLC, Inc, Corp, ...). Descriptive
    words such as "Restaurant", "Bar", or "Kitchen" are never removed -- they
    are meaningful and can be the only thing distinguishing two establishments.
    """
    text = normalize_text(value)
    tokens = [token for token in text.split() if token not in _BUSINESS_SUFFIXES]
    return _collapse_whitespace(" ".join(tokens))


def distinctive_name_tokens(normalized_name: str) -> frozenset[str]:
    """Tokens useful for *candidate generation* (blocking) only -- excludes
    generic stopwords like "the"/"restaurant"/"cafe" so that candidate lookup
    does not depend on the first word of a name. Never used for scoring."""
    tokens = [t for t in normalized_name.split() if t not in BLOCKING_STOPWORDS and len(t) > 1]
    return frozenset(tokens) if tokens else frozenset(normalized_name.split())


@dataclass(frozen=True)
class NormalizedAddress:
    """An address normalized for comparison, with the original preserved."""

    original: str
    normalized: str
    unit: str | None


def normalize_address(value: str | None) -> NormalizedAddress | None:
    """Normalize a street address: directional and street-suffix abbreviation,
    with any unit/suite/floor/building token extracted (not deleted) into
    ``unit`` so two suites at one street address stay distinguishable."""
    if not value or not value.strip():
        return None
    original = value.strip()
    unit_match = _UNIT_PATTERN.search(original)
    unit = _collapse_whitespace(unit_match.group(0)) if unit_match else None
    base = _UNIT_PATTERN.sub(" ", original)
    text = normalize_text(base)
    tokens = [_STREET_SUFFIXES.get(t, _DIRECTIONALS.get(t, t)) for t in text.split()]
    normalized = _collapse_whitespace(" ".join(tokens))
    return NormalizedAddress(original=original, normalized=normalized, unit=unit)


def street_number(normalized_address: str | None) -> str | None:
    """The leading street number of an already-normalized address, if any."""
    if not normalized_address:
        return None
    match = re.match(r"^(\d+)\b", normalized_address)
    return match.group(1) if match else None


def normalize_zip(value: str | None) -> tuple[str | None, str | None]:
    """Split a ZIP or ZIP+4 string into ``(postal5, postal4)``. Comparisons
    elsewhere use ``postal5`` only."""
    if not value:
        return None, None
    digits = re.sub(r"[^0-9]", "", value)
    if len(digits) >= 9:
        return digits[:5], digits[5:9]
    if len(digits) >= 5:
        return digits[:5], None
    return None, None


def extract_florida_numeric_suffix(value: str | None) -> str | None:
    """Trailing digits of a Florida license-number-like string.

    This is documented, weak, non-authoritative corroborating evidence only.
    Florida's inspection extract carries a plain numeric License Number (e.g.
    ``"2300027"``); Florida's separate license extract has been observed to
    format the same underlying identifier with a rank-code prefix (e.g.
    ``"SEA2300159"``) -- see ``plateproof/ingestion/florida.py``. This function
    makes NO claim that a numeric and a prefixed form with the same trailing
    digits refer to the same establishment; it is not used as an identity
    mechanism anywhere in this codebase.
    """
    if not value:
        return None
    match = re.search(r"(\d+)$", value.strip())
    return match.group(1) if match else None
