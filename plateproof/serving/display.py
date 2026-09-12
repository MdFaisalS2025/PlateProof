"""Shared, jurisdiction-aware presentation rules used by both the FastAPI
routes and the Streamlit pages -- the single place NYC/Florida display
semantics and official-source-link construction live, so neither surface
reimplements them independently.
"""

from __future__ import annotations

from typing import Any, Literal

Jurisdiction = Literal["nyc", "florida"]

MICHELIN_CONTEXT_NOTE = (
    "Michelin recognition is culinary context, not evidence of food safety, and PlateProof "
    "is independent -- not affiliated with or endorsed by Michelin."
)

PREDICTION_DISCLAIMER = "This is a PlateProof estimate, not an official inspection result."

INDEPENDENCE_STATEMENT = (
    "PlateProof is an independent project and is not endorsed by or affiliated with "
    "Michelin, Google, New York City, the Florida Department of Business and Professional "
    "Regulation, or any health department."
)

# Fixed, allowlisted official government dataset landing pages -- never built
# from a caller- or table-supplied URL fragment.
_NYC_SOURCE_URL = "https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j"
_FLORIDA_SOURCE_URL = "https://www2.myfloridalicense.com/hotels-restaurants/public-records/"


def official_source_link(jurisdiction: Jurisdiction) -> str:
    """The fixed official dataset landing page for a jurisdiction. Never
    constructed from a database row's own value -- both URLs are hardcoded
    constants pointing at the two datasets Task 2/3 ingest from."""
    if jurisdiction == "nyc":
        return _NYC_SOURCE_URL
    if jurisdiction == "florida":
        return _FLORIDA_SOURCE_URL
    raise ValueError(f"unknown jurisdiction: {jurisdiction!r}")


def is_safe_external_link(url: str) -> bool:
    """True only for an ``http``/``https`` URL. Used to validate a
    curator-supplied Michelin ``source_url`` before it is ever rendered --
    never renders raw HTML, and never a ``file://`` or other local scheme."""
    return url.startswith("https://") or url.startswith("http://")


def latest_documented_distinctions(history: list[dict[str, Any]]) -> tuple[list[str], int | None]:
    """From a restaurant's full Michelin distinction history (already
    accept-only), return the distinction label(s) from its latest
    *documented* guide edition only -- never claimed to be "current". Empty
    history returns ``([], None)``.
    """
    if not history:
        return [], None
    latest_year = max(item["guide_year"] for item in history)
    labels = sorted({item["distinction"] for item in history if item["guide_year"] == latest_year})
    return labels, latest_year
