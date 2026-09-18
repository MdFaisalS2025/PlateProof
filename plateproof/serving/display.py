"""Shared, jurisdiction-aware presentation rules used by both the FastAPI
routes and the Streamlit pages -- the single place NYC/Florida display
semantics and official-source-link construction live, so neither surface
reimplements them independently.
"""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

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

#: Task 9B: shown before an owner ever uploads a document, and repeated on
#: every extraction result and the downloadable JSON artifact. A machine
#: extraction (or a user's own correction to it) is never an official
#: inspection record, and it never feeds PlateProof's risk models.
DOCUMENT_PRIVACY_NOTICE = (
    "PlateProof does not verify restaurant ownership, and this tool does not create an "
    "official inspection record. Uploaded documents are processed only to help you review "
    "your own extracted data; PlateProof does not persist your upload, and no uploaded "
    "content is ever used to train or update PlateProof's prediction models."
)

#: Task 10: shown directly beside a constructed Google Maps search link.
#: Deliberately states three things in one place: (1) the link is a
#: *search query* PlateProof built from its own already-known data, not a
#: Google-verified match -- the user must confirm the result themselves;
#: (2) PlateProof's server never contacts Google -- opening the link sends
#: the query to Google from the user's OWN browser, a client-side action
#: PlateProof does not perform on anyone's behalf; (3) PlateProof remains
#: independent of and not affiliated with Google, matching the same
#: non-affiliation framing already used for Michelin/NYC/Florida elsewhere
#: in this module.
GOOGLE_SEARCH_LINK_ATTRIBUTION = (
    "This is a Google Maps search built from PlateProof's own restaurant data, not a "
    "Google-verified match -- please confirm it's the right result yourself. Opening it "
    "sends the search from your browser to Google; PlateProof's server never contacts "
    "Google. PlateProof is independent and not affiliated with or endorsed by Google."
)

#: Google's own documented ceiling for Maps URLs: "URLs are limited to
#: 2,048 characters for each request." A query that would exceed this is
#: refused outright (returns None) rather than truncated -- a truncated
#: name/address could silently become a different, misleading query.
GOOGLE_MAPS_URL_MAX_LENGTH = 2048

_GOOGLE_MAPS_SEARCH_BASE = "https://www.google.com/maps/search/?api=1&query="


def google_maps_search_link(
    *, name: str, address: str | None, city: str | None, region: str | None
) -> str | None:
    """A plain Google Maps URL (``developers.google.com/maps/documentation/urls``)
    built entirely from data PlateProof already has -- no API key, no
    billing account, no network call PlateProof itself makes. Returns
    ``None`` for a blank/whitespace-only name (never guesses a fallback
    query) and for a query that would exceed Google's documented 2,048-
    character URL ceiling (refused, never truncated -- see
    :data:`GOOGLE_MAPS_URL_MAX_LENGTH`)."""
    if not name or not name.strip():
        return None
    parts = [name.strip()]
    for part in (address, city, region):
        if part and part.strip():
            parts.append(part.strip())
    query = ", ".join(parts)
    url = _GOOGLE_MAPS_SEARCH_BASE + quote(query, safe="")
    if len(url) > GOOGLE_MAPS_URL_MAX_LENGTH:
        return None
    return url


USER_SUBMITTED_RECORD_DISCLAIMER = (
    "This is a user-submitted record produced from a machine-assisted extraction of an "
    "uploaded document, optionally corrected by the uploader. It is not an official "
    "inspection record from any health department, and it does not affect PlateProof's "
    "risk predictions for this restaurant."
)

# Fixed, allowlisted official government dataset landing pages -- never built
# from a caller- or table-supplied URL fragment.
_NYC_SOURCE_URL = "https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j"
_FLORIDA_SOURCE_URL = "https://www2.myfloridalicense.com/hotels-restaurants/public-records/"


#: Task 11: the spec's Privacy-and-safety section requires "a correction/
#: contact path" alongside the official-source link. PlateProof mirrors
#: public government records it does not control and operates no support
#: inbox of its own -- the honest, implementable correction path is the
#: issuing agency's own record, not an invented PlateProof-run intake.
DATA_CORRECTION_NOTE = (
    "To correct or dispute this record, contact the issuing government agency directly "
    "-- use the official source link above. PlateProof mirrors public records and "
    "cannot change them."
)


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
