"""RED-first tests for Task 10's link-only Google Maps feature
(``plateproof.serving.display.google_maps_search_link``).

This is a constructed *search* URL, not a Google-verified restaurant match --
Google may resolve it to a different or no result at all. The function makes
no network request of any kind; PlateProof's server never contacts Google.
Opening the returned link is a client-side action the user's own browser
performs, not something PlateProof does on their behalf.
"""

from __future__ import annotations

import socket

import pytest


def test_link_includes_encoded_name_and_address() -> None:
    from plateproof.serving.display import google_maps_search_link

    link = google_maps_search_link(
        name="Joe's Pizza", address="123 Main St", city="New York", region="NY"
    )
    assert link is not None
    assert link.startswith("https://www.google.com/maps/search/?api=1&query=")
    assert "Joe" in link
    assert "%27" in link or "27" in link  # apostrophe is percent-encoded, never raw
    assert " " not in link  # spaces must be encoded, never left raw in a URL


def test_link_url_encodes_special_characters() -> None:
    from plateproof.serving.display import google_maps_search_link

    link = google_maps_search_link(
        name="Tom & Jerry's Café", address="1 <Test> Ave", city=None, region=None
    )
    assert link is not None
    assert "&" not in link.split("query=", 1)[1].replace("%26", "")  # raw '&' never in the query
    assert "<" not in link
    assert ">" not in link


def test_blank_name_returns_none() -> None:
    from plateproof.serving.display import google_maps_search_link

    assert google_maps_search_link(name="", address="1 Main St", city="NY", region="NY") is None
    assert google_maps_search_link(name="   ", address=None, city=None, region=None) is None


def test_missing_optional_fields_degrade_gracefully() -> None:
    from plateproof.serving.display import google_maps_search_link

    link = google_maps_search_link(name="Joe's Diner", address=None, city=None, region=None)
    assert link is not None
    assert "None" not in link


def test_link_refused_when_it_would_exceed_googles_documented_url_length_limit() -> None:
    from plateproof.serving.display import GOOGLE_MAPS_URL_MAX_LENGTH, google_maps_search_link

    absurdly_long_name = "A" * 4000
    link = google_maps_search_link(
        name=absurdly_long_name, address="1 Main St", city="NY", region="NY"
    )
    assert link is None
    # A short, ordinary name/address never trips the ceiling.
    ordinary = google_maps_search_link(
        name="Joe's Pizza", address="123 Main St", city="New York", region="NY"
    )
    assert ordinary is not None
    assert len(ordinary) <= GOOGLE_MAPS_URL_MAX_LENGTH


def test_google_maps_url_max_length_matches_googles_documented_limit() -> None:
    from plateproof.serving.display import GOOGLE_MAPS_URL_MAX_LENGTH

    # Google's own Maps URLs documentation: "URLs are limited to 2,048
    # characters for each request."
    assert GOOGLE_MAPS_URL_MAX_LENGTH == 2048


def test_attribution_states_the_link_is_not_a_verified_match() -> None:
    from plateproof.serving.display import GOOGLE_SEARCH_LINK_ATTRIBUTION

    lowered = GOOGLE_SEARCH_LINK_ATTRIBUTION.lower()
    assert "not" in lowered and "verif" in lowered
    assert "confirm" in lowered


def test_attribution_discloses_the_browser_sends_the_query_to_google() -> None:
    from plateproof.serving.display import GOOGLE_SEARCH_LINK_ATTRIBUTION

    lowered = GOOGLE_SEARCH_LINK_ATTRIBUTION.lower()
    assert "browser" in lowered
    assert "google" in lowered


def test_attribution_states_independence() -> None:
    from plateproof.serving.display import GOOGLE_SEARCH_LINK_ATTRIBUTION

    lowered = GOOGLE_SEARCH_LINK_ATTRIBUTION.lower()
    assert "not affiliated" in lowered or "independent" in lowered


def test_no_network_call_is_ever_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hard tripwire, not just an absence-of-import check: if this
    function ever grows a network call, the test fails immediately rather
    than relying on someone noticing an added import."""
    from plateproof.serving.display import google_maps_search_link

    def _tripwire(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("google_maps_search_link attempted a socket connection")

    monkeypatch.setattr(socket.socket, "connect", _tripwire)
    link = google_maps_search_link(
        name="Joe's Pizza", address="123 Main St", city="New York", region="NY"
    )
    assert link is not None
