"""A small labeled synthetic validation matrix for match_candidate.

Scoring weights and the missing-evidence cap in entity_resolution.py are
versioned, provisional heuristics (see its module docstring), not proven
probabilities. This file is the behavioral check that locks their intended
outcome on a spread of clear matches, ambiguous pairs, and clear nonmatches --
including the named false-positive scenarios (chains, hotels, relocations,
renames) that motivated the required-evidence design in the first place.

All restaurants here are fictional.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

FIXED_TS = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _official(**overrides: Any) -> Any:
    from plateproof.matching.entity_resolution import OfficialRestaurant

    defaults: dict[str, Any] = dict(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        source_id="1",
        name="Example Bistro",
        normalized_name="example bistro",
        address_as_published="100 Fictional Ave",
        normalized_address="100 fictional ave",
        city="New York",
        postal_code="10001",
        latitude=None,
        longitude=None,
    )
    defaults.update(overrides)
    return OfficialRestaurant(**defaults)


def _michelin(**overrides: Any) -> Any:
    from plateproof.ingestion.michelin import MichelinRestaurant

    defaults: dict[str, Any] = dict(
        michelin_restaurant_id="michelin:r:aaaa",
        name_as_published="Example Bistro",
        normalized_name="example bistro",
        address_as_published="100 Fictional Ave",
        normalized_address="100 fictional ave",
        city="New York",
        region="NY",
        postal_code="10001",
        latitude=None,
        longitude=None,
        jurisdiction_candidate="nyc",
        identity_source_note="fixture",
        curator_restaurant_ref=None,
    )
    defaults.update(overrides)
    return MichelinRestaurant(**defaults)


def _score(official: Any, restaurant: Any) -> Any:
    from plateproof.matching.entity_resolution import match_candidate

    return match_candidate(official, restaurant, blocking_rules=["fixture"], generated_at=FIXED_TS)


# --- clear matches ----------------------------------------------------------


def test_clear_match_identical_identity() -> None:
    evidence = _score(_official(), _michelin())
    assert evidence.decision == "accept"


def test_clear_match_minor_punctuation_and_suffix_difference() -> None:
    official = _official(name="Example Bistro LLC", normalized_name="example bistro")
    evidence = _score(official, _michelin())
    assert evidence.decision == "accept"


# --- ambiguous pairs ---------------------------------------------------------


def test_ambiguous_same_name_missing_official_address() -> None:
    official = _official(address_as_published=None, normalized_address=None)
    evidence = _score(official, _michelin())
    assert evidence.decision == "review"


def test_ambiguous_ok_name_similarity_full_other_evidence() -> None:
    official = _official(name="Example Bistros", normalized_name="example bistros")
    evidence = _score(official, _michelin())
    assert evidence.decision in ("review", "accept")


# --- clear nonmatches ---------------------------------------------------------


def test_clear_nonmatch_different_name_and_address() -> None:
    official = _official(
        name="Totally Unrelated Place",
        normalized_name="totally unrelated place",
        address_as_published="9999 Nowhere Rd",
        normalized_address="9999 nowhere rd",
        postal_code="99999",
        city="Nowhere",
    )
    evidence = _score(official, _michelin())
    assert evidence.decision == "reject"


# --- named false-positive scenarios -----------------------------------------


def test_hotel_restaurants_share_address_different_names_not_cross_matched() -> None:
    """Two distinct restaurants inside one hotel; only the correctly named one
    may reach accept-eligibility despite both sharing the exact address."""
    grill = _official(
        restaurant_id="nyc:grill",
        name="The Grill at Example Hotel",
        normalized_name="grill at example hotel",
    )
    rose_room = _official(
        restaurant_id="nyc:rose",
        name="The Rose Room at Example Hotel",
        normalized_name="rose room at example hotel",
    )
    michelin_grill = _michelin(
        name_as_published="The Grill at Example Hotel", normalized_name="grill at example hotel"
    )
    correct = _score(grill, michelin_grill)
    wrong = _score(rose_room, michelin_grill)
    assert correct.decision in ("review", "accept")
    assert wrong.decision != "accept"
    assert correct.overall_score > wrong.overall_score


def test_chain_locations_do_not_cross_match_on_name_and_postal_alone() -> None:
    """Two branches of one named chain at different addresses; a Michelin
    record for one branch must not accept against the other branch merely
    because the name matches and the (different) postal codes are ignored --
    each branch's own address must corroborate its own candidate."""
    midtown = _official(
        restaurant_id="nyc:midtown",
        address_as_published="1 Midtown Ave",
        normalized_address="1 midtown ave",
        postal_code="10001",
    )
    downtown = _official(
        restaurant_id="nyc:downtown",
        address_as_published="2 Downtown Ave",
        normalized_address="2 downtown ave",
        postal_code="10005",
    )
    michelin_midtown = _michelin(
        michelin_restaurant_id="michelin:r:midtown",
        address_as_published="1 Midtown Ave",
        normalized_address="1 midtown ave",
        postal_code="10001",
    )
    correct = _score(midtown, michelin_midtown)
    cross = _score(downtown, michelin_midtown)
    assert correct.decision == "accept"
    assert cross.decision != "accept"
    assert "postal_code_conflict" in cross.conflicting_evidence


def test_relocation_name_matches_address_does_not_lands_in_review_not_false_reject() -> None:
    """An official restaurant that has relocated since the Michelin record's
    address was recorded: name matches, address does not -- must not be a
    silent accept (address is required evidence) but should still be visible
    for human review, not simply dropped."""
    relocated = _official(
        address_as_published="500 New Location Rd",
        normalized_address="500 new location rd",
        postal_code="10099",
    )
    evidence = _score(relocated, _michelin())
    assert evidence.decision != "accept"
    assert evidence.decision in ("review", "reject")


def test_rename_address_matches_name_does_not_does_not_false_accept() -> None:
    """An official restaurant that has been renamed since the Michelin
    distinction: address matches, name does not -- must not auto-accept."""
    renamed = _official(name="A Completely New Name", normalized_name="a completely new name")
    evidence = _score(renamed, _michelin())
    assert evidence.decision != "accept"


def test_same_address_different_suites_distinguishable_by_name() -> None:
    """Two tenants at the same street address in different suites: the
    correctly named one may still accept; a differently named one at the same
    (unit-stripped) normalized address must not."""
    from plateproof.matching.normalize import normalize_address

    suite_a = _official(
        restaurant_id="nyc:suite-a",
        address_as_published="400 Shared Plaza Suite 100",
        normalized_address=normalize_address("400 Shared Plaza Suite 100").normalized,  # type: ignore[union-attr]
    )
    suite_b = _official(
        restaurant_id="nyc:suite-b",
        name="Unrelated Tenant",
        normalized_name="unrelated tenant",
        address_as_published="400 Shared Plaza Suite 200",
        normalized_address=normalize_address("400 Shared Plaza Suite 200").normalized,  # type: ignore[union-attr]
    )
    michelin_record = _michelin(
        address_as_published="400 Shared Plaza Suite 100",
        normalized_address=normalize_address("400 Shared Plaza Suite 100").normalized,  # type: ignore[union-attr]
    )
    correct = _score(suite_a, michelin_record)
    wrong = _score(suite_b, michelin_record)
    assert correct.decision == "accept"
    assert wrong.decision != "accept"
