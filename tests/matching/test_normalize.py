"""Tests for plateproof.matching.normalize: deterministic, locally testable rules."""

from __future__ import annotations


def test_punctuation_differences_normalize_equal() -> None:
    from plateproof.matching.normalize import normalize_name

    assert normalize_name("Joe's Diner, Inc.") == normalize_name("Joes Diner Inc")


def test_accented_and_unaccented_names_normalize_equal() -> None:
    from plateproof.matching.normalize import normalize_name

    assert normalize_name("Café Boulud") == normalize_name("Cafe Boulud")


def test_ampersand_vs_and_normalize_equal() -> None:
    from plateproof.matching.normalize import normalize_name

    assert normalize_name("Smith & Wollensky") == normalize_name("Smith and Wollensky")


def test_business_suffix_removed() -> None:
    from plateproof.matching.normalize import normalize_name

    assert normalize_name("Example Bistro LLC") == normalize_name("Example Bistro")
    assert normalize_name("Example Corp") == normalize_name("Example")


def test_business_suffix_does_not_strip_meaningful_words() -> None:
    from plateproof.matching.normalize import normalize_name

    # "Restaurant"/"Bar"/"Kitchen" are meaningful, not legal-entity suffixes.
    assert normalize_name("Example Restaurant") != normalize_name("Example Bar")
    assert normalize_name("Example Kitchen") != normalize_name("Example")


def test_street_suffix_abbreviations_normalize_equal() -> None:
    from plateproof.matching.normalize import normalize_address

    a = normalize_address("123 Main Street")
    b = normalize_address("123 Main St")
    assert a is not None and b is not None
    assert a.normalized == b.normalized
    assert a.original == "123 Main Street"  # original preserved verbatim


def test_directional_abbreviations_normalize_equal() -> None:
    from plateproof.matching.normalize import normalize_address

    a = normalize_address("100 North Broadway")
    b = normalize_address("100 N Broadway")
    assert a is not None and b is not None
    assert a.normalized == b.normalized


def test_suite_unit_extracted_not_deleted() -> None:
    from plateproof.matching.normalize import normalize_address

    result = normalize_address("400 Park Ave Suite 200")
    assert result is not None
    assert result.unit is not None and "200" in result.unit
    assert "suite" not in result.normalized
    assert result.normalized.startswith("400 park ave")


def test_zip_plus4_splits_into_five_and_four() -> None:
    from plateproof.matching.normalize import normalize_zip

    postal5, postal4 = normalize_zip("10023-1234")
    assert postal5 == "10023"
    assert postal4 == "1234"


def test_zip_five_digit_only() -> None:
    from plateproof.matching.normalize import normalize_zip

    postal5, postal4 = normalize_zip("33602")
    assert postal5 == "33602"
    assert postal4 is None


def test_zip_blank_is_null() -> None:
    from plateproof.matching.normalize import normalize_zip

    assert normalize_zip(None) == (None, None)
    assert normalize_zip("") == (None, None)


def test_florida_license_numeric_suffix_extracted_but_not_equivalence_claimed() -> None:
    from plateproof.matching.normalize import extract_florida_numeric_suffix

    assert extract_florida_numeric_suffix("SEA2300159") == "2300159"
    assert extract_florida_numeric_suffix("2300027") == "2300027"
    assert extract_florida_numeric_suffix(None) is None
    # The function only extracts digits; it makes no claim the two forms above
    # refer to the same establishment (see plateproof.ingestion.florida docs).


def test_original_values_preserved_alongside_normalized() -> None:
    from plateproof.matching.normalize import normalize_address

    result = normalize_address("400 PARK AVE STE 200")
    assert result is not None
    assert result.original == "400 PARK AVE STE 200"
    assert result.normalized != result.original


def test_chef_or_hotel_name_tokens_not_stripped() -> None:
    from plateproof.matching.normalize import normalize_name

    # Two distinct restaurants inside/associated with the same hotel must remain
    # distinguishable -- normalization must not remove "hotel"/proper-noun tokens.
    a = normalize_name("The Grill at The Plaza Hotel")
    b = normalize_name("The Rose Club at The Plaza Hotel")
    assert a != b


def test_distinctive_name_tokens_ignore_stopwords() -> None:
    from plateproof.matching.normalize import distinctive_name_tokens, normalize_name

    tokens = distinctive_name_tokens(normalize_name("The Grill Room"))
    assert "the" not in tokens
    assert "room" not in tokens
    assert "grill" in tokens


def test_street_number_extraction() -> None:
    from plateproof.matching.normalize import normalize_address, street_number

    addr = normalize_address("400 Park Ave")
    assert addr is not None
    assert street_number(addr.normalized) == "400"
    assert street_number(None) is None


def test_normalize_address_none_and_blank_return_none() -> None:
    from plateproof.matching.normalize import normalize_address

    assert normalize_address(None) is None
    assert normalize_address("   ") is None
