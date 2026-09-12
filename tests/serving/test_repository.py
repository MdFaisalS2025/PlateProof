"""Tests for the DuckDB-over-Parquet repository: the single place all SQL
lives. Every fixture here is tiny and fictional; no production data.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest


def _settings(processed_dir: Path, **overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(_env_file=None, processed_data_dir=processed_dir, **overrides)


# --------------------------------------------------------------------------- #
# Health / datastore state                                                    #
# --------------------------------------------------------------------------- #


def test_health_unavailable_when_processed_dir_does_not_exist(tmp_path: Path) -> None:
    from plateproof.serving.repository import open_repository

    settings = _settings(tmp_path / "does_not_exist")
    repo = open_repository(settings)
    snapshot = repo.health_snapshot()
    assert snapshot["datastore"] == "empty"


def test_health_empty_when_processed_dir_exists_but_uninitialized(processed_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    snapshot = repo.health_snapshot()
    assert snapshot["datastore"] == "empty"
    assert snapshot["nyc_data"] == "unavailable"
    assert snapshot["florida_data"] == "unavailable"
    assert snapshot["michelin"] == "disabled"


def test_health_ok_when_restaurants_table_present(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row()])
    repo = open_repository(_settings(processed_dir))
    snapshot = repo.health_snapshot()
    assert snapshot["datastore"] == "ok"
    assert snapshot["nyc_data"] == "ok"
    assert snapshot["florida_data"] == "unavailable"


# --------------------------------------------------------------------------- #
# Path safety                                                                  #
# --------------------------------------------------------------------------- #


def test_processed_dir_with_space_and_apostrophe_works(tmp_path: Path, restaurant_row: Any) -> None:
    import polars as pl

    from plateproof.serving.repository import open_repository

    weird = tmp_path / "o'brien's data dir"
    weird.mkdir()
    pl.DataFrame([restaurant_row()]).write_parquet(weird / "restaurants.parquet")

    repo = open_repository(_settings(weird))
    result = repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=0)
    assert result.total == 1


def test_missing_configured_table_file_is_absent_not_fatal(processed_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=0)
    assert result.total == 0


def test_unsupported_file_content_is_treated_as_absent(processed_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    (processed_dir / "restaurants.parquet").write_text("not actually parquet", encoding="utf-8")
    repo = open_repository(_settings(processed_dir))
    snapshot = repo.health_snapshot()
    assert snapshot["nyc_data"] == "unavailable"


# --------------------------------------------------------------------------- #
# Search                                                                       #
# --------------------------------------------------------------------------- #


def test_search_by_name_is_accent_case_punctuation_tolerant(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [
            restaurant_row(
                restaurant_id="nyc:1", name="Café José's Diner", normalized_name="cafe joses diner"
            )
        ]
    )
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(query="cafe jose", jurisdiction=None, limit=10, offset=0)
    assert result.total == 1


def test_search_by_official_source_id_exact_lookup(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", source_id="1", normalized_name="one"),
            restaurant_row(restaurant_id="nyc:2", source_id="2", normalized_name="two"),
        ]
    )
    repo = open_repository(_settings(processed_dir))
    row = repo.get_restaurant("nyc:2")
    assert row is not None
    assert row["restaurant_id"] == "nyc:2"


def test_search_jurisdiction_filter(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc", normalized_name="a"),
            restaurant_row(
                restaurant_id="florida:1", jurisdiction="florida", normalized_name="b", region="FL"
            ),
        ]
    )
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(query="", jurisdiction="florida", limit=10, offset=0)
    assert result.total == 1
    assert result.results[0]["jurisdiction"] == "florida"


def test_search_unsupported_jurisdiction_raises(processed_dir: Path) -> None:
    from plateproof.serving.errors import InvalidQueryError
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    with pytest.raises(InvalidQueryError):
        repo.search_restaurants(query="", jurisdiction="texas", limit=10, offset=0)


def test_sql_injection_style_text_is_treated_as_literal(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [restaurant_row(restaurant_id="nyc:1", name="Bobby Tables", normalized_name="bobby tables")]
    )
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(
        query="'; DROP TABLE restaurants; --", jurisdiction=None, limit=10, offset=0
    )
    assert result.total == 0
    # table must still exist and be queryable afterward
    result2 = repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=0)
    assert result2.total == 1


# --------------------------------------------------------------------------- #
# Pagination                                                                   #
# --------------------------------------------------------------------------- #


def test_pagination_limit_below_one_raises(processed_dir: Path) -> None:
    from plateproof.serving.errors import InvalidPaginationError
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    with pytest.raises(InvalidPaginationError):
        repo.search_restaurants(query="", jurisdiction=None, limit=0, offset=0)


def test_pagination_limit_over_max_raises(processed_dir: Path) -> None:
    from plateproof.serving.errors import InvalidPaginationError
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir, max_page_size=50))
    with pytest.raises(InvalidPaginationError):
        repo.search_restaurants(query="", jurisdiction=None, limit=51, offset=0)


def test_pagination_negative_offset_raises(processed_dir: Path) -> None:
    from plateproof.serving.errors import InvalidPaginationError
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    with pytest.raises(InvalidPaginationError):
        repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=-1)


def test_pagination_deterministic_ordering_with_restaurant_id_tiebreak(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:2", normalized_name="same"),
            restaurant_row(restaurant_id="nyc:1", normalized_name="same"),
        ]
    )
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=0)
    assert [r["restaurant_id"] for r in result.results] == ["nyc:1", "nyc:2"]


# --------------------------------------------------------------------------- #
# Michelin filter transparency                                                #
# --------------------------------------------------------------------------- #


def test_michelin_filter_unavailable_returns_typed_transparent_result(
    processed_dir: Path, write_restaurants: Any, restaurant_row: Any
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row()])
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(
        query="", jurisdiction=None, michelin_category="one_star", limit=10, offset=0
    )
    assert result.optional_data_status["michelin"] == "unavailable"
    assert result.filters_applied["michelin_category"] is False
    assert any("michelin" in w.lower() for w in result.warnings)
    # never silently returns unfiltered restaurants as though the filter applied
    assert result.total == 1  # still returned, but flagged as filter not applied


def test_michelin_accept_only_filter(
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
    write_michelin: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", normalized_name="accepted place"),
            restaurant_row(restaurant_id="nyc:2", normalized_name="reviewed place"),
        ]
    )
    write_michelin(
        restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Accepted Place",
                "normalized_name": "accepted place",
            }
        ],
        distinctions=[
            {
                "michelin_distinction_event_id": "michelin:e:aaa",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "MICHELIN Guide NYC",
                "guide_year": 2025,
                "announced_date": date(2025, 1, 1),
                "source_url": "https://example.org/guide",
                "source_publisher": "Example Publisher",
            }
        ],
        matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
            },
            {
                "official_restaurant_id": "nyc:2",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "review",
            },
        ],
    )
    repo = open_repository(_settings(processed_dir))
    result = repo.search_restaurants(
        query="", jurisdiction=None, michelin_category="one_star", limit=10, offset=0
    )
    assert result.optional_data_status["michelin"] == "ok"
    assert [r["restaurant_id"] for r in result.results] == ["nyc:1"]

    history = repo.michelin_history("nyc:1")
    assert len(history) == 1
    assert history[0]["guide_year"] == 2025

    assert repo.michelin_history("nyc:2") == []  # review decision never public


def test_michelin_guide_year_filter_uses_explicit_distinction_event(
    processed_dir: Path,
    write_restaurants: Any,
    restaurant_row: Any,
    write_michelin: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row(restaurant_id="nyc:1", normalized_name="a")])
    write_michelin(
        restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "A",
                "normalized_name": "a",
            }
        ],
        distinctions=[
            {
                "michelin_distinction_event_id": "michelin:e:2024",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "MICHELIN Guide NYC",
                "guide_year": 2024,
                "announced_date": date(2024, 1, 1),
                "source_url": "https://example.org/2024",
                "source_publisher": "Example Publisher",
            },
        ],
        matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
            }
        ],
    )
    repo = open_repository(_settings(processed_dir))
    result_2024 = repo.search_restaurants(
        query="",
        jurisdiction=None,
        michelin_category="one_star",
        michelin_guide_year=2024,
        limit=10,
        offset=0,
    )
    assert result_2024.total == 1

    result_2023 = repo.search_restaurants(
        query="",
        jurisdiction=None,
        michelin_category="one_star",
        michelin_guide_year=2023,
        limit=10,
        offset=0,
    )
    assert result_2023.total == 0  # no distinction event recorded for 2023 -- never inferred


# --------------------------------------------------------------------------- #
# Restaurant detail / inspections / violations                                #
# --------------------------------------------------------------------------- #


def test_restaurant_not_found_returns_none(processed_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir))
    assert repo.get_restaurant("nyc:999") is None


def test_nyc_inspection_native_fields(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row()])
    write_inspections([inspection_row()])
    repo = open_repository(_settings(processed_dir))
    inspections = repo.list_inspections("nyc:1")
    assert inspections[0]["score"] == 13.0
    assert inspections[0]["grade"] == "A"


def test_florida_inspection_native_fields_and_no_letter_grade(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants(
        [restaurant_row(restaurant_id="florida:1", jurisdiction="florida", region="FL")]
    )
    write_inspections(
        [
            inspection_row(
                inspection_id="florida:v1",
                restaurant_id="florida:1",
                jurisdiction="florida",
                score=None,
                grade=None,
                critical_violation_count=None,
                high_priority_count=1,
                intermediate_count=2,
                basic_count=0,
                total_violation_count=3,
                disposition_status="met_standards",
            )
        ]
    )
    repo = open_repository(_settings(processed_dir))
    inspections = repo.list_inspections("florida:1")
    assert inspections[0]["high_priority_count"] == 1
    assert inspections[0]["grade"] is None


def test_inspection_ordering_newest_first(
    processed_dir: Path,
    write_restaurants: Any,
    write_inspections: Any,
    restaurant_row: Any,
    inspection_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row()])
    write_inspections(
        [
            inspection_row(inspection_id="i1", inspection_date=date(2024, 1, 1)),
            inspection_row(inspection_id="i2", inspection_date=date(2025, 1, 1)),
        ]
    )
    repo = open_repository(_settings(processed_dir))
    inspections = repo.list_inspections("nyc:1")
    assert [i["inspection_id"] for i in inspections] == ["i2", "i1"]


def test_violation_ordering_newest_first(
    processed_dir: Path,
    write_restaurants: Any,
    write_violations: Any,
    restaurant_row: Any,
    violation_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row()])
    write_violations(
        [
            violation_row(violation_event_id="v1", inspection_date=date(2024, 1, 1)),
            violation_row(violation_event_id="v2", inspection_date=date(2025, 1, 1)),
        ]
    )
    repo = open_repository(_settings(processed_dir))
    violations = repo.list_violations("nyc:1")
    assert [v["violation_event_id"] for v in violations] == ["v2", "v1"]


def test_florida_category_classification_unavailable_wording(
    processed_dir: Path,
    write_restaurants: Any,
    write_violations: Any,
    restaurant_row: Any,
    violation_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row(restaurant_id="florida:1", jurisdiction="florida")])
    write_violations(
        [
            violation_row(
                violation_event_id="v1",
                restaurant_id="florida:1",
                violation_code="12",
                severity="other",
            )
        ]
    )
    repo = open_repository(_settings(processed_dir))
    violations = repo.list_violations("florida:1")
    assert violations[0]["severity"] == "classification_unavailable_at_category_level"


def test_generic_other_violation_from_non_florida_keeps_plain_wording(
    processed_dir: Path,
    write_restaurants: Any,
    write_violations: Any,
    restaurant_row: Any,
    violation_row: Any,
) -> None:
    from plateproof.serving.repository import open_repository

    write_restaurants([restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc")])
    write_violations(
        [violation_row(violation_event_id="v1", restaurant_id="nyc:1", severity="other")]
    )
    repo = open_repository(_settings(processed_dir))
    violations = repo.list_violations("nyc:1")
    assert violations[0]["severity"] == "other"


def test_containment_check_tolerates_trailing_whitespace_in_configured_dir(
    tmp_path: Path, restaurant_row: Any
) -> None:
    """A configured processed-data directory with incidental trailing
    whitespace (e.g. from an env var assembled by a shell script) must not
    cause an existing table to resolve consistently while a not-yet-written
    one spuriously "escapes" the same base directory."""
    import polars as pl

    from plateproof.serving.repository import open_repository

    real_dir = tmp_path / "processed"
    real_dir.mkdir()
    pl.DataFrame([restaurant_row()]).write_parquet(real_dir / "restaurants.parquet")

    padded_dir = Path(str(real_dir) + " ")  # trailing space, same real directory on Windows
    repo = open_repository(_settings(padded_dir))
    result = repo.search_restaurants(query="", jurisdiction=None, limit=10, offset=0)
    assert result.total == 1
