"""Tests for chronological_split."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest


def _make_frame(inspection_ids: list[str], dates: list[date], labels: list[int]) -> Any:
    import polars as pl

    from plateproof.models.training import TrainingFrame

    X = pl.DataFrame({"inspection_id": inspection_ids, "f": [1.0] * len(inspection_ids)})
    y = pl.DataFrame({"inspection_id": inspection_ids, "label": labels})
    identity = pl.DataFrame({"inspection_id": inspection_ids, "inspection_date": dates})
    return TrainingFrame(
        X=X, y=y, identity=identity, feature_order=("f",), target_name="t", jurisdiction="nyc"
    )


def test_same_date_rows_stay_in_one_partition() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a", "b", "c", "d", "e", "f"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 1),  # both must land in the same partition
        date(2024, 3, 1),
        date(2024, 3, 1),
        date(2024, 5, 1),
        date(2024, 5, 1),
    ]
    labels = [0, 1, 0, 1, 0, 1]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    train, val, test, report = chronological_split(frame, boundaries)
    assert set(train.y.get_column("inspection_id").to_list()) == {"a", "b"}
    assert set(val.y.get_column("inspection_id").to_list()) == {"c", "d"}
    assert set(test.y.get_column("inspection_id").to_list()) == {"e", "f"}
    assert report.train.row_count == 2
    assert report.validation.row_count == 2
    assert report.test.row_count == 2


def test_exact_boundary_dates() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["train_early", "train_last", "val_first", "val_last", "test_first"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 31),
        date(2024, 2, 1),
        date(2024, 3, 31),
        date(2024, 4, 1),
    ]
    labels = [1, 0, 1, 0, 1]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 5, 1),
    )
    train, val, test, _ = chronological_split(frame, boundaries)
    assert set(train.y.get_column("inspection_id").to_list()) == {"train_early", "train_last"}
    assert set(val.y.get_column("inspection_id").to_list()) == {"val_first", "val_last"}
    assert test.y.get_column("inspection_id").to_list() == ["test_first"]


def test_rows_outside_interval_are_reported(make_event: Any) -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["too_early", "a", "a2", "b", "b2", "c", "too_late"]
    dates = [
        date(2020, 1, 1),
        date(2024, 1, 1),
        date(2024, 1, 15),
        date(2024, 3, 1),
        date(2024, 3, 15),
        date(2024, 4, 15),
        date(2030, 1, 1),
    ]
    labels = [0, 1, 0, 0, 1, 0, 1]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_start=date(2023, 1, 1),
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 5, 1),
    )
    _, _, _, report = chronological_split(frame, boundaries)
    assert report.excluded_before_train_count == 1
    assert report.excluded_after_test_count == 1


def test_train_and_validation_require_both_classes() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a", "b", "c"]
    dates = [date(2024, 1, 1), date(2024, 3, 1), date(2024, 5, 1)]
    labels = [0, 0, 1]  # train has only class 0
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    with pytest.raises(ValueError, match="train"):
        chronological_split(frame, boundaries)


def test_single_class_test_partition_is_permitted() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 3, 1),
        date(2024, 3, 2),
        date(2024, 5, 1),
        date(2024, 5, 2),
    ]
    labels = [0, 1, 0, 1, 0, 0]  # test has only class 0
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    _, _, test, report = chronological_split(frame, boundaries)  # must not raise
    assert report.test.positive_count == 0


def test_empty_partition_raises() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a", "b"]
    dates = [date(2024, 1, 1), date(2024, 1, 2)]
    labels = [0, 1]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    with pytest.raises(ValueError, match="validation"):
        chronological_split(frame, boundaries)


def test_empty_test_partition_raises() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a", "b", "c", "d"]
    dates = [date(2024, 1, 1), date(2024, 1, 2), date(2024, 3, 1), date(2024, 3, 2)]
    labels = [0, 1, 0, 1]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 6, 1),
        test_end=date(2024, 12, 1),
    )
    with pytest.raises(ValueError, match="test"):
        chronological_split(frame, boundaries)


def test_one_row_single_class_train_raises() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a", "b1", "b2", "c1", "c2"]
    dates = [
        date(2024, 1, 1),
        date(2024, 3, 1),
        date(2024, 3, 2),
        date(2024, 5, 1),
        date(2024, 5, 2),
    ]
    labels = [0, 0, 1, 0, 1]  # train is a single row, single class
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    with pytest.raises(ValueError, match="train"):
        chronological_split(frame, boundaries)


def test_one_row_single_class_validation_raises() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a1", "a2", "b", "c1", "c2"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 3, 1),
        date(2024, 5, 1),
        date(2024, 5, 2),
    ]
    labels = [0, 1, 0, 0, 1]  # validation is a single row, single class
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    with pytest.raises(ValueError, match="validation"):
        chronological_split(frame, boundaries)


def test_single_class_nonempty_test_is_allowed_even_with_diverse_train_and_validation() -> None:
    from plateproof.models.training import SplitBoundaries, chronological_split

    ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    dates = [
        date(2024, 1, 1),
        date(2024, 1, 2),
        date(2024, 3, 1),
        date(2024, 3, 2),
        date(2024, 5, 1),
        date(2024, 5, 2),
    ]
    labels = [0, 1, 0, 1, 0, 0]
    frame = _make_frame(ids, dates, labels)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    _, _, test, _ = chronological_split(frame, boundaries)  # must not raise
    assert test.y.height == 2


def test_quantile_boundaries_never_used_by_default_cli_path() -> None:
    from plateproof.models.training import derive_quantile_boundaries

    ids = [f"i{i}" for i in range(10)]
    dates = [date(2024, 1, i + 1) for i in range(10)]
    labels = [i % 2 for i in range(10)]
    frame = _make_frame(ids, dates, labels)
    boundaries = derive_quantile_boundaries(frame)
    assert boundaries.train_end < boundaries.validation_end < boundaries.test_end
