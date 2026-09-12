"""Tests for evaluation metrics: AP/Brier, undefined-ROC handling, top-decile recall, subgroups."""

from __future__ import annotations

import numpy as np


def test_ap_and_brier_computed() -> None:
    from plateproof.models.training import compute_metrics

    y = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([0.1, 0.8, 0.2, 0.9, 0.6, 0.3, 0.7, 0.1, 0.4, 0.85])
    metrics = compute_metrics(y, p, threshold=0.5, partition="validation")
    assert metrics.average_precision is not None
    assert metrics.brier_score is not None
    assert 0.0 <= metrics.brier_score <= 1.0


def test_single_class_partition_returns_unavailable_with_reason() -> None:
    from plateproof.models.training import compute_metrics

    y = np.zeros(20, dtype=int)
    p = np.random.RandomState(0).rand(20)
    metrics = compute_metrics(y, p, threshold=0.5, partition="test")
    assert metrics.average_precision is None
    assert metrics.average_precision_unavailable_reason is not None
    assert metrics.roc_auc is None
    assert metrics.roc_auc_unavailable_reason is not None
    assert metrics.brier_score is not None  # brier is always defined


def test_top_decile_recall_deterministic_tie_break() -> None:
    from plateproof.models.training import compute_metrics

    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.full(10, 0.5)  # all tied -> tie-break must be deterministic, not order-dependent
    a = compute_metrics(y, p, threshold=0.5, partition="validation")
    b = compute_metrics(y, p, threshold=0.5, partition="validation")
    assert a.top_decile_recall == b.top_decile_recall
    assert a.top_decile_row_count == b.top_decile_row_count


def test_top_decile_recall_unstable_for_tiny_partition_returns_none() -> None:
    from plateproof.models.training import compute_metrics

    y = np.array([1, 0, 1])
    p = np.array([0.9, 0.1, 0.8])
    metrics = compute_metrics(y, p, threshold=0.5, partition="validation")
    assert metrics.top_decile_recall is None


def test_subgroup_report_suppresses_small_groups() -> None:
    from plateproof.models.training import subgroup_report

    rng = np.random.RandomState(0)
    n = 100
    y = (rng.rand(n) > 0.7).astype(int)
    p = rng.rand(n)
    groups = np.array(["Manhattan"] * 40 + ["Brooklyn"] * 55 + ["Rare"] * 5)
    report = subgroup_report(y, p, groups, group_name="borough")
    entries = {entry["group_value"]: entry for entry in report}
    assert entries["Rare"]["suppressed"] is True
    assert entries["Rare"]["suppression_reason"] is not None
    assert entries["Manhattan"]["suppressed"] is False
