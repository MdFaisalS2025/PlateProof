"""Task 6 correction pass: bootstrap alignment, weighting fidelity, member
metadata persistence, readiness, complete artifact bundle, model cards, and
recoverable overwrite. No production data or network access is used here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Item 3: bootstrap row-to-restaurant alignment                              #
# --------------------------------------------------------------------------- #


def test_aligned_restaurant_ids_follows_given_id_order_not_identity_order() -> None:
    """A shuffled identity frame must not corrupt the restaurant-id mapping."""
    import polars as pl

    from plateproof.models.training import aligned_restaurant_ids

    inspection_ids = ["a", "b", "c", "d"]
    # identity intentionally NOT in the same order as inspection_ids
    identity = pl.DataFrame(
        {
            "inspection_id": ["d", "b", "a", "c"],
            "restaurant_id": ["r_d", "r_b", "r_a", "r_c"],
        }
    )
    result = aligned_restaurant_ids(inspection_ids, identity)
    assert list(result) == ["r_a", "r_b", "r_c", "r_d"]


def test_aligned_restaurant_ids_raises_on_missing_inspection_id() -> None:
    import polars as pl

    from plateproof.models.training import aligned_restaurant_ids

    identity = pl.DataFrame({"inspection_id": ["a", "b"], "restaurant_id": ["r_a", "r_b"]})
    with pytest.raises(ValueError, match="missing"):
        aligned_restaurant_ids(["a", "b", "c"], identity)


def test_aligned_restaurant_ids_raises_on_duplicate_identity_rows() -> None:
    import polars as pl

    from plateproof.models.training import aligned_restaurant_ids

    identity = pl.DataFrame(
        {"inspection_id": ["a", "a", "b"], "restaurant_id": ["r_a", "r_a2", "r_b"]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        aligned_restaurant_ids(["a", "b"], identity)


def test_assert_bootstrap_inputs_aligned_detects_length_mismatch() -> None:
    from plateproof.models.training import assert_bootstrap_inputs_aligned

    X = np.zeros((3, 2))
    y = np.zeros(3)
    ids = ["a", "b", "c"]
    restaurant_ids = np.array(["r1", "r2"])  # wrong length
    with pytest.raises(ValueError, match="misaligned"):
        assert_bootstrap_inputs_aligned(X, y, ids, restaurant_ids)


def test_assert_bootstrap_inputs_aligned_detects_duplicate_inspection_id() -> None:
    from plateproof.models.training import assert_bootstrap_inputs_aligned

    X = np.zeros((3, 2))
    y = np.zeros(3)
    ids = ["a", "a", "b"]
    restaurant_ids = np.array(["r1", "r1", "r2"])
    with pytest.raises(ValueError, match="duplicate"):
        assert_bootstrap_inputs_aligned(X, y, ids, restaurant_ids)


def test_assert_bootstrap_inputs_aligned_passes_for_valid_input() -> None:
    from plateproof.models.training import assert_bootstrap_inputs_aligned

    X = np.zeros((3, 2))
    y = np.zeros(3)
    ids = ["a", "b", "c"]
    restaurant_ids = np.array(["r1", "r2", "r3"])
    assert_bootstrap_inputs_aligned(X, y, ids, restaurant_ids)  # must not raise


# --------------------------------------------------------------------------- #
# Item 4: preserve the selected weighting configuration through bootstrap    #
# --------------------------------------------------------------------------- #


def _toy_data(n_restaurants: int = 20, rows_per: int = 4, seed: int = 0) -> Any:
    rng = np.random.RandomState(seed)
    restaurant_ids = np.repeat(np.arange(n_restaurants), rows_per)
    X = rng.rand(n_restaurants * rows_per, 2)
    y = (X[:, 0] + rng.rand(n_restaurants * rows_per) * 0.3 > 0.6).astype(int)
    return restaurant_ids, X, y


def test_weighted_selected_model_produces_weighted_bootstrap_fits() -> None:
    from plateproof.models.training import build_logistic_pipeline, fit_bootstrap_ensemble

    restaurant_ids, X, y = _toy_data(n_restaurants=30, rows_per=4)
    Xval, yval = X[:20], y[:20]
    captured_kwargs: list[dict[str, Any]] = []
    real_builder = build_logistic_pipeline

    class _Capturing:
        def __init__(self) -> None:
            self._inner = real_builder()

        def fit(self, X_boot: Any, y_boot: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return self._inner.fit(X_boot, y_boot, **kwargs)

        def predict_proba(self, X: Any) -> Any:
            return self._inner.predict_proba(X)

    fit_bootstrap_ensemble(
        _Capturing,
        X,
        y,
        restaurant_ids,
        Xval,
        yval,
        n_members=3,
        min_successful=1,
        weighted=True,
    )
    assert any(
        "sample_weight" in kwargs or "clf__sample_weight" in kwargs for kwargs in captured_kwargs
    )


def test_unweighted_selected_model_never_passes_sample_weight() -> None:
    from plateproof.models.training import build_logistic_pipeline, fit_bootstrap_ensemble

    restaurant_ids, X, y = _toy_data(n_restaurants=30, rows_per=4)
    Xval, yval = X[:20], y[:20]
    captured_kwargs: list[dict[str, Any]] = []
    real_builder = build_logistic_pipeline

    class _Capturing:
        def __init__(self) -> None:
            self._inner = real_builder()

        def fit(self, X_boot: Any, y_boot: Any, **kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return self._inner.fit(X_boot, y_boot, **kwargs)

        def predict_proba(self, X: Any) -> Any:
            return self._inner.predict_proba(X)

    fit_bootstrap_ensemble(
        _Capturing,
        X,
        y,
        restaurant_ids,
        Xval,
        yval,
        n_members=3,
        min_successful=1,
        weighted=False,
    )
    assert all(kwargs == {} for kwargs in captured_kwargs)


def test_build_selected_model_configuration_captures_weighting_and_metadata() -> None:
    from plateproof.models.training import (
        SelectedModelConfiguration,
        build_selected_model_configuration,
    )

    class _FakeSelection:
        selected_candidate = "logistic_regression"
        logistic_weighted = True
        hgb_weighted = False
        material_improvement_threshold = 0.01

    config = build_selected_model_configuration(
        _FakeSelection(), feature_order=("a", "b"), random_seed=42
    )
    assert isinstance(config, SelectedModelConfiguration)
    assert config.candidate_name == "logistic_regression"
    assert config.weighted is True
    assert config.random_seed == 42
    assert config.feature_order == ("a", "b")
    assert config.selection_metric
    assert config.material_improvement_threshold == 0.01
    assert config.preprocessing_description


# --------------------------------------------------------------------------- #
# Item 5: persisted bootstrap-member metadata survives a trusted reload      #
# --------------------------------------------------------------------------- #


def test_bootstrap_member_metadata_round_trips_through_artifact_reload(tmp_path: Any) -> None:
    from plateproof.models.training import (
        build_logistic_pipeline,
        deserialize_bootstrap_members,
        fit_bootstrap_ensemble,
        load_artifact,
        serialize_bootstrap_members,
        write_artifact,
    )

    restaurant_ids, X, y = _toy_data(n_restaurants=30, rows_per=4)
    Xval, yval = X[:20], y[:20]
    members, config = fit_bootstrap_ensemble(
        build_logistic_pipeline, X, y, restaurant_ids, Xval, yval, n_members=6, min_successful=3
    )
    pre_save_predictions = {
        m.member_index: m.estimator.predict_proba(X[:3])[:, 1].tolist()
        for m in members
        if m.success
    }

    bundle = {
        "bootstrap_members.joblib": serialize_bootstrap_members(members),
        "uncertainty_config.json": config.model_dump(),
    }
    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=bundle
    )
    loaded = load_artifact(output, trusted=True, require_ready=False)
    reloaded_members = deserialize_bootstrap_members(loaded["bootstrap_members.joblib"])

    # seeds and estimators remain associated
    original_by_seed = {m.seed: m for m in members}
    for reloaded in reloaded_members:
        original = original_by_seed[reloaded.seed]
        assert reloaded.success == original.success
        assert reloaded.member_index == original.member_index
        assert reloaded.failure_reason == original.failure_reason

    # predictions match pre-save predictions
    for reloaded in reloaded_members:
        if reloaded.success:
            assert reloaded.estimator.predict_proba(X[:3])[:, 1].tolist() == pytest.approx(
                pre_save_predictions[reloaded.member_index]
            )

    # failed-member counts and reasons remain accurate
    reloaded_failed = [m for m in reloaded_members if not m.success]
    original_failed = [m for m in members if not m.success]
    assert len(reloaded_failed) == len(original_failed)
    assert {m.failure_reason for m in reloaded_failed} == {
        m.failure_reason for m in original_failed
    }

    # the configured minimum-successful-member rule remains enforceable
    reloaded_config = loaded["uncertainty_config.json"]
    successful_count = sum(1 for m in reloaded_members if m.success)
    assert successful_count == reloaded_config["successful_members"]
    assert successful_count >= reloaded_config["min_successful_required"]


# --------------------------------------------------------------------------- #
# Item 6: machine-readable deployment/readiness status                       #
# --------------------------------------------------------------------------- #


def _split_report(*, train_n: int = 10, val_n: int = 10, test_n: int = 10) -> Any:
    from datetime import date

    from plateproof.models.training import (
        ChronologicalSplitReport,
        SplitBoundaries,
        SplitPartitionSummary,
    )

    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 1, 1),
        validation_end=date(2024, 2, 1),
        test_end=date(2024, 3, 1),
    )

    def _summary(name: str, n: int) -> Any:
        return SplitPartitionSummary(
            name=name,
            row_count=n,
            positive_count=n // 2,
            negative_count=n - n // 2,
            prevalence=0.5 if n else None,
            earliest_date=date(2023, 1, 1) if n else None,
            latest_date=date(2023, 6, 1) if n else None,
        )

    return ChronologicalSplitReport(
        boundaries=boundaries,
        train=_summary("train", train_n),
        validation=_summary("validation", val_n),
        test=_summary("test", test_n),
        excluded_before_train_count=0,
        excluded_after_test_count=0,
    )


def _test_metrics(
    *,
    positive_count: int = 5,
    negative_count: int = 5,
    average_precision: float | None = 0.6,
    roc_auc: float | None = 0.7,
    brier_score: float | None = 0.15,
) -> Any:
    """A minimal, directly-constructed ``EvaluationMetrics`` for exercising
    ``determine_readiness``'s test-partition checks without needing to
    reverse-engineer inputs to ``compute_metrics`` that produce a specific
    (possibly non-finite/missing) metric value."""
    from plateproof.models.training import EvaluationMetrics

    row_count = positive_count + negative_count
    return EvaluationMetrics(
        partition="test",
        row_count=row_count,
        positive_count=positive_count,
        negative_count=negative_count,
        prevalence=(positive_count / row_count) if row_count else None,
        average_precision=average_precision,
        average_precision_unavailable_reason=None,
        roc_auc=roc_auc,
        roc_auc_unavailable_reason=None,
        brier_score=brier_score,
        log_loss_value=0.5,
        log_loss_unavailable_reason=None,
        expected_calibration_error=0.05,
        top_decile_recall=0.5,
        top_decile_row_count=1,
        top_decile_unavailable_reason=None,
        precision_at_threshold=0.5,
        recall_at_threshold=0.5,
        threshold=0.5,
        confusion_matrix={"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        prediction_mean=0.5,
        prediction_median=0.5,
    )


def test_underperforming_model_is_not_ready() -> None:
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="insufficient_performance",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(),
    )
    assert status == ModelReadinessStatus.INSUFFICIENT_PERFORMANCE
    assert reason


def test_uncalibrated_model_is_not_ready() -> None:
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="uncalibrated_insufficient_data",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(),
    )
    assert status == ModelReadinessStatus.UNCALIBRATED
    assert reason


def test_uncertainty_incomplete_model_is_not_ready() -> None:
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="insufficient_bootstrap_members",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(),
    )
    assert status == ModelReadinessStatus.INSUFFICIENT_UNCERTAINTY
    assert reason


def test_fully_validated_model_is_ready() -> None:
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(),
    )
    assert status == ModelReadinessStatus.READY
    assert reason


def test_empty_test_partition_is_insufficient_data() -> None:
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, _ = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(test_n=0),
        test_metrics=_test_metrics(),
    )
    assert status == ModelReadinessStatus.INSUFFICIENT_DATA


# --------------------------------------------------------------------------- #
# Follow-up correction: a single-class test partition must never be READY    #
# --------------------------------------------------------------------------- #


def test_single_class_test_partition_is_still_permitted_by_chronological_split() -> None:
    """Requirement 1: chronological_split keeps allowing a nonempty
    single-class test partition (evaluation/audit still possible)."""
    from datetime import date

    import polars as pl

    from plateproof.models.training import SplitBoundaries, TrainingFrame, chronological_split

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
    X = pl.DataFrame({"inspection_id": ids, "f": [1.0] * len(ids)})
    y = pl.DataFrame({"inspection_id": ids, "label": labels})
    identity = pl.DataFrame({"inspection_id": ids, "inspection_date": dates})
    frame = TrainingFrame(
        X=X, y=y, identity=identity, feature_order=("f",), target_name="t", jurisdiction="nyc"
    )
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2024, 2, 1),
        validation_end=date(2024, 4, 1),
        test_end=date(2024, 6, 1),
    )
    _, _, test, _ = chronological_split(frame, boundaries)  # must not raise
    assert test.y.height == 2
    assert set(test.y.get_column("label").to_list()) == {0}


def test_single_class_test_cannot_produce_ready() -> None:
    """Requirement 2."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, _ = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(
            positive_count=10, negative_count=0, average_precision=None, roc_auc=None
        ),
    )
    assert status != ModelReadinessStatus.READY
    assert status in (ModelReadinessStatus.EVALUATION_ONLY, ModelReadinessStatus.INSUFFICIENT_DATA)


def test_single_class_test_produces_a_precise_non_ready_reason() -> None:
    """Requirement 3."""
    from plateproof.models.training import determine_readiness

    _, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(
            positive_count=10, negative_count=0, average_precision=None, roc_auc=None
        ),
    )
    assert "single-class" in reason.lower() or "single class" in reason.lower()


def test_missing_test_average_precision_prevents_readiness() -> None:
    """Requirement 7."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(average_precision=None),
    )
    assert status != ModelReadinessStatus.READY
    assert "average precision" in reason.lower()


def test_non_finite_test_average_precision_prevents_readiness() -> None:
    """Requirement 7 (non-finite variant)."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(average_precision=float("nan")),
    )
    assert status != ModelReadinessStatus.READY
    assert "average precision" in reason.lower()


def test_missing_test_brier_score_prevents_readiness() -> None:
    """Requirement 8."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(brier_score=None),
    )
    assert status != ModelReadinessStatus.READY
    assert "brier" in reason.lower()


def test_non_finite_test_brier_score_prevents_readiness() -> None:
    """Requirement 8 (non-finite variant)."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(brier_score=float("inf")),
    )
    assert status != ModelReadinessStatus.READY
    assert "brier" in reason.lower()


def test_single_class_test_artifact_is_rejected_by_default_but_loads_for_audit(
    tmp_path: Any,
) -> None:
    """Requirements 4 and 5: an evaluation-only artifact produced from a
    single-class test partition is rejected by ``load_artifact`` under the
    default ``require_ready=True``, and loads under an explicit
    ``require_ready=False`` audit override."""
    from plateproof.models.training import (
        ModelReadinessStatus,
        assemble_production_bundle,
        load_artifact,
        write_artifact,
    )

    inputs = _minimal_bundle_inputs()
    inputs["test_metrics"] = _test_metrics(
        positive_count=8, negative_count=0, average_precision=None, roc_auc=None
    )
    bundle = assemble_production_bundle(**inputs)
    assert bundle["deployment_status.json"]["status"] == ModelReadinessStatus.EVALUATION_ONLY.value

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=bundle
    )

    with pytest.raises(ValueError, match="non-ready"):
        load_artifact(output, trusted=True)

    loaded = load_artifact(output, trusted=True, require_ready=False)
    assert loaded["deployment_status.json"]["status"] == ModelReadinessStatus.EVALUATION_ONLY.value


def test_two_class_test_with_finite_ap_roc_and_brier_can_become_ready() -> None:
    """Requirement 6."""
    from plateproof.models.training import ModelReadinessStatus, determine_readiness

    status, reason = determine_readiness(
        model_status="validated",
        calibration_status="calibrated_sigmoid",
        uncertainty_status="available",
        feature_order=("a",),
        split_report=_split_report(),
        test_metrics=_test_metrics(
            positive_count=6, negative_count=4, average_precision=0.6, roc_auc=0.7, brier_score=0.1
        ),
    )
    assert status == ModelReadinessStatus.READY
    assert reason


def test_load_artifact_rejects_non_ready_by_default(tmp_path: Any) -> None:
    from plateproof.models.training import load_artifact, write_artifact

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {
            "status": "insufficient_performance",
            "reason": "did not beat baseline",
        },
    }
    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=bundle
    )
    with pytest.raises(ValueError, match="non-ready"):
        load_artifact(output, trusted=True)
    loaded = load_artifact(output, trusted=True, require_ready=False)  # explicit override works
    assert loaded["deployment_status.json"]["status"] == "insufficient_performance"


def test_load_artifact_accepts_ready_artifact_by_default(tmp_path: Any) -> None:
    from plateproof.models.training import load_artifact, write_artifact

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "ready", "reason": "all readiness checks passed"},
    }
    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=bundle
    )
    loaded = load_artifact(output, trusted=True)  # require_ready=True is the default
    assert loaded["deployment_status.json"]["status"] == "ready"


# --------------------------------------------------------------------------- #
# Item 10: recoverable overwrite                                             #
# --------------------------------------------------------------------------- #


def test_failed_forced_replacement_leaves_previous_artifact_intact(
    tmp_path: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.training import load_artifact, write_artifact

    original_bundle = {"model.joblib": {"version": "original"}}
    write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=original_bundle
    )

    real_serialize_one = training_module._serialize_one
    call_count = {"n": 0}

    def _flaky_serialize_one(path: Any, value: Any, *, final_name: str) -> None:
        call_count["n"] += 1
        # Fail partway through the SECOND write_artifact call (the forced
        # replacement), after at least one file has already been written.
        if call_count["n"] > 1:
            raise RuntimeError("simulated disk failure during replacement")
        real_serialize_one(path, value, final_name=final_name)

    monkeypatch.setattr(training_module, "_serialize_one", _flaky_serialize_one)

    new_bundle = {"model.joblib": {"version": "new"}, "extra.json": {"note": "new content"}}
    with pytest.raises(RuntimeError, match="simulated disk failure"):
        write_artifact(
            tmp_path,
            jurisdiction="nyc",
            target_name="t",
            model_version="v1",
            bundle=new_bundle,
            force=True,
        )

    output = tmp_path / "nyc" / "t" / "v1"
    # the previous, complete artifact must be exactly as it was
    loaded = load_artifact(output, trusted=True, require_ready=False)
    assert loaded["model.joblib"] == {"version": "original"}
    # no stray replacement/backup directories left behind
    siblings = {p.name for p in (tmp_path / "nyc" / "t").iterdir()}
    assert siblings == {"v1"}


# --------------------------------------------------------------------------- #
# Items 7-9: complete artifact bundle, risk-band policy, model card          #
# --------------------------------------------------------------------------- #


def _minimal_bundle_inputs(jurisdiction: str = "nyc") -> dict[str, Any]:
    from datetime import date

    from plateproof.models.calibration import (
        CalibrationOutcome,
        CalibrationReport,
        CalibrationStatus,
    )
    from plateproof.models.training import (
        BootstrapMember,
        ModelSelectionResult,
        SplitBoundaries,
        UncertaintyConfig,
        UncertaintyStatus,
        build_selected_model_configuration,
        compute_metrics,
    )

    selection = ModelSelectionResult(
        prevalence_ap=0.2,
        logistic_ap=0.4,
        hgb_ap=0.41,
        logistic_weighted=True,
        hgb_weighted=False,
        logistic_beats_prevalence=True,
        boosted_beats_logistic=False,
        selected_candidate="logistic_regression",
        candidate_validation_ap=0.4,
        model_status="validated",
        material_improvement_threshold=0.01,
        selected_estimator=object(),
    )
    feature_order = (
        ("history_depth", "nyc_previous_day_score")
        if jurisdiction == "nyc"
        else ("history_depth", "fl_previous_day_basic_count")
    )
    selected_config = build_selected_model_configuration(selection, feature_order, random_seed=42)
    calibration_report = CalibrationReport(
        status=CalibrationStatus.CALIBRATED_SIGMOID,
        method="sigmoid",
        brier_before=0.2,
        brier_after=0.15,
        validation_row_count=100,
    )
    calibration_outcome = CalibrationOutcome(report=calibration_report, estimator=object())

    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.8, 0.2, 0.7, 0.9, 0.3, 0.6, 0.15, 0.85, 0.75])
    metrics = compute_metrics(y_true, y_prob, partition="x")

    members = [
        BootstrapMember(
            member_index=i,
            seed=100 + i,
            estimator=object() if i < 3 else None,
            training_restaurant_count=10,
            training_row_count=40,
            success=i < 3,
            calibration_status="calibrated_sigmoid" if i < 3 else None,
            failure_reason=None if i < 3 else "single_class_resample",
        )
        for i in range(4)
    ]
    uncertainty_config = UncertaintyConfig(
        requested_members=4,
        successful_members=3,
        failed_members=1,
        failure_reasons={"single_class_resample": 1},
        min_successful_required=2,
        status=UncertaintyStatus.AVAILABLE,
        seeds=[100, 101, 102, 103],
    )

    boundaries = SplitBoundaries(
        jurisdiction=jurisdiction,
        target_name="t",
        train_end=date(2024, 1, 1),
        validation_end=date(2024, 2, 1),
        test_end=date(2024, 3, 1),
    )

    return {
        "jurisdiction": jurisdiction,
        "target_name": "t",
        "model_version": "v1",
        "feature_order": feature_order,
        "target_definition": "label = 1 iff ...",
        "eligible_inspection_types": ["Routine"],
        "excluded_inspection_types": ["Complaint"],
        "feature_categories": ["shared history", "jurisdiction-native history"],
        "split_boundaries": boundaries,
        "split_report": _split_report(),
        "target_build_report": None,
        "selection": selection,
        "selected_config": selected_config,
        "calibration_outcome": calibration_outcome,
        "train_metrics": metrics,
        "validation_metrics": metrics,
        "test_metrics": metrics,
        "members": members,
        "uncertainty_config": uncertainty_config,
        "subgroup_limitations": "Subgroup metrics are suppressed below minimum counts.",
        "data_snapshot_note": "unavailable",
        "known_limitations": ["Small synthetic fixture only."],
        "source_provenance": None,
        "random_seeds": {"training": 42},
    }


def test_assemble_production_bundle_contains_all_required_fields() -> None:
    from plateproof.models.training import assemble_production_bundle

    bundle = assemble_production_bundle(**_minimal_bundle_inputs())
    required_keys = {
        "point_estimator.joblib",
        "bootstrap_members.joblib",
        "feature_order.json",
        "feature_dtypes.json",
        "jurisdiction.json",
        "target_definition.json",
        "selection_report.json",
        "selected_configuration.json",
        "calibration_report.json",
        "uncertainty_config.json",
        "split_report.json",
        "metrics.json",
        "risk_band_thresholds.json",
        "history_sufficiency_rule.json",
        "deployment_status.json",
        "environment.json",
        "source_provenance.json",
        "artifact_schema_version.json",
        "model_card.md",
    }
    assert required_keys <= set(bundle)
    assert bundle["feature_order.json"] == list(_minimal_bundle_inputs()["feature_order"])
    assert bundle["jurisdiction.json"] == "nyc"
    assert bundle["artifact_schema_version.json"]
    env = bundle["environment.json"]
    for key in ("python", "scikit_learn", "polars", "numpy"):
        assert key in env
    assert bundle["source_provenance.json"] == {"status": "unavailable"}
    assert bundle["deployment_status.json"]["status"] == "ready"


def test_assemble_production_bundle_records_insufficient_performance_honestly() -> None:
    from plateproof.models.training import ModelReadinessStatus, assemble_production_bundle

    inputs = _minimal_bundle_inputs()
    inputs["selection"] = inputs["selection"].model_copy(
        update={"model_status": "insufficient_performance", "selected_candidate": "prevalence"}
    )
    bundle = assemble_production_bundle(**inputs)
    expected_status = ModelReadinessStatus.INSUFFICIENT_PERFORMANCE.value
    assert bundle["deployment_status.json"]["status"] == expected_status
    assert "not approved for production" in bundle["model_card.md"].lower()


def test_nyc_bundle_carries_only_nyc_risk_policy() -> None:
    from plateproof.models.training import assemble_production_bundle

    bundle = assemble_production_bundle(**_minimal_bundle_inputs(jurisdiction="nyc"))
    assert bundle["risk_band_thresholds.json"]["jurisdiction"] == "nyc"
    assert bundle["history_sufficiency_rule.json"]["jurisdiction"] == "nyc"
    assert "nyc_prior_valid_score_count" in str(bundle["history_sufficiency_rule.json"])
    assert "fl_prior_valid_high_priority_count" not in str(bundle["history_sufficiency_rule.json"])


def test_florida_bundle_carries_only_florida_risk_policy() -> None:
    from plateproof.models.training import assemble_production_bundle

    bundle = assemble_production_bundle(**_minimal_bundle_inputs(jurisdiction="florida"))
    assert bundle["risk_band_thresholds.json"]["jurisdiction"] == "florida"
    assert bundle["history_sufficiency_rule.json"]["jurisdiction"] == "florida"
    assert "fl_prior_valid_high_priority_count" in str(bundle["history_sufficiency_rule.json"])
    assert "nyc_prior_valid_score_count" not in str(bundle["history_sufficiency_rule.json"])


def test_model_card_contains_required_sections() -> None:
    from plateproof.models.training import assemble_production_bundle

    bundle = assemble_production_bundle(**_minimal_bundle_inputs())
    card = bundle["model_card.md"]
    for required_substring in (
        "model version",
        "target definition",
        "eligible",
        "excluded",
        "feature categories",
        "temporal leakage",
        "split",
        "candidate comparison",
        "selected estimator",
        "calibration",
        "risk band",
        "bootstrap",
        "evaluation metrics",
        "subgroup",
        "snapshot",
        "known limitations",
        "non-causal",
        "michelin",
        "google",
        "not an official inspection result",
        "deployment status",
    ):
        assert required_substring in card.lower(), f"missing section: {required_substring!r}"


def test_insufficient_performance_model_card_states_not_approved() -> None:
    from plateproof.models.training import assemble_production_bundle

    inputs = _minimal_bundle_inputs()
    inputs["selection"] = inputs["selection"].model_copy(
        update={"model_status": "insufficient_performance", "selected_candidate": "prevalence"}
    )
    bundle = assemble_production_bundle(**inputs)
    assert "not approved for production" in bundle["model_card.md"].lower()


# --------------------------------------------------------------------------- #
# Item 13: small synthetic end-to-end training run, per jurisdiction         #
# --------------------------------------------------------------------------- #


def _run_synthetic_training_end_to_end(frame: Any, tmp_path: Any) -> dict[str, Any]:
    from datetime import timedelta

    from plateproof.models.calibration import calibrate_on_validation
    from plateproof.models.risk_bands import assign_risk_band, score_row
    from plateproof.models.training import (
        CANDIDATE_BUILDERS,
        SplitBoundaries,
        aligned_restaurant_ids,
        assemble_production_bundle,
        assert_bootstrap_inputs_aligned,
        build_selected_model_configuration,
        chronological_split,
        compute_metrics,
        feature_matrix,
        fit_and_select_model,
        fit_bootstrap_ensemble,
        load_artifact,
        write_artifact,
    )

    start = frame.identity.get_column("inspection_date").min()
    boundaries = SplitBoundaries(
        jurisdiction=frame.jurisdiction,
        target_name=frame.target_name,
        train_end=start + timedelta(days=120),
        validation_end=start + timedelta(days=160),
        test_end=start + timedelta(days=200),
    )
    train, validation, test, split_report = chronological_split(frame, boundaries)

    selection = fit_and_select_model(train, validation)
    selected_config = build_selected_model_configuration(
        selection, frame.feature_order, random_seed=123
    )

    Xtrain = feature_matrix(train.X, train.feature_order)
    ytrain = train.y.get_column("label").to_numpy()
    Xval = feature_matrix(validation.X, validation.feature_order)
    yval = validation.y.get_column("label").to_numpy()
    Xtest = feature_matrix(test.X, test.feature_order)
    ytest = test.y.get_column("label").to_numpy()

    calibration = calibrate_on_validation(selection.selected_estimator, Xval, yval)
    train_metrics = compute_metrics(
        ytrain, calibration.estimator.predict_proba(Xtrain)[:, 1], partition="train"
    )
    validation_metrics = compute_metrics(
        yval, calibration.estimator.predict_proba(Xval)[:, 1], partition="validation"
    )
    test_metrics = compute_metrics(
        ytest, calibration.estimator.predict_proba(Xtest)[:, 1], partition="test"
    )

    train_ids = train.X.get_column("inspection_id").to_list()
    restaurant_ids = aligned_restaurant_ids(train_ids, train.identity)
    assert_bootstrap_inputs_aligned(Xtrain, ytrain, train_ids, restaurant_ids)

    build_fn = CANDIDATE_BUILDERS[selected_config.candidate_name]
    members, uncertainty_config = fit_bootstrap_ensemble(
        build_fn,
        Xtrain,
        ytrain,
        restaurant_ids,
        Xval,
        yval,
        n_members=25,
        min_successful=15,
        weighted=selected_config.weighted,
        base_seed=selected_config.random_seed,
    )

    bundle = assemble_production_bundle(
        jurisdiction=frame.jurisdiction,
        target_name=frame.target_name,
        model_version="v1",
        feature_order=frame.feature_order,
        target_definition="synthetic end-to-end test target",
        eligible_inspection_types=["synthetic"],
        excluded_inspection_types=[],
        feature_categories=["synthetic signal", "synthetic noise"],
        split_boundaries=boundaries,
        split_report=split_report,
        target_build_report=None,
        selection=selection,
        selected_config=selected_config,
        calibration_outcome=calibration,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        members=members,
        uncertainty_config=uncertainty_config,
        subgroup_limitations="none (synthetic fixture)",
        data_snapshot_note="unavailable (synthetic fixture)",
        known_limitations=["synthetic data only"],
        source_provenance=None,
        random_seeds={"training": selected_config.random_seed},
    )

    output = write_artifact(
        tmp_path,
        jurisdiction=frame.jurisdiction,
        target_name=frame.target_name,
        model_version="v1",
        bundle=bundle,
    )

    loaded = load_artifact(
        output,
        trusted=True,
        expected_jurisdiction=frame.jurisdiction,
        require_ready=bundle["deployment_status.json"]["status"] == "ready",
    )
    assert loaded["manifest"]["jurisdiction"] == frame.jurisdiction

    point_estimator = loaded["point_estimator.joblib"]
    row = Xtest[:1]
    point = float(point_estimator.predict_proba(row)[:, 1][0])
    from plateproof.models.training import deserialize_bootstrap_members

    boot_members = deserialize_bootstrap_members(loaded["bootstrap_members.joblib"])
    successful = [m for m in boot_members if m.success]
    thresholds_dict = loaded["risk_band_thresholds.json"]

    from plateproof.models.risk_bands import RiskBandThresholds

    thresholds = RiskBandThresholds(**thresholds_dict)
    band = assign_risk_band(point, thresholds)

    scored = score_row(
        probability=point,
        lower=0.0,
        upper=1.0,
        features_row={"missing_history": False, "nyc_prior_valid_score_count": 1},
        jurisdiction="nyc",  # sufficiency rule is jurisdiction-keyed, not frame-keyed here
        thresholds=thresholds,
    )

    return {
        "bundle": bundle,
        "loaded": loaded,
        "point": point,
        "band": band,
        "scored": scored,
        "successful_members": successful,
    }


def test_synthetic_end_to_end_training_nyc(synthetic_frame: Any, tmp_path: Any) -> None:
    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    result = _run_synthetic_training_end_to_end(frame, tmp_path)
    assert result["bundle"]["jurisdiction.json"] == "nyc"
    assert result["loaded"]["manifest"]["jurisdiction"] == "nyc"
    assert 0.0 <= result["point"] <= 1.0
    assert result["band"] in ("low", "moderate", "high")
    assert result["scored"].risk_band == result["band"]
    assert len(result["successful_members"]) >= 15


def test_synthetic_end_to_end_training_florida(synthetic_frame: Any, tmp_path: Any) -> None:
    import dataclasses

    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    florida_frame = dataclasses.replace(frame, jurisdiction="florida")
    result = _run_synthetic_training_end_to_end(florida_frame, tmp_path)
    assert result["bundle"]["jurisdiction.json"] == "florida"
    assert result["loaded"]["manifest"]["jurisdiction"] == "florida"
    assert 0.0 <= result["point"] <= 1.0
    assert result["band"] in ("low", "moderate", "high")
    assert len(result["successful_members"]) >= 15
