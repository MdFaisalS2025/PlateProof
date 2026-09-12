"""CLI to train, calibrate, evaluate, and artifact one jurisdiction/target risk model.

One jurisdiction and one target per invocation. No auto-download, no Michelin
or Google content, no paid service. A full production training run is
intentionally not exercised by the automated test suite -- only --dry-run and
argument-validation paths are covered there.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from plateproof.models.calibration import calibrate_on_validation
from plateproof.models.training import (
    MIN_SUCCESSFUL_BOOTSTRAP_MEMBERS,
    N_BOOTSTRAP_PRODUCTION_DEFAULT,
    PrevalenceBaseline,
    SplitBoundaries,
    assemble_training_frame,
    build_hgb,
    build_logistic_pipeline,
    chronological_split,
    compute_metrics,
    feature_matrix,
    fit_and_select_model,
    fit_bootstrap_ensemble,
    write_artifact,
)

TARGET_JURISDICTION: dict[str, str] = {
    "nyc_next_initial_score_ge_14": "nyc",
    "nyc_next_score_ge_28": "nyc",
    "nyc_next_any_critical_violation": "nyc",
    "florida_next_routine_high_priority_or_follow_up": "florida",
    "florida_next_temporary_closure": "florida",
}

_CANDIDATE_BUILDERS: dict[str, Any] = {
    "prevalence": PrevalenceBaseline,
    "logistic_regression": build_logistic_pipeline,
    "hist_gradient_boosting": build_hgb,
}


def _build_target(target_name: str, events: Any) -> tuple[Any, Any | None]:
    from plateproof.models.florida_risk import (
        build_florida_primary_target,
        build_florida_temporary_closure_target,
    )
    from plateproof.models.nyc_risk import (
        build_nyc_any_critical_target,
        build_nyc_primary_target,
        build_nyc_score_ge_28_target,
    )

    if target_name == "nyc_next_initial_score_ge_14":
        return build_nyc_primary_target(events)
    if target_name == "nyc_next_score_ge_28":
        return build_nyc_score_ge_28_target(events), None
    if target_name == "nyc_next_any_critical_violation":
        return build_nyc_any_critical_target(events), None
    if target_name == "florida_next_routine_high_priority_or_follow_up":
        return build_florida_primary_target(events)
    if target_name == "florida_next_temporary_closure":
        return build_florida_temporary_closure_target(events), None
    raise ValueError(f"unknown target: {target_name!r}")


def _load_events(jurisdiction: str, events_path: Path) -> Any:
    now = datetime.now(UTC)
    if jurisdiction == "nyc":
        from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

        raw = load_nyc_raw(events_path)
        return build_nyc_inspection_events(raw, ingested_at=now).inspection_events
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    raw = load_florida_extracts([FloridaExtractSource(path=events_path)])
    return build_florida_inspection_events(raw, ingested_at=now).inspection_events


def _load_violations(jurisdiction: str, events_path: Path) -> Any:
    now = datetime.now(UTC)
    if jurisdiction == "nyc":
        from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

        raw = load_nyc_raw(events_path)
        return build_nyc_inspection_events(raw, ingested_at=now).violation_events
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    raw = load_florida_extracts([FloridaExtractSource(path=events_path)])
    return build_florida_inspection_events(raw, ingested_at=now).violation_events


def _feature_list(jurisdiction: str) -> tuple[str, ...]:
    if jurisdiction == "nyc":
        from plateproof.models.nyc_risk import NYC_FEATURE_LIST

        return NYC_FEATURE_LIST
    from plateproof.models.florida_risk import FL_FEATURE_LIST

    return FL_FEATURE_LIST


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, calibrate, and artifact one jurisdiction's risk model."
    )
    parser.add_argument("--jurisdiction", required=True, choices=["nyc", "florida"])
    parser.add_argument("--target", required=True, choices=sorted(TARGET_JURISDICTION))
    parser.add_argument("--events", required=True, help="Path to a raw NYC/Florida extract CSV.")
    parser.add_argument("--output", required=True, help="Artifact output root directory.")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--train-start", type=date.fromisoformat, default=None)
    parser.add_argument("--train-end", type=date.fromisoformat, default=None)
    parser.add_argument("--validation-end", type=date.fromisoformat, default=None)
    parser.add_argument("--test-end", type=date.fromisoformat, default=None)
    parser.add_argument("--cutoff-date", type=date.fromisoformat, default=None)
    parser.add_argument("--bootstrap-members", type=int, default=N_BOOTSTRAP_PRODUCTION_DEFAULT)
    parser.add_argument(
        "--min-successful-bootstrap-members", type=int, default=MIN_SUCCESSFUL_BOOTSTRAP_MEMBERS
    )
    parser.add_argument("--force", action="store_true", help="Overwrite a completed artifact.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration only; write nothing to disk.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run the full pipeline and print evaluation, but do not write an artifact.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    expected_jurisdiction = TARGET_JURISDICTION.get(args.target)
    if expected_jurisdiction != args.jurisdiction:
        print(
            f"target {args.target!r} belongs to jurisdiction {expected_jurisdiction!r}, "
            f"not {args.jurisdiction!r}",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("dry run: configuration is valid; no data was loaded and nothing was written")
        return 0

    if not (args.train_end and args.validation_end and args.test_end):
        print(
            "a production run requires --train-end, --validation-end, and --test-end",
            file=sys.stderr,
        )
        return 2

    events_path = Path(args.events)
    events = _load_events(args.jurisdiction, events_path)
    violations = _load_violations(args.jurisdiction, events_path)
    target, target_report = _build_target(args.target, events)

    max_date = events.get_column("inspection_date").max()
    cutoff_date = args.cutoff_date or max_date
    if cutoff_date is None:
        print("no rows in --events; cannot build temporal features", file=sys.stderr)
        return 2

    from plateproof.features.temporal import build_temporal_features

    temporal = build_temporal_features(events, violations, cutoff_date)
    feature_list = _feature_list(args.jurisdiction)
    frame = assemble_training_frame(temporal, target, feature_list, args.jurisdiction, args.target)

    boundaries = SplitBoundaries(
        jurisdiction=args.jurisdiction,
        target_name=args.target,
        train_start=args.train_start,
        train_end=args.train_end,
        validation_end=args.validation_end,
        test_end=args.test_end,
    )
    train, validation, test, split_report = chronological_split(frame, boundaries)

    selection = fit_and_select_model(train, validation)

    Xval = feature_matrix(validation.X, validation.feature_order)
    yval = validation.y.get_column("label").to_numpy()
    calibration = calibrate_on_validation(selection.selected_estimator, Xval, yval)

    Xtrain = feature_matrix(train.X, train.feature_order)
    ytrain = train.y.get_column("label").to_numpy()
    Xtest = feature_matrix(test.X, test.feature_order)
    ytest = test.y.get_column("label").to_numpy()

    train_metrics = compute_metrics(
        ytrain, calibration.estimator.predict_proba(Xtrain)[:, 1], partition="train"
    )
    validation_metrics = compute_metrics(
        yval, calibration.estimator.predict_proba(Xval)[:, 1], partition="validation"
    )
    test_metrics = compute_metrics(
        ytest, calibration.estimator.predict_proba(Xtest)[:, 1], partition="test"
    )

    build_fn = _CANDIDATE_BUILDERS[selection.selected_candidate]
    restaurant_ids = (
        train.identity.join(train.y.select("inspection_id"), on="inspection_id")
        .get_column("restaurant_id")
        .to_numpy()
    )
    members, uncertainty_config = fit_bootstrap_ensemble(
        build_fn,
        Xtrain,
        ytrain,
        restaurant_ids,
        Xval,
        yval,
        n_members=args.bootstrap_members,
        min_successful=args.min_successful_bootstrap_members,
    )

    print(
        f"selected={selection.selected_candidate} status={selection.model_status} "
        f"calibration={calibration.report.status.value} "
        f"train_ap={train_metrics.average_precision} "
        f"validation_ap={validation_metrics.average_precision} "
        f"test_ap={test_metrics.average_precision} "
        f"bootstrap_successful={uncertainty_config.successful_members}"
    )

    if args.validate_only:
        print("validate-only: evaluation complete; no artifact written")
        return 0

    bundle = {
        "point_estimator.joblib": calibration.estimator,
        "bootstrap_members.joblib": [m.estimator for m in members if m.success],
        "selection_report.json": selection.model_dump(exclude={"selected_estimator"}),
        "calibration_report.json": {
            "status": calibration.report.status.value,
            "method": calibration.report.method,
            "brier_before": calibration.report.brier_before,
            "brier_after": calibration.report.brier_after,
            "validation_row_count": calibration.report.validation_row_count,
        },
        "uncertainty_config.json": uncertainty_config.model_dump(),
        "split_report.json": split_report.model_dump(mode="json"),
        "target_build_report.json": target_report.model_dump()
        if target_report is not None
        else None,
        "metrics.json": {
            "train": train_metrics.model_dump(),
            "validation": validation_metrics.model_dump(),
            "test": test_metrics.model_dump(),
        },
        "feature_order.json": list(frame.feature_order),
    }
    output_path = write_artifact(
        args.output,
        jurisdiction=args.jurisdiction,
        target_name=args.target,
        model_version=args.model_version,
        bundle=bundle,
        force=args.force,
    )
    print(str(output_path))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
