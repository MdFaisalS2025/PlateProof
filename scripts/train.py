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
from plateproof.models.florida_risk import FL_QUALIFYING_TYPE
from plateproof.models.nyc_risk import NYC_PRIMARY_QUALIFYING_TYPES
from plateproof.models.training import (
    CANDIDATE_BUILDERS,
    MIN_SUCCESSFUL_BOOTSTRAP_MEMBERS,
    N_BOOTSTRAP_PRODUCTION_DEFAULT,
    RANDOM_SEED,
    ModelReadinessStatus,
    SplitBoundaries,
    aligned_restaurant_ids,
    assemble_production_bundle,
    assemble_training_frame,
    assert_bootstrap_inputs_aligned,
    build_selected_model_configuration,
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

_TARGET_METADATA: dict[str, dict[str, Any]] = {
    "nyc_next_initial_score_ge_14": {
        "definition": (
            "label = 1 iff an exactly-qualifying initial NYC inspection's own score is "
            ">= 14 and unconflicted; the row is excluded otherwise."
        ),
        "eligible_inspection_types": sorted(NYC_PRIMARY_QUALIFYING_TYPES),
        "excluded_inspection_types": [
            "all non-qualifying NYC inspection types",
            "qualifying inspections with a missing or conflicted score",
        ],
        "feature_categories": [
            "shared inspection-history features (days since prior visit, prior violation counts)",
            "NYC score history (previous score, prior mean/max/variance/trend)",
            "NYC critical-violation history",
        ],
    },
    "nyc_next_score_ge_28": {
        "definition": "label = 1 iff the next qualifying NYC inspection's own score is >= 28.",
        "eligible_inspection_types": sorted(NYC_PRIMARY_QUALIFYING_TYPES),
        "excluded_inspection_types": [
            "all non-qualifying NYC inspection types",
            "qualifying inspections with a missing or conflicted score",
        ],
        "feature_categories": [
            "shared inspection-history features",
            "NYC score history",
            "NYC critical-violation history",
        ],
    },
    "nyc_next_any_critical_violation": {
        "definition": (
            "label = 1 iff the next qualifying NYC inspection's own "
            "critical_violation_count is >= 1."
        ),
        "eligible_inspection_types": sorted(NYC_PRIMARY_QUALIFYING_TYPES),
        "excluded_inspection_types": ["all non-qualifying NYC inspection types"],
        "feature_categories": [
            "shared inspection-history features",
            "NYC score history",
            "NYC critical-violation history",
        ],
    },
    "florida_next_routine_high_priority_or_follow_up": {
        "definition": (
            "label = 1 iff the initial routine-food visit has any high-priority violation, "
            "requires follow-up, or resulted in temporary closure."
        ),
        "eligible_inspection_types": [FL_QUALIFYING_TYPE],
        "excluded_inspection_types": [
            "all non-routine-food Florida inspection types",
            "non-initial visits",
            "ambiguous duplicate initial-visit groups",
            "visits with a missing high-priority count or unrecognized disposition",
        ],
        "feature_categories": [
            "shared inspection-history features",
            "Florida violation-severity history (high-priority/intermediate/basic counts)",
        ],
    },
    "florida_next_temporary_closure": {
        "definition": (
            "label = 1 iff the initial routine-food visit resulted in temporary closure."
        ),
        "eligible_inspection_types": [FL_QUALIFYING_TYPE],
        "excluded_inspection_types": [
            "all non-routine-food Florida inspection types",
            "non-initial visits",
            "visits with an unrecognized disposition",
        ],
        "feature_categories": [
            "shared inspection-history features",
            "Florida violation-severity history",
        ],
    },
}

_SUBGROUP_LIMITATIONS = (
    "Subgroup metrics are descriptive and diagnostic only; a subgroup below the minimum "
    "row, positive, or negative count is suppressed rather than reported unreliably."
)
_KNOWN_LIMITATIONS = [
    "Absolute risk-band thresholds (0.25 / 0.50) are fixed, not validation-derived.",
    "The model is trained on one jurisdiction/target only and does not generalize across "
    "jurisdictions.",
    "Uncertainty intervals come from a restaurant-cluster bootstrap and reflect resampling "
    "variability only, not all sources of real-world uncertainty.",
]


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


def _load_ingestion_result(jurisdiction: str, events_path: Path) -> Any:
    now = datetime.now(UTC)
    if jurisdiction == "nyc":
        from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

        raw = load_nyc_raw(events_path)
        return build_nyc_inspection_events(raw, ingested_at=now)
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    raw = load_florida_extracts([FloridaExtractSource(path=events_path)])
    return build_florida_inspection_events(raw, ingested_at=now)


def _source_provenance(report: Any) -> dict[str, Any]:
    """Record what provenance the ingestion report actually carries; any
    field the report does not supply is recorded as ``"unavailable"``
    explicitly, never silently omitted."""
    snapshot_date = getattr(report, "snapshot_date", None)
    retrieved_at = getattr(report, "retrieved_at_utc", None)
    sha256 = getattr(report, "source_sha256", None)
    return {
        "source_snapshot_date": snapshot_date.isoformat() if snapshot_date else "unavailable",
        "source_retrieved_at_utc": retrieved_at.isoformat() if retrieved_at else "unavailable",
        "source_sha256": sha256 or "unavailable",
    }


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
    ingestion_result = _load_ingestion_result(args.jurisdiction, events_path)
    events = ingestion_result.inspection_events
    violations = ingestion_result.violation_events
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
    selected_config = build_selected_model_configuration(
        selection, frame.feature_order, random_seed=RANDOM_SEED
    )

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

    train_inspection_ids = train.X.get_column("inspection_id").to_list()
    restaurant_ids = aligned_restaurant_ids(train_inspection_ids, train.identity)
    assert_bootstrap_inputs_aligned(Xtrain, ytrain, train_inspection_ids, restaurant_ids)

    build_fn = CANDIDATE_BUILDERS[selected_config.candidate_name]
    members, uncertainty_config = fit_bootstrap_ensemble(
        build_fn,
        Xtrain,
        ytrain,
        restaurant_ids,
        Xval,
        yval,
        n_members=args.bootstrap_members,
        min_successful=args.min_successful_bootstrap_members,
        weighted=selected_config.weighted,
        base_seed=selected_config.random_seed,
    )

    metadata = _TARGET_METADATA[args.target]
    bundle = assemble_production_bundle(
        jurisdiction=args.jurisdiction,
        target_name=args.target,
        model_version=args.model_version,
        feature_order=frame.feature_order,
        target_definition=metadata["definition"],
        eligible_inspection_types=metadata["eligible_inspection_types"],
        excluded_inspection_types=metadata["excluded_inspection_types"],
        feature_categories=metadata["feature_categories"],
        split_boundaries=boundaries,
        split_report=split_report,
        target_build_report=target_report,
        selection=selection,
        selected_config=selected_config,
        calibration_outcome=calibration,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_metrics_computed=True,
        members=members,
        uncertainty_config=uncertainty_config,
        subgroup_limitations=_SUBGROUP_LIMITATIONS,
        data_snapshot_note=str(_source_provenance(ingestion_result.report)),
        known_limitations=_KNOWN_LIMITATIONS,
        source_provenance=_source_provenance(ingestion_result.report),
        random_seeds={"training": selected_config.random_seed},
    )

    deployment_status = bundle["deployment_status.json"]
    readiness_note = (
        "READY for production prediction"
        if deployment_status["status"] == ModelReadinessStatus.READY.value
        else f"NOT approved for production ({deployment_status['status']}): "
        f"{deployment_status['reason']}"
    )
    print(
        f"selected={selection.selected_candidate} status={selection.model_status} "
        f"calibration={calibration.report.status.value} "
        f"train_ap={train_metrics.average_precision} "
        f"validation_ap={validation_metrics.average_precision} "
        f"test_ap={test_metrics.average_precision} "
        f"bootstrap_successful={uncertainty_config.successful_members} "
        f"deployment_status={deployment_status['status']} -- {readiness_note}"
    )

    if args.validate_only:
        print("validate-only: evaluation complete; no artifact written")
        return 0

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
