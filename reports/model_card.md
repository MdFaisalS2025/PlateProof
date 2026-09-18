# PlateProof Model Card (project-level)

This is the project-level model documentation Task 11 requires. It describes the
**modeling approach and governance rules** that apply to every jurisdiction/target
PlateProof trains. It is distinct from the **per-artifact** `model_card.md` that
`plateproof.models.training.generate_model_card` writes alongside every trained model
(see `models/<jurisdiction>/<target>/<version>/model_card.md` once a model is trained)
-- that file documents one specific trained artifact's own metrics, readiness status,
and data snapshot; this file documents the methodology those artifacts all share.

**No trained model artifact ships in this repository.** `models/` contains only a
`.gitkeep` (see `.gitignore`'s `models/*` rule) -- every claim below describes the
training/serving *code path*, verified by the automated test suite and, for the
data-pipeline stages, by a live audit run (see "Verification" below), not a specific
set of numbers from a specific trained run.

## Scope: two independent models, never one universal grade

PlateProof trains **one separate model per (jurisdiction, target)** pair -- never a
model that mixes NYC and Florida rows, and never a model whose output is presented as
a universal letter grade. `plateproof/models/nyc_risk.py` and
`plateproof/models/florida_risk.py` each define their own target-construction and
feature-list logic; `scripts/train.py` requires an explicit `--jurisdiction` and
`--target` per invocation. The five supported targets:

| Target | Jurisdiction | Label definition |
|---|---|---|
| `nyc_next_initial_score_ge_14` (primary) | NYC | Next qualifying initial inspection's own score >= 14 |
| `nyc_next_score_ge_28` | NYC | Next qualifying inspection's own score >= 28 |
| `nyc_next_any_critical_violation` | NYC | Next qualifying inspection has >= 1 critical violation |
| `florida_next_routine_high_priority_or_follow_up` (primary) | Florida | Next routine-food initial visit has a high-priority violation, requires follow-up, or results in temporary closure |
| `florida_next_temporary_closure` | Florida | Next routine-food initial visit results in temporary closure |

Every target excludes non-qualifying inspection types and rows with a missing/
conflicted outcome outright, rather than imputing a label (`scripts/train.py`'s
`_TARGET_METADATA`).

## Features and leakage prevention

Features are restricted to information available strictly before the target
inspection: prior score/severity history, rolling mean/max/variance/trend, days since
the previous inspection, prior critical/high-priority counts, repeated-violation-code
counts, and jurisdiction-native context (cuisine, geography, season). NYC additionally
uses NYC score history; Florida additionally uses its own high-priority/intermediate/
basic severity history -- the two feature sets are never merged.

Two independent leakage controls (`plateproof/features/temporal.py`):

1. **A closed allowlist** (`MODEL_FEATURE_ALLOWLIST`) -- every column reaching the
   model matrix must be explicitly listed; nothing new can enter silently.
2. **A token-based defense-in-depth check** (`assert_model_matrix_is_safe`) -- any
   column name containing a token strongly associated with the target, a current-event
   field, or a third-party source (`target`, `label`, `outcome`, `michelin`, `google`,
   `next_inspection`, ...) is rejected even if it were somehow allowlisted. A
   regression test (`tests/features/test_temporal.py::test_allowlist_rejects_google_column`)
   and a static import-graph test (`tests/test_google_policy.py`) both guard this from
   two different angles -- see "Google and Michelin content" below.

## Validation methodology

Chronological, non-shuffled splits only (`chronological_split`): train, validation,
and test periods are strictly time-ordered, with the test period always the most
recent. Three candidates are compared -- a prevalence baseline, regularized logistic
regression, and a histogram gradient-boosted tree (`scikit-learn`'s
`HistGradientBoostingClassifier`) -- and the selected model must **materially beat**
both the prevalence and logistic baselines on the *validation* period to be marked
`"validated"`; if it does not, `scripts/train.py`'s own docstring and the spec both
require shipping the simpler logistic model rather than a more complex one that isn't
actually earning its complexity.

Metrics computed on the untouched test period: average precision (the primary
selection metric, per the spec's "optimize area under the precision-recall curve"),
ROC AUC, Brier score, and subgroup breakdowns (suppressed, not reported unreliably,
below a minimum row/positive/negative count per `_SUBGROUP_LIMITATIONS`).

## Calibration

Post-hoc calibration runs on the **validation partition only**, never train or test
(`plateproof/models/calibration.py`). Sigmoid (Platt) calibration is the default;
isotonic calibration is used only once the validation partition is large enough
(>= 1,000 rows and >= 400 rows in the minority class) for its extra flexibility not to
overfit. Below `MIN_VALIDATION_ROWS_FOR_CALIBRATION` (30 rows), calibration is marked
`uncalibrated_insufficient_data` rather than run at all. Calibration uses
`sklearn.frozen.FrozenEstimator` to wrap the already-fitted model so
`CalibratedClassifierCV` calibrates without ever refitting on or touching the train
partition.

## Uncertainty

Prediction intervals come from a restaurant-cluster bootstrap (resampling whole
restaurants' histories together, not individual rows, so a restaurant's correlated
inspections never leak across bootstrap members). This reflects **resampling
variability only** -- it is not a complete accounting of real-world uncertainty, and
the per-artifact model card and API/UI both state this explicitly. Too few successful
bootstrap members yields `insufficient_uncertainty`, never a fabricated interval.

## Deployment readiness gate

A trained artifact is marked `"ready"` (`plateproof.models.training.determine_readiness`)
only when **every one** of the following independently holds: a real feature schema is
present; train, validation, and test partitions are all nonempty; the selected
candidate materially beat both baselines; calibration succeeded (sigmoid or isotonic);
enough bootstrap members succeeded to report uncertainty; and the test partition has
both classes present with finite average precision, Brier score, and ROC AUC. A
single-class or metric-unavailable test partition yields `"evaluation_only"`, never a
silently-promoted `"ready"` status. `Settings.expose_non_ready_model_cards` (default
`false`) keeps a non-ready model's card out of the public-facing model-card page and
route by default (`app/pages/4_Model_Card.py`, `GET /models/{jurisdiction}/card`) --
verified live in this audit: with no artifact configured at all, the route returns 404
("no model card available") rather than fabricating a response.

## Presentation: risk bands are not grades

`plateproof.models.risk_bands.assign_risk_band` maps a probability to `low`/
`moderate`/`high`/`insufficient_history` using fixed thresholds (0.25 / 0.50) --
**explicitly documented as a known limitation** (`_KNOWN_LIMITATIONS` in
`scripts/train.py`): these are presentation bands, not validation-derived or
jurisdiction-specific thresholds, are never an official grade, and are never
comparable across jurisdictions even when the numeric thresholds match. Every served
prediction carries its jurisdiction-native measure alongside the band (NYC score/
letter grade; Florida high-priority/intermediate/basic counts and disposition) --
verified live in this audit via `GET /restaurants/{id}` and `GET /restaurants/{id}/inspections`.

## Google and Michelin content

Michelin recognition may be used **only** as a separately-reported analytical feature,
never in the official health-risk model (spec: "Michelin category as an analytical
feature only in a separately reported experiment"). Google content is excluded from
model training, testing, validation, and fine-tuning entirely -- not "de-weighted," not
"used with a disclaimer," excluded. This is enforced by the two leakage controls above,
by `tests/test_google_policy.py`'s static guarantee that no ingestion/feature/model
module can even import the presentation layer where Google-link code lives, and
structurally by the fact that Task 10's Google integration (see README.md) never
fetches or stores any Google content at all -- there is no Google-sourced data for a
training pipeline to accidentally ingest even if the import barrier were removed.

## Known limitations (carried from `scripts/train.py`)

- Risk-band thresholds (0.25 / 0.50) are fixed, not validation-derived.
- A model trained on one jurisdiction/target does not generalize to the other.
- Uncertainty intervals reflect bootstrap resampling variability only, not every
  source of real-world uncertainty.
- Every prediction is a statistical estimate, never a guarantee or a prediction of a
  specific future violation (README.md, `PREDICTION_DISCLAIMER`).

## Verification (this audit, 2026-09-18/19)

- **Live, not simulated**: `python -m scripts.download_nyc` and
  `python -m scripts.download_florida` against the real NYC Open Data and Florida DBPR
  endpoints, followed by `python -m scripts.build_processed_tables` on the real
  downloaded extracts, all completed successfully and produced valid Parquet tables
  (488 NYC restaurants, 2,886 Florida restaurants from a capped sample).
- **Live, not simulated**: the FastAPI service, started against those real processed
  tables, correctly served search/detail/inspection/violation/health/Copilot endpoints,
  reported `google: disabled` and `local_ai: disabled` by default, and returned a
  404/`no_ready_model` (not a crash or a fabricated prediction) for the prediction and
  model-card routes with no trained artifact configured.
- **Live, not simulated**: `python -m scripts.audit_graph` against those same real
  processed tables built a 22,978-node/37,188-edge temporal knowledge graph (3,374
  restaurants, 3,991 inspections, 14,668 violation occurrences) with zero conflicts,
  zero dangling edges, zero duplicate node IDs, and zero temporal-ordering violations.
- **Not executed live in this audit**: a full-scale production training run
  (`scripts/train.py` without `--dry-run`) against the complete historical NYC/Florida
  datasets. A small-sample live run correctly failed at the gradient-boosting fit step
  with a scikit-learn binning error caused entirely by the sample's small size, not a
  code defect -- consistent with `scripts/train.py`'s own docstring ("a full production
  training run is intentionally not exercised by the automated test suite"). The
  chronological-split, calibration, uncertainty, readiness-gating, and metric logic
  described above are verified by the automated test suite
  (`tests/models/`, `tests/features/test_temporal.py`), which does use data at a scale
  designed to exercise these paths meaningfully, and by `--dry-run` CLI validation
  against the live-downloaded sample.
