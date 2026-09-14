# PlateProof

PlateProof is an independent restaurant-intelligence project combining official health-inspection records, violation histories, Michelin recognition, and carefully governed Google business information.

The first release covers New York City and Florida. It preserves each jurisdiction's native inspection system instead of inventing a universal health grade.

## Start here

1. Read `CLAUDE.md`.
2. Read `docs/PlateProof_Implementation_Specification.md` completely.
3. Paste `CLAUDE_CODE_START_PROMPT.md` into Claude Code.
4. Ask Claude Code to plan first and wait for approval before implementation.

## Non-negotiable constraints

- The core MVP must work without paid APIs or a credit card.
- NYC and Florida use separate targets and models.
- Michelin recognition is context, not evidence of food safety.
- Google Maps content must not enter model training or evaluation.
- Predictions are estimates and must show uncertainty and source dates.

## Local development

Requires Python 3.12 (`requires-python = ">=3.12,<3.14"`).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Quality checks:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy plateproof scripts
```

## Data downloads

The NYC DOHMH inspection extract is fetched on demand into a timestamped
snapshot directory (`raw.csv` + `metadata.json` + a `_SUCCESS` marker). Raw
government data is never committed.

```powershell
python -m scripts.download_nyc --output data/raw/nyc
```

An optional free Socrata app token (`PLATEPROOF_SODA_APP_TOKEN`) raises rate
limits; the download works without one. Use `--max-rows` to cap the download and
`--where` to pass a SoQL filter.

Florida DBPR extracts (current-fiscal-year CSV and recent statewide XLSX
archives) download the same way via `python -m scripts.download_florida`; see
`plateproof/ingestion/florida.py` for the supported fiscal years and formats.

## Michelin recognition (optional)

Michelin data is entirely optional -- the application runs correctly with none
configured. PlateProof does not scrape Michelin Guide pages, use Michelin's
paid API, or use third-party Michelin datasets scraped from the Guide site.
The only supported source is a small, hand-maintained, provenance-rich CSV;
see `data/reference/README.md` for the source policy and column contract, and
`plateproof/ingestion/michelin.py` for the loader. The repository ships only a
header-only template (`data/reference/michelin_seed_template.csv`) -- no real
Michelin restaurant data is committed. A Michelin distinction is contextual
metadata and never implies food safety or affects health-inspection results.

## Running the API and the Streamlit MVP

Task 7 adds a read-only FastAPI service and a Streamlit interface, both thin
adapters over a shared `plateproof.serving` service/repository layer backed
by DuckDB over local Parquet tables. Neither ever downloads data, trains a
model, or deserializes a model artifact inside a request/page load.

**Trust boundary:** offline administrative scoring
(`plateproof/serving/scoring.py`, run via `scripts/score_predictions.py`) is
the only code permitted to call `load_artifact`/`joblib.load` against a
locally configured, administrator-trusted artifact path. The running web
application (every FastAPI route and every Streamlit page) consumes only
sanitized metadata (`plateproof.serving.model_registry_service
.ModelMetadataReader`, which parses just a fixed allowlist of JSON/Markdown
files and never opens `.joblib`) and precomputed prediction rows -- it never
deserializes or executes a model artifact, even if `load_artifact` is
broken, patched, or removed.

Build the processed tables from already-downloaded extracts (never
downloads anything itself):

```powershell
python -m scripts.build_processed_tables --nyc-events data\raw\nyc\raw.csv --florida-events data\raw\florida\*.csv --output data\processed
```

Score predictions offline (also never trains or downloads; requires an
explicit `--as-of-date`, never "today" by default):

```powershell
python -m scripts.score_predictions --jurisdiction nyc --artifact-path models\nyc\nyc_next_initial_score_ge_14\v1 --events data\processed\inspection_events.parquet --violations data\processed\violation_events.parquet --as-of-date 2026-01-01 --output data\processed
```

Run the API and/or the Streamlit UI locally (each works independently):

```powershell
python -m scripts.run_app --target api
python -m scripts.run_app --target streamlit
```

Configure local paths via `PLATEPROOF_`-prefixed environment variables (see
`plateproof/core/config.py`): `PROCESSED_DATA_DIR`, `NYC_MODEL_ARTIFACT_PATH`,
`FLORIDA_MODEL_ARTIFACT_PATH`, `PREDICTION_TABLE_PATH`,
`EXPOSE_NON_READY_MODEL_CARDS` (default `false`). All optional -- the API
and UI run correctly with none configured, showing clear "unavailable"
states instead of failing.

## PlateProof Copilot

The Copilot (`POST /copilot/query`, and the "Owner Copilot" Streamlit page)
answers restaurant-scoped questions about documented inspection history and
reviewed official guidance. It is deterministic by design: **every factual
sentence in an answer is built by a fixed claim builder and a fixed
rendering template from authorized evidence (the knowledge graph and the
reviewed guidance corpus) -- never freely generated text.** Nothing in
PlateProof ever asks a local or remote model to write the answer itself.

Supported questions map to one of 9 closed intents: latest inspection
summary, recurring violations, full violation history, inspection trend,
official guidance for documented codes, a preparation checklist from
official guidance, forecast explanation, restaurant identity, and Michelin
context. A question outside this set is honestly refused, never guessed.

**Optional local intent assistance.** When a free-text question is
ambiguous or unrecognized, PlateProof can optionally ask a local Ollama
server (disabled by default) to propose *which one of the 9 closed
intents* the question maps to -- and nothing more. That proposal is
strictly validated (closed enum membership, a minimum confidence, no
unexpected fields) before it is ever trusted; on any rejection, timeout, or
connection failure, PlateProof falls back to its normal deterministic
refusal. The same deterministic evidence pipeline answers the question
either way -- only the response's `generator_mode`/`local_helper_status`
metadata differ. This costs no API fee: it runs entirely on your own
machine via [Ollama](https://ollama.com), and disabling it does not remove
any core Copilot functionality. To try it, install Ollama yourself, pull a
small instruction-following model of your choice, and set
`PLATEPROOF_LOCAL_LLM_ENABLED=true` and `PLATEPROOF_LOCAL_LLM_MODEL` in your
`.env` -- PlateProof never installs Ollama or downloads a model for you.
The local server may only be reached at `http://localhost:<port>` or
`http://127.0.0.1:<port>`; see `plateproof/copilot/generators/ollama.py` for
the full network-boundary hardening.

**Official guidance corpus.** Guidance citations come from a small,
manually reviewed snapshot of official government text under
`data/reference/guidance/`, checksum-verified on load
(`plateproof/copilot/corpus.py`). Run `python -m scripts.review_guidance_corpus`
to audit it. If the corpus can't be loaded (not configured, or fails
validation), guidance-dependent questions honestly report that PlateProof
doesn't currently have mapped guidance -- never that no such guidance
exists anywhere, and never a fabricated answer.

**Limitations.** PlateProof does not verify restaurant ownership. A
forecast is a statistical estimate, never a guarantee or a prediction of a
specific violation. Michelin recognition is contextual culinary
information and never implies food safety. Refusal is the default for
anything PlateProof cannot ground in documented evidence.

## Status

Task 1 (project foundation), Task 2 (NYC ingestion), Task 3 (Florida
ingestion), Task 4 (optional Michelin ingestion and auditable entity
resolution), Task 5 (leakage-safe temporal features), Task 6 (calibrated
jurisdiction risk models), Task 7 (FastAPI service and Streamlit MVP), and
Task 8 (deterministic, evidence-grounded Copilot with optional local
intent assistance) are implemented. Task 9 (owner document extraction) is
reserved but not implemented -- its route returns an explicit `501`.
