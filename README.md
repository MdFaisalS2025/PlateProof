# PlateProof

PlateProof is an independent restaurant-intelligence project combining official health-inspection records, violation histories, Michelin recognition, and carefully governed Google business information.

The first release covers New York City and Florida. It preserves each jurisdiction's native inspection system instead of inventing a universal health grade.

**This is a solo, from-scratch engineering project built task-by-task under strict
test-driven development** (every feature below has a failing-test-first commit
history), not a forked template or a tutorial walkthrough.

## The problem

Diners and restaurant owners have no single, honest place to see a restaurant's
*official* health-inspection history alongside a calibrated, uncertainty-aware
forecast of what its next inspection might look like. Existing restaurant apps
either ignore inspection data entirely or flatten NYC's point-based grading and
Florida's violation-severity system into a single misleading "score." PlateProof's
core bet: keep each jurisdiction's own inspection system intact, never invent a
universal grade, and never let a prediction be mistaken for an official result.

## What I built

- Two independently trained, leakage-safe, calibrated risk models (NYC and Florida),
  never a shared model across jurisdictions, with a hard-enforced feature allowlist
  and token-based guard that provably keeps the target and third-party content
  (Michelin, Google) out of training data.
- A read-only FastAPI service and a Streamlit MVP over the same DuckDB/Parquet
  serving layer -- restaurant search, inspection/violation history, jurisdiction-native
  risk bands with uncertainty intervals, and source links back to the original
  government datasets.
- **PlateProof Copilot** -- a deterministic, citation-grounded Q&A layer over a
  temporal knowledge graph and a reviewed official-guidance corpus. Every factual
  sentence is built by a fixed claim builder from retrieved evidence; nothing is
  freely generated. An optional local Ollama adapter only ever classifies *which*
  of 9 closed intents a question maps to -- it never writes the answer itself.
- A process-isolated document-extraction pipeline letting an owner upload their own
  inspection PDF/image and review a machine-assisted, evidence-grounded extraction
  of it -- OCR/PDF parsing runs only inside short-lived, killable worker processes
  behind a hand-rolled non-pickle IPC protocol, specifically to keep a
  malicious/malformed upload from ever reaching the trusted parent process.
- A researched, honestly-scoped optional Google Maps integration: after verifying
  that Google's Places API requires a billing account (a credit card) before any
  call at all, it ships only a zero-key, zero-network outbound search link instead.
- A full release-readiness audit (`reports/model_card.md`, `docs/deployment.md`):
  live-verified download-to-serving pipeline on real government data, a researched
  conclusion that no genuinely free/no-card host can run this app's full worker-process
  architecture publicly, and an honest "local-only release" status rather than a
  claimed deployment that doesn't hold up.

## Architecture

```
NYC / Florida open data  ---->  ingestion + entity resolution  ---->  DuckDB/Parquet
   (download scripts)          (dedup, normalize, Michelin match)     processed tables
                                                                            |
                                                                            v
                                              +-----------------------------------------+
                                              |        plateproof.serving (shared)       |
                                              |  repository + service layer, no business |
                                              |  logic duplicated between entry points   |
                                              +-------------------+---------------------+
                                                                   |
                                   +-------------------------------+-------------------------------+
                                   |                                                               |
                          FastAPI service                                                Streamlit MVP
                    (read-only HTTP routes)                                        (thin UI over the same layer)
                                   |                                                               |
                    +--------------+--------------+                                +---------------+
                    |                             |                                |
        Copilot (deterministic,         Document extraction:                 Model card / risk-band
        graph + guidance corpus         isolated worker-process               pages, source links,
        retrieval, citation-only)       pool, non-pickle IPC,                 optional Google-link
                                        OCR/PDFium, never the                 attribution
                                        request-handling process
```

Offline-only, administrator-run stages (never touched by a live request): data
download, `build_processed_tables`, model training/calibration
(`plateproof/models/`), and `score_predictions`. The running web application only
ever reads precomputed Parquet tables and sanitized model metadata -- it never
deserializes or executes a model artifact itself.

## Technology stack

Python 3.12 · FastAPI + Uvicorn · Streamlit · DuckDB over local Parquet · Polars ·
Pydantic · scikit-learn (logistic regression + histogram gradient boosting) ·
NetworkX (temporal knowledge graph) · pypdfium2 + Pillow + RapidOCR/ONNX Runtime
(document extraction, worker-process only) · pytest/Ruff/mypy/coverage for
verification · optional local Ollama for intent classification only. Every
dependency is free and runs locally -- no paid API is required for any feature.

## Demo

There is no hosted public demo (see "Release verification and deployment" below
for exactly why, and what it would take to change that). To try it locally: follow
"Local development" and "Running the API and the Streamlit MVP" below, then open
`http://localhost:8501` for the Streamlit UI or `http://localhost:8000/docs` for
the interactive API docs.

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

## Owner document extraction (Task 9)

`POST /owners/documents/extract` and the "Document Reader" Streamlit page
let a restaurant owner upload their own PDF/PNG/JPEG inspection document
and review a machine-assisted, evidence-grounded extraction of it. Both
entry points call the exact same
`plateproof.documents.service.extract_document(...)` function -- neither
ever parses a document itself. All real parsing (PDFium, Pillow, RapidOCR)
happens only inside a short-lived, killable, non-pickle-protocol worker
process this function submits to; see `plateproof/documents/worker/` for
the process-isolation boundary and `plateproof/documents/limits.py` for
every configured ceiling. Nothing uploaded is ever persisted server-side,
used to train or update a model, or written to the graph/Copilot corpus --
a completed extraction is a `record_status="user_submitted"` artifact the
owner can download as JSON, never an official inspection record.

### Deployment requirements (read before exposing either entry point)

**A reverse proxy or ASGI body-size middleware is REQUIRED in front of the
FastAPI service.** Uvicorn has no built-in request-body-size limit of its
own (its `--h11-max-incomplete-event-size` flag bounds only the request
line and headers of an *incomplete* HTTP event, never the body) --
without a gateway (nginx `client_max_body_size`, a cloud load balancer's
request-size cap, or middleware such as `content-size-limit-asgi`) in
front of it, a bare `uvicorn` process cannot reject an oversized request
body before Starlette's own multipart parser has already accepted (and,
for a large enough file, spooled to disk) it. PlateProof's own code
(`plateproof/api/routes/documents.py`) still enforces the exact
`documents_max_upload_bytes` ceiling with a bounded read afterward, but
that check necessarily runs only once the file has already reached the
application. **This gateway limit must also cover chunked-transfer
requests (no `Content-Length` header)**, since PlateProof's own bounded
read is the only in-app enforcement either way.

The Streamlit page has its own, coarser, megabyte-granularity limit
(`.streamlit/config.toml`'s `server.maxUploadSize`, kept at the smallest
whole-megabyte value that is still >= `documents_max_upload_bytes`) as its
own first-layer defense; it is not a substitute for a real reverse-proxy
limit if the Streamlit app is itself directly internet-facing.

**Each application process owns its own document worker pool.** A
`multiprocessing` pool cannot be meaningfully shared across unrelated
parent processes, so `documents_worker_pool_size` (default 2, ceiling 4)
is a **per-process** limit, not a machine-wide one. Size deployment
resources for the **combined** worst-case worker count:
`(API processes + Streamlit processes) x documents_worker_pool_size`
(multiplied again by replica count, if running more than one of either).
There is no pure-Python way to enforce a true machine-wide ceiling across
independent processes -- **OS/container memory and CPU limits are a
required operational complement**, not optional hardening: each worker
process renders full-page rasters (bounded by `documents_max_pixels_per_page`)
and may load a RapidOCR/ONNX Runtime model, so provision each container/
VM with enough memory for its own worst-case concurrent worker count,
and CPU headroom for that many simultaneous OCR/PDFium calls.

**Preview validation is structural, not a full decode.** Task 9A's
preview validator (`plateproof/documents/worker/pool.py`) checks a
generated preview PNG's entire container structure -- signature, chunk
CRCs and ordering, declared dimensions -- using only `struct`/`zlib`; it
does **not** fully decompress the image's pixel data. The Streamlit page
renders previews via a `data:image/png;base64,...` URI passed to
`st.image()`, which Streamlit's own code returns unmodified without ever
invoking Pillow -- the actual pixel decode happens only in the viewer's
own browser, never inside the Streamlit process itself. This is a
deliberate, documented trust boundary (the standard one for any web app
displaying an image), not a claim that the preview bytes are exhaustively
proven safe to decode.

**Windows note:** Streamlit's script runner replaces `sys.modules["__main__"]`
with a bare module wrapping the *currently executing page script* on every
rerun. On Windows, `multiprocessing`'s spawn bootstrap would otherwise try
to reconstruct a spawned worker's `__main__` by re-executing that same
page file, which crashes (no `ScriptRunContext` outside a real page run).
`plateproof/documents/worker/pool.py` works around this by pointing
`__main__` at its own, real, properly-specced module for the duration of
`Process.start()` only, restoring whatever was there immediately
afterward -- verified directly against a real `AppTest` run.

## Google integration (optional, link-only)

Google's Places API requires a billing-enabled Cloud project -- and therefore a credit
card on file -- before a single request can be made, even to stay within its free usage
thresholds (verified directly against
[Google's own "Get started" docs](https://developers.google.com/maps/get-started) and
[Cloud Billing's payment-method requirements](https://docs.cloud.google.com/billing/docs/how-to/payment-methods)).
The Business Profile API has no card requirement, but is gated behind a discretionary
Google approval process this project has no realistic path through, and is scoped to an
individually-OAuth'd, Google-verified owner -- something PlateProof has no
authentication system to support yet. Both are therefore out of scope; see
`docs/PlateProof_Implementation_Specification.md`'s Task 10 entry and the planning
record for the full comparison.

What PlateProof implements instead, set `PLATEPROOF_GOOGLE_INTEGRATION_ENABLED=true` to
enable: a plain
[Google Maps URL](https://developers.google.com/maps/documentation/urls/get-started)
("You don't need a Google API key to use Maps URLs.") built from a restaurant's own
already-known name/address, shown as a "View on Google Maps" link on the restaurant
search page. **This is a constructed search query, not a Google-verified match** --
PlateProof's server never contacts Google; opening the link sends the query to Google
from the *user's own browser*. No API key, billing account, OAuth flow, Place ID lookup,
or cached Google content is ever involved. The application runs identically with this
disabled (the default) -- see `plateproof/serving/display.py`'s
`google_maps_search_link`/`GOOGLE_SEARCH_LINK_ATTRIBUTION` and
`plateproof/api/routes/restaurants.py`.

## Release verification and deployment

Task 11's release-readiness audit is recorded in full in `reports/model_card.md`
(project-level modeling methodology and what was verified live vs. by the automated
suite) and `docs/deployment.md` (what actually runs today -- a local release, verified
live end-to-end -- and the hosting research behind why no public deployment was made).
In short: **PlateProof is a verified local-only release.** No cloud account was
created, no paid service was activated, and nothing was deployed publicly -- current
Google Maps Platform terms and Hugging Face Spaces policy both require a paid plan
before hosting compute at all, and the one genuinely free option found (Streamlit
Community Cloud) cannot run the FastAPI service and is a poor fit for Task 9's
memory/process-isolation needs; see `docs/deployment.md` for the full comparison and
what it would take to change this decision. `.github/workflows/ci.yml` runs the full
quality suite (Ruff, mypy, pytest with coverage) on every push, using GitHub Actions'
own no-credit-card-required free tier. A `Dockerfile` is included for anyone who wants
to run PlateProof in a container on their own machine -- see `docs/deployment.md`
before doing more than that with it.

## Status

Task 1 (project foundation), Task 2 (NYC ingestion), Task 3 (Florida
ingestion), Task 4 (optional Michelin ingestion and auditable entity
resolution), Task 5 (leakage-safe temporal features), Task 6 (calibrated
jurisdiction risk models), Task 7 (FastAPI service and Streamlit MVP),
Task 8 (deterministic, evidence-grounded Copilot with optional local
intent assistance), Task 9 (owner document extraction: an isolated
worker-process core, the `POST /owners/documents/extract` route, and the
Streamlit Document Reader page), Task 10 (optional, link-only Google
Maps integration -- no API key, billing account, or OAuth), and Task 11
(release-readiness audit, CI, and local-only deployment documentation)
are implemented.
