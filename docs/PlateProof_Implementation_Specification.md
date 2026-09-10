# PlateProof Implementation Specification

> **For Claude Code:** Build this project in the order specified. Use test-driven development, keep the core application free of paid services, and do not proceed to a later phase until the current phase passes its acceptance tests.

## Goal

Build PlateProof, an AI-powered restaurant intelligence platform that combines official health-inspection records, violation histories, Michelin recognition, and carefully governed Google business information. The product serves diners with evidence-based restaurant context and serves restaurant owners with inspection-readiness guidance.

The first release covers New York City and Florida. It must preserve the meaning of each jurisdiction's inspection system instead of inventing a universal grade.

## Product identity

- Product: **PlateProof**
- Tagline: **Evidence behind every plate**
- Consumer score: **PlateProof Index**
- Owner assistant: **PlateProof Copilot**
- Prediction feature: **Inspection Risk Forecast**
- Review feature: **Customer Signal Monitor**
- Michelin comparison: **Prestige and Safety Analysis**

PlateProof is independent. The interface and documentation must not imply endorsement by Michelin, Google, NYC, Florida DBPR, or any health department.

## Core product decisions

1. Official inspection records are the source of truth for health-compliance claims.
2. NYC and Florida receive separate prediction models and outcome definitions.
3. The public interface must label predictions as estimates, not inspection results.
4. Michelin recognition is contextual metadata, not evidence of food safety.
5. Google Maps content must not train, test, validate, or fine-tune PlateProof models.
6. The MVP must work without a Google API key, a paid LLM API, or a credit card.
7. Advanced AI must improve a real decision: understanding risk, finding recurring violations, or preparing corrective action.

## User experiences

### Diner experience

A diner searches for a restaurant and sees its Michelin recognition, official inspection history, jurisdiction-native violation measures, a confidence-aware PlateProof risk band, and links to the original government and Google records. The diner can compare restaurants but must be warned that inspection systems differ across jurisdictions.

### Owner experience

An owner claims or selects a restaurant, reviews recurring violations, uploads an inspection report, receives a cited preparation checklist, and can optionally connect a verified Google Business Profile. Owner information remains private by default.

### Research experience

A researcher compares whether Michelin-recognized restaurants exhibit different subsequent inspection outcomes after controlling for cuisine, geography, price level, and previous inspection history. Results must be framed as associations rather than causal effects.

## Jurisdiction-specific outcomes

| Jurisdiction | Native measure | Primary prediction target | Secondary outcomes |
|---|---|---|---|
| NYC | Violation points and letter grade | Next initial inspection has 14 or more points | 28 or more points, critical violation, score change |
| Florida | High Priority, Intermediate, and Basic violations plus disposition | Next routine inspection has at least one High Priority violation or requires follow-up | Emergency closure, total violations, repeated violation class |

Never convert Florida results into NYC letter grades. The shared UI may show calibrated Low, Moderate, and High risk bands, but it must always display the underlying jurisdiction-native measure.

## Data sources

### Required

- NYC DOHMH Restaurant Inspection Results: `https://data.cityofnewyork.us/d/43nn-pn8j`
- Florida DBPR Restaurants and Food Service Public Records: `https://www2.myfloridalicense.com/hotels-restaurants/public-records/`
- Michelin Guide restaurant information: use a legally obtained, documented dataset or a small curated research file. Record the source date and terms. Do not scrape pages that prohibit automated extraction.

### Optional Google integration

- Public mode: use Places API only for live display of permitted fields with required Google attribution.
- Owner mode: use Google Business Profile API only after OAuth authorization for a verified location.
- The application must run when Google integration is disabled.
- Store stable Google Place IDs when permitted. Do not bulk-cache review text or copy Google content into the training dataset.
- Reviews from Places API are limited and relevance-selected; do not present them as a representative sample.

## Recommended technology stack

- Python 3.12
- FastAPI for the service layer
- Streamlit for the first web interface
- DuckDB and Parquet for local analytical storage
- PostgreSQL with pgvector only when multi-user deployment becomes necessary
- Polars or pandas for transformations
- scikit-learn and XGBoost for tabular models
- NetworkX for the MVP graph; PyTorch Geometric only after the baseline proves value
- sentence-transformers for local embeddings
- Ollama with a small open-source instruct model for optional local generation
- Docling or PyMuPDF plus PaddleOCR for inspection-report extraction
- Pydantic for typed contracts
- pytest, Ruff, and mypy for quality checks
- MLflow local file store for experiment tracking

## Repository structure

```text
plateproof/
  CLAUDE.md
  README.md
  pyproject.toml
  .env.example
  configs/
    base.yaml
  data/
    raw/.gitkeep
    interim/.gitkeep
    processed/.gitkeep
  plateproof/
    core/config.py
    core/contracts.py
    ingestion/nyc.py
    ingestion/florida.py
    ingestion/michelin.py
    ingestion/google_places.py
    matching/normalize.py
    matching/entity_resolution.py
    features/inspection_events.py
    features/temporal.py
    models/nyc_risk.py
    models/florida_risk.py
    models/calibration.py
    models/risk_bands.py
    graph/builder.py
    copilot/retrieval.py
    copilot/service.py
    documents/extractor.py
    api/main.py
    api/routes/restaurants.py
    api/routes/predictions.py
    app/Home.py
    app/pages/1_Restaurant_Search.py
    app/pages/2_Inspection_History.py
    app/pages/3_Owner_Copilot.py
    app/pages/4_Model_Card.py
  tests/
    ingestion/
    matching/
    features/
    models/
    graph/
    copilot/
    api/
  models/.gitkeep
  reports/model_card.md
  scripts/download_nyc.py
  scripts/download_florida.py
  scripts/train.py
  scripts/run_app.py
```

## Core data contracts

```python
from datetime import date, datetime
from typing import Literal
from pydantic import BaseModel

Jurisdiction = Literal["nyc", "florida"]

class Restaurant(BaseModel):
    restaurant_id: str
    jurisdiction: Jurisdiction
    source_id: str
    name: str
    normalized_name: str
    address: str
    city: str
    region: str
    postal_code: str | None
    latitude: float | None
    longitude: float | None
    cuisine: str | None

class InspectionEvent(BaseModel):
    inspection_id: str
    restaurant_id: str
    jurisdiction: Jurisdiction
    inspection_date: date
    inspection_type: str
    disposition: str | None
    score: float | None
    grade: str | None
    high_priority_count: int | None
    intermediate_count: int | None
    basic_count: int | None

class ViolationEvent(BaseModel):
    inspection_id: str
    violation_code: str
    description: str
    severity: str
    corrected_on_site: bool | None

class RiskPrediction(BaseModel):
    restaurant_id: str
    jurisdiction: Jurisdiction
    generated_at: datetime
    target_name: str
    probability: float
    lower_bound: float
    upper_bound: float
    risk_band: Literal["low", "moderate", "high", "insufficient_history"]
    model_version: str
    top_factors: list[str]
```

## Data processing rules

### NYC

NYC records can contain one row per violation, so aggregate them to one inspection event using CAMIS, inspection date, and inspection type. Remove the `1900-01-01` placeholder date. Deduplicate violations within an inspection. Preserve the published score rather than summing duplicated rows. Only use information available before the target inspection.

### Florida

Normalize license number as the restaurant source identifier. Preserve the original inspection type and disposition. Map violation classifications only to `high_priority`, `intermediate`, `basic`, and `other`; retain the original code and text. Do not invent a numerical grade.

### Entity resolution

Matching must use normalized name, standardized address, postal code, and geographic distance. Produce a score and evidence for every match. Auto-accept only scores at or above `0.92`, reject below `0.75`, and place the middle range in a review queue. A Michelin or Google match must never overwrite an official restaurant identity.

## Modeling design

### Feature availability

For inspection event `t`, calculate features using events strictly earlier than `t`. Use `groupby(restaurant_id).shift(1)` before rolling calculations. Prohibited features include the target inspection's score, grade, violations, action, disposition, and adjudication results.

### Features

- Previous score or severity counts
- Rolling mean, maximum, variance, and trend
- Days since previous inspection
- Prior critical or High Priority violation counts
- Repeated violation-code counts
- Inspection-history depth
- Cuisine, jurisdiction-native geography, and season
- Trailing neighborhood and cuisine rates calculated without future data
- Michelin category as an analytical feature only in a separately reported experiment
- Google rating information excluded from the official health-risk model

### Validation

Use chronological train, validation, and test periods. Compare a prevalence baseline, regularized logistic regression, and gradient-boosted trees. Optimize area under the precision-recall curve. Report ROC AUC, Brier score, calibration plots, top-decile recall, and subgroup results. Calibrate probabilities on the validation period only.

The production model must beat the prevalence baseline and logistic baseline on the untouched test period. If it does not, ship the simpler logistic model.

### Uncertainty

Add split-conformal or bootstrap prediction intervals. Return `insufficient_history` when required history is unavailable. Never display a probability without its model date and uncertainty range.

## PlateProof Index

The PlateProof Index is a presentation layer, not a new health grade. It contains separately labeled dimensions:

- Official inspection history
- Predicted next-inspection risk
- Operational consistency
- Michelin recognition
- Public popularity information

Do not collapse all dimensions into a single unexplained number. If a composite is implemented, show the formula, allow users to inspect every component, and never label it an official safety score.

## Advanced AI modules

### PlateProof Copilot

Use retrieval-augmented generation over official inspection rules, violation definitions, corrective guidance, and the selected restaurant's structured history. Retrieval results must carry source URL, title, jurisdiction, effective date, and section. Generated answers must cite their evidence and refuse unsupported conclusions.

### Temporal knowledge graph

Create nodes for restaurants, inspections, violations, cuisines, locations, and Michelin distinctions. Create timestamped relationships. Begin with graph queries and graph-derived features. Add a graph neural network only if it beats the tabular baseline in an out-of-time ablation study.

### Multimodal inspection reader

Allow an owner to upload a PDF or image. Extract text locally, identify violation codes and deadlines, and require owner confirmation before saving structured results. Never treat extracted content as official data until confirmed.

### Customer Signal Monitor

This is owner-only. With authorization, classify incoming reviews into operational topics and urgency. Customer statements are perceptions, not verified violations. Keep this module isolated from the government-inspection prediction model and recheck Google policies before production release.

## API interfaces

```text
GET  /health
GET  /restaurants?query=&jurisdiction=&michelin_category=
GET  /restaurants/{restaurant_id}
GET  /restaurants/{restaurant_id}/inspections
GET  /restaurants/{restaurant_id}/violations
GET  /restaurants/{restaurant_id}/prediction
POST /owners/documents/extract
POST /copilot/query
GET  /models/{jurisdiction}/card
```

The prediction endpoint must return `RiskPrediction`. The Copilot endpoint must return an answer, citations, retrieved passages, and a safety status.

## Privacy and safety

- Do not expose owner OAuth tokens to the browser or logs.
- Encrypt stored credentials and provide disconnect and deletion controls.
- Strip reviewer names and profile information from analytics unless display is explicitly required and permitted.
- Do not publish restaurant blacklists.
- Do not make illness, negligence, or legal-liability claims from reviews or predictions.
- Display links to the original inspection source and a correction/contact path.
- Record model version, source snapshot, and generation timestamp for every prediction.

## Implementation plan

### Task 1 Project foundation

**Files:** create `pyproject.toml`, `CLAUDE.md`, `README.md`, `.env.example`, `plateproof/core/config.py`, and `tests/test_smoke.py`.

- [ ] Write `test_import_plateproof` and `test_settings_default_to_google_disabled`.
- [ ] Run `pytest tests/test_smoke.py -v` and confirm failure before implementation.
- [ ] Add typed settings with `GOOGLE_INTEGRATION_ENABLED=false` by default.
- [ ] Configure Ruff, mypy, pytest, and coverage in `pyproject.toml`.
- [ ] Run `ruff check .`, `mypy plateproof`, and `pytest`.
- [ ] Commit with `chore: establish PlateProof project foundation`.

### Task 2 NYC ingestion and event construction

**Files:** create `plateproof/ingestion/nyc.py`, `plateproof/features/inspection_events.py`, `scripts/download_nyc.py`, and tests under `tests/ingestion/`.

- [ ] Test placeholder-date removal, inspection-level aggregation, violation deduplication, and preservation of published scores using a five-row fixture.
- [ ] Implement `load_nyc_raw(path) -> pl.DataFrame` and `build_nyc_inspection_events(raw) -> tuple[pl.DataFrame, pl.DataFrame]`.
- [ ] Assert uniqueness of `(restaurant_id, inspection_date, inspection_type)` in the event table.
- [ ] Run `pytest tests/ingestion/test_nyc.py -v` and commit `feat: add NYC inspection pipeline`.

### Task 3 Florida ingestion and normalization

**Files:** create `plateproof/ingestion/florida.py`, `scripts/download_florida.py`, and `tests/ingestion/test_florida.py`.

- [ ] Test license normalization, severity mapping, disposition retention, and unknown violation handling.
- [ ] Implement `load_florida_extracts(paths) -> pl.DataFrame` and `build_florida_inspection_events(raw) -> tuple[pl.DataFrame, pl.DataFrame]`.
- [ ] Assert nonnegative severity counts and retain unmodified source fields for audit.
- [ ] Run the ingestion tests and commit `feat: add Florida inspection pipeline`.

### Task 4 Michelin ingestion and entity resolution

**Files:** create `plateproof/ingestion/michelin.py`, `plateproof/matching/normalize.py`, `plateproof/matching/entity_resolution.py`, and matching tests.

- [ ] Test punctuation, accents, abbreviations, suite numbers, and nearby restaurants with similar names.
- [ ] Implement `match_candidate(restaurant, candidate) -> MatchEvidence` with name, address, postal, and distance components.
- [ ] Implement thresholds `accept >= 0.92`, `review 0.75-0.9199`, and `reject < 0.75`.
- [ ] Export the review queue without silently accepting ambiguous matches.
- [ ] Run matching tests and commit `feat: add auditable Michelin entity resolution`.

### Task 5 Leakage-safe temporal features

**Files:** create `plateproof/features/temporal.py` and `tests/features/test_temporal.py`.

- [ ] Create a fixture where a future inspection contains an extreme score.
- [ ] Prove that features for earlier rows do not change when that future score changes.
- [ ] Implement `build_temporal_features(events, violations, cutoff_date)` using shifted histories.
- [ ] Add a feature-availability report and fail when prohibited target columns enter the matrix.
- [ ] Run tests and commit `feat: create leakage-safe temporal features`.

### Task 6 Separate jurisdiction models

**Files:** create both risk-model modules, calibration, risk bands, `scripts/train.py`, and model tests.

- [ ] Test target construction independently for NYC and Florida.
- [ ] Test chronological splits and probability bounds.
- [ ] Train prevalence, logistic, and boosted-tree candidates.
- [ ] Calibrate on validation only and evaluate once on the newest test period.
- [ ] Save model, schema, metrics, data snapshot, and model card together.
- [ ] Run model tests and commit `feat: train calibrated jurisdiction risk models`.

### Task 7 API and consumer interface

**Files:** create FastAPI routes, Streamlit pages, and API tests.

- [ ] Test restaurant search, inspection history, violations, predictions, missing restaurants, and insufficient history.
- [ ] Implement the API contracts exactly as defined above.
- [ ] Build pages for search, restaurant detail, history, risk explanation, and model cards.
- [ ] Display jurisdiction-native measures beside every risk band.
- [ ] Run API and browser smoke tests and commit `feat: deliver PlateProof consumer experience`.

### Task 8 Knowledge graph and Copilot

**Files:** create graph builder, retrieval, Copilot service, tests, and a versioned official-guidance corpus manifest.

- [ ] Test graph node uniqueness, timestamped edges, jurisdiction filtering, citation completeness, and unsupported-answer refusal.
- [ ] Implement deterministic retrieval before adding generation.
- [ ] Add a local-model adapter behind an interface; keep a citation-only fallback.
- [ ] Require every factual answer to cite retrieved official evidence.
- [ ] Run Copilot tests and commit `feat: add evidence-grounded PlateProof Copilot`.

### Task 9 Multimodal owner workflow

**Files:** create `plateproof/documents/extractor.py`, upload route, owner page, and document fixtures.

- [ ] Test text PDF, scanned image, unreadable document, ambiguous code, and owner correction.
- [ ] Implement local extraction into a draft `InspectionEvent` and `ViolationEvent` set.
- [ ] Require confirmation before persistence and record extraction confidence.
- [ ] Run tests and commit `feat: add owner inspection document extraction`.

### Task 10 Optional Google integration

**Files:** create `plateproof/ingestion/google_places.py`, owner OAuth adapter, attribution component, and policy tests.

- [ ] Test that the complete application works with Google disabled.
- [ ] Test field masks, attribution, source links, token redaction, and cache restrictions.
- [ ] Implement live public display separately from verified-owner review access.
- [ ] Prevent Google review text from entering model-training tables with a schema-level test.
- [ ] Run the full suite and commit `feat: add policy-aware Google integration`.

### Task 11 Final verification and deployment

**Files:** update `README.md`, `reports/model_card.md`, CI workflow, and deployment configuration.

- [ ] Run `ruff check .` and `ruff format --check .`.
- [ ] Run `mypy plateproof` with no errors.
- [ ] Run `pytest --cov=plateproof --cov-report=term-missing` with at least 85% coverage for core ingestion, feature, and prediction modules.
- [ ] Execute a clean download-to-prediction pipeline for both jurisdictions.
- [ ] Verify no secrets, raw Google reviews, or prohibited cached content are committed.
- [ ] Deploy the Streamlit MVP and verify search, history, prediction, and source links.
- [ ] Commit `docs: finalize PlateProof release documentation`.

## MVP acceptance criteria

- NYC and Florida pipelines build reproducible inspection and violation tables.
- Each jurisdiction has a separate leakage-safe, calibrated prediction model.
- The interface shows original measures, prediction uncertainty, model version, and source links.
- Michelin records are matched with auditable confidence and ambiguous matches require review.
- PlateProof Copilot answers from official evidence and returns citations.
- The system runs without Google or paid AI services.
- No Google review content appears in model training or evaluation data.
- Automated tests cover aggregation, temporal leakage, entity matching, prediction contracts, and citations.

## Explicitly deferred features

- Native mobile applications
- Restaurant reservations or payments
- Automated public accusations or blacklists
- Nationwide inspection normalization
- A graph neural network without baseline evidence
- Automated replies to Google reviews
- Production multi-tenant billing

## Sources and policies

- NYC Open Data, DOHMH New York City Restaurant Inspection Results: https://data.cityofnewyork.us/d/43nn-pn8j
- NYC Health, The Inspection Process: https://www.nyc.gov/site/doh/business/food-operators/the-inspection-process.page
- Florida DBPR, Restaurants and Food Service Public Records: https://www2.myfloridalicense.com/hotels-restaurants/public-records/
- Florida DBPR, Food and Lodging Inspections: https://www2.myfloridalicense.com/hotels-restaurants/inspections/
- Michelin, Florida expansion announcement: https://www.michelin.com/en/publications/products-and-services/the-michelin-guide-welcomes-new-cities-in-florida-expansion
- Google Maps Platform Terms: https://cloud.google.com/maps-platform/terms
- Google Places API policies: https://developers.google.com/maps/documentation/places/web-service/policies
- Google Business Profile review API: https://developers.google.com/my-business/reference/rest/v4/accounts.locations.reviews/list

## Initial Claude Code instruction

Copy the following message into Claude Code after placing this specification in the repository:

```text
Read PlateProof_Claude_Code_Build_Specification.md completely. Begin with Task 1 only. Use test-driven development: write the specified failing tests, show that they fail for the expected reason, implement the minimum required code, and run all Task 1 checks. Do not implement later tasks. Preserve the global constraints, particularly separate NYC and Florida semantics, optional Google integration, and the prohibition against using Google Maps content for model training. At the end, summarize created files, test results, assumptions, and any decision that requires approval before Task 2.
```


