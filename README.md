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

## Status

Planning scaffold plus Task 1 (project foundation) and Task 2 (NYC ingestion and
event construction). Florida ingestion, modeling, API, and interface code have
not been implemented.
