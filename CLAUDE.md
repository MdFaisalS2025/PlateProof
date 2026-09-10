# PlateProof Project Instructions

## Required reading

Before planning or changing files, read:

- `README.md`
- `docs/PlateProof_Implementation_Specification.md`
- `CLAUDE_CODE_START_PROMPT.md`

The specification is the design authority. If the repository and specification disagree, stop and explain the conflict before changing code.

## Workflow

1. Plan before implementation and obtain user approval for the plan.
2. Work on one numbered task from the specification at a time.
3. Use test-driven development for application behavior: write a failing test, run it and confirm the expected failure, add the minimum implementation, then rerun the test.
4. Run the relevant focused tests after each change and the full quality suite at task boundaries.
5. Do not silently broaden scope or implement deferred features.
6. Summarize changed files, commands run, results, assumptions, and unresolved decisions after each task.

## Product rules

- Keep NYC violation points and Florida violation classifications semantically separate.
- Never convert Florida results into NYC A, B, or C grades.
- Official government records are the source of truth for health-compliance claims.
- Michelin recognition is contextual metadata and does not imply food safety.
- The public product must not imply affiliation with Michelin, Google, NYC, Florida DBPR, or a health department.
- Do not train, test, validate, or fine-tune models using Google Maps content.
- Google integration is optional; the complete MVP must run with it disabled.
- Do not publish restaurant blacklists or infer illness, negligence, or legal liability.
- Every prediction must include model version, generation time, source snapshot, uncertainty, and the jurisdiction-native measure.

## Technical defaults

- Python 3.12
- FastAPI service layer
- Streamlit MVP interface
- DuckDB and Parquet for local analytics
- Polars for data transformations unless a library requires pandas
- Pydantic for typed boundaries
- scikit-learn baseline models before XGBoost
- pytest, Ruff, mypy, and coverage for verification
- Local/open-source AI adapters with deterministic fallbacks

## Quality commands

Once Task 1 creates the environment, use:

```powershell
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy plateproof
```

Do not claim a task is complete without fresh successful command output.

## Data hygiene

- Do not commit raw datasets, credentials, OAuth tokens, model binaries, or Google review content.
- Keep reproducible download scripts and source metadata.
- Preserve original source identifiers and fields in auditable staging tables.
- All temporal features must use information strictly earlier than their prediction event.

## Git discipline

- Keep commits small and scoped to one specification task.
- Use descriptive commit messages such as `feat: add NYC inspection pipeline`.
- Never rewrite or discard user changes without explicit approval.
