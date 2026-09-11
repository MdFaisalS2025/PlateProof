# Michelin seed data (optional)

PlateProof's Michelin recognition feature is **optional**: the complete
application runs correctly with no Michelin data at all
(`load_michelin_seed(None)` or a header-only file both return an empty
result, no error).

## Source policy

PlateProof does not scrape Michelin Guide pages, hidden endpoints, search
systems, or APIs; does not use Michelin's paid API tiers; and does not use
third-party "Michelin restaurant" datasets whose own documented origin is
scraping the Guide site (most publicly circulated ones are — verify before
ever considering one). The only supported source is this optional,
hand-maintained CSV, populated by a maintainer transcribing **facts** (name,
address, distinction, guide edition/year, and a citation URL) from a source
they have personally verified is appropriate to reuse.

**This repository intentionally ships this file empty** — `michelin_seed_template.csv`
contains only the header row. No real Michelin restaurant data is committed
by Task 4. Populating it with real rows is a separate, manual, human
decision, made outside of this codebase's automated changes.

### Responsibility

- **Dataset maintainers are responsible for confirming their own reuse rights**
  before adding any row. This module records what a maintainer tells it
  (`source_url`, `source_title`, `source_publisher`, `source_access_date`,
  `source_license_note`, `provenance_confidence`); it cannot and does not
  verify that a cited source's license actually permits the use made of it.
  Recording a `source_license_note` is not a legal opinion — it is a
  transparency record of what the maintainer believes the source permits.
- Only **factual metadata** belongs in this file: name, location, distinction,
  dates, and a citation. Never add review text, editorial descriptions,
  photographs, or logos — those are Michelin's (or a cited source's)
  expressive content, not fact, and are out of scope regardless of the
  source's license.
- If a cited source itself requires attribution (e.g. Wikipedia content is
  CC BY-SA 4.0 and expects attribution), preserve that attribution in
  `source_publisher`/`source_title`/`source_license_note` — do not strip it.
- Nothing produced from this file may state or imply endorsement by Michelin,
  by Wikimedia, or by any other cited source. PlateProof is independent.
- A Michelin distinction is contextual metadata. It must never be presented
  as, merged with, or used to infer health-inspection results; it does not
  imply food safety (see `CLAUDE.md`).

### Recommended reference (not automated)

When a maintainer wants a well-documented, openly licensed reference to
transcribe from, Wikipedia's *"List of Michelin-starred restaurants in New
York City"* and *"List of Michelin-starred restaurants in Florida"* articles
(CC BY-SA 4.0) are a reasonable starting point for name/distinction/year
history. **This is a citation recommendation for a human, not something
PlateProof fetches or parses automatically.** Wikipedia's own tables
typically lack street-level address and coordinates; those fields may be left
blank (`address_as_published`, `postal_code`, `latitude`, `longitude` are all
optional) rather than guessed.

## Columns

See `plateproof/ingestion/michelin.py` for the authoritative contract
(`MICHELIN_SEED_REQUIRED_COLUMNS`, `MICHELIN_SEED_OPTIONAL_COLUMNS`,
`MichelinRestaurant`, `MichelinDistinctionEvent`). Required columns:
`name_as_published, city, region, jurisdiction_candidate, distinction,
guide_name, guide_year, source_url, source_title, source_publisher,
source_access_date, source_license_note, provenance_confidence`. A row
failing validation (unsupported `distinction`, `provenance_confidence` other
than `verified_primary`/`verified_secondary`, missing provenance fields, or a
malformed `guide_year`/date) is rejected and counted in the load report — it
is never silently coerced or dropped without a trace.

One row = one **edition-specific distinction** for one restaurant (e.g. a
2024 one-star and a 2025 two-star for the same restaurant are two rows, and a
star plus a Green Star in the same year are two more rows). The loader
derives a stable restaurant identity from each row's normalized
name/address/postal/jurisdiction and attaches every matching row's
distinction as a separate, dated fact — nothing here computes or stores a
mutable "current distinction."

## Using a populated file

```python
from plateproof.ingestion.michelin import load_michelin_seed

result = load_michelin_seed("path/to/your/populated_seed.csv")
result.restaurants  # list[MichelinRestaurant]
result.distinction_events  # list[MichelinDistinctionEvent]
result.report  # MichelinLoadReport -- counts, rejections, conflicts
```

Pass `None` (or omit the path entirely from a caller that doesn't have one
configured) to run with Michelin disabled.
