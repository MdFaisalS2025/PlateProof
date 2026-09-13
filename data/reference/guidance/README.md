# Official guidance corpus (Task 8A)

This is the small, versioned, human-reviewable corpus of official
inspection-guidance material the PlateProof Copilot cites from. It is
deliberately small: three documents, five passages, all currently official
public records.

## Source policy

Every document here is a short, section-scoped extract from a current
official government page, browsed and verified against the live site at
`access_date` -- never reconstructed from memory, never a search-result
summary, never an SEO article or AI-generated paraphrase. Only the
following are eligible sources:

- NYC Department of Health and Mental Hygiene (nyc.gov)
- NYC Open Data documentation (data.cityofnewyork.us)
- Florida Department of Business and Professional Regulation
  (myfloridalicense.com)
- FDA Food Code or other directly relevant federal guidance (fda.gov) --
  included only where clearly identified as federal model guidance, never
  presented as an NYC- or Florida-specific enforcement rule

No Google review text, Michelin editorial content, blog post, or other
non-official material may ever be added here.

## What's in this corpus right now

| document_id | jurisdiction | authority | passages |
|---|---|---|---|
| `nyc-doh-inspection-process` | nyc | NYC DOHMH | 1 |
| `fl-dbpr-violation-classifications` | florida | Florida DBPR | 3 |
| `fda-food-code-overview` | federal | FDA | 1 |

Every passage's `applicable_violation_codes` is currently empty: no
document reviewed so far supports a direct, defensible code-level mapping.
The Florida violation-classification passages are linked to the broader
topics `high_priority`/`intermediate`/`basic` instead -- matching the
severity categories `plateproof.ingestion.florida` already assigns -- which
is an honest, supportable topic-level linkage rather than an invented
code-level one. A future addition may add a code mapping only when a
reviewed source states one explicitly.

## Provenance fields

Every document entry in `manifest.json` records: `document_id`,
`jurisdiction`, `title`, `authority`, `source_url`, `access_date`,
`effective_date`/`revision_date` (when the source publishes one),
`superseded`/`superseded_by`, a `provenance_note`, the `file` it lives in,
and a `sha256` of that file's exact bytes. Every passage inside a document
file additionally records its own `passage_id`, `section_locator`, exact
`text`, a `sha256` of that exact text, `applicable_violation_codes`, and
`topics`.

## Adding a document

1. Browse the live official page yourself; do not transcribe from memory
   or a search snippet.
2. Copy a short, clearly-bounded extract -- never an entire manual or PDF.
3. Compute the passage's own `sha256` over its exact UTF-8 text, and the
   document file's `sha256` over its exact bytes, and record both.
4. Run `python -m scripts.review_guidance_corpus` (see below) and confirm
   it reports every field before committing.
5. Only link a passage to a violation code when the source itself states
   that connection explicitly -- otherwise link it to a topic and say so.

## Corpus-review command

```powershell
python -m scripts.review_guidance_corpus
```

Prints one row per passage -- document, jurisdiction, title, URL (with
official-domain-allowlist pass/fail), access date, checksum-verified
(bool), superseded, topics, and code associations -- plus a final
PASS/FAIL line. This is the human sign-off gate; it performs no network
access and never mutates the manifest.

## Trust boundary

Loading this corpus at application startup (`plateproof.copilot.corpus`)
never makes a network request -- it only reads this committed, reviewed
snapshot. Updating the corpus is always a separate, manual, reviewed step,
exactly like `data/reference/michelin_seed_template.csv`; nothing in the
running application refreshes it automatically.
