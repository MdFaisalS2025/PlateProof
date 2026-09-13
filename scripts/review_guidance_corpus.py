"""Human corpus-review CLI for the Task 8A official-guidance corpus.

Prints one row per passage -- document, jurisdiction, title, source URL
(with official-domain-allowlist pass/fail), access date, checksum-verified
status, superseded flag, topics, and violation-code associations -- plus a
final PASS/FAIL line. This is the sign-off gate a human runs before
trusting the committed corpus; it performs no network access and never
mutates the manifest. Never used by the running web application -- see
``plateproof.copilot.corpus.load_corpus`` for that.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from plateproof.copilot.corpus import audit_corpus, load_corpus, validate_official_url

_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent / "data" / "reference" / "guidance" / "manifest.json"
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a human-reviewable table of every passage in the "
        "official-guidance corpus and report PASS/FAIL."
    )
    parser.add_argument("--manifest-path", default=str(_DEFAULT_MANIFEST))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.manifest_path)

    report = audit_corpus(manifest_path)
    result = load_corpus(manifest_path)

    print(f"documents examined: {report.documents_examined}")
    print(f"passages examined:  {report.passages_examined}")
    print()

    if result.store is not None:
        header = (
            f"{'document_id':<40} {'jurisdiction':<10} {'url_ok':<7} "
            f"{'topics':<30} {'codes':<20} {'superseded':<10}"
        )
        print(header)
        print("-" * len(header))
        for passage in result.store.passages:
            print(
                f"{passage.document_id:<40} {passage.jurisdiction:<10} "
                f"{str(validate_official_url(passage.source_url)):<7} "
                f"{','.join(passage.topics):<30} "
                f"{','.join(passage.applicable_violation_codes) or '-':<20} "
                f"{str(passage.superseded):<10}"
            )
        print()

    if report.findings:
        print("findings:")
        for finding in report.findings:
            print(
                f"  [{finding.document_id or '-'}/{finding.passage_id or '-'}] "
                f"{finding.field}: {finding.problem}"
            )
        print()

    passed = report.passed and result.outcome.value == "loaded"
    print(f"{'PASS' if passed else 'FAIL'}: guidance corpus review")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
