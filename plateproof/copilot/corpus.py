"""Secure, atomic loader for the Task 8A official-guidance corpus.

Carries forward the exact trust-boundary pattern
``plateproof.serving.model_registry_service`` was hardened to use for
Task 7 model metadata: bounded reads (never read-then-check), manifest
entries validated as safe relative paths before any file operation,
symlink/containment checks, checksum verification, and narrow exception
handling around only the specific untrusted filesystem/parsing operation
that can fail. The two subsystems intentionally do not share code --
each is independently reviewable, exactly like
``plateproof.models.training.verify_artifact_checksums`` (offline, full
artifact) and ``model_registry_service`` (web-facing, metadata-only)
already don't share code with each other.

Loading is atomic and fail-closed: any single invalid document or passage
anywhere in the manifest rejects the *entire* corpus
(:data:`CorpusLoadOutcome.REJECTED_INVALID`) -- never a partially loaded,
silently smaller corpus. Authoritative safety guidance must never be
silently incomplete. The separate, exhaustive :func:`audit_corpus` exists
only for a human reviewer (``scripts/review_guidance_corpus.py``); it is
never used to serve a real request.

No network access ever happens here -- this only reads an already-
committed, already-reviewed snapshot. Updating the corpus is always a
separate, manual, human-reviewed step (see
``data/reference/guidance/README.md``), exactly like the Michelin seed
CSV.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

GuidanceJurisdiction = Literal["nyc", "florida", "federal"]

# A single corpus file has no legitimate reason to be large -- generous
# headroom over real usage, bounding how much untrusted content this
# reader ever materializes in memory for one file.
_MAX_FILE_BYTES = 2_000_000
_MAX_DOCUMENTS = 200
_MAX_PASSAGES_PER_DOCUMENT = 100
_MAX_TOTAL_CORPUS_BYTES = 20_000_000
_MAX_PASSAGE_LENGTH = 2000

# Reviewed, human-maintained list of official domains this corpus may ever
# cite. Updated only as part of the same human-reviewed PR that adds a new
# document -- see data/reference/guidance/README.md.
OFFICIAL_DOMAIN_ALLOWLIST: frozenset[str] = frozenset(
    {
        "www.nyc.gov",
        "data.cityofnewyork.us",
        "www2.myfloridalicense.com",
        "www.myfloridalicense.com",
        "www.fda.gov",
    }
)


def validate_official_url(url: object) -> bool:
    """Never raises. True only for an ``https`` URL with no embedded
    credentials, no control characters, and a hostname exactly in
    :data:`OFFICIAL_DOMAIN_ALLOWLIST`."""
    if not isinstance(url, str) or not url or "\x00" in url:
        return False
    if any(ord(char) < 0x20 for char in url):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    if "@" in parsed.netloc:  # userinfo (embedded credentials) present
        return False
    host = parsed.hostname
    if host is None or host not in OFFICIAL_DOMAIN_ALLOWLIST:
        return False
    return True


class GuidanceManifestDocumentEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{2,63}$")
    jurisdiction: GuidanceJurisdiction
    title: str = Field(min_length=1, max_length=200)
    authority: str = Field(min_length=1, max_length=200)
    source_url: str
    access_date: date
    effective_date: date | None = None
    revision_date: date | None = None
    superseded: bool = False
    superseded_by: str | None = None
    provenance_note: str = Field(min_length=1, max_length=500)
    file: str = Field(pattern=r"^documents/[a-z0-9][a-z0-9\-]{2,63}\.json$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GuidanceManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    generated_at: datetime
    documents: tuple[GuidanceManifestDocumentEntry, ...]


class GuidancePassageRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    passage_id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]{2,63}#[a-z0-9][a-z0-9\-]{0,63}$")
    section_locator: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=_MAX_PASSAGE_LENGTH)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    applicable_violation_codes: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()


class GuidanceDocumentFile(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    jurisdiction: GuidanceJurisdiction
    passages: tuple[GuidancePassageRecord, ...]


@dataclass(frozen=True)
class CorpusPassage:
    """One retrievable, fully-verified passage -- everything a citation
    needs, already flattened from its parent document entry."""

    passage_id: str
    document_id: str
    jurisdiction: GuidanceJurisdiction
    title: str
    authority: str
    source_url: str
    section_locator: str
    text: str
    applicable_violation_codes: tuple[str, ...]
    topics: tuple[str, ...]
    superseded: bool


class CorpusStore:
    """Immutable, fully-validated corpus. Constructed only by
    :func:`load_corpus` -- if you have one of these, every passage in it
    already passed checksum, path-safety, and URL-policy validation."""

    def __init__(self, passages: tuple[CorpusPassage, ...]) -> None:
        self._passages = passages

    @property
    def passages(self) -> tuple[CorpusPassage, ...]:
        return self._passages

    def passages_for_jurisdiction(self, jurisdiction: str) -> tuple[CorpusPassage, ...]:
        """Current (non-superseded) passages for one jurisdiction only --
        the normal retrieval path. Superseded documents are retained in
        the store for audit purposes but never returned here."""
        return tuple(
            p for p in self._passages if p.jurisdiction == jurisdiction and not p.superseded
        )


class CorpusLoadOutcome(StrEnum):
    LOADED = "loaded"
    UNAVAILABLE_NOT_CONFIGURED = "unavailable_not_configured"
    REJECTED_INVALID = "rejected_invalid"


@dataclass(frozen=True)
class CorpusLoadResult:
    outcome: CorpusLoadOutcome
    store: CorpusStore | None
    rejection_reason: str | None


def _is_safe_relative_file(relative: object) -> bool:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        return False
    if "\\" in relative:
        return False
    if PureWindowsPath(relative).drive or PureWindowsPath(relative).root:
        return False
    posix = PurePosixPath(relative)
    if posix.is_absolute():
        return False
    if ".." in posix.parts or "." in posix.parts:
        return False
    return True


def _safe_corpus_path(corpus_dir: Path, relative: str) -> Path | None:
    """Returns the path to ``relative`` beneath ``corpus_dir`` only if it
    is provably safe to open: not absolute/traversing/UNC-qualified, not a
    symlink or reparse point, a regular file, and resolves to a path
    actually contained in ``corpus_dir``. Returns ``None`` for anything
    else -- including a file that disappears or changes type between
    checks -- rather than ever raising."""
    if not _is_safe_relative_file(relative):
        return None
    candidate = corpus_dir / relative
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved_dir = corpus_dir.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved_candidate.is_relative_to(resolved_dir):
        return None
    return candidate


def _read_bounded_bytes(path: Path, max_bytes: int) -> tuple[bytes | None, str | None]:
    """Reads at most ``max_bytes + 1`` bytes -- never the whole file first
    and only checking its length afterward. Returns ``(None, reason)`` on
    any expected failure; never raises."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        return None, "file could not be read"
    if len(raw) > max_bytes:
        return None, "file exceeds the maximum allowed size"
    return raw, None


def _load_json_bytes(raw: bytes) -> tuple[Any, str | None]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "is not valid UTF-8"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        return None, "is not valid JSON"


def load_corpus(manifest_path: Path | None) -> CorpusLoadResult:
    """Loads and fully validates the guidance corpus rooted at
    ``manifest_path``. ``manifest_path=None`` means the corpus is not
    configured -- a normal, non-error condition, exactly like a Task 7
    model artifact path left unset. Any other failure is atomic and
    fail-closed: the whole corpus is rejected, never served partially."""
    if manifest_path is None:
        return CorpusLoadResult(CorpusLoadOutcome.UNAVAILABLE_NOT_CONFIGURED, None, None)

    corpus_dir = manifest_path.parent

    manifest_raw, error = _read_bounded_bytes(manifest_path, _MAX_FILE_BYTES)
    if error is not None:
        return CorpusLoadResult(CorpusLoadOutcome.REJECTED_INVALID, None, f"manifest {error}")
    assert manifest_raw is not None
    manifest_json, error = _load_json_bytes(manifest_raw)
    if error is not None:
        return CorpusLoadResult(CorpusLoadOutcome.REJECTED_INVALID, None, f"manifest {error}")
    try:
        manifest = GuidanceManifest.model_validate(manifest_json)
    except ValidationError:
        return CorpusLoadResult(
            CorpusLoadOutcome.REJECTED_INVALID, None, "manifest schema is invalid"
        )

    if len(manifest.documents) > _MAX_DOCUMENTS:
        return CorpusLoadResult(
            CorpusLoadOutcome.REJECTED_INVALID, None, "manifest exceeds the maximum document count"
        )

    seen_document_ids: set[str] = set()
    seen_document_ids_normalized: set[str] = set()
    seen_passage_ids: set[str] = set()
    seen_passage_ids_normalized: set[str] = set()
    total_bytes = len(manifest_raw)
    passages: list[CorpusPassage] = []

    for entry in manifest.documents:
        normalized_doc_id = entry.document_id.strip().casefold()
        already_seen = (
            entry.document_id in seen_document_ids
            or normalized_doc_id in seen_document_ids_normalized
        )
        if already_seen:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID, None, "duplicate document id in manifest"
            )
        seen_document_ids.add(entry.document_id)
        seen_document_ids_normalized.add(normalized_doc_id)

        if not validate_official_url(entry.source_url):
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID,
                None,
                "a document's source_url failed policy validation",
            )

        safe_path = _safe_corpus_path(corpus_dir, entry.file)
        if safe_path is None:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID,
                None,
                "a document's file path failed safety validation",
            )

        doc_raw, error = _read_bounded_bytes(safe_path, _MAX_FILE_BYTES)
        if error is not None:
            return CorpusLoadResult(CorpusLoadOutcome.REJECTED_INVALID, None, f"document {error}")
        assert doc_raw is not None

        total_bytes += len(doc_raw)
        if total_bytes > _MAX_TOTAL_CORPUS_BYTES:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID, None, "corpus exceeds the maximum total size"
            )

        if hashlib.sha256(doc_raw).hexdigest() != entry.sha256:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID, None, "document checksum verification failed"
            )

        doc_json, error = _load_json_bytes(doc_raw)
        if error is not None:
            return CorpusLoadResult(CorpusLoadOutcome.REJECTED_INVALID, None, f"document {error}")
        try:
            doc = GuidanceDocumentFile.model_validate(doc_json)
        except ValidationError:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID, None, "document schema is invalid"
            )

        if doc.document_id != entry.document_id or doc.jurisdiction != entry.jurisdiction:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID,
                None,
                "document identity mismatch between manifest and file",
            )
        if len(doc.passages) > _MAX_PASSAGES_PER_DOCUMENT:
            return CorpusLoadResult(
                CorpusLoadOutcome.REJECTED_INVALID,
                None,
                "a document exceeds the maximum passage count",
            )

        for passage in doc.passages:
            normalized_passage_id = passage.passage_id.strip().casefold()
            if (
                passage.passage_id in seen_passage_ids
                or normalized_passage_id in seen_passage_ids_normalized
            ):
                return CorpusLoadResult(
                    CorpusLoadOutcome.REJECTED_INVALID, None, "duplicate passage id in corpus"
                )
            seen_passage_ids.add(passage.passage_id)
            seen_passage_ids_normalized.add(normalized_passage_id)

            if hashlib.sha256(passage.text.encode("utf-8")).hexdigest() != passage.sha256:
                return CorpusLoadResult(
                    CorpusLoadOutcome.REJECTED_INVALID,
                    None,
                    "passage checksum verification failed",
                )

            passages.append(
                CorpusPassage(
                    passage_id=passage.passage_id,
                    document_id=entry.document_id,
                    jurisdiction=entry.jurisdiction,
                    title=entry.title,
                    authority=entry.authority,
                    source_url=entry.source_url,
                    section_locator=passage.section_locator,
                    text=passage.text,
                    applicable_violation_codes=passage.applicable_violation_codes,
                    topics=passage.topics,
                    superseded=entry.superseded,
                )
            )

    return CorpusLoadResult(CorpusLoadOutcome.LOADED, CorpusStore(tuple(passages)), None)


@dataclass(frozen=True)
class CorpusAuditFinding:
    document_id: str | None
    passage_id: str | None
    field: str
    problem: str


@dataclass(frozen=True)
class CorpusAuditReport:
    """Exhaustive, best-effort report for the human corpus-review tool --
    collects every problem found, unlike :func:`load_corpus`, which stops
    at the first failure and refuses atomically. Never used to serve a
    real request."""

    documents_examined: int
    passages_examined: int
    findings: tuple[CorpusAuditFinding, ...]
    passed: bool


def audit_corpus(manifest_path: Path) -> CorpusAuditReport:
    findings: list[CorpusAuditFinding] = []
    documents_examined = 0
    passages_examined = 0

    manifest_raw, error = _read_bounded_bytes(manifest_path, _MAX_FILE_BYTES)
    if error is not None or manifest_raw is None:
        findings.append(
            CorpusAuditFinding(None, None, "manifest", error or "manifest could not be read")
        )
        return CorpusAuditReport(0, 0, tuple(findings), passed=False)

    manifest_json, error = _load_json_bytes(manifest_raw)
    if error is not None:
        findings.append(CorpusAuditFinding(None, None, "manifest", error))
        return CorpusAuditReport(0, 0, tuple(findings), passed=False)
    try:
        manifest = GuidanceManifest.model_validate(manifest_json)
    except ValidationError as exc:
        findings.append(
            CorpusAuditFinding(
                None, None, "manifest", f"schema invalid: {exc.error_count()} error(s)"
            )
        )
        return CorpusAuditReport(0, 0, tuple(findings), passed=False)

    corpus_dir = manifest_path.parent
    for entry in manifest.documents:
        documents_examined += 1
        if not validate_official_url(entry.source_url):
            findings.append(
                CorpusAuditFinding(
                    entry.document_id, None, "source_url", "failed policy validation"
                )
            )

        safe_path = _safe_corpus_path(corpus_dir, entry.file)
        if safe_path is None:
            findings.append(
                CorpusAuditFinding(entry.document_id, None, "file", "failed path safety validation")
            )
            continue

        doc_raw, error = _read_bounded_bytes(safe_path, _MAX_FILE_BYTES)
        if error is not None or doc_raw is None:
            findings.append(
                CorpusAuditFinding(entry.document_id, None, "file", error or "unreadable")
            )
            continue

        if hashlib.sha256(doc_raw).hexdigest() != entry.sha256:
            findings.append(
                CorpusAuditFinding(entry.document_id, None, "sha256", "checksum mismatch")
            )

        doc_json, error = _load_json_bytes(doc_raw)
        if error is not None:
            findings.append(CorpusAuditFinding(entry.document_id, None, "file", error))
            continue
        try:
            doc = GuidanceDocumentFile.model_validate(doc_json)
        except ValidationError as exc:
            findings.append(
                CorpusAuditFinding(
                    entry.document_id, None, "file", f"schema invalid: {exc.error_count()} error(s)"
                )
            )
            continue

        for passage in doc.passages:
            passages_examined += 1
            if hashlib.sha256(passage.text.encode("utf-8")).hexdigest() != passage.sha256:
                findings.append(
                    CorpusAuditFinding(
                        entry.document_id, passage.passage_id, "sha256", "checksum mismatch"
                    )
                )
            if not passage.applicable_violation_codes and not passage.topics:
                findings.append(
                    CorpusAuditFinding(
                        entry.document_id,
                        passage.passage_id,
                        "topics",
                        "no topic or violation-code association recorded",
                    )
                )

    return CorpusAuditReport(
        documents_examined, passages_examined, tuple(findings), passed=not findings
    )
