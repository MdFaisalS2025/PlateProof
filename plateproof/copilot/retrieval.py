"""Jurisdiction-filtered lexical retrieval over the guidance corpus.

Retrieval only ever *proposes candidate evidence* -- it never authorizes a
factual claim by itself. Two different kinds of "match" live here, and
they authorize two different (and only two different) claim types:

* :func:`passages_for_codes` / :func:`passages_for_topics` are exact,
  curated-field matches (``applicable_violation_codes`` / ``topics``, both
  set by a human reviewer when the passage was added to the corpus --
  never inferred). Only *these* results may back a
  ``GUIDANCE_FOR_CODE``/``GUIDANCE_FOR_TOPIC`` claim
  (``plateproof.copilot.claims``).
* :func:`search_by_topic_or_text` is TF-IDF lexical similarity. It is
  candidate-discovery tooling only (e.g. for a human curator deciding
  whether to add a topic/code tag to a passage) -- ``CopilotService``
  never calls it to authorize an answer, and no claim builder accepts its
  output. A high similarity score is a ranking signal, never proof that a
  passage's guidance applies to a specific code or topic.

Every function here additionally filters by jurisdiction *before*
matching or scoring -- an NYC query never even considers a Florida
passage, regardless of code/topic/text overlap.
"""

from __future__ import annotations

from collections.abc import Sequence

from plateproof.copilot.corpus import CorpusPassage, CorpusStore, PermittedUse
from plateproof.copilot.models import Citation, EvidenceType, RetrievedEvidenceItem

MIN_RELEVANCE_SCORE = 0.05


def _passage_citation(passage: CorpusPassage) -> Citation:
    return Citation(
        citation_id=passage.passage_id,
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
        title=passage.title,
        url=passage.source_url,
        excerpt=passage.text,
        jurisdiction=passage.jurisdiction,
        # The corpus is a static, human-reviewed snapshot -- its own
        # access_date is the most accurate "as of" signal available for a
        # guidance-passage citation (there is no dynamic score/record date
        # the way there is for a restaurant-record citation).
        as_of_date=passage.access_date,
        superseded=passage.superseded,
        issuing_authority=passage.authority,
        section_locator=passage.section_locator,
        access_date=passage.access_date,
        effective_date=passage.effective_date,
        revision_date=passage.revision_date,
    )


def _eligible_passages(
    store: CorpusStore, jurisdiction: str, *, required_use: PermittedUse | None
) -> tuple[CorpusPassage, ...]:
    passages = store.passages_for_jurisdiction(jurisdiction)
    if required_use is None:
        return passages
    return tuple(p for p in passages if required_use in p.permitted_uses)


def passages_for_codes(
    store: CorpusStore,
    jurisdiction: str,
    violation_codes: Sequence[str],
    *,
    required_use: PermittedUse | None = None,
) -> tuple[RetrievedEvidenceItem, ...]:
    """Exact, curated ``applicable_violation_codes`` match only -- never a
    text/topic similarity result. When ``required_use`` is given, a
    passage is eligible only if it also declares that use (e.g. a
    preparation checklist must pass
    ``required_use=PermittedUse.PREPARATION_ACTION``). Deterministic
    tie-break by ``citation_id``."""
    codes = {c.upper() for c in violation_codes}
    candidates: list[RetrievedEvidenceItem] = []
    for passage in _eligible_passages(store, jurisdiction, required_use=required_use):
        matched = codes & {c.upper() for c in passage.applicable_violation_codes}
        if matched:
            candidates.append(
                RetrievedEvidenceItem(
                    citation=_passage_citation(passage),
                    relevance_score=1.0,
                    matched_terms=tuple(sorted(matched)),
                    evidence_type=EvidenceType.GUIDANCE_PASSAGE,
                )
            )
    candidates.sort(key=lambda item: item.citation.citation_id)
    return tuple(candidates)


def passages_for_topics(
    store: CorpusStore,
    jurisdiction: str,
    topics: Sequence[str],
    *,
    required_use: PermittedUse | None = None,
) -> tuple[RetrievedEvidenceItem, ...]:
    """Exact, curated ``topics`` match only -- a deterministic set
    intersection, never TF-IDF/text similarity. ``topics`` here must
    already be a controlled vocabulary value the caller derived
    deterministically from documented data (e.g. a Florida violation's own
    ``severity`` classification), never freeform text. Deterministic
    tie-break by ``citation_id``."""
    wanted = {t.lower() for t in topics}
    candidates: list[RetrievedEvidenceItem] = []
    for passage in _eligible_passages(store, jurisdiction, required_use=required_use):
        matched = wanted & {t.lower() for t in passage.topics}
        if matched:
            candidates.append(
                RetrievedEvidenceItem(
                    citation=_passage_citation(passage),
                    relevance_score=1.0,
                    matched_terms=tuple(sorted(matched)),
                    evidence_type=EvidenceType.GUIDANCE_PASSAGE,
                )
            )
    candidates.sort(key=lambda item: item.citation.citation_id)
    return tuple(candidates)


def search_by_topic_or_text(
    store: CorpusStore, jurisdiction: str, query_text: str, *, top_k: int = 5
) -> tuple[RetrievedEvidenceItem, ...]:
    """TF-IDF cosine-similarity ranking over jurisdiction-filtered
    passages only. Deterministic tie-break: ``(score desc, passage_id
    asc)``. Below :data:`MIN_RELEVANCE_SCORE`, a passage is not returned
    as a candidate at all.

    Candidate-discovery tooling only: no claim builder accepts this
    function's output, and ``CopilotService`` never calls it while
    constructing an answer -- see the module docstring."""
    passages = store.passages_for_jurisdiction(jurisdiction)
    if not passages:
        return ()

    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = [p.text for p in passages]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform([*documents, query_text])
    except ValueError:
        # Empty vocabulary (e.g. an all-stopword/empty query) -- no
        # candidates rather than a crash.
        return ()

    query_vector = matrix[-1]
    doc_vectors = matrix[:-1]
    similarities = (doc_vectors @ query_vector.T).toarray().ravel()

    scored = sorted(
        zip(passages, similarities, strict=True), key=lambda pair: (-pair[1], pair[0].passage_id)
    )
    results: list[RetrievedEvidenceItem] = []
    for passage, score in scored[:top_k]:
        if score < MIN_RELEVANCE_SCORE:
            continue
        results.append(
            RetrievedEvidenceItem(
                citation=_passage_citation(passage),
                relevance_score=float(score),
                matched_terms=(),
                evidence_type=EvidenceType.GUIDANCE_PASSAGE,
            )
        )
    return tuple(results)
