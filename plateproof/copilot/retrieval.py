"""Jurisdiction-filtered lexical retrieval over the guidance corpus.

Retrieval only ever *proposes candidate evidence* -- it never authorizes a
factual claim by itself. A claim builder (``plateproof.copilot.claims``)
decides whether a retrieved passage is actually used, and the
``AUTHORIZED_EVIDENCE_TYPES`` matrix decides whether the resulting claim
type may cite it at all. A TF-IDF score is a ranking signal only: always
exposed for audit, never presented to the user as a probability, and never
sufficient on its own to authorize an answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from plateproof.copilot.corpus import CorpusPassage, CorpusStore
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
        as_of_date=None,
        superseded=passage.superseded,
    )


def passages_for_codes(
    store: CorpusStore, jurisdiction: str, violation_codes: Sequence[str]
) -> tuple[RetrievedEvidenceItem, ...]:
    """Deterministic, code-first retrieval: a passage whose
    ``applicable_violation_codes`` intersects the restaurant's documented
    codes is always preferred over a generic keyword match -- no TF-IDF
    needed for this exact-match case. Deterministic tie-break by
    ``citation_id``."""
    codes = {c.upper() for c in violation_codes}
    candidates: list[RetrievedEvidenceItem] = []
    for passage in store.passages_for_jurisdiction(jurisdiction):
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


def search_by_topic_or_text(
    store: CorpusStore, jurisdiction: str, query_text: str, *, top_k: int = 5
) -> tuple[RetrievedEvidenceItem, ...]:
    """TF-IDF cosine-similarity ranking over jurisdiction-filtered
    passages only. Deterministic tie-break: ``(score desc, passage_id
    asc)``. Below :data:`MIN_RELEVANCE_SCORE`, a passage is not returned
    as a candidate at all."""
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
