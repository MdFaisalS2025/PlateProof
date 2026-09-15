"""Restaurant-identity corroboration for owner-uploaded documents.

Deterministic only: reuses the exact same mechanical name normalization the
matching pipeline already uses for entity resolution
(``plateproof.matching.normalize.normalize_name``) -- never
``distinctive_name_tokens`` (blocking-only, not for equality), and no
fuzzy/ML scoring. A document either normalizes to the same name as the
restaurant the owner selected, or it doesn't; there is no partial-credit
corroboration in Task 9A.
"""

from __future__ import annotations

from plateproof.matching.normalize import normalize_name


def corroborate_restaurant_identity(
    candidate_name_text: str | None, expected_restaurant_name: str
) -> bool:
    if not candidate_name_text:
        return False
    return normalize_name(candidate_name_text) == normalize_name(expected_restaurant_name)
