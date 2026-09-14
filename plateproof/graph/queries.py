"""Typed, read-only query helpers over a built
:class:`~plateproof.graph.store.PlateProofGraph`. Every function here is a
pure function of the graph plus its arguments -- no caching, no I/O, no
wall-clock use except where the caller explicitly passes an ``as_of_date``.

Query semantics deliberately match the rest of the project's existing
tie-break/exclusion rules rather than inventing new ones:

* "latest" always means ``(date desc, id desc)``, exactly like
  ``plateproof.serving.repository.Repository.list_inspections``.
* a Michelin distinction event with no recorded ``announced_date`` is
  always excluded from an "as of" result, exactly like
  ``plateproof.ingestion.michelin.distinction_events_as_of``.
* location/cuisine are reported as "most recently documented", never as
  full history, because the upstream ``restaurants`` table itself only
  ever carries a restaurant's latest known snapshot (see
  ``plateproof.serving.processed_tables``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from plateproof.graph.builder import violation_code_node_id
from plateproof.graph.models import EdgeType, NodeType
from plateproof.graph.store import PlateProofGraph


@dataclass(frozen=True)
class LatestInspectionResult:
    inspection_id: str
    restaurant_id: str
    inspection_date: date
    inspection_type: str | None
    score: float | None
    grade: str | None
    same_day_count: int


@dataclass(frozen=True)
class RecurringViolation:
    violation_code: str
    violation_code_norm: str
    jurisdiction: str
    occurrence_count: int
    total_cited_count: int
    inspection_dates: tuple[date, ...]
    most_recent_description: str | None
    most_recent_description_date: date | None
    severity: str | None


def restaurant_inspections(graph: PlateProofGraph, restaurant_id: str) -> list[dict[str, Any]]:
    """Every ``Inspection`` node documented for this restaurant, ordered
    ``(inspection_date desc, inspection_id desc)`` -- the same tie-break
    ``Repository.list_inspections`` uses."""
    edges = graph.out_edges_of_type(restaurant_id, EdgeType.HAS_INSPECTION)
    rows: list[dict[str, Any]] = []
    for _, inspection_id, _edge_attrs in edges:
        node = graph.node(inspection_id)
        if node is None:
            continue
        rows.append({"inspection_id": inspection_id, "restaurant_id": restaurant_id, **node})
    rows.sort(key=lambda r: (r["inspection_date"], r["inspection_id"]), reverse=True)
    return rows


def latest_inspection(graph: PlateProofGraph, restaurant_id: str) -> LatestInspectionResult | None:
    rows = restaurant_inspections(graph, restaurant_id)
    if not rows:
        return None
    latest_date = rows[0]["inspection_date"]
    same_day = [r for r in rows if r["inspection_date"] == latest_date]
    chosen = rows[0]
    return LatestInspectionResult(
        inspection_id=chosen["inspection_id"],
        restaurant_id=restaurant_id,
        inspection_date=latest_date,
        inspection_type=chosen.get("inspection_type"),
        score=chosen.get("score"),
        grade=chosen.get("grade"),
        same_day_count=len(same_day),
    )


def restaurant_violation_occurrences(
    graph: PlateProofGraph, restaurant_id: str
) -> list[dict[str, Any]]:
    """Every ``ViolationOccurrence`` documented across all of this
    restaurant's inspections, each still carrying its own dated
    description -- never a single mutable "current" description."""
    occurrences: list[dict[str, Any]] = []
    for _, inspection_id, _ in graph.out_edges_of_type(restaurant_id, EdgeType.HAS_INSPECTION):
        for _, violation_event_id, _ in graph.out_edges_of_type(
            inspection_id, EdgeType.DOCUMENTED_VIOLATION
        ):
            node = graph.node(violation_event_id)
            if node is not None:
                occurrences.append({"violation_event_id": violation_event_id, **node})
    return occurrences


def recurring_violation_codes(
    graph: PlateProofGraph, restaurant_id: str, *, min_occurrences: int = 2
) -> list[RecurringViolation]:
    """One :class:`RecurringViolation` per code cited on ``min_occurrences``
    or more distinct inspections. ``occurrence_count`` (distinct citing
    inspections) and ``total_cited_count`` (sum of each occurrence's own
    ``count``) are always reported separately -- never conflated."""
    by_code: dict[str, list[dict[str, Any]]] = {}
    for occurrence in restaurant_violation_occurrences(graph, restaurant_id):
        code_norm = occurrence["violation_code_norm"]
        by_code.setdefault(code_norm, []).append(occurrence)

    results: list[RecurringViolation] = []
    for code_norm, occurrences in by_code.items():
        if len(occurrences) < min_occurrences:
            continue
        occurrences_sorted = sorted(occurrences, key=lambda o: o["inspection_date"], reverse=True)
        most_recent = occurrences_sorted[0]
        results.append(
            RecurringViolation(
                violation_code=most_recent["violation_code"],
                violation_code_norm=code_norm,
                jurisdiction=most_recent["jurisdiction"],
                occurrence_count=len(occurrences),
                total_cited_count=sum(o.get("count") or 1 for o in occurrences),
                inspection_dates=tuple(sorted(o["inspection_date"] for o in occurrences)),
                most_recent_description=most_recent.get("violation_description"),
                most_recent_description_date=most_recent.get("inspection_date"),
                severity=most_recent.get("severity"),
            )
        )
    results.sort(key=lambda r: (-r.occurrence_count, r.violation_code_norm))
    return results


def most_recent_description_for_code(
    graph: PlateProofGraph, restaurant_id: str, violation_code_norm: str
) -> tuple[str, date] | None:
    """The description text and date of the most recent occurrence of this
    code for this restaurant. Prior, differing descriptions are never
    overwritten -- they remain queryable on their own occurrence node via
    :func:`restaurant_violation_occurrences`; this only picks the newest as
    a dated label."""
    matching = [
        o
        for o in restaurant_violation_occurrences(graph, restaurant_id)
        if o["violation_code_norm"] == violation_code_norm
    ]
    if not matching:
        return None
    newest = max(matching, key=lambda o: o["inspection_date"])
    description = newest.get("violation_description")
    if description is None:
        return None
    return description, newest["inspection_date"]


def violation_code_exists(
    graph: PlateProofGraph, jurisdiction: str, violation_code_norm: str
) -> bool:
    return graph.has_node(violation_code_node_id(jurisdiction, violation_code_norm))


def accepted_michelin_match(graph: PlateProofGraph, restaurant_id: str) -> str | None:
    """The matched ``MichelinRestaurant`` node id, or ``None`` if no
    ``decision == "accept"`` match exists -- ``review``/``reject`` rows
    never reach this far because the builder never creates an edge for
    them."""
    edges = graph.out_edges_of_type(restaurant_id, EdgeType.MATCHED_TO_MICHELIN)
    if not edges:
        return None
    return edges[0][1]


def distinctions_as_of(
    graph: PlateProofGraph, restaurant_id: str, as_of_date: date
) -> list[dict[str, Any]]:
    """Distinction events for this restaurant's accepted Michelin match
    known to have been announced on or before ``as_of_date``. An event with
    an unknown ``announced_date`` is always excluded -- never assumed to
    have been in effect -- exactly matching
    ``plateproof.ingestion.michelin.distinction_events_as_of``."""
    michelin_id = accepted_michelin_match(graph, restaurant_id)
    if michelin_id is None:
        return []
    results: list[dict[str, Any]] = []
    for _, event_id, _ in graph.out_edges_of_type(michelin_id, EdgeType.HAS_DISTINCTION_EVENT):
        node = graph.node(event_id)
        if node is None:
            continue
        announced_date = node.get("announced_date")
        if announced_date is None or announced_date > as_of_date:
            continue
        results.append({"michelin_distinction_event_id": event_id, **node})
    results.sort(key=lambda r: r["guide_year"], reverse=True)
    return results


def current_documented_location(
    graph: PlateProofGraph, restaurant_id: str
) -> dict[str, Any] | None:
    """The restaurant's most recently documented location -- never
    presented as full address history, since the upstream ``restaurants``
    table itself retains only the latest known snapshot."""
    edges = graph.out_edges_of_type(restaurant_id, EdgeType.LOCATED_IN)
    if not edges:
        return None
    _, location_id, edge_attrs = edges[0]
    node = graph.node(location_id)
    if node is None:
        return None
    return {"as_of_date": edge_attrs.get("as_of_date"), **node}


def current_documented_cuisine(graph: PlateProofGraph, restaurant_id: str) -> str | None:
    edges = graph.out_edges_of_type(restaurant_id, EdgeType.HAS_CUISINE)
    if not edges:
        return None
    _, cuisine_id, _ = edges[0]
    node = graph.node(cuisine_id)
    return node.get("name") if node else None


def restaurant_node(graph: PlateProofGraph, restaurant_id: str) -> dict[str, Any] | None:
    node = graph.node(restaurant_id)
    if node is None or node.get("node_type") != NodeType.RESTAURANT.value:
        return None
    return dict(node)
