"""Deterministic construction of a :class:`~plateproof.graph.store.PlateProofGraph`
from Task 7's already-processed Parquet tables.

``build_graph`` is the pure core: given already-loaded frames, it produces
the same graph every time regardless of input row order (every frame is
sorted by its own primary key before iteration), uses no wall-clock or
random value in any node id or ordering decision, and never raises for
ordinary missing-optional-data conditions -- only ``GraphScaleExceededError``
for a genuinely oversized result.

``build_graph_from_processed_dir``/``GraphService`` add the thin I/O layer:
reading the six Parquet files (each individually optional, exactly like
``plateproof.serving.repository``'s per-table tolerance) and caching the
built graph by a cheap source fingerprint so a long-lived process rebuilds
only when the underlying files actually change.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl

from plateproof.graph.integrity import find_dangling_edges, find_temporal_violations
from plateproof.graph.models import (
    EdgeType,
    GraphBuildInput,
    GraphBuildResult,
    GraphConflict,
    GraphIntegrityReport,
    GraphScaleExceededError,
    GraphScaleLimits,
    GraphSourceFingerprint,
    NodeType,
)
from plateproof.graph.store import PlateProofGraph

_TABLE_FILES: tuple[str, ...] = (
    "restaurants.parquet",
    "inspection_events.parquet",
    "violation_events.parquet",
    "michelin_restaurants.parquet",
    "michelin_distinction_events.parquet",
    "restaurant_michelin_matches.parquet",
)


def violation_code_node_id(jurisdiction: str, violation_code_norm: str) -> str:
    """NYC and Florida codes are never merged merely because their text is
    similar -- the jurisdiction is always part of the id."""
    return f"{jurisdiction}:vc:{violation_code_norm}"


def cuisine_node_id(normalized_cuisine: str) -> str:
    return f"cuisine:{normalized_cuisine}"


def location_node_id(jurisdiction: str, city: str | None, postal_code: str | None) -> str:
    return f"{jurisdiction}:loc:{city or '_unknown'}:{postal_code or '_unknown'}"


def _normalize_cuisine(name: str) -> str:
    return " ".join(name.strip().lower().split())


def _normalize_violation_code(row: Mapping[str, Any], has_norm_column: bool) -> str:
    if has_norm_column:
        norm = row.get("violation_code_norm")
        if norm:
            return str(norm)
    return str(row.get("violation_code") or "").strip().upper()


def _check_node_capacity(graph: nx.MultiDiGraph, limits: GraphScaleLimits, node_id: str) -> None:
    """Raises before a genuinely new node would be added if the graph is
    already at its configured node limit. A node id already present never
    consumes new capacity -- checked first, cheaply, via ``has_node``."""
    if graph.has_node(node_id):
        return
    if graph.number_of_nodes() >= limits.max_nodes:
        raise GraphScaleExceededError("graph exceeded the configured maximum node count")


class _EdgeBudget:
    """A manual, O(1) edge counter.

    ``networkx``'s own ``MultiDiGraph.number_of_edges()`` (no arguments)
    calls ``Graph.size()``, which sums degree over every node -- O(V), not
    O(1). Calling it once per edge addition while building a graph with V
    nodes and E edges costs O(V*E) overall, which is enough to make a
    legitimately-sized (tens of thousands of nodes/edges) build take
    minutes instead of seconds. This counter is incremented exactly once
    per successful ``add_edge`` call, so checking capacity is a single
    integer comparison regardless of graph size."""

    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0


def _add_new_node(
    graph: nx.MultiDiGraph, limits: GraphScaleLimits, node_id: str, **attrs: Any
) -> None:
    """For the simple, non-conflict-tracked node types (location, cuisine,
    violation code) -- adds ``node_id`` only if not already present,
    enforcing the node limit first."""
    if graph.has_node(node_id):
        return
    _check_node_capacity(graph, limits, node_id)
    graph.add_node(node_id, **attrs)


def _add_new_edge(
    graph: nx.MultiDiGraph,
    limits: GraphScaleLimits,
    edge_budget: _EdgeBudget,
    source: str,
    target: str,
    **attrs: Any,
) -> None:
    """Raises before a new edge would be added if the graph is already at
    its configured edge limit. Every edge in this builder connects two
    nodes added via :func:`_add_new_node`/:func:`_add_node_with_conflict_check`
    beforehand, so this is never reachable via ``networkx``'s own
    endpoint-auto-creation on ``add_edge`` -- both endpoints already exist
    as real, accounted-for nodes by the time any edge is added."""
    if edge_budget.count >= limits.max_edges:
        raise GraphScaleExceededError("graph exceeded the configured maximum edge count")
    graph.add_edge(source, target, **attrs)
    edge_budget.count += 1


def _add_node_with_conflict_check(
    graph: nx.MultiDiGraph,
    tracking: dict[str, list[str]],
    conflicts: list[GraphConflict],
    node_id: str,
    node_type: NodeType,
    attrs: Mapping[str, Any],
    *,
    source_ref: str,
    limits: GraphScaleLimits,
) -> None:
    """Adds ``node_id`` with ``attrs`` the first time it's seen. A later row
    mapping to the same id is never silently overwritten: any field where
    both rows supply a non-null, disagreeing value is recorded as a
    :class:`GraphConflict`. ``tracking`` accumulates every source ref for
    this id purely for the report; it is never stored on the graph node
    itself. Raises :class:`GraphScaleExceededError` *before* adding a
    genuinely new node if the graph is already at its configured limit --
    an id already present (a repeated reference) never consumes new
    capacity and never raises."""
    is_new = not graph.has_node(node_id)
    if is_new:
        _check_node_capacity(graph, limits, node_id)
    refs = tracking.setdefault(node_id, [])
    refs.append(source_ref)
    if is_new:
        graph.add_node(node_id, **attrs)
        return
    existing = graph.nodes[node_id]
    for field, value in attrs.items():
        if field == "node_type":
            continue
        existing_value = existing.get(field)
        if value is not None and existing_value is not None and existing_value != value:
            conflicts.append(
                GraphConflict(
                    node_id=node_id,
                    node_type=node_type.value,
                    field=field,
                    values=tuple(sorted({str(existing_value), str(value)})),
                    source_row_refs=tuple(refs),
                )
            )


def build_graph(
    input_data: GraphBuildInput, *, limits: GraphScaleLimits = GraphScaleLimits()
) -> GraphBuildResult:
    start = time.perf_counter()
    graph = nx.MultiDiGraph()
    conflicts: list[GraphConflict] = []
    tracking: dict[str, list[str]] = {}
    input_row_counts: dict[str, int] = {}
    edge_budget = _EdgeBudget()

    restaurants_df = input_data.restaurants
    input_row_counts["restaurants"] = restaurants_df.height if restaurants_df is not None else 0
    if restaurants_df is not None and restaurants_df.height > 0:
        for row in restaurants_df.sort("restaurant_id").iter_rows(named=True):
            node_id = row["restaurant_id"]
            jurisdiction = row["jurisdiction"]
            attrs = {
                "node_type": NodeType.RESTAURANT.value,
                "jurisdiction": jurisdiction,
                "name": row.get("name"),
                "normalized_name": row.get("normalized_name"),
                "city": row.get("city"),
                "region": row.get("region"),
                "postal_code": row.get("postal_code"),
            }
            _add_node_with_conflict_check(
                graph,
                tracking,
                conflicts,
                node_id,
                NodeType.RESTAURANT,
                attrs,
                source_ref=node_id,
                limits=limits,
            )

            as_of = row.get("source_snapshot_date") or row.get("latest_inspection_date")
            loc_id = location_node_id(jurisdiction, row.get("city"), row.get("postal_code"))
            _add_new_node(
                graph,
                limits,
                loc_id,
                node_type=NodeType.LOCATION.value,
                jurisdiction=jurisdiction,
                city=row.get("city"),
                postal_code=row.get("postal_code"),
            )
            _add_new_edge(
                graph,
                limits,
                edge_budget,
                node_id,
                loc_id,
                edge_type=EdgeType.LOCATED_IN.value,
                as_of_date=as_of,
            )

            cuisine = row.get("cuisine")
            if cuisine:
                norm_cuisine = _normalize_cuisine(cuisine)
                cuisine_id = cuisine_node_id(norm_cuisine)
                _add_new_node(
                    graph, limits, cuisine_id, node_type=NodeType.CUISINE.value, name=norm_cuisine
                )
                _add_new_edge(
                    graph,
                    limits,
                    edge_budget,
                    node_id,
                    cuisine_id,
                    edge_type=EdgeType.HAS_CUISINE.value,
                    as_of_date=as_of,
                )

    inspections_df = input_data.inspection_events
    input_row_counts["inspection_events"] = (
        inspections_df.height if inspections_df is not None else 0
    )
    if inspections_df is not None and inspections_df.height > 0:
        for row in inspections_df.sort(["restaurant_id", "inspection_id"]).iter_rows(named=True):
            node_id = row["inspection_id"]
            attrs = {
                "node_type": NodeType.INSPECTION.value,
                "jurisdiction": row.get("jurisdiction"),
                "inspection_date": row.get("inspection_date"),
                "inspection_type": row.get("inspection_type"),
                "score": row.get("score"),
                "grade": row.get("grade"),
                "high_priority_count": row.get("high_priority_count"),
                "intermediate_count": row.get("intermediate_count"),
                "basic_count": row.get("basic_count"),
                "critical_violation_count": row.get("critical_violation_count"),
            }
            _add_node_with_conflict_check(
                graph,
                tracking,
                conflicts,
                node_id,
                NodeType.INSPECTION,
                attrs,
                source_ref=node_id,
                limits=limits,
            )
            restaurant_id = row["restaurant_id"]
            if graph.has_node(restaurant_id):
                _add_new_edge(
                    graph,
                    limits,
                    edge_budget,
                    restaurant_id,
                    node_id,
                    edge_type=EdgeType.HAS_INSPECTION.value,
                    inspection_date=row.get("inspection_date"),
                )

    violations_df = input_data.violation_events
    input_row_counts["violation_events"] = violations_df.height if violations_df is not None else 0
    if violations_df is not None and violations_df.height > 0:
        has_norm_column = "violation_code_norm" in violations_df.columns
        for row in violations_df.sort(["inspection_id", "violation_event_id"]).iter_rows(
            named=True
        ):
            node_id = row["violation_event_id"]
            code_norm = _normalize_violation_code(row, has_norm_column)
            attrs = {
                "node_type": NodeType.VIOLATION_OCCURRENCE.value,
                "jurisdiction": row.get("jurisdiction"),
                "inspection_date": row.get("inspection_date"),
                "violation_code": row.get("violation_code"),
                "violation_code_norm": code_norm,
                "violation_description": row.get("violation_description"),
                "severity": row.get("severity"),
                "count": row.get("count") or 1,
                "corrected_on_site": row.get("corrected_on_site"),
            }
            _add_node_with_conflict_check(
                graph,
                tracking,
                conflicts,
                node_id,
                NodeType.VIOLATION_OCCURRENCE,
                attrs,
                source_ref=node_id,
                limits=limits,
            )

            inspection_id = row["inspection_id"]
            if graph.has_node(inspection_id):
                _add_new_edge(
                    graph,
                    limits,
                    edge_budget,
                    inspection_id,
                    node_id,
                    edge_type=EdgeType.DOCUMENTED_VIOLATION.value,
                    inspection_date=row.get("inspection_date"),
                )

            jurisdiction = str(row.get("jurisdiction"))
            code_id = violation_code_node_id(jurisdiction, code_norm)
            _add_new_node(
                graph,
                limits,
                code_id,
                node_type=NodeType.VIOLATION_CODE.value,
                jurisdiction=jurisdiction,
                code=code_norm,
            )
            _add_new_edge(
                graph,
                limits,
                edge_budget,
                node_id,
                code_id,
                edge_type=EdgeType.USES_VIOLATION_CODE.value,
                observed_date=row.get("inspection_date"),
                source_violation_event_id=node_id,
            )

    michelin_restaurants_df = input_data.michelin_restaurants
    input_row_counts["michelin_restaurants"] = (
        michelin_restaurants_df.height if michelin_restaurants_df is not None else 0
    )
    if michelin_restaurants_df is not None and michelin_restaurants_df.height > 0:
        for row in michelin_restaurants_df.sort("michelin_restaurant_id").iter_rows(named=True):
            node_id = row["michelin_restaurant_id"]
            attrs = {
                "node_type": NodeType.MICHELIN_RESTAURANT.value,
                "name_as_published": row.get("name_as_published"),
                "jurisdiction_candidate": row.get("jurisdiction_candidate"),
            }
            _add_node_with_conflict_check(
                graph,
                tracking,
                conflicts,
                node_id,
                NodeType.MICHELIN_RESTAURANT,
                attrs,
                source_ref=node_id,
                limits=limits,
            )

    michelin_events_df = input_data.michelin_distinction_events
    input_row_counts["michelin_distinction_events"] = (
        michelin_events_df.height if michelin_events_df is not None else 0
    )
    if michelin_events_df is not None and michelin_events_df.height > 0:
        for row in michelin_events_df.sort("michelin_distinction_event_id").iter_rows(named=True):
            node_id = row["michelin_distinction_event_id"]
            attrs = {
                "node_type": NodeType.MICHELIN_DISTINCTION_EVENT.value,
                "distinction": row.get("distinction"),
                "guide_name": row.get("guide_name"),
                "guide_year": row.get("guide_year"),
                "announced_date": row.get("announced_date"),
                "source_url": row.get("source_url"),
            }
            _add_node_with_conflict_check(
                graph,
                tracking,
                conflicts,
                node_id,
                NodeType.MICHELIN_DISTINCTION_EVENT,
                attrs,
                source_ref=node_id,
                limits=limits,
            )
            michelin_restaurant_id = row["michelin_restaurant_id"]
            if graph.has_node(michelin_restaurant_id):
                _add_new_edge(
                    graph,
                    limits,
                    edge_budget,
                    michelin_restaurant_id,
                    node_id,
                    edge_type=EdgeType.HAS_DISTINCTION_EVENT.value,
                    announced_date=row.get("announced_date"),
                )

    matches_df = input_data.restaurant_michelin_matches
    input_row_counts["restaurant_michelin_matches"] = (
        matches_df.height if matches_df is not None else 0
    )
    if matches_df is not None and matches_df.height > 0:
        for row in matches_df.sort(["official_restaurant_id", "michelin_restaurant_id"]).iter_rows(
            named=True
        ):
            # Accepted-only: a "review" or "reject" decision must never
            # produce an edge -- mirrors Repository.michelin_history().
            if row.get("decision") != "accept":
                continue
            official_id = row["official_restaurant_id"]
            michelin_id = row["michelin_restaurant_id"]
            if graph.has_node(official_id) and graph.has_node(michelin_id):
                _add_new_edge(
                    graph,
                    limits,
                    edge_budget,
                    official_id,
                    michelin_id,
                    edge_type=EdgeType.MATCHED_TO_MICHELIN.value,
                    generated_at=row.get("generated_at"),
                )

    # By this point every node/edge addition above already enforced its
    # limit incrementally (raising and unwinding before ever exceeding
    # it), so this is a redundant, defense-in-depth confirmation -- never
    # the primary enforcement point. A failed build never reaches here at
    # all, so no partial graph is ever returned.
    plate_graph = PlateProofGraph(graph)
    node_count = plate_graph.number_of_nodes
    edge_count = plate_graph.number_of_edges
    within_limits = node_count <= limits.max_nodes and edge_count <= limits.max_edges
    if not within_limits:  # pragma: no cover - unreachable given the incremental checks above
        raise GraphScaleExceededError("graph exceeded the configured scale limits")

    duplicate_node_ids = tuple(
        sorted(node_id for node_id, refs in tracking.items() if len(refs) > 1)
    )
    report = GraphIntegrityReport(
        node_count=node_count,
        edge_count=edge_count,
        node_counts_by_type=plate_graph.node_counts_by_type(),
        edge_counts_by_type=plate_graph.edge_counts_by_type(),
        conflicts=tuple(conflicts),
        dangling_edge_keys=find_dangling_edges(plate_graph),
        duplicate_node_ids=duplicate_node_ids,
        temporal_violation_descriptions=find_temporal_violations(plate_graph),
        within_scale_limits=within_limits,
        build_duration_ms=(time.perf_counter() - start) * 1000,
        input_row_counts=input_row_counts,
        built_at=datetime.now(UTC),
    )
    return GraphBuildResult(graph=plate_graph, report=report)


def _read_optional_parquet(path: Path) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pl.read_parquet(path)
    except Exception:
        # Narrowly scoped to this one library call, mirroring
        # plateproof.serving.repository's `except duckdb.Error: continue`:
        # a corrupted or foreign Parquet file means "table not available",
        # never a crash. polars/pyarrow can raise several exception types
        # for a malformed file; this is not a broad catch around unrelated
        # application logic.
        return None


def compute_source_fingerprint(processed_dir: Path) -> GraphSourceFingerprint:
    """Cheap per-file (mtime, size) identity -- decides only whether to
    rebuild the cached graph, never whether to trust a file's content."""
    entries: list[tuple[str, int, int]] = []
    for name in _TABLE_FILES:
        path = processed_dir / name
        try:
            stat = path.stat()
            entries.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            entries.append((name, -1, -1))
    return GraphSourceFingerprint(entries=tuple(entries))


def build_graph_from_processed_dir(
    processed_dir: Path, *, limits: GraphScaleLimits = GraphScaleLimits()
) -> GraphBuildResult:
    input_data = GraphBuildInput(
        restaurants=_read_optional_parquet(processed_dir / "restaurants.parquet"),
        inspection_events=_read_optional_parquet(processed_dir / "inspection_events.parquet"),
        violation_events=_read_optional_parquet(processed_dir / "violation_events.parquet"),
        michelin_restaurants=_read_optional_parquet(processed_dir / "michelin_restaurants.parquet"),
        michelin_distinction_events=_read_optional_parquet(
            processed_dir / "michelin_distinction_events.parquet"
        ),
        restaurant_michelin_matches=_read_optional_parquet(
            processed_dir / "restaurant_michelin_matches.parquet"
        ),
    )
    return build_graph(input_data, limits=limits)


class GraphService:
    """Caches one :class:`GraphBuildResult` per source fingerprint --
    rebuilds only when the underlying Parquet files actually change
    (mtime/size), not on every access. One instance per app/process
    lifetime, exactly like ``plateproof.serving.repository.Repository``
    (never a cross-process singleton)."""

    def __init__(
        self, processed_dir: Path, *, limits: GraphScaleLimits = GraphScaleLimits()
    ) -> None:
        self._processed_dir = processed_dir
        self._limits = limits
        self._cached: GraphBuildResult | None = None
        self._cached_fingerprint: GraphSourceFingerprint | None = None

    def get(self) -> GraphBuildResult:
        fingerprint = compute_source_fingerprint(self._processed_dir)
        if self._cached is None or fingerprint != self._cached_fingerprint:
            self._cached = build_graph_from_processed_dir(self._processed_dir, limits=self._limits)
            self._cached_fingerprint = fingerprint
        return self._cached
