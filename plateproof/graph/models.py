"""Typed contracts for the Task 8A knowledge graph: node/edge type enums,
build input/result, and the integrity report. No business logic lives
here -- see ``builder.py``, ``store.py``, ``queries.py``, ``integrity.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from plateproof.graph.store import PlateProofGraph


class NodeType(StrEnum):
    RESTAURANT = "restaurant"
    INSPECTION = "inspection"
    VIOLATION_OCCURRENCE = "violation_occurrence"
    VIOLATION_CODE = "violation_code"
    CUISINE = "cuisine"
    LOCATION = "location"
    MICHELIN_RESTAURANT = "michelin_restaurant"
    MICHELIN_DISTINCTION_EVENT = "michelin_distinction_event"


class EdgeType(StrEnum):
    HAS_INSPECTION = "has_inspection"
    DOCUMENTED_VIOLATION = "documented_violation"
    USES_VIOLATION_CODE = "uses_violation_code"
    LOCATED_IN = "located_in"
    HAS_CUISINE = "has_cuisine"
    MATCHED_TO_MICHELIN = "matched_to_michelin"
    HAS_DISTINCTION_EVENT = "has_distinction_event"


@dataclass(frozen=True)
class GraphBuildInput:
    """Immutable snapshot of the exact tables the graph is built from.

    Each field is ``None`` when that processed table is not configured or
    could not be read -- mirroring ``plateproof.serving.repository``'s
    per-table tolerance (a missing optional table is never a build
    failure). ``restaurants``/``inspection_events``/``violation_events``
    being ``None`` simply yields an empty graph, not an error.
    """

    restaurants: pl.DataFrame | None
    inspection_events: pl.DataFrame | None
    violation_events: pl.DataFrame | None
    michelin_restaurants: pl.DataFrame | None
    michelin_distinction_events: pl.DataFrame | None
    restaurant_michelin_matches: pl.DataFrame | None


@dataclass(frozen=True)
class GraphSourceFingerprint:
    """Cheap, deterministic identity for 'has the input changed since the
    graph was last built' -- (logical_name, mtime_ns, size_bytes) per
    source file. A missing file is recorded as ``(-1, -1)``. This decides
    only whether to rebuild; it is never a substitute for the checksum
    verification a metadata reader performs before trusting content."""

    entries: tuple[tuple[str, int, int], ...]


class GraphScaleExceededError(Exception):
    """Raised by ``build_graph`` when the resulting graph would exceed
    ``GraphScaleLimits``. Callers must treat this as a typed 'graph
    temporarily unavailable' condition, never let it propagate as a 500."""


@dataclass(frozen=True)
class GraphScaleLimits:
    max_nodes: int = 500_000
    max_edges: int = 2_000_000


@dataclass(frozen=True)
class GraphConflict:
    """A same-node-id disagreement on a field not covered by the node's
    identity key -- reported, never silently overwritten (mirrors
    ``RestaurantConflict`` / Michelin's ``conflicting_restaurant_ids``)."""

    node_id: str
    node_type: str
    field: str
    values: tuple[str, ...]
    source_row_refs: tuple[str, ...]


@dataclass(frozen=True)
class GraphIntegrityReport:
    node_count: int
    edge_count: int
    node_counts_by_type: Mapping[str, int]
    edge_counts_by_type: Mapping[str, int]
    conflicts: tuple[GraphConflict, ...]
    dangling_edge_keys: tuple[str, ...]
    duplicate_node_ids: tuple[str, ...]
    temporal_violation_descriptions: tuple[str, ...]
    within_scale_limits: bool
    build_duration_ms: float
    input_row_counts: Mapping[str, int]
    built_at: datetime


@dataclass(frozen=True)
class GraphBuildResult:
    graph: PlateProofGraph
    report: GraphIntegrityReport
