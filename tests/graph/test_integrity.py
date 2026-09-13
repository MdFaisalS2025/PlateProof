"""Direct unit tests for plateproof.graph.integrity's pure check functions,
proving they actually detect what they claim to (not merely that the
production builder happens not to trigger them)."""

from __future__ import annotations

from datetime import date

import networkx as nx

from plateproof.graph.integrity import find_dangling_edges, find_temporal_violations
from plateproof.graph.models import EdgeType, NodeType
from plateproof.graph.store import PlateProofGraph


def test_find_dangling_edges_detects_a_hand_built_dangling_edge() -> None:
    raw = nx.MultiDiGraph()
    raw.add_node("nyc:1", node_type=NodeType.RESTAURANT.value)
    # "nyc:1:missing" is never added as a node -- a dangling target.
    raw.add_edge(
        "nyc:1", "nyc:1:missing", edge_type=EdgeType.HAS_INSPECTION.value, inspection_date=None
    )
    graph = PlateProofGraph(raw)
    dangling = find_dangling_edges(graph)
    assert len(dangling) == 1
    assert "nyc:1->nyc:1:missing" in dangling[0]


def test_find_dangling_edges_reports_nothing_for_a_consistent_graph() -> None:
    raw = nx.MultiDiGraph()
    raw.add_node("nyc:1", node_type=NodeType.RESTAURANT.value)
    raw.add_node("nyc:1:1", node_type=NodeType.INSPECTION.value)
    raw.add_edge(
        "nyc:1",
        "nyc:1:1",
        edge_type=EdgeType.HAS_INSPECTION.value,
        inspection_date=date(2025, 1, 1),
    )
    graph = PlateProofGraph(raw)
    assert find_dangling_edges(graph) == ()


def test_find_temporal_violations_detects_mismatched_guide_year() -> None:
    raw = nx.MultiDiGraph()
    raw.add_node(
        "michelin:e:1",
        node_type=NodeType.MICHELIN_DISTINCTION_EVENT.value,
        guide_year=2025,
        announced_date=date(2019, 11, 1),
    )
    graph = PlateProofGraph(raw)
    violations = find_temporal_violations(graph)
    assert len(violations) == 1
    assert "michelin:e:1" in violations[0]
    assert "2019" in violations[0]
    assert "2025" in violations[0]


def test_find_temporal_violations_accepts_matching_guide_year() -> None:
    raw = nx.MultiDiGraph()
    raw.add_node(
        "michelin:e:1",
        node_type=NodeType.MICHELIN_DISTINCTION_EVENT.value,
        guide_year=2025,
        announced_date=date(2025, 11, 1),
    )
    graph = PlateProofGraph(raw)
    assert find_temporal_violations(graph) == ()


def test_find_temporal_violations_tolerates_unknown_announced_date() -> None:
    raw = nx.MultiDiGraph()
    raw.add_node(
        "michelin:e:1",
        node_type=NodeType.MICHELIN_DISTINCTION_EVENT.value,
        guide_year=2025,
        announced_date=None,
    )
    graph = PlateProofGraph(raw)
    assert find_temporal_violations(graph) == ()
