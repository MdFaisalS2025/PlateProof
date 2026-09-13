"""Thin, typed wrapper over ``networkx.MultiDiGraph``. Nothing outside this
module touches a raw ``networkx`` object -- callers ask for nodes/edges by
:class:`~plateproof.graph.models.NodeType`/``EdgeType`` and get back plain
mappings, never a live-editable graph handle.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import networkx as nx

from plateproof.graph.models import EdgeType, NodeType


class PlateProofGraph:
    """Read-oriented view over a built ``networkx.MultiDiGraph``. Instances
    are produced only by ``plateproof.graph.builder.build_graph`` -- this
    class itself never reads a file or a Parquet table."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph

    @property
    def number_of_nodes(self) -> int:
        return int(self._graph.number_of_nodes())

    @property
    def number_of_edges(self) -> int:
        return int(self._graph.number_of_edges())

    def has_node(self, node_id: str) -> bool:
        return bool(self._graph.has_node(node_id))

    def has_typed_node(self, node_id: str) -> bool:
        """True only for a node that was actually constructed with real
        attributes -- ``networkx.add_edge`` silently creates an empty,
        attribute-less node for any endpoint that doesn't already exist, so
        ``has_node`` alone cannot distinguish a real node from one that
        merely exists because some edge happened to reference it."""
        return self._graph.has_node(node_id) and "node_type" in self._graph.nodes[node_id]

    def node(self, node_id: str) -> Mapping[str, Any] | None:
        if not self._graph.has_node(node_id):
            return None
        return dict(self._graph.nodes[node_id])

    def nodes_of_type(self, node_type: NodeType) -> Iterator[tuple[str, Mapping[str, Any]]]:
        for node_id, attrs in self._graph.nodes(data=True):
            if attrs.get("node_type") == node_type.value:
                yield node_id, dict(attrs)

    def node_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, attrs in self._graph.nodes(data=True):
            node_type = str(attrs.get("node_type"))
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def edge_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, attrs in self._graph.edges(data=True):
            edge_type = str(attrs.get("edge_type"))
            counts[edge_type] = counts.get(edge_type, 0) + 1
        return counts

    def out_edges_of_type(
        self, node_id: str, edge_type: EdgeType
    ) -> list[tuple[str, str, Mapping[str, Any]]]:
        if not self._graph.has_node(node_id):
            return []
        result: list[tuple[str, str, Mapping[str, Any]]] = []
        for _, target, attrs in self._graph.out_edges(node_id, data=True):
            if attrs.get("edge_type") == edge_type.value:
                result.append((node_id, target, dict(attrs)))
        return result

    def in_edges_of_type(
        self, node_id: str, edge_type: EdgeType
    ) -> list[tuple[str, str, Mapping[str, Any]]]:
        if not self._graph.has_node(node_id):
            return []
        result: list[tuple[str, str, Mapping[str, Any]]] = []
        for source, _, attrs in self._graph.in_edges(node_id, data=True):
            if attrs.get("edge_type") == edge_type.value:
                result.append((source, node_id, dict(attrs)))
        return result

    def all_edges(self) -> Iterator[tuple[str, str, Mapping[str, Any]]]:
        for source, target, attrs in self._graph.edges(data=True):
            yield source, target, dict(attrs)
