"""Post-construction integrity checks over an already-built
:class:`~plateproof.graph.store.PlateProofGraph`. Pure functions only --
never mutate the graph, never read a file. ``builder.build_graph`` calls
these to assemble the final :class:`~plateproof.graph.models.GraphIntegrityReport`;
duplicate/conflict detection happens during construction instead (see
``builder.py``), since it needs the original source rows, not just the
final graph shape.
"""

from __future__ import annotations

from plateproof.graph.models import NodeType
from plateproof.graph.store import PlateProofGraph


def find_dangling_edges(graph: PlateProofGraph) -> tuple[str, ...]:
    """Edge keys (``"{source}->{target} ({edge_type})"``) whose source or
    target is not a *real*, fully-constructed node.

    ``networkx.add_edge`` silently creates an empty, attribute-less node
    for any endpoint that doesn't already exist, so a plain "does the node
    exist" check can never detect a dangling reference -- it must check
    for the ``node_type`` attribute every real node carries
    (:meth:`PlateProofGraph.has_typed_node`). The production builder never
    produces one (every node is added with its attributes before any edge
    referencing it), but this check runs unconditionally as a
    defense-in-depth invariant, not merely a construction-time assumption.
    """
    dangling: list[str] = []
    for source, target, attrs in graph.all_edges():
        if not graph.has_typed_node(source) or not graph.has_typed_node(target):
            dangling.append(f"{source}->{target} ({attrs.get('edge_type')})")
    return tuple(sorted(dangling))


def find_temporal_violations(graph: PlateProofGraph) -> tuple[str, ...]:
    """Descriptions of dated facts whose own recorded dates are internally
    inconsistent.

    Currently checks: a ``MichelinDistinctionEvent`` whose ``announced_date``
    falls in a calendar year different from its own ``guide_year`` -- a
    guide edition's announcement year and the edition year it names should
    correspond, so a mismatch is a strong signal of a transcription error
    in the curated seed data, reported for human review rather than
    silently trusted.
    """
    violations: list[str] = []
    for node_id, attrs in graph.nodes_of_type(NodeType.MICHELIN_DISTINCTION_EVENT):
        announced_date = attrs.get("announced_date")
        guide_year = attrs.get("guide_year")
        if announced_date is not None and guide_year is not None:
            if announced_date.year != guide_year:
                violations.append(
                    f"{node_id}: announced_date year {announced_date.year} "
                    f"does not match guide_year {guide_year}"
                )
    return tuple(sorted(violations))
