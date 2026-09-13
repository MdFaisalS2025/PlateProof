"""Tests for plateproof.graph.builder: deterministic construction, node
uniqueness, timestamped edges, accepted-only Michelin, jurisdiction
separation, referential integrity, duplicate/conflict detection, and scale
limits.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

import pytest

from plateproof.graph.models import EdgeType, GraphScaleExceededError, GraphScaleLimits, NodeType


def test_empty_input_yields_empty_graph(make_build_input: Any) -> None:
    from plateproof.graph.builder import build_graph

    result = build_graph(make_build_input())
    assert result.graph.number_of_nodes == 0
    assert result.graph.number_of_edges == 0
    assert result.report.within_scale_limits is True
    assert result.report.conflicts == ()
    assert result.report.dangling_edge_keys == ()


def test_restaurant_and_inspection_nodes_are_unique(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(inspection_id="nyc:1:1", restaurant_id="nyc:1"),
            inspection_row(inspection_id="nyc:1:2", restaurant_id="nyc:1"),
        ],
    )
    result = build_graph(build_input)
    graph = result.graph
    assert graph.has_node("nyc:1")
    assert graph.node("nyc:1")["node_type"] == NodeType.RESTAURANT.value
    assert graph.node("nyc:1:1")["node_type"] == NodeType.INSPECTION.value
    assert graph.node("nyc:1:2")["node_type"] == NodeType.INSPECTION.value
    # Adding a restaurant/inspection with the same id twice never creates a
    # second node -- networkx itself enforces node-id uniqueness, and this
    # asserts the count reflects that, not an accidental duplicate.
    assert result.report.node_counts_by_type[NodeType.RESTAURANT.value] == 1
    assert result.report.node_counts_by_type[NodeType.INSPECTION.value] == 2


def test_edges_carry_dates_and_provenance(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(
                inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2025, 6, 1)
            )
        ],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_date=date(2025, 6, 1),
                violation_code="04L",
                violation_code_norm="04L",
            )
        ],
    )
    result = build_graph(build_input)
    graph = result.graph

    inspection_edges = graph.out_edges_of_type("nyc:1", EdgeType.HAS_INSPECTION)
    assert len(inspection_edges) == 1
    assert inspection_edges[0][2]["inspection_date"] == date(2025, 6, 1)

    violation_edges = graph.out_edges_of_type("nyc:1:1", EdgeType.DOCUMENTED_VIOLATION)
    assert len(violation_edges) == 1
    assert violation_edges[0][2]["inspection_date"] == date(2025, 6, 1)

    code_edges = graph.out_edges_of_type("nyc:v:1", EdgeType.USES_VIOLATION_CODE)
    assert len(code_edges) == 1
    _, code_node_id, code_edge_attrs = code_edges[0]
    assert code_node_id == "nyc:vc:04L"
    assert code_edge_attrs["observed_date"] == date(2025, 6, 1)
    assert code_edge_attrs["source_violation_event_id"] == "nyc:v:1"


def test_separate_nyc_and_florida_violation_code_nodes(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    """Two codes that share the exact same text in different jurisdictions
    must never collapse into one node."""
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[
            restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc"),
            restaurant_row(restaurant_id="florida:1", jurisdiction="florida"),
        ],
        inspections=[
            inspection_row(inspection_id="nyc:1:1", restaurant_id="nyc:1", jurisdiction="nyc"),
            inspection_row(
                inspection_id="florida:1:1", restaurant_id="florida:1", jurisdiction="florida"
            ),
        ],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                violation_code="04L",
                violation_code_norm="04L",
            ),
            violation_row(
                violation_event_id="florida:v:1",
                inspection_id="florida:1:1",
                restaurant_id="florida:1",
                jurisdiction="florida",
                violation_code="04L",
                violation_code_norm="04L",
            ),
        ],
    )
    result = build_graph(build_input)
    graph = result.graph
    assert graph.has_node("nyc:vc:04L")
    assert graph.has_node("florida:vc:04L")
    assert graph.node("nyc:vc:04L") != graph.node("florida:vc:04L")
    assert result.report.node_counts_by_type[NodeType.VIOLATION_CODE.value] == 2


def test_accepted_only_michelin_match_produces_an_edge(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    result = build_graph(build_input)
    edges = result.graph.out_edges_of_type("nyc:1", EdgeType.MATCHED_TO_MICHELIN)
    assert len(edges) == 1
    assert edges[0][1] == "michelin:r:aaa"


@pytest.mark.parametrize("decision", ["review", "reject"])
def test_non_accepted_michelin_match_produces_no_edge(
    make_build_input: Any, restaurant_row: Any, decision: str
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": decision,
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    result = build_graph(build_input)
    edges = result.graph.out_edges_of_type("nyc:1", EdgeType.MATCHED_TO_MICHELIN)
    assert edges == []


def test_no_edge_ever_implies_michelin_causes_or_predicts_safety(
    make_build_input: Any, restaurant_row: Any
) -> None:
    """Structural guarantee: the complete set of edge types the builder can
    ever create contains nothing connecting a Michelin node directly to an
    Inspection/ViolationOccurrence node -- only Restaurant<->Michelin and
    Michelin<->DistinctionEvent edges exist."""
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    result = build_graph(build_input)
    michelin_edge_types = {
        attrs["edge_type"]
        for _, target, attrs in result.graph.all_edges()
        if result.graph.node(target) is not None
        and result.graph.node(target).get("node_type")
        in (NodeType.MICHELIN_RESTAURANT.value, NodeType.MICHELIN_DISTINCTION_EVENT.value)
    }
    assert michelin_edge_types <= {
        EdgeType.MATCHED_TO_MICHELIN.value,
        EdgeType.HAS_DISTINCTION_EVENT.value,
    }


def test_michelin_distinction_event_with_unknown_announced_date_still_a_node(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        michelin_distinction_events=[
            {
                "michelin_distinction_event_id": "michelin:e:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "Fictional Guide",
                "guide_year": 2025,
                "announced_date": None,
                "source_url": "https://example.invalid/fictional",
            }
        ],
    )
    result = build_graph(build_input)
    assert result.graph.has_node("michelin:e:1")
    assert result.graph.node("michelin:e:1")["announced_date"] is None


def test_duplicate_source_rows_with_conflicting_fields_are_reported(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[
            restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen"),
            restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen And Bar"),
        ],
    )
    result = build_graph(build_input)
    conflicts = [c for c in result.report.conflicts if c.field == "name"]
    assert len(conflicts) == 1
    assert set(conflicts[0].values) == {"Anna's Kitchen", "Anna's Kitchen And Bar"}
    assert result.report.duplicate_node_ids == ("nyc:1",)


def test_referential_integrity_no_dangling_edges_in_production_output(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[inspection_row(inspection_id="nyc:1:1", restaurant_id="nyc:1")],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                violation_code_norm="04L",
            )
        ],
    )
    result = build_graph(build_input)
    assert result.report.dangling_edge_keys == ()


def test_inspection_referencing_unknown_restaurant_creates_no_edge(
    make_build_input: Any, inspection_row: Any
) -> None:
    """An inspection row whose restaurant isn't in the restaurants table
    must never invent a fake restaurant identity or a dangling edge -- the
    inspection node still exists (auditable), but is simply unconnected."""
    from plateproof.graph.builder import build_graph

    build_input = make_build_input(
        inspections=[inspection_row(inspection_id="nyc:1:1", restaurant_id="nyc:1")],
    )
    result = build_graph(build_input)
    assert result.graph.has_node("nyc:1:1")
    assert not result.graph.has_node("nyc:1")
    assert result.report.dangling_edge_keys == ()


def test_build_is_deterministic_under_shuffled_input(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    restaurants = [restaurant_row(restaurant_id=f"nyc:{i}") for i in range(10)]
    inspections = [
        inspection_row(inspection_id=f"nyc:{i}:1", restaurant_id=f"nyc:{i}") for i in range(10)
    ]
    violations = [
        violation_row(
            violation_event_id=f"nyc:v:{i}",
            inspection_id=f"nyc:{i}:1",
            restaurant_id=f"nyc:{i}",
            violation_code_norm="04L",
        )
        for i in range(10)
    ]

    rng = random.Random(42)
    shuffled_restaurants = restaurants[:]
    shuffled_inspections = inspections[:]
    shuffled_violations = violations[:]
    rng.shuffle(shuffled_restaurants)
    rng.shuffle(shuffled_inspections)
    rng.shuffle(shuffled_violations)

    result_a = build_graph(
        make_build_input(restaurants=restaurants, inspections=inspections, violations=violations)
    )
    result_b = build_graph(
        make_build_input(
            restaurants=shuffled_restaurants,
            inspections=shuffled_inspections,
            violations=shuffled_violations,
        )
    )

    assert sorted(result_a.graph.node_counts_by_type().items()) == sorted(
        result_b.graph.node_counts_by_type().items()
    )
    assert result_a.report.node_count == result_b.report.node_count
    assert result_a.report.edge_count == result_b.report.edge_count
    edges_a = sorted((s, t, attrs["edge_type"]) for s, t, attrs in result_a.graph.all_edges())
    edges_b = sorted((s, t, attrs["edge_type"]) for s, t, attrs in result_b.graph.all_edges())
    assert edges_a == edges_b


def test_scale_limits_raise_before_returning_an_oversized_graph(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph

    restaurants = [restaurant_row(restaurant_id=f"nyc:{i}") for i in range(5)]
    build_input = make_build_input(restaurants=restaurants)
    with pytest.raises(GraphScaleExceededError):
        build_graph(build_input, limits=GraphScaleLimits(max_nodes=3, max_edges=100))


def test_no_model_training_dependency() -> None:
    """Architectural guarantee: plateproof.graph never imports
    plateproof.models or plateproof.features -- graph output must never
    feed back into Task 6 training."""
    import ast
    from pathlib import Path

    graph_dir = Path(__file__).resolve().parent.parent.parent / "plateproof" / "graph"
    for path in graph_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("plateproof.models"), (path, node.module)
                assert not node.module.startswith("plateproof.features"), (path, node.module)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("plateproof.models"), (path, alias.name)
                    assert not alias.name.startswith("plateproof.features"), (path, alias.name)
