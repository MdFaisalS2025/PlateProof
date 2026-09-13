"""Offline audit CLI for the Task 8A knowledge graph.

Builds the graph from a processed-data directory and prints its integrity
report (node/edge counts, conflicts, dangling edges, duplicate ids,
temporal-consistency findings, scale-limit status) for an administrator to
review before trusting the data the Owner Copilot will eventually read.
Never used by the running web application -- see
``plateproof.graph.builder.GraphService`` for that.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from plateproof.graph.builder import build_graph_from_processed_dir
from plateproof.graph.models import GraphScaleExceededError, GraphScaleLimits


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the PlateProof knowledge graph from a processed-data "
        "directory and print its integrity report."
    )
    parser.add_argument("--processed-data-dir", required=True)
    parser.add_argument("--max-nodes", type=int, default=GraphScaleLimits().max_nodes)
    parser.add_argument("--max-edges", type=int, default=GraphScaleLimits().max_edges)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    limits = GraphScaleLimits(max_nodes=args.max_nodes, max_edges=args.max_edges)

    try:
        result = build_graph_from_processed_dir(Path(args.processed_data_dir), limits=limits)
    except GraphScaleExceededError as exc:
        print(json.dumps({"status": "scale_exceeded", "detail": str(exc)}, indent=2))
        return 1

    report = asdict(result.report)
    report["built_at"] = result.report.built_at.isoformat()
    print(json.dumps(report, indent=2, default=str))

    passed = (
        not result.report.conflicts
        and not result.report.dangling_edge_keys
        and not result.report.duplicate_node_ids
        and not result.report.temporal_violation_descriptions
        and result.report.within_scale_limits
    )
    print(f"\n{'PASS' if passed else 'FAIL'}: graph audit")
    return 0 if passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
