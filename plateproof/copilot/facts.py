"""Builds typed :class:`~plateproof.copilot.models.RestaurantFact` objects
from the knowledge graph -- the only input restaurant-scoped
:class:`~plateproof.copilot.models.Claim` objects are ever built from. No
claim builder reads the graph directly; everything goes through this
module first, so every fact used in an answer has a consistent, auditable
citation shape (``citation_id`` always ``"record:<restaurant_id>:<label>"``,
``evidence_type`` always ``restaurant_record``).

Every function here validates the jurisdiction it reads off the graph via
:func:`~plateproof.copilot.models.parse_restaurant_jurisdiction` rather
than casting a raw string -- an unrecognized/missing jurisdiction value
means no fact is produced (fail closed), never a guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from plateproof.copilot.models import (
    Citation,
    EvidenceType,
    RestaurantFact,
    RestaurantJurisdiction,
    parse_restaurant_jurisdiction,
)
from plateproof.graph import queries
from plateproof.graph.store import PlateProofGraph


def _record_citation(
    restaurant_id: str, jurisdiction: RestaurantJurisdiction, as_of: date, label: str, excerpt: str
) -> Citation:
    return Citation(
        citation_id=f"record:{restaurant_id}:{label}",
        evidence_type=EvidenceType.RESTAURANT_RECORD,
        title=f"PlateProof documented record ({label})",
        url=None,
        excerpt=excerpt,
        jurisdiction=jurisdiction,
        as_of_date=as_of,
    )


def restaurant_identity_fact(graph: PlateProofGraph, restaurant_id: str) -> RestaurantFact | None:
    restaurant = queries.restaurant_node(graph, restaurant_id)
    if restaurant is None:
        return None
    jurisdiction = parse_restaurant_jurisdiction(restaurant.get("jurisdiction"))
    if jurisdiction is None:
        return None
    location = queries.current_documented_location(graph, restaurant_id)
    cuisine = queries.current_documented_cuisine(graph, restaurant_id)
    as_of = (location or {}).get("as_of_date") or date.today()
    excerpt = f"Documented identity for restaurant {restaurant_id} as of {as_of.isoformat()}."
    return RestaurantFact(
        fact_type="restaurant_identity",
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        as_of_date=as_of,
        values={
            "name": restaurant.get("name"),
            "jurisdiction": jurisdiction,
            "city": (location or {}).get("city"),
            "postal_code": (location or {}).get("postal_code"),
            "cuisine": cuisine,
        },
        source_node_ids=(restaurant_id,),
        citation=_record_citation(restaurant_id, jurisdiction, as_of, "identity", excerpt),
    )


def latest_inspection_fact(graph: PlateProofGraph, restaurant_id: str) -> RestaurantFact | None:
    result = queries.latest_inspection(graph, restaurant_id)
    if result is None:
        return None
    restaurant = queries.restaurant_node(graph, restaurant_id)
    jurisdiction = parse_restaurant_jurisdiction((restaurant or {}).get("jurisdiction"))
    if jurisdiction is None:
        return None
    excerpt = (
        f"Documented inspection {result.inspection_id} on "
        f"{result.inspection_date.isoformat()} for restaurant {restaurant_id}."
    )
    return RestaurantFact(
        fact_type="inspection_summary",
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        as_of_date=result.inspection_date,
        values={
            "inspection_id": result.inspection_id,
            "inspection_date": result.inspection_date,
            "inspection_type": result.inspection_type,
            "score": result.score,
            "grade": result.grade,
            "same_day_count": result.same_day_count,
        },
        source_node_ids=(result.inspection_id,),
        citation=_record_citation(
            restaurant_id, jurisdiction, result.inspection_date, "latest_inspection", excerpt
        ),
    )


def recurring_violation_facts(
    graph: PlateProofGraph, restaurant_id: str, *, min_occurrences: int = 2
) -> tuple[RestaurantFact, ...]:
    items = queries.recurring_violation_codes(graph, restaurant_id, min_occurrences=min_occurrences)
    facts: list[RestaurantFact] = []
    for item in items:
        jurisdiction = parse_restaurant_jurisdiction(item.jurisdiction)
        if jurisdiction is None:
            continue
        as_of = max(item.inspection_dates) if item.inspection_dates else date.today()
        label = f"violation_code_{item.violation_code_norm}"
        excerpt = (
            f"Violation code {item.violation_code} documented on "
            f"{item.occurrence_count} distinct inspection(s) for restaurant {restaurant_id}, "
            f"most recently on {as_of.isoformat()}."
        )
        facts.append(
            RestaurantFact(
                fact_type="violation_code_frequency",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                as_of_date=as_of,
                values={
                    "violation_code": item.violation_code,
                    "violation_code_norm": item.violation_code_norm,
                    "occurrence_count": item.occurrence_count,
                    "total_cited_count": item.total_cited_count,
                    "inspection_dates": item.inspection_dates,
                    "description": item.most_recent_description,
                    "description_date": item.most_recent_description_date,
                    "severity": item.severity,
                },
                source_node_ids=(),
                citation=_record_citation(restaurant_id, jurisdiction, as_of, label, excerpt),
            )
        )
    return tuple(facts)


def inspection_trend_fact(graph: PlateProofGraph, restaurant_id: str) -> RestaurantFact | None:
    rows = queries.restaurant_inspections(graph, restaurant_id)
    if len(rows) < 2:
        return None
    restaurant = queries.restaurant_node(graph, restaurant_id)
    jurisdiction = parse_restaurant_jurisdiction((restaurant or {}).get("jurisdiction"))
    if jurisdiction is None:
        return None
    by_date = sorted(rows, key=lambda r: r["inspection_date"])
    earliest, latest = by_date[0], by_date[-1]
    excerpt = (
        f"{len(rows)} documented inspections for restaurant {restaurant_id} between "
        f"{earliest['inspection_date'].isoformat()} and {latest['inspection_date'].isoformat()}."
    )
    return RestaurantFact(
        fact_type="inspection_trend",
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        as_of_date=latest["inspection_date"],
        values={
            "inspection_count": len(rows),
            "earliest_date": earliest["inspection_date"],
            "earliest_score": earliest.get("score"),
            "earliest_grade": earliest.get("grade"),
            "latest_date": latest["inspection_date"],
            "latest_score": latest.get("score"),
            "latest_grade": latest.get("grade"),
        },
        source_node_ids=(earliest["inspection_id"], latest["inspection_id"]),
        citation=_record_citation(
            restaurant_id, jurisdiction, latest["inspection_date"], "trend", excerpt
        ),
    )


def michelin_context_facts(
    graph: PlateProofGraph, restaurant_id: str, as_of_date: date
) -> tuple[RestaurantFact, ...]:
    restaurant = queries.restaurant_node(graph, restaurant_id)
    jurisdiction = parse_restaurant_jurisdiction((restaurant or {}).get("jurisdiction"))
    if jurisdiction is None:
        return ()
    events = queries.distinctions_as_of(graph, restaurant_id, as_of_date)
    facts: list[RestaurantFact] = []
    for event in events:
        announced = event["announced_date"]
        label = f"michelin_{event['michelin_distinction_event_id']}"
        excerpt = (
            f"Michelin {event['distinction']} recognition documented in the "
            f"{event['guide_year']} {event['guide_name']}, announced {announced.isoformat()}. "
            "This is contextual recognition, not a health-safety indicator."
        )
        facts.append(
            RestaurantFact(
                fact_type="michelin_distinction",
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                as_of_date=announced,
                values={
                    "distinction": event["distinction"],
                    "guide_name": event["guide_name"],
                    "guide_year": event["guide_year"],
                    "announced_date": announced,
                },
                source_node_ids=(event["michelin_distinction_event_id"],),
                citation=_record_citation(restaurant_id, jurisdiction, announced, label, excerpt),
            )
        )
    return tuple(facts)


def forecast_fact_from_row(
    restaurant_id: str, jurisdiction: RestaurantJurisdiction, row: Mapping[str, object]
) -> RestaurantFact:
    """Wraps one already-precomputed Task 7 prediction row as a typed
    fact. Never computes a prediction and never touches a model artifact
    -- the row must already come from
    ``plateproof.serving.prediction_service.resolve_prediction`` or an
    equivalent precomputed source."""
    as_of = row["as_of_date"]
    excerpt = (
        f"PlateProof forecast for restaurant {restaurant_id}, generated "
        f"{row['generated_at']}, model version {row['model_version']}. This is a "
        "statistical estimate, not a guarantee of any future inspection outcome."
    )
    citation = Citation(
        citation_id=f"forecast:{restaurant_id}:{row['model_version']}:{as_of}",
        evidence_type=EvidenceType.MODEL_FORECAST,
        title="PlateProof Inspection Risk Forecast",
        url=None,
        excerpt=excerpt,
        jurisdiction=jurisdiction,
        as_of_date=as_of,  # type: ignore[arg-type]
    )
    return RestaurantFact(
        fact_type="model_forecast",
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        as_of_date=as_of,  # type: ignore[arg-type]
        values=dict(row),
        source_node_ids=(),
        citation=citation,
    )
