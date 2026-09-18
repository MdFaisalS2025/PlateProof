"""Restaurant search, detail, inspection history, and violation history
routes. All SQL lives in the repository; routes only translate between HTTP
and the repository's typed methods."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from plateproof.api.dependencies import get_app_settings, get_repository
from plateproof.api.schemas import (
    InspectionHistoryItem,
    MichelinDistinctionItem,
    PaginationMeta,
    RestaurantDetail,
    RestaurantSearchResponse,
    RestaurantSummary,
    ViolationHistoryItem,
)
from plateproof.core.config import Settings
from plateproof.serving.display import (
    GOOGLE_SEARCH_LINK_ATTRIBUTION,
    MICHELIN_CONTEXT_NOTE,
    google_maps_search_link,
    latest_documented_distinctions,
    official_source_link,
)
from plateproof.serving.errors import RestaurantNotFoundError
from plateproof.serving.repository import Repository

router = APIRouter()


def _to_summary(repository: Repository, row: dict[str, object]) -> RestaurantSummary:
    restaurant_id = str(row["restaurant_id"])
    history = repository.michelin_history(restaurant_id)
    labels, guide_year = latest_documented_distinctions(history)
    return RestaurantSummary(
        restaurant_id=restaurant_id,
        jurisdiction=row["jurisdiction"],
        name=str(row.get("name") or ""),
        address=row.get("address"),
        city=row.get("city"),
        region=row.get("region"),
        cuisine=row.get("cuisine"),
        latest_documented_michelin_distinctions=labels,
        latest_documented_guide_year=guide_year,
    )


@router.get("/restaurants", response_model=RestaurantSearchResponse)
def search_restaurants(
    query: str = "",
    jurisdiction: str | None = None,
    michelin_category: str | None = None,
    michelin_guide_year: int | None = None,
    limit: int = Query(20),
    offset: int = Query(0),
    repository: Repository = Depends(get_repository),
) -> RestaurantSearchResponse:
    result = repository.search_restaurants(
        query=query,
        jurisdiction=jurisdiction,
        michelin_category=michelin_category,
        michelin_guide_year=michelin_guide_year,
        limit=limit,
        offset=offset,
    )
    summaries = [_to_summary(repository, row) for row in result.results]
    pagination = PaginationMeta(
        total=result.total,
        limit=result.limit,
        offset=result.offset,
        has_more=(result.offset + len(result.results)) < result.total,
    )
    return RestaurantSearchResponse(
        results=summaries,
        pagination=pagination,
        filters_applied=result.filters_applied,
        warnings=result.warnings,
        optional_data_status=result.optional_data_status,
    )


@router.get("/restaurants/{restaurant_id}", response_model=RestaurantDetail)
def get_restaurant_detail(
    restaurant_id: str,
    repository: Repository = Depends(get_repository),
    settings: Settings = Depends(get_app_settings),
) -> RestaurantDetail:
    row = repository.get_restaurant(restaurant_id)
    if row is None:
        raise RestaurantNotFoundError(restaurant_id)
    history = repository.michelin_history(restaurant_id)
    summary = _to_summary(repository, row)

    # Task 10: a constructed Google Maps search link, never a Google API
    # call -- see plateproof.serving.display.google_maps_search_link. Both
    # fields stay None when the flag is off (the default) or when no
    # usable link could be built, so the response shape is identical
    # either way (only the values differ).
    google_search_link: str | None = None
    google_attribution: str | None = None
    if settings.google_integration_enabled:
        google_search_link = google_maps_search_link(
            name=summary.name, address=summary.address, city=summary.city, region=summary.region
        )
        if google_search_link is not None:
            google_attribution = GOOGLE_SEARCH_LINK_ATTRIBUTION

    return RestaurantDetail(
        restaurant=summary,
        official_source_links=[official_source_link(row["jurisdiction"])],
        michelin_history=[MichelinDistinctionItem(**item) for item in history],
        michelin_context_note=MICHELIN_CONTEXT_NOTE if history else None,
        google_search_link=google_search_link,
        google_attribution=google_attribution,
    )


@router.get("/restaurants/{restaurant_id}/inspections", response_model=list[InspectionHistoryItem])
def get_inspections(
    restaurant_id: str, repository: Repository = Depends(get_repository)
) -> list[InspectionHistoryItem]:
    if repository.get_restaurant(restaurant_id) is None:
        raise RestaurantNotFoundError(restaurant_id)
    rows = repository.list_inspections(restaurant_id)
    return [InspectionHistoryItem(**row) for row in rows]


@router.get("/restaurants/{restaurant_id}/violations", response_model=list[ViolationHistoryItem])
def get_violations(
    restaurant_id: str,
    inspection_id: str | None = None,
    repository: Repository = Depends(get_repository),
) -> list[ViolationHistoryItem]:
    if repository.get_restaurant(restaurant_id) is None:
        raise RestaurantNotFoundError(restaurant_id)
    rows = repository.list_violations(restaurant_id, inspection_id)
    return [
        ViolationHistoryItem(
            inspection_id=row["inspection_id"],
            inspection_date=row["inspection_date"],
            violation_code=row["violation_code"],
            description=row.get("violation_description"),
            severity=row.get("severity"),
            count=row.get("count") or 0,
            corrected_on_site=row.get("corrected_on_site"),
        )
        for row in rows
    ]
