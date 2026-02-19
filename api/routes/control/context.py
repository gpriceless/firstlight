"""
Context Data Lakehouse Query Endpoints.

Provides read-only endpoints for querying accumulated context data
(datasets, buildings, infrastructure, weather) and lakehouse statistics.

All endpoints are under /control/v1/context and require CONTEXT_READ permission.
"""

import logging
from datetime import datetime
from typing import Annotated, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request

from api.models.context import (
    BuildingItem,
    DatasetItem,
    InfrastructureItem,
    LakehouseSummaryResponse,
    PaginatedBuildingsResponse,
    PaginatedDatasetsResponse,
    PaginatedInfrastructureResponse,
    PaginatedWeatherResponse,
    TableStats,
    WeatherItem,
)
from api.models.errors import ValidationError
from api.routes.control import get_current_customer

logger = logging.getLogger(__name__)

context_router = APIRouter(prefix="/context", tags=["LLM Control - Context Lakehouse"])


# =============================================================================
# Backend Helpers
# =============================================================================


def _get_context_repo():
    """
    Get a ContextRepository instance.

    Uses the same DB connection parameters as the PostGIS state backend,
    sourced from api/config.py DatabaseSettings.
    """
    from core.context.repository import ContextRepository
    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    repo = ContextRepository(
        host=db.host,
        port=db.port,
        database=db.name,
        user=db.user,
        password=db.password,
    )
    return repo


async def _get_connected_repo():
    """Get a connected ContextRepository instance."""
    repo = _get_context_repo()
    await repo.connect()
    return repo


def _parse_bbox(bbox: Optional[str]):
    """
    Parse and validate a bbox query parameter.

    Args:
        bbox: Comma-separated string "west,south,east,north" in WGS84.

    Returns:
        Tuple of (west, south, east, north) or None if bbox is None.

    Raises:
        ValidationError if the bbox is malformed.
    """
    if bbox is None:
        return None

    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
    except (ValueError, AttributeError):
        raise ValidationError(
            message="bbox must be four comma-separated numbers: west,south,east,north"
        )
    if len(parts) != 4:
        raise ValidationError(
            message="bbox must be four comma-separated numbers: west,south,east,north"
        )
    west, south, east, north = parts
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValidationError(
            message="bbox longitude values must be between -180 and 180"
        )
    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValidationError(
            message="bbox latitude values must be between -90 and 90"
        )
    if south > north:
        raise ValidationError(
            message="bbox south must be less than or equal to north"
        )
    return (west, south, east, north)


# =============================================================================
# Task 3.2: GET /context/datasets
# =============================================================================


@context_router.get(
    "/datasets",
    response_model=PaginatedDatasetsResponse,
    summary="Query accumulated datasets",
    description=(
        "Query satellite scene datasets with spatial, temporal, and source filters. "
        "Returns paginated results with GeoJSON geometry."
    ),
)
async def list_datasets(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
    bbox: Annotated[
        Optional[str],
        Query(
            description="Bounding box filter as west,south,east,north in WGS84"
        ),
    ] = None,
    date_start: Annotated[
        Optional[datetime],
        Query(description="Minimum acquisition date (inclusive)"),
    ] = None,
    date_end: Annotated[
        Optional[datetime],
        Query(description="Maximum acquisition date (inclusive)"),
    ] = None,
    source: Annotated[
        Optional[str],
        Query(description="Filter by source catalog"),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)"),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=200, description="Items per page (max 200)"),
    ] = 50,
) -> PaginatedDatasetsResponse:
    """Query accumulated datasets with optional filters."""
    bbox_values = _parse_bbox(bbox)
    offset = (page - 1) * page_size

    repo = await _get_connected_repo()
    try:
        records, total = await repo.query_datasets(
            bbox=bbox_values,
            date_start=date_start,
            date_end=date_end,
            source=source,
            limit=page_size,
            offset=offset,
        )

        items = [
            DatasetItem(
                source=r.source,
                source_id=r.source_id,
                geometry=r.geometry,
                properties=r.properties,
                acquisition_date=r.acquisition_date,
                cloud_cover=r.cloud_cover,
                resolution_m=r.resolution_m,
                bands=r.bands,
                file_path=r.file_path,
            )
            for r in records
        ]

        return PaginatedDatasetsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        await repo.close()


# =============================================================================
# Task 3.3: GET /context/buildings
# =============================================================================


@context_router.get(
    "/buildings",
    response_model=PaginatedBuildingsResponse,
    summary="Query building footprints",
    description="Query building footprints with spatial filter.",
)
async def list_buildings(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
    bbox: Annotated[
        Optional[str],
        Query(description="Bounding box filter as west,south,east,north in WGS84"),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)"),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=200, description="Items per page (max 200)"),
    ] = 50,
) -> PaginatedBuildingsResponse:
    """Query building footprints by bbox."""
    bbox_values = _parse_bbox(bbox)
    offset = (page - 1) * page_size

    repo = await _get_connected_repo()
    try:
        records, total = await repo.query_buildings(
            bbox=bbox_values,
            limit=page_size,
            offset=offset,
        )

        items = [
            BuildingItem(
                source=r.source,
                source_id=r.source_id,
                geometry=r.geometry,
                properties=r.properties,
            )
            for r in records
        ]

        return PaginatedBuildingsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        await repo.close()


# =============================================================================
# Task 3.4: GET /context/infrastructure
# =============================================================================


@context_router.get(
    "/infrastructure",
    response_model=PaginatedInfrastructureResponse,
    summary="Query infrastructure facilities",
    description=(
        "Query critical infrastructure facilities with spatial and type filters."
    ),
)
async def list_infrastructure(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
    bbox: Annotated[
        Optional[str],
        Query(description="Bounding box filter as west,south,east,north in WGS84"),
    ] = None,
    type: Annotated[
        Optional[str],
        Query(
            description="Filter by infrastructure type (e.g., 'hospital', 'fire_station')"
        ),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)"),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=200, description="Items per page (max 200)"),
    ] = 50,
) -> PaginatedInfrastructureResponse:
    """Query infrastructure facilities by bbox and type."""
    bbox_values = _parse_bbox(bbox)
    offset = (page - 1) * page_size

    repo = await _get_connected_repo()
    try:
        records, total = await repo.query_infrastructure(
            bbox=bbox_values,
            type_filter=type,
            limit=page_size,
            offset=offset,
        )

        items = [
            InfrastructureItem(
                source=r.source,
                source_id=r.source_id,
                geometry=r.geometry,
                properties=r.properties,
            )
            for r in records
        ]

        return PaginatedInfrastructureResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        await repo.close()


# =============================================================================
# Task 3.5: GET /context/weather
# =============================================================================


@context_router.get(
    "/weather",
    response_model=PaginatedWeatherResponse,
    summary="Query weather observations",
    description="Query weather observations with spatial and temporal filters.",
)
async def list_weather(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
    bbox: Annotated[
        Optional[str],
        Query(description="Bounding box filter as west,south,east,north in WGS84"),
    ] = None,
    time_start: Annotated[
        Optional[datetime],
        Query(description="Minimum observation time (inclusive)"),
    ] = None,
    time_end: Annotated[
        Optional[datetime],
        Query(description="Maximum observation time (inclusive)"),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)"),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=200, description="Items per page (max 200)"),
    ] = 50,
) -> PaginatedWeatherResponse:
    """Query weather observations by bbox and time range."""
    bbox_values = _parse_bbox(bbox)
    offset = (page - 1) * page_size

    repo = await _get_connected_repo()
    try:
        records, total = await repo.query_weather(
            bbox=bbox_values,
            time_start=time_start,
            time_end=time_end,
            limit=page_size,
            offset=offset,
        )

        items = [
            WeatherItem(
                source=r.source,
                source_id=r.source_id,
                geometry=r.geometry,
                properties=r.properties,
                observation_time=r.observation_time,
            )
            for r in records
        ]

        return PaginatedWeatherResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        await repo.close()


# =============================================================================
# Task 3.6: GET /context/summary
# =============================================================================


@context_router.get(
    "/summary",
    response_model=LakehouseSummaryResponse,
    summary="Get lakehouse statistics",
    description=(
        "Returns overall lakehouse statistics: row counts per table, "
        "spatial extent, distinct sources, and usage statistics."
    ),
)
async def get_summary(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> LakehouseSummaryResponse:
    """Get lakehouse-wide statistics."""
    repo = await _get_connected_repo()
    try:
        stats = await repo.get_lakehouse_stats()

        # Convert raw dict to response model
        tables = {}
        for label, info in stats.get("tables", {}).items():
            tables[label] = TableStats(
                row_count=info["row_count"],
                sources=info.get("sources", []),
            )

        return LakehouseSummaryResponse(
            tables=tables,
            total_rows=stats.get("total_rows", 0),
            spatial_extent=stats.get("spatial_extent"),
            usage_stats=stats.get("usage_stats", {}),
        )
    finally:
        await repo.close()
