"""
Context Data Lakehouse API Response Models.

Pydantic models for the context query endpoints under /control/v1/context.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Dataset Response Models
# =============================================================================


class DatasetItem(BaseModel):
    """A dataset record in API responses."""

    source: str = Field(..., description="Origin catalog (e.g., 'earth_search')")
    source_id: str = Field(..., description="STAC Item ID or equivalent")
    geometry: Dict[str, Any] = Field(..., description="Scene footprint as GeoJSON")
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="STAC properties"
    )
    acquisition_date: datetime = Field(..., description="Scene acquisition datetime")
    cloud_cover: Optional[float] = Field(
        default=None, description="Cloud cover percentage"
    )
    resolution_m: Optional[float] = Field(
        default=None, description="Ground sample distance in metres"
    )
    bands: Optional[List[str]] = Field(
        default=None, description="Available band names"
    )
    file_path: Optional[str] = Field(
        default=None, description="Local path to downloaded data"
    )


class PaginatedDatasetsResponse(BaseModel):
    """Paginated response for dataset listing."""

    items: List[DatasetItem] = Field(..., description="List of dataset records")
    total: int = Field(..., description="Total number of matching datasets")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")


# =============================================================================
# Building Response Models
# =============================================================================


class BuildingItem(BaseModel):
    """A building footprint record in API responses."""

    source: str = Field(..., description="Data source (e.g., 'osm', 'overture')")
    source_id: str = Field(..., description="Feature ID from the source")
    geometry: Dict[str, Any] = Field(
        ..., description="Building footprint as GeoJSON"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Building tags"
    )


class PaginatedBuildingsResponse(BaseModel):
    """Paginated response for building listing."""

    items: List[BuildingItem] = Field(..., description="List of building records")
    total: int = Field(..., description="Total number of matching buildings")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")


# =============================================================================
# Infrastructure Response Models
# =============================================================================


class InfrastructureItem(BaseModel):
    """An infrastructure facility record in API responses."""

    source: str = Field(..., description="Data source")
    source_id: str = Field(..., description="Feature ID from the source")
    geometry: Dict[str, Any] = Field(
        ..., description="Facility geometry as GeoJSON"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Type, name, capacity, etc."
    )


class PaginatedInfrastructureResponse(BaseModel):
    """Paginated response for infrastructure listing."""

    items: List[InfrastructureItem] = Field(
        ..., description="List of infrastructure records"
    )
    total: int = Field(..., description="Total number of matching facilities")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")


# =============================================================================
# Weather Response Models
# =============================================================================


class WeatherItem(BaseModel):
    """A weather observation record in API responses."""

    source: str = Field(..., description="Data source (e.g., 'noaa', 'era5')")
    source_id: str = Field(..., description="Station/grid ID")
    geometry: Dict[str, Any] = Field(
        ..., description="Observation location as GeoJSON"
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Temperature, precipitation, etc."
    )
    observation_time: datetime = Field(
        ..., description="When the observation was recorded"
    )


class PaginatedWeatherResponse(BaseModel):
    """Paginated response for weather listing."""

    items: List[WeatherItem] = Field(
        ..., description="List of weather observation records"
    )
    total: int = Field(..., description="Total number of matching observations")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")


# =============================================================================
# Summary / Stats Response Models
# =============================================================================


class TableStats(BaseModel):
    """Statistics for a single context table."""

    row_count: int = Field(..., description="Total rows in this table")
    sources: List[str] = Field(
        default_factory=list, description="Distinct data sources"
    )


class LakehouseSummaryResponse(BaseModel):
    """Lakehouse-wide statistics."""

    tables: Dict[str, TableStats] = Field(
        ..., description="Per-table statistics"
    )
    total_rows: int = Field(..., description="Total rows across all tables")
    spatial_extent: Optional[Dict[str, Any]] = Field(
        default=None, description="Bounding box of all data as GeoJSON"
    )
    usage_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="Usage counts grouped by type (ingested, reused)",
    )


class JobContextSummaryResponse(BaseModel):
    """Context usage summary for a specific job."""

    job_id: str = Field(..., description="Job identifier")
    datasets_ingested: int = Field(default=0, description="Datasets freshly ingested")
    datasets_reused: int = Field(default=0, description="Datasets reused")
    buildings_ingested: int = Field(default=0, description="Buildings freshly ingested")
    buildings_reused: int = Field(default=0, description="Buildings reused")
    infrastructure_ingested: int = Field(
        default=0, description="Infrastructure freshly ingested"
    )
    infrastructure_reused: int = Field(default=0, description="Infrastructure reused")
    weather_ingested: int = Field(
        default=0, description="Weather observations freshly ingested"
    )
    weather_reused: int = Field(default=0, description="Weather observations reused")
    total_ingested: int = Field(default=0, description="Total items freshly ingested")
    total_reused: int = Field(default=0, description="Total items reused")
    total: int = Field(default=0, description="Grand total of all context items")
