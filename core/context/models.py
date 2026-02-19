"""
Pydantic models for the Context Data Lakehouse.

All context record models use Dict[str, Any] for geometry fields (GeoJSON
dicts). Geometry conversion to/from PostGIS happens in the repository layer
using ST_GeomFromGeoJSON / ST_AsGeoJSON. Models remain pure Pydantic with
no PostGIS dependencies.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Context Record Models
# =============================================================================


class DatasetRecord(BaseModel):
    """A satellite scene discovered and used by the pipeline.

    Corresponds to a row in the ``datasets`` table.
    """

    source: str = Field(
        ...,
        description="Origin catalog (e.g., 'earth_search', 'planetary_computer')",
        max_length=100,
    )
    source_id: str = Field(
        ...,
        description="STAC Item ID or equivalent identifier from the source system",
        max_length=500,
    )
    geometry: Dict[str, Any] = Field(
        ...,
        description="Scene footprint as GeoJSON (Polygon or MultiPolygon)",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full STAC Item properties (platform, constellation, etc.)",
    )
    acquisition_date: datetime = Field(
        ...,
        description="Scene acquisition datetime",
    )
    cloud_cover: Optional[float] = Field(
        default=None,
        description="Cloud cover percentage (0-100)",
    )
    resolution_m: Optional[float] = Field(
        default=None,
        description="Ground sample distance in metres",
    )
    bands: Optional[List[str]] = Field(
        default=None,
        description="Available band names (e.g., ['B02', 'B03', 'B04', 'B08'])",
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Local path to downloaded data (if acquired)",
    )


class BuildingRecord(BaseModel):
    """A building footprint from OSM, Overture, or other sources.

    Corresponds to a row in the ``context_buildings`` table.
    """

    source: str = Field(
        ...,
        description="Data source (e.g., 'osm', 'overture', 'microsoft_footprints')",
        max_length=100,
    )
    source_id: str = Field(
        ...,
        description="Feature ID from the source system (e.g., OSM way ID)",
        max_length=500,
    )
    geometry: Dict[str, Any] = Field(
        ...,
        description="Building footprint as GeoJSON Polygon",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tags: building type, height, name, etc.",
    )


class InfrastructureRecord(BaseModel):
    """A critical infrastructure facility (hospital, fire station, etc.).

    Corresponds to a row in the ``context_infrastructure`` table.
    """

    source: str = Field(
        ...,
        description="Data source (e.g., 'osm', 'overture', 'hifld')",
        max_length=100,
    )
    source_id: str = Field(
        ...,
        description="Feature ID from the source system",
        max_length=500,
    )
    geometry: Dict[str, Any] = Field(
        ...,
        description="Facility geometry as GeoJSON (Point or Polygon)",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Type, name, capacity, etc.",
    )


class WeatherRecord(BaseModel):
    """A weather observation from NOAA, ERA5, or other meteorological sources.

    Corresponds to a row in the ``context_weather`` table.
    """

    source: str = Field(
        ...,
        description="Data source (e.g., 'noaa', 'era5', 'openmeteo')",
        max_length=100,
    )
    source_id: str = Field(
        ...,
        description="Station ID + timestamp hash or grid cell ID",
        max_length=500,
    )
    geometry: Dict[str, Any] = Field(
        ...,
        description="Observation location as GeoJSON Point",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Temperature, precipitation, wind, humidity, etc.",
    )
    observation_time: datetime = Field(
        ...,
        description="When the observation was recorded",
    )


# =============================================================================
# Result / Summary Models
# =============================================================================


class ContextResult(BaseModel):
    """Result of a context store operation.

    Returned by store_* methods to indicate whether the row was freshly
    ingested or reused from a previous job.
    """

    context_id: UUID = Field(
        ...,
        description="UUID of the context row in the relevant table",
    )
    usage_type: Literal["ingested", "reused"] = Field(
        ...,
        description="Whether this row was freshly ingested or reused from a prior job",
    )


class ContextSummary(BaseModel):
    """Context usage summary for a job.

    Aggregates counts from job_context_usage grouped by context_table and
    usage_type.
    """

    datasets_ingested: int = Field(default=0, description="Datasets freshly ingested")
    datasets_reused: int = Field(default=0, description="Datasets reused from prior jobs")
    buildings_ingested: int = Field(default=0, description="Buildings freshly ingested")
    buildings_reused: int = Field(default=0, description="Buildings reused from prior jobs")
    infrastructure_ingested: int = Field(default=0, description="Infrastructure freshly ingested")
    infrastructure_reused: int = Field(default=0, description="Infrastructure reused from prior jobs")
    weather_ingested: int = Field(default=0, description="Weather observations freshly ingested")
    weather_reused: int = Field(default=0, description="Weather observations reused from prior jobs")
    total_ingested: int = Field(default=0, description="Total items freshly ingested")
    total_reused: int = Field(default=0, description="Total items reused from prior jobs")
    total: int = Field(default=0, description="Grand total of all context items")


class JobContextUsage(BaseModel):
    """A single row from the job_context_usage junction table."""

    job_id: UUID = Field(..., description="The consuming job")
    context_table: str = Field(
        ...,
        description="Which context table ('datasets', 'context_buildings', etc.)",
    )
    context_id: UUID = Field(
        ...,
        description="ID of the row in the context table",
    )
    usage_type: Literal["ingested", "reused"] = Field(
        ...,
        description="Was this freshly fetched or reused?",
    )
    linked_at: datetime = Field(
        ...,
        description="When the link was created",
    )
