"""
API Response Models.

Pydantic models for API response serialization.
Provides consistent response structure across all endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, computed_field

from .requests import (
    BoundingBox,
    DataConstraints,
    DataTypeCategory,
    EventPriority,
    Geometry,
    TemporalExtent,
)


class EventStatus(str, Enum):
    """Event processing status."""

    PENDING = "pending"
    QUEUED = "queued"
    DISCOVERING = "discovering"
    ACQUIRING = "acquiring"
    PROCESSING = "processing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProductStatus(str, Enum):
    """Product generation status."""

    PENDING = "pending"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    EXPIRED = "expired"


class ProductFormat(str, Enum):
    """Product output formats."""

    GEOTIFF = "geotiff"
    COG = "cog"
    NETCDF = "netcdf"
    ZARR = "zarr"
    GEOJSON = "geojson"
    GEOPARQUET = "geoparquet"
    PNG = "png"
    PDF = "pdf"


class QualityFlag(str, Enum):
    """Quality/confidence flags for products."""

    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"
    RESOLUTION_DEGRADED = "RESOLUTION_DEGRADED"
    SINGLE_SENSOR_MODE = "SINGLE_SENSOR_MODE"
    TEMPORALLY_INTERPOLATED = "TEMPORALLY_INTERPOLATED"
    HISTORICAL_PROXY = "HISTORICAL_PROXY"
    MISSING_OBSERVABLE = "MISSING_OBSERVABLE"
    FORECAST_DISCREPANCY = "FORECAST_DISCREPANCY"
    SPATIAL_UNCERTAINTY = "SPATIAL_UNCERTAINTY"
    MAGNITUDE_CONFLICT = "MAGNITUDE_CONFLICT"
    CONSERVATIVE_ESTIMATE = "CONSERVATIVE_ESTIMATE"


# Generic type for paginated responses
T = TypeVar("T")


class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Current offset")
    has_more: bool = Field(..., description="Whether more items exist")

    @computed_field
    @property
    def page(self) -> int:
        """Current page number (1-indexed)."""
        if self.limit == 0:
            return 1
        return (self.offset // self.limit) + 1

    @computed_field
    @property
    def total_pages(self) -> int:
        """Total number of pages."""
        if self.limit == 0:
            return 1
        return (self.total + self.limit - 1) // self.limit


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated response wrapper."""

    items: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")


class ResolvedIntent(BaseModel):
    """Resolved event intent information."""

    event_class: str = Field(
        ...,
        description="Resolved event class",
    )
    source: str = Field(
        ...,
        description="Resolution source: 'explicit' or 'inferred'",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Confidence score for inferred intents",
    )
    original_input: Optional[str] = Field(
        default=None,
        description="Original natural language input if inferred",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Extracted event parameters",
    )


class SpatialInfo(BaseModel):
    """Spatial information for an event."""

    geometry: Optional[Geometry] = Field(
        default=None,
        description="GeoJSON geometry",
    )
    bbox: BoundingBox = Field(..., description="Bounding box")
    crs: str = Field(default="EPSG:4326", description="Coordinate reference system")
    area_km2: Optional[float] = Field(
        default=None,
        description="Area in square kilometers",
    )


class EventSummary(BaseModel):
    """Summary event information for list responses."""

    id: str = Field(..., description="Unique event identifier")
    status: EventStatus = Field(..., description="Processing status")
    priority: EventPriority = Field(..., description="Processing priority")
    event_class: str = Field(..., description="Resolved event class")
    bbox: BoundingBox = Field(..., description="Bounding box")
    temporal: TemporalExtent = Field(..., description="Temporal extent")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    name: Optional[str] = Field(default=None, description="Event name")
    tags: Optional[List[str]] = Field(default=None, description="Event tags")
    product_count: int = Field(default=0, description="Number of products")


class EventResponse(BaseModel):
    """Full event details response."""

    id: str = Field(..., description="Unique event identifier")
    status: EventStatus = Field(..., description="Processing status")
    priority: EventPriority = Field(..., description="Processing priority")
    intent: ResolvedIntent = Field(..., description="Resolved intent information")
    spatial: SpatialInfo = Field(..., description="Spatial information")
    temporal: TemporalExtent = Field(..., description="Temporal extent")
    constraints: Optional[DataConstraints] = Field(
        default=None,
        description="Data constraints",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    started_at: Optional[datetime] = Field(
        default=None,
        description="Processing start timestamp",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Processing completion timestamp",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Event metadata",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is failed",
    )

    model_config = {"json_schema_extra": {"example": {
        "id": "evt_2024_florida_flood_001",
        "status": "completed",
        "priority": "high",
        "intent": {
            "event_class": "flood.coastal.storm_surge",
            "source": "explicit",
            "confidence": None,
            "parameters": {"causation": "tropical_cyclone"}
        },
        "spatial": {
            "bbox": {
                "west": -80.5,
                "south": 25.5,
                "east": -80.0,
                "north": 26.0
            },
            "crs": "EPSG:4326",
            "area_km2": 2500.0
        },
        "temporal": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z",
            "reference_time": "2024-09-17T12:00:00Z"
        },
        "created_at": "2024-09-16T08:00:00Z",
        "updated_at": "2024-09-17T14:30:00Z",
        "started_at": "2024-09-16T08:01:00Z",
        "completed_at": "2024-09-17T14:30:00Z"
    }}}


class PipelineStepProgress(BaseModel):
    """Progress information for a pipeline step."""

    step_id: str = Field(..., description="Step identifier")
    step_name: str = Field(..., description="Step name")
    status: str = Field(..., description="Step status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    progress_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Progress percentage",
    )
    message: Optional[str] = Field(default=None, description="Status message")


class StatusResponse(BaseModel):
    """Event processing status response."""

    event_id: str = Field(..., description="Event identifier")
    status: EventStatus = Field(..., description="Current status")
    progress_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall progress percentage",
    )
    current_stage: Optional[str] = Field(
        default=None,
        description="Current processing stage",
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message",
    )
    started_at: Optional[datetime] = Field(default=None)
    estimated_completion: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(..., description="Last update timestamp")


class ProgressResponse(BaseModel):
    """Detailed progress response with pipeline step information."""

    event_id: str = Field(..., description="Event identifier")
    status: EventStatus = Field(..., description="Current status")
    progress_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall progress percentage",
    )
    stages: List[PipelineStepProgress] = Field(
        default_factory=list,
        description="Progress for each pipeline stage",
    )
    data_sources_discovered: int = Field(
        default=0,
        description="Number of data sources discovered",
    )
    data_sources_acquired: int = Field(
        default=0,
        description="Number of data sources acquired",
    )
    products_generated: int = Field(
        default=0,
        description="Number of products generated",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Processing warnings",
    )
    started_at: Optional[datetime] = Field(default=None)
    estimated_completion: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(..., description="Last update timestamp")


class ProductMetadata(BaseModel):
    """Product metadata information."""

    format: ProductFormat = Field(..., description="Product format")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: Optional[float] = Field(
        default=None,
        description="Resolution in meters",
    )
    bbox: BoundingBox = Field(..., description="Product bounding box")
    file_size_bytes: int = Field(..., description="File size in bytes")
    checksum: Optional[str] = Field(
        default=None,
        description="SHA-256 checksum",
    )
    bands: Optional[List[str]] = Field(
        default=None,
        description="Band names (for raster products)",
    )
    statistics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Product statistics",
    )


class ProductQuality(BaseModel):
    """Product quality information."""

    overall_confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Overall confidence score",
    )
    flags: List[QualityFlag] = Field(
        default_factory=list,
        description="Quality flags",
    )
    validation_status: Optional[str] = Field(
        default=None,
        description="Validation status",
    )
    uncertainty_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Uncertainty metrics",
    )


class ProductResponse(BaseModel):
    """Product information response."""

    id: str = Field(..., description="Product identifier")
    event_id: str = Field(..., description="Parent event identifier")
    product_type: str = Field(..., description="Product type")
    name: str = Field(..., description="Product name")
    status: ProductStatus = Field(..., description="Product status")
    format: ProductFormat = Field(..., description="Output format")
    metadata: Optional[ProductMetadata] = Field(
        default=None,
        description="Product metadata",
    )
    quality: Optional[ProductQuality] = Field(
        default=None,
        description="Quality information",
    )
    download_url: Optional[str] = Field(
        default=None,
        description="Download URL (if ready)",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Download URL expiration",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    ready_at: Optional[datetime] = Field(
        default=None,
        description="Ready timestamp",
    )


class DataSourceInfo(BaseModel):
    """Data source information for catalog."""

    id: str = Field(..., description="Data source identifier")
    name: str = Field(..., description="Human-readable name")
    provider: str = Field(..., description="Data provider")
    data_type: DataTypeCategory = Field(..., description="Data type category")
    description: Optional[str] = Field(default=None, description="Description")
    resolution_m: Optional[float] = Field(
        default=None,
        description="Spatial resolution in meters",
    )
    revisit_days: Optional[int] = Field(
        default=None,
        description="Temporal revisit in days",
    )
    bands: Optional[List[str]] = Field(
        default=None,
        description="Available bands",
    )
    coverage: Optional[str] = Field(
        default=None,
        description="Geographic coverage description",
    )
    license: Optional[str] = Field(
        default=None,
        description="Data license",
    )
    open_data: bool = Field(
        default=False,
        description="Whether data is freely available",
    )
    applicable_event_classes: Optional[List[str]] = Field(
        default=None,
        description="Applicable event class patterns",
    )


class AlgorithmInfo(BaseModel):
    """Algorithm information for catalog."""

    id: str = Field(..., description="Algorithm identifier")
    name: str = Field(..., description="Human-readable name")
    category: str = Field(..., description="Category: baseline, advanced, experimental")
    version: str = Field(..., description="Algorithm version")
    description: Optional[str] = Field(default=None, description="Description")
    event_types: List[str] = Field(..., description="Supported event type patterns")
    required_data_types: List[DataTypeCategory] = Field(
        ...,
        description="Required data types",
    )
    outputs: Optional[List[str]] = Field(
        default=None,
        description="Output product types",
    )
    deterministic: bool = Field(
        default=True,
        description="Whether algorithm is deterministic",
    )


class EventTypeInfo(BaseModel):
    """Event type information for catalog."""

    class_name: str = Field(..., description="Event class name")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(default=None, description="Description")
    parent: Optional[str] = Field(
        default=None,
        description="Parent event class",
    )
    children: Optional[List[str]] = Field(
        default=None,
        description="Child event classes",
    )
    required_data_types: List[DataTypeCategory] = Field(
        default_factory=list,
        description="Required data types",
    )
    optional_data_types: List[DataTypeCategory] = Field(
        default_factory=list,
        description="Optional data types",
    )
    keywords: Optional[List[str]] = Field(
        default=None,
        description="Keywords for NLP matching",
    )


class CatalogSourcesResponse(BaseModel):
    """Response for available data sources."""

    items: List[DataSourceInfo] = Field(..., description="Data sources")
    total: int = Field(..., description="Total count")


class CatalogAlgorithmsResponse(BaseModel):
    """Response for available algorithms."""

    items: List[AlgorithmInfo] = Field(..., description="Algorithms")
    total: int = Field(..., description="Total count")


class CatalogEventTypesResponse(BaseModel):
    """Response for supported event types."""

    items: List[EventTypeInfo] = Field(..., description="Event types")
    total: int = Field(..., description="Total count")


class HealthCheckComponent(BaseModel):
    """Health check status for a component."""

    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Status: healthy, degraded, unhealthy")
    latency_ms: Optional[float] = Field(
        default=None,
        description="Response latency in milliseconds",
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Overall status")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment")
    timestamp: datetime = Field(..., description="Check timestamp")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool = Field(..., description="Whether service is ready")
    components: List[HealthCheckComponent] = Field(
        default_factory=list,
        description="Component statuses",
    )
    timestamp: datetime = Field(..., description="Check timestamp")


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    alive: bool = Field(..., description="Whether service is alive")
    uptime_seconds: float = Field(..., description="Uptime in seconds")
    timestamp: datetime = Field(..., description="Check timestamp")
