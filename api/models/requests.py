"""
API Request Models.

Pydantic models for API request validation and serialization.
Based on OpenSpec schema definitions for event specifications.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class EventPriority(str, Enum):
    """Event processing priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class DataTypeCategory(str, Enum):
    """Semantic data categories."""

    OPTICAL = "optical"
    SAR = "sar"
    DEM = "dem"
    WEATHER = "weather"
    ANCILLARY = "ancillary"


class Polarization(str, Enum):
    """SAR polarization modes."""

    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"


class SortOrder(str, Enum):
    """Sort order for list queries."""

    ASC = "asc"
    DESC = "desc"


class EventSortField(str, Enum):
    """Fields available for sorting events."""

    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"
    PRIORITY = "priority"
    STATUS = "status"


class GeometryType(str, Enum):
    """GeoJSON geometry types."""

    POINT = "Point"
    LINESTRING = "LineString"
    POLYGON = "Polygon"
    MULTIPOINT = "MultiPoint"
    MULTILINESTRING = "MultiLineString"
    MULTIPOLYGON = "MultiPolygon"


class Geometry(BaseModel):
    """GeoJSON geometry object."""

    type: GeometryType = Field(..., description="Geometry type")
    coordinates: List[Any] = Field(..., description="Geometry coordinates")

    model_config = {"json_schema_extra": {"example": {
        "type": "Polygon",
        "coordinates": [[
            [-80.5, 25.5],
            [-80.0, 25.5],
            [-80.0, 26.0],
            [-80.5, 26.0],
            [-80.5, 25.5]
        ]]
    }}}


class BoundingBox(BaseModel):
    """Bounding box specification."""

    west: float = Field(..., ge=-180, le=180, description="Western longitude")
    south: float = Field(..., ge=-90, le=90, description="Southern latitude")
    east: float = Field(..., ge=-180, le=180, description="Eastern longitude")
    north: float = Field(..., ge=-90, le=90, description="Northern latitude")

    @model_validator(mode="after")
    def validate_bounds(self) -> "BoundingBox":
        """Validate that bounds are properly ordered."""
        if self.south > self.north:
            raise ValueError("South latitude must be less than north latitude")
        return self

    def to_list(self) -> List[float]:
        """Convert to [west, south, east, north] list."""
        return [self.west, self.south, self.east, self.north]


class TemporalExtent(BaseModel):
    """Time window specification with optional reference point."""

    start: datetime = Field(..., description="Start of time window")
    end: datetime = Field(..., description="End of time window")
    reference_time: Optional[datetime] = Field(
        default=None, description="Event peak or reference timestamp"
    )

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "TemporalExtent":
        """Validate that start is before end."""
        if self.start > self.end:
            raise ValueError("Start time must be before end time")
        if self.reference_time:
            if self.reference_time < self.start or self.reference_time > self.end:
                raise ValueError("Reference time must be within the time window")
        return self


class SpatialSpec(BaseModel):
    """Spatial specification for event area of interest."""

    geometry: Optional[Geometry] = Field(
        default=None, description="GeoJSON geometry defining the area"
    )
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box (alternative to geometry)"
    )
    crs: str = Field(
        default="EPSG:4326",
        pattern=r"^EPSG:[0-9]+$",
        description="Coordinate reference system",
    )

    @model_validator(mode="after")
    def require_spatial_definition(self) -> "SpatialSpec":
        """Require either geometry or bbox."""
        if self.geometry is None and self.bbox is None:
            raise ValueError("Either geometry or bbox must be provided")
        return self


class DataConstraints(BaseModel):
    """Data acquisition constraints for event processing."""

    max_cloud_cover: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Maximum acceptable cloud cover (0-1)",
    )
    min_resolution_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum acceptable resolution in meters",
    )
    max_resolution_m: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum acceptable resolution in meters",
    )
    required_bands: Optional[List[str]] = Field(
        default=None,
        description="Required spectral bands (e.g., ['nir', 'swir'])",
    )
    required_data_types: Optional[List[DataTypeCategory]] = Field(
        default=None,
        description="Required data type categories",
    )
    polarization: Optional[List[Polarization]] = Field(
        default=None,
        description="Required SAR polarization modes",
    )

    @model_validator(mode="after")
    def validate_resolution_order(self) -> "DataConstraints":
        """Validate that min resolution is less than max resolution."""
        if self.min_resolution_m and self.max_resolution_m:
            if self.min_resolution_m > self.max_resolution_m:
                raise ValueError(
                    "min_resolution_m must be less than or equal to max_resolution_m"
                )
        return self


class IntentSpec(BaseModel):
    """Event intent specification for type resolution."""

    event_class: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z]+\.[a-z_]+(\.[a-z_]+)*$",
        description="Explicit event class (e.g., 'flood.coastal.storm_surge')",
    )
    natural_language: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Natural language description for intent inference",
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional event-class-specific parameters",
    )

    @model_validator(mode="after")
    def require_intent_source(self) -> "IntentSpec":
        """Require either event_class or natural_language."""
        if self.event_class is None and self.natural_language is None:
            raise ValueError(
                "Either event_class or natural_language must be provided"
            )
        return self


class EventMetadata(BaseModel):
    """Optional event metadata."""

    name: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Human-readable event name",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Detailed event description",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        max_length=20,
        description="Tags for categorization and search",
    )
    external_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="External reference ID",
    )
    callback_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for status notifications",
    )


class EventSubmitRequest(BaseModel):
    """Request model for submitting a new event specification."""

    intent: IntentSpec = Field(..., description="Event intent specification")
    spatial: SpatialSpec = Field(..., description="Spatial area of interest")
    temporal: TemporalExtent = Field(..., description="Temporal extent")
    constraints: Optional[DataConstraints] = Field(
        default=None,
        description="Data acquisition constraints",
    )
    priority: EventPriority = Field(
        default=EventPriority.NORMAL,
        description="Processing priority",
    )
    metadata: Optional[EventMetadata] = Field(
        default=None,
        description="Optional event metadata",
    )

    model_config = {"json_schema_extra": {"example": {
        "intent": {
            "event_class": "flood.coastal.storm_surge",
            "parameters": {"causation": "tropical_cyclone"}
        },
        "spatial": {
            "bbox": {
                "west": -80.5,
                "south": 25.5,
                "east": -80.0,
                "north": 26.0
            },
            "crs": "EPSG:4326"
        },
        "temporal": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z",
            "reference_time": "2024-09-17T12:00:00Z"
        },
        "constraints": {
            "max_cloud_cover": 0.3,
            "required_data_types": ["sar", "dem"]
        },
        "priority": "high",
        "metadata": {
            "name": "Hurricane Milton Flooding",
            "tags": ["hurricane", "florida", "2024"]
        }
    }}}


class EventQueryParams(BaseModel):
    """Query parameters for listing events."""

    # Filtering
    status: Optional[List[str]] = Field(
        default=None,
        description="Filter by status (pending, processing, completed, failed)",
    )
    event_class: Optional[str] = Field(
        default=None,
        description="Filter by event class pattern (supports wildcards)",
    )
    priority: Optional[List[EventPriority]] = Field(
        default=None,
        description="Filter by priority levels",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags (any match)",
    )
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter events created after this time",
    )
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter events created before this time",
    )

    # Spatial filtering
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Filter by bounding box intersection",
    )

    # Pagination
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip",
    )

    # Sorting
    sort_by: EventSortField = Field(
        default=EventSortField.CREATED_AT,
        description="Field to sort by",
    )
    sort_order: SortOrder = Field(
        default=SortOrder.DESC,
        description="Sort order",
    )


class ProductQueryParams(BaseModel):
    """Query parameters for listing products."""

    product_type: Optional[List[str]] = Field(
        default=None,
        description="Filter by product type",
    )
    format: Optional[List[str]] = Field(
        default=None,
        description="Filter by output format (geotiff, cog, geojson, etc.)",
    )
    ready_only: bool = Field(
        default=False,
        description="Only return products that are ready for download",
    )

    # Pagination
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip",
    )


class ProductDownloadParams(BaseModel):
    """Parameters for product download."""

    format: Optional[str] = Field(
        default=None,
        description="Desired output format (if conversion is supported)",
    )
    crs: Optional[str] = Field(
        default=None,
        pattern=r"^EPSG:[0-9]+$",
        description="Target CRS for reprojection",
    )
    resolution: Optional[float] = Field(
        default=None,
        gt=0,
        description="Target resolution in meters",
    )
    clip_to_aoi: bool = Field(
        default=True,
        description="Clip product to original area of interest",
    )


class CatalogQueryParams(BaseModel):
    """Query parameters for catalog endpoints."""

    # Filtering
    data_type: Optional[List[DataTypeCategory]] = Field(
        default=None,
        description="Filter by data type category",
    )
    event_class: Optional[str] = Field(
        default=None,
        description="Filter by compatible event class",
    )
    available_only: bool = Field(
        default=False,
        description="Only return currently available sources",
    )

    # Pagination
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip",
    )
