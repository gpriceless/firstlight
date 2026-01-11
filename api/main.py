"""
Multiverse Dive API - Geospatial Event Intelligence Platform.

This FastAPI application provides RESTful endpoints for:
- Event submission and retrieval
- Status monitoring and progress tracking
- Product download and metadata
- Data catalog browsing
- Health checks and readiness probes
- Webhook registration and management
- Authentication and rate limiting

See /docs for interactive Swagger documentation.
"""

import asyncio
import hashlib
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Header,
    Query,
    Request,
    BackgroundTasks,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Application Configuration
# ============================================================================

# OpenAPI metadata
API_TITLE = "Multiverse Dive API"
API_DESCRIPTION = """
## Geospatial Event Intelligence Platform

Multiverse Dive transforms (area, time window, event type) specifications into
actionable decision products for emergency response and hazard monitoring.

### Key Features

- **Situation-agnostic**: Same pipelines handle floods, wildfires, and storms
- **Reproducible**: Full provenance tracking from input to output
- **Intelligent**: Automated data discovery and algorithm selection
- **Quality-assured**: Multi-level validation and uncertainty quantification

### Quick Start

1. Obtain an API key
2. Submit an event specification to `/events`
3. Monitor progress via `/events/{id}/status`
4. Download products from `/events/{id}/products`

### Authentication

All endpoints (except `/health`) require an API key passed in the `X-API-Key` header.

### Rate Limiting

API requests are rate-limited per API key:
- Standard tier: 100 requests/minute
- Premium tier: 1000 requests/minute

### Webhooks

Register webhooks to receive real-time notifications when events complete or fail.
"""

API_VERSION = "1.0.0"

TAGS_METADATA = [
    {
        "name": "Events",
        "description": "Submit and manage event specifications. Events define the area, "
                       "time window, and type of hazard to analyze.",
    },
    {
        "name": "Status",
        "description": "Monitor event processing status and progress through pipeline phases.",
    },
    {
        "name": "Products",
        "description": "Access analysis outputs including raster data, vector files, and reports.",
    },
    {
        "name": "Catalog",
        "description": "Browse available data sources, algorithms, and supported event types.",
    },
    {
        "name": "Health",
        "description": "Health checks, readiness, and liveness probes for monitoring.",
    },
    {
        "name": "Webhooks",
        "description": "Register webhooks for event notifications and delivery tracking.",
    },
]

CONTACT_INFO = {
    "name": "Multiverse Dive Support",
    "url": "https://github.com/gpriceless/multiverse_dive",
    "email": "support@multiverse-dive.io",
}

LICENSE_INFO = {
    "name": "Apache 2.0",
    "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
}


# ============================================================================
# Request/Response Models
# ============================================================================

class IntentModel(BaseModel):
    """Event intent specification."""
    event_class: str = Field(
        ...,
        alias="class",
        description="Event class (e.g., flood.coastal.storm_surge)",
        pattern=r"^[a-z]+\.[a-z_]+(\.[a-z_]+)*$",
        examples=["flood.coastal.storm_surge", "wildfire.forest.active"]
    )
    source: str = Field(
        default="explicit",
        description="How intent was determined",
        examples=["explicit", "inferred"]
    )
    original_input: Optional[str] = Field(
        default=None,
        description="Original natural language input if inferred"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for inferred intents"
    )

    model_config = {"populate_by_name": True}


class SpatialModel(BaseModel):
    """Spatial extent specification."""
    type: str = Field(
        ...,
        description="GeoJSON geometry type",
        examples=["Polygon", "MultiPolygon"]
    )
    coordinates: List = Field(
        ...,
        description="GeoJSON coordinates array"
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate reference system",
        pattern=r"^EPSG:\d+$"
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Bounding box [minx, miny, maxx, maxy]"
    )


class TemporalModel(BaseModel):
    """Temporal window specification."""
    start: str = Field(
        ...,
        description="Start time (ISO 8601 format)",
        examples=["2024-09-15T00:00:00Z"]
    )
    end: str = Field(
        ...,
        description="End time (ISO 8601 format)",
        examples=["2024-09-20T23:59:59Z"]
    )
    reference_time: Optional[str] = Field(
        default=None,
        description="Event peak or reference timestamp"
    )


class ConstraintsModel(BaseModel):
    """Data acquisition constraints."""
    max_cloud_cover: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable cloud cover (0-1)"
    )
    min_resolution_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum spatial resolution in meters"
    )
    max_resolution_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum spatial resolution in meters"
    )
    required_data_types: Optional[List[str]] = Field(
        default=None,
        description="Required data types",
        examples=[["sar", "dem", "weather"]]
    )
    optional_data_types: Optional[List[str]] = Field(
        default=None,
        description="Optional data types"
    )
    polarization: Optional[List[str]] = Field(
        default=None,
        description="Required SAR polarizations",
        examples=[["VV", "VH"]]
    )


class MetadataModel(BaseModel):
    """Event metadata."""
    created_at: Optional[str] = Field(
        default=None,
        description="Creation timestamp"
    )
    created_by: Optional[str] = Field(
        default=None,
        description="Creator identifier"
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Classification tags",
        examples=[["hurricane", "miami", "critical"]]
    )


class EventSubmission(BaseModel):
    """Event submission request body."""
    id: Optional[str] = Field(
        default=None,
        description="Event ID (auto-generated if not provided)",
        pattern=r"^evt_[a-z0-9_]+$"
    )
    intent: IntentModel = Field(
        ...,
        description="Event intent specification"
    )
    spatial: SpatialModel = Field(
        ...,
        description="Area of interest"
    )
    temporal: TemporalModel = Field(
        ...,
        description="Time window"
    )
    priority: str = Field(
        default="normal",
        description="Processing priority",
        examples=["critical", "high", "normal", "low"]
    )
    constraints: Optional[ConstraintsModel] = Field(
        default=None,
        description="Data acquisition constraints"
    )
    metadata: Optional[MetadataModel] = Field(
        default=None,
        description="Additional metadata"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "evt_miami_flood_001",
                    "intent": {
                        "class": "flood.coastal.storm_surge",
                        "source": "explicit",
                        "confidence": 0.95
                    },
                    "spatial": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-80.3, 25.7], [-80.1, 25.7],
                            [-80.1, 25.9], [-80.3, 25.9],
                            [-80.3, 25.7]
                        ]],
                        "crs": "EPSG:4326"
                    },
                    "temporal": {
                        "start": "2024-09-15T00:00:00Z",
                        "end": "2024-09-20T23:59:59Z"
                    },
                    "priority": "high",
                    "constraints": {
                        "max_cloud_cover": 0.4,
                        "required_data_types": ["sar", "dem"]
                    }
                }
            ]
        }
    }


class EventResponse(BaseModel):
    """Event submission response."""
    event_id: str = Field(description="Assigned event ID")
    status: str = Field(description="Submission status")
    message: str = Field(description="Status message")
    links: Dict[str, str] = Field(description="Related endpoint links")


class StatusResponse(BaseModel):
    """Event status response."""
    event_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    updated_at: str


class ProgressPhase(BaseModel):
    """Progress for a single phase."""
    status: str
    progress: float = Field(ge=0.0, le=1.0)


class ProgressResponse(BaseModel):
    """Detailed progress response."""
    event_id: str
    status: str
    progress: float
    phases: Dict[str, ProgressPhase]
    estimated_completion: Optional[str] = None


class ProductMetadata(BaseModel):
    """Product metadata."""
    id: str
    name: str
    type: str
    format: str
    size_bytes: int
    created_at: str


class ProductListResponse(BaseModel):
    """Product list response."""
    event_id: str
    products: List[ProductMetadata]
    total: int


class WebhookRegistration(BaseModel):
    """Webhook registration request."""
    url: str = Field(
        ...,
        description="Webhook URL to receive notifications",
        examples=["https://example.com/webhook"]
    )
    events: List[str] = Field(
        default=["*"],
        description="Event types to subscribe to",
        examples=[["event.completed", "event.failed"]]
    )
    secret: Optional[str] = Field(
        default=None,
        description="Secret for HMAC signature verification"
    )
    active: bool = Field(
        default=True,
        description="Whether webhook is active"
    )


class WebhookResponse(BaseModel):
    """Webhook registration response."""
    id: str
    url: str
    events: List[str]
    active: bool
    message: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    error_code: Optional[str] = None


# ============================================================================
# Application Setup
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=TAGS_METADATA,
    contact=CONTACT_INFO,
    license_info=LICENSE_INFO,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# In-Memory Storage (Replace with database in production)
# ============================================================================

events_db: Dict[str, Dict[str, Any]] = {}
webhooks_db: Dict[str, Dict[str, Any]] = {}
rate_limit_tracker: Dict[str, List[float]] = {}


# ============================================================================
# Configuration
# ============================================================================

def get_api_keys() -> set:
    """Get valid API keys from environment or config."""
    keys_str = os.environ.get("API_KEYS", "")
    if keys_str:
        return set(keys_str.split(","))
    # Default key for development
    return {"dev_api_key_12345", "test_api_key"}


def get_rate_limit() -> int:
    """Get rate limit from environment."""
    return int(os.environ.get("RATE_LIMIT_PER_MINUTE", "100"))


def get_products_dir() -> Path:
    """Get products directory path."""
    return Path(os.environ.get("PRODUCTS_DIR", "./products"))


# ============================================================================
# Dependencies
# ============================================================================

async def verify_api_key(
    x_api_key: Optional[str] = Header(None, description="API key for authentication")
) -> str:
    """Verify API key from header."""
    valid_keys = get_api_keys()

    if not valid_keys:
        return "anonymous"

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Pass it in X-API-Key header.",
        )

    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return x_api_key


async def check_rate_limit(request: Request) -> bool:
    """Check rate limiting."""
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()
    window_start = current_time - 60

    if client_ip not in rate_limit_tracker:
        rate_limit_tracker[client_ip] = []

    # Clean old entries
    rate_limit_tracker[client_ip] = [
        t for t in rate_limit_tracker[client_ip] if t > window_start
    ]

    rate_limit = get_rate_limit()
    if len(rate_limit_tracker[client_ip]) >= rate_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded ({rate_limit} requests/minute)",
            headers={"Retry-After": "60"},
        )

    rate_limit_tracker[client_ip].append(current_time)
    return True


# ============================================================================
# Event Endpoints
# ============================================================================

@app.post(
    "/events",
    response_model=EventResponse,
    tags=["Events"],
    summary="Submit a new event",
    description="Submit an event specification for processing. "
                "The system will automatically discover data, select algorithms, "
                "and generate analysis products.",
    responses={
        200: {"description": "Event accepted for processing"},
        401: {"description": "Missing API key"},
        403: {"description": "Invalid API key"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def submit_event(
    event: EventSubmission,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
    _rate_limit: bool = Depends(check_rate_limit),
):
    """
    Submit a new event for processing.

    The event specification defines:
    - **Intent**: What type of hazard to analyze
    - **Spatial**: The area of interest (GeoJSON polygon)
    - **Temporal**: The time window for analysis
    - **Constraints**: Data requirements and quality thresholds

    Returns an event ID for tracking status and retrieving products.
    """
    event_id = event.id or f"evt_{uuid.uuid4().hex[:16]}"

    event_data = {
        "id": event_id,
        "intent": event.intent.model_dump(by_alias=True),
        "spatial": event.spatial.model_dump(),
        "temporal": event.temporal.model_dump(),
        "priority": event.priority,
        "constraints": event.constraints.model_dump() if event.constraints else None,
        "metadata": event.metadata.model_dump() if event.metadata else None,
        "status": "submitted",
        "progress": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    events_db[event_id] = event_data

    logger.info(f"Event submitted: {event_id}")

    # Start background processing (placeholder)
    background_tasks.add_task(process_event_background, event_id)

    return EventResponse(
        event_id=event_id,
        status="accepted",
        message="Event submitted successfully. Use /events/{event_id}/status to monitor progress.",
        links={
            "self": f"/events/{event_id}",
            "status": f"/events/{event_id}/status",
            "progress": f"/events/{event_id}/progress",
            "products": f"/events/{event_id}/products",
            "cancel": f"/events/{event_id}/cancel",
        },
    )


async def process_event_background(event_id: str):
    """Background task to process event (placeholder)."""
    # This would integrate with the agent orchestrator
    logger.info(f"Background processing started for {event_id}")
    await asyncio.sleep(0.1)  # Placeholder


@app.get(
    "/events/{event_id}",
    tags=["Events"],
    summary="Get event details",
    description="Retrieve full details of an event by ID.",
    responses={
        200: {"description": "Event details"},
        404: {"description": "Event not found"},
    },
)
async def get_event(
    event_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get event details by ID."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return events_db[event_id]


@app.get(
    "/events",
    tags=["Events"],
    summary="List events",
    description="List events with optional filtering and pagination.",
)
async def list_events(
    event_status: Optional[str] = Query(None, alias="status", description="Filter by status"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    limit: int = Query(10, ge=1, le=100, description="Maximum results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    api_key: str = Depends(verify_api_key),
):
    """List events with optional filtering."""
    events = list(events_db.values())

    if event_status:
        events = [e for e in events if e.get("status") == event_status]
    if priority:
        events = [e for e in events if e.get("priority") == priority]

    total = len(events)
    events = events[offset:offset + limit]

    return {
        "events": events,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post(
    "/events/{event_id}/cancel",
    tags=["Events"],
    summary="Cancel an event",
    description="Cancel a pending or in-progress event.",
    responses={
        200: {"description": "Event cancelled"},
        400: {"description": "Event cannot be cancelled"},
        404: {"description": "Event not found"},
    },
)
async def cancel_event(
    event_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Cancel an event."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    event = events_db[event_id]
    if event["status"] in ["completed", "cancelled"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel event in {event['status']} status",
        )

    events_db[event_id]["status"] = "cancelled"
    events_db[event_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(f"Event cancelled: {event_id}")

    return {
        "event_id": event_id,
        "status": "cancelled",
        "message": "Event cancelled successfully",
    }


# ============================================================================
# Status Endpoints
# ============================================================================

@app.get(
    "/events/{event_id}/status",
    response_model=StatusResponse,
    tags=["Status"],
    summary="Get event status",
    description="Get the current processing status of an event.",
)
async def get_status(
    event_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get event processing status."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    event = events_db[event_id]
    return StatusResponse(
        event_id=event_id,
        status=event["status"],
        progress=event.get("progress", 0.0),
        updated_at=event.get("updated_at", ""),
    )


@app.get(
    "/events/{event_id}/progress",
    response_model=ProgressResponse,
    tags=["Status"],
    summary="Get detailed progress",
    description="Get detailed progress information including phase-level status.",
)
async def get_progress(
    event_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get detailed progress information."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    event = events_db[event_id]

    # Calculate estimated completion based on progress
    progress = event.get("progress", 0.0)
    if progress > 0 and progress < 1.0:
        remaining_factor = (1.0 - progress) / progress
        estimated_minutes = 15 * remaining_factor
        estimated_completion = (
            datetime.now(timezone.utc) + timedelta(minutes=estimated_minutes)
        ).isoformat()
    else:
        estimated_completion = None

    return ProgressResponse(
        event_id=event_id,
        status=event["status"],
        progress=progress,
        phases={
            "discovery": ProgressPhase(status="completed", progress=1.0),
            "ingestion": ProgressPhase(status="in_progress", progress=0.6),
            "analysis": ProgressPhase(status="pending", progress=0.0),
            "quality": ProgressPhase(status="pending", progress=0.0),
            "reporting": ProgressPhase(status="pending", progress=0.0),
        },
        estimated_completion=estimated_completion,
    )


# ============================================================================
# Product Endpoints
# ============================================================================

@app.get(
    "/events/{event_id}/products",
    response_model=ProductListResponse,
    tags=["Products"],
    summary="List products",
    description="List available products for an event.",
)
async def list_products(
    event_id: str,
    api_key: str = Depends(verify_api_key),
):
    """List available products for an event."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    # Return sample products (replace with actual product lookup)
    products = [
        ProductMetadata(
            id="prod_flood_extent_001",
            name="flood_extent.tif",
            type="raster",
            format="GeoTIFF",
            size_bytes=1024000,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        ProductMetadata(
            id="prod_flood_depth_001",
            name="flood_depth.tif",
            type="raster",
            format="GeoTIFF",
            size_bytes=512000,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        ProductMetadata(
            id="prod_report_001",
            name="analysis_report.pdf",
            type="document",
            format="PDF",
            size_bytes=256000,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
    ]

    return ProductListResponse(
        event_id=event_id,
        products=products,
        total=len(products),
    )


@app.get(
    "/events/{event_id}/products/{product_id}",
    tags=["Products"],
    summary="Get product metadata",
    description="Get detailed metadata for a specific product.",
)
async def get_product_metadata(
    event_id: str,
    product_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get product metadata."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    return {
        "id": product_id,
        "event_id": event_id,
        "name": "flood_extent.tif",
        "type": "raster",
        "format": "GeoTIFF",
        "size_bytes": 1024000,
        "checksum": f"sha256:{hashlib.sha256(product_id.encode()).hexdigest()[:16]}...",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {
            "crs": "EPSG:4326",
            "resolution_m": 10,
            "bands": 1,
            "dtype": "float32",
            "nodata": -9999,
        },
        "quality": {
            "completeness": 0.95,
            "accuracy": 0.88,
            "confidence": 0.92,
        },
        "provenance": {
            "algorithms": ["sar_threshold_v1.2.0"],
            "data_sources": ["sentinel1_iw_grd"],
            "processing_time_seconds": 45.2,
        },
    }


@app.get(
    "/events/{event_id}/products/{product_id}/download",
    tags=["Products"],
    summary="Download product",
    description="Download a product file.",
    responses={
        200: {"description": "Product file"},
        404: {"description": "Product not found"},
    },
)
async def download_product(
    event_id: str,
    product_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Download a product file."""
    if event_id not in events_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )

    products_dir = get_products_dir()
    product_file = products_dir / "flood_extent.tif"

    if product_file.exists():
        return FileResponse(
            path=str(product_file),
            media_type="image/tiff",
            filename="flood_extent.tif",
        )

    # Return placeholder for testing
    return JSONResponse(
        content={"message": "Product data placeholder", "product_id": product_id},
        headers={"Content-Disposition": "attachment; filename=flood_extent.tif"},
    )


# ============================================================================
# Catalog Endpoints
# ============================================================================

@app.get(
    "/catalog/sources",
    tags=["Catalog"],
    summary="List data sources",
    description="List available data sources with optional type filtering.",
)
async def list_sources(
    source_type: Optional[str] = Query(None, alias="type", description="Filter by type"),
    api_key: str = Depends(verify_api_key),
):
    """List available data sources."""
    sources = [
        {"id": "sentinel2", "name": "Sentinel-2", "type": "optical", "provider": "ESA",
         "resolution_m": 10, "revisit_days": 5},
        {"id": "sentinel1", "name": "Sentinel-1", "type": "sar", "provider": "ESA",
         "resolution_m": 10, "revisit_days": 6},
        {"id": "landsat8", "name": "Landsat-8", "type": "optical", "provider": "USGS",
         "resolution_m": 30, "revisit_days": 16},
        {"id": "modis", "name": "MODIS", "type": "optical", "provider": "NASA",
         "resolution_m": 250, "revisit_days": 1},
        {"id": "copernicus_dem", "name": "Copernicus DEM", "type": "dem", "provider": "ESA",
         "resolution_m": 30, "revisit_days": None},
        {"id": "era5", "name": "ERA5", "type": "weather", "provider": "ECMWF",
         "resolution_m": 27000, "revisit_days": 0},
    ]

    if source_type:
        sources = [s for s in sources if s["type"] == source_type]

    return {"sources": sources, "total": len(sources)}


@app.get(
    "/catalog/algorithms",
    tags=["Catalog"],
    summary="List algorithms",
    description="List available analysis algorithms with optional event type filtering.",
)
async def list_algorithms(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    api_key: str = Depends(verify_api_key),
):
    """List available algorithms."""
    algorithms = [
        {
            "id": "sar_threshold",
            "name": "SAR Backscatter Threshold",
            "version": "1.2.0",
            "description": "Water detection using SAR backscatter thresholding",
            "event_types": ["flood.*"],
            "inputs": ["sar"],
            "outputs": ["flood_extent"],
        },
        {
            "id": "ndwi_optical",
            "name": "NDWI Optical Flood Detection",
            "version": "1.1.0",
            "description": "Water detection using Normalized Difference Water Index",
            "event_types": ["flood.*"],
            "inputs": ["optical"],
            "outputs": ["water_mask"],
        },
        {
            "id": "hand_model",
            "name": "Height Above Nearest Drainage",
            "version": "1.0.0",
            "description": "Flood susceptibility based on terrain analysis",
            "event_types": ["flood.*"],
            "inputs": ["dem"],
            "outputs": ["flood_susceptibility"],
        },
        {
            "id": "dnbr_severity",
            "name": "Differenced NBR Burn Severity",
            "version": "1.0.0",
            "description": "Burn severity mapping using dNBR",
            "event_types": ["wildfire.*"],
            "inputs": ["optical"],
            "outputs": ["burn_severity"],
        },
        {
            "id": "thermal_anomaly",
            "name": "Thermal Anomaly Detection",
            "version": "1.0.0",
            "description": "Active fire detection using thermal bands",
            "event_types": ["wildfire.*"],
            "inputs": ["optical"],
            "outputs": ["active_fire"],
        },
        {
            "id": "wind_damage",
            "name": "Wind Damage Assessment",
            "version": "1.0.0",
            "description": "Storm damage detection using change analysis",
            "event_types": ["storm.*"],
            "inputs": ["optical", "sar"],
            "outputs": ["damage_assessment"],
        },
    ]

    if event_type:
        algorithms = [
            a for a in algorithms
            if any(event_type.startswith(et.replace(".*", "")) for et in a["event_types"])
        ]

    return {"algorithms": algorithms, "total": len(algorithms)}


@app.get(
    "/catalog/event-types",
    tags=["Catalog"],
    summary="List event types",
    description="List supported event types and their hierarchical structure.",
)
async def list_event_types(
    api_key: str = Depends(verify_api_key),
):
    """List supported event types."""
    return {
        "event_types": [
            {
                "class": "flood",
                "description": "Flood events from various causes",
                "subclasses": [
                    {"class": "flood.coastal", "description": "Coastal flooding from storm surge or tides"},
                    {"class": "flood.riverine", "description": "River/fluvial flooding from overflow"},
                    {"class": "flood.pluvial", "description": "Urban/rainfall flooding from runoff"},
                ],
            },
            {
                "class": "wildfire",
                "description": "Wildfire and burn events",
                "subclasses": [
                    {"class": "wildfire.forest", "description": "Forest fires"},
                    {"class": "wildfire.grassland", "description": "Grassland/brush fires"},
                    {"class": "wildfire.prescribed", "description": "Prescribed/controlled burns"},
                ],
            },
            {
                "class": "storm",
                "description": "Storm and severe weather events",
                "subclasses": [
                    {"class": "storm.tropical_cyclone", "description": "Hurricanes, typhoons, cyclones"},
                    {"class": "storm.tornado", "description": "Tornado damage paths"},
                    {"class": "storm.severe_weather", "description": "Severe thunderstorm damage"},
                ],
            },
        ],
    }


# ============================================================================
# Health Endpoints
# ============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Basic health check endpoint. Does not require authentication.",
)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=API_VERSION,
    )


@app.get(
    "/health/ready",
    tags=["Health"],
    summary="Readiness probe",
    description="Readiness probe for Kubernetes/orchestration systems.",
)
async def readiness_check():
    """Readiness probe."""
    checks = {
        "database": True,
        "cache": True,
        "agents": True,
    }

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get(
    "/health/live",
    tags=["Health"],
    summary="Liveness probe",
    description="Liveness probe for Kubernetes/orchestration systems.",
)
async def liveness_check():
    """Liveness probe."""
    return {
        "alive": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Webhook Endpoints
# ============================================================================

@app.post(
    "/webhooks",
    response_model=WebhookResponse,
    tags=["Webhooks"],
    summary="Register webhook",
    description="Register a webhook to receive event notifications.",
)
async def register_webhook(
    webhook: WebhookRegistration,
    api_key: str = Depends(verify_api_key),
):
    """Register a webhook for event notifications."""
    webhook_id = f"wh_{uuid.uuid4().hex[:12]}"

    webhook_data = {
        "id": webhook_id,
        "url": webhook.url,
        "events": webhook.events,
        "secret": webhook.secret,
        "active": webhook.active,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "deliveries": [],
    }

    webhooks_db[webhook_id] = webhook_data

    logger.info(f"Webhook registered: {webhook_id} -> {webhook.url}")

    return WebhookResponse(
        id=webhook_id,
        url=webhook.url,
        events=webhook.events,
        active=webhook.active,
        message="Webhook registered successfully",
    )


@app.get(
    "/webhooks",
    tags=["Webhooks"],
    summary="List webhooks",
    description="List all registered webhooks.",
)
async def list_webhooks(
    api_key: str = Depends(verify_api_key),
):
    """List registered webhooks."""
    return {
        "webhooks": list(webhooks_db.values()),
        "total": len(webhooks_db),
    }


@app.get(
    "/webhooks/{webhook_id}",
    tags=["Webhooks"],
    summary="Get webhook details",
    description="Get details of a specific webhook.",
)
async def get_webhook(
    webhook_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get webhook details."""
    if webhook_id not in webhooks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )
    return webhooks_db[webhook_id]


@app.delete(
    "/webhooks/{webhook_id}",
    tags=["Webhooks"],
    summary="Delete webhook",
    description="Delete a registered webhook.",
)
async def delete_webhook(
    webhook_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Delete a webhook."""
    if webhook_id not in webhooks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    del webhooks_db[webhook_id]
    logger.info(f"Webhook deleted: {webhook_id}")

    return {"message": "Webhook deleted successfully"}


@app.post(
    "/webhooks/{webhook_id}/test",
    tags=["Webhooks"],
    summary="Test webhook",
    description="Send a test delivery to a webhook.",
)
async def test_webhook(
    webhook_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Send a test delivery to a webhook."""
    if webhook_id not in webhooks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    webhook = webhooks_db[webhook_id]

    delivery = {
        "id": str(uuid.uuid4()),
        "event_type": "test",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {"message": "Test webhook delivery"},
        "status": "delivered",
    }

    webhook["deliveries"].append(delivery)

    return {
        "delivery_id": delivery["id"],
        "status": "delivered",
        "message": "Test webhook delivered successfully",
    }


# ============================================================================
# Application Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {API_TITLE} v{API_VERSION}")
    logger.info(f"Docs available at /docs and /redoc")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {API_TITLE}")


# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
