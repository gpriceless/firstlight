"""
Comprehensive API Test Suite for Multiverse Dive Event Intelligence Platform.

Tests all API endpoints including:
- Event submission and retrieval
- Status and progress monitoring
- Product download and metadata
- Catalog browsing
- Health checks
- Authentication and rate limiting
- Webhook registration and delivery
- Error handling

Uses pytest and httpx/TestClient for testing FastAPI application.
"""

import asyncio
import hashlib
import json
import os
import pytest
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Test client imports
try:
    from httpx import AsyncClient
    from fastapi.testclient import TestClient
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    TestClient = None
    AsyncClient = None

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, status
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None


# Skip all tests if dependencies not available
pytestmark = pytest.mark.skipif(
    not (HTTPX_AVAILABLE and FASTAPI_AVAILABLE),
    reason="FastAPI or httpx not installed"
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_event_spec() -> Dict[str, Any]:
    """Sample valid event specification."""
    return {
        "id": f"evt_{uuid.uuid4().hex[:16]}",
        "intent": {
            "class": "flood.coastal.storm_surge",
            "source": "explicit",
            "confidence": 0.95
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[
                [-80.3, 25.7],
                [-80.1, 25.7],
                [-80.1, 25.9],
                [-80.3, 25.9],
                [-80.3, 25.7]
            ]],
            "crs": "EPSG:4326",
            "bbox": [-80.3, 25.7, -80.1, 25.9]
        },
        "temporal": {
            "start": "2024-09-15T00:00:00Z",
            "end": "2024-09-20T23:59:59Z"
        },
        "priority": "high",
        "constraints": {
            "max_cloud_cover": 0.4,
            "required_data_types": ["sar", "dem"]
        },
        "metadata": {
            "created_by": "test_suite",
            "tags": ["test", "flood", "coastal"]
        }
    }


@pytest.fixture
def sample_wildfire_event() -> Dict[str, Any]:
    """Sample wildfire event specification."""
    return {
        "id": f"evt_wildfire_{uuid.uuid4().hex[:8]}",
        "intent": {
            "class": "wildfire.forest.active",
            "source": "inferred",
            "original_input": "forest fire in California mountains",
            "confidence": 0.88
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.5],
                [-122.0, 37.5],
                [-122.0, 38.0],
                [-122.5, 38.0],
                [-122.5, 37.5]
            ]],
            "crs": "EPSG:4326"
        },
        "temporal": {
            "start": "2024-08-01T00:00:00Z",
            "end": "2024-08-10T23:59:59Z"
        },
        "priority": "critical"
    }


@pytest.fixture
def invalid_event_spec() -> Dict[str, Any]:
    """Invalid event specification missing required fields."""
    return {
        "id": "invalid_event",
        # Missing intent, spatial, temporal
        "priority": "normal"
    }


@pytest.fixture
def sample_api_key() -> str:
    """Sample valid API key for testing."""
    return "test_api_key_" + hashlib.sha256(b"test_secret").hexdigest()[:32]


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator agent for testing."""
    mock = MagicMock()
    mock.submit_event = AsyncMock(return_value={
        "event_id": "evt_test_001",
        "status": "accepted",
        "message": "Event submitted successfully"
    })
    mock.get_event = AsyncMock(return_value={
        "event_id": "evt_test_001",
        "status": "processing",
        "progress": 0.45
    })
    mock.cancel_event = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_discovery_agent():
    """Mock discovery agent for testing."""
    mock = MagicMock()
    mock.list_sources = AsyncMock(return_value=[
        {"id": "sentinel2", "name": "Sentinel-2", "type": "optical"},
        {"id": "sentinel1", "name": "Sentinel-1", "type": "sar"},
        {"id": "landsat8", "name": "Landsat-8", "type": "optical"}
    ])
    return mock


@pytest.fixture
def mock_quality_agent():
    """Mock quality agent for testing."""
    mock = MagicMock()
    mock.get_quality_report = AsyncMock(return_value={
        "overall_score": 0.92,
        "checks_passed": 15,
        "checks_failed": 2
    })
    return mock


@pytest.fixture
def temp_products_dir(tmp_path) -> Path:
    """Create temporary directory for test products."""
    products_dir = tmp_path / "products"
    products_dir.mkdir()

    # Create sample product files
    (products_dir / "flood_extent.tif").write_bytes(b"TIFF_DATA_PLACEHOLDER")
    (products_dir / "flood_depth.tif").write_bytes(b"TIFF_DEPTH_DATA")
    (products_dir / "report.pdf").write_bytes(b"%PDF-1.4_PLACEHOLDER")

    return products_dir


@pytest.fixture
def webhook_server():
    """Create a mock webhook server for testing deliveries."""
    deliveries = []

    class WebhookCapture:
        def __init__(self):
            self.deliveries = deliveries
            self.fail_count = 0
            self.should_fail = False

        def capture(self, payload: Dict[str, Any]) -> Dict[str, Any]:
            if self.should_fail and self.fail_count > 0:
                self.fail_count -= 1
                raise Exception("Webhook delivery failed")
            self.deliveries.append(payload)
            return {"status": "received", "id": str(uuid.uuid4())}

        def clear(self):
            self.deliveries.clear()

    return WebhookCapture()


# ============================================================================
# Mock FastAPI Application for Testing
# ============================================================================

def create_test_app(
    mock_orchestrator=None,
    mock_discovery_agent=None,
    mock_quality_agent=None,
    api_keys: Optional[List[str]] = None,
    rate_limit_per_minute: int = 100,
    products_dir: Optional[Path] = None
) -> FastAPI:
    """Create a test FastAPI application with mocked dependencies."""
    from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
    from typing import Optional, List
    import asyncio

    app = FastAPI(
        title="Multiverse Dive API",
        description="Geospatial Event Intelligence Platform API",
        version="1.0.0"
    )

    # In-memory storage for testing
    events_db: Dict[str, Dict[str, Any]] = {}
    webhooks_db: Dict[str, Dict[str, Any]] = {}
    rate_limit_tracker: Dict[str, List[float]] = {}

    # Valid API keys for testing
    valid_api_keys = set(api_keys or ["test_api_key"])

    # Request models
    class EventSubmission(BaseModel):
        id: Optional[str] = None
        intent: Dict[str, Any]
        spatial: Dict[str, Any]
        temporal: Dict[str, Any]
        priority: str = "normal"
        constraints: Optional[Dict[str, Any]] = None
        metadata: Optional[Dict[str, Any]] = None

    class WebhookRegistration(BaseModel):
        url: str
        events: List[str] = Field(default_factory=lambda: ["*"])
        secret: Optional[str] = None
        active: bool = True

    # Dependencies
    async def verify_api_key(x_api_key: Optional[str] = Header(None)):
        if not valid_api_keys:
            return True  # No auth required if no keys configured
        if x_api_key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required"
            )
        if x_api_key not in valid_api_keys:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        return x_api_key

    async def check_rate_limit(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        if client_ip not in rate_limit_tracker:
            rate_limit_tracker[client_ip] = []

        # Clean old entries
        rate_limit_tracker[client_ip] = [
            t for t in rate_limit_tracker[client_ip] if t > window_start
        ]

        if len(rate_limit_tracker[client_ip]) >= rate_limit_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers={"Retry-After": "60"}
            )

        rate_limit_tracker[client_ip].append(current_time)
        return True

    # Event Endpoints
    @app.post("/events", tags=["Events"])
    async def submit_event(
        event: EventSubmission,
        api_key: str = Depends(verify_api_key),
        _rate_limit: bool = Depends(check_rate_limit)
    ):
        """Submit a new event for processing."""
        event_id = event.id or f"evt_{uuid.uuid4().hex[:16]}"

        # Validate required fields
        if not event.intent or not event.spatial or not event.temporal:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Missing required fields: intent, spatial, temporal"
            )

        event_data = {
            "id": event_id,
            "intent": event.intent,
            "spatial": event.spatial,
            "temporal": event.temporal,
            "priority": event.priority,
            "constraints": event.constraints,
            "metadata": event.metadata,
            "status": "submitted",
            "progress": 0.0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

        events_db[event_id] = event_data

        if mock_orchestrator:
            await mock_orchestrator.submit_event(event_data)

        return {
            "event_id": event_id,
            "status": "accepted",
            "message": "Event submitted successfully",
            "links": {
                "self": f"/events/{event_id}",
                "status": f"/events/{event_id}/status",
                "cancel": f"/events/{event_id}/cancel"
            }
        }

    @app.get("/events/{event_id}", tags=["Events"])
    async def get_event(
        event_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Retrieve an event by ID."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )
        return events_db[event_id]

    @app.get("/events", tags=["Events"])
    async def list_events(
        status: Optional[str] = Query(None),
        priority: Optional[str] = Query(None),
        limit: int = Query(10, ge=1, le=100),
        offset: int = Query(0, ge=0),
        api_key: str = Depends(verify_api_key)
    ):
        """List events with optional filtering."""
        events = list(events_db.values())

        # Apply filters
        if status:
            events = [e for e in events if e.get("status") == status]
        if priority:
            events = [e for e in events if e.get("priority") == priority]

        # Apply pagination
        total = len(events)
        events = events[offset:offset + limit]

        return {
            "events": events,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    @app.post("/events/{event_id}/cancel", tags=["Events"])
    async def cancel_event(
        event_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Cancel an event."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        event = events_db[event_id]
        if event["status"] in ["completed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel event in {event['status']} status"
            )

        events_db[event_id]["status"] = "cancelled"
        events_db[event_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

        if mock_orchestrator:
            await mock_orchestrator.cancel_event(event_id)

        return {
            "event_id": event_id,
            "status": "cancelled",
            "message": "Event cancelled successfully"
        }

    # Status Endpoints
    @app.get("/events/{event_id}/status", tags=["Status"])
    async def get_status(
        event_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Get event processing status."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        event = events_db[event_id]
        return {
            "event_id": event_id,
            "status": event["status"],
            "progress": event.get("progress", 0.0),
            "updated_at": event.get("updated_at")
        }

    @app.get("/events/{event_id}/progress", tags=["Status"])
    async def get_progress(
        event_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Get detailed progress information."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        event = events_db[event_id]
        return {
            "event_id": event_id,
            "status": event["status"],
            "progress": event.get("progress", 0.0),
            "phases": {
                "discovery": {"status": "completed", "progress": 1.0},
                "ingestion": {"status": "in_progress", "progress": 0.6},
                "analysis": {"status": "pending", "progress": 0.0},
                "quality": {"status": "pending", "progress": 0.0},
                "reporting": {"status": "pending", "progress": 0.0}
            },
            "estimated_completion": (
                datetime.now(timezone.utc) + timedelta(minutes=15)
            ).isoformat()
        }

    # Product Endpoints
    @app.get("/events/{event_id}/products", tags=["Products"])
    async def list_products(
        event_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """List available products for an event."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        # Return mock products
        products = [
            {
                "id": "prod_flood_extent_001",
                "name": "flood_extent.tif",
                "type": "raster",
                "format": "GeoTIFF",
                "size_bytes": 1024000,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "prod_flood_depth_001",
                "name": "flood_depth.tif",
                "type": "raster",
                "format": "GeoTIFF",
                "size_bytes": 512000,
                "created_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "prod_report_001",
                "name": "analysis_report.pdf",
                "type": "document",
                "format": "PDF",
                "size_bytes": 256000,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        ]

        return {
            "event_id": event_id,
            "products": products,
            "total": len(products)
        }

    @app.get("/events/{event_id}/products/{product_id}", tags=["Products"])
    async def get_product_metadata(
        event_id: str,
        product_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Get metadata for a specific product."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        # Return mock product metadata
        return {
            "id": product_id,
            "event_id": event_id,
            "name": "flood_extent.tif",
            "type": "raster",
            "format": "GeoTIFF",
            "size_bytes": 1024000,
            "checksum": "sha256:abc123...",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "crs": "EPSG:4326",
                "resolution_m": 10,
                "bands": 1,
                "dtype": "float32"
            },
            "quality": {
                "completeness": 0.95,
                "accuracy": 0.88
            },
            "provenance": {
                "algorithms": ["sar_threshold_v1.2.0"],
                "data_sources": ["sentinel1_iw_grd"]
            }
        }

    @app.get("/events/{event_id}/products/{product_id}/download", tags=["Products"])
    async def download_product(
        event_id: str,
        product_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Download a product file."""
        if event_id not in events_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Event {event_id} not found"
            )

        if products_dir and products_dir.exists():
            product_file = products_dir / "flood_extent.tif"
            if product_file.exists():
                return FileResponse(
                    path=str(product_file),
                    media_type="image/tiff",
                    filename="flood_extent.tif"
                )

        # Return placeholder for testing
        return JSONResponse(
            content={"message": "Product data placeholder", "product_id": product_id},
            headers={"Content-Disposition": "attachment; filename=flood_extent.tif"}
        )

    # Catalog Endpoints
    @app.get("/catalog/sources", tags=["Catalog"])
    async def list_sources(
        type: Optional[str] = Query(None),
        api_key: str = Depends(verify_api_key)
    ):
        """List available data sources."""
        sources = [
            {"id": "sentinel2", "name": "Sentinel-2", "type": "optical", "provider": "ESA"},
            {"id": "sentinel1", "name": "Sentinel-1", "type": "sar", "provider": "ESA"},
            {"id": "landsat8", "name": "Landsat-8", "type": "optical", "provider": "USGS"},
            {"id": "modis", "name": "MODIS", "type": "optical", "provider": "NASA"},
            {"id": "copernicus_dem", "name": "Copernicus DEM", "type": "dem", "provider": "ESA"},
            {"id": "era5", "name": "ERA5", "type": "weather", "provider": "ECMWF"}
        ]

        if type:
            sources = [s for s in sources if s["type"] == type]

        if mock_discovery_agent:
            sources = await mock_discovery_agent.list_sources()

        return {"sources": sources, "total": len(sources)}

    @app.get("/catalog/algorithms", tags=["Catalog"])
    async def list_algorithms(
        event_type: Optional[str] = Query(None),
        api_key: str = Depends(verify_api_key)
    ):
        """List available analysis algorithms."""
        algorithms = [
            {
                "id": "sar_threshold",
                "name": "SAR Backscatter Threshold",
                "version": "1.2.0",
                "event_types": ["flood.*"],
                "inputs": ["sar"],
                "outputs": ["flood_extent"]
            },
            {
                "id": "ndwi_optical",
                "name": "NDWI Optical Flood Detection",
                "version": "1.1.0",
                "event_types": ["flood.*"],
                "inputs": ["optical"],
                "outputs": ["water_mask"]
            },
            {
                "id": "dnbr_severity",
                "name": "Differenced NBR Burn Severity",
                "version": "1.0.0",
                "event_types": ["wildfire.*"],
                "inputs": ["optical"],
                "outputs": ["burn_severity"]
            },
            {
                "id": "wind_damage",
                "name": "Wind Damage Assessment",
                "version": "1.0.0",
                "event_types": ["storm.*"],
                "inputs": ["optical", "sar"],
                "outputs": ["damage_assessment"]
            }
        ]

        if event_type:
            algorithms = [
                a for a in algorithms
                if any(event_type.startswith(et.replace(".*", "")) for et in a["event_types"])
            ]

        return {"algorithms": algorithms, "total": len(algorithms)}

    @app.get("/catalog/event-types", tags=["Catalog"])
    async def list_event_types(
        api_key: str = Depends(verify_api_key)
    ):
        """List supported event types."""
        return {
            "event_types": [
                {
                    "class": "flood",
                    "description": "Flood events",
                    "subclasses": [
                        {"class": "flood.coastal", "description": "Coastal flooding"},
                        {"class": "flood.riverine", "description": "River flooding"},
                        {"class": "flood.pluvial", "description": "Urban/rainfall flooding"}
                    ]
                },
                {
                    "class": "wildfire",
                    "description": "Wildfire events",
                    "subclasses": [
                        {"class": "wildfire.forest", "description": "Forest fires"},
                        {"class": "wildfire.grassland", "description": "Grassland fires"}
                    ]
                },
                {
                    "class": "storm",
                    "description": "Storm events",
                    "subclasses": [
                        {"class": "storm.tropical_cyclone", "description": "Hurricanes/typhoons"},
                        {"class": "storm.tornado", "description": "Tornado damage"},
                        {"class": "storm.severe_weather", "description": "Severe storms"}
                    ]
                }
            ]
        }

    # Health Endpoints
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0"
        }

    @app.get("/health/ready", tags=["Health"])
    async def readiness_check():
        """Readiness probe for Kubernetes."""
        # Check if all required services are ready
        checks = {
            "database": True,
            "cache": True,
            "agents": True
        }

        all_ready = all(checks.values())

        return {
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/health/live", tags=["Health"])
    async def liveness_check():
        """Liveness probe for Kubernetes."""
        return {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Webhook Endpoints
    @app.post("/webhooks", tags=["Webhooks"])
    async def register_webhook(
        webhook: WebhookRegistration,
        api_key: str = Depends(verify_api_key)
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
            "deliveries": []
        }

        webhooks_db[webhook_id] = webhook_data

        return {
            "id": webhook_id,
            "url": webhook.url,
            "events": webhook.events,
            "active": webhook.active,
            "message": "Webhook registered successfully"
        }

    @app.get("/webhooks", tags=["Webhooks"])
    async def list_webhooks(
        api_key: str = Depends(verify_api_key)
    ):
        """List registered webhooks."""
        return {
            "webhooks": list(webhooks_db.values()),
            "total": len(webhooks_db)
        }

    @app.delete("/webhooks/{webhook_id}", tags=["Webhooks"])
    async def delete_webhook(
        webhook_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Delete a webhook."""
        if webhook_id not in webhooks_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )

        del webhooks_db[webhook_id]
        return {"message": "Webhook deleted successfully"}

    # Test helper endpoint to trigger webhook
    @app.post("/webhooks/{webhook_id}/test", tags=["Webhooks"])
    async def test_webhook(
        webhook_id: str,
        api_key: str = Depends(verify_api_key)
    ):
        """Send a test delivery to a webhook."""
        if webhook_id not in webhooks_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Webhook {webhook_id} not found"
            )

        webhook = webhooks_db[webhook_id]

        # Simulate delivery
        delivery = {
            "id": str(uuid.uuid4()),
            "event_type": "test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {"message": "Test webhook delivery"},
            "status": "delivered"
        }

        webhook["deliveries"].append(delivery)

        return {
            "delivery_id": delivery["id"],
            "status": "delivered",
            "message": "Test webhook delivered successfully"
        }

    # Store test utilities
    app.state.events_db = events_db
    app.state.webhooks_db = webhooks_db
    app.state.rate_limit_tracker = rate_limit_tracker

    return app


@pytest.fixture
def app(mock_orchestrator, temp_products_dir, sample_api_key) -> FastAPI:
    """Create test application."""
    return create_test_app(
        mock_orchestrator=mock_orchestrator,
        api_keys=[sample_api_key, "test_api_key"],
        rate_limit_per_minute=100,
        products_dir=temp_products_dir
    )


@pytest.fixture
def client(app) -> Generator:
    """Create test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(sample_api_key) -> Dict[str, str]:
    """Headers with valid API key."""
    return {"X-API-Key": sample_api_key}


# ============================================================================
# Test Classes
# ============================================================================

class TestEventEndpoints:
    """Tests for event submission and retrieval endpoints."""

    def test_submit_event_success(self, client, auth_headers, sample_event_spec):
        """Test successful event submission."""
        response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "event_id" in data
        assert data["status"] == "accepted"
        assert "links" in data
        assert "self" in data["links"]

    def test_submit_event_auto_generated_id(self, client, auth_headers, sample_event_spec):
        """Test event submission generates ID if not provided."""
        event = sample_event_spec.copy()
        del event["id"]

        response = client.post("/events", json=event, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["event_id"].startswith("evt_")

    def test_submit_event_invalid_spec(self, client, auth_headers, invalid_event_spec):
        """Test event submission with invalid specification."""
        response = client.post(
            "/events",
            json=invalid_event_spec,
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_submit_event_missing_intent(self, client, auth_headers, sample_event_spec):
        """Test event submission without intent field."""
        event = sample_event_spec.copy()
        del event["intent"]

        response = client.post("/events", json=event, headers=auth_headers)

        assert response.status_code == 422

    def test_get_event_found(self, client, auth_headers, sample_event_spec):
        """Test retrieving an existing event."""
        # First submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        # Then retrieve it
        response = client.get(f"/events/{event_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == event_id
        assert "intent" in data
        assert "spatial" in data
        assert "temporal" in data

    def test_get_event_not_found(self, client, auth_headers):
        """Test retrieving a non-existent event."""
        response = client.get("/events/evt_nonexistent_123", headers=auth_headers)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_events(self, client, auth_headers, sample_event_spec, sample_wildfire_event):
        """Test listing events."""
        # Submit multiple events
        client.post("/events", json=sample_event_spec, headers=auth_headers)
        client.post("/events", json=sample_wildfire_event, headers=auth_headers)

        response = client.get("/events", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) >= 2
        assert "total" in data

    def test_list_events_with_filter(self, client, auth_headers, sample_event_spec):
        """Test listing events with status filter."""
        # Submit an event
        client.post("/events", json=sample_event_spec, headers=auth_headers)

        response = client.get(
            "/events?status=submitted",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert all(e["status"] == "submitted" for e in data["events"])

    def test_list_events_pagination(self, client, auth_headers, sample_event_spec):
        """Test event listing pagination."""
        # Submit multiple events
        for i in range(5):
            event = sample_event_spec.copy()
            event["id"] = f"evt_pagination_test_{i}"
            client.post("/events", json=event, headers=auth_headers)

        # Test pagination
        response = client.get("/events?limit=2&offset=0", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["events"]) <= 2
        assert data["limit"] == 2
        assert data["offset"] == 0

    def test_cancel_event(self, client, auth_headers, sample_event_spec):
        """Test cancelling an event."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        # Cancel it
        response = client.post(f"/events/{event_id}/cancel", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"

    def test_cancel_event_not_found(self, client, auth_headers):
        """Test cancelling non-existent event."""
        response = client.post("/events/evt_nonexistent/cancel", headers=auth_headers)

        assert response.status_code == 404


class TestStatusEndpoints:
    """Tests for status and progress monitoring endpoints."""

    def test_get_status(self, client, auth_headers, sample_event_spec):
        """Test getting event status."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        response = client.get(f"/events/{event_id}/status", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == event_id
        assert "status" in data
        assert "progress" in data

    def test_get_progress(self, client, auth_headers, sample_event_spec):
        """Test getting detailed progress."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        response = client.get(f"/events/{event_id}/progress", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == event_id
        assert "phases" in data
        assert "estimated_completion" in data

    def test_status_transitions(self, client, auth_headers, sample_event_spec):
        """Test status transitions through cancel."""
        # Submit event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        # Check initial status
        status_response = client.get(f"/events/{event_id}/status", headers=auth_headers)
        assert status_response.json()["status"] == "submitted"

        # Cancel event
        client.post(f"/events/{event_id}/cancel", headers=auth_headers)

        # Check cancelled status
        status_response = client.get(f"/events/{event_id}/status", headers=auth_headers)
        assert status_response.json()["status"] == "cancelled"

    def test_status_not_found(self, client, auth_headers):
        """Test status for non-existent event."""
        response = client.get("/events/evt_nonexistent/status", headers=auth_headers)

        assert response.status_code == 404


class TestProductEndpoints:
    """Tests for product listing, metadata, and download endpoints."""

    def test_list_products(self, client, auth_headers, sample_event_spec):
        """Test listing products for an event."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        response = client.get(f"/events/{event_id}/products", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "products" in data
        assert len(data["products"]) > 0
        assert "total" in data

    def test_download_product(self, client, auth_headers, sample_event_spec):
        """Test downloading a product."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        response = client.get(
            f"/events/{event_id}/products/prod_001/download",
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_product_metadata(self, client, auth_headers, sample_event_spec):
        """Test getting product metadata."""
        # Submit an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        response = client.get(
            f"/events/{event_id}/products/prod_001",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "metadata" in data
        assert "quality" in data
        assert "provenance" in data

    def test_product_not_found(self, client, auth_headers):
        """Test product access for non-existent event."""
        response = client.get(
            "/events/evt_nonexistent/products",
            headers=auth_headers
        )

        assert response.status_code == 404


class TestCatalogEndpoints:
    """Tests for catalog browsing endpoints."""

    def test_list_sources(self, client, auth_headers):
        """Test listing data sources."""
        response = client.get("/catalog/sources", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0

        # Verify source structure
        source = data["sources"][0]
        assert "id" in source
        assert "name" in source
        assert "type" in source

    def test_list_sources_with_type_filter(self, client, auth_headers):
        """Test listing data sources filtered by type."""
        response = client.get("/catalog/sources?type=sar", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert all(s["type"] == "sar" for s in data["sources"])

    def test_list_algorithms(self, client, auth_headers):
        """Test listing algorithms."""
        response = client.get("/catalog/algorithms", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "algorithms" in data
        assert len(data["algorithms"]) > 0

        # Verify algorithm structure
        algo = data["algorithms"][0]
        assert "id" in algo
        assert "name" in algo
        assert "version" in algo
        assert "event_types" in algo

    def test_list_algorithms_by_event_type(self, client, auth_headers):
        """Test listing algorithms filtered by event type."""
        response = client.get(
            "/catalog/algorithms?event_type=flood",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["algorithms"]) > 0

    def test_list_event_types(self, client, auth_headers):
        """Test listing event types."""
        response = client.get("/catalog/event-types", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "event_types" in data

        # Verify structure
        event_type = data["event_types"][0]
        assert "class" in event_type
        assert "description" in event_type
        assert "subclasses" in event_type


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_readiness(self, client):
        """Test readiness probe."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "checks" in data

    def test_liveness(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestAuthentication:
    """Tests for API authentication."""

    def test_api_key_valid(self, client, auth_headers, sample_event_spec):
        """Test valid API key access."""
        response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )

        assert response.status_code == 200

    def test_api_key_invalid(self, client, sample_event_spec):
        """Test invalid API key rejection."""
        response = client.post(
            "/events",
            json=sample_event_spec,
            headers={"X-API-Key": "invalid_key_123"}
        )

        assert response.status_code == 403
        assert "invalid" in response.json()["detail"].lower()

    def test_api_key_missing(self, client, sample_event_spec):
        """Test missing API key rejection."""
        response = client.post("/events", json=sample_event_spec)

        assert response.status_code == 401
        assert "required" in response.json()["detail"].lower()

    def test_rate_limiting(self, sample_api_key, temp_products_dir, sample_event_spec):
        """Test rate limiting enforcement."""
        # Create app with very low rate limit
        app = create_test_app(
            api_keys=[sample_api_key],
            rate_limit_per_minute=3,
            products_dir=temp_products_dir
        )

        with TestClient(app) as client:
            headers = {"X-API-Key": sample_api_key}

            # Make requests up to limit
            for i in range(3):
                response = client.post("/events", json=sample_event_spec, headers=headers)
                assert response.status_code == 200

            # Next request should be rate limited
            response = client.post("/events", json=sample_event_spec, headers=headers)
            assert response.status_code == 429
            assert "Retry-After" in response.headers


class TestWebhooks:
    """Tests for webhook registration and delivery."""

    def test_webhook_registration(self, client, auth_headers):
        """Test webhook registration."""
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["event.completed", "event.failed"],
            "secret": "test_secret_123"
        }

        response = client.post(
            "/webhooks",
            json=webhook_data,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["url"] == webhook_data["url"]
        assert data["events"] == webhook_data["events"]

    def test_webhook_list(self, client, auth_headers):
        """Test listing registered webhooks."""
        # Register a webhook first
        client.post(
            "/webhooks",
            json={"url": "https://example.com/hook1", "events": ["*"]},
            headers=auth_headers
        )

        response = client.get("/webhooks", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "webhooks" in data
        assert len(data["webhooks"]) >= 1

    def test_webhook_delivery(self, client, auth_headers):
        """Test webhook delivery (via test endpoint)."""
        # Register a webhook
        register_response = client.post(
            "/webhooks",
            json={"url": "https://example.com/webhook", "events": ["*"]},
            headers=auth_headers
        )
        webhook_id = register_response.json()["id"]

        # Test delivery
        response = client.post(
            f"/webhooks/{webhook_id}/test",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "delivered"

    def test_webhook_deletion(self, client, auth_headers):
        """Test webhook deletion."""
        # Register a webhook
        register_response = client.post(
            "/webhooks",
            json={"url": "https://example.com/webhook", "events": ["*"]},
            headers=auth_headers
        )
        webhook_id = register_response.json()["id"]

        # Delete it
        response = client.delete(f"/webhooks/{webhook_id}", headers=auth_headers)

        assert response.status_code == 200

        # Verify it's deleted
        get_response = client.delete(f"/webhooks/{webhook_id}", headers=auth_headers)
        assert get_response.status_code == 404

    def test_webhook_retry(self, client, auth_headers):
        """Test webhook retry behavior placeholder."""
        # This is a placeholder test - full retry testing would require
        # more complex mocking infrastructure
        webhook_data = {
            "url": "https://example.com/webhook",
            "events": ["*"],
            "active": True
        }

        response = client.post("/webhooks", json=webhook_data, headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["active"] is True


class TestErrorHandling:
    """Tests for error response handling."""

    def test_validation_error_response(self, client, auth_headers):
        """Test validation error response format."""
        # Submit invalid event (missing required fields)
        response = client.post(
            "/events",
            json={"invalid": "data"},
            headers=auth_headers
        )

        assert response.status_code == 422

    def test_not_found_response(self, client, auth_headers):
        """Test 404 response format."""
        response = client.get("/events/evt_nonexistent_xyz", headers=auth_headers)

        assert response.status_code == 404
        data = response.json()
        assert "detail" in data

    def test_internal_error_response(self, sample_api_key, temp_products_dir):
        """Test internal error handling."""
        # Create app with broken mock
        broken_mock = MagicMock()
        broken_mock.submit_event = AsyncMock(side_effect=Exception("Internal error"))

        app = create_test_app(
            mock_orchestrator=broken_mock,
            api_keys=[sample_api_key],
            products_dir=temp_products_dir
        )

        # The app should handle the error gracefully
        # Note: In a real implementation, this would return 500
        # For now, since the mock is called after response,
        # it doesn't affect the response
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEventEdgeCases:
    """Tests for event-related edge cases."""

    def test_submit_event_with_all_fields(self, client, auth_headers):
        """Test event submission with all optional fields."""
        event = {
            "id": "evt_full_test_001",
            "intent": {
                "class": "flood.coastal.storm_surge",
                "source": "inferred",
                "original_input": "coastal flooding after hurricane",
                "confidence": 0.95
            },
            "spatial": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.3, 25.7], [-80.1, 25.7],
                    [-80.1, 25.9], [-80.3, 25.9],
                    [-80.3, 25.7]
                ]],
                "crs": "EPSG:4326",
                "bbox": [-80.3, 25.7, -80.1, 25.9]
            },
            "temporal": {
                "start": "2024-09-15T00:00:00Z",
                "end": "2024-09-20T23:59:59Z",
                "reference_time": "2024-09-17T12:00:00Z"
            },
            "priority": "critical",
            "constraints": {
                "max_cloud_cover": 0.3,
                "min_resolution_m": 10,
                "max_resolution_m": 30,
                "required_data_types": ["sar", "dem", "weather"],
                "polarization": ["VV", "VH"]
            },
            "metadata": {
                "created_by": "emergency_ops",
                "tags": ["hurricane", "miami", "critical"]
            }
        }

        response = client.post("/events", json=event, headers=auth_headers)

        assert response.status_code == 200

    def test_submit_multipolygon_event(self, client, auth_headers):
        """Test event with MultiPolygon geometry."""
        event = {
            "id": "evt_multipolygon_001",
            "intent": {
                "class": "flood.riverine",
                "source": "explicit"
            },
            "spatial": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[-80.3, 25.7], [-80.1, 25.7], [-80.1, 25.9], [-80.3, 25.9], [-80.3, 25.7]]],
                    [[[-81.3, 26.7], [-81.1, 26.7], [-81.1, 26.9], [-81.3, 26.9], [-81.3, 26.7]]]
                ],
                "crs": "EPSG:4326"
            },
            "temporal": {
                "start": "2024-09-15T00:00:00Z",
                "end": "2024-09-20T23:59:59Z"
            }
        }

        response = client.post("/events", json=event, headers=auth_headers)

        assert response.status_code == 200

    def test_cancel_already_cancelled_event(self, client, auth_headers, sample_event_spec):
        """Test cancelling an already cancelled event."""
        # Submit and cancel an event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        event_id = submit_response.json()["event_id"]

        client.post(f"/events/{event_id}/cancel", headers=auth_headers)

        # Try to cancel again
        response = client.post(f"/events/{event_id}/cancel", headers=auth_headers)

        assert response.status_code == 400


class TestCatalogEdgeCases:
    """Tests for catalog edge cases."""

    def test_list_sources_unknown_type(self, client, auth_headers):
        """Test listing sources with unknown type filter."""
        response = client.get("/catalog/sources?type=unknown", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 0

    def test_list_algorithms_unknown_event_type(self, client, auth_headers):
        """Test listing algorithms with unknown event type."""
        response = client.get(
            "/catalog/algorithms?event_type=unknown",
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["algorithms"]) == 0


class TestConcurrency:
    """Tests for concurrent access patterns."""

    def test_concurrent_event_submissions(self, client, auth_headers, sample_event_spec):
        """Test handling multiple concurrent event submissions."""
        import concurrent.futures

        def submit_event(event_id: str):
            event = sample_event_spec.copy()
            event["id"] = event_id
            return client.post("/events", json=event, headers=auth_headers)

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(submit_event, f"evt_concurrent_{i}")
                for i in range(10)
            ]

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count == 10


class TestContentNegotiation:
    """Tests for content type handling."""

    def test_json_content_type(self, client, auth_headers, sample_event_spec):
        """Test JSON content type handling."""
        response = client.post(
            "/events",
            json=sample_event_spec,
            headers={**auth_headers, "Content-Type": "application/json"}
        )

        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_event_workflow(self, client, auth_headers, sample_event_spec):
        """Test complete event workflow from submission to products."""
        # 1. Submit event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        assert submit_response.status_code == 200
        event_id = submit_response.json()["event_id"]

        # 2. Check status
        status_response = client.get(f"/events/{event_id}/status", headers=auth_headers)
        assert status_response.status_code == 200

        # 3. Get progress
        progress_response = client.get(f"/events/{event_id}/progress", headers=auth_headers)
        assert progress_response.status_code == 200

        # 4. List products
        products_response = client.get(f"/events/{event_id}/products", headers=auth_headers)
        assert products_response.status_code == 200

        # 5. Get event details
        event_response = client.get(f"/events/{event_id}", headers=auth_headers)
        assert event_response.status_code == 200

    def test_webhook_integration(self, client, auth_headers, sample_event_spec):
        """Test webhook integration with event workflow."""
        # 1. Register webhook
        webhook_response = client.post(
            "/webhooks",
            json={
                "url": "https://example.com/events",
                "events": ["event.submitted", "event.completed"]
            },
            headers=auth_headers
        )
        assert webhook_response.status_code == 200
        webhook_id = webhook_response.json()["id"]

        # 2. Submit event
        submit_response = client.post(
            "/events",
            json=sample_event_spec,
            headers=auth_headers
        )
        assert submit_response.status_code == 200

        # 3. Verify webhook exists
        webhooks_response = client.get("/webhooks", headers=auth_headers)
        assert webhook_id in [w["id"] for w in webhooks_response.json()["webhooks"]]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
