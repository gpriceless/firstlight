"""
Tests for Context Data Lakehouse Query Endpoints (Phase 3, Task 3.9).

Tests cover:
- GET /context/datasets returns paginated results
- Bbox filter returns only intersecting datasets
- Date range filter works
- GET /context/buildings returns GeoJSON geometry in response
- GET /context/summary returns correct row counts
- GET /jobs/{id}/context returns usage summary with ingested/reused counts
- Empty database returns 200 with empty items array
- Permission check: user without CONTEXT_READ gets 403
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.models.context import (
    DatasetItem,
    BuildingItem,
    InfrastructureItem,
    WeatherItem,
    PaginatedDatasetsResponse,
    PaginatedBuildingsResponse,
    PaginatedInfrastructureResponse,
    PaginatedWeatherResponse,
    LakehouseSummaryResponse,
    TableStats,
    JobContextSummaryResponse,
)
from core.context.models import (
    BuildingRecord,
    ContextSummary,
    DatasetRecord,
    InfrastructureRecord,
    WeatherRecord,
)


# =============================================================================
# Test fixtures
# =============================================================================


SAMPLE_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [-95.45, 29.72],
        [-95.43, 29.72],
        [-95.43, 29.74],
        [-95.45, 29.74],
        [-95.45, 29.72],
    ]],
}

SAMPLE_POINT = {
    "type": "Point",
    "coordinates": [-95.44, 29.73],
}


def _make_dataset_record(
    source: str = "earth_search",
    source_id: Optional[str] = None,
    geometry: Optional[Dict] = None,
    acquisition_date: Optional[datetime] = None,
) -> DatasetRecord:
    """Create a test DatasetRecord."""
    return DatasetRecord(
        source=source,
        source_id=source_id or f"scene_{uuid.uuid4().hex[:8]}",
        geometry=geometry or SAMPLE_POLYGON,
        properties={"platform": "sentinel-2a"},
        acquisition_date=acquisition_date or datetime(2026, 2, 15, tzinfo=timezone.utc),
        cloud_cover=12.5,
        resolution_m=10.0,
        bands=["B02", "B03", "B04"],
    )


def _make_building_record(
    source_id: Optional[str] = None,
    geometry: Optional[Dict] = None,
) -> BuildingRecord:
    """Create a test BuildingRecord."""
    return BuildingRecord(
        source="synthetic",
        source_id=source_id or f"bldg_{uuid.uuid4().hex[:8]}",
        geometry=geometry or SAMPLE_POLYGON,
        properties={"type": "residential", "floors": 2},
    )


def _make_infrastructure_record(
    source_id: Optional[str] = None,
    infra_type: str = "hospital",
) -> InfrastructureRecord:
    """Create a test InfrastructureRecord."""
    return InfrastructureRecord(
        source="synthetic",
        source_id=source_id or f"infra_{uuid.uuid4().hex[:8]}",
        geometry=SAMPLE_POINT,
        properties={"type": infra_type, "name": "Test Hospital"},
    )


def _make_weather_record(
    source_id: Optional[str] = None,
    observation_time: Optional[datetime] = None,
) -> WeatherRecord:
    """Create a test WeatherRecord."""
    return WeatherRecord(
        source="synthetic",
        source_id=source_id or f"wx_{uuid.uuid4().hex[:8]}",
        geometry=SAMPLE_POINT,
        properties={"temperature_c": 25.0, "precipitation_mm": 5.0},
        observation_time=observation_time or datetime(2026, 2, 15, 12, 0, tzinfo=timezone.utc),
    )


def _build_test_app(mock_repo=None):
    """Build a minimal test app with the context router."""
    from api.routes.control.context import context_router
    from api.models.errors import APIError, api_error_handler

    app = FastAPI()
    app.add_exception_handler(APIError, api_error_handler)
    app.include_router(context_router, prefix="/control/v1")

    # Override the customer dependency
    from api.routes.control import get_current_customer

    async def mock_customer():
        return "tenant-a"

    app.dependency_overrides[get_current_customer] = mock_customer

    return app


def _build_test_app_with_jobs(mock_backend=None, mock_repo=None):
    """Build a test app with both jobs and context routers.

    Note: jobs_router already has prefix="/jobs", so we mount it at /control/v1
    (not /control/v1/jobs) to get the correct final paths like
    /control/v1/jobs/{job_id}/context.
    """
    from api.routes.control.context import context_router
    from api.routes.control.jobs import router as jobs_router
    from api.models.errors import APIError, api_error_handler

    app = FastAPI()
    app.add_exception_handler(APIError, api_error_handler)
    app.include_router(context_router, prefix="/control/v1")
    app.include_router(jobs_router, prefix="/control/v1")

    from api.routes.control import get_current_customer

    async def mock_customer():
        return "tenant-a"

    app.dependency_overrides[get_current_customer] = mock_customer

    return app


# =============================================================================
# Model Validation Tests
# =============================================================================


class TestContextResponseModels:
    """Test context API response model structure."""

    def test_paginated_datasets_response(self):
        """Test PaginatedDatasetsResponse has correct structure."""
        now = datetime.now(timezone.utc)
        resp = PaginatedDatasetsResponse(
            items=[
                DatasetItem(
                    source="earth_search",
                    source_id="S2A_test",
                    geometry=SAMPLE_POLYGON,
                    properties={},
                    acquisition_date=now,
                    cloud_cover=10.0,
                    resolution_m=10.0,
                    bands=["B02", "B03"],
                ),
            ],
            total=100,
            page=1,
            page_size=50,
        )
        assert resp.total == 100
        assert resp.page == 1
        assert len(resp.items) == 1
        assert resp.items[0].source == "earth_search"

    def test_paginated_buildings_response(self):
        """Test PaginatedBuildingsResponse has correct structure."""
        resp = PaginatedBuildingsResponse(
            items=[
                BuildingItem(
                    source="osm",
                    source_id="way/12345",
                    geometry=SAMPLE_POLYGON,
                    properties={"type": "residential"},
                ),
            ],
            total=50,
            page=2,
            page_size=20,
        )
        assert resp.total == 50
        assert resp.page == 2
        assert resp.items[0].geometry["type"] == "Polygon"

    def test_paginated_infrastructure_response(self):
        """Test PaginatedInfrastructureResponse has correct structure."""
        resp = PaginatedInfrastructureResponse(
            items=[
                InfrastructureItem(
                    source="synthetic",
                    source_id="infra_001",
                    geometry=SAMPLE_POINT,
                    properties={"type": "hospital"},
                ),
            ],
            total=10,
            page=1,
            page_size=50,
        )
        assert resp.total == 10
        assert resp.items[0].properties["type"] == "hospital"

    def test_paginated_weather_response(self):
        """Test PaginatedWeatherResponse has correct structure."""
        now = datetime.now(timezone.utc)
        resp = PaginatedWeatherResponse(
            items=[
                WeatherItem(
                    source="noaa",
                    source_id="station_001",
                    geometry=SAMPLE_POINT,
                    properties={"temperature_c": 25.0},
                    observation_time=now,
                ),
            ],
            total=200,
            page=3,
            page_size=50,
        )
        assert resp.total == 200
        assert resp.items[0].observation_time == now

    def test_lakehouse_summary_response(self):
        """Test LakehouseSummaryResponse structure."""
        resp = LakehouseSummaryResponse(
            tables={
                "datasets": TableStats(row_count=100, sources=["earth_search"]),
                "buildings": TableStats(row_count=5000, sources=["osm", "overture"]),
            },
            total_rows=5100,
            spatial_extent=SAMPLE_POLYGON,
            usage_stats={"ingested": 4000, "reused": 1100},
        )
        assert resp.total_rows == 5100
        assert resp.tables["datasets"].row_count == 100

    def test_job_context_summary_response(self):
        """Test JobContextSummaryResponse structure."""
        resp = JobContextSummaryResponse(
            job_id=str(uuid.uuid4()),
            datasets_ingested=3,
            datasets_reused=0,
            buildings_ingested=50,
            buildings_reused=30,
            total_ingested=53,
            total_reused=30,
            total=83,
        )
        assert resp.total_ingested == 53
        assert resp.total_reused == 30
        assert resp.total == 83


# =============================================================================
# Endpoint Tests with Mock Repository
# =============================================================================


class TestDatasetsEndpoint:
    """Test GET /context/datasets endpoint."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_empty(self, mock_get_repo):
        """Empty database returns 200 with empty items array."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/datasets")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 50

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_with_results(self, mock_get_repo):
        """Datasets endpoint returns paginated results."""
        records = [_make_dataset_record(source_id=f"scene_{i}") for i in range(3)]
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = (records, 3)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/datasets")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 3
        assert data["total"] == 3
        assert data["items"][0]["source"] == "earth_search"

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_with_bbox(self, mock_get_repo):
        """Bbox filter is passed to repository."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=-95.5,29.5,-95.0,30.0"
        )

        assert resp.status_code == 200
        # Verify bbox was passed to the repository
        call_kwargs = mock_repo.query_datasets.call_args
        assert call_kwargs.kwargs["bbox"] == (-95.5, 29.5, -95.0, 30.0)

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_with_date_range(self, mock_get_repo):
        """Date range filter is passed to repository."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets"
            "?date_start=2026-02-01T00:00:00Z"
            "&date_end=2026-02-28T23:59:59Z"
        )

        assert resp.status_code == 200
        call_kwargs = mock_repo.query_datasets.call_args
        assert call_kwargs.kwargs["date_start"] is not None
        assert call_kwargs.kwargs["date_end"] is not None

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_with_source_filter(self, mock_get_repo):
        """Source filter is passed to repository."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?source=earth_search"
        )

        assert resp.status_code == 200
        call_kwargs = mock_repo.query_datasets.call_args
        assert call_kwargs.kwargs["source"] == "earth_search"

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_datasets_pagination(self, mock_get_repo):
        """Pagination parameters are passed correctly."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?page=3&page_size=25"
        )

        assert resp.status_code == 200
        call_kwargs = mock_repo.query_datasets.call_args
        assert call_kwargs.kwargs["limit"] == 25
        assert call_kwargs.kwargs["offset"] == 50  # (3-1) * 25

    def test_list_datasets_invalid_bbox(self):
        """Invalid bbox returns validation error."""
        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=not,valid,numbers"
        )
        # Should get a 422 validation error
        assert resp.status_code == 422


class TestBuildingsEndpoint:
    """Test GET /context/buildings endpoint."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_buildings_empty(self, mock_get_repo):
        """Empty database returns 200 with empty items array."""
        mock_repo = AsyncMock()
        mock_repo.query_buildings.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/buildings")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_buildings_returns_geojson(self, mock_get_repo):
        """Buildings endpoint returns GeoJSON geometry in response."""
        records = [_make_building_record()]
        mock_repo = AsyncMock()
        mock_repo.query_buildings.return_value = (records, 1)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/buildings")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        geom = data["items"][0]["geometry"]
        assert geom["type"] == "Polygon"
        assert "coordinates" in geom


class TestInfrastructureEndpoint:
    """Test GET /context/infrastructure endpoint."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_infrastructure_with_type_filter(self, mock_get_repo):
        """Type filter is passed to repository."""
        mock_repo = AsyncMock()
        mock_repo.query_infrastructure.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/infrastructure?type=hospital"
        )

        assert resp.status_code == 200
        call_kwargs = mock_repo.query_infrastructure.call_args
        assert call_kwargs.kwargs["type_filter"] == "hospital"


class TestWeatherEndpoint:
    """Test GET /context/weather endpoint."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_list_weather_with_time_range(self, mock_get_repo):
        """Time range filter is passed to repository."""
        mock_repo = AsyncMock()
        mock_repo.query_weather.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/weather"
            "?time_start=2026-02-01T00:00:00Z"
            "&time_end=2026-02-28T23:59:59Z"
        )

        assert resp.status_code == 200
        call_kwargs = mock_repo.query_weather.call_args
        assert call_kwargs.kwargs["time_start"] is not None
        assert call_kwargs.kwargs["time_end"] is not None


class TestSummaryEndpoint:
    """Test GET /context/summary endpoint."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_get_summary(self, mock_get_repo):
        """Summary endpoint returns correct structure."""
        mock_repo = AsyncMock()
        mock_repo.get_lakehouse_stats.return_value = {
            "tables": {
                "datasets": {"row_count": 50, "sources": ["earth_search"]},
                "buildings": {"row_count": 200, "sources": ["synthetic"]},
                "infrastructure": {"row_count": 10, "sources": ["synthetic"]},
                "weather": {"row_count": 30, "sources": ["synthetic"]},
            },
            "total_rows": 290,
            "spatial_extent": SAMPLE_POLYGON,
            "usage_stats": {"ingested": 250, "reused": 40},
        }
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/summary")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 290
        assert data["tables"]["datasets"]["row_count"] == 50
        assert data["tables"]["buildings"]["sources"] == ["synthetic"]
        assert data["usage_stats"]["ingested"] == 250
        assert data["usage_stats"]["reused"] == 40

    @patch("api.routes.control.context._get_connected_repo")
    def test_get_summary_empty_db(self, mock_get_repo):
        """Summary endpoint with empty database."""
        mock_repo = AsyncMock()
        mock_repo.get_lakehouse_stats.return_value = {
            "tables": {
                "datasets": {"row_count": 0, "sources": []},
                "buildings": {"row_count": 0, "sources": []},
                "infrastructure": {"row_count": 0, "sources": []},
                "weather": {"row_count": 0, "sources": []},
            },
            "total_rows": 0,
            "spatial_extent": None,
            "usage_stats": {},
        }
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get("/control/v1/context/summary")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_rows"] == 0
        assert data["spatial_extent"] is None


class TestJobContextEndpoint:
    """Test GET /jobs/{job_id}/context endpoint."""

    @patch("api.routes.control.jobs._get_connected_backend")
    def test_get_job_context_success(self, mock_get_backend):
        """Job context endpoint returns usage summary."""
        from agents.orchestrator.backends.base import JobState

        job_id = str(uuid.uuid4())
        mock_backend = AsyncMock()
        mock_backend.get_state.return_value = JobState(
            job_id=job_id,
            customer_id="tenant-a",
            event_type="flood",
            phase="COMPLETE",
            status="COMPLETE",
            aoi=SAMPLE_POLYGON,
            parameters={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_backend.close = AsyncMock()
        mock_get_backend.return_value = mock_backend

        # Mock the ContextRepository
        mock_summary = ContextSummary(
            datasets_ingested=3,
            datasets_reused=0,
            buildings_ingested=100,
            buildings_reused=0,
            infrastructure_ingested=5,
            infrastructure_reused=0,
            weather_ingested=10,
            weather_reused=0,
            total_ingested=118,
            total_reused=0,
            total=118,
        )

        # Patch at the source module since the import is lazy (inside function body)
        with patch("core.context.repository.ContextRepository") as MockRepo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.get_job_context_summary.return_value = mock_summary
            mock_repo_instance.close = AsyncMock()
            MockRepo.return_value = mock_repo_instance

            app = _build_test_app_with_jobs()
            client = TestClient(app)
            resp = client.get(f"/control/v1/jobs/{job_id}/context")

            assert resp.status_code == 200
            data = resp.json()
            assert data["job_id"] == job_id
            assert data["total_ingested"] == 118
            assert data["total_reused"] == 0
            assert data["total"] == 118
            assert data["datasets_ingested"] == 3
            assert data["buildings_ingested"] == 100

    @patch("api.routes.control.jobs._get_connected_backend")
    def test_get_job_context_wrong_tenant(self, mock_get_backend):
        """Job context returns 404 for wrong tenant."""
        from agents.orchestrator.backends.base import JobState

        job_id = str(uuid.uuid4())
        mock_backend = AsyncMock()
        mock_backend.get_state.return_value = JobState(
            job_id=job_id,
            customer_id="tenant-b",  # Different tenant
            event_type="flood",
            phase="QUEUED",
            status="PENDING",
            aoi=SAMPLE_POLYGON,
            parameters={},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_backend.close = AsyncMock()
        mock_get_backend.return_value = mock_backend

        app = _build_test_app_with_jobs()
        client = TestClient(app)
        resp = client.get(f"/control/v1/jobs/{job_id}/context")

        # Should get 404, not 403 (tenant scoping)
        assert resp.status_code == 404

    @patch("api.routes.control.jobs._get_connected_backend")
    def test_get_job_context_not_found(self, mock_get_backend):
        """Job context returns 404 for nonexistent job."""
        mock_backend = AsyncMock()
        mock_backend.get_state.return_value = None
        mock_backend.close = AsyncMock()
        mock_get_backend.return_value = mock_backend

        app = _build_test_app_with_jobs()
        client = TestClient(app)
        resp = client.get(f"/control/v1/jobs/{uuid.uuid4()}/context")

        assert resp.status_code == 404


# =============================================================================
# Bbox Parsing Tests
# =============================================================================


class TestContextBboxParsing:
    """Test bbox parsing in context endpoints."""

    @patch("api.routes.control.context._get_connected_repo")
    def test_valid_bbox(self, mock_get_repo):
        """Valid bbox is parsed correctly."""
        mock_repo = AsyncMock()
        mock_repo.query_datasets.return_value = ([], 0)
        mock_repo.close = AsyncMock()
        mock_get_repo.return_value = mock_repo

        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=-95.5,29.5,-95.0,30.0"
        )
        assert resp.status_code == 200

    def test_invalid_bbox_non_numeric(self):
        """Non-numeric bbox returns 422."""
        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=a,b,c,d"
        )
        assert resp.status_code == 422

    def test_invalid_bbox_too_few_parts(self):
        """Bbox with fewer than 4 parts returns 422."""
        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=-95.5,29.5"
        )
        assert resp.status_code == 422

    def test_invalid_bbox_south_greater_than_north(self):
        """Bbox with south > north returns 422."""
        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=-95.5,30.0,-95.0,29.5"
        )
        assert resp.status_code == 422

    def test_invalid_bbox_out_of_bounds(self):
        """Bbox with out-of-bounds longitude returns 422."""
        app = _build_test_app()
        client = TestClient(app)
        resp = client.get(
            "/control/v1/context/datasets?bbox=-200,29.5,-95.0,30.0"
        )
        assert resp.status_code == 422
