"""
Integration tests for Phase 2 pipeline context storage.

Tests the Context Data Lakehouse integration in the discovery and pipeline
agents: dataset storage from discovery results, synthetic stub generation
for buildings/infrastructure/weather, deduplication (reuse), and the
guarantee that context storage failures never break the pipeline.

These tests exercise the ContextRepository directly rather than spinning up
the full DiscoveryAgent (which has complex constructor dependencies).

Requires a live PostGIS instance. Marked with @pytest.mark.integration.
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Skip all tests if asyncpg is not installed
try:
    import asyncpg

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed"),
]


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------

POSTGIS_HOST = os.environ.get("POSTGIS_TEST_HOST", "localhost")
POSTGIS_PORT = int(os.environ.get("POSTGIS_TEST_PORT", "5433"))
POSTGIS_DB = os.environ.get("POSTGIS_TEST_DB", "firstlight_test")
POSTGIS_USER = os.environ.get("POSTGIS_TEST_USER", "postgres")
POSTGIS_PASSWORD = os.environ.get("POSTGIS_TEST_PASSWORD", "testpass")


# ---------------------------------------------------------------------------
# Sample geometries
# ---------------------------------------------------------------------------

SAMPLE_AOI = {
    "type": "Polygon",
    "coordinates": [[
        [-95.5, 29.7],
        [-95.4, 29.7],
        [-95.4, 29.8],
        [-95.5, 29.8],
        [-95.5, 29.7],
    ]],
}

SAMPLE_MULTIPOLYGON = {
    "type": "MultiPolygon",
    "coordinates": [[[
        [-95.5, 29.7],
        [-95.4, 29.7],
        [-95.4, 29.8],
        [-95.5, 29.8],
        [-95.5, 29.7],
    ]]],
}

HOUSTON_BBOX: Tuple[float, float, float, float] = (-95.5, 29.7, -95.4, 29.8)


# ---------------------------------------------------------------------------
# Fixtures (same schema setup pattern as test_context_repository.py)
# ---------------------------------------------------------------------------

async def _create_test_pool():
    """Create a test connection pool."""
    dsn = (
        f"postgresql://{POSTGIS_USER}:{POSTGIS_PASSWORD}"
        f"@{POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DB}"
    )
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        return pool
    except (ConnectionRefusedError, asyncpg.InvalidCatalogNameError, OSError):
        return None


async def _setup_schema(pool):
    """Apply all migrations (000-007) to the test database."""
    import pathlib

    migrations_dir = pathlib.Path(__file__).parents[2] / "db" / "migrations"

    migration_files = [
        "000_add_customer_id.sql",
        "001_control_plane_schema.sql",
        "002_job_events_notify.sql",
        "003_webhook_tables.sql",
        "004_materialized_views.sql",
        "005_partition_job_events.sql",
        "006_pgstac_init.sql",
        "007_context_data.sql",
    ]

    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        # Drop existing tables for a clean slate
        await conn.execute("DROP TABLE IF EXISTS job_context_usage CASCADE")
        await conn.execute("DROP TABLE IF EXISTS context_weather CASCADE")
        await conn.execute("DROP TABLE IF EXISTS context_infrastructure CASCADE")
        await conn.execute("DROP TABLE IF EXISTS context_buildings CASCADE")
        await conn.execute("DROP TABLE IF EXISTS datasets CASCADE")
        await conn.execute("DROP TABLE IF EXISTS webhook_deliveries CASCADE")
        await conn.execute("DROP TABLE IF EXISTS webhook_configs CASCADE")
        await conn.execute("DROP TABLE IF EXISTS escalations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS job_checkpoints CASCADE")
        await conn.execute("DROP TABLE IF EXISTS job_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS jobs CASCADE")
        await conn.execute(
            "DROP FUNCTION IF EXISTS update_updated_at_column CASCADE"
        )
        await conn.execute(
            "DROP FUNCTION IF EXISTS compute_aoi_area_km2 CASCADE"
        )
        await conn.execute(
            "DROP FUNCTION IF EXISTS notify_job_event CASCADE"
        )
        await conn.execute(
            "DROP MATERIALIZED VIEW IF EXISTS mv_job_summary CASCADE"
        )
        await conn.execute(
            "DROP MATERIALIZED VIEW IF EXISTS mv_event_counts CASCADE"
        )

        for mig_file in migration_files:
            mig_path = migrations_dir / mig_file
            if mig_path.exists():
                sql = mig_path.read_text()
                try:
                    await conn.execute(sql)
                except Exception:
                    pass


@pytest_asyncio.fixture
async def pool():
    """Provide a test connection pool, skip if PostGIS is unavailable."""
    p = await _create_test_pool()
    if p is None:
        pytest.skip("PostGIS not available for integration tests")

    await _setup_schema(p)
    yield p
    await p.close()


@pytest_asyncio.fixture
async def test_job_id(pool) -> uuid.UUID:
    """Create a test job and return its UUID."""
    job_id = uuid.uuid4()
    aoi_json = json.dumps(SAMPLE_AOI)

    await pool.execute(
        """
        INSERT INTO jobs (job_id, customer_id, event_type, aoi, phase, status, parameters)
        VALUES ($1, 'test-tenant', 'flood',
                ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($2), 4326)),
                'QUEUED', 'PENDING', '{}'::jsonb)
        """,
        job_id,
        aoi_json,
    )
    return job_id


@pytest_asyncio.fixture
async def second_job_id(pool) -> uuid.UUID:
    """Create a second test job for reuse testing."""
    job_id = uuid.uuid4()
    aoi_json = json.dumps(SAMPLE_AOI)

    await pool.execute(
        """
        INSERT INTO jobs (job_id, customer_id, event_type, aoi, phase, status, parameters)
        VALUES ($1, 'test-tenant', 'flood',
                ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($2), 4326)),
                'QUEUED', 'PENDING', '{}'::jsonb)
        """,
        job_id,
        aoi_json,
    )
    return job_id


@pytest_asyncio.fixture
async def repo(pool):
    """Create a ContextRepository sharing the test pool."""
    from core.context.repository import ContextRepository

    repo = ContextRepository(pool=pool)
    yield repo
    # Pool is closed by the pool fixture


# ===========================================================================
# Task 2.1 — Discovery stores dataset records
# ===========================================================================


@pytest.mark.asyncio
async def test_discovery_stores_dataset_record(repo, test_job_id):
    """Storing a DatasetRecord via store_dataset works end-to-end."""
    from core.context.models import DatasetRecord

    record = DatasetRecord(
        source="earth_search",
        source_id="S2A_MSIL1C_20230101T000000",
        geometry=SAMPLE_MULTIPOLYGON,
        properties={"platform": "sentinel-2a"},
        acquisition_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        cloud_cover=12.5,
        resolution_m=10.0,
        bands=["B02", "B03", "B04", "B08"],
    )

    result = await repo.store_dataset(test_job_id, record)

    assert result.usage_type == "ingested"
    assert result.context_id is not None


@pytest.mark.asyncio
async def test_duplicate_discovery_results_in_reused(repo, test_job_id, second_job_id):
    """
    Same (source, source_id) stored by two jobs: first is 'ingested',
    second is 'reused'.
    """
    from core.context.models import DatasetRecord

    record = DatasetRecord(
        source="earth_search",
        source_id="S2A_MSIL1C_20230615T120000_DEDUP",
        geometry=SAMPLE_MULTIPOLYGON,
        properties={"platform": "sentinel-2a"},
        acquisition_date=datetime(2023, 6, 15, tzinfo=timezone.utc),
        cloud_cover=5.0,
        resolution_m=10.0,
    )

    r1 = await repo.store_dataset(test_job_id, record)
    r2 = await repo.store_dataset(second_job_id, record)

    assert r1.usage_type == "ingested"
    assert r2.usage_type == "reused"
    # Both point to the same context row
    assert r1.context_id == r2.context_id


@pytest.mark.asyncio
async def test_job_context_summary_after_discovery(repo, test_job_id):
    """get_job_context_summary returns correct counts after storing datasets."""
    from core.context.models import DatasetRecord

    for i in range(3):
        await repo.store_dataset(
            test_job_id,
            DatasetRecord(
                source="earth_search",
                source_id=f"scene_summary_{i}",
                geometry=SAMPLE_MULTIPOLYGON,
                properties={},
                acquisition_date=datetime(2023, 1, i + 1, tzinfo=timezone.utc),
            ),
        )

    summary = await repo.get_job_context_summary(test_job_id)
    assert summary.datasets_ingested == 3
    assert summary.total_ingested == 3
    assert summary.total == 3


# ===========================================================================
# Task 2.1 — DiscoveryAgent geometry derivation
# ===========================================================================


def test_derive_geometry_from_polygon():
    """_derive_geometry_from_spatial converts a Polygon to MultiPolygon."""
    from agents.discovery.main import DiscoveryAgent

    spatial = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    geom = DiscoveryAgent._derive_geometry_from_spatial(spatial)
    assert geom is not None
    assert geom["type"] == "MultiPolygon"
    assert len(geom["coordinates"]) == 1


def test_derive_geometry_from_bbox():
    """_derive_geometry_from_spatial converts a bbox dict to MultiPolygon."""
    from agents.discovery.main import DiscoveryAgent

    spatial = {"bbox": [-95.5, 29.5, -95.0, 30.0]}
    geom = DiscoveryAgent._derive_geometry_from_spatial(spatial)
    assert geom is not None
    assert geom["type"] == "MultiPolygon"


def test_derive_geometry_from_multipolygon():
    """_derive_geometry_from_spatial passes through MultiPolygon."""
    from agents.discovery.main import DiscoveryAgent

    geom = DiscoveryAgent._derive_geometry_from_spatial(SAMPLE_MULTIPOLYGON)
    assert geom is SAMPLE_MULTIPOLYGON


def test_derive_geometry_returns_none_for_empty():
    """_derive_geometry_from_spatial returns None for unsupported input."""
    from agents.discovery.main import DiscoveryAgent

    assert DiscoveryAgent._derive_geometry_from_spatial({}) is None


# ===========================================================================
# Tasks 2.2-2.4 — Synthetic stubs store into repo
# ===========================================================================


@pytest.mark.asyncio
async def test_synthetic_buildings_stored(repo, test_job_id):
    """Synthetic building stubs can be batch-stored into the lakehouse."""
    from core.context.stubs import generate_buildings

    buildings = generate_buildings(HOUSTON_BBOX, count=5, seed=42)
    assert len(buildings) == 5
    assert all(b.source == "synthetic" for b in buildings)

    results = await repo.store_batch(
        test_job_id, "context_buildings", buildings
    )
    assert len(results) == 5
    assert all(r.usage_type == "ingested" for r in results)


@pytest.mark.asyncio
async def test_synthetic_infrastructure_mixed_geom(repo, test_job_id):
    """Infrastructure stubs contain both POINT and POLYGON geometries."""
    from core.context.stubs import generate_infrastructure

    infra = generate_infrastructure(HOUSTON_BBOX, count=8, seed=42)
    geom_types = {f.geometry["type"] for f in infra}
    assert "Point" in geom_types
    assert "Polygon" in geom_types

    results = await repo.store_batch(
        test_job_id, "context_infrastructure", infra
    )
    assert len(results) == 8


@pytest.mark.asyncio
async def test_synthetic_weather_stored(repo, test_job_id):
    """Weather stubs with realistic properties can be stored."""
    from core.context.stubs import generate_weather

    ref_time = datetime(2023, 8, 1, 12, 0, tzinfo=timezone.utc)
    weather = generate_weather(HOUSTON_BBOX, count=5, reference_time=ref_time, seed=42)

    assert len(weather) == 5
    # All should be Points
    assert all(w.geometry["type"] == "Point" for w in weather)
    # Properties should have weather fields
    for w in weather:
        assert "temperature_c" in w.properties
        assert "precipitation_mm" in w.properties
        assert "wind_speed_ms" in w.properties

    results = await repo.store_batch(test_job_id, "context_weather", weather)
    assert len(results) == 5


@pytest.mark.asyncio
async def test_synthetic_reuse_across_jobs(repo, test_job_id, second_job_id):
    """
    Synthetic stubs with the same source_id are reused (not duplicated)
    across jobs.
    """
    from core.context.models import BuildingRecord

    # Same building record stored by two jobs
    record = BuildingRecord(
        source="synthetic",
        source_id="synth_bldg_shared_001",
        geometry={
            "type": "Polygon",
            "coordinates": [[
                [-95.45, 29.73],
                [-95.44, 29.73],
                [-95.44, 29.74],
                [-95.45, 29.74],
                [-95.45, 29.73],
            ]],
        },
        properties={"type": "residential"},
    )

    r1 = await repo.store_building(test_job_id, record)
    r2 = await repo.store_building(second_job_id, record)

    assert r1.usage_type == "ingested"
    assert r2.usage_type == "reused"
    assert r1.context_id == r2.context_id


# ===========================================================================
# Task 2.5 — Context storage failure doesn't fail pipeline
# ===========================================================================


@pytest.mark.asyncio
async def test_context_storage_failure_does_not_block_discovery():
    """
    If the ContextRepository raises an exception during store_dataset,
    _store_discovery_context logs a warning but does NOT raise.
    """
    from agents.discovery.main import DiscoveryAgent
    from core.data.discovery.base import DiscoveryResult

    # Create a DiscoveryAgent with a broken context repo
    broken_repo = MagicMock()
    broken_repo.store_dataset = AsyncMock(
        side_effect=RuntimeError("DB connection lost")
    )
    job_id = uuid.uuid4()

    agent = DiscoveryAgent(
        context_repo=broken_repo,
        job_id=job_id,
    )

    # Build a fake CatalogQueryResult
    from agents.discovery.catalog import CatalogQueryResult, CatalogQueryStatus

    fake_result = CatalogQueryResult(
        catalog_id="test",
        provider_id="test_provider",
        status=CatalogQueryStatus.COMPLETED,
        results=[
            DiscoveryResult(
                dataset_id="broken_scene_001",
                provider="test_provider",
                data_type="optical",
                source_uri="s3://bucket/scene.tif",
                format="cog",
                acquisition_time=datetime(2023, 1, 1, tzinfo=timezone.utc),
                spatial_coverage_percent=90.0,
                resolution_m=10.0,
                metadata={"stac_item": "broken_scene_001"},
            ),
        ],
    )

    spatial = {"bbox": [-95.5, 29.5, -95.0, 30.0]}

    # This should NOT raise, even though the repo is broken
    await agent._store_discovery_context([fake_result], spatial)

    # Verify the call was attempted
    assert broken_repo.store_dataset.called


@pytest.mark.asyncio
async def test_context_storage_failure_does_not_block_pipeline():
    """
    If the ContextRepository raises during batch store,
    PipelineAgent._store_synthetic_context logs a warning but does NOT raise.
    """
    from agents.pipeline.main import PipelineAgent

    broken_repo = MagicMock()
    broken_repo.store_batch = AsyncMock(
        side_effect=RuntimeError("DB connection lost")
    )
    job_id = uuid.uuid4()

    agent = PipelineAgent(
        context_repo=broken_repo,
        job_id=job_id,
    )

    # This should NOT raise
    await agent._store_synthetic_context(HOUSTON_BBOX)

    # store_batch was called (and failed silently for each type)
    assert broken_repo.store_batch.call_count == 3  # buildings, infra, weather


# ===========================================================================
# PipelineAgent bbox extraction
# ===========================================================================


def test_extract_bbox_from_direct_bbox():
    """_extract_bbox_from_inputs extracts a direct bbox key."""
    from agents.pipeline.main import PipelineAgent

    inputs = {"bbox": [-95.5, 29.7, -95.4, 29.8]}
    result = PipelineAgent._extract_bbox_from_inputs(inputs)
    assert result == (-95.5, 29.7, -95.4, 29.8)


def test_extract_bbox_from_aoi_geojson():
    """_extract_bbox_from_inputs extracts bbox from an aoi GeoJSON Polygon."""
    from agents.pipeline.main import PipelineAgent

    inputs = {"aoi": SAMPLE_AOI}
    result = PipelineAgent._extract_bbox_from_inputs(inputs)
    assert result is not None
    west, south, east, north = result
    assert west == pytest.approx(-95.5)
    assert south == pytest.approx(29.7)


def test_extract_bbox_returns_none_for_empty():
    """_extract_bbox_from_inputs returns None when no geometry is found."""
    from agents.pipeline.main import PipelineAgent

    assert PipelineAgent._extract_bbox_from_inputs({}) is None
    assert PipelineAgent._extract_bbox_from_inputs({"data": "foo"}) is None


# ===========================================================================
# Full context summary after mixed storage
# ===========================================================================


@pytest.mark.asyncio
async def test_full_context_summary(repo, test_job_id):
    """
    After storing datasets + buildings + infrastructure + weather for one job,
    get_job_context_summary returns correct per-table counts.
    """
    from core.context.models import DatasetRecord
    from core.context.stubs import (
        generate_buildings,
        generate_infrastructure,
        generate_weather,
    )

    # Store 2 datasets
    for i in range(2):
        await repo.store_dataset(
            test_job_id,
            DatasetRecord(
                source="earth_search",
                source_id=f"full_summary_scene_{i}",
                geometry=SAMPLE_MULTIPOLYGON,
                properties={},
                acquisition_date=datetime(2023, 1, i + 1, tzinfo=timezone.utc),
            ),
        )

    # Store 3 buildings
    buildings = generate_buildings(HOUSTON_BBOX, count=3, seed=100)
    await repo.store_batch(test_job_id, "context_buildings", buildings)

    # Store 2 infrastructure
    infra = generate_infrastructure(HOUSTON_BBOX, count=2, seed=100)
    await repo.store_batch(test_job_id, "context_infrastructure", infra)

    # Store 4 weather
    weather = generate_weather(HOUSTON_BBOX, count=4, seed=100)
    await repo.store_batch(test_job_id, "context_weather", weather)

    summary = await repo.get_job_context_summary(test_job_id)

    assert summary.datasets_ingested == 2
    assert summary.buildings_ingested == 3
    assert summary.infrastructure_ingested == 2
    assert summary.weather_ingested == 4
    assert summary.total_ingested == 11
    assert summary.total == 11
