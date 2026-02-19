"""
Tests for the lakehouse effect: context data reuse across overlapping jobs.

These tests require a live PostGIS instance. They are marked with
@pytest.mark.integration so they can be skipped in CI without PostGIS.

Tests cover:
- Two jobs with overlapping AOIs result in shared context rows with correct
  usage_type values (first job: 'ingested', second job: 'reused')
- Non-overlapping jobs do not share context
- Junction table correctly tracks provenance (each job has its own link)
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

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

# Houston area polygon A (west Houston)
HOUSTON_POLYGON_A = {
    "type": "Polygon",
    "coordinates": [[
        [-95.45, 29.72],
        [-95.40, 29.72],
        [-95.40, 29.76],
        [-95.45, 29.76],
        [-95.45, 29.72],
    ]],
}

# Houston area polygon B (overlapping -- shifted east by ~0.03 degrees)
HOUSTON_POLYGON_B = {
    "type": "Polygon",
    "coordinates": [[
        [-95.42, 29.73],
        [-95.37, 29.73],
        [-95.37, 29.77],
        [-95.42, 29.77],
        [-95.42, 29.73],
    ]],
}

# Far-away polygon (no overlap with Houston)
REMOTE_POLYGON = {
    "type": "Polygon",
    "coordinates": [[
        [-80.0, 40.0],
        [-79.95, 40.0],
        [-79.95, 40.05],
        [-80.0, 40.05],
        [-80.0, 40.0],
    ]],
}

# Building inside the overlap zone (inside both A and B)
OVERLAP_BUILDING = {
    "type": "Polygon",
    "coordinates": [[
        [-95.415, 29.735],
        [-95.414, 29.735],
        [-95.414, 29.736],
        [-95.415, 29.736],
        [-95.415, 29.735],
    ]],
}

# Building inside only polygon A (not in B)
A_ONLY_BUILDING = {
    "type": "Polygon",
    "coordinates": [[
        [-95.448, 29.725],
        [-95.447, 29.725],
        [-95.447, 29.726],
        [-95.448, 29.726],
        [-95.448, 29.725],
    ]],
}

# Weather point in the overlap zone
OVERLAP_POINT = {
    "type": "Point",
    "coordinates": [-95.41, 29.74],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pool():
    """Create an asyncpg connection pool and apply migrations."""
    try:
        dsn = (
            f"postgresql://{POSTGIS_USER}:{POSTGIS_PASSWORD}"
            f"@{POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DB}"
        )
        _pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
    except (asyncpg.InvalidCatalogNameError, OSError, asyncpg.CannotConnectNowError):
        pytest.skip(f"PostGIS not available at {POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DB}")
        return

    # Apply all migrations
    import glob
    migration_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "db", "migrations"
    )
    migration_files = sorted(glob.glob(os.path.join(migration_dir, "*.sql")))

    async with _pool.acquire() as conn:
        for mfile in migration_files:
            with open(mfile, "r") as f:
                sql = f.read()
            try:
                await conn.execute(sql)
            except Exception:
                pass  # Idempotent migrations may warn on re-run

    yield _pool

    # Clean up test data
    async with _pool.acquire() as conn:
        await conn.execute("DELETE FROM job_context_usage")
        await conn.execute("DELETE FROM context_buildings")
        await conn.execute("DELETE FROM context_infrastructure")
        await conn.execute("DELETE FROM context_weather")
        await conn.execute("DELETE FROM datasets")
        # Clean up test jobs
        await conn.execute(
            "DELETE FROM jobs WHERE customer_id = 'test-lakehouse'"
        )

    await _pool.close()


@pytest_asyncio.fixture
async def repo(pool):
    """Create a ContextRepository backed by the test pool."""
    from core.context.repository import ContextRepository

    r = ContextRepository(pool=pool)
    yield r


@pytest_asyncio.fixture
async def job_ids(pool):
    """Create two test jobs and return their UUIDs."""
    from agents.orchestrator.backends.postgis_backend import PostGISStateBackend

    backend = PostGISStateBackend(pool=pool)

    job_a = await backend.create_job(
        customer_id="test-lakehouse",
        event_type="flood",
        aoi_geojson=HOUSTON_POLYGON_A,
        phase="QUEUED",
        status="PENDING",
    )
    job_b = await backend.create_job(
        customer_id="test-lakehouse",
        event_type="flood",
        aoi_geojson=HOUSTON_POLYGON_B,
        phase="QUEUED",
        status="PENDING",
    )

    return uuid.UUID(job_a.job_id), uuid.UUID(job_b.job_id)


@pytest_asyncio.fixture
async def remote_job_id(pool):
    """Create a job with a remote (non-overlapping) AOI."""
    from agents.orchestrator.backends.postgis_backend import PostGISStateBackend

    backend = PostGISStateBackend(pool=pool)

    job = await backend.create_job(
        customer_id="test-lakehouse",
        event_type="flood",
        aoi_geojson=REMOTE_POLYGON,
        phase="QUEUED",
        status="PENDING",
    )

    return uuid.UUID(job.job_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_overlapping_jobs_share_context(repo, job_ids):
    """
    Two jobs with overlapping AOIs share context rows. The first job
    gets usage_type='ingested', the second gets 'reused'.
    """
    from core.context.models import BuildingRecord

    job_a, job_b = job_ids

    # Job A ingests a building in the overlap zone
    building = BuildingRecord(
        source="synthetic",
        source_id="overlap_bldg_001",
        geometry=OVERLAP_BUILDING,
        properties={"type": "residential"},
    )

    result_a = await repo.store_building(job_a, building)
    assert result_a.usage_type == "ingested"

    # Job B tries to store the SAME building (same source + source_id)
    result_b = await repo.store_building(job_b, building)
    assert result_b.usage_type == "reused"

    # Both results reference the same context_id
    assert result_a.context_id == result_b.context_id


@pytest.mark.asyncio
async def test_overlapping_jobs_correct_summaries(repo, job_ids):
    """
    Job summaries correctly reflect ingested vs reused counts.
    """
    from core.context.models import BuildingRecord, WeatherRecord

    job_a, job_b = job_ids

    # Job A: ingest 3 buildings + 2 weather
    for i in range(3):
        building = BuildingRecord(
            source="synthetic",
            source_id=f"shared_bldg_{i:03d}",
            geometry=OVERLAP_BUILDING,
            properties={"type": "residential"},
        )
        await repo.store_building(job_a, building)

    for i in range(2):
        weather = WeatherRecord(
            source="synthetic",
            source_id=f"shared_wx_{i:03d}",
            geometry=OVERLAP_POINT,
            properties={"temperature_c": 25.0},
            observation_time=datetime(2026, 2, 15, 12 + i, tzinfo=timezone.utc),
        )
        await repo.store_weather(job_a, weather)

    # Job B: store the same 3 buildings + 2 weather (should be reused)
    for i in range(3):
        building = BuildingRecord(
            source="synthetic",
            source_id=f"shared_bldg_{i:03d}",
            geometry=OVERLAP_BUILDING,
            properties={"type": "residential"},
        )
        await repo.store_building(job_b, building)

    for i in range(2):
        weather = WeatherRecord(
            source="synthetic",
            source_id=f"shared_wx_{i:03d}",
            geometry=OVERLAP_POINT,
            properties={"temperature_c": 25.0},
            observation_time=datetime(2026, 2, 15, 12 + i, tzinfo=timezone.utc),
        )
        await repo.store_weather(job_b, weather)

    # Verify summaries
    summary_a = await repo.get_job_context_summary(job_a)
    assert summary_a.buildings_ingested == 3
    assert summary_a.buildings_reused == 0
    assert summary_a.weather_ingested == 2
    assert summary_a.weather_reused == 0
    assert summary_a.total_ingested == 5
    assert summary_a.total == 5

    summary_b = await repo.get_job_context_summary(job_b)
    assert summary_b.buildings_ingested == 0
    assert summary_b.buildings_reused == 3
    assert summary_b.weather_ingested == 0
    assert summary_b.weather_reused == 2
    assert summary_b.total_reused == 5
    assert summary_b.total == 5


@pytest.mark.asyncio
async def test_non_overlapping_jobs_no_sharing(repo, job_ids, remote_job_id):
    """
    A remote job that stores different context data does not share with
    the Houston jobs.
    """
    from core.context.models import BuildingRecord

    job_a, _ = job_ids

    # Job A ingests a Houston building
    building_houston = BuildingRecord(
        source="synthetic",
        source_id="houston_only_bldg_001",
        geometry=A_ONLY_BUILDING,
        properties={"type": "commercial"},
    )
    result_houston = await repo.store_building(job_a, building_houston)
    assert result_houston.usage_type == "ingested"

    # Remote job ingests a DIFFERENT building (different source_id)
    building_remote = BuildingRecord(
        source="synthetic",
        source_id="remote_bldg_001",
        geometry=REMOTE_POLYGON,
        properties={"type": "warehouse"},
    )
    result_remote = await repo.store_building(remote_job_id, building_remote)
    assert result_remote.usage_type == "ingested"

    # The context IDs should be different (no sharing)
    assert result_houston.context_id != result_remote.context_id

    # Each job should show 1 ingested, 0 reused
    summary_houston = await repo.get_job_context_summary(job_a)
    assert summary_houston.buildings_ingested == 1
    assert summary_houston.buildings_reused == 0

    summary_remote = await repo.get_job_context_summary(remote_job_id)
    assert summary_remote.buildings_ingested == 1
    assert summary_remote.buildings_reused == 0


@pytest.mark.asyncio
async def test_junction_table_tracks_provenance(repo, job_ids, pool):
    """
    The junction table correctly tracks which jobs use which context rows.
    Both jobs have links to the same context_id but with different usage_types.
    """
    from core.context.models import BuildingRecord

    job_a, job_b = job_ids

    # Both jobs store the same building
    building = BuildingRecord(
        source="synthetic",
        source_id="provenance_bldg_001",
        geometry=OVERLAP_BUILDING,
        properties={"type": "hospital"},
    )

    result_a = await repo.store_building(job_a, building)
    result_b = await repo.store_building(job_b, building)
    context_id = result_a.context_id

    # Verify junction table entries directly
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT job_id, context_table, context_id, usage_type
            FROM job_context_usage
            WHERE context_id = $1 AND context_table = 'context_buildings'
            ORDER BY linked_at
            """,
            context_id,
        )

    assert len(rows) == 2

    # First entry: Job A ingested
    assert rows[0]["job_id"] == job_a
    assert rows[0]["usage_type"] == "ingested"

    # Second entry: Job B reused
    assert rows[1]["job_id"] == job_b
    assert rows[1]["usage_type"] == "reused"


@pytest.mark.asyncio
async def test_batch_insert_with_mixed_reuse(repo, job_ids):
    """
    Batch insert for Job B with a mix of new and existing records
    produces correct ingested/reused counts.
    """
    from core.context.models import BuildingRecord

    job_a, job_b = job_ids

    # Job A ingests 3 buildings
    buildings_a = [
        BuildingRecord(
            source="synthetic",
            source_id=f"batch_bldg_{i:03d}",
            geometry=OVERLAP_BUILDING,
            properties={"type": "residential"},
        )
        for i in range(3)
    ]
    results_a = await repo.store_batch(job_a, "context_buildings", buildings_a)
    assert all(r.usage_type == "ingested" for r in results_a)

    # Job B batch-inserts 5 buildings: 3 overlapping + 2 new
    buildings_b = [
        BuildingRecord(
            source="synthetic",
            source_id=f"batch_bldg_{i:03d}",
            geometry=OVERLAP_BUILDING,
            properties={"type": "residential"},
        )
        for i in range(3)
    ] + [
        BuildingRecord(
            source="synthetic",
            source_id=f"batch_bldg_new_{j:03d}",
            geometry=OVERLAP_BUILDING,
            properties={"type": "commercial"},
        )
        for j in range(2)
    ]
    results_b = await repo.store_batch(job_b, "context_buildings", buildings_b)

    reused_count = sum(1 for r in results_b if r.usage_type == "reused")
    ingested_count = sum(1 for r in results_b if r.usage_type == "ingested")

    assert reused_count == 3
    assert ingested_count == 2

    # Verify summary
    summary_b = await repo.get_job_context_summary(job_b)
    assert summary_b.buildings_reused == 3
    assert summary_b.buildings_ingested == 2
