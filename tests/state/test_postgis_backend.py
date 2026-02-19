"""
Tests for the PostGIS state backend.

These tests require a live PostGIS instance. They are marked with
@pytest.mark.integration so they can be skipped in CI without PostGIS.

When running locally, ensure PostGIS is available:
    docker run -d --name test-postgis \
        -e POSTGRES_DB=firstlight_test \
        -e POSTGRES_USER=postgres \
        -e POSTGRES_PASSWORD=testpass \
        -p 5433:5432 \
        postgis/postgis:15-3.4-alpine
"""

import json
import os
import uuid

import pytest
import pytest_asyncio

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)
from agents.orchestrator.state_model import (
    JobPhase,
    JobStatus,
    is_valid_phase_status,
)

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


# Sample GeoJSON for a ~100 km2 area in Houston, TX
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

SAMPLE_MULTIPOLYGON_AOI = {
    "type": "MultiPolygon",
    "coordinates": [[[
        [-95.5, 29.7],
        [-95.4, 29.7],
        [-95.4, 29.8],
        [-95.5, 29.8],
        [-95.5, 29.7],
    ]]],
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


async def _create_test_pool():
    """Create a test connection pool."""
    dsn = f"postgresql://{POSTGIS_USER}:{POSTGIS_PASSWORD}@{POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DB}"
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5)
        return pool
    except (ConnectionRefusedError, asyncpg.InvalidCatalogNameError, OSError):
        return None


async def _setup_schema(pool):
    """Apply the schema migration to the test database."""
    import pathlib

    migration_path = (
        pathlib.Path(__file__).parents[2]
        / "db"
        / "migrations"
        / "001_control_plane_schema.sql"
    )
    if not migration_path.exists():
        pytest.skip(f"Migration file not found: {migration_path}")

    sql = migration_path.read_text()

    async with pool.acquire() as conn:
        # Ensure PostGIS extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        # Drop existing tables for a clean slate
        await conn.execute("DROP TABLE IF EXISTS escalations CASCADE")
        await conn.execute("DROP TABLE IF EXISTS job_checkpoints CASCADE")
        await conn.execute("DROP TABLE IF EXISTS job_events CASCADE")
        await conn.execute("DROP TABLE IF EXISTS jobs CASCADE")
        # Drop functions before recreating
        await conn.execute("DROP FUNCTION IF EXISTS update_updated_at_column CASCADE")
        await conn.execute("DROP FUNCTION IF EXISTS compute_aoi_area_km2 CASCADE")
        # Apply migration
        await conn.execute(sql)


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
async def backend(pool):
    """Provide a configured PostGISStateBackend."""
    from agents.orchestrator.backends.postgis_backend import PostGISStateBackend

    b = PostGISStateBackend(pool=pool)
    yield b


@pytest_asyncio.fixture
async def sample_job(backend):
    """Create a sample job and return its state."""
    return await backend.create_job(
        customer_id="tenant-1",
        event_type="flood",
        aoi_geojson=SAMPLE_AOI,
        phase=JobPhase.QUEUED.value,
        status=JobStatus.PENDING.value,
        parameters={"threshold": 0.5},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostGISBackendInsert:
    """Tests for job creation and AOI handling."""

    @pytest.mark.asyncio
    async def test_create_job_with_polygon_promotes_to_multi(self, backend):
        """A Polygon AOI should be promoted to MultiPolygon via ST_Multi."""
        job = await backend.create_job(
            customer_id="tenant-1",
            event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase=JobPhase.QUEUED.value,
            status=JobStatus.PENDING.value,
        )
        assert job.aoi is not None
        assert job.aoi["type"] == "MultiPolygon"

    @pytest.mark.asyncio
    async def test_create_job_with_multipolygon(self, backend):
        """A MultiPolygon AOI should be stored as-is."""
        job = await backend.create_job(
            customer_id="tenant-1",
            event_type="wildfire",
            aoi_geojson=SAMPLE_MULTIPOLYGON_AOI,
            phase=JobPhase.QUEUED.value,
            status=JobStatus.PENDING.value,
        )
        assert job.aoi is not None
        assert job.aoi["type"] == "MultiPolygon"

    @pytest.mark.asyncio
    async def test_create_job_fields(self, backend):
        """All fields should be correctly stored and retrieved."""
        job = await backend.create_job(
            customer_id="tenant-42",
            event_type="storm",
            aoi_geojson=SAMPLE_AOI,
            phase=JobPhase.QUEUED.value,
            status=JobStatus.PENDING.value,
            parameters={"wind_speed_threshold": 75},
            orchestrator_id="orch-abc",
        )
        assert job.customer_id == "tenant-42"
        assert job.event_type == "storm"
        assert job.phase == "QUEUED"
        assert job.status == "PENDING"
        assert job.parameters == {"wind_speed_threshold": 75}
        assert job.orchestrator_id == "orch-abc"
        assert job.created_at is not None
        assert job.updated_at is not None


class TestPostGISBackendTransition:
    """Tests for TOCTOU-safe state transitions."""

    @pytest.mark.asyncio
    async def test_transition_success(self, backend, sample_job):
        """Transition should succeed when expected state matches."""
        result = await backend.transition(
            sample_job.job_id,
            expected_phase="QUEUED",
            expected_status="PENDING",
            new_phase="QUEUED",
            new_status="VALIDATING",
            reason="Starting validation",
            actor="orchestrator-1",
        )
        assert result.phase == "QUEUED"
        assert result.status == "VALIDATING"

    @pytest.mark.asyncio
    async def test_transition_conflict(self, backend, sample_job):
        """Transition should raise StateConflictError on mismatch."""
        with pytest.raises(StateConflictError) as exc_info:
            await backend.transition(
                sample_job.job_id,
                expected_phase="DISCOVERING",
                expected_status="DISCOVERING",
                new_phase="INGESTING",
                new_status="INGESTING",
            )
        assert exc_info.value.actual_phase == "QUEUED"
        assert exc_info.value.actual_status == "PENDING"

    @pytest.mark.asyncio
    async def test_transition_records_event(self, backend, sample_job, pool):
        """A successful transition should insert a STATE_TRANSITION event."""
        await backend.transition(
            sample_job.job_id,
            expected_phase="QUEUED",
            expected_status="PENDING",
            new_phase="DISCOVERING",
            new_status="DISCOVERING",
            reason="Starting discovery",
            actor="test-actor",
        )
        row = await pool.fetchrow(
            "SELECT * FROM job_events WHERE job_id = $1 ORDER BY event_seq DESC LIMIT 1",
            uuid.UUID(sample_job.job_id),
        )
        assert row is not None
        assert row["event_type"] == "STATE_TRANSITION"
        assert row["phase"] == "DISCOVERING"
        assert row["status"] == "DISCOVERING"
        assert row["actor"] == "test-actor"
        assert row["reasoning"] == "Starting discovery"

    @pytest.mark.asyncio
    async def test_transition_nonexistent_job(self, backend):
        """Transition on a nonexistent job should raise KeyError."""
        fake_id = str(uuid.uuid4())
        with pytest.raises(KeyError):
            await backend.transition(
                fake_id,
                expected_phase="QUEUED",
                expected_status="PENDING",
                new_phase="DISCOVERING",
                new_status="DISCOVERING",
            )


class TestPostGISBackendCheckpoint:
    """Tests for checkpoint round-trip."""

    @pytest.mark.asyncio
    async def test_checkpoint_roundtrip(self, backend, sample_job):
        """Checkpoint should be stored and retrievable."""
        payload = {
            "stage": "QUEUED",
            "progress": 0.0,
            "data": {"key": "value"},
        }
        await backend.checkpoint(sample_job.job_id, payload)

        restored = await backend.get_latest_checkpoint(sample_job.job_id)
        assert restored is not None
        assert restored["stage"] == "QUEUED"
        assert restored["data"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_checkpoint_latest(self, backend, sample_job):
        """get_latest_checkpoint should return the most recent checkpoint."""
        await backend.checkpoint(sample_job.job_id, {"version": 1})
        await backend.checkpoint(sample_job.job_id, {"version": 2})

        restored = await backend.get_latest_checkpoint(sample_job.job_id)
        assert restored is not None
        assert restored["version"] == 2

    @pytest.mark.asyncio
    async def test_checkpoint_nonexistent_job(self, backend):
        """Checkpoint on a nonexistent job should raise KeyError."""
        with pytest.raises(KeyError):
            await backend.checkpoint(str(uuid.uuid4()), {"data": "test"})


class TestPostGISBackendListJobs:
    """Tests for job listing with filters."""

    @pytest.mark.asyncio
    async def test_list_all(self, backend):
        """list_jobs with no filters should return all jobs."""
        await backend.create_job(
            customer_id="t1", event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase="QUEUED", status="PENDING",
        )
        await backend.create_job(
            customer_id="t2", event_type="wildfire",
            aoi_geojson=SAMPLE_AOI,
            phase="DISCOVERING", status="DISCOVERING",
        )
        jobs = await backend.list_jobs()
        assert len(jobs) >= 2

    @pytest.mark.asyncio
    async def test_list_filter_by_phase(self, backend):
        """list_jobs should filter by phase."""
        await backend.create_job(
            customer_id="t1", event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase="QUEUED", status="PENDING",
        )
        await backend.create_job(
            customer_id="t1", event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase="DISCOVERING", status="DISCOVERING",
        )
        jobs = await backend.list_jobs(phase="DISCOVERING")
        assert all(j.phase == "DISCOVERING" for j in jobs)

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self, backend):
        """list_jobs should filter by status."""
        await backend.create_job(
            customer_id="t1", event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase="QUEUED", status="PENDING",
        )
        jobs = await backend.list_jobs(status="PENDING")
        assert all(j.status == "PENDING" for j in jobs)

    @pytest.mark.asyncio
    async def test_list_filter_by_customer(self, backend):
        """list_jobs should filter by customer_id."""
        await backend.create_job(
            customer_id="isolated-tenant", event_type="flood",
            aoi_geojson=SAMPLE_AOI,
            phase="QUEUED", status="PENDING",
        )
        jobs = await backend.list_jobs(customer_id="isolated-tenant")
        assert all(j.customer_id == "isolated-tenant" for j in jobs)

    @pytest.mark.asyncio
    async def test_list_pagination(self, backend):
        """list_jobs should support limit and offset."""
        for i in range(5):
            await backend.create_job(
                customer_id="paginate-test", event_type="flood",
                aoi_geojson=SAMPLE_AOI,
                phase="QUEUED", status="PENDING",
            )
        page1 = await backend.list_jobs(customer_id="paginate-test", limit=2, offset=0)
        page2 = await backend.list_jobs(customer_id="paginate-test", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # Pages should have different jobs
        ids1 = {j.job_id for j in page1}
        ids2 = {j.job_id for j in page2}
        assert ids1.isdisjoint(ids2)


class TestPostGISBackendGeoJSON:
    """Tests for GeoJSON round-trip fidelity."""

    @pytest.mark.asyncio
    async def test_geojson_coordinates_preserved(self, backend):
        """Coordinates should be preserved to at least 5 decimal places."""
        precise_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-95.12345, 29.67890],
                [-95.01234, 29.67890],
                [-95.01234, 29.78901],
                [-95.12345, 29.78901],
                [-95.12345, 29.67890],
            ]],
        }
        job = await backend.create_job(
            customer_id="t1", event_type="flood",
            aoi_geojson=precise_aoi,
            phase="QUEUED", status="PENDING",
        )
        assert job.aoi is not None
        # After ST_Multi promotion, it's a MultiPolygon
        coords = job.aoi["coordinates"][0][0]  # First polygon, exterior ring
        # Check first coordinate preserved to 5 decimal places
        assert round(coords[0][0], 5) == -95.12345
        assert round(coords[0][1], 5) == 29.67890


class TestPostGISBackendGetAndSet:
    """Tests for get_state and set_state."""

    @pytest.mark.asyncio
    async def test_get_state(self, backend, sample_job):
        """get_state should return the job."""
        job = await backend.get_state(sample_job.job_id)
        assert job is not None
        assert job.job_id == sample_job.job_id

    @pytest.mark.asyncio
    async def test_get_state_nonexistent(self, backend):
        """get_state for a nonexistent job should return None."""
        result = await backend.get_state(str(uuid.uuid4()))
        assert result is None

    @pytest.mark.asyncio
    async def test_set_state(self, backend, sample_job):
        """set_state should update phase and status."""
        updated = await backend.set_state(
            sample_job.job_id,
            phase="DISCOVERING",
            status="DISCOVERING",
        )
        assert updated.phase == "DISCOVERING"
        assert updated.status == "DISCOVERING"

    @pytest.mark.asyncio
    async def test_set_state_invalid_pair(self, backend, sample_job):
        """set_state with invalid pair should raise ValueError."""
        with pytest.raises(ValueError):
            await backend.set_state(
                sample_job.job_id,
                phase="QUEUED",
                status="DISCOVERING",  # Wrong phase for this status
            )

    @pytest.mark.asyncio
    async def test_set_state_nonexistent(self, backend):
        """set_state on nonexistent job should raise KeyError."""
        with pytest.raises(KeyError):
            await backend.set_state(
                str(uuid.uuid4()),
                phase="QUEUED",
                status="PENDING",
            )


class TestPostGISBackendRecordEvent:
    """Tests for event recording (extension method)."""

    @pytest.mark.asyncio
    async def test_record_event(self, backend, sample_job, pool):
        """record_event should insert into job_events."""
        seq = await backend.record_event(
            job_id=sample_job.job_id,
            customer_id="tenant-1",
            event_type="REASONING",
            phase="QUEUED",
            status="PENDING",
            actor="llm-agent-1",
            reasoning="Analyzing flood patterns",
            payload={"confidence": 0.85},
        )
        assert isinstance(seq, int)
        assert seq > 0

        row = await pool.fetchrow(
            "SELECT * FROM job_events WHERE event_seq = $1", seq
        )
        assert row["event_type"] == "REASONING"
        assert row["reasoning"] == "Analyzing flood patterns"
        assert row["actor"] == "llm-agent-1"
