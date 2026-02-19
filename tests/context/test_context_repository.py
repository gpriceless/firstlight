"""
Integration tests for the Context Data Lakehouse repository.

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
# Sample GeoJSON geometries for testing
# ---------------------------------------------------------------------------

# Polygon in Houston, TX area
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

# Polygon outside Houston (far away — should not intersect Houston bbox)
SAMPLE_POLYGON_OUTSIDE = {
    "type": "Polygon",
    "coordinates": [[
        [-80.0, 40.0],
        [-79.98, 40.0],
        [-79.98, 40.02],
        [-80.0, 40.02],
        [-80.0, 40.0],
    ]],
}

# Point inside Houston
SAMPLE_POINT = {
    "type": "Point",
    "coordinates": [-95.44, 29.73],
}

# Point outside Houston
SAMPLE_POINT_OUTSIDE = {
    "type": "Point",
    "coordinates": [-80.0, 40.0],
}

# MultiPolygon for datasets (scene footprint)
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

# AOI for creating a test job
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

# Houston bbox for spatial queries
HOUSTON_BBOX = (-95.5, 29.7, -95.4, 29.8)


# ---------------------------------------------------------------------------
# Fixtures
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

    # Ordered migration files
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
        # Ensure PostGIS extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")

        # Drop existing tables for a clean slate (reverse order of creation)
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
        # Drop functions before recreating
        await conn.execute(
            "DROP FUNCTION IF EXISTS update_updated_at_column CASCADE"
        )
        await conn.execute(
            "DROP FUNCTION IF EXISTS compute_aoi_area_km2 CASCADE"
        )
        await conn.execute(
            "DROP FUNCTION IF EXISTS notify_job_event CASCADE"
        )
        # Drop materialized views
        await conn.execute(
            "DROP MATERIALIZED VIEW IF EXISTS mv_job_summary CASCADE"
        )
        await conn.execute(
            "DROP MATERIALIZED VIEW IF EXISTS mv_event_counts CASCADE"
        )

        # Apply each migration
        for mig_file in migration_files:
            mig_path = migrations_dir / mig_file
            if mig_path.exists():
                sql = mig_path.read_text()
                try:
                    await conn.execute(sql)
                except Exception:
                    # Some migrations may fail on re-run (e.g., partition
                    # migrations that rely on specific table states).
                    # Continue with remaining migrations.
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
    """
    Create a test job in the jobs table and return its UUID.

    Context tables have FK references to jobs(job_id), so tests need
    a valid job_id.
    """
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
    """Provide a configured ContextRepository."""
    from core.context.repository import ContextRepository

    r = ContextRepository(pool=pool)
    yield r


# ---------------------------------------------------------------------------
# Helper: sample record factories
# ---------------------------------------------------------------------------


def _make_dataset_record(
    source: str = "earth_search",
    source_id: str = None,
    geometry: dict = None,
    acquisition_date: datetime = None,
):
    from core.context.models import DatasetRecord

    return DatasetRecord(
        source=source,
        source_id=source_id or f"S2A_{uuid.uuid4().hex[:16]}",
        geometry=geometry or SAMPLE_MULTIPOLYGON,
        properties={"platform": "sentinel-2a"},
        acquisition_date=acquisition_date or datetime(2024, 1, 15, tzinfo=timezone.utc),
        cloud_cover=12.5,
        resolution_m=10.0,
        bands=["B02", "B03", "B04", "B08"],
        file_path="/data/scenes/test.tif",
    )


def _make_building_record(
    source: str = "osm",
    source_id: str = None,
    geometry: dict = None,
):
    from core.context.models import BuildingRecord

    return BuildingRecord(
        source=source,
        source_id=source_id or f"way_{uuid.uuid4().hex[:12]}",
        geometry=geometry or SAMPLE_POLYGON,
        properties={"building": "residential", "height": "10m"},
    )


def _make_infrastructure_record(
    source: str = "osm",
    source_id: str = None,
    geometry: dict = None,
    infra_type: str = "hospital",
):
    from core.context.models import InfrastructureRecord

    return InfrastructureRecord(
        source=source,
        source_id=source_id or f"node_{uuid.uuid4().hex[:12]}",
        geometry=geometry or SAMPLE_POINT,
        properties={"type": infra_type, "name": "Test Hospital"},
    )


def _make_weather_record(
    source: str = "noaa",
    source_id: str = None,
    geometry: dict = None,
    observation_time: datetime = None,
):
    from core.context.models import WeatherRecord

    return WeatherRecord(
        source=source,
        source_id=source_id or f"station_{uuid.uuid4().hex[:10]}",
        geometry=geometry or SAMPLE_POINT,
        properties={"temperature_c": 28.5, "precipitation_mm": 5.2},
        observation_time=observation_time or datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests: Store dataset
# ---------------------------------------------------------------------------


class TestStoreDataset:
    """Tests for store_dataset insert and dedup."""

    @pytest.mark.asyncio
    async def test_store_dataset_new_returns_ingested(self, repo, test_job_id):
        """Storing a new dataset returns usage_type='ingested'."""
        record = _make_dataset_record()
        result = await repo.store_dataset(test_job_id, record)

        assert result.context_id is not None
        assert result.usage_type == "ingested"

    @pytest.mark.asyncio
    async def test_store_dataset_duplicate_returns_reused(
        self, repo, test_job_id, second_job_id
    ):
        """Storing the same source+source_id returns usage_type='reused'."""
        record = _make_dataset_record(source_id="S2A_DUPLICATE_001")

        result1 = await repo.store_dataset(test_job_id, record)
        assert result1.usage_type == "ingested"

        result2 = await repo.store_dataset(second_job_id, record)
        assert result2.usage_type == "reused"
        assert result2.context_id == result1.context_id

    @pytest.mark.asyncio
    async def test_store_dataset_writes_to_junction_table(
        self, repo, test_job_id, pool
    ):
        """store_dataset creates a junction table entry."""
        record = _make_dataset_record()
        result = await repo.store_dataset(test_job_id, record)

        row = await pool.fetchrow(
            """
            SELECT * FROM job_context_usage
            WHERE job_id = $1 AND context_table = 'datasets'
              AND context_id = $2
            """,
            test_job_id,
            result.context_id,
        )
        assert row is not None
        assert row["usage_type"] == "ingested"


# ---------------------------------------------------------------------------
# Tests: Store building
# ---------------------------------------------------------------------------


class TestStoreBuilding:
    """Tests for store_building with polygon geometry."""

    @pytest.mark.asyncio
    async def test_store_building_with_polygon(self, repo, test_job_id):
        """Storing a building with a valid polygon succeeds."""
        record = _make_building_record()
        result = await repo.store_building(test_job_id, record)

        assert result.context_id is not None
        assert result.usage_type == "ingested"

    @pytest.mark.asyncio
    async def test_store_building_duplicate_reused(
        self, repo, test_job_id, second_job_id
    ):
        """Duplicate building returns reused."""
        record = _make_building_record(source_id="way_duplicate_001")

        r1 = await repo.store_building(test_job_id, record)
        r2 = await repo.store_building(second_job_id, record)

        assert r1.usage_type == "ingested"
        assert r2.usage_type == "reused"
        assert r2.context_id == r1.context_id


# ---------------------------------------------------------------------------
# Tests: Store weather
# ---------------------------------------------------------------------------


class TestStoreWeather:
    """Tests for store_weather with point geometry."""

    @pytest.mark.asyncio
    async def test_store_weather_with_point(self, repo, test_job_id):
        """Storing weather with a valid point succeeds."""
        record = _make_weather_record()
        result = await repo.store_weather(test_job_id, record)

        assert result.context_id is not None
        assert result.usage_type == "ingested"


# ---------------------------------------------------------------------------
# Tests: Store infrastructure
# ---------------------------------------------------------------------------


class TestStoreInfrastructure:
    """Tests for store_infrastructure with mixed geometry types."""

    @pytest.mark.asyncio
    async def test_store_infrastructure_with_point(self, repo, test_job_id):
        """Storing infrastructure with a point geometry succeeds."""
        record = _make_infrastructure_record(geometry=SAMPLE_POINT)
        result = await repo.store_infrastructure(test_job_id, record)

        assert result.context_id is not None
        assert result.usage_type == "ingested"

    @pytest.mark.asyncio
    async def test_store_infrastructure_with_polygon(self, repo, test_job_id):
        """Storing infrastructure with a polygon geometry succeeds."""
        record = _make_infrastructure_record(
            source_id="way_campus_001",
            geometry=SAMPLE_POLYGON,
            infra_type="hospital_campus",
        )
        result = await repo.store_infrastructure(test_job_id, record)

        assert result.context_id is not None
        assert result.usage_type == "ingested"


# ---------------------------------------------------------------------------
# Tests: Batch store
# ---------------------------------------------------------------------------


class TestStoreBatch:
    """Tests for store_batch with mixed new and existing records."""

    @pytest.mark.asyncio
    async def test_batch_store_mixed_records(
        self, repo, test_job_id, second_job_id
    ):
        """Batch with some new and some existing records returns correct types."""
        # First, store one building with job 1
        existing_record = _make_building_record(source_id="batch_existing_001")
        r1 = await repo.store_building(test_job_id, existing_record)
        assert r1.usage_type == "ingested"

        # Now batch-store 3 buildings with job 2: 1 existing + 2 new
        new_record_1 = _make_building_record(source_id="batch_new_001")
        new_record_2 = _make_building_record(source_id="batch_new_002")

        results = await repo.store_batch(
            second_job_id,
            "context_buildings",
            [existing_record, new_record_1, new_record_2],
        )

        assert len(results) == 3
        # First should be reused (already existed)
        assert results[0].usage_type == "reused"
        assert results[0].context_id == r1.context_id
        # Second and third should be ingested
        assert results[1].usage_type == "ingested"
        assert results[2].usage_type == "ingested"


# ---------------------------------------------------------------------------
# Tests: Query buildings by bbox
# ---------------------------------------------------------------------------


class TestQueryBuildings:
    """Tests for query_buildings spatial filtering."""

    @pytest.mark.asyncio
    async def test_query_buildings_by_bbox(self, repo, test_job_id):
        """Only buildings intersecting the bbox are returned."""
        # Insert one building inside Houston
        inside = _make_building_record(
            source_id="inside_houston_001",
            geometry=SAMPLE_POLYGON,
        )
        await repo.store_building(test_job_id, inside)

        # Insert one building outside Houston
        outside = _make_building_record(
            source_id="outside_houston_001",
            geometry=SAMPLE_POLYGON_OUTSIDE,
        )
        await repo.store_building(test_job_id, outside)

        # Query with Houston bbox
        records, total = await repo.query_buildings(bbox=HOUSTON_BBOX)

        assert total >= 1
        sources = [r.source_id for r in records]
        assert "inside_houston_001" in sources
        assert "outside_houston_001" not in sources

    @pytest.mark.asyncio
    async def test_query_buildings_returns_geojson(self, repo, test_job_id):
        """Query results include geometry as GeoJSON dict."""
        record = _make_building_record(source_id="geojson_test_001")
        await repo.store_building(test_job_id, record)

        records, _ = await repo.query_buildings(bbox=HOUSTON_BBOX)
        assert len(records) >= 1

        geom = records[0].geometry
        assert isinstance(geom, dict)
        assert "type" in geom
        assert "coordinates" in geom


# ---------------------------------------------------------------------------
# Tests: Query datasets by date range
# ---------------------------------------------------------------------------


class TestQueryDatasets:
    """Tests for query_datasets temporal and spatial filtering."""

    @pytest.mark.asyncio
    async def test_query_datasets_by_date_range(self, repo, test_job_id):
        """Only datasets within the date range are returned."""
        jan = _make_dataset_record(
            source_id="jan_scene",
            acquisition_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        mar = _make_dataset_record(
            source_id="mar_scene",
            acquisition_date=datetime(2024, 3, 15, tzinfo=timezone.utc),
        )
        await repo.store_dataset(test_job_id, jan)
        await repo.store_dataset(test_job_id, mar)

        # Query only January-February
        records, total = await repo.query_datasets(
            date_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            date_end=datetime(2024, 2, 28, tzinfo=timezone.utc),
        )

        source_ids = [r.source_id for r in records]
        assert "jan_scene" in source_ids
        assert "mar_scene" not in source_ids

    @pytest.mark.asyncio
    async def test_query_datasets_by_bbox(self, repo, test_job_id):
        """Only datasets intersecting the bbox are returned."""
        inside = _make_dataset_record(
            source_id="inside_ds",
            geometry=SAMPLE_MULTIPOLYGON,
        )
        outside = _make_dataset_record(
            source_id="outside_ds",
            geometry={
                "type": "MultiPolygon",
                "coordinates": [[[
                    [-80.0, 40.0],
                    [-79.98, 40.0],
                    [-79.98, 40.02],
                    [-80.0, 40.02],
                    [-80.0, 40.0],
                ]]],
            },
        )
        await repo.store_dataset(test_job_id, inside)
        await repo.store_dataset(test_job_id, outside)

        records, total = await repo.query_datasets(bbox=HOUSTON_BBOX)
        source_ids = [r.source_id for r in records]
        assert "inside_ds" in source_ids
        assert "outside_ds" not in source_ids


# ---------------------------------------------------------------------------
# Tests: Query weather
# ---------------------------------------------------------------------------


class TestQueryWeather:
    """Tests for query_weather spatial and temporal filtering."""

    @pytest.mark.asyncio
    async def test_query_weather_by_time_range(self, repo, test_job_id):
        """Only weather observations within time range are returned."""
        morning = _make_weather_record(
            source_id="morning_obs",
            observation_time=datetime(2024, 1, 15, 8, 0, tzinfo=timezone.utc),
        )
        evening = _make_weather_record(
            source_id="evening_obs",
            observation_time=datetime(2024, 1, 15, 20, 0, tzinfo=timezone.utc),
        )
        await repo.store_weather(test_job_id, morning)
        await repo.store_weather(test_job_id, evening)

        records, total = await repo.query_weather(
            time_start=datetime(2024, 1, 15, 6, 0, tzinfo=timezone.utc),
            time_end=datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc),
        )
        source_ids = [r.source_id for r in records]
        assert "morning_obs" in source_ids
        assert "evening_obs" not in source_ids


# ---------------------------------------------------------------------------
# Tests: Context summary
# ---------------------------------------------------------------------------


class TestContextSummary:
    """Tests for get_job_context_summary."""

    @pytest.mark.asyncio
    async def test_context_summary_counts(
        self, repo, test_job_id, second_job_id
    ):
        """Summary returns correct ingested/reused counts."""
        # Job 1: ingest 2 buildings and 1 dataset
        b1 = _make_building_record(source_id="summary_bldg_1")
        b2 = _make_building_record(source_id="summary_bldg_2")
        d1 = _make_dataset_record(source_id="summary_ds_1")
        await repo.store_building(test_job_id, b1)
        await repo.store_building(test_job_id, b2)
        await repo.store_dataset(test_job_id, d1)

        # Job 2: reuse 1 building, ingest 1 new building
        b3 = _make_building_record(source_id="summary_bldg_3")
        await repo.store_building(second_job_id, b1)  # reuse
        await repo.store_building(second_job_id, b3)  # new

        # Check summary for job 1
        summary1 = await repo.get_job_context_summary(test_job_id)
        assert summary1.buildings_ingested == 2
        assert summary1.datasets_ingested == 1
        assert summary1.total_ingested == 3
        assert summary1.total == 3

        # Check summary for job 2
        summary2 = await repo.get_job_context_summary(second_job_id)
        assert summary2.buildings_ingested == 1
        assert summary2.buildings_reused == 1
        assert summary2.total == 2


# ---------------------------------------------------------------------------
# Tests: Lakehouse stats
# ---------------------------------------------------------------------------


class TestLakehouseStats:
    """Tests for get_lakehouse_stats."""

    @pytest.mark.asyncio
    async def test_lakehouse_stats_structure(self, repo, test_job_id):
        """Stats dict has the expected structure."""
        # Insert some data
        await repo.store_building(
            test_job_id,
            _make_building_record(source_id="stats_bldg"),
        )
        await repo.store_weather(
            test_job_id,
            _make_weather_record(source_id="stats_weather"),
        )

        stats = await repo.get_lakehouse_stats()

        assert "tables" in stats
        assert "total_rows" in stats
        assert "spatial_extent" in stats
        assert "usage_stats" in stats

        assert "buildings" in stats["tables"]
        assert "weather" in stats["tables"]
        assert stats["tables"]["buildings"]["row_count"] >= 1
        assert stats["tables"]["weather"]["row_count"] >= 1
        assert stats["total_rows"] >= 2

    @pytest.mark.asyncio
    async def test_lakehouse_stats_sources(self, repo, test_job_id):
        """Stats include distinct source names."""
        await repo.store_building(
            test_job_id,
            _make_building_record(source="osm", source_id="src_test_1"),
        )
        await repo.store_building(
            test_job_id,
            _make_building_record(source="overture", source_id="src_test_2"),
        )

        stats = await repo.get_lakehouse_stats()
        sources = stats["tables"]["buildings"]["sources"]
        assert "osm" in sources
        assert "overture" in sources
