"""
Context Data Lakehouse repository.

Provides asyncpg-based storage and query methods for context data tables
(datasets, buildings, infrastructure, weather) with insert-or-link
deduplication via ON CONFLICT DO NOTHING and a junction table for
provenance tracking.

Follows the same asyncpg pool management pattern as PostGISStateBackend
(agents/orchestrator/backends/postgis_backend.py).
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

from core.context.models import (
    BuildingRecord,
    ContextResult,
    ContextSummary,
    DatasetRecord,
    InfrastructureRecord,
    WeatherRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Table metadata for the _insert_or_link helper
# =============================================================================

# Mapping from table name to (columns, geometry_sql_fragment)
# The geometry SQL fragment handles the ST_SetSRID(ST_GeomFromGeoJSON(...), 4326)
# conversion, with ST_Multi wrapping for datasets only.
_TABLE_META: Dict[str, Dict[str, Any]] = {
    "datasets": {
        "columns": [
            "source", "source_id", "geometry", "properties",
            "acquisition_date", "cloud_cover", "resolution_m",
            "bands", "file_path", "ingested_by_job_id",
        ],
        "geom_sql": "ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON({param}), 4326))",
    },
    "context_buildings": {
        "columns": [
            "source", "source_id", "geometry", "properties",
            "ingested_by_job_id",
        ],
        "geom_sql": "ST_SetSRID(ST_GeomFromGeoJSON({param}), 4326)",
    },
    "context_infrastructure": {
        "columns": [
            "source", "source_id", "geometry", "properties",
            "ingested_by_job_id",
        ],
        "geom_sql": "ST_SetSRID(ST_GeomFromGeoJSON({param}), 4326)",
    },
    "context_weather": {
        "columns": [
            "source", "source_id", "geometry", "properties",
            "observation_time", "ingested_by_job_id",
        ],
        "geom_sql": "ST_SetSRID(ST_GeomFromGeoJSON({param}), 4326)",
    },
}


class ContextRepository:
    """
    PostGIS-backed repository for Context Data Lakehouse tables.

    Handles insert-or-link deduplication and spatial queries for all
    context types: datasets, buildings, infrastructure, weather.

    Args:
        host: PostgreSQL host.
        port: PostgreSQL port.
        database: Database name.
        user: Database user.
        password: Database password.
        pool: An existing asyncpg pool. If provided, host/port/etc. are ignored.
        min_pool_size: Minimum connections in the pool.
        max_pool_size: Maximum connections in the pool.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "firstlight",
        user: str = "postgres",
        password: str = "",
        pool: Optional["asyncpg.Pool"] = None,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ):
        if asyncpg is None:
            raise ImportError(
                "asyncpg is required for ContextRepository. "
                "Install it with: pip install asyncpg"
            )
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool: Optional["asyncpg.Pool"] = pool
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._owns_pool = pool is None

    async def connect(self) -> None:
        """Create the connection pool if not already provided."""
        if self._pool is None:
            dsn = (
                f"postgresql://{self._user}:{self._password}"
                f"@{self._host}:{self._port}/{self._database}"
            )
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
            logger.info("ContextRepository connection pool created")

    async def close(self) -> None:
        """Close the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None
            logger.info("ContextRepository connection pool closed")

    def _ensure_pool(self) -> "asyncpg.Pool":
        """Return the pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError(
                "ContextRepository is not connected. Call connect() first."
            )
        return self._pool

    # =========================================================================
    # Internal helper: insert-or-link
    # =========================================================================

    async def _insert_or_link(
        self,
        conn: "asyncpg.Connection",
        job_id: UUID,
        table: str,
        values: Dict[str, Any],
    ) -> ContextResult:
        """
        Insert a context row or link to an existing one.

        Uses INSERT ... ON CONFLICT (source, source_id) DO NOTHING RETURNING id.
        If no row is returned (conflict), SELECTs the existing row id.
        Then inserts into job_context_usage with the appropriate usage_type.

        Args:
            conn: An asyncpg connection (should be inside a transaction).
            job_id: The job that is storing this context data.
            table: The context table name (e.g., 'datasets').
            values: Column name -> value mapping. Must include 'source' and
                    'source_id'. Geometry value must already be a JSON string.

        Returns:
            ContextResult with context_id and usage_type.
        """
        meta = _TABLE_META[table]
        columns = meta["columns"]
        geom_sql = meta["geom_sql"]

        # Build parameterized INSERT
        col_list = []
        val_placeholders = []
        params: list = []
        idx = 1

        for col in columns:
            col_list.append(col)
            if col == "geometry":
                val_placeholders.append(geom_sql.format(param=f"${idx}"))
            elif col == "properties":
                val_placeholders.append(f"${idx}::jsonb")
            else:
                val_placeholders.append(f"${idx}")
            params.append(values[col])
            idx += 1

        insert_sql = (
            f"INSERT INTO {table} ({', '.join(col_list)}) "
            f"VALUES ({', '.join(val_placeholders)}) "
            f"ON CONFLICT (source, source_id) DO NOTHING "
            f"RETURNING id"
        )

        row = await conn.fetchrow(insert_sql, *params)

        if row is not None:
            # New row inserted
            context_id = row["id"]
            usage_type = "ingested"
        else:
            # Conflict: row already exists, fetch its id
            existing = await conn.fetchrow(
                f"SELECT id FROM {table} WHERE source = $1 AND source_id = $2",
                values["source"],
                values["source_id"],
            )
            context_id = existing["id"]
            usage_type = "reused"

        # Link in junction table
        await conn.execute(
            """
            INSERT INTO job_context_usage (job_id, context_table, context_id, usage_type)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (job_id, context_table, context_id) DO NOTHING
            """,
            job_id,
            table,
            context_id,
            usage_type,
        )

        return ContextResult(context_id=context_id, usage_type=usage_type)

    # =========================================================================
    # Store methods
    # =========================================================================

    def _dataset_values(
        self, job_id: UUID, record: DatasetRecord
    ) -> Dict[str, Any]:
        """Build the values dict for a DatasetRecord insert."""
        return {
            "source": record.source,
            "source_id": record.source_id,
            "geometry": json.dumps(record.geometry),
            "properties": json.dumps(record.properties),
            "acquisition_date": record.acquisition_date,
            "cloud_cover": record.cloud_cover,
            "resolution_m": record.resolution_m,
            "bands": record.bands,
            "file_path": record.file_path,
            "ingested_by_job_id": job_id,
        }

    def _building_values(
        self, job_id: UUID, record: BuildingRecord
    ) -> Dict[str, Any]:
        """Build the values dict for a BuildingRecord insert."""
        return {
            "source": record.source,
            "source_id": record.source_id,
            "geometry": json.dumps(record.geometry),
            "properties": json.dumps(record.properties),
            "ingested_by_job_id": job_id,
        }

    def _infrastructure_values(
        self, job_id: UUID, record: InfrastructureRecord
    ) -> Dict[str, Any]:
        """Build the values dict for an InfrastructureRecord insert."""
        return {
            "source": record.source,
            "source_id": record.source_id,
            "geometry": json.dumps(record.geometry),
            "properties": json.dumps(record.properties),
            "ingested_by_job_id": job_id,
        }

    def _weather_values(
        self, job_id: UUID, record: WeatherRecord
    ) -> Dict[str, Any]:
        """Build the values dict for a WeatherRecord insert."""
        return {
            "source": record.source,
            "source_id": record.source_id,
            "geometry": json.dumps(record.geometry),
            "properties": json.dumps(record.properties),
            "observation_time": record.observation_time,
            "ingested_by_job_id": job_id,
        }

    async def store_dataset(
        self, job_id: UUID, record: DatasetRecord
    ) -> ContextResult:
        """
        Store a dataset record with deduplication.

        Inserts into the datasets table. If a row with the same
        (source, source_id) already exists, links to the existing row
        with usage_type='reused'.

        Args:
            job_id: The job storing this dataset.
            record: The dataset record to store.

        Returns:
            ContextResult with context_id and usage_type.
        """
        pool = self._ensure_pool()
        values = self._dataset_values(job_id, record)

        async with pool.acquire() as conn:
            async with conn.transaction():
                return await self._insert_or_link(conn, job_id, "datasets", values)

    async def store_building(
        self, job_id: UUID, record: BuildingRecord
    ) -> ContextResult:
        """
        Store a building footprint record with deduplication.

        Args:
            job_id: The job storing this building.
            record: The building record to store.

        Returns:
            ContextResult with context_id and usage_type.
        """
        pool = self._ensure_pool()
        values = self._building_values(job_id, record)

        async with pool.acquire() as conn:
            async with conn.transaction():
                return await self._insert_or_link(
                    conn, job_id, "context_buildings", values
                )

    async def store_infrastructure(
        self, job_id: UUID, record: InfrastructureRecord
    ) -> ContextResult:
        """
        Store an infrastructure facility record with deduplication.

        Args:
            job_id: The job storing this infrastructure facility.
            record: The infrastructure record to store.

        Returns:
            ContextResult with context_id and usage_type.
        """
        pool = self._ensure_pool()
        values = self._infrastructure_values(job_id, record)

        async with pool.acquire() as conn:
            async with conn.transaction():
                return await self._insert_or_link(
                    conn, job_id, "context_infrastructure", values
                )

    async def store_weather(
        self, job_id: UUID, record: WeatherRecord
    ) -> ContextResult:
        """
        Store a weather observation record with deduplication.

        Args:
            job_id: The job storing this weather observation.
            record: The weather record to store.

        Returns:
            ContextResult with context_id and usage_type.
        """
        pool = self._ensure_pool()
        values = self._weather_values(job_id, record)

        async with pool.acquire() as conn:
            async with conn.transaction():
                return await self._insert_or_link(
                    conn, job_id, "context_weather", values
                )

    async def store_batch(
        self,
        job_id: UUID,
        table_name: str,
        records: list,
    ) -> List[ContextResult]:
        """
        Batch-insert context records with deduplication.

        Loops individual inserts within a single transaction because asyncpg
        executemany does NOT support RETURNING clauses.

        Args:
            job_id: The job storing these records.
            table_name: The context table ('datasets', 'context_buildings',
                        'context_infrastructure', 'context_weather').
            records: List of record objects (DatasetRecord, BuildingRecord, etc.).

        Returns:
            List of ContextResult, one per record.
        """
        pool = self._ensure_pool()

        # Build values function based on table
        values_fn = {
            "datasets": self._dataset_values,
            "context_buildings": self._building_values,
            "context_infrastructure": self._infrastructure_values,
            "context_weather": self._weather_values,
        }[table_name]

        results: List[ContextResult] = []

        async with pool.acquire() as conn:
            async with conn.transaction():
                for record in records:
                    values = values_fn(job_id, record)
                    result = await self._insert_or_link(
                        conn, job_id, table_name, values
                    )
                    results.append(result)

        return results

    # =========================================================================
    # Query methods
    # =========================================================================

    async def query_datasets(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        date_start: Optional[datetime] = None,
        date_end: Optional[datetime] = None,
        source: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[DatasetRecord], int]:
        """
        Query accumulated datasets with spatial and temporal filters.

        Args:
            bbox: (west, south, east, north) bounding box in WGS84.
            date_start: Minimum acquisition date (inclusive).
            date_end: Maximum acquisition date (inclusive).
            source: Filter by source catalog.
            limit: Max records to return.
            offset: Number of records to skip.

        Returns:
            Tuple of (list of DatasetRecord, total_count).
        """
        pool = self._ensure_pool()

        conditions: List[str] = []
        params: list = []
        idx = 1

        if bbox is not None:
            west, south, east, north = bbox
            conditions.append(
                f"ST_Intersects(geometry, ST_MakeEnvelope(${idx}, ${idx+1}, ${idx+2}, ${idx+3}, 4326))"
            )
            params.extend([west, south, east, north])
            idx += 4

        if date_start is not None:
            conditions.append(f"acquisition_date >= ${idx}")
            params.append(date_start)
            idx += 1

        if date_end is not None:
            conditions.append(f"acquisition_date <= ${idx}")
            params.append(date_end)
            idx += 1

        if source is not None:
            conditions.append(f"source = ${idx}")
            params.append(source)
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        sql = f"""
            SELECT
                source, source_id,
                ST_AsGeoJSON(geometry)::text AS geometry_geojson,
                properties,
                acquisition_date, cloud_cover, resolution_m,
                bands, file_path,
                COUNT(*) OVER() AS total_count
            FROM datasets
            WHERE {where_clause}
            ORDER BY acquisition_date DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        rows = await pool.fetch(sql, *params)

        total = rows[0]["total_count"] if rows else 0

        records = []
        for row in rows:
            props = row["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            records.append(
                DatasetRecord(
                    source=row["source"],
                    source_id=row["source_id"],
                    geometry=json.loads(row["geometry_geojson"]),
                    properties=props if props else {},
                    acquisition_date=row["acquisition_date"],
                    cloud_cover=float(row["cloud_cover"]) if row["cloud_cover"] is not None else None,
                    resolution_m=float(row["resolution_m"]) if row["resolution_m"] is not None else None,
                    bands=list(row["bands"]) if row["bands"] is not None else None,
                    file_path=row["file_path"],
                )
            )

        return records, total

    async def query_buildings(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[BuildingRecord], int]:
        """
        Query building footprints with spatial filter.

        Args:
            bbox: (west, south, east, north) bounding box in WGS84.
            limit: Max records to return.
            offset: Number of records to skip.

        Returns:
            Tuple of (list of BuildingRecord, total_count).
        """
        pool = self._ensure_pool()

        conditions: List[str] = []
        params: list = []
        idx = 1

        if bbox is not None:
            west, south, east, north = bbox
            conditions.append(
                f"ST_Intersects(geometry, ST_MakeEnvelope(${idx}, ${idx+1}, ${idx+2}, ${idx+3}, 4326))"
            )
            params.extend([west, south, east, north])
            idx += 4

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        sql = f"""
            SELECT
                source, source_id,
                ST_AsGeoJSON(geometry)::text AS geometry_geojson,
                properties,
                COUNT(*) OVER() AS total_count
            FROM context_buildings
            WHERE {where_clause}
            ORDER BY ingested_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        rows = await pool.fetch(sql, *params)

        total = rows[0]["total_count"] if rows else 0

        records = []
        for row in rows:
            props = row["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            records.append(
                BuildingRecord(
                    source=row["source"],
                    source_id=row["source_id"],
                    geometry=json.loads(row["geometry_geojson"]),
                    properties=props if props else {},
                )
            )

        return records, total

    async def query_infrastructure(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        type_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[InfrastructureRecord], int]:
        """
        Query infrastructure facilities with spatial and type filters.

        Args:
            bbox: (west, south, east, north) bounding box in WGS84.
            type_filter: Filter by properties->>'type' value.
            limit: Max records to return.
            offset: Number of records to skip.

        Returns:
            Tuple of (list of InfrastructureRecord, total_count).
        """
        pool = self._ensure_pool()

        conditions: List[str] = []
        params: list = []
        idx = 1

        if bbox is not None:
            west, south, east, north = bbox
            conditions.append(
                f"ST_Intersects(geometry, ST_MakeEnvelope(${idx}, ${idx+1}, ${idx+2}, ${idx+3}, 4326))"
            )
            params.extend([west, south, east, north])
            idx += 4

        if type_filter is not None:
            conditions.append(f"properties->>'type' = ${idx}")
            params.append(type_filter)
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        sql = f"""
            SELECT
                source, source_id,
                ST_AsGeoJSON(geometry)::text AS geometry_geojson,
                properties,
                COUNT(*) OVER() AS total_count
            FROM context_infrastructure
            WHERE {where_clause}
            ORDER BY ingested_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        rows = await pool.fetch(sql, *params)

        total = rows[0]["total_count"] if rows else 0

        records = []
        for row in rows:
            props = row["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            records.append(
                InfrastructureRecord(
                    source=row["source"],
                    source_id=row["source_id"],
                    geometry=json.loads(row["geometry_geojson"]),
                    properties=props if props else {},
                )
            )

        return records, total

    async def query_weather(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        time_start: Optional[datetime] = None,
        time_end: Optional[datetime] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[WeatherRecord], int]:
        """
        Query weather observations with spatial and temporal filters.

        Args:
            bbox: (west, south, east, north) bounding box in WGS84.
            time_start: Minimum observation time (inclusive).
            time_end: Maximum observation time (inclusive).
            limit: Max records to return.
            offset: Number of records to skip.

        Returns:
            Tuple of (list of WeatherRecord, total_count).
        """
        pool = self._ensure_pool()

        conditions: List[str] = []
        params: list = []
        idx = 1

        if bbox is not None:
            west, south, east, north = bbox
            conditions.append(
                f"ST_Intersects(geometry, ST_MakeEnvelope(${idx}, ${idx+1}, ${idx+2}, ${idx+3}, 4326))"
            )
            params.extend([west, south, east, north])
            idx += 4

        if time_start is not None:
            conditions.append(f"observation_time >= ${idx}")
            params.append(time_start)
            idx += 1

        if time_end is not None:
            conditions.append(f"observation_time <= ${idx}")
            params.append(time_end)
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        sql = f"""
            SELECT
                source, source_id,
                ST_AsGeoJSON(geometry)::text AS geometry_geojson,
                properties,
                observation_time,
                COUNT(*) OVER() AS total_count
            FROM context_weather
            WHERE {where_clause}
            ORDER BY observation_time DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
        """

        rows = await pool.fetch(sql, *params)

        total = rows[0]["total_count"] if rows else 0

        records = []
        for row in rows:
            props = row["properties"]
            if isinstance(props, str):
                props = json.loads(props)
            records.append(
                WeatherRecord(
                    source=row["source"],
                    source_id=row["source_id"],
                    geometry=json.loads(row["geometry_geojson"]),
                    properties=props if props else {},
                    observation_time=row["observation_time"],
                )
            )

        return records, total

    async def get_job_context_summary(self, job_id: UUID) -> ContextSummary:
        """
        Get context usage summary for a specific job.

        Aggregates counts from job_context_usage grouped by context_table
        and usage_type.

        Args:
            job_id: The job to summarize.

        Returns:
            ContextSummary with per-table and total counts.
        """
        pool = self._ensure_pool()

        rows = await pool.fetch(
            """
            SELECT context_table, usage_type, COUNT(*) AS cnt
            FROM job_context_usage
            WHERE job_id = $1
            GROUP BY context_table, usage_type
            """,
            job_id,
        )

        # Build summary from aggregated rows
        counts: Dict[str, Dict[str, int]] = {}
        for row in rows:
            table = row["context_table"]
            utype = row["usage_type"]
            if table not in counts:
                counts[table] = {"ingested": 0, "reused": 0}
            counts[table][utype] = row["cnt"]

        # Map table names to summary field prefixes
        table_to_prefix = {
            "datasets": "datasets",
            "context_buildings": "buildings",
            "context_infrastructure": "infrastructure",
            "context_weather": "weather",
        }

        kwargs: Dict[str, int] = {}
        total_ingested = 0
        total_reused = 0

        for table_name, prefix in table_to_prefix.items():
            ingested = counts.get(table_name, {}).get("ingested", 0)
            reused = counts.get(table_name, {}).get("reused", 0)
            kwargs[f"{prefix}_ingested"] = ingested
            kwargs[f"{prefix}_reused"] = reused
            total_ingested += ingested
            total_reused += reused

        kwargs["total_ingested"] = total_ingested
        kwargs["total_reused"] = total_reused
        kwargs["total"] = total_ingested + total_reused

        return ContextSummary(**kwargs)

    async def get_lakehouse_stats(self) -> Dict[str, Any]:
        """
        Get overall lakehouse statistics.

        Returns total row counts per table, total spatial extent,
        and distinct sources per table.

        Returns:
            Dict with keys: tables (per-table stats), total_rows,
            spatial_extent, usage_stats.
        """
        pool = self._ensure_pool()

        tables_info = {
            "datasets": "datasets",
            "buildings": "context_buildings",
            "infrastructure": "context_infrastructure",
            "weather": "context_weather",
        }

        stats: Dict[str, Any] = {"tables": {}, "total_rows": 0}

        for label, table_name in tables_info.items():
            row = await pool.fetchrow(
                f"""
                SELECT
                    COUNT(*) AS row_count,
                    array_agg(DISTINCT source) AS sources
                FROM {table_name}
                """
            )
            count = row["row_count"]
            sources = row["sources"] if row["sources"] and row["sources"][0] is not None else []
            stats["tables"][label] = {
                "row_count": count,
                "sources": list(sources),
            }
            stats["total_rows"] += count

        # Total spatial extent across all tables
        extent_row = await pool.fetchrow(
            """
            SELECT
                ST_AsGeoJSON(ST_Extent(geom))::text AS extent_geojson
            FROM (
                SELECT geometry AS geom FROM datasets
                UNION ALL
                SELECT geometry AS geom FROM context_buildings
                UNION ALL
                SELECT geometry AS geom FROM context_infrastructure
                UNION ALL
                SELECT geometry AS geom FROM context_weather
            ) sub
            """
        )
        if extent_row and extent_row["extent_geojson"]:
            stats["spatial_extent"] = json.loads(extent_row["extent_geojson"])
        else:
            stats["spatial_extent"] = None

        # Usage stats
        usage_row = await pool.fetch(
            """
            SELECT usage_type, COUNT(*) AS cnt
            FROM job_context_usage
            GROUP BY usage_type
            """
        )
        stats["usage_stats"] = {
            row["usage_type"]: row["cnt"] for row in usage_row
        }

        return stats
