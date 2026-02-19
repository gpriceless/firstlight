"""
PostGIS state backend implementation.

Uses asyncpg for native async PostgreSQL access with connection pooling.
All geospatial operations use PostGIS functions (ST_Multi, ST_GeomFromGeoJSON,
ST_AsGeoJSON). State transitions are atomic via conditional UPDATE ... RETURNING.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)
from agents.orchestrator.state_model import (
    is_valid_phase_status,
)

logger = logging.getLogger(__name__)


def _row_to_job_state(row: "asyncpg.Record") -> JobState:
    """Convert an asyncpg Row from the jobs table to a JobState."""
    aoi_geojson = None
    if row["aoi_geojson"] is not None:
        aoi_geojson = json.loads(row["aoi_geojson"])

    return JobState(
        job_id=str(row["job_id"]),
        customer_id=row["customer_id"],
        event_type=row["event_type"],
        phase=row["phase"],
        status=row["status"],
        aoi=aoi_geojson,
        parameters=json.loads(row["parameters"]) if isinstance(row["parameters"], str) else (row["parameters"] or {}),
        orchestrator_id=row["orchestrator_id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# Standard SELECT columns with AOI as GeoJSON
_SELECT_COLS = """
    job_id, customer_id, event_type,
    ST_AsGeoJSON(aoi)::text AS aoi_geojson,
    phase, status, orchestrator_id,
    parameters, created_at, updated_at
"""


class PostGISStateBackend(StateBackend):
    """
    PostGIS-backed state backend using asyncpg.

    Uses a connection pool for efficient async database access. All state
    transitions are atomic via conditional UPDATE ... RETURNING queries.

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
                "asyncpg is required for PostGISStateBackend. "
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
            dsn = f"postgresql://{self._user}:{self._password}@{self._host}:{self._port}/{self._database}"
            self._pool = await asyncpg.create_pool(
                dsn,
                min_size=self._min_pool_size,
                max_size=self._max_pool_size,
            )
            logger.info("PostGIS connection pool created")

    async def close(self) -> None:
        """Close the connection pool if we own it."""
        if self._pool is not None and self._owns_pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostGIS connection pool closed")

    def _ensure_pool(self) -> "asyncpg.Pool":
        """Return the pool, raising if not connected."""
        if self._pool is None:
            raise RuntimeError(
                "PostGISStateBackend is not connected. Call connect() first."
            )
        return self._pool

    async def get_state(self, job_id: str) -> Optional[JobState]:
        """Retrieve job state by ID."""
        pool = self._ensure_pool()
        row = await pool.fetchrow(
            f"SELECT {_SELECT_COLS} FROM jobs WHERE job_id = $1",
            uuid.UUID(job_id),
        )
        if row is None:
            return None
        return _row_to_job_state(row)

    async def create_job(
        self,
        customer_id: str,
        event_type: str,
        aoi_geojson: Dict[str, Any],
        phase: str,
        status: str,
        parameters: Optional[Dict[str, Any]] = None,
        orchestrator_id: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> JobState:
        """
        Create a new job in the database.

        This is an extension beyond the base ABC (which assumes jobs already
        exist). The Control API uses this to create jobs.

        Args:
            customer_id: Tenant identifier.
            event_type: Event type (e.g., 'flood', 'wildfire').
            aoi_geojson: Area of Interest as GeoJSON dict.
            phase: Initial phase.
            status: Initial status.
            parameters: Optional job parameters.
            orchestrator_id: Optional orchestrator ID.
            job_id: Optional specific job ID. Generated if None.

        Returns:
            The created JobState.
        """
        if not is_valid_phase_status(phase, status):
            raise ValueError(f"Invalid (phase, status) pair: ({phase}, {status})")

        pool = self._ensure_pool()
        jid = uuid.UUID(job_id) if job_id else uuid.uuid4()
        params_json = json.dumps(parameters or {})
        aoi_json_str = json.dumps(aoi_geojson)

        row = await pool.fetchrow(
            f"""
            INSERT INTO jobs (job_id, customer_id, event_type, aoi, phase, status,
                              orchestrator_id, parameters)
            VALUES ($1, $2, $3, ST_Multi(ST_GeomFromGeoJSON($4)), $5, $6, $7, $8::jsonb)
            RETURNING {_SELECT_COLS}
            """,
            jid,
            customer_id,
            event_type,
            aoi_json_str,
            phase,
            status,
            orchestrator_id,
            params_json,
        )
        return _row_to_job_state(row)

    async def set_state(
        self,
        job_id: str,
        phase: str,
        status: str,
        **kwargs: Any,
    ) -> JobState:
        """Set the phase and status for a job."""
        if not is_valid_phase_status(phase, status):
            raise ValueError(f"Invalid (phase, status) pair: ({phase}, {status})")

        pool = self._ensure_pool()

        # Build dynamic SET clause for extra kwargs
        set_parts = ["phase = $2", "status = $3", "updated_at = now()"]
        params: list = [uuid.UUID(job_id), phase, status]
        idx = 4

        if "orchestrator_id" in kwargs:
            set_parts.append(f"orchestrator_id = ${idx}")
            params.append(kwargs["orchestrator_id"])
            idx += 1
        if "parameters" in kwargs:
            set_parts.append(f"parameters = ${idx}::jsonb")
            params.append(json.dumps(kwargs["parameters"]))
            idx += 1

        set_clause = ", ".join(set_parts)

        row = await pool.fetchrow(
            f"""
            UPDATE jobs SET {set_clause}
            WHERE job_id = $1
            RETURNING {_SELECT_COLS}
            """,
            *params,
        )
        if row is None:
            raise KeyError(f"Job {job_id} not found")

        return _row_to_job_state(row)

    async def transition(
        self,
        job_id: str,
        expected_phase: str,
        expected_status: str,
        new_phase: str,
        new_status: str,
        *,
        reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> JobState:
        """
        Atomically transition a job from one state to another.

        Uses a conditional UPDATE ... RETURNING for atomic TOCTOU protection.
        On success, inserts a STATE_TRANSITION event in the same transaction.
        """
        if not is_valid_phase_status(expected_phase, expected_status):
            raise ValueError(
                f"Invalid expected (phase, status): ({expected_phase}, {expected_status})"
            )
        if not is_valid_phase_status(new_phase, new_status):
            raise ValueError(
                f"Invalid target (phase, status): ({new_phase}, {new_status})"
            )

        pool = self._ensure_pool()
        jid = uuid.UUID(job_id)

        async with pool.acquire() as conn:
            async with conn.transaction():
                # Atomic conditional update
                row = await conn.fetchrow(
                    f"""
                    UPDATE jobs
                    SET phase = $2, status = $3, updated_at = now()
                    WHERE job_id = $1 AND phase = $4 AND status = $5
                    RETURNING {_SELECT_COLS}
                    """,
                    jid,
                    new_phase,
                    new_status,
                    expected_phase,
                    expected_status,
                )

                if row is None:
                    # Transition failed -- fetch current state for error reporting
                    current = await conn.fetchrow(
                        "SELECT phase, status FROM jobs WHERE job_id = $1",
                        jid,
                    )
                    if current is None:
                        raise KeyError(f"Job {job_id} not found")
                    raise StateConflictError(
                        job_id=job_id,
                        expected_phase=expected_phase,
                        expected_status=expected_status,
                        actual_phase=current["phase"],
                        actual_status=current["status"],
                    )

                # Insert event within the same transaction
                await conn.execute(
                    """
                    INSERT INTO job_events
                        (job_id, customer_id, event_type, phase, status,
                         reasoning, actor, payload)
                    VALUES ($1, $2, 'STATE_TRANSITION', $3, $4, $5, $6, $7::jsonb)
                    """,
                    jid,
                    row["customer_id"],
                    new_phase,
                    new_status,
                    reason,
                    actor or "system",
                    json.dumps({
                        "from_phase": expected_phase,
                        "from_status": expected_status,
                    }),
                )

        return _row_to_job_state(row)

    async def list_jobs(
        self,
        *,
        customer_id: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobState]:
        """List jobs with optional filtering."""
        pool = self._ensure_pool()

        conditions = []
        params: list = []
        idx = 1

        if customer_id is not None:
            conditions.append(f"customer_id = ${idx}")
            params.append(customer_id)
            idx += 1
        if phase is not None:
            conditions.append(f"phase = ${idx}")
            params.append(phase)
            idx += 1
        if status is not None:
            conditions.append(f"status = ${idx}")
            params.append(status)
            idx += 1
        if event_type is not None:
            conditions.append(f"event_type = ${idx}")
            params.append(event_type)
            idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        limit_idx = idx
        idx += 1
        params.append(offset)
        offset_idx = idx

        rows = await pool.fetch(
            f"""
            SELECT {_SELECT_COLS}
            FROM jobs
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${limit_idx} OFFSET ${offset_idx}
            """,
            *params,
        )

        return [_row_to_job_state(row) for row in rows]

    async def checkpoint(
        self,
        job_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Save a checkpoint snapshot for a job."""
        pool = self._ensure_pool()
        jid = uuid.UUID(job_id)

        # Verify job exists and get its current phase
        current = await pool.fetchrow(
            "SELECT phase FROM jobs WHERE job_id = $1", jid
        )
        if current is None:
            raise KeyError(f"Job {job_id} not found")

        await pool.execute(
            """
            INSERT INTO job_checkpoints (job_id, phase, state_snapshot)
            VALUES ($1, $2, $3::jsonb)
            """,
            jid,
            current["phase"],
            json.dumps(payload),
        )

    async def record_event(
        self,
        job_id: str,
        customer_id: str,
        event_type: str,
        phase: str,
        status: str,
        actor: str,
        reasoning: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Record an event in the job_events table.

        This is an extension beyond the base ABC, used for recording
        reasoning entries, errors, and other non-transition events.

        Returns:
            The event_seq of the inserted event.
        """
        pool = self._ensure_pool()
        row = await pool.fetchrow(
            """
            INSERT INTO job_events
                (job_id, customer_id, event_type, phase, status,
                 reasoning, actor, payload)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            RETURNING event_seq
            """,
            uuid.UUID(job_id),
            customer_id,
            event_type,
            phase,
            status,
            reasoning,
            actor,
            json.dumps(payload or {}),
        )
        return row["event_seq"]

    async def get_latest_checkpoint(
        self,
        job_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest checkpoint for a job.

        Returns:
            The checkpoint payload dict, or None if no checkpoints exist.
        """
        pool = self._ensure_pool()
        row = await pool.fetchrow(
            """
            SELECT state_snapshot FROM job_checkpoints
            WHERE job_id = $1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            uuid.UUID(job_id),
        )
        if row is None:
            return None
        snapshot = row["state_snapshot"]
        if isinstance(snapshot, str):
            return json.loads(snapshot)
        return dict(snapshot) if snapshot else None
