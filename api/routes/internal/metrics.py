"""
Internal Metrics and Queue Summary Endpoints.

Provides pipeline health metrics and queue summary from materialized views
that are refreshed every 30 seconds.

Task 3.8
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.routes.internal.deps import get_current_customer, require_internal_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Partner Integration - Metrics"])

# Runtime counter for active SSE connections (set by events.py)
_active_sse_connections = 0


def increment_sse_connections():
    """Increment active SSE connection counter."""
    global _active_sse_connections
    _active_sse_connections += 1


def decrement_sse_connections():
    """Decrement active SSE connection counter."""
    global _active_sse_connections
    _active_sse_connections = max(0, _active_sse_connections - 1)


# =============================================================================
# Response Models
# =============================================================================


class PipelineHealthMetrics(BaseModel):
    """Pipeline health metrics response."""

    jobs_completed_1h: int = Field(default=0, description="Jobs completed in last hour")
    jobs_failed_1h: int = Field(default=0, description="Jobs failed in last hour")
    jobs_completed_24h: int = Field(default=0, description="Jobs completed in last 24h")
    jobs_failed_24h: int = Field(default=0, description="Jobs failed in last 24h")
    p50_duration_s: float = Field(default=0.0, description="P50 job duration (seconds)")
    p95_duration_s: float = Field(default=0.0, description="P95 job duration (seconds)")
    active_sse_connections: int = Field(
        default=0, description="Currently active SSE connections"
    )
    webhook_success_rate_1h: Optional[float] = Field(
        default=None, description="Webhook delivery success rate (1h)"
    )


class OldestStuckJob(BaseModel):
    """Oldest stuck job detail."""

    job_id: str = Field(..., description="Stuck job UUID")
    stuck_since: datetime = Field(..., description="When the job became stuck")


class QueueSummary(BaseModel):
    """Queue summary response."""

    per_phase_counts: Dict[str, int] = Field(
        default_factory=dict, description="Active job counts by phase"
    )
    stuck_count: int = Field(default=0, description="Number of stuck jobs")
    awaiting_review_count: int = Field(
        default=0, description="Jobs with open escalations"
    )
    oldest_stuck_job: Optional[OldestStuckJob] = Field(
        default=None, description="Oldest stuck job details (null if none)"
    )


# =============================================================================
# Helper
# =============================================================================


async def _get_pool():
    """Get an asyncpg connection pool."""
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError("asyncpg is required for metrics endpoints")

    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    return await asyncpg.create_pool(dsn, min_size=1, max_size=3)


# =============================================================================
# Endpoints
# =============================================================================


@router.get(
    "/metrics",
    response_model=PipelineHealthMetrics,
    summary="Pipeline health metrics",
    description=(
        "Returns aggregated pipeline health metrics from the pipeline_health_metrics "
        "materialized view. Refreshed every 30 seconds. Responds within 100ms."
    ),
)
async def get_metrics(
    request: Request,
    user=Depends(require_internal_scope),
) -> JSONResponse:
    """Get pipeline health metrics."""
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            "SELECT * FROM pipeline_health_metrics LIMIT 1"
        )

        if row is None:
            metrics = PipelineHealthMetrics(
                active_sse_connections=_active_sse_connections,
            )
        else:
            metrics = PipelineHealthMetrics(
                jobs_completed_1h=row["jobs_completed_1h"] or 0,
                jobs_failed_1h=row["jobs_failed_1h"] or 0,
                jobs_completed_24h=row["jobs_completed_24h"] or 0,
                jobs_failed_24h=row["jobs_failed_24h"] or 0,
                p50_duration_s=float(row["p50_duration_s"] or 0),
                p95_duration_s=float(row["p95_duration_s"] or 0),
                active_sse_connections=_active_sse_connections,
                webhook_success_rate_1h=(
                    float(row["webhook_success_rate_1h"])
                    if row["webhook_success_rate_1h"] is not None
                    else None
                ),
            )

        refreshed_at = row["refreshed_at"].isoformat() if row and row["refreshed_at"] else None

        return JSONResponse(
            content=metrics.model_dump(),
            headers={
                "Cache-Control": "max-age=30",
                "X-Metrics-Refreshed-At": refreshed_at or "unknown",
            },
        )
    finally:
        await pool.close()


@router.get(
    "/queue/summary",
    response_model=QueueSummary,
    summary="Queue summary",
    description=(
        "Returns per-phase job counts, stuck job count, and awaiting review count "
        "from the queue_summary materialized view."
    ),
)
async def get_queue_summary(
    request: Request,
    user=Depends(require_internal_scope),
) -> QueueSummary:
    """Get queue summary."""
    pool = await _get_pool()
    try:
        row = await pool.fetchrow(
            "SELECT * FROM queue_summary LIMIT 1"
        )

        if row is None:
            return QueueSummary()

        per_phase_counts = {}
        if row["per_phase_counts"]:
            if isinstance(row["per_phase_counts"], str):
                per_phase_counts = json.loads(row["per_phase_counts"])
            else:
                per_phase_counts = dict(row["per_phase_counts"])

        oldest_stuck = None
        if row["oldest_stuck_job_id"] is not None:
            oldest_stuck = OldestStuckJob(
                job_id=str(row["oldest_stuck_job_id"]),
                stuck_since=row["oldest_stuck_since"],
            )

        return QueueSummary(
            per_phase_counts=per_phase_counts,
            stuck_count=row["stuck_count"] or 0,
            awaiting_review_count=row["awaiting_review_count"] or 0,
            oldest_stuck_job=oldest_stuck,
        )
    finally:
        await pool.close()
