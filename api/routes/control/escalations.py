"""
Control Plane Escalation Endpoints.

Provides endpoints for creating, resolving, and listing
escalations scoped to jobs.

Task 2.8: POST, PATCH, GET /control/v1/jobs/{job_id}/escalations
"""

import json
import logging
import uuid
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Path, Query, Request, status

from api.models.control import (
    EscalationRequest,
    EscalationResolveRequest,
    EscalationResponse,
    PaginatedEscalationsResponse,
)
from api.models.errors import (
    ConflictError,
    NotFoundError,
)
from api.routes.control import get_current_customer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["LLM Control - Escalations"])


# =============================================================================
# Backend Helpers
# =============================================================================


def _get_backend():
    """Get the active PostGIS backend instance."""
    from agents.orchestrator.backends.postgis_backend import PostGISStateBackend
    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    return PostGISStateBackend(
        host=db.host,
        port=db.port,
        database=db.name,
        user=db.user,
        password=db.password,
    )


async def _get_connected_backend():
    """Get a connected backend instance."""
    backend = _get_backend()
    await backend.connect()
    return backend


async def _verify_job_access(backend, job_id: str, customer_id: str) -> None:
    """Verify that the job exists and belongs to the customer."""
    job = await backend.get_state(job_id)
    if job is None or job.customer_id != customer_id:
        raise NotFoundError(message=f"Job '{job_id}' not found")


# =============================================================================
# Task 2.8: Escalation Endpoints
# =============================================================================


@router.post(
    "/jobs/{job_id}/escalations",
    response_model=EscalationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an escalation for a job",
    description=(
        "Create an escalation for a job with a severity level, reason, "
        "and optional context. Requires escalation:manage permission."
    ),
)
async def create_escalation(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    body: EscalationRequest,
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> EscalationResponse:
    """Create an escalation for a job."""
    backend = await _get_connected_backend()
    try:
        await _verify_job_access(backend, job_id, customer_id)

        pool = backend._ensure_pool()
        escalation_id = uuid.uuid4()
        context_json = json.dumps(body.context) if body.context else None

        row = await pool.fetchrow(
            """
            INSERT INTO escalations
                (escalation_id, job_id, customer_id, reason, severity, context)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb)
            RETURNING escalation_id, job_id, customer_id, reason, severity,
                      context, created_at, resolved_at, resolution, resolved_by
            """,
            escalation_id,
            uuid.UUID(job_id),
            customer_id,
            body.reason,
            body.severity.value,
            context_json,
        )

        # Record escalation event
        job = await backend.get_state(job_id)
        if job is not None:
            await backend.record_event(
                job_id=job_id,
                customer_id=customer_id,
                event_type="escalation.created",
                phase=job.phase,
                status=job.status,
                actor=customer_id,
                payload={
                    "escalation_id": str(escalation_id),
                    "severity": body.severity.value,
                    "reason": body.reason,
                },
            )

        return _row_to_escalation_response(row)
    finally:
        await backend.close()


@router.patch(
    "/jobs/{job_id}/escalations/{escalation_id}",
    response_model=EscalationResponse,
    summary="Resolve an escalation",
    description=(
        "Resolve an escalation with a resolution text. "
        "A second resolve attempt returns HTTP 409. "
        "Requires escalation:manage permission."
    ),
    responses={
        409: {"description": "Escalation already resolved"},
        404: {"description": "Escalation not found"},
    },
)
async def resolve_escalation(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    escalation_id: Annotated[str, Path(description="Escalation identifier (UUID)")],
    body: EscalationResolveRequest,
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> EscalationResponse:
    """Resolve an escalation."""
    backend = await _get_connected_backend()
    try:
        await _verify_job_access(backend, job_id, customer_id)

        pool = backend._ensure_pool()

        # Check if escalation exists and belongs to this job+customer
        existing = await pool.fetchrow(
            """
            SELECT escalation_id, resolved_at, resolution, resolved_by
            FROM escalations
            WHERE escalation_id = $1 AND job_id = $2 AND customer_id = $3
            """,
            uuid.UUID(escalation_id),
            uuid.UUID(job_id),
            customer_id,
        )

        if existing is None:
            raise NotFoundError(
                message=f"Escalation '{escalation_id}' not found for job '{job_id}'"
            )

        # Check if already resolved
        if existing["resolved_at"] is not None:
            raise ConflictError(
                message=(
                    f"Escalation '{escalation_id}' is already resolved. "
                    f"Original resolution: {existing['resolution']}"
                )
            )

        # Resolve the escalation
        row = await pool.fetchrow(
            """
            UPDATE escalations
            SET resolved_at = now(), resolution = $1, resolved_by = $2
            WHERE escalation_id = $3
            RETURNING escalation_id, job_id, customer_id, reason, severity,
                      context, created_at, resolved_at, resolution, resolved_by
            """,
            body.resolution,
            customer_id,
            uuid.UUID(escalation_id),
        )

        # Record resolution event
        job = await backend.get_state(job_id)
        if job is not None:
            await backend.record_event(
                job_id=job_id,
                customer_id=customer_id,
                event_type="escalation.resolved",
                phase=job.phase,
                status=job.status,
                actor=customer_id,
                payload={
                    "escalation_id": escalation_id,
                    "resolution": body.resolution,
                },
            )

        return _row_to_escalation_response(row)
    finally:
        await backend.close()


@router.get(
    "/jobs/{job_id}/escalations",
    response_model=PaginatedEscalationsResponse,
    summary="List escalations for a job",
    description=(
        "List escalations for a job, optionally filtered by severity. "
        "Requires state:read permission."
    ),
)
async def list_escalations(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    customer_id: Annotated[str, Depends(get_current_customer)],
    severity: Annotated[
        Optional[str],
        Query(description="Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)"),
    ] = None,
) -> PaginatedEscalationsResponse:
    """List escalations for a job."""
    backend = await _get_connected_backend()
    try:
        await _verify_job_access(backend, job_id, customer_id)

        pool = backend._ensure_pool()

        # Build query
        conditions = ["job_id = $1", "customer_id = $2"]
        params: list = [uuid.UUID(job_id), customer_id]
        idx = 3

        if severity is not None:
            conditions.append(f"severity = ${idx}")
            params.append(severity.upper())
            idx += 1

        where_clause = " AND ".join(conditions)

        rows = await pool.fetch(
            f"""
            SELECT escalation_id, job_id, customer_id, reason, severity,
                   context, created_at, resolved_at, resolution, resolved_by
            FROM escalations
            WHERE {where_clause}
            ORDER BY created_at DESC
            """,
            *params,
        )

        items = [_row_to_escalation_response(row) for row in rows]

        return PaginatedEscalationsResponse(
            items=items,
            total=len(items),
        )
    finally:
        await backend.close()


# =============================================================================
# Helpers
# =============================================================================


def _row_to_escalation_response(row) -> EscalationResponse:
    """Convert an asyncpg row to an EscalationResponse."""
    context = None
    if row["context"] is not None:
        if isinstance(row["context"], str):
            context = json.loads(row["context"])
        else:
            context = dict(row["context"]) if row["context"] else None

    return EscalationResponse(
        escalation_id=str(row["escalation_id"]),
        job_id=str(row["job_id"]),
        customer_id=row["customer_id"],
        severity=row["severity"],
        reason=row["reason"],
        context=context,
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
        resolution=row["resolution"],
        resolved_by=row["resolved_by"],
    )
