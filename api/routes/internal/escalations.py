"""
Internal Open Escalations List Endpoint.

Provides GET /internal/v1/escalations for listing open escalations
for the authenticated customer. Reuses the escalations table from Phase 2.

Supports severity and since query parameters for filtering.

Task 3.9
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field

from api.models.control import EscalationResponse
from api.routes.internal.deps import get_current_customer, require_internal_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Partner Integration - Escalations"])


class OpenEscalationsResponse(BaseModel):
    """Response for open escalations list."""

    items: List[EscalationResponse] = Field(..., description="Open escalations")
    total: int = Field(..., description="Total count of open escalations")


async def _get_pool():
    """Get an asyncpg connection pool."""
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError("asyncpg is required")

    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    return await asyncpg.create_pool(dsn, min_size=1, max_size=3)


@router.get(
    "/escalations",
    response_model=OpenEscalationsResponse,
    summary="List open escalations",
    description=(
        "List open (unresolved) escalations for the authenticated customer. "
        "Supports filtering by severity and creation time."
    ),
)
async def list_open_escalations(
    request: Request,
    user=Depends(require_internal_scope),
    customer_id: str = Depends(get_current_customer),
    severity: Optional[str] = Query(
        default=None,
        description="Filter by severity (LOW, MEDIUM, HIGH, CRITICAL)",
    ),
    since: Optional[datetime] = Query(
        default=None,
        description="Only return escalations created after this timestamp (ISO 8601)",
    ),
) -> OpenEscalationsResponse:
    """List open escalations for the authenticated customer."""
    pool = await _get_pool()
    try:
        conditions = ["customer_id = $1", "resolved_at IS NULL"]
        params: list = [customer_id]
        idx = 2

        if severity is not None:
            conditions.append(f"severity = ${idx}")
            params.append(severity.upper())
            idx += 1

        if since is not None:
            conditions.append(f"created_at >= ${idx}")
            params.append(since)
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

        items = []
        for row in rows:
            context = None
            if row["context"] is not None:
                if isinstance(row["context"], str):
                    context = json.loads(row["context"])
                else:
                    context = dict(row["context"]) if row["context"] else None

            items.append(
                EscalationResponse(
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
            )

        return OpenEscalationsResponse(
            items=items,
            total=len(items),
        )
    finally:
        await pool.close()
