"""
Internal Webhook Management Endpoints.

Provides CRUD endpoints for webhook subscriptions with SSRF protection.
Registration validates HTTPS URLs and rejects private/loopback IP addresses.
Listing uses cursor-based pagination.

Task 3.5
"""

import json
import logging
import secrets
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Path, Query, Request, status

from api.models.internal import (
    CreateWebhookRequest,
    CursorPaginatedWebhooksResponse,
    WebhookResponse,
)
from api.models.errors import NotFoundError
from api.routes.internal.deps import get_current_customer, require_internal_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Partner Integration - Webhooks"])


async def _get_pool():
    """Get an asyncpg connection pool."""
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError("asyncpg is required for webhook management")

    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    return await asyncpg.create_pool(dsn, min_size=1, max_size=5)


def _row_to_response(row) -> WebhookResponse:
    """Convert an asyncpg row to a WebhookResponse."""
    event_filter = list(row["event_filter"]) if row["event_filter"] else []
    return WebhookResponse(
        subscription_id=str(row["subscription_id"]),
        customer_id=row["customer_id"],
        target_url=row["target_url"],
        event_filter=event_filter,
        active=row["active"],
        created_at=row["created_at"],
    )


@router.post(
    "/webhooks",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a webhook subscription",
    description=(
        "Register a new webhook endpoint for event delivery. "
        "URL must be HTTPS and must not resolve to private/loopback addresses."
    ),
)
async def create_webhook(
    body: CreateWebhookRequest,
    request: Request,
    user=Depends(require_internal_scope),
    customer_id: str = Depends(get_current_customer),
) -> WebhookResponse:
    """Register a new webhook subscription."""
    pool = await _get_pool()
    try:
        # Generate a secure secret key for HMAC signing
        secret_key = secrets.token_urlsafe(32)

        row = await pool.fetchrow(
            """
            INSERT INTO webhook_subscriptions
                (customer_id, target_url, secret_key, event_filter)
            VALUES ($1, $2, $3, $4)
            RETURNING subscription_id, customer_id, target_url, event_filter,
                      active, created_at
            """,
            customer_id,
            body.target_url,
            secret_key,
            body.event_filter,
        )

        logger.info(
            "Created webhook subscription %s for customer %s -> %s",
            row["subscription_id"],
            customer_id,
            body.target_url,
        )

        return _row_to_response(row)
    finally:
        await pool.close()


@router.get(
    "/webhooks",
    response_model=CursorPaginatedWebhooksResponse,
    summary="List webhook subscriptions",
    description=(
        "List webhook subscriptions for the authenticated customer. "
        "Uses cursor-based pagination."
    ),
)
async def list_webhooks(
    request: Request,
    user=Depends(require_internal_scope),
    customer_id: str = Depends(get_current_customer),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of items to return",
    ),
    after: Optional[str] = Query(
        default=None,
        description="Cursor to start after (subscription_id from previous page)",
    ),
) -> CursorPaginatedWebhooksResponse:
    """List webhook subscriptions with cursor-based pagination."""
    pool = await _get_pool()
    try:
        # Fetch limit + 1 to determine has_more
        fetch_limit = limit + 1

        if after:
            # Get the created_at for the cursor subscription to paginate
            cursor_row = await pool.fetchrow(
                """
                SELECT created_at, subscription_id
                FROM webhook_subscriptions
                WHERE subscription_id = $1 AND customer_id = $2
                """,
                uuid.UUID(after),
                customer_id,
            )

            if cursor_row:
                rows = await pool.fetch(
                    """
                    SELECT subscription_id, customer_id, target_url, event_filter,
                           active, created_at
                    FROM webhook_subscriptions
                    WHERE customer_id = $1
                      AND (created_at, subscription_id) > ($2, $3)
                    ORDER BY created_at ASC, subscription_id ASC
                    LIMIT $4
                    """,
                    customer_id,
                    cursor_row["created_at"],
                    cursor_row["subscription_id"],
                    fetch_limit,
                )
            else:
                rows = []
        else:
            rows = await pool.fetch(
                """
                SELECT subscription_id, customer_id, target_url, event_filter,
                       active, created_at
                FROM webhook_subscriptions
                WHERE customer_id = $1
                ORDER BY created_at ASC, subscription_id ASC
                LIMIT $2
                """,
                customer_id,
                fetch_limit,
            )

        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        items = [_row_to_response(row) for row in rows]
        next_cursor = items[-1].subscription_id if has_more and items else None

        return CursorPaginatedWebhooksResponse(
            items=items,
            next_cursor=next_cursor,
            has_more=has_more,
        )
    finally:
        await pool.close()


@router.delete(
    "/webhooks/{subscription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a webhook subscription",
    description="Deregister a webhook subscription.",
)
async def delete_webhook(
    subscription_id: Annotated[str, Path(description="Subscription UUID")],
    request: Request,
    user=Depends(require_internal_scope),
    customer_id: str = Depends(get_current_customer),
):
    """Delete a webhook subscription."""
    pool = await _get_pool()
    try:
        result = await pool.execute(
            """
            DELETE FROM webhook_subscriptions
            WHERE subscription_id = $1 AND customer_id = $2
            """,
            uuid.UUID(subscription_id),
            customer_id,
        )

        if result == "DELETE 0":
            raise NotFoundError(
                message=f"Webhook subscription '{subscription_id}' not found"
            )

        logger.info(
            "Deleted webhook subscription %s for customer %s",
            subscription_id,
            customer_id,
        )
    finally:
        await pool.close()
