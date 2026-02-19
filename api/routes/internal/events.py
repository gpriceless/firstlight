"""
Internal SSE Event Streaming Endpoint.

Provides GET /internal/v1/events/stream for real-time event streaming
using Server-Sent Events (SSE) with CloudEvents v1.0 envelopes.

Events are sourced from PostgreSQL LISTEN/NOTIFY on the 'job_events' channel
(set up by migration 002_job_events_notify.sql). Each event is wrapped in a
CloudEvents v1.0 envelope before streaming.

Supports:
- Last-Event-ID header for replay of missed events
- customer_id query parameter for tenant filtering
- type query parameter for event type filtering
- Backpressure handling via per-connection Queue(maxsize=500)

Tasks 3.3 + 3.4
"""

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse

from api.routes.internal.deps import get_current_customer, require_internal_scope

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Partner Integration - Events"])

# Maximum events buffered per SSE connection before backpressure kicks in
SSE_QUEUE_MAX_SIZE = 500


async def _get_dedicated_connection():
    """
    Create a dedicated asyncpg connection for LISTEN (not from pool).

    The LISTEN connection must be separate from the connection pool because
    it holds a persistent server-side cursor for notifications.
    """
    try:
        import asyncpg
    except ImportError:
        raise RuntimeError(
            "asyncpg is required for SSE streaming. "
            "Install: pip install firstlight[control-plane]"
        )

    from api.config import get_settings

    settings = get_settings()
    db = settings.database

    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
    return await asyncpg.connect(dsn)


async def _replay_events(
    conn,
    last_event_id: int,
    customer_id: Optional[str] = None,
    event_type_filter: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Replay events from the job_events table starting after the given event_seq.

    Used when the client sends a Last-Event-ID header to catch up on missed events.
    """
    from core.events.cloudevents import build_cloudevent

    conditions = ["event_seq > $1"]
    params: list = [last_event_id]
    idx = 2

    if customer_id:
        conditions.append(f"customer_id = ${idx}")
        params.append(customer_id)
        idx += 1

    if event_type_filter:
        conditions.append(f"event_type = ${idx}")
        params.append(event_type_filter)
        idx += 1

    where_clause = " AND ".join(conditions)

    rows = await conn.fetch(
        f"""
        SELECT event_seq, job_id, customer_id, event_type, phase, status,
               reasoning, actor, payload, occurred_at
        FROM job_events
        WHERE {where_clause}
        ORDER BY event_seq ASC
        """,
        *params,
    )

    for row in rows:
        payload = {}
        if row["payload"] is not None:
            if isinstance(row["payload"], str):
                payload = json.loads(row["payload"])
            else:
                payload = dict(row["payload"]) if row["payload"] else {}

        envelope = build_cloudevent(
            event_seq=row["event_seq"],
            job_id=str(row["job_id"]),
            customer_id=row["customer_id"],
            event_type=row["event_type"],
            phase=row["phase"],
            status=row["status"],
            occurred_at=row["occurred_at"],
            payload=payload,
            reasoning=row["reasoning"],
            actor=row["actor"],
        )
        yield envelope


async def _stream_events(
    customer_id: Optional[str] = None,
    event_type_filter: Optional[str] = None,
    last_event_id: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Main SSE event generator.

    1. Establishes a dedicated LISTEN connection
    2. Replays missed events if Last-Event-ID is provided
    3. Streams new events in real time via pg_notify
    4. Implements backpressure with per-connection Queue(maxsize=500)
    """
    from core.events.cloudevents import (
        build_cloudevent,
        cloudevent_to_sse_frame,
        parse_job_event_notification,
    )

    conn = await _get_dedicated_connection()
    event_queue: asyncio.Queue = asyncio.Queue(maxsize=SSE_QUEUE_MAX_SIZE)
    disconnected = False

    async def notification_handler(conn_ref, pid, channel, payload_str):
        """Handle incoming pg_notify notifications."""
        nonlocal disconnected
        if disconnected:
            return

        try:
            event_data = parse_job_event_notification(payload_str)

            # Apply customer_id filter
            if customer_id and event_data.get("customer_id") != customer_id:
                return

            # Apply event_type filter
            if event_type_filter and event_data.get("event_type") != event_type_filter:
                return

            payload = {}
            if event_data.get("payload") is not None:
                if isinstance(event_data["payload"], str):
                    payload = json.loads(event_data["payload"])
                elif isinstance(event_data["payload"], dict):
                    payload = event_data["payload"]

            envelope = build_cloudevent(
                event_seq=event_data["event_seq"],
                job_id=str(event_data["job_id"]),
                customer_id=event_data["customer_id"],
                event_type=event_data["event_type"],
                phase=event_data["phase"],
                status=event_data["status"],
                occurred_at=event_data.get("occurred_at"),
                payload=payload,
                reasoning=event_data.get("reasoning"),
                actor=event_data.get("actor"),
            )

            try:
                event_queue.put_nowait(envelope)
            except asyncio.QueueFull:
                # Backpressure: queue is full, signal disconnect
                logger.warning(
                    "SSE backpressure: queue full (%d events), disconnecting slow consumer",
                    SSE_QUEUE_MAX_SIZE,
                )
                disconnected = True

        except Exception as e:
            logger.error("Error processing notification: %s", e)

    try:
        # Register the notification handler and start listening
        await conn.add_listener("job_events", notification_handler)

        # Send initial comment to establish SSE connection
        yield ": connected\n\n"

        # Replay missed events if Last-Event-ID was provided
        if last_event_id is not None:
            async for envelope in _replay_events(
                conn, last_event_id, customer_id, event_type_filter
            ):
                yield cloudevent_to_sse_frame(envelope)

        # Stream new events in real time
        while not disconnected:
            try:
                # Wait for events with a periodic heartbeat
                envelope = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                yield cloudevent_to_sse_frame(envelope)
            except asyncio.TimeoutError:
                # Send heartbeat comment to keep connection alive
                yield ": heartbeat\n\n"

        # If we got here due to backpressure, send 503 comment frame
        if disconnected:
            yield ": 503 Service Unavailable - backpressure limit reached\n\n"

    except asyncio.CancelledError:
        logger.info("SSE connection cancelled by client")
    except Exception as e:
        logger.error("SSE stream error: %s", e)
        yield f": error {str(e)}\n\n"
    finally:
        try:
            await conn.remove_listener("job_events", notification_handler)
            await conn.close()
        except Exception:
            pass


@router.get(
    "/events/stream",
    summary="Stream job events via SSE",
    description=(
        "Server-Sent Events endpoint that streams job events in real time "
        "using CloudEvents v1.0 envelopes. Supports Last-Event-ID replay, "
        "customer_id scoping, and event type filtering."
    ),
    responses={
        200: {
            "description": "SSE event stream",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_events(
    request: Request,
    user=Depends(require_internal_scope),
    customer_id: Optional[str] = Query(
        default=None,
        description="Filter events by customer ID",
    ),
    type: Optional[str] = Query(
        default=None,
        description="Filter events by event type",
    ),
    last_event_id: Optional[str] = Header(
        default=None,
        alias="Last-Event-ID",
        description="Resume from this event sequence number",
    ),
):
    """Stream job events as Server-Sent Events with CloudEvents envelopes."""
    # Parse Last-Event-ID header
    replay_from: Optional[int] = None
    if last_event_id is not None:
        try:
            replay_from = int(last_event_id)
        except (ValueError, TypeError):
            pass

    # If customer_id not explicitly provided, use the authenticated customer
    if customer_id is None:
        customer_id = getattr(request.state, "customer_id", None)

    return StreamingResponse(
        _stream_events(
            customer_id=customer_id,
            event_type_filter=type,
            last_event_id=replay_from,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
