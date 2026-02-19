"""
Status API Routes.

Provides endpoints for monitoring event processing status,
including real-time updates via WebSocket.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Dict

from fastapi import APIRouter, Depends, Path, WebSocket, WebSocketDisconnect, status

from api.dependencies import AuthDep, DBSessionDep, SettingsDep
from api.models.errors import EventNotFoundError
from api.models.responses import (
    EventStatus,
    PipelineStepProgress,
    ProgressResponse,
    StatusResponse,
)
from api.routes.events import deserialize_event_from_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Status"])


# Simulated pipeline stages for progress tracking
_PIPELINE_STAGES = [
    {"id": "discovery", "name": "Data Discovery", "weight": 0.15},
    {"id": "acquisition", "name": "Data Acquisition", "weight": 0.25},
    {"id": "preprocessing", "name": "Preprocessing", "weight": 0.15},
    {"id": "analysis", "name": "Analysis", "weight": 0.25},
    {"id": "validation", "name": "Quality Validation", "weight": 0.10},
    {"id": "product_generation", "name": "Product Generation", "weight": 0.10},
]


def _get_stage_status(
    event_status: EventStatus,
    stage_index: int,
    current_stage_index: int,
) -> str:
    """Determine the status of a pipeline stage based on event status."""
    if event_status == EventStatus.COMPLETED:
        return "completed"
    elif event_status == EventStatus.FAILED:
        if stage_index < current_stage_index:
            return "completed"
        elif stage_index == current_stage_index:
            return "failed"
        else:
            return "pending"
    elif event_status == EventStatus.CANCELLED:
        if stage_index < current_stage_index:
            return "completed"
        else:
            return "cancelled"
    else:
        if stage_index < current_stage_index:
            return "completed"
        elif stage_index == current_stage_index:
            return "in_progress"
        else:
            return "pending"


def _calculate_progress(event_status: EventStatus, current_stage_index: int) -> float:
    """Calculate overall progress percentage based on completed stages."""
    if event_status == EventStatus.COMPLETED:
        return 100.0
    elif event_status == EventStatus.PENDING:
        return 0.0
    elif event_status in [EventStatus.FAILED, EventStatus.CANCELLED]:
        # Progress at failure point
        completed_weight = sum(
            stage["weight"] for i, stage in enumerate(_PIPELINE_STAGES)
            if i < current_stage_index
        )
        return completed_weight * 100

    # In progress - calculate based on completed stages
    completed_weight = sum(
        stage["weight"] for i, stage in enumerate(_PIPELINE_STAGES)
        if i < current_stage_index
    )

    # Add partial progress for current stage (assume 50% through)
    if current_stage_index < len(_PIPELINE_STAGES):
        completed_weight += _PIPELINE_STAGES[current_stage_index]["weight"] * 0.5

    return min(completed_weight * 100, 99.9)


def _get_current_stage(event_status: EventStatus) -> tuple[int, str]:
    """Get the current stage index and name based on event status."""
    status_to_stage = {
        EventStatus.PENDING: (0, "Pending"),
        EventStatus.QUEUED: (0, "Queued"),
        EventStatus.DISCOVERING: (0, "Data Discovery"),
        EventStatus.ACQUIRING: (1, "Data Acquisition"),
        EventStatus.PROCESSING: (3, "Analysis"),
        EventStatus.VALIDATING: (4, "Quality Validation"),
        EventStatus.COMPLETED: (5, "Completed"),
        EventStatus.FAILED: (3, "Failed"),  # Assume failure during analysis
        EventStatus.CANCELLED: (0, "Cancelled"),
    }
    return status_to_stage.get(event_status, (0, "Unknown"))


@router.get(
    "/{event_id}/status",
    response_model=StatusResponse,
    summary="Get processing status",
    description="Get the current processing status for an event.",
    responses={
        200: {"description": "Event status"},
        404: {"description": "Event not found"},
    },
)
async def get_event_status(
    event_id: Annotated[str, Path(description="Event identifier")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> StatusResponse:
    """
    Get the current processing status for an event.

    Returns:
    - Current status (pending, processing, completed, failed)
    - Overall progress percentage
    - Current processing stage
    - Timing information
    """
    row = await db_session.get_event(event_id)
    if not row:
        raise EventNotFoundError(event_id)

    event = deserialize_event_from_db(row)
    stage_index, stage_name = _get_current_stage(event.status)
    progress = _calculate_progress(event.status, stage_index)

    # Determine message based on status
    if event.status == EventStatus.COMPLETED:
        message = "Processing completed successfully"
    elif event.status == EventStatus.FAILED:
        message = getattr(event, "error_message", None) or "Processing failed"
    elif event.status == EventStatus.CANCELLED:
        message = "Processing was cancelled"
    elif event.status == EventStatus.PENDING:
        message = "Waiting to start processing"
    else:
        message = f"Currently: {stage_name}"

    return StatusResponse(
        event_id=event_id,
        status=event.status,
        progress_percent=round(progress, 1),
        current_stage=stage_name,
        message=message,
        started_at=getattr(event, "started_at", None),
        estimated_completion=None,  # Would be calculated from historical data
        updated_at=event.updated_at,
    )


@router.get(
    "/{event_id}/progress",
    response_model=ProgressResponse,
    summary="Get detailed progress",
    description="Get detailed progress information including pipeline stage breakdown.",
    responses={
        200: {"description": "Detailed progress"},
        404: {"description": "Event not found"},
    },
)
async def get_event_progress(
    event_id: Annotated[str, Path(description="Event identifier")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> ProgressResponse:
    """
    Get detailed progress information for an event.

    Returns:
    - Overall progress percentage
    - Status of each pipeline stage
    - Data source discovery and acquisition counts
    - Product generation counts
    - Any processing warnings
    """
    row = await db_session.get_event(event_id)
    if not row:
        raise EventNotFoundError(event_id)

    event = deserialize_event_from_db(row)
    current_stage_index, _ = _get_current_stage(event.status)
    progress = _calculate_progress(event.status, current_stage_index)

    started_at = getattr(event, "started_at", None)
    completed_at = getattr(event, "completed_at", None)

    # Build stage progress information
    stages: list[PipelineStepProgress] = []
    for i, stage in enumerate(_PIPELINE_STAGES):
        stage_status = _get_stage_status(event.status, i, current_stage_index)

        # Determine stage progress
        if stage_status == "completed":
            stage_progress = 100.0
        elif stage_status == "in_progress":
            stage_progress = 50.0  # Simulated
        else:
            stage_progress = 0.0

        stages.append(
            PipelineStepProgress(
                step_id=stage["id"],
                step_name=stage["name"],
                status=stage_status,
                started_at=started_at if i <= current_stage_index else None,
                completed_at=completed_at if stage_status == "completed" and event.status == EventStatus.COMPLETED else None,
                progress_percent=stage_progress,
                message=None,
            )
        )

    # Simulated counts (would come from actual processing state)
    data_sources_discovered = 5 if current_stage_index > 0 else 0
    data_sources_acquired = 4 if current_stage_index > 1 else 0
    products_generated = 2 if event.status == EventStatus.COMPLETED else 0

    # Simulated warnings
    warnings = []
    if event.status == EventStatus.COMPLETED:
        warnings = [
            "Cloud cover exceeded threshold for 2 optical scenes, using SAR backup",
        ]

    return ProgressResponse(
        event_id=event_id,
        status=event.status,
        progress_percent=round(progress, 1),
        stages=stages,
        data_sources_discovered=data_sources_discovered,
        data_sources_acquired=data_sources_acquired,
        products_generated=products_generated,
        warnings=warnings,
        started_at=started_at,
        estimated_completion=None,
        updated_at=event.updated_at,
    )


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self) -> None:
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, event_id: str) -> None:
        """Accept and register a WebSocket connection."""
        await websocket.accept()
        if event_id not in self.active_connections:
            self.active_connections[event_id] = []
        self.active_connections[event_id].append(websocket)
        logger.info(f"WebSocket connected for event {event_id}")

    def disconnect(self, websocket: WebSocket, event_id: str) -> None:
        """Remove a WebSocket connection."""
        if event_id in self.active_connections:
            if websocket in self.active_connections[event_id]:
                self.active_connections[event_id].remove(websocket)
            if not self.active_connections[event_id]:
                del self.active_connections[event_id]
        logger.info(f"WebSocket disconnected for event {event_id}")

    async def send_update(self, event_id: str, data: Dict[str, Any]) -> None:
        """Send update to all connections for an event."""
        if event_id in self.active_connections:
            message = json.dumps(data, default=str)
            disconnected = []
            for connection in self.active_connections[event_id]:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.append(connection)
            # Clean up disconnected
            for conn in disconnected:
                self.disconnect(conn, event_id)


_connection_manager = ConnectionManager()


@router.websocket("/{event_id}/stream")
async def stream_event_updates(
    websocket: WebSocket,
    event_id: str,
    db_session: DBSessionDep = None,
) -> None:
    """
    WebSocket endpoint for real-time event status updates.

    Streams status updates for the specified event until:
    - The event completes or fails
    - The client disconnects
    - A timeout occurs

    Message format:
    ```json
    {
        "type": "status_update",
        "event_id": "evt_xxx",
        "status": "processing",
        "progress_percent": 45.5,
        "current_stage": "Analysis",
        "message": "Processing SAR imagery...",
        "timestamp": "2024-01-15T12:30:00Z"
    }
    ```
    """
    # Check if event exists via DB lookup
    if db_session is not None:
        row = await db_session.get_event(event_id)
        if not row:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        event = deserialize_event_from_db(row)
    else:
        # No DB session available -- accept connection but cannot verify event
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await _connection_manager.connect(websocket, event_id)

    try:
        # Send initial status
        stage_index, stage_name = _get_current_stage(event.status)
        progress = _calculate_progress(event.status, stage_index)

        await websocket.send_json({
            "type": "status_update",
            "event_id": event_id,
            "status": event.status.value,
            "progress_percent": round(progress, 1),
            "current_stage": stage_name,
            "message": f"Connected - Current status: {event.status.value}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Keep connection alive and send updates
        # In production, this would subscribe to a message queue
        while True:
            try:
                # Wait for client messages (ping/pong, close, etc.)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # 30 second timeout for keepalive
                )

                # Handle client messages
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send heartbeat -- re-fetch event from DB
                row = await db_session.get_event(event_id)
                if row:
                    event = deserialize_event_from_db(row)
                    stage_index, stage_name = _get_current_stage(event.status)
                    progress = _calculate_progress(event.status, stage_index)

                    await websocket.send_json({
                        "type": "heartbeat",
                        "event_id": event_id,
                        "status": event.status.value,
                        "progress_percent": round(progress, 1),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                    # Close if event is terminal
                    if event.status in [
                        EventStatus.COMPLETED,
                        EventStatus.FAILED,
                        EventStatus.CANCELLED,
                    ]:
                        await websocket.send_json({
                            "type": "complete",
                            "event_id": event_id,
                            "status": event.status.value,
                            "message": "Event processing finished",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        break
                else:
                    # Event was deleted
                    await websocket.send_json({
                        "type": "error",
                        "event_id": event_id,
                        "message": "Event no longer exists",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected for event {event_id}")
    except Exception as e:
        logger.error(f"WebSocket error for event {event_id}: {e}")
    finally:
        _connection_manager.disconnect(websocket, event_id)
