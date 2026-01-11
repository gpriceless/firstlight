"""
Events API Routes.

Provides endpoints for submitting, listing, retrieving, and cancelling
event specifications for processing.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, Path, Query, status

from api.dependencies import (
    AgentRegistryDep,
    AuthDep,
    CommonDep,
    CorrelationIdDep,
    DBSessionDep,
    RateLimitDep,
    SchemaValidatorDep,
    SettingsDep,
)
from api.models.errors import (
    ConflictError,
    EventNotFoundError,
    SchemaValidationError,
    ValidationError,
)
from api.models.requests import (
    BoundingBox,
    EventPriority,
    EventQueryParams,
    EventSubmitRequest,
    SortOrder,
    EventSortField,
)
from api.models.responses import (
    EventResponse,
    EventStatus,
    EventSummary,
    PaginatedResponse,
    PaginationMeta,
    ResolvedIntent,
    SpatialInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Events"])


def generate_event_id() -> str:
    """Generate a unique event identifier."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    unique_id = uuid.uuid4().hex[:8]
    return f"evt_{timestamp}_{unique_id}"


def calculate_area_km2(bbox: BoundingBox) -> float:
    """
    Calculate approximate area of bounding box in square kilometers.

    Uses a simple spherical Earth approximation.
    """
    import math

    # Earth's radius in km
    R = 6371.0

    # Convert to radians
    lat1 = math.radians(bbox.south)
    lat2 = math.radians(bbox.north)
    lon1 = math.radians(bbox.west)
    lon2 = math.radians(bbox.east)

    # Calculate area using spherical approximation
    # A = R^2 * |sin(lat1) - sin(lat2)| * |lon1 - lon2|
    area = (
        R * R
        * abs(math.sin(lat1) - math.sin(lat2))
        * abs(lon1 - lon2)
    )

    return round(area, 2)


# In-memory event store (would be replaced with database in production)
_events_store: Dict[str, EventResponse] = {}


@router.post(
    "",
    response_model=EventResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit new event specification",
    description=(
        "Submit a new event specification for processing. "
        "The event will be validated against OpenSpec schemas and queued for processing."
    ),
    responses={
        201: {"description": "Event created successfully"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
    },
)
async def create_event(
    request: EventSubmitRequest,
    settings: SettingsDep,
    db_session: DBSessionDep,
    schema_validator: SchemaValidatorDep,
    agent_registry: AgentRegistryDep,
    auth: AuthDep,
    correlation_id: CorrelationIdDep,
    _rate_limit: RateLimitDep,
) -> EventResponse:
    """
    Submit a new event specification for processing.

    The event specification includes:
    - Intent: Event type (explicit class or natural language for inference)
    - Spatial: Area of interest (geometry or bounding box)
    - Temporal: Time window with optional reference time
    - Constraints: Data acquisition constraints (cloud cover, resolution, etc.)
    - Priority: Processing priority level
    - Metadata: Optional descriptive metadata

    The event will be:
    1. Validated against OpenSpec schemas
    2. Intent will be resolved (if natural language provided)
    3. Queued for processing by the orchestrator agent
    """
    # Generate unique event ID
    event_id = generate_event_id()

    # Build event data for schema validation
    event_data = {
        "id": event_id,
        "intent": {
            "class": request.intent.event_class or "pending_resolution",
            "source": "explicit" if request.intent.event_class else "inferred",
        },
        "spatial": {
            "type": "Polygon",
            "coordinates": [],  # Would be filled from geometry
            "crs": request.spatial.crs,
        },
        "temporal": {
            "start": request.temporal.start.isoformat(),
            "end": request.temporal.end.isoformat(),
        },
        "priority": request.priority.value,
    }

    if request.temporal.reference_time:
        event_data["temporal"]["reference_time"] = request.temporal.reference_time.isoformat()

    # Validate against OpenSpec event schema
    valid, errors = schema_validator.validate_event(event_data)
    if not valid:
        raise SchemaValidationError(
            message="Event specification failed schema validation",
            schema_name="event",
            errors=errors,
        )

    # Calculate bounding box from geometry if not provided
    if request.spatial.bbox:
        bbox = request.spatial.bbox
    elif request.spatial.geometry:
        # Extract bbox from geometry coordinates
        # This is a simplified version - would need proper GeoJSON parsing
        coords = request.spatial.geometry.coordinates
        if request.spatial.geometry.type.value == "Polygon":
            lons = [c[0] for c in coords[0]]
            lats = [c[1] for c in coords[0]]
            bbox = BoundingBox(
                west=min(lons),
                south=min(lats),
                east=max(lons),
                north=max(lats),
            )
        else:
            # Fallback for other geometry types
            bbox = BoundingBox(west=-180, south=-90, east=180, north=90)
    else:
        raise ValidationError(message="Either geometry or bbox must be provided")

    # Resolve intent if natural language provided
    resolved_intent = ResolvedIntent(
        event_class=request.intent.event_class or "pending_resolution",
        source="explicit" if request.intent.event_class else "inferred",
        confidence=1.0 if request.intent.event_class else None,
        original_input=request.intent.natural_language,
        parameters=request.intent.parameters,
    )

    # If natural language intent, would trigger NLP resolution here
    if not request.intent.event_class and request.intent.natural_language:
        # TODO: Call intent resolution service
        # For now, default to generic flood
        resolved_intent.event_class = "flood.riverine"
        resolved_intent.confidence = 0.75
        logger.info(
            f"Intent resolved from NL: '{request.intent.natural_language}' -> "
            f"'{resolved_intent.event_class}' (confidence: {resolved_intent.confidence})"
        )

    now = datetime.now(timezone.utc)

    # Create event response
    event = EventResponse(
        id=event_id,
        status=EventStatus.PENDING,
        priority=request.priority,
        intent=resolved_intent,
        spatial=SpatialInfo(
            geometry=request.spatial.geometry,
            bbox=bbox,
            crs=request.spatial.crs,
            area_km2=calculate_area_km2(bbox),
        ),
        temporal=request.temporal,
        constraints=request.constraints,
        created_at=now,
        updated_at=now,
        metadata={
            "name": request.metadata.name if request.metadata else None,
            "description": request.metadata.description if request.metadata else None,
            "tags": request.metadata.tags if request.metadata else None,
            "external_id": request.metadata.external_id if request.metadata else None,
            "callback_url": request.metadata.callback_url if request.metadata else None,
        } if request.metadata else None,
    )

    # Store event (in production, this would go to database)
    _events_store[event_id] = event

    # Queue for processing (would trigger orchestrator agent)
    logger.info(
        f"Event {event_id} created and queued for processing | "
        f"Class: {resolved_intent.event_class} | "
        f"Priority: {request.priority.value} | "
        f"Correlation: {correlation_id}"
    )

    return event


@router.get(
    "",
    response_model=PaginatedResponse[EventSummary],
    summary="List events",
    description="List all events with filtering, sorting, and pagination.",
)
async def list_events(
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
    # Filtering parameters
    status_filter: Annotated[
        Optional[List[str]],
        Query(alias="status", description="Filter by status"),
    ] = None,
    event_class: Annotated[
        Optional[str],
        Query(description="Filter by event class pattern"),
    ] = None,
    priority: Annotated[
        Optional[List[EventPriority]],
        Query(description="Filter by priority"),
    ] = None,
    tags: Annotated[
        Optional[List[str]],
        Query(description="Filter by tags"),
    ] = None,
    created_after: Annotated[
        Optional[datetime],
        Query(description="Filter events created after this time"),
    ] = None,
    created_before: Annotated[
        Optional[datetime],
        Query(description="Filter events created before this time"),
    ] = None,
    # Pagination
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Maximum number of results"),
    ] = 20,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of results to skip"),
    ] = 0,
    # Sorting
    sort_by: Annotated[
        EventSortField,
        Query(description="Field to sort by"),
    ] = EventSortField.CREATED_AT,
    sort_order: Annotated[
        SortOrder,
        Query(description="Sort order"),
    ] = SortOrder.DESC,
) -> PaginatedResponse[EventSummary]:
    """
    List events with optional filtering.

    Supports filtering by:
    - Status (pending, processing, completed, failed)
    - Event class (with wildcard support)
    - Priority level
    - Tags
    - Creation time range
    """
    # Get all events (in production, would query database)
    events = list(_events_store.values())

    # Apply filters
    if status_filter:
        events = [e for e in events if e.status.value in status_filter]

    if event_class:
        if event_class.endswith(".*"):
            prefix = event_class[:-2]
            events = [e for e in events if e.intent.event_class.startswith(prefix)]
        else:
            events = [e for e in events if e.intent.event_class == event_class]

    if priority:
        events = [e for e in events if e.priority in priority]

    if tags:
        events = [
            e for e in events
            if e.metadata and e.metadata.get("tags")
            and any(t in e.metadata["tags"] for t in tags)
        ]

    if created_after:
        events = [e for e in events if e.created_at >= created_after]

    if created_before:
        events = [e for e in events if e.created_at <= created_before]

    # Sort events
    reverse = sort_order == SortOrder.DESC
    if sort_by == EventSortField.CREATED_AT:
        events.sort(key=lambda e: e.created_at, reverse=reverse)
    elif sort_by == EventSortField.UPDATED_AT:
        events.sort(key=lambda e: e.updated_at, reverse=reverse)
    elif sort_by == EventSortField.PRIORITY:
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        events.sort(key=lambda e: priority_order.get(e.priority.value, 2), reverse=reverse)
    elif sort_by == EventSortField.STATUS:
        events.sort(key=lambda e: e.status.value, reverse=reverse)

    # Apply pagination
    total = len(events)
    events = events[offset : offset + limit]

    # Convert to summaries
    summaries = [
        EventSummary(
            id=e.id,
            status=e.status,
            priority=e.priority,
            event_class=e.intent.event_class,
            bbox=e.spatial.bbox,
            temporal=e.temporal,
            created_at=e.created_at,
            updated_at=e.updated_at,
            name=e.metadata.get("name") if e.metadata else None,
            tags=e.metadata.get("tags") if e.metadata else None,
            product_count=0,  # Would be calculated from products
        )
        for e in events
    ]

    return PaginatedResponse(
        items=summaries,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total,
        ),
    )


@router.get(
    "/{event_id}",
    response_model=EventResponse,
    summary="Get event details",
    description="Get full details for a specific event.",
    responses={
        200: {"description": "Event details"},
        404: {"description": "Event not found"},
    },
)
async def get_event(
    event_id: Annotated[str, Path(description="Event identifier")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> EventResponse:
    """
    Get full details for a specific event.

    Returns:
    - Event specification (intent, spatial, temporal, constraints)
    - Processing status and timestamps
    - Metadata and error information (if failed)
    """
    if event_id not in _events_store:
        raise EventNotFoundError(event_id)

    return _events_store[event_id]


@router.delete(
    "/{event_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Cancel event processing",
    description="Cancel an event that is pending or in progress.",
    responses={
        204: {"description": "Event cancelled successfully"},
        404: {"description": "Event not found"},
        409: {"description": "Event cannot be cancelled (already completed or failed)"},
    },
)
async def cancel_event(
    event_id: Annotated[str, Path(description="Event identifier")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    agent_registry: AgentRegistryDep,
    auth: AuthDep,
    correlation_id: CorrelationIdDep,
) -> None:
    """
    Cancel an event that is pending or in progress.

    Events that are already completed or failed cannot be cancelled.
    Cancelling an event will:
    - Stop any ongoing processing
    - Clean up intermediate data
    - Mark the event as cancelled
    """
    if event_id not in _events_store:
        raise EventNotFoundError(event_id)

    event = _events_store[event_id]

    # Check if event can be cancelled
    if event.status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED]:
        raise ConflictError(
            message=f"Event cannot be cancelled: status is {event.status.value}",
        )

    # Update event status
    event.status = EventStatus.CANCELLED
    event.updated_at = datetime.now(timezone.utc)

    # Would trigger cancellation in orchestrator agent
    logger.info(
        f"Event {event_id} cancelled | "
        f"Previous status: {event.status.value} | "
        f"Correlation: {correlation_id}"
    )

    # Delete from store (or keep with cancelled status, depending on requirements)
    # For now, keep the event but mark as cancelled
    _events_store[event_id] = event
