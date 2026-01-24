"""
Events API Routes.

Provides endpoints for submitting, listing, retrieving, and cancelling
event specifications for processing.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

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
from core.intent.resolver import IntentResolver
from core.intent.registry import get_registry
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


# Helper functions for database serialization

def serialize_event_for_db(event: EventResponse) -> Dict[str, Any]:
    """Convert EventResponse to database row format."""
    import json
    from typing import Any

    return {
        "id": event.id,
        "status": event.status.value,
        "priority": event.priority.value,
        "intent_json": json.dumps({
            "event_class": event.intent.event_class,
            "source": event.intent.source,
            "confidence": event.intent.confidence,
            "original_input": event.intent.original_input,
            "parameters": event.intent.parameters,
        }),
        "spatial_json": json.dumps({
            "geometry": event.spatial.geometry.model_dump() if event.spatial.geometry else None,
            "bbox": event.spatial.bbox.model_dump(),
            "crs": event.spatial.crs,
            "area_km2": event.spatial.area_km2,
        }),
        "temporal_json": json.dumps({
            "start": event.temporal.start.isoformat(),
            "end": event.temporal.end.isoformat(),
            "reference_time": event.temporal.reference_time.isoformat() if event.temporal.reference_time else None,
        }),
        "constraints_json": json.dumps(event.constraints.model_dump()) if event.constraints else None,
        "metadata_json": json.dumps(event.metadata) if event.metadata else None,
        "created_at": event.created_at.isoformat(),
        "updated_at": event.updated_at.isoformat(),
    }


def deserialize_event_from_db(row: Any) -> EventResponse:
    """Convert database row to EventResponse."""
    import json
    from typing import Any
    from api.models.requests import GeoJSONGeometry, GeometryType, TemporalWindow, DataConstraints

    intent_data = json.loads(row["intent_json"])
    spatial_data = json.loads(row["spatial_json"])
    temporal_data = json.loads(row["temporal_json"])
    constraints_data = json.loads(row["constraints_json"]) if row["constraints_json"] else None
    metadata_data = json.loads(row["metadata_json"]) if row["metadata_json"] else None

    return EventResponse(
        id=row["id"],
        status=EventStatus(row["status"]),
        priority=EventPriority(row["priority"]),
        intent=ResolvedIntent(
            event_class=intent_data["event_class"],
            source=intent_data["source"],
            confidence=intent_data.get("confidence"),
            original_input=intent_data.get("original_input"),
            parameters=intent_data.get("parameters"),
        ),
        spatial=SpatialInfo(
            geometry=GeoJSONGeometry(**spatial_data["geometry"]) if spatial_data.get("geometry") else None,
            bbox=BoundingBox(**spatial_data["bbox"]),
            crs=spatial_data["crs"],
            area_km2=spatial_data["area_km2"],
        ),
        temporal=TemporalWindow(
            start=datetime.fromisoformat(temporal_data["start"]),
            end=datetime.fromisoformat(temporal_data["end"]),
            reference_time=datetime.fromisoformat(temporal_data["reference_time"]) if temporal_data.get("reference_time") else None,
        ),
        constraints=DataConstraints(**constraints_data) if constraints_data else None,
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        metadata=metadata_data,
    )


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

    # If natural language intent, trigger NLP resolution
    if not request.intent.event_class and request.intent.natural_language:
        resolver = IntentResolver()

        try:
            resolution = resolver.resolve(
                natural_language=request.intent.natural_language,
                explicit_class=None,
                allow_override=False
            )

            if resolution and resolution.confidence > 0.0:
                resolved_intent.event_class = resolution.resolved_class
                resolved_intent.confidence = resolution.confidence

                # Merge extracted parameters
                if resolution.parameters:
                    if resolved_intent.parameters is None:
                        resolved_intent.parameters = {}
                    resolved_intent.parameters.update(resolution.parameters)

                logger.info(
                    f"Intent resolved from NL: '{request.intent.natural_language}' -> "
                    f"'{resolved_intent.event_class}' (confidence: {resolved_intent.confidence:.2f})"
                )
            else:
                # Resolution failed - reject with 400
                raise ValidationError(
                    message=f"Could not resolve intent from natural language: '{request.intent.natural_language}'. "
                    "Please provide an explicit event_class or rephrase your description."
                )
        except Exception as e:
            # NLP failure - log and provide helpful error
            logger.error(f"Intent resolution failed: {e}")
            raise ValidationError(
                message=f"Intent resolution failed: {str(e)}. "
                "Please provide an explicit event_class instead of natural language."
            )
    elif request.intent.event_class:
        # Validate explicit class against taxonomy
        registry = get_registry()
        event_class_obj = registry.get_class(request.intent.event_class)

        if not event_class_obj:
            available_classes = registry.list_all_classes()
            raise ValidationError(
                message=f"Invalid event class: '{request.intent.event_class}'. "
                f"Must be a valid class from the taxonomy. "
                f"Available classes include: {', '.join(available_classes[:10])}..."
            )

        logger.info(f"Using explicit event class: {request.intent.event_class}")

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

    # Store event in database
    await db_session.create_event(serialize_event_for_db(event))

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
    # Build database filters
    filters = {}

    if status_filter:
        filters["status"] = ",".join(status_filter)

    if priority:
        filters["priority"] = ",".join([p.value for p in priority])

    if created_after:
        filters["created_after"] = created_after.isoformat()

    if created_before:
        filters["created_before"] = created_before.isoformat()

    # Determine order field
    order_by_map = {
        EventSortField.CREATED_AT: "created_at",
        EventSortField.UPDATED_AT: "updated_at",
        EventSortField.PRIORITY: "priority",
        EventSortField.STATUS: "status",
    }
    order_by = order_by_map.get(sort_by, "created_at")
    order_desc = sort_order == SortOrder.DESC

    # Query database
    rows, total = await db_session.list_events(
        filters=filters,
        limit=limit,
        offset=offset,
        order_by=order_by,
        order_desc=order_desc
    )

    # Deserialize events
    events = [deserialize_event_from_db(row) for row in rows]

    # Apply in-memory filters for complex queries not supported by SQL
    if event_class:
        if event_class.endswith(".*"):
            prefix = event_class[:-2]
            events = [e for e in events if e.intent.event_class.startswith(prefix)]
        else:
            events = [e for e in events if e.intent.event_class == event_class]

    if tags:
        events = [
            e for e in events
            if e.metadata and e.metadata.get("tags")
            and any(t in e.metadata["tags"] for t in tags)
        ]

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
    row = await db_session.get_event(event_id)

    if not row:
        raise EventNotFoundError(event_id)

    return deserialize_event_from_db(row)


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
    row = await db_session.get_event(event_id)

    if not row:
        raise EventNotFoundError(event_id)

    event = deserialize_event_from_db(row)

    # Check if event can be cancelled
    if event.status in [EventStatus.COMPLETED, EventStatus.FAILED, EventStatus.CANCELLED]:
        raise ConflictError(
            message=f"Event cannot be cancelled: status is {event.status.value}",
        )

    # Update event status
    now = datetime.now(timezone.utc)
    await db_session.update_event(
        event_id,
        {
            "status": EventStatus.CANCELLED.value,
            "updated_at": now.isoformat(),
        }
    )

    # Would trigger cancellation in orchestrator agent
    logger.info(
        f"Event {event_id} cancelled | "
        f"Previous status: {event.status.value} | "
        f"Correlation: {correlation_id}"
    )
