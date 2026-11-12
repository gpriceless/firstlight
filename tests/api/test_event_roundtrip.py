"""An event written to the database reads back unchanged."""

from datetime import datetime, timezone

from api.models.requests import BoundingBox, EventPriority, Geometry, TemporalExtent
from api.models.responses import EventResponse, EventStatus, ResolvedIntent, SpatialInfo
from api.routes.events import deserialize_event_from_db, serialize_event_for_db


def test_event_survives_serialize_then_deserialize():
    now = datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc)
    event = EventResponse(
        id="evt_roundtrip",
        status=EventStatus.PENDING,
        priority=EventPriority.NORMAL,
        intent=ResolvedIntent(event_class="flood.riverine", source="explicit"),
        spatial=SpatialInfo(
            geometry=Geometry(
                type="Polygon",
                coordinates=[[[-80.5, 25.5], [-80.0, 25.5], [-80.0, 26.0], [-80.5, 25.5]]],
            ),
            bbox=BoundingBox(west=-80.5, south=25.5, east=-80.0, north=26.0),
            area_km2=2800.0,
        ),
        temporal=TemporalExtent(start=now, end=now),
        created_at=now,
        updated_at=now,
    )

    restored = deserialize_event_from_db(serialize_event_for_db(event))

    assert restored == event
