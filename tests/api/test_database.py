"""
Tests for database persistence layer.

Verifies CRUD operations, connection management, and data persistence.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from api.database import DatabaseManager, DatabaseSession


@pytest.fixture
async def db_path():
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_file = Path(f.name)

    yield db_file

    # Cleanup
    if db_file.exists():
        db_file.unlink()


@pytest.fixture
async def db_manager(db_path):
    """Create database manager with temporary database."""
    manager = DatabaseManager(db_path)
    await manager.initialize()
    return manager


@pytest.fixture
async def db_session(db_manager):
    """Create database session."""
    session = DatabaseSession(db_manager)
    await session.connect()
    yield session
    await session.disconnect()


@pytest.mark.asyncio
async def test_database_initialization(db_path):
    """Test database is initialized with correct schema."""
    manager = DatabaseManager(db_path)
    await manager.initialize()

    # Verify database file exists
    assert db_path.exists()

    # Verify tables exist
    async with manager.connect() as conn:
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = await cursor.fetchall()
        table_names = [row["name"] for row in tables]

        assert "events" in table_names
        assert "executions" in table_names
        assert "products" in table_names


@pytest.mark.asyncio
async def test_create_and_get_event(db_session):
    """Test creating and retrieving an event."""
    event_id = "evt_20260123_abc123"
    now = datetime.now(timezone.utc)

    event_data = {
        "id": event_id,
        "status": "pending",
        "priority": "normal",
        "intent_json": json.dumps({
            "event_class": "flood.riverine",
            "source": "explicit",
            "confidence": 1.0,
        }),
        "spatial_json": json.dumps({
            "bbox": {"west": -122.5, "south": 37.5, "east": -122.0, "north": 38.0},
            "crs": "EPSG:4326",
            "area_km2": 100.5,
        }),
        "temporal_json": json.dumps({
            "start": "2026-01-20T00:00:00Z",
            "end": "2026-01-22T00:00:00Z",
        }),
        "constraints_json": None,
        "metadata_json": json.dumps({"name": "Test Event"}),
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    # Create event
    await db_session.create_event(event_data)

    # Retrieve event
    row = await db_session.get_event(event_id)

    assert row is not None
    assert row["id"] == event_id
    assert row["status"] == "pending"
    assert row["priority"] == "normal"

    intent = json.loads(row["intent_json"])
    assert intent["event_class"] == "flood.riverine"


@pytest.mark.asyncio
async def test_update_event(db_session):
    """Test updating event fields."""
    event_id = "evt_20260123_update"
    now = datetime.now(timezone.utc)

    # Create initial event
    await db_session.create_event({
        "id": event_id,
        "status": "pending",
        "priority": "normal",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    # Update status
    new_time = datetime.now(timezone.utc)
    await db_session.update_event(
        event_id,
        {
            "status": "processing",
            "updated_at": new_time.isoformat(),
        }
    )

    # Verify update
    row = await db_session.get_event(event_id)
    assert row["status"] == "processing"
    assert row["updated_at"] == new_time.isoformat()


@pytest.mark.asyncio
async def test_delete_event(db_session):
    """Test deleting an event."""
    event_id = "evt_20260123_delete"
    now = datetime.now(timezone.utc)

    # Create event
    await db_session.create_event({
        "id": event_id,
        "status": "pending",
        "priority": "normal",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    # Verify it exists
    row = await db_session.get_event(event_id)
    assert row is not None

    # Delete event
    await db_session.delete_event(event_id)

    # Verify it's gone
    row = await db_session.get_event(event_id)
    assert row is None


@pytest.mark.asyncio
async def test_list_events_pagination(db_session):
    """Test listing events with pagination."""
    now = datetime.now(timezone.utc)

    # Create multiple events
    for i in range(5):
        await db_session.create_event({
            "id": f"evt_test_{i}",
            "status": "pending",
            "priority": "normal",
            "intent_json": "{}",
            "spatial_json": "{}",
            "temporal_json": "{}",
            "constraints_json": None,
            "metadata_json": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        })

    # List first page
    rows, total = await db_session.list_events(limit=2, offset=0)
    assert len(rows) == 2
    assert total == 5

    # List second page
    rows, total = await db_session.list_events(limit=2, offset=2)
    assert len(rows) == 2
    assert total == 5


@pytest.mark.asyncio
async def test_list_events_filtering(db_session):
    """Test filtering events by status and priority."""
    now = datetime.now(timezone.utc)

    # Create events with different statuses
    await db_session.create_event({
        "id": "evt_pending_1",
        "status": "pending",
        "priority": "high",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    await db_session.create_event({
        "id": "evt_processing_1",
        "status": "processing",
        "priority": "normal",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    # Filter by status
    rows, total = await db_session.list_events(
        filters={"status": "pending"}
    )
    assert total == 1
    assert rows[0]["id"] == "evt_pending_1"

    # Filter by priority
    rows, total = await db_session.list_events(
        filters={"priority": "high"}
    )
    assert total == 1
    assert rows[0]["id"] == "evt_pending_1"


@pytest.mark.asyncio
async def test_persistence_across_sessions(db_manager):
    """Test that data persists across database sessions."""
    event_id = "evt_persist_test"
    now = datetime.now(timezone.utc)

    # Create event in first session
    session1 = DatabaseSession(db_manager)
    await session1.connect()
    await session1.create_event({
        "id": event_id,
        "status": "pending",
        "priority": "normal",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })
    await session1.disconnect()

    # Retrieve in second session
    session2 = DatabaseSession(db_manager)
    await session2.connect()
    row = await session2.get_event(event_id)
    await session2.disconnect()

    assert row is not None
    assert row["id"] == event_id


@pytest.mark.asyncio
async def test_transaction_rollback(db_session):
    """Test transaction rollback on error."""
    event_id = "evt_rollback_test"
    now = datetime.now(timezone.utc)

    # Create event
    await db_session.create_event({
        "id": event_id,
        "status": "pending",
        "priority": "normal",
        "intent_json": "{}",
        "spatial_json": "{}",
        "temporal_json": "{}",
        "constraints_json": None,
        "metadata_json": None,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    })

    # Try to create duplicate (should fail)
    with pytest.raises(Exception):
        await db_session.create_event({
            "id": event_id,  # Duplicate ID
            "status": "pending",
            "priority": "normal",
            "intent_json": "{}",
            "spatial_json": "{}",
            "temporal_json": "{}",
            "constraints_json": None,
            "metadata_json": None,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        })

    # Original event should still exist
    row = await db_session.get_event(event_id)
    assert row is not None
