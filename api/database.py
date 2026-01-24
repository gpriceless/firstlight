"""
Database Connection and Session Management.

Provides async SQLite database operations using aiosqlite.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from api.models.database import INIT_DATABASE_SQL

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Async database manager with connection pooling.

    Handles database initialization, migrations, and connection lifecycle.
    """

    def __init__(self, db_path: Path):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema if not exists."""
        if self._initialized:
            return

        # Ensure data directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Run initialization SQL
        async with aiosqlite.connect(self.db_path) as db:
            # Enable foreign keys
            await db.execute("PRAGMA foreign_keys = ON")

            # Create tables and indexes
            for sql in INIT_DATABASE_SQL:
                await db.executescript(sql)

            await db.commit()

        self._initialized = True
        logger.info(f"Database initialized at {self.db_path}")

    async def connect(self) -> aiosqlite.Connection:
        """
        Create a new database connection.

        Returns:
            Async database connection
        """
        await self.initialize()
        conn = await aiosqlite.connect(self.db_path)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA foreign_keys = ON")
        return conn


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(db_path: Optional[Path] = None) -> DatabaseManager:
    """
    Get or create global database manager instance.

    Args:
        db_path: Path to database file (only used on first call)

    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        if db_path is None:
            db_path = Path("data/firstlight.db")
        _db_manager = DatabaseManager(db_path)
    return _db_manager


class DatabaseSession:
    """
    Async database session with transaction support.

    Provides high-level database operations for events.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize database session.

        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self._connection: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Establish database connection."""
        if self._connection is None:
            self._connection = await self.db_manager.connect()
            logger.debug("Database session connected")

    async def disconnect(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.debug("Database session disconnected")

    async def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> aiosqlite.Cursor:
        """
        Execute a database query.

        Args:
            query: SQL query with named parameters (:param_name)
            params: Dictionary of parameter values

        Returns:
            Database cursor
        """
        if not self._connection:
            raise RuntimeError("Database session not connected")

        cursor = await self._connection.execute(query, params or {})
        return cursor

    async def fetchone(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[aiosqlite.Row]:
        """
        Execute query and fetch one row.

        Args:
            query: SQL query with named parameters
            params: Dictionary of parameter values

        Returns:
            Single row or None
        """
        cursor = await self.execute(query, params)
        return await cursor.fetchone()

    async def fetchall(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[aiosqlite.Row]:
        """
        Execute query and fetch all rows.

        Args:
            query: SQL query with named parameters
            params: Dictionary of parameter values

        Returns:
            List of rows
        """
        cursor = await self.execute(query, params)
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit current transaction."""
        if not self._connection:
            raise RuntimeError("Database session not connected")
        await self._connection.commit()

    async def rollback(self) -> None:
        """Rollback current transaction."""
        if not self._connection:
            raise RuntimeError("Database session not connected")
        await self._connection.rollback()

    # Event-specific operations

    async def create_event(self, event_data: Dict[str, Any]) -> None:
        """
        Insert a new event into database.

        Args:
            event_data: Event data dictionary
        """
        await self.execute(
            """
            INSERT INTO events (
                id, status, priority, intent_json, spatial_json,
                temporal_json, constraints_json, metadata_json,
                created_at, updated_at
            ) VALUES (
                :id, :status, :priority, :intent_json, :spatial_json,
                :temporal_json, :constraints_json, :metadata_json,
                :created_at, :updated_at
            )
            """,
            event_data
        )
        await self.commit()

    async def get_event(self, event_id: str) -> Optional[aiosqlite.Row]:
        """
        Get event by ID.

        Args:
            event_id: Event identifier

        Returns:
            Event row or None if not found
        """
        return await self.fetchone(
            "SELECT * FROM events WHERE id = :id",
            {"id": event_id}
        )

    async def update_event(
        self,
        event_id: str,
        updates: Dict[str, Any]
    ) -> None:
        """
        Update event fields.

        Args:
            event_id: Event identifier
            updates: Dictionary of field updates
        """
        # Build SET clause from updates
        set_clauses = [f"{key} = :{key}" for key in updates.keys()]
        query = f"UPDATE events SET {', '.join(set_clauses)} WHERE id = :id"

        params = {**updates, "id": event_id}
        await self.execute(query, params)
        await self.commit()

    async def delete_event(self, event_id: str) -> None:
        """
        Delete event from database.

        Args:
            event_id: Event identifier
        """
        await self.execute(
            "DELETE FROM events WHERE id = :id",
            {"id": event_id}
        )
        await self.commit()

    async def list_events(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True
    ) -> tuple[List[aiosqlite.Row], int]:
        """
        List events with filtering and pagination.

        Args:
            filters: Optional filter conditions
            limit: Maximum results to return
            offset: Number of results to skip
            order_by: Field to sort by
            order_desc: Sort descending if True

        Returns:
            Tuple of (rows, total_count)
        """
        # Build WHERE clause from filters
        where_clauses = []
        params = {}

        if filters:
            if "status" in filters:
                # Split comma-separated values and create placeholders
                status_values = filters["status"].split(",")
                placeholders = ",".join([f":status{i}" for i in range(len(status_values))])
                where_clauses.append(f"status IN ({placeholders})")
                for i, val in enumerate(status_values):
                    params[f"status{i}"] = val

            if "priority" in filters:
                # Split comma-separated values and create placeholders
                priority_values = filters["priority"].split(",")
                placeholders = ",".join([f":priority{i}" for i in range(len(priority_values))])
                where_clauses.append(f"priority IN ({placeholders})")
                for i, val in enumerate(priority_values):
                    params[f"priority{i}"] = val

            if "created_after" in filters:
                where_clauses.append("created_at >= :created_after")
                params["created_after"] = filters["created_after"]

            if "created_before" in filters:
                where_clauses.append("created_at <= :created_before")
                params["created_before"] = filters["created_before"]

        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        order_clause = f"ORDER BY {order_by} {'DESC' if order_desc else 'ASC'}"

        # Get total count
        count_query = f"SELECT COUNT(*) as count FROM events {where_clause}"
        count_row = await self.fetchone(count_query, params)
        total = count_row["count"] if count_row else 0

        # Get rows
        query = f"""
            SELECT * FROM events
            {where_clause}
            {order_clause}
            LIMIT :limit OFFSET :offset
        """
        params.update({"limit": limit, "offset": offset})
        rows = await self.fetchall(query, params)

        return rows, total
