"""
Database Schema Definitions.

Provides SQLite schema for events, executions, and products.
"""

# Event table schema
CREATE_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    priority TEXT NOT NULL,
    intent_json TEXT NOT NULL,
    spatial_json TEXT NOT NULL,
    temporal_json TEXT NOT NULL,
    constraints_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""

# Index for faster queries
CREATE_EVENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_priority ON events(priority);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
CREATE INDEX IF NOT EXISTS idx_events_updated_at ON events(updated_at);
"""

# Execution tracking table (for future use)
CREATE_EXECUTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS executions (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error_message TEXT,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
)
"""

# Products table (for future use)
CREATE_PRODUCTS_TABLE = """
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    product_type TEXT NOT NULL,
    format TEXT NOT NULL,
    storage_path TEXT,
    size_bytes INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
)
"""

# All initialization SQL
INIT_DATABASE_SQL = [
    CREATE_EVENTS_TABLE,
    CREATE_EVENTS_INDEXES,
    CREATE_EXECUTIONS_TABLE,
    CREATE_PRODUCTS_TABLE,
]
