-- Migration 006: Initialize pgSTAC schema
--
-- pgSTAC uses its own 'pgstac' schema namespace, so it does not conflict
-- with FirstLight application tables in the 'public' schema.
--
-- This migration sets up the pgSTAC extension via pypgstac CLI. In Docker,
-- the init sequence runs: pypgstac migrate --dsn $DATABASE_URL
--
-- If pypgstac is not available, create minimal placeholder tables so the
-- application can start without the STAC layer.
--
-- Task 4.5

-- Ensure the pgstac schema exists (pypgstac create their own, but we
-- create it defensively)
CREATE SCHEMA IF NOT EXISTS pgstac;

-- Grant usage to the application user
GRANT USAGE ON SCHEMA pgstac TO CURRENT_USER;

-- Create placeholder tables if pypgstac hasn't been run yet.
-- pypgstac migrate will handle the real schema creation.
-- These are only created IF NOT EXISTS so they won't conflict with pypgstac.

CREATE TABLE IF NOT EXISTS pgstac.collections (
    id TEXT PRIMARY KEY,
    content JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pgstac.items (
    id TEXT NOT NULL,
    collection TEXT NOT NULL REFERENCES pgstac.collections(id),
    content JSONB NOT NULL,
    geometry GEOMETRY(GEOMETRY, 4326),
    datetime TIMESTAMPTZ,
    end_datetime TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (id, collection)
);

CREATE INDEX IF NOT EXISTS idx_pgstac_items_collection
    ON pgstac.items (collection);
CREATE INDEX IF NOT EXISTS idx_pgstac_items_datetime
    ON pgstac.items (datetime);
CREATE INDEX IF NOT EXISTS idx_pgstac_items_geometry
    ON pgstac.items USING GIST (geometry);

COMMENT ON SCHEMA pgstac IS 'pgSTAC schema for STAC collections and items. Managed by pypgstac migrate.';
