-- Migration 000: Add customer_id to api_keys table
-- This migration supports the transition from in-memory to persistent API key storage.
-- For existing deployments using the in-memory APIKeyStore, this migration prepares
-- the schema for when a persistent key store is introduced.
--
-- The customer_id field enables multi-tenant isolation: every API key belongs to
-- exactly one customer, and all downstream queries are scoped by customer_id.

-- Add customer_id column with NOT NULL constraint and backfill existing rows with "legacy"
ALTER TABLE IF EXISTS api_keys
    ADD COLUMN IF NOT EXISTS customer_id TEXT NOT NULL DEFAULT 'legacy';

-- Remove the default after backfill so future inserts must provide a customer_id explicitly
-- (Commented out: only run this after confirming all existing rows are backfilled)
-- ALTER TABLE api_keys ALTER COLUMN customer_id DROP DEFAULT;

-- Index for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_api_keys_customer_id ON api_keys (customer_id);

-- If the api_keys table does not yet exist (in-memory store is still in use),
-- create it with the customer_id column from the start.
CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    customer_id TEXT NOT NULL DEFAULT 'legacy',
    role TEXT NOT NULL DEFAULT 'user',
    permissions TEXT[] DEFAULT '{}',
    name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT true,
    rate_limit INTEGER,
    metadata JSONB DEFAULT '{}'
);

-- Ensure index exists on the new table as well
CREATE INDEX IF NOT EXISTS idx_api_keys_customer_id ON api_keys (customer_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys (user_id);
