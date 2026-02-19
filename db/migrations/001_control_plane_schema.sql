-- Control Plane Schema Migration
-- Creates the core tables for the LLM Control Plane state externalization.
--
-- Prerequisites:
--   - PostGIS extension must be installed (see db/init.sql)
--
-- Tables:
--   - jobs: Primary job state table with geospatial AOI
--   - job_events: Append-only event log for state transitions and reasoning
--   - job_checkpoints: Point-in-time state snapshots for recovery
--   - escalations: Human-in-the-loop escalation records

-- ============================================================================
-- Utility: updated_at trigger function
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- Table: jobs
-- ============================================================================
-- NOTE on generated columns:
--   ST_Envelope(aoi) is IMMUTABLE and works as a GENERATED ALWAYS AS column.
--   ST_Area(aoi::geography) involves a geography cast that is NOT marked
--   IMMUTABLE in PostGIS 3.4, so aoi_area_km2 uses a trigger instead.
-- ============================================================================

CREATE TABLE IF NOT EXISTS jobs (
    job_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    aoi             GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    aoi_area_km2    NUMERIC(12, 4),
    bbox            GEOMETRY(POLYGON, 4326) GENERATED ALWAYS AS (ST_Envelope(aoi)) STORED,
    phase           TEXT NOT NULL,
    status          TEXT NOT NULL,
    orchestrator_id TEXT,
    parameters      JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),

    -- Geometry validity constraints
    CONSTRAINT chk_aoi_valid CHECK (ST_IsValid(aoi)),
    CONSTRAINT chk_aoi_nonempty CHECK (NOT ST_IsEmpty(aoi)),
    -- Area bounds: 0.01 km2 to 5,000,000 km2
    -- Enforced via trigger since ST_Area(geography) is not immutable
    CONSTRAINT chk_phase_nonempty CHECK (phase <> ''),
    CONSTRAINT chk_status_nonempty CHECK (status <> '')
);

-- Spatial index on AOI for geospatial queries
CREATE INDEX IF NOT EXISTS idx_jobs_aoi_gist ON jobs USING GIST (aoi);

-- Index for listing by phase/status
CREATE INDEX IF NOT EXISTS idx_jobs_phase_status ON jobs (phase, status);

-- Index for tenant-scoped queries
CREATE INDEX IF NOT EXISTS idx_jobs_customer_id ON jobs (customer_id);

-- Index for ordering by creation time
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs (created_at DESC);

-- Trigger: auto-update updated_at
CREATE TRIGGER trg_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


-- ============================================================================
-- Trigger: maintain aoi_area_km2 column
-- ============================================================================
-- Uses ST_Area(aoi::geography) / 1e6 to compute area in km2.
-- Also enforces the area bounds constraint (0.01 to 5,000,000 km2).

CREATE OR REPLACE FUNCTION compute_aoi_area_km2()
RETURNS TRIGGER AS $$
DECLARE
    area_km2 NUMERIC(12, 4);
BEGIN
    area_km2 := ST_Area(NEW.aoi::geography) / 1000000.0;
    IF area_km2 < 0.01 OR area_km2 > 5000000 THEN
        RAISE EXCEPTION 'AOI area % km2 is outside allowed range [0.01, 5000000]', area_km2;
    END IF;
    NEW.aoi_area_km2 := area_km2;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_jobs_compute_area
    BEFORE INSERT OR UPDATE OF aoi ON jobs
    FOR EACH ROW
    EXECUTE FUNCTION compute_aoi_area_km2();


-- ============================================================================
-- Table: job_events
-- ============================================================================
-- Append-only event log for state transitions, reasoning entries, and errors.

CREATE TABLE IF NOT EXISTS job_events (
    event_seq   BIGSERIAL PRIMARY KEY,
    job_id      UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    customer_id TEXT NOT NULL,
    event_type  TEXT NOT NULL DEFAULT 'STATE_TRANSITION',
    phase       TEXT NOT NULL,
    status      TEXT NOT NULL,
    reasoning   TEXT,
    actor       TEXT NOT NULL,
    payload     JSONB DEFAULT '{}',
    occurred_at TIMESTAMPTZ DEFAULT now()
);

-- Composite index for per-job event queries (ordered by sequence)
CREATE INDEX IF NOT EXISTS idx_job_events_job_seq ON job_events (job_id, event_seq);

-- Index for event replay from a specific sequence number
CREATE INDEX IF NOT EXISTS idx_job_events_seq ON job_events (event_seq);

-- Index for customer-scoped event queries
CREATE INDEX IF NOT EXISTS idx_job_events_customer ON job_events (customer_id);


-- ============================================================================
-- Table: job_checkpoints
-- ============================================================================
-- Point-in-time state snapshots for recovery and debugging.

CREATE TABLE IF NOT EXISTS job_checkpoints (
    checkpoint_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id         UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    phase          TEXT NOT NULL,
    state_snapshot JSONB NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- Index for finding latest checkpoint per job
CREATE INDEX IF NOT EXISTS idx_job_checkpoints_job_time
    ON job_checkpoints (job_id, created_at DESC);


-- ============================================================================
-- Table: escalations
-- ============================================================================
-- Human-in-the-loop escalation records for jobs requiring manual review.

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id        UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    customer_id   TEXT NOT NULL,
    reason        TEXT NOT NULL,
    severity      TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    context       JSONB,
    created_at    TIMESTAMPTZ DEFAULT now(),
    resolved_at   TIMESTAMPTZ,
    resolution    TEXT,
    resolved_by   TEXT
);

-- Index for open escalation queries per customer (unresolved first)
CREATE INDEX IF NOT EXISTS idx_escalations_customer_open
    ON escalations (customer_id, resolved_at NULLS FIRST);

-- Index for per-job escalation lookups
CREATE INDEX IF NOT EXISTS idx_escalations_job_id ON escalations (job_id);
