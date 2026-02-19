-- Migration 005: Convert job_events to declarative range partitioning
--
-- Partitions job_events by occurred_at (monthly). Uses a shared sequence
-- for event_seq to ensure globally unique sequence numbers across partitions.
-- Last-Event-ID replay works across partition boundaries because event_seq
-- is ordered by the shared sequence.
--
-- Strategy:
-- 1. Create the shared sequence
-- 2. Rename original table
-- 3. Create new partitioned table
-- 4. Migrate data
-- 5. Swap names
-- 6. Create initial monthly partitions
--
-- Depends on: 001_control_plane_schema.sql, 002_job_events_notify.sql

-- Step 1: Create a shared sequence for event_seq across all partitions
CREATE SEQUENCE IF NOT EXISTS job_events_event_seq_shared
    AS BIGINT
    START WITH 1
    INCREMENT BY 1
    NO CYCLE;

-- Advance the sequence past any existing event_seq values
DO $$
DECLARE
    max_seq BIGINT;
BEGIN
    SELECT COALESCE(MAX(event_seq), 0) INTO max_seq FROM job_events;
    IF max_seq > 0 THEN
        PERFORM setval('job_events_event_seq_shared', max_seq);
    END IF;
END $$;

-- Step 2: Rename the original table
ALTER TABLE IF EXISTS job_events RENAME TO job_events_old;

-- Drop the trigger on the old table (will be recreated on the new one)
DROP TRIGGER IF EXISTS trg_notify_job_event ON job_events_old;

-- Step 3: Create the new partitioned table
CREATE TABLE job_events (
    event_seq   BIGINT NOT NULL DEFAULT nextval('job_events_event_seq_shared'),
    job_id      UUID NOT NULL,
    customer_id TEXT NOT NULL,
    event_type  TEXT NOT NULL DEFAULT 'STATE_TRANSITION',
    phase       TEXT NOT NULL,
    status      TEXT NOT NULL,
    reasoning   TEXT,
    actor       TEXT NOT NULL,
    payload     JSONB DEFAULT '{}',
    occurred_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (occurred_at, event_seq)
) PARTITION BY RANGE (occurred_at);

-- Note: Foreign key to jobs(job_id) cannot be on a partitioned table in PG15.
-- Enforcement is done at the application level.

-- Index for Last-Event-ID replay (works across partitions via event_seq ordering)
CREATE INDEX idx_job_events_seq ON job_events (event_seq);

-- Index for job-scoped queries
CREATE INDEX idx_job_events_job_id ON job_events (job_id, event_seq);

-- Step 4: Create initial monthly partitions
-- Current month and next 3 months
DO $$
DECLARE
    start_date DATE;
    end_date DATE;
    partition_name TEXT;
    i INT;
BEGIN
    FOR i IN -1..3 LOOP
        start_date := date_trunc('month', current_date + (i || ' months')::interval)::date;
        end_date := (start_date + interval '1 month')::date;
        partition_name := 'job_events_' || to_char(start_date, 'YYYY_MM');

        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS %I PARTITION OF job_events
             FOR VALUES FROM (%L) TO (%L)',
            partition_name, start_date, end_date
        );
    END LOOP;
END $$;

-- Also create a default partition for any data outside the defined ranges
CREATE TABLE IF NOT EXISTS job_events_default PARTITION OF job_events DEFAULT;

-- Step 5: Migrate data from old table
INSERT INTO job_events (
    event_seq, job_id, customer_id, event_type, phase, status,
    reasoning, actor, payload, occurred_at
)
SELECT
    event_seq, job_id, customer_id, event_type, phase, status,
    reasoning, actor, payload, occurred_at
FROM job_events_old
ORDER BY event_seq;

-- Step 6: Drop the old table
DROP TABLE IF EXISTS job_events_old;

-- Step 7: Recreate the NOTIFY trigger on the new partitioned table
CREATE OR REPLACE FUNCTION notify_job_event()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'job_events',
        json_build_object(
            'event_seq', NEW.event_seq,
            'job_id', NEW.job_id,
            'customer_id', NEW.customer_id,
            'event_type', NEW.event_type,
            'phase', NEW.phase,
            'status', NEW.status,
            'reasoning', NEW.reasoning,
            'actor', NEW.actor,
            'payload', NEW.payload,
            'occurred_at', NEW.occurred_at
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_notify_job_event
    AFTER INSERT ON job_events
    FOR EACH ROW
    EXECUTE FUNCTION notify_job_event();

-- Comments
COMMENT ON TABLE job_events IS
    'Partitioned job events table (by occurred_at, monthly partitions). '
    'Uses shared sequence job_events_event_seq_shared for globally ordered event_seq.';
