-- Migration 002: NOTIFY trigger on job_events INSERT
--
-- Sends a pg_notify on the 'job_events' channel whenever a new row is
-- inserted into the job_events table. The notification payload is the
-- JSON representation of the inserted row, enabling SSE endpoints and
-- webhook delivery workers to react in real time without polling.
--
-- Depends on: 001_control_plane_schema.sql (job_events table)

-- Create the trigger function
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

-- Attach the trigger to the job_events table
DROP TRIGGER IF EXISTS trg_notify_job_event ON job_events;
CREATE TRIGGER trg_notify_job_event
    AFTER INSERT ON job_events
    FOR EACH ROW
    EXECUTE FUNCTION notify_job_event();

-- Add a comment for documentation
COMMENT ON FUNCTION notify_job_event() IS
    'Sends pg_notify on job_events channel with JSON payload for each new event. '
    'Used by SSE streaming endpoint and webhook delivery workers.';
