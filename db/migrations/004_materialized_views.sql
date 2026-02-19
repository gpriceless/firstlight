-- Migration 004: Materialized views for pipeline health metrics and queue summary
--
-- These views are refreshed every 30 seconds by a background asyncio task
-- registered in api/main.py lifespan.
--
-- Depends on: 001_control_plane_schema.sql (jobs, job_events tables)

-- =============================================================================
-- Pipeline Health Metrics View
-- =============================================================================

-- Drop existing view if present (for idempotent re-runs)
DROP MATERIALIZED VIEW IF EXISTS pipeline_health_metrics;

CREATE MATERIALIZED VIEW pipeline_health_metrics AS
SELECT
    -- Completed/failed counts for 1h and 24h windows
    COUNT(*) FILTER (
        WHERE phase = 'COMPLETE' AND status = 'COMPLETE'
        AND updated_at >= now() - interval '1 hour'
    ) AS jobs_completed_1h,

    COUNT(*) FILTER (
        WHERE status = 'FAILED'
        AND updated_at >= now() - interval '1 hour'
    ) AS jobs_failed_1h,

    COUNT(*) FILTER (
        WHERE phase = 'COMPLETE' AND status = 'COMPLETE'
        AND updated_at >= now() - interval '24 hours'
    ) AS jobs_completed_24h,

    COUNT(*) FILTER (
        WHERE status = 'FAILED'
        AND updated_at >= now() - interval '24 hours'
    ) AS jobs_failed_24h,

    -- Duration percentiles (for completed jobs in last 24h)
    COALESCE(
        percentile_cont(0.5) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (updated_at - created_at))
        ) FILTER (
            WHERE phase = 'COMPLETE' AND status = 'COMPLETE'
            AND updated_at >= now() - interval '24 hours'
        ),
        0
    ) AS p50_duration_s,

    COALESCE(
        percentile_cont(0.95) WITHIN GROUP (
            ORDER BY EXTRACT(EPOCH FROM (updated_at - created_at))
        ) FILTER (
            WHERE phase = 'COMPLETE' AND status = 'COMPLETE'
            AND updated_at >= now() - interval '24 hours'
        ),
        0
    ) AS p95_duration_s,

    -- Webhook success rate (1h) — computed from job_events
    -- Using a subquery since we need the webhook_subscriptions table
    (
        SELECT CASE
            WHEN COUNT(*) = 0 THEN 1.0
            ELSE COUNT(*) FILTER (WHERE event_type = 'webhook.delivered')::float
                 / NULLIF(COUNT(*) FILTER (
                     WHERE event_type IN ('webhook.delivered', 'webhook.failed')
                 ), 0)
        END
        FROM job_events
        WHERE occurred_at >= now() - interval '1 hour'
          AND event_type IN ('webhook.delivered', 'webhook.failed')
    ) AS webhook_success_rate_1h,

    now() AS refreshed_at

FROM jobs;

-- UNIQUE index required for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX IF NOT EXISTS idx_pipeline_health_singleton
    ON pipeline_health_metrics (refreshed_at);


-- =============================================================================
-- Queue Summary View
-- =============================================================================

DROP MATERIALIZED VIEW IF EXISTS queue_summary;

CREATE MATERIALIZED VIEW queue_summary AS
WITH phase_counts AS (
    SELECT
        phase,
        COUNT(*) AS job_count
    FROM jobs
    WHERE status NOT IN ('COMPLETE', 'FAILED', 'CANCELLED')
    GROUP BY phase
),
stuck_jobs AS (
    -- Jobs not updated within the stuck threshold (default 30 min from app config)
    SELECT
        job_id,
        phase,
        status,
        updated_at
    FROM jobs
    WHERE status NOT IN ('COMPLETE', 'FAILED', 'CANCELLED')
      AND updated_at < now() - interval '30 minutes'
),
oldest_stuck AS (
    SELECT
        job_id,
        updated_at AS stuck_since
    FROM stuck_jobs
    ORDER BY updated_at ASC
    LIMIT 1
)
SELECT
    -- Per-phase counts as JSONB
    COALESCE(
        (SELECT jsonb_object_agg(phase, job_count) FROM phase_counts),
        '{}'::jsonb
    ) AS per_phase_counts,

    -- Stuck job count
    (SELECT COUNT(*) FROM stuck_jobs) AS stuck_count,

    -- Awaiting review count (jobs with open escalations)
    (
        SELECT COUNT(DISTINCT j.job_id)
        FROM jobs j
        JOIN escalations e ON j.job_id = e.job_id
        WHERE e.resolved_at IS NULL
          AND j.status NOT IN ('COMPLETE', 'FAILED', 'CANCELLED')
    ) AS awaiting_review_count,

    -- Oldest stuck job
    (SELECT job_id FROM oldest_stuck) AS oldest_stuck_job_id,
    (SELECT stuck_since FROM oldest_stuck) AS oldest_stuck_since,

    now() AS refreshed_at;

-- UNIQUE index for CONCURRENTLY refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_queue_summary_singleton
    ON queue_summary (refreshed_at);


-- Comments
COMMENT ON MATERIALIZED VIEW pipeline_health_metrics IS
    'Aggregated pipeline health metrics refreshed every 30 seconds';
COMMENT ON MATERIALIZED VIEW queue_summary IS
    'Queue status summary refreshed every 30 seconds';
