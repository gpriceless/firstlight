-- Migration 003: Webhook subscriptions and dead letter queue tables
--
-- Supports the partner webhook delivery system (Phase 3).
-- webhook_subscriptions stores registered webhook endpoints with HMAC secrets.
-- webhook_dlq stores failed deliveries after all retry attempts are exhausted.
--
-- Depends on: 001_control_plane_schema.sql

-- Webhook subscription registrations
CREATE TABLE IF NOT EXISTS webhook_subscriptions (
    subscription_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id     TEXT NOT NULL,
    target_url      TEXT NOT NULL,
    secret_key      TEXT NOT NULL,
    event_filter    TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now(),
    active          BOOLEAN DEFAULT true
);

-- Index for efficient lookups by customer
CREATE INDEX IF NOT EXISTS idx_webhook_subs_customer
    ON webhook_subscriptions (customer_id)
    WHERE active = true;

-- Index for cursor-based pagination
CREATE INDEX IF NOT EXISTS idx_webhook_subs_created
    ON webhook_subscriptions (created_at, subscription_id);

-- Dead letter queue for failed webhook deliveries
CREATE TABLE IF NOT EXISTS webhook_dlq (
    dlq_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subscription_id UUID NOT NULL REFERENCES webhook_subscriptions(subscription_id),
    event_seq       BIGINT NOT NULL,
    payload         JSONB NOT NULL,
    last_error      TEXT,
    attempt_count   INT DEFAULT 0,
    failed_at       TIMESTAMPTZ DEFAULT now()
);

-- Index for DLQ lookups by subscription
CREATE INDEX IF NOT EXISTS idx_webhook_dlq_subscription
    ON webhook_dlq (subscription_id, failed_at DESC);

-- Comments
COMMENT ON TABLE webhook_subscriptions IS
    'Registered webhook endpoints for partner event delivery';
COMMENT ON TABLE webhook_dlq IS
    'Dead letter queue for webhook deliveries that failed after all retry attempts';
