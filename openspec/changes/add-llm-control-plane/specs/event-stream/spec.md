## ADDED Requirements

### Requirement: SSE Event Stream Endpoint
The system SHALL expose a Server-Sent Events endpoint at `GET /internal/v1/events/stream` that pushes pipeline events to connected partners in real time.

The endpoint MUST accept optional query parameters `customer_id` (restrict events to one tenant) and `type` (comma-separated list of event type prefixes to filter, e.g. `job.*,llm.*`). Unauthenticated requests MUST be rejected with 401.

#### Scenario: Partner connects and receives live events
- **WHEN** a partner opens a persistent GET connection to `/internal/v1/events/stream`
- **THEN** the server holds the connection open, sends a `text/event-stream` content-type header, and pushes each new event as it is appended to the `job_events` table via PostgreSQL LISTEN/NOTIFY

#### Scenario: Customer-scoped connection
- **WHEN** a request includes a valid `customer_id` query parameter
- **THEN** only events belonging to that customer's jobs are emitted on the stream

#### Scenario: Type filtering
- **WHEN** a request includes `type=job.*,quality.*`
- **THEN** only events whose type matches one of the supplied prefixes are emitted; unmatched events are silently dropped at the server

#### Scenario: Unauthenticated request rejected
- **WHEN** a request arrives without a valid API key or bearer token
- **THEN** the server responds with HTTP 401 before establishing the stream

---

### Requirement: CloudEvents v1.0 Envelope
Every event emitted on the SSE stream and delivered via webhook MUST be wrapped in a CloudEvents v1.0 envelope.

Required attributes: `specversion` (fixed `"1.0"`), `type` (namespaced event type string), `source` (URI identifying the FirstLight instance), `id` (UUID v4, unique per event), `time` (RFC 3339 timestamp). The envelope MUST include the following FirstLight extension attributes: `firstlight_job_id`, `firstlight_customer_id`, `firstlight_phase`, and `firstlight_status`. Event-specific data MUST appear in the `data` field as a JSON object.

#### Scenario: Well-formed envelope on stream
- **WHEN** a `job.state_changed` event is appended to `job_events`
- **THEN** the SSE `data:` line contains a JSON object with `specversion`, `type`, `source`, `id`, `time`, `firstlight_job_id`, `firstlight_customer_id`, `firstlight_phase`, `firstlight_status`, and a `data` object containing the state transition details

#### Scenario: Extension attributes present
- **WHEN** any event is emitted (SSE or webhook)
- **THEN** `firstlight_job_id` and `firstlight_customer_id` are non-empty strings and `firstlight_phase`/`firstlight_status` reflect the job's state at the moment the event was written

---

### Requirement: Last-Event-ID Replay
The SSE endpoint MUST support reconnection replay using the standard `Last-Event-ID` HTTP header so that partners do not miss events during transient disconnects.

Each SSE event MUST carry an `id:` field set to the `job_events` row's sequential integer primary key. On reconnect, the server MUST query `job_events` WHERE id > Last-Event-ID and stream all missed events before resuming live LISTEN/NOTIFY delivery.

#### Scenario: Client reconnects after gap
- **WHEN** a partner reconnects with `Last-Event-ID: 4172`
- **THEN** the server immediately streams all events with id > 4172 from `job_events` in ascending order, then transitions to live delivery without duplicating any event

#### Scenario: No Last-Event-ID header
- **WHEN** a partner connects without a `Last-Event-ID` header
- **THEN** the server begins delivery with the next new event and does not replay historical events

#### Scenario: Last-Event-ID beyond known range
- **WHEN** a partner sends a `Last-Event-ID` value greater than the current maximum id in `job_events`
- **THEN** the server treats it as a fresh connection and begins live delivery from the next event

---

### Requirement: Event Types Catalog
The system MUST emit a defined set of structured event types. All type strings MUST follow the reverse-DNS namespacing convention `io.firstlight.<domain>.<verb>`.

**Job lifecycle events** (emitted by the pipeline orchestrator):
- `io.firstlight.job.created` — job record inserted, includes AOI and event type
- `io.firstlight.job.state_changed` — phase or status transition, includes previous and next values
- `io.firstlight.job.completed` — terminal success, includes result STAC item reference
- `io.firstlight.job.failed` — terminal failure, includes error category and message

**LLM decision events** (emitted by the control plane on agent action):
- `io.firstlight.llm.decision_made` — agent selected an algorithm or parameter; includes reasoning text and confidence
- `io.firstlight.llm.escalated` — agent could not resolve ambiguity and escalated to a human operator; includes the open question

**Quality events** (emitted by the quality stage):
- `io.firstlight.quality.check_passed` — quality gate passed, includes metric summary
- `io.firstlight.quality.check_failed` — quality gate failed, includes failing metric names and thresholds
- `io.firstlight.quality.manual_review_requested` — result flagged for human review

#### Scenario: Job progresses through pipeline
- **WHEN** a job moves from NORMALIZING to ANALYZING
- **THEN** a `io.firstlight.job.state_changed` event is written to `job_events` with `data.previous_phase = "NORMALIZING"` and `data.next_phase = "ANALYZING"`

#### Scenario: LLM agent makes a decision
- **WHEN** the LLM control plane selects an algorithm for a job
- **THEN** an `io.firstlight.llm.decision_made` event is emitted containing the algorithm identifier, the reasoning text, and a confidence score

#### Scenario: Quality gate fails
- **WHEN** the quality stage determines a result does not meet thresholds
- **THEN** an `io.firstlight.quality.check_failed` event is emitted listing each failing metric and its actual vs. expected value

---

### Requirement: Webhook Registration
Partners MUST be able to register webhook endpoints that receive event push delivery without maintaining a persistent SSE connection.

The system SHALL expose `POST /internal/v1/webhooks` accepting a JSON body with: `url` (HTTPS endpoint), `events` (array of event type prefixes to subscribe to, e.g. `["io.firstlight.job.*"]`), `secret` (string used for HMAC signing), and optionally `customer_id` to scope delivery to one tenant. The response MUST return a `webhook_id` and the registration's active status. Webhooks MUST be retrievable via `GET /internal/v1/webhooks` and deletable via `DELETE /internal/v1/webhooks/{webhook_id}`.

**Webhook listing pagination:** `GET /internal/v1/webhooks` MUST support cursor-based pagination using the following query parameters:
- `limit` — maximum number of subscriptions to return per page (default: 20, max: 100)
- `after` — opaque cursor value returned in the previous response's `next_cursor` field; omit to start from the beginning

The response MUST include:
- `items` — array of webhook subscription objects, each containing `webhook_id`, `url`, `events`, `active`, and `created_at`
- `next_cursor` — opaque string to pass as `after` on the next request; `null` when no further pages exist
- `has_more` — boolean indicating whether additional pages exist

Results MUST be ordered by `created_at` ascending (oldest first) and MUST be scoped to the authenticated customer's `customer_id`.

#### Scenario: Partner registers a webhook
- **WHEN** a POST to `/internal/v1/webhooks` is made with a valid HTTPS URL, event list, and secret
- **THEN** the server stores the registration, returns a 201 response with a `webhook_id`, and begins delivering matching events to that URL

#### Scenario: Non-HTTPS URL rejected
- **WHEN** a POST to `/internal/v1/webhooks` includes an HTTP (non-TLS) URL
- **THEN** the server responds with 422 and does not create the registration

#### Scenario: Prefix wildcard subscription
- **WHEN** a webhook is registered with `events: ["io.firstlight.job.*"]`
- **THEN** all job lifecycle events are delivered to that webhook; llm and quality events are not

#### Scenario: List webhooks with pagination
- **WHEN** a customer has 45 registered webhooks and calls `GET /internal/v1/webhooks?limit=20`
- **THEN** the response contains 20 items, `has_more: true`, and a non-null `next_cursor`; calling `GET /internal/v1/webhooks?limit=20&after=<next_cursor>` returns the next 20 items

#### Scenario: No webhooks registered
- **WHEN** no webhook subscriptions exist for the authenticated customer
- **WHEN** the customer sends `GET /internal/v1/webhooks`
- **THEN** the response is HTTP 200 with `items: []`, `has_more: false`, and `next_cursor: null`

#### Scenario: Tenant isolation on listing
- **WHEN** customer "acme" lists their webhooks
- **THEN** only webhooks registered by "acme" appear; webhooks belonging to other customers are never returned

---

### Requirement: Webhook Delivery with HMAC Signing and Retry
The system MUST deliver webhook payloads reliably using signed requests, exponential backoff retries, and a dead letter queue for undeliverable events.

Each outbound webhook request MUST include an `X-FirstLight-Signature-256` header containing `sha256=<HMAC-SHA256 hex digest>` computed over the raw request body using the registration secret. Delivery MUST be attempted up to 5 times with exponential backoff starting at 5 seconds and doubling with random jitter capped at 5 minutes per interval. If all retries are exhausted, the event MUST be written to the `webhook_dlq` PostgreSQL table with the failure reason and timestamp. Idempotency keys stored in Redis MUST prevent duplicate delivery within a 24-hour window.

#### Scenario: Successful delivery
- **WHEN** the partner's endpoint returns a 2xx response within 10 seconds
- **THEN** the delivery is marked successful and no retry is scheduled

#### Scenario: Endpoint temporarily unavailable
- **WHEN** the partner's endpoint returns a 5xx or times out
- **THEN** the system retries with exponential backoff (5s, 10s, 20s, 40s, 80s with jitter) up to 5 times before moving the event to the dead letter queue

#### Scenario: Signature verification
- **WHEN** a partner receives a webhook request
- **THEN** they can verify authenticity by computing HMAC-SHA256 of the raw body with their secret and comparing it to the `X-FirstLight-Signature-256` header value

#### Scenario: Dead letter queue capture
- **WHEN** all retry attempts for a webhook event are exhausted
- **THEN** a row is inserted into `webhook_dlq` containing the `webhook_id`, event `id`, failure reason, attempt count, and `failed_at` timestamp; the operator can inspect and replay from this table

---

### Requirement: Metrics Endpoint
The system SHALL expose `GET /internal/v1/metrics` returning a JSON summary of pipeline throughput and health, backed by a PostgreSQL materialized view refreshed every 30 seconds.

The response MUST include: jobs completed and failed in the last 1 hour and 24 hours, median and 95th-percentile end-to-end processing duration (in seconds) over the last 24 hours, active SSE connection count, and webhook delivery success rate over the last hour. The endpoint MUST respond within 100 ms under normal load because it reads from a materialized view rather than executing live aggregation queries.

#### Scenario: Operator polls for health
- **WHEN** a GET request is made to `/internal/v1/metrics`
- **THEN** the response contains `jobs_completed_1h`, `jobs_failed_1h`, `jobs_completed_24h`, `jobs_failed_24h`, `p50_duration_s`, `p95_duration_s`, `active_sse_connections`, and `webhook_success_rate_1h` as numeric fields

#### Scenario: Response latency
- **WHEN** the materialized view was last refreshed within the past 30 seconds
- **THEN** the endpoint responds in under 100 ms with a `Cache-Control: max-age=30` header and an `X-Metrics-Refreshed-At` timestamp

#### Scenario: View not yet populated
- **WHEN** the materialized view has never been refreshed (e.g., immediately after deployment)
- **THEN** the endpoint returns zeroes for all count and duration fields rather than an error

---

### Requirement: Queue Summary Endpoint
The system SHALL expose `GET /internal/v1/queue/summary` returning a real-time count of jobs in each processing state, including stuck and awaiting-review jobs requiring operator attention.

The response MUST include: count of in-flight jobs per phase, count of jobs that have not progressed in more than 30 minutes (stuck), count of jobs awaiting human review (escalated by LLM agent), and the oldest stuck job's `job_id` and `stuck_since` timestamp if any exist. The stuck threshold MUST be configurable via environment variable `FIRSTLIGHT_STUCK_JOB_THRESHOLD_MINUTES` with a default of 30.

#### Scenario: Healthy queue
- **WHEN** no jobs are stuck and no jobs await review
- **THEN** the response contains per-phase counts and `stuck_count: 0`, `awaiting_review_count: 0`, `oldest_stuck_job: null`

#### Scenario: Stuck job detected
- **WHEN** a job has been in the same phase for longer than the configured threshold
- **THEN** the response reflects `stuck_count >= 1` and `oldest_stuck_job` contains the relevant `job_id` and `stuck_since` timestamp

#### Scenario: LLM escalations surfaced
- **WHEN** one or more jobs have an active escalation (LLM agent could not decide)
- **THEN** `awaiting_review_count` reflects the number of open escalations so the operator knows action is required

---

### Requirement: Event Retention Policy
The `job_events` table SHALL be partitioned by month using PostgreSQL declarative partitioning (`PARTITION BY RANGE (occurred_at)`) to prevent unbounded growth and maintain query performance for SSE replay. Partitions older than the configured retention period MUST be automatically detached and archived.

The retention period SHALL default to 90 days and be configurable via the environment variable `FIRSTLIGHT_EVENT_RETENTION_DAYS`. A scheduled maintenance task (cron or pg_cron) MUST run daily to:
1. Create next month's partition if it does not exist (ensuring partition availability ahead of time)
2. Detach partitions older than the retention threshold
3. Archive detached partitions to cold storage as compressed pg_dump files
4. Drop archived partitions after successful archive verification

The `event_seq` BIGSERIAL primary key MUST remain globally unique across partitions (using a single shared sequence). SSE `Last-Event-ID` replay MUST work correctly across partition boundaries.

#### Scenario: Partition created automatically
- **WHEN** the maintenance task runs and next month's partition does not exist
- **THEN** a new partition is created covering the next calendar month's `occurred_at` range

#### Scenario: Old partition archived and dropped
- **WHEN** a partition contains only events older than the configured retention period
- **THEN** the partition is detached, compressed, archived to cold storage, and dropped from the live database

#### Scenario: Replay works within retention window
- **WHEN** a partner reconnects with a `Last-Event-ID` from 60 days ago (within 90-day retention)
- **THEN** replay succeeds and delivers all events with `event_seq > Last-Event-ID`

#### Scenario: Replay beyond retention window
- **WHEN** a partner reconnects with a `Last-Event-ID` from 120 days ago (beyond retention)
- **THEN** the server returns events starting from the oldest available partition; no error is raised, but the gap is indicated via a `X-Events-Gap: true` response header
