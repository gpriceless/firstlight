## ADDED Requirements

### Requirement: Job State Read
The control API SHALL expose a GET endpoint at `/control/v1/jobs/{id}` that returns the full observable state of a job for LLM agent consumption, including the current phase, status, active parameters, and the complete reasoning history in chronological order.

The response MUST include:
- `job_id` — UUID of the job
- `phase` — coarse phase from the seven-value `JobPhase` enum (QUEUED, DISCOVERING, INGESTING, NORMALIZING, ANALYZING, REPORTING, COMPLETE)
- `status` — fine-grained per-phase substate (e.g., DISCOVERING, DISCOVERED, FAILED)
- `customer_id` — tenant identifier; requests MUST only return jobs belonging to the authenticated customer
- `parameters` — current effective algorithm parameters
- `reasoning_history` — ordered list of reasoning entries, each with `agent_id`, `confidence`, `content`, and `created_at`
- `created_at`, `updated_at` — ISO 8601 timestamps

#### Scenario: Agent reads job state mid-pipeline
- **WHEN** an authenticated LLM agent sends `GET /control/v1/jobs/{id}` for a job it owns
- **THEN** the API returns HTTP 200 with `phase`, `status`, `parameters`, and all prior reasoning entries

#### Scenario: Job not found or wrong tenant
- **WHEN** the job ID does not exist or belongs to a different `customer_id`
- **THEN** the API returns HTTP 404 with no information leakage about the foreign job

#### Scenario: Unauthenticated request
- **WHEN** no valid API key or bearer token is present
- **THEN** the API returns HTTP 401

---

### Requirement: State Transition
The control API SHALL expose a POST endpoint at `/control/v1/jobs/{id}/transition` that atomically advances a job from one phase/status to another, enforcing an expected-state guard to prevent races.

The request body MUST include:
- `expected_phase` — the phase the caller believes the job is in
- `expected_status` — the status the caller believes the job is in
- `target_phase` — the desired phase after transition
- `target_status` — the desired status after transition
- `reason` — human-readable string explaining the transition (recorded to reasoning history)

The transition MUST be executed as an atomic `UPDATE ... WHERE phase = :expected_phase AND status = :expected_status`. If the row does not match, the endpoint MUST return HTTP 409 with the current state so the caller can reconcile.

#### Scenario: Successful transition
- **WHEN** the job's current phase and status match `expected_phase` / `expected_status`
- **THEN** the job is atomically moved to `target_phase` / `target_status`, the `reason` is appended to the job event log, and HTTP 200 is returned with the new state

#### Scenario: Stale expected state (TOCTOU guard)
- **WHEN** another agent has already transitioned the job before this request arrives
- **THEN** the `UPDATE WHERE` clause matches zero rows and the API returns HTTP 409 with the current `phase` and `status`

#### Scenario: Invalid target state
- **WHEN** `target_phase` / `target_status` is not a legal value in the Phase+Status model
- **THEN** the API returns HTTP 422 before touching the database

---

### Requirement: Parameter Adjustment
The control API SHALL expose a PATCH endpoint at `/control/v1/jobs/{id}/parameters` that replaces one or more algorithm parameters on a running or paused job and records a before/after audit entry in the job event log.

The request body MUST include:
- `updates` — a map of parameter key to new value (partial update; unspecified keys are unchanged)
- `reason` — explanation for the change (stored in the audit entry)

The endpoint MUST:
- Validate each key against the algorithm's `parameter_schema` before writing
- Store `{before, after, reason, agent_id, timestamp}` in the `job_events` table
- Return the full updated parameter map in the response

#### Scenario: Agent adjusts a threshold parameter
- **WHEN** a valid PATCH request supplies `{"updates": {"flood_threshold": 0.45}, "reason": "Reducing sensitivity for dry season"}`
- **THEN** the parameter is updated, a before/after audit record is written, and HTTP 200 returns the complete updated parameters

#### Scenario: Parameter key not in schema
- **WHEN** the `updates` map contains a key not defined in the algorithm's `parameter_schema`
- **THEN** the API returns HTTP 422 listing the unknown keys; no changes are written

#### Scenario: Job in terminal state
- **WHEN** the job phase is COMPLETE or the status is FAILED and cannot accept changes
- **THEN** the API returns HTTP 409 with a message indicating the job is not mutable

---

### Requirement: Reasoning Injection
The control API SHALL expose a POST endpoint at `/control/v1/jobs/{id}/reasoning` that allows an LLM agent to append a reasoning entry to a job's history, including a confidence score and free-text content.

The request body MUST include:
- `agent_id` — identifier of the submitting agent
- `confidence` — float in `[0.0, 1.0]`
- `content` — free-text reasoning narrative (no max length enforced at API layer; storage layer truncates at 64KB)
- `phase` — the phase this reasoning applies to (defaults to current phase if omitted)

Reasoning entries are append-only. The endpoint MUST NOT allow deletion or modification of existing entries.

#### Scenario: Agent submits analysis reasoning
- **WHEN** a POST with valid `agent_id`, `confidence: 0.87`, and `content` is received
- **THEN** the entry is appended to the job's reasoning history, assigned a UUID and `created_at` timestamp, and HTTP 201 is returned with the new entry

#### Scenario: Invalid confidence value
- **WHEN** `confidence` is outside `[0.0, 1.0]` (e.g., `1.5`)
- **THEN** the API returns HTTP 422 without writing any entry

#### Scenario: Job does not exist
- **WHEN** the job ID resolves to no record or a record outside the caller's `customer_id`
- **THEN** the API returns HTTP 404

---

### Requirement: Escalation Management
The control API SHALL expose endpoints for creating and resolving escalations on jobs, enabling LLM agents to flag decision-critical situations that require human or supervisory review.

**Create:** `POST /control/v1/jobs/{id}/escalations`
- Body MUST include `severity` (one of: `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`), `reason`, and an optional `context` payload (arbitrary JSON, max 16KB)
- Returns HTTP 201 with the escalation ID and `created_at`

**Resolve:** `PATCH /control/v1/jobs/{id}/escalations/{escalation_id}`
- Body MUST include `resolution` — a free-text description of how the escalation was addressed
- Sets `resolved_at` and `resolved_by` (the authenticated agent or user identity)
- An escalation MUST NOT be resolvable twice; a second resolve attempt returns HTTP 409

#### Scenario: Agent raises a critical data quality escalation
- **WHEN** `POST /control/v1/jobs/{id}/escalations` is sent with `severity: "CRITICAL"` and a `reason`
- **THEN** the escalation is created, assigned an ID, and HTTP 201 is returned; a corresponding event is appended to `job_events` so the SSE stream notifies subscribers

#### Scenario: Operator resolves the escalation
- **WHEN** `PATCH /control/v1/jobs/{id}/escalations/{escalation_id}` is sent with a `resolution` string
- **THEN** the escalation record is marked resolved with `resolved_at` and `resolved_by`, and HTTP 200 is returned

#### Scenario: Duplicate resolve attempt
- **WHEN** a PATCH is sent for an escalation that is already resolved
- **THEN** the API returns HTTP 409 with the original resolution details

---

### Requirement: LLM Tool Schema Generation
The control API SHALL expose a GET endpoint at `/control/v1/tools` that returns auto-generated JSON tool schemas for all non-deprecated algorithms in the `AlgorithmRegistry`, formatted for direct consumption by an LLM agent's tool-use interface.

Each tool schema MUST include:
- `name` — the algorithm ID (e.g., `flood.baseline.threshold_sar`), normalized to a valid tool name
- `description` — the algorithm's `description` field from the registry
- `parameters` — derived from the algorithm's `parameter_schema`, with required parameters marked
- `event_types` — the list of event type patterns the algorithm handles
- `category` — baseline, advanced, or experimental

The schemas MUST be regenerated from the live registry state on each request (no static cache older than process startup). Deprecated algorithms MUST be omitted.

#### Scenario: Agent fetches available tools at session start
- **WHEN** `GET /control/v1/tools` is called by an authenticated agent
- **THEN** the response returns one tool schema per non-deprecated algorithm, in a JSON array under a `tools` key

#### Scenario: Registry has no algorithms loaded
- **WHEN** the AlgorithmRegistry is empty (e.g., definitions directory missing)
- **THEN** the endpoint returns HTTP 200 with an empty `tools` array and a `warning` field indicating the registry is empty

#### Scenario: New algorithm registered at runtime
- **WHEN** an algorithm is registered into the registry after server startup and a subsequent call to `/control/v1/tools` is made
- **THEN** the new algorithm's schema appears in the response

---

### Requirement: Job Listing with Filtering
The control API SHALL expose a GET endpoint at `/control/v1/jobs` that returns a paginated list of jobs visible to the authenticated customer, supporting server-side filtering to reduce the result set for agent queries.

Supported query parameters:
- `phase` — filter by `JobPhase` value (exact match)
- `status` — filter by `JobStatus` value (exact match)
- `customer_id` — available only to internal service accounts; regular agents are implicitly scoped to their own `customer_id`
- `event_type` — filter by event type string (exact or prefix match)
- `bbox` — spatial filter expressed as four comma-separated decimal values in WGS84: `west,south,east,north` (i.e., `minlon,minlat,maxlon,maxlat`). Matches jobs whose AOI bounding box overlaps the supplied envelope. Example: `bbox=-122.5,37.5,-121.5,38.5`. Values outside WGS84 bounds (longitude outside [-180, 180] or latitude outside [-90, 90]) MUST be rejected with HTTP 422.
- `page` — 1-based page number (default: 1)
- `page_size` — results per page (default: 20, max: 100)

The response MUST include `items`, `total`, `page`, and `page_size` fields. Each item in `items` MUST be a summary object (not the full job state) containing `job_id`, `phase`, `status`, `event_type`, `created_at`, and `updated_at`.

#### Scenario: Agent lists all in-progress jobs
- **WHEN** `GET /control/v1/jobs?phase=ANALYZING` is called
- **THEN** only jobs in the ANALYZING phase belonging to the caller's `customer_id` are returned, paginated

#### Scenario: Filter by event type
- **WHEN** `GET /control/v1/jobs?event_type=flood` is called
- **THEN** only jobs whose `event_type` starts with `flood` are returned

#### Scenario: page_size exceeds maximum
- **WHEN** `page_size=500` is supplied
- **THEN** the API returns HTTP 422 indicating the maximum is 100

---

### Requirement: Rate Limiting
The control API SHALL enforce rate limiting on all endpoints to prevent a single LLM agent or customer from degrading service for other tenants. Rate limits MUST be applied at two levels: per-agent (identified by API key) and per-customer (identified by `customer_id`). When a rate limit is exceeded, the API MUST return HTTP 429 with a `Retry-After` header indicating the number of seconds the caller should wait before retrying.

Rate limit configuration MUST be injectable via environment variables:
- `FIRSTLIGHT_RATE_LIMIT_PER_AGENT` — maximum requests per minute per API key (default: 120)
- `FIRSTLIGHT_RATE_LIMIT_PER_CUSTOMER` — maximum requests per minute per customer_id (default: 600)
- `FIRSTLIGHT_RATE_LIMIT_BURST` — maximum burst size above the sustained rate (default: 20)

Rate limit state MUST be stored in Redis using a sliding window counter. Rate limit headers MUST be included on every response: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.

#### Scenario: Agent exceeds per-key rate limit
- **WHEN** a single API key sends more than the configured per-agent limit within a one-minute window
- **THEN** subsequent requests from that key receive HTTP 429 with a `Retry-After` header indicating seconds until the window resets, and requests from other API keys under the same customer are unaffected

#### Scenario: Customer exceeds aggregate rate limit
- **WHEN** the combined requests from all API keys belonging to a single `customer_id` exceed the configured per-customer limit within a one-minute window
- **THEN** all API keys under that customer receive HTTP 429 with a `Retry-After` header, and requests from other customers are unaffected

#### Scenario: Burst allowance
- **WHEN** an agent sends a burst of requests that exceeds the sustained rate but stays within the burst allowance
- **THEN** all requests in the burst are accepted; the burst capacity refills at the sustained rate

#### Scenario: Rate limit headers on normal response
- **WHEN** a request is accepted within rate limits
- **THEN** the response includes `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers reflecting the caller's current usage against their per-agent limit
