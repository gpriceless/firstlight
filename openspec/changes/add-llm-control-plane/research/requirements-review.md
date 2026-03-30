# Requirements Review: add-llm-control-plane

## Verdict: READY WITH NOTES

## Summary

The LLM Control Plane specification is comprehensive, well-structured, and shows strong engineering maturity across 7 spec files covering 30+ distinct requirements. The Phase+Status state model, StateBackend ABC, four API surfaces, and CloudEvents-based event stream are clearly specified with testable scenarios. Three issues must be resolved before engineering handoff: (1) the StateBackend ABC method signatures in the state-management spec do not match the existing StateManager public interface as tasks.md requires, (2) the Taskiq broker is used in Phase 3 (task 3.6, webhook delivery) but formally introduced in Phase 3 task 3.0 -- the value chain's dependency inversion concern (G2) has been partially addressed by moving installation to task 3.0, but task 4.8 still references "verify Taskiq broker created in 3.0" which should be a Phase 3 task, and (3) six open questions in design.md remain unresolved and several directly affect schema design and API behavior.

---

## Completeness Findings

### C1: StateBackend ABC Signature Mismatch with Existing StateManager [MUST FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/state-management/spec.md` (line 26)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 1.1)

The state-management spec declares the StateBackend ABC with methods: `get_state`, `set_state`, `transition`, `list_jobs`, `checkpoint`. Task 1.1 says "the signature must match the existing StateManager public interface exactly." But the existing `StateManager` in `/home/gprice/projects/firstlight/agents/orchestrator/state.py` has these public methods:
- `create_state(event_id, event_spec, orchestrator_id)` -- NOT in the spec ABC
- `get_state(event_id)` -- present
- `update_state(state: ExecutionState)` -- spec calls it `set_state(job_id, phase, status)` with different signature
- `update_stage(event_id, stage, status, progress_percent, message, metrics)` -- NOT in the spec ABC
- `checkpoint(state: ExecutionState)` -- spec calls it `checkpoint(job_id, payload)` with different signature
- `restore_checkpoint(event_id)` -- NOT in the spec ABC
- `record_error(event_id, stage, error, stack_trace)` -- task 1.1 includes this, spec ABC does not
- `set_degraded_mode(event_id, level, reason, fallback, confidence_impact)` -- NOT in spec ABC
- `list_active_executions()` -- spec calls it `list_jobs(filters)` with different signature
- `get_stage_history(event_id, stage)` -- NOT in spec ABC
- `get_error_history(event_id)` -- NOT in spec ABC

The spec ABC is a new simplified interface, not a 1:1 match. Task 1.1's claim that "the signature must match the existing StateManager public interface exactly" contradicts the spec. Either the task or the spec needs to be updated. Recommendation: Update task 1.1 to say the ABC is a new interface and the SQLiteStateBackend adapter (task 1.2) adapts the existing StateManager to this new interface. The ABC should explicitly list all methods it includes vs. excludes.

### C2: Checkpoint Resume Contract Unspecified [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/state-management/spec.md` (lines 84-100)

The "Job Checkpoint and Resume" requirement says "the orchestrator MUST read the latest checkpoint for that job and pass it to the pipeline stage as initial context." But there is no specification of:
- What the checkpoint payload structure looks like for each pipeline stage
- How each pipeline stage determines what to skip vs. re-run from the checkpoint
- Whether the existing `StageProgress.metrics` data is included in the checkpoint payload

The existing `StateManager.checkpoint()` stores the entire `ExecutionState.to_dict()` as the checkpoint payload (line 664 of state.py). The new spec says `checkpoint(job_id, payload)` takes an arbitrary JSON payload. These are different approaches. The value chain analysis (G3) also flagged this.

### C3: No Specification for Job Deletion/Cancellation via Control API [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/control-api/spec.md`

The control API spec covers CREATE (POST /jobs), READ (GET /jobs, GET /jobs/{id}), UPDATE (transitions, parameters), but no DELETE or CANCEL endpoint. Jobs can reach FAILED or CANCELLED status per the state model, but there is no API endpoint for an LLM agent to cancel a running job. The OGC spec (ogc-integration/spec.md line 151) includes `DELETE /oapi/jobs/{id}` for job dismissal, but the control API does not.

### C4: Missing Error Response Body Format Specification [SHOULD FIX]

**All spec files**

Multiple specs reference HTTP error codes (401, 403, 404, 409, 422, 429) but none specify a consistent error response body format. Two engineers could implement different error shapes. Recommendation: Add a shared error envelope to the design document:

```json
{
  "error": {
    "code": "CONFLICT",
    "message": "Job is not in expected state",
    "details": { "current_phase": "ANALYZING", "current_status": "ANALYZING" }
  }
}
```

### C5: No Specification for Webhook Subscription Listing [NICE TO HAVE]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/event-stream/spec.md` (line 95)

The webhook registration requirement mentions `GET /internal/v1/webhooks` is available for retrieval, but there is no scenario specifying pagination, filtering, or the response shape for listing webhooks.

### C6: Health/Readiness Probe Paths Not Enumerated [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/security-hardening/spec.md` (line 40)

The security spec says "a request targets a designated liveness or readiness probe path" is exempt from auth, but does not enumerate which paths. The OGC auth boundary (ogc-integration/spec.md lines 140-153) enumerates its exempt paths precisely. The health probe exemption should do the same (e.g., `GET /health`, `GET /ready`).

### C7: Metrics Materialized View Does Not Match Task 3.8 [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/event-stream/spec.md` (lines 134-150)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 3.8)

The event-stream spec's Metrics Endpoint requirement specifies: `jobs_completed_1h`, `jobs_failed_1h`, `p50_duration_s`, `p95_duration_s`, `active_sse_connections`, `webhook_success_rate_1h`. Task 3.8 creates a materialized view with: `job_id`, `customer_id`, `phase`, `status`, `age_minutes`, `checkpoint_count`. These are entirely different schemas -- the task describes a queue view while the spec describes a health/throughput metrics view. The endpoint path also differs: spec says `GET /internal/v1/metrics`, task says `GET /internal/v1/metrics/queue`.

### C8: Event Retention Policy Not Specified [SHOULD FIX]

**Design file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/design.md` (open question 5)

Open question 5 asks "How long should job_events be retained?" This is still listed as a question, but the `job_events` table is the replay buffer for SSE. If it grows unbounded, replay queries degrade. At minimum, the spec should state a default retention period or partition strategy, even if configurable.

---

## Clarity Findings

### CL1: Webhook Retry Backoff Inconsistency Between Spec and Tasks [MUST FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/event-stream/spec.md` (line 114)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 3.6)

The spec says: "exponential backoff starting at 5 seconds and doubling with random jitter capped at 5 minutes per interval." The schedule would be: 5s, 10s, 20s, 40s, 80s (with jitter).

Task 3.6 says: "exponential backoff 2^n seconds with random.uniform(0, 1) jitter." The schedule would be: 1s, 2s, 4s, 8s, 16s.

These are different retry schedules. The spec's 5s base is more conservative. Pick one.

### CL2: Control API Transition Endpoint Path Inconsistency [MUST FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/control-api/spec.md` (line 30)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 2.5)

The spec defines the transition endpoint as: `POST /control/v1/jobs/{id}/transition` (singular).
Task 2.5 says: `POST /control/v1/jobs/{job_id}/transitions` (plural).

The request body also differs:
- Spec: `expected_phase`, `expected_status`, `target_phase`, `target_status`, `reason`
- Task: `target_phase`, `target_status`, `reasoning`, `actor` (no expected_phase/expected_status)

The spec is more correct here (the expected-state guard is the whole point of TOCTOU protection). The task description should match the spec.

### CL3: Escalation Endpoint Path Inconsistency [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/control-api/spec.md` (lines 109-128)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 2.8)

The spec nests escalations under a job: `POST /control/v1/jobs/{id}/escalations` and `PATCH /control/v1/jobs/{id}/escalations/{escalation_id}`. Task 2.8 uses a flat namespace: `POST /control/v1/escalations` and `PATCH /control/v1/escalations/{escalation_id}/resolve`. These are different API designs. The spec's job-scoped approach is more RESTful.

### CL4: `aoi_area_km2` Column Type Inconsistency [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/postgis-schema/spec.md` (line 56)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 1.4)

The postgis-schema spec says `aoi_area_km2` is `NUMERIC(12, 4)`. Task 1.4 says `aoi_area_km2 FLOAT`. These are different types -- NUMERIC(12,4) is exact precision, FLOAT is approximate. The spec's NUMERIC(12,4) is the better choice for a generated column. Update the task.

### CL5: Ambiguous `bbox` Query Parameter Format [SHOULD FIX]

**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 2.2)

Task 2.2 says the jobs list endpoint supports `bbox` as "(WKT or GeoJSON)". This is ambiguous -- should the API accept a WKT string as a query parameter, or a GeoJSON object, or both? Standard practice (OGC/STAC) is `bbox=minlon,minlat,maxlon,maxlat` as comma-separated values in the query string. Clarify the expected format.

### CL6: Per-Phase Substatus Values Not Enumerated [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/state-management/spec.md`
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md` (task 1.5)

The spec says `JobStatus` carries per-phase substates like `DISCOVERED`, `DISCOVERY_FAILED`. Task 1.5 says to add "per-phase substatus strings (e.g. DISCOVERING/DISCOVERED, DISCOVERING/DISCOVERY_FAILED) matching the state-management spec." But the spec never enumerates the full list of valid substatus values per phase. Engineers need to know the complete set. Without this, two engineers will invent different status strings for INGESTING, NORMALIZING, ANALYZING, REPORTING, etc.

---

## Testability Findings

### T1: SSE Backpressure Test Is Hard to Automate [NICE TO HAVE]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/event-stream/spec.md` (task 3.4)

The backpressure scenario ("slow consumer triggers 503 disconnect") requires simulating a slow consumer that does not read from the connection. This is testable but requires careful timing. Recommendation: Specify the queue size threshold (500 per task 3.4) in the spec itself, not just the task, so the test has a concrete threshold to verify.

### T2: TOCTOU Race Condition Test Requires Concurrent Callers [NICE TO HAVE]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/state-management/spec.md` (line 30)

The "Atomic transition prevents race condition" scenario requires two concurrent callers. This is testable with `asyncio.gather()` but the spec should note that the test must use actual concurrent database connections, not mocked backends, to verify atomicity.

### T3: Metrics Response Time Requirement (100ms) Needs Test Infrastructure [NICE TO HAVE]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/event-stream/spec.md` (line 144)

The metrics endpoint must respond within 100ms. This is testable but sensitive to CI runner speed. Recommendation: Frame this as "responds under 100ms on a system where direct DB reads take under 10ms" rather than an absolute threshold.

### T4: State-Based Tool Filtering Has No Test Task [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/ogc-integration/spec.md` (lines 114-133)
**Tasks file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/tasks.md`

The "State-Based Tool Filtering" requirement in the OGC spec has three scenarios but no corresponding test task in tasks.md. This feature is non-trivial (mapping JobPhase to algorithm categories) and needs test coverage.

---

## Feasibility Findings

### F1: AlgorithmRegistry Has No `PROCESS_METADATA` Attribute [MUST FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/ogc-integration/spec.md` (lines 22-25, 99)
**Codebase:** `/home/gprice/projects/firstlight/core/analysis/library/registry.py`

The OGC spec references `PROCESS_METADATA` from the `AlgorithmRegistry` multiple times: "using `PROCESS_METADATA` from the `AlgorithmRegistry` to populate the process id, title, description, inputs, and outputs fields" and "populated from `PROCESS_METADATA`." This attribute does not exist in the current `AlgorithmRegistry` class. The registry stores `AlgorithmMetadata` objects with fields like `id`, `name`, `description`, `parameter_schema`, `event_types`, `category`, `resources`, `validation`. The spec should reference these actual fields, not a non-existent `PROCESS_METADATA` attribute. Similarly, the `generate_tool_schema` function referenced in task 2.9 does not exist yet -- this is expected (it is a new function to be created), but the task should note it is a new function, not a modification of an existing one.

### F2: Existing Permission Enum Needs Extension, Not Replacement [SHOULD FIX]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/security-hardening/spec.md` (lines 66-84)
**Codebase:** `/home/gprice/projects/firstlight/api/auth.py` (lines 75-104)

The security spec says the `Permission` enum "SHALL include entries for all control plane operations: `state_read`, `state_write`, and `escalation_manage`." The existing `Permission` enum in `/home/gprice/projects/firstlight/api/auth.py` already has 16 values using the `namespace:action` naming convention (e.g., `event:create`, `product:read`). The new permissions should follow this convention: `state:read`, `state:write`, `escalation:manage`. The spec should explicitly state these are additions to the existing enum, not a replacement.

### F3: UserContext and APIKey Have No `customer_id` Field [EXPECTED -- This Is What Phase 0 Creates]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/security-hardening/spec.md` (lines 1-20)
**Codebase:** `/home/gprice/projects/firstlight/api/auth.py` (lines 151-160, 194-209)

Confirmed: The existing `UserContext` has no `customer_id` field. The existing `APIKey` has no `customer_id` field. Task 0.1 correctly identifies this as the first task. The existing `APIKey` also lives in `api/auth.py` as a dataclass, not in `api/models/database.py` as task 0.1 states. Task 0.1 should reference the correct file.

### F4: Hardcoded Credential Patterns Confirmed in Docker Compose [VERIFIED]

The review of docker-compose files confirms the security concerns:
- `docker-compose.yml`: `POSTGRES_PASSWORD:-devpassword` (hardcoded default)
- `deploy/docker-compose.yml`: `POSTGRES_PASSWORD:-firstlight_dev` (hardcoded default)
- `deploy/on-prem/standalone/docker-compose.yml`: `DB_PASSWORD:-changeme` (hardcoded default)
- No `REDIS_PASSWORD` variable in any compose file

Task 0.4 correctly identifies all of these. The variable naming is inconsistent across compose files (`POSTGRES_PASSWORD` vs `DB_PASSWORD`) -- the task should normalize this.

### F5: ExecutionStage Has Stages Not Mapped in JobPhase [SHOULD FIX]

**Codebase:** `/home/gprice/projects/firstlight/agents/orchestrator/state.py` (lines 32-43)
**Spec:** state-management/spec.md (line 5)
**Tasks:** tasks.md (task 1.5)

The existing `ExecutionStage` has 10 values: PENDING, VALIDATING, DISCOVERY, PIPELINE, QUALITY, REPORTING, ASSEMBLY, COMPLETED, FAILED, CANCELLED.

The proposed `JobPhase` has 7 values: QUEUED, DISCOVERING, INGESTING, NORMALIZING, ANALYZING, REPORTING, COMPLETE.

Mapping issues:
- PENDING -> QUEUED (OK)
- VALIDATING -> ? (no corresponding JobPhase -- is this folded into QUEUED?)
- DISCOVERY -> DISCOVERING (OK, but name change)
- PIPELINE -> split into INGESTING, NORMALIZING, ANALYZING (reasonable -- PIPELINE was coarse)
- QUALITY -> ? (no corresponding JobPhase -- is this folded into ANALYZING?)
- REPORTING -> REPORTING (OK)
- ASSEMBLY -> ? (no corresponding JobPhase -- is this folded into REPORTING?)
- COMPLETED -> COMPLETE (OK, but name change)
- FAILED -> terminal status, not a phase (OK)
- CANCELLED -> terminal status, not a phase (OK)

Three existing stages (VALIDATING, QUALITY, ASSEMBLY) are not represented in the new JobPhase enum. The spec says "7 phases mapping to the existing ExecutionStage" but the mapping is not 1:1. Engineers need an explicit mapping table showing which existing stages map to which new phases.

### F6: `bbox` Generated Column Feasibility [NICE TO HAVE]

**Spec file:** `/home/gprice/projects/firstlight/openspec/changes/add-llm-control-plane/specs/postgis-schema/spec.md` (lines 55-65)

The spec says `bbox` should be a generated column using `ST_Envelope(aoi)`. PostgreSQL generated columns require `IMMUTABLE` functions. `ST_Envelope` is IMMUTABLE in PostGIS, so this is feasible. However, `aoi_area_km2` using `ST_Area(aoi::geography)` may not work as a `GENERATED ALWAYS AS ... STORED` column because the `::geography` cast involves a non-IMMUTABLE function. Task 1.4 uses `FLOAT GENERATED ALWAYS AS (ST_Area(aoi::geography) / 1e6) STORED` -- this will fail if `ST_Area(geography)` is not marked IMMUTABLE. Recommendation: Verify this works with PostGIS 3.4, or fall back to a trigger-based computation.

---

## Consistency Findings

### CO1: Phase+Status Naming Consistency (Previously Flagged as G5) [VERIFIED AS FIXED]

The value chain analysis flagged inconsistent naming across specs (G5). Reviewing the current state:
- **state-management/spec.md**: QUEUED, DISCOVERING, INGESTING, NORMALIZING, ANALYZING, REPORTING, COMPLETE (line 5)
- **control-api/spec.md**: QUEUED, DISCOVERING, INGESTING, NORMALIZING, ANALYZING, REPORTING, COMPLETE (line 8)
- **tasks.md task 1.5**: QUEUED, DISCOVERING, INGESTING, NORMALIZING, ANALYZING, REPORTING, COMPLETE

These are now consistent across the spec files. However, the existing codebase uses PENDING (not QUEUED), DISCOVERY (not DISCOVERING), PIPELINE (not INGESTING/NORMALIZING/ANALYZING), ASSEMBLY (not present), and COMPLETED (not COMPLETE). Task 1.5 needs to document the mapping.

### CO2: HTTP Status Code Consistency Across Specs [PASS]

Reviewed all HTTP status codes across specs:
- 200: Successful operations -- consistent
- 201: Resource creation (jobs, escalations, reasoning, webhooks) -- consistent
- 401: Unauthenticated -- consistent
- 403: Insufficient permissions -- consistent (security spec only)
- 404: Not found / cross-tenant blocked -- consistent
- 409: State conflict / duplicate resolve -- consistent
- 422: Validation errors -- consistent
- 429: Rate limited -- consistent

### CO3: Naming Convention Consistency [PASS WITH NOTES]

- URL paths: kebab-case (`/control/v1/jobs/{id}/parameters`) -- consistent
- JSON fields: snake_case (`customer_id`, `job_id`, `event_type`) -- consistent
- Enum values: UPPER_SNAKE_CASE (`QUEUED`, `DISCOVERING`) -- consistent
- CloudEvents attributes: snake_case with `firstlight_` prefix -- consistent with CE spec

One minor note: The control API spec uses `reason` (singular) for transition payloads while the reasoning injection spec uses `content` for reasoning text. These serve different purposes so the naming difference is acceptable, but documenting the distinction would help.

### CO4: Taskiq Dependency Inversion (Previously G2) [PARTIALLY FIXED]

The value chain analysis flagged Taskiq as formally introduced in Phase 4 but used in Phase 3. The current tasks.md addresses this by adding task 3.0 (install Taskiq and create broker in Phase 3). However, task 4.8 still says "Verify Taskiq broker (created in task 3.0) is compatible with OGC async job execution." This is really a Phase 3 or Phase 4 wiring task, not a new installation. The dependency is now correctly ordered.

---

## Structural Audit

### Stub Traceability

No code stubs are introduced by this spec -- it is a greenfield addition. The specs define new interfaces (StateBackend ABC) rather than modifying existing code with stubs. The `SQLiteStateBackend` (task 1.2) is an adapter wrapping the existing `StateManager`, not a stub.

### API Surface Validation

| Reference | Where Referenced | Exists? | Issue |
|-----------|-----------------|---------|-------|
| `StateManager` | tasks.md 1.2 | Yes | `/home/gprice/projects/firstlight/agents/orchestrator/state.py` |
| `UserContext` | tasks.md 0.1 | Yes | `/home/gprice/projects/firstlight/api/auth.py:151` -- no `customer_id` field yet |
| `APIKey` | tasks.md 0.1 | Yes | `/home/gprice/projects/firstlight/api/auth.py:195` -- task says `api/models/database.py` which is WRONG |
| `AuthSettings.enabled` | tasks.md 0.3 | Yes | `/home/gprice/projects/firstlight/api/config.py:120` -- default is `False` as expected |
| `auth.secret_key` default | tasks.md 0.5 | Yes | `/home/gprice/projects/firstlight/api/config.py:121-124` -- default is `"dev-secret-key-change-in-production"` |
| `AlgorithmRegistry` | tasks.md 2.9 | Yes | `/home/gprice/projects/firstlight/core/analysis/library/registry.py` |
| `generate_tool_schema` | tasks.md 2.9 | No | New function to be created -- this is expected |
| `PROCESS_METADATA` | ogc-integration/spec.md | No | Does not exist in AlgorithmRegistry -- **HALLUCINATED** |
| `Permission` enum | security-hardening/spec.md | Yes | `/home/gprice/projects/firstlight/api/auth.py:75` -- needs extension |
| `OrchestratorAgent.__init__` | tasks.md 1.8 | Exists but not verified | Task assumes it instantiates StateManager directly |
| `api/models/requests.py` | tasks.md 0.1 | Not verified | Task references this for UserContext but actual location is `api/auth.py` |
| `api/models/database.py` | tasks.md 0.1 | Not verified | Task references this for APIKey but actual location is `api/auth.py` |
| `COGConfig` | stac-publishing/spec.md | Yes | `/home/gprice/projects/firstlight/agents/reporting/products.py` -- confirmed existing |
| `ProductGenerator.generate_raster()` | stac-publishing/spec.md | Yes | Confirmed with `as_cog=True` parameter |

### Tech Debt Risks

| Risk | Severity | Proposals Affected | Mitigation |
|------|----------|-------------------|------------|
| Six open questions in design.md remain unresolved | Important | All phases | Resolve questions 1 (checkpoint granularity), 2 (queue isolation), 5 (event retention), and 6 (rate limiting) before engineering starts. Questions 3 (STAC collection strategy) and 4 (OGC sync/async) can wait until Phase 4. |
| No SQLite-to-PostGIS data migration for existing jobs | Important | Phase 1 | Value chain G6 flagged this. Add a migration script task or document that historical jobs are not migrated. |
| Materialized view refresh interval (30s) creates stale metrics window | Minor | Phase 3 | Acceptable for MVP. Document the staleness bound in the metrics endpoint response headers. |
| No integration test between Control API and SSE stream | Important | Phases 2-3 | Task 4.11 (e2e smoke test) covers this, but only in Phase 4. Consider adding a simpler integration test at end of Phase 3. |
| Docker Compose variable naming inconsistency (`POSTGRES_PASSWORD` vs `DB_PASSWORD`) | Minor | Phase 0 | Task 0.4 should normalize variable names across all compose files. |

---

## Acceptance Criteria Generated

The following Gherkin scenarios are generated for requirements that had incomplete or missing acceptance criteria.

### Feature: Job Cancellation (Missing from Control API spec)

```gherkin
Feature: Job Cancellation via Control API
  As an LLM agent
  I want to cancel a running job
  So that I can stop wasted processing when conditions change

  Scenario: Cancel a running job
    Given a job exists in the ANALYZING phase
    When an authenticated agent sends POST /control/v1/jobs/{id}/transition
      with target_phase CANCELLED and target_status CANCELLED
    Then the job transitions to terminal state
    And a job.cancelled event is emitted to job_events
    And the STAC publisher does not publish a result for this job

  Scenario: Cancel an already-completed job
    Given a job exists in the COMPLETE phase
    When an authenticated agent sends a cancellation transition
    Then the API returns HTTP 409
    And the job remains in COMPLETE state
```

### Feature: Webhook Subscription Listing (Missing scenarios from event-stream spec)

```gherkin
Feature: Webhook Subscription Listing
  As a partner operator
  I want to list my webhook subscriptions
  So that I can audit and manage my event delivery configuration

  Scenario: List all active webhooks for customer
    Given two webhook subscriptions exist for customer "acme"
    And one webhook subscription exists for customer "beta"
    When customer "acme" sends GET /internal/v1/webhooks
    Then only the two "acme" subscriptions are returned
    And each subscription includes webhook_id, url, events, active status, and created_at

  Scenario: No webhooks registered
    Given no webhook subscriptions exist for the authenticated customer
    When the customer sends GET /internal/v1/webhooks
    Then the response is HTTP 200 with an empty array
```

### Feature: Health Probe Auth Exemption (Missing path enumeration from security spec)

```gherkin
Feature: Health Probe Auth Exemption
  As a load balancer
  I want to check application health without credentials
  So that I can route traffic based on application readiness

  Scenario: Liveness probe accessible without auth
    Given authentication is enabled
    When an unauthenticated request targets GET /health
    Then the response is HTTP 200 with a JSON body containing service status

  Scenario: Readiness probe accessible without auth
    Given authentication is enabled
    When an unauthenticated request targets GET /ready
    Then the response is HTTP 200 if all dependencies (PostgreSQL, Redis) are reachable
    And HTTP 503 if any dependency is unreachable

  Scenario: Non-exempt path still requires auth
    Given authentication is enabled
    When an unauthenticated request targets GET /control/v1/jobs
    Then the response is HTTP 401
```

### Feature: ExecutionStage to JobPhase Mapping (Missing from state-management spec)

```gherkin
Feature: ExecutionStage to JobPhase Mapping
  As the orchestrator pipeline
  I want stage transitions to map correctly to the new Phase+Status model
  So that existing pipeline code continues to work with the new state backend

  Scenario: PENDING maps to QUEUED
    Given a new job is created via the existing pipeline
    When the StateManager records ExecutionStage.PENDING
    Then the PostGIS jobs table shows phase=QUEUED

  Scenario: VALIDATING maps to QUEUED substatus
    Given a job advances to ExecutionStage.VALIDATING
    When the SQLiteStateBackend adapter translates the stage
    Then the PostGIS jobs table shows phase=QUEUED, status=VALIDATING

  Scenario: PIPELINE stage maps to multiple phases
    Given a job advances to ExecutionStage.PIPELINE
    When the adapter processes sub-stage progress
    Then the phase is set to INGESTING, NORMALIZING, or ANALYZING
      based on the pipeline sub-stage reported in metrics

  Scenario: QUALITY stage maps to ANALYZING substatus
    Given a job advances to ExecutionStage.QUALITY
    When the adapter translates the stage
    Then the PostGIS jobs table shows phase=ANALYZING, status=QUALITY_CHECK
```

*NOTE: These generated criteria are flagged for Product Queen approval. The ExecutionStage-to-JobPhase mapping is a design decision that should be confirmed before engineering.*

---

## Priority Findings

### Critical (must fix before engineering)

1. **CL2: Transition endpoint path and request body differ between spec and tasks.** The spec's `POST /control/v1/jobs/{id}/transition` with expected-state guard is correct. Update tasks.md task 2.5 to match.
2. **C1: StateBackend ABC signature does not match existing StateManager.** Update task 1.1 to clarify the ABC is a new simplified interface, not a 1:1 match with StateManager. Document which existing methods are excluded and why.
3. **F1: `PROCESS_METADATA` does not exist in AlgorithmRegistry.** Update ogc-integration/spec.md to reference `AlgorithmMetadata` fields (`id`, `name`, `description`, `parameter_schema`, `event_types`, `category`, `resources`) instead of a non-existent `PROCESS_METADATA` attribute.

### Important (should fix)

4. **CL1: Webhook retry backoff schedule differs between spec and tasks.** Pick one (spec's 5s base is recommended) and update the other.
5. **CL3: Escalation endpoint paths differ between spec and tasks.** Align on the spec's job-scoped design.
6. **CL4: `aoi_area_km2` column type differs (NUMERIC vs FLOAT).** Use the spec's NUMERIC(12,4).
7. **C7: Metrics materialized view schema differs between spec and tasks.** The spec describes health metrics; the task describes queue metrics. Either add both views or reconcile.
8. **F5: ExecutionStage-to-JobPhase mapping is incomplete.** Add an explicit mapping table for VALIDATING, QUALITY, and ASSEMBLY stages.
9. **CL6: Per-phase substatus values not enumerated.** Add a complete table of valid (phase, status) pairs.
10. **C4: No error response body format specified.** Add a standard error envelope to design.md.
11. **F3: Task 0.1 references wrong file paths for UserContext and APIKey.** Both live in `api/auth.py`, not `api/models/requests.py` and `api/models/database.py`.
12. **C6: Health probe exempt paths not enumerated.** List the specific paths.
13. **T4: State-based tool filtering has no test task.** Add a test task to Phase 4.
14. **C8: Event retention policy unresolved.** Set a default (e.g., 90 days) as a configurable value.

### Minor (could fix)

15. **CL5: `bbox` query parameter format ambiguous.** Specify comma-separated `minlon,minlat,maxlon,maxlat`.
16. **F6: `ST_Area(aoi::geography)` may not work as a generated column.** Verify with PostGIS 3.4 or use a trigger.
17. **C5: Webhook listing endpoint lacks pagination/filter spec.** Add basic pagination params.
18. **CO3: Docker Compose variable naming inconsistency.** Normalize in task 0.4.

---

## Recommended Next Steps

1. **Product Queen**: Resolve the three Critical findings (transition endpoint alignment, StateBackend ABC scope clarification, PROCESS_METADATA reference fix) and create an explicit ExecutionStage-to-JobPhase mapping table as a design decision.
2. **Product Queen**: Close open questions 1, 2, 5, and 6 from design.md before Phase 1 starts. Questions 3 and 4 can remain open until Phase 4.
3. **Product Queen**: Add a standard error response envelope to design.md that all specs reference.
4. **Product Queen**: Enumerate the complete set of valid (phase, status) pairs so engineers have an authoritative reference.
5. **Product Queen**: Update tasks.md to fix file path references (task 0.1), align endpoint paths (tasks 2.5, 2.8), align column types (task 1.4), and reconcile the metrics view schema (task 3.8).
6. **Engineering Manager**: Verify `ST_Area(aoi::geography)` works as a `GENERATED ALWAYS AS ... STORED` column with PostGIS 3.4 before task 1.4 begins.
7. **CTO**: After Critical and Important fixes are applied, this spec is ready for `/run-phase`.
