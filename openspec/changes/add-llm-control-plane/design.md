## Context

FirstLight is a geospatial event intelligence platform with a mature processing pipeline (170K+ lines, 518+ tests) but no external control surface. Pipeline state lives in SQLite, agent coordination is in-memory, and there is no HTTP interface for LLM agents to observe or drive the pipeline. MAIA Analytics needs to integrate FirstLight as a backend service where their conversational layer can submit jobs, monitor progress, and receive structured events.

Seven research reviews were conducted covering state architecture, geospatial schema, security, industry standards, event streaming, and MAIA's integration needs. This design synthesizes those findings.

### Stakeholders
- **MAIA Analytics:** Integration partner — needs event streams, job submission, result discovery
- **LLM agents:** Need HTTP tool-use endpoints to read state, trigger transitions, inject reasoning
- **Existing CLI users:** Must maintain backward compatibility with `flight` command
- **Operations:** Need metrics, queue visibility, escalation management

## Goals / Non-Goals

### Goals
- Externalize pipeline state into PostGIS so it survives process restarts and is queryable
- Expose four distinct API surfaces for different consumers (OGC, STAC, Control, Internal)
- Enable LLM agents to drive the pipeline via standard HTTP tool-use
- Emit structured events at every decision point for partner consumption
- Maintain backward compatibility with existing CLI and agent system
- Harden security to production-grade (tenant isolation, auth-on-by-default)

### Non-Goals
- Building dashboards or UIs (partners build on the event stream)
- Replacing the existing agent framework (BaseAgent, AgentRegistry, MessageBus stay)
- Real-time collaboration between multiple LLM agents on the same job (single-agent-per-job)
- Custom billing, Slack, or notification integrations (event stream covers this)

## Decisions

### D1: Phase+Status State Model (cites state-reconciliation.md)
**Decision:** Two-level state model — coarse `JobPhase` (7 phases mapping to existing ExecutionStage) and fine-grained `JobStatus` (per-phase substates like DISCOVERING/DISCOVERED).
**Why:** The codebase has 6 independent state enums (ExecutionStage, AgentState, EventStatus, DelegationStatus, WorkflowState, TaskStatus). A single flat enum cannot represent all granularities. Phase+Status gives LLM agents the detail they need while keeping the existing pipeline's coarse transitions working.
**Alternatives:** Single unified enum (loses granularity), full state machine library like transitions (over-engineering for HTTP-driven state).

### D2: StateBackend ABC with DualWriteBackend Migration (cites state-reconciliation.md)
**Decision:** Abstract `StateBackend` interface with `SQLiteStateBackend` (current) and `PostGISStateBackend` (target). During migration, `DualWriteBackend` writes to PostGIS as canonical and SQLite as best-effort fallback.
**Why:** Zero-downtime migration. Existing CLI and orchestrator code sees no change — they interact through the same interface. If PostGIS has issues, SQLite still has state.
**Alternatives:** Hard cutover (risky, breaks CLI), ORM abstraction (adds unnecessary layer).

### D3: MULTIPOLYGON(4326) PostGIS Schema (cites geospatial-review.md)
**Decision:** `aoi GEOMETRY(MULTIPOLYGON, 4326)` with `ST_Multi()` promotion at insert, `::geography` cast for metric calculations, GIST index, validity constraints, computed `aoi_area_km2` column.
**Why:** The codebase already handles MultiPolygon in CLI, API models, and discovery. Real wildfire/flood AOIs are often multi-part. POLYGON would force lossy casts. Geography casts give accurate area/distance without changing the storage SRID.
**Alternatives:** GEOMETRY(POLYGON) (lossy), GEOGRAPHY type (fewer functions, slower joins, incompatible with pgSTAC/titiler).

### D4: Taskiq Over Celery (cites standards-and-patterns.md)
**Decision:** Use Taskiq with Redis Streams broker for async task execution.
**Why:** FirstLight is fully async/await. Celery has no native asyncio support (open issue #3884 since 2017). Taskiq is 5.7x faster than Celery in benchmarks (2.03s vs 11.68s for 20K jobs), async-native, and has first-class FastAPI dependency injection via `taskiq-fastapi`.
**Alternatives:** Celery (sync mismatch), ARQ (async but 17x slower than Taskiq), Dramatiq (fast but not async-native).

### D5: Four API Surfaces (cites control-plane-review.md, event-stream-integration.md)
**Decision:** Four distinct API mounts:
- `/oapi/*` — pygeoapi (OGC API Processes) for standards-compliant job submission
- `/stac/*` — stac-fastapi-pgstac for result discovery and metadata
- `/control/v1/*` — Custom FastAPI for LLM agent tool-use (transitions, reasoning, parameters)
- `/internal/v1/*` — Partner integration (SSE events, metrics, queue, escalations)
**Why:** Each surface serves a different consumer with different needs. OGC gives interoperability. STAC gives result discoverability. Control gives LLM-specific affordances. Internal gives partner operational visibility.
**Alternatives:** Single unified API (conflates concerns), GraphQL (overkill for event-driven patterns).

### D6: SSE with CloudEvents over WebSocket (cites event-stream-integration.md)
**Decision:** Server-Sent Events at `/internal/v1/events/stream` with CloudEvents v1.0 envelope, LISTEN/NOTIFY from append-only `job_events` table, Last-Event-ID replay.
**Why:** Partners only need server-push (no bidirectional). SSE works through HTTP proxies/CDNs, has built-in reconnection, and is natively supported by browsers. CloudEvents is the CNCF standard envelope. The `job_events` table is both the audit log AND the replay buffer — no separate event bus needed.
**Alternatives:** WebSocket (bidirectional overhead, no built-in reconnect), Kafka (infrastructure overhead for this scale), polling (latency, waste).

### D7: pygeoapi Mount for OGC Processes (cites standards-and-patterns.md)
**Decision:** Mount pygeoapi as ASGI sub-application via `app.mount("/oapi", pygeoapi_starlette_app)`. Each algorithm becomes a `BaseProcessor` subclass.
**Why:** pygeoapi is the reference OGC API implementation. Mounting avoids rewriting the API layer. Vendor extensions (`x-firstlight-*`) add LLM-specific fields without breaking spec compliance. No standalone FastAPI-native OGC Processes implementation exists.
**Alternatives:** Custom OGC implementation (massive effort), fastgeoapi (wraps pygeoapi anyway).

### D8: pgSTAC for Result Publishing (cites standards-and-patterns.md)
**Decision:** Install pgSTAC schema into existing PostgreSQL, mount stac-fastapi-pgstac at `/stac`. Publish every analysis result as a STAC Item with `processing` extension.
**Why:** pgSTAC scales to hundreds of millions of items (NASA, USGS, Planetary Computer use it). Makes results discoverable by any STAC client. titiler-pgstac can generate dynamic map tiles. The existing pystac-client for input discovery is complementary — different direction of data flow.
**Alternatives:** Custom result catalog (reinventing the wheel), file-based STAC catalog (doesn't scale).

### D9: 4-Layer Webhook Delivery (cites standards-and-patterns.md, event-stream-integration.md)
**Decision:** HMAC-SHA256 signing, Taskiq-managed retries with exponential backoff + jitter, PostgreSQL dead letter queue, Redis idempotency keys.
**Why:** Industry standard (Stripe, GitHub, Twilio pattern). Fire-and-forget webhooks lose events. The Taskiq task queue handles retries naturally. The event log is source of truth — failed deliveries can always be replayed.
**Alternatives:** Fire-and-forget (lossy), dedicated webhook service (over-engineering at this scale).

### D10: Standard Error Response Envelope
**Decision:** All API surfaces (`/control/v1/*`, `/internal/v1/*`) SHALL return errors using a consistent JSON envelope:

```json
{
  "error": {
    "code": "CONFLICT",
    "message": "Job is not in expected state",
    "details": {
      "current_phase": "ANALYZING",
      "current_status": "ANALYZING"
    }
  }
}
```

The `error.code` field is a machine-readable string constant. The `error.message` field is a human-readable description. The `error.details` field is an optional object carrying structured context specific to the error type. HTTP status codes map to error codes as follows:

| HTTP Status | Error Code | Usage |
|-------------|------------|-------|
| 400 | `BAD_REQUEST` | Malformed request body or parameters |
| 401 | `UNAUTHENTICATED` | Missing or invalid API key / bearer token |
| 403 | `FORBIDDEN` | Authenticated but insufficient permissions |
| 404 | `NOT_FOUND` | Resource does not exist or is in another tenant |
| 409 | `CONFLICT` | State conflict (TOCTOU guard, duplicate resolve) |
| 422 | `VALIDATION_ERROR` | Schema validation failure (details lists individual field errors) |
| 429 | `RATE_LIMITED` | Rate limit exceeded (includes `Retry-After` header) |
| 500 | `INTERNAL_ERROR` | Unexpected server error (no internal details exposed) |

**Why:** Without a standard envelope, different engineers produce different error shapes, making client error handling brittle. This follows Stripe/GitHub patterns.
**Alternatives:** RFC 7807 Problem Details (heavier, less LLM-friendly).

### D11: Event Retention Policy (resolves Open Question 5)
**Decision:** The `job_events` table SHALL be partitioned by month using PostgreSQL declarative partitioning (`PARTITION BY RANGE (occurred_at)`). Active partitions (last 90 days) are retained in hot storage. Partitions older than 90 days are automatically detached and archived to cold storage (compressed pg_dump files) via a scheduled maintenance task. The retention period is configurable via `FIRSTLIGHT_EVENT_RETENTION_DAYS` (default: 90).
**Why:** The `job_events` table serves as both the audit log and SSE replay buffer. Unbounded growth degrades replay queries and increases backup sizes. 90 days provides sufficient replay window for partner reconnections while keeping the hot table performant. Monthly partitioning aligns with natural data lifecycle and allows instant partition drops without vacuum overhead.
**Alternatives:** Row-level DELETE with cron (vacuum overhead, table bloat), no retention (unbounded growth).

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| PostGIS migration corrupts state | Data loss during transition | DualWriteBackend pattern — SQLite remains as fallback |
| pygeoapi mount conflicts with FastAPI routes | Routing errors | Test mount points with path prefix isolation |
| Taskiq immaturity vs Celery ecosystem | Fewer community resources | ARQ as fallback; Taskiq has active development |
| TOCTOU race on state transitions | Invalid state changes | Atomic `UPDATE WHERE state = :expected` pattern |
| Event stream backpressure | Memory exhaustion on slow consumers | Per-connection buffer limits, disconnect stale clients |
| pgSTAC schema conflicts with existing tables | Migration failures | pgSTAC uses its own schema namespace, separate from app tables |

## Migration Plan

### Phase 0: Security Blockers (must complete first)
1. Add `customer_id` to UserContext and APIKey models
2. Enable auth by default (`AuthSettings.enabled = True`)
3. Rotate infrastructure credentials, remove defaults from docker-compose
4. Add tenant isolation middleware (customer_id scoping on all queries)

### Phase 1: State Externalization
1. Define `StateBackend` ABC and `SQLiteStateBackend` (wraps existing StateManager)
2. Create PostGIS schema (jobs, job_events, job_checkpoints tables)
3. Implement `PostGISStateBackend` with asyncpg
4. Add `DualWriteBackend` for migration period
5. Swap docker-compose postgres image to postgis/postgis:15-3.4-alpine

### Phase 2: Control API
1. Implement `/control/v1/jobs` CRUD endpoints
2. Add state transition endpoint with atomic WHERE clause
3. Add parameter adjustment and reasoning injection endpoints
4. Add escalation create/resolve endpoints
5. Wire AlgorithmRegistry to auto-generate LLM tool schemas

### Phase 3: Event Stream & Partner API
1. Create `job_events` INSERT trigger with NOTIFY
2. Implement SSE endpoint with Last-Event-ID replay
3. Add CloudEvents envelope formatting
4. Implement webhook registration, delivery, and DLQ
5. Add metrics endpoint (materialized view, 30s refresh)
6. Add queue summary and escalation list endpoints

### Phase 4: Standards Integration
1. Mount pygeoapi with algorithm BaseProcessor subclasses
2. Install pgSTAC schema, mount stac-fastapi-pgstac
3. Publish analysis results as STAC Items with processing extension
4. Wire Taskiq for internal pipeline orchestration

### Rollback
- DualWriteBackend can be switched to SQLite-only by config change
- API mounts can be disabled independently via feature flags
- Event stream is append-only — no destructive operations

## Open Questions

1. ~~**Checkpoint granularity:** How often should `job_checkpoints` be written? Every state transition, or only at phase boundaries?~~ **RESOLVED** -- Checkpoints are written at phase boundaries only (when JobPhase changes). Writing at every status change would produce excessive I/O. The `checkpoint(job_id, payload)` method is called by the orchestrator at phase entry, not by the state backend automatically. The pipeline stage is responsible for calling checkpoint with its resumable state. Note: the checkpoint payload structure per pipeline stage is NOT specified -- this is acceptable for MVP since checkpoint/resume is an optimization, not a correctness requirement. The existing `StateManager.checkpoint()` stores `ExecutionState.to_dict()` -- the new backend accepts arbitrary JSON payloads.
2. ~~**Multi-tenant queue isolation:** Should Taskiq use separate Redis databases per customer, or is queue-prefix isolation sufficient?~~ **RESOLVED** -- Queue-prefix isolation is sufficient for MVP. Taskiq tasks include `customer_id` in the task payload and labels. Separate Redis databases per customer is over-engineering at current scale (single integration partner). If multi-tenant queue starvation becomes an issue, add per-customer priority lanes via Taskiq middleware.
3. ~~**pgSTAC collection strategy:** One STAC collection per event type, or per customer, or per mission?~~ **RESOLVED** -- One collection per event type (flood, wildfire, storm). This matches the data model (algorithms are organized by event type) and is the simplest approach. Customer isolation is handled at the API layer (tenant-scoped queries), not at the collection level. Task 4.6 implements this.
4. ~~**OGC Process sync vs async:** Should all algorithms be async-execute only, or support sync for fast algorithms?~~ **RESOLVED** -- All algorithms are async-execute only. Sync execution is rejected with 400 for long-running algorithms (all FirstLight algorithms are long-running by nature -- satellite data processing). This simplifies the implementation and aligns with task 4.8 (all execution via Taskiq).
5. ~~**Event retention:** How long should `job_events` be retained?~~ **RESOLVED** -- See D11. Partition by month, retain 90 days, auto-archive to cold storage.
6. ~~**LLM agent rate limiting:** Per-agent or per-customer rate limits on control API? Token bucket or sliding window?~~ **RESOLVED** -- Both per-agent AND per-customer sliding window counters in Redis. See control-api spec (Rate Limiting requirement) and task 2.10.

## Engineering Notes (added by EM)

### EN1: Project Has No Migration Framework
The project has no `alembic/` directory and no database migration tooling. Schema changes will use raw SQL files in a new `db/migrations/` directory, applied manually or via Docker entrypoint init scripts. Tasks reference this directory. If the team later adopts alembic, the raw SQL files can be converted.

### EN2: `aoi_area_km2` Generated Column Feasibility
The `ST_Area(aoi::geography)` function may not be marked IMMUTABLE in PostGIS 3.4, which would prevent it from being used in a `GENERATED ALWAYS AS ... STORED` column. Task 1.4 instructs the coder to test this against the Docker PostGIS image first. Fallback: use an `AFTER INSERT OR UPDATE OF aoi` trigger. The `ST_Envelope(aoi)` function IS IMMUTABLE and should work as a generated column for the `bbox` field.

### EN3: Existing Error Handling Is Already Standardized
The design doc (D10) specifies a standard error response envelope. The codebase already has `api/models/errors.py` with `ErrorCode` enum, `ErrorResponse` model, and `APIError` exception hierarchy including `ConflictError` (409), `RateLimitError` (429), and `ValidationError` (422). New control plane endpoints should reuse these existing error classes rather than creating parallel error handling. The `CONFLICT` error code needed for state transition 409 responses should be added to `ErrorCode` if not already present (check -- `ALREADY_EXISTS` exists but `CONFLICT` does not; add `CONFLICT = "CONFLICT"` or reuse `ALREADY_EXISTS`).

### EN4: asyncpg vs SQLAlchemy
The existing `DatabaseSettings` in `api/config.py` constructs a SQLAlchemy-style URL (`postgresql+asyncpg://...`). The PostGIS state backend uses raw asyncpg (not SQLAlchemy). Coders must build asyncpg DSN from individual `DatabaseSettings` fields (host, port, name, user, password), not from the `url` property.

### EN5: StateManager Sync vs Async Mismatch
The existing `StateManager` in `agents/orchestrator/state.py` uses synchronous `sqlite3` (not `aiosqlite`). The `StateBackend` ABC declares async methods. The `SQLiteStateBackend` adapter (task 1.5) will need to wrap synchronous `StateManager` calls in `asyncio.to_thread()` to bridge this gap.

### EN6: API Key Storage Model
The `APIKey` dataclass in `api/auth.py` appears to be used with in-memory storage (no database ORM model found). The `customer_id` migration in task 0.1 assumes a persistent `api_keys` table. The coder needs to verify how API keys are actually stored before writing the migration. If keys are in-memory only, the migration adds a column to a table that may not exist yet. Consider creating the table as part of the migration if needed.
