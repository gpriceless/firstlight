# Tasks: add-llm-control-plane

<!-- phase: add-llm-control-plane -->

Ordered checklist for implementing the LLM Control Plane across five phases.
Each task is independently verifiable. Dependencies are called out inline.

Plane Issue: FIRSTLIGHT-2

---

## Phase 0: Security Blockers
<!-- execution: sequential -->

These tasks must be complete and merged before any Phase 1 work begins. They are breaking changes that affect every downstream consumer.

- [x] 0.1 Add `customer_id: str` field to `UserContext` dataclass and to `APIKey` dataclass in `api/auth.py`; add a database migration that adds the `customer_id` column to the `api_keys` table with a `NOT NULL` constraint and a backfill of `"legacy"` for existing rows
  <!-- files: api/auth.py (modify — UserContext at line 150, APIKey at line 194), db/migrations/000_add_customer_id.sql (new) -->
  <!-- pattern: follow api/auth.py for dataclass conventions (field order, type hints, defaults) -->
  <!-- test: tests/api/test_auth_enforcement.py (task 0.6) -->
  <!-- gotcha: Both UserContext and APIKey live in api/auth.py, NOT api/models/requests.py or api/models/database.py. There is no alembic/ directory — use raw SQL migration files in a new db/migrations/ directory. The APIKey is currently a dataclass with in-memory storage (see api/auth.py line 194-209), not a database model — the "api_keys table" migration only applies if there is a persistent key store. Verify whether keys are stored in SQLite or in-memory before writing the migration. -->

- [x] 0.2 Add `TenantMiddleware` to `api/middleware.py` that extracts `customer_id` from the resolved API key on every request and attaches it to `request.state.customer_id`; raise `HTTP 401` if the key resolves to no tenant; add auth exemption allowlist for health probes (`GET /health`, `GET /ready`) and OGC discovery paths (`GET /oapi/`, `GET /oapi/conformance`, `GET /oapi/processes`, `GET /oapi/processes/{id}`) (depends on 0.1)
  <!-- files: api/middleware.py (modify — add TenantMiddleware class following CorrelationIdMiddleware pattern at line 24), api/main.py (modify — add TenantMiddleware to setup_middleware) -->
  <!-- pattern: follow api/middleware.py CorrelationIdMiddleware for BaseHTTPMiddleware subclass structure -->
  <!-- test: tests/api/test_auth_enforcement.py (task 0.6) -->
  <!-- gotcha: The middleware stack is configured via setup_middleware(app, settings) in api/middleware.py — add TenantMiddleware there, not directly in api/main.py. The exemption allowlist must be configurable to allow Phase 4 OGC paths. -->

- [x] 0.3 Change `AuthSettings.enabled` default from `False` to `True` in `api/config.py`; update the dev `.env.example` to set `AUTH_ENABLED=true` and document the change in `CHANGELOG.md`
  <!-- files: api/config.py (modify — line 120: change default=False to default=True), .env.example (modify or create), CHANGELOG.md (modify or create) -->
  <!-- pattern: follow api/config.py AuthSettings pattern (Pydantic BaseSettings with env_prefix="AUTH_") -->
  <!-- test: tests/api/test_auth_enforcement.py (task 0.6) -->
  <!-- gotcha: This is a BREAKING change. Existing tests that do not set AUTH_ENABLED=false will fail after this. Test fixtures may need updating. Check tests/api/test_jwt_auth.py and tests/test_api.py for existing auth test patterns. -->

- [x] 0.4 Audit all three compose files for hardcoded credentials and normalize the database password variable name to `POSTGRES_PASSWORD` across all of them: (a) `docker-compose.yml` -- replace `POSTGRES_PASSWORD:-devpassword` with `${POSTGRES_PASSWORD}` (no default); (b) `deploy/docker-compose.yml` -- replace `POSTGRES_PASSWORD:-firstlight_dev` with `${POSTGRES_PASSWORD}` (no default); (c) `deploy/on-prem/standalone/docker-compose.yml` -- rename `DB_PASSWORD:-changeme` to `POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}` (no default); add `REDIS_PASSWORD: ${REDIS_PASSWORD}` to all three compose files (Redis currently has no auth — see redis command in compose files); add a startup healthcheck script that aborts with a clear error if either `POSTGRES_PASSWORD` or `REDIS_PASSWORD` is unset
  <!-- files: docker-compose.yml (modify — lines 44, 79, 113, 132), deploy/docker-compose.yml (modify — lines 137), deploy/on-prem/standalone/docker-compose.yml (modify — lines 33, 63, 98), scripts/check-env.sh (new) -->
  <!-- test: tests/test_docker_compose.py (verify — existing tests may validate compose structure) -->
  <!-- gotcha: The standalone compose at deploy/on-prem/standalone/ already uses postgis/postgis:15-3.4-alpine (line 93) — do not change the image back to postgres:15-alpine during credential cleanup. Also the worker service in standalone uses DATABASE_URL with DB_PASSWORD — update that too. The redis command at docker-compose.yml line 113 has --appendonly but no --requirepass — add --requirepass ${REDIS_PASSWORD} after credential injection. -->

- [x] 0.5 Replace the hardcoded `auth.secret_key` default `"dev-secret-key-change-in-production"` in `api/config.py` with a validator that raises `ValueError` when `FIRSTLIGHT_ENVIRONMENT=production` and the key matches the default
  <!-- files: api/config.py (modify — lines 121-124, add field_validator for secret_key) -->
  <!-- pattern: follow api/config.py field_validator pattern (see parse_api_keys at line 134) -->
  <!-- test: tests/api/test_auth_enforcement.py (task 0.6) -->

- [x] 0.6 Add `state:read`, `state:write`, `escalation:manage` permissions to the `Permission` enum in `api/auth.py`; update `ROLE_PERMISSIONS` to grant `state:read` to `readonly` and above, `state:write` to `operator` and above, and `escalation:manage` to `operator` and above
  <!-- files: api/auth.py (modify — Permission enum at line 75, ROLE_PERMISSIONS dict at line 107) -->
  <!-- pattern: follow existing namespace:action convention (e.g., event:create, webhook:read) -->
  <!-- test: tests/api/test_auth_enforcement.py (task 0.7) -->
  <!-- gotcha: The Permission enum already has 19 values using the namespace:action format. The security spec says state_read/state_write but the codebase convention is state:read/state:write — use the codebase convention. -->

- [x] 0.7 Write integration tests in `tests/api/test_auth_enforcement.py` that: (a) verify a request without an API key returns `HTTP 401` now that auth is on by default, (b) verify `customer_id` is present in `request.state` after successful auth, (c) verify the tenant middleware blocks cross-tenant job queries, (d) verify health probes are exempt from auth, (e) verify `state:write` permission is required for transition endpoints (depends on 0.1, 0.2, 0.3, 0.6)
  <!-- files: tests/api/test_auth_enforcement.py (new) -->
  <!-- pattern: follow tests/api/test_jwt_auth.py for API test fixture patterns -->

<!-- wave-gate -->

---

## Phase 1: State Externalization
<!-- execution: sequential -->

Goal: PostGIS becomes the canonical state store. SQLite remains live as fallback. No existing CLI or orchestrator code changes its call sites.

### Section 1A: Foundations (ABC + Enums + Schema)

- [ ] 1.1 Define `StateBackend` ABC in `agents/orchestrator/backends/base.py` with async abstract methods: `get_state(job_id) -> JobState`, `set_state(job_id, phase, status)`, `transition(job_id, expected_phase, expected_status, new_phase, new_status) -> JobState` (atomic, raises `StateConflictError` on mismatch), `list_jobs(filters) -> List[JobState]`, and `checkpoint(job_id, payload)`. Also define `StateConflictError` exception class and `JobState` dataclass in the same module. This is a new simplified interface -- it is NOT a 1:1 mirror of the existing `StateManager` class (which has 11+ methods including `create_state`, `update_stage`, `restore_checkpoint`, `record_error`, `set_degraded_mode`, `get_stage_history`, `get_error_history`). The existing `StateManager` methods not in the ABC are either handled by the adapter layer (task 1.2) or by dedicated tables (e.g., `record_error` writes to `job_events` in the PostGIS backend).
  <!-- files: agents/orchestrator/backends/__init__.py (new), agents/orchestrator/backends/base.py (new) -->
  <!-- pattern: use ABC from abc module; async abstract methods; type hints on all signatures -->
  <!-- gotcha: The existing StateManager is in agents/orchestrator/state.py. Do not modify it — the ABC is a new interface in a new subpackage. The adapter (task 1.2) bridges the two. -->

- [ ] 1.2 Define `JobPhase` and `JobStatus` enums in `agents/orchestrator/state_model.py` with full mapping table and per-phase substatus validation. `JobPhase` values: `QUEUED`, `DISCOVERING`, `INGESTING`, `NORMALIZING`, `ANALYZING`, `REPORTING`, `COMPLETE`. Terminal statuses `FAILED` and `CANCELLED` are `JobStatus` values, not phases. Include the complete ExecutionStage-to-JobPhase mapping:
  - PENDING -> QUEUED (status=PENDING)
  - VALIDATING -> QUEUED (status=VALIDATING)
  - DISCOVERY -> DISCOVERING (status=DISCOVERING)
  - PIPELINE -> INGESTING, NORMALIZING, or ANALYZING (determined by sub-stage progress)
  - QUALITY -> ANALYZING (status=QUALITY_CHECK)
  - REPORTING -> REPORTING (status=REPORTING)
  - ASSEMBLY -> REPORTING (status=ASSEMBLING)
  - COMPLETED -> COMPLETE (status=COMPLETE)
  - FAILED -> (any phase, status=FAILED)
  - CANCELLED -> (any phase, status=CANCELLED)

  Add a `VALID_PHASE_STATUS_PAIRS` dict that enumerates all legal (phase, status) combinations. Add a `validate_transition(from_phase, from_status, to_phase, to_status) -> bool` function.
  <!-- files: agents/orchestrator/state_model.py (new) -->
  <!-- pattern: use str Enums like existing ExecutionStage at agents/orchestrator/state.py line 32 -->
  <!-- gotcha: The per-phase substatus values from the state-management spec are: QUEUED={PENDING, VALIDATING, VALIDATED, VALIDATION_FAILED}, DISCOVERING={DISCOVERING, DISCOVERED, DISCOVERY_FAILED}, INGESTING={INGESTING, INGESTED, INGESTION_FAILED}, NORMALIZING={NORMALIZING, NORMALIZED, NORMALIZATION_FAILED}, ANALYZING={ANALYZING, QUALITY_CHECK, ANALYZED, ANALYSIS_FAILED}, REPORTING={REPORTING, ASSEMBLING, REPORTED, REPORTING_FAILED}, COMPLETE={COMPLETE}. Plus universal terminal statuses FAILED and CANCELLED valid in any phase. -->

- [ ] 1.3 Swap `docker-compose.yml` postgres image from `postgres:15-alpine` to `postgis/postgis:15-3.4-alpine`; update `deploy/docker-compose.yml`; add `CREATE EXTENSION IF NOT EXISTS postgis` to the init SQL; note that `deploy/on-prem/standalone/docker-compose.yml` already uses the PostGIS image
  <!-- files: docker-compose.yml (modify — line 125: change image), deploy/docker-compose.yml (modify — line 128: change image), db/init.sql (new — CREATE EXTENSION IF NOT EXISTS postgis) -->
  <!-- gotcha: The standalone compose already has postgis/postgis:15-3.4-alpine at line 93. deploy/docker-compose.yml has an init-db.sql mount at line 141 — use that path for the init SQL or create a new one. -->

- [ ] 1.4 Write the PostGIS schema migration as a raw SQL file at `db/migrations/001_control_plane_schema.sql` containing:
  - `jobs` table: `job_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `customer_id TEXT NOT NULL`, `event_type TEXT NOT NULL`, `aoi GEOMETRY(MULTIPOLYGON, 4326) NOT NULL`, `aoi_area_km2 NUMERIC(12, 4)` (use `GENERATED ALWAYS AS (ST_Area(aoi::geography) / 1e6) STORED` if PostGIS 3.4 supports it; otherwise fall back to an AFTER INSERT OR UPDATE trigger), `bbox GEOMETRY(POLYGON, 4326) GENERATED ALWAYS AS (ST_Envelope(aoi)) STORED`, `phase TEXT NOT NULL`, `status TEXT NOT NULL`, `orchestrator_id TEXT`, `parameters JSONB DEFAULT '{}'`, `created_at TIMESTAMPTZ DEFAULT now()`, `updated_at TIMESTAMPTZ DEFAULT now()`; add `CHECK (ST_IsValid(aoi))`, `CHECK (NOT ST_IsEmpty(aoi))`, `CHECK (ST_Area(aoi::geography) / 1e6 BETWEEN 0.01 AND 5000000)` area bounds constraint, and `GIST index` on `aoi`; add trigger for `updated_at` auto-update
  - `job_events` table: `event_seq BIGSERIAL PRIMARY KEY`, `job_id UUID NOT NULL REFERENCES jobs(job_id)`, `customer_id TEXT NOT NULL`, `event_type TEXT NOT NULL DEFAULT 'STATE_TRANSITION'`, `phase TEXT NOT NULL`, `status TEXT NOT NULL`, `reasoning TEXT`, `actor TEXT NOT NULL`, `payload JSONB DEFAULT '{}'`, `occurred_at TIMESTAMPTZ DEFAULT now()`; index on `(job_id, event_seq)`
  - `job_checkpoints` table: `checkpoint_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `job_id UUID NOT NULL REFERENCES jobs(job_id)`, `phase TEXT NOT NULL`, `state_snapshot JSONB NOT NULL`, `created_at TIMESTAMPTZ DEFAULT now()`; index on `(job_id, created_at DESC)`
  - `escalations` table: `escalation_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `job_id UUID NOT NULL REFERENCES jobs(job_id)`, `customer_id TEXT NOT NULL`, `reason TEXT NOT NULL`, `severity TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL'))`, `context JSONB`, `created_at TIMESTAMPTZ DEFAULT now()`, `resolved_at TIMESTAMPTZ`, `resolution TEXT`, `resolved_by TEXT`; index on `(customer_id, resolved_at NULLS FIRST)` for open escalation queries
  (depends on 1.3)
  <!-- files: db/migrations/001_control_plane_schema.sql (new) -->
  <!-- gotcha: ST_Area(aoi::geography) may not work as GENERATED ALWAYS AS ... STORED because the geography cast might not be marked IMMUTABLE in PostGIS 3.4. Test this first against the Docker PostGIS image. If it fails, use a trigger-maintained column instead per the postgis-schema spec implementation note. ST_Envelope(aoi) is IMMUTABLE and should work as a generated column. There is no alembic in this project — raw SQL migrations are the pattern. -->

<!-- wave-gate: 1A -->

### Section 1B: Backend Implementations

- [ ] 1.5 Create `agents/orchestrator/backends/sqlite_backend.py` -- wrap the existing `StateManager` class to implement the 5-method `StateBackend` ABC; this is an adapter that maps the ABC's simplified interface to the existing `StateManager` methods (e.g., `set_state` delegates to `StateManager.update_state`, `transition` does a `get_state` + compare + `update_state` with locking, `list_jobs` delegates to `list_active_executions` with filter post-processing, `checkpoint` delegates to `StateManager.checkpoint`); implement the ExecutionStage-to-JobPhase mapping from task 1.2 in both directions (read: ExecutionStage -> JobPhase for API consumers; write: JobPhase -> ExecutionStage for StateManager compatibility); no logic changes to `StateManager` itself, pure adapter (depends on 1.1, 1.2)
  <!-- files: agents/orchestrator/backends/sqlite_backend.py (new) -->
  <!-- pattern: follow agents/orchestrator/state.py StateManager interface -->
  <!-- gotcha: StateManager is instantiated with a db_path (see agents/orchestrator/main.py line 191: StateManager(self.config.state_db_path)). The adapter needs to accept a StateManager instance or create one. StateManager uses synchronous sqlite3 (not aiosqlite) despite the async codebase — the adapter may need to wrap sync calls in asyncio.to_thread(). -->

- [ ] 1.6 Implement `PostGISStateBackend` in `agents/orchestrator/backends/postgis_backend.py` using `asyncpg` (not SQLAlchemy) with connection pool; implement all `StateBackend` methods; use `ST_Multi(ST_GeomFromGeoJSON($1))` for AOI inserts; use atomic `UPDATE jobs SET phase=$1, status=$2, updated_at=now() WHERE job_id=$3 AND phase=$4 AND status=$5 RETURNING *` for transition updates; on successful transition, INSERT into `job_events` within the same transaction; validate (phase, status) pairs against `VALID_PHASE_STATUS_PAIRS` from task 1.2 before any write (depends on 1.1, 1.2, 1.4)
  <!-- files: agents/orchestrator/backends/postgis_backend.py (new) -->
  <!-- pattern: use asyncpg connection pool; follow api/config.py DatabaseSettings for connection params (host, port, name, user, password at lines 37-55) -->
  <!-- gotcha: asyncpg is not currently in pyproject.toml dependencies — add it. The DatabaseSettings.url property returns a SQLAlchemy-style URL (postgresql+asyncpg://...) — asyncpg needs a plain DSN (postgresql://...). Build the DSN from individual settings fields, not from the url property. -->

- [ ] 1.7 Implement `DualWriteBackend` in `agents/orchestrator/backends/dual_write.py`: PostGIS is the canonical write; SQLite write is wrapped in `try/except` and logged as warning on failure; reads always go to PostGIS; expose a `primary_healthy: bool` property (depends on 1.5, 1.6)
  <!-- files: agents/orchestrator/backends/dual_write.py (new) -->
  <!-- test: tests/state/test_dual_write_backend.py (task 1.10) -->

- [ ] 1.8 Add `STATE_BACKEND` setting to `api/config.py` with values `sqlite | postgis | dual`; update `OrchestratorAgent.__init__` to instantiate the correct backend from this setting; default to `dual` so no existing deployment breaks (depends on 1.5, 1.6, 1.7)
  <!-- files: api/config.py (modify — add StateBackendType enum and STATE_BACKEND field to Settings class), agents/orchestrator/main.py (modify — line 155-158, change __init__ to accept optional StateBackend; line 191, use factory based on config) -->
  <!-- pattern: follow api/config.py Environment enum pattern for StateBackendType -->
  <!-- gotcha: OrchestratorAgent.__init__ currently instantiates StateManager directly at line 191. The refactor should keep backward compatibility — if no backend is passed, instantiate based on the STATE_BACKEND config. Import the Settings lazily to avoid circular imports between agents/ and api/ modules. -->

- [ ] 1.9 Add `asyncpg` to `pyproject.toml` dependencies; add `cloudevents` SDK to the `[control-plane]` optional group; create this optional group if it does not exist
  <!-- files: pyproject.toml (modify — add asyncpg to dependencies list around line 66, add [control-plane] optional group after line 100) -->
  <!-- gotcha: pyproject.toml currently has no [control-plane] optional group. The group should include: asyncpg, taskiq, taskiq-fastapi, taskiq-redis, pygeoapi, stac-fastapi-pgstac, pypgstac, cloudevents. Install only asyncpg as a core dependency (needed for Phase 1); the rest go in the optional group for Phase 3-4. -->

<!-- wave-gate: 1B -->

### Section 1C: Tests

- [ ] 1.10 Write `tests/state/test_postgis_backend.py` covering: AOI insert with `ST_Multi` promotion, TOCTOU-safe transition rejection when current phase does not match expected, checkpoint round-trip, error recording via job_events, list_jobs filtering by phase/status, GeoJSON round-trip (coordinates preserved to 5 decimal places); run against a PostGIS container in CI (depends on 1.6)
  <!-- files: tests/state/__init__.py (new), tests/state/test_postgis_backend.py (new) -->
  <!-- pattern: follow tests/test_algorithm_registry.py for test structure; use pytest-asyncio for async tests -->
  <!-- gotcha: Tests need a live PostGIS instance. Use pytest fixtures with docker-compose or testcontainers-python. Mark with @pytest.mark.integration so they can be skipped in CI without PostGIS. -->

- [ ] 1.11 Write `tests/state/test_dual_write_backend.py` covering: PostGIS failure falls back to SQLite without raising, reads come from PostGIS even when SQLite write fails, `primary_healthy` reflects actual state (depends on 1.7, 1.10)
  <!-- files: tests/state/test_dual_write_backend.py (new) -->

- [ ] 1.12 Write `tests/state/test_state_model.py` covering: all valid (phase, status) pairs are accepted, invalid pairs are rejected, ExecutionStage-to-JobPhase mapping is bidirectional and correct for all 10 ExecutionStage values, validate_transition rejects impossible transitions (depends on 1.2)
  <!-- files: tests/state/test_state_model.py (new) -->

<!-- wave-gate: Phase 1 -->

---

## Phase 2: Control API
<!-- execution: parallel -->

Goal: LLM agents can read job state, trigger transitions, adjust parameters, and submit reasoning through `POST /control/v1/` endpoints. All endpoints are tenant-scoped.

### Track A: Core CRUD + Transitions (sequential within track)

- [ ] 2.1 Create `api/routes/control/` package with `__init__.py`; define a `control_router = APIRouter(prefix="/control/v1", tags=["LLM Control"])` that aggregates sub-routers for jobs, escalations, and tools; register the router in `api/main.py` by adding `app.include_router(control_router)` alongside the existing `api_router`; add `customer_id` scoping via a `get_current_customer` dependency that reads from `request.state.customer_id` (set by TenantMiddleware) (depends on Phase 0 complete, Phase 1 complete)
  <!-- files: api/routes/control/__init__.py (new), api/routes/control/jobs.py (new — empty router stub), api/routes/control/escalations.py (new — empty router stub), api/routes/control/tools.py (new — empty router stub), api/main.py (modify — import and include control_router) -->
  <!-- pattern: follow api/routes/__init__.py for router aggregation pattern; follow api/routes/events.py for individual route file structure -->
  <!-- gotcha: The existing routes are all under /api/v1 via a single api_router. The control API is a new top-level prefix /control/v1 — use app.include_router directly, not through api_router. This avoids /api/v1/control/v1 double-prefixing. -->

- [ ] 2.2 Implement `GET /control/v1/jobs` -- list jobs for the authenticated tenant; support query params: `phase`, `status`, `event_type`, `bbox` (comma-separated WGS84 decimal values in the format `west,south,east,north` -- e.g. `bbox=-122.5,37.5,-121.5,38.5`; reject with 422 if values are outside WGS84 bounds or the string cannot be parsed as four numbers), `page` (1-based, default 1), `page_size` (default 20, max 100); return paginated response with `items` (list of `JobSummary`), `total`, `page`, `page_size`; each `JobSummary` contains `job_id`, `phase`, `status`, `event_type`, `aoi_area_km2`, `created_at`, `updated_at` (depends on 2.1)
  <!-- files: api/routes/control/jobs.py (modify), api/models/control.py (new — Pydantic models for JobSummary, PaginatedResponse) -->
  <!-- pattern: follow api/models/responses.py for Pydantic response model conventions -->

- [ ] 2.3 Implement `POST /control/v1/jobs` -- create a new job; accept `event_type`, `aoi` as GeoJSON (any geometry type, promote to MultiPolygon via `ST_Multi`), `parameters: dict`, optional `reasoning: str`; validate GeoJSON: check WGS84 coordinate bounds, vertex count limit (configurable), `ST_IsValid` before insert; return `HTTP 201` with `JobResponse` containing `job_id`; emit a `job.created` event to `job_events` (depends on 2.1)
  <!-- files: api/routes/control/jobs.py (modify), api/models/control.py (modify — add CreateJobRequest, JobResponse models) -->
  <!-- gotcha: GeoJSON validation should happen at the API boundary (Pydantic model) before touching PostGIS. Use shapely for client-side validation of geometry validity and coordinate bounds. The security-hardening spec requires SSRF-like input validation on the AOI (max vertex count, max area, WGS84 bounds). -->

- [ ] 2.4 Implement `GET /control/v1/jobs/{job_id}` -- full job detail including current phase, status, `aoi` serialized as GeoJSON, parameters, and `_links` (self, events, checkpoints); return `HTTP 404` if `customer_id` does not match (depends on 2.1)
  <!-- files: api/routes/control/jobs.py (modify), api/models/control.py (modify — add JobDetailResponse model) -->

- [ ] 2.5 Implement `POST /control/v1/jobs/{job_id}/transition` -- body: `{ "expected_phase": "...", "expected_status": "...", "target_phase": "...", "target_status": "...", "reason": "..." }`; validate target (phase, status) pair against `VALID_PHASE_STATUS_PAIRS` before touching the database (return 422 if invalid); the `expected_phase` and `expected_status` fields are the TOCTOU guard -- execute via `StateBackend.transition()`; if `StateConflictError`, return `HTTP 409` with the current phase and status; on success, return `HTTP 200` with the new state; require `state:write` permission (depends on 2.4)
  <!-- files: api/routes/control/jobs.py (modify), api/models/control.py (modify — add TransitionRequest model) -->
  <!-- pattern: use the standard error envelope from api/models/errors.py — ConflictError for 409, ValidationError for 422 -->
  <!-- gotcha: The error response for 409 MUST include the current state in the details field so the caller can reconcile. Use ConflictError from api/models/errors.py (line 350) and pass current state in a custom details field. -->

- [ ] 2.6 Implement `PATCH /control/v1/jobs/{job_id}/parameters` -- body is a JSON merge-patch (`application/merge-patch+json`); validate keys against the algorithm's declared `parameter_schema` (from AlgorithmRegistry) before writing; reject if job is in terminal state (COMPLETE, FAILED, CANCELLED) with 409; return the updated parameters; emit `job.parameters_updated` event; require `state:write` permission (depends on 2.4)
  <!-- files: api/routes/control/jobs.py (modify) -->
  <!-- gotcha: AlgorithmMetadata.parameter_schema is a dict (JSON Schema) — use jsonschema.validate() to check submitted parameter values against it. The algorithm for a job needs to be determinable — either store algorithm_id on the jobs table or derive it from event_type via AlgorithmRegistry. -->

- [ ] 2.7 Implement `POST /control/v1/jobs/{job_id}/reasoning` -- append a reasoning entry to `job_events` with `actor` (from authenticated identity), `reasoning` text (max 64KB, reject null bytes per security spec), and optional `payload`; add `confidence` float field validated in [0.0, 1.0]; return `HTTP 201` with the inserted event's `event_seq` (depends on 2.4)
  <!-- files: api/routes/control/jobs.py (modify), api/models/control.py (modify — add ReasoningRequest model with Field(max_length=65536)) -->
  <!-- gotcha: The security-hardening spec requires reasoning field sanitization — reject null bytes (\x00), enforce max character length. Use a Pydantic validator on the model. -->

### Track B: Escalations + Tools (parallel with Track A after 2.1)

- [ ] 2.8 Implement escalation endpoints scoped to jobs: `POST /control/v1/jobs/{job_id}/escalations` (create — accept `severity` as enum LOW/MEDIUM/HIGH/CRITICAL, `reason`, optional `context` JSON max 16KB), `PATCH /control/v1/jobs/{job_id}/escalations/{escalation_id}` (resolve — accept `resolution` text, set `resolved_at = now()` and `resolved_by`), `GET /control/v1/jobs/{job_id}/escalations` (list — support `severity` filter); a second resolve attempt returns `HTTP 409`; emit escalation events to `job_events` on create and resolve; require `escalation:manage` permission for create/resolve, `state:read` for list (depends on 2.1)
  <!-- files: api/routes/control/escalations.py (modify), api/models/control.py (modify — add EscalationRequest, EscalationResponse models) -->
  <!-- pattern: follow api/models/errors.py ConflictError for 409 on duplicate resolve -->

- [ ] 2.9 Add `generate_tool_schema(algorithm_id: str) -> dict` function to `core/analysis/library/registry.py`; it reads the algorithm's `AlgorithmMetadata` fields (`id`, `name`, `description`, `parameter_schema`, `default_parameters`, `event_types`, `category`, `resources`, `validation`) and returns an OpenAI-compatible function-calling schema with `name`, `description`, `parameters` (JSON Schema derived from `parameter_schema`); add a `list_tool_schemas(exclude_deprecated=True) -> List[dict]` convenience method to `AlgorithmRegistry`. Register a `GET /control/v1/tools` endpoint that returns all available tool schemas for the authenticated tenant under a `{"tools": [...]}` response key (depends on 2.1)
  <!-- files: core/analysis/library/registry.py (modify — add generate_tool_schema function after line 210, add list_tool_schemas to AlgorithmRegistry class), api/routes/control/tools.py (modify) -->
  <!-- pattern: follow AlgorithmMetadata.to_dict() at line 124 for field serialization pattern -->
  <!-- gotcha: AlgorithmMetadata has a `deprecated` field (line 121) — filter out deprecated algorithms. The `parameter_schema` field (line 113) is already a dict that can be used as JSON Schema parameters directly. The function needs to normalize algorithm IDs with dots into valid function names (replace . with _). -->

- [ ] 2.10 Implement rate limiting middleware for the `/control/v1` router: use Redis sliding window counters keyed by API key (per-agent) and `customer_id` (per-customer); configure limits via `FIRSTLIGHT_RATE_LIMIT_PER_AGENT` (default 120 req/min), `FIRSTLIGHT_RATE_LIMIT_PER_CUSTOMER` (default 600 req/min), and `FIRSTLIGHT_RATE_LIMIT_BURST` (default 20); return `HTTP 429` with `Retry-After` header when exceeded; include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers on all responses (depends on 2.1, Phase 0 complete)
  <!-- files: api/rate_limit.py (modify — existing file has RateLimitConfig and infrastructure at line 32, extend with sliding window implementation), api/routes/control/__init__.py (modify — add rate limit dependency to control router) -->
  <!-- pattern: follow existing api/rate_limit.py RateLimitConfig pattern; reuse RateLimitError from api/models/errors.py (line 335) -->
  <!-- gotcha: api/rate_limit.py already exists with a RateLimitConfig class and memory fallback. Extend this rather than creating a new module. The existing rate_limit.py has burst_multiplier (line 41) — align the new burst config with this. The existing RateLimitError at api/models/errors.py line 335 already includes Retry-After header. -->

<!-- wave-gate: Phase 2 tests -->

### Section 2C: Tests (sequential, after Tracks A and B complete)

- [ ] 2.11 Write `tests/api/test_control_jobs.py` covering: create job with polygon AOI promotes to multipolygon, create job with invalid GeoJSON (out-of-bounds coords) returns 422, transition succeeds in expected state, transition returns 409 in wrong state (with current state in response body), PATCH parameters validates against algorithm schema and rejects unknown keys, cross-tenant access returns 404 (not 403), list jobs with phase filter, list jobs with bbox filter, paginated response has correct total/page fields (depends on 2.3, 2.5, 2.6)
  <!-- files: tests/api/test_control_jobs.py (new) -->
  <!-- pattern: follow tests/api/test_jwt_auth.py for FastAPI TestClient usage -->

- [ ] 2.12 Write `tests/api/test_control_escalations.py` covering: create escalation via `POST /control/v1/jobs/{job_id}/escalations`, resolve via `PATCH /control/v1/jobs/{job_id}/escalations/{escalation_id}`, duplicate resolve returns 409 with original resolution details, list filtered by severity, create requires escalation:manage permission (depends on 2.8)
  <!-- files: tests/api/test_control_escalations.py (new) -->

- [ ] 2.13 Write `tests/api/test_rate_limiting.py` covering: per-agent limit returns 429 with Retry-After header, per-customer aggregate limit triggers across multiple keys, burst allowance permits short spikes, rate limit headers (X-RateLimit-*) present on normal 200 responses, other tenants unaffected when one tenant is rate-limited (depends on 2.10)
  <!-- files: tests/api/test_rate_limiting.py (new) -->

- [ ] 2.14 Write `tests/api/test_control_tools.py` covering: GET /control/v1/tools returns tool schemas for all non-deprecated algorithms, each schema has name/description/parameters fields, deprecated algorithms are excluded, empty registry returns 200 with empty tools array and warning (depends on 2.9)
  <!-- files: tests/api/test_control_tools.py (new) -->

<!-- wave-gate: Phase 2 -->

---

## Phase 3: Event Stream and Partner API
<!-- execution: parallel -->

Goal: Partners (MAIA Analytics) can stream structured events in real time, inspect metrics, and receive webhooks. All event payloads use the CloudEvents v1.0 envelope.

### Track A: SSE Stream (sequential within track)

- [ ] 3.0 Install `taskiq`, `taskiq-fastapi`, `taskiq-redis` and add to `pyproject.toml` optional group `[control-plane]`; pin versions; create `workers/taskiq_app.py` defining a `TaskiqBroker` backed by Redis Streams (`RedisStreamBroker(redis_url)`); wire `taskiq-fastapi` lifespan startup into `api/main.py`; verify imports and broker connectivity in CI; update the worker Dockerfile to run `taskiq worker workers.taskiq_app:broker` (depends on Phase 0 complete -- Redis password must be injected via env)
  <!-- files: pyproject.toml (modify — add to [control-plane] group), workers/__init__.py (new), workers/taskiq_app.py (new), api/main.py (modify — add taskiq lifespan startup in lifespan() function at line 70), docker/worker/Dockerfile (modify — add taskiq worker command) -->
  <!-- gotcha: The workers/ directory does not exist — create it. The existing worker Dockerfile may need to be found at docker/worker/Dockerfile. The redis URL needs the REDIS_PASSWORD from Phase 0 credential changes. -->

- [ ] 3.1 Add a PostgreSQL NOTIFY trigger on `job_events` INSERT: `CREATE OR REPLACE FUNCTION notify_job_event() RETURNS TRIGGER AS $$ BEGIN PERFORM pg_notify('job_events', row_to_json(NEW)::text); RETURN NEW; END; $$ LANGUAGE plpgsql;` -- attach as `AFTER INSERT` trigger (depends on Phase 1 complete)
  <!-- files: db/migrations/002_job_events_notify.sql (new) -->

- [ ] 3.2 Create `api/routes/internal/` package; register under `/internal/v1` with tag `Partner Integration`; add `internal:read` permission to the `Permission` enum in `api/auth.py`; add a dedicated dependency `require_internal_scope` that checks for this permission (depends on Phase 0 complete)
  <!-- files: api/routes/internal/__init__.py (new), api/routes/internal/events.py (new — stub), api/routes/internal/webhooks.py (new — stub), api/routes/internal/metrics.py (new — stub), api/auth.py (modify — add INTERNAL_READ = "internal:read" to Permission enum), api/main.py (modify — include internal_router) -->
  <!-- pattern: follow api/routes/control/__init__.py pattern from Phase 2 -->

- [ ] 3.3 Implement `GET /internal/v1/events/stream` as an SSE endpoint using `asyncpg`'s `LISTEN` channel from task 3.1; wrap each event in a CloudEvents v1.0 envelope: `specversion: "1.0"`, `type: "io.firstlight.job.<event_type>"`, `source: "/jobs/{job_id}"`, `id: <event_seq>`, `time: <occurred_at as RFC 3339>`, `datacontenttype: "application/json"`, `firstlight_job_id`, `firstlight_customer_id`, `firstlight_phase`, `firstlight_status`, `data: <payload>`; support `Last-Event-ID` header for replay (query `job_events WHERE event_seq > $last_id ORDER BY event_seq`); support optional `customer_id` and `type` query params for filtering (depends on 3.1, 3.2)
  <!-- files: api/routes/internal/events.py (modify), core/events/cloudevents.py (new — CloudEvents envelope builder) -->
  <!-- gotcha: FastAPI SSE uses StreamingResponse with media_type="text/event-stream". Each SSE frame must have id: <event_seq>, event: <type>, data: <json>. The asyncpg LISTEN connection should be separate from the connection pool — use a dedicated connection for the LISTEN channel. -->

- [ ] 3.4 Add backpressure handling to the SSE endpoint: set a per-connection asyncio `Queue(maxsize=500)`; if the queue is full, drop the connection with a `503` comment frame and log the disconnect; test with a slow consumer (depends on 3.3)
  <!-- files: api/routes/internal/events.py (modify) -->

### Track B: Webhooks (parallel with Track A after 3.0 and 3.2)

- [ ] 3.5 Create `webhook_subscriptions` and `webhook_dlq` tables via SQL migration; `webhook_subscriptions`: `subscription_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `customer_id TEXT NOT NULL`, `target_url TEXT NOT NULL`, `secret_key TEXT NOT NULL`, `event_filter TEXT[] DEFAULT '{}'`, `created_at TIMESTAMPTZ DEFAULT now()`, `active BOOLEAN DEFAULT true`; `webhook_dlq`: `dlq_id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `subscription_id UUID NOT NULL REFERENCES webhook_subscriptions(subscription_id)`, `event_seq BIGINT NOT NULL`, `payload JSONB NOT NULL`, `last_error TEXT`, `attempt_count INT DEFAULT 0`, `failed_at TIMESTAMPTZ DEFAULT now()`; add `POST /internal/v1/webhooks` (register -- validate HTTPS URL, reject private/loopback IPs per SSRF spec), `GET /internal/v1/webhooks` (list -- cursor-based pagination with `limit` and `after` params, return `items`, `next_cursor`, `has_more`), and `DELETE /internal/v1/webhooks/{subscription_id}` (deregister) endpoints (depends on 3.2)
  <!-- files: db/migrations/003_webhook_tables.sql (new), api/routes/internal/webhooks.py (modify), api/models/internal.py (new — Pydantic models for webhook CRUD) -->
  <!-- gotcha: SSRF protection is required per security-hardening spec. Validate webhook URLs at registration time: reject http:// (require https://), reject URLs resolving to 127.0.0.0/8, ::1, 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16, 169.254.0.0/16, fe80::/10. Use socket.getaddrinfo() to resolve the hostname before registration. Do NOT follow HTTP redirects during delivery. -->

- [ ] 3.6 Implement webhook delivery as a Taskiq task `deliver_webhook` in `workers/tasks/webhooks.py`; payload: CloudEvents envelope (reuse builder from 3.3); signing: `X-FirstLight-Signature-256: sha256=HMAC-SHA256(secret_key, body)`; retry policy: 5 attempts, exponential backoff starting at 5 seconds and doubling with `random.uniform(0, 1)` jitter, capped at 5 minutes per interval (schedule: ~5s, ~10s, ~20s, ~40s, ~80s); after 5 failures, insert to `webhook_dlq` table; do NOT follow HTTP redirects during delivery (treat redirect as failure) (depends on 3.0, 3.5)
  <!-- files: workers/tasks/__init__.py (new), workers/tasks/webhooks.py (new) -->
  <!-- pattern: follow existing api/webhooks.py WebhookConfig for delivery configuration conventions -->
  <!-- gotcha: api/webhooks.py already has a WebhookConfig class with retry settings (line 33-40). Reuse those settings where applicable but override the backoff schedule to match the spec (5s base, not 30s). The existing webhooks.py has hmac signing infrastructure — reference but do not duplicate. -->

- [ ] 3.7 Add Redis idempotency key `webhook:delivered:{event_seq}:{subscription_id}` (TTL 24h) before delivering; skip delivery if key exists; this prevents duplicate delivery on Taskiq retry (depends on 3.6)
  <!-- files: workers/tasks/webhooks.py (modify) -->

### Track C: Metrics + Ops Endpoints (parallel with Tracks A and B after 3.2)

- [ ] 3.8 Create two materialized views via SQL migration, refreshed every 30 seconds via a background asyncio task registered in `api/main.py` lifespan:
  - `pipeline_health_metrics` view backing `GET /internal/v1/metrics`: response fields `jobs_completed_1h`, `jobs_failed_1h`, `jobs_completed_24h`, `jobs_failed_24h`, `p50_duration_s`, `p95_duration_s`, `active_sse_connections` (runtime counter, not from view), `webhook_success_rate_1h`; include `Cache-Control: max-age=30` and `X-Metrics-Refreshed-At` headers; must respond within 100ms
  - `queue_summary` view backing `GET /internal/v1/queue/summary`: response fields `per_phase_counts` (object keyed by phase), `stuck_count`, `awaiting_review_count`, `oldest_stuck_job` (object with `job_id` and `stuck_since` or null); stuck threshold configurable via `FIRSTLIGHT_STUCK_JOB_THRESHOLD_MINUTES` (default 30)
  (depends on 3.2)
  <!-- files: db/migrations/004_materialized_views.sql (new), api/routes/internal/metrics.py (modify), api/main.py (modify — add periodic refresh task to lifespan) -->
  <!-- gotcha: REFRESH MATERIALIZED VIEW CONCURRENTLY requires a UNIQUE index on the view. Add one. The background refresh task should use asyncpg to run the refresh, not block the event loop. -->

- [ ] 3.9 Implement `GET /internal/v1/escalations` -- list open escalations for the authenticated customer (reuses `escalations` table from Phase 2); support `severity` and `since` query params (depends on Phase 2 complete, 3.2)
  <!-- files: api/routes/internal/metrics.py (modify — or create api/routes/internal/escalations.py) -->

- [ ] 3.10 Convert `job_events` table to use PostgreSQL declarative partitioning (`PARTITION BY RANGE (occurred_at)`); create initial monthly partitions; add a scheduled maintenance task (via application-level scheduler using asyncio) that runs daily to: (a) create next month's partition if not exists, (b) detach partitions older than `FIRSTLIGHT_EVENT_RETENTION_DAYS` (default 90), (c) archive detached partitions as compressed pg_dump files, (d) drop archived partitions after verification; ensure `event_seq` BIGSERIAL uses a shared sequence across all partitions; verify `Last-Event-ID` replay works across partition boundaries (depends on 3.1, Phase 1 complete)
  <!-- files: db/migrations/005_partition_job_events.sql (new), workers/tasks/maintenance.py (new) -->
  <!-- gotcha: Partitioning an existing table requires: (1) create new partitioned table, (2) migrate data, (3) swap names. Cannot ALTER an existing table to add partitioning. The BIGSERIAL on event_seq needs CREATE SEQUENCE shared_event_seq; then each partition uses DEFAULT nextval('shared_event_seq'). This is a complex migration — consider making it a separate migration from 001_control_plane_schema.sql. -->

<!-- wave-gate: Phase 3 tests -->

### Section 3D: Tests (sequential, after all tracks complete)

- [ ] 3.11 Write `tests/api/test_event_stream.py` covering: SSE stream emits CloudEvents envelope on job_events INSERT, CloudEvents envelope has all required attributes (specversion, type, source, id, time, firstlight_* extensions), Last-Event-ID replay returns events after the given seq, slow consumer triggers 503 disconnect, customer_id scoping filters events (depends on 3.3, 3.4)
  <!-- files: tests/api/test_event_stream.py (new) -->

- [ ] 3.12 Write `tests/api/test_webhooks.py` covering: registration validates HTTPS URL, registration rejects private IP URLs (SSRF), webhook listing uses cursor-based pagination, HMAC signature matches expected, retry inserts to DLQ after 5 failures, idempotency key prevents double delivery, redirect response treated as failure (depends on 3.5, 3.6, 3.7)
  <!-- files: tests/api/test_webhooks.py (new) -->

<!-- wave-gate: Phase 3 -->

---

## Phase 4: Standards Integration
<!-- execution: parallel -->

Goal: FirstLight is accessible via OGC API Processes and STAC. Results are published as STAC Items. Task execution is managed by Taskiq.

### Track A: OGC Integration (sequential within track)

- [ ] 4.1 Install `pygeoapi` and `stac-fastapi-pgstac`, `pypgstac`; add to `pyproject.toml` optional group `[control-plane]`; pin versions; verify imports in CI
  <!-- files: pyproject.toml (modify — add to [control-plane] group) -->

- [ ] 4.2 Create `core/ogc/processors/` package; for each algorithm in `AlgorithmRegistry`, generate a `pygeoapi.process.base.BaseProcessor` subclass at startup: `id` = `AlgorithmMetadata.id`, `title` = `AlgorithmMetadata.name`, `description` = `AlgorithmMetadata.description`, `inputs` derived from `AlgorithmMetadata.parameter_schema`, `outputs` derived from `AlgorithmMetadata.outputs`; add vendor extension fields `x-firstlight-category`, `x-firstlight-resource-requirements`, `x-firstlight-reasoning`, `x-firstlight-confidence`, `x-firstlight-escalation` to the process description (depends on 2.9, 4.1)
  <!-- files: core/ogc/__init__.py (new), core/ogc/processors/__init__.py (new), core/ogc/processors/factory.py (new — dynamic BaseProcessor subclass factory) -->
  <!-- pattern: follow core/analysis/library/registry.py AlgorithmMetadata for field access -->
  <!-- gotcha: pygeoapi loads processors from a YAML config file or programmatic registration. The factory approach dynamically creates subclasses — verify pygeoapi supports this pattern (it may require a config file listing each processor). If config-file-based, generate a pygeoapi-local.yml at startup. -->

- [ ] 4.3 Mount pygeoapi as an ASGI sub-application at `/oapi` in `api/main.py` using `app.mount("/oapi", pygeoapi_starlette_app)`; configure pygeoapi to load processors from `core/ogc/processors/`; verify `GET /oapi/processes` returns the FirstLight algorithm list (depends on 4.2)
  <!-- files: api/main.py (modify — add app.mount for pygeoapi), core/ogc/config.py (new — generate pygeoapi config) -->
  <!-- gotcha: pygeoapi uses a Starlette-compatible ASGI app. The mount should not conflict with /api/v1/ or /control/v1/ routes. Test path isolation. -->

- [ ] 4.4 Configure OGC auth boundary: add OGC discovery paths (`GET /oapi/`, `GET /oapi/conformance`, `GET /oapi/processes`, `GET /oapi/processes/{id}`) to the `TenantMiddleware` auth exemption allowlist; ensure OGC execution endpoints (`POST /oapi/processes/{id}/execution`, `GET /oapi/jobs/*`) require authentication; disable pygeoapi's internal auth; document the auth boundary in `docs/design/api-surface-map.md` (depends on 4.3, Phase 0 complete)
  <!-- files: api/middleware.py (modify — update TenantMiddleware exemption allowlist), docs/design/api-surface-map.md (new) -->

### Track B: STAC Publishing (parallel with Track A)

- [ ] 4.5 Install pgSTAC schema into the existing PostgreSQL instance using `pypgstac migrate`; add the migration step to the Docker entrypoint init sequence; verify pgSTAC tables (`pgstac.items`, `pgstac.collections`) are created without conflicting with FirstLight app tables (depends on 1.3)
  <!-- files: db/migrations/006_pgstac_init.sql (new — or use pypgstac CLI in Docker entrypoint), docker/api/Dockerfile (modify — add pypgstac migrate step) -->
  <!-- gotcha: pgSTAC uses its own schema namespace which should not conflict with the public schema used by FirstLight app tables. Verify namespace isolation. pypgstac needs the DATABASE_URL to be set. -->

- [ ] 4.6 Mount `stac-fastapi-pgstac` at `/stac` in `api/main.py`; create one STAC Collection per event type (flood, wildfire, storm) with `id`, `description`, `extent`, `links`; run `pypgstac load` to register collections (depends on 4.5)
  <!-- files: api/main.py (modify — add app.mount for stac-fastapi), core/stac/__init__.py (new), core/stac/collections.py (new — collection definitions) -->

- [ ] 4.7 Write `publish_result_as_stac_item(job_id: str)` function in `core/stac/publisher.py`: read final products from `jobs` and `job_events`; build a STAC Item with `processing:lineage`, `processing:software`, `processing:datetime` extension fields; AOI becomes the Item `geometry` (from `ST_AsGeoJSON(aoi)`); add `derived_from` links for source imagery STAC Items if URIs were recorded during ingestion; register COG raster assets with `type=image/tiff; application=geotiff; profile=cloud-optimized` (reuse existing `COGConfig` output from `agents/reporting/products.py`); insert via `pypgstac.load` using upsert mode; call this function from the orchestrator after `COMPLETE` phase transition (depends on 4.6, Phase 2 complete)
  <!-- files: core/stac/publisher.py (new), agents/orchestrator/main.py (modify — add call to publish_result_as_stac_item after COMPLETE transition) -->
  <!-- pattern: follow agents/reporting/products.py for COGConfig conventions and output paths -->
  <!-- gotcha: The COGConfig in agents/reporting/products.py already produces COGs with 512x512 tiles, DEFLATE compression, and overview factors [2,4,8,16,32]. The STAC publisher should reference existing output files, NOT re-generate COGs. The stac-publishing spec confirms this. -->

### Track C: Taskiq Wiring + Integration (depends on 3.0 and Track A)

- [ ] 4.8 Register OGC process execution as Taskiq tasks alongside the existing `deliver_webhook` task; implement the `Prefer: respond-async` OGC job execution flow (return 201 with Location header, delegate to Taskiq); confirm task routing and priority isolation between webhook delivery and algorithm execution using Taskiq labels or separate queues (depends on 3.0, 4.3)
  <!-- files: workers/tasks/ogc_execution.py (new), core/ogc/processors/factory.py (modify — wire execute method to Taskiq) -->

<!-- wave-gate: Phase 4 tests -->

### Section 4D: Tests (sequential, after all tracks complete)

- [ ] 4.9 Write `tests/ogc/test_processors.py` covering: each registered algorithm appears in `GET /oapi/processes`, process description includes `x-firstlight-*` vendor fields, submitting an OGC execute request with `Prefer: respond-async` returns 201 with Location header, sync execute for long-running algorithm returns 400 (depends on 4.3, 4.8)
  <!-- files: tests/ogc/__init__.py (new), tests/ogc/test_processors.py (new) -->

- [ ] 4.10 Write `tests/stac/test_publisher.py` covering: completed job produces a valid STAC Item, Item geometry matches job AOI (GeoJSON round-trip preserved to 5 decimal places), processing extension fields are present (processing:lineage, processing:software, processing:datetime), `GET /stac/collections/{event_type}/items/{job_id}` returns the item, re-publishing same result upserts (no duplicate), derived_from links present when source URIs exist (depends on 4.7)
  <!-- files: tests/stac/__init__.py (new), tests/stac/test_publisher.py (new) -->

- [ ] 4.11 Write `tests/ogc/test_state_based_tool_filtering.py` covering: (a) job in `ANALYZING` phase returns only analysis-category algorithm tool schemas, excluding discovery and reporting algorithms; (b) job transitioning from `DISCOVERING` to `ANALYZING` causes the tool list to update accordingly; (c) job in a phase with no mapped algorithms returns an empty tool list rather than all algorithms as fallback (depends on 4.2, 4.3, Phase 2 complete)
  <!-- files: tests/ogc/test_state_based_tool_filtering.py (new) -->
  <!-- gotcha: This test requires a mapping from JobPhase to AlgorithmCategory. This mapping is not yet defined in any task — the coder should define it in core/ogc/processors/factory.py or agents/orchestrator/state_model.py. At minimum: DISCOVERING -> no algorithms, ANALYZING -> baseline + advanced, REPORTING -> none (or reporting-specific if they exist). -->

- [ ] 4.12 End-to-end smoke test in `tests/e2e/test_control_plane_e2e.py`: (1) POST a job via `/control/v1/jobs`, (2) verify SSE emits `job.created`, (3) transition through phases to COMPLETE via `/control/v1/jobs/{id}/transition`, (4) verify STAC item appears at `/stac/collections/...`, (5) verify OGC process list includes the algorithm used; requires PostGIS, Redis, and full service stack (depends on all prior phases complete)
  <!-- files: tests/e2e/test_control_plane_e2e.py (new) -->
  <!-- pattern: follow tests/e2e/test_tile_validation.py for e2e test structure -->
  <!-- gotcha: Mark with @pytest.mark.e2e and @pytest.mark.integration. This needs the full Docker stack running. Consider using docker-compose for test fixtures. -->
