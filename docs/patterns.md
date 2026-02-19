# Code Patterns -- FirstLight

## Backend

### Agent System
- **Shape**: BaseAgent ABC -> Specialized Agent -> Delegates to domain modules
- **Exemplar**: `agents/orchestrator/main.py` (OrchestratorAgent)
- **Where used**: discovery, pipeline, quality, reporting agents
- **Notes**: Agents use AgentRegistry + MessageBus for coordination; state tracked via StateManager

### State Management (PostGIS backend)
- **Shape**: StateBackend ABC -> PostGISStateBackend (asyncpg pool, atomic transitions)
- **Exemplar**: `agents/orchestrator/backends/postgis_backend.py`
- **Where used**: Control plane API, job state, SQLite adapter
- **Notes**: Constructor accepts optional pool param (line 86). Pool lifecycle: connect()/close(). Guard: _ensure_pool(). DSN from individual params. Conditional UPDATE ... RETURNING for atomic transitions.

### State Management (legacy SQLite)
- **Shape**: ExecutionStage enum + ExecutionState dataclass -> StateManager (SQLite)
- **Exemplar**: `agents/orchestrator/state.py`
- **Where used**: OrchestratorAgent, CLI workflow
- **Notes**: 6 independent state enums exist (see docs/design/state-reconciliation.md)

### Algorithm Registry
- **Shape**: AlgorithmMetadata dataclass -> AlgorithmRegistry (in-memory, loaded from YAML)
- **Exemplar**: `core/analysis/library/registry.py`
- **Where used**: Pipeline execution, intent resolution, API catalog
- **Notes**: Fields: id, name, category, event_types, parameter_schema, outputs, resources, validation

### Configuration
- **Shape**: Pydantic BaseSettings with env_prefix -> nested settings objects -> lru_cache singleton
- **Exemplar**: `api/config.py` (Settings, DatabaseSettings, RedisSettings, AuthSettings)
- **Where used**: All API and service code
- **Notes**: env_prefix pattern: FIRSTLIGHT_, DATABASE_, REDIS_, AUTH_, STORAGE_, CORS_

### Repository Layer (asyncpg)
- **Shape**: Repository class with optional pool param -> connect()/close() lifecycle -> _ensure_pool() guard
- **Exemplar**: `agents/orchestrator/backends/postgis_backend.py`
- **Where used**: PostGISStateBackend, ContextRepository (new)
- **Notes**: Geometry as GeoJSON string via json.dumps(), insert with ST_SetSRID(ST_GeomFromGeoJSON($1), 4326), return with ST_AsGeoJSON(geometry)::text. Parameterized queries with $N placeholders and idx tracking for dynamic WHERE clauses.

## API

### Route Organization
- **Shape**: APIRouter per domain -> aggregated in parent __init__.py -> included in FastAPI app
- **Exemplar**: `api/routes/control/__init__.py` (control_router aggregates jobs, escalations, tools sub-routers)
- **Where used**: /api/v1/* routes, /control/v1/* routes, /internal/v1/* routes
- **Notes**: Three top-level prefixes: /api/v1 (public), /control/v1 (LLM agents), /internal/v1 (partners)

### Error Handling
- **Shape**: ErrorCode enum + ErrorResponse model + APIError exception hierarchy -> registered handlers
- **Exemplar**: `api/models/errors.py`
- **Where used**: All API error responses
- **Notes**: Existing pattern already has CONFLICT, RATE_LIMITED, VALIDATION_ERROR codes

### Auth System
- **Shape**: Permission enum (namespace:action) + ROLE_PERMISSIONS dict + UserContext dataclass + APIKey dataclass
- **Exemplar**: `api/auth.py`
- **Where used**: Route dependencies
- **Notes**: 22+ permissions, 5 roles (admin, operator, user, readonly, anonymous); customer_id on UserContext and APIKey

### Middleware Stack
- **Shape**: BaseHTTPMiddleware subclass -> setup_middleware(app, settings) in api/middleware.py
- **Exemplar**: `api/middleware.py` (CorrelationIdMiddleware, TenantMiddleware)
- **Where used**: api/main.py via setup_middleware()

### Pydantic Response Models
- **Shape**: BaseModel per response type -> paginated wrapper with items/total/page/page_size
- **Exemplar**: `api/models/control.py` (JobSummary, PaginatedJobsResponse)
- **Where used**: All API responses
- **Notes**: Field(description=...) on every field. Paginated responses: items list + total + page + page_size.

## Test Structure
- **Convention**: `tests/test_<domain>.py` for unit tests; `tests/api/test_<feature>.py` for API tests; `tests/state/test_*.py` for backend tests; `tests/e2e/` for end-to-end
- **Exemplar**: `tests/state/test_postgis_backend.py` (PostGIS integration), `tests/api/test_control_jobs.py` (API tests)
- **Notes**: pytest markers: integration, e2e, flood, wildfire, schema. PostGIS tests use port 5433, env var config, @pytest.mark.integration.

## Infrastructure
- **Docker**: 3 compose files, postgis/postgis:15-3.4-alpine
- **Migrations**: Raw SQL in `db/migrations/NNN_name.sql` (manual ordering, no Alembic). Idempotent via CREATE TABLE/INDEX IF NOT EXISTS.
- **Migration sequence**: 000 (customer_id), 001 (control plane schema), 002 (notify trigger), 003 (webhooks), 004 (materialized views), 005 (partitioning), 006 (pgSTAC)
