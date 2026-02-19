# Tasks: add-context-data-lakehouse

<!-- phase: add-context-data-lakehouse -->

Ordered checklist for implementing the Context Data Lakehouse across four phases.
Each task is independently verifiable. Dependencies are called out inline.

Plane Issue: FIRSTLIGHT-3

---

## Phase 0: Schema Design + Migration
<!-- execution: sequential -->

Goal: All context tables and the junction table exist in PostGIS, ready for inserts.

- [x] 0.1 Write the SQL migration `db/migrations/007_context_data.sql` containing all five tables:
  - `datasets` table: `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `source VARCHAR(100) NOT NULL`, `source_id VARCHAR(500) NOT NULL`, `geometry GEOMETRY(MULTIPOLYGON, 4326) NOT NULL`, `properties JSONB DEFAULT '{}'`, `acquisition_date TIMESTAMPTZ NOT NULL`, `cloud_cover NUMERIC(5, 2)`, `resolution_m NUMERIC(8, 2)`, `bands TEXT[]`, `file_path TEXT`, `ingested_at TIMESTAMPTZ DEFAULT now()`, `ingested_by_job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL`; add `UNIQUE (source, source_id)`, GIST index on `geometry`, B-tree index on `acquisition_date`
  - `context_buildings` table: `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `source VARCHAR(100) NOT NULL`, `source_id VARCHAR(500) NOT NULL`, `geometry GEOMETRY(POLYGON, 4326) NOT NULL`, `properties JSONB DEFAULT '{}'`, `ingested_at TIMESTAMPTZ DEFAULT now()`, `ingested_by_job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL`; add `UNIQUE (source, source_id)`, GIST index on `geometry`
  - `context_infrastructure` table: `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `source VARCHAR(100) NOT NULL`, `source_id VARCHAR(500) NOT NULL`, `geometry GEOMETRY(GEOMETRY, 4326) NOT NULL`, `properties JSONB DEFAULT '{}'`, `ingested_at TIMESTAMPTZ DEFAULT now()`, `ingested_by_job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL`; add `UNIQUE (source, source_id)`, GIST index on `geometry`
  - `context_weather` table: `id UUID PRIMARY KEY DEFAULT gen_random_uuid()`, `source VARCHAR(100) NOT NULL`, `source_id VARCHAR(500) NOT NULL`, `geometry GEOMETRY(POINT, 4326) NOT NULL`, `properties JSONB DEFAULT '{}'`, `observation_time TIMESTAMPTZ NOT NULL`, `ingested_at TIMESTAMPTZ DEFAULT now()`, `ingested_by_job_id UUID REFERENCES jobs(job_id) ON DELETE SET NULL`; add `UNIQUE (source, source_id)`, GIST index on `geometry`, B-tree index on `observation_time`
  - `job_context_usage` junction table: `job_id UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE`, `context_table VARCHAR(50) NOT NULL CHECK (context_table IN ('datasets', 'context_buildings', 'context_infrastructure', 'context_weather'))`, `context_id UUID NOT NULL`, `usage_type VARCHAR(20) NOT NULL CHECK (usage_type IN ('ingested', 'reused'))`, `linked_at TIMESTAMPTZ DEFAULT now()`; add `PRIMARY KEY (job_id, context_table, context_id)`, B-tree index on `(context_table, context_id)` for reverse lookups
  All statements must use `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` for idempotency; all FKs to `jobs(job_id)` use `ON DELETE SET NULL` for context tables (context data survives job deletion) and `ON DELETE CASCADE` for the junction table (usage records are meaningless without the job)
  <!-- files: db/migrations/007_context_data.sql (new) -->
  <!-- pattern: follow db/migrations/001_control_plane_schema.sql for style (section comments, constraint naming, index naming). See also 005_partition_job_events.sql for the most recent migration style. -->
  <!-- gotcha: The jobs table is created in 001_control_plane_schema.sql and must exist before this migration runs. Migration ordering is manual (no Alembic) -- the 007 numbering enforces order. Existing migrations 000-006 are already applied on feature/llm-control-plane branch. Also: the B-tree index on (source, source_id) is implicit via the UNIQUE constraint -- do NOT add a separate index for it. -->

- [x] 0.2 Verify the migration runs cleanly against a fresh PostGIS instance: spin up `docker compose up -d postgres`, apply migrations 000-006, then apply 007; confirm all tables exist with correct columns, constraints, and indexes; confirm `INSERT ... ON CONFLICT (source, source_id) DO NOTHING` works for each table; confirm the junction table FK to `jobs(job_id)` works (depends on 0.1)
  <!-- files: db/migrations/007_context_data.sql (verify) -->
  <!-- test: manual verification via psql or a script. The coder should run: psql -h localhost -p 5433 -U postgres -d firstlight_test -f db/migrations/007_context_data.sql and then exercise INSERT/ON CONFLICT for each table. -->
  <!-- gotcha: Use port 5433 for the test PostGIS instance (same as test_postgis_backend.py convention at tests/state/test_postgis_backend.py line 53). The test database name is firstlight_test. -->

<!-- wave-gate: Phase 0 -->

---

## Phase 1: Context Table Models + Repository Layer
<!-- execution: sequential -->

Goal: Python models and a repository class that handle insert-or-link and spatial queries for all context types.

- [x] 1.1 Define Pydantic models for context records in `core/context/models.py`: `DatasetRecord`, `BuildingRecord`, `InfrastructureRecord`, `WeatherRecord`, `ContextResult` (with `context_id: UUID`, `usage_type: Literal['ingested', 'reused']`), `ContextSummary` (with per-table counts and total for a job), and `JobContextUsage` (with `job_id`, `context_table`, `context_id`, `usage_type`, `linked_at`); each record model must have a `source: str`, `source_id: str`, and a GeoJSON-compatible `geometry` field (Dict[str, Any]); `DatasetRecord` additionally has `acquisition_date`, `cloud_cover`, `resolution_m`, `bands`, `file_path`; `WeatherRecord` additionally has `observation_time`
  <!-- files: core/context/__init__.py (new), core/context/models.py (new) -->
  <!-- pattern: follow api/models/control.py for Pydantic model conventions (type hints, Field validators, Optional fields). See JobSummary at api/models/control.py line 23 for the Field(description=...) pattern. -->
  <!-- gotcha: The geometry field should be typed as Dict[str, Any] (GeoJSON dict), NOT a shapely object. All geometry conversion to/from PostGIS happens in the repository layer using ST_GeomFromGeoJSON/ST_AsGeoJSON. Keep models pure Pydantic with no PostGIS dependencies. -->

- [x] 1.2 Implement `ContextRepository` in `core/context/repository.py` using asyncpg (same connection pool as PostGISStateBackend); implement these methods:
  - `store_dataset(job_id, record) -> ContextResult` -- INSERT with ON CONFLICT DO NOTHING RETURNING id; if no id returned, SELECT existing row id; INSERT into job_context_usage with usage_type='ingested' or 'reused' accordingly
  - `store_building(job_id, record) -> ContextResult` -- same pattern
  - `store_infrastructure(job_id, record) -> ContextResult` -- same pattern
  - `store_weather(job_id, record) -> ContextResult` -- same pattern
  - `store_batch(job_id, table_name, records) -> List[ContextResult]` -- batch insert for performance; same dedup pattern per row
  - Internal helper `_insert_or_link(job_id, table, columns, values, source, source_id) -> ContextResult` to DRY the pattern
  All geometry values are passed as GeoJSON strings and converted via `ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)` (with `ST_Multi()` wrapping for datasets only); all inserts happen within a single transaction per store call (depends on 1.1)
  <!-- files: core/context/repository.py (new) -->
  <!-- pattern: follow agents/orchestrator/backends/postgis_backend.py for asyncpg pool management. Key patterns to replicate: (1) constructor accepts optional pool parameter (line 86), (2) connect()/close() lifecycle (lines 105-121), (3) _ensure_pool() guard (lines 123-129), (4) DSN construction from individual params (line 108). -->
  <!-- gotcha: (1) asyncpg executemany does NOT support RETURNING clauses -- it silently drops them. For store_batch, loop individual inserts within a single transaction (use `async with pool.acquire() as conn: async with conn.transaction():` -- see postgis_backend.py lines 269-270 for the pattern). (2) The asyncpg pool should be shared with the existing PostGIS state backend, not a separate pool. ContextRepository should accept an optional pool: asyncpg.Pool parameter in its constructor. (3) Use ST_SetSRID(ST_GeomFromGeoJSON($1), 4326) to ensure SRID is set -- plain ST_GeomFromGeoJSON may not set SRID from GeoJSON. (4) Geometry must be serialized to JSON string via json.dumps() before passing to asyncpg (see postgis_backend.py line 178: aoi_json_str = json.dumps(aoi_geojson)). -->

- [x] 1.3 Add query methods to `ContextRepository`:
  - `query_datasets(bbox, date_start, date_end, source, limit, offset) -> List[DatasetRecord]` -- spatial query using `ST_Intersects(geometry, ST_MakeEnvelope(west, south, east, north, 4326))` with optional date range and source filters
  - `query_buildings(bbox, limit, offset) -> List[BuildingRecord]` -- spatial query
  - `query_infrastructure(bbox, type_filter, limit, offset) -> List[InfrastructureRecord]` -- spatial query with optional JSONB property filter (`properties->>'type' = $1`)
  - `query_weather(bbox, time_start, time_end, limit, offset) -> List[WeatherRecord]` -- spatial + temporal query
  - `get_job_context_summary(job_id) -> ContextSummary` -- aggregate counts from job_context_usage grouped by context_table and usage_type
  - `get_lakehouse_stats() -> dict` -- total row counts per table, total spatial extent, distinct sources
  All spatial queries use the GIST index; geometry is returned as GeoJSON via `ST_AsGeoJSON(geometry)::text` and parsed via `json.loads()` (depends on 1.2)
  <!-- files: core/context/repository.py (modify) -->
  <!-- pattern: follow the _SELECT_COLS pattern from postgis_backend.py (line 53) for SELECT column lists with ST_AsGeoJSON. Also follow the dynamic WHERE clause builder pattern from postgis_backend.py list_jobs() (lines 324-377). -->
  <!-- gotcha: (1) query_datasets needs a total count for pagination -- use a COUNT(*) subquery or window function (COUNT(*) OVER() AS total_count). (2) ST_MakeEnvelope(west, south, east, north, 4326) is the correct bbox approach -- do not use ST_GeomFromText. (3) LIMIT/OFFSET parameters must be appended after all filter params in the parameterized query (see postgis_backend.py list_jobs() for the idx-tracking pattern). -->

- [x] 1.4 Write `tests/context/test_context_repository.py` covering: store_dataset inserts new row and returns usage_type='ingested', store_dataset with same source+source_id returns usage_type='reused', store_building with valid polygon, store_weather with point geometry, query_buildings by bbox returns only intersecting buildings, query_datasets by date range, get_job_context_summary returns correct counts, batch insert with mixed new and existing records; run against PostGIS container; mark with `@pytest.mark.integration` (depends on 1.2, 1.3)
  <!-- files: tests/context/__init__.py (new), tests/context/test_context_repository.py (new) -->
  <!-- pattern: follow tests/state/test_postgis_backend.py for PostGIS test fixture patterns. Key elements: (1) POSTGIS_HOST/PORT/DB/USER/PASSWORD from env vars with defaults (lines 52-56), (2) pytestmark with pytest.mark.integration + skipif for asyncpg (lines 42-45), (3) pytest_asyncio fixtures for pool creation and schema setup. -->
  <!-- gotcha: Tests need a job_id in the jobs table for FK constraints. The fixture must: (a) apply all migrations 000-007, (b) insert a test job into the jobs table using PostGISStateBackend.create_job() or raw SQL, (c) tear down after tests. Use the SAMPLE_AOI GeoJSON from test_postgis_backend.py as the test AOI. -->

<!-- wave-gate: Phase 1 -->

---

## Phase 2: Pipeline Integration
<!-- execution: sequential -->

Goal: The pipeline stores context data into PostGIS as it flows through discovery and analysis. No change to pipeline behavior or outputs -- storage is a side effect.

- [x] 2.1 Integrate `ContextRepository` into `DiscoveryAgent`: after STAC catalog queries return scene metadata (around line 698 of agents/discovery/main.py, after `catalog_results = await self._catalog_manager.query_catalogs(...)`), call `store_dataset()` for each discovered scene; pass the current `job_id`, map discovery result fields to `DatasetRecord` (depends on Phase 1 complete)
  <!-- files: agents/discovery/main.py (modify -- add context storage after line 698 where catalog_results is populated) -->
  <!-- pattern: follow the existing try/except pattern in agents/discovery/main.py for catalog query failures. Wrap context storage in try/except to ensure discovery flow is never blocked. -->
  <!-- gotcha: CRITICAL -- The DiscoveryAgent does NOT receive raw pystac Items. It receives CatalogQueryResult objects (agents/discovery/catalog.py line 46) containing List[DiscoveryResult] (core/data/discovery/base.py line 14). DiscoveryResult has: dataset_id, provider, source_uri, acquisition_time, resolution_m, cloud_cover_percent, metadata (optional dict). Map these fields: provider -> source, dataset_id -> source_id, acquisition_time -> acquisition_date, cloud_cover_percent -> cloud_cover, resolution_m -> resolution_m. There is NO geometry field on DiscoveryResult -- you will need to extract it from the metadata dict if present, or skip the geometry. If geometry is unavailable, you cannot insert into datasets (geometry is NOT NULL). Check if CatalogQueryResult.metadata or DiscoveryResult.metadata contains spatial_extent or geometry. The ContextRepository must be injected via the agent's constructor -- add an optional context_repo parameter to DiscoveryAgent.__init__. -->

- [x] 2.1b **Recon: Verify pipeline context data fetch paths.** Before implementing 2.2-2.4, inspect `agents/pipeline/main.py`, `agents/discovery/main.py`, and any context enrichment modules to document: (a) which context data types are actually fetched today, (b) file locations and entry points for each fetch path, (c) which fetch paths don't exist yet and need stubs. Record findings as comments in a `docs/design/context-fetch-paths.md` (new file). This informs whether 2.2-2.4 are "wire existing fetch to repository" or "create stub fetch + wire to repository." (depends on Phase 1 complete)
  <!-- files: docs/design/context-fetch-paths.md (new -- recon findings), agents/pipeline/main.py (read-only), agents/discovery/main.py (read-only) -->
  <!-- gotcha: EM audit result -- the pipeline agent at agents/pipeline/main.py does NOT have any existing building, infrastructure, or weather fetch paths. A grep for "buildings|infrastructure|weather|context|footprint|overture|osm" in agents/pipeline/main.py returned zero matches (only "context" in a generic "execution context" sense). Tasks 2.2-2.4 will almost certainly need stubs. The recon task should confirm this and document the decision. -->

- [x] 2.2 Integrate `ContextRepository` into `PipelineAgent` for building footprints: based on 2.1b findings (expected: no existing fetch path), create a stub fetcher at `core/context/stubs.py` that generates synthetic building data for a given bbox for demo purposes; wire to `store_building()` or `store_batch()`; same fire-and-forget pattern as 2.1 (depends on 2.1b)
  <!-- files: core/context/stubs.py (new -- synthetic data generators), agents/pipeline/main.py (modify -- add context storage hook) -->
  <!-- pattern: follow the fire-and-forget try/except pattern from task 2.1 -->
  <!-- gotcha: The stub should generate realistic building polygons inside the given bbox -- use simple rectangles with random offsets. Each synthetic building should have source="synthetic", source_id="synth_bldg_{uuid}", and a valid POLYGON geometry. Real OSM/Overture integration is a future task. -->

- [x] 2.3 Integrate `ContextRepository` into `PipelineAgent` for infrastructure data: same pattern as 2.2 but for critical infrastructure facilities; based on 2.1b findings (expected: no existing fetch path), add infrastructure stub to `core/context/stubs.py`; store via `store_infrastructure()` (depends on 2.1b)
  <!-- files: core/context/stubs.py (modify -- add infrastructure generator), agents/pipeline/main.py (modify) -->
  <!-- gotcha: Infrastructure stubs should generate POINT geometries (hospitals, fire stations, etc.) and a few POLYGON geometries (hospital campuses) to exercise the GEOMETRY(GEOMETRY, 4326) mixed type. Use source="synthetic". -->

- [x] 2.4 Integrate `ContextRepository` into `PipelineAgent` for weather observations: based on 2.1b findings (expected: no existing fetch path), add weather stub to `core/context/stubs.py`; store via `store_weather()` (depends on 2.1b)
  <!-- files: core/context/stubs.py (modify -- add weather generator), agents/pipeline/main.py (modify) -->
  <!-- gotcha: Weather stubs should generate POINT geometries with realistic observation_time values and properties containing temperature, precipitation, wind. Use source="synthetic". -->

- [x] 2.5 Write integration tests in `tests/context/test_pipeline_integration.py` covering: running a discovery through the DiscoveryAgent stores dataset records in PostGIS, running the same discovery twice for overlapping areas results in 'reused' usage_type entries in the junction table, context storage failure does not fail the pipeline job (error is logged and swallowed); use mock STAC responses and a live PostGIS instance (depends on 2.1, 2.2, 2.3, 2.4)
  <!-- files: tests/context/test_pipeline_integration.py (new) -->
  <!-- pattern: follow tests/state/test_postgis_backend.py for PostGIS fixture patterns. For mocking the discovery agent, use unittest.mock to patch the catalog_manager. -->
  <!-- gotcha: The DiscoveryAgent requires multiple constructor dependencies (catalog_manager, provider_registry, etc.). You may need to mock these or create a minimal test harness. Consider testing the ContextRepository integration layer directly (call store_dataset with sample data) rather than spinning up the full DiscoveryAgent. -->

<!-- wave-gate: Phase 2 -->

---

## Phase 3: Query API + Demo
<!-- execution: parallel -->

Goal: MAIA can query accumulated context data. A demo script demonstrates the lakehouse effect.

### Track A: Query Endpoints

- [x] 3.1 Create `api/routes/control/context.py` with a `context_router` scoped under `/context`; register the router in `api/routes/control/__init__.py` via `control_router.include_router(context_router)`; add `Permission.CONTEXT_READ` to the `Permission` enum in `api/auth.py` and add it to `ROLE_PERMISSIONS` for `readonly` and above (`readonly`, `user`, `operator`, `admin`), following the `state:read` pattern; require `CONTEXT_READ` for all context endpoints (depends on Phase 2 complete)
  <!-- files: api/routes/control/context.py (new), api/routes/control/__init__.py (modify -- add import and include_router for context_router), api/auth.py (modify -- add CONTEXT_READ = "context:read" to Permission enum at line 113, add Permission.CONTEXT_READ to readonly/user/operator role sets in ROLE_PERMISSIONS) -->
  <!-- pattern: follow api/routes/control/jobs.py (line 47: router = APIRouter(prefix="/jobs", tags=["LLM Control - Jobs"])) for router structure. Follow api/routes/control/__init__.py (lines 50-52) for router registration. Follow api/auth.py Permission enum (lines 75-113) and ROLE_PERMISSIONS (lines 117-158) for permission pattern. -->
  <!-- gotcha: The context router prefix should be "/context" (not "/control/v1/context") because the parent control_router already has prefix="/control/v1" (see api/routes/control/__init__.py line 43-47). So endpoint paths resolve to /control/v1/context/datasets etc. Also: the backend needs a ContextRepository instance -- follow the _get_backend() / _get_connected_backend() pattern from api/routes/control/jobs.py (lines 55-81) but instantiate ContextRepository instead. The ContextRepository and PostGISStateBackend should share the same DB connection parameters from api/config.py DatabaseSettings. -->

- [x] 3.2 Implement `GET /control/v1/context/datasets` -- query accumulated datasets; support query params: `bbox` (west,south,east,north), `date_start`, `date_end`, `source`, `page` (1-based, default 1), `page_size` (default 50, max 200); return paginated response with `items` (list of dataset records with geometry as GeoJSON), `total`, `page`, `page_size`; delegate to `ContextRepository.query_datasets()` (depends on 3.1)
  <!-- files: api/routes/control/context.py (modify), api/models/context.py (new -- Pydantic response models for context endpoints) -->
  <!-- pattern: follow api/routes/control/jobs.py list_jobs() (lines 89-208) for bbox parsing, pagination, and response construction. Follow api/models/control.py PaginatedJobsResponse (line 37) for paginated response model. -->
  <!-- gotcha: Reuse the bbox parsing logic from jobs.py list_jobs() (lines 133-158). Consider extracting it to a shared utility since both context and jobs endpoints parse bbox the same way. -->

- [x] 3.3 Implement `GET /control/v1/context/buildings` -- query building footprints by bbox; same pagination pattern as 3.2; return building records with geometry as GeoJSON (depends on 3.1)
  <!-- files: api/routes/control/context.py (modify), api/models/context.py (modify) -->

- [x] 3.4 Implement `GET /control/v1/context/infrastructure` -- query infrastructure by bbox with optional `type` filter (passed as `properties->>'type'` query); same pagination pattern (depends on 3.1)
  <!-- files: api/routes/control/context.py (modify) -->

- [x] 3.5 Implement `GET /control/v1/context/weather` -- query weather observations by bbox and time range (`time_start`, `time_end`); same pagination pattern (depends on 3.1)
  <!-- files: api/routes/control/context.py (modify) -->

- [x] 3.6 Implement `GET /control/v1/context/summary` -- return lakehouse statistics: total row count per context table, total spatial extent (bounding box of all data), list of distinct sources per table, total `job_context_usage` entries grouped by usage_type; delegate to `ContextRepository.get_lakehouse_stats()` (depends on 3.1)
  <!-- files: api/routes/control/context.py (modify) -->

- [x] 3.7 Implement `GET /control/v1/jobs/{job_id}/context` -- return context usage summary for a specific job: per-table counts of ingested vs reused items, total counts; delegate to `ContextRepository.get_job_context_summary()` (depends on 3.1)
  <!-- files: api/routes/control/jobs.py (modify -- add GET /jobs/{job_id}/context route after existing reasoning endpoint, around line 543) -->
  <!-- pattern: follow the get_job() endpoint pattern (api/routes/control/jobs.py lines 278-320) for tenant scoping and backend lifecycle. -->
  <!-- gotcha: This endpoint adds a route to the existing jobs router, not the context router. The URL pattern is /control/v1/jobs/{job_id}/context (under the jobs prefix). The coder must verify the job belongs to the authenticated customer before returning context data (same tenant scoping as get_job at line 299). -->

### Track B: Demo + Verification

- [x] 3.8 Create a demo script `scripts/demo_lakehouse.py` that demonstrates the lakehouse effect:
  1. Submit Job A for a bounding box (e.g., Houston flood zone) via `/control/v1/jobs`
  2. Simulate pipeline execution by inserting synthetic context data (100 buildings, 5 infrastructure facilities, 10 weather observations, 3 satellite scenes) via the ContextRepository
  3. Query `/control/v1/context/summary` to show accumulated data
  4. Submit Job B for an overlapping bounding box
  5. Simulate pipeline for Job B -- show that context rows are reused (usage_type='reused')
  6. Query `/control/v1/jobs/{job_b_id}/context` to show reuse stats
  7. Print a summary: "Job A ingested 118 context items. Job B reused 95 of them."
  (depends on Track A complete, Phase 2 complete)
  <!-- files: scripts/demo_lakehouse.py (new) -->
  <!-- pattern: follow scripts/demo_control_plane.py for demo script structure (existing demo script for the control plane). Use httpx or requests for API calls. -->
  <!-- gotcha: The demo needs a running API server and PostGIS. Either use docker-compose or connect to localhost. The demo needs an API key for authenticated requests -- follow the dev key creation pattern from scripts/demo_control_plane.py. The Houston flood zone bbox is approximately: -95.5, 29.5, -95.0, 30.0. -->

<!-- wave-gate: Phase 3 tests -->

### Section 3C: Tests (sequential, after both tracks complete)

- [x] 3.9 Write `tests/api/test_context_query.py` covering: GET /context/datasets returns paginated results, bbox filter returns only intersecting datasets, date range filter works, GET /context/buildings returns GeoJSON geometry in response, GET /context/summary returns correct row counts, GET /jobs/{id}/context returns usage summary with ingested/reused counts, empty database returns 200 with empty items array (depends on Track A)
  <!-- files: tests/api/test_context_query.py (new) -->
  <!-- pattern: follow tests/api/test_control_jobs.py for API test structure (FastAPI TestClient, auth fixtures, PostGIS test database). -->
  <!-- gotcha: Tests need seeded data in the context tables. Create a fixture that inserts known context rows and junction entries before running queries. The permission check test should verify that a user without CONTEXT_READ gets 403. -->

- [x] 3.10 Write `tests/context/test_lakehouse_effect.py` covering: two jobs with overlapping AOIs result in shared context rows with correct usage_type values, non-overlapping jobs do not share context, junction table correctly tracks provenance (depends on Phase 2 complete, Track A)
  <!-- files: tests/context/test_lakehouse_effect.py (new) -->
  <!-- pattern: follow tests/state/test_postgis_backend.py for PostGIS integration test patterns. -->

<!-- wave-gate: Phase 3 -->
