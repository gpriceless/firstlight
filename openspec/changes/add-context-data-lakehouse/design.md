## Context

FirstLight's LLM Control Plane (migrations 000-006) externalizes job state, events, escalations, webhooks, and metrics into PostGIS. The pipeline processes satellite imagery, building footprints, infrastructure, and weather data during each job -- but this context data is ephemeral. It exists only in memory or temporary files during pipeline execution and is discarded after the job completes.

This design adds PostGIS tables that capture context data as it flows through the pipeline, creating a persistent geospatial knowledge base. The design sits on top of the existing control plane schema and reuses its PostGIS patterns (asyncpg, GIST indexes, SRID 4326, geography casts).

### Stakeholders
- **MAIA Analytics:** Needs to query accumulated context for situational awareness (e.g., "what buildings are near this flood zone?")
- **Pipeline:** Stores data during normal execution; benefits from reuse on overlapping jobs
- **Demo audience:** Needs to see the "lakehouse effect" -- data accumulates, future jobs get faster

## Goals / Non-Goals

### Goals
- Store all data the pipeline fetches during analysis in durable PostGIS tables
- Deduplicate on insert using natural keys from source systems
- Track which jobs used which context data, and whether it was fresh or reused
- Expose query endpoints for MAIA to search accumulated context by geometry
- Single migration file, consistent with existing migration sequence

### Non-Goals
- Freshness/staleness logic (future product decision)
- Smart spatial gap analysis (future optimization)
- Roads, population, administrative boundaries (future context types)
- SSE events for context accumulation (silent growth, queryable on demand)
- Multi-tenant context isolation (context is shared for demo)

## Schema Design

### Table: `datasets`
Satellite scenes discovered and used by the pipeline. Each row represents one scene from a STAC catalog.

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `UUID` | `PK DEFAULT gen_random_uuid()` | Internal ID |
| `source` | `VARCHAR(100)` | `NOT NULL` | Origin catalog: `earth_search`, `planetary_computer`, `copernicus` |
| `source_id` | `VARCHAR(500)` | `NOT NULL` | STAC Item ID (e.g., `S2A_MSIL2A_20240115T...`) |
| `geometry` | `GEOMETRY(MULTIPOLYGON, 4326)` | `NOT NULL` | Scene footprint, promoted via `ST_Multi()` |
| `properties` | `JSONB` | `DEFAULT '{}'` | Full STAC Item properties (platform, constellation, etc.) |
| `acquisition_date` | `TIMESTAMPTZ` | `NOT NULL` | Scene acquisition datetime |
| `cloud_cover` | `NUMERIC(5, 2)` | | Cloud cover percentage (0-100) |
| `resolution_m` | `NUMERIC(8, 2)` | | Ground sample distance in metres |
| `bands` | `TEXT[]` | | Available band names (e.g., `{B02, B03, B04, B08}`) |
| `file_path` | `TEXT` | | Local path to downloaded data (if acquired) |
| `ingested_at` | `TIMESTAMPTZ` | `DEFAULT now()` | When this row was created |
| `ingested_by_job_id` | `UUID` | `FK -> jobs(job_id)` | Which job first brought this in |
| **UNIQUE** | | `(source, source_id)` | Natural key for dedup |

Indexes: GIST on `geometry`, B-tree on `(source, source_id)`, B-tree on `acquisition_date`.

### Table: `context_buildings`
Building footprints from OSM, Overture, or other sources. Each row is one building polygon.

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `UUID` | `PK DEFAULT gen_random_uuid()` | Internal ID |
| `source` | `VARCHAR(100)` | `NOT NULL` | `osm`, `overture`, `microsoft_footprints` |
| `source_id` | `VARCHAR(500)` | `NOT NULL` | OSM way ID, Overture feature ID, etc. |
| `geometry` | `GEOMETRY(POLYGON, 4326)` | `NOT NULL` | Building footprint (single polygon) |
| `properties` | `JSONB` | `DEFAULT '{}'` | Tags: building type, height, name, etc. |
| `ingested_at` | `TIMESTAMPTZ` | `DEFAULT now()` | |
| `ingested_by_job_id` | `UUID` | `FK -> jobs(job_id)` | |
| **UNIQUE** | | `(source, source_id)` | |

Indexes: GIST on `geometry`.

Note: Buildings use `POLYGON` not `MULTIPOLYGON` because individual building footprints are single polygons. If a source provides multi-part buildings, they should be decomposed into individual rows via `ST_Dump()`.

### Table: `context_infrastructure`
Critical infrastructure facilities (hospitals, fire stations, power plants, bridges, etc.).

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `UUID` | `PK DEFAULT gen_random_uuid()` | Internal ID |
| `source` | `VARCHAR(100)` | `NOT NULL` | `osm`, `overture`, `hifld` |
| `source_id` | `VARCHAR(500)` | `NOT NULL` | Feature ID from source |
| `geometry` | `GEOMETRY(GEOMETRY, 4326)` | `NOT NULL` | Point or polygon (facilities vary) |
| `properties` | `JSONB` | `DEFAULT '{}'` | Type, name, capacity, etc. |
| `ingested_at` | `TIMESTAMPTZ` | `DEFAULT now()` | |
| `ingested_by_job_id` | `UUID` | `FK -> jobs(job_id)` | |
| **UNIQUE** | | `(source, source_id)` | |

Indexes: GIST on `geometry`.

Note: Infrastructure uses `GEOMETRY(GEOMETRY, 4326)` (generic geometry) because facilities can be points (fire stations) or polygons (hospital campuses). The GIST index handles both.

### Table: `context_weather`
Weather observations from NOAA, ERA5, or other meteorological sources.

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `id` | `UUID` | `PK DEFAULT gen_random_uuid()` | Internal ID |
| `source` | `VARCHAR(100)` | `NOT NULL` | `noaa`, `era5`, `openmeteo` |
| `source_id` | `VARCHAR(500)` | `NOT NULL` | Station ID + timestamp hash, or grid cell ID |
| `geometry` | `GEOMETRY(POINT, 4326)` | `NOT NULL` | Observation location (station or grid centroid) |
| `properties` | `JSONB` | `DEFAULT '{}'` | Temperature, precipitation, wind, humidity, etc. |
| `observation_time` | `TIMESTAMPTZ` | `NOT NULL` | When the observation was recorded |
| `ingested_at` | `TIMESTAMPTZ` | `DEFAULT now()` | |
| `ingested_by_job_id` | `UUID` | `FK -> jobs(job_id)` | |
| **UNIQUE** | | `(source, source_id)` | |

Indexes: GIST on `geometry`, B-tree on `observation_time`.

### Table: `job_context_usage` (Junction)
Links jobs to context rows they consumed. Tracks provenance and the "lakehouse effect."

| Column | Type | Constraints | Notes |
|--------|------|-------------|-------|
| `job_id` | `UUID` | `NOT NULL FK -> jobs(job_id)` | The consuming job |
| `context_table` | `VARCHAR(50)` | `NOT NULL` | Enum-like: `datasets`, `context_buildings`, `context_infrastructure`, `context_weather` |
| `context_id` | `UUID` | `NOT NULL` | ID of the row in the context table |
| `usage_type` | `VARCHAR(20)` | `NOT NULL CHECK (usage_type IN ('ingested', 'reused'))` | Was this freshly fetched or reused? |
| `linked_at` | `TIMESTAMPTZ` | `DEFAULT now()` | When the link was created |
| **PK** | | `(job_id, context_table, context_id)` | Composite primary key |

Indexes: B-tree on `(context_table, context_id)` for reverse lookups ("which jobs used this building?"), B-tree on `job_id` for forward lookups ("what context did this job use?").

## Decisions

### D1: Option B Ownership Model -- Tag at Ingest, Dedup by Natural Key
**Decision:** Context data is tagged with `ingested_by_job_id` on first insert. Subsequent jobs that encounter the same data (same `source` + `source_id`) link to the existing row via `job_context_usage` with `usage_type = 'reused'` instead of re-inserting.
**Why:** This is the simplest model that achieves the lakehouse effect. No complex ownership tracking, no merge logic, no conflict resolution. The junction table provides full provenance.
**Alternatives:** Per-job copies (wastes storage, no reuse), shared-nothing with soft references (complex, no dedup benefit).

### D2: INSERT ... ON CONFLICT DO NOTHING for Dedup
**Decision:** All context table inserts use `INSERT INTO ... ON CONFLICT (source, source_id) DO NOTHING RETURNING id`. If the row already exists, a follow-up `SELECT id FROM ... WHERE source = $1 AND source_id = $2` retrieves the existing ID for the junction table link.
**Why:** Simplest dedup strategy. No upsert needed because context data from the same source+ID is assumed identical (immutable at source). No freshness checks needed for demo scope.
**Alternatives:** `ON CONFLICT DO UPDATE SET properties = EXCLUDED.properties` (updates on re-encounter; unnecessary complexity for demo), application-level EXISTS check (race condition under concurrent inserts).

### D3: Geometry Types Per Table
**Decision:** Use the most specific geometry type per table rather than generic `GEOMETRY` everywhere.
- `datasets` -> `MULTIPOLYGON` (scene footprints can be multi-part)
- `context_buildings` -> `POLYGON` (individual building footprints)
- `context_infrastructure` -> `GEOMETRY` (mixed points and polygons)
- `context_weather` -> `POINT` (station locations or grid centroids)

**Why:** Specific types prevent accidental insertion of wrong geometry types and enable PostGIS to optimize storage and indexing. Infrastructure is the exception because facilities genuinely vary between point and polygon representations.
**Alternatives:** All `GEOMETRY` (loses type safety), all `MULTIPOLYGON` with promotion (overkill for points).

### D4: Silent Accumulation -- No SSE Events
**Decision:** Context data inserts do not emit events to `job_events` or the SSE stream. The lakehouse grows silently and is queryable on demand via the context query API.
**Why:** Context inserts are high-volume (hundreds of buildings per job). Streaming every insert would flood the event stream with noise that MAIA does not need. MAIA queries context when it needs situational awareness, not as a live feed.
**Alternatives:** Batch summary events ("job ingested 347 buildings") -- nice-to-have but not needed for demo.

### D5: No Freshness Logic
**Decision:** No TTL, no "stale data" checks, no re-fetch triggers. Data inserted by a previous job is always considered valid for reuse.
**Why:** Freshness is a product decision that depends on data type (satellite imagery goes stale in days, building footprints are stable for years). Deferring this keeps the demo scope tight. The `ingested_at` timestamp is stored so freshness logic can be added later without schema changes.
**Alternatives:** Configurable TTL per source (premature complexity), always re-fetch and upsert (defeats the lakehouse purpose).

## Pipeline Integration Pattern

### Discovery Phase (datasets)
When the discovery agent queries STAC catalogs, it receives scene metadata. Before returning results to the orchestrator:

```
1. For each discovered scene:
   a. INSERT INTO datasets ... ON CONFLICT (source, source_id) DO NOTHING RETURNING id
   b. If RETURNING is empty (conflict), SELECT id WHERE source = $1 AND source_id = $2
   c. INSERT INTO job_context_usage (job_id, 'datasets', context_id, usage_type)
      - usage_type = 'ingested' if step (a) returned an id
      - usage_type = 'reused' if step (b) was needed
2. Continue with normal discovery flow (no change to return values)
```

### Pipeline Phase (buildings, infrastructure, weather)
When the pipeline agent fetches supplementary data for analysis context:

```
1. For each context data item:
   a. INSERT INTO context_{type} ... ON CONFLICT DO NOTHING RETURNING id
   b. If conflict, SELECT existing id
   c. INSERT INTO job_context_usage with appropriate usage_type
2. Continue with normal pipeline flow
```

### Repository Layer
A new `ContextRepository` class wraps the insert-or-link pattern:

```python
class ContextRepository:
    async def store_dataset(self, job_id: UUID, dataset: DatasetRecord) -> ContextResult
    async def store_building(self, job_id: UUID, building: BuildingRecord) -> ContextResult
    async def store_infrastructure(self, job_id: UUID, facility: InfrastructureRecord) -> ContextResult
    async def store_weather(self, job_id: UUID, observation: WeatherRecord) -> ContextResult
    async def query_buildings(self, bbox: BBox, limit: int = 100) -> List[BuildingRecord]
    async def query_infrastructure(self, bbox: BBox, limit: int = 100) -> List[InfrastructureRecord]
    async def query_datasets(self, bbox: BBox, ...) -> List[DatasetRecord]
    async def query_weather(self, bbox: BBox, ...) -> List[WeatherRecord]
    async def get_job_context_summary(self, job_id: UUID) -> ContextSummary
```

`ContextResult` includes `context_id`, `usage_type` ('ingested' or 'reused'), and the stored record.

## Migration Strategy

Single SQL migration file: `db/migrations/007_context_data.sql`

This migration:
1. Creates all four context tables with constraints and indexes
2. Creates the `job_context_usage` junction table
3. Is idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`)
4. References `jobs(job_id)` via FK, which exists from `001_control_plane_schema.sql`

No data migration needed -- these are new tables with no existing data.

The migration numbering follows the existing sequence: 000 (customer_id), 001 (control plane), 002 (notify trigger), 003 (webhooks), 004 (materialized views), 005 (partitioning), 006 (pgSTAC).

## Query API Design

New endpoints under `/control/v1/context/`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/control/v1/context/datasets` | GET | Query accumulated datasets by bbox, date range, source |
| `/control/v1/context/buildings` | GET | Query building footprints by bbox |
| `/control/v1/context/infrastructure` | GET | Query infrastructure by bbox, type filter |
| `/control/v1/context/weather` | GET | Query weather observations by bbox, time range |
| `/control/v1/context/summary` | GET | Get lakehouse statistics (row counts, spatial extent, sources) |
| `/control/v1/jobs/{job_id}/context` | GET | Get context usage summary for a specific job |

All spatial queries accept `bbox` as `west,south,east,north` (WGS84), consistent with the existing jobs list endpoint. Results are paginated (default 50, max 200). Geometry is returned as GeoJSON.

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| High insert volume during pipeline execution | Increased job latency | Batch inserts (50-100 rows per statement), async fire-and-forget pattern |
| Context tables grow unbounded | Disk usage | `ingested_at` column enables future cleanup; demo scope is small |
| No FK on `job_context_usage.context_id` | Orphan references if context rows deleted | Acceptable for demo; context rows are never deleted in current design |
| Junction table `context_table` is a string, not a real FK | No referential integrity to context tables | Trade-off for simplicity; a polymorphic FK would require table inheritance |
| Concurrent inserts from parallel jobs | Potential deadlocks on unique constraint | `ON CONFLICT DO NOTHING` is deadlock-safe in PostgreSQL |

## Open Questions

1. **Batch insert size:** What is the optimal batch size for context inserts? Start with 50 rows per `INSERT ... VALUES` statement, tune based on demo performance.
2. ~~**Context query pagination:** Should context queries support cursor-based pagination (like webhooks) or offset-based (like jobs list)?~~ **RESOLVED** -- offset-based pagination (`page`, `page_size`) for simplicity. Cursor-based is a future optimization for large result sets.

---

## Engineering Notes (added by EM during readiness review)

### EN1: DiscoveryResult does NOT have geometry

The proposal and task 2.1 assume the discovery agent receives STAC Items with `item.geometry`. This is incorrect. The discovery agent works with `DiscoveryResult` objects (`core/data/discovery/base.py` line 14), which have:
- `dataset_id: str`
- `provider: str`
- `source_uri: str`
- `acquisition_time: datetime`
- `resolution_m: float`
- `cloud_cover_percent: Optional[float]`
- `metadata: Optional[Dict[str, Any]]`

There is NO `geometry` field. The coder implementing task 2.1 must either:
(a) Check if `metadata` contains a geometry/spatial_extent field and use it, OR
(b) Skip datasets without geometry (log and continue), OR
(c) Derive geometry from the request's spatial bbox (the area being queried) as a rough proxy.

Option (a) is preferred if the data is there. The recon in task 2.1b should also look at what the `CatalogQueryResult.metadata` contains.

### EN2: Pipeline has NO existing context fetch paths

A grep of `agents/pipeline/main.py` for buildings, infrastructure, weather, OSM, Overture, NOAA found zero matches. The pipeline agent assembles and executes analysis pipelines but does not fetch supplementary context data today.

Tasks 2.2-2.4 will need synthetic data stubs (`core/context/stubs.py`). The design's integration pattern ("when the pipeline agent fetches supplementary data") describes a future state. For the demo, the stubs generate realistic synthetic data and store it via `ContextRepository`.

### EN3: asyncpg pool sharing

The `ContextRepository` and `PostGISStateBackend` should share the same asyncpg connection pool in production. Both accept an optional `pool` parameter in their constructors (the pattern is already established in `PostGISStateBackend.__init__` at `agents/orchestrator/backends/postgis_backend.py` line 86).

In the API layer, the `_get_backend()` helper in `api/routes/control/jobs.py` creates a new backend per request (lines 55-74). This is acceptable for a demo but creates a pool per request. The context endpoints should follow the same pattern. A future optimization would be to create the pool once in the FastAPI lifespan and share it via `app.state`.

### EN4: Permission enum addition

Adding `CONTEXT_READ = "context:read"` to the Permission enum in `api/auth.py` at line ~113. Insert it after `INTERNAL_READ` to maintain grouping. Add to `ROLE_PERMISSIONS`:
- `readonly` set (line 147): add `Permission.CONTEXT_READ`
- `user` set (line 137): add `Permission.CONTEXT_READ`
- `operator` set (line 119): add `Permission.CONTEXT_READ`
- `admin` already gets all permissions via `set(Permission)` (line 118)

### EN5: Context router registration

In `api/routes/control/__init__.py`, add:
```python
from api.routes.control.context import router as context_router
control_router.include_router(context_router)
```
After the existing `tools_router` include at line 52. The context_router prefix should be `"/context"` because `control_router` already has `prefix="/control/v1"`.

### EN6: Total count for paginated queries

The context query endpoints need total counts for pagination. Options:
(a) Run a separate `SELECT COUNT(*)` query (simple, two round trips)
(b) Use `COUNT(*) OVER() AS total_count` window function (one round trip, but scans all matching rows even if LIMIT is small)
(c) Use `SELECT COUNT(*) ... ; SELECT ... LIMIT ... OFFSET ...` in a single transaction

Option (a) is simplest and matches the pattern in `api/routes/control/jobs.py` (which fetches all jobs then slices -- though that pattern is inefficient for large sets). For the context tables, option (b) is recommended to avoid fetching all rows.

### EN7: Branch strategy

This work builds on `feature/llm-control-plane` which has migrations 000-006 and all control plane code. Create branch `feature/context-data-lakehouse` from `feature/llm-control-plane`. Phase 3 Track A and Track B can be parallel but they are small enough that worktrees are unnecessary -- sequential execution within a single branch is cleaner for this scope (10 tasks, no file conflicts between tracks). See full branching plan in the EM verdict.
