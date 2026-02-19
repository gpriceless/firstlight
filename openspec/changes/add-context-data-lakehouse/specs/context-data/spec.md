## ADDED Requirements

### Requirement: Context Data Tables
The system SHALL maintain four PostGIS tables for storing geospatial context data acquired during pipeline execution: `datasets` (satellite scenes), `context_buildings` (building footprints), `context_infrastructure` (critical facilities), and `context_weather` (meteorological observations). Each table SHALL have a UUID primary key, a `source` column identifying the origin system, a `source_id` column containing the natural key from that system, a `geometry` column with a GIST index (SRID 4326), a `properties` JSONB column for flexible attributes, an `ingested_at` timestamp, and an `ingested_by_job_id` foreign key to the jobs table. Each table SHALL enforce a UNIQUE constraint on `(source, source_id)` for deduplication.

#### Scenario: Dataset scene stored during discovery
- **WHEN** the discovery agent finds a satellite scene in a STAC catalog
- **THEN** the scene metadata is stored in the `datasets` table with source, source_id, geometry, acquisition_date, and properties populated from the STAC Item

#### Scenario: Building footprint stored during pipeline
- **WHEN** the pipeline fetches building footprints from OSM or Overture for analysis context
- **THEN** each building is stored in `context_buildings` with its polygon geometry and source attributes

#### Scenario: Duplicate context data absorbed on insert
- **WHEN** a second job encounters the same source+source_id as an existing context row
- **THEN** the insert is absorbed via `ON CONFLICT DO NOTHING` and no error is raised

---

### Requirement: Job Context Usage Junction
The system SHALL maintain a `job_context_usage` junction table that links each job to the context rows it consumed during execution. Each row SHALL record the `job_id`, `context_table` (name of the context table), `context_id` (UUID of the context row), `usage_type` (either `ingested` for freshly fetched data or `reused` for pre-existing data), and `linked_at` timestamp. The composite primary key SHALL be `(job_id, context_table, context_id)`.

#### Scenario: Freshly ingested context linked to job
- **WHEN** a job fetches data that does not yet exist in the context tables
- **THEN** the new context row is inserted and a junction entry with `usage_type = 'ingested'` is created

#### Scenario: Reused context linked to job
- **WHEN** a job encounters data that already exists in a context table (matching source+source_id)
- **THEN** no new context row is inserted and a junction entry with `usage_type = 'reused'` links the job to the existing row

#### Scenario: Job context summary
- **WHEN** MAIA queries the context usage for a completed job
- **THEN** the response includes per-table counts of ingested versus reused context items

---

### Requirement: Silent Lakehouse Accumulation
Context data SHALL accumulate silently over time as jobs execute. The system SHALL NOT emit SSE events or job_events entries when context rows are inserted. Context data SHALL be queryable on demand via API endpoints but SHALL NOT be streamed to partners. Over time, the context tables SHALL grow into a geospatial knowledge base that reduces redundant data fetching for overlapping analysis areas.

#### Scenario: Context accumulates across multiple jobs
- **WHEN** three jobs are executed for overlapping geographic areas over time
- **THEN** the context tables contain the union of all unique context data across all three jobs, with no duplicate rows

#### Scenario: No events emitted for context inserts
- **WHEN** the pipeline stores context data during job execution
- **THEN** no entries are added to the `job_events` table and no SSE events are emitted for the context inserts

---

### Requirement: Context Query API
The system SHALL expose query endpoints under `/control/v1/context/` that allow MAIA to search accumulated context data by geographic bounding box. The endpoints SHALL support pagination and return geometry as GeoJSON. Available endpoints SHALL include datasets (with date range and source filters), buildings (bbox only), infrastructure (with type filter), weather (with time range), a lakehouse summary (row counts, spatial extent, sources), and a per-job context usage summary. All context query endpoints SHALL require the `context:read` permission.

#### Scenario: Query buildings by bounding box
- **WHEN** MAIA queries `/control/v1/context/buildings?bbox=-95.5,29.5,-95.0,30.0`
- **THEN** the response contains paginated building footprints whose geometry intersects the bounding box, with geometry serialized as GeoJSON

#### Scenario: Query lakehouse summary
- **WHEN** MAIA queries `/control/v1/context/summary`
- **THEN** the response contains total row counts per context table, distinct source systems, and total spatial extent of all accumulated data

#### Scenario: Empty lakehouse returns valid response
- **WHEN** MAIA queries a context endpoint before any jobs have run
- **THEN** the response is HTTP 200 with an empty items array and zero counts

---

### Requirement: Pipeline Context Storage
The pipeline SHALL store context data into PostGIS as a non-blocking side effect during normal job execution. Context storage failures SHALL be logged but SHALL NOT fail the pipeline job. The discovery agent SHALL store dataset metadata after STAC catalog queries. The pipeline agent SHALL store building footprints, infrastructure data, and weather observations as they are fetched during analysis preparation.

#### Scenario: Context storage does not block pipeline
- **WHEN** the PostGIS context insert fails (e.g., connection timeout)
- **THEN** the error is logged, the pipeline continues normally, and the job is not marked as failed

#### Scenario: Discovery agent stores datasets
- **WHEN** the discovery agent completes STAC catalog queries
- **THEN** all discovered scenes are stored in the `datasets` table before returning results to the orchestrator

---

### Requirement: Context Data Migration
The context data schema SHALL be created via a single SQL migration file (`007_context_data.sql`) that follows the existing migration sequence (000-006). The migration SHALL be idempotent using `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS`. The migration SHALL reference the `jobs` table via foreign keys, requiring that migration `001_control_plane_schema.sql` has already been applied.

#### Scenario: Migration applies cleanly on fresh database
- **WHEN** migrations 000-006 have been applied and 007 is executed
- **THEN** all five tables are created with correct columns, constraints, indexes, and foreign keys

#### Scenario: Migration is idempotent
- **WHEN** migration 007 is applied twice
- **THEN** the second application succeeds without error and the schema is unchanged
