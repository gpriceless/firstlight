# Change: Add Context Data Lakehouse

## Why
FirstLight's pipeline fetches satellite imagery, building footprints, infrastructure data, and weather observations during every analysis job -- then throws it away. Each job pays the full latency and API cost of re-fetching the same data that a previous job already retrieved for an overlapping area. There is no queryable knowledge base that accumulates over time, and the LLM (MAIA) cannot ask "what buildings are in this area?" without triggering a new pipeline run.

Adding PostGIS-backed context tables turns every job's data fetches into durable, deduplicated rows. Over time, the database silently grows into a geospatial lakehouse. Future jobs reuse existing context instead of re-fetching. MAIA can query accumulated context directly for situational awareness without running a full analysis.

## What Changes
- **Context tables:** Four new tables in PostGIS (`datasets`, `context_buildings`, `context_infrastructure`, `context_weather`) storing data the pipeline pulls during analysis
- **Junction table:** `job_context_usage` links each job to the context rows it consumed, tracking whether data was freshly ingested or reused from a prior job
- **Dedup on insert:** All context tables use `INSERT ... ON CONFLICT (source, source_id) DO NOTHING` so duplicate data from overlapping jobs is absorbed without error
- **Migration:** Single SQL migration `007_context_data.sql` added to the existing `db/migrations/` sequence
- **Pipeline integration:** Ingestion and discovery agents store context data as it flows through the pipeline
- **Query API:** New `/control/v1/context/` endpoints let MAIA query accumulated context by geometry and type

## Impact
- Affected specs: context-data (new capability)
- Affected code: `agents/discovery/main.py` (store discovered datasets), `agents/pipeline/main.py` (store context data during ingestion), `api/routes/control/` (new context query endpoints), `db/migrations/` (new migration)
- Non-breaking: Existing jobs, API surfaces, and CLI are unchanged. New tables are additive.
- Dependencies: Builds on existing PostGIS schema from `001_control_plane_schema.sql` (jobs table FK), existing asyncpg patterns from the control plane backends

## Non-Goals
- **Freshness/staleness logic:** No TTL, no "is this data too old to reuse?" checks. That is a future product decision.
- **Smart spatial gap analysis:** No "fetch only the tiles we don't already have" optimization. Full re-fetch with dedup-on-insert is sufficient for demo.
- **Additional context types:** Roads, population, administrative boundaries are future additions. Demo scope is datasets, buildings, infrastructure, weather.
- **SSE events for lakehouse:** The lakehouse grows silently. MAIA does not receive streamed events when context rows are inserted. Lakehouse data is queryable on demand.
- **Multi-tenant isolation of context:** For the demo, context data is shared across all jobs (deduped by natural key). Tenant-scoped context isolation is a future consideration.

## Success Criteria
- All four context tables created via migration and accepting inserts
- Pipeline stores context data during normal job execution without impacting job latency
- Jobs that overlap geographically with prior jobs reuse existing context rows (verified via `job_context_usage.usage_type = 'reused'`)
- MAIA can query `/control/v1/context/buildings?bbox=...` and receive accumulated building footprints
- Demo script shows the lakehouse effect: first job ingests data, second overlapping job reuses it
