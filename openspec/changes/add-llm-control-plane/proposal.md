# Change: Add LLM Control Plane

## Why
FirstLight's pipeline state is trapped inside a SQLite-backed state manager and in-memory agent registries. LLM agents cannot read pipeline state, trigger transitions, or inject reasoning through HTTP — they must be co-located in the Python process. This blocks integration with MAIA Analytics and any external orchestration layer. Externalizing state into PostGIS and exposing it via HTTP endpoints turns FirstLight from a monolithic pipeline into a controllable service that LLM agents can drive through standard tool-use patterns.

## What Changes
- **State externalization:** Migrate pipeline state from SQLite (`StateManager`) to PostGIS with a Phase+Status model that reconciles 6 existing state enums into a unified schema
- **Control API:** New `/control/v1/` endpoints for LLM agents to read job state, trigger transitions, adjust parameters, and submit reasoning
- **Event stream:** New `/internal/v1/` endpoints (SSE stream, metrics, queue summary, escalations) for partner integration using CloudEvents envelope
- **OGC compliance:** Mount pygeoapi at `/oapi/` to expose algorithms as OGC API Processes
- **Result publishing:** Mount stac-fastapi-pgstac at `/stac/` to publish analysis results as STAC Items
- **Task queue:** Replace planned Celery with Taskiq (async-native, 5x faster, FastAPI DI integration)
- **Security hardening:** **BREAKING** — Enable auth by default, add tenant isolation via `customer_id` scoping, fix infrastructure credentials
- **PostGIS schema:** MULTIPOLYGON(4326) with geography casts, validity constraints, computed area columns

## Impact
- Affected specs: state-management (new), postgis-schema (new), control-api (new), event-stream (new), ogc-integration (new), stac-publishing (new), security-hardening (new)
- Affected code: `agents/orchestrator/state.py`, `agents/base.py`, `api/config.py`, `api/routes/`, `docker-compose.yml`, `core/analysis/library/registry.py`
- Breaking: Auth enabled by default changes deployment behavior for existing users
- Dependencies: PostGIS extension in PostgreSQL, asyncpg driver, Taskiq, pygeoapi, stac-fastapi-pgstac, cloudevents SDK

## Research Completed
Seven review documents produced by specialized agents:
1. `docs/design/control-plane-review.md` — Initial gap analysis of wireframe vs. codebase
2. `docs/design/state-reconciliation.md` — Phase+Status pattern, StateBackend ABC, DualWriteBackend migration
3. `docs/design/geospatial-review.md` — PostGIS schema corrections (Tobler agent)
4. `docs/design/control-plane-security-review.md` — 25 findings, 3 blockers (Security Expert)
5. `docs/research/2026-02-18-standards-and-patterns.md` — OGC/Taskiq/pgSTAC/webhook patterns (E-S-R-E-V-E-R)
6. `docs/design/event-stream-integration.md` — SSE, CloudEvents, webhook delivery architecture
7. `docs/research/2026-02-18-maia-analytics-geospatial-control-plane.md` — MAIA/industry context
