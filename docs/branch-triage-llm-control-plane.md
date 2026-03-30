# Branch Triage: feature/llm-control-plane

**Date**: 2026-03-18
**Branch**: `feature/llm-control-plane` vs `main`
**Scope**: 77 commits, 143 files changed, +27,069 / -6,985 lines
**Board directive**: LLM control plane deprecated. MAIA-specific showcase deprecated. DB store and data collection backend could be interesting. Move away from anything created for MAIA.

---

## Category 1: KEEP — Core FirstLight Improvements

These changes are independent of the LLM control plane and MAIA. They improve the core platform.

### 1a. Security Scrub & PII Removal (CLEAN CHERRY-PICK)

| File | Status | Commit | What Changed |
|------|--------|--------|-------------|
| AUDIT_ADDENDUM.md | Deleted | `54d8464` | Removed internal audit doc |
| AUDIT_REPORT.md | Deleted | `54d8464` | Removed internal audit doc |
| IMPLEMENTATION_SPECS.md | Deleted | `54d8464` | Removed internal spec doc |
| OPENSPEC_PRODUCTION.md | Deleted | `54d8464` | Removed internal production spec |
| QA_REPORT.md | Deleted | `54d8464` | Removed internal QA report |
| REQUIREMENTS_ANALYSIS.md | Deleted | `54d8464` | Removed internal requirements doc |
| TASK_R2.4.1_COMPLETION_SUMMARY.md | Deleted | `54d8464` | Removed task completion doc |
| WORK_ASSIGNMENTS.md | Deleted | `54d8464` | Removed work assignments doc |
| docs/R2.2.1_COMPLETION_SUMMARY.md | Deleted | `54d8464` | Removed completion summary |
| docs/TRACK_B_ROADMAP.md | Deleted | `54d8464` | Removed track B roadmap |
| docs/qa/QA_REPORT_R2_BATCH1.md | Deleted | `54d8464` | Removed QA report |
| docs/qa/QA_REPORT_R2_BATCH2.md | Deleted | `54d8464` | Removed QA report |
| docs/qa/QA_REPORT_R2_FINAL.md | Deleted | `54d8464` | Removed QA report |
| docs/specs/REPORT_2.0_WORK_ASSIGNMENTS.md | Deleted | `54d8464` | Removed work assignments |
| README.md | Modified | `54d8464` | Repo URL: personal → org |
| GUIDELINES.md | Modified | `54d8464` | Repo URL: personal → org |
| PROJECT_STATUS.md | Modified | `54d8464` | Repo URL: personal → org |
| CLAUDE.md | Modified | `54d8464`, `dc3c527` | Removed email/git creds, added OpenSpec block |
| .gitignore | Modified | `54d8464` | Block internal docs from commit |
| api/models/errors.py | Modified | `54d8464` | URL → example.com |

**Commits**: `54d8464`, `dc3c527`
**Cherry-pick clean?**: YES — these two commits are self-contained security scrubs.

### 1b. Bug Fix — Broken In-Memory Store Imports (CLEAN CHERRY-PICK)

| File | Status | Commit | What Changed |
|------|--------|--------|-------------|
| api/routes/products.py | Modified | `2803a76` | Fixed broken `_events_store` import, switched to DB lookup via `db_session.get_event()` |
| api/routes/status.py | Modified | `2803a76` | Same fix — DB lookup instead of broken in-memory store import |

**Commit**: `2803a76`
**Cherry-pick clean?**: YES — isolated bug fix, no control plane dependencies.

### 1c. Useful Infrastructure (MANUAL EXTRACTION)

These are valuable but mixed into commits with control plane code. Manual extraction needed.

| File | What's Useful | What's Mixed In |
|------|--------------|----------------|
| .env.example | Environment variable documentation | References control plane env vars |
| scripts/check-env.sh | Environment validation utility | Clean, no CP deps |
| docker/api/entrypoint.sh | Proper Docker entrypoint | Clean, no CP deps |
| CHANGELOG.md | Change history | References control plane features |
| tests/api/__init__.py | Deleted (cleanup) | Clean |

### 1d. Auth Hardening (MANUAL EXTRACTION)

These improvements in `api/config.py` are genuinely useful but interleaved with control plane additions in the same file:

| Improvement | Location | Coupled to CP? |
|------------|----------|---------------|
| `auth.enabled = True` by default | api/config.py | No — standalone |
| Production secret_key validator | api/config.py | No — standalone |
| CORS origins default `[]` instead of `["*"]` | api/config.py | No — standalone |
| Storage bucket default `""` (require explicit) | api/config.py | No — standalone |
| `StateBackendType` enum | api/config.py | YES — CP only |
| `state_backend` setting | api/config.py | YES — CP only |

**Recommendation**: Manually apply the 4 non-coupled improvements to main. Skip StateBackendType and state_backend.

---

## Category 2: DISCARD — LLM Control Plane

All new control plane infrastructure. Deprecated per board directive.

### 2a. Control Plane API (39 files, ~6,500 lines)

| Directory/File | Lines | Purpose |
|----------------|-------|---------|
| api/routes/control/__init__.py | 61 | Control router registration |
| api/routes/control/jobs.py | 839 | Job CRUD, state transitions |
| api/routes/control/escalations.py | 308 | Escalation management |
| api/routes/control/tools.py | 158 | Tool discovery/registry |
| api/models/control.py | 388 | Control plane Pydantic models |
| api/routes/internal/__init__.py | 34 | Internal router |
| api/routes/internal/deps.py | 35 | Internal dependencies |
| api/routes/internal/escalations.py | 132 | Escalation list endpoint |
| api/routes/internal/events.py | 296 | SSE event streaming |
| api/routes/internal/metrics.py | 207 | Pipeline health metrics |
| api/routes/internal/webhooks.py | 233 | Webhook management |
| api/models/internal.py | 138 | Internal API models |
| api/rate_limit.py (additions) | ~210 | Rate limiting middleware |

### 2b. State Backends (6 files, ~1,700 lines)

| File | Lines | Purpose |
|------|-------|---------|
| agents/orchestrator/backends/__init__.py | 21 | Backend factory |
| agents/orchestrator/backends/base.py | 239 | StateBackend ABC |
| agents/orchestrator/backends/dual_write.py | 216 | DualWrite migration backend |
| agents/orchestrator/backends/postgis_backend.py | 470 | PostGIS state backend |
| agents/orchestrator/backends/sqlite_backend.py | 279 | SQLite adapter |
| agents/orchestrator/state_model.py | 377 | Job phase/status enums |

### 2c. Workers / Task Queue (4 files, ~1,600 lines)

| File | Lines | Purpose |
|------|-------|---------|
| workers/__init__.py | 7 | Package init |
| workers/taskiq_app.py | 89 | Taskiq broker (Redis Streams) |
| workers/tasks/maintenance.py | 194 | Partition maintenance |
| workers/tasks/ogc_execution.py | 276 | OGC execution task |
| workers/tasks/pipeline_execution.py | 718 | Pipeline execution task |
| workers/tasks/webhooks.py | 333 | Webhook delivery |

### 2d. OGC / STAC Integration (7 files, ~1,300 lines)

| File | Lines | Purpose |
|------|-------|---------|
| core/ogc/__init__.py | 7 | Package |
| core/ogc/config.py | 184 | pygeoapi configuration |
| core/ogc/processors/__init__.py | 20 | Package |
| core/ogc/processors/factory.py | 398 | BaseProcessor factory |
| core/stac/__init__.py | 6 | Package |
| core/stac/collections.py | 258 | STAC collection definitions |
| core/stac/mount.py | 107 | stac-fastapi mount |
| core/stac/publisher.py | 353 | STAC Item publisher |

### 2e. CloudEvents (2 files, ~130 lines)

| File | Lines | Purpose |
|------|-------|---------|
| core/events/__init__.py | 5 | Package |
| core/events/cloudevents.py | 122 | CloudEvent model/factory |

### 2f. DB Migrations — Control Plane (6 files, ~600 lines)

| File | Purpose |
|------|---------|
| db/migrations/001_control_plane_schema.sql | Core CP tables |
| db/migrations/002_job_events_notify.sql | PG NOTIFY trigger |
| db/migrations/003_webhook_tables.sql | Webhook subscriptions |
| db/migrations/004_materialized_views.sql | Pipeline metrics views |
| db/migrations/005_partition_job_events.sql | Table partitioning |
| db/migrations/006_pgstac_init.sql | pgSTAC initialization |

### 2g. Tests — Control Plane (~16 files, ~6,500 lines)

| File | Lines |
|------|-------|
| tests/api/test_auth_enforcement.py | 473 |
| tests/api/test_control_escalations.py | 214 |
| tests/api/test_control_jobs.py | 427 |
| tests/api/test_control_tools.py | 268 |
| tests/api/test_event_stream.py | 312 |
| tests/api/test_rate_limiting.py | 289 |
| tests/api/test_webhooks.py | 372 |
| tests/e2e/test_control_plane_e2e.py | 330 |
| tests/geospatial/test_geospatial_edge_cases.py | 1,490 |
| tests/ogc/test_processors.py | 324 |
| tests/ogc/test_state_based_tool_filtering.py | 263 |
| tests/stac/test_publisher.py | 380 |
| tests/state/test_dual_write_backend.py | 319 |
| tests/state/test_postgis_backend.py | 499 |
| tests/state/test_state_model.py | 347 |

### 2h. Control Plane OpenSpec & Docs

| File | Purpose |
|------|---------|
| openspec/changes/add-llm-control-plane/tasks.md | CP task tracking |
| docs/design/api-surface-map.md | API surface documentation |
| docs/design/context-fetch-paths.md | Context fetch path design |
| docs/patterns.md | Design patterns doc |

### 2i. Modifications to Existing Files (MIXED — requires revert)

These existing files were modified to integrate the control plane. The CP additions must be reverted when cherry-picking KEEP changes.

| File | KEEP portion | DISCARD portion |
|------|-------------|----------------|
| api/main.py | None | Control/internal router mounts, Taskiq broker, mat view refresh, OGC/STAC mounts |
| api/middleware.py | None | TenantMiddleware class, middleware registration |
| api/config.py | Auth hardening (see 1d) | StateBackendType, state_backend field |
| api/auth.py | customer_id field (see Evaluate) | STATE_READ/WRITE, ESCALATION_MANAGE, INTERNAL_READ permissions |
| api/dependencies.py | load_default_algorithms change | None (change is benign) |
| agents/orchestrator/main.py | None | State backend factory wiring |
| agents/pipeline/main.py | None | Pipeline execution wiring, context repo integration |
| agents/discovery/main.py | None | Context repo integration |
| pyproject.toml | asyncpg core dep | control-plane optional group, workers package inclusion |
| core/analysis/library/registry.py | load_default_algorithms function | None (useful independently) |

---

## Category 3: DISCARD — MAIA-Specific

Demo scripts and notebooks created for MAIA showcase. All deprecated.

| File | Lines | Purpose |
|------|-------|---------|
| scripts/demo_control_plane.py | 1,277 | Control plane demo script |
| scripts/agent_demo.py | 763 | Agent orchestration demo |
| scripts/claude_agent_demo.py | 625 | Claude agent demo |
| scripts/demo_lakehouse.py | 784 | Lakehouse demo script |
| notebooks/demo_control_plane.ipynb | 954 | Interactive demo notebook |

**Total**: ~4,400 lines of demo code. All discard.

---

## Category 4: EVALUATE — Context Data Lakehouse

Board noted "DB store and data collection backend could be interesting." The context data lakehouse is exactly this — a data collection backend that stores discovery results and processing context for reuse across jobs.

### Files

| File | Lines | Purpose |
|------|-------|---------|
| core/context/__init__.py | 27 | Package init |
| core/context/models.py | 211 | Pydantic models (DatasetRecord, ContextQuery, etc.) |
| core/context/repository.py | 899 | ContextRepository — store/query/expire context data |
| core/context/stubs.py | 255 | Stub implementations for testing |
| api/models/context.py | 187 | API request/response models |
| api/routes/control/context.py | 430 | Context query API endpoints |
| db/migrations/000_add_customer_id.sql | 41 | Multi-tenant customer_id column |
| db/migrations/007_context_data.sql | 191 | Context data schema |
| tests/context/__init__.py | 0 | Package |
| tests/context/test_context_repository.py | 738 | Repository unit tests |
| tests/context/test_lakehouse_effect.py | 462 | Lakehouse effect tests |
| tests/context/test_pipeline_integration.py | 593 | Pipeline integration tests |
| openspec/changes/add-context-data-lakehouse/* | ~600 | Full OpenSpec (proposal, design, tasks, spec) |

**Total**: ~4,600 lines

### Coupling Assessment

| Coupling Point | Severity | Notes |
|----------------|----------|-------|
| API routes under `/control/v1/context/` | High | Mounted on control router |
| Uses control plane auth (CONTEXT_READ permission) | Medium | Permission could be re-added to auth.py |
| References job_id from control plane job model | Medium | Could use generic UUID |
| Repository uses asyncpg (same DB as CP) | Low | asyncpg is useful independently |
| Pipeline integration hooks into agents | Medium | discovery/pipeline agents modified |

### Recommendation: REFERENCE ONLY — Do Not Cherry-Pick

The concepts are sound (context caching for geospatial discovery is genuinely valuable), but the implementation is too tightly coupled to the control plane to extract cleanly. **Use as a design reference** when building a standalone context caching system on main:

- `core/context/models.py` — good Pydantic model patterns
- `core/context/repository.py` — good repository pattern with PostGIS queries
- `db/migrations/007_context_data.sql` — good schema design for spatial context storage

### Also EVALUATE: Algorithm Registry YAML

| File | Lines | Purpose |
|------|-------|---------|
| core/analysis/library/definitions/algorithms.yaml | 529 | Algorithm definitions (YAML) |
| core/analysis/library/registry.py (changes) | ~46 | load_default_algorithms function |

**Recommendation: KEEP** — The algorithms.yaml file documents all FirstLight algorithms in a structured format. The registry change (`load_default_algorithms`) is a clean improvement. Both are independent of the control plane. Cherry-pick or manually apply.

### Also EVALUATE: OpenSpec Tooling

| File | Lines | Purpose |
|------|-------|---------|
| openspec/AGENTS.md | 456 | OpenSpec CLI agent instructions |
| openspec/project.md | 58 | Project metadata for OpenSpec |

**Recommendation: KEEP** — OpenSpec tooling is independent of the control plane and useful for future spec work. These files can be cherry-picked or manually applied.

### Also EVALUATE: Multi-Tenant Foundation (customer_id)

The `customer_id` additions to `api/auth.py` (UserContext, APIKey, APIKeyStore) and `db/migrations/000_add_customer_id.sql` lay groundwork for multi-tenancy.

**Recommendation: KEEP the concept, rebuild cleanly** — The customer_id field is useful for future multi-tenant support, but it's interleaved with control plane permission additions in auth.py. Extract the customer_id additions manually.

---

## Impact Assessment: Abandoning the Entire Branch

### What is Lost

| Category | Lines | Impact |
|----------|-------|--------|
| Security scrub + PII removal | ~6,985 deletions | **HIGH** — these deletions must be applied to main |
| Bug fixes (products/status) | ~100 lines | **HIGH** — broken imports crash those routes |
| Auth hardening | ~80 lines | **MEDIUM** — auth defaults to disabled on main |
| Control plane infrastructure | ~18,000 lines | **NONE** — deprecated by board |
| MAIA demos | ~4,400 lines | **NONE** — deprecated by board |
| Context lakehouse | ~4,600 lines | **LOW** — use as reference for future work |
| Algorithm YAML + registry | ~575 lines | **MEDIUM** — useful structured data |
| OpenSpec tooling | ~514 lines | **LOW** — can recreate |

### What Breaks on Main Without These Fixes

1. **`api/routes/products.py`** — imports `_events_store` from events module which no longer exists → crash
2. **`api/routes/status.py`** — same broken import → crash
3. **Internal docs with PII** — still present on main (emails, local paths, personal GitHub URL)

---

## Recommended Approach

### Strategy: Targeted Cherry-Pick + Manual Extraction

Do NOT merge the branch. Do NOT start fresh. Cherry-pick the clean commits, then manually apply a small set of improvements.

### Step 1: Cherry-Pick Clean Commits (3 commits)

```bash
git checkout main
git cherry-pick 54d8464   # security: Scrub PII, credentials, and sensitive references
git cherry-pick dc3c527   # security: Remove remaining internal doc with local paths
git cherry-pick 2803a76   # fix: Remove broken _events_store import from products and status routes
```

**Note**: Check for conflicts on each. The security commits delete files and modify URLs — should apply cleanly. The products/status fix may need minor conflict resolution if main has diverged.

### Step 2: Manual Extraction (new commit on main)

Apply these improvements manually (not cherry-pick — too mixed with CP code):

1. **Auth hardening** from `api/config.py`:
   - `auth.enabled = True` (was False)
   - Production secret_key validator
   - CORS origins default `[]` (was `["*"]`)
   - Storage bucket default `""` (was `"firstlight-products"`)

2. **Algorithm registry** from `core/analysis/library/`:
   - Copy `definitions/algorithms.yaml`
   - Apply `load_default_algorithms` changes to `registry.py`

3. **Infrastructure files** (copy directly):
   - `scripts/check-env.sh`
   - `docker/api/entrypoint.sh`

4. **OpenSpec tooling** (copy directly):
   - `openspec/AGENTS.md`
   - `openspec/project.md`

### Step 3: Do NOT Cherry-Pick

Everything else stays on the branch as-is. The branch is archived, not deleted — it serves as a design reference for:
- Context data lakehouse patterns
- PostGIS state backend patterns
- OGC/STAC integration patterns
- Multi-tenant architecture

---

## P0 Recommendation: Priority Engineering Work on Main

Based on this triage and the board directive, the immediate priorities for main are:

1. **Apply the 3 cherry-picks above** — security scrub and bug fixes are blocking
2. **Apply auth hardening** — auth disabled by default is a security gap
3. **Algorithm registry YAML** — structured algorithm data improves the platform independently
4. **Decide on context caching** — if "DB store and data collection backend" is a priority, spec it fresh on main using the lakehouse code as reference (no control plane coupling)

### What NOT to Build on Main

- LLM control plane (deprecated)
- OGC API Processes / pygeoapi integration (coupled to CP)
- STAC catalog / pgSTAC (coupled to CP)
- Taskiq/Redis worker queue (coupled to CP)
- Multi-tenant TenantMiddleware (premature without new product direction)
