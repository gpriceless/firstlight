# FirstLight Production Readiness - OpenSpec

**Version:** 1.0
**Date:** 2026-01-23
**Status:** Active
**Owner:** Product Queen

---

## Executive Summary

The FirstLight repository contains **production-ready geospatial algorithms** and **solid architectural foundations**, but the user-facing interfaces (CLI and API) are largely disconnected from the working core. This OpenSpec defines all work required to achieve true production readiness.

### Current State
- **Core Library:** Real implementations - algorithms work, validated with real analysis
- **CLI Layer:** ~60% stub/mock - commands return fake data instead of calling core
- **API Layer:** ~90% complete - missing database persistence and JWT validation
- **Agent System:** ~95% complete - pipeline agent has stub algorithm execution

### Total Effort Estimate
- **P0 (Critical):** 8-10 days - Wire CLI and Agent to Core
- **P1 (High):** 4-6 days - API production hardening
- **P2 (Medium):** 2-3 days - Integration testing and polish
- **Total:** 15-21 developer days to full production readiness

### Work Structure
- **Phase 1:** CLI-to-Core Wiring (P0) - Can parallelize across commands
- **Phase 2:** Agent-to-Core Wiring (P0) - Sequential after Phase 1 starts
- **Phase 3:** API Production Hardening (P1) - Can run parallel to Phase 1/2
- **Phase 4:** Integration Testing (P2) - Sequential after Phases 1-3
- **Phase 5:** Documentation and Polish (P3) - Can run parallel

---

## Phase 1: CLI-to-Core Wiring (P0 - Critical)

**Goal:** Replace all CLI stubs with real core library calls
**Effort:** 8-10 days total
**Parallelization:** Each epic can run in parallel

### Epic 1.1: Wire CLI Analyze Command

**File:** `cli/commands/analyze.py`
**Current State:** Uses `MockAlgorithm` class, returns random data
**Target State:** Call real algorithms from `core/analysis/library/`

#### Tasks

- [ ] **Task 1.1.1:** Remove MockAlgorithm class (L394-408)
  - Size: S (0.5 day)
  - Acceptance: No MockAlgorithm in codebase

- [ ] **Task 1.1.2:** Import algorithm registry from core
  - Size: S (0.5 day)
  - Import: `from core.analysis.library import AlgorithmRegistry`
  - Acceptance: Registry loads all baseline algorithms

- [ ] **Task 1.1.3:** Wire algorithm lookup by event type
  - Size: M (1 day)
  - Map event type to algorithm: flood -> ThresholdSAR/NDWI, wildfire -> dNBR
  - Acceptance: Correct algorithm selected for each event type

- [ ] **Task 1.1.4:** Wire algorithm execution with real data
  - Size: M (1 day)
  - Replace random output (L523-528) with real algorithm.run()
  - Acceptance: Real raster output from algorithm

- [ ] **Task 1.1.5:** Add unit tests for wired analyze command
  - Size: S (0.5 day)
  - Acceptance: Tests verify real algorithm execution

**Dependencies:** None - can start immediately
**Blockers:** None identified

---

### Epic 1.2: Wire CLI Ingest Command

**File:** `cli/commands/ingest.py`
**Current State:** "Downloads" by writing text placeholder to file (L429-432)
**Target State:** Use rasterio to fetch real COG/GeoTIFF data

#### Tasks

- [ ] **Task 1.2.1:** Remove placeholder download logic
  - Size: S (0.5 day)
  - Acceptance: No text-file placeholder writes

- [ ] **Task 1.2.2:** Import streaming ingester from core
  - Size: S (0.5 day)
  - Import: `from core.data.ingestion import StreamingIngester`
  - Acceptance: StreamingIngester available in CLI

- [ ] **Task 1.2.3:** Wire download to StreamingIngester
  - Size: M (1 day)
  - Use COG format handler for cloud-optimized downloads
  - Acceptance: Real satellite data downloaded as GeoTIFF

- [ ] **Task 1.2.4:** Wire normalization to real normalizer (L496)
  - Size: M (1 day)
  - Import: `from core.data.ingestion.normalization import DataNormalizer`
  - Acceptance: Data properly reprojected and resampled

- [ ] **Task 1.2.5:** Integrate image validation
  - Size: S (0.5 day)
  - Call: `core.data.ingestion.validation.ImageValidator`
  - Acceptance: Blank/corrupt images detected and reported

- [ ] **Task 1.2.6:** Add integration tests for ingest
  - Size: M (1 day)
  - Acceptance: End-to-end ingest test with real STAC data

**Dependencies:** None - can start immediately
**Blockers:** None identified

---

### Epic 1.3: Wire CLI Validate Command

**File:** `cli/commands/validate.py`
**Current State:** All quality checks return `random.uniform()` scores (L235-275)
**Target State:** Call real QC modules from `core/quality/`

#### Tasks

- [ ] **Task 1.3.1:** Remove random score generation
  - Size: S (0.5 day)
  - Acceptance: No random.uniform() in validation code

- [ ] **Task 1.3.2:** Import sanity suite from core
  - Size: S (0.5 day)
  - Import: `from core.quality.sanity import SanitySuite`
  - Acceptance: Sanity checks available

- [ ] **Task 1.3.3:** Wire spatial validation
  - Size: S (0.5 day)
  - Call: `sanity.spatial.SpatialCoherenceCheck`
  - Acceptance: Real spatial coherence scores

- [ ] **Task 1.3.4:** Wire value validation
  - Size: S (0.5 day)
  - Call: `sanity.values.ValueRangeCheck`
  - Acceptance: Real value range validation

- [ ] **Task 1.3.5:** Wire artifact detection
  - Size: S (0.5 day)
  - Call: `sanity.artifacts.ArtifactDetector`
  - Acceptance: Stripe/hot pixel detection working

- [ ] **Task 1.3.6:** Wire temporal consistency check
  - Size: S (0.5 day)
  - Call: `sanity.temporal.TemporalConsistencyCheck`
  - Acceptance: Temporal validation scores

- [ ] **Task 1.3.7:** Add tests for real validation
  - Size: S (0.5 day)
  - Acceptance: Tests confirm real QC execution

**Dependencies:** None - can start immediately
**Blockers:** None identified

---

### Epic 1.4: Wire CLI Export Command

**File:** `cli/commands/export.py`
**Current State:** Creates mock GeoJSON, placeholder PNG, text-based "PDF" (L419+)
**Target State:** Use real ProductGenerator from core

#### Tasks

- [ ] **Task 1.4.1:** Remove mock export implementations
  - Size: S (0.5 day)
  - Acceptance: No placeholder file creation

- [ ] **Task 1.4.2:** Import ProductGenerator from core
  - Size: S (0.5 day)
  - Import: `from core.quality.reporting import ProductGenerator`
  - Acceptance: ProductGenerator available

- [ ] **Task 1.4.3:** Wire GeoTIFF/COG export
  - Size: M (1 day)
  - Use rasterio for proper GeoTIFF with metadata
  - Acceptance: Valid GeoTIFF with CRS and geotransform

- [ ] **Task 1.4.4:** Wire GeoJSON export
  - Size: S (0.5 day)
  - Use proper vectorization from raster results
  - Acceptance: Valid GeoJSON with feature properties

- [ ] **Task 1.4.5:** Wire PDF report generation
  - Size: M (1 day)
  - Use QA reporter for real PDF generation
  - Acceptance: Proper PDF with maps and statistics

- [ ] **Task 1.4.6:** Add export format tests
  - Size: S (0.5 day)
  - Acceptance: All export formats validated

**Dependencies:** Epic 1.1 (needs real analysis output to export)
**Blockers:** None identified

---

### Epic 1.5: Wire CLI Run Command

**File:** `cli/commands/run.py`
**Current State:** Multiple stubs returning random/empty data
- L509-514: `run_analyze()` returns `np.random.randint`
- L533-536: `run_validate()` returns `random.uniform`
- L562-565: `run_export()` creates empty 0-byte files
- L445-458: Falls back to hardcoded mock discovery

**Target State:** Orchestrate real pipeline through all stages

#### Tasks

- [ ] **Task 1.5.1:** Remove all random/mock fallbacks
  - Size: M (1 day)
  - Acceptance: No random.* or np.random.* in run.py

- [ ] **Task 1.5.2:** Wire discovery to real STAC client
  - Size: M (1 day)
  - Replace mock discovery fallback (L445-458)
  - Acceptance: Real STAC queries executed

- [ ] **Task 1.5.3:** Wire run_analyze to real pipeline runner
  - Size: M (1 day)
  - Import: `from core.analysis.execution import PipelineRunner`
  - Acceptance: Real algorithm execution with progress

- [ ] **Task 1.5.4:** Wire run_validate to real QC
  - Size: S (0.5 day)
  - Reuse wiring from Epic 1.3
  - Acceptance: Real QC scores in pipeline

- [ ] **Task 1.5.5:** Wire run_export to real products
  - Size: S (0.5 day)
  - Reuse wiring from Epic 1.4
  - Acceptance: Real products generated

- [ ] **Task 1.5.6:** Add end-to-end run tests
  - Size: M (1 day)
  - Acceptance: Full pipeline test with real data

**Dependencies:** Epics 1.1, 1.3, 1.4
**Blockers:** Must complete after dependent epics

---

### Epic 1.6: Wire CLI Discover Command

**File:** `cli/commands/discover.py`
**Current State:** Falls back to `generate_mock_results()` (L334-340)
**Target State:** Always use real STAC discovery

#### Tasks

- [ ] **Task 1.6.1:** Remove generate_mock_results fallback
  - Size: S (0.5 day)
  - Acceptance: No mock result generation

- [ ] **Task 1.6.2:** Improve error handling for STAC failures
  - Size: M (1 day)
  - Raise proper exceptions instead of returning mocks
  - Acceptance: Clear error messages on STAC failure

- [ ] **Task 1.6.3:** Add retry logic for transient failures
  - Size: S (0.5 day)
  - Use exponential backoff for network errors
  - Acceptance: Transient failures retried automatically

- [ ] **Task 1.6.4:** Add discover command tests
  - Size: S (0.5 day)
  - Acceptance: Tests for success and error cases

**Dependencies:** None - can start immediately
**Blockers:** None identified

---

## Phase 2: Agent-to-Core Wiring (P0 - Critical)

**Goal:** Connect agent orchestration to real algorithm execution
**Effort:** 1-2 days
**Parallelization:** Must complete after Phase 1 starts (shared patterns)

### Epic 2.1: Wire Pipeline Agent to Algorithms

**File:** `agents/pipeline/main.py`
**Current State:** `_execute_algorithm()` (L811-834) returns stub data
**Target State:** Call real algorithms from algorithm registry

#### Tasks

- [ ] **Task 2.1.1:** Import algorithm registry in pipeline agent
  - Size: S (0.5 day)
  - Import: `from core.analysis.library import AlgorithmRegistry`
  - Acceptance: Registry available to agent

- [ ] **Task 2.1.2:** Replace stub _execute_algorithm implementation
  - Size: M (1 day)
  - Remove `await asyncio.sleep(0.01)` placeholder
  - Call `algorithm.run(data)` with real inputs
  - Acceptance: Real algorithm output from agent

- [ ] **Task 2.1.3:** Handle algorithm configuration
  - Size: S (0.5 day)
  - Pass algorithm parameters from pipeline spec
  - Acceptance: Algorithm config respected

- [ ] **Task 2.1.4:** Add error handling for algorithm failures
  - Size: S (0.5 day)
  - Proper exception handling and retry logic
  - Acceptance: Failed algorithms reported correctly

- [ ] **Task 2.1.5:** Add agent-algorithm integration tests
  - Size: M (1 day)
  - Acceptance: End-to-end agent pipeline test

**Dependencies:** Patterns established in Phase 1
**Blockers:** None identified

---

## Phase 3: API Production Hardening (P1 - High)

**Goal:** Make API production-ready with persistence and security
**Effort:** 4-6 days
**Parallelization:** Can run parallel to Phases 1 and 2

### Epic 3.1: Database Persistence

**File:** `api/dependencies.py`
**Current State:** `execute()` raises `NotImplementedError` (L67-70)
**Target State:** Real database persistence (PostgreSQL or SQLite)

#### Tasks

- [ ] **Task 3.1.1:** Choose and add database library
  - Size: S (0.5 day)
  - Options: SQLAlchemy (ORM), asyncpg (raw), SQLite (simple)
  - Decision criteria: Deployment target complexity
  - Acceptance: Database library in requirements

- [ ] **Task 3.1.2:** Define database schema
  - Size: M (1 day)
  - Tables: events, executions, products, users
  - Acceptance: Schema migration file created

- [ ] **Task 3.1.3:** Implement database connection pool
  - Size: M (1 day)
  - Async connection pool with health checks
  - Acceptance: Connections managed properly

- [ ] **Task 3.1.4:** Replace _events_store dict
  - Size: M (1 day)
  - File: `api/routes/events.py`
  - Acceptance: Events persisted to database

- [ ] **Task 3.1.5:** Add database tests
  - Size: S (0.5 day)
  - Acceptance: CRUD operations tested

**Dependencies:** None - can start immediately
**Blockers:** Database choice decision required

---

### Epic 3.2: JWT Authentication

**File:** `api/dependencies.py`
**Current State:** TODO at L356 - tokens accepted but not validated
**Target State:** Proper JWT validation with claims

#### Tasks

- [ ] **Task 3.2.1:** Add PyJWT dependency
  - Size: S (0.5 day)
  - Acceptance: PyJWT in requirements

- [ ] **Task 3.2.2:** Implement JWT validation
  - Size: M (1 day)
  - Verify signature, expiration, issuer
  - Acceptance: Invalid tokens rejected

- [ ] **Task 3.2.3:** Add role-based access control
  - Size: M (1 day)
  - Roles: admin, analyst, viewer
  - Acceptance: Endpoints protected by role

- [ ] **Task 3.2.4:** Add refresh token support
  - Size: S (0.5 day)
  - Acceptance: Tokens can be refreshed

- [ ] **Task 3.2.5:** Add auth tests
  - Size: S (0.5 day)
  - Acceptance: Auth flow fully tested

**Dependencies:** None - can start immediately
**Blockers:** JWT secret management strategy needed

---

### Epic 3.3: Wire Intent Resolution API

**File:** `api/routes/events.py`
**Current State:** TODO at L201 - defaults to "flood.riverine"
**Target State:** Call real intent resolver from core

#### Tasks

- [ ] **Task 3.3.1:** Import intent resolver from core
  - Size: S (0.5 day)
  - Import: `from core.intent import IntentResolver`
  - Acceptance: Resolver available in API

- [ ] **Task 3.3.2:** Wire natural language parsing
  - Size: M (1 day)
  - Parse user descriptions to event types
  - Acceptance: "flooding after hurricane" -> flood.coastal

- [ ] **Task 3.3.3:** Add intent validation
  - Size: S (0.5 day)
  - Validate resolved intent against taxonomy
  - Acceptance: Invalid intents rejected

- [ ] **Task 3.3.4:** Add intent resolution tests
  - Size: S (0.5 day)
  - Acceptance: Common phrases resolve correctly

**Dependencies:** None - can start immediately
**Blockers:** None identified

---

## Phase 4: Integration Testing (P2 - Medium)

**Goal:** Verify end-to-end system works
**Effort:** 3-5 days
**Parallelization:** Must complete after Phases 1-3

### Epic 4.1: CLI Integration Tests

#### Tasks

- [ ] **Task 4.1.1:** Create CLI test harness
  - Size: M (1 day)
  - Use CliRunner from Typer
  - Acceptance: CLI commands testable

- [ ] **Task 4.1.2:** Add discover-to-analyze flow test
  - Size: M (1 day)
  - Discover data, run analysis, verify output
  - Acceptance: Full flow works

- [ ] **Task 4.1.3:** Add analyze-to-export flow test
  - Size: M (1 day)
  - Run analysis, export products, validate files
  - Acceptance: Products are valid

- [ ] **Task 4.1.4:** Add error handling tests
  - Size: S (0.5 day)
  - Test invalid inputs, network failures
  - Acceptance: Errors handled gracefully

**Dependencies:** Phase 1 complete
**Blockers:** None identified

---

### Epic 4.2: API Integration Tests

#### Tasks

- [ ] **Task 4.2.1:** Create API test client
  - Size: S (0.5 day)
  - Use TestClient from FastAPI
  - Acceptance: API testable end-to-end

- [ ] **Task 4.2.2:** Add event submission flow test
  - Size: M (1 day)
  - Submit event, poll status, get results
  - Acceptance: Full API flow works

- [ ] **Task 4.2.3:** Add concurrent request test
  - Size: M (1 day)
  - Multiple simultaneous events
  - Acceptance: No race conditions

- [ ] **Task 4.2.4:** Add database persistence test
  - Size: S (0.5 day)
  - Restart API, verify data persists
  - Acceptance: Data survives restart

**Dependencies:** Phase 3 complete
**Blockers:** None identified

---

### Epic 4.3: Agent Integration Tests

#### Tasks

- [ ] **Task 4.3.1:** Add orchestrator flow test
  - Size: M (1 day)
  - Submit event, verify agent coordination
  - Acceptance: Agents communicate correctly

- [ ] **Task 4.3.2:** Add pipeline execution test
  - Size: M (1 day)
  - Run full pipeline through agents
  - Acceptance: Real results from agent system

- [ ] **Task 4.3.3:** Add failure recovery test
  - Size: S (0.5 day)
  - Simulate failures, verify recovery
  - Acceptance: System recovers gracefully

**Dependencies:** Phase 2 complete
**Blockers:** None identified

---

## Phase 5: Documentation and Polish (P3 - Low)

**Goal:** Update documentation to reflect production state
**Effort:** 1-2 days
**Parallelization:** Can run parallel to Phases 1-4

### Epic 5.1: Update Documentation

#### Tasks

- [ ] **Task 5.1.1:** Update CLAUDE.md
  - Size: S (0.5 day)
  - Reflect new CLI capabilities
  - Acceptance: Accurate CLI documentation

- [ ] **Task 5.1.2:** Update API documentation
  - Size: S (0.5 day)
  - OpenAPI spec reflects real endpoints
  - Acceptance: Swagger UI accurate

- [ ] **Task 5.1.3:** Update ROADMAP.md
  - Size: S (0.5 day)
  - Mark production readiness complete
  - Acceptance: Roadmap current

- [ ] **Task 5.1.4:** Archive this OpenSpec
  - Size: S (0.5 day)
  - Move to OPENSPEC_ARCHIVE.md
  - Acceptance: OpenSpec archived

**Dependencies:** All phases complete
**Blockers:** None identified

---

## Dependency Graph

```
Phase 1 (CLI Wiring) ─────────────────┐
├── Epic 1.1 (Analyze) ───────────────├──> Epic 1.4 (Export)
├── Epic 1.2 (Ingest) ────────────────│
├── Epic 1.3 (Validate) ──────────────├──> Epic 1.5 (Run)
├── Epic 1.6 (Discover) ──────────────┘
│
├──> Phase 2 (Agent Wiring)
│    └── Epic 2.1 (Pipeline Agent)
│
├──> Phase 4 (Integration Testing)
     ├── Epic 4.1 (CLI Tests)
     ├── Epic 4.3 (Agent Tests)
     │
Phase 3 (API Hardening) ──────────────┤
├── Epic 3.1 (Database) ──────────────├──> Epic 4.2 (API Tests)
├── Epic 3.2 (JWT) ───────────────────┘
├── Epic 3.3 (Intent)
│
Phase 5 (Documentation) ──────────────> Final (after all phases)
```

---

## Parallel Work Streams

### Stream A: CLI Commands (Can all run in parallel)
- Epic 1.1 (Analyze) - Independent
- Epic 1.2 (Ingest) - Independent
- Epic 1.3 (Validate) - Independent
- Epic 1.6 (Discover) - Independent

### Stream B: API Hardening (Can run parallel to Stream A)
- Epic 3.1 (Database) - Independent
- Epic 3.2 (JWT) - Independent
- Epic 3.3 (Intent) - Independent

### Stream C: Sequential (Must wait)
- Epic 1.4 (Export) - After 1.1
- Epic 1.5 (Run) - After 1.1, 1.3, 1.4
- Epic 2.1 (Agent) - After Phase 1 patterns established
- Epic 4.x (Testing) - After relevant phases complete

### Optimal Parallelization
With 3 developers:
- Developer 1: Epic 1.1 -> 1.4 -> 2.1
- Developer 2: Epic 1.2 -> 1.3 -> 1.5
- Developer 3: Epic 3.1 -> 3.2 -> 3.3

With 2 developers:
- Developer 1: Epics 1.1, 1.2, 1.4, 2.1
- Developer 2: Epics 1.3, 1.6, 3.1, 3.2, 3.3

With 1 developer:
- Week 1: Epics 1.1, 1.2, 1.3
- Week 2: Epics 1.4, 1.5, 1.6
- Week 3: Epics 2.1, 3.1, 3.2, 3.3

---

## Risk Assessment

### High Risk Items
1. **Database migration complexity** - If existing in-memory data must be preserved
   - Mitigation: Clean slate for production deployment

2. **STAC catalog availability** - Real tests depend on external services
   - Mitigation: VCR-style response recording for tests

### Medium Risk Items
1. **Algorithm performance** - Real algorithms may be slower than stubs
   - Mitigation: Already have Dask parallelization

2. **Memory usage** - Real data processing needs more memory
   - Mitigation: Tiled processing already implemented

### Low Risk Items
1. **Test coverage** - 518+ tests already exist
2. **Architecture** - Clean separation already in place

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `flight analyze` produces real classification results
- [ ] `flight ingest` downloads real satellite data
- [ ] `flight validate` produces real QC scores
- [ ] `flight export` creates valid GeoTIFF/GeoJSON/PDF
- [ ] `flight run` executes full pipeline end-to-end
- [ ] `flight discover` never falls back to mocks

### Phase 2 Complete When:
- [ ] Agent pipeline executes real algorithms
- [ ] Agent output matches direct algorithm call output

### Phase 3 Complete When:
- [ ] Events persist across API restarts
- [ ] Invalid JWT tokens are rejected
- [ ] Natural language intent is resolved correctly

### Phase 4 Complete When:
- [ ] All integration tests pass
- [ ] No mock/stub code remains in production paths

### Production Ready When:
- [ ] All phases complete
- [ ] Documentation updated
- [ ] This OpenSpec archived

---

## Appendix A: File Change Summary

| File | Current State | Changes Required |
|------|---------------|------------------|
| `cli/commands/analyze.py` | MockAlgorithm | Wire to AlgorithmRegistry |
| `cli/commands/ingest.py` | Text placeholder | Wire to StreamingIngester |
| `cli/commands/validate.py` | random.uniform() | Wire to SanitySuite |
| `cli/commands/export.py` | Mock files | Wire to ProductGenerator |
| `cli/commands/run.py` | Multiple stubs | Wire all sub-commands |
| `cli/commands/discover.py` | Mock fallback | Remove fallback |
| `agents/pipeline/main.py` | Stub execution | Wire to algorithms |
| `api/dependencies.py` | NotImplementedError | Add database layer |
| `api/dependencies.py` | TODO JWT | Implement validation |
| `api/routes/events.py` | TODO intent | Wire to IntentResolver |

---

## Appendix B: Core Libraries to Wire

| Core Module | CLI Target | API Target | Agent Target |
|-------------|------------|------------|--------------|
| `core.analysis.library.AlgorithmRegistry` | analyze.py | - | pipeline/main.py |
| `core.data.ingestion.StreamingIngester` | ingest.py | - | - |
| `core.quality.sanity.SanitySuite` | validate.py | - | quality/main.py |
| `core.quality.reporting.ProductGenerator` | export.py | products.py | reporting/main.py |
| `core.intent.IntentResolver` | - | events.py | - |
| `core.data.discovery.STACClient` | discover.py | - | discovery/main.py |

---

*OpenSpec created by Product Queen. Last updated 2026-01-23.*
