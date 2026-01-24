# FirstLight - Requirements Analysis

**Date:** 2026-01-23
**Analyst:** Requirements Reviewer Agent
**Sources:** AUDIT_REPORT.md, AUDIT_ADDENDUM.md, OPENSPEC.md, codebase inspection
**Scope:** CLI-to-Core integration, API completion, agent wiring

---

## Executive Summary

### Overall Assessment: **NEEDS WORK - Moderate Integration Gaps**

The FirstLight platform has a **complete and tested core layer** (170K+ lines, 518+ tests) but suffers from a **disconnected CLI layer** that returns mock/random data instead of invoking real algorithms. The API layer has placeholder components that need production implementation.

| Layer | Status | Primary Gap |
|-------|--------|-------------|
| Core Algorithms | **COMPLETE** | None - fully functional |
| Data Discovery | **COMPLETE** | None - STAC integration works |
| Quality System | **COMPLETE** | None - sanity checks work |
| CLI Commands | **60% STUB** | Mock implementations instead of core calls |
| API Layer | **90% Complete** | Database placeholder, JWT stub |
| Agent System | **95% Complete** | Algorithm execution stubbed |

### Blocking Issues
1. CLI commands produce fake outputs (random data, empty files)
2. Pipeline agent `_execute_algorithm()` is a stub
3. Database layer raises `NotImplementedError`
4. JWT validation is `TODO`

---

## 1. CLI-to-Core Integration Requirements

The CLI layer exists as a demo facade. Commands accept parameters but return mock data instead of invoking the real core processing modules.

### 1.1 Analyze Command Integration

**Current State:** `cli/commands/analyze.py:394-408` falls back to `MockAlgorithm` class when imports fail or no input data exists.

- **REQ-CLI-001**: Wire analyze command to real algorithm registry
  - **Description:** Replace `MockAlgorithm` fallback with proper error handling and actual algorithm invocation via `core.analysis.library.registry`
  - **Acceptance Criteria:**
    - Given valid input data exists
    - When `flight analyze --algorithm sar_threshold` is run
    - Then the real `ThresholdSARAlgorithm` from `core.analysis.library.baseline.flood.threshold_sar` is invoked
    - And output contains actual flood extent data (not random)
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-002 (ingest must work first)
  - **Agent Type:** coder
  - **Parallelizable:** No (depends on ingest)

- **REQ-CLI-002**: Remove mock algorithm fallback
  - **Description:** When algorithm import fails, raise clear error instead of silently using mock
  - **Acceptance Criteria:**
    - Given algorithm module cannot be imported
    - When analyze command runs
    - Then descriptive ImportError is raised with module path
    - And no mock data is produced
  - **Complexity:** Small
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-CLI-003**: Connect analyze to TiledRunner for large datasets
  - **Description:** Use `core.analysis.execution.tiled_runner.TiledRunner` for memory-efficient processing
  - **Acceptance Criteria:**
    - Given input raster is larger than configured memory limit
    - When analyze command runs with `--parallel N`
    - Then TiledRunner processes tiles in parallel
    - And tiles are properly stitched into final output
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-001
  - **Agent Type:** coder
  - **Parallelizable:** No

### 1.2 Ingest Command Integration

**Current State:** `cli/commands/ingest.py:429-432` writes text comments to files instead of downloading actual satellite data.

- **REQ-CLI-004**: Wire ingest to real data download
  - **Description:** Use `core.data.ingestion.streaming.StreamingIngester` or rasterio to fetch COG tiles
  - **Acceptance Criteria:**
    - Given valid STAC discovery results
    - When `flight ingest` is run
    - Then satellite imagery is downloaded as GeoTIFF/COG
    - And file size is non-zero and contains raster data
  - **Complexity:** Large
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-CLI-005**: Implement proper normalization pipeline
  - **Description:** Apply `core.data.ingestion.normalization` modules (projection, resolution, temporal alignment)
  - **Acceptance Criteria:**
    - Given multi-source data is ingested
    - When normalization runs
    - Then all rasters share same CRS and resolution
    - And temporal alignment is recorded in metadata
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-004
  - **Agent Type:** coder
  - **Parallelizable:** No

### 1.3 Validate Command Integration

**Current State:** `cli/commands/validate.py:235-275` returns `random.uniform()` scores instead of calling quality modules.

- **REQ-CLI-006**: Wire validate to core quality modules
  - **Description:** Replace mock random scores with calls to `core.quality.sanity` and `core.quality.validation`
  - **Acceptance Criteria:**
    - Given analysis results exist
    - When `flight validate` is run
    - Then `SanitySuite.run_check()` is called for each sanity check
    - And actual quality metrics are computed from raster data
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-001 (analyze must produce real output)
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-CLI-007**: Integrate artifact detection
  - **Description:** Use `core.quality.sanity.artifacts` for stripe/saturation detection
  - **Acceptance Criteria:**
    - Given satellite imagery with processing artifacts
    - When artifact detection runs
    - Then stripe patterns are detected
    - And quality report includes artifact locations
  - **Complexity:** Small
  - **Dependencies:** REQ-CLI-006
  - **Agent Type:** coder
  - **Parallelizable:** Yes

### 1.4 Export Command Integration

**Current State:** `cli/commands/export.py:419+` creates empty 0-byte files and text-based "PDFs".

- **REQ-CLI-008**: Wire export to real product generator
  - **Description:** Use `core.quality.reporting.qa_report.ProductGenerator` for proper output formats
  - **Acceptance Criteria:**
    - Given valid analysis results
    - When `flight export --format geotiff` is run
    - Then valid GeoTIFF with proper georeferencing is created
    - And file can be opened in QGIS/ArcGIS
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-001
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-CLI-009**: Implement proper GeoJSON export
  - **Description:** Convert raster flood extent to vector polygons with attributes
  - **Acceptance Criteria:**
    - Given flood extent raster
    - When GeoJSON export runs
    - Then flood polygons are properly vectorized
    - And GeoJSON contains valid geometry and properties
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-008
  - **Agent Type:** coder
  - **Parallelizable:** Yes

### 1.5 Run Command Integration

**Current State:** `cli/commands/run.py` orchestrates the pipeline but individual stages use mocks.

- **REQ-CLI-010**: Wire run_analyze() to real algorithm execution
  - **Description:** Replace `np.random.randint(0,2,size=(100,100))` at line 509-514 with actual algorithm call
  - **Acceptance Criteria:**
    - Given `flight run --event flood` is executed
    - When analyze stage runs
    - Then real flood detection algorithm produces output
    - And output dimensions match input data
  - **Complexity:** Medium
  - **Dependencies:** REQ-CLI-001, REQ-CLI-004
  - **Agent Type:** coder
  - **Parallelizable:** No

- **REQ-CLI-011**: Wire run_validate() to real quality checks
  - **Description:** Replace `random.uniform(0.7, 1.0)` at line 533-536 with quality module calls
  - **Acceptance Criteria:**
    - Given analysis results exist
    - When validate stage runs
    - Then actual quality scores are computed
    - And validation report contains real metrics
  - **Complexity:** Small
  - **Dependencies:** REQ-CLI-006
  - **Agent Type:** coder
  - **Parallelizable:** No

- **REQ-CLI-012**: Wire run_export() to ProductGenerator
  - **Description:** Replace empty file creation at line 562-565 with real export
  - **Acceptance Criteria:**
    - Given valid analysis results
    - When export stage runs
    - Then output files contain actual data
    - And metadata is properly embedded
  - **Complexity:** Small
  - **Dependencies:** REQ-CLI-008
  - **Agent Type:** coder
  - **Parallelizable:** No

### 1.6 Discover Command Integration

**Current State:** `cli/commands/discover.py:334-340` falls back to `generate_mock_results()` on any error.

- **REQ-CLI-013**: Improve error handling in discover command
  - **Description:** Only fall back to mock on specific network/timeout errors, not on all exceptions
  - **Acceptance Criteria:**
    - Given STAC catalog is reachable
    - When discovery fails due to query parameters
    - Then descriptive error is shown (not mock results)
    - And user can correct parameters
  - **Complexity:** Small
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

---

## 2. API Layer Requirements

### 2.1 Database Persistence

**Current State:** `api/dependencies.py:67-70` raises `NotImplementedError`. Events stored in memory dict.

- **REQ-API-001**: Implement database persistence layer
  - **Description:** Replace in-memory `_events_store` dict with PostgreSQL/SQLite via SQLAlchemy or asyncpg
  - **Acceptance Criteria:**
    - Given API server restarts
    - When events are retrieved
    - Then previously submitted events persist
    - And multi-instance deployments share state
  - **Complexity:** Large
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-API-002**: Add event migration for existing data
  - **Description:** Migration script to move from in-memory to persistent storage
  - **Acceptance Criteria:**
    - Given existing events in memory
    - When migration runs
    - Then all events are persisted to database
    - And no data loss occurs
  - **Complexity:** Small
  - **Dependencies:** REQ-API-001
  - **Agent Type:** coder
  - **Parallelizable:** No

- **REQ-API-003**: Implement async database session management
  - **Description:** Complete `DatabaseSession.execute()` and connection pooling
  - **Acceptance Criteria:**
    - Given concurrent API requests
    - When database operations run
    - Then connections are properly pooled
    - And no connection exhaustion occurs
  - **Complexity:** Medium
  - **Dependencies:** REQ-API-001
  - **Agent Type:** coder
  - **Parallelizable:** No

### 2.2 Authentication

**Current State:** `api/dependencies.py:356` has `TODO: Implement actual JWT validation`.

- **REQ-API-004**: Implement JWT token validation
  - **Description:** Add PyJWT dependency and validate bearer tokens with signature verification
  - **Acceptance Criteria:**
    - Given bearer token in Authorization header
    - When token is expired or invalid signature
    - Then 401 Unauthorized is returned
    - And valid tokens grant access
  - **Complexity:** Medium
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-API-005**: Add role-based access control
  - **Description:** Implement user roles (admin, analyst, viewer) with route-level authorization
  - **Acceptance Criteria:**
    - Given user with "viewer" role
    - When attempting to delete event
    - Then 403 Forbidden is returned
    - And "admin" role can delete
  - **Complexity:** Medium
  - **Dependencies:** REQ-API-004
  - **Agent Type:** coder
  - **Parallelizable:** No

### 2.3 Intent Resolution

**Current State:** `api/routes/events.py:201` has `TODO: Call intent resolution service`.

- **REQ-API-006**: Wire intent resolution to API
  - **Description:** Connect natural language intent to `core.intent.resolver.IntentResolver`
  - **Acceptance Criteria:**
    - Given natural language like "flooded areas near Miami"
    - When event is submitted
    - Then `IntentResolver` classifies as `flood.coastal` or `flood.riverine`
    - And confidence score is returned
  - **Complexity:** Medium
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

### 2.4 Agent Integration

- **REQ-API-007**: Connect API to Orchestrator agent
  - **Description:** Event submission should trigger `OrchestratorAgent` processing
  - **Acceptance Criteria:**
    - Given event is submitted via POST /events
    - When event status changes to PROCESSING
    - Then OrchestratorAgent is invoked
    - And status updates are reflected in API
  - **Complexity:** Large
  - **Dependencies:** REQ-AGENT-001
  - **Agent Type:** coder
  - **Parallelizable:** No

---

## 3. Agent System Requirements

### 3.1 Pipeline Agent Algorithm Wiring

**Current State:** `agents/pipeline/main.py:811-834` `_execute_algorithm()` is a stub returning mock data.

- **REQ-AGENT-001**: Wire pipeline agent to algorithm registry
  - **Description:** Replace stub with actual algorithm module loading and execution
  - **Acceptance Criteria:**
    - Given pipeline step specifies `sar_threshold` algorithm
    - When `_execute_algorithm()` is called
    - Then `ThresholdSARAlgorithm.run()` is invoked
    - And real algorithm result is returned
  - **Complexity:** Medium
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

- **REQ-AGENT-002**: Implement algorithm result handling
  - **Description:** Convert algorithm output to pipeline step format with proper typing
  - **Acceptance Criteria:**
    - Given algorithm returns numpy array and metadata
    - When result is processed
    - Then proper `StepResult` is created
    - And output data is accessible to next step
  - **Complexity:** Small
  - **Dependencies:** REQ-AGENT-001
  - **Agent Type:** coder
  - **Parallelizable:** No

### 3.2 Quality Agent Integration

**Current State:** `agents/pipeline/main.py:836-856` `_run_quality_checks()` returns hardcoded scores.

- **REQ-AGENT-003**: Wire quality agent to sanity suite
  - **Description:** Connect quality checks to `core.quality.sanity.SanitySuite`
  - **Acceptance Criteria:**
    - Given step produces output data
    - When inline quality checks run
    - Then actual sanity checks are performed
    - And real quality scores are returned
  - **Complexity:** Medium
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

### 3.3 Orchestrator Webhook Delivery

- **REQ-AGENT-004**: Implement webhook notifications
  - **Description:** Deliver status updates to callback URLs specified in event metadata
  - **Acceptance Criteria:**
    - Given event has `callback_url` in metadata
    - When event status changes
    - Then POST request is sent to callback URL
    - And retry logic handles failures
  - **Complexity:** Medium
  - **Dependencies:** None
  - **Agent Type:** coder
  - **Parallelizable:** Yes

---

## 4. Testing Requirements

### 4.1 Integration Tests for CLI-Core Wiring

- **REQ-TEST-001**: CLI-to-Core integration test suite
  - **Description:** Tests that verify CLI commands invoke real core modules
  - **Acceptance Criteria:**
    - Given test fixtures with sample satellite data
    - When CLI commands are run
    - Then core modules are actually invoked
    - And outputs are validated against expected results
  - **Complexity:** Large
  - **Dependencies:** All REQ-CLI-* requirements
  - **Agent Type:** qa
  - **Parallelizable:** Yes

- **REQ-TEST-002**: End-to-end pipeline test
  - **Description:** Test complete workflow from discovery to export
  - **Acceptance Criteria:**
    - Given sample GeoJSON area
    - When `flight run --area sample.geojson --event flood` executes
    - Then all stages complete with real data
    - And final products are valid
  - **Complexity:** Large
  - **Dependencies:** REQ-TEST-001
  - **Agent Type:** qa
  - **Parallelizable:** No

### 4.2 API Integration Tests

- **REQ-TEST-003**: API database integration tests
  - **Description:** Tests for database persistence layer
  - **Acceptance Criteria:**
    - Given test database
    - When CRUD operations are performed
    - Then data persists correctly
    - And concurrent access is handled
  - **Complexity:** Medium
  - **Dependencies:** REQ-API-001
  - **Agent Type:** qa
  - **Parallelizable:** Yes

- **REQ-TEST-004**: API authentication tests
  - **Description:** Tests for JWT validation and RBAC
  - **Acceptance Criteria:**
    - Given valid/invalid tokens
    - When requests are made
    - Then proper authentication behavior occurs
    - And roles are enforced
  - **Complexity:** Medium
  - **Dependencies:** REQ-API-004, REQ-API-005
  - **Agent Type:** qa
  - **Parallelizable:** Yes

---

## 5. Documentation Requirements

- **REQ-DOC-001**: Update CLI documentation
  - **Description:** Document that CLI commands now use real core modules
  - **Acceptance Criteria:**
    - Given README and help text
    - When user reads documentation
    - Then expected behavior matches actual behavior
    - And error messages are documented
  - **Complexity:** Small
  - **Dependencies:** All REQ-CLI-* requirements
  - **Agent Type:** documentation
  - **Parallelizable:** Yes

- **REQ-DOC-002**: API migration guide
  - **Description:** Document database migration from in-memory to persistent
  - **Acceptance Criteria:**
    - Given existing deployments
    - When migration guide is followed
    - Then data is preserved
    - And upgrade is seamless
  - **Complexity:** Small
  - **Dependencies:** REQ-API-001
  - **Agent Type:** documentation
  - **Parallelizable:** Yes

---

## 6. Agent Work Breakdown

### Phase 1: Foundation (Can Run in Parallel)

| Requirement | Agent Type | Parallelizable | Dependencies |
|-------------|------------|----------------|--------------|
| REQ-CLI-002 | coder | Yes | None |
| REQ-CLI-004 | coder | Yes | None |
| REQ-CLI-013 | coder | Yes | None |
| REQ-API-001 | coder | Yes | None |
| REQ-API-004 | coder | Yes | None |
| REQ-API-006 | coder | Yes | None |
| REQ-AGENT-001 | coder | Yes | None |
| REQ-AGENT-003 | coder | Yes | None |
| REQ-AGENT-004 | coder | Yes | None |

**Estimated Agent Capacity:** 4-5 parallel coder agents

### Phase 2: Integration (Sequential Dependencies)

| Requirement | Agent Type | Dependencies |
|-------------|------------|--------------|
| REQ-CLI-001 | coder | REQ-CLI-002 |
| REQ-CLI-005 | coder | REQ-CLI-004 |
| REQ-CLI-006 | coder | Phase 1 |
| REQ-CLI-007 | coder | REQ-CLI-006 |
| REQ-CLI-008 | coder | REQ-CLI-001 |
| REQ-CLI-009 | coder | REQ-CLI-008 |
| REQ-API-002 | coder | REQ-API-001 |
| REQ-API-003 | coder | REQ-API-001 |
| REQ-API-005 | coder | REQ-API-004 |
| REQ-AGENT-002 | coder | REQ-AGENT-001 |

### Phase 3: Pipeline Completion (Sequential)

| Requirement | Agent Type | Dependencies |
|-------------|------------|--------------|
| REQ-CLI-003 | coder | REQ-CLI-001 |
| REQ-CLI-010 | coder | REQ-CLI-001, REQ-CLI-004 |
| REQ-CLI-011 | coder | REQ-CLI-006 |
| REQ-CLI-012 | coder | REQ-CLI-008 |
| REQ-API-007 | coder | REQ-AGENT-001 |

### Phase 4: Testing & Documentation

| Requirement | Agent Type | Dependencies |
|-------------|------------|--------------|
| REQ-TEST-001 | qa | Phase 3 |
| REQ-TEST-002 | qa | REQ-TEST-001 |
| REQ-TEST-003 | qa | REQ-API-001 |
| REQ-TEST-004 | qa | REQ-API-004, REQ-API-005 |
| REQ-DOC-001 | documentation | Phase 3 |
| REQ-DOC-002 | documentation | REQ-API-001 |

---

## 7. Priority Matrix

### P0 - Critical (Must Complete Before Production Use)

| Requirement | Impact | Reason |
|-------------|--------|--------|
| REQ-CLI-004 | High | Without real ingest, no real data processing |
| REQ-CLI-001 | High | Core analysis is blocked without algorithm wiring |
| REQ-AGENT-001 | High | Agent orchestration depends on this |
| REQ-API-001 | High | Multi-instance deployment requires persistence |

### P1 - High Priority (Should Complete)

| Requirement | Impact | Reason |
|-------------|--------|--------|
| REQ-CLI-006 | Medium | Quality validation ensures output reliability |
| REQ-CLI-008 | Medium | Export is final user-facing output |
| REQ-API-004 | Medium | Production security requirement |
| REQ-TEST-001 | Medium | Prevents regression |

### P2 - Medium Priority (Nice to Have)

| Requirement | Impact | Reason |
|-------------|--------|--------|
| REQ-API-005 | Low | Multi-tenant security enhancement |
| REQ-API-006 | Low | NL intent is optional path |
| REQ-AGENT-004 | Low | Webhook notifications are convenience |

### P3 - Low Priority (Future Enhancement)

| Requirement | Impact | Reason |
|-------------|--------|--------|
| REQ-DOC-001 | Low | Documentation follows implementation |
| REQ-DOC-002 | Low | Migration guide follows DB implementation |

---

## 8. Success Criteria

The requirements are complete when:

1. **CLI produces real outputs**: All `flight` commands invoke core modules and produce actual satellite analysis results
2. **End-to-end pipeline works**: `flight run` executes complete workflow from discovery to export with real data
3. **API persists data**: Events survive server restart and work across multiple instances
4. **Authentication is secure**: JWT tokens are properly validated with signature verification
5. **Agent orchestration is functional**: Pipeline agent executes real algorithms, not stubs
6. **Tests pass**: Integration tests verify CLI-Core wiring and API functionality
7. **Quality checks are real**: Validation uses actual sanity suite, not random scores

---

## 9. Gap Verification - Cross-Reference with Audit

| Gap from AUDIT_ADDENDUM | Addressed by Requirement |
|-------------------------|-------------------------|
| `run.py` random analysis output | REQ-CLI-010 |
| `run.py` random validation score | REQ-CLI-011 |
| `run.py` empty export files | REQ-CLI-012 |
| `analyze.py` MockAlgorithm | REQ-CLI-001, REQ-CLI-002 |
| `ingest.py` text placeholder | REQ-CLI-004 |
| `validate.py` random scores | REQ-CLI-006 |
| `export.py` mock outputs | REQ-CLI-008, REQ-CLI-009 |
| `discover.py` mock fallback | REQ-CLI-013 |
| Pipeline agent stub | REQ-AGENT-001 |
| Database NotImplementedError | REQ-API-001 |
| JWT TODO | REQ-API-004 |
| Intent resolution TODO | REQ-API-006 |

---

## 10. Appendix: File Locations

### CLI Layer (Requires Modification)
- `/home/gprice/projects/firstlight/cli/commands/analyze.py` - Lines 394-408, 523-528
- `/home/gprice/projects/firstlight/cli/commands/run.py` - Lines 509-514, 533-536, 562-565
- `/home/gprice/projects/firstlight/cli/commands/ingest.py` - Lines 429-432, 496
- `/home/gprice/projects/firstlight/cli/commands/validate.py` - Lines 235-275
- `/home/gprice/projects/firstlight/cli/commands/export.py` - Lines 419+
- `/home/gprice/projects/firstlight/cli/commands/discover.py` - Lines 334-340

### API Layer (Requires Modification)
- `/home/gprice/projects/firstlight/api/dependencies.py` - Lines 67-70 (DB), 356 (JWT)
- `/home/gprice/projects/firstlight/api/routes/events.py` - Lines 201 (intent), 89-90 (in-memory store)

### Agent Layer (Requires Modification)
- `/home/gprice/projects/firstlight/agents/pipeline/main.py` - Lines 811-834 (`_execute_algorithm`), 836-856 (`_run_quality_checks`)

### Core Layer (Reference - No Changes Needed)
- `/home/gprice/projects/firstlight/core/analysis/library/registry.py` - Algorithm registry
- `/home/gprice/projects/firstlight/core/analysis/library/baseline/` - Algorithm implementations
- `/home/gprice/projects/firstlight/core/quality/sanity/` - Quality checks
- `/home/gprice/projects/firstlight/core/data/ingestion/streaming.py` - Data ingestion
- `/home/gprice/projects/firstlight/core/intent/resolver.py` - Intent resolution

---

*Requirements Analysis completed by Requirements Reviewer agent. This document should be reviewed by Product Queen before engineering handoff.*
