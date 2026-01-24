# FirstLight: Implementation Roadmap

**Last Updated:** 2026-01-23
**Status:** Core Platform Complete, Interface Wiring In Progress

---

## Executive Summary

The FirstLight platform has a **production-ready core** (170K+ lines, 518+ tests) with working geospatial algorithms. The primary remaining work is **wiring the user-facing interfaces** (CLI, API, Agents) to the working core library.

### Current Completion Status

| Layer | Status | Notes |
|-------|--------|-------|
| Core Analysis Library | 100% | All algorithms working, validated |
| Data Ingestion | 100% | Streaming, validation, normalization |
| Quality Assurance | 100% | Sanity checks, QC metrics |
| Distributed Processing | 100% | Dask and Sedona backends |
| CLI Commands | 40% | Commands exist but use stubs/mocks |
| API Endpoints | 90% | Missing database persistence, JWT |
| Agent System | 95% | Pipeline agent has stub execution |

---

## Production Readiness Work

### Overview

| Phase | Priority | Description | Dependencies |
|-------|----------|-------------|--------------|
| Phase 1 | P0 | CLI-to-Core Wiring | None |
| Phase 2 | P0 | Agent-to-Core Wiring | Phase 1 patterns |
| Phase 3 | P1 | API Production Hardening | None |
| Phase 4 | P2 | Integration Testing | Phases 1-3 |
| Phase 5 | P3 | Documentation and Polish | All phases |

---

### Phase 1: CLI-to-Core Wiring (P0 - Critical)

**Goal:** Replace all CLI stubs with real core library calls
**Status:** Not Started
**Parallelization:** Each epic can run in parallel

#### Epic 1.1: Wire CLI Analyze Command

**File:** `cli/commands/analyze.py`
**Current State:** Uses `MockAlgorithm` class, returns random data
**Target State:** Call real algorithms from `core/analysis/library/`

| Task | Description | Status |
|------|-------------|--------|
| 1.1.1 | Remove MockAlgorithm class (L394-408) | [x] |
| 1.1.2 | Import algorithm registry from core | [x] |
| 1.1.3 | Wire algorithm lookup by event type | [x] |
| 1.1.4 | Wire algorithm execution with real data | [x] |
| 1.1.5 | Add unit tests for wired analyze command | [x] |

**Dependencies:** None - can start immediately

---

#### Epic 1.2: Wire CLI Ingest Command

**File:** `cli/commands/ingest.py`
**Current State:** "Downloads" by writing text placeholder to file
**Target State:** Use rasterio to fetch real COG/GeoTIFF data

| Task | Description | Status |
|------|-------------|--------|
| 1.2.1 | Remove placeholder download logic | [ ] |
| 1.2.2 | Import streaming ingester from core | [ ] |
| 1.2.3 | Wire download to StreamingIngester | [ ] |
| 1.2.4 | Wire normalization to real normalizer | [ ] |
| 1.2.5 | Integrate image validation | [ ] |
| 1.2.6 | Add integration tests for ingest | [ ] |

**Dependencies:** None - can start immediately

---

#### Epic 1.3: Wire CLI Validate Command ✅ COMPLETE

**File:** `cli/commands/validate.py`
**Current State:** ~~All quality checks return `random.uniform()` scores~~ **FIXED**
**Target State:** Call real QC modules from `core/quality/` ✅

| Task | Description | Status |
|------|-------------|--------|
| 1.3.1 | Remove random score generation | [x] Complete |
| 1.3.2 | Import sanity suite from core | [x] Complete |
| 1.3.3 | Wire spatial validation | [x] Complete |
| 1.3.4 | Wire value validation | [x] Complete |
| 1.3.5 | Wire artifact detection | [x] Complete |
| 1.3.6 | Wire temporal consistency check | [x] Complete |
| 1.3.7 | Add tests for real validation | [x] Complete |

**Dependencies:** None - can start immediately
**Completed:** 2026-01-23

---

#### Epic 1.4: Wire CLI Export Command

**File:** `cli/commands/export.py`
**Current State:** Creates mock GeoJSON, placeholder PNG, text-based "PDF"
**Target State:** Use real ProductGenerator from core

| Task | Description | Status |
|------|-------------|--------|
| 1.4.1 | Remove mock export implementations | [ ] |
| 1.4.2 | Import ProductGenerator from core | [ ] |
| 1.4.3 | Wire GeoTIFF/COG export | [ ] |
| 1.4.4 | Wire GeoJSON export | [ ] |
| 1.4.5 | Wire PDF report generation | [ ] |
| 1.4.6 | Add export format tests | [ ] |

**Dependencies:** Epic 1.1 (needs real analysis output to export)

---

#### Epic 1.5: Wire CLI Run Command

**File:** `cli/commands/run.py`
**Current State:** Multiple stubs returning random/empty data
**Target State:** Orchestrate real pipeline through all stages

| Task | Description | Status |
|------|-------------|--------|
| 1.5.1 | Remove all random/mock fallbacks | [ ] |
| 1.5.2 | Wire discovery to real STAC client | [ ] |
| 1.5.3 | Wire run_analyze to real pipeline runner | [ ] |
| 1.5.4 | Wire run_validate to real QC | [ ] |
| 1.5.5 | Wire run_export to real products | [ ] |
| 1.5.6 | Add end-to-end run tests | [ ] |

**Dependencies:** Epics 1.1, 1.3, 1.4

---

#### Epic 1.6: Wire CLI Discover Command

**File:** `cli/commands/discover.py`
**Current State:** Falls back to `generate_mock_results()`
**Target State:** Always use real STAC discovery
**Status:** COMPLETE (2026-01-23)

| Task | Description | Status |
|------|-------------|--------|
| 1.6.1 | Remove generate_mock_results fallback | [x] *(2026-01-23)* |
| 1.6.2 | Improve error handling for STAC failures | [x] *(2026-01-23)* |
| 1.6.3 | Add retry logic for transient failures | [x] *(2026-01-23)* |
| 1.6.4 | Add discover command tests | [x] *(2026-01-23)* |

**Dependencies:** None - can start immediately
**Completion Summary:** See `EPIC_1.6_COMPLETION_SUMMARY.md`

---

### Phase 2: Agent-to-Core Wiring (P0 - Critical)

**Goal:** Connect agent orchestration to real algorithm execution
**Status:** Not Started
**Parallelization:** Must complete after Phase 1 starts (shared patterns)

#### Epic 2.1: Wire Pipeline Agent to Algorithms

**File:** `agents/pipeline/main.py`
**Current State:** `_execute_algorithm()` returns stub data
**Target State:** Call real algorithms from algorithm registry

| Task | Description | Status |
|------|-------------|--------|
| 2.1.1 | Import algorithm registry in pipeline agent | [ ] |
| 2.1.2 | Replace stub _execute_algorithm implementation | [ ] |
| 2.1.3 | Handle algorithm configuration | [ ] |
| 2.1.4 | Add error handling for algorithm failures | [ ] |
| 2.1.5 | Add agent-algorithm integration tests | [ ] |

**Dependencies:** Patterns established in Phase 1

---

### Phase 3: API Production Hardening (P1 - High)

**Goal:** Make API production-ready with persistence and security
**Status:** Not Started
**Parallelization:** Can run parallel to Phases 1 and 2

#### Epic 3.1: Database Persistence

**File:** `api/dependencies.py`
**Current State:** `execute()` raises `NotImplementedError`
**Target State:** Real database persistence (PostgreSQL or SQLite)

| Task | Description | Status |
|------|-------------|--------|
| 3.1.1 | Choose and add database library | [ ] |
| 3.1.2 | Define database schema | [ ] |
| 3.1.3 | Implement database connection pool | [ ] |
| 3.1.4 | Replace _events_store dict | [ ] |
| 3.1.5 | Add database tests | [ ] |

**Dependencies:** None - can start immediately
**Decision Required:** Database choice (PostgreSQL vs SQLite)

---

#### Epic 3.2: JWT Authentication

**File:** `api/dependencies.py`
**Current State:** Tokens accepted but not validated
**Target State:** Proper JWT validation with claims

| Task | Description | Status |
|------|-------------|--------|
| 3.2.1 | Add PyJWT dependency | [x] Completed 2026-01-23 |
| 3.2.2 | Implement JWT validation | [x] Completed 2026-01-23 |
| 3.2.3 | Add role-based access control | [ ] Deferred (not in spec) |
| 3.2.4 | Add refresh token support | [ ] Deferred (not in spec) |
| 3.2.5 | Add auth tests | [x] Completed 2026-01-23 |

**Dependencies:** None - can start immediately
**Decision Required:** JWT secret management strategy - RESOLVED: Environment variables (JWT_SECRET, JWT_ALGORITHM, JWT_ISSUER)

---

#### Epic 3.3: Wire Intent Resolution API

**File:** `api/routes/events.py`
**Current State:** ✅ **COMPLETE** - Real intent resolution wired
**Target State:** Call real intent resolver from core

| Task | Description | Status |
|------|-------------|--------|
| 3.3.1 | Import intent resolver from core | [x] *(Completed 2026-01-23)* |
| 3.3.2 | Wire natural language parsing | [x] *(Completed 2026-01-23)* |
| 3.3.3 | Add intent validation | [x] *(Completed 2026-01-23)* |
| 3.3.4 | Add intent resolution tests | [x] *(Completed 2026-01-23)* |

**Dependencies:** None - can start immediately

---

### Phase 4: Integration Testing (P2 - Medium)

**Goal:** Verify end-to-end system works
**Status:** Not Started
**Parallelization:** Must complete after Phases 1-3

#### Epic 4.1: CLI Integration Tests

| Task | Description | Status |
|------|-------------|--------|
| 4.1.1 | Create CLI test harness | [ ] |
| 4.1.2 | Add discover-to-analyze flow test | [ ] |
| 4.1.3 | Add analyze-to-export flow test | [ ] |
| 4.1.4 | Add error handling tests | [ ] |

**Dependencies:** Phase 1 complete

---

#### Epic 4.2: API Integration Tests

| Task | Description | Status |
|------|-------------|--------|
| 4.2.1 | Create API test client | [ ] |
| 4.2.2 | Add event submission flow test | [ ] |
| 4.2.3 | Add concurrent request test | [ ] |
| 4.2.4 | Add database persistence test | [ ] |

**Dependencies:** Phase 3 complete

---

#### Epic 4.3: Agent Integration Tests

| Task | Description | Status |
|------|-------------|--------|
| 4.3.1 | Add orchestrator flow test | [ ] |
| 4.3.2 | Add pipeline execution test | [ ] |
| 4.3.3 | Add failure recovery test | [ ] |

**Dependencies:** Phase 2 complete

---

### Phase 5: Documentation and Polish (P3 - Low)

**Goal:** Update documentation to reflect production state
**Status:** Not Started
**Parallelization:** Can run parallel to Phases 1-4

#### Epic 5.1: Update Documentation

| Task | Description | Status |
|------|-------------|--------|
| 5.1.1 | Update CLAUDE.md | [ ] |
| 5.1.2 | Update API documentation | [ ] |
| 5.1.3 | Update ROADMAP.md | [ ] |
| 5.1.4 | Archive production OpenSpec | [ ] |

**Dependencies:** All phases complete

---

## Parallel Work Streams

### Stream A: CLI Commands (Independent)
- Epic 1.1 (Analyze)
- Epic 1.2 (Ingest)
- Epic 1.3 (Validate)
- Epic 1.6 (Discover)

### Stream B: API Hardening (Independent)
- Epic 3.1 (Database)
- Epic 3.2 (JWT)
- Epic 3.3 (Intent)

### Stream C: Sequential Work
- Epic 1.4 (Export) - After 1.1
- Epic 1.5 (Run) - After 1.1, 1.3, 1.4
- Epic 2.1 (Agent) - After Phase 1 patterns established
- Epic 4.x (Testing) - After relevant phases complete

---

## Success Criteria

### Phase 1 Complete When:
- [ ] `flight analyze` produces real classification results
- [ ] `flight ingest` downloads real satellite data
- [x] `flight validate` produces real QC scores *(Completed 2026-01-23)*
- [ ] `flight export` creates valid GeoTIFF/GeoJSON/PDF
- [ ] `flight run` executes full pipeline end-to-end
- [x] `flight discover` never falls back to mocks *(Completed 2026-01-23)*

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
- [ ] OpenSpec archived

---

## Completed Work

### Implementation History

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Foundation (schemas, validation, taxonomy) | COMPLETE |
| Phase 2 | Intelligence (intent, discovery, selection) | COMPLETE |
| Phase 3 | Analysis Pipeline (algorithms, execution) | COMPLETE |
| Phase 4 | Data Engineering (ingestion, caching) | COMPLETE |
| Phase 5 | Quality & Resilience (QC, fallbacks) | COMPLETE |
| Phase 6 | Orchestration & Deployment (agents, API) | COMPLETE |

### P0 Bug Fixes - COMPLETE

All critical bugs have been resolved.

| Task | Bug ID | Description | File | Status |
|------|--------|-------------|------|--------|
| BUG-001 | FIX-003 | WCS duplicate dict key | `core/data/discovery/wms_wcs.py:374-382` | COMPLETE |
| BUG-002 | FIX-004 | scipy grey_dilation | `core/analysis/library/baseline/flood/hand_model.py:307` | COMPLETE |
| BUG-003 | FIX-005 | distance_transform_edt tuple unpacking | `core/analysis/library/baseline/flood/hand_model.py:384-387` | COMPLETE |
| BUG-004 | FIX-006 | processing_level schema definition | `openspec/schemas/common.schema.json:115-119` | COMPLETE |

**Details:** See `FIXES.md` for documentation of all fixes.

### Stream A: Image Validation - COMPLETE

Production-grade image validation for the ingestion workflow.

**Implemented Files:**
- `core/data/ingestion/validation/exceptions.py` - 10 exception types
- `core/data/ingestion/validation/config.py` - Configuration with YAML/env support
- `core/data/ingestion/validation/image_validator.py` - Main validator + dataclasses
- `core/data/ingestion/validation/band_validator.py` - Optical band validation
- `core/data/ingestion/validation/sar_validator.py` - SAR speckle-aware validation
- `core/data/ingestion/validation/screenshot_generator.py` - Matplotlib screenshots
- `config/ingestion.yaml` - Validation configuration
- `tests/test_image_validation.py` - Unit tests
- `tests/integration/test_validation_integration.py` - Integration tests

**Full Requirements:** `docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`

### Stream B: Distributed Raster Processing (Dask) - COMPLETE

Enable parallel tile processing on laptops and cloud clusters.

**Implemented Files:**
- `core/data/ingestion/virtual_index.py` - VirtualRasterIndex, STACVRTBuilder, TileAccessor
- `core/analysis/execution/dask_tiled.py` - DaskTileProcessor, DaskProcessingConfig
- `core/analysis/execution/dask_adapters.py` - DaskAlgorithmAdapter, wrap_algorithm_for_dask
- `core/analysis/execution/router.py` - ExecutionRouter, auto_route, ResourceEstimator
- `tests/test_dask_execution.py` - Comprehensive test suite

**Key Features:**
- Automatic backend selection (serial -> tiled -> Dask local -> distributed)
- Tile-level parallelization with configurable workers (4-8x speedup target)
- Memory-efficient streaming for datasets larger than RAM
- Algorithm adapters to wrap any existing algorithm for Dask
- Progress tracking and callbacks
- Multiple blend modes for tile stitching

**Configuration Presets:**
- `DaskProcessingConfig.for_laptop(memory_gb=4.0)`
- `DaskProcessingConfig.for_workstation(memory_gb=16.0)`
- `DaskProcessingConfig.for_cluster(scheduler_address)`

### Stream C: Distributed Processing - Cloud (Sedona) - COMPLETE

Apache Sedona integration for continental-scale geospatial processing on Spark clusters.

**Implemented Files:**
- `core/analysis/execution/sedona_backend.py` - SedonaBackend, SedonaTileProcessor, SedonaConfig
- `core/analysis/execution/sedona_adapters.py` - SedonaAlgorithmAdapter, AlgorithmSerializer, ResultCollector
- `core/analysis/execution/router.py` - SEDONA/SEDONA_CLUSTER profiles
- `tests/test_sedona_execution.py` - Comprehensive test suite

**Key Features:**
- Continental-scale processing (100,000 km^2, 10,000+ tiles)
- Apache Sedona raster functions (RS_FromGeoTiff, RS_Tile, RS_MapAlgebra)
- Automatic Spark cluster detection and configuration
- Mock mode for development without Spark installation
- Graceful fallback to Dask when Spark unavailable
- Algorithm adapters for flood, wildfire, and storm detection

**Execution Profiles:**
- `ExecutionProfile.SEDONA` - Sedona local mode
- `ExecutionProfile.SEDONA_CLUSTER` - Sedona on remote Spark cluster
- `ExecutionProfile.CONTINENTAL` - Auto-select best backend for 10,000+ tiles

---

## Platform Metrics

- **82,776 lines** of core processing code
- **53,294 lines** of comprehensive tests
- **20,343 lines** of agent orchestration code
- **13,109 lines** of API and CLI interfaces
- **8 baseline algorithms** production-ready
- **518+ passing tests** across all subsystems
- **Deployment-ready** with Docker, Kubernetes, and cloud configurations

### Implemented Algorithms

**Flood:**
- SAR threshold detection
- NDWI optical detection
- Change detection (pre/post)
- HAND model (Height Above Nearest Drainage)

**Wildfire:**
- Thermal anomaly detection
- dNBR burn severity
- Burned area classification

**Storm:**
- Wind damage assessment
- Structural damage analysis

**Advanced:**
- UNet segmentation (experimental)
- Ensemble fusion (experimental)

---

## Deployment Targets

| Environment | Status |
|-------------|--------|
| Laptop (4GB RAM) | COMPLETE |
| Workstation (16GB RAM) | COMPLETE |
| Docker Compose | COMPLETE |
| Kubernetes | COMPLETE |
| AWS Lambda | COMPLETE |
| AWS ECS/Batch | COMPLETE |
| GCP Cloud Run | COMPLETE |
| Edge (Raspberry Pi) | COMPLETE |
| Spark Cluster | COMPLETE |
| Dask Cluster | COMPLETE |

---

## Project Principles

1. **Situation-Agnostic:** Same pipeline handles floods, fires, storms
2. **Reproducible:** Deterministic selections, version pinning, provenance tracking
3. **Resilient:** Graceful degradation, comprehensive fallback strategies
4. **Scalable:** Laptop to cloud with same codebase
5. **Open-First:** Prefer open data and open-source tools
6. **Fast Response:** Optimized for emergency scenarios

---

## Getting Started

```bash
# Run tests
./run_tests.py                    # All 518+ tests
./run_tests.py flood              # Flood-specific tests

# Run analysis (laptop mode)
python run_real_analysis.py       # Miami flood analysis

# Start API
docker-compose up                 # Full stack
# OR
uvicorn api.main:app --reload     # Development mode

# Use CLI
flight run --event examples/flood_event.yaml --profile laptop
```

---

## Agent Coordination

See `.claude/agents/PROJECT_MEMORY.md` for:
- Current project context and history
- Active work groups and their status
- Agent assignment tracking
- Decision log

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

## File Change Summary

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

## Core Libraries to Wire

| Core Module | CLI Target | API Target | Agent Target |
|-------------|------------|------------|--------------|
| `core.analysis.library.AlgorithmRegistry` | analyze.py | - | pipeline/main.py |
| `core.data.ingestion.StreamingIngester` | ingest.py | - | - |
| `core.quality.sanity.SanitySuite` | validate.py | - | quality/main.py |
| `core.quality.reporting.ProductGenerator` | export.py | products.py | reporting/main.py |
| `core.intent.IntentResolver` | - | events.py | - |
| `core.data.discovery.STACClient` | discover.py | - | discovery/main.py |

---

**Full Specification:** `OPENSPEC_PRODUCTION.md`
**Next Review:** After Phase 1 complete
