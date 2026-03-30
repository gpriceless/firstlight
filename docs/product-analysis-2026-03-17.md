# FirstLight Product Analysis

**Date:** 2026-03-17
**Author:** Product Queen (Sentinel Pipeline)
**Issue:** LGT-20

---

## 1. Module Ownership Map

### Core Domain (102,392 LOC across 212 files)

| Module | Files | LOC | What It Does | Maturity | Test Coverage | Maintainability |
|--------|-------|-----|-------------|----------|---------------|-----------------|
| **core/analysis** | 48 | 31,371 | Algorithm library (flood/wildfire/storm), pipeline assembly, execution routing, distributed processing (Dask + Sedona) | **High** -- 8 baseline algorithms validated, assembly engine complete, execution router with 5 profiles | Strong -- dedicated markers per algorithm, selection tests, fusion tests | Good -- registry pattern, clean abstractions, algorithm adapters |
| **core/data** | 76 | 32,621 | Data discovery (STAC, WMS/WCS), ingestion (streaming, validation, band stacking), provider registry, normalization | **High** -- 13 data source integrations, multi-band ingestion complete | Strong -- ingestion tests, provider tests, validation tests | Good but large -- 76 files is the biggest module, could benefit from further decomposition |
| **core/quality** | 22 | 15,329 | Sanity checks, QC metrics, diagnostics (spatial/temporal), flagging, uncertainty estimation, QA reporting | **High** -- all 36 bugs fixed, production-grade | Moderate -- tests exist but some edge cases were bug sources | Good -- flag registry, clean separation of concerns |
| **core/resilience** | 20 | 10,505 | Failure handling, degraded modes, fallback strategies, risk assessments, circuit breakers | **Medium-High** -- implemented but less battle-tested than analysis/data | Low-Moderate -- resilience patterns hard to test without production load | Solid design -- patterns well-structured |
| **core/reporting** | 22 | 5,441 | Maps (static + interactive), PDF generation, web reports, design system, templates | **Medium** -- REPORT-2.0 partially complete; maps, PDF, web reports done; data integrations partial | Moderate -- 29 interactive report tests, 13 PDF tests, map tests | Good -- Jinja2 templates, WeasyPrint PDF, Folium maps |
| **core/context** | 4 | 1,392 | Context data lakehouse (buildings, infrastructure, datasets, weather) | **Low** -- stub/early implementation | Minimal | Needs work -- only 4 files for a large domain |
| **core/execution** | 4 | 3,280 | State management, tiling, execution profiles | **Medium** -- functional but tightly coupled to control plane | Low | Adequate |
| **core/intent** | 4 | 993 | Event classification, keyword matching, NLP intent resolution | **High** -- wired to API, working in production path | Good -- dedicated intent tests | Simple and clean |
| **core/ogc** | 4 | 609 | OGC API Processes adapters (pygeoapi) | **Low-Medium** -- adapters exist but lightly tested | Low | Small scope, maintainable |
| **core/stac** | 4 | 724 | STAC catalog generation, metadata publishing | **Low-Medium** -- catalog generation works, publishing is new | Low | Small scope |
| **core/events** | 2 | 127 | Event submission models | **Low** -- minimal implementation | None visible | Trivial scope |
| **core/provenance** | 1 | 0 | Lineage tracking | **None** -- empty file, placeholder only | None | Not implemented |

### Interface Layers

| Module | Files | LOC | What It Does | Maturity | Notes |
|--------|-------|-----|-------------|----------|-------|
| **api/** | 37 | 13,579 | FastAPI REST API with 5 route groups (v1, control, internal, OGC, STAC) | **Medium-High** -- core routes working, control plane routes new | JWT auth implemented, DB persistence still TODO |
| **agents/** | 33 | 22,306 | LLM orchestration layer (base agent, discovery, pipeline, quality, reporting agents) | **Medium** -- base framework solid, individual agents at varying completion | Agent pipeline has stub execution in places |
| **cli/** | 11 | 5,412 | Click-based CLI (`flight` command) | **Medium** -- 3/6 commands fully wired (analyze, validate, discover); ingest/export/run still have stubs | Critical path: CLI-to-core wiring is the #1 remaining work |
| **workers/** | 7 | 1,622 | Taskiq async workers (webhooks, OGC, pipeline, maintenance) | **Low-Medium** -- infrastructure exists but lightly tested | New addition from control plane work |
| **tests/** | 90 | 70,339 | Comprehensive test suite organized by hazard type, component, and test type | **High** -- 518+ tests passing, excellent coverage of core algorithms | Gap: reporting module lacks dedicated test category |

### Deployment (non-Python)

| Artifact | What It Does | Maturity |
|----------|-------------|----------|
| **docker/** | 5 Docker images (base, api, cli, core, worker) | High -- multi-stage builds, production-ready |
| **docker-compose** | 3 configs (full, dev, minimal) | High |
| **deploy/kubernetes/** | Full K8s manifests (deployments, HPA, PVCs, secrets, ingress) | Medium-High |
| **deploy/gcp/** | Cloud Run + GKE configs | Medium |
| **deploy/aws/** | Lambda serverless config | Medium |
| **deploy/azure/** | AKS deployment | Medium |
| **deploy/on-prem/** | Ansible playbooks (standalone + cluster) | Medium |
| **.github/workflows/** | CI/CD (build, push, staging, production) | Medium-High |

---

## 2. Architecture Assessment

### Strengths

1. **Clean domain separation** -- Core domain logic (102K LOC) is fully isolated from interface layers (API, CLI, agents). This is textbook architecture for a processing platform.

2. **Registry-driven extensibility** -- Four registries (Algorithm, EventClass, Provider, Flag) enable plugin-style extension without modifying core code. Adding a new algorithm or data provider is a well-defined process.

3. **Pipeline assembly engine** -- The assembler/graph/optimizer/validator pattern in `core/analysis/assembly/` (3.5K LOC) is sophisticated. It builds dynamic DAGs based on available data and event type, which is the right approach for a multi-hazard platform.

4. **Scale-aware execution** -- The execution router automatically selects backend (serial, tiled, Dask local, Dask distributed, Sedona) based on data size. This is production-grade scaling.

5. **Multi-cloud deployment** -- Docker, K8s, Lambda, Cloud Run, AKS, on-prem -- comprehensive coverage for a geospatial tool that needs to run where the data is.

6. **Test discipline** -- 518+ tests, 70K LOC of test code, organized by hazard type and component with pytest markers. Strong foundation for CI/CD.

### Concerns

1. **CLI wiring gap (P0)** -- 3 of 6 CLI commands still use stubs or mocks. `flight ingest`, `flight export`, and `flight run` don't use real core library calls. This is the single biggest gap between "core works" and "product works."

2. **No database persistence** -- The API stores events in an in-memory dict. `api/dependencies.py execute()` raises `NotImplementedError`. This blocks any deployment where events need to survive a restart.

3. **Provenance is empty** -- `core/provenance/` contains 0 LOC. For a disaster response tool, lineage tracking ("which algorithm, which data, which parameters produced this result") is not optional -- it's a regulatory and trust requirement.

4. **Agent pipeline has stubs** -- `agents/pipeline/main.py _execute_algorithm()` returns stub data. The agent layer looks complete from the outside but doesn't execute real analysis.

5. **Context data lakehouse is embryonic** -- 4 files, 1,392 LOC for what should be a rich contextual data layer (buildings, infrastructure, weather, population). REPORT-2.0 data integrations (Census, OSM, emergency resources) are partially done but not connected to context module.

6. **Resilience is untested under load** -- 10.5K LOC of resilience code (circuit breakers, fallbacks, degraded modes) but no load testing or chaos engineering evidence. These patterns only prove their value under stress.

### Technical Debt

| Area | Severity | Description |
|------|----------|-------------|
| CLI stubs | **High** | 3 commands use mock/placeholder logic |
| DB persistence | **High** | API stores data in-memory only |
| Provenance | **Medium** | Empty module, no lineage tracking |
| Agent stubs | **Medium** | Pipeline agent doesn't execute real algorithms |
| REPORT-2.0 data gaps | **Medium** | Census/OSM integrations partial (R2.4.4-R2.4.10 incomplete) |
| WCAG compliance | **Low** | R2.1.7 (contrast verification) and R2.1.9 (icon library) incomplete |
| B&W print patterns | **Low** | R2.3.10 and R2.6.7 deferred |

---

## 3. Product Recommendations (Prioritized)

### P0: Ship-Blocking -- Must Complete Before Any Production Use

**1. Finish CLI-to-Core Wiring (Phase 1 remaining)**
- Wire `flight ingest` (Epic 1.2, depends on completed Epic 1.7)
- Wire `flight export` (Epic 1.4)
- Wire `flight run` (Epic 1.5 -- orchestrates the full pipeline)
- **Why:** Without this, the CLI is a demo, not a tool. Users cannot run an end-to-end analysis.
- **Effort:** Medium -- the hardest part (Epic 1.7, multi-band ingestion) is already done.

**2. Add Database Persistence (Epic 3.1)**
- Choose PostgreSQL (aligns with PostGIS already in stack)
- Implement event storage, job tracking, result persistence
- **Why:** The API cannot be deployed without this. Events vanish on restart.
- **Effort:** Medium -- schema design is straightforward given existing Pydantic models.

### P1: High Value -- Should Follow Immediately

**3. Wire Agent Pipeline to Real Algorithms (Epic 2.1)**
- Replace stub `_execute_algorithm()` with real AlgorithmRegistry calls
- **Why:** The agent layer is the autonomous execution path. Without this, LLM orchestration is decorative.
- **Effort:** Low-Medium -- patterns established by CLI wiring.

**4. Implement Provenance Tracking**
- Record: input data sources, algorithm versions, parameters, execution timestamps, quality scores
- Store alongside results (JSON sidecar or DB records)
- **Why:** Disaster response products need audit trails. "Where did this flood map come from?" must be answerable.
- **Effort:** Medium -- touches multiple modules but well-scoped.

**5. Complete REPORT-2.0 Data Integrations (R2.4.4-R2.4.10)**
- Population estimates, hospital/school/shelter locations, facilities-in-flood-zone calculations
- **Why:** The reporting layer generates beautiful maps but the "Who Is Affected" section needs real data.
- **Effort:** Medium -- clients exist (Census, OSM), individual queries need implementation.

### P2: Strategic -- Important for Positioning

**6. Integration Testing Suite (Phase 4)**
- End-to-end CLI, API, and agent flow tests
- Remove all remaining mock/stub code from production paths
- **Why:** Confidence gate before any real deployment.

**7. Context Data Lakehouse Completion**
- Expand from 4 files to a real contextual layer
- Connect REPORT-2.0 data clients to the context module
- **Why:** Contextual enrichment (population, infrastructure, weather) is what separates a "flood detection tool" from a "flood intelligence platform."

### What Looks Good (Keep/Invest)

- **Algorithm library** -- 8 validated algorithms across 3 hazard types. This is the crown jewel.
- **Pipeline assembly** -- Dynamic DAG construction is sophisticated and correct.
- **Execution routing** -- Automatic backend selection from laptop to Spark cluster.
- **Multi-band ingestion** -- Epic 1.7 completion was a critical milestone.
- **Map visualization** -- Static and interactive maps with proper cartographic furniture.
- **Test suite** -- 518+ tests is a strong foundation.

### What to Deprecate/Remove

- **MockAlgorithm references** -- Any remaining mock/stub code in production paths should be deleted, not commented out.
- **generate_mock_results()** in discover.py -- Already removed but verify no other mock generators remain.
- **Text-based "PDF" export** -- Replace entirely with WeasyPrint pipeline from REPORT-2.0.

---

## 4. Integration Assessment

### Portfolio Fit

FirstLight occupies the **geospatial event intelligence** niche -- it converts satellite imagery + event context into decision products for emergency managers. Within the broader project portfolio:

| Project | Domain | Potential Synergy |
|---------|--------|-------------------|
| **MAIA Pipeline** | Solar site feasibility (geospatial ML) | **High** -- shared geospatial stack (rasterio, STAC, PostGIS), shared satellite data sources (Sentinel-2, Landsat). MAIA's medallion architecture (Bronze/Silver/Gold) could inform FirstLight's context data lakehouse design. |
| **detr_geo** | Geospatial ML (object detection) | **Medium-High** -- detr_geo's ML models could enhance FirstLight's advanced algorithms (UNet segmentation, ensemble fusion). FirstLight's pipeline assembly could orchestrate detr_geo inference as an algorithm plugin. |
| **Second Brain** | Knowledge management | **Low** -- no direct technical overlap, but FirstLight analysis results could be indexed as knowledge artifacts. |

### Cross-Project Opportunities

1. **Shared STAC Discovery Layer** -- Both FirstLight and MAIA need STAC catalog discovery for satellite imagery. FirstLight's `core/data/discovery/stac_client.py` (with multi-band support) could become a shared library.

2. **Shared Geospatial Validation** -- FirstLight's image validation pipeline (`core/data/ingestion/validation/`) is production-grade and sensor-aware. MAIA will need similar validation for solar irradiance data.

3. **Shared PostGIS Schema Patterns** -- FirstLight's LLM control plane uses PostGIS for state management. MAIA uses PostGIS (Gold tier) for feasibility results. Common schema patterns and migration tooling could be shared.

4. **ML Model Integration** -- FirstLight already has experimental UNet and ensemble fusion. detr_geo's object detection could plug into the algorithm registry as a "structural damage detection" or "infrastructure identification" algorithm.

---

## 5. Risk Assessment

### High Severity

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **CLI stubs in production path** | Users run `flight run` and get mock data without knowing it | High (current state) | P0: Complete CLI wiring. Add runtime warnings if any stub is hit. |
| **No DB persistence** | API data loss on restart; cannot track historical events | High (current state) | P0: Implement PostgreSQL persistence (Epic 3.1) |
| **External STAC catalog dependency** | Real analysis fails if Earth Search / Planetary Computer is down | Medium | VCR-style test recordings exist; add runtime fallback messaging and retry logic |

### Medium Severity

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **No provenance/audit trail** | Cannot explain or reproduce results; liability in emergency response context | Medium | Implement provenance module before any real-world deployment |
| **Agent pipeline stubs** | LLM control plane routes exist but agent execution is decorative | Medium | Wire after CLI wiring (shared patterns) |
| **Memory pressure with real data** | Satellite imagery is large; in-memory processing may OOM on constrained hardware | Low-Medium | Dask tiling already implemented; validate with real Sentinel-2 scenes |
| **36 bugs were found and fixed** | Indicates code was written fast; similar bugs may lurk in less-tested paths | Medium | Focus testing on REPORT-2.0 data integrations and context module (newer, less tested) |

### Low Severity

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **WCAG compliance gaps** | Reports not accessible to all users | Low (not deployed yet) | Schedule accessibility audit before public release |
| **Resilience code untested** | Circuit breakers and fallbacks may not work as designed | Low (no production load yet) | Add chaos testing before production deployment |
| **Geospatial dependency complexity** | GDAL/rasterio/pyproj version conflicts | Low (managed via Docker) | Pin versions in Docker base image |

### Security Considerations

| Area | Status | Notes |
|------|--------|-------|
| **JWT Auth** | Implemented | Core validation done; refresh tokens and RBAC deferred |
| **API Input Validation** | Pydantic models | Good -- FastAPI + Pydantic provides automatic validation |
| **CORS** | Configured | Review allowed origins before production |
| **Secrets Management** | Environment variables | Adequate for current stage; consider Vault for production |
| **Docker Images** | Multi-stage builds | Good -- minimizes attack surface |
| **Dependency Scanning** | Not visible | Recommend adding `pip-audit` or Snyk to CI pipeline |

---

## Summary

FirstLight is a **substantial and well-architected** geospatial platform. The core domain (102K LOC) is mature, with validated algorithms, sophisticated pipeline assembly, and scale-aware execution. The main gap is **interface wiring** -- the CLI, agent, and API layers don't fully use the core they wrap.

**Top 3 actions:**
1. Finish CLI wiring (especially `flight run` end-to-end)
2. Add PostgreSQL persistence
3. Implement provenance tracking

The platform is closer to production-ready than it appears -- the hard geospatial work (algorithms, data ingestion, distributed processing) is done. What remains is plumbing, persistence, and polish.
