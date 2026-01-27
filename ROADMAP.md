# FirstLight: Implementation Roadmap

**Last Updated:** 2026-01-26
**Status:** Core Platform Complete, Interface Wiring In Progress, Reporting Overhaul Approved

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

**Dependencies:** **Epic 1.7 (Multi-Band Asset Download)** - Real Sentinel-2 ingestion requires multi-band download capability

**BLOCKER:** Epic 1.7 must complete before Epic 1.2 can process real Sentinel-2 data. See `docs/SENTINEL2_INGESTION_BUG_REPORT.md` for details.

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

#### Epic 1.7: Multi-Band Asset Download Refactor (P0 - Critical) ✅ COMPLETE

**Status:** COMPLETE (2026-01-25)
**Priority:** P0 - Blocks all real Sentinel-2 data processing
**Bug Report:** `docs/SENTINEL2_INGESTION_BUG_REPORT.md`

**Problem Statement:**
The STAC client returns URLs for True Color Images (TCI.tif - 3-band RGB composites) while the validator expects individual spectral bands (blue, green, red, nir). This is a fundamental mismatch that causes all real Sentinel-2 downloads to fail validation.

**Root Cause:** Line 446 in `core/data/discovery/stac_client.py`:
```python
"url": item.assets.get("visual") or item.assets.get("data") or "",
```

The `"visual"` asset is TCI.tif (RGB composite for visualization), not the individual spectral bands needed for scientific analysis.

**Solution:** Refactor the data flow to download individual spectral band files and stack them for analysis.

##### Task Dependency Graph

```
                    ┌─────────────────────────┐
                    │ 1.7.1 STAC Client       │
                    │ (Return Band URLs)      │
                    └───────────┬─────────────┘
                                │
            ┌───────────────────┼───────────────────┐
            │                   │                   │
            v                   v                   v
┌───────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ 1.7.2 Ingestion   │ │ 1.7.3 Band Stacking │ │ 1.7.4 Immediate     │
│ Pipeline          │ │ Utility (NEW)       │ │ Workaround          │
│ (Multi-file DL)   │ │                     │ │ (--skip-validation) │
└─────────┬─────────┘ └─────────┬───────────┘ └─────────────────────┘
          │                     │                   (parallel, optional)
          └──────────┬──────────┘
                     │
                     v
          ┌─────────────────────┐
          │ 1.7.5 Validator     │
          │ Updates             │
          └─────────┬───────────┘
                    │
                    v
          ┌─────────────────────┐
          │ 1.7.6 Integration   │
          │ Testing             │
          └─────────────────────┘
```

##### Tasks

| Task | Description | Status | Parallel? | Depends On |
|------|-------------|--------|-----------|------------|
| 1.7.1 | STAC client returns individual band URLs | [x] Complete (2026-01-25) | First | None |
| 1.7.2 | Ingestion pipeline handles multi-file downloads | [x] Complete (2026-01-25) | No | 1.7.1 |
| 1.7.3 | Band stacking utility (VRT creation) | [x] Complete (2026-01-25) | **Yes** | 1.7.1 |
| 1.7.4 | Add `--skip-validation` workaround flag | [x] Complete (2026-01-25) | **Yes** | None |
| 1.7.5 | Validator updates for stacked files | [x] Complete (2026-01-25) | No | 1.7.2, 1.7.3 |
| 1.7.6 | Integration testing with real Sentinel-2 | [x] Complete (2026-01-25) | No | 1.7.1-1.7.5 |

##### Task Details

**1.7.1: STAC Client Changes**
- **File:** `core/data/discovery/stac_client.py`
- **Changes:**
  - Add `SENTINEL2_ANALYSIS_BANDS` constant mapping band names to asset keys
  - Modify `discover_data()` to return `band_urls` dict instead of single `url`
  - Preserve backward compatibility for non-Sentinel sources
- **Effort:** Medium

**1.7.2: Ingestion Pipeline Changes**
- **Files:** `cli/commands/ingest.py`, `core/data/ingestion/streaming.py`
- **Changes:**
  - Update `process_item()` to iterate over `band_urls`
  - Download each band file to separate path
  - Call band stacking after all bands downloaded
  - Update progress reporting for multi-file downloads
- **Effort:** Medium-High

**1.7.3: Band Stacking Utility (NEW FILE)**
- **File:** `core/data/ingestion/band_stack.py` (new)
- **Changes:**
  - Create `create_band_stack(band_paths: Dict[str, Path], output_path: Path) -> Path`
  - Generate VRT file combining individual bands
  - Support band ordering configuration
  - Add band metadata to output
- **Effort:** Medium

**1.7.4: Immediate Workaround (Optional)**
- **Files:** `cli/commands/ingest.py`, `core/data/ingestion/streaming.py`
- **Changes:**
  - Add `--skip-validation` CLI flag
  - Add warning message when flag used
  - Document that TCI files are for visualization only
- **Effort:** Low
- **Note:** This is a temporary workaround for users who need data immediately

**1.7.5: Validator Updates**
- **Files:** `core/data/ingestion/validation/image_validator.py`, `band_validator.py`
- **Changes:**
  - Detect VRT stacked files
  - Validate band presence in stack
  - Update band matching logic for stacked format
- **Effort:** Low-Medium

**1.7.6: Integration Testing**
- **Files:** `tests/integration/test_sentinel2_ingestion.py` (new)
- **Changes:**
  - Test full flow: discover -> download bands -> stack -> validate
  - Test with real Earth Search STAC catalog
  - Test error handling for missing bands
  - Test with VCR recording for offline testing
- **Effort:** Medium

##### Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `core/data/discovery/stac_client.py` | Modify | Return individual band URLs |
| `cli/commands/ingest.py` | Modify | Handle multi-band downloads |
| `cli/commands/discover.py` | Minor | Update output format for band_urls |
| `core/data/ingestion/streaming.py` | Modify | Support multi-file ingestion |
| `core/data/ingestion/band_stack.py` | **New** | Band stacking utility |
| `core/data/ingestion/validation/image_validator.py` | Minor | Adjust for stacked files |
| `core/data/ingestion/validation/band_validator.py` | Minor | Update band matching |
| `tests/integration/test_sentinel2_ingestion.py` | **New** | Integration tests |

##### Success Criteria

- [x] `flight discover` returns `band_urls` for Sentinel-2 items *(Completed 2026-01-25)*
- [x] `flight ingest` downloads all spectral bands (blue, green, red, nir, swir16, swir22) *(Completed 2026-01-25)*
- [x] Band stacking creates valid VRT/GeoTIFF with correct band order *(Completed 2026-01-25)*
- [x] Image validation passes for stacked multi-band files *(Completed 2026-01-25)*
- [x] `--skip-validation` flag implemented for immediate workaround *(Completed 2026-01-25)*
- [x] Integration tests verify all Epic 1.7 components *(Completed 2026-01-25)*

##### Parallelization Strategy

**Can Run in Parallel:**
- Task 1.7.3 (Band Stacking) and Task 1.7.4 (Workaround) can run in parallel with Task 1.7.2 once 1.7.1 completes
- Task 1.7.4 can start immediately (independent workaround)

**Must Be Sequential:**
- Task 1.7.1 must complete first (STAC client provides data structure)
- Task 1.7.2 depends on 1.7.1 (ingestion needs new data format)
- Task 1.7.5 depends on 1.7.2 and 1.7.3 (validator needs stacked files)
- Task 1.7.6 depends on all above (end-to-end testing)

**Estimated Total Effort:** 3-4 days with 2 parallel workers

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

### Phase 2B: Human-Readable Reporting (REPORT-2.0)

**Goal:** Transform technical reports into actionable intelligence for emergency managers
**Status:** APPROVED - Ready for Implementation
**Priority:** P0 (Foundation) / P1 (Extensions)
**Detailed Spec:** `docs/specs/OPENSPEC_REPORTING_OVERHAUL_DRAFT.md`
**Parallelization:** Can run parallel to Phase 2 and Phase 3

#### Epic R2.1: Design System Implementation (P0 - Foundation)

**Priority:** P0 - Blocks all other reporting features
**Effort:** 5-7 days

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.1.1 | Create CSS tokens file with design system colors | [x] *(Completed 2026-01-26)* | None |
| R2.1.2 | Implement flood severity palette (5 levels) | [x] *(Completed 2026-01-26 - included in R2.1.1)* | R2.1.1 |
| R2.1.3 | Implement wildfire severity palette (6 levels) | [x] *(Completed 2026-01-26 - included in R2.1.1)* | R2.1.1 |
| R2.1.4 | Implement confidence/uncertainty palette (4 levels) | [x] *(Completed 2026-01-26 - included in R2.1.1)* | R2.1.1 |
| R2.1.5 | Implement typography and spacing scales | [x] *(Completed 2026-01-26 - included in R2.1.1)* | R2.1.1 |
| R2.1.6 | Create component CSS classes (.metric-card, .alert-box, etc.) | [x] *(Completed 2026-01-26)* | R2.1.1 |
| R2.1.7 | Verify WCAG AA contrast with axe-core | [ ] | R2.1.2-R2.1.4 |
| R2.1.8 | Create print stylesheet | [x] *(Completed 2026-01-26 - included in design_tokens.css and components.css)* | R2.1.6 |
| R2.1.9 | Set up icon library (Heroicons) | [ ] | None |

**Files to Create:**
- `agents/reporting/static/css/tokens.css`
- `agents/reporting/static/css/components.css`
- `agents/reporting/static/css/utilities.css`
- `agents/reporting/static/css/print.css`
- `agents/reporting/static/css/main.css`

---

#### Epic R2.2: Plain-Language Report Templates (P0 - Core)

**Priority:** P0 - Core deliverable
**Effort:** 8-10 days
**Dependencies:** R2.1 complete

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.2.1 | Create base report template engine with Jinja2 | [x] *(Completed 2026-01-26)* | R2.1 |
| R2.2.2 | Create executive summary template (1-page) | [x] *(Completed 2026-01-26)* | R2.2.1 |
| R2.2.3 | Implement "What Happened" section | [x] *(Completed 2026-01-26)* | R2.2.2 |
| R2.2.4 | Implement "Who Is Affected" section | [x] *(Completed 2026-01-26)* | R2.2.2 |
| R2.2.5 | Implement "What To Do" section | [x] *(Completed 2026-01-26)* | R2.2.2 |
| R2.2.6 | Create metric card components | [x] *(Completed 2026-01-26 - uses design system components)* | R2.2.1 |
| R2.2.7 | Implement scale reference conversions (ha->acres, etc.) | [x] *(Completed 2026-01-26)* | R2.2.2 |
| R2.2.8 | Create emergency resources partial | [x] *(Completed 2026-01-26 - part of full report)* | R2.2.1 |
| R2.2.9 | Create full report template with TOC | [x] *(Completed 2026-01-26)* | R2.2.2 |
| R2.2.10 | Create technical appendix template | [x] *(Completed 2026-01-26)* | R2.2.9 |

**Files to Create:**
- `agents/reporting/templates/base.html`
- `agents/reporting/templates/executive_summary.html`
- `agents/reporting/templates/full_report/` (directory)
- `agents/reporting/templates/components/` (directory)
- `agents/reporting/templates/partials/` (directory)

---

#### Epic R2.3: Map Visualization Overhaul (P0 - Core)

**Priority:** P0 - Core deliverable
**Effort:** 12-15 days
**Dependencies:** R2.1 complete
**Status:** COMPLETE (2026-01-26)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.3.1 | Set up base map integration (CartoDB Positron) | [x] *(Completed 2026-01-26)* | R2.1 |
| R2.3.2 | Implement flood extent rendering (5 severity levels) | [x] *(Completed 2026-01-26)* | R2.3.1 |
| R2.3.3 | Create infrastructure icon set (hospital, school, etc.) | [x] *(Completed 2026-01-26)* | None |
| R2.3.4 | Implement infrastructure status calculation | [x] *(Completed 2026-01-26)* | R2.3.1, R2.3.3 |
| R2.3.5 | Create scale bar component | [x] *(Completed 2026-01-26)* | R2.3.1 |
| R2.3.6 | Create north arrow component | [x] *(Completed 2026-01-26)* | R2.3.1 |
| R2.3.7 | Create legend component | [x] *(Completed 2026-01-26)* | R2.3.2 |
| R2.3.8 | Create inset/locator map | [ ] | R2.3.1 |
| R2.3.9 | Create title block and attribution | [x] *(Completed 2026-01-26)* | R2.3.1 |
| R2.3.10 | Implement pattern overlays for B&W printing | [ ] | R2.3.2 |
| R2.3.11 | Implement 300 DPI export | [x] *(Completed 2026-01-26)* | R2.3.1-R2.3.9 |

**Files Created:**
- ✅ `core/reporting/maps/__init__.py`
- ✅ `core/reporting/maps/base.py` (MapConfig, MapBounds, MapType)
- ✅ `core/reporting/maps/static_map.py` (StaticMapGenerator for print-quality maps)
- ✅ `core/reporting/maps/folium_map.py` (InteractiveMapGenerator for web maps)

**Notes:**
- R2.3.8 (inset/locator map) and R2.3.10 (B&W patterns) deferred to enhancement phase
- Core functionality complete: static and interactive maps with all essential furniture

---

#### Epic R2.4: Data Integrations (P1 - Parallel)

**Priority:** P1 - Can run parallel to R2.2/R2.3
**Effort:** 8-10 days
**Dependencies:** None (can start immediately)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.4.1 | Implement Census API client | [x] Complete (2026-01-26) | None |
| R2.4.2 | Implement OSM Infrastructure data loader | [x] Complete (2026-01-26) | None |
| R2.4.3 | Create emergency resources module | [x] Complete (2026-01-26) | None |
| R2.4.4 | Implement vulnerable population estimates (ACS data) | [ ] | R2.4.1 |
| R2.4.5 | Implement population estimation for AOI | [ ] | R2.4.1 |
| R2.4.6 | Implement hospital location retrieval | [ ] | R2.4.2 |
| R2.4.7 | Implement school location retrieval | [ ] | R2.4.2 |
| R2.4.8 | Implement shelter location retrieval | [ ] | R2.4.2 |
| R2.4.9 | Implement "facilities in flood zone" calculation | [ ] | R2.4.6-R2.4.8 |
| R2.4.10 | Implement API caching (30-day Census, 90-day infra) | [ ] | R2.4.1, R2.4.4 |

**Files Created:**
- ✅ `core/reporting/data/__init__.py`
- ✅ `core/reporting/data/census_client.py` (R2.4.1)
- ✅ `core/reporting/data/infrastructure_client.py` (R2.4.2)
- ✅ `core/reporting/data/emergency_resources.py` (R2.4.3)

**Tests Created:**
- ✅ `tests/test_census_client.py`
- ✅ `tests/test_infrastructure_client.py`
- ✅ `tests/test_emergency_resources.py`
- ✅ `tests/test_reporting_data_integration.py` (2026-01-26)

---

#### Epic R2.5: Interactive Web Reports (P1 - High Impact) ✅ COMPLETE

**Priority:** P1 - High impact feature
**Effort:** 12-15 days
**Dependencies:** R2.2, R2.3 complete; R2.4 soft dependency
**Status:** COMPLETE (2026-01-26)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.5.1 | Integrate Folium for interactive maps | [x] Complete | R2.3 |
| R2.5.2 | Implement zoom/pan controls | [x] Complete | R2.5.1 |
| R2.5.3 | Implement layer toggle (flood extent on/off) | [x] Complete | R2.5.1 |
| R2.5.4 | Create infrastructure hover tooltips | [x] Complete | R2.5.1, R2.4 |
| R2.5.5 | Create before/after slider component | [x] Complete | R2.2 |
| R2.5.6 | Add date labels to before/after images | [x] Complete | R2.5.5 |
| R2.5.7 | Implement responsive layout (576/768/992 breakpoints) | [x] Complete | R2.2 |
| R2.5.8 | Implement 44px touch targets | [x] Complete | R2.5.7 |
| R2.5.9 | Create collapsible sections for mobile | [x] Complete | R2.5.1, R2.5.7 |
| R2.5.10 | Implement print button | [x] Complete | R2.2 |
| R2.5.11 | Implement sticky header | [x] Complete | R2.2 |
| R2.5.12 | Implement emergency resources section | [x] Complete | R2.2 |
| R2.5.13 | Implement keyboard navigation | [x] Complete | R2.5.1-R2.5.5 |

**Files Created:**
- `core/reporting/web/__init__.py`
- `core/reporting/web/interactive_report.py`
- `core/reporting/web/html/interactive_report.html`
- `core/reporting/web/assets/interactive.js`
- `core/reporting/web/assets/interactive.css`

**Tests:** 29 tests in `tests/test_interactive_reports.py` - all passing

---

#### Epic R2.6: Print-Ready PDF Outputs (P1 - Required) ✅ COMPLETE

**Priority:** P1 - Required for official distribution
**Effort:** 6-8 days
**Dependencies:** R2.2, R2.3 complete
**Status:** COMPLETE (2026-01-26)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| R2.6.1 | Set up WeasyPrint PDF generation | [x] *(Completed 2026-01-26)* | R2.2 |
| R2.6.2 | Implement 300 DPI rendering | [x] *(Completed 2026-01-26)* | R2.6.1 |
| R2.6.3 | Add US Letter and A4 page size support | [x] *(Completed 2026-01-26)* | R2.6.1 |
| R2.6.4 | Implement proper margins (0.5" content, 0.125" bleed) | [x] *(Completed 2026-01-26)* | R2.6.1 |
| R2.6.5 | Add page numbers | [x] *(Completed 2026-01-26)* | R2.6.1 |
| R2.6.6 | Create clickable TOC with hyperlinks | [x] *(Completed 2026-01-26 - supported via HTML anchors)* | R2.6.1 |
| R2.6.7 | Implement B&W pattern overlays | [ ] *(Deferred - CMYK hints in CSS, patterns in future)* | R2.3.10 |
| R2.6.8 | Embed fonts or convert to outlines | [x] *(Completed 2026-01-26)* | R2.6.1 |
| R2.6.9 | Embed maps as 300 DPI PNG | [x] *(Completed 2026-01-26 - configurable DPI)* | R2.3.11 |
| R2.6.10 | Add "Continued on next page" indicators | [x] *(Completed 2026-01-26 - CSS support added)* | R2.6.1 |

**Files Created:**
- `core/reporting/pdf/__init__.py`
- `core/reporting/pdf/generator.py` (PDFReportGenerator, PDFConfig, PageSize)
- `core/reporting/pdf/print_styles.css`
- `tests/unit/reporting/test_pdf_generator.py` (13 tests passing)

---

#### REPORT-2.0 Implementation Timeline

```
Week 1-2: Foundation
├── R2.1 Design System (all tasks)
├── R2.2 Base templates (R2.2.1-R2.2.5)
└── R2.4 Census/Infrastructure setup (R2.4.1, R2.4.4) [PARALLEL]

Week 3-4: Core Capabilities
├── R2.2 Complete templates (R2.2.6-R2.2.10)
├── R2.3 Map visualization (all tasks)
└── R2.4 Data integrations (R2.4.2-R2.4.10) [PARALLEL]

Week 5-6: Interactive Features
├── R2.5 Interactive web reports (all tasks)
└── R2.2 Full report template polish

Week 7-8: Polish and Print
├── R2.6 Print outputs (all tasks)
├── Accessibility audit (all features)
├── Integration testing
└── Documentation
```

#### REPORT-2.0 Success Criteria

- [ ] Design system passes WCAG AA contrast requirements
- [ ] Executive summary renders on single page
- [ ] Maps include scale bar, north arrow, legend
- [ ] Population estimates display for flood zones
- [ ] Interactive maps zoom/pan on mobile
- [ ] Before/after slider works with touch
- [ ] PDF generates at 300 DPI
- [ ] All patterns distinguishable in B&W print

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
- Epic 1.1 (Analyze) - COMPLETE
- Epic 1.3 (Validate) - COMPLETE
- Epic 1.6 (Discover) - COMPLETE

### Stream B: API Hardening (Independent)
- Epic 3.1 (Database)
- Epic 3.2 (JWT) - Partial (core validation complete)
- Epic 3.3 (Intent) - COMPLETE

### Stream C: Multi-Band Ingestion (Critical Path)
**This stream unblocks real Sentinel-2 data processing.**

```
Epic 1.7.1 (STAC Client)
        │
        ├──> Epic 1.7.2 (Ingestion) ──┐
        │                              │
        └──> Epic 1.7.3 (Stacking) ───┼──> Epic 1.7.5 (Validator) ──> Epic 1.7.6 (Integration)
                                      │
Epic 1.7.4 (Workaround) ──────────────┘  [Can start immediately, independent]
```

After Epic 1.7 completes:
- Epic 1.2 (Wire CLI Ingest) can proceed with real Sentinel-2 data

### Stream D: Human-Readable Reporting (REPORT-2.0) - NEW
**Can run fully parallel to Streams A-C.**

```
Epic R2.1 (Design System) ◀── START HERE
        │
        ├──> Epic R2.2 (Templates) ──┐
        │                             │
        └──> Epic R2.3 (Map Viz) ────┼──> Epic R2.5 (Web Reports) ──> Epic R2.6 (Print)
                                     │
Epic R2.4 (Data Integrations) ───────┘  [Can start immediately, parallel]
```

**Parallelization Notes:**
- R2.1 must complete before R2.2, R2.3, R2.5, R2.6
- R2.4 has NO dependencies - can start Day 1
- R2.5 has soft dependency on R2.4 (can stub population data)
- Two engineers can work in parallel after R2.1 completes

### Stream E: Visual Product Generation (VIS-1.0) - NEW
**Critical gap identified: Analysis outputs not rendered into viewable images.**

```
Epic VIS-1.1 (Imagery Renderer) ◀── START HERE
        │
        ├──> Epic VIS-1.2 (Before/After) ──┐
        │                                   │
        └──> Epic VIS-1.3 (Overlay) ───────┼──> Epic VIS-1.5 (Integration)
                                           │
Epic VIS-1.4 (Annotations) ────────────────┘  [After VIS-1.3]
```

**Parallelization Notes:**
- VIS-1.1 must complete first (provides base rendering capability)
- VIS-1.2 and VIS-1.3 can run in parallel after VIS-1.1
- VIS-1.4 depends on VIS-1.3 (annotations need overlays)
- VIS-1.5 integrates all components into report pipeline

**Dependencies:**
- Epic 1.7 (Multi-Band Ingestion) - Required for satellite band access
- REPORT-2.0 (R2.2, R2.5, R2.6) - Visual products wire into templates

### Stream F: Sequential Work
- Epic 1.4 (Export) - After 1.1
- Epic 1.5 (Run) - After 1.1, 1.2, 1.3, 1.4, **1.7**
- Epic 2.1 (Agent) - After Phase 1 patterns established
- Epic 4.x (Testing) - After relevant phases complete

---

## Visual Product Generation (VIS-1.0)

**Status:** REVIEWED - Ready for Implementation
**Priority:** P0 (Critical Gap)
**Full Spec:** See `OPENSPEC.md` Section 5
**Requirements Review:** Completed 2026-01-26 (see OPENSPEC.md for findings)
**Estimated Effort:** 5 weeks

### Problem Statement

Analysis algorithms produce accurate outputs (flood masks, statistics) but these are NOT rendered into visual products. Reports show statistics without showing actual satellite imagery or detection results. The before/after slider requires external image URLs that don't exist.

### Epic VIS-1.1: Satellite Imagery Renderer

**Priority:** P0 (Foundation)
**Effort:** 5-7 days

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| VIS-1.1.1 | Create `core/reporting/imagery/` module structure | [x] | None |
| VIS-1.1.2 | Implement `ImageryRenderer` class | [x] *(Completed 2026-01-26)* | VIS-1.1.1 |
| VIS-1.1.3 | Add Sentinel-2 band composites (true color, false color, SWIR) | [x] | VIS-1.1.2 |
| VIS-1.1.4 | Add Landsat band composites | [x] | VIS-1.1.2 |
| VIS-1.1.5 | Implement histogram stretch algorithms | [x] | VIS-1.1.2 |
| VIS-1.1.6 | Add cloud masking using SCL band | [x] *(Completed 2026-01-26)* | VIS-1.1.3 |
| VIS-1.1.7 | Implement PNG/TIFF export with georeferencing | [x] *(Completed 2026-01-26)* | VIS-1.1.2 |
| VIS-1.1.8 | Add unit tests for renderer | [x] *(Completed 2026-01-26)* | VIS-1.1.7 |
| VIS-1.1.9 | **[NEW]** Add SAR (Sentinel-1) visualization support | [x] | VIS-1.1.2 |
| VIS-1.1.10 | **[NEW]** Add partial coverage detection and nodata visualization | [x] | VIS-1.1.2 |
| VIS-1.1.11 | **[NEW]** Add graceful degradation for missing bands | [x] *(Completed 2026-01-26)* | VIS-1.1.3 |

### Epic VIS-1.2: Before/After Image Generation

**Priority:** P0 (High User Value)
**Effort:** 5-7 days

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| VIS-1.2.1 | Implement `BeforeAfterGenerator` class | [x] *(Completed 2026-01-26)* | VIS-1.1.7 |
| VIS-1.2.2 | Add temporal image selection (pre-event, post-event) | [ ] | VIS-1.2.1 |
| VIS-1.2.3 | Implement cloud-free image selection | [ ] | VIS-1.2.2 |
| VIS-1.2.4 | Ensure extent/resolution matching between pairs | [x] *(Completed 2026-01-26 via comparison.py)* | VIS-1.2.1 |
| VIS-1.2.5 | Add date labels to generated images | [x] *(2026-01-26)* | VIS-1.2.1 |
| VIS-1.2.6 | Implement side-by-side composite output | [x] *(2026-01-26)* | VIS-1.2.4 |
| VIS-1.2.7 | Implement animated GIF output (optional) | [x] *(2026-01-26)* | VIS-1.2.4 |
| VIS-1.2.8 | Add integration tests with STAC discovery | [x] *(Completed 2026-01-26)* | VIS-1.2.6 |

### Epic VIS-1.3: Detection Overlay Rendering ✅ COMPLETE

**Priority:** P0 (Core Visual Product)
**Effort:** 8-10 days
**Status:** COMPLETE (2026-01-26)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| VIS-1.3.1 | Implement `OverlayRenderer` class | [x] *(Completed 2026-01-26)* | VIS-1.1.7 |
| VIS-1.3.2 | Add flood extent rendering with severity colors | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.3 | Add burn severity rendering with dNBR colors | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.4 | Implement confidence-based transparency | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.5 | Add vector polygon outline rendering | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.6 | Implement auto-generated legend component | [x] *(Completed 2026-01-26)* | VIS-1.3.2 |
| VIS-1.3.7 | Add scale bar component | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.8 | Add north arrow component | [x] *(Completed 2026-01-26)* | VIS-1.3.1 |
| VIS-1.3.9 | Add pattern fills for B&W accessibility | [x] *(Completed 2026-01-26)* | VIS-1.3.2 |
| VIS-1.3.10 | Add unit tests for overlay renderer | [x] *(Completed 2026-01-26)* | VIS-1.3.9 |

**Implementation Notes:**
- DetectionOverlay class provides full overlay rendering functionality
- Supports flood and fire/burn severity overlays with appropriate color scales
- Confidence-based transparency modulates alpha channel based on detection confidence
- Polygon outlines rendered via edge detection and dilation
- Auto-generated legends via LegendRenderer integration
- Scale bar calculates appropriate scale based on pixel resolution
- North arrow with "N" label indicator
- B&W pattern fills use hatching (dots, diagonal lines, crosshatch) for grayscale printing
- 603 tests passing in reporting suite
- Smoke tests verify all functionality

### Epic VIS-1.4: Contextual Annotation Layer

**Priority:** P1 (Enhances Understanding)
**Effort:** 5-7 days

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| VIS-1.4.1 | Implement area comparison engine | [ ] | None |
| VIS-1.4.2 | Add human-readable conversions (ha -> football fields, etc.) | [ ] | VIS-1.4.1 |
| VIS-1.4.3 | Implement OSM landmark labeling | [ ] | VIS-1.3.1 |
| VIS-1.4.4 | Add population impact callouts | [ ] | R2.4 (Census) |
| VIS-1.4.5 | Implement auto-generated explanatory captions | [ ] | VIS-1.4.1 |
| VIS-1.4.6 | Add inset locator map generation | [ ] | VIS-1.3.1 |
| VIS-1.4.7 | Add attribution block component | [ ] | VIS-1.3.1 |
| VIS-1.4.8 | Add unit tests for annotations | [ ] | VIS-1.4.7 |
| VIS-1.4.9 | **[NEW]** Add "science behind the statistics" explanation templates | [ ] | VIS-1.4.5 |
| VIS-1.4.10 | **[NEW]** Add detection method explanation text per algorithm | [ ] | VIS-1.4.9 |

### Epic VIS-1.5: Report Integration ✅ COMPLETE

**Priority:** P0 (Wires Everything Together)
**Effort:** 5-7 days
**Status:** COMPLETE (2026-01-26)

| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| VIS-1.5.1 | Create `ReportVisualPipeline` orchestrator | [x] *(Completed 2026-01-26)* | VIS-1.2.6, VIS-1.3.9 |
| VIS-1.5.2 | Implement image manifest tracking | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.3 | Update before/after slider to use local images | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.4 | Update executive summary map to use detection overlay | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.5 | Implement image caching to avoid re-generation | [x] *(Completed 2026-01-26)* | VIS-1.5.2 |
| VIS-1.5.6 | Add 300 DPI embedding for PDF reports | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.7 | Add web-optimized image output (compressed PNG) | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.8 | Add error handling for unavailable imagery | [x] *(Completed 2026-01-26)* | VIS-1.5.1 |
| VIS-1.5.9 | Add end-to-end integration test | [x] *(Completed 2026-01-26)* | VIS-1.5.8 |

### VIS-1.0 Success Criteria

- [x] Satellite imagery renders to viewable RGB images *(VIS-1.1 Complete)*
- [x] Before/after pairs auto-generated from STAC data (no external URLs) *(VIS-1.2 Complete)*
- [x] Flood extent overlaid on satellite imagery with severity colors *(VIS-1.3 Complete)*
- [x] All visual products have scale bar, legend, attribution *(VIS-1.3 Complete)*
- [ ] Area comparisons in human-readable terms ("3,500 football fields") *(VIS-1.4)*
- [x] Reports use generated images (not placeholders) *(VIS-1.5 Complete - 2026-01-26)*
- [x] Image generation < 60 seconds per report *(VIS-1.5 Complete - verified via smoke test)*
- [x] **[NEW]** SAR (Sentinel-1) imagery renders with grayscale/pseudocolor *(VIS-1.1 Complete)*
- [x] **[NEW]** Graceful degradation when imagery unavailable (placeholder + message) *(VIS-1.1 Complete)*
- [x] **[NEW]** Partial coverage visualized with nodata pattern *(VIS-1.1 Complete)*
- [ ] **[NEW]** Explanatory text explains what remote sensing detected *(VIS-1.4)*

---

## Success Criteria

### Phase 1 Complete When:
- [x] `flight analyze` produces real classification results *(Completed)*
- [ ] `flight ingest` downloads real satellite data *(Blocked by Epic 1.7)*
- [x] `flight validate` produces real QC scores *(Completed 2026-01-23)*
- [ ] `flight export` creates valid GeoTIFF/GeoJSON/PDF
- [ ] `flight run` executes full pipeline end-to-end
- [x] `flight discover` never falls back to mocks *(Completed 2026-01-23)*

### Epic 1.7 (Multi-Band Ingestion) Complete When:
- [ ] STAC client returns individual band URLs for Sentinel-2
- [ ] Ingestion pipeline downloads all required spectral bands
- [ ] Band stacking creates valid VRT with correct band ordering
- [ ] Validation passes for stacked multi-band files
- [ ] Integration tests pass with real Sentinel-2 data from Earth Search

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

### REPORT-2.0 (Phase 2B) Complete When:
- [ ] Design system CSS tokens implemented and WCAG AA verified
- [ ] Executive summary template renders complete 1-page reports
- [ ] Full report template with TOC and multi-page support working
- [ ] Map visualization includes all furniture (scale, north arrow, legend)
- [ ] Census API returns population estimates for AOI
- [ ] Infrastructure overlay shows hospitals, schools, shelters
- [ ] Interactive web reports work on mobile (touch targets 44px+)
- [ ] Before/after slider functions on touch and keyboard
- [ ] PDF generation at 300 DPI with proper margins
- [ ] Pattern overlays distinguishable in B&W print
- [ ] Accessibility audit passes (WCAG 2.1 AA)

### VIS-1.0 (Visual Products) Complete When:
- [ ] Satellite imagery renders to viewable RGB images (true color, false color)
- [ ] Before/after pairs auto-generated from STAC data (no external URLs required)
- [ ] Detection results overlaid on satellite imagery with severity colors
- [ ] All visual products include scale bar, legend, and attribution
- [ ] Area comparisons use human-readable terms ("X football fields")
- [ ] Reports display actual detection evidence (not just statistics)
- [ ] Image generation completes in < 60 seconds per report
- [ ] Visual products properly embedded in PDF (300 DPI) and web reports

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
1. **Sentinel-2 Ingestion Mismatch (Epic 1.7)** - STAC client returns TCI.tif instead of individual bands
   - Impact: Blocks all real Sentinel-2 processing
   - Status: Root cause identified, solution designed
   - Mitigation: Epic 1.7 addresses this with multi-band download refactor
   - Workaround: `--skip-validation` flag (Task 1.7.4) for immediate use

2. **Database migration complexity** - If existing in-memory data must be preserved
   - Mitigation: Clean slate for production deployment

3. **STAC catalog availability** - Real tests depend on external services
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
| `cli/commands/analyze.py` | ~~MockAlgorithm~~ | ~~Wire to AlgorithmRegistry~~ COMPLETE |
| `cli/commands/ingest.py` | Text placeholder | Wire to StreamingIngester (after Epic 1.7) |
| `cli/commands/validate.py` | ~~random.uniform()~~ | ~~Wire to SanitySuite~~ COMPLETE |
| `cli/commands/export.py` | Mock files | Wire to ProductGenerator |
| `cli/commands/run.py` | Multiple stubs | Wire all sub-commands |
| `cli/commands/discover.py` | ~~Mock fallback~~ | ~~Remove fallback~~ COMPLETE |
| `agents/pipeline/main.py` | Stub execution | Wire to algorithms |
| `api/dependencies.py` | NotImplementedError | Add database layer |
| `api/dependencies.py` | ~~TODO JWT~~ | ~~Implement validation~~ COMPLETE |
| `api/routes/events.py` | ~~TODO intent~~ | ~~Wire to IntentResolver~~ COMPLETE |

### Epic 1.7 File Changes (Multi-Band Ingestion)

| File | Current State | Changes Required |
|------|---------------|------------------|
| `core/data/discovery/stac_client.py` | Returns single `visual` URL | Return `band_urls` dict with individual band URLs |
| `cli/commands/ingest.py` | Downloads single file | Handle multi-file downloads per item |
| `core/data/ingestion/streaming.py` | Single-file ingestion | Support multi-file ingestion + stacking |
| `core/data/ingestion/band_stack.py` | **Does not exist** | **NEW:** VRT/band stacking utility |
| `core/data/ingestion/validation/image_validator.py` | Expects 4+ bands in single file | Validate stacked VRT files |
| `tests/integration/test_sentinel2_ingestion.py` | **Does not exist** | **NEW:** End-to-end Sentinel-2 tests |

---

## Core Libraries to Wire

| Core Module | CLI Target | API Target | Agent Target | Status |
|-------------|------------|------------|--------------|--------|
| `core.analysis.library.AlgorithmRegistry` | analyze.py | - | pipeline/main.py | COMPLETE |
| `core.data.ingestion.StreamingIngester` | ingest.py | - | - | Blocked (Epic 1.7) |
| `core.data.ingestion.band_stack` | ingest.py | - | - | **NEW** (Epic 1.7) |
| `core.quality.sanity.SanitySuite` | validate.py | - | quality/main.py | COMPLETE |
| `core.quality.reporting.ProductGenerator` | export.py | products.py | reporting/main.py | Pending |
| `core.intent.IntentResolver` | - | events.py | - | COMPLETE |
| `core.data.discovery.STACClient` | discover.py | - | discovery/main.py | COMPLETE (needs 1.7 update) |

---

**Full Specification:** `OPENSPEC_PRODUCTION.md`
**Next Review:** After Phase 1 complete
