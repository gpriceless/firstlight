# REPORT-2.0 Work Assignments

**Epic ID:** REPORT-2.0
**Status:** Ready for Implementation
**Created:** 2026-01-26
**Author:** Ratchet (Engineering Manager)
**Branch:** feature/report-2.0-human-readable-reporting

---

## Executive Summary

This document breaks down the Human-Readable Reporting Overhaul into three parallel tracks, each managed by a dedicated Project Manager. The total implementation timeline is **8 weeks**.

### Track Overview

| Track | PM | Features | Start Condition | Duration |
|-------|----|---------:|-----------------|----------|
| **A: Foundation** | PM-1 | R2.1, R2.2 | Immediate | 3 weeks |
| **B: Geospatial** | PM-2 | R2.3, R2.4 | After R2.1 (1 week) | 3 weeks |
| **C: Output Formats** | PM-3 | R2.5, R2.6 | After R2.2+R2.3 | 3 weeks |

### Dependency Graph

```
Week 1       Week 2       Week 3       Week 4       Week 5       Week 6       Week 7       Week 8
  |            |            |            |            |            |            |            |
  v            v            v            v            v            v            v            v
+--------+  +--------+  +--------+
| R2.1   |  | R2.1   |  |        |  Track A: Foundation (PM-1)
| Design |->| cont.  |  |        |
+--------+  +--------+  +--------+
     |           |           |
     v           v           v
            +--------+  +--------+  +--------+
            | R2.2   |  | R2.2   |  | R2.2   |  Track A: Templates (PM-1)
            | Start  |->| cont.  |->| finish |
            +--------+  +--------+  +--------+
                             |           |
                             v           v
+--------+  +--------+  +--------+  +--------+
| R2.4   |  | R2.4   |  | R2.4   |  |        |  Track B: Data (PM-2) [PARALLEL]
| Census |->| Infra  |->| finish |  |        |
+--------+  +--------+  +--------+  +--------+
     |           |           |
+--------+  +--------+  +--------+  +--------+
| R2.3   |  | R2.3   |  | R2.3   |  |        |  Track B: Maps (PM-2)
| (wait) |  | Start  |->| cont.  |->| finish |
+--------+  +--------+  +--------+  +--------+
                                         |
                                         v
                        +--------+  +--------+  +--------+  +--------+
                        |        |  | R2.5   |  | R2.5   |  | R2.5   |  Track C: Web (PM-3)
                        |        |  | Start  |->| cont.  |->| finish |
                        +--------+  +--------+  +--------+  +--------+
                                                     |           |
                                                     v           v
                                         +--------+  +--------+  +--------+
                                         |        |  | R2.6   |  | R2.6   |  Track C: PDF (PM-3)
                                         |        |  | Start  |->| finish |
                                         +--------+  +--------+  +--------+
                                                                      |
                                                                      v
                                                          +--------+  +--------+
                                                          | QA &   |  | Release|
                                                          | Access |  |        |
                                                          +--------+  +--------+
```

---

## Track A: Foundation (PM-1)

**Project Manager:** PM-1
**Features:** R2.1 (Design System), R2.2 (Plain-Language Templates)
**Duration:** Weeks 1-4
**Priority:** P0 (Critical Path - blocks all other tracks)

### Start Condition
- Immediate start
- No dependencies

### Track A Deliverables

| Feature | Deliverable Files | Acceptance Criteria |
|---------|-------------------|---------------------|
| R2.1 | `agents/reporting/static/css/tokens.css` | All design tokens from DESIGN_SYSTEM.md |
| R2.1 | `agents/reporting/static/css/components.css` | .metric-card, .alert-box, .severity-badge, .data-table |
| R2.1 | `agents/reporting/static/css/utilities.css` | Margin, padding, text alignment utilities |
| R2.1 | `agents/reporting/static/css/print.css` | Print-specific styles, page breaks |
| R2.1 | `agents/reporting/static/css/main.css` | Master stylesheet importing all |
| R2.1 | `agents/reporting/static/icons/heroicons/` | SVG icons for UI |
| R2.1 | `agents/reporting/static/icons/firstlight/` | Brand assets |
| R2.2 | `agents/reporting/templates/base.html` | Base template with header/footer |
| R2.2 | `agents/reporting/templates/executive_summary.html` | 1-page summary template |
| R2.2 | `agents/reporting/templates/full_report/` | Multi-page report structure |
| R2.2 | `agents/reporting/templates/components/` | Reusable component templates |
| R2.2 | `agents/reporting/templates/partials/` | Header, footer, emergency resources |
| R2.2 | `tests/test_design_system.py` | CSS token and component tests |
| R2.2 | `tests/test_templates.py` | Template rendering tests |

### QA Checkpoints

| Checkpoint | When | Criteria |
|------------|------|----------|
| **QA-A1** | End of Week 2 | Design system CSS complete, WCAG AA verified with axe-core |
| **QA-A2** | End of Week 3 | Executive summary template renders correctly |
| **QA-A3** | End of Week 4 | Full report template complete, plain language conversions working |

---

## Feature R2.1: Design System Implementation

**Assigned to:** Track A (PM-1)
**Effort:** 5-7 days
**Dependencies:** None
**Blocks:** R2.2, R2.3, R2.5, R2.6

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.1.1 | Create CSS custom properties file with brand colors | 2h | None | Coder-A1 |
| R2.1.2 | Implement flood severity palette (5 levels + colorblind safe) | 2h | R2.1.1 | Coder-A1 |
| R2.1.3 | Implement wildfire severity palette (6 levels) | 2h | R2.1.1 | Coder-A1 |
| R2.1.4 | Implement confidence/uncertainty palette (4 levels) | 1h | R2.1.1 | Coder-A1 |
| R2.1.5 | Implement typography scale (text-xs through text-5xl) | 2h | R2.1.1 | Coder-A2 |
| R2.1.6 | Implement spacing scale (space-1 through space-24) | 1h | R2.1.1 | Coder-A2 |
| R2.1.7 | Create metric card component CSS | 3h | R2.1.1, R2.1.5 | Coder-A1 |
| R2.1.8 | Create alert/warning box component CSS | 3h | R2.1.1, R2.1.5 | Coder-A2 |
| R2.1.9 | Create severity badge component CSS | 2h | R2.1.2, R2.1.3 | Coder-A1 |
| R2.1.10 | Create data table component CSS | 3h | R2.1.1, R2.1.5 | Coder-A2 |
| R2.1.11 | Create legend component CSS | 2h | R2.1.2, R2.1.3 | Coder-A1 |
| R2.1.12 | Create utility classes (margins, padding, alignment) | 2h | R2.1.6 | Coder-A2 |
| R2.1.13 | Create print stylesheet with page break handling | 4h | R2.1.7-R2.1.12 | Coder-A1 |
| R2.1.14 | Set up Heroicons SVG library | 2h | None | Coder-A2 |
| R2.1.15 | Create FirstLight brand assets folder | 1h | None | Coder-A2 |
| R2.1.16 | Run axe-core WCAG AA contrast verification | 2h | R2.1.2-R2.1.4 | Coder-A1 |
| R2.1.17 | Create main.css master stylesheet | 1h | R2.1.1-R2.1.15 | Coder-A1 |
| R2.1.18 | Write unit tests for token completeness | 3h | R2.1.17 | Coder-A2 |

### R2.1 Parallel Work Strategy

```
Coder-A1                          Coder-A2
--------                          --------
R2.1.1 (tokens) ─────────────────> R2.1.5 (typography)
    │                                  │
    v                                  v
R2.1.2 (flood palette)            R2.1.6 (spacing)
R2.1.3 (wildfire palette)             │
R2.1.4 (confidence palette)           v
    │                             R2.1.8 (alert box)
    v                             R2.1.10 (data table)
R2.1.7 (metric card)              R2.1.12 (utilities)
R2.1.9 (severity badge)           R2.1.14 (heroicons)
R2.1.11 (legend)                  R2.1.15 (brand assets)
    │                                  │
    v                                  v
R2.1.16 (WCAG verification)       R2.1.18 (tests)
R2.1.13 (print stylesheet)
R2.1.17 (main.css)
```

### R2.1 Acceptance Criteria

- [ ] CSS custom properties file contains all tokens from DESIGN_SYSTEM.md
- [ ] Flood severity palette: 5 levels (minor through extreme)
- [ ] Wildfire severity palette: 6 levels
- [ ] Confidence palette: 4 levels
- [ ] Typography scale: text-xs through text-5xl
- [ ] Spacing scale: space-1 through space-24
- [ ] All colors pass WCAG 2.1 AA contrast (4.5:1 normal, 3:1 large)
- [ ] Component classes: .metric-card, .alert-box, .severity-badge, .data-table
- [ ] Print stylesheet handles page breaks
- [ ] [STRETCH] Dark mode support

---

## Feature R2.2: Plain-Language Report Templates

**Assigned to:** Track A (PM-1)
**Effort:** 8-10 days
**Dependencies:** R2.1
**Blocks:** R2.5, R2.6

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.2.1 | Create base.html template structure | 3h | R2.1 | Coder-A1 |
| R2.2.2 | Create header partial with FirstLight branding | 2h | R2.2.1 | Coder-A1 |
| R2.2.3 | Create footer partial with attribution/timestamp | 2h | R2.2.1 | Coder-A1 |
| R2.2.4 | Create emergency_resources.html partial | 3h | R2.2.1 | Coder-A2 |
| R2.2.5 | Create metric_card.html component | 2h | R2.2.1 | Coder-A1 |
| R2.2.6 | Create alert_box.html component | 2h | R2.2.1 | Coder-A2 |
| R2.2.7 | Create severity_badge.html component | 1h | R2.2.1 | Coder-A1 |
| R2.2.8 | Create facility_card.html component | 2h | R2.2.1 | Coder-A2 |
| R2.2.9 | Create executive_summary.html template structure | 4h | R2.2.1-R2.2.8 | Coder-A1 |
| R2.2.10 | Implement "What Happened" section | 3h | R2.2.9 | Coder-A1 |
| R2.2.11 | Implement "Who Is Affected" section | 3h | R2.2.9 | Coder-A2 |
| R2.2.12 | Implement "What To Do" section | 3h | R2.2.9 | Coder-A1 |
| R2.2.13 | Implement scale reference conversions utility | 4h | None | Coder-A2 |
| R2.2.14 | Create full_report/cover.html | 2h | R2.2.1 | Coder-A1 |
| R2.2.15 | Create full_report/toc.html | 3h | R2.2.1 | Coder-A2 |
| R2.2.16 | Create full_report/what_happened.html | 3h | R2.2.10 | Coder-A1 |
| R2.2.17 | Create full_report/who_affected.html | 3h | R2.2.11 | Coder-A2 |
| R2.2.18 | Create full_report/maps.html placeholder | 2h | R2.2.1 | Coder-A1 |
| R2.2.19 | Create full_report/recommendations.html | 3h | R2.2.12 | Coder-A2 |
| R2.2.20 | Create full_report/appendix_technical.html | 3h | R2.2.1 | Coder-A1 |
| R2.2.21 | Write template rendering tests | 4h | R2.2.9, R2.2.14-R2.2.20 | Coder-A2 |
| R2.2.22 | Write plain language conversion tests | 2h | R2.2.13 | Coder-A2 |

### R2.2 Plain Language Conversion Rules

The scale reference utility (R2.2.13) must implement these conversions:

| Technical Term | Plain Language |
|----------------|----------------|
| X hectares | "About Y acres (roughly Z football fields)" |
| X% coverage | "Nearly 1 in Y acres in the study area" |
| IoU 85% | "85% agreement between detection methods" |
| >85% confidence | "HIGH CONFIDENCE: Flooding confirmed" |
| 60-85% confidence | "MODERATE: Likely flooded" |
| <60% confidence | "UNCERTAIN: Unable to confirm" |

### R2.2 Acceptance Criteria

- [ ] Executive summary template renders 1-page layout per REPORT_VISUAL_SPEC.md
- [ ] Template includes "What Happened" with plain-language event description
- [ ] Template includes "Who Is Affected" with population and facility counts
- [ ] Template includes "What To Do" with actionable guidance
- [ ] Key metrics displayed as visual cards (not tables)
- [ ] Severity badges use text labels, not just colors
- [ ] Scale references include: ha->acres, large areas->football fields, percentages->fractions
- [ ] Emergency resource section includes FEMA, local EM, disaster hotlines
- [ ] Full report template has TOC and multi-page support
- [ ] Technical appendix template available

---

## Track B: Geospatial (PM-2)

**Project Manager:** PM-2
**Features:** R2.3 (Map Visualization), R2.4 (Data Integrations)
**Duration:** Weeks 1-4
**Priority:** P0 (R2.3), P1 (R2.4)

### Start Condition
- R2.4: Immediate start (no dependencies)
- R2.3: Waits for R2.1 (design system colors) - ~1 week

### Track B Deliverables

| Feature | Deliverable Files | Acceptance Criteria |
|---------|-------------------|---------------------|
| R2.3 | `agents/reporting/cartography/__init__.py` | Module initialization |
| R2.3 | `agents/reporting/cartography/config.py` | CRS settings, default styles |
| R2.3 | `agents/reporting/cartography/base_maps.py` | Tile source management |
| R2.3 | `agents/reporting/cartography/flood_renderer.py` | Flood extent styling |
| R2.3 | `agents/reporting/cartography/infrastructure.py` | Icon rendering, status indication |
| R2.3 | `agents/reporting/cartography/map_furniture.py` | Legend, scale bar, north arrow |
| R2.3 | `agents/reporting/cartography/export.py` | PNG/PDF map export |
| R2.3 | `agents/reporting/static/icons/infrastructure/` | Hospital, school, shelter SVGs |
| R2.4 | `core/data/context/__init__.py` | Module initialization |
| R2.4 | `core/data/context/census.py` | Census API client |
| R2.4 | `core/data/context/infrastructure.py` | HIFLD/OSM data loading |
| R2.4 | `core/data/context/emergency_resources.py` | Resource link configuration |
| R2.4 | `core/data/context/population.py` | Population impact estimation |
| R2.4 | `tests/test_cartography.py` | Map rendering tests |
| R2.4 | `tests/test_data_integrations.py` | API integration tests |

### QA Checkpoints

| Checkpoint | When | Criteria |
|------------|------|----------|
| **QA-B1** | End of Week 2 | Census API returns population for test AOI |
| **QA-B2** | End of Week 3 | Base maps rendering with flood extent overlay |
| **QA-B3** | End of Week 4 | Infrastructure overlays complete, 300 DPI export working |

---

## Feature R2.3: Map Visualization Overhaul

**Assigned to:** Track B (PM-2)
**Effort:** 12-15 days
**Dependencies:** R2.1 (color palettes)
**Blocks:** R2.5, R2.6

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.3.1 | Create cartography module structure | 2h | None | Coder-B1 |
| R2.3.2 | Implement config.py with CRS settings | 3h | R2.3.1 | Coder-B1 |
| R2.3.3 | Implement base_maps.py with CartoDB Positron | 4h | R2.3.1 | Coder-B1 |
| R2.3.4 | Create tile source fallback system (OSM backup) | 3h | R2.3.3 | Coder-B1 |
| R2.3.5 | Implement flood_renderer.py basic structure | 3h | R2.1, R2.3.1 | Coder-B2 |
| R2.3.6 | Add 5-level severity coloring per design system | 4h | R2.3.5 | Coder-B2 |
| R2.3.7 | Add edge treatment for confidence levels | 3h | R2.3.6 | Coder-B2 |
| R2.3.8 | Create infrastructure icon SVG set | 4h | None | Coder-B1 |
| R2.3.9 | Implement infrastructure.py icon rendering | 4h | R2.3.8 | Coder-B1 |
| R2.3.10 | Implement infrastructure status calculation | 4h | R2.3.9 | Coder-B2 |
| R2.3.11 | Create scale bar component in map_furniture.py | 3h | R2.3.1 | Coder-B1 |
| R2.3.12 | Create north arrow component | 2h | R2.3.11 | Coder-B1 |
| R2.3.13 | Create legend component (categorical) | 4h | R2.3.6, R2.3.9 | Coder-B2 |
| R2.3.14 | Create legend component (continuous) | 3h | R2.3.13 | Coder-B2 |
| R2.3.15 | Create title block and attribution | 2h | R2.3.1 | Coder-B1 |
| R2.3.16 | Create inset/locator map | 4h | R2.3.3 | Coder-B2 |
| R2.3.17 | Implement pattern overlays for B&W | 4h | R2.3.6 | Coder-B1 |
| R2.3.18 | Implement export.py with 300 DPI PNG | 4h | R2.3.3-R2.3.16 | Coder-B2 |
| R2.3.19 | Add PDF export capability | 3h | R2.3.18 | Coder-B2 |
| R2.3.20 | Write projection accuracy tests | 3h | R2.3.2, R2.3.11 | Coder-B1 |
| R2.3.21 | Write visual layer stacking tests | 3h | R2.3.5, R2.3.9 | Coder-B2 |
| R2.3.22 | Write colorblind simulation tests | 2h | R2.3.6 | Coder-B1 |

### R2.3 Infrastructure Status Calculation

Task R2.3.10 must implement these status classifications:

| Status | Definition | Visual Treatment |
|--------|------------|------------------|
| "In flood zone" | Facility geometry intersects flood polygon | Full color, pulsing ring |
| "Adjacent" | Within 500m buffer of flood zone | Full color, orange outline |
| "Outside flood zone" | Beyond 500m buffer | Grayscale, no outline |
| "Data unavailable" | Location unknown | Question mark overlay |

### R2.3 Acceptance Criteria

- [ ] Base map integration with CartoDB Positron (OSM fallback)
- [ ] Flood extent renders with 5-level severity coloring
- [ ] Infrastructure icons: hospital, school, shelter, fire station, police
- [ ] Infrastructure status calculated correctly (in zone/adjacent/outside/unknown)
- [ ] Scale bar with metric units, alternating black/white
- [ ] North arrow on all print maps
- [ ] Legend with severity levels AND infrastructure icons
- [ ] Inset/locator map showing regional context
- [ ] Title block with event name, location, date
- [ ] Attribution with data sources, CRS, timestamp
- [ ] Pattern overlays for B&W printing
- [ ] Maps render at 300 DPI for print

---

## Feature R2.4: Data Integrations

**Assigned to:** Track B (PM-2)
**Effort:** 8-10 days
**Dependencies:** None (can start immediately)
**Soft dependency for:** R2.5 (population data in web reports)

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.4.1 | Create context module structure | 1h | None | Coder-B1 |
| R2.4.2 | Implement census.py API client skeleton | 3h | R2.4.1 | Coder-B1 |
| R2.4.3 | Implement Census population retrieval for AOI | 4h | R2.4.2 | Coder-B1 |
| R2.4.4 | Implement housing unit counts from Census | 2h | R2.4.3 | Coder-B1 |
| R2.4.5 | Implement vulnerable population estimates (ACS) | 4h | R2.4.3 | Coder-B2 |
| R2.4.6 | Create infrastructure.py HIFLD loader | 4h | R2.4.1 | Coder-B2 |
| R2.4.7 | Implement OSM fallback for infrastructure | 3h | R2.4.6 | Coder-B2 |
| R2.4.8 | Implement hospital location retrieval | 2h | R2.4.6 | Coder-B1 |
| R2.4.9 | Implement school location retrieval | 2h | R2.4.6 | Coder-B2 |
| R2.4.10 | Implement shelter location retrieval | 2h | R2.4.6 | Coder-B1 |
| R2.4.11 | Implement fire station location retrieval | 2h | R2.4.6 | Coder-B2 |
| R2.4.12 | Implement population.py impact estimation | 4h | R2.4.3, R2.4.8-R2.4.11 | Coder-B1 |
| R2.4.13 | Implement "facilities in flood zone" calculation | 4h | R2.4.12 | Coder-B2 |
| R2.4.14 | Create emergency_resources.py config system | 3h | R2.4.1 | Coder-B1 |
| R2.4.15 | Add state/region emergency resource links | 2h | R2.4.14 | Coder-B2 |
| R2.4.16 | Implement API caching (30-day Census) | 3h | R2.4.3 | Coder-B1 |
| R2.4.17 | Implement API caching (90-day infrastructure) | 2h | R2.4.6 | Coder-B2 |
| R2.4.18 | Implement rate limiting for Census API | 2h | R2.4.3 | Coder-B1 |
| R2.4.19 | Write Census API integration tests | 3h | R2.4.3-R2.4.5 | Coder-B2 |
| R2.4.20 | Write infrastructure query tests | 3h | R2.4.8-R2.4.11 | Coder-B1 |
| R2.4.21 | Write cache behavior tests | 2h | R2.4.16, R2.4.17 | Coder-B2 |

### R2.4 Vulnerable Population Specification

Task R2.4.5 must retrieve these ACS 5-year data points:

| Population | ACS Table | Notes |
|------------|-----------|-------|
| Elderly (65+) | B01001 | Sum of age 65+ categories |
| Children (<18) | B01001 | Sum of under 18 categories |
| Below Poverty | B17001 | Poverty status |

Display format: "X residents (margin of error: +/- Y)" when margin available.

### R2.4 Acceptance Criteria

- [ ] Census API returns population for AOI
- [ ] Housing unit counts derived from Census
- [ ] Vulnerable populations: Elderly 65+, Children <18, Below Poverty
- [ ] Infrastructure source: HIFLD primary, OSM fallback
- [ ] Hospital locations and names retrieved
- [ ] School locations retrieved
- [ ] Shelter locations retrieved (FEMA NSS if available)
- [ ] Fire station locations retrieved
- [ ] "Facilities in flood zone" spatial intersection working
- [ ] Emergency resource links configurable per state/region
- [ ] API rate limiting implemented (Census 500/day)
- [ ] Caching: 30-day Census, 90-day infrastructure

---

## Track C: Output Formats (PM-3)

**Project Manager:** PM-3
**Features:** R2.5 (Interactive Web Reports), R2.6 (Print-Ready PDF)
**Duration:** Weeks 4-8
**Priority:** P1

### Start Condition
- R2.5: Waits for R2.2 (templates) + R2.3 (map visualization) - ~3 weeks
- R2.6: Waits for R2.5 (shared learnings) - ~2 weeks after R2.5 starts

### Track C Deliverables

| Feature | Deliverable Files | Acceptance Criteria |
|---------|-------------------|---------------------|
| R2.5 | `agents/reporting/templates/web_report/index.html` | Main web report template |
| R2.5 | `agents/reporting/templates/web_report/map_embed.html` | Folium map iframe |
| R2.5 | `agents/reporting/templates/web_report/slider.html` | Before/after slider |
| R2.5 | `agents/reporting/static/js/map-controls.js` | Map interaction handlers |
| R2.5 | `agents/reporting/static/js/slider.js` | Slider logic |
| R2.5 | `agents/reporting/static/js/share.js` | Share/download functionality |
| R2.5 | `agents/reporting/static/css/web-report.css` | Web-specific styles |
| R2.5 | `agents/reporting/generators/web_report.py` | Web report generation |
| R2.6 | `agents/reporting/generators/pdf_report.py` | PDF generation |
| R2.6 | `agents/reporting/generators/print_config.py` | Print configuration |
| R2.6 | `tests/test_web_reports.py` | Web report tests |
| R2.6 | `tests/test_pdf_generation.py` | PDF tests |

### QA Checkpoints

| Checkpoint | When | Criteria |
|------------|------|----------|
| **QA-C1** | End of Week 5 | Interactive map zoom/pan working |
| **QA-C2** | End of Week 6 | Before/after slider functional on touch |
| **QA-C3** | End of Week 7 | PDF generates at 300 DPI with correct margins |
| **QA-C4** | End of Week 8 | Full accessibility audit passes |

---

## Feature R2.5: Interactive Web Reports

**Assigned to:** Track C (PM-3)
**Effort:** 12-15 days
**Dependencies:** R2.2, R2.3, R2.4 (soft)
**Blocks:** None

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.5.1 | Create web_report module structure | 2h | R2.2, R2.3 | Coder-C1 |
| R2.5.2 | Implement Folium map integration | 4h | R2.3, R2.5.1 | Coder-C1 |
| R2.5.3 | Add zoom/pan controls | 2h | R2.5.2 | Coder-C1 |
| R2.5.4 | Implement flood extent layer toggle | 3h | R2.5.2 | Coder-C1 |
| R2.5.5 | Add infrastructure markers with hover tooltips | 4h | R2.5.2, R2.4 | Coder-C2 |
| R2.5.6 | Create map_embed.html iframe template | 2h | R2.5.2 | Coder-C1 |
| R2.5.7 | Create slider.html component structure | 3h | R2.5.1 | Coder-C2 |
| R2.5.8 | Implement slider.js drag logic | 4h | R2.5.7 | Coder-C2 |
| R2.5.9 | Implement touch support for slider | 3h | R2.5.8 | Coder-C2 |
| R2.5.10 | Add date labels to before/after images | 2h | R2.5.7 | Coder-C1 |
| R2.5.11 | Create web-report.css responsive base | 4h | R2.2 | Coder-C1 |
| R2.5.12 | Implement 576px breakpoint (mobile) | 3h | R2.5.11 | Coder-C2 |
| R2.5.13 | Implement 768px breakpoint (tablet portrait) | 2h | R2.5.11 | Coder-C1 |
| R2.5.14 | Implement 992px breakpoint (tablet landscape) | 2h | R2.5.11 | Coder-C2 |
| R2.5.15 | Ensure 44px minimum touch targets | 2h | R2.5.11-R2.5.14 | Coder-C1 |
| R2.5.16 | Create collapsible legend for mobile | 3h | R2.5.12 | Coder-C2 |
| R2.5.17 | Implement share.js with copyable URL | 3h | R2.5.1 | Coder-C1 |
| R2.5.18 | Implement PDF download button | 3h | R2.5.17 | Coder-C2 |
| R2.5.19 | Implement PNG export button | 2h | R2.5.17 | Coder-C1 |
| R2.5.20 | Implement sticky emergency footer | 3h | R2.2, R2.5.11 | Coder-C2 |
| R2.5.21 | Hide footer during print (full footer on page) | 2h | R2.5.20 | Coder-C2 |
| R2.5.22 | Implement keyboard navigation (ARIA) | 4h | R2.5.2, R2.5.8 | Coder-C1 |
| R2.5.23 | Create web_report.py generator | 4h | R2.5.1-R2.5.22 | Coder-C2 |
| R2.5.24 | Write map interaction tests | 3h | R2.5.2-R2.5.5 | Coder-C1 |
| R2.5.25 | Write slider functionality tests | 3h | R2.5.8-R2.5.9 | Coder-C2 |
| R2.5.26 | Write responsive layout tests | 2h | R2.5.11-R2.5.15 | Coder-C1 |

### R2.5 Acceptance Criteria

- [ ] Interactive map with zoom/pan controls (Leaflet via Folium)
- [ ] Flood extent layer toggleable
- [ ] Infrastructure markers with hover tooltips (name, address, status)
- [ ] Before/after slider with draggable divider
- [ ] Date labels on before/after images
- [ ] Responsive: 576px, 768px, 992px breakpoints
- [ ] 44px minimum touch targets
- [ ] Legend collapsible on mobile
- [ ] Share button with copyable URL
- [ ] Download buttons for PDF, PNG
- [ ] Sticky emergency footer (hidden in print)
- [ ] Full keyboard navigation

---

## Feature R2.6: Print-Ready PDF Outputs

**Assigned to:** Track C (PM-3)
**Effort:** 6-8 days
**Dependencies:** R2.2, R2.3
**Blocks:** None

### Task Breakdown

| Task ID | Description | Effort | Dependencies | Coder Assignment |
|---------|-------------|--------|--------------|------------------|
| R2.6.1 | Create pdf_report.py module structure | 2h | R2.2, R2.3 | Coder-C1 |
| R2.6.2 | Set up WeasyPrint integration | 3h | R2.6.1 | Coder-C1 |
| R2.6.3 | Implement 300 DPI rendering | 3h | R2.6.2 | Coder-C1 |
| R2.6.4 | Create print_config.py with page settings | 2h | R2.6.1 | Coder-C2 |
| R2.6.5 | Add US Letter page size support | 2h | R2.6.4 | Coder-C2 |
| R2.6.6 | Add A4 page size support | 2h | R2.6.4 | Coder-C1 |
| R2.6.7 | Implement margins (0.5" content, 0.125" bleed) | 2h | R2.6.4 | Coder-C2 |
| R2.6.8 | Add page numbers to multi-page reports | 3h | R2.6.2 | Coder-C1 |
| R2.6.9 | Create clickable TOC with hyperlinks | 4h | R2.6.8 | Coder-C2 |
| R2.6.10 | Implement B&W pattern overlays for flood | 4h | R2.3 | Coder-C1 |
| R2.6.11 | Document CMYK color values | 2h | R2.6.1 | Coder-C2 |
| R2.6.12 | Embed fonts or convert to outlines | 3h | R2.6.2 | Coder-C1 |
| R2.6.13 | Embed maps as 300 DPI PNG | 3h | R2.3, R2.6.3 | Coder-C2 |
| R2.6.14 | Add "Continued on next page" indicators | 2h | R2.6.8 | Coder-C1 |
| R2.6.15 | Write DPI verification tests | 2h | R2.6.3 | Coder-C2 |
| R2.6.16 | Write page margin tests | 2h | R2.6.7 | Coder-C1 |
| R2.6.17 | Write font embedding tests | 2h | R2.6.12 | Coder-C2 |
| R2.6.18 | Write B&W pattern distinguishability tests | 2h | R2.6.10 | Coder-C1 |

### R2.6 Pattern Overlays Specification

Task R2.6.10 must implement these B&W patterns:

| Severity | Pattern | Line Weight | Spacing |
|----------|---------|-------------|---------|
| Minor | Horizontal lines | 0.5pt | 6pt |
| Moderate | Diagonal lines 45deg | 0.5pt | 4pt |
| Significant | Cross-hatch | 0.5pt | 3pt |
| Severe | Dense dots | - | 2pt |
| Extreme | Solid fill | - | - |

### R2.6 Acceptance Criteria

- [ ] PDF at 300 DPI resolution
- [ ] US Letter (8.5x11") and A4 supported
- [ ] Margins: 0.5" content, 0.125" bleed
- [ ] Page numbers on multi-page reports
- [ ] TOC with clickable hyperlinks
- [ ] Pattern overlays for B&W printing
- [ ] CMYK values documented
- [ ] Fonts embedded or outlined
- [ ] Maps embedded at 300 DPI PNG
- [ ] "Continued on next page" indicators

---

## Implementation Sequence

### Week-by-Week Timeline

#### Week 1
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| A | R2.1 | R2.1.1-R2.1.12 (tokens, palettes, typography, components) | Coder-A1, A2 |
| B | R2.4 | R2.4.1-R2.4.7 (module setup, Census API, infrastructure loaders) | Coder-B1, B2 |

**Week 1 Gate:** Design tokens complete, Census API skeleton working

#### Week 2
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| A | R2.1 | R2.1.13-R2.1.18 (print stylesheet, icons, tests) | Coder-A1, A2 |
| B | R2.3 | R2.3.1-R2.3.4 (module structure, base maps) | Coder-B1 |
| B | R2.4 | R2.4.8-R2.4.13 (facility retrieval, impact estimation) | Coder-B2 |

**Week 2 Gate:** R2.1 COMPLETE, WCAG AA verified (QA-A1)

#### Week 3
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| A | R2.2 | R2.2.1-R2.2.12 (base template, components, exec summary) | Coder-A1, A2 |
| B | R2.3 | R2.3.5-R2.3.10 (flood renderer, infrastructure overlay) | Coder-B1, B2 |
| B | R2.4 | R2.4.14-R2.4.21 (emergency resources, caching, tests) | Coder-B1, B2 |

**Week 3 Gate:** Executive summary renders (QA-A2), Census returns population (QA-B1)

#### Week 4
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| A | R2.2 | R2.2.13-R2.2.22 (scale conversions, full report, tests) | Coder-A1, A2 |
| B | R2.3 | R2.3.11-R2.3.22 (map furniture, export, tests) | Coder-B1, B2 |

**Week 4 Gate:** R2.2 COMPLETE (QA-A3), R2.3 COMPLETE (QA-B3), R2.4 COMPLETE

#### Week 5
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| C | R2.5 | R2.5.1-R2.5.10 (Folium integration, slider component) | Coder-C1, C2 |

**Week 5 Gate:** Interactive map zoom/pan working (QA-C1)

#### Week 6
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| C | R2.5 | R2.5.11-R2.5.20 (responsive layout, share/download) | Coder-C1, C2 |

**Week 6 Gate:** Slider works on touch (QA-C2)

#### Week 7
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| C | R2.5 | R2.5.21-R2.5.26 (keyboard nav, generator, tests) | Coder-C1, C2 |
| C | R2.6 | R2.6.1-R2.6.9 (WeasyPrint, page setup, TOC) | Coder-C1, C2 |

**Week 7 Gate:** PDF 300 DPI with margins (QA-C3)

#### Week 8
| Track | Feature | Tasks | Coders |
|-------|---------|-------|--------|
| C | R2.6 | R2.6.10-R2.6.18 (patterns, fonts, tests) | Coder-C1, C2 |
| All | Integration | Full system integration testing | All |
| All | Accessibility | WCAG 2.1 AA audit | QA Team |

**Week 8 Gate:** Accessibility audit passes (QA-C4), RELEASE READY

---

## Quality Gates Summary

| Gate | Week | Feature | Owner | Criteria |
|------|------|---------|-------|----------|
| QA-A1 | 2 | R2.1 | PM-1 | WCAG AA contrast verified |
| QA-A2 | 3 | R2.2 | PM-1 | Executive summary renders |
| QA-A3 | 4 | R2.2 | PM-1 | Full report template complete |
| QA-B1 | 3 | R2.4 | PM-2 | Census API returns population |
| QA-B2 | 3 | R2.3 | PM-2 | Base maps with flood overlay |
| QA-B3 | 4 | R2.3 | PM-2 | 300 DPI export working |
| QA-C1 | 5 | R2.5 | PM-3 | Map zoom/pan working |
| QA-C2 | 6 | R2.5 | PM-3 | Slider touch functional |
| QA-C3 | 7 | R2.6 | PM-3 | PDF 300 DPI, correct margins |
| QA-C4 | 8 | All | PM-3 | Accessibility audit passes |

### Sentinel Review Points

Sentinel security review required at:

1. **Before Track C starts (Week 4):** Review R2.4 data integrations (Census API, external data)
2. **Before Release (Week 8):** Full security review of web reports (XSS, data exposure)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner |
|------|------------|--------|------------|-------|
| WeasyPrint rendering issues | Medium | Medium | Fallback to ReportLab | PM-3 |
| Census API downtime | Low | High | Aggressive caching, mock data mode | PM-2 |
| Map tile provider changes | Low | Medium | Abstract tile source, easy swap | PM-2 |
| Folium/Jinja template conflicts | Medium | Medium | Test integration early in Week 5 | PM-3 |
| Large map OOM | Medium | Medium | Tile-based rendering, limit extent | PM-2 |
| Browser compatibility | Low | Low | Test Chrome, Firefox, Safari | PM-3 |

---

## Communication Protocol

### Daily Standups
- Track A/B: Combined standup Weeks 1-4
- Track C: Joins Week 4
- Time: 9:00 AM daily

### Progress Updates
Post to #firstlight Matrix channel:
- Feature completion (major milestones)
- QA gate results
- Blockers requiring escalation

### Escalation Path
1. PM resolves within track
2. Cross-track issue -> Ratchet coordinates
3. Spec clarification -> Product Queen
4. Technical decision -> CTO

---

## Metadata

```yaml
document: REPORT_2.0_WORK_ASSIGNMENTS
epic: REPORT-2.0
version: 1.0.0
created: 2026-01-26
author: Ratchet (Engineering Manager)
branch: feature/report-2.0-human-readable-reporting

tracks:
  - id: A
    pm: PM-1
    features: [R2.1, R2.2]
    coders: [Coder-A1, Coder-A2]
    duration: 4 weeks

  - id: B
    pm: PM-2
    features: [R2.3, R2.4]
    coders: [Coder-B1, Coder-B2]
    duration: 4 weeks

  - id: C
    pm: PM-3
    features: [R2.5, R2.6]
    coders: [Coder-C1, Coder-C2]
    duration: 4 weeks

total_tasks: 112
total_effort: ~8 weeks with parallel execution
```

---

*Document prepared by Ratchet, Engineering Manager. Ready for Project Manager assignment.*
