# Track B Roadmap: Geospatial

## Overview
**Track ID**: B
**Spec**: /home/gprice/projects/firstlight/docs/specs/REPORT_2.0_WORK_ASSIGNMENTS.md
**Priority**: P0 (R2.3), P1 (R2.4)
**PM**: Project Manager PM-2
**Branch**: feature/report-2.0-human-readable-reporting

## Current Status
Track B started - implementing R2.4 Data Integrations (no dependencies). R2.3 Map Visualization waits for R2.1 design system completion (~1 week).

## Tasks

### Batch 1: R2.4 Data Integrations (Parallelizable - NO DEPENDENCIES)

#### Task R2.4.1: Census API Client ⚪
- **Task ID**: #4
- **Assignee**: Coder-B1
- **Effort**: Small (3-4 hours)
- **Dependencies**: None
- **Files**: `/home/gprice/projects/firstlight/core/reporting/data/census_client.py`
- **Deliverables**:
  - Census Bureau API client
  - Population retrieval by bounding box
  - Housing unit counts
  - 30-day response caching
  - Rate limiting (500 req/day)
- **Acceptance Criteria**:
  - [ ] Census API returns population for test AOI
  - [ ] Housing unit counts available
  - [ ] Responses cached (30 days)
  - [ ] Rate limiting implemented
  - [ ] Error handling and timeouts
  - [ ] Unit tests pass

#### Task R2.4.2: Infrastructure Data Client ✅
- **Task ID**: #5
- **Assignee**: Coder-B1
- **Effort**: Small (3-4 hours)
- **Dependencies**: None
- **Files**: `/home/gprice/projects/firstlight/core/reporting/data/infrastructure_client.py`
- **Deliverables**:
  - OpenStreetMap Overpass API client
  - Query hospitals, schools, fire stations, shelters
  - Parse and normalize to GeoJSON
  - 90-day caching
- **Acceptance Criteria**:
  - [x] OSM Overpass queries work for all facility types *(Completed 2026-01-26)*
  - [x] Results normalized to consistent schema *(Completed 2026-01-26)*
  - [x] GeoJSON output format *(Completed 2026-01-26)*
  - [x] 90-day caching implemented *(Completed 2026-01-26)*
  - [x] Error handling for API downtime *(Completed 2026-01-26)*
  - [ ] Unit tests pass *(Pending R2.4.4 - integration tests)*

#### Task R2.4.3: Emergency Resources Module ⚪
- **Task ID**: #6
- **Assignee**: Coder-B1
- **Effort**: Small (2-3 hours)
- **Dependencies**: None
- **Files**: `/home/gprice/projects/firstlight/core/reporting/data/emergency_resources.py`
- **Deliverables**:
  - FEMA contact information
  - State emergency management links
  - Red Cross shelter finder links
  - Road closure info sources
  - Configurable per state/region
- **Acceptance Criteria**:
  - [ ] State-specific emergency links available
  - [ ] FEMA contact info included
  - [ ] Configurable resource system
  - [ ] Easy to add new states/regions
  - [ ] Unit tests pass

#### Task R2.4.4: Module Init and Integration Tests ⚪
- **Task ID**: #7
- **Assignee**: Coder-B1
- **Effort**: Small (2-3 hours)
- **Dependencies**: R2.4.1, R2.4.2, R2.4.3
- **Files**:
  - `/home/gprice/projects/firstlight/core/reporting/data/__init__.py`
  - `/home/gprice/projects/firstlight/tests/test_data_integrations.py`
- **Deliverables**:
  - Module initialization
  - Integration tests for all R2.4 modules
  - Mock API tests for offline testing
- **Acceptance Criteria**:
  - [ ] Module imports work correctly
  - [ ] Integration tests pass
  - [ ] Mock API tests work offline
  - [ ] Cache behavior verified
  - [ ] All R2.4 acceptance criteria met

### Batch 2: R2.3 Map Visualization (Waits for R2.1 Design System)

#### Status: ⚪ WAITING ON R2.1
Map visualization tasks will begin once Track A completes R2.1 (design system with color palettes).

**Planned Tasks**:
- R2.3.1-R2.3.4: Base maps and tile sources (Coder-B1)
- R2.3.5-R2.3.7: Flood renderer with severity coloring (Coder-B2)
- R2.3.8-R2.3.10: Infrastructure icon rendering (Coder-B1, B2)
- R2.3.11-R2.3.16: Map furniture (scale, legend, locator map)
- R2.3.17-R2.3.19: Pattern overlays and export
- R2.3.20-R2.3.22: Tests

## Integration Points

### R2.4 Data → R2.5 Web Reports
- Population data used in "Who Is Affected" section
- Infrastructure locations rendered on maps
- Emergency resources shown in web footer

### R2.3 Maps → R2.5/R2.6 Output Formats
- Map exports embedded in web reports
- 300 DPI PNG for PDF reports
- Interactive maps for web version

### Cross-Track Dependencies
- **R2.1 (Track A)** → R2.3: Color palettes for flood severity
- **R2.2 (Track A)** → R2.3: Template structure for map placement
- **R2.3 + R2.4** → R2.5 (Track C): Maps and data for web reports

## Risks

| Risk | Likelihood | Impact | Mitigation | Status |
|------|------------|--------|------------|--------|
| Census API downtime | Low | High | Aggressive caching, mock data mode | Monitoring |
| OSM Overpass rate limits | Medium | Medium | 90-day cache, HIFLD fallback | In progress |
| Large map OOM | Medium | Medium | Tile-based rendering, limit extent | Planned for R2.3 |
| Map tile provider changes | Low | Medium | Abstract tile source, easy swap | Planned for R2.3 |

## Completion Criteria

### R2.4 Complete When:
- [ ] All tasks R2.4.1 through R2.4.4 complete
- [ ] Census API integration tested
- [ ] Infrastructure queries working
- [ ] Emergency resources configurable
- [ ] All integration tests passing
- [ ] Ready for Track C (R2.5) to consume data

### R2.3 Complete When:
- [ ] All tasks R2.3.1 through R2.3.22 complete
- [ ] Base maps rendering correctly
- [ ] Flood extent with 5-level severity coloring
- [ ] Infrastructure overlays functional
- [ ] 300 DPI export working
- [ ] All tests passing
- [ ] Ready for Sentinel security review

### Track B Complete When:
- [ ] R2.4 Complete
- [ ] R2.3 Complete
- [ ] Integration verified with Track A templates
- [ ] QA checkpoints passed:
  - [ ] QA-B1 (Week 2): Census API returns population
  - [ ] QA-B2 (Week 3): Base maps with flood overlay
  - [ ] QA-B3 (Week 4): 300 DPI export working
- [ ] Ready to hand off to Track C

## Progress Tracking

### Week 1 (Current)
- ⚪ R2.4.1: Census client
- ⚪ R2.4.2: Infrastructure client
- ⚪ R2.4.3: Emergency resources
- ⚪ R2.4.4: Integration tests

### Week 2 (Pending R2.1)
- Waiting: R2.3 map visualization tasks

### Week 3
- Continue: R2.3 implementation
- Complete: R2.4 integration

### Week 4
- Finish: R2.3 tasks
- QA: Full Track B verification
- Handoff: Ready for Track C

---

## Metadata

```yaml
track: B
pm: PM-2
features: [R2.3, R2.4]
duration: 4 weeks
status: In Progress
current_batch: R2.4 (Batch 1)
blocked_tasks: R2.3 (waiting on R2.1)
```

---

*PM-2 tracking geospatial features for REPORT-2.0 epic. Starting with R2.4 Data Integrations.*
