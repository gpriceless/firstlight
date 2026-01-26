# OpenSpec: Human-Readable Reporting Overhaul

**Epic ID:** REPORT-2.0
**Status:** APPROVED
**Created:** 2026-01-26
**Author:** Product Queen
**Version:** 1.0.0
**Approved By:** Product Queen (2026-01-26)

---

## Document Status

| Review Stage | Status | Reviewer | Date |
|--------------|--------|----------|------|
| Product Specification | COMPLETE | Product Queen | 2026-01-26 |
| Requirements Review | COMPLETE | Requirements Reviewer | 2026-01-26 |
| Technical Feasibility | APPROVED | Engineering | 2026-01-26 |
| Design Validation | APPROVED | Design-Sorcerer | 2026-01-26 |
| Final Approval | APPROVED | Product Queen | 2026-01-26 |

---

## 1. Epic Overview

### 1.1 Title

**Human-Readable Reporting Overhaul**

Transform FirstLight's technically excellent but expert-oriented reports into accessible, actionable intelligence products that serve emergency managers, local officials, affected residents, and media.

### 1.2 Business Justification

**The Problem:**

FirstLight produces geospatially accurate flood, wildfire, and storm analysis, but the output reports are designed for technical users who understand satellite imagery, spectral indices, and GIS concepts. A geospatial engineer assessment revealed critical gaps:

> "A flooded area statistic is technically accurate but operationally incomplete without context about affected populations, infrastructure, and actionable response guidance."

**Current Report Limitations:**
- No pre-event imagery for comparison (missing "before" context)
- No infrastructure overlays (hospitals, schools, evacuation routes)
- No population impact estimates (just hectares, not people)
- No plain-language explanations (technical jargon throughout)
- No interactive maps (static outputs only)
- No emergency resource links (no actionable response information)

**The Opportunity:**

Emergency managers need to make decisions in minutes, not hours. A properly designed report can answer "How bad is it?" and "What should I do?" in under 30 seconds. Current reports require technical interpretation that delays response.

**Business Value:**
1. **Faster Decision-Making:** Reduce time-to-decision from hours to minutes
2. **Broader Audience:** Extend platform value beyond technical GIS specialists
3. **Increased Adoption:** Emergency management agencies prefer actionable intelligence
4. **Reduced Training:** Self-explanatory reports require less user education
5. **Media-Ready:** Reports can be shared with press without modification

### 1.3 Target Users

| User Type | Primary Needs | Report Format |
|-----------|---------------|---------------|
| **Emergency Managers** | Fast situational awareness, facility status, resource allocation | Executive Summary (1-page PDF) |
| **Local Officials** | Constituent impact, infrastructure status, media talking points | Full Report (PDF) |
| **Affected Residents** | "Am I in danger?", "What should I do?", emergency resources | Interactive Web Report |
| **Media/Journalists** | Shareable graphics, plain-language quotes, before/after visuals | Web Report + PNG exports |
| **Technical Analysts** | Full methodology, raw data access, QC details | Technical Appendix |

### 1.4 Success Metrics

| Metric | Current Baseline | Target | Measurement Method |
|--------|------------------|--------|-------------------|
| **Time to Key Finding** | 5-10 minutes (requires interpretation) | < 30 seconds | User testing |
| **Audience Accessibility** | GIS specialists only | Non-technical officials | Comprehension test |
| **Report Completeness** | Technical metrics only | People + Places + Actions | Checklist audit |
| **Emergency Resource Visibility** | 0% (not included) | 100% (prominent) | Template audit |
| **Mobile Readability** | 0% (no responsive design) | 100% (all reports) | Device testing |
| **Colorblind Accessibility** | Unknown | WCAG AA compliant | Automated testing |
| **Print Quality** | Variable | 300 DPI standard | Output sampling |

### 1.5 Scope Boundaries

**In Scope:**
- Executive summary templates (1-page)
- Full report templates (multi-page)
- Interactive web reports with maps
- Print-ready PDF generation
- Before/after comparison visualizations
- Infrastructure overlay capability
- Population impact estimates (Census integration)
- Emergency resource links
- Design system implementation
- Accessibility compliance (WCAG 2.1 AA)

**Out of Scope:**
- Real-time dashboards (separate initiative)
- Multi-language support (future phase)
- Social media integration (future phase)
- Mobile native apps (web-only for now)
- Historical event comparison module (future phase)
- Crowdsource validation (future phase)

---

## 2. Feature Breakdown

### Feature R2.1: Design System Implementation

**Feature ID:** R2.1
**Priority:** P0 (Foundation - blocks all other features)
**Complexity:** Medium (M)
**Estimated Effort:** 5-7 days

#### Description

Implement the FirstLight Design System as CSS custom properties, component classes, and reusable styling utilities. This provides the visual foundation for all report outputs.

#### User Stories

**US-R2.1.1:** As a report generator, I want consistent styling across all outputs so that reports have a professional, unified appearance.

**US-R2.1.2:** As a developer, I want design tokens as CSS variables so that I can easily apply brand-consistent colors without hardcoding hex values.

**US-R2.1.3:** As an accessibility user, I want colorblind-safe palettes so that I can distinguish flood severity levels without relying on color alone.

#### Acceptance Criteria

- [ ] **AC-1:** CSS custom properties file created with all design tokens from `DESIGN_SYSTEM.md`
- [ ] **AC-2:** Flood severity palette implemented with 5 levels (minor through extreme)
- [ ] **AC-3:** Wildfire severity palette implemented with 6 levels
- [ ] **AC-4:** Confidence/uncertainty palette implemented with 4 levels
- [ ] **AC-5:** Typography scale implemented (text-xs through text-5xl)
- [ ] **AC-6:** Spacing scale implemented (space-1 through space-24)
- [ ] **AC-7:** All colors pass WCAG 2.1 AA contrast requirements (4.5:1 for normal text, 3:1 for large text), verified using axe-core with test results logged in CI [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-8:** Component CSS classes created: `.metric-card`, `.alert-box`, `.severity-badge`, `.data-table`
- [ ] **AC-9:** Print stylesheet created with proper page break handling
- [ ] **AC-10:** [STRETCH] Dark mode support (optional, lower priority)

#### Technical Requirements

**New Files:**
```
agents/reporting/static/
├── css/
│   ├── tokens.css           # CSS custom properties (colors, spacing, typography)
│   ├── components.css       # Reusable component classes
│   ├── utilities.css        # Utility classes (margins, padding, text align)
│   ├── print.css            # Print-specific styles
│   └── main.css             # Master stylesheet importing all above
└── icons/
    ├── heroicons/           # SVG icons for infrastructure, status
    └── firstlight/          # Brand assets
```

**CSS Token Structure:**
```css
:root {
  /* Brand Colors */
  --color-brand-navy: #1A365D;
  --color-brand-blue: #2C5282;
  --color-brand-sky: #3182CE;

  /* Flood Severity */
  --color-flood-minor: #90CDF4;
  --color-flood-moderate: #4299E1;
  --color-flood-significant: #2B6CB0;
  --color-flood-severe: #2C5282;
  --color-flood-extreme: #1A365D;

  /* Typography */
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  /* ... etc */
}
```

#### Dependencies

- None (foundation feature)

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| Token completeness | Unit | Verify all tokens from DESIGN_SYSTEM.md are implemented |
| Contrast ratios | Automated | Run axe-core or similar on sample HTML |
| Component rendering | Visual | Screenshot comparison of components |
| Print rendering | Manual | Print to PDF, verify styling |

---

### Feature R2.2: Plain-Language Report Templates

**Feature ID:** R2.2
**Priority:** P0 (Core deliverable)
**Complexity:** Medium (M)
**Estimated Effort:** 8-10 days

#### Description

Create Jinja2 HTML templates for human-readable reports that translate technical metrics into plain language. Templates support multiple audience levels from executive summary to full analysis.

#### User Stories

**US-R2.2.1:** As an emergency manager, I want a 1-page executive summary so that I can understand the situation in 30 seconds.

**US-R2.2.2:** As a local official, I want "What This Means" sections so that I can explain impacts to constituents without geospatial expertise.

**US-R2.2.3:** As a resident, I want plain-language severity descriptions so that I understand danger levels without technical knowledge.

**US-R2.2.4:** As anyone affected, I want prominent emergency contact information so that I know who to call for help.

#### Acceptance Criteria

- [ ] **AC-1:** Executive summary template created (1-page layout per `REPORT_VISUAL_SPEC.md`)
- [ ] **AC-2:** Template includes "What Happened" section with plain-language event description
- [ ] **AC-3:** Template includes "Who Is Affected" section with population and facility counts
- [ ] **AC-4:** Template includes "What To Do" section with actionable guidance
- [ ] **AC-5:** Key metrics displayed as visual cards (not tables)
- [ ] **AC-6:** Severity badges use text labels, not just colors
- [ ] **AC-7:** Scale references include conversions: hectares to acres (1 ha = 2.47 acres), large areas to football fields (1 ha = 1.9 fields), percentages to fractions. Each numeric metric shows at least 2 relatable comparisons. [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-8:** Emergency resource section includes FEMA, local EM, and disaster hotlines
- [ ] **AC-9:** Full report template created with TOC and multi-page support
- [ ] **AC-10:** Technical appendix template available for detailed methodology

#### Template Structure

```
agents/reporting/templates/
├── base.html                    # Base template with header/footer
├── executive_summary.html       # 1-page summary
├── full_report/
│   ├── cover.html              # Cover page
│   ├── toc.html                # Table of contents
│   ├── what_happened.html      # Event description section
│   ├── who_affected.html       # Population/infrastructure impact
│   ├── maps.html               # Map page layouts
│   ├── recommendations.html    # Action items
│   └── appendix_technical.html # Technical details for analysts
├── components/
│   ├── metric_card.html        # Reusable stat card
│   ├── alert_box.html          # Warning/info boxes
│   ├── severity_badge.html     # Severity indicator
│   └── facility_card.html      # Infrastructure status card
└── partials/
    ├── header.html             # FirstLight branded header
    ├── footer.html             # Attribution, timestamp, disclaimer
    └── emergency_resources.html # Emergency contact block
```

#### Plain Language Conversion Rules

| Technical Term | Plain Language |
|---------------|----------------|
| 3,026 hectares | About 7,500 acres (roughly 5,700 football fields) |
| 28.9% coverage | Nearly 1 in 3 acres in the study area |
| IoU 85% | 85% agreement between detection methods |
| >85% confidence | HIGH CONFIDENCE: Flooding confirmed |
| 60-85% confidence | MODERATE: Likely flooded |
| <60% confidence | UNCERTAIN: Unable to confirm |

#### Dependencies

- R2.1 (Design System) - for consistent styling

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| Template rendering | Unit | Render with sample data, verify output |
| Plain language | Manual | Have non-technical user read and explain back |
| Mobile responsive | Visual | Test on phone viewport |
| Print layout | Manual | Generate PDF, check page breaks |

---

### Feature R2.3: Map Visualization Overhaul

**Feature ID:** R2.3
**Priority:** P0 (Core deliverable)
**Complexity:** Large (L)
**Estimated Effort:** 12-15 days

#### Description

Implement cartographic standards from `CARTOGRAPHIC_SPEC.md` to produce professional, accessible map products. Includes base map integration, flood extent styling, infrastructure overlays, and map furniture (legend, scale bar, north arrow).

#### User Stories

**US-R2.3.1:** As an emergency manager, I want infrastructure overlays on flood maps so that I can see which hospitals, schools, and shelters are affected.

**US-R2.3.2:** As a user, I want consistent map styling so that all FirstLight maps have the same professional appearance.

**US-R2.3.3:** As a print user, I want proper map furniture (scale bar, legend, north arrow) so that maps are self-explanatory.

**US-R2.3.4:** As a colorblind user, I want pattern overlays for severity levels so that I can distinguish categories without color.

#### Acceptance Criteria

- [ ] **AC-1:** Base map integration with CartoDB Positron tiles (or equivalent muted style)
- [ ] **AC-2:** Flood extent rendering with 5-level severity coloring per design system
- [ ] **AC-3:** Infrastructure icon set implemented (hospital, school, shelter, fire station, police)
- [ ] **AC-4:** Infrastructure status calculated as: "In flood zone" (intersects flood polygon), "Adjacent" (within 500m buffer), "Outside flood zone" (beyond 500m), "Data unavailable" (location unknown). Displayed with color-coded icons per CARTOGRAPHIC_SPEC Section 3.4. [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-5:** Scale bar with metric units, alternating black/white style
- [ ] **AC-6:** North arrow present on all print maps
- [ ] **AC-7:** Legend with severity levels AND infrastructure icons
- [ ] **AC-8:** Inset/locator map showing regional context
- [ ] **AC-9:** Title block with event name, location, date
- [ ] **AC-10:** Attribution block with data sources, CRS, timestamp
- [ ] **AC-11:** Pattern overlays available for B&W printing
- [ ] **AC-12:** Maps render at 300 DPI for print quality

#### Technical Requirements

**New Files:**
```
agents/reporting/
├── cartography/
│   ├── __init__.py
│   ├── config.py              # CRS settings, default styles
│   ├── base_maps.py           # Tile source management
│   ├── flood_renderer.py      # Flood extent styling
│   ├── infrastructure.py      # Icon rendering, status indication
│   ├── map_furniture.py       # Legend, scale bar, north arrow, title block
│   └── export.py              # PNG/PDF map export
└── static/
    └── icons/
        └── infrastructure/    # Hospital, school, shelter SVGs
```

**Map Generation Stack:**
- Matplotlib + Cartopy for static maps (print quality)
- Contextily for base map tiles
- Folium for interactive web maps (R2.5)

**Projection Strategy:**
| Scale | CRS | Rationale |
|-------|-----|-----------|
| Local (<100km) | UTM Zone (auto-detected) | Minimal distortion |
| Regional | State Plane or UTM | Agency familiarity |
| Web display | Web Mercator (EPSG:3857) | Tile compatibility |
| Data exchange | WGS 84 (EPSG:4326) | Interoperability |

#### Dependencies

- R2.1 (Design System) - color palettes, typography

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| Projection accuracy | Unit | Verify scale bar matches known distance |
| Layer stacking | Visual | Confirm proper z-order (base < flood < infrastructure < labels) |
| Icon rendering | Visual | All icons render at correct size |
| Print quality | Manual | 300 DPI export, verify no artifacts |
| Colorblind | Automated | Simulate deuteranopia, verify distinguishable |

---

### Feature R2.4: Data Integrations

**Feature ID:** R2.4
**Priority:** P1 (High value, parallel work)
**Complexity:** Medium (M)
**Estimated Effort:** 8-10 days

#### Description

Integrate external data sources to provide human context for technical analysis: population estimates (Census), infrastructure locations (HIFLD/OSM), and emergency resource links.

#### User Stories

**US-R2.4.1:** As an emergency manager, I want population impact estimates so that I know how many people are affected, not just acres.

**US-R2.4.2:** As a responder, I want critical infrastructure locations so that I can prioritize facilities in the flood zone.

**US-R2.4.3:** As a resident, I want emergency resource links so that I know where to get help.

#### Acceptance Criteria

- [ ] **AC-1:** Census API integration returns population estimates for AOI
- [ ] **AC-2:** Housing unit counts derived from Census data
- [ ] **AC-3:** Vulnerable population estimates from ACS 5-year data: Elderly (65+, Table B01001), Children (under 18, Table B01001), Poverty (below poverty level, Table B17001). Displayed as counts with margin of error when available. [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-4:** Infrastructure data source configured (HIFLD primary, OSM fallback)
- [ ] **AC-5:** Hospital locations and names retrieved for AOI
- [ ] **AC-6:** School locations retrieved for AOI
- [ ] **AC-7:** Emergency shelter locations retrieved (FEMA National Shelter System if available)
- [ ] **AC-8:** Fire station locations retrieved for AOI
- [ ] **AC-9:** "Facilities in flood zone" calculation implemented (spatial intersection)
- [ ] **AC-10:** Emergency resource links configurable per state/region
- [ ] **AC-11:** API rate limiting and caching implemented

#### Technical Requirements

**New Files:**
```
core/data/context/
├── __init__.py
├── census.py               # Census API client
├── infrastructure.py       # HIFLD/OSM infrastructure data
├── emergency_resources.py  # Resource link configuration
└── population.py           # Population impact estimation
```

**Data Sources:**

| Source | Data Type | Access Method | Rate Limit |
|--------|-----------|---------------|------------|
| US Census | Population, demographics | REST API (api.census.gov) | 500/day default |
| HIFLD | Critical infrastructure | Geospatial download | None (static) |
| OpenStreetMap | POIs, buildings | Overpass API | Fair use |
| FEMA NSS | Shelter locations | API (if available) | TBD |

**Caching Strategy:**
- Census data: 30-day cache (slow-changing)
- Infrastructure: 90-day cache (rarely changes)
- Emergency resources: 7-day cache (may update in disaster)

#### Dependencies

- None (can work in parallel with other features)

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| Census API | Integration | Verify population retrieval for known area |
| Infrastructure query | Integration | Verify hospital count matches known data |
| Spatial intersection | Unit | Test "in flood zone" calculation |
| Cache behavior | Unit | Verify cache hit/miss/expiry |

---

### Feature R2.5: Interactive Web Reports

**Feature ID:** R2.5
**Priority:** P1 (High impact)
**Complexity:** Large (L)
**Estimated Effort:** 12-15 days

#### Description

Create interactive HTML reports with embedded maps, before/after sliders, hover tooltips, and mobile-responsive layouts. This is the primary delivery format for web users.

#### User Stories

**US-R2.5.1:** As a user, I want to zoom and pan the flood map so that I can find my specific location.

**US-R2.5.2:** As a user, I want a before/after slider so that I can see what changed visually.

**US-R2.5.3:** As a mobile user, I want reports that work on my phone so that I can access information in the field.

**US-R2.5.4:** As a user, I want to hover over facilities to see details so that I can understand their status.

#### Acceptance Criteria

- [ ] **AC-1:** Interactive map with zoom/pan controls (Leaflet or Folium)
- [ ] **AC-2:** Flood extent layer toggleable on/off
- [ ] **AC-3:** Infrastructure markers with hover tooltips showing name, address, status
- [ ] **AC-4:** Before/after slider component with draggable divider
- [ ] **AC-5:** Date labels on before/after images
- [ ] **AC-6:** Mobile-responsive layout (breakpoints: 576px, 768px, 992px)
- [ ] **AC-7:** Touch-friendly controls (44px minimum touch targets)
- [ ] **AC-8:** Legend collapsible on mobile
- [ ] **AC-9:** Share button with copyable URL
- [ ] **AC-10:** Download buttons for PDF, PNG exports
- [ ] **AC-11:** Emergency contact section uses CSS `position: sticky` at viewport bottom on web; hidden during print (full footer on each page instead). Verified by manual scroll test on mobile and desktop. [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-12:** Keyboard navigation for accessibility

#### Technical Requirements

**New Files:**
```
agents/reporting/
├── templates/
│   └── web_report/
│       ├── index.html          # Main web report template
│       ├── map_embed.html      # Folium map iframe
│       └── slider.html         # Before/after slider component
├── static/
│   ├── js/
│   │   ├── map-controls.js     # Map interaction handlers
│   │   ├── slider.js           # Before/after slider logic
│   │   └── share.js            # Share/download functionality
│   └── css/
│       └── web-report.css      # Web-specific styles
└── generators/
    └── web_report.py           # Web report generation logic
```

**JavaScript Dependencies:**
- Leaflet.js (map rendering)
- No jQuery (vanilla JS for bundle size)

**Responsive Breakpoints:**
```css
@media (max-width: 576px) { /* Mobile */ }
@media (max-width: 768px) { /* Tablet portrait */ }
@media (max-width: 992px) { /* Tablet landscape */ }
```

#### Dependencies

- R2.1 (Design System)
- R2.2 (Templates)
- R2.3 (Map Visualization)

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| Map interaction | Manual | Verify zoom, pan, click behavior |
| Slider functionality | Manual | Test drag, touch, keyboard |
| Responsive layout | Visual | Test at all breakpoints |
| Touch targets | Automated | Verify minimum 44px |
| Load performance | Automated | Page load < 3s on 3G |

---

### Feature R2.6: Print-Ready Outputs

**Feature ID:** R2.6
**Priority:** P1 (Required for official distribution)
**Complexity:** Medium (M)
**Estimated Effort:** 6-8 days

#### Description

Generate print-quality PDF reports with proper DPI, CMYK color conversion capability, and pattern overlays for black-and-white printing.

#### User Stories

**US-R2.6.1:** As an official, I want print-quality PDFs so that I can distribute physical copies at emergency briefings.

**US-R2.6.2:** As a print shop, I want proper margins and bleed so that documents print correctly.

**US-R2.6.3:** As a field responder, I want reports that work in B&W so that I can use any available printer.

#### Acceptance Criteria

- [ ] **AC-1:** PDF generation at 300 DPI resolution
- [ ] **AC-2:** US Letter (8.5x11") and A4 page sizes supported
- [ ] **AC-3:** Proper margins (0.5" content margin, 0.125" bleed for professional print)
- [ ] **AC-4:** Page numbers on multi-page reports
- [ ] **AC-5:** Table of contents with hyperlinks (clickable PDF)
- [ ] **AC-6:** Pattern overlays available for flood severity (for B&W)
- [ ] **AC-7:** CMYK color values documented for offset printing
- [ ] **AC-8:** Fonts embedded or converted to outlines
- [ ] **AC-9:** Map exports maintain quality: static maps embedded as PNG at 300 DPI, scale bars/legends as SVG where supported, text elements preserved as selectable text. [UPDATED BY REQUIREMENTS REVIEW]
- [ ] **AC-10:** "Continued on next page" indicators for long tables

#### Technical Requirements

**PDF Generation Stack:**
- WeasyPrint (CSS-to-PDF rendering)
- Alternative: Paged.js for complex layouts

**Print Configuration:**
```python
class PrintConfig:
    dpi: int = 300
    page_size: str = "letter"  # or "a4"
    margin_inches: float = 0.5
    bleed_inches: float = 0.125  # for professional print
    color_space: str = "rgb"  # or "cmyk" for offset
```

**Pattern Overlays (B&W):**
| Severity | Pattern |
|----------|---------|
| Minor | Horizontal lines, 6pt spacing |
| Moderate | Diagonal lines 45deg, 4pt spacing |
| Significant | Cross-hatch, 3pt spacing |
| Severe | Dense dots, 2pt spacing |
| Extreme | Solid fill |

#### Dependencies

- R2.1 (Design System)
- R2.2 (Templates)
- R2.3 (Map Visualization)

#### Test Plan

| Test | Type | Description |
|------|------|-------------|
| DPI verification | Manual | Check image resolution in PDF |
| Page margins | Manual | Measure printed output |
| B&W patterns | Visual | Print B&W, verify distinguishable |
| Font embedding | Automated | PDF/A validation |
| File size | Automated | < 10MB for standard report |

---

## 3. Technical Requirements

### 3.1 Required Libraries/Tools

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| **Jinja2** | 3.x | Template rendering | BSD |
| **Matplotlib** | 3.x | Static map generation | BSD |
| **Cartopy** | 0.21+ | Map projections, features | LGPL |
| **Contextily** | 1.x | Base map tiles | BSD |
| **Folium** | 0.14+ | Interactive web maps | MIT |
| **WeasyPrint** | 59+ | PDF generation | BSD |
| **Requests** | 2.28+ | API calls (Census, etc.) | Apache 2.0 |
| **Heroicons** | 2.x | Icon library | MIT |

### 3.2 API Integrations

| API | Purpose | Auth Required | Rate Limit |
|-----|---------|---------------|------------|
| Census API | Population data | API key (free) | 500/day |
| HIFLD Open Data | Infrastructure | None | None |
| Overpass (OSM) | POI fallback | None | Fair use |
| CartoDB Tiles | Base maps | None | Fair use |

### 3.3 File Format Specifications

| Format | Resolution | Use Case | Max Size |
|--------|------------|----------|----------|
| HTML | N/A | Web reports | No limit |
| PDF | 300 DPI | Print reports | 50MB |
| PNG | 144 DPI (web), 300 DPI (print) | Map exports | 10MB |
| GeoJSON | N/A | Data exchange | No limit |

### 3.4 Performance Requirements

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| Report generation time | < 30 seconds | End-to-end timing |
| Web report load time | < 3 seconds on 3G | Lighthouse audit |
| PDF file size | < 10MB for standard report | File size check |
| Map render time | < 5 seconds | Timing instrumentation |
| API response caching | 90%+ cache hit rate | Cache metrics |

### 3.5 Accessibility Requirements

| Requirement | Standard | Test Method |
|-------------|----------|-------------|
| Color contrast | WCAG 2.1 AA (4.5:1 text) | Automated (axe-core) |
| Color independence | No color-only meaning | Manual review |
| Keyboard navigation | Full functionality | Manual testing |
| Screen reader support | Proper ARIA labels | Screen reader testing |
| Alt text | All images described | Template audit |
| Focus indicators | Visible 2px outline | Visual inspection |

---

## 4. Dependencies Graph

### 4.1 Feature Dependencies

```
                    ┌─────────────────────────┐
                    │ R2.1 Design System      │ ◀── START HERE (Foundation)
                    │ Priority: P0            │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│ R2.2 Templates    │ │ R2.3 Map Viz        │ │ R2.4 Data Integr.   │
│ Priority: P0      │ │ Priority: P0        │ │ Priority: P1        │
│                   │ │                     │ │ (Parallel OK)       │
└─────────┬─────────┘ └─────────┬───────────┘ └─────────┬───────────┘
          │                     │                       │
          └──────────┬──────────┘                       │
                     │                                  │
                     ▼                                  │
          ┌─────────────────────┐                       │
          │ R2.5 Web Reports    │◀──────────────────────┘
          │ Priority: P1        │  (soft dependency: uses population data, can stub for development) [UPDATED BY REQUIREMENTS REVIEW]
          └─────────┬───────────┘
                    │
                    ▼
          ┌─────────────────────┐
          │ R2.6 Print Outputs  │
          │ Priority: P1        │
          └─────────────────────┘
```

### 4.2 External Dependencies

| Feature | External Dependency | Risk | Mitigation |
|---------|---------------------|------|------------|
| R2.3 | CartoDB tile availability | Low | Fallback to OSM tiles |
| R2.4 | Census API availability | Medium | Cache aggressively, mock data fallback |
| R2.4 | HIFLD data updates | Low | Use static download, refresh quarterly |
| R2.5 | Leaflet.js CDN | Low | Bundle locally |
| R2.6 | WeasyPrint system deps | Medium | Document installation, use Docker |

### 4.3 Parallelization Opportunities

**Can Run in Parallel:**
- R2.4 (Data Integrations) can run parallel to R2.2, R2.3
- CSS development (R2.1) and Python map code (R2.3) can be split between developers

**Must Be Sequential:**
- R2.1 must complete before R2.2, R2.3, R2.5, R2.6
- R2.2 + R2.3 must complete before R2.5
- R2.5 should complete before R2.6 (shared learnings)

---

## 5. Implementation Roadmap

### 5.1 Phase 1: Foundation (Week 1-2)

| Task | Feature | Effort | Dependencies |
|------|---------|--------|--------------|
| CSS tokens implementation | R2.1 | 2 days | None |
| Component CSS classes | R2.1 | 2 days | Tokens |
| Print stylesheet | R2.1 | 1 day | Component CSS |
| Icon library setup | R2.1 | 1 day | None |
| Base template structure | R2.2 | 2 days | R2.1 |
| Executive summary template | R2.2 | 3 days | Base template |

### 5.2 Phase 2: Core Capabilities (Week 3-4)

| Task | Feature | Effort | Dependencies |
|------|---------|--------|--------------|
| Map rendering pipeline | R2.3 | 4 days | R2.1 |
| Infrastructure overlays | R2.3 | 3 days | Map pipeline |
| Map furniture (legend, scale) | R2.3 | 2 days | Map pipeline |
| Census API integration | R2.4 | 3 days | None |
| Infrastructure data loading | R2.4 | 3 days | None |

### 5.3 Phase 3: Interactive Features (Week 5-6)

| Task | Feature | Effort | Dependencies |
|------|---------|--------|--------------|
| Folium map integration | R2.5 | 4 days | R2.3 |
| Before/after slider | R2.5 | 3 days | R2.5 map |
| Mobile responsive layout | R2.5 | 2 days | R2.2 |
| Full report template | R2.2 | 3 days | Exec summary |

### 5.4 Phase 4: Polish and Print (Week 7-8)

| Task | Feature | Effort | Dependencies |
|------|---------|--------|--------------|
| PDF generation pipeline | R2.6 | 4 days | R2.2, R2.3 |
| Pattern overlays (B&W) | R2.6 | 2 days | PDF pipeline |
| Accessibility audit | All | 2 days | All features |
| Integration testing | All | 3 days | All features |
| Documentation | All | 2 days | All features |

### 5.5 Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1 | 2 weeks | Design system, basic templates |
| Phase 2 | 2 weeks | Map visualization, data integrations |
| Phase 3 | 2 weeks | Interactive web reports |
| Phase 4 | 2 weeks | Print outputs, polish |
| **Total** | **8 weeks** | Complete reporting overhaul |

---

## 6. Testing Strategy [ADDED BY REQUIREMENTS REVIEW]

### 6.1 Test Environment Requirements

| Environment | Configuration |
|-------------|---------------|
| **Browsers** | Chrome 120+, Firefox 115+, Safari 17+, Edge 120+ |
| **Devices** | Desktop, Tablet (768px), Mobile (375px) |
| **PDF Viewers** | Adobe Reader, macOS Preview, Chrome built-in |
| **OS** | Linux (CI), macOS (development), Windows (validation) |

### 6.2 Accessibility Testing

| Method | Tool | Frequency |
|--------|------|-----------|
| Automated contrast | axe-core v4.x | Every PR (CI) |
| Screen reader | NVDA, VoiceOver | Per feature release |
| Colorblind simulation | Stark plugin | Design review stage |
| Keyboard navigation | Manual | Per interactive feature |

### 6.3 Performance Benchmarks

| Metric | Target | Measurement |
|--------|--------|-------------|
| Report generation (standard AOI) | < 30 seconds | End-to-end timing |
| Web report initial load (3G) | < 3 seconds | Lighthouse audit |
| PDF file size (standard report) | < 10 MB | File size check |
| PDF file size (full technical) | < 50 MB | File size check |
| Map render time | < 5 seconds | Instrumented timing |

### 6.4 Test Types by Feature

| Feature | Unit | Integration | Visual | Manual |
|---------|------|-------------|--------|--------|
| R2.1 Design System | Token completeness | CSS loading | Screenshot comparison | Print test |
| R2.2 Templates | Render validation | Full report generation | Layout review | Readability test |
| R2.3 Map Viz | Projection accuracy | Layer stacking | Icon rendering | Print quality |
| R2.4 Data Integration | API mocking | Cache behavior | N/A | Data validation |
| R2.5 Web Reports | Component tests | Full page load | Responsive breakpoints | Touch interaction |
| R2.6 Print Outputs | PDF structure | Multi-page generation | DPI verification | Physical print |

---

## 7. Deployment Strategy [ADDED BY REQUIREMENTS REVIEW]

### 7.1 Integration Points

| Component | Integration Method |
|-----------|-------------------|
| **CLI** | New `flight report` subcommand under existing CLI framework |
| **Output Location** | `{output_dir}/reports/` alongside existing outputs |
| **Web Reports** | Static HTML files, no server component required |
| **Configuration** | New `reporting` section in flight config |

### 7.2 Rollout Plan

| Phase | Scope | Validation |
|-------|-------|------------|
| **Alpha** | Internal team with Hurricane Ian sample data | Functionality complete |
| **Beta** | 3 pilot events (1 flood, 1 wildfire, 1 storm) | Stakeholder feedback |
| **GA** | General availability | Documentation complete |

### 7.3 Backwards Compatibility

- Existing `flight run` command unchanged
- New `flight report` command additive
- Existing output formats preserved
- New reporting features opt-in

---

## 8. Risk Assessment

### 8.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WeasyPrint rendering issues | Medium | Medium | Have fallback to ReportLab |
| Census API downtime | Low | High | Aggressive caching, mock data mode |
| Map tile provider changes | Low | Medium | Abstract tile source, easy swap |
| Large map rendering OOM | Medium | Medium | Tile-based rendering, limit extent |
| Browser compatibility | Low | Low | Test in Chrome, Firefox, Safari |
| Folium/template integration | Medium | Medium | Test Folium HTML injection into Jinja templates early in R2.5 [ADDED BY REQUIREMENTS REVIEW] |

### 8.2 Schedule Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Scope creep (more templates) | High | Medium | Strict feature freeze after spec approval |
| Design iteration delays | Medium | Medium | Use design system as single source of truth |
| Integration complexity | Medium | Medium | Early integration testing, not waterfall |

### 8.3 Quality Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Accessibility violations | Medium | High | Automated testing in CI |
| Print/screen parity issues | Medium | Medium | Regular cross-format testing |
| Performance degradation | Low | Medium | Performance budgets, monitoring |

---

## 9. Open Questions

### 9.1 Design Decisions Needed

| Question | Options | Recommendation | Status |
|----------|---------|----------------|--------|
| PDF library choice | WeasyPrint vs ReportLab vs Paged.js | WeasyPrint (CSS-based) | PENDING |
| Interactive map library | Leaflet vs Mapbox GL vs OpenLayers | Leaflet (lightweight, free) | PENDING |
| Infrastructure data source | HIFLD vs OSM vs both | HIFLD primary, OSM fallback | PENDING |
| Before/after implementation | Slider vs toggle vs animation | Slider (most intuitive) | PENDING |

### 9.2 Clarifications Needed

1. **Q:** Should reports support event comparison (Hurricane Ian vs Irma)?
   - **A:** Out of scope for this epic. Future enhancement.

2. **Q:** What languages should reports support?
   - **A:** English only for v1. Multi-language is future phase.

3. **Q:** Should we include real-time data (NWS alerts)?
   - **A:** Out of scope. Reports are point-in-time snapshots.

4. **Q:** How should we handle areas with no Census data (territories)?
   - **A:** Graceful degradation - show "Population data unavailable" and proceed.

5. **Q:** Should emergency resources be editable by users?
   - **A:** Configuration file only for v1. Admin UI is future work.

### 9.3 Technical Clarifications Needed

1. **Q:** Where should report templates be stored?
   - **Proposed:** `agents/reporting/templates/`

2. **Q:** Should maps be embedded or linked?
   - **Proposed:** Embedded for web, embedded-as-image for PDF

3. **Q:** What's the maximum AOI size for population estimates?
   - **Proposed:** 10,000 km^2 (beyond this, generalize to county level)

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2026-01-26 | Product Queen | Initial draft |
| 0.2.0 | 2026-01-26 | Requirements Reviewer | Requirements review - added Testing Strategy, Deployment Strategy, clarified 6 ACs, updated dependencies |

---

## Appendix A: Template Mockup Descriptions

See `docs/design/REPORT_VISUAL_SPEC.md` Section 5 for detailed mockup descriptions including:
- Executive Summary - Hurricane Ian (pixel-level spec)
- Full Report - Cover Page
- Flood Extent Map - Full Page

## Appendix B: Color Accessibility Verification

See `docs/design/DESIGN_SYSTEM.md` Appendix A for:
- Flood palette contrast ratios
- Colorblind simulation results
- WCAG compliance verification

## Appendix C: Cartographic Standards Reference

See `docs/design/CARTOGRAPHIC_SPEC.md` for:
- CRS recommendations by scale
- Map furniture specifications
- Before/after comparison requirements
- Print vs digital considerations

---

---

## Product Queen Sign-Off

**Decision:** APPROVED FOR IMPLEMENTATION

This specification has been reviewed by the Requirements Reviewer and all major recommendations have been incorporated:

1. **6 Acceptance Criteria Clarified** - Updated per Section 3.1 recommendations
2. **Testing Strategy Added** - Section 6 covers test environment, accessibility testing, and performance benchmarks
3. **Deployment Strategy Added** - Section 7 covers integration points and rollout plan
4. **Dependency Graph Updated** - R2.4 -> R2.5 soft dependency documented
5. **Folium Integration Risk Added** - Risk register updated in Section 8.1
6. **Stretch Goals Marked** - R2.1-AC10 (dark mode) marked as [STRETCH]

**Implementation may begin immediately** with the following notes:

- **Start with R2.1 (Design System)** - Foundation for all other features
- **R2.4 (Data Integrations) can run in parallel** - No dependencies
- **Resolve open questions in Section 9** before reaching those implementation points

**Estimated Timeline:** 8 weeks (as specified in Section 5)

---

*Specification approved by Product Queen on 2026-01-26*
