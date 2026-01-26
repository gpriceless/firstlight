# FirstLight Cartographic Specification

**Version:** 1.0.0
**Last Updated:** 2026-01-26
**Companion Documents:** `DESIGN_SYSTEM.md`, `REPORT_VISUAL_SPEC.md`

---

## Overview

This specification defines cartographic standards for all FirstLight map outputs. It bridges professional cartographic principles with the visual framework established in the Design System, ensuring all map products meet both technical accuracy requirements and accessibility standards for emergency response communication.

**Guiding Principles:**

1. **Accuracy first** - Projections and scales must be appropriate for the use case
2. **Clarity over decoration** - Every map element serves communication
3. **Accessibility** - Colorblind-safe, print-ready, screen reader describable
4. **Consistency** - All maps follow the same visual language
5. **Actionability** - Maps enable decisions, not just display data

---

## 1. Projection and CRS Recommendations

### 1.1 Scale-Appropriate Projections

| Scale | Coverage | Recommended CRS | EPSG | Rationale |
|-------|----------|-----------------|------|-----------|
| **Local** | City/County (<100 km) | UTM Zone (local) | 326xx/327xx | Minimal distortion, metric units |
| **Regional** | Multi-county/State | State Plane or UTM | Varies | Familiar to local agencies |
| **National** | CONUS | Albers Equal Area Conic | 5070 | Equal-area for thematic mapping |
| **Web Display** | Any | Web Mercator | 3857 | Tile compatibility, familiar to users |
| **Data Exchange** | Any | WGS 84 | 4326 | Universal interoperability |

### 1.2 Projection Selection by Use Case

#### Emergency Response Maps (Primary Use Case)

**Local Incident Maps (Fort Myers flood, single city fire):**
```
Recommended: UTM Zone (determined by centroid)
- Fort Myers, FL: UTM Zone 17N (EPSG:32617)
- Los Angeles, CA: UTM Zone 11N (EPSG:32611)
- Houston, TX: UTM Zone 15N (EPSG:32615)

Rationale:
- Metric units for accurate area/distance measurements
- Minimal distortion within zone (<0.04% at center)
- Familiar to surveyors and field teams
```

**State/Regional Response Maps:**
```
Recommended: State Plane Coordinate System or UTM
- Florida: Florida State Plane West (EPSG:2237) for Lee County
- California: California State Plane Zone III (EPSG:2227)
- General: UTM zone covering majority of area

Rationale:
- State agencies often use State Plane
- Feet units may be preferred by some US agencies
- Better area preservation than Web Mercator
```

**National Overview Maps:**
```
Recommended: Albers Equal Area Conic (EPSG:5070)
- Also known as NAD83 / Conus Albers

Rationale:
- Equal-area projection preserves relative sizes
- Standard for USGS and many federal agencies
- Appropriate for choropleth and thematic maps
```

#### Web/Interactive Maps

```
Required: Web Mercator (EPSG:3857)

Rationale:
- All major tile providers use Web Mercator
- Users expect north-up orientation
- Seamless zoom from global to local

Caveat:
- NEVER use Web Mercator for area calculations
- NEVER use Web Mercator for distance measurements at high latitudes
- Always reproject to equal-area for analysis
```

#### Data Storage and Exchange

```
Required: WGS 84 (EPSG:4326)

Rationale:
- GeoJSON specification requires WGS 84
- Maximum interoperability
- GPS native coordinate system

Implementation:
- Store all vector data in WGS 84
- Transform to display CRS at render time
- Document original CRS if different
```

### 1.3 UTM Zone Quick Reference (Common US Locations)

| Region | UTM Zone | EPSG (North) | Cities |
|--------|----------|--------------|--------|
| Pacific Northwest | 10N | 32610 | Seattle, Portland |
| California | 10N/11N | 32610/32611 | San Francisco (10N), LA (11N) |
| Mountain West | 12N/13N | 32612/32613 | Phoenix (12N), Denver (13N) |
| Central | 14N/15N | 32614/32615 | Dallas (14N), Houston (15N) |
| Midwest | 15N/16N | 32615/32616 | Chicago (16N), Minneapolis (15N) |
| Southeast | 16N/17N | 32616/32617 | Atlanta (16N), Miami (17N) |
| Northeast | 18N/19N | 32618/32619 | NYC (18N), Boston (19N) |
| Florida | 17N | 32617 | Fort Myers, Tampa, Jacksonville |
| Puerto Rico | 19N | 32619 | San Juan |
| Hawaii | 4N/5N | 32604/32605 | Honolulu (4N) |
| Alaska | 4N-10N | Varies | Anchorage (6N) |

### 1.4 CRS Transformation Guidelines

**Critical Rules:**

1. **Always specify axis order explicitly**
   ```python
   from pyproj import Transformer
   transformer = Transformer.from_crs(
       "EPSG:4326",
       "EPSG:32617",
       always_xy=True  # CRITICAL: lon, lat order
   )
   x, y = transformer.transform(lon, lat)
   ```

2. **Verify transformation with known points**
   - Before processing, transform a known landmark
   - Verify result against expected coordinates
   - Example: Fort Myers City Hall at (-81.8724, 26.6406) WGS84

3. **Document transformations in metadata**
   ```json
   {
     "source_crs": "EPSG:4326",
     "display_crs": "EPSG:32617",
     "transformation_method": "pyproj 3.x default",
     "accuracy_m": 0.01
   }
   ```

4. **Handle datum shifts properly**
   - NAD27 to NAD83 requires NADCON grid shift
   - Do not assume simple reprojection
   - Use authoritative transformation pipelines

### 1.5 Common Pitfalls to Avoid

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| **Mixed CRS in overlay** | Features offset from basemap | Verify all layers in same CRS |
| **Axis order confusion** | Features in wrong hemisphere | Use `always_xy=True` consistently |
| **Web Mercator for analysis** | Incorrect area calculations | Reproject to equal-area for analysis |
| **Missing CRS definition** | Features at Null Island (0,0) | Always define CRS in output files |
| **Degree-based buffers** | Elliptical buffers | Project to metric CRS before buffering |

---

## 2. Base Map Design

### 2.1 Recommended Tile Sources

| Source | Style | URL Pattern | Best For | Attribution |
|--------|-------|-------------|----------|-------------|
| **CartoDB Positron** | Light, muted | `cartodb.com/light_all` | Data overlays | CartoDB |
| **CartoDB Voyager** | Subtle labels | `cartodb.com/voyager` | General use | CartoDB |
| **Stamen Toner Lite** | Grayscale | `stamen.com/toner-lite` | High-contrast data | Stamen |
| **OpenStreetMap** | Standard | `tile.openstreetmap.org` | Detailed context | OSM Contributors |
| **ESRI World Imagery** | Satellite | `services.arcgisonline.com` | Before/after | Esri, Maxar |
| **ESRI World Topo** | Topographic | `services.arcgisonline.com` | Terrain context | Esri |

**Primary Recommendation:** CartoDB Positron for flood maps, CartoDB Voyager for general use.

### 2.2 Custom Base Map Styling

When using vector tiles or custom styling, apply the Design System base map palette:

| Layer | Fill Color | Stroke Color | Stroke Width | Opacity |
|-------|------------|--------------|--------------|---------|
| **Land** | `#EDF2F7` | None | - | 100% |
| **Water (existing)** | `#E2E8F0` | `#CBD5E0` | 0.5px | 100% |
| **Urban areas** | `#CBD5E0` | None | - | 80% |
| **Parks/green** | `#E2E8F0` | None | - | 50% |
| **Buildings** | `#CBD5E0` | `#A0AEC0` | 0.5px | 60% |
| **Highways** | `#A0AEC0` | None | 3px | 100% |
| **Major roads** | `#A0AEC0` | None | 1.5px | 100% |
| **Minor roads** | `#CBD5E0` | None | 0.75px | 70% |
| **Boundaries** | None | `#718096` | 1px dashed | 100% |

### 2.3 Label Hierarchy

| Feature Type | Font Size | Weight | Color | Halo | Show at Zoom |
|--------------|-----------|--------|-------|------|--------------|
| **State name** | 16px | 400 | `#718096` | None | 5-7 |
| **County name** | 14px | 500 | `#4A5568` | 1px white | 7-9 |
| **City (major)** | 14px | 700 | `#1A202C` | 2px white | 8-12 |
| **City (minor)** | 12px | 600 | `#2D3748` | 1.5px white | 10-14 |
| **Neighborhood** | 11px | 500 | `#4A5568` | 1px white | 12-16 |
| **Highway number** | 10px | 600 | `#4A5568` | 1px white | 10-16 |
| **Street name** | 10px | 400 | `#4A5568` | 1px white | 14-18 |
| **Water body** | 11px | 400 italic | `#2B6CB0` | 1px white | 10-16 |

### 2.4 Zoom Level Guidelines

| Zoom | Ground Resolution | Coverage | Show |
|------|-------------------|----------|------|
| 6 | ~2.5 km/px | State | State boundaries, major cities |
| 8 | ~600 m/px | Region | Counties, highways, cities |
| 10 | ~150 m/px | Metro | Major roads, neighborhoods |
| 12 | ~40 m/px | City | All roads, points of interest |
| 14 | ~10 m/px | Neighborhood | Buildings, street names |
| 16 | ~2.5 m/px | Block | Building outlines, addresses |

**Flood Map Defaults:**
- Overview map: Zoom 10-11
- Detail map: Zoom 13-14
- Building-level: Zoom 15-16

### 2.5 Attribution Requirements

All maps MUST include appropriate attribution. Place in bottom-right corner in `text-xs` (12px), `#A0AEC0`:

```
Base map: (C) OpenStreetMap contributors
Imagery: (C) Esri, Maxar, GeoEye
Tiles: (C) CartoDB
```

---

## 3. Thematic Layer Specifications

### 3.1 Flood Extent Layer Stack

Render layers in this order (bottom to top):

| Order | Layer | Purpose |
|-------|-------|---------|
| 1 | Base map tiles | Context |
| 2 | Pre-existing water mask | Distinguish new vs. permanent water |
| 3 | **Flood extent (primary)** | Main data layer |
| 4 | Uncertainty overlay | Show confidence |
| 5 | Roads | Transportation context |
| 6 | Infrastructure icons | Critical facilities |
| 7 | Labels | Place names |
| 8 | Map furniture | Legend, scale, north arrow |

### 3.2 Flood Extent Rendering

**Solid Fill (Primary Method):**

| Severity | Color (from Design System) | Opacity | Pattern |
|----------|---------------------------|---------|---------|
| Minor (0-0.3m) | `#90CDF4` | 70% | None |
| Moderate (0.3-1m) | `#4299E1` | 75% | None |
| Significant (1-2m) | `#2B6CB0` | 80% | None |
| Severe (2-3m) | `#2C5282` | 85% | None |
| Extreme (>3m) | `#1A365D` | 90% | None |

**Edge Treatment:**

| Confidence Level | Edge Style |
|------------------|------------|
| High (>85%) | Solid edge, no blur |
| Medium (60-85%) | 1px feathered edge |
| Low (<60%) | 2px feathered edge, dashed outline |

**Uncertainty Zones:**

```css
/* CSS for uncertainty overlay */
.uncertainty-high {
  fill: var(--color-conf-high);
  fill-opacity: 0.1;
  stroke: var(--color-conf-high);
  stroke-width: 1px;
  stroke-dasharray: none;
}

.uncertainty-low {
  fill: var(--color-conf-low);
  fill-opacity: 0.2;
  stroke: var(--color-conf-low);
  stroke-width: 1px;
  stroke-dasharray: 4,4;
}
```

### 3.3 Depth Classification Breaks

When depth data is available, use these classification breaks:

| Class | Depth Range | Color | Label | Human Context |
|-------|-------------|-------|-------|---------------|
| 1 | 0 - 0.15 m | `#BEE3F8` | Shallow | Ankle deep |
| 2 | 0.15 - 0.3 m | `#90CDF4` | Minor | Knee deep |
| 3 | 0.3 - 0.6 m | `#63B3ED` | Moderate | Thigh deep |
| 4 | 0.6 - 1.0 m | `#4299E1` | Significant | Waist deep |
| 5 | 1.0 - 1.5 m | `#3182CE` | Deep | Chest deep |
| 6 | 1.5 - 2.0 m | `#2B6CB0` | Very deep | Over head |
| 7 | 2.0 - 3.0 m | `#2C5282` | Severe | First floor |
| 8 | > 3.0 m | `#1A365D` | Extreme | Multi-story |

**Rationale:** Breaks align with pedestrian safety thresholds and building impact levels.

### 3.4 Infrastructure Overlay Specifications

#### Icon Sizes by Zoom Level

| Zoom | Icon Size | Label | Clustering |
|------|-----------|-------|------------|
| 8-10 | 16px | None | Yes, 50px radius |
| 11-12 | 20px | Major only | Yes, 30px radius |
| 13-14 | 24px | All visible | No |
| 15+ | 28px | All with details | No |

#### Icon Specifications

| Facility | Shape | Fill | Icon | Stroke |
|----------|-------|------|------|--------|
| **Hospital** | Circle | `#E53E3E` | White cross | 2px white |
| **School** | Circle | `#3182CE` | White cap | 2px white |
| **Fire Station** | Diamond | `#F97316` | White flame | 2px white |
| **Police** | Shield | `#2C5282` | White star | 2px white |
| **Shelter** | House | `#38A169` | White house | 2px white |
| **Power** | Square | `#D69E2E` | White bolt | 2px white |
| **Water Treatment** | Circle | `#4299E1` | White drop | 2px white |

#### Status Indication

| Status | Visual Treatment |
|--------|------------------|
| In flood zone | Full color, pulsing ring |
| Adjacent (<500m) | Full color, orange outline |
| Outside flood zone | Grayscale, no outline |
| Status unknown | Question mark overlay |

### 3.5 Label Placement Rules

**Point Features:**
1. Prefer right of icon
2. Fallback: above, below, left (in order)
3. Never overlap flood extent
4. Use leader line if displaced >20px

**Line Features (roads, rivers):**
1. Center along line
2. Follow curve orientation
3. Repeat every 200px on long features
4. Skip labels where overlapping data

**Area Features:**
1. Center of largest polygon
2. Curved text for large features
3. Multiple labels for complex shapes

---

## 4. Map Furniture Standards

### 4.1 Scale Bar

**Style:** Alternating black and white segments

```
|====|====|====|====|
0    1    2    3    4 km
```

**Specifications:**

| Property | Value |
|----------|-------|
| Position | Bottom-left, 16px from edge |
| Height | 8px |
| Segment count | 4-5 |
| Tick height | 12px |
| Label font | `text-xs` (12px), weight 500, `#4A5568` |
| Units | km for regional, m for local |
| Background | Semi-transparent white (`rgba(255,255,255,0.8)`) |
| Padding | 8px |
| Border radius | 4px |

**Unit Selection:**

| Map extent | Primary unit | Secondary unit |
|------------|--------------|----------------|
| > 50 km | kilometers | miles |
| 5-50 km | kilometers | none |
| 1-5 km | meters | feet |
| < 1 km | meters | none |

**Implementation (matplotlib):**

```python
from matplotlib_scalebar.scalebar import ScaleBar

scalebar = ScaleBar(
    1,  # 1 pixel = 1 meter (adjust for CRS)
    location='lower left',
    length_fraction=0.25,
    height_fraction=0.01,
    pad=0.5,
    color='#4A5568',
    box_alpha=0.8,
    font_properties={'size': 12, 'weight': 500}
)
ax.add_artist(scalebar)
```

### 4.2 North Arrow

**Style:** Simple arrow with N indicator

**Specifications:**

| Property | Value |
|----------|-------|
| Position | Top-right, 16px from edge |
| Size | 32px height |
| Arrow color | `#4A5568` |
| N label | `text-sm` (14px), weight 600, `#2D3748` |
| Background | Semi-transparent white (`rgba(255,255,255,0.8)`) |
| Padding | 8px |
| Border radius | 4px |

**When Required:**
- Always on print maps
- Always on maps with rotation
- Optional on web maps (north usually up)
- Not needed on small thumbnails

**SVG Template:**

```svg
<svg width="32" height="48" viewBox="0 0 32 48">
  <path d="M16 0 L24 20 L16 16 L8 20 Z" fill="#4A5568"/>
  <path d="M16 16 L24 20 L16 48 L8 20 Z" fill="#A0AEC0"/>
  <text x="16" y="20" text-anchor="middle"
        font-size="12" font-weight="600" fill="#2D3748">N</text>
</svg>
```

### 4.3 Legend Design

**Continuous Scale (flood depth):**

```
+----------------------------------+
|  FLOOD DEPTH                     |
|  --------------------------------|
|  [gradient bar ░░▒▒▓▓██]         |
|  0 m      1.5 m      3.0+ m      |
|                                  |
|  Data confidence indicated by    |
|  edge sharpness                  |
+----------------------------------+
```

**Categorical (severity levels):**

```
+----------------------------------+
|  FLOOD SEVERITY                  |
|  --------------------------------|
|  [█] Extreme (>3 m)              |
|  [█] Severe (2-3 m)              |
|  [█] Significant (1-2 m)         |
|  [█] Moderate (0.3-1 m)          |
|  [█] Minor (<0.3 m)              |
|  --------------------------------|
|  INFRASTRUCTURE                  |
|  --------------------------------|
|  [+] Hospital                    |
|  [cap] School                    |
|  [house] Shelter                 |
+----------------------------------+
```

**Specifications:**

| Property | Value |
|----------|-------|
| Position | Bottom-left (after scale bar) or right side |
| Background | White, 95% opacity |
| Border | `1px solid #E2E8F0` |
| Border radius | 8px |
| Padding | 16px |
| Title | `text-sm` (14px), weight 600, `#2D3748`, uppercase |
| Divider | `1px solid #E2E8F0`, 8px margin |
| Item gap | 8px |
| Symbol size | 14px |
| Label | `text-xs` (12px), weight 400, `#4A5568` |
| Max width | 200px |

### 4.4 Title Block

**Layout:**

```
+--------------------------------------------------+
|  FLOOD EXTENT MAP                                |
|  Hurricane Ian | Fort Myers, FL | Sep 28, 2022   |
+--------------------------------------------------+
```

**Specifications:**

| Element | Specification |
|---------|---------------|
| Main title | `text-2xl` (24px), weight 700, `#1A365D` |
| Subtitle | `text-sm` (14px), weight 400, `#718096` |
| Separator | Bullet character `\u2022` |
| Position | Top-center, outside map frame |
| Gap | 8px between title and subtitle |
| Margin bottom | 16px to map frame |

### 4.5 Source Attribution Block

**Layout:**

```
Data: Sentinel-1 SAR, Sentinel-2 Optical | Copernicus Open Access Hub
Analysis: Sep 29, 2022 | Generated: Jan 26, 2026 | CRS: EPSG:32617
```

**Specifications:**

| Property | Value |
|----------|-------|
| Position | Bottom-center, below map frame |
| Font | `text-xs` (12px), weight 400, `#A0AEC0` |
| Line height | 1.5 |
| Margin top | 8px from map frame |

**Required Information:**

1. Data sources (satellite, date)
2. Processing/analysis date
3. Generation timestamp
4. Coordinate Reference System
5. Confidence level (if applicable)

### 4.6 Inset/Locator Map

**When Required:**
- Local maps (city/county scale)
- Isolated areas (islands, remote locations)
- Multi-location reports

**Specifications:**

| Property | Value |
|----------|-------|
| Position | Top-left corner, inside map frame |
| Size | 80-120px square |
| Border | `2px solid #718096` |
| Background | Light base map |
| Extent indicator | Red rectangle showing main map extent |
| Context | State or region outline |
| Offset from edge | 16px |

**Implementation:**

```python
# Create inset axes
inset_ax = fig.add_axes([0.12, 0.65, 0.2, 0.2])  # [left, bottom, width, height]
inset_ax.set_extent([-88, -79, 24, 32], crs=ccrs.PlateCarree())  # Florida
inset_ax.add_feature(cfeature.STATES, edgecolor='#718096', linewidth=0.5)
inset_ax.add_patch(Rectangle(
    (main_west, main_south),
    main_east - main_west,
    main_north - main_south,
    fill=False, edgecolor='#E53E3E', linewidth=2
))
```

---

## 5. Before/After Comparison Cartography

### 5.1 Critical Requirements

**Alignment Checklist:**
- [ ] Identical geographic extent (bounds)
- [ ] Identical pixel resolution
- [ ] Identical projection/CRS
- [ ] Identical map dimensions
- [ ] North arrow position matches
- [ ] Scale bar position matches
- [ ] Legend position matches (if present)

### 5.2 Side-by-Side Layout

```
+---------------------------+  +---------------------------+
|   BEFORE                  |  |   AFTER                   |
|   September 25, 2022      |  |   September 29, 2022      |
|                           |  |                           |
|   +-------------------+   |  |   +-------------------+   |
|   |                   |   |  |   |                   |   |
|   |   [Pre-event      |   |  |   |   [Post-event     |   |
|   |    satellite      |   |  |   |    with flood     |   |
|   |    imagery]       |   |  |   |    overlay]       |   |
|   |                   |   |  |   |                   |   |
|   +-------------------+   |  |   +-------------------+   |
|                           |  |                           |
+---------------------------+  +---------------------------+

              [Shared legend]     [Shared scale bar]
```

**Specifications:**

| Element | Specification |
|---------|---------------|
| Container gap | 16px |
| Frame aspect ratio | Must match (e.g., 4:3) |
| Date label position | Top-left, inside frame |
| Date label background | `rgba(0,0,0,0.7)` |
| Date label text | `text-sm` (14px), weight 600, white |
| Date label padding | 8px 12px |
| Date label border radius | 0 0 4px 0 (bottom-right only) |

### 5.3 Slider/Swipe Layout (Interactive)

**Implementation Approach:**

```html
<div class="comparison-container">
  <div class="comparison-image before">
    <img src="before.png" alt="Pre-event satellite imagery, September 25, 2022">
    <span class="date-label">BEFORE - Sep 25, 2022</span>
  </div>
  <div class="comparison-image after">
    <img src="after.png" alt="Post-event imagery showing flood extent, September 29, 2022">
    <span class="date-label">AFTER - Sep 29, 2022</span>
  </div>
  <div class="comparison-slider">
    <div class="slider-handle"></div>
  </div>
</div>
```

**CSS Specifications:**

```css
.comparison-container {
  position: relative;
  width: 100%;
  aspect-ratio: 16/9;
  overflow: hidden;
}

.comparison-slider {
  position: absolute;
  top: 0;
  left: 50%;
  width: 4px;
  height: 100%;
  background: var(--color-brand-sky);
  cursor: ew-resize;
}

.slider-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 40px;
  height: 40px;
  background: white;
  border: 2px solid var(--color-brand-sky);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

### 5.4 Animated GIF Specifications

**When to Use:**
- Social media sharing
- Email attachments
- Situations without JavaScript

**Specifications:**

| Property | Value |
|----------|-------|
| Frame 1 duration | 2500ms |
| Frame 2 duration | 2500ms |
| Transition | None (hard cut) |
| Date label | Prominent, top-left |
| Loop | Infinite |
| File size target | < 5 MB |
| Resolution | 1200 x 800px for full, 600 x 400px for thumbnail |

**Frame Requirements:**
- Frame 1: Pre-event imagery with "BEFORE" label
- Frame 2: Post-event with flood overlay and "AFTER" label
- Both frames must have identical extent, scale, and base elements

---

## 6. Data Visualization on Maps

### 6.1 Choropleth vs. Graduated Symbols

**Decision Tree:**

```
Is data bounded by geographic units (counties, ZIP codes)?
  |
  +-- YES --> Do units have similar sizes?
  |             |
  |             +-- YES --> CHOROPLETH
  |             |
  |             +-- NO --> Consider CARTOGRAM or GRADUATED SYMBOLS
  |
  +-- NO --> Is data associated with point locations?
              |
              +-- YES --> GRADUATED SYMBOLS or PROPORTIONAL SYMBOLS
              |
              +-- NO --> Consider ISOLINE or HEAT MAP
```

**For Flood Data:**
- Flood extent: Direct polygon rendering (not choropleth)
- Population affected by county: Choropleth
- Facilities affected: Graduated symbols
- Flood depth continuous: Classified or continuous color ramp

### 6.2 Classification Methods

| Method | Best For | Avoid When |
|--------|----------|------------|
| **Natural Breaks (Jenks)** | Varied data, identifying clusters | Very skewed distributions |
| **Quantile** | Equal number per class, rankings | Large variation in class ranges |
| **Equal Interval** | Uniform distributions | Highly skewed data |
| **Standard Deviation** | Statistical analysis | Non-normal distributions |
| **Manual** | Meaningful thresholds (e.g., flood depths) | Unknown data characteristics |

**Recommended for Flood Depth:** Manual classification with human-meaningful breaks (see Section 3.3).

### 6.3 Number of Classes

| Data complexity | Recommended classes |
|-----------------|---------------------|
| Simple binary | 2 (yes/no) |
| Basic severity | 3-4 |
| Standard thematic | 5-6 |
| Detailed analysis | 7-8 (maximum) |

**Rationale:** Human perception limits distinguish ~7 color steps reliably. More classes reduce map readability.

### 6.4 Label Density Guidelines

| Map scale | Max labels per cm^2 | Strategy |
|-----------|---------------------|----------|
| Overview (1:500,000) | 0.5 | Major features only |
| Regional (1:100,000) | 1.5 | Cities, highways |
| Local (1:25,000) | 3.0 | All named features |
| Detail (1:10,000) | 5.0 | Streets, addresses |

**Label Priority (descending):**
1. Affected critical infrastructure (hospitals, shelters)
2. Major roads and highways
3. City/town names
4. Neighborhood names
5. Minor roads
6. Minor landmarks

---

## 7. Print vs. Digital Considerations

### 7.1 Resolution Requirements

| Output | Resolution | Color Space | Notes |
|--------|------------|-------------|-------|
| Screen (web) | 144 DPI | sRGB | Standard retina |
| Screen (presentation) | 150 DPI | sRGB | Projection quality |
| Print (report) | 300 DPI | sRGB or Adobe RGB | Standard print |
| Print (poster) | 150-200 DPI | Adobe RGB | Large format |
| Print (professional) | 300+ DPI | CMYK | Offset printing |

### 7.2 Color Space Conversion

**CMYK Equivalents for Primary Flood Colors:**

| Severity | RGB Hex | CMYK (approximate) |
|----------|---------|-------------------|
| Minor | `#90CDF4` | C:42 M:17 Y:0 K:4 |
| Moderate | `#4299E1` | C:71 M:33 Y:0 K:12 |
| Significant | `#2B6CB0` | C:76 M:40 Y:0 K:31 |
| Severe | `#2C5282` | C:67 M:38 Y:0 K:49 |
| Extreme | `#1A365D` | C:73 M:43 Y:0 K:64 |

**Note:** Professional print production should use Pantone or verified CMYK swatches. These approximations are for reference.

### 7.3 Font Size Minimums

| Element | Screen minimum | Print minimum |
|---------|----------------|---------------|
| Body text | 14px | 10pt (13.3px) |
| Captions | 12px | 8pt (10.7px) |
| Labels on map | 10px | 7pt (9.3px) |
| Fine print | 10px | 6pt (8px) |

### 7.4 Pattern Overlays for B&W Printing

When color printing is not available, use pattern overlays:

| Severity | Pattern | Line weight | Spacing |
|----------|---------|-------------|---------|
| Minor | Horizontal lines | 0.5pt | 6pt |
| Moderate | Diagonal lines (45 deg) | 0.5pt | 4pt |
| Significant | Cross-hatch | 0.5pt | 3pt |
| Severe | Dense dots | - | 2pt |
| Extreme | Solid fill | - | - |

### 7.5 Bleed and Margin Requirements

**US Letter (8.5" x 11"):**
- Bleed: 0.125" (3mm) beyond trim on all sides
- Safe area: 0.25" (6mm) inside trim
- Margins: 0.5" (12.7mm) minimum for content

**A4 (210mm x 297mm):**
- Bleed: 3mm beyond trim on all sides
- Safe area: 5mm inside trim
- Margins: 15mm minimum for content

---

## 8. Implementation Recommendations

### 8.1 Python Library Stack

**Core Libraries:**

| Library | Purpose | Install |
|---------|---------|---------|
| `geopandas` | Vector data manipulation | `pip install geopandas` |
| `rasterio` | Raster I/O | `pip install rasterio` |
| `pyproj` | CRS transformations | `pip install pyproj` |
| `shapely` | Geometry operations | `pip install shapely` |

**Visualization Libraries:**

| Library | Purpose | Best For |
|---------|---------|----------|
| `matplotlib` + `cartopy` | Static maps | Print-quality, custom styling |
| `folium` | Interactive web maps | HTML export, quick prototypes |
| `contextily` | Basemap tiles | Adding base maps to matplotlib |
| `geoviews` + `holoviews` | Interactive exploration | Data analysis |
| `pydeck` | 3D visualization | Large-scale, impressive visuals |
| `leafmap` | Interactive mapping | Jupyter notebooks |

**Recommended Stack for FirstLight:**

```python
# Static maps (print-quality)
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar

# Interactive maps (web export)
import folium
from folium.plugins import MousePosition, MeasureControl

# Data handling
import geopandas as gpd
import rasterio
from pyproj import Transformer
```

### 8.2 Folium Implementation Example

```python
import folium
from folium.plugins import MousePosition

def create_flood_map(
    flood_gdf: gpd.GeoDataFrame,
    infrastructure_gdf: gpd.GeoDataFrame,
    bounds: tuple,
    title: str = "Flood Extent Map"
) -> folium.Map:
    """Create interactive flood map with infrastructure overlay."""

    # Calculate center
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Create map with CartoDB Positron base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron',
        attr='CartoDB'
    )

    # Add flood extent with severity styling
    severity_colors = {
        'minor': '#90CDF4',
        'moderate': '#4299E1',
        'significant': '#2B6CB0',
        'severe': '#2C5282',
        'extreme': '#1A365D'
    }

    for _, row in flood_gdf.iterrows():
        severity = row.get('severity', 'moderate')
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda x, sev=severity: {
                'fillColor': severity_colors.get(sev, '#4299E1'),
                'fillOpacity': 0.7,
                'color': '#2D3748',
                'weight': 1
            }
        ).add_to(m)

    # Add infrastructure icons
    icon_map = {
        'hospital': ('plus', 'red'),
        'school': ('graduation-cap', 'blue'),
        'shelter': ('home', 'green'),
        'fire_station': ('fire-extinguisher', 'orange')
    }

    for _, row in infrastructure_gdf.iterrows():
        fac_type = row.get('type', 'other')
        icon_name, color = icon_map.get(fac_type, ('info', 'gray'))

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=f"<b>{row.get('name', 'Facility')}</b><br>{row.get('address', '')}",
            icon=folium.Icon(icon=icon_name, prefix='fa', color=color)
        ).add_to(m)

    # Add mouse position display
    MousePosition().add_to(m)

    # Add scale control
    folium.plugins.MeasureControl(position='bottomleft').add_to(m)

    return m
```

### 8.3 Matplotlib/Cartopy Implementation Example

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import geopandas as gpd

def create_print_flood_map(
    flood_gdf: gpd.GeoDataFrame,
    bounds: tuple,
    output_path: str,
    title: str = "Flood Extent Map",
    dpi: int = 300
):
    """Create print-quality flood map."""

    # Set up figure with proper projection
    fig = plt.figure(figsize=(11, 8.5))  # Letter landscape

    # Use UTM for local accuracy
    utm_zone = int((bounds[0] + bounds[2]) / 2 + 180) // 6 + 1
    proj = ccrs.UTM(zone=utm_zone, southern_hemisphere=bounds[1] < 0)

    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

    # Add basemap
    ctx.add_basemap(
        ax,
        crs=proj,
        source=ctx.providers.CartoDB.Positron,
        zoom=12
    )

    # Reproject flood data to map CRS
    flood_proj = flood_gdf.to_crs(proj.proj4_init)

    # Plot flood extent with severity colors
    severity_colors = {
        'minor': '#90CDF4',
        'moderate': '#4299E1',
        'significant': '#2B6CB0',
        'severe': '#2C5282',
        'extreme': '#1A365D'
    }

    for severity, color in severity_colors.items():
        subset = flood_proj[flood_proj['severity'] == severity]
        if not subset.empty:
            subset.plot(
                ax=ax,
                color=color,
                alpha=0.75,
                edgecolor='#2D3748',
                linewidth=0.5,
                label=severity.capitalize()
            )

    # Add scale bar
    scalebar = ScaleBar(
        1, 'm',
        location='lower left',
        length_fraction=0.2,
        height_fraction=0.01,
        pad=0.5,
        color='#4A5568',
        box_alpha=0.8,
        font_properties={'size': 10}
    )
    ax.add_artist(scalebar)

    # Add north arrow
    ax.annotate(
        'N', xy=(0.95, 0.95), xycoords='axes fraction',
        fontsize=14, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1A365D', pad=20)

    # Add legend
    ax.legend(
        loc='lower right',
        title='Flood Severity',
        framealpha=0.9,
        fontsize=9
    )

    # Save
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
```

### 8.4 Current FirstLight Integration

**Recommended Additions to Codebase:**

1. **New module:** `agents/reporting/cartography.py`
   - Projection selection utilities
   - Map generation functions
   - Legend/scale bar helpers

2. **Update:** `agents/reporting/products.py`
   - Add map generation methods
   - Integrate with existing thumbnail generation

3. **New templates:** `agents/reporting/templates/`
   - HTML templates for interactive maps
   - CSS for map styling consistent with Design System

4. **Configuration:** `core/config/cartography.yaml`
   - Default projections by region
   - Color palettes from Design System
   - Scale bar/legend preferences

### 8.5 JavaScript Library Options

For web-based interactive maps:

| Library | Pros | Cons | Best For |
|---------|------|------|----------|
| **Leaflet** | Simple, lightweight, plugins | Less performant with large data | Quick implementations |
| **MapLibre GL JS** | Vector tiles, smooth rendering | More complex setup | Production web apps |
| **OpenLayers** | Full-featured, standards-compliant | Steeper learning curve | GIS-heavy applications |
| **Deck.gl** | Large data visualization | Requires WebGL | Big data, 3D |

**Recommendation:** Leaflet for initial implementation, MapLibre for production upgrade.

---

## 9. Example Map Specifications

### 9.1 Hurricane Ian Flood Extent Map (8.5x11 Print)

**Page Setup:**
- Orientation: Landscape
- Margins: 0.5" all sides
- Bleed: 0.125" for professional printing

**Geographic Specifications:**

| Property | Value |
|----------|-------|
| Extent (WGS84) | West: -82.2, South: 26.3, East: -81.7, North: 26.8 |
| Display CRS | EPSG:32617 (UTM Zone 17N) |
| Scale | Approximately 1:75,000 |
| Ground coverage | ~50 km x 55 km |

**Layer Stack:**

| Order | Layer | Source | Styling |
|-------|-------|--------|---------|
| 1 | Basemap | CartoDB Positron tiles | Default |
| 2 | County boundaries | US Census TIGER | `#718096`, 1px dashed |
| 3 | Pre-existing water | OSM water polygons | `#E2E8F0`, 100% |
| 4 | Flood extent | FirstLight analysis | Severity palette, 75% opacity |
| 5 | Major roads | OSM highways | `#A0AEC0`, 2px |
| 6 | Road labels | OSM | 10px, white halo |
| 7 | City labels | Gazetteer | Per hierarchy spec |

**Map Furniture Placement:**

| Element | Position | Size/Style |
|---------|----------|------------|
| Title | Top center | 24px, weight 700, `#1A365D` |
| Subtitle | Below title | 14px, weight 400, `#718096` |
| Legend | Bottom left | 180px wide, per spec |
| Scale bar | Bottom center-left | 4 segments, km |
| North arrow | Top right | 32px, per spec |
| Inset (Florida) | Top left | 100px, red extent box |
| Attribution | Bottom center | 12px, `#A0AEC0` |

**Color Specifications (from Design System):**

```css
--flood-minor: #90CDF4;
--flood-moderate: #4299E1;
--flood-significant: #2B6CB0;
--flood-severe: #2C5282;
--flood-extreme: #1A365D;
```

### 9.2 Infrastructure Impact Map

**Purpose:** Show which critical facilities are in or near flood zone.

**Geographic Specifications:**
- Same extent as flood extent map
- Same CRS and scale

**Layer Stack:**

| Order | Layer | Styling |
|-------|-------|---------|
| 1 | Basemap | CartoDB Voyager (more labels) |
| 2 | Flood extent (simplified) | Single color `#90CDF4`, 50% opacity |
| 3 | 500m flood buffer | Dashed orange outline, no fill |
| 4 | Hospital markers | Red circles, 24px, white cross |
| 5 | School markers | Blue circles, 20px, white cap |
| 6 | Shelter markers | Green houses, 20px |
| 7 | Fire station markers | Orange diamonds, 20px |
| 8 | Callout labels | White boxes with facility names |

**Callout Design:**

```
+----------------------+
| LEE MEMORIAL         |
| HOSPITAL             |
| [IN FLOOD ZONE]      |
+----------------------+
        |
        v (leader line)
      [+]
```

| Property | Value |
|----------|-------|
| Background | White, 90% opacity |
| Border | `1px solid #E2E8F0` |
| Padding | 8px 12px |
| Title | 12px, weight 600, `#2D3748` |
| Status | 10px, weight 500, status color |
| Leader line | 1px, `#718096` |
| Max callouts visible | 10-15 (prioritize in flood zone) |

### 9.3 Before/After Comparison

**Layout:** Side-by-side, landscape orientation

**Frame Specifications:**

| Property | Before Frame | After Frame |
|----------|--------------|-------------|
| Position | Left half | Right half |
| Size | 5" x 3.75" | 5" x 3.75" |
| Gap between | 0.25" | - |
| Date label | "BEFORE - Sep 25, 2022" | "AFTER - Sep 29, 2022" |
| Label position | Top left, inside | Top left, inside |
| Label background | `rgba(0,0,0,0.7)` | `rgba(0,0,0,0.7)` |
| Label text | White, 14px, weight 600 | White, 14px, weight 600 |

**Content:**

| Frame | Content |
|-------|---------|
| Before | Sentinel-2 true color composite, cloud-free |
| After | Sentinel-2 true color with flood overlay |

**Shared Elements (below both frames):**

| Element | Position | Notes |
|---------|----------|-------|
| Legend | Center-left | Flood severity only |
| Scale bar | Center-right | Single, applies to both |
| North arrow | Far right | Single |
| Caption | Center bottom | "Both images at same scale and extent" |

**Alignment Verification Checklist:**

- [ ] Both frames show identical geographic bounds
- [ ] Both frames have identical pixel dimensions
- [ ] Zoom level matches (same scale)
- [ ] Key landmarks visible in both (for reference)
- [ ] Date labels use same formatting
- [ ] No data gaps in comparison areas

---

## 10. Quality Checklist

### 10.1 Pre-Publication QA Checklist

**Projection & Accuracy:**

- [ ] CRS documented in metadata
- [ ] CRS appropriate for map scale and purpose
- [ ] Scale bar accurate (verified with known distance)
- [ ] No obvious misalignment between layers
- [ ] Coordinate display (if shown) uses correct format

**Visual Quality:**

- [ ] All flood extent visible (not cut off by frame)
- [ ] Colors match Design System specification
- [ ] Sufficient contrast between flood and base map
- [ ] Labels legible (minimum sizes met)
- [ ] No label/symbol collisions
- [ ] Appropriate visual hierarchy (data > context)

**Map Furniture:**

- [ ] Title present and descriptive
- [ ] Scale bar present and accurate
- [ ] North arrow present (for print maps)
- [ ] Legend complete and accurate
- [ ] Data source attribution present
- [ ] Date/timestamp present
- [ ] CRS noted (for technical audiences)

**Accessibility:**

- [ ] Colorblind simulation test passed
- [ ] Patterns available for B&W printing
- [ ] Alt text prepared for digital distribution
- [ ] Contrast ratios meet WCAG AA
- [ ] No reliance on color alone for meaning

**Print-Readiness (if applicable):**

- [ ] Resolution >= 300 DPI
- [ ] Bleed added if needed
- [ ] Fonts embedded or outlined
- [ ] CMYK conversion verified (if required)
- [ ] File size acceptable for delivery method

### 10.2 Automated Validation Checks

```python
def validate_map_output(
    output_path: str,
    expected_crs: str,
    expected_bounds: tuple,
    min_dpi: int = 300
) -> dict:
    """Validate map output meets specifications."""

    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check file exists
    if not Path(output_path).exists():
        results['valid'] = False
        results['errors'].append("Output file not found")
        return results

    # Check resolution (for raster outputs)
    if output_path.endswith(('.png', '.tif', '.jpg')):
        from PIL import Image
        with Image.open(output_path) as img:
            dpi = img.info.get('dpi', (72, 72))
            if dpi[0] < min_dpi:
                results['warnings'].append(
                    f"DPI ({dpi[0]}) below minimum ({min_dpi})"
                )

    # Check CRS (for geospatial outputs)
    if output_path.endswith(('.tif', '.geojson', '.gpkg')):
        import rasterio
        import geopandas as gpd

        if output_path.endswith('.tif'):
            with rasterio.open(output_path) as src:
                actual_crs = str(src.crs)
        else:
            gdf = gpd.read_file(output_path)
            actual_crs = str(gdf.crs)

        if actual_crs != expected_crs:
            results['errors'].append(
                f"CRS mismatch: expected {expected_crs}, got {actual_crs}"
            )
            results['valid'] = False

    return results
```

### 10.3 Visual Inspection Checklist

For human review before publication:

**Overall Impression:**
- [ ] Map communicates its message clearly
- [ ] Professional appearance
- [ ] Appropriate level of detail for audience
- [ ] No visual clutter

**Geographic Accuracy:**
- [ ] Flood extent matches expected areas
- [ ] Infrastructure locations correct
- [ ] No obvious errors (features in ocean, etc.)
- [ ] Boundaries align with base map

**Text & Labels:**
- [ ] No spelling errors
- [ ] Dates formatted consistently
- [ ] Numbers formatted appropriately
- [ ] All acronyms explained in legend

**Final Sign-Off:**
- [ ] Technical review complete
- [ ] Editorial review complete
- [ ] Accessibility review complete
- [ ] Approved for distribution

---

## Appendix A: Coordinate System Quick Reference

### A.1 Common EPSG Codes

| EPSG | Name | Use Case |
|------|------|----------|
| 4326 | WGS 84 | GPS, data exchange, GeoJSON |
| 3857 | Web Mercator | Web map tiles |
| 5070 | NAD83 Conus Albers | US national equal-area |
| 4269 | NAD83 | US survey/government |
| 32601-32660 | UTM Zones 1-60 North | Local high-precision |
| 32701-32760 | UTM Zones 1-60 South | Local high-precision |

### A.2 Florida-Specific CRS

| EPSG | Name | Best For |
|------|------|----------|
| 32617 | UTM Zone 17N | Most of Florida |
| 2236 | Florida East (ftUS) | East coast, metric |
| 2237 | Florida West (ftUS) | West coast (Fort Myers) |
| 2238 | Florida North (ftUS) | Panhandle |
| 3086 | Florida Albers | Statewide thematic |

### A.3 PyProj Transformation Examples

```python
from pyproj import Transformer, CRS

# WGS84 to UTM Zone 17N
wgs_to_utm = Transformer.from_crs(
    "EPSG:4326",
    "EPSG:32617",
    always_xy=True
)
x, y = wgs_to_utm.transform(-81.87, 26.64)  # Fort Myers

# UTM to WGS84
utm_to_wgs = Transformer.from_crs(
    "EPSG:32617",
    "EPSG:4326",
    always_xy=True
)
lon, lat = utm_to_wgs.transform(x, y)

# Web Mercator to WGS84
web_to_wgs = Transformer.from_crs(
    "EPSG:3857",
    "EPSG:4326",
    always_xy=True
)
```

---

## Appendix B: Color Palettes (Design System Reference)

### B.1 Flood Severity Palette

```css
:root {
  --flood-none: #F7FAFC;
  --flood-minor: #90CDF4;
  --flood-moderate: #4299E1;
  --flood-significant: #2B6CB0;
  --flood-severe: #2C5282;
  --flood-extreme: #1A365D;
}
```

### B.2 Infrastructure Status

```css
:root {
  --status-in-flood: #E53E3E;
  --status-adjacent: #F97316;
  --status-safe: #38A169;
  --status-unknown: #718096;
}
```

### B.3 Base Map Elements

```css
:root {
  --basemap-land: #EDF2F7;
  --basemap-water: #E2E8F0;
  --basemap-urban: #CBD5E0;
  --basemap-road-major: #A0AEC0;
  --basemap-road-minor: #CBD5E0;
  --basemap-boundary: #718096;
  --basemap-label: #4A5568;
}
```

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **CRS** | Coordinate Reference System - defines how coordinates map to locations on Earth |
| **EPSG** | European Petroleum Survey Group - maintains database of CRS definitions |
| **UTM** | Universal Transverse Mercator - global grid of local projections |
| **Web Mercator** | Projection used by most web mapping services (Google, Bing, OSM) |
| **Choropleth** | Thematic map where areas are shaded by data values |
| **COG** | Cloud-Optimized GeoTIFF - efficient format for web serving |
| **Datum** | Reference model for the shape of the Earth (e.g., WGS84, NAD83) |
| **Projection** | Mathematical transformation from sphere to flat surface |
| **Scale bar** | Map element showing relationship between map distance and ground distance |
| **Tile** | Pre-rendered map image at a specific zoom level and location |
| **Vector tiles** | Tiles containing geometric data rather than rendered images |
| **Jenks breaks** | Classification method that minimizes within-class variance |

---

*FirstLight Cartographic Specification v1.0.0*
*Companion to: DESIGN_SYSTEM.md, REPORT_VISUAL_SPEC.md*
