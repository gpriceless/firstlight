# FirstLight Map Visualization Module

Production-ready map visualization for flood extent, infrastructure overlays, and before/after comparisons.

## Overview

This module provides both **static** (matplotlib-based) and **interactive** (Folium-based) map generation following the FirstLight Cartographic Specification.

## Components

### Base Configuration (`base.py`)

- **MapConfig**: Map generation settings (size, DPI, title, map furniture)
- **MapBounds**: Geographic extent (WGS 84 coordinates)
- **MapType**: Enum for map types (flood_extent, infrastructure, before_after)

### Static Maps (`static_map.py`)

Print-quality PNG/PDF maps using matplotlib/cartopy/contextily.

**Features:**
- CartoDB Positron base maps
- Flood extent visualization with design system colors
- Infrastructure overlays (hospitals, schools, shelters, etc.)
- Scale bar, north arrow, legend
- 300 DPI export for print
- Before/after side-by-side comparisons

**Dependencies:**
- matplotlib (required)
- cartopy (optional, for projection support)
- contextily (optional, for base map tiles)
- matplotlib-scalebar (optional, for scale bars)

### Interactive Maps (`folium_map.py`)

Web-based interactive maps with zoom, pan, and layer controls.

**Features:**
- Folium-based HTML maps
- Interactive tooltips and popups
- Layer toggle controls
- Infrastructure markers with FontAwesome icons
- Mouse position display
- Embeddable HTML output

**Dependencies:**
- folium (required)
- folium.plugins (optional, for advanced features)

## Usage Examples

### Static Flood Map

```python
from pathlib import Path
import numpy as np
from core.reporting.maps import StaticMapGenerator, MapBounds, MapConfig

# Configure map (print quality)
config = MapConfig(
    width=3300,
    height=2550,
    dpi=300,
    title="Hurricane Ian Flood Extent",
    show_scale_bar=True,
    show_north_arrow=True,
    show_legend=True
)

# Create generator
generator = StaticMapGenerator(config)

# Generate flood map
flood_extent = np.load("flood_extent.npy")  # Boolean array
bounds = MapBounds(min_lon=-82.2, min_lat=26.3, max_lon=-81.7, max_lat=26.8)

generator.generate_flood_map(
    flood_extent=flood_extent,
    bounds=bounds,
    output_path=Path("flood_map.png"),
    title="Hurricane Ian - Fort Myers Flooding"
)
```

### Interactive Web Map

```python
from pathlib import Path
import json
from core.reporting.maps import InteractiveMapGenerator, MapBounds

# Load flood extent GeoJSON
with open("flood_extent.geojson") as f:
    flood_geojson = json.load(f)

# Infrastructure data
infrastructure = [
    {
        'lon': -81.87,
        'lat': 26.64,
        'type': 'hospital',
        'name': 'Lee Memorial Hospital',
        'address': '2776 Cleveland Ave'
    },
    {
        'lon': -81.85,
        'lat': 26.62,
        'type': 'school',
        'name': 'Fort Myers High School',
        'address': '2635 Cortez Blvd'
    }
]

# Create generator
generator = InteractiveMapGenerator()

# Generate interactive map
bounds = MapBounds(min_lon=-82.2, min_lat=26.3, max_lon=-81.7, max_lat=26.8)

html = generator.generate_flood_map(
    flood_geojson=flood_geojson,
    bounds=bounds,
    infrastructure=infrastructure,
    title="Hurricane Ian Interactive Flood Map"
)

# Save to file
Path("interactive_map.html").write_text(html)
```

### Before/After Comparison

```python
from pathlib import Path
import numpy as np
from core.reporting.maps import StaticMapGenerator, MapBounds, MapConfig

# Load imagery
before_image = np.load("before_rgb.npy")  # Shape: (H, W, 3)
after_image = np.load("after_rgb.npy")    # Shape: (H, W, 3)

# Configure
config = MapConfig(width=1600, height=600, dpi=150)
generator = StaticMapGenerator(config)

# Generate comparison
bounds = MapBounds(min_lon=-82.2, min_lat=26.3, max_lon=-81.7, max_lat=26.8)

generator.generate_before_after(
    before_image=before_image,
    after_image=after_image,
    bounds=bounds,
    output_path=Path("before_after.png"),
    before_date="September 25, 2022",
    after_date="September 29, 2022"
)
```

## Color Palette

Uses FirstLight design system colors from `core.reporting.utils.color_utils`:

**Flood Severity:**
- Minor: `#90CDF4`
- Moderate: `#4299E1`
- Significant: `#2B6CB0`
- Severe: `#2C5282`
- Extreme: `#1A365D`

**Infrastructure:**
- Hospital: `#E53E3E` (red)
- School: `#3182CE` (blue)
- Shelter: `#38A169` (green)
- Fire Station: `#F97316` (orange)
- Police: `#2C5282` (dark blue)
- Power: `#D69E2E` (yellow)

## Installation

### Required Dependencies

```bash
# Static maps
pip install matplotlib

# Interactive maps
pip install folium
```

### Optional Dependencies

```bash
# Enhanced static maps
pip install cartopy contextily matplotlib-scalebar

# Enhanced interactive maps
pip install folium[plugins]
```

## Cartographic Standards

Maps follow the FirstLight Cartographic Specification (`docs/design/CARTOGRAPHIC_SPEC.md`):

- **Projections:** WGS 84 (EPSG:4326) for data exchange, UTM for local analysis
- **Base Maps:** CartoDB Positron (light, muted for data overlays)
- **Resolution:** 150 DPI for screen, 300 DPI for print
- **Accessibility:** WCAG AA contrast ratios, colorblind-safe palettes
- **Map Furniture:** Scale bar, north arrow, legend, attribution

## Testing

```bash
# Test imports
python -c "from core.reporting.maps import *; print('OK')"

# Test static generator (requires matplotlib)
python -c "
from core.reporting.maps import StaticMapGenerator, MapConfig
gen = StaticMapGenerator(MapConfig())
print(f'Static generator ready (cartopy: {gen.has_cartopy})')
"

# Test interactive generator (requires folium)
python -c "
from core.reporting.maps import InteractiveMapGenerator
gen = InteractiveMapGenerator()
print(f'Interactive generator ready (plugins: {gen.has_plugins})')
"
```

## File Structure

```
core/reporting/maps/
├── __init__.py           # Public API exports
├── base.py               # MapConfig, MapBounds, MapType
├── static_map.py         # StaticMapGenerator
├── folium_map.py         # InteractiveMapGenerator
└── README.md             # This file
```

## Next Steps

See `ROADMAP.md` Epic R2.3 for planned enhancements:
- Inset/locator maps (R2.3.8)
- Pattern overlays for B&W printing (R2.3.10)

## References

- Cartographic Spec: `docs/design/CARTOGRAPHIC_SPEC.md`
- Design System: `docs/design/DESIGN_SYSTEM.md`
- Color Utilities: `core/reporting/utils/color_utils.py`
