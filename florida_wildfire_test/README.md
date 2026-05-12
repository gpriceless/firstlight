# Florida Wildfire Test — Max Road Fire, May 2026

## Overview
FirstLight platform test using active Florida wildfire data.

**Target**: Max Road Fire, western Broward County (Everglades)
- ~11,000 acres burned, 60% contained (as of May 11, 2026)
- Approximate center: 25.98°N, 80.40°W
- Started: May 10, 2026

## Data Sources
Sentinel-2 L2A via Element84 Earth Search STAC API (Cloud Optimized GeoTIFF):

| Scene | Date | Cloud | MGRS Tile |
|-------|------|-------|-----------|
| S2B_17RNJ_20260404 | Apr 4, 2026 | 20% | 17RNJ |
| S2C_17RNJ_20260509 | May 9, 2026 | 24% | 17RNJ |
| S2A_17RNJ_20260511 | May 11, 2026 | 85% | 17RNJ |

Bands: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR, 10m), B12 (SWIR2, 20m), SCL

## Analysis Products

### Visualizations
- `01_true_color_comparison.png` — RGB before/after comparison (Apr 4 vs May 9)
- `02_false_color_vegetation.png` — NIR-R-G false color highlighting vegetation health
- `03_ndvi_vegetation_map.png` — NDVI vegetation density comparison with legend
- `04_dnbr_severity_map.png` — dNBR burn severity map with legend and statistics
- `05_confidence_map.png` — Algorithm confidence raster (green=high, red=low)
- `06_fuel_risk_map.png` — Pre-fire fuel load risk assessment

### GeoTIFFs (georeferenced, EPSG:4326)
- `dnbr_map.tif` — Differenced NBR values (Apr 4 → May 9)
- `nbr_pre_fire.tif` — Pre-fire NBR baseline
- `ndvi_pre_fire.tif` — Pre-fire NDVI at 10m resolution

## Key Findings

### Pre-fire Landscape
- 23% dense vegetation (NDVI > 0.50) — Everglades sawgrass, mangrove, hammocks
- 29% moderate vegetation — mixed marsh and developed landscape
- 7.1% water — canals, retention ponds, Everglades waterways
- Mean NDVI: 0.3216

### Drought Stress (Apr 4 → May 9)
- Mean NDVI decline: +0.0173 (vegetation losing vigor)
- Confirms extreme drought conditions statewide

### Vegetation Change (dNBR)
- 12.9% of area showed burn-like spectral change (drought stress)
- 513 high-severity pixels, 6,720 moderate-high, 23,443 moderate-low
- Algorithm: FirstLight DifferencedNBRAlgorithm (USGS severity thresholds)

### Fuel Risk
- 25% of vegetated area at high fire risk
- 10% at extreme risk — concentrated at urban-wildland interface

## Algorithms Used
- `wildfire.baseline.nbr_differenced` — DifferencedNBRAlgorithm (citations: Key & Benson 2006, Miller & Thode 2007)
- Custom NDVI vegetation characterization
- Custom fuel load assessment (NDVI × SWIR reflectance)

## Limitations
1. Post-fire Sentinel-2 (May 11) was 96.4% cloud-covered — insufficient for definitive burn scar mapping
2. NASA FIRMS thermal data requires API key registration
3. Rerun recommended when cloud-free post-fire imagery available (~May 14–16)
