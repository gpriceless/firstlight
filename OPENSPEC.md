# OpenSpec: Active Specifications

**Status:** In-Progress Specifications Only
**Last Updated:** 2026-01-23
**Archive:** See `docs/OPENSPEC_ARCHIVE.md` for completed specifications

---

## CRITICAL: Production Readiness Audit

**Date:** 2026-01-23

A comprehensive audit revealed that while core algorithms work, the CLI layer is approximately 60% stub/mock implementations. User-facing commands return fake data instead of calling real core libraries.

**See:** `OPENSPEC_PRODUCTION.md` for the full production readiness specification.

### Summary of Findings

| Layer | Audit Assessment | Evidence |
|-------|------------------|----------|
| Core Algorithms | REAL - Works | Verified with Camp Fire analysis |
| CLI Commands | ~60% STUB | Commands return random/mock data |
| API Layer | ~90% Complete | Missing database, JWT validation |
| Agent System | ~95% Complete | Pipeline agent has stub execution |

### Required Work (15-21 days total)

| Phase | Description | Priority | Effort |
|-------|-------------|----------|--------|
| Phase 1 | Wire CLI commands to core libraries | P0 | 8-10 days |
| Phase 2 | Wire agent pipeline to algorithms | P0 | 1-2 days |
| Phase 3 | API database and JWT | P1 | 4-6 days |
| Phase 4 | Integration testing | P2 | 3-5 days |
| Phase 5 | Documentation | P3 | 1-2 days |

**Action Required:** Complete Phase 1 and Phase 2 before any new feature work.

---

## Completed Work (Archived)

The following have been fully implemented and moved to the archive:

- Core Schemas (event, intent, datasource, pipeline, provenance, quality)
- Data Broker Architecture (13 provider integrations)
- Data Ingestion & Normalization Pipeline
- Analysis & Modeling Layer (8 baseline algorithms)
- Multi-Sensor Fusion Engine
- Forecast & Scenario Integration
- Quality Control & Validation
- Agent Architecture (orchestrator, discovery, pipeline, quality, reporting)
- API & Deployment (FastAPI, Docker, Kubernetes, cloud configs)

**Total:** 170K+ lines, 518+ passing tests

---

## Active Specifications

### 1. Distributed Raster Processing

**Status:** COMPLETE (2026-01-13)

#### Problem Statement

Earth observation datasets are massive:
- Single Sentinel-2 scene: 500MB-5GB (100km x 100km at 10m resolution)
- Continental analysis: 100,000km² = 1,000+ scenes = 500GB-5TB
- Current serial tiled processing: 20-30 minutes for 100km² on laptop
- Memory constraints: Existing tiling works but doesn't parallelize across cores
- Download bottleneck: Must download entire scenes before processing begins

#### Goals

1. **Laptop-Scale Parallelization:** Leverage all CPU cores for 4-8x speedup
2. **Cloud-Scale Distribution:** Process continental areas on Spark/Flink clusters
3. **Streaming Ingestion:** Never download full scenes - stream only needed tiles
4. **Transparent Scaling:** Same API works on laptop or 100-node cluster

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Event Specification                          │
│         (Area: 1000km², Event: flood.coastal)                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Discovery (STAC)                        │
│  Finds: 25 Sentinel-1 scenes, 30 Sentinel-2 scenes, DEM        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Virtual Raster Index (NEW)                      │
│  - Build GDAL VRT from STAC query results                      │
│  - No download - just index tile locations                      │
│  - Track which tiles needed for AOI                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Execution Router (NEW)                         │
│  ├─ Small (<100 tiles): Serial execution                       │
│  ├─ Medium (100-1000): Dask local cluster                      │
│  └─ Large (1000+): Sedona on Spark                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│               Distributed Tile Processing                       │
│  - Stream tiles via HTTP range requests                        │
│  - Process in parallel across cores/workers                    │
│  - Memory-mapped intermediate results                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Streamed Results                             │
│  - Mosaic tiles on-the-fly                                     │
│  - Output as COG with overviews                                │
└─────────────────────────────────────────────────────────────────┘
```

#### Technology Stack

| Technology | Use Case | Status |
|-----------|----------|--------|
| **Dask + Rasterio** | Local parallelization (laptop/workstation) | COMPLETE |
| **Apache Sedona 1.5+** | Cloud-scale Spark-based processing | Planned |
| **GDAL Virtual Rasters** | Lightweight tile indexing | COMPLETE |

#### New Files Required

```
core/data/ingestion/
└── virtual_index.py          # GDAL VRT from STAC results

core/analysis/execution/
├── dask_tiled.py             # Dask-based parallel tile processing
├── router.py                 # Execution environment router
└── sedona_backend.py         # Apache Sedona integration (future)
```

#### Success Metrics

- [x] Laptop: 1000km² analysis in <10 minutes (currently 30+ min) - IMPLEMENTED
- [ ] Cloud: 100,000km² analysis in <1 hour (requires Sedona)
- [x] Memory: Peak <4GB on laptop regardless of AOI size - IMPLEMENTED
- [x] Streaming: Zero full scene downloads (tile streaming only) - IMPLEMENTED
- [x] Parallelization: 80%+ CPU utilization on multi-core laptops - IMPLEMENTED

#### Implementation Summary (2026-01-13)

**Files Created:**
- `core/data/ingestion/virtual_index.py` - VirtualRasterIndex, STACVRTBuilder (DP-1)
- `core/analysis/execution/dask_tiled.py` - DaskTileProcessor, parallel engine (DP-2)
- `core/analysis/execution/dask_adapters.py` - Algorithm adapters (DP-3)
- `core/analysis/execution/router.py` - ExecutionRouter, auto selection (DP-4)
- `tests/test_dask_execution.py` - Integration tests (DP-5)

**Key Classes:**
- `VirtualRasterIndex` - Lazy tile access via VRT
- `DaskTileProcessor` - Parallel tile processing
- `ExecutionRouter` - Automatic backend selection
- `DaskAlgorithmAdapter` - Wrap any algorithm for Dask

---

### 2. Production Image Validation

**Status:** Requirements defined, implementation pending

#### Overview

Add production-grade image validation to the ingestion workflow. When satellite imagery is downloaded for algorithm processing, validate band integrity and optionally capture screenshots BEFORE images are processed or merged.

#### Requirements

- Validate band presence and content for optical imagery (Sentinel-2, Landsat)
- Ensure images are not blank before merging into mosaics
- Handle SAR imagery differently (speckle-aware validation)
- Optional screenshot capture for debugging/audit trails

#### Validation Thresholds

```yaml
optical_validation:
  min_std_dev: 1.0          # Band not blank if std dev > 1.0
  max_nodata_percent: 50    # Fail if >50% nodata
  expected_value_range:
    sentinel2_l1c: [0, 10000]
    sentinel2_l2a: [0, 10000]
    landsat8_l1: [0, 65535]
    landsat8_l2: [0, 10000]

sar_validation:
  min_std_dev_db: 2.0       # Higher threshold due to speckle
  backscatter_range_db: [-30, 10]
  required_polarizations: ["VV", "VH"]
```

#### New Files Required

```
core/data/ingestion/validation/
├── __init__.py
├── exceptions.py             # ValidationError, BlankBandError, etc.
├── config.py                 # Validation configuration schema
├── image_validator.py        # Base ImageValidator class
├── band_validator.py         # OpticalBandValidator
├── sar_validator.py          # SARValidator (speckle-aware)
└── screenshot_generator.py   # Optional screenshot capture

config/
└── ingestion.yaml            # Validation settings
```

#### Integration Points

1. `core/data/ingestion/streaming.py` (~line 1048) - Post-download validation
2. `core/analysis/execution/tiled_runner.py` (~line 509) - Pre-merge validation
3. `core/analysis/execution/runner.py` (~line 1047) - Pipeline integration

#### Success Criteria

- [ ] All optical bands validated before processing
- [ ] Blank band detection catches >95% of corrupt/missing data
- [ ] SAR validation correctly handles speckle noise
- [ ] Screenshots captured when enabled (configurable)
- [ ] Zero false positives on valid imagery
- [ ] <500ms validation overhead per image

**Full Requirements:** See `docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md`

---

## Architecture Decisions Pending

### Distributed Processing

- [ ] Tile caching strategy: S3 vs local SSD vs memory
- [ ] Spark session lifecycle: per-job vs persistent pool
- [ ] Failure handling: retry tiles vs fail entire job
- [ ] Progress reporting: polling vs push notifications
- [ ] Tile size optimization: 256x256 vs 512x512 pixels (TBD via profiling)

### Image Validation

- [ ] Screenshot storage location and retention policy
- [ ] Integration with existing QC pipeline
- [ ] Alerting strategy for validation failures

---

## Reference

### Related Documentation

- **`OPENSPEC_PRODUCTION.md`** - **CRITICAL: Production readiness specification (current priority)**
- `docs/OPENSPEC_ARCHIVE.md` - Complete specifications for implemented features
- `docs/PRODUCTION_IMAGE_VALIDATION_REQUIREMENTS.md` - Detailed validation requirements
- `ROADMAP.md` - Implementation roadmap and task breakdown
- `FIXES.md` - Bug tracking (P0 bugs must be fixed before new features)
- `AUDIT_REPORT.md` - Initial comprehensive audit
- `AUDIT_ADDENDUM.md` - CLI layer corrections

### Test Commands

```bash
# Run all tests
./run_tests.py

# Run specific categories
./run_tests.py flood
./run_tests.py wildfire
./run_tests.py schemas

# Run with specific algorithm
./run_tests.py --algorithm sar
./run_tests.py --algorithm ndwi
```

---

## 3. Multi-Band Asset Download

**Status:** Specification Complete - Ready for Implementation
**Priority:** P0 - Blocks all real Sentinel-2 processing
**Bug Report:** `docs/SENTINEL2_INGESTION_BUG_REPORT.md`
**Date Added:** 2026-01-25

---

### Problem Statement

The current ingestion pipeline cannot process real Sentinel-2 satellite imagery from STAC catalogs. The STAC client retrieves URLs for True Color Images (TCI.tif) - 3-band RGB composites - while downstream algorithms and validators expect individual spectral bands (blue, green, red, nir, swir16, swir22).

**Root Cause Location:** `core/data/discovery/stac_client.py`, line 446:

```python
"url": item.assets.get("visual") or item.assets.get("data") or "",
```

This line prioritizes the `visual` asset, which for Sentinel-2 is a pre-rendered RGB composite intended for visualization, not scientific analysis. The composite lacks the spectral bands (especially NIR and SWIR) required for:

- NDVI calculation (requires NIR)
- NDWI calculation (requires NIR, SWIR)
- Burn severity analysis (requires NIR, SWIR)
- Flood detection (requires NIR, SWIR)
- All spectral index algorithms

**Consequence:** Every Sentinel-2 ingestion fails with validation errors:

```
Image validation failed: [
  "Required band 'blue' not found in dataset",
  "Required band 'green' not found in dataset",
  "Required band 'red' not found in dataset",
  "Required band 'nir' not found in dataset"
]
```

This blocks all real-world Sentinel-2 analysis, making the platform unusable for its primary purpose.

---

### Solution Overview

Implement a multi-band download architecture that:

1. **Discovers** individual spectral band URLs from STAC item assets
2. **Downloads** each required band as a separate GeoTIFF
3. **Stacks** bands into a single analysis-ready file using GDAL VRT
4. **Validates** the stacked multi-band file against expected band requirements

```
                         PROPOSED FLOW (FIXED)

  STAC Catalog              stac_client.py                 ingest.py
+---------------+         +------------------+         +------------------+
| Sentinel-2    | search  | discover_data()  | results | process_item()   |
| L2A Items     |-------->| returns band_urls|-------->| downloads each   |
|               |         | for each asset   |         | band separately  |
| Assets:       |         |                  |         |                  |
|  - visual     |         | band_urls: {     |         | blue.tif         |
|  - blue  -----|---------|-- blue: URL      |         | green.tif        |
|  - green -----|---------|-- green: URL     |         | red.tif          |
|  - red   -----|---------|-- red: URL       |         | nir.tif          |
|  - nir   -----|---------|-- nir: URL       |         | swir16.tif       |
|  - swir16-----|---------|-- swir16: URL    |         | swir22.tif       |
|  - swir22-----|---------|-- swir22: URL    |         |                  |
+---------------+         +------------------+         +--------+---------+
                                                                |
                                                                v
                                                       +------------------+
                                                       | band_stack.py    |
                                                       | create_vrt()     |
                                                       |                  |
                                                       | Output:          |
                                                       | scene_stack.vrt  |
                                                       | (6+ bands)       |
                                                       +--------+---------+
                                                                |
                                                                v
                                                       +------------------+
                                                       | ImageValidator   |
                                                       | validates stack: |
                                                       |  - blue    [OK]  |
                                                       |  - green   [OK]  |
                                                       |  - red     [OK]  |
                                                       |  - nir     [OK]  |
                                                       |  - swir16  [OK]  |
                                                       |  - swir22  [OK]  |
                                                       |                  |
                                                       | VALIDATION PASS  |
                                                       +------------------+
```

---

### User Stories

#### US-1: Analyst Downloads Sentinel-2 for Flood Analysis

**As** a disaster response analyst
**I want** to ingest Sentinel-2 imagery with all required spectral bands
**So that** I can run NDWI and water detection algorithms that require NIR/SWIR bands

**Acceptance Criteria:**
- [ ] `flight discover` returns individual band URLs for Sentinel-2 items
- [ ] `flight ingest` downloads blue, green, red, nir, swir16, swir22 bands
- [ ] Downloaded bands are automatically stacked into analysis-ready format
- [ ] Image validation passes with all required bands present
- [ ] NDWI algorithm runs successfully on ingested data

#### US-2: Operator Configures Which Bands to Download

**As** a system operator
**I want** to configure which spectral bands are downloaded per data source
**So that** I can optimize storage and bandwidth for specific analysis workflows

**Acceptance Criteria:**
- [ ] Configuration file specifies required bands per sensor type
- [ ] Bands not in configuration are skipped (not downloaded)
- [ ] CLI provides `--bands` override option
- [ ] Default configuration includes standard analysis bands

#### US-3: Pipeline Recovers from Partial Band Download Failures

**As** a pipeline operator
**I want** the ingestion to handle partial download failures gracefully
**So that** transient network issues don't require full re-downloads

**Acceptance Criteria:**
- [ ] Each band download is retried independently (up to 3 attempts)
- [ ] Successfully downloaded bands are preserved on failure
- [ ] Resume capability for interrupted multi-band downloads
- [ ] Clear error reporting identifies which specific bands failed

#### US-4: Analyst Uses Existing Composite for Visualization

**As** an analyst
**I want** the option to download TCI composite for quick visualization
**So that** I can preview data without downloading full spectral bands

**Acceptance Criteria:**
- [ ] `--visualization-only` flag downloads only TCI composite
- [ ] Validation mode relaxed for visualization downloads
- [ ] Clear warning that visualization data cannot be used for spectral analysis

---

### Functional Requirements

#### FR-1: STAC Client Changes

**File:** `core/data/discovery/stac_client.py`

**FR-1.1:** Return individual band URLs instead of single composite URL

```python
# Replace line 446:
# "url": item.assets.get("visual") or item.assets.get("data") or "",

# With structured band URLs:
"url": None,  # Deprecated - use band_urls
"band_urls": self._extract_band_urls(item, source),
```

**FR-1.2:** Implement band URL extraction per sensor type

| Sensor | Required Band Assets |
|--------|---------------------|
| Sentinel-2 L2A | blue, green, red, nir, swir16, swir22 |
| Sentinel-2 L1C | blue, green, red, nir, swir16, swir22 |
| Landsat 8/9 | blue, green, red, nir08, swir16, swir22 |
| Sentinel-1 | vv, vh (polarizations, not spectral bands) |

**FR-1.3:** Preserve backward compatibility

- Keep `url` field populated with primary band (blue) for backward compat
- Add `band_urls` dictionary with all analysis bands
- Add `visualization_url` field with TCI/visual asset URL

**FR-1.4:** Handle missing band assets gracefully

- Log warning for expected but missing bands
- Continue with available bands
- Include `missing_bands` list in result

**FR-1.5:** Band configuration

Define sensor band mappings in new configuration:

```python
# core/data/discovery/band_config.py

SENTINEL2_BANDS = {
    "blue": "blue",       # B02, 490nm, 10m
    "green": "green",     # B03, 560nm, 10m
    "red": "red",         # B04, 665nm, 10m
    "nir": "nir",         # B08, 842nm, 10m
    "swir16": "swir16",   # B11, 1610nm, 20m
    "swir22": "swir22",   # B12, 2190nm, 20m
}

SENTINEL2_OPTIONAL_BANDS = {
    "coastal": "coastal",     # B01, 443nm, 60m
    "rededge1": "rededge1",   # B05, 705nm, 20m
    "rededge2": "rededge2",   # B06, 740nm, 20m
    "rededge3": "rededge3",   # B07, 783nm, 20m
    "nir08": "nir08",         # B8A, 865nm, 20m
    "nir09": "nir09",         # B09, 940nm, 60m
    "scl": "scl",             # Scene classification
}

LANDSAT_BANDS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
}
```

---

#### FR-2: Ingestion Pipeline Changes

**File:** `cli/commands/ingest.py`

**FR-2.1:** Update `process_item()` to handle multi-band downloads

```python
def process_item(item: Dict, output_path: Path, ...) -> bool:
    band_urls = item.get("band_urls", {})

    if band_urls:
        # Multi-band download path
        band_paths = download_bands(band_urls, item_dir, item)
        stack_path = create_band_stack(band_paths, item_dir / f"{item_id}_stack.vrt")
        return validate_and_finalize(stack_path, item)
    else:
        # Legacy single-URL path (backward compat)
        url = item.get("url")
        if url:
            return download_single_file(url, output_path, item)
```

**FR-2.2:** Implement `download_bands()` function

```python
def download_bands(
    band_urls: Dict[str, str],
    output_dir: Path,
    item: Dict[str, Any],
    max_retries: int = 3,
    parallel: bool = True,
) -> Dict[str, Path]:
    """
    Download multiple band files for a single scene.

    Args:
        band_urls: Mapping of band name to URL
        output_dir: Directory to save band files
        item: Item metadata
        max_retries: Retry attempts per band
        parallel: Download bands in parallel

    Returns:
        Mapping of band name to downloaded file path

    Raises:
        IngestionError: If required bands fail to download
    """
```

**FR-2.3:** Parallel download support

- Use ThreadPoolExecutor for concurrent downloads
- Default: 4 parallel downloads
- Configurable via `--parallel-downloads N`
- Respect rate limits (configurable delay between requests)

**FR-2.4:** Progress reporting

- Report per-band download progress
- Show overall scene progress (X of Y bands complete)
- Display total bytes downloaded vs expected

**FR-2.5:** Retry and resume logic

- Each band independently retried on failure
- Existing complete band files skipped on retry
- Partial downloads detected and restarted

---

#### FR-3: Band Stacking Utility

**New File:** `core/data/ingestion/band_stack.py`

**FR-3.1:** Create GDAL VRT from individual band files

```python
def create_band_stack(
    band_paths: Dict[str, Path],
    output_path: Path,
    band_order: Optional[List[str]] = None,
    band_names: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Create a virtual raster (VRT) stacking individual band files.

    The VRT file references the original band files without duplicating data,
    providing a single multi-band interface for downstream processing.

    Args:
        band_paths: Mapping of band name to file path
        output_path: Path for output VRT file
        band_order: Order of bands in output (default: alphabetical)
        band_names: Band names to write in VRT metadata

    Returns:
        Path to created VRT file

    Raises:
        BandStackError: If stacking fails
    """
```

**FR-3.2:** Validate band compatibility before stacking

- Check CRS matches across all bands
- Check pixel resolution compatibility (handle 10m vs 20m)
- Check spatial extent overlap
- Warn on dimension mismatches

**FR-3.3:** Handle mixed resolution bands

Sentinel-2 bands have different native resolutions:
- 10m: blue, green, red, nir
- 20m: swir16, swir22, rededge bands

Options (configurable):
1. **Resample to common resolution** (default: 10m) using bilinear interpolation
2. **Preserve native resolution** with separate VRTs per resolution
3. **Crop to intersection** of all band extents

**FR-3.4:** VRT band metadata

Set band descriptions in VRT for downstream identification:

```xml
<VRTRasterBand dataType="UInt16" band="1">
  <Description>blue</Description>
  <SimpleSource>
    <SourceFilename relativeToVRT="1">blue.tif</SourceFilename>
    <SourceBand>1</SourceBand>
  </SimpleSource>
</VRTRasterBand>
```

**FR-3.5:** Optional GeoTIFF output

For workflows requiring a single file (not VRT):

```python
def stack_to_geotiff(
    band_paths: Dict[str, Path],
    output_path: Path,
    compress: str = "LZW",
    tiled: bool = True,
) -> Path:
    """
    Create a multi-band GeoTIFF from individual bands.

    Note: This copies all data, increasing storage requirements.
    Use VRT when possible.
    """
```

---

#### FR-4: Validator Updates

**File:** `core/data/ingestion/validation/image_validator.py`

**FR-4.1:** Detect VRT files and validate referenced bands

```python
def _validate_vrt(self, vrt_path: Path, data_source_spec: Dict) -> ValidationResult:
    """
    Validate a VRT file by checking:
    1. VRT structure is valid
    2. All referenced files exist and are readable
    3. Band metadata matches expected bands
    4. Each band passes content validation
    """
```

**FR-4.2:** Match bands by VRT description metadata

The validator currently tries to match bands by filename patterns. Update to also check VRT band descriptions:

```python
def _get_band_name_from_vrt(self, dataset, band_index: int) -> Optional[str]:
    """Get band name from VRT Description tag."""
    desc = dataset.descriptions[band_index - 1]
    if desc and desc.lower() in KNOWN_BAND_NAMES:
        return desc.lower()
    return None
```

**FR-4.3:** Validate band order consistency

When processing stacked files, ensure band order matches expected configuration.

**File:** `core/data/ingestion/validation/band_validator.py`

**FR-4.4:** Update band matching to use VRT metadata

```python
def _identify_bands(self, dataset) -> Dict[str, int]:
    """
    Identify bands in dataset by multiple methods:
    1. VRT Description tags (most reliable for stacked files)
    2. GeoTIFF band descriptions
    3. Filename patterns
    4. Positional fallback (only if band count matches)
    """
```

---

### Non-Functional Requirements

#### NFR-1: Performance

| Metric | Requirement |
|--------|-------------|
| Band download parallelism | 4 concurrent downloads default |
| Per-band download timeout | 5 minutes per band |
| VRT creation time | <5 seconds for 6-band stack |
| Validation overhead | <500ms per stacked file |
| Memory usage | <500MB for 6-band stack validation |

#### NFR-2: Storage

| Scenario | Storage Requirement |
|----------|---------------------|
| VRT stack | Negligible (<10KB VRT file + original bands) |
| GeoTIFF stack | ~6x single band size (bands copied) |
| Band cache | 30-day retention default (configurable) |

**Storage Calculation for Sentinel-2:**
- Single band (10m, 100km tile): ~100MB
- 6 analysis bands: ~600MB per scene
- Full scene (all bands): ~1.5GB

#### NFR-3: Error Handling

| Failure Mode | Behavior |
|--------------|----------|
| Network timeout (single band) | Retry 3 times with exponential backoff |
| Band download failure (after retries) | Fail ingestion, report which band(s) failed |
| Missing band in STAC item | Log warning, proceed with available bands if non-critical |
| VRT creation failure | Fail ingestion with detailed error |
| Validation failure | Fail ingestion with band-specific errors |

#### NFR-4: Logging

All band operations must be logged at appropriate levels:

| Level | Events |
|-------|--------|
| INFO | Band download started, completed, VRT created |
| WARNING | Optional band missing, retry attempted |
| ERROR | Required band failed, validation error |
| DEBUG | Download progress, byte counts, timing |

---

### API Changes

#### CLI Interface

**`flight discover`** - No changes required (band_urls included in output)

**`flight ingest`** - New options:

```bash
flight ingest --area area.geojson --event flood.coastal \
    --bands blue,green,red,nir,swir16    # Override default bands
    --parallel-downloads 4               # Concurrent band downloads
    --visualization-only                 # Download TCI only (no validation)
    --output-format vrt                  # VRT (default) or geotiff
    --skip-validation                    # Skip validation (emergency workaround)
```

**`flight info`** - Add band configuration display:

```bash
flight info --bands sentinel2
# Shows: blue(B02), green(B03), red(B04), nir(B08), swir16(B11), swir22(B12)
```

#### Python API

**Discovery Result Format:**

```python
{
    "id": "S2B_MSIL2A_20260125T...",
    "source": "sentinel2",
    "datetime": "2026-01-25T10:45:00Z",
    "cloud_cover": 5.2,
    "resolution_m": 10.0,
    "size_bytes": 600000000,  # Estimated total for configured bands

    # NEW: Individual band URLs
    "band_urls": {
        "blue": "https://sentinel-cogs.s3.../B02.tif",
        "green": "https://sentinel-cogs.s3.../B03.tif",
        "red": "https://sentinel-cogs.s3.../B04.tif",
        "nir": "https://sentinel-cogs.s3.../B08.tif",
        "swir16": "https://sentinel-cogs.s3.../B11.tif",
        "swir22": "https://sentinel-cogs.s3.../B12.tif",
    },

    # NEW: Visualization URL (separate from analysis bands)
    "visualization_url": "https://sentinel-cogs.s3.../TCI.tif",

    # DEPRECATED: Single URL (kept for backward compat)
    "url": "https://sentinel-cogs.s3.../B02.tif",

    # Existing: Full asset dictionary
    "assets": {...},

    # NEW: Missing expected bands
    "missing_bands": [],
}
```

**Ingestion Result Format:**

```python
{
    "status": "completed",
    "item_id": "S2B_MSIL2A_20260125T...",
    "output_path": "/data/scenes/S2B_.../S2B_stack.vrt",

    # NEW: Band download details
    "bands_downloaded": {
        "blue": {"path": "blue.tif", "size_bytes": 105000000, "time_s": 12.3},
        "green": {"path": "green.tif", "size_bytes": 102000000, "time_s": 11.8},
        # ...
    },
    "bands_failed": [],
    "total_download_time_s": 45.2,
    "total_size_bytes": 612000000,

    # Validation result
    "validation": {
        "is_valid": True,
        "bands_validated": ["blue", "green", "red", "nir", "swir16", "swir22"],
        "warnings": [],
        "errors": [],
    },
}
```

---

### Data Model

#### New Classes

**`BandDownloadResult`** (dataclass)

```python
@dataclass
class BandDownloadResult:
    """Result of downloading a single band."""
    band_name: str
    url: str
    local_path: Optional[Path]
    size_bytes: int
    download_time_s: float
    success: bool
    error: Optional[str] = None
    retries: int = 0
```

**`StackedRaster`** (dataclass)

```python
@dataclass
class StackedRaster:
    """Metadata for a stacked multi-band raster."""
    path: Path
    format: str  # "vrt" or "geotiff"
    bands: Dict[str, int]  # band name -> band index
    crs: str
    resolution: Tuple[float, float]
    bounds: Tuple[float, float, float, float]
    source_files: Dict[str, Path]  # band name -> source file
```

**`SensorBandConfig`** (dataclass)

```python
@dataclass
class SensorBandConfig:
    """Band configuration for a specific sensor."""
    sensor_name: str
    required_bands: List[str]
    optional_bands: List[str]
    stac_asset_mapping: Dict[str, str]  # generic name -> STAC asset key
    native_resolutions: Dict[str, float]  # band name -> resolution in meters
```

#### Configuration Schema

**`config/band_config.yaml`**

```yaml
sensors:
  sentinel2:
    required_bands:
      - blue
      - green
      - red
      - nir
    optional_bands:
      - coastal
      - rededge1
      - rededge2
      - rededge3
      - nir08
      - swir16
      - swir22
      - scl
    default_bands:
      - blue
      - green
      - red
      - nir
      - swir16
      - swir22
    stac_assets:
      blue: blue
      green: green
      red: red
      nir: nir
      swir16: swir16
      swir22: swir22
    resolutions:
      blue: 10
      green: 10
      red: 10
      nir: 10
      swir16: 20
      swir22: 20

  sentinel1:
    required_bands:
      - vv
      - vh
    stac_assets:
      vv: vv
      vh: vh

  landsat:
    required_bands:
      - blue
      - green
      - red
      - nir08
    default_bands:
      - blue
      - green
      - red
      - nir08
      - swir16
      - swir22

download:
  parallel_downloads: 4
  retry_attempts: 3
  retry_delay_s: 2
  timeout_per_band_s: 300

stacking:
  default_format: vrt
  resample_method: bilinear
  target_resolution: null  # null = preserve native
```

---

### Success Criteria

#### Functional Acceptance

- [ ] **SC-1:** `flight discover` returns `band_urls` for Sentinel-2 items with all 6 default bands
- [ ] **SC-2:** `flight ingest` downloads each band file separately to scene directory
- [ ] **SC-3:** Band files are stacked into VRT with correct band metadata
- [ ] **SC-4:** Image validation passes for stacked VRT files
- [ ] **SC-5:** NDVI algorithm executes successfully on ingested Sentinel-2 data
- [ ] **SC-6:** NDWI algorithm executes successfully (proves NIR/SWIR present)
- [ ] **SC-7:** Burn severity algorithm executes successfully (proves SWIR bands)
- [ ] **SC-8:** Partial band failures are handled with proper retry and error reporting
- [ ] **SC-9:** `--visualization-only` flag downloads TCI with relaxed validation
- [ ] **SC-10:** Backward compatibility maintained for single-URL discovery results

#### Performance Acceptance

- [ ] **SC-11:** 6-band Sentinel-2 scene downloads in <5 minutes (100Mbps connection)
- [ ] **SC-12:** VRT creation completes in <5 seconds
- [ ] **SC-13:** Validation completes in <500ms
- [ ] **SC-14:** Memory usage <500MB during validation

#### Quality Acceptance

- [ ] **SC-15:** All existing tests continue to pass
- [ ] **SC-16:** New tests cover multi-band download path
- [ ] **SC-17:** New tests cover VRT stacking
- [ ] **SC-18:** New tests cover band validation for stacked files
- [ ] **SC-19:** Integration test with real STAC catalog passes

---

### Dependencies

#### Must Be Complete First

| Dependency | Status | Notes |
|------------|--------|-------|
| Production Image Validation | COMPLETE | Required for stacked file validation |
| CLI wired to core libraries | COMPLETE | Required for real downloads |

#### External Dependencies

| Dependency | Version | Purpose |
|------------|---------|---------|
| GDAL | 3.0+ | VRT creation and manipulation |
| rasterio | 1.3+ | Raster I/O and band operations |
| pystac-client | 0.5+ | STAC catalog queries |
| requests | 2.28+ | HTTP downloads |

---

### Out of Scope

The following are explicitly NOT part of this specification:

1. **Cloud-native streaming (VSICURL)** - Future enhancement; this spec focuses on download-first approach
2. **Mosaic creation** - Handled by existing mosaic utilities; this spec handles single-scene ingestion
3. **Atmospheric correction** - Assumes L2A (already corrected) data
4. **Band math/indices calculation** - Handled by analysis algorithms
5. **Data archival and lifecycle** - Handled by existing storage management
6. **Authentication/authorization for private STAC catalogs** - Future enhancement
7. **Support for non-COG formats** - Assumes Cloud-Optimized GeoTIFF source files

---

### Implementation Plan

#### Phase 1: STAC Client Updates (2-3 days)

1. Create `core/data/discovery/band_config.py` with sensor band mappings
2. Update `stac_client.py` to return `band_urls` dictionary
3. Add tests for multi-band URL extraction
4. Verify backward compatibility with existing discovery calls

#### Phase 2: Band Download Infrastructure (3-4 days)

1. Create band download utilities in `cli/commands/ingest.py`
2. Implement parallel download with retry logic
3. Add progress reporting
4. Add tests for download success/failure scenarios

#### Phase 3: Band Stacking (2-3 days)

1. Create `core/data/ingestion/band_stack.py`
2. Implement VRT creation with band metadata
3. Handle mixed resolution bands
4. Add tests for stacking scenarios

#### Phase 4: Validator Updates (1-2 days)

1. Update `image_validator.py` to handle VRT files
2. Update `band_validator.py` to use VRT metadata
3. Add tests for stacked file validation

#### Phase 5: Integration and Testing (2-3 days)

1. End-to-end integration test with real STAC catalog
2. Test all analysis algorithms with ingested data
3. Performance benchmarking
4. Documentation updates

**Total Estimated Effort:** 10-15 days

---

### Test Plan

#### Unit Tests

```
tests/
├── test_band_config.py           # Sensor band configuration
├── test_stac_band_extraction.py  # STAC client band URL extraction
├── test_band_download.py         # Download with retry logic
├── test_band_stack.py            # VRT creation and manipulation
└── test_vrt_validation.py        # Validation of stacked files
```

#### Integration Tests

```
tests/integration/
├── test_sentinel2_ingestion.py   # Full S2 download and stack
├── test_landsat_ingestion.py     # Full Landsat download and stack
└── test_algorithm_on_stack.py    # NDVI/NDWI on stacked data
```

#### Test Data

- Mock STAC responses with individual band assets
- Sample single-band GeoTIFFs for stacking tests
- Pre-built VRT files for validation tests

---

### Rollback Plan

If issues arise post-deployment:

1. **Immediate:** Add `--legacy-download` flag to use original single-URL path
2. **Short-term:** Config toggle to disable multi-band download globally
3. **Validation bypass:** `--skip-validation` flag already specified as workaround

---

### Open Questions

1. **Q:** Should we support partial stacks (e.g., RGB only without NIR)?
   **A:** Yes, via `--bands` CLI option. Validation will warn about missing bands but not fail unless configured as required.

2. **Q:** How to handle rate limiting from STAC data providers?
   **A:** Configurable delay between requests; default 100ms. Provider-specific configs possible.

3. **Q:** Should VRT or GeoTIFF be the default output format?
   **A:** VRT (default) for storage efficiency. GeoTIFF available via `--output-format geotiff`.

---

## 3.1 Multi-Band Asset Download - Implementation Addendum

**Date:** 2026-01-25
**Author:** Requirements Reviewer
**Purpose:** Fill gaps for sequential and parallel implementation

---

### Overview: What This Addendum Covers

The main specification (Section 3) is comprehensive for describing WHAT needs to be built. This addendum addresses:

1. **Interface Contracts** - Exact function signatures and data structures engineers must share
2. **Hidden Dependencies** - Subtle dependencies not obvious from task descriptions
3. **Edge Cases and Error Handling** - Detailed scenarios for robustness
4. **Test Fixtures** - Specific test data requirements
5. **Rollback Procedures** - How to safely back out partial implementations
6. **Configuration Precedence** - How config sources interact

---

### 3.1.1 Interface Contracts (CRITICAL FOR PARALLEL WORK)

Engineers working on different components MUST agree on these interfaces BEFORE starting work.

#### Contract A: STAC Client to Ingestion Pipeline

**File:** `core/data/discovery/stac_client.py`

The `discover_data()` function must return items with this exact structure:

```python
@dataclass
class DiscoveryResultItem:
    """
    INTERFACE CONTRACT: Do not change field names without updating all consumers.

    Consumers:
    - cli/commands/discover.py (display)
    - cli/commands/ingest.py (download orchestration)
    - core/data/ingestion/streaming.py (download execution)
    """
    id: str                              # e.g., "S2B_MSIL2A_20260125T..."
    source: str                          # e.g., "sentinel2", "landsat", "sentinel1"
    datetime: str                        # ISO 8601 format
    cloud_cover: Optional[float]         # Percentage, 0-100
    resolution_m: float                  # Native resolution in meters
    size_bytes: int                      # Estimated total size
    bbox: Tuple[float, float, float, float]  # (west, south, east, north) in WGS84
    priority: str                        # "primary" or "secondary"

    # NEW FIELDS (Epic 1.7)
    url: Optional[str]                   # DEPRECATED - backward compat only; primary band URL
    band_urls: Dict[str, str]            # band_name -> URL mapping
    visualization_url: Optional[str]     # TCI/visual asset URL
    missing_bands: List[str]             # Expected bands not found in STAC item
    assets: Dict[str, Any]               # Raw STAC assets (preserve for debugging)
```

**Serialization Note:** When returned as dict (for JSON), all fields remain the same. The `band_urls` dict keys are lowercase band names: `blue`, `green`, `red`, `nir`, `swir16`, `swir22`.

#### Contract B: Band Download Result

**File:** `cli/commands/ingest.py` (or new `core/data/ingestion/band_downloader.py`)

```python
@dataclass
class BandDownloadResult:
    """
    Result from downloading a single band file.

    INTERFACE CONTRACT: Used by:
    - download_bands() function
    - progress reporting
    - band stacking utility
    - error recovery logic
    """
    band_name: str               # e.g., "blue", "nir"
    url: str                     # Source URL
    local_path: Optional[Path]   # None if download failed
    size_bytes: int              # Actual bytes downloaded
    download_time_s: float       # Wall-clock time
    success: bool
    error: Optional[str]         # Error message if success=False
    retries: int                 # Number of retry attempts made
    checksum: Optional[str]      # MD5 hash of downloaded file (for resume detection)
```

#### Contract C: Band Stacking Input/Output

**File:** `core/data/ingestion/band_stack.py` (NEW)

```python
def create_band_stack(
    band_paths: Dict[str, Path],
    output_path: Path,
    band_order: Optional[List[str]] = None,
    band_descriptions: Optional[Dict[str, str]] = None,
    resolution_mode: str = "resample",  # "resample" | "preserve" | "intersect"
    target_resolution: Optional[float] = None,
    resample_method: str = "bilinear",
) -> StackResult:
    """
    Create VRT stacking individual band files.

    INTERFACE CONTRACT:

    Args:
        band_paths: Keys are lowercase band names (e.g., "blue", "nir")
                   Values are absolute Path objects to .tif files
        output_path: Must end in .vrt; parent directory must exist
        band_order: Order of bands in output VRT; default alphabetical
        band_descriptions: Override band descriptions in VRT metadata
        resolution_mode:
            - "resample": Resample all bands to target_resolution (default: 10m)
            - "preserve": Keep native resolutions (creates separate VRTs)
            - "intersect": Crop to intersection of all extents
        target_resolution: Target resolution in meters (only for resample mode)
        resample_method: GDAL resampling method name

    Returns:
        StackResult with:
        - path: Path to created VRT
        - band_count: Number of bands
        - band_mapping: Dict[str, int] mapping band names to 1-indexed positions
        - crs: Output CRS as WKT
        - bounds: (west, south, east, north)
        - resolution: (x_res, y_res) in CRS units
        - warnings: List of non-fatal issues

    Raises:
        BandStackError: Base exception for stacking failures
        CRSMismatchError: Bands have incompatible CRS
        ExtentMismatchError: Bands don't overlap
        ResolutionError: Cannot resample to target resolution
    """
```

```python
@dataclass
class StackResult:
    """Result from band stacking operation."""
    path: Path
    band_count: int
    band_mapping: Dict[str, int]  # band_name -> 1-indexed position
    crs: str                      # WKT
    bounds: Tuple[float, float, float, float]
    resolution: Tuple[float, float]
    warnings: List[str]
    source_files: Dict[str, Path]  # Preserved for debugging
```

#### Contract D: Validator Interface for VRT

**File:** `core/data/ingestion/validation/image_validator.py`

The existing `validate()` method signature remains unchanged. New behavior for VRT files:

```python
def validate(
    self,
    raster_path: Union[str, Path],
    data_source_spec: Optional[Dict[str, Any]] = None,
    dataset_id: str = "",
    capture_screenshot: bool = False,
) -> ValidationResult:
    """
    VRT-SPECIFIC BEHAVIOR (NEW):

    When raster_path ends in .vrt:
    1. Validates VRT XML structure is valid
    2. Verifies all referenced source files exist and are readable
    3. Opens VRT with rasterio and validates as a single multi-band dataset
    4. Extracts band names from VRT <Description> tags
    5. Uses data_source_spec["bands"] if provided, else derives from VRT

    VRT-specific validations:
    - Source file accessibility (local paths must exist)
    - Band count matches VRT definition
    - Band descriptions match expected band names

    Returns ValidationResult with:
    - vrt_sources: List of source file paths (new field)
    - band_names_from_vrt: Dict mapping index -> name (new field)
    """
```

---

### 3.1.2 Hidden Dependencies Between Tasks

#### Dependency Graph (Expanded)

```
                    ┌─────────────────────────────────────────┐
                    │ 1.7.0 Define Interface Contracts        │ ◀── START HERE
                    │ (This addendum, must review first)      │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
   ┌────────────────────────────┐           ┌────────────────────────────┐
   │ 1.7.1 STAC Client          │           │ 1.7.4 Skip-Validation Flag │
   │ band_urls extraction       │           │ (independent workaround)   │
   └────────────┬───────────────┘           └────────────────────────────┘
                │                                         │
    ┌───────────┴───────────────────┬────────────────────┴──┐
    │                               │                        │
    ▼                               ▼                        │
┌───────────────────┐   ┌───────────────────────┐            │
│ 1.7.2 Ingestion   │   │ 1.7.3 Band Stacking   │            │
│ Multi-file DL     │   │ VRT creation          │            │
│                   │   │                       │            │
│ HIDDEN DEP:       │   │ HIDDEN DEP:           │            │
│ - Uses band_urls  │   │ - Uses virtual_index  │            │
│   from 1.7.1      │   │   module patterns     │            │
│ - Needs download  │   │ - Must match Contract │            │
│   checksum logic  │   │   C exactly           │            │
└────────┬──────────┘   └───────────┬───────────┘            │
         │                          │                        │
         └──────────┬───────────────┘                        │
                    │                                        │
                    ▼                                        │
         ┌─────────────────────┐                             │
         │ 1.7.5 Validator     │◀────────────────────────────┘
         │ VRT support         │     (workaround informs
         │                     │      what to skip)
         │ HIDDEN DEPS:        │
         │ - Contract D        │
         │ - band_validator.py │
         │   _identify_bands   │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ 1.7.6 Integration   │
         │ Testing             │
         └─────────────────────┘
```

#### Critical Hidden Dependencies

| Task | Hidden Dependency | Impact if Missed |
|------|-------------------|------------------|
| 1.7.2 | Must handle items where some bands have `None` URLs | Crash on missing bands |
| 1.7.2 | Must preserve existing file check logic for resume | Re-downloads complete files |
| 1.7.3 | Must use GDAL VRT creation (not just rasterio WarpedVRT) | Missing band metadata |
| 1.7.3 | Must match CRS of first band for all other bands | Silent projection errors |
| 1.7.5 | Must check for `.vrt` extension BEFORE opening with rasterio | VRT opened as raster |
| 1.7.5 | The `_identify_bands()` method in band_validator.py must be updated | Bands not recognized |

---

### 3.1.3 Edge Cases and Error Handling

#### Scenario Matrix

| Scenario | Expected Behavior | Test Case |
|----------|-------------------|-----------|
| **Network Errors** | | |
| Single band timeout | Retry 3x with exponential backoff (2s, 4s, 8s) | `test_band_download_timeout` |
| All bands timeout | Fail with `IngestionError("All band downloads failed")` | `test_all_bands_timeout` |
| Network drops mid-download | Detect via size mismatch, retry from start | `test_partial_download_recovery` |
| Rate limit (429) | Exponential backoff, max 60s delay | `test_rate_limit_handling` |
| **Data Errors** | | |
| Missing required band in STAC | Include in `missing_bands`, log warning, continue | `test_missing_required_band` |
| Missing optional band in STAC | Include in `missing_bands`, no warning | `test_missing_optional_band` |
| Band file is 0 bytes | Delete and retry, then fail if still 0 | `test_empty_band_file` |
| Band file is corrupt (not valid COG) | Fail validation with specific error | `test_corrupt_band_file` |
| **Resolution Mismatches** | | |
| 10m + 20m bands together | Resample 20m to 10m by default | `test_mixed_resolution_stack` |
| SWIR at 20m needs 10m output | Bilinear resampling | `test_resample_swir` |
| **CRS Mismatches** | | |
| All bands same UTM zone | No reprojection needed | `test_same_crs_stack` |
| Bands cross UTM zones | Reproject to first band's CRS | `test_cross_zone_stack` |
| One band in WGS84, rest UTM | Reproject WGS84 to UTM | `test_mixed_crs_stack` |
| **VRT Validation** | | |
| VRT references non-existent file | Fail with `VRTSourceNotFoundError` | `test_vrt_missing_source` |
| VRT band count != expected | Warning, not error (bands may be optional) | `test_vrt_band_count_mismatch` |
| VRT has no band descriptions | Fall back to positional matching | `test_vrt_no_descriptions` |

#### Error Message Standards

All error messages must include:
1. What failed (band name, URL, file path)
2. Why it failed (HTTP status, validation rule, exception type)
3. What to try (retry suggestion, workaround flag, documentation link)

```python
# GOOD error message:
f"Band '{band_name}' download failed from {url}: HTTP 503. "
f"Retried {retries}x. Try --retry-failed or check Earth Search status."

# BAD error message:
f"Download failed for {band_name}"
```

---

### 3.1.4 Test Fixtures Required

#### Mock STAC Response Fixtures

Create fixture file: `tests/fixtures/stac_responses/`

```python
# tests/fixtures/stac_responses/sentinel2_l2a_single_item.json
{
    "type": "Feature",
    "stac_version": "1.0.0",
    "id": "S2B_MSIL2A_20260125T103209_N0510_R108_T32UQD_20260125T141200",
    "geometry": {...},
    "bbox": [11.5, 46.0, 12.5, 47.0],
    "properties": {
        "datetime": "2026-01-25T10:32:09.000Z",
        "eo:cloud_cover": 5.2,
        "platform": "sentinel-2b",
        "instruments": ["msi"],
        "proj:epsg": 32632
    },
    "assets": {
        "blue": {
            "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B02.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized"
        },
        "green": {"href": "..."},
        "red": {"href": "..."},
        "nir": {"href": "..."},
        "swir16": {"href": "..."},
        "swir22": {"href": "..."},
        "visual": {
            "href": "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../TCI.tif",
            "type": "image/tiff; application=geotiff; profile=cloud-optimized"
        }
    }
}
```

#### Sample Band GeoTIFF Fixtures

Create fixture files: `tests/fixtures/bands/`

```
tests/fixtures/bands/
├── blue_10m.tif       # 100x100 pixels, 10m resolution, UInt16, EPSG:32632
├── green_10m.tif      # Same dimensions/CRS
├── red_10m.tif        # Same dimensions/CRS
├── nir_10m.tif        # Same dimensions/CRS
├── swir16_20m.tif     # 50x50 pixels, 20m resolution, UInt16, EPSG:32632
├── swir22_20m.tif     # Same as swir16
├── blue_corrupt.tif   # Invalid GeoTIFF header
└── blue_empty.tif     # Valid header, 0 bytes data
```

Generation script (add to `tests/fixtures/generate_band_fixtures.py`):

```python
def generate_test_bands():
    """Generate synthetic band files for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    bounds = (11.5, 46.0, 12.5, 47.0)  # WGS84
    crs = "EPSG:32632"

    # 10m bands (100x100)
    for band_name in ["blue", "green", "red", "nir"]:
        transform = from_bounds(*bounds, 100, 100)
        data = np.random.randint(0, 10000, (1, 100, 100), dtype=np.uint16)
        # Write with band description
        profile = {
            "driver": "GTiff",
            "dtype": "uint16",
            "width": 100,
            "height": 100,
            "count": 1,
            "crs": crs,
            "transform": transform,
        }
        with rasterio.open(f"{band_name}_10m.tif", "w", **profile) as dst:
            dst.write(data)
            dst.set_band_description(1, band_name)

    # 20m bands (50x50)
    for band_name in ["swir16", "swir22"]:
        transform = from_bounds(*bounds, 50, 50)
        data = np.random.randint(0, 10000, (1, 50, 50), dtype=np.uint16)
        profile["width"] = 50
        profile["height"] = 50
        profile["transform"] = transform
        with rasterio.open(f"{band_name}_20m.tif", "w", **profile) as dst:
            dst.write(data)
            dst.set_band_description(1, band_name)
```

#### Pre-built VRT Fixtures

```
tests/fixtures/vrts/
├── valid_6band.vrt       # References 6 band files, correct metadata
├── missing_source.vrt    # References non-existent file
├── no_descriptions.vrt   # Valid but no band descriptions
└── mixed_crs.vrt         # Bands in different CRS (edge case)
```

---

### 3.1.5 Rollback Procedures

#### Partial Implementation Rollback

If implementation must be stopped mid-way:

| Completed Tasks | Rollback Procedure |
|-----------------|-------------------|
| 1.7.1 only (STAC client) | No rollback needed - backward compat maintained via `url` field |
| 1.7.1 + 1.7.2 (downloads) | Rollback: Restore original `process_item()` from git |
| 1.7.1 + 1.7.2 + 1.7.3 (stacking) | Rollback: `band_stack.py` is isolated; delete file |
| 1.7.1-1.7.5 (all but tests) | Rollback: Use feature flag to disable multi-band path |

#### Feature Flag for Safe Deployment

Add to `config/ingestion.yaml`:

```yaml
multi_band_download:
  enabled: true  # Set false to use legacy single-URL path
  fallback_to_visual: true  # If band_urls empty, use visual asset
```

Implementation in `cli/commands/ingest.py`:

```python
def process_item(item: Dict, ...):
    config = load_ingestion_config()

    if config.multi_band_download.enabled and item.get("band_urls"):
        # New multi-band path
        return process_item_multiband(item, ...)
    elif item.get("url"):
        # Legacy single-file path
        return process_item_single(item, ...)
    elif config.multi_band_download.fallback_to_visual:
        # Fallback to TCI
        item["url"] = item.get("visualization_url")
        return process_item_single(item, ...)
    else:
        raise IngestionError(f"No download URL for item {item['id']}")
```

---

### 3.1.6 Configuration Precedence

When multiple configuration sources exist, use this precedence (highest wins):

1. **CLI flags** (e.g., `--bands blue,green,red`)
2. **Environment variables** (e.g., `FIRSTLIGHT_BANDS=blue,green,red`)
3. **config/ingestion.yaml**
4. **Hardcoded defaults in code**

Example configuration resolution:

```python
def resolve_bands_to_download(
    cli_bands: Optional[List[str]],
    env_bands: Optional[str],
    config_bands: List[str],
    default_bands: List[str],
) -> List[str]:
    """
    Resolve which bands to download with proper precedence.

    Returns list of lowercase band names.
    """
    if cli_bands:
        return [b.lower().strip() for b in cli_bands]

    if env_bands:
        return [b.lower().strip() for b in env_bands.split(",")]

    if config_bands:
        return config_bands

    return default_bands
```

---

### 3.1.7 Gherkin Acceptance Criteria (Missing from Main Spec)

The following scenarios were not explicitly defined in the main spec.

#### Feature: Band Download Progress Reporting

```gherkin
Feature: Band Download Progress Reporting
  As an operator watching ingestion
  I want to see per-band download progress
  So that I can estimate time remaining and identify slow bands

  Scenario: Normal multi-band download
    Given a Sentinel-2 item with 6 bands
    When I run `flight ingest` with default settings
    Then I should see progress for each band like:
      | Band   | Progress |
      | blue   | 45%      |
      | green  | 100%     |
      | red    | 78%      |
    And I should see overall progress like "3/6 bands complete (50%)"

  Scenario: One band fails
    Given a Sentinel-2 item where "swir22" URL returns 404
    When I run `flight ingest`
    Then I should see "swir22: FAILED (HTTP 404)" in progress output
    And I should see "5/6 bands complete, 1 failed"
    And the exit code should be non-zero
```

#### Feature: Resume Interrupted Downloads

```gherkin
Feature: Resume Interrupted Downloads
  As an operator with unreliable network
  I want to resume interrupted band downloads
  So that I don't re-download already complete bands

  Scenario: Resume after partial download
    Given a previous ingest attempt that downloaded blue, green, red
    And the download was interrupted during "nir"
    When I run `flight ingest` again for the same item
    Then blue, green, red should show "Already exists, skipping"
    And nir, swir16, swir22 should download normally
    And the VRT should be created with all 6 bands

  Scenario: Corrupt partial file detected
    Given a previous ingest that wrote 50% of "nir.tif"
    When I run `flight ingest` again
    Then "nir.tif" should be deleted and re-downloaded
    And a warning should be logged: "Partial file detected, re-downloading"
```

#### Feature: Visualization-Only Mode

```gherkin
Feature: Visualization-Only Download
  As an analyst who just needs a quick preview
  I want to download only the TCI composite
  So that I can see the area quickly without full band download

  Scenario: Download TCI only
    Given a Sentinel-2 item with TCI available
    When I run `flight ingest --visualization-only`
    Then only TCI.tif should be downloaded
    And validation should accept 3-band RGB file
    And a warning should be logged: "Visualization mode: spectral analysis not possible"

  Scenario: TCI not available
    Given a Sentinel-2 item without visual asset
    When I run `flight ingest --visualization-only`
    Then the command should fail with error
    And the error should say "No visualization asset available for this item"
```

---

### 3.1.8 File Location Summary

For engineers starting work, here's the exact file map:

| Component | File Path | Exists? | Action |
|-----------|-----------|---------|--------|
| STAC Client | `/core/data/discovery/stac_client.py` | Yes | Modify L439-450 |
| Band Config | `/core/data/discovery/band_config.py` | **No** | **Create** |
| Ingest Command | `/cli/commands/ingest.py` | Yes | Modify `process_item()` |
| Band Downloader | `/core/data/ingestion/band_downloader.py` | **No** | **Create** (optional, could be in ingest.py) |
| Band Stacker | `/core/data/ingestion/band_stack.py` | **No** | **Create** |
| Image Validator | `/core/data/ingestion/validation/image_validator.py` | Yes | Modify for VRT support |
| Band Validator | `/core/data/ingestion/validation/band_validator.py` | Yes | Modify `_identify_bands()` |
| Ingestion Config | `/config/ingestion.yaml` | Yes | Add `multi_band_download` section |
| Unit Tests | `/tests/test_band_download.py` | **No** | **Create** |
| Unit Tests | `/tests/test_band_stack.py` | **No** | **Create** |
| Integration Tests | `/tests/integration/test_sentinel2_ingestion.py` | **No** | **Create** |
| Fixtures | `/tests/fixtures/stac_responses/` | **No** | **Create directory** |
| Fixtures | `/tests/fixtures/bands/` | **No** | **Create directory** |

---

### 3.1.9 Existing Code to Reuse

Do NOT reinvent these existing utilities:

| Need | Existing Code | Location |
|------|---------------|----------|
| VRT creation from files | `STACVRTBuilder` patterns | `/core/data/ingestion/virtual_index.py` L200-400 |
| Chunked HTTP download | `StreamingDownloader` | `/core/data/ingestion/streaming.py` L100-250 |
| Retry with backoff | `RetryConfig` | `/core/data/ingestion/streaming.py` L50-80 |
| Band descriptions | `_get_band_descriptions()` | `/core/data/ingestion/validation/band_validator.py` L140-152 |
| CRS handling | Rasterio patterns | Throughout `/core/data/` |

---

### 3.1.10 Parallel Work Boundaries

For two engineers working simultaneously:

#### Engineer A: Download Path (Tasks 1.7.1, 1.7.2, 1.7.4)

**Scope:**
- Modify STAC client to return `band_urls`
- Implement band download logic in ingest.py
- Add `--skip-validation` flag

**Interface to Engineer B:**
- Must produce files in predictable locations: `{item_dir}/{band_name}.tif`
- Must write band descriptions to GeoTIFF metadata
- Must pass `Dict[str, Path]` to stacking function

**Can work independently on:**
- Download retry logic
- Progress reporting
- Resume detection

#### Engineer B: Stacking and Validation Path (Tasks 1.7.3, 1.7.5)

**Scope:**
- Create `band_stack.py` with VRT creation
- Update validators to handle VRT files

**Interface to Engineer A:**
- Expects `Dict[str, Path]` with band_name keys
- Expects GeoTIFF files with band descriptions set

**Can work independently on:**
- VRT XML generation
- Resolution resampling logic
- VRT-specific validation rules

**Sync Point:**
- Both engineers must agree on Contract C (band_paths structure) before starting
- Integration (Task 1.7.6) requires both paths complete

---

### 3.1.11 ROADMAP.md Additions

The following tasks should be added to ROADMAP.md Epic 1.7:

```markdown
| Task | Description | Status | Depends On |
|------|-------------|--------|------------|
| 1.7.0 | Review interface contracts in OPENSPEC addendum | [ ] | None |
| 1.7.1a | Create `/core/data/discovery/band_config.py` | [ ] | 1.7.0 |
| 1.7.1b | Modify `stac_client.py` discover_data() | [ ] | 1.7.1a |
| 1.7.2a | Add BandDownloadResult dataclass | [ ] | 1.7.0 |
| 1.7.2b | Implement download_bands() function | [ ] | 1.7.1b, 1.7.2a |
| 1.7.2c | Add resume detection logic | [ ] | 1.7.2b |
| 1.7.3a | Create band_stack.py with create_band_stack() | [ ] | 1.7.0 |
| 1.7.3b | Add mixed resolution handling | [ ] | 1.7.3a |
| 1.7.5a | Add VRT detection in image_validator.py | [ ] | 1.7.3a |
| 1.7.5b | Update _identify_bands() for VRT descriptions | [ ] | 1.7.5a |
| 1.7.6a | Create test fixtures (bands, VRTs, STAC responses) | [ ] | 1.7.0 |
| 1.7.6b | Write unit tests for band download | [ ] | 1.7.2c |
| 1.7.6c | Write unit tests for band stacking | [ ] | 1.7.3b |
| 1.7.6d | Write integration test with real STAC | [ ] | 1.7.5b |
```

---

### 3.1.12 Open Implementation Questions

Questions that implementers should resolve with Product/Tech Lead:

1. **Band download order:** Should bands download in alphabetical order, by resolution (10m first), or by file size (smallest first for quick progress feedback)?
   - **Recommendation:** By resolution (10m first) - gets most useful bands quickly.

2. **Checksum verification:** Should downloaded bands be verified against STAC item checksums (if available)?
   - **Recommendation:** Yes, if `file:checksum` is present in STAC asset metadata.

3. **Partial stack creation:** If 5 of 6 bands download successfully, should VRT be created with available bands?
   - **Recommendation:** Yes, with warning, unless the missing band is in `required_bands` config.

4. **Concurrent downloads:** What's the right default for parallel band downloads - 4, 6, or match band count?
   - **Recommendation:** 4 (balances throughput vs. rate limits).

5. **VRT vs GeoTIFF default:** Should we default to VRT (space efficient) or GeoTIFF (more portable)?
   - **Recommendation:** VRT, with CLI flag for GeoTIFF when needed.

---

## 3.2 Engineering Implementation Notes

**Date:** 2026-01-25
**Author:** Engineering Manager (Ratchet)
**Purpose:** Clarify gaps identified during implementation planning

---

### 3.2.1 VRT Source File Validation

**Gap:** How to validate VRT source files exist before opening with rasterio.

**Resolution:** Parse VRT XML directly before rasterio open:

```python
import xml.etree.ElementTree as ET
from pathlib import Path

def validate_vrt_sources(vrt_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate all VRT source files exist before opening.

    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
    tree = ET.parse(vrt_path)
    root = tree.getroot()
    vrt_dir = vrt_path.parent
    missing = []

    for source in root.iter('SourceFilename'):
        relative_to_vrt = source.get('relativeToVRT', '0') == '1'
        filename = source.text

        if relative_to_vrt:
            source_path = vrt_dir / filename
        else:
            source_path = Path(filename)

        if not source_path.exists():
            missing.append(str(source_path))

    return len(missing) == 0, missing
```

**Integration Point:** Call this in `image_validator.py` BEFORE `rasterio.open()` for `.vrt` files.

---

### 3.2.2 Band Name Case Sensitivity

**Gap:** Case sensitivity of band names not explicitly defined.

**Resolution:**
- **STAC asset keys:** Lowercase (e.g., `blue`, `nir`, `swir16`) - this is the STAC convention
- **VRT band descriptions:** Store as lowercase for consistency
- **Validation matching:** Case-insensitive comparison with `.lower()`
- **Internal representation:** Always lowercase

**Implementation Rule:** All band name comparisons must use `.lower()`:

```python
# CORRECT
if band_name.lower() in KNOWN_BANDS:
    ...

# INCORRECT - will miss "Blue" or "BLUE"
if band_name in KNOWN_BANDS:
    ...
```

---

### 3.2.3 STAC Authentication

**Gap:** What happens when STAC catalog requires authentication?

**Resolution:** This is explicitly OUT OF SCOPE for this epic. However, for robustness:

1. If authentication is required, the STAC client will raise `pystac_client.exceptions.APIError`
2. The error message should be surfaced to the user with guidance:
   ```
   STAC catalog requires authentication. This catalog is not supported.
   Public catalogs (Earth Search, Planetary Computer) do not require auth.
   ```
3. Do NOT attempt to handle auth in this epic - defer to future work.

---

### 3.2.4 Checksum Verification

**Gap:** Which checksum algorithm to use, and fallback behavior.

**Resolution:**
- **Primary:** Use `file:checksum` from STAC asset metadata if present
- **Algorithm Detection:** Parse the checksum string format:
  - `md5:abc123...` -> MD5
  - `sha256:abc123...` -> SHA256
  - `multihash:...` -> Parse multihash header
- **Fallback:** If no checksum available, skip verification (log warning)
- **Mismatch:** If checksum fails, delete file and retry download once

```python
def verify_checksum(file_path: Path, stac_checksum: Optional[str]) -> bool:
    """Verify file checksum against STAC metadata."""
    if not stac_checksum:
        logger.warning(f"No checksum available for {file_path}, skipping verification")
        return True

    if stac_checksum.startswith('md5:'):
        expected = stac_checksum[4:]
        actual = hashlib.md5(file_path.read_bytes()).hexdigest()
    elif stac_checksum.startswith('sha256:'):
        expected = stac_checksum[7:]
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
    else:
        logger.warning(f"Unknown checksum format: {stac_checksum}")
        return True

    return actual.lower() == expected.lower()
```

---

### 3.2.5 Canonical Band Order for VRT

**Gap:** What order should bands appear in VRT for Sentinel-2?

**Resolution:** Use wavelength order (shortest to longest) as the canonical order:

| Position | Band Name | Wavelength (nm) | Sentinel-2 Band |
|----------|-----------|-----------------|-----------------|
| 1 | blue | 490 | B02 |
| 2 | green | 560 | B03 |
| 3 | red | 665 | B04 |
| 4 | nir | 842 | B08 |
| 5 | swir16 | 1610 | B11 |
| 6 | swir22 | 2190 | B12 |

**Constant Definition:**

```python
# core/data/discovery/band_config.py

SENTINEL2_CANONICAL_ORDER = ["blue", "green", "red", "nir", "swir16", "swir22"]

LANDSAT_CANONICAL_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]

SENTINEL1_CANONICAL_ORDER = ["vv", "vh"]
```

This order matches the positional fallback in `band_validator.py` (blue=1, green=2, red=3, nir=4).

---

### 3.2.6 Sentinel-1 Handling

**Gap:** How does multi-band flow handle Sentinel-1 (SAR) data?

**Resolution:** Sentinel-1 is fundamentally different:

1. **Asset Keys:** `vv`, `vh` (polarizations, not spectral bands)
2. **Validation:** Uses `SARValidator`, not `BandValidator`
3. **Stacking:** Two bands (VV, VH) instead of 6

**STAC Client Update:**

```python
def _extract_band_urls(self, item, source: str) -> Dict[str, str]:
    """Extract band URLs based on sensor type."""
    if source == "sentinel2":
        band_keys = SENTINEL2_BANDS.keys()
    elif source == "sentinel1":
        band_keys = ["vv", "vh"]
    elif source == "landsat":
        band_keys = LANDSAT_BANDS.keys()
    else:
        # Unknown sensor - return empty, fall back to visual/data
        return {}

    return {
        band: item.assets[band].href
        for band in band_keys
        if band in item.assets and hasattr(item.assets[band], 'href')
    }
```

---

### 3.2.7 Error Message Template

**Gap:** Consistent error messages for multi-band failures.

**Resolution:** All error messages must follow this pattern:

```
[COMPONENT] [OPERATION] failed for [TARGET]: [REASON]. [ACTION]

Examples:
- "Band download failed for 'nir' from https://...: HTTP 503 Service Unavailable. Retried 3 times. Try again later or use --skip-validation."
- "Band stacking failed for scene S2B_...: CRS mismatch between bands (blue=EPSG:32632, nir=EPSG:32633). Contact support."
- "VRT validation failed for /path/to/stack.vrt: Source file '/path/to/blue.tif' not found. Re-run ingest."
```

---

### 3.2.8 Test Data Generation Script

**Gap:** Need clear instructions for generating test fixtures.

**Resolution:** Create script at `tests/fixtures/generate_band_fixtures.py`:

```python
#!/usr/bin/env python3
"""Generate synthetic band files for testing multi-band ingestion."""

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path

def main():
    output_dir = Path(__file__).parent / "bands"
    output_dir.mkdir(exist_ok=True)

    # Common parameters
    bounds = (11.5, 46.0, 12.5, 47.0)  # WGS84 bbox
    crs = "EPSG:32632"  # UTM zone 32N

    # 10m resolution bands (100x100 pixels for test)
    for band_name in ["blue", "green", "red", "nir"]:
        create_test_band(output_dir / f"{band_name}_10m.tif",
                        bounds, crs, 100, 100, band_name)

    # 20m resolution bands (50x50 pixels for test)
    for band_name in ["swir16", "swir22"]:
        create_test_band(output_dir / f"{band_name}_20m.tif",
                        bounds, crs, 50, 50, band_name)

    # Corrupt file for error testing
    create_corrupt_file(output_dir / "blue_corrupt.tif")

    print(f"Generated test fixtures in {output_dir}")

def create_test_band(path, bounds, crs, width, height, band_name):
    """Create a synthetic single-band GeoTIFF."""
    transform = from_bounds(*bounds, width, height)
    data = np.random.randint(0, 10000, (1, height, width), dtype=np.uint16)

    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        dst.set_band_description(1, band_name)

def create_corrupt_file(path):
    """Create a file with invalid GeoTIFF header."""
    with open(path, "wb") as f:
        f.write(b"NOT A VALID GEOTIFF")

if __name__ == "__main__":
    main()
```

Run with: `python tests/fixtures/generate_band_fixtures.py`

---

### 3.2.9 Resume Detection Algorithm

**Gap:** How to detect partial downloads for resume capability.

**Resolution:**

```python
def is_download_complete(file_path: Path, expected_size: Optional[int]) -> bool:
    """
    Check if a downloaded file is complete.

    A file is considered complete if:
    1. It exists
    2. Size matches expected (if known)
    3. It's a valid GeoTIFF (can be opened)
    """
    if not file_path.exists():
        return False

    actual_size = file_path.stat().st_size

    # Size check
    if expected_size and actual_size != expected_size:
        logger.info(f"Partial download detected: {actual_size}/{expected_size} bytes")
        return False

    # Validity check - try to open with rasterio
    try:
        with rasterio.open(file_path) as ds:
            # Check we can read metadata
            _ = ds.count
            return True
    except Exception as e:
        logger.info(f"Invalid file detected: {e}")
        return False
```

---

### Revision History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-25 | 1.0 | Initial specification |
| 2026-01-25 | 1.1 | Added Implementation Addendum (Section 3.1) |
| 2026-01-25 | 1.2 | Added Engineering Implementation Notes (Section 3.2) |
