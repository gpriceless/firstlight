# Work Assignments: Multi-Band Asset Download (Epic 1.7)

**Feature Branch:** `feature/multi-band-asset-download`
**Created:** 2026-01-25
**Engineering Manager:** Ratchet

---

## Overview

This document defines the work breakdown for Epic 1.7: Multi-Band Asset Download. The work is organized into three parallel tracks to maximize throughput while respecting dependencies.

```
                     TRACK DEPENDENCY DIAGRAM

    ┌─────────────────┐
    │  TRACK A        │ ◀── Can start IMMEDIATELY (independent)
    │  Workaround     │
    │  (1.7.4)        │
    └────────┬────────┘
             │
             │ (ships independently, no blocker)
             │
    ┌────────┴────────────────────────────────────────────┐
    │                                                      │
    │   ┌─────────────────┐                               │
    │   │  TRACK B        │ ◀── Sequential core work      │
    │   │  Core Flow      │                               │
    │   │                 │                               │
    │   │  1.7.1 STAC ────┼───┐                           │
    │   │       │         │   │                           │
    │   │       ▼         │   │                           │
    │   │  1.7.2 Ingest ──┼───┼───┐                       │
    │   │       │         │   │   │                       │
    │   │       ▼         │   │   │                       │
    │   │  1.7.5 Valid ───┼───┼───┼───┐                   │
    │   │       │         │   │   │   │                   │
    │   │       ▼         │   │   │   │                   │
    │   │  1.7.6 Tests    │   │   │   │                   │
    │   └─────────────────┘   │   │   │                   │
    │                         │   │   │                   │
    │   ┌─────────────────┐   │   │   │                   │
    │   │  TRACK C        │◀──┘   │   │                   │
    │   │  Band Stacking  │       │   │                   │
    │   │                 │       │   │                   │
    │   │  1.7.3 Stack ───┼───────┘   │                   │
    │   │  (VRT creation) │           │                   │
    │   └─────────────────┘           │                   │
    │                                 │                   │
    └─────────────────────────────────┘                   │
                                                          │
    INTEGRATION ◀─────────────────────────────────────────┘
    (All tracks merge at Task 1.7.5)
```

---

## Track A: Independent Workaround

**Assigned Agent:** Coder Agent
**Priority:** P0 - Ship immediately
**Can Start:** NOW (no dependencies)
**Can Ship:** Independently (doesn't need Tracks B/C)

### Task 1.7.4: Add --skip-validation Flag

**Purpose:** Provide immediate workaround for users blocked on Sentinel-2 ingestion.

#### Scope

Add a `--skip-validation` flag to the ingest command that bypasses image validation. This allows users to download TCI files for visualization while the proper multi-band fix is implemented.

#### Files to Modify

| File | Changes |
|------|---------|
| `/home/gprice/projects/firstlight/cli/commands/ingest.py` | Add `--skip-validation` flag, wire to skip validation logic |
| `/home/gprice/projects/firstlight/core/data/ingestion/streaming.py` | Add `skip_validation` parameter to `ingest()` method |

#### Implementation Details

**1. CLI Flag (ingest.py)**

```python
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip image validation (WARNING: may download unusable files)"
)
def ingest(..., skip_validation: bool):
    if skip_validation:
        console.print(
            "[yellow]WARNING: Validation skipped. Downloaded files may not be "
            "suitable for spectral analysis (missing NIR/SWIR bands).[/yellow]"
        )
```

**2. StreamingIngester Parameter**

```python
def ingest(
    self,
    source_url: str,
    ...,
    skip_validation: bool = False,
) -> Dict[str, Any]:
    ...
    if not skip_validation:
        validation_result = self._validate_image(...)
```

#### Definition of Done

- [ ] `flight ingest --skip-validation` flag is recognized
- [ ] Warning message displayed when flag is used
- [ ] Validation is actually skipped (TCI downloads succeed)
- [ ] Help text explains the trade-off
- [ ] Unit test for flag behavior

#### How to Test

```bash
# Manual test
flight ingest --area tests/fixtures/areas/small_area.geojson \
    --event flood.coastal \
    --skip-validation

# Expected: Download succeeds (even if TCI file)
# Expected: Warning message displayed

# Unit test
pytest tests/test_cli_ingest.py -k "skip_validation" -v
```

#### Estimated Effort

Low (0.5 day)

---

## Track B: Sequential Core Implementation

**Assigned Agent:** Project Manager + Coder Agents
**Priority:** P0 - Critical path
**Can Start:** NOW (Task 1.7.1)
**Ships:** With Track C completion

### Task 1.7.1: STAC Client Band URL Extraction

**Purpose:** Update STAC client to return individual band URLs instead of single composite URL.

#### Scope

Modify `discover_data()` to extract URLs for each spectral band and return them in a new `band_urls` field. Maintain backward compatibility with the existing `url` field.

#### Files to Create/Modify

| File | Action | Changes |
|------|--------|---------|
| `/home/gprice/projects/firstlight/core/data/discovery/band_config.py` | CREATE | Band configuration constants |
| `/home/gprice/projects/firstlight/core/data/discovery/stac_client.py` | MODIFY | Add band URL extraction (L439-450) |

#### Implementation Details

**1. Create band_config.py:**

```python
"""
Band configuration for satellite sensors.

Defines which bands to extract from STAC assets for each sensor type.
"""

from dataclasses import dataclass
from typing import Dict, List

# Sentinel-2 L2A band configuration
SENTINEL2_BANDS = {
    "blue": "blue",       # B02, 490nm, 10m
    "green": "green",     # B03, 560nm, 10m
    "red": "red",         # B04, 665nm, 10m
    "nir": "nir",         # B08, 842nm, 10m
    "swir16": "swir16",   # B11, 1610nm, 20m
    "swir22": "swir22",   # B12, 2190nm, 20m
}

SENTINEL2_CANONICAL_ORDER = ["blue", "green", "red", "nir", "swir16", "swir22"]

# Landsat 8/9 band configuration
LANDSAT_BANDS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
}

LANDSAT_CANONICAL_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Sentinel-1 (SAR) configuration
SENTINEL1_BANDS = {
    "vv": "vv",
    "vh": "vh",
}

SENTINEL1_CANONICAL_ORDER = ["vv", "vh"]

def get_band_config(source: str) -> Dict[str, str]:
    """Get band configuration for a sensor type."""
    configs = {
        "sentinel2": SENTINEL2_BANDS,
        "landsat": LANDSAT_BANDS,
        "sentinel1": SENTINEL1_BANDS,
    }
    return configs.get(source, {})

def get_canonical_order(source: str) -> List[str]:
    """Get canonical band order for a sensor type."""
    orders = {
        "sentinel2": SENTINEL2_CANONICAL_ORDER,
        "landsat": LANDSAT_CANONICAL_ORDER,
        "sentinel1": SENTINEL1_CANONICAL_ORDER,
    }
    return orders.get(source, [])
```

**2. Modify stac_client.py (around line 439-450):**

```python
from core.data.discovery.band_config import get_band_config

def _extract_band_urls(self, item, source: str) -> Dict[str, str]:
    """Extract individual band URLs from STAC item assets."""
    band_config = get_band_config(source)
    band_urls = {}

    for band_name, asset_key in band_config.items():
        if asset_key in item.assets:
            asset = item.assets[asset_key]
            if hasattr(asset, 'href'):
                band_urls[band_name] = asset.href

    return band_urls

# In discover_data(), update the result dictionary:
results.append({
    "id": item.id,
    "source": source,
    "datetime": item.datetime.isoformat(),
    "cloud_cover": item.cloud_cover,
    "resolution_m": 10.0 if source in ["sentinel1", "sentinel2"] else 30.0,
    "size_bytes": size_estimate,
    "url": item.assets.get("blue", {}).get("href") or item.assets.get("visual", {}).get("href") or "",  # Backward compat
    "band_urls": self._extract_band_urls(item, source),  # NEW
    "visualization_url": item.assets.get("visual", {}).get("href"),  # NEW
    "missing_bands": [],  # NEW - populated if expected bands missing
    "priority": "primary" if source in ["sentinel1", "sentinel2"] else "secondary",
    "bbox": item.bbox,
    "assets": item.assets,
})
```

#### Definition of Done

- [ ] `band_config.py` created with Sentinel-2, Landsat, Sentinel-1 configs
- [ ] `stac_client.py` returns `band_urls` dict for Sentinel-2 items
- [ ] Backward compatibility: `url` field still populated (with blue band or TCI fallback)
- [ ] `visualization_url` field populated with TCI asset
- [ ] Unit tests for band URL extraction
- [ ] Existing STAC client tests still pass

#### How to Test

```bash
# Unit test for band extraction
pytest tests/test_stac_client.py -k "band_urls" -v

# Manual integration test
flight discover --area tests/fixtures/areas/small_area.geojson \
    --event flood.coastal \
    --start-date 2026-01-01 \
    --end-date 2026-01-25 \
    --output json

# Expected: JSON output includes "band_urls" with blue, green, red, nir, swir16, swir22
```

#### Estimated Effort

Medium (1-2 days)

---

### Task 1.7.2: Ingestion Pipeline Multi-Band Download

**Purpose:** Update ingestion to download each band file separately.

**Depends On:** Task 1.7.1 (needs `band_urls` in discovery results)

#### Scope

Modify `process_item()` to iterate over `band_urls`, download each band to a separate file, and coordinate parallel downloads.

#### Files to Modify

| File | Changes |
|------|---------|
| `/home/gprice/projects/firstlight/cli/commands/ingest.py` | Update `process_item()` for multi-band downloads |
| `/home/gprice/projects/firstlight/core/data/ingestion/streaming.py` | Add `download_band()` and resume detection |

#### Implementation Details

**1. New download_bands() function in ingest.py:**

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

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

def download_bands(
    band_urls: Dict[str, str],
    output_dir: Path,
    item_id: str,
    parallel_downloads: int = 4,
    max_retries: int = 3,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, BandDownloadResult]:
    """
    Download multiple band files for a single scene.

    Args:
        band_urls: Mapping of band name to URL
        output_dir: Directory to save band files
        item_id: Item ID for logging
        parallel_downloads: Number of concurrent downloads
        max_retries: Retry attempts per band
        progress_callback: Called with (band_name, status, progress)

    Returns:
        Mapping of band name to download result
    """
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_single_band(band_name: str, url: str) -> BandDownloadResult:
        band_path = output_dir / f"{band_name}.tif"
        start_time = time.time()

        # Check for existing complete file
        if is_download_complete(band_path, expected_size=None):
            return BandDownloadResult(
                band_name=band_name,
                url=url,
                local_path=band_path,
                size_bytes=band_path.stat().st_size,
                download_time_s=0,
                success=True,
            )

        # Download with retries
        for attempt in range(max_retries):
            try:
                _download_real_data(url, band_path, {})
                elapsed = time.time() - start_time

                if progress_callback:
                    progress_callback(band_name, "complete", 1.0)

                return BandDownloadResult(
                    band_name=band_name,
                    url=url,
                    local_path=band_path,
                    size_bytes=band_path.stat().st_size,
                    download_time_s=elapsed,
                    success=True,
                    retries=attempt,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return BandDownloadResult(
                    band_name=band_name,
                    url=url,
                    local_path=None,
                    size_bytes=0,
                    download_time_s=time.time() - start_time,
                    success=False,
                    error=str(e),
                    retries=attempt + 1,
                )

    # Download bands in parallel
    with ThreadPoolExecutor(max_workers=parallel_downloads) as executor:
        futures = {
            executor.submit(download_single_band, name, url): name
            for name, url in band_urls.items()
        }

        for future in as_completed(futures):
            band_name = futures[future]
            results[band_name] = future.result()

    return results
```

**2. Update process_item() to use multi-band path:**

```python
def process_item(item: Dict, output_path: Path, ...) -> bool:
    band_urls = item.get("band_urls", {})

    if band_urls:
        # Multi-band download path (NEW)
        item_dir = output_path / item["id"]
        band_results = download_bands(band_urls, item_dir, item["id"])

        # Check for failures
        failed_bands = [r for r in band_results.values() if not r.success]
        if failed_bands:
            for r in failed_bands:
                console.print(f"[red]Failed to download {r.band_name}: {r.error}[/red]")
            return False

        # Get paths for successful downloads
        band_paths = {
            name: result.local_path
            for name, result in band_results.items()
            if result.success and result.local_path
        }

        # Create band stack (calls Track C implementation)
        stack_path = item_dir / f"{item['id']}_stack.vrt"
        from core.data.ingestion.band_stack import create_band_stack
        stack_result = create_band_stack(band_paths, stack_path)

        # Validate the stack
        if not _validate_downloaded_image(stack_result.path, item):
            return False

        return True

    else:
        # Legacy single-URL path (backward compat)
        url = item.get("url")
        if url:
            return download_single_file(url, output_path, item)
        return False
```

#### Definition of Done

- [ ] `BandDownloadResult` dataclass defined
- [ ] `download_bands()` function implements parallel downloads
- [ ] `process_item()` uses multi-band path when `band_urls` present
- [ ] Progress reporting shows per-band status
- [ ] Retry logic works with exponential backoff
- [ ] Resume detection skips complete files
- [ ] Legacy single-URL path still works

#### How to Test

```bash
# Unit test for download_bands
pytest tests/test_band_download.py -v

# Integration test (requires network)
flight ingest --area tests/fixtures/areas/small_area.geojson \
    --event flood.coastal \
    --parallel-downloads 4 \
    --output /tmp/ingest_test

# Check output structure
ls /tmp/ingest_test/S2*/
# Expected: blue.tif, green.tif, red.tif, nir.tif, swir16.tif, swir22.tif, *_stack.vrt
```

#### Estimated Effort

Medium-High (2-3 days)

---

### Task 1.7.5: Validator Updates for VRT

**Purpose:** Update image validator to handle VRT stacked files.

**Depends On:** Tasks 1.7.2 and 1.7.3 (needs stacked VRT files to validate)

#### Scope

Update `ImageValidator` and `BandValidator` to:
1. Detect VRT files by extension
2. Validate VRT XML structure and source file existence
3. Extract band names from VRT Description tags
4. Apply band validation to each band in the stack

#### Files to Modify

| File | Changes |
|------|---------|
| `/home/gprice/projects/firstlight/core/data/ingestion/validation/image_validator.py` | Add VRT detection and source validation |
| `/home/gprice/projects/firstlight/core/data/ingestion/validation/band_validator.py` | Update `_match_bands()` for VRT descriptions |

#### Implementation Details

**1. VRT detection in image_validator.py:**

```python
import xml.etree.ElementTree as ET

def _validate_vrt_sources(self, vrt_path: Path) -> Tuple[bool, List[str]]:
    """Validate all VRT source files exist."""
    try:
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
    except ET.ParseError as e:
        return False, [f"Invalid VRT XML: {e}"]

def validate(self, raster_path: Union[str, Path], ...):
    raster_path = Path(raster_path)

    # VRT-specific pre-validation
    if raster_path.suffix.lower() == '.vrt':
        sources_valid, missing = self._validate_vrt_sources(raster_path)
        if not sources_valid:
            result.is_valid = False
            result.errors.extend([
                f"VRT source file not found: {m}" for m in missing
            ])
            return result

    # Continue with normal validation...
```

**2. Update _match_bands() in band_validator.py:**

Add VRT Description tag as the first matching method (highest priority):

```python
def _match_bands(
    self,
    expected_bands: Dict[str, List[str]],
    band_descriptions: Dict[int, str],
    band_count: int,
) -> Dict[str, int]:
    mapping = {}

    # Priority 1: Direct match on VRT Description (case-insensitive)
    for generic_name in expected_bands.keys():
        for band_idx, desc in band_descriptions.items():
            if desc and desc.lower() == generic_name.lower():
                mapping[generic_name] = band_idx
                break

    # Priority 2: Match by possible sensor band names
    for generic_name, possible_names in expected_bands.items():
        if generic_name in mapping:
            continue
        for band_idx, desc in band_descriptions.items():
            if desc:
                desc_upper = desc.upper()
                for possible in possible_names:
                    if possible.upper() in desc_upper:
                        mapping[generic_name] = band_idx
                        break
            if generic_name in mapping:
                break

    # Priority 3: Positional fallback (existing logic)
    # ...
```

#### Definition of Done

- [ ] VRT files detected by `.vrt` extension
- [ ] VRT XML parsed to validate source file existence
- [ ] Band names extracted from VRT Description tags
- [ ] Band validation works for stacked VRT files
- [ ] Clear error messages for missing sources
- [ ] Unit tests for VRT validation

#### How to Test

```bash
# Unit test for VRT validation
pytest tests/test_vrt_validation.py -v

# Test with generated VRT fixture
python -c "
from core.data.ingestion.validation.image_validator import ImageValidator
from core.data.ingestion.validation.config import ValidationConfig

config = ValidationConfig()
validator = ImageValidator(config)
result = validator.validate('/tmp/test_stack.vrt')
print(result)
"
```

#### Estimated Effort

Low-Medium (1-2 days)

---

### Task 1.7.6: Integration Testing

**Purpose:** Verify end-to-end multi-band ingestion works with real STAC catalogs.

**Depends On:** Tasks 1.7.1-1.7.5 (all core implementation complete)

#### Scope

Create comprehensive integration tests that verify:
1. Full discovery -> download -> stack -> validate flow
2. Real Sentinel-2 data from Earth Search
3. Algorithm execution on ingested data
4. Error handling for edge cases

#### Files to Create

| File | Purpose |
|------|---------|
| `/home/gprice/projects/firstlight/tests/integration/test_sentinel2_ingestion.py` | Main integration tests |
| `/home/gprice/projects/firstlight/tests/fixtures/stac_responses/sentinel2_l2a_single_item.json` | Mock STAC response |

#### Implementation Details

```python
# tests/integration/test_sentinel2_ingestion.py
"""Integration tests for multi-band Sentinel-2 ingestion."""

import pytest
from pathlib import Path
import tempfile

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestSentinel2Ingestion:
    """Test complete Sentinel-2 ingestion flow."""

    @pytest.fixture
    def temp_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_discover_returns_band_urls(self, small_area_geojson):
        """Discovery should return individual band URLs for Sentinel-2."""
        from core.data.discovery.stac_client import discover_data

        results = discover_data(
            area=small_area_geojson,
            event_type="flood.coastal",
            start_date="2026-01-01",
            end_date="2026-01-25",
        )

        # Filter to Sentinel-2 only
        s2_results = [r for r in results if r["source"] == "sentinel2"]

        if s2_results:
            item = s2_results[0]
            assert "band_urls" in item
            assert "blue" in item["band_urls"]
            assert "nir" in item["band_urls"]
            assert "swir16" in item["band_urls"]

    def test_full_ingest_flow(self, small_area_geojson, temp_output_dir):
        """Full ingestion should download all bands and create stack."""
        from cli.commands.ingest import process_item
        from core.data.discovery.stac_client import discover_data

        # Discover
        results = discover_data(
            area=small_area_geojson,
            event_type="flood.coastal",
            start_date="2026-01-01",
            end_date="2026-01-25",
        )

        s2_results = [r for r in results if r["source"] == "sentinel2"]
        if not s2_results:
            pytest.skip("No Sentinel-2 data available in time range")

        # Ingest first item
        item = s2_results[0]
        success = process_item(item, temp_output_dir)

        assert success, "Ingestion should succeed"

        # Verify output structure
        item_dir = temp_output_dir / item["id"]
        assert item_dir.exists()
        assert (item_dir / "blue.tif").exists()
        assert (item_dir / "nir.tif").exists()
        assert len(list(item_dir.glob("*_stack.vrt"))) == 1

    def test_ndwi_on_ingested_data(self, small_area_geojson, temp_output_dir):
        """NDWI algorithm should work on ingested multi-band data."""
        # Similar to above, but also run NDWI
        # This proves NIR band is present and valid
        pass

    def test_retry_on_partial_failure(self, temp_output_dir):
        """Partial download failure should retry failed bands."""
        pass

    def test_resume_skips_complete_files(self, temp_output_dir):
        """Resume should skip already-downloaded files."""
        pass
```

#### Definition of Done

- [ ] Integration test file created
- [ ] Tests cover discovery -> download -> stack -> validate flow
- [ ] Tests verify NDWI works (proves NIR band present)
- [ ] Tests verify error handling
- [ ] Tests can run with VCR recordings for CI (no network required)
- [ ] All tests pass

#### How to Test

```bash
# Run integration tests (requires network)
pytest tests/integration/test_sentinel2_ingestion.py -v

# Run with VCR recording (creates fixtures for CI)
pytest tests/integration/test_sentinel2_ingestion.py -v --vcr-record=new_episodes

# Run in CI (uses recorded fixtures)
pytest tests/integration/test_sentinel2_ingestion.py -v
```

#### Estimated Effort

Medium (2-3 days)

---

## Track C: Band Stacking (Parallel with Track B after 1.7.1)

**Assigned Agent:** Coder Agent
**Priority:** P0 - Required for core flow
**Can Start:** After Task 1.7.1 completes
**Ships:** With Track B (feeds into Task 1.7.5)

### Task 1.7.3: Band Stacking Utility (VRT Creation)

**Purpose:** Create utility to stack individual band files into a single VRT.

**Depends On:** Task 1.7.1 (needs band config for canonical order)

#### Scope

Create a new module that combines individual band GeoTIFF files into a GDAL Virtual Raster (VRT) with proper band ordering and metadata.

#### Files to Create

| File | Purpose |
|------|---------|
| `/home/gprice/projects/firstlight/core/data/ingestion/band_stack.py` | Band stacking utility |
| `/home/gprice/projects/firstlight/tests/test_band_stack.py` | Unit tests |

#### Implementation Details

```python
# core/data/ingestion/band_stack.py
"""
Band Stacking Utility for Multi-Band Raster Creation.

Creates GDAL Virtual Rasters (VRT) from individual band files,
combining them into a single multi-band analysis-ready file.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional imports
try:
    import rasterio
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False

try:
    from osgeo import gdal
    gdal.UseExceptions()
    HAS_GDAL = True
except ImportError:
    gdal = None
    HAS_GDAL = False


class BandStackError(Exception):
    """Base exception for band stacking errors."""
    pass


class CRSMismatchError(BandStackError):
    """Bands have incompatible coordinate reference systems."""
    pass


class ExtentMismatchError(BandStackError):
    """Bands don't have overlapping extents."""
    pass


@dataclass
class StackResult:
    """Result from band stacking operation."""
    path: Path
    band_count: int
    band_mapping: Dict[str, int]  # band_name -> 1-indexed position
    crs: str
    bounds: Tuple[float, float, float, float]
    resolution: Tuple[float, float]
    warnings: List[str] = field(default_factory=list)
    source_files: Dict[str, Path] = field(default_factory=dict)


def create_band_stack(
    band_paths: Dict[str, Path],
    output_path: Path,
    band_order: Optional[List[str]] = None,
    band_descriptions: Optional[Dict[str, str]] = None,
    resolution_mode: str = "resample",
    target_resolution: Optional[float] = None,
    resample_method: str = "bilinear",
) -> StackResult:
    """
    Create a virtual raster (VRT) stacking individual band files.

    The VRT file references the original band files without duplicating data,
    providing a single multi-band interface for downstream processing.

    Args:
        band_paths: Keys are lowercase band names (e.g., "blue", "nir")
                   Values are absolute Path objects to .tif files
        output_path: Must end in .vrt; parent directory must exist
        band_order: Order of bands in output VRT; default by canonical order
        band_descriptions: Override band descriptions in VRT metadata
        resolution_mode:
            - "resample": Resample all bands to target_resolution (default)
            - "preserve": Keep native resolutions
            - "intersect": Crop to intersection of all extents
        target_resolution: Target resolution in meters (default: 10m for Sentinel-2)
        resample_method: GDAL resampling method name

    Returns:
        StackResult with path, band_mapping, and metadata

    Raises:
        BandStackError: Base exception for stacking failures
        CRSMismatchError: Bands have incompatible CRS
        ExtentMismatchError: Bands don't overlap
    """
    if not HAS_GDAL:
        raise BandStackError("GDAL is required for VRT creation")

    if not band_paths:
        raise BandStackError("No band files provided")

    output_path = Path(output_path)
    if output_path.suffix.lower() != '.vrt':
        raise BandStackError(f"Output must be .vrt file, got: {output_path.suffix}")

    # Default band order: canonical Sentinel-2 order, then alphabetical for unknown
    if band_order is None:
        from core.data.discovery.band_config import SENTINEL2_CANONICAL_ORDER
        band_order = [b for b in SENTINEL2_CANONICAL_ORDER if b in band_paths]
        band_order += sorted([b for b in band_paths.keys() if b not in band_order])

    warnings = []

    # Validate and collect band metadata
    band_info = {}
    reference_crs = None
    reference_bounds = None

    for band_name in band_order:
        if band_name not in band_paths:
            warnings.append(f"Band '{band_name}' in order but not in paths")
            continue

        band_path = band_paths[band_name]
        if not band_path.exists():
            raise BandStackError(f"Band file not found: {band_path}")

        with rasterio.open(band_path) as src:
            band_info[band_name] = {
                "path": band_path,
                "crs": src.crs,
                "bounds": src.bounds,
                "resolution": src.res,
                "width": src.width,
                "height": src.height,
                "dtype": src.dtypes[0],
            }

            if reference_crs is None:
                reference_crs = src.crs
                reference_bounds = src.bounds
            elif src.crs != reference_crs:
                warnings.append(
                    f"CRS mismatch for {band_name}: {src.crs} vs {reference_crs}"
                )

    if not band_info:
        raise BandStackError("No valid band files found")

    # Build VRT using GDAL
    vrt_options = gdal.BuildVRTOptions(
        separate=True,  # Separate bands (not mosaic)
        resolution='highest' if resolution_mode == 'resample' else 'average',
        resampleAlg=resample_method,
    )

    # Order band files according to band_order
    ordered_paths = [str(band_info[b]["path"]) for b in band_order if b in band_info]

    vrt_ds = gdal.BuildVRT(str(output_path), ordered_paths, options=vrt_options)

    if vrt_ds is None:
        raise BandStackError(f"GDAL BuildVRT failed for {output_path}")

    # Set band descriptions
    for i, band_name in enumerate([b for b in band_order if b in band_info], 1):
        band = vrt_ds.GetRasterBand(i)
        desc = band_descriptions.get(band_name, band_name) if band_descriptions else band_name
        band.SetDescription(desc)

    vrt_ds.FlushCache()
    vrt_ds = None  # Close dataset

    # Build result
    band_mapping = {
        band_name: i + 1
        for i, band_name in enumerate([b for b in band_order if b in band_info])
    }

    return StackResult(
        path=output_path,
        band_count=len(band_mapping),
        band_mapping=band_mapping,
        crs=str(reference_crs),
        bounds=(
            reference_bounds.left,
            reference_bounds.bottom,
            reference_bounds.right,
            reference_bounds.top,
        ),
        resolution=band_info[list(band_info.keys())[0]]["resolution"],
        warnings=warnings,
        source_files={b: info["path"] for b, info in band_info.items()},
    )


def stack_to_geotiff(
    band_paths: Dict[str, Path],
    output_path: Path,
    band_order: Optional[List[str]] = None,
    compress: str = "LZW",
    tiled: bool = True,
) -> StackResult:
    """
    Create a multi-band GeoTIFF from individual bands.

    Note: This copies all data, increasing storage requirements.
    Use create_band_stack() (VRT) when possible for space efficiency.
    """
    # First create VRT, then translate to GeoTIFF
    with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as tmp:
        tmp_vrt = Path(tmp.name)

    try:
        vrt_result = create_band_stack(band_paths, tmp_vrt, band_order)

        translate_options = gdal.TranslateOptions(
            format='GTiff',
            creationOptions=[
                f'COMPRESS={compress}',
                f'TILED={"YES" if tiled else "NO"}',
            ],
        )

        gdal.Translate(str(output_path), str(tmp_vrt), options=translate_options)

        return StackResult(
            path=output_path,
            band_count=vrt_result.band_count,
            band_mapping=vrt_result.band_mapping,
            crs=vrt_result.crs,
            bounds=vrt_result.bounds,
            resolution=vrt_result.resolution,
            warnings=vrt_result.warnings,
            source_files=vrt_result.source_files,
        )
    finally:
        tmp_vrt.unlink(missing_ok=True)
```

#### Definition of Done

- [ ] `band_stack.py` created with `create_band_stack()` function
- [ ] `stack_to_geotiff()` function for GeoTIFF output option
- [ ] VRT has correct band descriptions set
- [ ] Band order follows canonical order (blue, green, red, nir, swir16, swir22)
- [ ] Handles mixed resolution bands (10m + 20m)
- [ ] Unit tests cover all scenarios
- [ ] Existing virtual_index.py patterns reused where applicable

#### How to Test

```bash
# Unit test
pytest tests/test_band_stack.py -v

# Manual test with fixture files
python -c "
from pathlib import Path
from core.data.ingestion.band_stack import create_band_stack

band_paths = {
    'blue': Path('tests/fixtures/bands/blue_10m.tif'),
    'green': Path('tests/fixtures/bands/green_10m.tif'),
    'red': Path('tests/fixtures/bands/red_10m.tif'),
    'nir': Path('tests/fixtures/bands/nir_10m.tif'),
}

result = create_band_stack(band_paths, Path('/tmp/test_stack.vrt'))
print(f'Created VRT with {result.band_count} bands')
print(f'Band mapping: {result.band_mapping}')
"
```

#### Estimated Effort

Medium (2 days)

---

## Summary: Parallel Execution Timeline

```
Day 0 (Today):
├── Track A: Start Task 1.7.4 (--skip-validation) ──────────────────────┐
│                                                                        │
├── Track B: Start Task 1.7.1 (STAC client) ─────────────────────┐      │
│                                                                 │      │
Day 1:                                                            │      │
├── Track A: COMPLETE Task 1.7.4 ────────────────────────────────┼──────┤ SHIP workaround
│                                                                 │
├── Track B: COMPLETE Task 1.7.1 ─────────────────────────────────┤
│            Start Task 1.7.2 (Ingestion) ────────────────────────┤
│                                                                  │
├── Track C: Start Task 1.7.3 (Band stacking) ◀───────────────────┘
│            (Can start once 1.7.1 provides band_config)
│
Day 2-3:
├── Track B: COMPLETE Task 1.7.2 ─────────────────────────────────┐
│            Start Task 1.7.5 (Validator) ────────────────────────┤
│                                                                  │
├── Track C: COMPLETE Task 1.7.3 ─────────────────────────────────┘
│
Day 4:
├── Track B: COMPLETE Task 1.7.5 ─────────────────────────────────┐
│            Start Task 1.7.6 (Integration tests) ────────────────┤
│
Day 5:
└── Track B: COMPLETE Task 1.7.6 ─────────────────────────────────┘ SHIP epic
```

**Total Estimated Duration:** 5 days with parallel execution

---

## Agent Spawn Order

### Recommended Sequence

1. **Spawn Coder Agent for Track A (Task 1.7.4)**
   - Can start immediately
   - Ships independently
   - Provides immediate user value

2. **Spawn Coder Agent for Track B (Task 1.7.1)**
   - Can start immediately
   - Blocks Track C and subsequent B tasks

3. **Spawn Coder Agent for Track C (Task 1.7.3)**
   - Start after Task 1.7.1 completes
   - Runs parallel to Task 1.7.2

4. **Track B continues sequentially (Tasks 1.7.2, 1.7.5, 1.7.6)**
   - Same agent or spawn new for each task

### Spawn Commands

```bash
# Track A - Start immediately
spawn coder --task "Task 1.7.4: Add --skip-validation flag" \
    --files "cli/commands/ingest.py,core/data/ingestion/streaming.py" \
    --branch feature/multi-band-asset-download

# Track B.1 - Start immediately
spawn coder --task "Task 1.7.1: STAC client band URL extraction" \
    --files "core/data/discovery/band_config.py,core/data/discovery/stac_client.py" \
    --branch feature/multi-band-asset-download

# Track C - Start after 1.7.1 complete
spawn coder --task "Task 1.7.3: Band stacking utility" \
    --files "core/data/ingestion/band_stack.py" \
    --branch feature/multi-band-asset-download

# Track B.2 - Start after 1.7.1 complete
spawn coder --task "Task 1.7.2: Multi-band download" \
    --files "cli/commands/ingest.py" \
    --branch feature/multi-band-asset-download

# Track B.3 - Start after 1.7.2 and 1.7.3 complete
spawn coder --task "Task 1.7.5: VRT validator updates" \
    --files "core/data/ingestion/validation/image_validator.py,core/data/ingestion/validation/band_validator.py" \
    --branch feature/multi-band-asset-download

# Track B.4 - Start after 1.7.5 complete
spawn coder --task "Task 1.7.6: Integration testing" \
    --files "tests/integration/test_sentinel2_ingestion.py" \
    --branch feature/multi-band-asset-download
```

---

## Quality Gates

Before merging to main, all of the following must pass:

### Gate 1: Unit Tests
```bash
./run_tests.py
# All 518+ existing tests must pass
# New tests for Epic 1.7 must pass
```

### Gate 2: Integration Tests
```bash
pytest tests/integration/test_sentinel2_ingestion.py -v
# Full flow must work with real STAC catalog
```

### Gate 3: Security Review
```bash
invoke sentinel --scope "core/data/discovery,core/data/ingestion,cli/commands/ingest.py" --gate pre-merge
# No critical/high severity issues
```

### Gate 4: Manual Verification
```bash
# Discover Sentinel-2 data
flight discover --area tests/fixtures/areas/small_area.geojson --event flood.coastal --output json

# Ingest with multi-band download
flight ingest --area tests/fixtures/areas/small_area.geojson --event flood.coastal --output /tmp/test

# Verify output
ls /tmp/test/S2*/
# Should see: blue.tif, green.tif, red.tif, nir.tif, swir16.tif, swir22.tif, *_stack.vrt
```

---

## Rollback Plan

If issues arise post-merge:

1. **Feature Flag:** Set `multi_band_download.enabled: false` in `config/ingestion.yaml`
2. **CLI Flag:** Users can use `--skip-validation` for TCI downloads
3. **Git Revert:** If critical, revert the merge commit
4. **Escalation:** Contact Engineering Manager (Ratchet) for coordination

---

## References

- **Specification:** `OPENSPEC.md` Sections 3, 3.1, 3.2
- **Bug Report:** `docs/SENTINEL2_INGESTION_BUG_REPORT.md`
- **Roadmap:** `ROADMAP.md` Epic 1.7
- **Existing VRT Code:** `core/data/ingestion/virtual_index.py`
- **Validator Code:** `core/data/ingestion/validation/`
