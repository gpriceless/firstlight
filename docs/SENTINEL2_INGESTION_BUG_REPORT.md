# Sentinel-2 Image Ingestion Bug Report

**Date:** 2026-01-25
**Status:** Root Cause Identified
**Severity:** High - Blocks all real Sentinel-2 data processing

---

## Executive Summary

The FirstLight image ingestion pipeline rejects all Sentinel-2 downloads with validation errors because the STAC discovery returns URLs for True Color Images (TCI.tif) - 3-band RGB composites - while the validator expects individual spectral bands (blue, green, red, nir). This is a fundamental mismatch between what the STAC client retrieves and what the downstream processing requires.

---

## Error Observed

```
Image validation failed: [
  "Required band 'blue' not found in dataset",
  "Required band 'green' not found in dataset",
  "Required band 'red' not found in dataset",
  "Required band 'nir' not found in dataset"
]
```

---

## Data Flow Analysis

### 1. STAC Discovery (`core/data/discovery/stac_client.py`)

The `discover_data()` function searches STAC catalogs and builds result objects. The critical code is at **lines 446-450**:

```python
results.append({
    "id": item.id,
    "source": source,
    "datetime": item.datetime.isoformat(),
    "cloud_cover": item.cloud_cover,
    "resolution_m": 10.0 if source in ["sentinel1", "sentinel2"] else 30.0,
    "size_bytes": size_estimate,
    "url": item.assets.get("visual") or item.assets.get("data") or "",  # <-- ROOT CAUSE
    "priority": "primary" if source in ["sentinel1", "sentinel2"] else "secondary",
    "bbox": item.bbox,
    "assets": item.assets,
})
```

**Root Cause Identified:** Line 446 selects the URL using:
```python
"url": item.assets.get("visual") or item.assets.get("data") or "",
```

For Sentinel-2 data in Earth Search STAC catalog:
- `"visual"` asset = TCI.tif (True Color Image - RGB composite, 3 bands)
- The individual band assets are named: `"blue"`, `"green"`, `"red"`, `"nir"`, `"swir16"`, `"swir22"`, etc.

The code prioritizes the `"visual"` asset, which is a pre-rendered composite image meant for visualization, NOT for scientific analysis.

### 2. STAC Item Asset Structure

A Sentinel-2 L2A item from Earth Search has these assets (partial list):

| Asset Key | Description | Bands | Use Case |
|-----------|-------------|-------|----------|
| `visual` | True Color Image (TCI) | 3 (RGB composite) | Visualization |
| `blue` | Band 2 (490nm) | 1 | Analysis |
| `green` | Band 3 (560nm) | 1 | Analysis |
| `red` | Band 4 (665nm) | 1 | Analysis |
| `nir` | Band 8 (842nm) | 1 | Analysis |
| `swir16` | Band 11 (1610nm) | 1 | Analysis |
| `swir22` | Band 12 (2190nm) | 1 | Analysis |

The `assets` dictionary IS being passed through (line 449), but only a single URL is selected for download.

### 3. Ingestion Flow (`cli/commands/ingest.py`)

The ingest command calls `process_item()` which:

1. Gets the single URL from the discovery result (line 479)
2. Downloads using `_download_real_data()` (lines 387-415)
3. Validates using `_validate_downloaded_image()` (lines 418-456)

```python
def process_item(...):
    url = item.get("url")  # Single URL - the TCI.tif

    if url:
        _download_real_data(url, download_path, item)

    if not _validate_downloaded_image(download_path, item):
        return False  # FAILS HERE
```

### 4. Streaming Ingester (`core/data/ingestion/streaming.py`)

The `StreamingIngester.ingest()` method downloads and validates:

```python
# Lines 1078-1085
validation_result = self._validate_image(source_path, result)
if validation_result is not None and not validation_result.is_valid:
    result["status"] = "failed"
    result["errors"].extend(validation_result.errors)
    result["validation"] = validation_result.to_dict()
    logger.error(f"Image validation failed for {source}: {validation_result.errors}")
    return result
```

### 5. Image Validator (`core/data/ingestion/validation/image_validator.py`)

The validator expects specific bands based on configuration:

**Default Required Bands (from `config.py` lines 188-189):**
```python
required_optical_bands: List[str] = field(
    default_factory=lambda: ["blue", "green", "red", "nir"]
)
```

**Band Matching Logic (from `_get_expected_optical_bands()` lines 565-581):**
```python
def _get_expected_optical_bands(self, data_source_spec):
    if data_source_spec and "bands" in data_source_spec:
        return data_source_spec["bands"]

    # Default band mapping for common sensors
    return {
        "blue": ["B02", "B2"],
        "green": ["B03", "B3"],
        "red": ["B04", "B4"],
        "nir": ["B08", "B8", "B8A", "B5"],
        "swir1": ["B11", "B6"],
        "swir2": ["B12", "B7"],
    }
```

### 6. Band Validator (`core/data/ingestion/validation/band_validator.py`)

The `validate_bands()` method (lines 81-138):

1. Gets band descriptions from dataset metadata
2. Tries to match expected bands (blue, green, red, nir) to actual bands
3. For TCI.tif (3 bands with no metadata names), falls back to positional matching
4. **Critical:** Positional fallback only works if there are >= 4 bands (line 186):

```python
if band_count >= 4:
    positional_defaults = {
        "blue": 1,
        "green": 2,
        "red": 3,
        "nir": 4,  # <-- TCI only has 3 bands!
    }
```

Since TCI.tif has only 3 bands, NIR cannot be mapped, and validation fails for all 4 required bands.

---

## Visual Data Flow

```
                                    CURRENT FLOW (BROKEN)

  STAC Catalog                   stac_client.py                    ingest.py
+---------------+              +------------------+              +------------------+
| Sentinel-2    |  search()    | discover_data()  |  discovery   | process_item()   |
| L2A Items     |------------->| selects "visual" |------------->| downloads single |
|               |              | asset = TCI.tif  |   results    | TCI.tif file     |
| Assets:       |              |                  |              |                  |
|  - visual     |              | url: TCI.tif     |              |                  |
|  - blue       |              | assets: {...}    |              |                  |
|  - green      |              |   (passed but    |              |                  |
|  - red        |              |    not used)     |              |                  |
|  - nir        |              |                  |              |                  |
|  - swir16     |              |                  |              |                  |
|  - swir22     |              |                  |              |                  |
+---------------+              +------------------+              +------------------+
                                                                        |
                                                                        v
                                                               +------------------+
                                                               | ImageValidator   |
                                                               | expects bands:   |
                                                               |  - blue          |
                                                               |  - green         |
                                                               |  - red           |
                                                               |  - nir           |
                                                               |                  |
                                                               | TCI.tif has:     |
                                                               |  - band 1 (R)    |
                                                               |  - band 2 (G)    |
                                                               |  - band 3 (B)    |
                                                               |                  |
                                                               | MISMATCH -> FAIL |
                                                               +------------------+
```

---

## Fix Options Analysis

### Option A: Modify STAC Client to Return Individual Band URLs

**Location:** `core/data/discovery/stac_client.py`, lines 430-450

**Change:** Instead of selecting a single "visual" URL, return URLs for individual spectral bands needed for analysis.

```python
# Current (broken):
"url": item.assets.get("visual") or item.assets.get("data") or "",

# Proposed:
# Return URLs for individual bands needed for analysis
analysis_bands = ["blue", "green", "red", "nir", "swir16", "swir22"]
band_urls = {
    band: item.assets.get(band)
    for band in analysis_bands
    if band in item.assets
}
# Store band_urls instead of single url
```

**Pros:**
- Correct fix for scientific analysis pipelines
- Downloads exactly the bands needed
- Preserves spectral integrity for algorithms (NDVI, burn severity, etc.)
- Smaller downloads (only needed bands vs. full scene)

**Cons:**
- Requires changes to ingestion to handle multiple URLs per item
- Multiple HTTP requests per scene
- Needs VRT or stacking step to combine bands
- More complex error handling

**Effort:** Medium-High

---

### Option B: Modify Validator to Accept Composite Files

**Location:** `core/data/ingestion/validation/image_validator.py` and `band_validator.py`

**Change:** Detect TCI/composite files and use relaxed validation.

```python
def _validate_optical_bands(self, dataset, data_source_spec, result):
    # Check if this is a composite file
    if self._is_composite_image(dataset):
        # Use simplified validation for composites
        return self._validate_composite_image(dataset, result)

    # Normal band validation...
```

**Pros:**
- Minimal code changes
- Backward compatible
- TCI files can be used for visualization workflows

**Cons:**
- TCI files cannot be used for spectral analysis (no NIR, SWIR)
- Defeats purpose of validation (masking real issues)
- Creates two code paths to maintain
- Scientific workflows still broken

**Effort:** Low

**Recommendation:** NOT RECOMMENDED for scientific analysis platform

---

### Option C: Add Skip-Validation Flag

**Location:** CLI commands and `StreamingIngester`

**Change:** Add `--skip-validation` or `--validation-mode lenient` flag.

```python
@click.option("--skip-validation", is_flag=True, help="Skip image validation")
def ingest(..., skip_validation: bool):
    if skip_validation:
        ingester._validation_enabled = False
```

**Pros:**
- Quick fix for users who need data now
- Doesn't require architectural changes
- Useful for testing and debugging

**Cons:**
- Masks the underlying problem
- Users may get corrupt/invalid data
- Requires users to know about the workaround
- Not a real fix

**Effort:** Very Low

---

### Option D: Multi-Band Asset Download (Recommended)

**Location:** Multiple files - most comprehensive fix

**Changes:**

1. **`stac_client.py`:** Return all analysis assets
2. **`ingest.py`:** Download multiple band files per item
3. **Add VRT/stack step:** Combine bands into analysis-ready stack
4. **Validator:** Validate the combined stack

```python
# stac_client.py - discover_data()
results.append({
    "id": item.id,
    "source": source,
    # Remove single URL, use assets dict
    "assets": item.assets,
    "analysis_bands": {
        "blue": item.assets.get("blue"),
        "green": item.assets.get("green"),
        "red": item.assets.get("red"),
        "nir": item.assets.get("nir"),
        "swir16": item.assets.get("swir16"),
        "swir22": item.assets.get("swir22"),
    },
    ...
})
```

**Pros:**
- Correct solution for scientific analysis
- Downloads all spectral bands needed
- Full validation possible
- Supports all algorithms (NDVI, NBR, etc.)
- Future-proof architecture

**Cons:**
- Largest code change
- Multiple downloads per scene
- Needs band stacking logic
- More complex error handling

**Effort:** High

---

### Option E: Use COG Virtual Filesystem (VSICURL)

**Location:** `streaming.py` and analysis code

**Change:** Instead of downloading, access COGs directly via GDAL's /vsicurl/ virtual filesystem.

```python
# Access individual bands directly without full download
blue_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/.../B02.tif"
vrt_path = create_vrt([
    f"/vsicurl/{blue_url}",
    f"/vsicurl/{green_url}",
    f"/vsicurl/{red_url}",
    f"/vsicurl/{nir_url}",
])
```

**Pros:**
- No download storage needed
- Stream processing of remote data
- Modern cloud-native approach
- Fast for regional analysis (only reads tiles needed)

**Cons:**
- Requires network connectivity during analysis
- Slower for repeated access
- May hit rate limits
- Needs GDAL/rasterio configuration

**Effort:** Medium

---

## Recommended Solution

**Primary Fix: Option D (Multi-Band Asset Download)**

This is the correct architectural fix for a scientific analysis platform. The current design assumes a single file per item, which doesn't match how Sentinel-2 data is distributed (separate files per band).

**Immediate Workaround: Option C (Skip-Validation Flag)**

Add `--skip-validation` flag for users who need to proceed immediately, with clear documentation that TCI files are for visualization only.

**Implementation Priority:**

1. **(Immediate)** Add `--skip-validation` flag with warning message
2. **(Short-term)** Update STAC client to expose individual band URLs
3. **(Medium-term)** Update ingestion pipeline to download multiple bands
4. **(Medium-term)** Add VRT/band stacking capability
5. **(Long-term)** Consider VSICURL for cloud-native workflows

---

## Technical Details for Implementation

### Required Changes for Option D

#### 1. `stac_client.py` Changes

```python
# Add constant for analysis bands
SENTINEL2_ANALYSIS_BANDS = {
    "blue": "blue",      # B02
    "green": "green",    # B03
    "red": "red",        # B04
    "nir": "nir",        # B08
    "swir16": "swir16",  # B11
    "swir22": "swir22",  # B12
}

# In discover_data(), change URL selection:
results.append({
    ...
    "url": None,  # No single URL
    "band_urls": {
        name: item.assets.get(asset_key).href
        for name, asset_key in SENTINEL2_ANALYSIS_BANDS.items()
        if asset_key in item.assets and item.assets.get(asset_key)
    },
    "assets": item.assets,
    ...
})
```

#### 2. `ingest.py` Changes

```python
def process_item(...):
    band_urls = item.get("band_urls", {})

    if band_urls:
        # Download each band
        band_paths = {}
        for band_name, url in band_urls.items():
            band_path = item_dir / f"{band_name}.tif"
            _download_real_data(url, band_path, item)
            band_paths[band_name] = band_path

        # Create VRT stack
        stack_path = item_dir / f"{item_id}_stack.vrt"
        create_band_stack(band_paths, stack_path)

        # Validate the stack
        if not _validate_downloaded_image(stack_path, item):
            return False
```

#### 3. New Band Stacking Utility

```python
# core/data/ingestion/band_stack.py
def create_band_stack(band_paths: Dict[str, Path], output_path: Path) -> Path:
    """Create a VRT stacking individual band files."""
    import rasterio
    from rasterio.vrt import WarpedVRT

    # Implementation details...
```

---

## Files Affected

| File | Change Type | Description |
|------|-------------|-------------|
| `core/data/discovery/stac_client.py` | Modify | Return individual band URLs |
| `cli/commands/ingest.py` | Modify | Handle multi-band downloads |
| `cli/commands/discover.py` | Minor | Update output format |
| `core/data/ingestion/streaming.py` | Modify | Support multi-file ingestion |
| `core/data/ingestion/band_stack.py` | New | Band stacking utility |
| `core/data/ingestion/validation/image_validator.py` | Minor | Adjust for stacked files |

---

## Appendix: Sentinel-2 STAC Asset Names

Earth Search STAC catalog uses these asset keys for Sentinel-2 L2A:

| Asset Key | Sentinel-2 Band | Wavelength (nm) | Resolution (m) |
|-----------|-----------------|-----------------|----------------|
| `coastal` | B01 | 443 | 60 |
| `blue` | B02 | 490 | 10 |
| `green` | B03 | 560 | 10 |
| `red` | B04 | 665 | 10 |
| `rededge1` | B05 | 705 | 20 |
| `rededge2` | B06 | 740 | 20 |
| `rededge3` | B07 | 783 | 20 |
| `nir` | B08 | 842 | 10 |
| `nir08` | B8A | 865 | 20 |
| `nir09` | B09 | 940 | 60 |
| `swir16` | B11 | 1610 | 20 |
| `swir22` | B12 | 2190 | 20 |
| `scl` | Scene Classification | - | 20 |
| `visual` | TCI (True Color) | RGB composite | 10 |

---

## Conclusion

The root cause is a single line of code in `stac_client.py` that prioritizes the "visual" asset (TCI.tif) for download. TCI files are 3-band RGB composites intended for visualization, not scientific analysis. The validator correctly identifies that required spectral bands (especially NIR) are missing.

The fix requires updating the data flow to download individual spectral band files and stack them for analysis. This is the correct approach for a geospatial analysis platform.
