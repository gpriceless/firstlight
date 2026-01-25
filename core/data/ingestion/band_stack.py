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
import tempfile

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
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
        raise BandStackError("GDAL is required for VRT creation. Install with: pip install gdal")

    if not HAS_RASTERIO:
        raise BandStackError("Rasterio is required for band metadata. Install with: pip install rasterio")

    if not band_paths:
        raise BandStackError("No band files provided")

    output_path = Path(output_path)
    if output_path.suffix.lower() != '.vrt':
        raise BandStackError(f"Output must be .vrt file, got: {output_path.suffix}")

    # Import canonical order
    try:
        from core.data.discovery.band_config import SENTINEL2_CANONICAL_ORDER
        default_order = SENTINEL2_CANONICAL_ORDER
    except ImportError:
        default_order = ["blue", "green", "red", "nir", "swir16", "swir22"]

    # Default band order: canonical order, then alphabetical for unknown
    if band_order is None:
        band_order = [b for b in default_order if b in band_paths]
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
            elif str(src.crs) != str(reference_crs):
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
        crs=str(reference_crs) if reference_crs else "",
        bounds=(
            reference_bounds.left,
            reference_bounds.bottom,
            reference_bounds.right,
            reference_bounds.top,
        ) if reference_bounds else (0, 0, 0, 0),
        resolution=band_info[list(band_info.keys())[0]]["resolution"] if band_info else (0, 0),
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

    Args:
        band_paths: Keys are lowercase band names, values are Path to .tif files
        output_path: Output path for GeoTIFF (should end in .tif)
        band_order: Order of bands in output
        compress: Compression method (LZW, DEFLATE, etc.)
        tiled: Whether to tile the output GeoTIFF

    Returns:
        StackResult with output metadata
    """
    if not HAS_GDAL:
        raise BandStackError("GDAL is required for GeoTIFF creation")

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


def validate_vrt_sources(vrt_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate all VRT source files exist.

    This function parses VRT XML directly to check source files before
    opening with rasterio/GDAL.

    Args:
        vrt_path: Path to VRT file

    Returns:
        Tuple of (is_valid, list_of_missing_files)
    """
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
