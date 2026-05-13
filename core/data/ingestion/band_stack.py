"""
Band Stacking Utility for Multi-Band Raster Creation.

Creates GDAL Virtual Rasters (VRT) from individual band files by writing
the VRT XML directly, using rasterio (which bundles libgdal) for the band
metadata. This avoids the optional `osgeo` Python bindings as a hard
runtime dependency — installing rasterio is sufficient.
"""

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.enums import Resampling
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    Resampling = None
    HAS_RASTERIO = False


# rasterio numpy-style dtype name -> GDAL VRT dataType attribute value.
_GDAL_DTYPE_NAMES = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


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


def _gdal_dtype_name(dtype: str) -> str:
    return _GDAL_DTYPE_NAMES.get(str(dtype).lower(), "Float32")


def _format_geotransform(transform) -> str:
    # rasterio Affine: a, b, c, d, e, f  ->  GDAL geotransform (c, a, b, f, d, e)
    gt = (transform.c, transform.a, transform.b, transform.f, transform.d, transform.e)
    return ", ".join(f"{v:.16e}" for v in gt)


def _block_size(src) -> Tuple[int, int]:
    try:
        block = src.block_shapes[0]
        return int(block[1]), int(block[0])  # block_shapes: (rows, cols)
    except Exception:
        return src.width, 1


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
            - "resample": Resample all bands to the highest-resolution band (default)
            - "preserve": Keep native grid of the first band (bands at other grids
              still use SrcRect/DstRect to align to it)
        target_resolution: Reserved for future use; currently unused
        resample_method: Reserved for future use; VRT consumers may override

    Returns:
        StackResult with path, band_mapping, and metadata

    Raises:
        BandStackError: Base exception for stacking failures
    """
    if not HAS_RASTERIO:
        raise BandStackError("Rasterio is required for band stacking. Install with: pip install rasterio")

    if not band_paths:
        raise BandStackError("No band files provided")

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".vrt":
        raise BandStackError(f"Output must be .vrt file, got: {output_path.suffix}")

    try:
        from core.data.discovery.band_config import SENTINEL2_CANONICAL_ORDER
        default_order = SENTINEL2_CANONICAL_ORDER
    except ImportError:
        default_order = ["blue", "green", "red", "nir", "swir16", "swir22"]

    if band_order is None:
        band_order = [b for b in default_order if b in band_paths]
        band_order += sorted([b for b in band_paths.keys() if b not in band_order])

    warnings: List[str] = []
    band_info: Dict[str, dict] = {}
    reference_crs = None

    for band_name in band_order:
        if band_name not in band_paths:
            warnings.append(f"Band '{band_name}' in order but not in paths")
            continue

        band_path = band_paths[band_name]
        if not band_path.exists():
            raise BandStackError(f"Band file not found: {band_path}")

        with rasterio.open(band_path) as src:
            block_x, block_y = _block_size(src)
            band_info[band_name] = {
                "path": band_path.resolve(),
                "crs": src.crs,
                "bounds": src.bounds,
                "transform": src.transform,
                "resolution": src.res,
                "width": src.width,
                "height": src.height,
                "dtype": src.dtypes[0],
                "nodata": src.nodata,
                "block_x": block_x,
                "block_y": block_y,
            }

            if reference_crs is None:
                reference_crs = src.crs
            elif str(src.crs) != str(reference_crs):
                warnings.append(
                    f"CRS mismatch for {band_name}: {src.crs} vs {reference_crs}"
                )

    if not band_info:
        raise BandStackError("No valid band files found")

    ordered_names = [b for b in band_order if b in band_info]

    # Pick reference grid.
    if resolution_mode == "preserve":
        ref_name = ordered_names[0]
    else:
        # "resample" (or any other value): use the band with the smallest pixel size.
        ref_name = min(
            ordered_names,
            key=lambda n: abs(band_info[n]["resolution"][0]) * abs(band_info[n]["resolution"][1]),
        )

    ref = band_info[ref_name]
    ref_width = ref["width"]
    ref_height = ref["height"]
    ref_transform = ref["transform"]
    ref_bounds = ref["bounds"]

    # Build VRT XML.
    root = ET.Element(
        "VRTDataset",
        {"rasterXSize": str(ref_width), "rasterYSize": str(ref_height)},
    )
    if reference_crs is not None:
        srs_el = ET.SubElement(root, "SRS")
        try:
            srs_el.text = reference_crs.to_wkt()
        except Exception:
            srs_el.text = str(reference_crs)
    gt_el = ET.SubElement(root, "GeoTransform")
    gt_el.text = _format_geotransform(ref_transform)

    for idx, band_name in enumerate(ordered_names, 1):
        info = band_info[band_name]
        gdal_dtype = _gdal_dtype_name(info["dtype"])

        rb = ET.SubElement(
            root,
            "VRTRasterBand",
            {"dataType": gdal_dtype, "band": str(idx)},
        )
        desc_text = band_descriptions.get(band_name, band_name) if band_descriptions else band_name
        desc_el = ET.SubElement(rb, "Description")
        desc_el.text = desc_text

        if info["nodata"] is not None:
            nd_el = ET.SubElement(rb, "NoDataValue")
            nd_el.text = repr(info["nodata"])

        same_grid = (
            info["width"] == ref_width
            and info["height"] == ref_height
            and info["transform"] == ref_transform
        )
        source_tag = "SimpleSource" if same_grid else "ComplexSource"
        src_el = ET.SubElement(rb, source_tag)

        fn_el = ET.SubElement(src_el, "SourceFilename", {"relativeToVRT": "0"})
        fn_el.text = str(info["path"])

        sb_el = ET.SubElement(src_el, "SourceBand")
        sb_el.text = "1"

        ET.SubElement(
            src_el,
            "SourceProperties",
            {
                "RasterXSize": str(info["width"]),
                "RasterYSize": str(info["height"]),
                "DataType": gdal_dtype,
                "BlockXSize": str(info["block_x"]),
                "BlockYSize": str(info["block_y"]),
            },
        )
        ET.SubElement(
            src_el,
            "SrcRect",
            {
                "xOff": "0",
                "yOff": "0",
                "xSize": str(info["width"]),
                "ySize": str(info["height"]),
            },
        )
        ET.SubElement(
            src_el,
            "DstRect",
            {
                "xOff": "0",
                "yOff": "0",
                "xSize": str(ref_width),
                "ySize": str(ref_height),
            },
        )
        if not same_grid and info["nodata"] is not None:
            nodata_el = ET.SubElement(src_el, "NODATA")
            nodata_el.text = repr(info["nodata"])

    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    output_path.write_bytes(xml_bytes)

    band_mapping = {name: i + 1 for i, name in enumerate(ordered_names)}

    return StackResult(
        path=output_path,
        band_count=len(band_mapping),
        band_mapping=band_mapping,
        crs=str(reference_crs) if reference_crs else "",
        bounds=(
            ref_bounds.left,
            ref_bounds.bottom,
            ref_bounds.right,
            ref_bounds.top,
        ),
        resolution=(abs(ref["resolution"][0]), abs(ref["resolution"][1])),
        warnings=warnings,
        source_files={name: info["path"] for name, info in band_info.items()},
    )


def stack_to_geotiff(
    band_paths: Dict[str, Path],
    output_path: Path,
    band_order: Optional[List[str]] = None,
    compress: str = "LZW",
    tiled: bool = True,
) -> StackResult:
    """
    Create a multi-band GeoTIFF from individual bands using rasterio.

    Bands at coarser resolutions are resampled to the highest-resolution band
    using bilinear resampling. This copies data into a single file.

    Args:
        band_paths: Keys are lowercase band names, values are Path to .tif files
        output_path: Output path for GeoTIFF (should end in .tif)
        band_order: Order of bands in output
        compress: Compression method (LZW, DEFLATE, etc.)
        tiled: Whether to tile the output GeoTIFF

    Returns:
        StackResult with output metadata
    """
    if not HAS_RASTERIO:
        raise BandStackError("Rasterio is required for GeoTIFF stacking. Install with: pip install rasterio")

    output_path = Path(output_path)

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as tmp:
        tmp_vrt = Path(tmp.name)

    try:
        vrt_result = create_band_stack(band_paths, tmp_vrt, band_order)

        with rasterio.open(tmp_vrt) as vrt_src:
            profile = vrt_src.profile.copy()
            profile.update(
                driver="GTiff",
                compress=compress,
                tiled=tiled,
                count=vrt_src.count,
            )

            with rasterio.open(output_path, "w", **profile) as dst:
                for i in range(1, vrt_src.count + 1):
                    dst.write(vrt_src.read(i, resampling=Resampling.bilinear), i)
                    desc = vrt_src.descriptions[i - 1] if i - 1 < len(vrt_src.descriptions) else None
                    if desc:
                        dst.set_band_description(i, desc)

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

        for source in root.iter("SourceFilename"):
            relative_to_vrt = source.get("relativeToVRT", "0") == "1"
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
