"""
Projection and CRS Handling for Data Normalization.

Provides tools for coordinate reference system (CRS) transformations,
reprojection of raster and vector data, and CRS validation/inference.

Key Capabilities:
- CRS parsing and validation from multiple formats (EPSG, WKT, PROJ)
- Raster reprojection with configurable resampling
- Vector geometry transformation
- Bounds transformation across CRS
- CRS compatibility checking
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ResamplingMethod(Enum):
    """Resampling methods for reprojection."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBICSPLINE = "cubicspline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    MODE = "mode"
    MIN = "min"
    MAX = "max"
    MED = "med"
    Q1 = "q1"
    Q3 = "q3"
    SUM = "sum"


class CRSType(Enum):
    """Types of coordinate reference systems."""

    GEOGRAPHIC = "geographic"  # Lat/lon (e.g., EPSG:4326)
    PROJECTED = "projected"  # Planar (e.g., UTM)
    COMPOUND = "compound"  # Horizontal + vertical
    UNKNOWN = "unknown"


@dataclass
class CRSInfo:
    """
    Information about a coordinate reference system.

    Attributes:
        code: EPSG code or authority:code (e.g., "EPSG:4326")
        wkt: Well-Known Text representation
        proj4: PROJ4 string representation
        crs_type: Type of CRS (geographic, projected, etc.)
        units: Linear units for projected CRS (meters, feet, etc.)
        datum: Geodetic datum name
        ellipsoid: Reference ellipsoid name
        is_geographic: True if geographic CRS (lat/lon)
        is_projected: True if projected CRS (planar)
        axis_order: Order of axes (e.g., ["lon", "lat"] or ["x", "y"])
    """

    code: str
    wkt: str
    proj4: str
    crs_type: CRSType
    units: Optional[str]
    datum: Optional[str]
    ellipsoid: Optional[str]
    is_geographic: bool
    is_projected: bool
    axis_order: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code": self.code,
            "wkt": self.wkt,
            "proj4": self.proj4,
            "crs_type": self.crs_type.value,
            "units": self.units,
            "datum": self.datum,
            "ellipsoid": self.ellipsoid,
            "is_geographic": self.is_geographic,
            "is_projected": self.is_projected,
            "axis_order": self.axis_order,
        }


@dataclass
class ReprojectionConfig:
    """
    Configuration for reprojection operations.

    Attributes:
        target_crs: Target coordinate reference system
        resampling: Resampling method for raster data
        resolution: Target resolution (None to preserve source)
        src_nodata: Source nodata value
        dst_nodata: Destination nodata value
        target_aligned_pixels: Align pixels to target grid
        num_threads: Number of threads for reprojection
        memory_limit: Memory limit in MB for operation
        bounds: Optional output bounds (minx, miny, maxx, maxy)
        width: Optional output width (overrides resolution)
        height: Optional output height (overrides resolution)
    """

    target_crs: str
    resampling: ResamplingMethod = ResamplingMethod.BILINEAR
    resolution: Optional[Tuple[float, float]] = None
    src_nodata: Optional[float] = None
    dst_nodata: Optional[float] = None
    target_aligned_pixels: bool = True
    num_threads: int = 4
    memory_limit: int = 512
    bounds: Optional[Tuple[float, float, float, float]] = None
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ReprojectionResult:
    """Result from a reprojection operation."""

    output_path: Optional[Path]
    source_crs: str
    target_crs: str
    source_bounds: Tuple[float, float, float, float]
    target_bounds: Tuple[float, float, float, float]
    source_resolution: Tuple[float, float]
    target_resolution: Tuple[float, float]
    resampling_method: str
    pixel_error_estimate: float  # Estimated max error in pixels
    processing_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "output_path": str(self.output_path) if self.output_path else None,
            "source_crs": self.source_crs,
            "target_crs": self.target_crs,
            "source_bounds": self.source_bounds,
            "target_bounds": self.target_bounds,
            "source_resolution": self.source_resolution,
            "target_resolution": self.target_resolution,
            "resampling_method": self.resampling_method,
            "pixel_error_estimate": self.pixel_error_estimate,
            "processing_time_seconds": self.processing_time_seconds,
            "metadata": self.metadata,
        }


class CRSHandler:
    """
    Handler for CRS operations and transformations.

    Provides utilities for CRS parsing, validation, comparison,
    and transformation between coordinate systems.

    Example:
        handler = CRSHandler()
        info = handler.parse_crs("EPSG:4326")
        print(info.is_geographic)  # True

        # Transform bounds
        wgs84_bounds = (0, 0, 10, 10)
        utm_bounds = handler.transform_bounds(
            wgs84_bounds, "EPSG:4326", "EPSG:32632"
        )
    """

    def __init__(self):
        """Initialize CRS handler."""
        # Common CRS definitions cache
        self._crs_cache: Dict[str, Any] = {}

    def parse_crs(self, crs: Union[str, int, Any]) -> CRSInfo:
        """
        Parse a CRS from various formats.

        Args:
            crs: CRS specification (EPSG code, WKT, PROJ4, or CRS object)

        Returns:
            CRSInfo with parsed CRS details

        Raises:
            ValueError: If CRS cannot be parsed
        """
        try:
            from pyproj import CRS
        except ImportError:
            raise ImportError("pyproj is required for CRS operations")

        # Handle different input formats
        if isinstance(crs, int):
            crs_str = f"EPSG:{crs}"
        elif isinstance(crs, str):
            crs_str = crs
        else:
            # Assume it's already a CRS-like object
            crs_str = str(crs)

        try:
            pyproj_crs = CRS.from_user_input(crs_str)
        except Exception as e:
            raise ValueError(f"Cannot parse CRS '{crs_str}': {e}")

        # Extract CRS information
        is_geographic = pyproj_crs.is_geographic
        is_projected = pyproj_crs.is_projected

        # Determine CRS type
        if is_geographic:
            crs_type = CRSType.GEOGRAPHIC
        elif is_projected:
            crs_type = CRSType.PROJECTED
        elif pyproj_crs.is_compound:
            crs_type = CRSType.COMPOUND
        else:
            crs_type = CRSType.UNKNOWN

        # Get axis info
        axis_order = []
        if pyproj_crs.axis_info:
            for axis in pyproj_crs.axis_info:
                axis_order.append(axis.name.lower())

        # Get authority code
        auth = pyproj_crs.to_authority()
        if auth:
            code = f"{auth[0]}:{auth[1]}"
        else:
            code = crs_str

        # Get datum and ellipsoid
        datum_name = None
        ellipsoid_name = None
        if pyproj_crs.datum:
            datum_name = pyproj_crs.datum.name
            if pyproj_crs.datum.ellipsoid:
                ellipsoid_name = pyproj_crs.datum.ellipsoid.name

        # Get units
        units = None
        if is_projected and pyproj_crs.axis_info:
            units = pyproj_crs.axis_info[0].unit_name

        return CRSInfo(
            code=code,
            wkt=pyproj_crs.to_wkt(),
            proj4=pyproj_crs.to_proj4(),
            crs_type=crs_type,
            units=units,
            datum=datum_name,
            ellipsoid=ellipsoid_name,
            is_geographic=is_geographic,
            is_projected=is_projected,
            axis_order=axis_order,
        )

    def are_equivalent(self, crs1: str, crs2: str) -> bool:
        """
        Check if two CRS definitions are equivalent.

        Args:
            crs1: First CRS specification
            crs2: Second CRS specification

        Returns:
            True if CRS are equivalent
        """
        try:
            from pyproj import CRS
        except ImportError:
            raise ImportError("pyproj is required for CRS operations")

        try:
            pyproj_crs1 = CRS.from_user_input(crs1)
            pyproj_crs2 = CRS.from_user_input(crs2)
            return pyproj_crs1.equals(pyproj_crs2)
        except (ValueError, TypeError) as e:
            logger.debug(f"CRS equivalence check failed: {e}")
            return False

    def suggest_utm_zone(
        self, longitude: float, latitude: float, southern_hemisphere: bool = None
    ) -> str:
        """
        Suggest appropriate UTM zone for a location.

        Args:
            longitude: Longitude in decimal degrees
            latitude: Latitude in decimal degrees
            southern_hemisphere: Override hemisphere detection

        Returns:
            EPSG code for appropriate UTM zone
        """
        # Calculate UTM zone number
        zone = int((longitude + 180) / 6) + 1

        # Handle edge cases
        if zone > 60:
            zone = 60
        elif zone < 1:
            zone = 1

        # Special zones for Norway/Svalbard
        if 56 <= latitude < 64 and 3 <= longitude < 12:
            zone = 32
        elif 72 <= latitude < 84:
            if 0 <= longitude < 9:
                zone = 31
            elif 9 <= longitude < 21:
                zone = 33
            elif 21 <= longitude < 33:
                zone = 35
            elif 33 <= longitude < 42:
                zone = 37

        # Determine hemisphere
        if southern_hemisphere is None:
            southern_hemisphere = latitude < 0

        # Calculate EPSG code
        if southern_hemisphere:
            epsg = 32700 + zone
        else:
            epsg = 32600 + zone

        return f"EPSG:{epsg}"

    def transform_bounds(
        self,
        bounds: Tuple[float, float, float, float],
        source_crs: str,
        target_crs: str,
        densify_points: int = 21,
    ) -> Tuple[float, float, float, float]:
        """
        Transform bounding box between coordinate systems.

        Args:
            bounds: Source bounds (minx, miny, maxx, maxy)
            source_crs: Source CRS
            target_crs: Target CRS
            densify_points: Number of points per edge for accuracy

        Returns:
            Transformed bounds (minx, miny, maxx, maxy)
        """
        try:
            from pyproj import CRS, Transformer
        except ImportError:
            raise ImportError("pyproj is required for CRS operations")

        minx, miny, maxx, maxy = bounds

        # Create transformer
        src_crs = CRS.from_user_input(source_crs)
        dst_crs = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(
            src_crs, dst_crs, always_xy=True
        )

        # Densify edges for better accuracy
        x_points = np.linspace(minx, maxx, densify_points)
        y_points = np.linspace(miny, maxy, densify_points)

        # Create points along all four edges
        edge_points_x = []
        edge_points_y = []

        # Top and bottom edges
        for x in x_points:
            edge_points_x.extend([x, x])
            edge_points_y.extend([miny, maxy])

        # Left and right edges
        for y in y_points:
            edge_points_x.extend([minx, maxx])
            edge_points_y.extend([y, y])

        # Transform all points
        tx, ty = transformer.transform(edge_points_x, edge_points_y)

        # Handle infinities and NaN
        tx = np.array(tx)
        ty = np.array(ty)
        valid = np.isfinite(tx) & np.isfinite(ty)

        if not np.any(valid):
            raise ValueError("Bounds transformation resulted in no valid points")

        return (
            float(np.min(tx[valid])),
            float(np.min(ty[valid])),
            float(np.max(tx[valid])),
            float(np.max(ty[valid])),
        )

    def transform_point(
        self,
        x: float,
        y: float,
        source_crs: str,
        target_crs: str,
    ) -> Tuple[float, float]:
        """
        Transform a single point between coordinate systems.

        Args:
            x: X coordinate
            y: Y coordinate
            source_crs: Source CRS
            target_crs: Target CRS

        Returns:
            Transformed (x, y) coordinates
        """
        try:
            from pyproj import CRS, Transformer
        except ImportError:
            raise ImportError("pyproj is required for CRS operations")

        src_crs = CRS.from_user_input(source_crs)
        dst_crs = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(
            src_crs, dst_crs, always_xy=True
        )

        tx, ty = transformer.transform(x, y)
        return (float(tx), float(ty))


class RasterReprojector:
    """
    Reprojects raster data between coordinate systems.

    Supports both file-based and array-based reprojection with
    configurable resampling methods and output options.

    Example:
        reprojector = RasterReprojector()
        result = reprojector.reproject_file(
            "input.tif",
            "output.tif",
            ReprojectionConfig(target_crs="EPSG:4326")
        )
    """

    def __init__(self, crs_handler: Optional[CRSHandler] = None):
        """
        Initialize reprojector.

        Args:
            crs_handler: CRS handler instance (creates new if None)
        """
        self.crs_handler = crs_handler or CRSHandler()

    def reproject_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        config: ReprojectionConfig,
        overwrite: bool = False,
    ) -> ReprojectionResult:
        """
        Reproject a raster file to a new CRS.

        Args:
            input_path: Path to input raster
            output_path: Path for output raster
            config: Reprojection configuration
            overwrite: Whether to overwrite existing output

        Returns:
            ReprojectionResult with details

        Raises:
            FileNotFoundError: If input file doesn't exist
            FileExistsError: If output exists and overwrite=False
        """
        import time

        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file exists: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.warp import (
                calculate_default_transform,
                reproject,
                Resampling as WarpResampling,
            )
        except ImportError:
            raise ImportError("rasterio is required for raster reprojection")

        start_time = time.time()
        logger.info(f"Reprojecting {input_path} to {config.target_crs}")

        with rasterio.open(input_path) as src:
            source_crs = str(src.crs) if src.crs else None
            if not source_crs:
                raise ValueError(f"Source file has no CRS: {input_path}")

            source_bounds = src.bounds
            source_res = (src.res[0], src.res[1])

            # Calculate target transform
            if config.bounds:
                dst_bounds = config.bounds
            else:
                dst_bounds = None

            if config.width and config.height:
                transform, width, height = calculate_default_transform(
                    src.crs,
                    config.target_crs,
                    config.width,
                    config.height,
                    *src.bounds,
                )
            elif config.resolution:
                transform, width, height = calculate_default_transform(
                    src.crs,
                    config.target_crs,
                    src.width,
                    src.height,
                    *src.bounds,
                    resolution=config.resolution,
                )
            else:
                transform, width, height = calculate_default_transform(
                    src.crs,
                    config.target_crs,
                    src.width,
                    src.height,
                    *(dst_bounds or src.bounds),
                )

            # Get resampling method
            resampling = getattr(
                WarpResampling, config.resampling.value.upper(), WarpResampling.bilinear
            )

            # Update profile
            profile = src.profile.copy()
            profile.update({
                "crs": config.target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            })

            if config.dst_nodata is not None:
                profile["nodata"] = config.dst_nodata

            # Perform reprojection
            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    src_data = src.read(band_idx)

                    dst_data = np.empty((height, width), dtype=src_data.dtype)

                    reproject(
                        source=src_data,
                        destination=dst_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=config.target_crs,
                        resampling=resampling,
                        src_nodata=config.src_nodata or src.nodata,
                        dst_nodata=config.dst_nodata,
                        num_threads=config.num_threads,
                    )

                    dst.write(dst_data, band_idx)

        processing_time = time.time() - start_time

        # Calculate target resolution
        target_res = (abs(transform[0]), abs(transform[4]))

        # Estimate pixel error (simplified)
        pixel_error = 0.5  # Conservative estimate

        # Get target bounds
        target_bounds_result = self.crs_handler.transform_bounds(
            (source_bounds.left, source_bounds.bottom,
             source_bounds.right, source_bounds.top),
            source_crs,
            config.target_crs,
        )

        logger.info(
            f"Reprojection complete: {output_path} "
            f"({processing_time:.2f}s)"
        )

        return ReprojectionResult(
            output_path=output_path,
            source_crs=source_crs,
            target_crs=config.target_crs,
            source_bounds=(
                source_bounds.left,
                source_bounds.bottom,
                source_bounds.right,
                source_bounds.top,
            ),
            target_bounds=target_bounds_result,
            source_resolution=source_res,
            target_resolution=target_res,
            resampling_method=config.resampling.value,
            pixel_error_estimate=pixel_error,
            processing_time_seconds=processing_time,
        )

    def reproject_array(
        self,
        data: np.ndarray,
        source_transform: Any,
        source_crs: str,
        config: ReprojectionConfig,
        source_nodata: Optional[float] = None,
    ) -> Tuple[np.ndarray, Any, Tuple[float, float, float, float]]:
        """
        Reproject a numpy array to a new CRS.

        Args:
            data: Input array (height, width) or (bands, height, width)
            source_transform: Affine transform of source data
            source_crs: Source CRS
            config: Reprojection configuration
            source_nodata: NoData value in source data

        Returns:
            Tuple of (reprojected_data, new_transform, new_bounds)
        """
        try:
            import rasterio
            from rasterio.warp import (
                calculate_default_transform,
                reproject,
                Resampling as WarpResampling,
            )
        except ImportError:
            raise ImportError("rasterio is required for raster reprojection")

        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        band_count, src_height, src_width = data.shape

        # Calculate source bounds
        src_bounds = rasterio.transform.array_bounds(
            src_height, src_width, source_transform
        )

        # Calculate target transform
        if config.resolution:
            transform, width, height = calculate_default_transform(
                source_crs,
                config.target_crs,
                src_width,
                src_height,
                *src_bounds,
                resolution=config.resolution,
            )
        else:
            transform, width, height = calculate_default_transform(
                source_crs,
                config.target_crs,
                src_width,
                src_height,
                *src_bounds,
            )

        # Get resampling method
        resampling = getattr(
            WarpResampling, config.resampling.value.upper(), WarpResampling.bilinear
        )

        # Allocate output
        dst_data = np.empty((band_count, height, width), dtype=data.dtype)

        # Reproject each band
        for i in range(band_count):
            reproject(
                source=data[i],
                destination=dst_data[i],
                src_transform=source_transform,
                src_crs=source_crs,
                dst_transform=transform,
                dst_crs=config.target_crs,
                resampling=resampling,
                src_nodata=source_nodata,
                dst_nodata=config.dst_nodata,
                num_threads=config.num_threads,
            )

        # Calculate output bounds
        dst_bounds = rasterio.transform.array_bounds(height, width, transform)

        return dst_data.squeeze(), transform, dst_bounds


class VectorReprojector:
    """
    Reprojects vector/geometry data between coordinate systems.

    Handles GeoJSON-like geometries and coordinate arrays.
    """

    def __init__(self, crs_handler: Optional[CRSHandler] = None):
        """
        Initialize vector reprojector.

        Args:
            crs_handler: CRS handler instance
        """
        self.crs_handler = crs_handler or CRSHandler()

    def transform_geometry(
        self,
        geometry: Dict[str, Any],
        source_crs: str,
        target_crs: str,
    ) -> Dict[str, Any]:
        """
        Transform a GeoJSON-like geometry.

        Args:
            geometry: GeoJSON geometry dict
            source_crs: Source CRS
            target_crs: Target CRS

        Returns:
            Transformed geometry dict
        """
        try:
            from pyproj import CRS, Transformer
        except ImportError:
            raise ImportError("pyproj is required for geometry transformation")

        src_crs = CRS.from_user_input(source_crs)
        dst_crs = CRS.from_user_input(target_crs)
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

        geom_type = geometry.get("type", "")
        coords = geometry.get("coordinates", [])

        transformed_coords = self._transform_coordinates(coords, transformer, geom_type)

        return {
            "type": geom_type,
            "coordinates": transformed_coords,
        }

    def _transform_coordinates(
        self,
        coords: Any,
        transformer: Any,
        geom_type: str,
    ) -> Any:
        """Recursively transform coordinates."""
        if geom_type == "Point":
            x, y = coords[0], coords[1]
            tx, ty = transformer.transform(x, y)
            if len(coords) > 2:
                return [tx, ty, coords[2]]
            return [tx, ty]

        elif geom_type in ["LineString", "MultiPoint"]:
            return [
                self._transform_coordinates(coord, transformer, "Point")
                for coord in coords
            ]

        elif geom_type in ["Polygon", "MultiLineString"]:
            return [
                self._transform_coordinates(ring, transformer, "LineString")
                for ring in coords
            ]

        elif geom_type == "MultiPolygon":
            return [
                self._transform_coordinates(poly, transformer, "Polygon")
                for poly in coords
            ]

        elif geom_type == "GeometryCollection":
            # Handle nested geometries
            return coords  # Would need special handling

        else:
            # Try to auto-detect based on structure
            if isinstance(coords, (int, float)):
                return coords
            elif isinstance(coords, list):
                if len(coords) >= 2 and all(isinstance(c, (int, float)) for c in coords[:2]):
                    # This is a coordinate pair
                    return self._transform_coordinates(coords, transformer, "Point")
                else:
                    # Recurse
                    return [
                        self._transform_coordinates(c, transformer, "")
                        for c in coords
                    ]
            return coords


def reproject_to_crs(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_crs: str,
    resampling: str = "bilinear",
    overwrite: bool = False,
) -> ReprojectionResult:
    """
    Convenience function to reproject a raster file.

    Args:
        input_path: Path to input raster
        output_path: Path for output raster
        target_crs: Target CRS (e.g., "EPSG:4326")
        resampling: Resampling method name
        overwrite: Whether to overwrite existing output

    Returns:
        ReprojectionResult with details
    """
    config = ReprojectionConfig(
        target_crs=target_crs,
        resampling=ResamplingMethod(resampling.lower()),
    )
    reprojector = RasterReprojector()
    return reprojector.reproject_file(input_path, output_path, config, overwrite)


def get_crs_info(crs: Union[str, int]) -> CRSInfo:
    """
    Get detailed information about a CRS.

    Args:
        crs: CRS specification (EPSG code or string)

    Returns:
        CRSInfo with CRS details
    """
    handler = CRSHandler()
    return handler.parse_crs(crs)


def suggest_target_crs(
    bounds: Tuple[float, float, float, float],
    source_crs: str = "EPSG:4326",
) -> str:
    """
    Suggest an appropriate projected CRS for given bounds.

    Args:
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        source_crs: Source CRS (default WGS84)

    Returns:
        Suggested EPSG code
    """
    handler = CRSHandler()

    # Transform to WGS84 if needed
    if source_crs != "EPSG:4326":
        bounds = handler.transform_bounds(bounds, source_crs, "EPSG:4326")

    # Calculate centroid
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    return handler.suggest_utm_zone(center_lon, center_lat)
