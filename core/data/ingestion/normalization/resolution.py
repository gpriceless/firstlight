"""
Resolution and Resampling Tools for Data Normalization.

Provides tools for spatial resampling of raster data including
upsampling, downsampling, and resolution harmonization across
multiple datasets.

Key Capabilities:
- Resolution harmonization across datasets
- Spatial resampling with multiple methods
- Resolution calculation and conversion
- Scale-aware processing for multi-resolution data
- Quality-aware resampling with degradation tracking
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ResamplingMethod(Enum):
    """Resampling methods for spatial resolution changes."""

    # Nearest neighbor - preserves values, suitable for categorical data
    NEAREST = "nearest"

    # Bilinear interpolation - smooth, suitable for continuous data
    BILINEAR = "bilinear"

    # Cubic convolution - smoother than bilinear
    CUBIC = "cubic"

    # Cubic spline - very smooth
    CUBICSPLINE = "cubicspline"

    # Lanczos - high quality for downsampling
    LANCZOS = "lanczos"

    # Average - for downsampling continuous data
    AVERAGE = "average"

    # Mode - most common value, for categorical downsampling
    MODE = "mode"

    # Min/max - extreme values
    MIN = "min"
    MAX = "max"

    # RMS - root mean square for signal data
    RMS = "rms"

    # Q1/Q3 - quartiles for statistical summary
    Q1 = "q1"
    Q3 = "q3"


class ResolutionUnit(Enum):
    """Units for resolution specification."""

    METERS = "meters"
    DEGREES = "degrees"
    FEET = "feet"
    ARCSECONDS = "arcseconds"
    PIXELS = "pixels"


@dataclass
class Resolution:
    """
    Represents spatial resolution.

    Attributes:
        x: Resolution in x direction
        y: Resolution in y direction
        unit: Unit of measurement
        crs: Coordinate reference system
    """

    x: float
    y: float
    unit: ResolutionUnit = ResolutionUnit.METERS
    crs: Optional[str] = None

    def __post_init__(self):
        """Validate resolution values."""
        if self.x <= 0 or self.y <= 0:
            raise ValueError("Resolution must be positive")

    @property
    def area(self) -> float:
        """Area of one pixel in squared units."""
        return self.x * self.y

    @property
    def is_square(self) -> bool:
        """Check if resolution is square."""
        return np.isclose(self.x, self.y)

    def to_meters(self, latitude: float = 0) -> "Resolution":
        """
        Convert resolution to meters.

        Args:
            latitude: Reference latitude for degree conversion

        Returns:
            Resolution in meters
        """
        if self.unit == ResolutionUnit.METERS:
            return Resolution(self.x, self.y, ResolutionUnit.METERS, self.crs)

        elif self.unit == ResolutionUnit.DEGREES:
            # Approximate conversion at latitude
            meters_per_degree_lat = 111320.0  # At equator
            meters_per_degree_lon = 111320.0 * np.cos(np.radians(latitude))
            return Resolution(
                self.x * meters_per_degree_lon,
                self.y * meters_per_degree_lat,
                ResolutionUnit.METERS,
                self.crs,
            )

        elif self.unit == ResolutionUnit.ARCSECONDS:
            # 1 arcsecond ≈ 30.87 meters at equator
            meters_per_arcsec_lat = 30.87
            meters_per_arcsec_lon = 30.87 * np.cos(np.radians(latitude))
            return Resolution(
                self.x * meters_per_arcsec_lon,
                self.y * meters_per_arcsec_lat,
                ResolutionUnit.METERS,
                self.crs,
            )

        elif self.unit == ResolutionUnit.FEET:
            meters_per_foot = 0.3048
            return Resolution(
                self.x * meters_per_foot,
                self.y * meters_per_foot,
                ResolutionUnit.METERS,
                self.crs,
            )

        else:
            raise ValueError(f"Cannot convert {self.unit} to meters")

    def to_degrees(self, latitude: float = 0) -> "Resolution":
        """
        Convert resolution to degrees.

        Args:
            latitude: Reference latitude for conversion

        Returns:
            Resolution in degrees
        """
        if self.unit == ResolutionUnit.DEGREES:
            return Resolution(self.x, self.y, ResolutionUnit.DEGREES, self.crs)

        meters_res = self.to_meters(latitude)
        meters_per_degree_lat = 111320.0
        meters_per_degree_lon = 111320.0 * np.cos(np.radians(latitude))

        return Resolution(
            meters_res.x / meters_per_degree_lon,
            meters_res.y / meters_per_degree_lat,
            ResolutionUnit.DEGREES,
            self.crs,
        )

    def scale_factor_to(self, target: "Resolution") -> Tuple[float, float]:
        """
        Calculate scale factor needed to match target resolution.

        Args:
            target: Target resolution

        Returns:
            Scale factors (x, y)

        Raises:
            ValueError: If resolutions have different units or target resolution is zero
        """
        # Ensure same units
        if self.unit != target.unit:
            raise ValueError("Resolutions must have same units for comparison")

        if target.x <= 0 or target.y <= 0:
            raise ValueError("Target resolution must be positive")

        return (self.x / target.x, self.y / target.y)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "unit": self.unit.value,
            "crs": self.crs,
        }


@dataclass
class ResamplingConfig:
    """
    Configuration for resampling operations.

    Attributes:
        target_resolution: Target resolution
        method: Resampling method
        nodata: NoData value
        preserve_nodata: Whether to preserve NoData regions
        quality_method: Method for quality layer resampling
        num_threads: Number of processing threads
    """

    target_resolution: Resolution
    method: ResamplingMethod = ResamplingMethod.BILINEAR
    nodata: Optional[float] = None
    preserve_nodata: bool = True
    quality_method: ResamplingMethod = ResamplingMethod.MIN
    num_threads: int = 4


@dataclass
class ResamplingResult:
    """Result from a resampling operation."""

    output_shape: Tuple[int, ...]
    source_resolution: Resolution
    target_resolution: Resolution
    actual_resolution: Resolution
    resampling_method: str
    scale_factors: Tuple[float, float]
    processing_time_seconds: float
    quality_degradation: float  # 0-1, higher means more degradation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_shape": self.output_shape,
            "source_resolution": self.source_resolution.to_dict(),
            "target_resolution": self.target_resolution.to_dict(),
            "actual_resolution": self.actual_resolution.to_dict(),
            "resampling_method": self.resampling_method,
            "scale_factors": self.scale_factors,
            "processing_time_seconds": self.processing_time_seconds,
            "quality_degradation": self.quality_degradation,
            "metadata": self.metadata,
        }


class ResolutionCalculator:
    """
    Utilities for resolution calculations and conversions.

    Handles resolution inference from transforms, comparison,
    and target resolution calculation.

    Example:
        calc = ResolutionCalculator()

        # Get resolution from transform
        res = calc.from_transform(transform)

        # Calculate target shape
        shape = calc.calculate_shape(bounds, resolution)
    """

    def from_transform(
        self,
        transform: Any,
        crs: Optional[str] = None,
    ) -> Resolution:
        """
        Extract resolution from an affine transform.

        Args:
            transform: Affine transform (rasterio/GDAL style)
            crs: Optional CRS for context

        Returns:
            Resolution extracted from transform
        """
        try:
            # rasterio/affine style transform
            res_x = abs(transform[0])  # a
            res_y = abs(transform[4])  # e
        except (TypeError, IndexError):
            try:
                # Try attribute access
                res_x = abs(transform.a)
                res_y = abs(transform.e)
            except AttributeError:
                raise ValueError("Cannot extract resolution from transform")

        # Infer unit from CRS
        unit = ResolutionUnit.METERS
        if crs:
            crs_lower = str(crs).lower()
            if "4326" in crs_lower or "geographic" in crs_lower:
                unit = ResolutionUnit.DEGREES

        return Resolution(res_x, res_y, unit, crs)

    def calculate_shape(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: Resolution,
    ) -> Tuple[int, int]:
        """
        Calculate raster shape for given bounds and resolution.

        Args:
            bounds: Bounds (minx, miny, maxx, maxy)
            resolution: Target resolution

        Returns:
            Shape (height, width)
        """
        minx, miny, maxx, maxy = bounds
        width = int(np.ceil((maxx - minx) / resolution.x))
        height = int(np.ceil((maxy - miny) / resolution.y))
        return (height, width)

    def calculate_transform(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: Resolution,
    ) -> Tuple[float, ...]:
        """
        Calculate affine transform for given bounds and resolution.

        Args:
            bounds: Bounds (minx, miny, maxx, maxy)
            resolution: Target resolution

        Returns:
            Affine transform coefficients (a, b, c, d, e, f)
        """
        minx, miny, maxx, maxy = bounds
        return (
            resolution.x,  # a: pixel width
            0.0,  # b: row rotation
            minx,  # c: x origin
            0.0,  # d: column rotation
            -resolution.y,  # e: pixel height (negative for north-up)
            maxy,  # f: y origin
        )

    def find_common_resolution(
        self,
        resolutions: List[Resolution],
        strategy: str = "finest",
    ) -> Resolution:
        """
        Find common resolution for multiple datasets.

        Args:
            resolutions: List of resolutions
            strategy: "finest", "coarsest", or "median"

        Returns:
            Common resolution
        """
        if not resolutions:
            raise ValueError("No resolutions provided")

        if len(resolutions) == 1:
            return resolutions[0]

        # Ensure all have same unit
        unit = resolutions[0].unit
        for res in resolutions[1:]:
            if res.unit != unit:
                raise ValueError("All resolutions must have same unit")

        x_values = [r.x for r in resolutions]
        y_values = [r.y for r in resolutions]

        if strategy == "finest":
            target_x = min(x_values)
            target_y = min(y_values)
        elif strategy == "coarsest":
            target_x = max(x_values)
            target_y = max(y_values)
        elif strategy == "median":
            target_x = float(np.median(x_values))
            target_y = float(np.median(y_values))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return Resolution(target_x, target_y, unit, resolutions[0].crs)


class SpatialResampler:
    """
    Resamples raster data to different spatial resolutions.

    Supports both upsampling (increasing resolution) and
    downsampling (decreasing resolution) with configurable methods.

    Example:
        resampler = SpatialResampler()

        # Resample array
        resampled, result = resampler.resample_array(
            data=image,
            source_resolution=Resolution(10, 10),
            config=ResamplingConfig(
                target_resolution=Resolution(30, 30),
                method=ResamplingMethod.AVERAGE
            )
        )
    """

    def __init__(self):
        """Initialize resampler."""
        self.resolution_calc = ResolutionCalculator()

    def resample_array(
        self,
        data: np.ndarray,
        source_resolution: Resolution,
        config: ResamplingConfig,
    ) -> Tuple[np.ndarray, ResamplingResult]:
        """
        Resample a numpy array to target resolution.

        Args:
            data: Input array (height, width) or (bands, height, width)
            source_resolution: Source resolution
            config: Resampling configuration

        Returns:
            Tuple of (resampled_data, result)
        """
        import time

        start_time = time.time()

        # Handle 2D arrays
        is_2d = data.ndim == 2
        if is_2d:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        bands, src_height, src_width = data.shape

        # Calculate scale factors (source/target: <1 = downsampling, >1 = upsampling)
        scale_x, scale_y = source_resolution.scale_factor_to(config.target_resolution)

        # Calculate output size
        # scale_factor = source_res / target_res
        # For 10m -> 30m: scale = 10/30 = 0.33, dst_height = 100 * 0.33 = 33
        # For 30m -> 10m: scale = 30/10 = 3.0, dst_height = 30 * 3.0 = 90
        dst_height = int(round(src_height * scale_y))
        dst_width = int(round(src_width * scale_x))

        if dst_height < 1 or dst_width < 1:
            raise ValueError("Target resolution too coarse for input data")

        # Choose resampling implementation
        if scale_x > 1 and scale_y > 1:
            # Downsampling
            output = self._downsample(
                data, (dst_height, dst_width), config.method, config.nodata
            )
            quality_degradation = 0.0  # Downsampling preserves info
        else:
            # Upsampling
            output = self._upsample(
                data, (dst_height, dst_width), config.method, config.nodata
            )
            # Quality degrades with upsampling
            quality_degradation = 1.0 - min(scale_x, scale_y)

        # Handle nodata preservation
        if config.preserve_nodata and config.nodata is not None:
            nodata_mask = self._create_nodata_mask(data, config.nodata)
            if np.any(nodata_mask):
                # Resample mask
                resampled_mask = self._upsample(
                    nodata_mask[np.newaxis, :, :].astype(float),
                    (dst_height, dst_width),
                    ResamplingMethod.NEAREST,
                    None,
                )
                output[:, resampled_mask[0] > 0.5] = config.nodata

        processing_time = time.time() - start_time

        # Calculate actual achieved resolution
        actual_res = Resolution(
            source_resolution.x * (src_width / dst_width),
            source_resolution.y * (src_height / dst_height),
            source_resolution.unit,
            source_resolution.crs,
        )

        result = ResamplingResult(
            output_shape=output.shape,
            source_resolution=source_resolution,
            target_resolution=config.target_resolution,
            actual_resolution=actual_res,
            resampling_method=config.method.value,
            scale_factors=(scale_x, scale_y),
            processing_time_seconds=processing_time,
            quality_degradation=quality_degradation,
        )

        if is_2d:
            output = output[0]

        return output, result

    def resample_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        config: ResamplingConfig,
        overwrite: bool = False,
    ) -> ResamplingResult:
        """
        Resample a raster file to target resolution.

        Args:
            input_path: Path to input raster
            output_path: Path for output raster
            config: Resampling configuration
            overwrite: Whether to overwrite existing output

        Returns:
            ResamplingResult with details

        Raises:
            FileNotFoundError: If input doesn't exist
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
        except ImportError:
            raise ImportError("rasterio is required for file resampling")

        start_time = time.time()
        logger.info(f"Resampling {input_path} to {config.target_resolution.x}x{config.target_resolution.y}")

        with rasterio.open(input_path) as src:
            source_res = self.resolution_calc.from_transform(src.transform, str(src.crs))

            # Calculate scale factors and output size
            scale_x, scale_y = source_res.scale_factor_to(config.target_resolution)
            dst_height = int(round(src.height / scale_y))
            dst_width = int(round(src.width / scale_x))

            # Map method to rasterio resampling
            resampling = self._get_rasterio_resampling(config.method)

            # Calculate new transform
            transform = src.transform * src.transform.scale(
                src.width / dst_width,
                src.height / dst_height,
            )

            # Update profile
            profile = src.profile.copy()
            profile.update({
                "height": dst_height,
                "width": dst_width,
                "transform": transform,
            })

            # Read and resample
            with rasterio.open(output_path, "w", **profile) as dst:
                for band_idx in range(1, src.count + 1):
                    data = src.read(
                        band_idx,
                        out_shape=(dst_height, dst_width),
                        resampling=resampling,
                    )
                    dst.write(data, band_idx)

        processing_time = time.time() - start_time

        # Calculate quality degradation
        if scale_x > 1 and scale_y > 1:
            quality_degradation = 0.0
        else:
            quality_degradation = 1.0 - min(scale_x, scale_y)

        # Calculate actual resolution
        actual_res = Resolution(
            source_res.x * scale_x,
            source_res.y * scale_y,
            source_res.unit,
            source_res.crs,
        )

        logger.info(f"Resampling complete: {output_path} ({processing_time:.2f}s)")

        return ResamplingResult(
            output_shape=(dst_height, dst_width),
            source_resolution=source_res,
            target_resolution=config.target_resolution,
            actual_resolution=actual_res,
            resampling_method=config.method.value,
            scale_factors=(scale_x, scale_y),
            processing_time_seconds=processing_time,
            quality_degradation=quality_degradation,
        )

    def _downsample(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, int],
        method: ResamplingMethod,
        nodata: Optional[float],
    ) -> np.ndarray:
        """Downsample array using specified method."""
        bands, src_height, src_width = data.shape
        dst_height, dst_width = target_shape

        # Calculate block sizes
        block_h = src_height / dst_height
        block_w = src_width / dst_width

        output = np.zeros((bands, dst_height, dst_width), dtype=data.dtype)

        for b in range(bands):
            for y in range(dst_height):
                for x in range(dst_width):
                    # Get block bounds
                    y0 = int(y * block_h)
                    y1 = int((y + 1) * block_h)
                    x0 = int(x * block_w)
                    x1 = int((x + 1) * block_w)

                    # Ensure bounds are valid
                    y1 = min(y1, src_height)
                    x1 = min(x1, src_width)

                    if y1 <= y0 or x1 <= x0:
                        continue

                    block = data[b, y0:y1, x0:x1].astype(float)

                    # Mask nodata if specified
                    if nodata is not None:
                        block = np.where(block == nodata, np.nan, block)

                    # Apply aggregation method
                    if method == ResamplingMethod.AVERAGE:
                        value = np.nanmean(block)
                    elif method == ResamplingMethod.MIN:
                        value = np.nanmin(block)
                    elif method == ResamplingMethod.MAX:
                        value = np.nanmax(block)
                    elif method == ResamplingMethod.MODE:
                        # For mode, use most common non-nan value
                        valid = block[~np.isnan(block)]
                        if len(valid) > 0:
                            values, counts = np.unique(valid, return_counts=True)
                            value = values[np.argmax(counts)]
                        else:
                            value = nodata if nodata is not None else 0
                    elif method == ResamplingMethod.NEAREST:
                        # Use center pixel
                        cy = (y0 + y1) // 2
                        cx = (x0 + x1) // 2
                        value = data[b, cy, cx]
                    else:
                        value = np.nanmean(block)

                    output[b, y, x] = value if not np.isnan(value) else (nodata or 0)

        return output

    def _upsample(
        self,
        data: np.ndarray,
        target_shape: Tuple[int, int],
        method: ResamplingMethod,
        nodata: Optional[float],
    ) -> np.ndarray:
        """Upsample array using specified method."""
        from scipy import ndimage

        bands, src_height, src_width = data.shape
        dst_height, dst_width = target_shape

        # Calculate zoom factors
        zoom_y = dst_height / src_height
        zoom_x = dst_width / src_width

        output = np.zeros((bands, dst_height, dst_width), dtype=data.dtype)

        # Map method to scipy order
        if method == ResamplingMethod.NEAREST:
            order = 0
        elif method == ResamplingMethod.BILINEAR:
            order = 1
        elif method in [ResamplingMethod.CUBIC, ResamplingMethod.CUBICSPLINE]:
            order = 3
        elif method == ResamplingMethod.LANCZOS:
            order = 5
        else:
            order = 1  # Default to bilinear

        for b in range(bands):
            band_data = data[b].astype(float)

            # Handle nodata
            if nodata is not None:
                mask = band_data == nodata
                band_data = np.where(mask, np.nan, band_data)

            # Apply zoom
            output[b] = ndimage.zoom(
                band_data,
                (zoom_y, zoom_x),
                order=order,
                mode="nearest",
            )

            # Restore nodata
            if nodata is not None:
                output[b] = np.where(np.isnan(output[b]), nodata, output[b])

        return output.astype(data.dtype)

    def _create_nodata_mask(
        self,
        data: np.ndarray,
        nodata: float,
    ) -> np.ndarray:
        """Create 2D mask of nodata pixels."""
        if data.ndim == 2:
            return data == nodata
        else:
            # True if any band has nodata
            return np.any(data == nodata, axis=0)

    def _get_rasterio_resampling(self, method: ResamplingMethod) -> Any:
        """Map ResamplingMethod to rasterio Resampling."""
        try:
            from rasterio.enums import Resampling
        except ImportError:
            raise ImportError("rasterio is required")

        mapping = {
            ResamplingMethod.NEAREST: Resampling.nearest,
            ResamplingMethod.BILINEAR: Resampling.bilinear,
            ResamplingMethod.CUBIC: Resampling.cubic,
            ResamplingMethod.CUBICSPLINE: Resampling.cubic_spline,
            ResamplingMethod.LANCZOS: Resampling.lanczos,
            ResamplingMethod.AVERAGE: Resampling.average,
            ResamplingMethod.MODE: Resampling.mode,
            ResamplingMethod.MIN: Resampling.min,
            ResamplingMethod.MAX: Resampling.max,
            ResamplingMethod.RMS: Resampling.rms,
            ResamplingMethod.Q1: Resampling.q1,
            ResamplingMethod.Q3: Resampling.q3,
        }

        return mapping.get(method, Resampling.bilinear)


class ResolutionHarmonizer:
    """
    Harmonizes multiple datasets to a common resolution.

    Used when combining data from different sensors or sources
    with varying spatial resolutions.

    Example:
        harmonizer = ResolutionHarmonizer()

        # Harmonize multiple datasets
        harmonized = harmonizer.harmonize(
            datasets={
                "sentinel2": (s2_data, Resolution(10, 10)),
                "sentinel1": (s1_data, Resolution(20, 20)),
            },
            target_resolution=Resolution(15, 15)
        )
    """

    def __init__(self):
        """Initialize harmonizer."""
        self.resampler = SpatialResampler()
        self.resolution_calc = ResolutionCalculator()

    def harmonize(
        self,
        datasets: Dict[str, Tuple[np.ndarray, Resolution]],
        target_resolution: Optional[Resolution] = None,
        strategy: str = "finest",
        method: ResamplingMethod = ResamplingMethod.BILINEAR,
    ) -> Dict[str, Tuple[np.ndarray, ResamplingResult]]:
        """
        Harmonize multiple datasets to common resolution.

        Args:
            datasets: Dict of name to (data, resolution) tuples
            target_resolution: Target resolution (auto if None)
            strategy: "finest", "coarsest", or "median" if target not specified
            method: Resampling method

        Returns:
            Dict of name to (resampled_data, result) tuples
        """
        if not datasets:
            return {}

        # Determine target resolution
        if target_resolution is None:
            resolutions = [res for _, res in datasets.values()]
            target_resolution = self.resolution_calc.find_common_resolution(
                resolutions, strategy
            )

        logger.info(
            f"Harmonizing {len(datasets)} datasets to "
            f"{target_resolution.x}x{target_resolution.y} {target_resolution.unit.value}"
        )

        config = ResamplingConfig(
            target_resolution=target_resolution,
            method=method,
        )

        results = {}
        for name, (data, source_res) in datasets.items():
            # Skip if already at target resolution
            if np.isclose(source_res.x, target_resolution.x) and np.isclose(
                source_res.y, target_resolution.y
            ):
                result = ResamplingResult(
                    output_shape=data.shape,
                    source_resolution=source_res,
                    target_resolution=target_resolution,
                    actual_resolution=source_res,
                    resampling_method="none",
                    scale_factors=(1.0, 1.0),
                    processing_time_seconds=0.0,
                    quality_degradation=0.0,
                )
                results[name] = (data, result)
            else:
                resampled, result = self.resampler.resample_array(
                    data, source_res, config
                )
                results[name] = (resampled, result)

        return results


def resample_to_resolution(
    data: np.ndarray,
    source_resolution: Tuple[float, float],
    target_resolution: Tuple[float, float],
    method: str = "bilinear",
    unit: str = "meters",
) -> Tuple[np.ndarray, ResamplingResult]:
    """
    Convenience function to resample an array.

    Args:
        data: Input array
        source_resolution: Source resolution (x, y)
        target_resolution: Target resolution (x, y)
        method: Resampling method name
        unit: Resolution unit

    Returns:
        Tuple of (resampled_data, result)
    """
    src_res = Resolution(
        source_resolution[0],
        source_resolution[1],
        ResolutionUnit(unit),
    )
    tgt_res = Resolution(
        target_resolution[0],
        target_resolution[1],
        ResolutionUnit(unit),
    )
    config = ResamplingConfig(
        target_resolution=tgt_res,
        method=ResamplingMethod(method.lower()),
    )
    resampler = SpatialResampler()
    return resampler.resample_array(data, src_res, config)


def calculate_resolution(
    bounds: Tuple[float, float, float, float],
    shape: Tuple[int, int],
) -> Resolution:
    """
    Calculate resolution from bounds and shape.

    Args:
        bounds: Bounds (minx, miny, maxx, maxy)
        shape: Shape (height, width)

    Returns:
        Resolution
    """
    minx, miny, maxx, maxy = bounds
    height, width = shape
    return Resolution(
        (maxx - minx) / width,
        (maxy - miny) / height,
    )
