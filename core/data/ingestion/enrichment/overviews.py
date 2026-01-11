"""
Image Pyramid (Overview) Generation for Raster Data.

Generates multi-resolution overviews (image pyramids) for efficient display
at different zoom levels. Overviews enable fast rendering of large datasets
by pre-computing downsampled versions.

Key Features:
- Multiple resampling methods (average, nearest, bilinear, cubic, etc.)
- Configurable overview levels (powers of 2, custom factors)
- Internal (embedded) or external overview files
- Memory-efficient tiled processing
- Support for COG-compatible overview generation
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class OverviewResampling(Enum):
    """Resampling methods for overview generation."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    CUBIC = "cubic"
    CUBICSPLINE = "cubicspline"
    LANCZOS = "lanczos"
    AVERAGE = "average"
    RMS = "rms"
    MODE = "mode"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    Q1 = "q1"
    Q3 = "q3"

    @classmethod
    def for_data_type(cls, data_type: str) -> "OverviewResampling":
        """
        Select appropriate resampling method based on data type.

        Args:
            data_type: Type of data (continuous, categorical, boolean, etc.)

        Returns:
            Recommended resampling method
        """
        type_mapping = {
            "continuous": cls.AVERAGE,
            "categorical": cls.MODE,
            "boolean": cls.MODE,
            "elevation": cls.BILINEAR,
            "imagery": cls.LANCZOS,
            "classification": cls.NEAREST,
            "index": cls.AVERAGE,
            "temperature": cls.BILINEAR,
            "count": cls.AVERAGE,
            "mask": cls.NEAREST,
        }
        return type_mapping.get(data_type.lower(), cls.AVERAGE)


class OverviewFormat(Enum):
    """Overview storage format."""

    INTERNAL = "internal"  # Embedded in the main file
    EXTERNAL = "external"  # Separate .ovr file
    SIDECAR = "sidecar"  # Separate files with same base name


@dataclass
class OverviewConfig:
    """
    Configuration for overview generation.

    Attributes:
        factors: List of overview reduction factors (e.g., [2, 4, 8, 16])
        resampling: Resampling method for overview generation
        format: Where to store overviews (internal, external, sidecar)
        blocksize: Tile size for overview computation
        compress: Enable compression for overviews
        photometric: Photometric interpretation (RGB, YCBCR, MINISBLACK)
        sparse: Enable sparse file support for large nodata areas
        min_size: Minimum dimension for smallest overview level
        force_power_of_two: Ensure factors are powers of 2
    """

    factors: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    resampling: OverviewResampling = OverviewResampling.AVERAGE
    format: OverviewFormat = OverviewFormat.INTERNAL
    blocksize: int = 512
    compress: bool = True
    photometric: Optional[str] = None
    sparse: bool = False
    min_size: int = 256
    force_power_of_two: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.factors:
            raise ValueError("At least one overview factor is required")

        for factor in self.factors:
            if factor < 2:
                raise ValueError(f"Overview factors must be >= 2, got {factor}")
            if self.force_power_of_two and (factor & (factor - 1)) != 0:
                raise ValueError(f"Factor {factor} is not a power of 2")

        if self.blocksize not in [128, 256, 512, 1024, 2048]:
            raise ValueError(
                f"blocksize must be 128, 256, 512, 1024, or 2048, got {self.blocksize}"
            )

    def auto_factors(
        self, width: int, height: int, min_size: Optional[int] = None
    ) -> List[int]:
        """
        Calculate overview factors based on image dimensions.

        Generates factors until the smallest dimension reaches min_size.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            min_size: Minimum dimension for smallest overview (defaults to config)

        Returns:
            List of overview factors
        """
        min_dim = min(width, height)
        target_min = min_size or self.min_size
        factors = []
        current_factor = 2

        while min_dim // current_factor >= target_min:
            factors.append(current_factor)
            current_factor *= 2

        return factors if factors else [2]


@dataclass
class OverviewLevel:
    """Information about a single overview level."""

    factor: int
    width: int
    height: int
    size_bytes: int
    resampling: str

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Return width, height tuple."""
        return (self.width, self.height)

    @property
    def reduction(self) -> str:
        """Return human-readable reduction description."""
        return f"1:{self.factor} ({self.width}x{self.height})"


@dataclass
class OverviewResult:
    """Result from overview generation."""

    input_path: Path
    output_path: Optional[Path]  # None if internal overviews
    levels: List[OverviewLevel]
    total_overview_bytes: int
    overhead_percent: float
    resampling_method: str
    format: OverviewFormat
    metadata: Dict[str, Any]

    @property
    def level_count(self) -> int:
        """Number of overview levels generated."""
        return len(self.levels)

    @property
    def factors(self) -> List[int]:
        """List of overview factors."""
        return [level.factor for level in self.levels]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path) if self.output_path else None,
            "levels": [
                {
                    "factor": level.factor,
                    "width": level.width,
                    "height": level.height,
                    "size_bytes": level.size_bytes,
                    "resampling": level.resampling,
                }
                for level in self.levels
            ],
            "total_overview_bytes": self.total_overview_bytes,
            "overhead_percent": self.overhead_percent,
            "resampling_method": self.resampling_method,
            "format": self.format.value,
            "metadata": self.metadata,
        }


class OverviewGenerator:
    """
    Generator for image pyramid (overview) levels.

    Creates multi-resolution versions of raster data for efficient
    display at various zoom levels. Supports both internal (embedded)
    and external overview storage.

    Example:
        generator = OverviewGenerator(OverviewConfig(
            factors=[2, 4, 8, 16],
            resampling=OverviewResampling.AVERAGE
        ))
        result = generator.generate("large_image.tif")

        # Or with automatic factor calculation
        result = generator.generate_auto("large_image.tif")
    """

    def __init__(self, config: Optional[OverviewConfig] = None):
        """
        Initialize overview generator.

        Args:
            config: Overview generation configuration (uses defaults if None)
        """
        self.config = config or OverviewConfig()

    def generate(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        factors: Optional[List[int]] = None,
    ) -> OverviewResult:
        """
        Generate overviews for a raster file.

        Args:
            input_path: Path to input raster file
            output_path: Path for external overviews (ignored if internal)
            factors: Override overview factors (uses config if None)

        Returns:
            OverviewResult with generation details

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input is not a valid raster
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError:
            raise ImportError("rasterio is required for overview generation")

        use_factors = factors or self.config.factors
        resampling = getattr(Resampling, self.config.resampling.value.upper())

        logger.info(f"Generating overviews for {input_path} at factors {use_factors}")

        # Get original file size
        original_size = input_path.stat().st_size

        # Generate overviews
        with rasterio.open(input_path, "r+") as src:
            src_width = src.width
            src_height = src.height
            src_count = src.count
            src_dtype = src.dtypes[0]

            # Build overviews
            src.build_overviews(use_factors, resampling)

            # Update tags to document overview creation
            src.update_tags(
                ns="rio_overview",
                resampling=self.config.resampling.value,
                factors=",".join(map(str, use_factors)),
            )

        # Calculate overview level info
        levels = []
        for factor in use_factors:
            level_width = max(1, src_width // factor)
            level_height = max(1, src_height // factor)

            # Estimate size (approximate - actual size depends on compression)
            bytes_per_pixel = np.dtype(src_dtype).itemsize * src_count
            level_size = level_width * level_height * bytes_per_pixel

            levels.append(
                OverviewLevel(
                    factor=factor,
                    width=level_width,
                    height=level_height,
                    size_bytes=level_size,
                    resampling=self.config.resampling.value,
                )
            )

        # Calculate total overhead
        new_size = input_path.stat().st_size
        total_overview_bytes = new_size - original_size
        overhead_percent = (
            (total_overview_bytes / original_size) * 100 if original_size > 0 else 0.0
        )

        logger.info(
            f"Overviews generated: {len(levels)} levels, "
            f"{total_overview_bytes / 1024 / 1024:.2f} MB overhead "
            f"({overhead_percent:.1f}%)"
        )

        return OverviewResult(
            input_path=input_path,
            output_path=None,  # Internal overviews
            levels=levels,
            total_overview_bytes=total_overview_bytes,
            overhead_percent=overhead_percent,
            resampling_method=self.config.resampling.value,
            format=OverviewFormat.INTERNAL,
            metadata={
                "original_size_bytes": original_size,
                "final_size_bytes": new_size,
                "band_count": src_count,
                "dtype": src_dtype,
            },
        )

    def generate_auto(
        self,
        input_path: Union[str, Path],
        min_size: Optional[int] = None,
    ) -> OverviewResult:
        """
        Generate overviews with automatically calculated factors.

        Factors are calculated to produce overviews down to a minimum
        dimension size.

        Args:
            input_path: Path to input raster file
            min_size: Minimum dimension for smallest overview

        Returns:
            OverviewResult with generation details
        """
        input_path = Path(input_path)

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for overview generation")

        # Get image dimensions
        with rasterio.open(input_path) as src:
            width = src.width
            height = src.height

        # Calculate appropriate factors
        factors = self.config.auto_factors(width, height, min_size)
        logger.info(f"Auto-calculated overview factors: {factors}")

        return self.generate(input_path, factors=factors)

    def generate_from_array(
        self,
        data: np.ndarray,
        factors: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """
        Generate in-memory overview arrays from a numpy array.

        Useful for creating overviews before writing to disk.

        Args:
            data: Input array (bands, height, width) or (height, width)
            factors: Overview factors (uses config if None)

        Returns:
            List of downsampled arrays, one per factor
        """
        use_factors = factors or self.config.factors

        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        _, height, width = data.shape
        overviews = []

        for factor in use_factors:
            new_height = max(1, height // factor)
            new_width = max(1, width // factor)

            # Create downsampled version
            overview = self._downsample_array(data, new_height, new_width)
            overviews.append(overview)

            logger.debug(f"Created overview at 1:{factor} ({new_width}x{new_height})")

        return overviews

    def _downsample_array(
        self, data: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        """
        Downsample an array using the configured resampling method.

        Args:
            data: Input array (bands, height, width)
            target_height: Target height
            target_width: Target width

        Returns:
            Downsampled array
        """
        bands, height, width = data.shape
        result = np.zeros((bands, target_height, target_width), dtype=data.dtype)

        # Calculate block size for averaging
        y_factor = height / target_height
        x_factor = width / target_width

        for b in range(bands):
            band_data = data[b]

            if self.config.resampling == OverviewResampling.NEAREST:
                # Nearest neighbor - simple indexing
                y_indices = (np.arange(target_height) * y_factor).astype(int)
                x_indices = (np.arange(target_width) * x_factor).astype(int)
                y_indices = np.clip(y_indices, 0, height - 1)
                x_indices = np.clip(x_indices, 0, width - 1)
                result[b] = band_data[np.ix_(y_indices, x_indices)]

            elif self.config.resampling == OverviewResampling.AVERAGE:
                # Block average
                for y in range(target_height):
                    for x in range(target_width):
                        y_start = int(y * y_factor)
                        y_end = int((y + 1) * y_factor)
                        x_start = int(x * x_factor)
                        x_end = int((x + 1) * x_factor)

                        block = band_data[y_start:y_end, x_start:x_end]
                        if block.size > 0:
                            result[b, y, x] = np.nanmean(block)

            elif self.config.resampling == OverviewResampling.MAX:
                # Block maximum
                for y in range(target_height):
                    for x in range(target_width):
                        y_start = int(y * y_factor)
                        y_end = int((y + 1) * y_factor)
                        x_start = int(x * x_factor)
                        x_end = int((x + 1) * x_factor)

                        block = band_data[y_start:y_end, x_start:x_end]
                        if block.size > 0:
                            result[b, y, x] = np.nanmax(block)

            elif self.config.resampling == OverviewResampling.MIN:
                # Block minimum
                for y in range(target_height):
                    for x in range(target_width):
                        y_start = int(y * y_factor)
                        y_end = int((y + 1) * y_factor)
                        x_start = int(x * x_factor)
                        x_end = int((x + 1) * x_factor)

                        block = band_data[y_start:y_end, x_start:x_end]
                        if block.size > 0:
                            result[b, y, x] = np.nanmin(block)

            elif self.config.resampling == OverviewResampling.MODE:
                # Block mode (for categorical data)
                for y in range(target_height):
                    for x in range(target_width):
                        y_start = int(y * y_factor)
                        y_end = int((y + 1) * y_factor)
                        x_start = int(x * x_factor)
                        x_end = int((x + 1) * x_factor)

                        block = band_data[y_start:y_end, x_start:x_end]
                        if block.size > 0:
                            values, counts = np.unique(block, return_counts=True)
                            result[b, y, x] = values[np.argmax(counts)]

            else:
                # Fall back to nearest for unsupported methods
                y_indices = (np.arange(target_height) * y_factor).astype(int)
                x_indices = (np.arange(target_width) * x_factor).astype(int)
                y_indices = np.clip(y_indices, 0, height - 1)
                x_indices = np.clip(x_indices, 0, width - 1)
                result[b] = band_data[np.ix_(y_indices, x_indices)]

        return result

    def check_overviews(self, path: Union[str, Path]) -> Tuple[bool, List[int]]:
        """
        Check if a file has overviews and return their factors.

        Args:
            path: Path to raster file

        Returns:
            Tuple of (has_overviews, list_of_factors)
        """
        path = Path(path)

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for overview checking")

        with rasterio.open(path) as src:
            overviews = src.overviews(1)
            has_overviews = len(overviews) > 0
            return has_overviews, list(overviews)

    def needs_overviews(
        self,
        path: Union[str, Path],
        min_dimension: int = 1024,
    ) -> bool:
        """
        Check if a file would benefit from overviews.

        Args:
            path: Path to raster file
            min_dimension: Minimum dimension to recommend overviews

        Returns:
            True if file should have overviews
        """
        path = Path(path)

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for overview checking")

        with rasterio.open(path) as src:
            # Check if already has overviews
            if src.overviews(1):
                return False

            # Check if dimensions warrant overviews
            return src.width >= min_dimension or src.height >= min_dimension


def generate_overviews(
    input_path: Union[str, Path],
    factors: Optional[List[int]] = None,
    resampling: str = "average",
    auto: bool = False,
    min_size: int = 256,
) -> OverviewResult:
    """
    Convenience function to generate overviews for a raster file.

    Args:
        input_path: Path to input raster file
        factors: Override overview factors (ignored if auto=True)
        resampling: Resampling method (average, nearest, bilinear, etc.)
        auto: Automatically calculate factors based on image dimensions
        min_size: Minimum dimension for auto factor calculation

    Returns:
        OverviewResult with generation details
    """
    config = OverviewConfig(
        factors=factors or [2, 4, 8, 16, 32],
        resampling=OverviewResampling(resampling.lower()),
        min_size=min_size,
    )
    generator = OverviewGenerator(config)

    if auto:
        return generator.generate_auto(input_path, min_size=min_size)
    else:
        return generator.generate(input_path, factors=factors)
