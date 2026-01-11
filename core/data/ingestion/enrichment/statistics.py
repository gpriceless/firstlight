"""
Band Statistics Computation for Raster Data.

Computes comprehensive statistics for each band of raster data, including:
- Basic statistics (min, max, mean, std, median)
- Histogram generation
- Percentile calculations
- NoData handling
- Valid pixel statistics
- Spatial distribution metrics

Statistics are essential for:
- Data quality assessment
- Visualization (stretch/contrast)
- Anomaly detection
- Cross-dataset comparison
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class HistogramType(Enum):
    """Histogram computation methods."""

    LINEAR = "linear"  # Equal-width bins
    LOGARITHMIC = "logarithmic"  # Log-scale bins
    PERCENTILE = "percentile"  # Percentile-based bins
    AUTOMATIC = "automatic"  # Auto-select based on data distribution


@dataclass
class StatisticsConfig:
    """
    Configuration for statistics computation.

    Attributes:
        compute_histogram: Whether to compute histograms
        histogram_bins: Number of histogram bins
        histogram_type: Type of histogram binning
        percentiles: List of percentiles to compute (0-100)
        sample_size: Max pixels to sample (None for full computation)
        nodata_values: List of values to treat as nodata
        approx_ok: Allow approximate statistics for large files
        per_block: Compute block-level statistics
    """

    compute_histogram: bool = True
    histogram_bins: int = 256
    histogram_type: HistogramType = HistogramType.LINEAR
    percentiles: List[float] = field(
        default_factory=lambda: [1, 5, 10, 25, 50, 75, 90, 95, 99]
    )
    sample_size: Optional[int] = None
    nodata_values: List[float] = field(default_factory=list)
    approx_ok: bool = True
    per_block: bool = False

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.histogram_bins < 1:
            raise ValueError(f"histogram_bins must be >= 1, got {self.histogram_bins}")

        for p in self.percentiles:
            if not 0 <= p <= 100:
                raise ValueError(f"percentiles must be in [0, 100], got {p}")


@dataclass
class BandStatistics:
    """
    Statistics for a single raster band.

    Comprehensive statistics including basic measures, percentiles,
    histogram, and data quality metrics.
    """

    band_index: int
    min: float
    max: float
    mean: float
    std: float
    median: float
    variance: float
    sum: float
    count: int
    valid_count: int
    nodata_count: int
    valid_percent: float
    percentiles: Dict[float, float]
    histogram: Optional[Tuple[np.ndarray, np.ndarray]]
    dtype: str
    unique_count: Optional[int]
    is_integer: bool
    is_categorical: bool

    @property
    def range(self) -> float:
        """Data range (max - min)."""
        return self.max - self.min

    @property
    def cv(self) -> float:
        """Coefficient of variation (std / mean)."""
        if self.mean != 0:
            return self.std / abs(self.mean)
        return float("inf") if self.std > 0 else 0.0

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio (mean / std)."""
        if self.std != 0:
            return abs(self.mean) / self.std
        return float("inf") if self.mean != 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "band_index": self.band_index,
            "min": float(self.min) if np.isfinite(self.min) else None,
            "max": float(self.max) if np.isfinite(self.max) else None,
            "mean": float(self.mean) if np.isfinite(self.mean) else None,
            "std": float(self.std) if np.isfinite(self.std) else None,
            "median": float(self.median) if np.isfinite(self.median) else None,
            "variance": float(self.variance) if np.isfinite(self.variance) else None,
            "sum": float(self.sum) if np.isfinite(self.sum) else None,
            "count": self.count,
            "valid_count": self.valid_count,
            "nodata_count": self.nodata_count,
            "valid_percent": round(self.valid_percent, 2),
            "percentiles": {k: float(v) for k, v in self.percentiles.items()},
            "dtype": self.dtype,
            "unique_count": self.unique_count,
            "is_integer": self.is_integer,
            "is_categorical": self.is_categorical,
            "range": float(self.range) if np.isfinite(self.range) else None,
            "cv": float(self.cv) if np.isfinite(self.cv) else None,
        }

        # Add histogram if present (as lists for JSON serialization)
        if self.histogram is not None:
            counts, edges = self.histogram
            result["histogram"] = {
                "counts": counts.tolist(),
                "bin_edges": edges.tolist(),
            }

        return result


@dataclass
class RasterStatistics:
    """Complete statistics for a raster file."""

    path: Path
    band_stats: List[BandStatistics]
    width: int
    height: int
    band_count: int
    crs: Optional[str]
    bounds: Optional[Tuple[float, float, float, float]]
    global_min: float
    global_max: float
    total_pixels: int
    total_valid_pixels: int
    overall_valid_percent: float
    metadata: Dict[str, Any]

    def get_band(self, index: int) -> BandStatistics:
        """
        Get statistics for a specific band.

        Args:
            index: Band index (1-based)

        Returns:
            BandStatistics for the requested band
        """
        if index < 1 or index > len(self.band_stats):
            raise IndexError(
                f"Band index {index} out of range [1, {len(self.band_stats)}]"
            )
        return self.band_stats[index - 1]

    @property
    def global_range(self) -> float:
        """Global data range across all bands."""
        return self.global_max - self.global_min

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "band_stats": [bs.to_dict() for bs in self.band_stats],
            "width": self.width,
            "height": self.height,
            "band_count": self.band_count,
            "crs": self.crs,
            "bounds": self.bounds,
            "global_min": float(self.global_min)
            if np.isfinite(self.global_min)
            else None,
            "global_max": float(self.global_max)
            if np.isfinite(self.global_max)
            else None,
            "global_range": float(self.global_range)
            if np.isfinite(self.global_range)
            else None,
            "total_pixels": self.total_pixels,
            "total_valid_pixels": self.total_valid_pixels,
            "overall_valid_percent": round(self.overall_valid_percent, 2),
            "metadata": self.metadata,
        }


class StatisticsCalculator:
    """
    Calculator for raster band statistics.

    Computes comprehensive statistics for each band of raster data,
    with support for large files, sampling, and nodata handling.

    Example:
        calculator = StatisticsCalculator(StatisticsConfig(
            compute_histogram=True,
            percentiles=[5, 25, 50, 75, 95]
        ))
        stats = calculator.compute("image.tif")

        for band in stats.band_stats:
            print(f"Band {band.band_index}: mean={band.mean:.2f}")
    """

    def __init__(self, config: Optional[StatisticsConfig] = None):
        """
        Initialize statistics calculator.

        Args:
            config: Statistics computation configuration
        """
        self.config = config or StatisticsConfig()

    def compute(self, path: Union[str, Path]) -> RasterStatistics:
        """
        Compute statistics for all bands of a raster file.

        Args:
            path: Path to raster file

        Returns:
            RasterStatistics with comprehensive band statistics

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input is not a valid raster
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for statistics computation")

        logger.info(f"Computing statistics for {path}")

        with rasterio.open(path) as src:
            width = src.width
            height = src.height
            band_count = src.count
            crs = str(src.crs) if src.crs else None
            bounds = src.bounds
            file_nodata = src.nodata

            band_stats = []
            global_min = float("inf")
            global_max = float("-inf")
            total_valid = 0

            for band_idx in range(1, band_count + 1):
                logger.debug(f"Computing statistics for band {band_idx}")

                # Read band data
                data = src.read(band_idx)
                dtype = str(data.dtype)

                # Create nodata mask
                nodata_values = list(self.config.nodata_values)
                if file_nodata is not None:
                    nodata_values.append(file_nodata)

                mask = np.zeros(data.shape, dtype=bool)
                for nd in nodata_values:
                    mask |= data == nd
                # Also mask NaN and Inf
                mask |= ~np.isfinite(data.astype(float))

                # Get valid data
                valid_data = data[~mask]
                valid_count = len(valid_data)
                nodata_count = mask.sum()
                total_count = data.size
                valid_percent = (valid_count / total_count * 100) if total_count > 0 else 0.0

                total_valid += valid_count

                # Compute statistics on valid data
                if valid_count > 0:
                    # Sample if requested and data is large
                    if (
                        self.config.sample_size
                        and valid_count > self.config.sample_size
                    ):
                        sample_indices = np.random.choice(
                            valid_count, self.config.sample_size, replace=False
                        )
                        sample_data = valid_data[sample_indices]
                    else:
                        sample_data = valid_data

                    min_val = float(np.min(valid_data))
                    max_val = float(np.max(valid_data))
                    mean_val = float(np.mean(sample_data))
                    std_val = float(np.std(sample_data))
                    median_val = float(np.median(sample_data))
                    variance_val = float(np.var(sample_data))
                    sum_val = float(np.sum(valid_data))

                    # Update global min/max
                    global_min = min(global_min, min_val)
                    global_max = max(global_max, max_val)

                    # Compute percentiles
                    percentiles = {}
                    for p in self.config.percentiles:
                        percentiles[p] = float(np.percentile(sample_data, p))

                    # Compute histogram
                    histogram = None
                    if self.config.compute_histogram:
                        histogram = self._compute_histogram(
                            sample_data, min_val, max_val
                        )

                    # Count unique values (for integer/categorical data)
                    is_integer = np.issubdtype(data.dtype, np.integer)
                    unique_count = None
                    is_categorical = False

                    if is_integer and valid_count <= 100000:
                        unique_vals = np.unique(valid_data)
                        unique_count = len(unique_vals)
                        # Consider categorical if few unique values relative to range
                        value_range = max_val - min_val + 1
                        if value_range > 0:
                            is_categorical = unique_count / value_range < 0.1
                else:
                    # All nodata
                    min_val = float("nan")
                    max_val = float("nan")
                    mean_val = float("nan")
                    std_val = float("nan")
                    median_val = float("nan")
                    variance_val = float("nan")
                    sum_val = 0.0
                    percentiles = {p: float("nan") for p in self.config.percentiles}
                    histogram = None
                    is_integer = np.issubdtype(data.dtype, np.integer)
                    unique_count = 0
                    is_categorical = False

                band_stat = BandStatistics(
                    band_index=band_idx,
                    min=min_val,
                    max=max_val,
                    mean=mean_val,
                    std=std_val,
                    median=median_val,
                    variance=variance_val,
                    sum=sum_val,
                    count=total_count,
                    valid_count=valid_count,
                    nodata_count=nodata_count,
                    valid_percent=valid_percent,
                    percentiles=percentiles,
                    histogram=histogram,
                    dtype=dtype,
                    unique_count=unique_count,
                    is_integer=is_integer,
                    is_categorical=is_categorical,
                )
                band_stats.append(band_stat)

        total_pixels = width * height * band_count
        overall_valid_percent = (
            (total_valid / total_pixels * 100) if total_pixels > 0 else 0.0
        )

        # Handle case where all values were nodata
        if global_min == float("inf"):
            global_min = float("nan")
        if global_max == float("-inf"):
            global_max = float("nan")

        logger.info(
            f"Statistics computed: {band_count} bands, "
            f"valid data: {overall_valid_percent:.1f}%"
        )

        return RasterStatistics(
            path=path,
            band_stats=band_stats,
            width=width,
            height=height,
            band_count=band_count,
            crs=crs,
            bounds=(
                (bounds.left, bounds.bottom, bounds.right, bounds.top)
                if bounds
                else None
            ),
            global_min=global_min,
            global_max=global_max,
            total_pixels=total_pixels,
            total_valid_pixels=total_valid,
            overall_valid_percent=overall_valid_percent,
            metadata={"computation_method": "full" if not self.config.sample_size else "sampled"},
        )

    def compute_from_array(
        self,
        data: np.ndarray,
        nodata: Optional[float] = None,
    ) -> List[BandStatistics]:
        """
        Compute statistics from a numpy array.

        Args:
            data: Input array (bands, height, width) or (height, width)
            nodata: Value to treat as nodata

        Returns:
            List of BandStatistics, one per band
        """
        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        if data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        band_count, height, width = data.shape
        band_stats = []

        nodata_values = list(self.config.nodata_values)
        if nodata is not None:
            nodata_values.append(nodata)

        for band_idx in range(band_count):
            band_data = data[band_idx]
            dtype = str(band_data.dtype)

            # Create nodata mask
            mask = np.zeros(band_data.shape, dtype=bool)
            for nd in nodata_values:
                mask |= band_data == nd
            mask |= ~np.isfinite(band_data.astype(float))

            valid_data = band_data[~mask]
            valid_count = len(valid_data)
            nodata_count = mask.sum()
            total_count = band_data.size
            valid_percent = (valid_count / total_count * 100) if total_count > 0 else 0.0

            if valid_count > 0:
                min_val = float(np.min(valid_data))
                max_val = float(np.max(valid_data))
                mean_val = float(np.mean(valid_data))
                std_val = float(np.std(valid_data))
                median_val = float(np.median(valid_data))
                variance_val = float(np.var(valid_data))
                sum_val = float(np.sum(valid_data))

                percentiles = {}
                for p in self.config.percentiles:
                    percentiles[p] = float(np.percentile(valid_data, p))

                histogram = None
                if self.config.compute_histogram:
                    histogram = self._compute_histogram(valid_data, min_val, max_val)

                is_integer = np.issubdtype(band_data.dtype, np.integer)
                unique_count = None
                is_categorical = False

                if is_integer and valid_count <= 100000:
                    unique_vals = np.unique(valid_data)
                    unique_count = len(unique_vals)
                    value_range = max_val - min_val + 1
                    if value_range > 0:
                        is_categorical = unique_count / value_range < 0.1
            else:
                min_val = float("nan")
                max_val = float("nan")
                mean_val = float("nan")
                std_val = float("nan")
                median_val = float("nan")
                variance_val = float("nan")
                sum_val = 0.0
                percentiles = {p: float("nan") for p in self.config.percentiles}
                histogram = None
                is_integer = np.issubdtype(band_data.dtype, np.integer)
                unique_count = 0
                is_categorical = False

            band_stats.append(
                BandStatistics(
                    band_index=band_idx + 1,
                    min=min_val,
                    max=max_val,
                    mean=mean_val,
                    std=std_val,
                    median=median_val,
                    variance=variance_val,
                    sum=sum_val,
                    count=total_count,
                    valid_count=valid_count,
                    nodata_count=nodata_count,
                    valid_percent=valid_percent,
                    percentiles=percentiles,
                    histogram=histogram,
                    dtype=dtype,
                    unique_count=unique_count,
                    is_integer=is_integer,
                    is_categorical=is_categorical,
                )
            )

        return band_stats

    def _compute_histogram(
        self, data: np.ndarray, min_val: float, max_val: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute histogram for data.

        Args:
            data: Input data array (1D)
            min_val: Minimum value for binning
            max_val: Maximum value for binning

        Returns:
            Tuple of (counts, bin_edges)
        """
        if self.config.histogram_type == HistogramType.LOGARITHMIC:
            if min_val > 0:
                log_bins = np.logspace(
                    np.log10(min_val),
                    np.log10(max_val),
                    self.config.histogram_bins + 1,
                )
                return np.histogram(data, bins=log_bins)
            else:
                # Fall back to linear if min is <= 0
                return np.histogram(
                    data, bins=self.config.histogram_bins, range=(min_val, max_val)
                )

        elif self.config.histogram_type == HistogramType.PERCENTILE:
            # Use percentile-based bins for better distribution
            percentile_bins = np.percentile(
                data, np.linspace(0, 100, self.config.histogram_bins + 1)
            )
            # Remove duplicate bin edges
            percentile_bins = np.unique(percentile_bins)
            if len(percentile_bins) < 2:
                percentile_bins = np.array([min_val, max_val])
            return np.histogram(data, bins=percentile_bins)

        elif self.config.histogram_type == HistogramType.AUTOMATIC:
            # Use numpy's auto bin selection
            return np.histogram(data, bins="auto")

        else:
            # Linear (default)
            # Guard against constant data where min_val == max_val
            if min_val == max_val:
                # Return single bin for constant data
                return np.array([len(data)]), np.array([min_val, max_val + 1e-10])
            return np.histogram(
                data, bins=self.config.histogram_bins, range=(min_val, max_val)
            )

    def get_stretch_params(
        self,
        stats: Union[BandStatistics, RasterStatistics],
        method: str = "minmax",
        percentile: float = 2.0,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Get parameters for contrast stretching based on statistics.

        Args:
            stats: BandStatistics or RasterStatistics
            method: Stretch method (minmax, percentile, stddev)
            percentile: Percentile value for percentile method

        Returns:
            Dict mapping band index to (min, max) stretch values
        """
        if isinstance(stats, BandStatistics):
            band_list = [stats]
        else:
            band_list = stats.band_stats

        result = {}
        for band in band_list:
            if method == "minmax":
                stretch_min = band.min
                stretch_max = band.max
            elif method == "percentile":
                low_p = percentile
                high_p = 100 - percentile
                stretch_min = band.percentiles.get(low_p, band.min)
                stretch_max = band.percentiles.get(high_p, band.max)
            elif method == "stddev":
                n_std = 2.0
                stretch_min = band.mean - n_std * band.std
                stretch_max = band.mean + n_std * band.std
            else:
                stretch_min = band.min
                stretch_max = band.max

            result[band.band_index] = (stretch_min, stretch_max)

        return result


def compute_statistics(
    path: Union[str, Path],
    compute_histogram: bool = True,
    percentiles: Optional[List[float]] = None,
) -> RasterStatistics:
    """
    Convenience function to compute statistics for a raster file.

    Args:
        path: Path to raster file
        compute_histogram: Whether to compute histograms
        percentiles: List of percentiles to compute

    Returns:
        RasterStatistics with comprehensive band statistics
    """
    config = StatisticsConfig(
        compute_histogram=compute_histogram,
        percentiles=percentiles or [1, 5, 10, 25, 50, 75, 90, 95, 99],
    )
    calculator = StatisticsCalculator(config)
    return calculator.compute(path)
