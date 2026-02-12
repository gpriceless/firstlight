"""
Histogram stretching algorithms for satellite imagery enhancement.

This module provides histogram stretching operations to improve visual contrast
in satellite imagery. Implements multiple stretch methods optimized for
geospatial raster data.

Part of VIS-1.1: Satellite Imagery Renderer

Performance target: <1 second for 10000x10000 arrays using vectorized numpy operations.
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import numpy.ma as ma


class HistogramStretch(Enum):
    """Available histogram stretch methods for imagery enhancement."""

    LINEAR_2PCT = "linear_2pct"
    """Linear stretch clipping 2% at each end of histogram (default)."""

    MIN_MAX = "min_max"
    """Linear stretch using absolute min/max values."""

    STDDEV = "stddev"
    """Stretch using mean +/- 2 standard deviations."""

    ADAPTIVE = "adaptive"
    """Adaptive histogram equalization (future implementation)."""


def calculate_stretch_bounds(
    array: np.ndarray,
    percentile: float = 2.0,
    nodata: Optional[float] = None,
    sample_size: int = 1_000_000
) -> Tuple[float, float]:
    """
    Calculate stretch bounds based on percentile clipping.

    Computes min/max values for histogram stretching by clipping outliers
    at the specified percentile. This removes extreme values that would
    otherwise compress the dynamic range of typical pixel values.

    For performance on large arrays (>1M pixels), uses statistical sampling
    to estimate percentiles rather than sorting all pixels.

    Args:
        array: Input array to analyze. Can be regular ndarray or masked array.
        percentile: Percentage of values to clip at each end (0-50).
                   Default 2.0 means clip bottom 2% and top 2%.
        nodata: Optional nodata value to exclude from calculation.
               For masked arrays, the mask is used instead.
        sample_size: Maximum number of pixels to use for percentile calculation.
                    Arrays larger than this are randomly sampled. Default 1M.

    Returns:
        Tuple of (min_val, max_val) for stretching.

    Raises:
        ValueError: If percentile is outside valid range [0, 50].
        ValueError: If array contains only nodata/masked values.

    Examples:
        >>> import numpy as np
        >>> data = np.random.normal(100, 20, (1000, 1000))
        >>> min_val, max_val = calculate_stretch_bounds(data, percentile=2.0)
        >>> # Clips bottom 2% and top 2% of histogram

        >>> # With nodata values
        >>> data[data < 50] = -9999
        >>> min_val, max_val = calculate_stretch_bounds(data, nodata=-9999)

        >>> # With masked array
        >>> masked_data = np.ma.masked_less(data, 50)
        >>> min_val, max_val = calculate_stretch_bounds(masked_data)
    """
    if not 0 <= percentile <= 50:
        raise ValueError(f"Percentile must be between 0 and 50, got {percentile}")

    # Handle masked arrays
    if ma.is_masked(array):
        valid_data = array.compressed()
    else:
        # Filter out nodata values if specified
        if nodata is not None:
            valid_data = array[array != nodata]
        else:
            # Filter out NaN and infinite values
            valid_data = array[np.isfinite(array)]

    if valid_data.size == 0:
        raise ValueError("Array contains only nodata/masked values")

    # For large arrays, use sampling for performance
    # Percentile calculation requires sorting which is O(n log n)
    # For 10000x10000 = 100M pixels, this is too slow
    if valid_data.size > sample_size:
        # Random sampling for statistical estimate
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        sample_indices = rng.choice(valid_data.size, size=sample_size, replace=False)
        sample_data = valid_data[sample_indices]
    else:
        sample_data = valid_data

    # Calculate percentile bounds using numpy's quantile (faster than percentile)
    # Single call with both quantiles is faster than two separate calls
    quantiles = np.quantile(sample_data, [percentile / 100, 1 - percentile / 100], method='linear')
    min_val, max_val = quantiles[0], quantiles[1]

    # Ensure min < max (handles edge case of uniform data)
    if min_val >= max_val:
        # Fall back to actual min/max if percentile bounds are identical
        min_val = float(valid_data.min())
        max_val = float(valid_data.max())

        # If still identical, add small epsilon to avoid division by zero
        if min_val >= max_val:
            max_val = min_val + 1e-10

    return float(min_val), float(max_val)


def apply_stretch(
    array: np.ndarray,
    min_val: float,
    max_val: float,
    nodata: Optional[float] = None
) -> np.ndarray:
    """
    Apply linear stretch to map values to 0-1 range.

    Performs vectorized linear scaling: (value - min) / (max - min).
    Values below min are clipped to 0, values above max are clipped to 1.

    Preserves masked arrays and nodata values.

    Args:
        array: Input array to stretch. Can be regular ndarray or masked array.
        min_val: Minimum value to map to 0.
        max_val: Maximum value to map to 1.
        nodata: Optional nodata value to preserve in output.
               For masked arrays, the mask is preserved.

    Returns:
        Stretched array with values in [0, 1] range.
        Masked arrays return masked arrays, regular arrays return regular arrays.

    Raises:
        ValueError: If min_val >= max_val.

    Examples:
        >>> import numpy as np
        >>> data = np.array([[0, 50, 100], [150, 200, 250]])
        >>> stretched = apply_stretch(data, min_val=50, max_val=200)
        >>> # 50 maps to 0.0, 200 maps to 1.0
        >>> # Values outside range are clipped

        >>> # Preserves nodata
        >>> data[0, 0] = -9999
        >>> stretched = apply_stretch(data, min_val=50, max_val=200, nodata=-9999)
        >>> stretched[0, 0]  # Still -9999
    """
    if min_val >= max_val:
        raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")

    # Handle masked arrays specially to preserve mask
    is_masked = ma.is_masked(array)

    if is_masked:
        # Work with masked array
        result = array.copy()
        # Apply stretch only to valid (unmasked) data
        valid_mask = ~result.mask
        result.data[valid_mask] = np.clip(
            (result.data[valid_mask] - min_val) / (max_val - min_val),
            0.0,
            1.0
        )
    else:
        # Regular array
        if nodata is not None:
            # Preserve nodata values
            nodata_mask = (array == nodata)
            result = np.clip((array - min_val) / (max_val - min_val), 0.0, 1.0)
            result[nodata_mask] = nodata
        else:
            # Simple vectorized stretch
            result = np.clip((array - min_val) / (max_val - min_val), 0.0, 1.0)

    return result


def calculate_stddev_bounds(
    array: np.ndarray,
    n_std: float = 2.0,
    nodata: Optional[float] = None
) -> Tuple[float, float]:
    """
    Calculate stretch bounds using standard deviation method.

    Computes min/max as mean +/- n_std * standard_deviation.
    Useful for normally distributed data.

    Args:
        array: Input array to analyze.
        n_std: Number of standard deviations from mean. Default 2.0.
        nodata: Optional nodata value to exclude from calculation.

    Returns:
        Tuple of (min_val, max_val) for stretching.

    Raises:
        ValueError: If array contains only nodata/masked values.

    Examples:
        >>> import numpy as np
        >>> data = np.random.normal(100, 20, (1000, 1000))
        >>> min_val, max_val = calculate_stddev_bounds(data, n_std=2.0)
        >>> # Approximately mean +/- 2*std (100 +/- 40)
    """
    # Handle masked arrays
    if ma.is_masked(array):
        valid_data = array.compressed()
    else:
        # Filter out nodata values if specified
        if nodata is not None:
            valid_data = array[array != nodata]
        else:
            # Filter out NaN and infinite values
            valid_data = array[np.isfinite(array)]

    if valid_data.size == 0:
        raise ValueError("Array contains only nodata/masked values")

    mean = np.mean(valid_data)
    std = np.std(valid_data)

    min_val = float(mean - n_std * std)
    max_val = float(mean + n_std * std)

    # Ensure min < max
    if min_val >= max_val:
        max_val = min_val + 1e-10

    return min_val, max_val


def stretch_band(
    array: np.ndarray,
    method: HistogramStretch = HistogramStretch.LINEAR_2PCT,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    nodata: Optional[float] = None,
    percentile: float = 2.0
) -> np.ndarray:
    """
    Apply histogram stretch to a single band of imagery.

    Main entry point for histogram stretching. Supports multiple stretch methods
    and manual min/max override for full control.

    Args:
        array: Input band array. Can be regular ndarray or masked array.
        method: Stretch method to use. Default LINEAR_2PCT.
        min_val: Optional manual minimum value. Overrides automatic calculation.
        max_val: Optional manual maximum value. Overrides automatic calculation.
        nodata: Optional nodata value to preserve and exclude from calculation.
        percentile: Percentile to use for LINEAR_2PCT method. Default 2.0.

    Returns:
        Stretched array with values in [0, 1] range.

    Raises:
        ValueError: If manual min_val >= max_val.
        ValueError: If array contains only nodata/masked values.
        NotImplementedError: If ADAPTIVE method is requested (not yet implemented).

    Examples:
        >>> import numpy as np
        >>> band = np.random.normal(100, 20, (1000, 1000))

        >>> # Automatic 2% stretch (default)
        >>> stretched = stretch_band(band)

        >>> # Manual min/max override
        >>> stretched = stretch_band(band, min_val=50, max_val=150)

        >>> # Standard deviation method
        >>> from core.reporting.imagery.histogram import HistogramStretch
        >>> stretched = stretch_band(band, method=HistogramStretch.STDDEV)

        >>> # With nodata handling
        >>> band[band < 50] = -9999
        >>> stretched = stretch_band(band, nodata=-9999)
    """
    # Manual min/max override takes precedence
    if min_val is not None and max_val is not None:
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        return apply_stretch(array, min_val, max_val, nodata=nodata)

    # Calculate bounds based on method
    if method == HistogramStretch.LINEAR_2PCT:
        bounds = calculate_stretch_bounds(array, percentile=percentile, nodata=nodata)
    elif method == HistogramStretch.MIN_MAX:
        # Use actual min/max (percentile=0 would clip nothing)
        if ma.is_masked(array):
            valid_data = array.compressed()
        else:
            if nodata is not None:
                valid_data = array[array != nodata]
            else:
                valid_data = array[np.isfinite(array)]

        if valid_data.size == 0:
            raise ValueError("Array contains only nodata/masked values")

        bounds = (float(valid_data.min()), float(valid_data.max()))

        # Ensure min < max
        if bounds[0] >= bounds[1]:
            bounds = (bounds[0], bounds[0] + 1e-10)
    elif method == HistogramStretch.STDDEV:
        bounds = calculate_stddev_bounds(array, n_std=2.0, nodata=nodata)
    elif method == HistogramStretch.ADAPTIVE:
        raise NotImplementedError(
            "Adaptive histogram equalization not yet implemented. "
            "Use LINEAR_2PCT, MIN_MAX, or STDDEV."
        )
    else:
        raise ValueError(f"Unknown stretch method: {method}")

    # Apply manual overrides if provided
    if min_val is not None:
        bounds = (min_val, bounds[1])
    if max_val is not None:
        bounds = (bounds[0], max_val)

    return apply_stretch(array, bounds[0], bounds[1], nodata=nodata)
