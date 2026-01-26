"""
Utility functions for imagery processing in VIS-1.1.

Provides core image manipulation functions including:
- Normalization to uint8 for PNG export
- Band stacking for RGB composites
- NoData value handling and masking
- Coverage detection for partial datasets
- Valid data bounds extraction

These utilities support graceful degradation when data is missing or incomplete.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_to_uint8(
    array: np.ndarray, min_val: float = 0.0, max_val: float = 1.0
) -> np.ndarray:
    """
    Convert float array to uint8 range (0-255) for image export.

    Performs linear scaling from [min_val, max_val] to [0, 255].
    Values outside the range are clipped.

    Args:
        array: Input array (typically float32/float64)
        min_val: Minimum value in input range (maps to 0)
        max_val: Maximum value in input range (maps to 255)

    Returns:
        uint8 array with values in [0, 255]

    Example:
        # Normalize reflectance data (0-1) to image
        rgb_uint8 = normalize_to_uint8(rgb_float, 0.0, 1.0)

        # Normalize with custom stretch
        stretched = normalize_to_uint8(data, percentile_2, percentile_98)
    """
    if array.size == 0:
        return array.astype(np.uint8)

    # Handle case where min_val == max_val (constant array)
    if np.isclose(min_val, max_val):
        # Return middle gray value
        return np.full(array.shape, 127, dtype=np.uint8)

    # Scale to [0, 1] then to [0, 255]
    scaled = (array - min_val) / (max_val - min_val)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def stack_bands_to_rgb(
    red: np.ndarray, green: np.ndarray, blue: np.ndarray
) -> np.ndarray:
    """
    Combine three bands into a single RGB array.

    Args:
        red: Red band (H, W)
        green: Green band (H, W)
        blue: Blue band (H, W)

    Returns:
        RGB array with shape (H, W, 3)

    Raises:
        ValueError: If bands have different shapes or aren't 2D

    Example:
        # Stack Sentinel-2 bands to true color
        rgb = stack_bands_to_rgb(b04, b03, b02)
    """
    # Validate inputs
    if red.ndim != 2 or green.ndim != 2 or blue.ndim != 2:
        raise ValueError(
            f"All bands must be 2D. Got shapes: "
            f"red={red.shape}, green={green.shape}, blue={blue.shape}"
        )

    if not (red.shape == green.shape == blue.shape):
        raise ValueError(
            f"All bands must have same shape. Got: "
            f"red={red.shape}, green={green.shape}, blue={blue.shape}"
        )

    # Stack along third dimension
    return np.stack([red, green, blue], axis=-1)


def handle_nodata(
    array: np.ndarray, nodata_value: Optional[float] = None, fill: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Handle nodata values in array by replacing them and creating a validity mask.

    Detects nodata values from:
    1. Explicit nodata_value parameter
    2. NaN values in array
    3. Masked array mask (if input is np.ma.MaskedArray)

    Args:
        array: Input array (may be masked array)
        nodata_value: Explicit nodata value to detect (optional)
        fill: Value to replace nodata with

    Returns:
        Tuple of (filled_array, valid_mask) where:
        - filled_array: Array with nodata replaced by fill value
        - valid_mask: Boolean array (True = valid data, False = nodata)

    Example:
        # Handle explicit nodata value
        filled, mask = handle_nodata(data, nodata_value=-9999, fill=0)

        # Handle NaN values
        filled, mask = handle_nodata(data_with_nan, fill=0)

        # Use mask to compute statistics on valid data only
        valid_mean = filled[mask].mean()
    """
    # Initialize validity mask as all True
    valid_mask = np.ones(array.shape, dtype=bool)

    # Handle masked arrays
    if isinstance(array, np.ma.MaskedArray):
        valid_mask &= ~array.mask
        array = array.filled(fill)

    # Detect explicit nodata value
    if nodata_value is not None:
        if np.isnan(nodata_value):
            valid_mask &= ~np.isnan(array)
        else:
            valid_mask &= array != nodata_value

    # Detect NaN values (always treat as nodata)
    if np.issubdtype(array.dtype, np.floating):
        valid_mask &= ~np.isnan(array)

    # Create filled array
    filled = array.copy()
    filled[~valid_mask] = fill

    return filled, valid_mask


def detect_partial_coverage(
    array: np.ndarray, nodata_value: Optional[float] = None
) -> float:
    """
    Calculate percentage of valid (non-nodata) pixels in array.

    Args:
        array: Input array
        nodata_value: Explicit nodata value (optional)

    Returns:
        Coverage percentage (0.0 to 100.0)

    Example:
        coverage = detect_partial_coverage(band, nodata_value=-9999)
        if coverage < 50.0:
            logger.warning(f"Low coverage: {coverage:.1f}%")
    """
    if array.size == 0:
        return 0.0

    # Use handle_nodata to detect all nodata types
    _, valid_mask = handle_nodata(array, nodata_value=nodata_value)

    # Calculate percentage
    valid_count = np.sum(valid_mask)
    total_count = array.size
    return 100.0 * valid_count / total_count


def get_valid_data_bounds(
    array: np.ndarray, nodata_value: Optional[float] = None
) -> Tuple[int, int, int, int]:
    """
    Find bounding box of valid (non-nodata) pixels.

    Returns the smallest rectangle that contains all valid data,
    useful for cropping to actual data extent.

    Args:
        array: Input array (2D)
        nodata_value: Explicit nodata value (optional)

    Returns:
        Tuple of (row_start, row_end, col_start, col_end) where:
        - row_start, row_end: Row indices (inclusive, exclusive)
        - col_start, col_end: Column indices (inclusive, exclusive)
        - Returns (0, 0, 0, 0) if no valid data found

    Example:
        # Find extent of valid data
        r0, r1, c0, c1 = get_valid_data_bounds(band, nodata_value=-9999)
        if r0 < r1 and c0 < c1:
            cropped = band[r0:r1, c0:c1]
    """
    if array.ndim != 2:
        raise ValueError(f"Array must be 2D, got shape {array.shape}")

    if array.size == 0:
        return (0, 0, 0, 0)

    # Get validity mask
    _, valid_mask = handle_nodata(array, nodata_value=nodata_value)

    # Find any valid pixels
    valid_pixels = np.where(valid_mask)
    if len(valid_pixels[0]) == 0:
        # No valid data
        return (0, 0, 0, 0)

    # Find bounding box
    rows = valid_pixels[0]
    cols = valid_pixels[1]

    row_min = int(np.min(rows))
    row_max = int(np.max(rows)) + 1  # Make exclusive
    col_min = int(np.min(cols))
    col_max = int(np.max(cols)) + 1  # Make exclusive

    return (row_min, row_max, col_min, col_max)
