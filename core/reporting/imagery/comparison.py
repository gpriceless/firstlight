"""
Image Co-registration and Comparison Utilities for VIS-1.2.

Provides tools to align and compare before/after satellite imagery:
- Spatial co-registration: Align images with different extents/resolutions
- Resampling: Convert to common resolution grid
- Histogram normalization: Match appearance for visual comparison
- Difference calculation: Compute pixel-wise changes

Used by BeforeAfterGenerator to ensure temporal image pairs are comparable.

Example:
    # Align two images with different resolutions
    aligned = coregister_images(
        before_array, after_array,
        before_transform, after_transform,
        config=ComparisonConfig(target_resolution=10.0)
    )

    # Compute difference for change detection
    diff = calculate_difference(aligned.before, aligned.after)

    # Normalize histograms for consistent appearance
    before_norm, after_norm = normalize_histograms(aligned.before, aligned.after)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ComparisonConfig:
    """Configuration for image comparison operations.

    Attributes:
        target_resolution: Target pixel resolution in meters for resampling
        resampling_method: Resampling algorithm ('bilinear', 'nearest', 'cubic')
        crop_to_intersection: If True, crop to overlapping extent; if False, pad to union
    """

    target_resolution: float = 10.0
    resampling_method: str = "bilinear"
    crop_to_intersection: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.target_resolution <= 0:
            raise ValueError(
                f"target_resolution must be positive, got {self.target_resolution}"
            )

        valid_methods = {"bilinear", "nearest", "cubic", "lanczos"}
        if self.resampling_method not in valid_methods:
            raise ValueError(
                f"resampling_method must be one of {valid_methods}, "
                f"got '{self.resampling_method}'"
            )


@dataclass
class AlignedPair:
    """Result of co-registering two images.

    Attributes:
        before: Before image array aligned to common grid
        after: After image array aligned to common grid
        bounds: Spatial bounds (minx, miny, maxx, maxy) in CRS units
        resolution: Pixel resolution in meters
        crs: Coordinate reference system (EPSG code or WKT string)
    """

    before: np.ndarray
    after: np.ndarray
    bounds: Tuple[float, float, float, float]
    resolution: float
    crs: str

    def __post_init__(self):
        """Validate aligned pair."""
        if self.before.shape != self.after.shape:
            raise ValueError(
                f"Before and after arrays must have same shape after alignment. "
                f"Got before={self.before.shape}, after={self.after.shape}"
            )

        if self.before.ndim not in (2, 3):
            raise ValueError(
                f"Arrays must be 2D (single band) or 3D (multi-band), "
                f"got shape {self.before.shape}"
            )


def coregister_images(
    before: np.ndarray,
    after: np.ndarray,
    before_transform,
    after_transform,
    before_crs: str = "EPSG:4326",
    after_crs: str = "EPSG:4326",
    config: Optional[ComparisonConfig] = None,
) -> AlignedPair:
    """
    Co-register two images to a common spatial grid.

    Aligns images with potentially different:
    - Spatial extents (different coverage areas)
    - Resolutions (different pixel sizes)
    - Coordinate systems (different projections)

    The output images will have:
    - Same dimensions
    - Same resolution (config.target_resolution)
    - Same spatial extent (intersection or union based on config)
    - Same coordinate system (uses before_crs as target)

    Args:
        before: Before image array (H, W) or (H, W, C)
        after: After image array (H, W) or (H, W, C)
        before_transform: Affine transform for before image (rasterio.Affine)
        after_transform: Affine transform for after image (rasterio.Affine)
        before_crs: CRS of before image (default: EPSG:4326)
        after_crs: CRS of after image (default: EPSG:4326)
        config: Comparison configuration (default: ComparisonConfig())

    Returns:
        AlignedPair with both images on common grid

    Raises:
        ValueError: If images have incompatible dimensions (e.g., different channel counts)
        ImportError: If rasterio is not available for resampling

    Example:
        # Align Sentinel-2 images from different dates
        from rasterio.transform import from_bounds

        before_tf = from_bounds(-180, -90, 180, 90, before.shape[1], before.shape[0])
        after_tf = from_bounds(-180, -90, 180, 90, after.shape[1], after.shape[0])

        aligned = coregister_images(before, after, before_tf, after_tf)

        # Now aligned.before and aligned.after have same shape and extent
    """
    if config is None:
        config = ComparisonConfig()

    # Validate inputs
    if before.ndim != after.ndim:
        raise ValueError(
            f"Before and after must have same number of dimensions. "
            f"Got before.ndim={before.ndim}, after.ndim={after.ndim}"
        )

    if before.ndim == 3 and before.shape[2] != after.shape[2]:
        raise ValueError(
            f"Before and after must have same number of channels. "
            f"Got before.shape[2]={before.shape[2]}, after.shape[2]={after.shape[2]}"
        )

    try:
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        from rasterio.transform import Affine
    except ImportError as e:
        raise ImportError(
            "rasterio is required for image co-registration. "
            "Install with: pip install rasterio"
        ) from e

    # Convert resampling method string to rasterio enum
    resampling_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "lanczos": Resampling.lanczos,
    }
    resampling = resampling_map[config.resampling_method]

    # Calculate bounds for both images in their respective CRS
    def get_bounds(transform, shape):
        """Get (minx, miny, maxx, maxy) from transform and shape."""
        height, width = shape[:2]
        minx = transform.c
        maxy = transform.f
        maxx = minx + width * transform.a
        miny = maxy + height * transform.e  # e is negative
        return (minx, miny, maxx, maxy)

    before_bounds = get_bounds(before_transform, before.shape)
    after_bounds = get_bounds(after_transform, after.shape)

    # If CRS differ, reproject after bounds to before CRS
    if before_crs != after_crs:
        from rasterio.warp import transform_bounds
        after_bounds_reprojected = transform_bounds(
            after_crs, before_crs, *after_bounds
        )
    else:
        after_bounds_reprojected = after_bounds

    # Calculate target bounds (intersection or union)
    if config.crop_to_intersection:
        # Intersection: crop to common area
        target_bounds = (
            max(before_bounds[0], after_bounds_reprojected[0]),  # minx
            max(before_bounds[1], after_bounds_reprojected[1]),  # miny
            min(before_bounds[2], after_bounds_reprojected[2]),  # maxx
            min(before_bounds[3], after_bounds_reprojected[3]),  # maxy
        )

        # Check for valid intersection
        if target_bounds[0] >= target_bounds[2] or target_bounds[1] >= target_bounds[3]:
            raise ValueError(
                "Images do not overlap. Cannot crop to intersection. "
                f"Before bounds: {before_bounds}, After bounds: {after_bounds_reprojected}"
            )
    else:
        # Union: pad to combined area
        target_bounds = (
            min(before_bounds[0], after_bounds_reprojected[0]),  # minx
            min(before_bounds[1], after_bounds_reprojected[1]),  # miny
            max(before_bounds[2], after_bounds_reprojected[2]),  # maxx
            max(before_bounds[3], after_bounds_reprojected[3]),  # maxy
        )

    # Calculate target dimensions based on target resolution
    target_width = int(np.ceil((target_bounds[2] - target_bounds[0]) / config.target_resolution))
    target_height = int(np.ceil((target_bounds[3] - target_bounds[1]) / config.target_resolution))

    # Create target transform
    target_transform = Affine.translation(target_bounds[0], target_bounds[3]) * Affine.scale(
        config.target_resolution, -config.target_resolution
    )

    # Prepare output arrays
    is_multiband = before.ndim == 3
    num_bands = before.shape[2] if is_multiband else 1

    if is_multiband:
        before_aligned = np.zeros((target_height, target_width, num_bands), dtype=before.dtype)
        after_aligned = np.zeros((target_height, target_width, num_bands), dtype=after.dtype)
    else:
        before_aligned = np.zeros((target_height, target_width), dtype=before.dtype)
        after_aligned = np.zeros((target_height, target_width), dtype=after.dtype)

    # Reproject each band
    for band_idx in range(num_bands):
        if is_multiband:
            before_band = before[:, :, band_idx]
            after_band = after[:, :, band_idx]
        else:
            before_band = before
            after_band = after

        # Reproject before image
        reproject(
            source=before_band,
            destination=before_aligned[:, :, band_idx] if is_multiband else before_aligned,
            src_transform=before_transform,
            src_crs=before_crs,
            dst_transform=target_transform,
            dst_crs=before_crs,  # Keep in before CRS
            resampling=resampling,
        )

        # Reproject after image
        reproject(
            source=after_band,
            destination=after_aligned[:, :, band_idx] if is_multiband else after_aligned,
            src_transform=after_transform,
            src_crs=after_crs,
            dst_transform=target_transform,
            dst_crs=before_crs,  # Reproject to before CRS
            resampling=resampling,
        )

    return AlignedPair(
        before=before_aligned,
        after=after_aligned,
        bounds=target_bounds,
        resolution=config.target_resolution,
        crs=before_crs,
    )


def calculate_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Calculate pixel-wise difference between before and after images.

    Computes simple arithmetic difference (after - before) for change detection.
    Positive values indicate increase, negative values indicate decrease.

    Args:
        before: Before image (H, W) or (H, W, C)
        after: After image (H, W) or (H, W, C)

    Returns:
        Difference array with same shape as inputs

    Raises:
        ValueError: If arrays have different shapes

    Example:
        # Detect changes
        diff = calculate_difference(before, after)

        # Find pixels with significant change
        changed_pixels = np.abs(diff) > threshold

        # Separate increases and decreases
        increases = diff > threshold
        decreases = diff < -threshold
    """
    if before.shape != after.shape:
        raise ValueError(
            f"Before and after must have same shape. "
            f"Got before={before.shape}, after={after.shape}"
        )

    # Simple difference
    return after.astype(np.float64) - before.astype(np.float64)


def normalize_histograms(
    before: np.ndarray, after: np.ndarray, method: str = "match"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize histograms to make images visually comparable.

    Adjusts the histogram of the 'after' image to match the 'before' image,
    ensuring consistent brightness and contrast for visual comparison.

    Methods:
        - 'match': Match after histogram to before (preserves before appearance)
        - 'standardize': Standardize both to zero mean and unit variance

    Args:
        before: Before image (H, W) or (H, W, C)
        after: After image (H, W) or (H, W, C)
        method: Normalization method ('match' or 'standardize')

    Returns:
        Tuple of (before_normalized, after_normalized)

    Raises:
        ValueError: If arrays have incompatible shapes or invalid method

    Example:
        # Match after image to before for consistent appearance
        before_norm, after_norm = normalize_histograms(before, after)

        # Now both images have similar brightness/contrast
        # Useful for visual side-by-side comparison
    """
    if before.shape != after.shape:
        raise ValueError(
            f"Before and after must have same shape. "
            f"Got before={before.shape}, after={after.shape}"
        )

    if method == "match":
        # Histogram matching: adjust after to match before distribution
        return _histogram_match(before, after)
    elif method == "standardize":
        # Standardization: zero mean, unit variance for both
        return _standardize_arrays(before, after)
    else:
        raise ValueError(
            f"Invalid normalization method: '{method}'. "
            f"Must be 'match' or 'standardize'"
        )


def _histogram_match(
    reference: np.ndarray, source: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match histogram of source image to reference image.

    Applies histogram matching per-band for multi-band images.
    Reference image is returned unchanged.

    Args:
        reference: Reference image (target histogram)
        source: Source image to adjust

    Returns:
        Tuple of (reference, matched_source)
    """
    # Work with copies to avoid modifying inputs
    reference_out = reference.copy()
    source_out = source.copy()

    is_multiband = reference.ndim == 3
    num_bands = reference.shape[2] if is_multiband else 1

    for band_idx in range(num_bands):
        if is_multiband:
            ref_band = reference[:, :, band_idx].ravel()
            src_band = source[:, :, band_idx].ravel()
        else:
            ref_band = reference.ravel()
            src_band = source.ravel()

        # Filter out zeros/nodata (assuming 0 is nodata)
        ref_valid = ref_band[ref_band > 0]
        src_valid = src_band[src_band > 0]

        if len(ref_valid) == 0 or len(src_valid) == 0:
            # Skip if no valid data
            continue

        # Calculate CDFs
        ref_values, ref_counts = np.unique(ref_valid, return_counts=True)
        src_values, src_counts = np.unique(src_valid, return_counts=True)

        ref_cdf = np.cumsum(ref_counts).astype(np.float64)
        ref_cdf /= ref_cdf[-1]

        src_cdf = np.cumsum(src_counts).astype(np.float64)
        src_cdf /= src_cdf[-1]

        # Build lookup table
        matched = np.interp(src_cdf, ref_cdf, ref_values)

        # Apply lookup to source, preserving zeros (nodata)
        lookup = np.interp(src_band, src_values, matched)
        # Set zeros back to zero (preserve nodata)
        lookup[src_band == 0] = 0

        if is_multiband:
            source_out[:, :, band_idx] = lookup.reshape(source.shape[:2])
        else:
            source_out = lookup.reshape(source.shape)

    return reference_out, source_out


def _standardize_arrays(
    arr1: np.ndarray, arr2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize both arrays to zero mean and unit variance.

    Applies per-band standardization for multi-band images.
    Preserves zeros (nodata) as zeros in the output.

    Args:
        arr1: First array
        arr2: Second array

    Returns:
        Tuple of (standardized_arr1, standardized_arr2)
    """
    def standardize_band(band: np.ndarray) -> np.ndarray:
        """Standardize single band, preserving zeros."""
        valid = band[band > 0]  # Exclude zeros/nodata
        if len(valid) == 0:
            return band.astype(np.float64)

        mean = np.mean(valid)
        std = np.std(valid)

        if std == 0:
            return band.astype(np.float64)

        # Initialize output with zeros (preserving nodata)
        result = np.zeros_like(band, dtype=np.float64)
        mask = band > 0
        result[mask] = (band[mask] - mean) / std

        return result

    is_multiband = arr1.ndim == 3
    num_bands = arr1.shape[2] if is_multiband else 1

    arr1_out = np.zeros_like(arr1, dtype=np.float64)
    arr2_out = np.zeros_like(arr2, dtype=np.float64)

    for band_idx in range(num_bands):
        if is_multiband:
            arr1_out[:, :, band_idx] = standardize_band(arr1[:, :, band_idx])
            arr2_out[:, :, band_idx] = standardize_band(arr2[:, :, band_idx])
        else:
            arr1_out = standardize_band(arr1)
            arr2_out = standardize_band(arr2)

    return arr1_out, arr2_out
