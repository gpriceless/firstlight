"""
Area Metrics Module (VIS-1.4 Task 3.1).

Factual area measurements for detected regions. NO arbitrary comparisons,
NO "football fields" - this is emergency response software requiring
precise, factual measurements only.

Provides:
- AreaMetrics: Dataclass containing factual measurements
- calculate_area_metrics: Calculate metrics from detection mask
- format_area: Format metrics as clean, factual text

All measurements are factual: hectares, acres, km², mi², and percent of AOI.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Conversion constants
SQMETER_TO_HECTARE = 1.0 / 10_000.0
SQMETER_TO_ACRE = 1.0 / 4_046.86
SQMETER_TO_SQKM = 1.0 / 1_000_000.0
SQMETER_TO_SQMI = 1.0 / 2_589_988.0


@dataclass
class AreaMetrics:
    """Factual area measurements for detected regions.

    All measurements are precise, factual values with no arbitrary comparisons.
    Suitable for emergency response and professional analysis.

    Attributes:
        hectares: Area in hectares
        acres: Area in acres
        square_kilometers: Area in square kilometers
        square_miles: Area in square miles
        percent_of_aoi: Percentage of analysis area (0-100), None if not available
    """

    hectares: float
    acres: float
    square_kilometers: float
    square_miles: float
    percent_of_aoi: Optional[float] = None


def calculate_area_metrics(
    detection_mask: np.ndarray,
    pixel_size_meters: float,
    aoi_area_sqm: Optional[float] = None,
) -> AreaMetrics:
    """Calculate area metrics from detection mask.

    Computes factual area measurements from a boolean detection mask,
    converting pixel counts to standard units.

    Args:
        detection_mask: Boolean mask of detected pixels (True = detected)
        pixel_size_meters: Size of each pixel in meters (assumes square pixels)
        aoi_area_sqm: Total area of interest in square meters (for % calculation).
                      If None, percent_of_aoi will be None.

    Returns:
        AreaMetrics with factual measurements in multiple units

    Raises:
        ValueError: If pixel_size_meters is not positive
        ValueError: If aoi_area_sqm is provided but not positive

    Example:
        >>> mask = np.ones((1000, 1000), dtype=bool)
        >>> metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)
        >>> metrics.hectares
        100.0
    """
    if pixel_size_meters <= 0:
        raise ValueError(f"pixel_size_meters must be positive, got {pixel_size_meters}")

    if aoi_area_sqm is not None and aoi_area_sqm <= 0:
        raise ValueError(f"aoi_area_sqm must be positive if provided, got {aoi_area_sqm}")

    # Count detected pixels
    detected_pixels = np.sum(detection_mask)

    # Calculate area in square meters
    area_per_pixel_sqm = pixel_size_meters * pixel_size_meters
    total_area_sqm = detected_pixels * area_per_pixel_sqm

    # Convert to all standard units
    hectares = total_area_sqm * SQMETER_TO_HECTARE
    acres = total_area_sqm * SQMETER_TO_ACRE
    square_kilometers = total_area_sqm * SQMETER_TO_SQKM
    square_miles = total_area_sqm * SQMETER_TO_SQMI

    # Calculate percentage if AOI area provided
    percent_of_aoi = None
    if aoi_area_sqm is not None and aoi_area_sqm > 0:
        percent_of_aoi = (total_area_sqm / aoi_area_sqm) * 100.0

    return AreaMetrics(
        hectares=hectares,
        acres=acres,
        square_kilometers=square_kilometers,
        square_miles=square_miles,
        percent_of_aoi=percent_of_aoi,
    )


def format_area(metrics: AreaMetrics, include_percentage: bool = True) -> str:
    """Format area metrics as clean, factual text.

    Produces human-readable area descriptions using appropriate units based
    on scale. Always provides both metric and imperial measurements.

    Unit selection:
        - < 100 hectares: use hectares/acres
        - >= 100 hectares: use km²/mi²

    Args:
        metrics: AreaMetrics to format
        include_percentage: Include percentage of AOI if available

    Returns:
        Formatted string with factual measurements

    Examples:
        >>> metrics = AreaMetrics(hectares=24.5, acres=60.5, ...)
        >>> format_area(metrics, include_percentage=False)
        "24.5 hectares (60.5 acres)"

        >>> metrics = AreaMetrics(hectares=245.0, acres=605.0,
        ...                       square_kilometers=2.45, square_miles=0.95,
        ...                       percent_of_aoi=18.0)
        >>> format_area(metrics)
        "2.5 km² (0.9 mi²), 18% of analysis area"
    """
    # Choose appropriate units based on scale
    if metrics.hectares < 100:
        # Small areas: use hectares/acres
        metric_str = _format_number(metrics.hectares, decimals=1)
        imperial_str = _format_number(metrics.acres, decimals=1)
        result = f"{metric_str} hectares ({imperial_str} acres)"
    else:
        # Large areas: use km²/mi²
        metric_str = _format_number(metrics.square_kilometers, decimals=1)
        imperial_str = _format_number(metrics.square_miles, decimals=1)
        result = f"{metric_str} km² ({imperial_str} mi²)"

    # Add percentage if available and requested
    if include_percentage and metrics.percent_of_aoi is not None:
        percent = round(metrics.percent_of_aoi)
        result += f", {percent}% of analysis area"

    return result


def _format_number(value: float, decimals: int = 1) -> str:
    """Format number with thousands separator and specified decimals.

    Args:
        value: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string with comma separators

    Examples:
        >>> _format_number(1234.567, decimals=1)
        "1,234.6"
        >>> _format_number(42.3, decimals=1)
        "42.3"
    """
    # Round to specified decimals
    rounded = round(value, decimals)

    # Format with thousands separator
    if decimals == 0:
        return f"{rounded:,.0f}"
    else:
        return f"{rounded:,.{decimals}f}"
