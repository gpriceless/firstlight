"""
Color scales for detection overlays (VIS-1.3 Task 1.1).

This module provides color palettes for visualizing flood/wildfire severity
and confidence levels in detection overlays.

Design System Colors (from REPORT-2.0):
- Flood severity: Blue gradient (minimal → severe)
- Wildfire severity: Green → Yellow → Orange → Red → Dark red (unburned → severe)
- Confidence: Grayscale (very low → high)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import numpy as np


class SeverityLevel(Enum):
    """Severity levels for flood and wildfire detection."""

    # Flood severity levels (5 levels)
    FLOOD_MINIMAL = "minimal"
    FLOOD_LOW = "low"
    FLOOD_MODERATE = "moderate"
    FLOOD_HIGH = "high"
    FLOOD_SEVERE = "severe"

    # Wildfire severity levels (6 levels, based on dNBR burn severity)
    FIRE_UNBURNED = "unburned"
    FIRE_UNBURNED_REGROWTH = "unburned_regrowth"
    FIRE_LOW = "low"
    FIRE_MODERATE_LOW = "moderate_low"
    FIRE_MODERATE_HIGH = "moderate_high"
    FIRE_HIGH = "high"

    # Confidence levels (4 levels)
    CONFIDENCE_VERY_LOW = "very_low"
    CONFIDENCE_LOW = "low"
    CONFIDENCE_MEDIUM = "medium"
    CONFIDENCE_HIGH = "high"


@dataclass
class ColorPalette:
    """
    Color palette for detection overlays.

    Attributes:
        name: Descriptive name of the palette
        colors: List of RGBA colors (0-255 for each channel)
        labels: Human-readable labels for each color level
        value_ranges: Value ranges (min, max) for each level
    """

    name: str
    colors: List[Tuple[int, int, int, int]]  # RGBA tuples
    labels: List[str]
    value_ranges: List[Tuple[float, float]]  # (min, max) for each level

    def __post_init__(self):
        """Validate palette consistency."""
        n_colors = len(self.colors)
        if len(self.labels) != n_colors:
            raise ValueError(
                f"Number of labels ({len(self.labels)}) must match number of colors ({n_colors})"
            )
        if len(self.value_ranges) != n_colors:
            raise ValueError(
                f"Number of value ranges ({len(self.value_ranges)}) must match number of colors ({n_colors})"
            )


def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """
    Convert hex color string to RGBA tuple.

    Args:
        hex_color: Hex color string (e.g., "#E3F2FD" or "E3F2FD")
        alpha: Alpha channel value (0-255)

    Returns:
        RGBA tuple (r, g, b, a) with values 0-255

    Examples:
        >>> hex_to_rgba("#E3F2FD")
        (227, 242, 253, 255)
        >>> hex_to_rgba("E3F2FD", alpha=128)
        (227, 242, 253, 128)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip("#")

    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b, alpha)


def get_flood_palette() -> ColorPalette:
    """
    Get flood severity color palette.

    Returns blue gradient from light (minimal) to dark (severe).
    Based on Material Design blue palette.

    Returns:
        ColorPalette with 5 flood severity levels
    """
    # Blue gradient: light → dark
    hex_colors = [
        "#E3F2FD",  # Minimal - very light blue
        "#90CAF9",  # Low - light blue
        "#42A5F5",  # Moderate - medium blue
        "#1976D2",  # High - dark blue
        "#0D47A1",  # Severe - very dark blue
    ]

    colors = [hex_to_rgba(color, alpha=200) for color in hex_colors]

    labels = ["Minimal", "Low", "Moderate", "High", "Severe"]

    # Value ranges: normalized 0-1 for flood severity
    value_ranges = [
        (0.0, 0.2),  # Minimal
        (0.2, 0.4),  # Low
        (0.4, 0.6),  # Moderate
        (0.6, 0.8),  # High
        (0.8, 1.0),  # Severe
    ]

    return ColorPalette(
        name="Flood Severity",
        colors=colors,
        labels=labels,
        value_ranges=value_ranges,
    )


def get_fire_palette() -> ColorPalette:
    """
    Get wildfire burn severity color palette.

    Returns colors based on dNBR (delta Normalized Burn Ratio) burn severity scale:
    - Unburned: Green
    - Low severity: Yellow
    - Moderate severity: Orange
    - High severity: Red to dark red

    Based on USGS burn severity classification.

    Returns:
        ColorPalette with 6 fire severity levels
    """
    # Green → Yellow → Orange → Red → Dark red
    hex_colors = [
        "#4CAF50",  # Unburned - green
        "#8BC34A",  # Unburned/regrowth - light green
        "#FFEB3B",  # Low - yellow
        "#FF9800",  # Moderate-low - orange
        "#F44336",  # Moderate-high - red
        "#B71C1C",  # High - dark red
    ]

    colors = [hex_to_rgba(color, alpha=200) for color in hex_colors]

    labels = [
        "Unburned",
        "Unburned/Regrowth",
        "Low Severity",
        "Moderate-Low",
        "Moderate-High",
        "High Severity",
    ]

    # Value ranges based on dNBR thresholds (USGS classification)
    value_ranges = [
        (-0.5, -0.1),  # Unburned (negative dNBR)
        (-0.1, 0.1),  # Unburned/regrowth
        (0.1, 0.27),  # Low severity
        (0.27, 0.44),  # Moderate-low severity
        (0.44, 0.66),  # Moderate-high severity
        (0.66, 1.3),  # High severity
    ]

    return ColorPalette(
        name="Fire Burn Severity",
        colors=colors,
        labels=labels,
        value_ranges=value_ranges,
    )


def get_confidence_palette() -> ColorPalette:
    """
    Get confidence level color palette.

    Returns grayscale gradient from light (very low) to dark (high).

    Returns:
        ColorPalette with 4 confidence levels
    """
    # Grayscale: light → dark
    hex_colors = [
        "#E0E0E0",  # Very low - very light gray
        "#9E9E9E",  # Low - light gray
        "#616161",  # Medium - medium gray
        "#212121",  # High - dark gray
    ]

    colors = [hex_to_rgba(color, alpha=200) for color in hex_colors]

    labels = ["Very Low", "Low", "Medium", "High"]

    # Value ranges: normalized 0-1 for confidence
    value_ranges = [
        (0.0, 0.25),  # Very low
        (0.25, 0.5),  # Low
        (0.5, 0.75),  # Medium
        (0.75, 1.0),  # High
    ]

    return ColorPalette(
        name="Confidence Level",
        colors=colors,
        labels=labels,
        value_ranges=value_ranges,
    )


def apply_colormap(
    values: np.ndarray, palette: ColorPalette, nodata_value: float = np.nan
) -> np.ndarray:
    """
    Apply color palette to value array.

    Maps input values to RGBA colors based on palette value ranges.
    Values outside all ranges are mapped to transparent.

    Args:
        values: 2D array of values to colorize
        palette: ColorPalette defining the color mapping
        nodata_value: Value to treat as nodata (will be transparent)

    Returns:
        RGBA image array with shape (H, W, 4) and dtype uint8

    Examples:
        >>> values = np.array([[0.1, 0.5], [0.7, np.nan]])
        >>> palette = get_flood_palette()
        >>> rgba = apply_colormap(values, palette)
        >>> rgba.shape
        (2, 2, 4)
        >>> rgba.dtype
        dtype('uint8')
    """
    if values.ndim != 2:
        raise ValueError(f"Input values must be 2D array, got shape {values.shape}")

    height, width = values.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Handle nodata
    if np.isnan(nodata_value):
        valid_mask = ~np.isnan(values)
    else:
        valid_mask = values != nodata_value

    # Map values to colors based on value ranges
    for i, (vmin, vmax) in enumerate(palette.value_ranges):
        # Find pixels in this value range
        in_range = valid_mask & (values >= vmin) & (values < vmax)

        # Handle last range specially to include upper bound
        if i == len(palette.value_ranges) - 1:
            in_range = valid_mask & (values >= vmin) & (values <= vmax)

        # Apply color to pixels in range
        if np.any(in_range):
            r, g, b, a = palette.colors[i]
            rgba[in_range, 0] = r
            rgba[in_range, 1] = g
            rgba[in_range, 2] = b
            rgba[in_range, 3] = a

    return rgba


def get_palette_by_name(name: str) -> ColorPalette:
    """
    Get color palette by name.

    Args:
        name: Palette name ("flood", "fire", or "confidence")

    Returns:
        Requested ColorPalette

    Raises:
        ValueError: If palette name is not recognized
    """
    name = name.lower()
    if name == "flood":
        return get_flood_palette()
    elif name == "fire":
        return get_fire_palette()
    elif name == "confidence":
        return get_confidence_palette()
    else:
        raise ValueError(
            f"Unknown palette name: {name}. Choose from: flood, fire, confidence"
        )
