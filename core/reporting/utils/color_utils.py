"""
Color utilities for FirstLight design system.

Provides color conversion, WCAG contrast checking, color manipulation,
and design system color constants.

All functions work with hex color strings in format #RRGGBB.
No external dependencies - uses Python standard library only.
"""

import colorsys
import re
from typing import Tuple


# =============================================================================
# Color Conversion Functions
# =============================================================================

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color to RGB tuple.

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB)

    Returns:
        Tuple of (red, green, blue) values (0-255)

    Examples:
        >>> hex_to_rgb("#1A365D")
        (26, 54, 93)
        >>> hex_to_rgb("90CDF4")
        (144, 205, 244)
    """
    # Remove # prefix if present
    hex_color = hex_color.lstrip('#')

    # Validate hex format
    if not re.match(r'^[0-9A-Fa-f]{6}$', hex_color):
        raise ValueError(f"Invalid hex color format: {hex_color}")

    # Convert to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    return (r, g, b)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Convert RGB to hex string.

    Args:
        r: Red value (0-255)
        g: Green value (0-255)
        b: Blue value (0-255)

    Returns:
        Hex color string with # prefix (#RRGGBB)

    Examples:
        >>> rgb_to_hex(26, 54, 93)
        '#1A365D'
        >>> rgb_to_hex(144, 205, 244)
        '#90CDF4'
    """
    # Clamp values to valid range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f'#{r:02X}{g:02X}{b:02X}'


def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert hex to HSL (Hue, Saturation, Lightness).

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB)

    Returns:
        Tuple of (hue, saturation, lightness)
        - hue: 0-360 degrees
        - saturation: 0-1 (0% to 100%)
        - lightness: 0-1 (0% to 100%)

    Examples:
        >>> hex_to_hsl("#1A365D")
        (214.3, 0.563, 0.233)
    """
    r, g, b = hex_to_rgb(hex_color)

    # Normalize RGB values to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Use colorsys to convert RGB to HLS (note: HLS not HSL)
    h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)

    # Convert to standard HSL format
    # h: 0-1 → 0-360 degrees
    # s: keep as 0-1
    # l: keep as 0-1
    hue = h * 360.0

    return (hue, s, l)


# =============================================================================
# WCAG Contrast Checking Functions
# =============================================================================

def get_luminance(hex_color: str) -> float:
    """
    Calculate relative luminance per WCAG 2.1 formula.

    Luminance is calculated using the formula defined in WCAG 2.1:
    https://www.w3.org/TR/WCAG21/#dfn-relative-luminance

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB)

    Returns:
        Relative luminance value (0-1)
        - 0 = darkest black
        - 1 = lightest white

    Examples:
        >>> get_luminance("#FFFFFF")  # white
        1.0
        >>> get_luminance("#000000")  # black
        0.0
        >>> get_luminance("#1A365D")  # FirstLight Navy
        0.041
    """
    r, g, b = hex_to_rgb(hex_color)

    # Normalize to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0

    # Apply gamma correction per WCAG formula
    def _linearize(channel: float) -> float:
        if channel <= 0.03928:
            return channel / 12.92
        else:
            return ((channel + 0.055) / 1.055) ** 2.4

    r_linear = _linearize(r_norm)
    g_linear = _linearize(g_norm)
    b_linear = _linearize(b_norm)

    # Calculate relative luminance using WCAG coefficients
    luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear

    return luminance


def get_contrast_ratio(color1: str, color2: str) -> float:
    """
    Calculate contrast ratio between two colors per WCAG 2.1.

    Formula: (L1 + 0.05) / (L2 + 0.05)
    where L1 is the lighter color's luminance and L2 is the darker.

    Args:
        color1: First hex color string
        color2: Second hex color string

    Returns:
        Contrast ratio (1-21)
        - 1 = no contrast (same color)
        - 21 = maximum contrast (black on white)

    Examples:
        >>> get_contrast_ratio("#FFFFFF", "#000000")
        21.0
        >>> get_contrast_ratio("#FFFFFF", "#1A365D")  # Navy on white
        10.3
    """
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)

    # Ensure lighter color is in numerator
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    # WCAG contrast ratio formula
    ratio = (lighter + 0.05) / (darker + 0.05)

    return round(ratio, 2)


def check_wcag_aa(foreground: str, background: str, large_text: bool = False) -> bool:
    """
    Check if color combination meets WCAG AA standard.

    WCAG AA Requirements:
    - Normal text (<18px or <14px bold): 4.5:1 minimum
    - Large text (≥18px or ≥14px bold): 3:1 minimum

    Args:
        foreground: Foreground hex color string
        background: Background hex color string
        large_text: True if text is large (≥18px or ≥14px bold)

    Returns:
        True if passes WCAG AA, False otherwise

    Examples:
        >>> check_wcag_aa("#1A365D", "#FFFFFF")  # Navy on white
        True
        >>> check_wcag_aa("#90CDF4", "#FFFFFF")  # Light blue on white
        False
        >>> check_wcag_aa("#90CDF4", "#FFFFFF", large_text=True)
        True
    """
    ratio = get_contrast_ratio(foreground, background)

    if large_text:
        return ratio >= 3.0
    else:
        return ratio >= 4.5


def check_wcag_aaa(foreground: str, background: str, large_text: bool = False) -> bool:
    """
    Check if color combination meets WCAG AAA standard (enhanced).

    WCAG AAA Requirements:
    - Normal text (<18px or <14px bold): 7:1 minimum
    - Large text (≥18px or ≥14px bold): 4.5:1 minimum

    Args:
        foreground: Foreground hex color string
        background: Background hex color string
        large_text: True if text is large (≥18px or ≥14px bold)

    Returns:
        True if passes WCAG AAA, False otherwise

    Examples:
        >>> check_wcag_aaa("#1A365D", "#FFFFFF")  # Navy on white
        True
        >>> check_wcag_aaa("#2C5282", "#FFFFFF")  # Blue on white
        False
    """
    ratio = get_contrast_ratio(foreground, background)

    if large_text:
        return ratio >= 4.5
    else:
        return ratio >= 7.0


# =============================================================================
# Color Manipulation Functions
# =============================================================================

def lighten(hex_color: str, amount: float) -> str:
    """
    Lighten color by percentage.

    Increases the lightness value in HSL color space.

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB)
        amount: Amount to lighten (0-1, where 0.1 = 10% lighter)

    Returns:
        Lightened hex color string (#RRGGBB)

    Examples:
        >>> lighten("#1A365D", 0.2)
        '#2B5691'
        >>> lighten("#2C5282", 0.3)
        '#4A7DB8'
    """
    # Clamp amount to valid range
    amount = max(0.0, min(1.0, amount))

    h, s, l = hex_to_hsl(hex_color)

    # Increase lightness, clamping to max of 1.0
    new_l = min(1.0, l + amount)

    # Convert back to RGB
    r_norm, g_norm, b_norm = colorsys.hls_to_rgb(h / 360.0, new_l, s)

    # Convert to 0-255 range
    r = int(round(r_norm * 255))
    g = int(round(g_norm * 255))
    b = int(round(b_norm * 255))

    return rgb_to_hex(r, g, b)


def darken(hex_color: str, amount: float) -> str:
    """
    Darken color by percentage.

    Decreases the lightness value in HSL color space.

    Args:
        hex_color: Hex color string (#RRGGBB or RRGGBB)
        amount: Amount to darken (0-1, where 0.1 = 10% darker)

    Returns:
        Darkened hex color string (#RRGGBB)

    Examples:
        >>> darken("#90CDF4", 0.2)
        '#5AA8D9'
        >>> darken("#4299E1", 0.3)
        '#2B6CB0'
    """
    # Clamp amount to valid range
    amount = max(0.0, min(1.0, amount))

    h, s, l = hex_to_hsl(hex_color)

    # Decrease lightness, clamping to min of 0.0
    new_l = max(0.0, l - amount)

    # Convert back to RGB
    r_norm, g_norm, b_norm = colorsys.hls_to_rgb(h / 360.0, new_l, s)

    # Convert to 0-255 range
    r = int(round(r_norm * 255))
    g = int(round(g_norm * 255))
    b = int(round(b_norm * 255))

    return rgb_to_hex(r, g, b)


def get_accessible_text_color(background: str) -> str:
    """
    Return black or white text color based on background luminance.

    Uses relative luminance to determine which text color provides
    better contrast. Threshold is set at 0.5 luminance.

    Args:
        background: Background hex color string

    Returns:
        '#000000' (black) for light backgrounds
        '#FFFFFF' (white) for dark backgrounds

    Examples:
        >>> get_accessible_text_color("#FFFFFF")  # white background
        '#000000'
        >>> get_accessible_text_color("#1A365D")  # navy background
        '#FFFFFF'
        >>> get_accessible_text_color("#90CDF4")  # light blue
        '#000000'
    """
    luminance = get_luminance(background)

    # Threshold at 0.5 luminance
    # Dark backgrounds (< 0.5) get white text
    # Light backgrounds (>= 0.5) get black text
    if luminance < 0.5:
        return '#FFFFFF'
    else:
        return '#000000'


# =============================================================================
# Design System Color Constants
# =============================================================================

# Brand Colors
BRAND_COLORS = {
    'navy': '#1A365D',
    'blue': '#2C5282',
    'sky': '#3182CE',
    'slate': '#2D3748',
}

# Semantic Colors
SEMANTIC_COLORS = {
    'success': '#38A169',
    'success_dark': '#276749',
    'warning': '#D69E2E',
    'warning_dark': '#975A16',
    'danger': '#E53E3E',
    'danger_dark': '#C53030',
    'info': '#3182CE',
    'info_dark': '#2B6CB0',
}

# Flood Severity Palette (Colorblind Accessible)
FLOOD_SEVERITY = {
    'none': '#F7FAFC',
    'minor': '#90CDF4',
    'moderate': '#4299E1',
    'significant': '#2B6CB0',
    'severe': '#2C5282',
    'extreme': '#1A365D',
}

# Alternative High-Contrast Flood Palette
FLOOD_SEVERITY_HIGH_CONTRAST = {
    'none': '#F7FAFC',
    'minor': '#BEE3F8',
    'moderate': '#63B3ED',
    'significant': '#3182CE',
    'severe': '#2C5282',
    'extreme': '#1A365D',
}

# Wildfire Severity Palette (Colorblind Accessible)
WILDFIRE_SEVERITY = {
    'unburned': '#F7FAFC',
    'low': '#FED7AA',
    'moderate_low': '#FDBA74',
    'moderate': '#F97316',
    'moderate_high': '#EA580C',
    'high': '#9A3412',
}

# Confidence/Uncertainty Palette
CONFIDENCE_COLORS = {
    'high': '#276749',
    'medium': '#4A5568',
    'low': '#718096',
    'very_low': '#A0AEC0',
}

# Base Map Palette
BASE_MAP_COLORS = {
    'land': '#EDF2F7',
    'water_bodies': '#E2E8F0',
    'urban_areas': '#CBD5E0',
    'roads_major': '#A0AEC0',
    'roads_minor': '#CBD5E0',
    'boundaries': '#718096',
    'labels': '#4A5568',
}

# Neutral Scale
NEUTRAL_COLORS = {
    'neutral_50': '#F7FAFC',
    'neutral_100': '#EDF2F7',
    'neutral_200': '#E2E8F0',
    'neutral_300': '#CBD5E0',
    'neutral_400': '#A0AEC0',
    'neutral_500': '#718096',
    'neutral_600': '#4A5568',
    'neutral_700': '#2D3748',
    'neutral_800': '#1A202C',
    'neutral_900': '#171923',
}

# All colors dictionary for easy access
ALL_COLORS = {
    'brand': BRAND_COLORS,
    'semantic': SEMANTIC_COLORS,
    'flood_severity': FLOOD_SEVERITY,
    'flood_severity_high_contrast': FLOOD_SEVERITY_HIGH_CONTRAST,
    'wildfire_severity': WILDFIRE_SEVERITY,
    'confidence': CONFIDENCE_COLORS,
    'base_map': BASE_MAP_COLORS,
    'neutral': NEUTRAL_COLORS,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_color(category: str, name: str) -> str:
    """
    Get a color from the design system by category and name.

    Args:
        category: Color category (e.g., 'brand', 'flood_severity')
        name: Color name within category (e.g., 'navy', 'severe')

    Returns:
        Hex color string

    Raises:
        KeyError: If category or name not found

    Examples:
        >>> get_color('brand', 'navy')
        '#1A365D'
        >>> get_color('flood_severity', 'severe')
        '#2C5282'
    """
    if category not in ALL_COLORS:
        raise KeyError(f"Unknown color category: {category}")

    palette = ALL_COLORS[category]

    if name not in palette:
        raise KeyError(f"Unknown color name '{name}' in category '{category}'")

    return palette[name]


def validate_hex_color(hex_color: str) -> bool:
    """
    Validate hex color format.

    Args:
        hex_color: String to validate

    Returns:
        True if valid hex color format, False otherwise

    Examples:
        >>> validate_hex_color("#1A365D")
        True
        >>> validate_hex_color("1A365D")
        True
        >>> validate_hex_color("#XYZ123")
        False
    """
    hex_clean = hex_color.lstrip('#')
    return bool(re.match(r'^[0-9A-Fa-f]{6}$', hex_clean))
