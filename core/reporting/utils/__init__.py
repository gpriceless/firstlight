"""
Reporting utilities for FirstLight.

Provides color utilities, formatting helpers, and other utilities
for generating human-readable reports.
"""

from .color_utils import (
    # Color conversion functions
    hex_to_rgb,
    rgb_to_hex,
    hex_to_hsl,

    # WCAG contrast checking
    get_luminance,
    get_contrast_ratio,
    check_wcag_aa,
    check_wcag_aaa,

    # Color manipulation
    lighten,
    darken,
    get_accessible_text_color,

    # Design system colors
    BRAND_COLORS,
    SEMANTIC_COLORS,
    FLOOD_SEVERITY,
    FLOOD_SEVERITY_HIGH_CONTRAST,
    WILDFIRE_SEVERITY,
    CONFIDENCE_COLORS,
    BASE_MAP_COLORS,
    NEUTRAL_COLORS,
    ALL_COLORS,

    # Helper functions
    get_color,
    validate_hex_color,
)

__all__ = [
    # Color conversion
    'hex_to_rgb',
    'rgb_to_hex',
    'hex_to_hsl',

    # WCAG contrast
    'get_luminance',
    'get_contrast_ratio',
    'check_wcag_aa',
    'check_wcag_aaa',

    # Color manipulation
    'lighten',
    'darken',
    'get_accessible_text_color',

    # Design system colors
    'BRAND_COLORS',
    'SEMANTIC_COLORS',
    'FLOOD_SEVERITY',
    'FLOOD_SEVERITY_HIGH_CONTRAST',
    'WILDFIRE_SEVERITY',
    'CONFIDENCE_COLORS',
    'BASE_MAP_COLORS',
    'NEUTRAL_COLORS',
    'ALL_COLORS',

    # Helpers
    'get_color',
    'validate_hex_color',
]
