"""Band combination definitions for satellite imagery rendering.

Defines spectral band combinations for visualizing multi-band satellite imagery
across different sensors (Sentinel-2, Landsat 8/9, Sentinel-1 SAR).

Each composite specifies which bands to use for red, green, and blue channels,
enabling different visualization modes to highlight specific features:
- True color: Natural appearance (visible spectrum)
- False color infrared: Vegetation and water contrast
- SWIR: Burn scars, soil moisture, water bodies
- SAR: Radar backscatter for all-weather imaging
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class BandComposite:
    """Definition of a band combination for multi-spectral imagery rendering.

    Attributes:
        name: Identifier for this composite (e.g., 'true_color', 'false_color_ir')
        bands: Tuple of 3 band identifiers (red, green, blue) for RGB rendering
        description: Human-readable description of what this composite shows
        sensor: Sensor type this composite applies to ('sentinel2', 'landsat8', etc.)
        stretch_range: Optional (min, max) percentile for histogram stretch (default: 2, 98)
    """

    name: str
    bands: tuple[str, str, str]
    description: str
    sensor: str
    stretch_range: Optional[tuple[float, float]] = None

    def __post_init__(self):
        """Validate band composite definition."""
        if len(self.bands) != 3:
            raise ValueError(f"Band composite must have exactly 3 bands, got {len(self.bands)}")

        # Set default stretch range if not provided
        if self.stretch_range is None:
            self.stretch_range = (2.0, 98.0)


# Sentinel-2 Band Combinations
# Sentinel-2 has 13 bands from visible to SWIR
# Key bands: B02 (blue), B03 (green), B04 (red), B08 (NIR), B8A (narrow NIR), B11/B12 (SWIR)

SENTINEL2_COMPOSITES = {
    "true_color": BandComposite(
        name="true_color",
        bands=("B04", "B03", "B02"),  # Red, Green, Blue
        description="Natural color composite using visible spectrum bands",
        sensor="sentinel2"
    ),
    "false_color_ir": BandComposite(
        name="false_color_ir",
        bands=("B08", "B04", "B03"),  # NIR, Red, Green
        description="False color infrared - highlights vegetation (red) and water (dark blue/black)",
        sensor="sentinel2"
    ),
    "swir": BandComposite(
        name="swir",
        bands=("B12", "B8A", "B04"),  # SWIR2, Narrow NIR, Red
        description="Short-wave infrared composite - highlights burn scars, soil moisture, water penetration",
        sensor="sentinel2"
    ),
    "agriculture": BandComposite(
        name="agriculture",
        bands=("B11", "B08", "B02"),  # SWIR1, NIR, Blue
        description="Agriculture composite - crop health and soil moisture",
        sensor="sentinel2"
    ),
    "geology": BandComposite(
        name="geology",
        bands=("B12", "B11", "B02"),  # SWIR2, SWIR1, Blue
        description="Geology composite - rock types and mineral mapping",
        sensor="sentinel2"
    ),
}


# Landsat 8/9 Band Combinations
# Landsat Collection 2 Level-2 band naming: SR_B1, SR_B2, etc.
# B1 (coastal/aerosol), B2 (blue), B3 (green), B4 (red), B5 (NIR), B6 (SWIR1), B7 (SWIR2)

LANDSAT_COMPOSITES = {
    "true_color": BandComposite(
        name="true_color",
        bands=("SR_B4", "SR_B3", "SR_B2"),  # Red, Green, Blue
        description="Natural color composite using visible spectrum bands",
        sensor="landsat8"
    ),
    "false_color_ir": BandComposite(
        name="false_color_ir",
        bands=("SR_B5", "SR_B4", "SR_B3"),  # NIR, Red, Green
        description="False color infrared - highlights vegetation (red) and water (dark blue/black)",
        sensor="landsat8"
    ),
    "swir": BandComposite(
        name="swir",
        bands=("SR_B7", "SR_B5", "SR_B4"),  # SWIR2, NIR, Red
        description="Short-wave infrared composite - highlights burn scars, soil moisture, water penetration",
        sensor="landsat8"
    ),
    "agriculture": BandComposite(
        name="agriculture",
        bands=("SR_B6", "SR_B5", "SR_B2"),  # SWIR1, NIR, Blue
        description="Agriculture composite - crop health and soil moisture",
        sensor="landsat8"
    ),
    "geology": BandComposite(
        name="geology",
        bands=("SR_B7", "SR_B6", "SR_B2"),  # SWIR2, SWIR1, Blue
        description="Geology composite - rock types and mineral mapping",
        sensor="landsat8"
    ),
}


# Sentinel-1 SAR Visualizations
# SAR imagery uses polarizations instead of spectral bands
# VV: vertical transmit, vertical receive
# VH: vertical transmit, horizontal receive
# VV is typically stronger backscatter, VH shows surface roughness

SAR_VISUALIZATIONS = {
    "vv_single": BandComposite(
        name="vv_single",
        bands=("VV", "VV", "VV"),  # Single polarization grayscale
        description="VV polarization - water detection and flood mapping (water appears dark)",
        sensor="sentinel1",
        stretch_range=(-20.0, 5.0)  # dB scale typical range
    ),
    "vh_single": BandComposite(
        name="vh_single",
        bands=("VH", "VH", "VH"),  # Single polarization grayscale
        description="VH polarization - surface roughness and vegetation structure",
        sensor="sentinel1",
        stretch_range=(-25.0, 0.0)  # dB scale typical range
    ),
    "dual_pol": BandComposite(
        name="dual_pol",
        bands=("VV", "VH", "VV/VH"),  # Dual polarization composite
        description="Dual polarization composite - combines VV and VH for feature discrimination",
        sensor="sentinel1",
        stretch_range=(-20.0, 5.0)  # dB scale typical range
    ),
    "flood_detection": BandComposite(
        name="flood_detection",
        bands=("VV", "VV", "VH"),  # VV dominant for water detection
        description="Optimized for flood detection - flooded areas appear dark in VV",
        sensor="sentinel1",
        stretch_range=(-20.0, 5.0)  # dB scale typical range
    ),
}


def get_bands_for_composite(sensor: str, composite_name: str) -> BandComposite:
    """Get band composite definition for a given sensor and composite type.

    Args:
        sensor: Sensor identifier ('sentinel2', 'landsat8', 'landsat9', 'sentinel1')
        composite_name: Name of the composite (e.g., 'true_color', 'false_color_ir')

    Returns:
        BandComposite definition with band identifiers and metadata

    Raises:
        ValueError: If sensor or composite_name is not recognized

    Example:
        >>> composite = get_bands_for_composite('sentinel2', 'true_color')
        >>> print(composite.bands)
        ('B04', 'B03', 'B02')
    """
    sensor_lower = sensor.lower()

    # Map sensors to their composite dictionaries
    sensor_map = {
        'sentinel2': SENTINEL2_COMPOSITES,
        'sentinel-2': SENTINEL2_COMPOSITES,
        's2': SENTINEL2_COMPOSITES,
        'landsat8': LANDSAT_COMPOSITES,
        'landsat-8': LANDSAT_COMPOSITES,
        'landsat9': LANDSAT_COMPOSITES,
        'landsat-9': LANDSAT_COMPOSITES,
        'l8': LANDSAT_COMPOSITES,
        'l9': LANDSAT_COMPOSITES,
        'sentinel1': SAR_VISUALIZATIONS,
        'sentinel-1': SAR_VISUALIZATIONS,
        's1': SAR_VISUALIZATIONS,
    }

    if sensor_lower not in sensor_map:
        supported = ['sentinel2', 'landsat8', 'landsat9', 'sentinel1']
        raise ValueError(
            f"Unsupported sensor '{sensor}'. Supported sensors: {supported}"
        )

    composites = sensor_map[sensor_lower]

    if composite_name not in composites:
        available = list(composites.keys())
        raise ValueError(
            f"Composite '{composite_name}' not available for sensor '{sensor}'. "
            f"Available composites: {available}"
        )

    return composites[composite_name]


def get_available_composites(sensor: str) -> list[str]:
    """Get list of available composite names for a given sensor.

    Args:
        sensor: Sensor identifier ('sentinel2', 'landsat8', 'landsat9', 'sentinel1')

    Returns:
        List of composite names available for this sensor

    Raises:
        ValueError: If sensor is not recognized

    Example:
        >>> composites = get_available_composites('sentinel2')
        >>> 'true_color' in composites
        True
    """
    sensor_lower = sensor.lower()

    # Map sensors to their composite dictionaries
    sensor_map = {
        'sentinel2': SENTINEL2_COMPOSITES,
        'sentinel-2': SENTINEL2_COMPOSITES,
        's2': SENTINEL2_COMPOSITES,
        'landsat8': LANDSAT_COMPOSITES,
        'landsat-8': LANDSAT_COMPOSITES,
        'landsat9': LANDSAT_COMPOSITES,
        'landsat-9': LANDSAT_COMPOSITES,
        'l8': LANDSAT_COMPOSITES,
        'l9': LANDSAT_COMPOSITES,
        'sentinel1': SAR_VISUALIZATIONS,
        'sentinel-1': SAR_VISUALIZATIONS,
        's1': SAR_VISUALIZATIONS,
    }

    if sensor_lower not in sensor_map:
        supported = ['sentinel2', 'landsat8', 'landsat9', 'sentinel1']
        raise ValueError(
            f"Unsupported sensor '{sensor}'. Supported sensors: {supported}"
        )

    composites = sensor_map[sensor_lower]
    return list(composites.keys())
