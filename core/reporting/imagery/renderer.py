"""
Main ImageryRenderer class for satellite imagery visualization.

Orchestrates band selection, histogram stretching, and RGB composition to convert
multi-band satellite imagery into displayable RGB images. Supports multiple sensors
and graceful degradation when data is incomplete.

Part of VIS-1.1: Satellite Imagery Renderer

Example:
    # Basic usage with defaults
    renderer = ImageryRenderer()
    result = renderer.render("sentinel2_scene.tif", sensor="sentinel2")

    # Custom configuration
    config = RendererConfig(
        composite_name="false_color_ir",
        stretch_method=HistogramStretch.STDDEV,
        output_format="uint8"
    )
    renderer = ImageryRenderer(config)
    result = renderer.render(band_data_dict, sensor="sentinel2")

    # Access rendered output
    rgb_image = result.rgb_array  # (H, W, 3) uint8 or float
    valid_mask = result.valid_mask  # Boolean mask
    coverage = result.metadata['coverage_percent']
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from core.reporting.imagery.band_combinations import (
    BandComposite,
    get_bands_for_composite,
    get_available_composites,
)
from core.reporting.imagery.histogram import HistogramStretch, stretch_band
from core.reporting.imagery.utils import (
    detect_partial_coverage,
    handle_nodata,
    normalize_to_uint8,
    stack_bands_to_rgb,
)


@dataclass
class RendererConfig:
    """Configuration for ImageryRenderer.

    Attributes:
        composite_name: Name of band composite to use (e.g., 'true_color', 'false_color_ir')
        stretch_method: Histogram stretch method to apply
        output_format: Output format - 'uint8' (0-255) or 'float' (0.0-1.0)
        apply_cloud_mask: Whether to apply cloud masking if available
        custom_min: Optional manual minimum for histogram stretch
        custom_max: Optional manual maximum for histogram stretch
    """

    composite_name: str = "true_color"
    stretch_method: HistogramStretch = HistogramStretch.LINEAR_2PCT
    output_format: str = "uint8"
    apply_cloud_mask: bool = False
    custom_min: Optional[float] = None
    custom_max: Optional[float] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.output_format not in ("uint8", "float"):
            raise ValueError(
                f"output_format must be 'uint8' or 'float', got '{self.output_format}'"
            )

        if self.custom_min is not None and self.custom_max is not None:
            if self.custom_min >= self.custom_max:
                raise ValueError(
                    f"custom_min ({self.custom_min}) must be less than "
                    f"custom_max ({self.custom_max})"
                )


@dataclass
class RenderedImage:
    """Result of rendering imagery to RGB.

    Attributes:
        rgb_array: Rendered RGB image with shape (H, W, 3)
        metadata: Rendering metadata including sensor, composite, stretch params, coverage
        valid_mask: Boolean mask indicating valid pixels (True = valid, False = nodata)
    """

    rgb_array: np.ndarray
    metadata: Dict[str, any] = field(default_factory=dict)
    valid_mask: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Validate rendered image."""
        if self.rgb_array.ndim != 3 or self.rgb_array.shape[2] != 3:
            raise ValueError(
                f"rgb_array must have shape (H, W, 3), got {self.rgb_array.shape}"
            )


class ImageryRenderer:
    """Main renderer for converting multi-band satellite imagery to RGB visualizations.

    This class orchestrates the complete rendering pipeline:
    1. Load or extract required bands for the composite
    2. Apply histogram stretching for visual enhancement
    3. Stack bands into RGB array
    4. Handle missing data and compute coverage statistics

    Supports multiple input formats:
    - File path to GeoTIFF
    - Numpy array with shape (H, W, bands)
    - Dictionary mapping band names to 2D arrays

    Examples:
        # Render from file
        renderer = ImageryRenderer()
        result = renderer.render("sentinel2_scene.tif", sensor="sentinel2")

        # Render from band dictionary
        bands = {"B04": red_array, "B03": green_array, "B02": blue_array}
        result = renderer.render(bands, sensor="sentinel2")

        # Custom configuration
        config = RendererConfig(
            composite_name="false_color_ir",
            stretch_method=HistogramStretch.STDDEV
        )
        renderer = ImageryRenderer(config)
        result = renderer.render(bands, sensor="sentinel2")
    """

    def __init__(self, config: Optional[RendererConfig] = None):
        """Initialize renderer with optional configuration.

        Args:
            config: Renderer configuration. If None, uses default config.
        """
        self.config = config if config is not None else RendererConfig()

    def render(
        self,
        data: Union[str, Path, np.ndarray, Dict[str, np.ndarray]],
        sensor: str,
        composite_name: Optional[str] = None,
        nodata_value: Optional[float] = None
    ) -> RenderedImage:
        """Main entry point for rendering imagery.

        Args:
            data: Input data - file path (str/Path), numpy array (H,W,bands),
                  or dict of band arrays
            sensor: Sensor identifier ('sentinel2', 'landsat8', 'landsat9', 'sentinel1')
            composite_name: Override composite name from config (optional)
            nodata_value: Explicit nodata value to handle (optional)

        Returns:
            RenderedImage with RGB array, metadata, and valid mask

        Raises:
            ValueError: If sensor is not supported or composite is unavailable
            FileNotFoundError: If file path doesn't exist

        Examples:
            >>> renderer = ImageryRenderer()
            >>> result = renderer.render("scene.tif", sensor="sentinel2")
            >>> print(f"Coverage: {result.metadata['coverage_percent']:.1f}%")
        """
        # Use composite from argument or config
        composite_name_to_use = composite_name or self.config.composite_name

        # Get composite definition
        composite = get_bands_for_composite(sensor, composite_name_to_use)

        # Load bands from source
        band_data = self._load_bands(data, sensor, composite, nodata_value)

        # Handle missing bands gracefully
        composite = self._handle_missing_bands(band_data, composite)

        # Select the three bands for RGB
        r_band, g_band, b_band = self._select_bands(band_data, composite)

        # Apply histogram stretching
        r_stretched, g_stretched, b_stretched = self._apply_stretch(
            (r_band, g_band, b_band), nodata_value
        )

        # Stack to RGB and convert to output format
        rgb_array = self._to_rgb(r_stretched, g_stretched, b_stretched)

        # Compute valid data mask and coverage
        valid_mask = self._compute_valid_mask(r_band, g_band, b_band, nodata_value)
        coverage_percent = detect_partial_coverage(r_band, nodata_value)

        # Build metadata
        metadata = {
            "sensor": sensor,
            "composite_name": composite.name,
            "composite_description": composite.description,
            "bands_used": composite.bands,
            "stretch_method": self.config.stretch_method.value,
            "output_format": self.config.output_format,
            "coverage_percent": coverage_percent,
            "shape": rgb_array.shape,
        }

        # Add stretch parameters if custom values were used
        if self.config.custom_min is not None:
            metadata["custom_min"] = self.config.custom_min
        if self.config.custom_max is not None:
            metadata["custom_max"] = self.config.custom_max

        return RenderedImage(
            rgb_array=rgb_array,
            metadata=metadata,
            valid_mask=valid_mask
        )

    def _load_bands(
        self,
        source: Union[str, Path, np.ndarray, Dict[str, np.ndarray]],
        sensor: str,
        composite: BandComposite,
        nodata_value: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """Load bands from various source formats.

        Args:
            source: Data source - file path, array, or band dict
            sensor: Sensor identifier
            composite: Band composite definition
            nodata_value: Optional nodata value

        Returns:
            Dictionary mapping band names to 2D arrays

        Raises:
            ValueError: If source format is not supported
            FileNotFoundError: If file doesn't exist
        """
        # Case 1: Dictionary of band arrays (already in correct format)
        if isinstance(source, dict):
            return source

        # Case 2: File path - load with rasterio
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            return self._load_from_geotiff(path, composite)

        # Case 3: Numpy array (H, W, bands) - need to map to band names
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                # Single band - replicate for grayscale
                return {
                    composite.bands[0]: source,
                    composite.bands[1]: source,
                    composite.bands[2]: source,
                }
            elif source.ndim == 3:
                # Multi-band - map bands by index
                # This requires knowing the band order, which is sensor-specific
                raise NotImplementedError(
                    "Loading from multi-band numpy array requires explicit band "
                    "mapping. Please provide a dictionary mapping band names to arrays."
                )
            else:
                raise ValueError(
                    f"Array must be 2D or 3D, got shape {source.shape}"
                )

        raise ValueError(
            f"Unsupported data source type: {type(source)}. "
            "Expected str, Path, np.ndarray, or dict."
        )

    def _load_from_geotiff(
        self,
        path: Path,
        composite: BandComposite
    ) -> Dict[str, np.ndarray]:
        """Load bands from GeoTIFF file using rasterio.

        Args:
            path: Path to GeoTIFF file
            composite: Band composite to determine which bands to load

        Returns:
            Dictionary mapping band names to 2D arrays
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError(
                "rasterio is required for loading GeoTIFF files. "
                "Install with: pip install rasterio"
            )

        band_data = {}

        with rasterio.open(path) as src:
            # Get band descriptions/names from metadata
            band_descriptions = src.descriptions

            # Try to find bands by name in descriptions
            for band_name in composite.bands:
                band_idx = None

                # Search for band in descriptions
                if band_descriptions:
                    for idx, desc in enumerate(band_descriptions, start=1):
                        if desc and band_name in desc:
                            band_idx = idx
                            break

                # If not found in descriptions, try to parse band number
                # e.g., "B04" -> band 4
                if band_idx is None:
                    # Try to extract number from band name
                    import re
                    match = re.search(r'\d+', band_name)
                    if match:
                        potential_idx = int(match.group())
                        if 1 <= potential_idx <= src.count:
                            band_idx = potential_idx

                # Load the band if found
                if band_idx is not None:
                    band_data[band_name] = src.read(band_idx)
                # Otherwise, leave missing (will be handled by _handle_missing_bands)

        return band_data

    def _select_bands(
        self,
        band_data: Dict[str, np.ndarray],
        composite: BandComposite
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select the three bands needed for RGB composite.

        Args:
            band_data: Dictionary of available bands
            composite: Band composite definition

        Returns:
            Tuple of (red, green, blue) arrays

        Raises:
            ValueError: If required bands are missing from band_data
        """
        required_bands = composite.bands

        # Check all required bands are present
        missing = [b for b in required_bands if b not in band_data]
        if missing:
            raise ValueError(
                f"Missing required bands for composite '{composite.name}': {missing}. "
                f"Available bands: {list(band_data.keys())}"
            )

        # Extract the three bands
        r_band = band_data[required_bands[0]]
        g_band = band_data[required_bands[1]]
        b_band = band_data[required_bands[2]]

        # Validate shapes match
        if not (r_band.shape == g_band.shape == b_band.shape):
            raise ValueError(
                f"Band shapes must match. Got: "
                f"{required_bands[0]}={r_band.shape}, "
                f"{required_bands[1]}={g_band.shape}, "
                f"{required_bands[2]}={b_band.shape}"
            )

        return r_band, g_band, b_band

    def _apply_stretch(
        self,
        bands: Tuple[np.ndarray, np.ndarray, np.ndarray],
        nodata_value: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply histogram stretching to all three bands.

        Args:
            bands: Tuple of (red, green, blue) arrays
            nodata_value: Optional nodata value to exclude

        Returns:
            Tuple of stretched (red, green, blue) arrays in [0, 1] range
        """
        r_band, g_band, b_band = bands

        # Apply stretch to each band
        r_stretched = stretch_band(
            r_band,
            method=self.config.stretch_method,
            min_val=self.config.custom_min,
            max_val=self.config.custom_max,
            nodata=nodata_value
        )

        g_stretched = stretch_band(
            g_band,
            method=self.config.stretch_method,
            min_val=self.config.custom_min,
            max_val=self.config.custom_max,
            nodata=nodata_value
        )

        b_stretched = stretch_band(
            b_band,
            method=self.config.stretch_method,
            min_val=self.config.custom_min,
            max_val=self.config.custom_max,
            nodata=nodata_value
        )

        return r_stretched, g_stretched, b_stretched

    def _to_rgb(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """Stack bands to RGB and convert to output format.

        Args:
            r: Red band (values in [0, 1])
            g: Green band (values in [0, 1])
            b: Blue band (values in [0, 1])

        Returns:
            RGB array in configured output format (uint8 or float)
        """
        # Stack to RGB
        rgb = stack_bands_to_rgb(r, g, b)

        # Convert to output format
        if self.config.output_format == "uint8":
            # Normalize assumes input is in [0, 1] and outputs [0, 255] uint8
            rgb = normalize_to_uint8(rgb, 0.0, 1.0)
        # else: keep as float in [0, 1]

        return rgb

    def _handle_missing_bands(
        self,
        band_data: Dict[str, np.ndarray],
        composite: BandComposite
    ) -> BandComposite:
        """Handle missing bands with graceful degradation.

        Attempts to find fallback composites if required bands are missing.
        For example, if SWIR bands are missing, fall back to true color.

        Args:
            band_data: Available band data
            composite: Requested composite

        Returns:
            Composite to use (may be different from requested if bands missing)

        Raises:
            ValueError: If no suitable composite can be found
        """
        required_bands = composite.bands
        available_bands = set(band_data.keys())

        # Check if all required bands are present
        if all(band in available_bands for band in required_bands):
            # All good, use requested composite
            return composite

        # Missing some bands - try to find a fallback
        missing = [b for b in required_bands if b not in available_bands]

        # Get all available composites for this sensor
        sensor = composite.sensor
        available_composites = get_available_composites(sensor)

        # Try fallback to true_color as it's most commonly available
        if composite.name != "true_color" and "true_color" in available_composites:
            fallback = get_bands_for_composite(sensor, "true_color")
            fallback_bands = fallback.bands

            # Check if fallback composite has all required bands
            if all(band in available_bands for band in fallback_bands):
                # Use fallback
                return fallback

        # No fallback found - raise error
        raise ValueError(
            f"Cannot render composite '{composite.name}' - missing bands: {missing}. "
            f"Available bands: {list(available_bands)}. "
            f"No suitable fallback composite found."
        )

    def _compute_valid_mask(
        self,
        r_band: np.ndarray,
        g_band: np.ndarray,
        b_band: np.ndarray,
        nodata_value: Optional[float] = None
    ) -> np.ndarray:
        """Compute combined valid data mask from all three bands.

        A pixel is valid only if it's valid in all three bands.

        Args:
            r_band: Red band
            g_band: Green band
            b_band: Blue band
            nodata_value: Optional nodata value

        Returns:
            Boolean array (True = valid, False = nodata)
        """
        # Get validity mask for each band
        _, r_valid = handle_nodata(r_band, nodata_value)
        _, g_valid = handle_nodata(g_band, nodata_value)
        _, b_valid = handle_nodata(b_band, nodata_value)

        # Combine masks - pixel is valid only if valid in all bands
        combined_mask = r_valid & g_valid & b_valid

        return combined_mask
