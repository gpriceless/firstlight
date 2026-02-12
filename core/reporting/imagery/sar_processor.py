"""
SAR (Synthetic Aperture Radar) imagery processing for Sentinel-1.

This module provides specialized processing for SAR imagery, which uses radar
backscatter instead of optical reflectance. Key differences from optical imagery:
- Linear backscatter values converted to dB scale for visualization
- VV and VH polarizations instead of spectral bands
- Speckle noise requiring optional filtering
- Effective flood detection through low backscatter in water

SAR is critical for all-weather imaging, especially flood detection through clouds.

Part of VIS-1.1: Satellite Imagery Renderer
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Try to import scipy's median filter, fall back to numpy implementation
try:
    from scipy.ndimage import median_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class SARConfig:
    """Configuration for SAR imagery processing.

    Attributes:
        visualization: Visualization mode to use.
            - "vv": VV polarization grayscale
            - "vh": VH polarization grayscale
            - "dual_pol": VV/VH/ratio RGB composite
            - "flood_detection": VV-optimized for flood mapping
        db_min: Minimum dB value for stretching (default -20.0 dB)
        db_max: Maximum dB value for stretching (default 5.0 dB)
        apply_speckle_filter: Whether to apply speckle reduction (default False)
        colormap: Colormap for pseudocolor rendering.
            - "grayscale": Standard grayscale (default)
            - "viridis": Perceptually uniform colormap
            - "water": Blue-based colormap for water visualization
    """

    visualization: str = "vv"
    db_min: float = -20.0
    db_max: float = 5.0
    apply_speckle_filter: bool = False
    colormap: str = "grayscale"

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_viz = ["vv", "vh", "dual_pol", "flood_detection"]
        if self.visualization not in valid_viz:
            raise ValueError(
                f"Invalid visualization '{self.visualization}'. "
                f"Must be one of {valid_viz}"
            )

        valid_colormaps = ["grayscale", "viridis", "water"]
        if self.colormap not in valid_colormaps:
            raise ValueError(
                f"Invalid colormap '{self.colormap}'. "
                f"Must be one of {valid_colormaps}"
            )

        if self.db_min >= self.db_max:
            raise ValueError(
                f"db_min ({self.db_min}) must be less than db_max ({self.db_max})"
            )


@dataclass
class SARResult:
    """Result of SAR imagery processing.

    Attributes:
        rgb_array: Processed imagery array.
            - For grayscale: (H, W) with values in [0, 1]
            - For RGB/pseudocolor: (H, W, 3) with values in [0, 1]
        metadata: Processing metadata including:
            - db_range: Tuple of (min, max) dB values used for stretching
            - visualization: Visualization type applied
            - speckle_filtered: Whether speckle filtering was applied
            - colormap: Colormap used (if applicable)
        valid_mask: Boolean mask indicating valid pixels (True = valid data)
    """

    rgb_array: np.ndarray
    metadata: dict
    valid_mask: np.ndarray


class SARProcessor:
    """Processor for Sentinel-1 SAR imagery.

    Handles conversion from linear backscatter to dB scale, applies visualization
    modes optimized for SAR interpretation, and optionally reduces speckle noise.

    SAR backscatter characteristics:
    - VV polarization: Strong for surface scattering, good for water detection
    - VH polarization: Sensitive to volume scattering (vegetation, roughness)
    - Water appears dark (low backscatter) in both polarizations
    - Urban areas appear bright (high backscatter)

    Examples:
        >>> # VV single polarization for flood mapping
        >>> config = SARConfig(visualization="vv", db_min=-20, db_max=5)
        >>> processor = SARProcessor(config)
        >>> result = processor.process(vv_array)

        >>> # Dual polarization composite
        >>> config = SARConfig(visualization="dual_pol", apply_speckle_filter=True)
        >>> processor = SARProcessor(config)
        >>> result = processor.process(vv_array, vh_array)

        >>> # Custom colormap for water visualization
        >>> config = SARConfig(visualization="vv", colormap="water")
        >>> processor = SARProcessor(config)
        >>> result = processor.process(vv_array)
    """

    def __init__(self, config: Optional[SARConfig] = None):
        """Initialize SAR processor.

        Args:
            config: SAR processing configuration. If None, uses default config.
        """
        self.config = config or SARConfig()

    def process(
        self,
        vv: np.ndarray,
        vh: Optional[np.ndarray] = None
    ) -> SARResult:
        """Process SAR imagery to visualization-ready format.

        Converts linear backscatter to dB scale, applies stretching, and creates
        the requested visualization composite.

        Note: For "vh" visualization mode, the VH data can be passed as either
        the first parameter (vv) OR as the vh parameter. This allows single-band
        processing to work intuitively.

        Args:
            vv: VV polarization array (linear backscatter, not dB).
                For "vh" mode, this can contain VH data.
                Shape: (H, W), dtype: float32/float64
            vh: VH polarization array (linear backscatter, not dB). Optional.
                Required for "dual_pol" and "flood_detection" visualizations.
                Shape: (H, W), dtype: float32/float64

        Returns:
            SARResult with processed imagery, metadata, and valid data mask

        Raises:
            ValueError: If vh is required but not provided
            ValueError: If vv and vh have different shapes
            ValueError: If arrays are not 2D

        Examples:
            >>> # VV single polarization
            >>> result = processor.process(vv_array)
            >>> print(result.rgb_array.shape)  # (H, W) for grayscale

            >>> # VH single polarization (VH data in first param)
            >>> config = SARConfig(visualization="vh")
            >>> processor = SARProcessor(config)
            >>> result = processor.process(vh_array)

            >>> # Dual polarization
            >>> result = processor.process(vv_array, vh_array)
            >>> print(result.rgb_array.shape)  # (H, W, 3) for RGB composite
        """
        # Validate inputs
        if vv.ndim != 2:
            raise ValueError(f"VV array must be 2D, got shape {vv.shape}")

        # Check if VH is required for dual-pol modes
        requires_vh = self.config.visualization in ["dual_pol", "flood_detection"]
        if requires_vh and vh is None:
            raise ValueError(
                f"Visualization '{self.config.visualization}' requires VH polarization"
            )

        if vh is not None:
            if vh.ndim != 2:
                raise ValueError(f"VH array must be 2D, got shape {vh.shape}")
            if vv.shape != vh.shape:
                raise ValueError(
                    f"VV and VH must have same shape. "
                    f"Got VV={vv.shape}, VH={vh.shape}"
                )

        # Create valid data mask (exclude NaN, inf, and negative/zero values)
        valid_mask = np.isfinite(vv) & (vv > 0)
        if vh is not None:
            valid_mask &= np.isfinite(vh) & (vh > 0)

        # Convert to dB scale
        vv_db = self._to_db(vv)
        vh_db = self._to_db(vh) if vh is not None else None

        # Apply speckle filtering if requested
        if self.config.apply_speckle_filter:
            vv_db = self._speckle_filter(vv_db)
            if vh_db is not None:
                vh_db = self._speckle_filter(vh_db)

        # Apply dB stretch to 0-1 range
        vv_stretched = self._apply_db_stretch(vv_db)
        vh_stretched = self._apply_db_stretch(vh_db) if vh_db is not None else None

        # Create visualization based on mode
        if self.config.visualization == "vv":
            result_array = vv_stretched
        elif self.config.visualization == "vh":
            # For VH mode, use VH data if provided, otherwise treat vv param as VH
            # This allows single-band VH processing to work intuitively
            result_array = vh_stretched if vh_stretched is not None else vv_stretched
        elif self.config.visualization == "dual_pol":
            result_array = self._create_dual_pol_rgb(vv_stretched, vh_stretched)
        elif self.config.visualization == "flood_detection":
            # Flood detection: emphasize VV in red, VH in blue
            # Flooded areas (dark VV) appear dark, vegetation (bright VH) appears blue
            result_array = self._create_flood_detection_rgb(vv_stretched, vh_stretched)

        # Apply colormap if not already RGB and colormap is not grayscale
        colormap_applied = False
        if result_array.ndim == 2 and self.config.colormap != "grayscale":
            result_array = self._apply_colormap(result_array, self.config.colormap)
            colormap_applied = True

        # Apply valid mask
        if result_array.ndim == 2:
            result_array[~valid_mask] = 0.0
        else:
            result_array[~valid_mask] = [0.0, 0.0, 0.0]

        # Compile metadata
        metadata = {
            "db_range": (self.config.db_min, self.config.db_max),
            "visualization": self.config.visualization,
            "speckle_filtered": self.config.apply_speckle_filter,
            "colormap": self.config.colormap if colormap_applied else None,
        }

        return SARResult(
            rgb_array=result_array,
            metadata=metadata,
            valid_mask=valid_mask
        )

    def _to_db(self, linear_array: np.ndarray) -> np.ndarray:
        """Convert linear backscatter to dB scale.

        Formula: dB = 10 * log10(linear)

        Invalid values (NaN, inf, <=0) are preserved as NaN.

        Args:
            linear_array: Linear backscatter values

        Returns:
            Array in dB scale
        """
        # Create output array
        db_array = np.full_like(linear_array, np.nan, dtype=np.float32)

        # Only convert valid positive values
        valid = (linear_array > 0) & np.isfinite(linear_array)
        db_array[valid] = 10.0 * np.log10(linear_array[valid])

        return db_array

    def _apply_db_stretch(self, db_array: np.ndarray) -> np.ndarray:
        """Apply linear stretch to dB array, mapping to 0-1 range.

        Maps [db_min, db_max] to [0, 1] with clipping.

        Args:
            db_array: Array in dB scale

        Returns:
            Stretched array with values in [0, 1]
        """
        # Handle case where all values are invalid
        if db_array is None or not np.any(np.isfinite(db_array)):
            return np.zeros_like(db_array)

        # Linear stretch from db_min to db_max
        stretched = (db_array - self.config.db_min) / (
            self.config.db_max - self.config.db_min
        )

        # Clip to [0, 1]
        stretched = np.clip(stretched, 0.0, 1.0)

        # Preserve NaN values
        stretched[~np.isfinite(db_array)] = 0.0

        return stretched.astype(np.float32)

    def _create_dual_pol_rgb(
        self,
        vv_db: np.ndarray,
        vh_db: np.ndarray
    ) -> np.ndarray:
        """Create dual polarization RGB composite.

        Standard SAR RGB composite:
        - R: VV (surface scattering)
        - G: VH (volume scattering)
        - B: VV/VH ratio (depolarization)

        This combination allows visual discrimination of different surface types:
        - Water: Dark in all channels (low backscatter)
        - Vegetation: Green (high VH, lower VV)
        - Urban: Red-white (high VV, moderate VH)
        - Bare soil: Yellow (high VV and VH)

        Args:
            vv_db: VV stretched to [0, 1]
            vh_db: VH stretched to [0, 1]

        Returns:
            RGB array with shape (H, W, 3)
        """
        # Calculate VV/VH ratio
        # Avoid division by zero - use small epsilon
        epsilon = 1e-10
        ratio = vv_db / (vh_db + epsilon)

        # Normalize ratio to 0-1 range
        # Typical ratio range is 0.5 to 3.0, but clip to be safe
        ratio_normalized = np.clip((ratio - 0.5) / 2.5, 0.0, 1.0)

        # Stack into RGB
        rgb = np.stack([vv_db, vh_db, ratio_normalized], axis=-1)

        return rgb.astype(np.float32)

    def _create_flood_detection_rgb(
        self,
        vv_db: np.ndarray,
        vh_db: np.ndarray
    ) -> np.ndarray:
        """Create flood detection optimized RGB composite.

        Optimized for flood mapping:
        - R: VV (water appears dark)
        - G: VV (emphasize VV for water detection)
        - B: VH (vegetation appears bright)

        This makes:
        - Flooded areas: Dark gray/black (low VV)
        - Open water: Very dark (low VV and VH)
        - Vegetation: Blue-green (high VH)
        - Urban/bare: Yellow-white (high VV)

        Args:
            vv_db: VV stretched to [0, 1]
            vh_db: VH stretched to [0, 1]

        Returns:
            RGB array with shape (H, W, 3)
        """
        rgb = np.stack([vv_db, vv_db, vh_db], axis=-1)
        return rgb.astype(np.float32)

    def _apply_colormap(
        self,
        array: np.ndarray,
        colormap: str
    ) -> np.ndarray:
        """Apply pseudocolor colormap to grayscale array.

        Converts grayscale [0, 1] values to RGB using specified colormap.

        Args:
            array: Grayscale array in [0, 1] range
            colormap: Colormap name ("viridis", "water")

        Returns:
            RGB array with shape (H, W, 3)
        """
        if colormap == "viridis":
            # Simplified viridis: smooth purple->blue->green->yellow
            # Using piecewise linear interpolation
            r = np.where(array < 0.5,
                         array * 0.5,  # 0->0.25
                         0.25 + (array - 0.5) * 1.5)  # 0.25->1.0
            g = np.where(array < 0.25,
                         array * 0.5,  # 0->0.125
                         np.where(array < 0.75,
                                  0.125 + (array - 0.25) * 1.5,  # 0.125->0.875
                                  0.875 + (array - 0.75) * 0.5))  # 0.875->1.0
            b = np.where(array < 0.5,
                         0.5 + array * 0.5,  # 0.5->0.75
                         0.75 - (array - 0.5) * 1.5)  # 0.75->0
            rgb = np.stack([r, g, b], axis=-1)

        elif colormap == "water":
            # Blue-based colormap: dark blue -> cyan -> white
            r = array * array  # Quadratic for smoother transition
            g = array
            b = 0.3 + 0.7 * array  # Always has blue component
            rgb = np.stack([r, g, b], axis=-1)

        else:
            raise ValueError(f"Unsupported colormap: {colormap}")

        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def _speckle_filter(
        self,
        array: np.ndarray,
        window: int = 3
    ) -> np.ndarray:
        """Apply speckle reduction filter.

        Uses median filter to reduce speckle noise in SAR imagery while
        preserving edges better than mean filtering.

        Note: For production use, more sophisticated filters like Lee or
        Frost filters would be better, but median is simple and effective.

        Args:
            array: Input array (typically in dB scale)
            window: Filter window size (default 3x3)

        Returns:
            Filtered array
        """
        # Preserve NaN values
        valid = np.isfinite(array)

        # Apply median filter to valid data
        filtered = array.copy()

        # Only filter if we have valid data
        if np.any(valid):
            if HAS_SCIPY:
                # Use scipy's optimized median filter
                masked = np.where(valid, array, 0.0)
                from scipy.ndimage import median_filter
                filtered = median_filter(masked, size=window, mode='nearest')
                # Restore NaN values
                filtered[~valid] = np.nan
            else:
                # Fall back to numpy-based median filter
                filtered = self._numpy_median_filter(array, window, valid)

        return filtered

    def _numpy_median_filter(
        self,
        array: np.ndarray,
        window: int,
        valid: np.ndarray
    ) -> np.ndarray:
        """Numpy-based median filter fallback.

        Slower than scipy but works without scipy dependency.

        Args:
            array: Input array
            window: Filter window size
            valid: Boolean mask of valid (finite) values

        Returns:
            Filtered array
        """
        filtered = array.copy()
        h, w = array.shape
        pad = window // 2

        # Pad the array with edge values
        padded = np.pad(array, pad, mode='edge')
        valid_padded = np.pad(valid, pad, mode='edge')

        # Apply median filter
        for i in range(h):
            for j in range(w):
                if valid[i, j]:
                    # Extract window
                    window_data = padded[i:i+window, j:j+window]
                    window_valid = valid_padded[i:i+window, j:j+window]

                    # Compute median of valid values in window
                    valid_window = window_data[window_valid]
                    if len(valid_window) > 0:
                        filtered[i, j] = np.median(valid_window)

        return filtered
