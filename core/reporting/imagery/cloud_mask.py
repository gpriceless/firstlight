"""Cloud masking for Sentinel-2 optical imagery using Scene Classification Layer.

Sentinel-2 Level-2A data includes a Scene Classification Layer (SCL) band that
identifies clouds, cloud shadows, water, vegetation, and other surface types.
This module provides cloud masking capabilities to filter cloudy pixels from
rendered imagery.

SCL Classification Values:
    0: No data
    1: Saturated or defective
    2: Dark area pixels
    3: Cloud shadows
    4: Vegetation
    5: Bare soils
    6: Water
    7: Unclassified
    8: Cloud medium probability
    9: Cloud high probability
    10: Thin cirrus
    11: Snow/ice

Part of VIS-1.1: Satellite Imagery Renderer
Task 3.2: Cloud masking for optical imagery
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import numpy as np


class SCLClass(IntEnum):
    """Sentinel-2 Scene Classification Layer values.

    Defines all 12 classification values from the Sentinel-2 SCL band.
    Values 0-11 map to specific surface types and conditions.
    """

    NO_DATA = 0
    SATURATED_DEFECTIVE = 1
    DARK_AREA = 2
    CLOUD_SHADOWS = 3
    VEGETATION = 4
    BARE_SOILS = 5
    WATER = 6
    UNCLASSIFIED = 7
    CLOUD_MEDIUM_PROBABILITY = 8
    CLOUD_HIGH_PROBABILITY = 9
    THIN_CIRRUS = 10
    SNOW_ICE = 11


@dataclass
class CloudMaskConfig:
    """Configuration for cloud masking behavior.

    Controls which SCL classes to mask and how masked pixels are handled.

    Attributes:
        mask_clouds: Mask medium and high probability clouds (SCL 8, 9). Default True.
        mask_cirrus: Mask thin cirrus clouds (SCL 10). Default True.
        mask_shadows: Mask cloud shadows (SCL 3). Default False.
        mask_snow: Mask snow and ice (SCL 11). Default False.
        transparency: Transparency for masked pixels (0.0 = fully masked, 1.0 = show through).
                     Default 0.0 (fully transparent/masked).
    """

    mask_clouds: bool = True
    mask_cirrus: bool = True
    mask_shadows: bool = False
    mask_snow: bool = False
    transparency: float = 0.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.transparency <= 1.0:
            raise ValueError(
                f"transparency must be between 0.0 and 1.0, got {self.transparency}"
            )


@dataclass
class CloudMaskResult:
    """Result from cloud mask generation.

    Attributes:
        mask: Boolean array where True = masked/invalid pixel, False = valid pixel.
              Shape matches input SCL band.
        cloud_percentage: Percentage of scene covered by clouds (0-100).
        classes_masked: List of SCL classes that were masked.
    """

    mask: np.ndarray
    cloud_percentage: float
    classes_masked: List[SCLClass]

    def __post_init__(self):
        """Validate result attributes."""
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2D array, got {self.mask.ndim}D")

        if not 0.0 <= self.cloud_percentage <= 100.0:
            raise ValueError(
                f"cloud_percentage must be between 0 and 100, got {self.cloud_percentage}"
            )


class CloudMask:
    """Cloud masking for Sentinel-2 optical imagery.

    Generates cloud masks from Sentinel-2 Scene Classification Layer (SCL) band
    and applies masks to RGB imagery. Supports configurable masking of clouds,
    cirrus, shadows, and snow.

    Example:
        >>> from core.reporting.imagery.cloud_mask import CloudMask, CloudMaskConfig
        >>> import numpy as np
        >>>
        >>> # Create cloud masker with default config (mask clouds and cirrus)
        >>> masker = CloudMask()
        >>>
        >>> # Load SCL band from Sentinel-2 scene
        >>> scl = ...  # Shape (height, width), values 0-11
        >>>
        >>> # Generate cloud mask
        >>> result = masker.from_scl_band(scl)
        >>> print(f"Cloud coverage: {result.cloud_percentage:.1f}%")
        >>>
        >>> # Apply mask to RGB image
        >>> rgb = ...  # Shape (height, width, 3), values 0-255
        >>> masked_rgb = masker.apply_to_image(rgb, result.mask)
    """

    def __init__(self, config: Optional[CloudMaskConfig] = None):
        """Initialize cloud masker.

        Args:
            config: Cloud masking configuration. If None, uses default config.
        """
        self.config = config or CloudMaskConfig()

    def from_scl_band(self, scl: np.ndarray) -> CloudMaskResult:
        """Generate cloud mask from Scene Classification Layer band.

        Creates a boolean mask identifying pixels to exclude based on SCL
        classification and the configured masking rules.

        Args:
            scl: Scene Classification Layer array. Shape (height, width).
                Values should be 0-11 according to Sentinel-2 SCL specification.

        Returns:
            CloudMaskResult with:
                - mask: Boolean array (True = masked pixel)
                - cloud_percentage: Percentage of scene covered by clouds
                - classes_masked: List of SCL classes that were masked

        Raises:
            ValueError: If scl is not a 2D array.

        Example:
            >>> masker = CloudMask(CloudMaskConfig(mask_clouds=True, mask_cirrus=True))
            >>> scl = np.array([[4, 4, 8], [9, 10, 4]])  # vegetation, clouds, cirrus
            >>> result = masker.from_scl_band(scl)
            >>> result.mask
            array([[False, False,  True],
                   [ True,  True, False]])
            >>> result.cloud_percentage
            50.0
        """
        if scl.ndim != 2:
            raise ValueError(f"SCL band must be 2D array, got {scl.ndim}D with shape {scl.shape}")

        # Initialize mask (all False = all valid)
        mask = np.zeros(scl.shape, dtype=bool)
        classes_masked = []

        # Always mask no data and saturated/defective pixels
        mask |= (scl == SCLClass.NO_DATA)
        mask |= (scl == SCLClass.SATURATED_DEFECTIVE)
        classes_masked.extend([SCLClass.NO_DATA, SCLClass.SATURATED_DEFECTIVE])

        # Mask clouds (medium and high probability)
        if self.config.mask_clouds:
            mask |= (scl == SCLClass.CLOUD_MEDIUM_PROBABILITY)
            mask |= (scl == SCLClass.CLOUD_HIGH_PROBABILITY)
            classes_masked.extend([
                SCLClass.CLOUD_MEDIUM_PROBABILITY,
                SCLClass.CLOUD_HIGH_PROBABILITY
            ])

        # Mask cirrus clouds
        if self.config.mask_cirrus:
            mask |= (scl == SCLClass.THIN_CIRRUS)
            classes_masked.append(SCLClass.THIN_CIRRUS)

        # Mask cloud shadows
        if self.config.mask_shadows:
            mask |= (scl == SCLClass.CLOUD_SHADOWS)
            classes_masked.append(SCLClass.CLOUD_SHADOWS)

        # Mask snow and ice
        if self.config.mask_snow:
            mask |= (scl == SCLClass.SNOW_ICE)
            classes_masked.append(SCLClass.SNOW_ICE)

        # Calculate cloud coverage percentage
        # Only count cloud classes (8, 9, 10), not all masked pixels
        cloud_pixels = (
            (scl == SCLClass.CLOUD_MEDIUM_PROBABILITY) |
            (scl == SCLClass.CLOUD_HIGH_PROBABILITY) |
            (scl == SCLClass.THIN_CIRRUS)
        )
        total_pixels = scl.size
        cloud_percentage = (np.sum(cloud_pixels) / total_pixels) * 100.0

        return CloudMaskResult(
            mask=mask,
            cloud_percentage=cloud_percentage,
            classes_masked=classes_masked
        )

    def apply_to_image(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        fill_value: int = 255
    ) -> np.ndarray:
        """Apply cloud mask to RGB image.

        Masked pixels are set to fill_value (default white for transparency).
        If transparency > 0, masked pixels are blended with fill_value.

        Args:
            rgb: RGB image array. Shape (height, width, 3), dtype uint8, values 0-255.
            mask: Boolean mask array. Shape (height, width). True = masked pixel.
            fill_value: Value to set masked pixels to (0-255). Default 255 (white).
                       Use 0 for black, 255 for white (transparent in PNG with alpha).

        Returns:
            Masked RGB image. Same shape and dtype as input.
            Masked pixels are set to fill_value.

        Raises:
            ValueError: If rgb and mask shapes don't match.
            ValueError: If rgb is not 3D with 3 channels.
            ValueError: If fill_value is outside 0-255 range.

        Example:
            >>> rgb = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
            >>> mask = np.zeros((10, 10), dtype=bool)
            >>> mask[5:, 5:] = True  # Mask bottom-right quadrant
            >>>
            >>> masker = CloudMask()
            >>> masked_rgb = masker.apply_to_image(rgb, mask, fill_value=255)
            >>> # Bottom-right quadrant is now white (255, 255, 255)
            >>> np.all(masked_rgb[5:, 5:] == 255)
            True
        """
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must be 3D array with 3 channels, got shape {rgb.shape}"
            )

        if mask.shape != rgb.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} doesn't match rgb spatial shape {rgb.shape[:2]}"
            )

        if not 0 <= fill_value <= 255:
            raise ValueError(f"fill_value must be 0-255, got {fill_value}")

        # Create output copy
        output = rgb.copy()

        if self.config.transparency == 0.0:
            # Fully mask pixels
            output[mask] = fill_value
        elif self.config.transparency == 1.0:
            # No masking (show through completely)
            pass
        else:
            # Blend between original and fill_value
            alpha = self.config.transparency
            output[mask] = (
                alpha * rgb[mask] + (1 - alpha) * fill_value
            ).astype(np.uint8)

        return output

    def get_cloud_percentage(self, scl: np.ndarray) -> float:
        """Calculate percentage of scene covered by clouds.

        Counts pixels classified as cloud medium probability (8),
        cloud high probability (9), and thin cirrus (10).

        Args:
            scl: Scene Classification Layer array. Shape (height, width).

        Returns:
            Cloud coverage percentage (0-100).

        Raises:
            ValueError: If scl is not a 2D array.

        Example:
            >>> scl = np.array([[4, 8, 9], [10, 4, 4]])
            >>> masker = CloudMask()
            >>> masker.get_cloud_percentage(scl)
            50.0
        """
        if scl.ndim != 2:
            raise ValueError(f"SCL band must be 2D array, got {scl.ndim}D")

        cloud_pixels = (
            (scl == SCLClass.CLOUD_MEDIUM_PROBABILITY) |
            (scl == SCLClass.CLOUD_HIGH_PROBABILITY) |
            (scl == SCLClass.THIN_CIRRUS)
        )

        total_pixels = scl.size
        cloud_percentage = (np.sum(cloud_pixels) / total_pixels) * 100.0

        return cloud_percentage

    def get_classification_stats(self, scl: np.ndarray) -> dict:
        """Calculate percentage of each SCL class in the scene.

        Provides detailed breakdown of all classification types in the scene.
        Useful for understanding scene composition and data quality.

        Args:
            scl: Scene Classification Layer array. Shape (height, width).

        Returns:
            Dictionary mapping SCL class names to percentages (0-100).
            Includes all 12 SCL classes.

        Raises:
            ValueError: If scl is not a 2D array.

        Example:
            >>> scl = np.array([[4, 4, 8], [9, 10, 4]])
            >>> masker = CloudMask()
            >>> stats = masker.get_classification_stats(scl)
            >>> stats['VEGETATION']
            50.0
            >>> stats['CLOUD_HIGH_PROBABILITY']
            16.666666666666664
        """
        if scl.ndim != 2:
            raise ValueError(f"SCL band must be 2D array, got {scl.ndim}D")

        total_pixels = scl.size
        stats = {}

        for scl_class in SCLClass:
            count = np.sum(scl == scl_class.value)
            percentage = (count / total_pixels) * 100.0
            stats[scl_class.name] = percentage

        return stats
