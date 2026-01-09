"""
Wind Damage Vegetation Detection Algorithm

Detects vegetation damage from high winds using change detection
in vegetation indices (NDVI, EVI). Identifies areas of vegetation loss
caused by storm events such as hurricanes, cyclones, and tornadoes.

Algorithm ID: storm.baseline.wind_damage
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Named constants for algorithm thresholds
NDVI_HEALTHY_MIN = 0.3  # Minimum NDVI for healthy vegetation
NDVI_SPARSE_MIN = 0.1   # Minimum NDVI for sparse vegetation
NDVI_CHANGE_SEVERE = -0.4  # NDVI change threshold for severe damage
NDVI_CHANGE_MODERATE = -0.2  # NDVI change threshold for moderate damage
NDVI_CHANGE_MINOR = -0.1  # NDVI change threshold for minor damage
EVI_SCALE_FACTOR = 2.5  # EVI formula scaling factor
EVI_COEFF_RED = 6.0  # EVI red band coefficient
EVI_COEFF_BLUE = 7.5  # EVI blue band coefficient
EVI_OFFSET = 1.0  # EVI offset constant


@dataclass
class WindDamageConfig:
    """Configuration for wind damage vegetation detection."""

    ndvi_change_threshold: float = -0.2  # NDVI change threshold for damage detection
    min_damage_area_ha: float = 0.5  # Minimum damage polygon area (hectares)
    vegetation_index: Literal["ndvi", "evi"] = "ndvi"  # Which index to use
    pre_event_ndvi_min: float = 0.3  # Minimum pre-event NDVI for vegetated area
    use_severity_classification: bool = True  # Enable damage severity levels
    cloud_mask_threshold: float = 0.2  # Max cloud cover fraction to process

    def __post_init__(self):
        """Validate configuration parameters."""
        if not -1.0 <= self.ndvi_change_threshold <= 0.0:
            raise ValueError(
                f"ndvi_change_threshold must be in [-1.0, 0.0], got {self.ndvi_change_threshold}"
            )
        if self.min_damage_area_ha < 0:
            raise ValueError(
                f"min_damage_area_ha must be non-negative, got {self.min_damage_area_ha}"
            )
        if self.vegetation_index not in ["ndvi", "evi"]:
            raise ValueError(
                f"vegetation_index must be 'ndvi' or 'evi', got {self.vegetation_index}"
            )
        if not 0.0 <= self.pre_event_ndvi_min <= 1.0:
            raise ValueError(
                f"pre_event_ndvi_min must be in [0.0, 1.0], got {self.pre_event_ndvi_min}"
            )


@dataclass
class WindDamageResult:
    """Results from wind damage vegetation detection."""

    damage_extent: np.ndarray  # Binary mask of damage extent
    damage_severity: np.ndarray  # Severity classification (0=none, 1=minor, 2=moderate, 3=severe)
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    ndvi_change: np.ndarray  # NDVI change map (post - pre)
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, float]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "damage_extent": self.damage_extent,
            "damage_severity": self.damage_severity,
            "confidence_raster": self.confidence_raster,
            "ndvi_change": self.ndvi_change,
            "metadata": self.metadata,
            "statistics": self.statistics,
        }


class WindDamageDetection:
    """
    Wind Damage Vegetation Detection Algorithm.

    This algorithm detects vegetation damage from storm events by analyzing
    changes in vegetation indices (NDVI or EVI) between pre-event and post-event
    optical imagery. Significant decreases in vegetation indices indicate
    damage from high winds, which can strip leaves, break branches, or
    completely destroy trees and crops.

    Requirements:
        - Pre-event optical imagery (red + NIR bands, optionally blue for EVI)
        - Post-event optical imagery (same bands)

    Outputs:
        - damage_extent: Binary mask of damaged areas
        - damage_severity: Classified severity levels (minor/moderate/severe)
        - confidence_raster: Per-pixel confidence (0-1)
        - ndvi_change: Change in vegetation index
    """

    METADATA = {
        "id": "storm.baseline.wind_damage",
        "name": "Wind Damage Vegetation Detection",
        "category": "baseline",
        "event_types": ["storm.*", "hurricane.*", "cyclone.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "optical_pre": {
                    "bands": ["red", "nir"],
                    "temporal": "pre_event",
                },
                "optical_post": {
                    "bands": ["red", "nir"],
                    "temporal": "post_event",
                },
            },
            "optional": {
                "blue_band": {"benefit": "enables_evi_calculation"},
                "cloud_mask": {"benefit": "improves_accuracy"},
            },
            "compute": {"memory_gb": 2, "gpu": False},
        },
        "validation": {
            "accuracy_range": [0.72, 0.88],
            "validated_regions": ["caribbean", "southeast_asia"],
            "citations": ["doi:10.1016/j.rse.2018.05.003"],
        },
    }

    def __init__(self, config: Optional[WindDamageConfig] = None):
        """
        Initialize wind damage detection algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or WindDamageConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(
            f"Configuration: ndvi_change_threshold={self.config.ndvi_change_threshold}, "
            f"min_area={self.config.min_damage_area_ha}ha, "
            f"index={self.config.vegetation_index}"
        )

    def execute(
        self,
        red_pre: np.ndarray,
        nir_pre: np.ndarray,
        red_post: np.ndarray,
        nir_post: np.ndarray,
        blue_pre: Optional[np.ndarray] = None,
        blue_post: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
        pixel_size_m: float = 10.0,
        nodata_value: Optional[float] = None,
    ) -> WindDamageResult:
        """
        Execute wind damage vegetation detection.

        Args:
            red_pre: Pre-event red band reflectance (0-1), shape (H, W)
            nir_pre: Pre-event NIR band reflectance (0-1), shape (H, W)
            red_post: Post-event red band reflectance (0-1), shape (H, W)
            nir_post: Post-event NIR band reflectance (0-1), shape (H, W)
            blue_pre: Optional pre-event blue band for EVI (0-1), shape (H, W)
            blue_post: Optional post-event blue band for EVI (0-1), shape (H, W)
            cloud_mask: Optional cloud mask (True = cloudy/invalid), shape (H, W)
            pixel_size_m: Pixel size in meters (for area calculation)
            nodata_value: NoData value to mask out

        Returns:
            WindDamageResult containing damage extent, severity, and confidence
        """
        logger.info("Starting wind damage vegetation detection")

        # Validate inputs
        self._validate_inputs(
            red_pre, nir_pre, red_post, nir_post, blue_pre, blue_post
        )

        # Create valid data mask
        valid_mask = self._create_valid_mask(
            red_pre, nir_pre, red_post, nir_post, cloud_mask, nodata_value
        )

        # Calculate vegetation indices
        if self.config.vegetation_index == "evi" and blue_pre is not None and blue_post is not None:
            vi_pre = self._calculate_evi(red_pre, nir_pre, blue_pre)
            vi_post = self._calculate_evi(red_post, nir_post, blue_post)
            index_name = "EVI"
        else:
            vi_pre = self._calculate_ndvi(red_pre, nir_pre)
            vi_post = self._calculate_ndvi(red_post, nir_post)
            index_name = "NDVI"

        logger.info(f"Using {index_name} for vegetation change detection")

        # Calculate change (negative = vegetation loss)
        vi_change = vi_post - vi_pre

        # Identify vegetated areas in pre-event image
        vegetated_mask = vi_pre >= self.config.pre_event_ndvi_min

        # Detect damage: significant decrease in vegetated areas
        damage_extent = (
            (vi_change <= self.config.ndvi_change_threshold)
            & vegetated_mask
            & valid_mask
        )

        # Calculate severity classification
        if self.config.use_severity_classification:
            damage_severity = self._classify_severity(vi_change, valid_mask, vegetated_mask)
        else:
            damage_severity = damage_extent.astype(np.uint8)

        # Calculate confidence
        confidence = self._calculate_confidence(
            vi_change, vi_pre, valid_mask, vegetated_mask
        )

        # Calculate statistics
        pixel_area_ha = (pixel_size_m**2) / 10000.0
        damage_pixels = int(np.sum(damage_extent))
        damage_area_ha = damage_pixels * pixel_area_ha
        vegetated_pixels = int(np.sum(vegetated_mask & valid_mask))

        # Count pixels by severity
        severity_counts = {
            "none": int(np.sum(damage_severity == 0)),
            "minor": int(np.sum(damage_severity == 1)),
            "moderate": int(np.sum(damage_severity == 2)),
            "severe": int(np.sum(damage_severity == 3)),
        }

        statistics = {
            "total_pixels": int(damage_extent.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "vegetated_pixels": vegetated_pixels,
            "damage_pixels": damage_pixels,
            "damage_area_ha": float(damage_area_ha),
            "damage_percent_of_vegetated": (
                float(100.0 * damage_pixels / vegetated_pixels)
                if vegetated_pixels > 0
                else 0.0
            ),
            "mean_vi_change": (
                float(np.nanmean(vi_change[valid_mask]))
                if np.any(valid_mask)
                else 0.0
            ),
            "mean_damage_confidence": (
                float(np.mean(confidence[damage_extent]))
                if damage_pixels > 0
                else 0.0
            ),
            "severity_counts": severity_counts,
            "minor_damage_ha": float(severity_counts["minor"] * pixel_area_ha),
            "moderate_damage_ha": float(severity_counts["moderate"] * pixel_area_ha),
            "severe_damage_ha": float(severity_counts["severe"] * pixel_area_ha),
        }

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "ndvi_change_threshold": self.config.ndvi_change_threshold,
                "min_damage_area_ha": self.config.min_damage_area_ha,
                "vegetation_index": self.config.vegetation_index,
                "pre_event_ndvi_min": self.config.pre_event_ndvi_min,
                "use_severity_classification": self.config.use_severity_classification,
                "pixel_size_m": pixel_size_m,
            },
            "execution": {
                "index_used": index_name,
                "evi_available": blue_pre is not None and blue_post is not None,
            },
        }

        logger.info(
            f"Detection complete: {damage_area_ha:.2f} ha damage extent "
            f"({statistics['damage_percent_of_vegetated']:.1f}% of vegetated area)"
        )
        logger.info(
            f"Severity breakdown: minor={severity_counts['minor']}, "
            f"moderate={severity_counts['moderate']}, severe={severity_counts['severe']} pixels"
        )

        return WindDamageResult(
            damage_extent=damage_extent,
            damage_severity=damage_severity,
            confidence_raster=confidence,
            ndvi_change=vi_change,
            metadata=metadata,
            statistics=statistics,
        )

    def _validate_inputs(
        self,
        red_pre: np.ndarray,
        nir_pre: np.ndarray,
        red_post: np.ndarray,
        nir_post: np.ndarray,
        blue_pre: Optional[np.ndarray],
        blue_post: Optional[np.ndarray],
    ) -> None:
        """Validate input array shapes and dimensions."""
        if red_pre.ndim != 2:
            raise ValueError(f"Expected 2D red_pre array, got shape {red_pre.shape}")

        shape = red_pre.shape
        for name, arr in [
            ("nir_pre", nir_pre),
            ("red_post", red_post),
            ("nir_post", nir_post),
        ]:
            if arr.shape != shape:
                raise ValueError(
                    f"{name} shape {arr.shape} doesn't match red_pre shape {shape}"
                )

        if blue_pre is not None and blue_pre.shape != shape:
            raise ValueError(
                f"blue_pre shape {blue_pre.shape} doesn't match expected {shape}"
            )
        if blue_post is not None and blue_post.shape != shape:
            raise ValueError(
                f"blue_post shape {blue_post.shape} doesn't match expected {shape}"
            )

    def _create_valid_mask(
        self,
        red_pre: np.ndarray,
        nir_pre: np.ndarray,
        red_post: np.ndarray,
        nir_post: np.ndarray,
        cloud_mask: Optional[np.ndarray],
        nodata_value: Optional[float],
    ) -> np.ndarray:
        """Create combined valid data mask."""
        valid_mask = np.ones_like(red_pre, dtype=bool)

        # Check for nodata
        if nodata_value is not None:
            valid_mask &= red_pre != nodata_value
            valid_mask &= nir_pre != nodata_value
            valid_mask &= red_post != nodata_value
            valid_mask &= nir_post != nodata_value

        # Check for finite values
        valid_mask &= np.isfinite(red_pre)
        valid_mask &= np.isfinite(nir_pre)
        valid_mask &= np.isfinite(red_post)
        valid_mask &= np.isfinite(nir_post)

        # Apply cloud mask if provided
        if cloud_mask is not None:
            valid_mask &= ~cloud_mask

        return valid_mask

    def _calculate_ndvi(self, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index.

        NDVI = (NIR - Red) / (NIR + Red)

        Args:
            red: Red band reflectance
            nir: NIR band reflectance

        Returns:
            NDVI values (-1 to 1)
        """
        denominator = nir + red
        ndvi = np.zeros_like(red, dtype=np.float32)

        # Avoid division by zero
        valid = denominator != 0
        ndvi[valid] = (nir[valid] - red[valid]) / denominator[valid]

        return np.clip(ndvi, -1.0, 1.0)

    def _calculate_evi(
        self, red: np.ndarray, nir: np.ndarray, blue: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Enhanced Vegetation Index.

        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

        Args:
            red: Red band reflectance
            nir: NIR band reflectance
            blue: Blue band reflectance

        Returns:
            EVI values (approximately -1 to 1)
        """
        denominator = (
            nir
            + EVI_COEFF_RED * red
            - EVI_COEFF_BLUE * blue
            + EVI_OFFSET
        )
        evi = np.zeros_like(red, dtype=np.float32)

        # Avoid division by zero
        valid = denominator != 0
        evi[valid] = EVI_SCALE_FACTOR * (nir[valid] - red[valid]) / denominator[valid]

        return np.clip(evi, -1.0, 1.0)

    def _classify_severity(
        self,
        vi_change: np.ndarray,
        valid_mask: np.ndarray,
        vegetated_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Classify damage severity based on vegetation index change.

        Severity levels:
            0 = No damage (change > minor threshold or not vegetated)
            1 = Minor damage (change between minor and moderate thresholds)
            2 = Moderate damage (change between moderate and severe thresholds)
            3 = Severe damage (change below severe threshold)

        Args:
            vi_change: Vegetation index change (post - pre)
            valid_mask: Valid data mask
            vegetated_mask: Pre-event vegetated area mask

        Returns:
            Severity classification array (uint8)
        """
        severity = np.zeros_like(vi_change, dtype=np.uint8)

        # Only classify vegetated valid areas
        analysis_mask = valid_mask & vegetated_mask

        # Minor damage: -0.1 to -0.2 change
        minor_mask = (
            (vi_change <= NDVI_CHANGE_MINOR)
            & (vi_change > NDVI_CHANGE_MODERATE)
            & analysis_mask
        )
        severity[minor_mask] = 1

        # Moderate damage: -0.2 to -0.4 change
        moderate_mask = (
            (vi_change <= NDVI_CHANGE_MODERATE)
            & (vi_change > NDVI_CHANGE_SEVERE)
            & analysis_mask
        )
        severity[moderate_mask] = 2

        # Severe damage: < -0.4 change
        severe_mask = (vi_change <= NDVI_CHANGE_SEVERE) & analysis_mask
        severity[severe_mask] = 3

        return severity

    def _calculate_confidence(
        self,
        vi_change: np.ndarray,
        vi_pre: np.ndarray,
        valid_mask: np.ndarray,
        vegetated_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate per-pixel confidence scores.

        Confidence is based on:
        1. Magnitude of change (larger decrease = higher confidence)
        2. Pre-event vegetation density (denser vegetation = higher confidence)

        Args:
            vi_change: Vegetation index change
            vi_pre: Pre-event vegetation index
            valid_mask: Valid data mask
            vegetated_mask: Pre-event vegetated area mask

        Returns:
            Confidence scores (0-1)
        """
        confidence = np.zeros_like(vi_change, dtype=np.float32)

        analysis_mask = valid_mask & vegetated_mask

        # Change magnitude confidence (normalized by threshold)
        # More negative change = higher confidence
        # Guard against division by zero when threshold is 0
        threshold_divisor = abs(self.config.ndvi_change_threshold * 2)
        if threshold_divisor < 1e-10:
            threshold_divisor = 0.2  # Use default threshold for normalization
        change_conf = np.clip(
            -vi_change / threshold_divisor,
            0.0,
            1.0,
        )

        # Pre-event vegetation confidence (denser = more confident)
        # Normalized to [0, 1] based on NDVI range
        veg_conf = np.clip((vi_pre - NDVI_SPARSE_MIN) / (1.0 - NDVI_SPARSE_MIN), 0.0, 1.0)

        # Combine: weighted average (change is more important)
        confidence[analysis_mask] = (
            0.7 * change_conf[analysis_mask] + 0.3 * veg_conf[analysis_mask]
        )

        return confidence

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return WindDamageDetection.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> "WindDamageDetection":
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = WindDamageConfig(**params)
        return WindDamageDetection(config)
