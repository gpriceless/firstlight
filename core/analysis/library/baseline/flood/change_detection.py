"""
Pre/Post Change Detection Flood Algorithm

Detects flood extent by comparing pre-event and post-event imagery
to identify significant changes indicative of flooding.

Algorithm ID: flood.baseline.change_detection
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChangeDetectionConfig:
    """Configuration for pre/post change detection."""

    change_threshold: float = 0.15  # Change magnitude threshold
    method: Literal["ratio", "difference", "normalized_difference"] = "normalized_difference"
    min_area_ha: float = 0.5  # Minimum flood polygon area (hectares)
    use_multiple_bands: bool = True  # Use multiple bands for robust detection
    outlier_removal: bool = True  # Remove statistical outliers
    outlier_sigma: float = 3.0  # Sigma threshold for outlier removal

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.change_threshold < 0:
            raise ValueError(f"change_threshold must be non-negative, got {self.change_threshold}")
        if self.min_area_ha < 0:
            raise ValueError(f"min_area_ha must be non-negative, got {self.min_area_ha}")
        if self.method not in ["ratio", "difference", "normalized_difference"]:
            raise ValueError(f"Invalid method: {self.method}")


@dataclass
class ChangeDetectionResult:
    """Results from change detection flood mapping."""

    flood_extent: np.ndarray  # Binary mask of flood extent
    confidence_raster: np.ndarray  # Confidence scores (0-1)
    change_magnitude: np.ndarray  # Change magnitude values
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, float]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "flood_extent": self.flood_extent,
            "confidence_raster": self.confidence_raster,
            "change_magnitude": self.change_magnitude,
            "metadata": self.metadata,
            "statistics": self.statistics
        }


class ChangeDetectionAlgorithm:
    """
    Pre/Post Change Detection Flood Mapping.

    Identifies flood extent by comparing pre-event and post-event imagery.
    Supports multiple change detection methods:
    - Ratio: post / pre (for SAR, decrease indicates flooding)
    - Difference: post - pre (for optical NDVI, decrease indicates flooding)
    - Normalized Difference: (post - pre) / (post + pre)

    The algorithm is sensor-agnostic and works with:
    - SAR backscatter (VV, VH)
    - Optical indices (NDVI, NDWI)
    - Raw optical bands

    Requirements:
        - Pre-event imagery
        - Post-event imagery
        - Matching spatial resolution and coverage

    Outputs:
        - flood_extent: Binary vector polygon of flood extent
        - confidence_raster: Per-pixel confidence (0-1)
        - change_magnitude: Raw change values for analysis
    """

    METADATA = {
        "id": "flood.baseline.change_detection",
        "name": "Pre/Post Change Detection",
        "category": "baseline",
        "event_types": ["flood.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "pre_event": {"temporal": "pre_event", "any_sensor": True},
                "post_event": {"temporal": "post_event", "any_sensor": True}
            },
            "compute": {"memory_gb": 6, "gpu": False}
        },
        "validation": {
            "accuracy_range": [0.72, 0.88],
            "validated_regions": ["global"],
            "citations": ["doi:10.1016/j.rse.2018.05.032"]
        }
    }

    def __init__(self, config: Optional[ChangeDetectionConfig] = None):
        """
        Initialize change detection algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or ChangeDetectionConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(f"Configuration: method={self.config.method}, "
                   f"threshold={self.config.change_threshold}")

    def execute(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        pixel_size_m: float = 10.0,
        nodata_value: Optional[float] = None,
        sensor_type: Literal["sar", "optical"] = "sar"
    ) -> ChangeDetectionResult:
        """
        Execute pre/post change detection.

        Args:
            pre_image: Pre-event imagery, shape (H, W) or (C, H, W)
            post_image: Post-event imagery, shape (H, W) or (C, H, W)
            pixel_size_m: Pixel size in meters (for area calculation)
            nodata_value: NoData value to mask out
            sensor_type: Type of sensor ("sar" or "optical") for interpretation

        Returns:
            ChangeDetectionResult containing flood extent and confidence
        """
        logger.info(f"Starting change detection with {self.config.method} method")
        logger.info(f"Sensor type: {sensor_type}")

        # Validate inputs
        if pre_image.shape != post_image.shape:
            raise ValueError(f"Pre/post shape mismatch: {pre_image.shape} vs {post_image.shape}")

        # Handle multi-band imagery
        if pre_image.ndim == 3:
            logger.info(f"Multi-band input detected: {pre_image.shape[0]} bands")
            if self.config.use_multiple_bands:
                change_magnitude, valid_mask = self._detect_multiband(
                    pre_image, post_image, nodata_value, sensor_type
                )
            else:
                # Use first band only
                change_magnitude, valid_mask = self._detect_singleband(
                    pre_image[0], post_image[0], nodata_value, sensor_type
                )
        else:
            change_magnitude, valid_mask = self._detect_singleband(
                pre_image, post_image, nodata_value, sensor_type
            )

        # Remove outliers if enabled
        if self.config.outlier_removal:
            change_magnitude = self._remove_outliers(change_magnitude, valid_mask)

        # Detect flood extent
        flood_extent, confidence = self._threshold_change(
            change_magnitude, valid_mask, sensor_type
        )

        # Calculate statistics
        pixel_area_ha = (pixel_size_m ** 2) / 10000.0  # m² to hectares
        flood_pixels = np.sum(flood_extent)
        flood_area_ha = flood_pixels * pixel_area_ha

        statistics = {
            "total_pixels": int(change_magnitude.size),
            "valid_pixels": int(np.sum(valid_mask)),
            "flood_pixels": int(flood_pixels),
            "flood_area_ha": float(flood_area_ha),
            "flood_percent": float(100.0 * flood_pixels / np.sum(valid_mask)) if np.sum(valid_mask) > 0 else 0.0,
            "mean_change": float(np.mean(change_magnitude[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "std_change": float(np.std(change_magnitude[valid_mask])) if np.sum(valid_mask) > 0 else 0.0,
            "mean_change_flood": float(np.mean(change_magnitude[flood_extent])) if flood_pixels > 0 else 0.0,
            "mean_confidence": float(np.mean(confidence[flood_extent])) if flood_pixels > 0 else 0.0
        }

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "change_threshold": self.config.change_threshold,
                "method": self.config.method,
                "min_area_ha": self.config.min_area_ha,
                "use_multiple_bands": self.config.use_multiple_bands,
                "outlier_removal": self.config.outlier_removal,
                "pixel_size_m": pixel_size_m,
                "sensor_type": sensor_type
            },
            "execution": {
                "bands_used": pre_image.shape[0] if pre_image.ndim == 3 else 1
            }
        }

        logger.info(f"Detection complete: {flood_area_ha:.2f} ha flood extent "
                   f"({statistics['flood_percent']:.1f}% of valid area)")
        logger.info(f"Mean change: {statistics['mean_change']:.3f} ± {statistics['std_change']:.3f}")

        return ChangeDetectionResult(
            flood_extent=flood_extent,
            confidence_raster=confidence,
            change_magnitude=change_magnitude,
            metadata=metadata,
            statistics=statistics
        )

    def _detect_singleband(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        nodata_value: Optional[float],
        sensor_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Single-band change detection."""
        # Create valid mask
        valid_mask = np.ones_like(pre, dtype=bool)
        if nodata_value is not None:
            valid_mask &= (pre != nodata_value) & (post != nodata_value)
        valid_mask &= np.isfinite(pre) & np.isfinite(post)

        # Calculate change
        change = self._calculate_change(pre, post, valid_mask)

        return change, valid_mask

    def _detect_multiband(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        nodata_value: Optional[float],
        sensor_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-band change detection (average across bands)."""
        n_bands = pre.shape[0]
        changes = []
        valid_masks = []

        for i in range(n_bands):
            change, valid_mask = self._detect_singleband(
                pre[i], post[i], nodata_value, sensor_type
            )
            changes.append(change)
            valid_masks.append(valid_mask)

        # Combine bands: average change, intersect valid masks
        change_combined = np.mean(changes, axis=0)
        valid_mask_combined = np.all(valid_masks, axis=0)

        logger.info(f"Combined {n_bands} bands for change detection")

        return change_combined, valid_mask_combined

    def _calculate_change(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Calculate change magnitude based on configured method.

        Args:
            pre: Pre-event values
            post: Post-event values
            valid_mask: Valid data mask

        Returns:
            Change magnitude array
        """
        change = np.zeros_like(pre, dtype=np.float32)

        if self.config.method == "difference":
            # Simple difference: post - pre
            change[valid_mask] = post[valid_mask] - pre[valid_mask]

        elif self.config.method == "ratio":
            # Ratio: post / pre (avoid division by zero)
            safe_pre = np.where(np.abs(pre) < 1e-6, 1e-6, pre)
            change[valid_mask] = post[valid_mask] / safe_pre[valid_mask]

        elif self.config.method == "normalized_difference":
            # Normalized difference: (post - pre) / (post + pre)
            denominator = np.abs(post) + np.abs(pre)
            safe_calc = valid_mask & (denominator > 1e-6)
            change[safe_calc] = (post[safe_calc] - pre[safe_calc]) / denominator[safe_calc]

        return change

    def _remove_outliers(
        self,
        change: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """Remove statistical outliers using sigma clipping."""
        if np.sum(valid_mask) == 0:
            return change

        valid_changes = change[valid_mask]
        mean_change = np.mean(valid_changes)
        std_change = np.std(valid_changes)

        lower_bound = mean_change - self.config.outlier_sigma * std_change
        upper_bound = mean_change + self.config.outlier_sigma * std_change

        # Clip outliers to bounds
        change_cleaned = np.copy(change)
        change_cleaned = np.clip(change_cleaned, lower_bound, upper_bound)

        n_outliers = np.sum((change < lower_bound) | (change > upper_bound))
        if n_outliers > 0:
            logger.info(f"Removed {n_outliers} outliers ({100.0 * n_outliers / change.size:.2f}%)")

        return change_cleaned

    def _threshold_change(
        self,
        change: np.ndarray,
        valid_mask: np.ndarray,
        sensor_type: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Threshold change magnitude to detect flood extent.

        For SAR: decrease (negative change) indicates flooding
        For optical: depends on index (NDVI decrease, NDWI increase)

        Args:
            change: Change magnitude
            valid_mask: Valid data mask
            sensor_type: Sensor type for interpretation

        Returns:
            Tuple of (flood_extent, confidence)
        """
        if sensor_type == "sar":
            # SAR: flooding causes backscatter decrease
            if self.config.method == "ratio":
                # Ratio < 1 indicates decrease
                flood_extent = (change < (1.0 - self.config.change_threshold)) & valid_mask
                confidence = np.clip((1.0 - change) / self.config.change_threshold, 0.0, 1.0)
            else:
                # Difference/normalized difference: negative = decrease
                flood_extent = (change < -self.config.change_threshold) & valid_mask
                confidence = np.clip(-change / self.config.change_threshold, 0.0, 1.0)
        else:
            # Optical: depends on index, default to magnitude of change
            flood_extent = (np.abs(change) > self.config.change_threshold) & valid_mask
            confidence = np.clip(np.abs(change) / self.config.change_threshold, 0.0, 1.0)

        confidence[~valid_mask] = 0.0

        return flood_extent, confidence

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return ChangeDetectionAlgorithm.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> 'ChangeDetectionAlgorithm':
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = ChangeDetectionConfig(**params)
        return ChangeDetectionAlgorithm(config)
