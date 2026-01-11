"""
Optical Imagery Quality Assessment.

Evaluates the quality of optical satellite imagery including:
- Cloud cover percentage and distribution
- Cloud shadow detection
- Haze/aerosol assessment
- Sun glint detection (for water bodies)
- Saturation check (overexposed pixels)
- Overall optical quality score (0-1)

The assessor uses spectral properties of imagery to detect quality issues
and provides actionable metrics for decision-making.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class CloudDetectionMethod(Enum):
    """Methods for cloud detection."""
    THRESHOLD = "threshold"          # Simple brightness threshold
    SPECTRAL = "spectral"            # Multi-band spectral analysis
    MACHINE_LEARNING = "ml"          # ML-based detection
    METADATA = "metadata"            # Use provided metadata
    COMBINED = "combined"            # Combination of methods


class OpticalIssueSeverity(Enum):
    """Severity levels for optical quality issues."""
    CRITICAL = "critical"   # Image unusable
    HIGH = "high"           # Significant quality degradation
    MEDIUM = "medium"       # Moderate impact on analysis
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational only


@dataclass
class OpticalQualityIssue:
    """
    An optical quality issue found during assessment.

    Attributes:
        issue_type: Type of quality issue
        severity: Issue severity level
        description: Human-readable description
        affected_area_percent: Percentage of image affected
        location: Affected region (bbox or mask)
        recommendation: Suggested action
    """
    issue_type: str
    severity: OpticalIssueSeverity
    description: str
    affected_area_percent: float = 0.0
    location: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "description": self.description,
            "affected_area_percent": self.affected_area_percent,
            "location": self.location,
            "recommendation": self.recommendation,
        }


@dataclass
class OpticalQualityConfig:
    """
    Configuration for optical quality assessment.

    Attributes:
        cloud_detection_method: Method for detecting clouds
        cloud_threshold_percent: Max acceptable cloud cover percentage
        shadow_threshold_percent: Max acceptable shadow percentage
        haze_threshold: Haze detection threshold (0-1)
        glint_threshold: Sun glint detection threshold (0-1)
        saturation_threshold_percent: Max acceptable saturation percentage

        check_clouds: Enable cloud detection
        check_shadows: Enable shadow detection
        check_haze: Enable haze assessment
        check_glint: Enable sun glint detection
        check_saturation: Enable saturation check
    """
    cloud_detection_method: CloudDetectionMethod = CloudDetectionMethod.COMBINED
    cloud_threshold_percent: float = 20.0
    shadow_threshold_percent: float = 10.0
    haze_threshold: float = 0.3
    glint_threshold: float = 0.5
    saturation_threshold_percent: float = 5.0

    check_clouds: bool = True
    check_shadows: bool = True
    check_haze: bool = True
    check_glint: bool = True
    check_saturation: bool = True

    # Band indices for different sensors (0-indexed)
    blue_band: int = 1
    green_band: int = 2
    red_band: int = 3
    nir_band: int = 7
    swir_band: int = 10

    # Thresholds for spectral detection
    cloud_brightness_threshold: float = 0.3
    cloud_blue_threshold: float = 0.2
    shadow_brightness_threshold: float = 0.08
    glint_nir_threshold: float = 0.3


@dataclass
class OpticalQualityResult:
    """
    Result of optical quality assessment.

    Attributes:
        overall_score: Overall quality score (0-1)
        is_usable: Whether the image is usable for analysis
        cloud_cover_percent: Detected cloud cover percentage
        cloud_shadow_percent: Detected cloud shadow percentage
        haze_score: Haze severity score (0-1)
        sun_glint_percent: Sun glint affected percentage
        saturation_percent: Saturated pixel percentage
        clear_pixel_percent: Clear usable pixel percentage
        issues: List of quality issues found
        metrics: Detailed quality metrics
        duration_seconds: Assessment duration
    """
    overall_score: float
    is_usable: bool
    cloud_cover_percent: float
    cloud_shadow_percent: float
    haze_score: float
    sun_glint_percent: float
    saturation_percent: float
    clear_pixel_percent: float
    issues: List[OpticalQualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == OpticalIssueSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 3),
            "is_usable": self.is_usable,
            "cloud_cover_percent": round(self.cloud_cover_percent, 2),
            "cloud_shadow_percent": round(self.cloud_shadow_percent, 2),
            "haze_score": round(self.haze_score, 3),
            "sun_glint_percent": round(self.sun_glint_percent, 2),
            "saturation_percent": round(self.saturation_percent, 2),
            "clear_pixel_percent": round(self.clear_pixel_percent, 2),
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class OpticalQualityAssessor:
    """
    Assesses optical imagery quality for analysis suitability.

    Evaluates cloud cover, shadows, haze, sun glint, and saturation
    to produce an overall quality score and actionable recommendations.

    Example:
        assessor = OpticalQualityAssessor()
        result = assessor.assess(
            image_data=multispectral_array,
            metadata={"cloud_cover": 15.0}
        )
        if not result.is_usable:
            print(f"Image unusable: {result.issues[0].description}")
    """

    def __init__(self, config: Optional[OpticalQualityConfig] = None):
        """
        Initialize the optical quality assessor.

        Args:
            config: Configuration options
        """
        self.config = config or OpticalQualityConfig()

    def assess(
        self,
        image_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        valid_mask: Optional[np.ndarray] = None,
        water_mask: Optional[np.ndarray] = None,
    ) -> OpticalQualityResult:
        """
        Assess optical image quality.

        Args:
            image_data: Multi-band image array (bands, height, width) or (height, width)
            metadata: Optional image metadata including existing cloud cover
            valid_mask: Optional mask of valid data pixels
            water_mask: Optional water body mask for glint detection

        Returns:
            OpticalQualityResult with detailed assessment
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}

        # Validate input
        if image_data.size == 0:
            return OpticalQualityResult(
                overall_score=0.0,
                is_usable=False,
                cloud_cover_percent=100.0,
                cloud_shadow_percent=0.0,
                haze_score=1.0,
                sun_glint_percent=0.0,
                saturation_percent=0.0,
                clear_pixel_percent=0.0,
                issues=[OpticalQualityIssue(
                    issue_type="empty_image",
                    severity=OpticalIssueSeverity.CRITICAL,
                    description="Image data is empty",
                    recommendation="Verify data acquisition and loading"
                )],
                duration_seconds=time.time() - start_time
            )

        # Ensure 3D array (bands, height, width)
        if image_data.ndim == 2:
            image_data = image_data[np.newaxis, :, :]

        # Apply valid mask
        if valid_mask is None:
            valid_mask = ~np.any(np.isnan(image_data), axis=0)

        num_valid_pixels = valid_mask.sum()
        if num_valid_pixels == 0:
            return OpticalQualityResult(
                overall_score=0.0,
                is_usable=False,
                cloud_cover_percent=100.0,
                cloud_shadow_percent=0.0,
                haze_score=1.0,
                sun_glint_percent=0.0,
                saturation_percent=0.0,
                clear_pixel_percent=0.0,
                issues=[OpticalQualityIssue(
                    issue_type="no_valid_pixels",
                    severity=OpticalIssueSeverity.CRITICAL,
                    description="No valid pixels in image",
                    recommendation="Check data source and acquisition parameters"
                )],
                duration_seconds=time.time() - start_time
            )

        metrics["num_valid_pixels"] = int(num_valid_pixels)
        metrics["total_pixels"] = int(image_data.shape[1] * image_data.shape[2])

        # Initialize masks
        cloud_mask = np.zeros_like(valid_mask, dtype=bool)
        shadow_mask = np.zeros_like(valid_mask, dtype=bool)
        glint_mask = np.zeros_like(valid_mask, dtype=bool)
        saturation_mask = np.zeros_like(valid_mask, dtype=bool)
        haze_score = 0.0

        metadata = metadata or {}

        # Cloud detection
        if self.config.check_clouds:
            cloud_mask, cloud_metrics = self._detect_clouds(
                image_data, valid_mask, metadata
            )
            metrics["cloud"] = cloud_metrics

        # Shadow detection
        if self.config.check_shadows:
            shadow_mask, shadow_metrics = self._detect_shadows(
                image_data, valid_mask, cloud_mask
            )
            metrics["shadow"] = shadow_metrics

        # Haze assessment
        if self.config.check_haze:
            haze_score, haze_metrics = self._assess_haze(image_data, valid_mask)
            metrics["haze"] = haze_metrics

        # Sun glint detection
        if self.config.check_glint and water_mask is not None:
            glint_mask, glint_metrics = self._detect_sun_glint(
                image_data, valid_mask, water_mask
            )
            metrics["glint"] = glint_metrics

        # Saturation check
        if self.config.check_saturation:
            saturation_mask, saturation_metrics = self._detect_saturation(
                image_data, valid_mask
            )
            metrics["saturation"] = saturation_metrics

        # Calculate percentages
        cloud_percent = 100.0 * cloud_mask.sum() / num_valid_pixels
        shadow_percent = 100.0 * shadow_mask.sum() / num_valid_pixels
        glint_percent = 100.0 * glint_mask.sum() / num_valid_pixels if water_mask is not None else 0.0
        saturation_percent = 100.0 * saturation_mask.sum() / num_valid_pixels

        # Calculate clear pixels
        affected_mask = cloud_mask | shadow_mask | glint_mask | saturation_mask
        clear_pixels = valid_mask & ~affected_mask
        clear_percent = 100.0 * clear_pixels.sum() / num_valid_pixels

        # Generate issues
        if cloud_percent > self.config.cloud_threshold_percent:
            severity = (
                OpticalIssueSeverity.CRITICAL if cloud_percent > 80
                else OpticalIssueSeverity.HIGH if cloud_percent > 50
                else OpticalIssueSeverity.MEDIUM
            )
            issues.append(OpticalQualityIssue(
                issue_type="excessive_cloud_cover",
                severity=severity,
                description=f"Cloud cover {cloud_percent:.1f}% exceeds threshold {self.config.cloud_threshold_percent}%",
                affected_area_percent=cloud_percent,
                recommendation="Consider SAR alternative or cloud-free composite"
            ))

        if shadow_percent > self.config.shadow_threshold_percent:
            issues.append(OpticalQualityIssue(
                issue_type="excessive_shadow",
                severity=OpticalIssueSeverity.MEDIUM,
                description=f"Shadow coverage {shadow_percent:.1f}% exceeds threshold",
                affected_area_percent=shadow_percent,
                recommendation="Apply shadow correction or use alternative acquisition"
            ))

        if haze_score > self.config.haze_threshold:
            issues.append(OpticalQualityIssue(
                issue_type="high_haze",
                severity=OpticalIssueSeverity.MEDIUM,
                description=f"Haze score {haze_score:.2f} indicates atmospheric interference",
                recommendation="Apply atmospheric correction or use clearer acquisition"
            ))

        if glint_percent > 10.0:
            issues.append(OpticalQualityIssue(
                issue_type="sun_glint",
                severity=OpticalIssueSeverity.LOW,
                description=f"Sun glint affects {glint_percent:.1f}% of water areas",
                affected_area_percent=glint_percent,
                recommendation="Apply sun glint correction"
            ))

        if saturation_percent > self.config.saturation_threshold_percent:
            issues.append(OpticalQualityIssue(
                issue_type="saturation",
                severity=OpticalIssueSeverity.MEDIUM,
                description=f"Saturation affects {saturation_percent:.1f}% of pixels",
                affected_area_percent=saturation_percent,
                recommendation="Check sensor calibration or use HDR processing"
            ))

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            cloud_percent,
            shadow_percent,
            haze_score,
            glint_percent,
            saturation_percent,
            clear_percent
        )

        # Determine usability
        is_usable = (
            overall_score >= 0.3 and
            cloud_percent <= 80.0 and
            clear_percent >= 20.0
        )

        duration = time.time() - start_time
        logger.info(
            f"Optical quality assessment: score={overall_score:.2f}, "
            f"cloud={cloud_percent:.1f}%, clear={clear_percent:.1f}%"
        )

        return OpticalQualityResult(
            overall_score=overall_score,
            is_usable=is_usable,
            cloud_cover_percent=cloud_percent,
            cloud_shadow_percent=shadow_percent,
            haze_score=haze_score,
            sun_glint_percent=glint_percent,
            saturation_percent=saturation_percent,
            clear_pixel_percent=clear_percent,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _detect_clouds(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect clouds in optical imagery.

        Uses multiple methods based on configuration:
        - Metadata: Use provided cloud cover percentage
        - Threshold: Simple brightness thresholding
        - Spectral: Multi-band analysis for better accuracy
        - Combined: Weighted combination of methods
        """
        metrics = {}
        cloud_mask = np.zeros_like(valid_mask, dtype=bool)
        method = self.config.cloud_detection_method

        # Try metadata first if available
        if method in [CloudDetectionMethod.METADATA, CloudDetectionMethod.COMBINED]:
            if "cloud_cover" in metadata:
                metrics["metadata_cloud_cover"] = metadata["cloud_cover"]
                # Create uniform mask based on metadata percentage
                if metadata["cloud_cover"] > 0:
                    # For metadata-only, we estimate but can't localize
                    pass

        # Brightness threshold method
        if method in [CloudDetectionMethod.THRESHOLD, CloudDetectionMethod.COMBINED]:
            brightness_mask = self._cloud_brightness_threshold(image_data, valid_mask)
            metrics["brightness_cloud_percent"] = 100.0 * brightness_mask.sum() / valid_mask.sum()
            cloud_mask |= brightness_mask

        # Spectral method (requires multiple bands)
        if method in [CloudDetectionMethod.SPECTRAL, CloudDetectionMethod.COMBINED]:
            if image_data.shape[0] >= 4:
                spectral_mask = self._cloud_spectral_detection(image_data, valid_mask)
                metrics["spectral_cloud_percent"] = 100.0 * spectral_mask.sum() / valid_mask.sum()

                if method == CloudDetectionMethod.COMBINED:
                    # Combine with AND logic (more conservative)
                    cloud_mask = cloud_mask | spectral_mask
                else:
                    cloud_mask = spectral_mask

        return cloud_mask, metrics

    def _cloud_brightness_threshold(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """Simple cloud detection using brightness threshold."""
        # Calculate brightness (average across visible bands)
        num_bands = min(3, image_data.shape[0])
        brightness = np.nanmean(image_data[:num_bands, :, :], axis=0)

        # Normalize to 0-1 range
        valid_values = brightness[valid_mask]
        if len(valid_values) == 0:
            return np.zeros_like(valid_mask, dtype=bool)

        # Use percentile-based normalization
        p2, p98 = np.nanpercentile(valid_values, [2, 98])
        if p98 - p2 > 1e-6:
            brightness_norm = (brightness - p2) / (p98 - p2)
        else:
            brightness_norm = brightness

        # Clouds are bright
        cloud_mask = (brightness_norm > self.config.cloud_brightness_threshold) & valid_mask

        return cloud_mask

    def _cloud_spectral_detection(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """
        Spectral cloud detection using blue band and NIR.

        Clouds are bright in visible and have high reflectance in blue.
        """
        cloud_mask = np.zeros_like(valid_mask, dtype=bool)

        try:
            # Get bands (handle varying band configurations)
            blue_idx = min(self.config.blue_band, image_data.shape[0] - 1)
            nir_idx = min(self.config.nir_band, image_data.shape[0] - 1)

            blue = image_data[blue_idx, :, :]
            nir = image_data[nir_idx, :, :]

            # Normalize bands
            def normalize_band(band: np.ndarray) -> np.ndarray:
                valid_vals = band[valid_mask & ~np.isnan(band)]
                if len(valid_vals) == 0:
                    return band
                p2, p98 = np.nanpercentile(valid_vals, [2, 98])
                if p98 - p2 > 1e-6:
                    return np.clip((band - p2) / (p98 - p2), 0, 1)
                return band

            blue_norm = normalize_band(blue)
            nir_norm = normalize_band(nir)

            # Clouds: high blue reflectance and moderate-high NIR
            cloud_mask = (
                valid_mask &
                (blue_norm > self.config.cloud_blue_threshold) &
                (nir_norm > 0.2)
            )

        except Exception as e:
            logger.warning(f"Spectral cloud detection failed: {e}")

        return cloud_mask

    def _detect_shadows(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray,
        cloud_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect cloud shadows.

        Shadows are dark areas adjacent to (or offset from) clouds.
        """
        metrics = {}

        # Calculate brightness
        num_bands = min(3, image_data.shape[0])
        brightness = np.nanmean(image_data[:num_bands, :, :], axis=0)

        # Normalize
        valid_values = brightness[valid_mask & ~cloud_mask]
        if len(valid_values) == 0:
            return np.zeros_like(valid_mask, dtype=bool), metrics

        p2, p98 = np.nanpercentile(valid_values, [2, 98])
        if p98 - p2 > 1e-6:
            brightness_norm = (brightness - p2) / (p98 - p2)
        else:
            brightness_norm = brightness

        # Dark pixels (potential shadows)
        dark_mask = (brightness_norm < self.config.shadow_brightness_threshold) & valid_mask

        # Shadow must not be cloud
        shadow_mask = dark_mask & ~cloud_mask

        # Additional validation: shadows should be near clouds
        if cloud_mask.any():
            try:
                from scipy import ndimage
                # Dilate cloud mask to find potential shadow zones
                cloud_buffer = ndimage.binary_dilation(
                    cloud_mask,
                    iterations=20
                )
                # Only keep shadows in buffer zone
                # shadow_mask = shadow_mask & cloud_buffer
                # Actually, shadows can be offset significantly, so keep all dark pixels
            except ImportError:
                pass

        metrics["dark_pixel_percent"] = 100.0 * dark_mask.sum() / valid_mask.sum()

        return shadow_mask, metrics

    def _assess_haze(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Assess atmospheric haze level.

        Haze reduces contrast and adds a uniform blue tint.
        """
        metrics = {}

        try:
            # Get blue and red bands
            blue_idx = min(self.config.blue_band, image_data.shape[0] - 1)
            red_idx = min(self.config.red_band, image_data.shape[0] - 1)

            blue = image_data[blue_idx, :, :]
            red = image_data[red_idx, :, :]

            # Haze increases blue/red ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                blue_red_ratio = np.where(
                    (red > 0.01) & valid_mask,
                    blue / red,
                    np.nan
                )

            valid_ratios = blue_red_ratio[~np.isnan(blue_red_ratio)]
            if len(valid_ratios) == 0:
                return 0.0, metrics

            mean_ratio = np.nanmean(valid_ratios)
            metrics["blue_red_ratio_mean"] = float(mean_ratio)

            # Calculate contrast (lower contrast = more haze)
            brightness = np.nanmean(image_data[:min(3, image_data.shape[0])], axis=0)
            valid_brightness = brightness[valid_mask]
            if len(valid_brightness) > 0:
                contrast = np.nanstd(valid_brightness) / (np.nanmean(valid_brightness) + 1e-6)
                metrics["contrast"] = float(contrast)
            else:
                contrast = 0.5

            # Haze score combines ratio and contrast
            # High ratio and low contrast = high haze
            ratio_score = min(1.0, max(0.0, (mean_ratio - 1.0) / 0.5))
            contrast_score = max(0.0, 1.0 - contrast * 2)

            haze_score = 0.6 * ratio_score + 0.4 * contrast_score
            haze_score = np.clip(haze_score, 0.0, 1.0)

            metrics["ratio_score"] = float(ratio_score)
            metrics["contrast_score"] = float(contrast_score)

            return float(haze_score), metrics

        except Exception as e:
            logger.warning(f"Haze assessment failed: {e}")
            return 0.0, {"error": str(e)}

    def _detect_sun_glint(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray,
        water_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect sun glint in water areas.

        Sun glint appears as very bright pixels in NIR band over water.
        """
        metrics = {}
        glint_mask = np.zeros_like(valid_mask, dtype=bool)

        try:
            # Get NIR band
            nir_idx = min(self.config.nir_band, image_data.shape[0] - 1)
            nir = image_data[nir_idx, :, :]

            # Normalize NIR
            water_nir = nir[water_mask & valid_mask]
            if len(water_nir) == 0:
                return glint_mask, metrics

            p50, p98 = np.nanpercentile(water_nir, [50, 98])

            # Water should be dark in NIR, glint is bright
            if p98 - p50 > 1e-6:
                nir_norm = (nir - p50) / (p98 - p50)
            else:
                nir_norm = nir

            # Glint: bright NIR over water
            glint_mask = (
                water_mask &
                valid_mask &
                (nir_norm > self.config.glint_nir_threshold)
            )

            metrics["water_pixel_count"] = int(water_mask.sum())
            metrics["glint_pixel_count"] = int(glint_mask.sum())

        except Exception as e:
            logger.warning(f"Sun glint detection failed: {e}")
            metrics["error"] = str(e)

        return glint_mask, metrics

    def _detect_saturation(
        self,
        image_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect saturated (overexposed) pixels.

        Saturated pixels have values at or near maximum in multiple bands.
        """
        metrics = {}

        # Find maximum values in each band
        max_vals = np.nanmax(image_data, axis=(1, 2))
        metrics["band_max_values"] = max_vals.tolist()

        # Pixel is saturated if it's at max in multiple bands
        near_max_mask = np.zeros_like(valid_mask, dtype=int)

        for band_idx in range(image_data.shape[0]):
            band_max = max_vals[band_idx]
            if band_max > 0:
                # Consider near-max (within 1% of max)
                threshold = band_max * 0.99
                near_max_mask += (image_data[band_idx] >= threshold).astype(int)

        # Saturated if multiple bands at max
        saturation_mask = (near_max_mask >= 2) & valid_mask

        metrics["saturated_pixel_count"] = int(saturation_mask.sum())

        return saturation_mask, metrics

    def _calculate_overall_score(
        self,
        cloud_percent: float,
        shadow_percent: float,
        haze_score: float,
        glint_percent: float,
        saturation_percent: float,
        clear_percent: float
    ) -> float:
        """
        Calculate overall quality score (0-1).

        Weights different factors based on impact on analysis quality.
        """
        # Component scores
        cloud_score = max(0.0, 1.0 - cloud_percent / 100.0)
        shadow_score = max(0.0, 1.0 - shadow_percent / 50.0)
        haze_score_inv = 1.0 - haze_score
        glint_score = max(0.0, 1.0 - glint_percent / 30.0)
        saturation_score = max(0.0, 1.0 - saturation_percent / 10.0)
        clear_score = clear_percent / 100.0

        # Weighted combination
        weights = {
            "cloud": 0.35,
            "shadow": 0.15,
            "haze": 0.15,
            "glint": 0.10,
            "saturation": 0.10,
            "clear": 0.15
        }

        overall = (
            weights["cloud"] * cloud_score +
            weights["shadow"] * shadow_score +
            weights["haze"] * haze_score_inv +
            weights["glint"] * glint_score +
            weights["saturation"] * saturation_score +
            weights["clear"] * clear_score
        )

        return float(np.clip(overall, 0.0, 1.0))


def assess_optical_quality(
    image_data: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[OpticalQualityConfig] = None,
) -> OpticalQualityResult:
    """
    Convenience function to assess optical image quality.

    Args:
        image_data: Multi-band image array
        metadata: Optional image metadata
        config: Optional configuration

    Returns:
        OpticalQualityResult with assessment
    """
    assessor = OpticalQualityAssessor(config)
    return assessor.assess(image_data, metadata)
