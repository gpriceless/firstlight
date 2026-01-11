"""
SAR Imagery Quality Assessment.

Evaluates the quality of Synthetic Aperture Radar imagery including:
- Speckle noise level
- Geometric distortion (layover, foreshortening)
- Radiometric calibration accuracy
- Terrain-induced artifacts (layover, shadow)
- Overall SAR quality score (0-1)

SAR quality assessment considers the unique characteristics of radar
imaging including coherent noise and terrain-dependent effects.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SpeckleFilterType(Enum):
    """Types of speckle filters that can be applied."""
    NONE = "none"           # No filtering
    LEE = "lee"             # Lee filter (adaptive)
    FROST = "frost"         # Frost filter
    GAMMA_MAP = "gamma_map" # Gamma MAP filter
    REFINED_LEE = "refined_lee"  # Refined Lee filter
    BOXCAR = "boxcar"       # Simple mean filter


class SARIssueSeverity(Enum):
    """Severity levels for SAR quality issues."""
    CRITICAL = "critical"   # Image unusable
    HIGH = "high"           # Significant quality degradation
    MEDIUM = "medium"       # Moderate impact
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational


class GeometricDistortionType(Enum):
    """Types of geometric distortion in SAR imagery."""
    LAYOVER = "layover"           # Near-range slope effect
    FORESHORTENING = "foreshortening"  # Moderate slope compression
    SHADOW = "shadow"             # Far-range no-return zones


@dataclass
class SARQualityIssue:
    """
    A SAR quality issue found during assessment.

    Attributes:
        issue_type: Type of quality issue
        severity: Issue severity level
        description: Human-readable description
        affected_area_percent: Percentage of image affected
        location: Affected region information
        recommendation: Suggested action
    """
    issue_type: str
    severity: SARIssueSeverity
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
class SARQualityConfig:
    """
    Configuration for SAR quality assessment.

    Attributes:
        max_speckle_enl: Maximum acceptable equivalent number of looks
        min_speckle_enl: Minimum acceptable ENL (below = too noisy)
        calibration_tolerance_db: Acceptable radiometric calibration error (dB)
        max_layover_percent: Maximum acceptable layover percentage
        max_shadow_percent: Maximum acceptable radar shadow percentage

        check_speckle: Enable speckle noise assessment
        check_calibration: Enable radiometric calibration check
        check_geometry: Enable geometric distortion check
        check_terrain: Enable terrain artifact detection
    """
    max_speckle_enl: float = 50.0
    min_speckle_enl: float = 3.0
    calibration_tolerance_db: float = 1.0
    max_layover_percent: float = 20.0
    max_shadow_percent: float = 15.0

    check_speckle: bool = True
    check_calibration: bool = True
    check_geometry: bool = True
    check_terrain: bool = True

    # Expected backscatter ranges by surface type (dB)
    expected_water_sigma0_db: Tuple[float, float] = (-25.0, -15.0)
    expected_vegetation_sigma0_db: Tuple[float, float] = (-15.0, -5.0)
    expected_urban_sigma0_db: Tuple[float, float] = (-10.0, 5.0)

    # Filter recommendation thresholds
    recommend_filter_enl_threshold: float = 5.0


@dataclass
class SARQualityResult:
    """
    Result of SAR quality assessment.

    Attributes:
        overall_score: Overall quality score (0-1)
        is_usable: Whether the image is usable
        speckle_enl: Equivalent number of looks
        speckle_std: Speckle standard deviation
        calibration_error_db: Estimated calibration error
        layover_percent: Layover affected percentage
        shadow_percent: Radar shadow percentage
        recommended_filter: Recommended speckle filter
        issues: List of quality issues
        metrics: Detailed metrics
        duration_seconds: Assessment duration
    """
    overall_score: float
    is_usable: bool
    speckle_enl: float
    speckle_std: float
    calibration_error_db: float
    layover_percent: float
    shadow_percent: float
    recommended_filter: SpeckleFilterType
    issues: List[SARQualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == SARIssueSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 3),
            "is_usable": self.is_usable,
            "speckle_enl": round(self.speckle_enl, 2),
            "speckle_std": round(self.speckle_std, 3),
            "calibration_error_db": round(self.calibration_error_db, 2),
            "layover_percent": round(self.layover_percent, 2),
            "shadow_percent": round(self.shadow_percent, 2),
            "recommended_filter": self.recommended_filter.value,
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class SARQualityAssessor:
    """
    Assesses SAR imagery quality for analysis suitability.

    Evaluates speckle noise, radiometric calibration, and geometric
    distortions to produce an overall quality score and recommendations.

    Example:
        assessor = SARQualityAssessor()
        result = assessor.assess(
            intensity_data=sigma0_array,
            dem_data=elevation_array,
            metadata={"orbit": "ascending", "look_angle": 35.0}
        )
        if result.recommended_filter != SpeckleFilterType.NONE:
            print(f"Apply {result.recommended_filter.value} filter")
    """

    def __init__(self, config: Optional[SARQualityConfig] = None):
        """
        Initialize the SAR quality assessor.

        Args:
            config: Configuration options
        """
        self.config = config or SARQualityConfig()

    def assess(
        self,
        intensity_data: np.ndarray,
        dem_data: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
        valid_mask: Optional[np.ndarray] = None,
        reference_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> SARQualityResult:
        """
        Assess SAR image quality.

        Args:
            intensity_data: SAR intensity/backscatter array (linear or dB)
            dem_data: Optional DEM for terrain analysis
            metadata: Optional metadata (orbit, look angle, etc.)
            valid_mask: Optional mask of valid pixels
            reference_targets: Optional dict of reference target masks for calibration

        Returns:
            SARQualityResult with detailed assessment
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}
        metadata = metadata or {}

        # Validate input
        if intensity_data.size == 0:
            return SARQualityResult(
                overall_score=0.0,
                is_usable=False,
                speckle_enl=0.0,
                speckle_std=1.0,
                calibration_error_db=999.0,
                layover_percent=0.0,
                shadow_percent=0.0,
                recommended_filter=SpeckleFilterType.LEE,
                issues=[SARQualityIssue(
                    issue_type="empty_image",
                    severity=SARIssueSeverity.CRITICAL,
                    description="SAR data is empty",
                    recommendation="Verify data acquisition"
                )],
                duration_seconds=time.time() - start_time
            )

        # Handle 2D/3D arrays
        if intensity_data.ndim == 3:
            # Use first band or average
            intensity_data = intensity_data[0] if intensity_data.shape[0] <= 2 else np.nanmean(intensity_data, axis=0)

        # Create valid mask
        if valid_mask is None:
            valid_mask = ~np.isnan(intensity_data) & (intensity_data > 0)

        num_valid = valid_mask.sum()
        if num_valid == 0:
            return SARQualityResult(
                overall_score=0.0,
                is_usable=False,
                speckle_enl=0.0,
                speckle_std=1.0,
                calibration_error_db=999.0,
                layover_percent=0.0,
                shadow_percent=0.0,
                recommended_filter=SpeckleFilterType.LEE,
                issues=[SARQualityIssue(
                    issue_type="no_valid_pixels",
                    severity=SARIssueSeverity.CRITICAL,
                    description="No valid SAR pixels",
                    recommendation="Check data format and processing"
                )],
                duration_seconds=time.time() - start_time
            )

        metrics["num_valid_pixels"] = int(num_valid)

        # Convert to linear if in dB
        is_db = metadata.get("scale", "linear") == "dB"
        if is_db:
            intensity_linear = np.power(10, intensity_data / 10)
        else:
            intensity_linear = intensity_data

        # Initialize results
        speckle_enl = 0.0
        speckle_std = 1.0
        calibration_error_db = 0.0
        layover_percent = 0.0
        shadow_percent = 0.0
        recommended_filter = SpeckleFilterType.NONE

        # Speckle assessment
        if self.config.check_speckle:
            speckle_enl, speckle_std, speckle_metrics = self._assess_speckle(
                intensity_linear, valid_mask
            )
            metrics["speckle"] = speckle_metrics

            if speckle_enl < self.config.min_speckle_enl:
                issues.append(SARQualityIssue(
                    issue_type="high_speckle",
                    severity=SARIssueSeverity.HIGH,
                    description=f"High speckle noise (ENL={speckle_enl:.1f})",
                    recommendation="Apply speckle filtering before analysis"
                ))
                recommended_filter = SpeckleFilterType.REFINED_LEE

            elif speckle_enl < self.config.recommend_filter_enl_threshold:
                issues.append(SARQualityIssue(
                    issue_type="moderate_speckle",
                    severity=SARIssueSeverity.MEDIUM,
                    description=f"Moderate speckle (ENL={speckle_enl:.1f})",
                    recommendation="Consider Lee or Frost filter"
                ))
                recommended_filter = SpeckleFilterType.LEE

        # Radiometric calibration check
        if self.config.check_calibration:
            calibration_error_db, cal_metrics = self._check_calibration(
                intensity_linear, valid_mask, reference_targets
            )
            metrics["calibration"] = cal_metrics

            if abs(calibration_error_db) > self.config.calibration_tolerance_db:
                issues.append(SARQualityIssue(
                    issue_type="calibration_error",
                    severity=SARIssueSeverity.MEDIUM,
                    description=f"Radiometric calibration error: {calibration_error_db:.1f} dB",
                    recommendation="Apply radiometric terrain flattening"
                ))

        # Geometric distortion analysis
        if self.config.check_geometry and dem_data is not None:
            layover_percent, shadow_percent, geo_metrics = self._analyze_geometry(
                intensity_linear, dem_data, valid_mask, metadata
            )
            metrics["geometry"] = geo_metrics

            if layover_percent > self.config.max_layover_percent:
                issues.append(SARQualityIssue(
                    issue_type="excessive_layover",
                    severity=SARIssueSeverity.HIGH,
                    description=f"Layover affects {layover_percent:.1f}% of image",
                    affected_area_percent=layover_percent,
                    recommendation="Use ascending/descending orbit fusion or different sensor"
                ))

            if shadow_percent > self.config.max_shadow_percent:
                issues.append(SARQualityIssue(
                    issue_type="excessive_shadow",
                    severity=SARIssueSeverity.MEDIUM,
                    description=f"Radar shadow affects {shadow_percent:.1f}% of image",
                    affected_area_percent=shadow_percent,
                    recommendation="Consider opposite orbit direction"
                ))

        # Terrain artifacts
        if self.config.check_terrain and dem_data is not None:
            terrain_issues = self._detect_terrain_artifacts(
                intensity_linear, dem_data, valid_mask
            )
            issues.extend(terrain_issues)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            speckle_enl,
            calibration_error_db,
            layover_percent,
            shadow_percent
        )

        # Determine usability
        is_usable = (
            overall_score >= 0.3 and
            layover_percent <= 50.0 and
            speckle_enl >= 1.0
        )

        duration = time.time() - start_time
        logger.info(
            f"SAR quality assessment: score={overall_score:.2f}, "
            f"ENL={speckle_enl:.1f}, layover={layover_percent:.1f}%"
        )

        return SARQualityResult(
            overall_score=overall_score,
            is_usable=is_usable,
            speckle_enl=speckle_enl,
            speckle_std=speckle_std,
            calibration_error_db=calibration_error_db,
            layover_percent=layover_percent,
            shadow_percent=shadow_percent,
            recommended_filter=recommended_filter,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _assess_speckle(
        self,
        intensity: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Assess speckle noise level.

        Uses Equivalent Number of Looks (ENL) as the primary metric.
        ENL = (mean/std)^2 for intensity data under multiplicative noise model.
        """
        metrics = {}

        valid_data = intensity[valid_mask]
        if len(valid_data) == 0:
            return 0.0, 1.0, metrics

        # Calculate statistics
        mean_val = np.nanmean(valid_data)
        std_val = np.nanstd(valid_data)

        metrics["mean_intensity"] = float(mean_val)
        metrics["std_intensity"] = float(std_val)

        if mean_val < 1e-10:
            return 0.0, std_val, metrics

        # ENL calculation
        enl = (mean_val / std_val) ** 2 if std_val > 1e-10 else 0.0
        metrics["enl"] = float(enl)

        # Coefficient of variation
        cv = std_val / mean_val
        metrics["coefficient_of_variation"] = float(cv)

        # Local ENL estimation (more robust)
        local_enl = self._estimate_local_enl(intensity, valid_mask)
        metrics["local_enl"] = float(local_enl)

        # Use average of global and local ENL
        final_enl = (enl + local_enl) / 2

        return float(final_enl), float(cv), metrics

    def _estimate_local_enl(
        self,
        intensity: np.ndarray,
        valid_mask: np.ndarray,
        window_size: int = 7,
        num_samples: int = 1000
    ) -> float:
        """Estimate ENL using local windows for robustness."""
        height, width = intensity.shape
        half_win = window_size // 2

        # Sample random locations
        valid_rows, valid_cols = np.where(valid_mask)
        if len(valid_rows) < num_samples:
            indices = np.arange(len(valid_rows))
        else:
            indices = np.random.choice(len(valid_rows), num_samples, replace=False)

        local_enls = []

        for idx in indices:
            row, col = valid_rows[idx], valid_cols[idx]

            # Skip edge pixels
            if (row < half_win or row >= height - half_win or
                col < half_win or col >= width - half_win):
                continue

            # Extract window
            window = intensity[
                row - half_win:row + half_win + 1,
                col - half_win:col + half_win + 1
            ]

            window_mask = valid_mask[
                row - half_win:row + half_win + 1,
                col - half_win:col + half_win + 1
            ]

            valid_window = window[window_mask]
            if len(valid_window) < 10:
                continue

            # Local ENL
            mean_val = np.nanmean(valid_window)
            std_val = np.nanstd(valid_window)

            if std_val > 1e-10 and mean_val > 1e-10:
                local_enl = (mean_val / std_val) ** 2
                if 0.5 < local_enl < 100:  # Reasonable range
                    local_enls.append(local_enl)

        if not local_enls:
            return 1.0

        return float(np.median(local_enls))

    def _check_calibration(
        self,
        intensity: np.ndarray,
        valid_mask: np.ndarray,
        reference_targets: Optional[Dict[str, np.ndarray]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Check radiometric calibration accuracy.

        Uses known surface types to validate backscatter values.
        """
        metrics = {}

        # Convert to dB
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma0_db = np.where(
                intensity > 0,
                10 * np.log10(intensity),
                np.nan
            )

        # Overall statistics
        valid_db = sigma0_db[valid_mask & ~np.isnan(sigma0_db)]
        if len(valid_db) == 0:
            return 0.0, metrics

        metrics["sigma0_mean_db"] = float(np.nanmean(valid_db))
        metrics["sigma0_std_db"] = float(np.nanstd(valid_db))
        metrics["sigma0_range_db"] = [float(np.nanmin(valid_db)), float(np.nanmax(valid_db))]

        calibration_error = 0.0

        # Check against reference targets if provided
        if reference_targets:
            for target_type, target_mask in reference_targets.items():
                combined_mask = target_mask & valid_mask
                if combined_mask.sum() < 10:
                    continue

                target_db = sigma0_db[combined_mask]
                target_mean = np.nanmean(target_db)
                target_std = np.nanstd(target_db)

                metrics[f"{target_type}_mean_db"] = float(target_mean)
                metrics[f"{target_type}_std_db"] = float(target_std)

                # Compare to expected range
                expected_key = f"expected_{target_type}_sigma0_db"
                expected_range = getattr(self.config, expected_key, None)

                if expected_range:
                    exp_min, exp_max = expected_range
                    expected_mid = (exp_min + exp_max) / 2

                    if target_mean < exp_min:
                        calibration_error = max(calibration_error, target_mean - exp_min)
                    elif target_mean > exp_max:
                        calibration_error = max(calibration_error, target_mean - exp_max)

                    metrics[f"{target_type}_error_db"] = float(target_mean - expected_mid)

        # If no reference targets, use histogram analysis
        if not reference_targets:
            # Check for bimodal distribution (water + land)
            hist, bin_edges = np.histogram(valid_db, bins=50)
            metrics["histogram_peak_db"] = float(bin_edges[np.argmax(hist)])

            # Simple check: mean should be in reasonable range for mixed scenes
            if metrics["sigma0_mean_db"] < -20 or metrics["sigma0_mean_db"] > 0:
                calibration_error = abs(metrics["sigma0_mean_db"] - (-10))

        return float(calibration_error), metrics

    def _analyze_geometry(
        self,
        intensity: np.ndarray,
        dem: np.ndarray,
        valid_mask: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Analyze geometric distortions from terrain.

        Detects layover and shadow regions based on slope and look angle.
        """
        metrics = {}

        # Get look angle (default to 35 degrees if not provided)
        look_angle = metadata.get("look_angle", 35.0)
        look_angle_rad = np.radians(look_angle)
        metrics["look_angle_deg"] = look_angle

        # Get orbit direction
        orbit = metadata.get("orbit", "descending")
        metrics["orbit"] = orbit

        # Calculate slope and aspect from DEM
        slope, aspect = self._calculate_terrain(dem)
        metrics["mean_slope_deg"] = float(np.nanmean(np.degrees(slope)))
        metrics["max_slope_deg"] = float(np.nanmax(np.degrees(slope)))

        # Calculate local incidence angle
        # For simplicity, use slope as proxy for terrain effect
        # Real implementation would consider aspect relative to look direction

        # Layover: occurs when terrain slope > 90 - look_angle
        layover_threshold = np.pi/2 - look_angle_rad
        layover_mask = (slope > layover_threshold) & valid_mask

        # Shadow: occurs when backslope is steep and away from radar
        shadow_threshold = look_angle_rad
        shadow_mask = (slope > shadow_threshold) & valid_mask

        # Refine shadow detection using intensity
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity_db = np.where(intensity > 0, 10 * np.log10(intensity), -40)

        very_dark = intensity_db < -30
        shadow_mask = shadow_mask | (very_dark & valid_mask)

        num_valid = valid_mask.sum()
        layover_percent = 100.0 * layover_mask.sum() / num_valid if num_valid > 0 else 0.0
        shadow_percent = 100.0 * shadow_mask.sum() / num_valid if num_valid > 0 else 0.0

        metrics["layover_pixels"] = int(layover_mask.sum())
        metrics["shadow_pixels"] = int(shadow_mask.sum())

        return float(layover_percent), float(shadow_percent), metrics

    def _calculate_terrain(
        self,
        dem: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate slope and aspect from DEM."""
        # Calculate gradients
        gy, gx = np.gradient(dem)

        # Slope (radians)
        slope = np.arctan(np.sqrt(gx**2 + gy**2))

        # Aspect (radians from north)
        aspect = np.arctan2(-gx, gy)

        return slope, aspect

    def _detect_terrain_artifacts(
        self,
        intensity: np.ndarray,
        dem: np.ndarray,
        valid_mask: np.ndarray
    ) -> List[SARQualityIssue]:
        """Detect terrain-induced artifacts."""
        issues = []

        # Calculate local slope variability
        slope, _ = self._calculate_terrain(dem)

        # High slope variance indicates complex terrain
        slope_std = np.nanstd(np.degrees(slope[valid_mask]))

        if slope_std > 15.0:
            issues.append(SARQualityIssue(
                issue_type="complex_terrain",
                severity=SARIssueSeverity.INFO,
                description=f"Complex terrain detected (slope std={slope_std:.1f} deg)",
                recommendation="Consider terrain correction and orthorectification"
            ))

        # Check for extremely bright pixels (potential artifacts)
        with np.errstate(divide='ignore', invalid='ignore'):
            intensity_db = np.where(intensity > 0, 10 * np.log10(intensity), np.nan)

        valid_db = intensity_db[valid_mask & ~np.isnan(intensity_db)]
        if len(valid_db) > 0:
            p99 = np.percentile(valid_db, 99)
            if p99 > 5.0:
                issues.append(SARQualityIssue(
                    issue_type="bright_artifacts",
                    severity=SARIssueSeverity.LOW,
                    description=f"Very bright pixels detected (99th percentile={p99:.1f} dB)",
                    recommendation="Check for double bounce or corner reflector effects"
                ))

        return issues

    def _calculate_overall_score(
        self,
        speckle_enl: float,
        calibration_error_db: float,
        layover_percent: float,
        shadow_percent: float
    ) -> float:
        """Calculate overall quality score (0-1)."""
        # Speckle score: better with higher ENL
        if speckle_enl < 1:
            speckle_score = 0.0
        elif speckle_enl < 3:
            speckle_score = 0.3 * speckle_enl / 3
        elif speckle_enl < 10:
            speckle_score = 0.3 + 0.5 * (speckle_enl - 3) / 7
        else:
            speckle_score = min(1.0, 0.8 + 0.2 * min(1, (speckle_enl - 10) / 40))

        # Calibration score
        cal_score = max(0.0, 1.0 - abs(calibration_error_db) / 5.0)

        # Geometric scores
        layover_score = max(0.0, 1.0 - layover_percent / 50.0)
        shadow_score = max(0.0, 1.0 - shadow_percent / 30.0)

        # Weighted combination
        weights = {"speckle": 0.35, "calibration": 0.25, "layover": 0.25, "shadow": 0.15}

        overall = (
            weights["speckle"] * speckle_score +
            weights["calibration"] * cal_score +
            weights["layover"] * layover_score +
            weights["shadow"] * shadow_score
        )

        return float(np.clip(overall, 0.0, 1.0))


def assess_sar_quality(
    intensity_data: np.ndarray,
    dem_data: Optional[np.ndarray] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[SARQualityConfig] = None,
) -> SARQualityResult:
    """
    Convenience function to assess SAR image quality.

    Args:
        intensity_data: SAR intensity array
        dem_data: Optional DEM for terrain analysis
        metadata: Optional metadata
        config: Optional configuration

    Returns:
        SARQualityResult with assessment
    """
    assessor = SARQualityAssessor(config)
    return assessor.assess(intensity_data, dem_data, metadata)
