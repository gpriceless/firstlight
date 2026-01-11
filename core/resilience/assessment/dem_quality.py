"""
DEM Quality Assessment.

Evaluates the quality of Digital Elevation Model data including:
- Void detection (missing data areas)
- Artifact detection (stripes, spikes, pits)
- Resolution adequacy for intended use
- Vertical accuracy estimation
- Overall DEM quality score (0-1)

DEM quality is critical for terrain analysis, flood modeling,
and geometric correction of optical/SAR imagery.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DEMArtifactType(Enum):
    """Types of DEM artifacts."""
    VOID = "void"               # Missing data (no-data values)
    STRIPE = "stripe"           # Linear artifacts from sensor issues
    SPIKE = "spike"             # Anomalously high values
    PIT = "pit"                 # Anomalously low values (sinks)
    NOISE = "noise"             # Random noise
    STEP = "step"               # Elevation discontinuities
    EDGE_ARTIFACT = "edge"      # Artifacts at tile boundaries


class DEMIssueSeverity(Enum):
    """Severity levels for DEM quality issues."""
    CRITICAL = "critical"   # DEM unusable
    HIGH = "high"           # Significant quality issues
    MEDIUM = "medium"       # Moderate impact
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational


@dataclass
class DEMQualityIssue:
    """
    A DEM quality issue found during assessment.

    Attributes:
        issue_type: Type of quality issue (artifact type)
        severity: Issue severity level
        description: Human-readable description
        affected_area_percent: Percentage of DEM affected
        location: Affected region information
        recommendation: Suggested action
    """
    issue_type: str
    severity: DEMIssueSeverity
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
class DEMQualityConfig:
    """
    Configuration for DEM quality assessment.

    Attributes:
        max_void_percent: Maximum acceptable void percentage
        spike_threshold_m: Height threshold for spike detection (m)
        pit_threshold_m: Depth threshold for pit detection (m)
        min_resolution_m: Minimum acceptable resolution (m)
        expected_vertical_accuracy_m: Expected vertical accuracy (m)

        check_voids: Enable void detection
        check_artifacts: Enable artifact detection
        check_resolution: Enable resolution adequacy check
        check_accuracy: Enable vertical accuracy estimation
    """
    max_void_percent: float = 5.0
    spike_threshold_m: float = 100.0
    pit_threshold_m: float = 50.0
    min_resolution_m: float = 90.0
    expected_vertical_accuracy_m: float = 10.0

    check_voids: bool = True
    check_artifacts: bool = True
    check_resolution: bool = True
    check_accuracy: bool = True

    # Analysis parameters
    smoothness_window: int = 5
    stripe_detection_length: int = 20
    noise_threshold_m: float = 5.0

    # Expected elevation range (for sanity checks)
    expected_min_elevation_m: float = -500.0
    expected_max_elevation_m: float = 9000.0


@dataclass
class DEMQualityResult:
    """
    Result of DEM quality assessment.

    Attributes:
        overall_score: Overall quality score (0-1)
        is_usable: Whether the DEM is usable
        void_percent: Void (no-data) percentage
        artifact_count: Number of artifacts detected
        resolution_adequate: Whether resolution is adequate
        estimated_accuracy_m: Estimated vertical accuracy
        coverage_percent: Valid data coverage percentage
        issues: List of quality issues
        metrics: Detailed metrics
        duration_seconds: Assessment duration
    """
    overall_score: float
    is_usable: bool
    void_percent: float
    artifact_count: int
    resolution_adequate: bool
    estimated_accuracy_m: float
    coverage_percent: float
    issues: List[DEMQualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == DEMIssueSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 3),
            "is_usable": self.is_usable,
            "void_percent": round(self.void_percent, 2),
            "artifact_count": self.artifact_count,
            "resolution_adequate": self.resolution_adequate,
            "estimated_accuracy_m": round(self.estimated_accuracy_m, 2),
            "coverage_percent": round(self.coverage_percent, 2),
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class DEMQualityAssessor:
    """
    Assesses DEM quality for terrain analysis and modeling.

    Evaluates voids, artifacts, resolution, and accuracy to produce
    an overall quality score and recommendations for DEM use.

    Example:
        assessor = DEMQualityAssessor()
        result = assessor.assess(
            dem_data=elevation_array,
            resolution_m=30.0,
            metadata={"source": "SRTM"}
        )
        if not result.resolution_adequate:
            print("DEM resolution insufficient for analysis")
    """

    def __init__(self, config: Optional[DEMQualityConfig] = None):
        """
        Initialize the DEM quality assessor.

        Args:
            config: Configuration options
        """
        self.config = config or DEMQualityConfig()

    def assess(
        self,
        dem_data: np.ndarray,
        resolution_m: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        reference_dem: Optional[np.ndarray] = None,
        nodata_value: Optional[float] = None,
    ) -> DEMQualityResult:
        """
        Assess DEM quality.

        Args:
            dem_data: DEM elevation array (meters)
            resolution_m: DEM resolution in meters
            metadata: Optional metadata (source, accuracy specs)
            reference_dem: Optional higher-quality reference DEM for accuracy assessment
            nodata_value: Value used for no-data (default: -9999 or NaN)

        Returns:
            DEMQualityResult with detailed assessment
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}
        metadata = metadata or {}

        # Validate input
        if dem_data.size == 0:
            return DEMQualityResult(
                overall_score=0.0,
                is_usable=False,
                void_percent=100.0,
                artifact_count=0,
                resolution_adequate=False,
                estimated_accuracy_m=999.0,
                coverage_percent=0.0,
                issues=[DEMQualityIssue(
                    issue_type="empty_dem",
                    severity=DEMIssueSeverity.CRITICAL,
                    description="DEM data is empty",
                    recommendation="Verify data source"
                )],
                duration_seconds=time.time() - start_time
            )

        # Handle 3D arrays
        if dem_data.ndim == 3:
            dem_data = dem_data[0]

        # Create valid mask (handle various nodata representations)
        if nodata_value is not None:
            valid_mask = (dem_data != nodata_value) & ~np.isnan(dem_data)
        else:
            # Common nodata values
            valid_mask = (
                ~np.isnan(dem_data) &
                (dem_data != -9999) &
                (dem_data != -32768) &
                (dem_data > -500)
            )

        total_pixels = dem_data.size
        valid_pixels = valid_mask.sum()

        if valid_pixels == 0:
            return DEMQualityResult(
                overall_score=0.0,
                is_usable=False,
                void_percent=100.0,
                artifact_count=0,
                resolution_adequate=False,
                estimated_accuracy_m=999.0,
                coverage_percent=0.0,
                issues=[DEMQualityIssue(
                    issue_type="no_valid_data",
                    severity=DEMIssueSeverity.CRITICAL,
                    description="No valid elevation data",
                    recommendation="Check data format and nodata values"
                )],
                duration_seconds=time.time() - start_time
            )

        void_percent = 100.0 * (total_pixels - valid_pixels) / total_pixels
        coverage_percent = 100.0 - void_percent

        metrics["total_pixels"] = int(total_pixels)
        metrics["valid_pixels"] = int(valid_pixels)
        metrics["shape"] = list(dem_data.shape)

        # Elevation statistics
        valid_elevations = dem_data[valid_mask]
        metrics["elevation_min_m"] = float(np.nanmin(valid_elevations))
        metrics["elevation_max_m"] = float(np.nanmax(valid_elevations))
        metrics["elevation_mean_m"] = float(np.nanmean(valid_elevations))
        metrics["elevation_std_m"] = float(np.nanstd(valid_elevations))

        # Initialize results
        artifact_count = 0
        resolution_adequate = True
        estimated_accuracy_m = self.config.expected_vertical_accuracy_m

        # Void detection
        if self.config.check_voids:
            void_issues, void_metrics = self._detect_voids(
                dem_data, valid_mask, void_percent
            )
            issues.extend(void_issues)
            metrics["voids"] = void_metrics

        # Artifact detection
        if self.config.check_artifacts:
            artifacts, artifact_metrics = self._detect_artifacts(
                dem_data, valid_mask
            )
            issues.extend(artifacts)
            artifact_count = len(artifacts)
            metrics["artifacts"] = artifact_metrics

        # Resolution check
        if self.config.check_resolution and resolution_m is not None:
            resolution_adequate, resolution_issues = self._check_resolution(
                resolution_m, metadata
            )
            issues.extend(resolution_issues)
            metrics["resolution_m"] = resolution_m

        # Accuracy estimation
        if self.config.check_accuracy:
            estimated_accuracy_m, accuracy_metrics = self._estimate_accuracy(
                dem_data, valid_mask, reference_dem, metadata
            )
            metrics["accuracy"] = accuracy_metrics

            if estimated_accuracy_m > self.config.expected_vertical_accuracy_m * 2:
                issues.append(DEMQualityIssue(
                    issue_type="low_accuracy",
                    severity=DEMIssueSeverity.MEDIUM,
                    description=f"Estimated vertical accuracy {estimated_accuracy_m:.1f}m exceeds expected",
                    recommendation="Consider using higher-quality DEM source"
                ))

        # Sanity checks
        sanity_issues = self._sanity_check(dem_data, valid_mask, metrics)
        issues.extend(sanity_issues)

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            void_percent,
            artifact_count,
            resolution_adequate,
            estimated_accuracy_m,
            coverage_percent
        )

        # Determine usability
        is_usable = (
            overall_score >= 0.3 and
            void_percent <= 50.0 and
            coverage_percent >= 50.0
        )

        duration = time.time() - start_time
        logger.info(
            f"DEM quality assessment: score={overall_score:.2f}, "
            f"voids={void_percent:.1f}%, artifacts={artifact_count}"
        )

        return DEMQualityResult(
            overall_score=overall_score,
            is_usable=is_usable,
            void_percent=void_percent,
            artifact_count=artifact_count,
            resolution_adequate=resolution_adequate,
            estimated_accuracy_m=estimated_accuracy_m,
            coverage_percent=coverage_percent,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _detect_voids(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray,
        void_percent: float
    ) -> Tuple[List[DEMQualityIssue], Dict[str, Any]]:
        """
        Detect and characterize void regions.

        Analyzes spatial distribution of voids to assess impact.
        """
        issues = []
        metrics = {}

        void_mask = ~valid_mask

        if void_percent > self.config.max_void_percent:
            severity = (
                DEMIssueSeverity.CRITICAL if void_percent > 30
                else DEMIssueSeverity.HIGH if void_percent > 15
                else DEMIssueSeverity.MEDIUM
            )
            issues.append(DEMQualityIssue(
                issue_type=DEMArtifactType.VOID.value,
                severity=severity,
                description=f"Void coverage {void_percent:.1f}% exceeds threshold",
                affected_area_percent=void_percent,
                recommendation="Apply void filling or use alternative DEM"
            ))

        # Analyze void distribution
        if void_mask.any():
            try:
                from scipy import ndimage
                labeled, num_voids = ndimage.label(void_mask)
                metrics["num_void_regions"] = int(num_voids)

                if num_voids > 0:
                    void_sizes = ndimage.sum(void_mask, labeled, range(1, num_voids + 1))
                    metrics["largest_void_pixels"] = int(np.max(void_sizes))
                    metrics["mean_void_size_pixels"] = float(np.mean(void_sizes))

            except ImportError:
                # Simple void analysis without scipy
                metrics["num_void_regions"] = -1

        return issues, metrics

    def _detect_artifacts(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> Tuple[List[DEMQualityIssue], Dict[str, Any]]:
        """
        Detect various DEM artifacts.

        Looks for spikes, pits, stripes, and other anomalies.
        """
        issues = []
        metrics = {}

        valid_elevations = dem_data[valid_mask]
        if len(valid_elevations) == 0:
            return issues, metrics

        # Calculate local statistics
        mean_elev = np.nanmean(valid_elevations)
        std_elev = np.nanstd(valid_elevations)

        # Detect spikes (anomalously high)
        spike_threshold = mean_elev + 3 * std_elev + self.config.spike_threshold_m
        spike_mask = (dem_data > spike_threshold) & valid_mask
        spike_count = spike_mask.sum()

        if spike_count > 0:
            spike_percent = 100.0 * spike_count / valid_mask.sum()
            metrics["spike_count"] = int(spike_count)
            metrics["spike_percent"] = float(spike_percent)

            if spike_percent > 0.1:
                issues.append(DEMQualityIssue(
                    issue_type=DEMArtifactType.SPIKE.value,
                    severity=DEMIssueSeverity.MEDIUM,
                    description=f"Detected {spike_count} spike artifacts",
                    affected_area_percent=spike_percent,
                    recommendation="Apply spike removal filter"
                ))

        # Detect pits (anomalously low)
        pit_threshold = mean_elev - 3 * std_elev - self.config.pit_threshold_m
        pit_mask = (dem_data < pit_threshold) & valid_mask
        pit_count = pit_mask.sum()

        if pit_count > 0:
            pit_percent = 100.0 * pit_count / valid_mask.sum()
            metrics["pit_count"] = int(pit_count)
            metrics["pit_percent"] = float(pit_percent)

            if pit_percent > 0.1:
                issues.append(DEMQualityIssue(
                    issue_type=DEMArtifactType.PIT.value,
                    severity=DEMIssueSeverity.MEDIUM,
                    description=f"Detected {pit_count} pit artifacts",
                    affected_area_percent=pit_percent,
                    recommendation="Apply pit filling or depression breaching"
                ))

        # Detect stripes
        stripe_issues = self._detect_stripes(dem_data, valid_mask)
        issues.extend(stripe_issues)
        metrics["stripe_detected"] = len(stripe_issues) > 0

        # Detect noise
        noise_level = self._estimate_noise(dem_data, valid_mask)
        metrics["noise_level_m"] = float(noise_level)

        if noise_level > self.config.noise_threshold_m:
            issues.append(DEMQualityIssue(
                issue_type=DEMArtifactType.NOISE.value,
                severity=DEMIssueSeverity.LOW,
                description=f"Elevated noise level ({noise_level:.1f}m)",
                recommendation="Apply smoothing filter if needed"
            ))

        return issues, metrics

    def _detect_stripes(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray
    ) -> List[DEMQualityIssue]:
        """Detect linear stripe artifacts."""
        issues = []

        height, width = dem_data.shape

        # Check for row-wise stripes
        row_means = []
        for r in range(height):
            row_valid = valid_mask[r, :]
            if row_valid.sum() > 0:
                row_means.append(np.nanmean(dem_data[r, row_valid]))
            else:
                row_means.append(np.nan)

        row_means = np.array(row_means)
        valid_rows = ~np.isnan(row_means)

        if valid_rows.sum() > 10:
            # Check for abrupt changes between rows
            row_diff = np.abs(np.diff(row_means[valid_rows]))
            if len(row_diff) > 0:
                mean_diff = np.nanmean(row_diff)
                max_diff = np.nanmax(row_diff)

                if max_diff > mean_diff * 5 and max_diff > 10:
                    issues.append(DEMQualityIssue(
                        issue_type=DEMArtifactType.STRIPE.value,
                        severity=DEMIssueSeverity.MEDIUM,
                        description="Horizontal stripe artifacts detected",
                        recommendation="Apply destriping filter"
                    ))

        # Similar check for columns
        col_means = []
        for c in range(width):
            col_valid = valid_mask[:, c]
            if col_valid.sum() > 0:
                col_means.append(np.nanmean(dem_data[col_valid, c]))
            else:
                col_means.append(np.nan)

        col_means = np.array(col_means)
        valid_cols = ~np.isnan(col_means)

        if valid_cols.sum() > 10:
            col_diff = np.abs(np.diff(col_means[valid_cols]))
            if len(col_diff) > 0:
                mean_diff = np.nanmean(col_diff)
                max_diff = np.nanmax(col_diff)

                if max_diff > mean_diff * 5 and max_diff > 10:
                    issues.append(DEMQualityIssue(
                        issue_type=DEMArtifactType.STRIPE.value,
                        severity=DEMIssueSeverity.MEDIUM,
                        description="Vertical stripe artifacts detected",
                        recommendation="Apply destriping filter"
                    ))

        return issues

    def _estimate_noise(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray,
        window_size: int = 3
    ) -> float:
        """
        Estimate DEM noise level using local variance.

        Uses the median absolute deviation of local differences
        as a robust noise estimator.
        """
        # Calculate local differences
        if dem_data.shape[0] < 3 or dem_data.shape[1] < 3:
            return 0.0

        # Gradient-based noise estimation
        gy, gx = np.gradient(dem_data)
        gradient_mag = np.sqrt(gx**2 + gy**2)

        valid_gradients = gradient_mag[valid_mask]
        if len(valid_gradients) == 0:
            return 0.0

        # Use MAD as robust noise estimator
        median_grad = np.nanmedian(valid_gradients)
        mad = np.nanmedian(np.abs(valid_gradients - median_grad))

        # Convert MAD to standard deviation equivalent
        noise_estimate = mad * 1.4826

        return float(noise_estimate)

    def _check_resolution(
        self,
        resolution_m: float,
        metadata: Dict[str, Any]
    ) -> Tuple[bool, List[DEMQualityIssue]]:
        """Check if DEM resolution is adequate."""
        issues = []

        adequate = resolution_m <= self.config.min_resolution_m

        if not adequate:
            issues.append(DEMQualityIssue(
                issue_type="insufficient_resolution",
                severity=DEMIssueSeverity.MEDIUM,
                description=f"Resolution {resolution_m}m exceeds maximum {self.config.min_resolution_m}m",
                recommendation="Use higher resolution DEM source"
            ))

        # Check if resolution matches intended use
        intended_use = metadata.get("intended_use", "general")
        if intended_use == "flood_modeling" and resolution_m > 30:
            issues.append(DEMQualityIssue(
                issue_type="resolution_for_use",
                severity=DEMIssueSeverity.INFO,
                description="Consider higher resolution DEM for detailed flood modeling",
                recommendation="Use 10-30m DEM for urban flood modeling"
            ))

        return adequate, issues

    def _estimate_accuracy(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray,
        reference_dem: Optional[np.ndarray],
        metadata: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate vertical accuracy of DEM.

        Uses reference DEM if available, otherwise uses metadata
        or heuristics based on noise and consistency.
        """
        metrics = {}

        # If reference DEM is provided, calculate actual accuracy
        if reference_dem is not None:
            if reference_dem.shape == dem_data.shape:
                combined_valid = valid_mask & ~np.isnan(reference_dem)
                if combined_valid.sum() > 100:
                    diff = dem_data[combined_valid] - reference_dem[combined_valid]

                    rmse = np.sqrt(np.nanmean(diff**2))
                    mae = np.nanmean(np.abs(diff))
                    bias = np.nanmean(diff)

                    metrics["rmse_m"] = float(rmse)
                    metrics["mae_m"] = float(mae)
                    metrics["bias_m"] = float(bias)
                    metrics["method"] = "reference_comparison"

                    return float(rmse), metrics

        # Use metadata accuracy if provided
        if "vertical_accuracy_m" in metadata:
            metrics["method"] = "metadata"
            return float(metadata["vertical_accuracy_m"]), metrics

        # Estimate from noise level and source
        noise_level = self._estimate_noise(dem_data, valid_mask)

        source = metadata.get("source", "unknown")
        source_accuracy = {
            "SRTM": 16.0,
            "ASTER": 17.0,
            "Copernicus": 4.0,
            "ALOS": 5.0,
            "NASADEM": 12.0,
            "LiDAR": 0.15,
        }

        base_accuracy = source_accuracy.get(source, 10.0)
        estimated = np.sqrt(base_accuracy**2 + noise_level**2)

        metrics["method"] = "estimated"
        metrics["noise_contribution_m"] = float(noise_level)
        metrics["base_accuracy_m"] = float(base_accuracy)

        return float(estimated), metrics

    def _sanity_check(
        self,
        dem_data: np.ndarray,
        valid_mask: np.ndarray,
        metrics: Dict[str, Any]
    ) -> List[DEMQualityIssue]:
        """Perform sanity checks on elevation values."""
        issues = []

        min_elev = metrics.get("elevation_min_m", 0)
        max_elev = metrics.get("elevation_max_m", 0)

        # Check for unrealistic elevations
        if min_elev < self.config.expected_min_elevation_m:
            issues.append(DEMQualityIssue(
                issue_type="unrealistic_minimum",
                severity=DEMIssueSeverity.HIGH,
                description=f"Minimum elevation {min_elev:.0f}m below expected range",
                recommendation="Check for nodata values or datum issues"
            ))

        if max_elev > self.config.expected_max_elevation_m:
            issues.append(DEMQualityIssue(
                issue_type="unrealistic_maximum",
                severity=DEMIssueSeverity.HIGH,
                description=f"Maximum elevation {max_elev:.0f}m above expected range",
                recommendation="Check for spike artifacts or datum issues"
            ))

        # Check for flat areas (constant elevation)
        valid_elevations = dem_data[valid_mask]
        if len(valid_elevations) > 100:
            unique_values = len(np.unique(np.round(valid_elevations, 1)))
            if unique_values < 10:
                issues.append(DEMQualityIssue(
                    issue_type="flat_area",
                    severity=DEMIssueSeverity.INFO,
                    description="DEM shows very little elevation variation",
                    recommendation="Verify area is truly flat or check for interpolation issues"
                ))

        return issues

    def _calculate_overall_score(
        self,
        void_percent: float,
        artifact_count: int,
        resolution_adequate: bool,
        estimated_accuracy_m: float,
        coverage_percent: float
    ) -> float:
        """Calculate overall quality score (0-1)."""
        # Void score
        void_score = max(0.0, 1.0 - void_percent / 30.0)

        # Coverage score
        coverage_score = coverage_percent / 100.0

        # Artifact score
        artifact_score = max(0.0, 1.0 - artifact_count * 0.1)

        # Resolution score
        resolution_score = 1.0 if resolution_adequate else 0.6

        # Accuracy score (assuming 10m is good, 50m is poor)
        if estimated_accuracy_m <= 5:
            accuracy_score = 1.0
        elif estimated_accuracy_m <= 10:
            accuracy_score = 0.9
        elif estimated_accuracy_m <= 20:
            accuracy_score = 0.7
        elif estimated_accuracy_m <= 50:
            accuracy_score = 0.4
        else:
            accuracy_score = 0.2

        # Weighted combination
        weights = {
            "void": 0.25,
            "coverage": 0.20,
            "artifact": 0.15,
            "resolution": 0.20,
            "accuracy": 0.20
        }

        overall = (
            weights["void"] * void_score +
            weights["coverage"] * coverage_score +
            weights["artifact"] * artifact_score +
            weights["resolution"] * resolution_score +
            weights["accuracy"] * accuracy_score
        )

        return float(np.clip(overall, 0.0, 1.0))


def assess_dem_quality(
    dem_data: np.ndarray,
    resolution_m: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[DEMQualityConfig] = None,
) -> DEMQualityResult:
    """
    Convenience function to assess DEM quality.

    Args:
        dem_data: DEM elevation array
        resolution_m: DEM resolution in meters
        metadata: Optional metadata
        config: Optional configuration

    Returns:
        DEMQualityResult with assessment
    """
    assessor = DEMQualityAssessor(config)
    return assessor.assess(dem_data, resolution_m, metadata)
