"""
Quality Summary Generation for Ingested Data.

Generates comprehensive quality summaries for ingested raster and vector data,
combining multiple quality metrics into a unified assessment.

Quality Metrics Include:
- Data completeness (valid pixel coverage)
- Spatial quality (resolution, alignment, edge artifacts)
- Temporal quality (freshness, consistency)
- Radiometric quality (noise, saturation, striping)
- Geometric quality (projection accuracy, georeferencing)
- Cloud/atmospheric quality (for optical data)
- Overall quality score with confidence

Quality summaries enable:
- Automated data acceptance/rejection
- Quality-based data ranking and selection
- User communication of data limitations
- Provenance documentation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Overall quality assessment levels."""

    EXCELLENT = "excellent"  # > 90%
    GOOD = "good"  # 70-90%
    ACCEPTABLE = "acceptable"  # 50-70%
    DEGRADED = "degraded"  # 30-50%
    POOR = "poor"  # < 30%

    @classmethod
    def from_score(cls, score: float) -> "QualityLevel":
        """
        Convert a quality score (0-1) to a quality level.

        Args:
            score: Quality score between 0 and 1

        Returns:
            Corresponding quality level
        """
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.7:
            return cls.GOOD
        elif score >= 0.5:
            return cls.ACCEPTABLE
        elif score >= 0.3:
            return cls.DEGRADED
        else:
            return cls.POOR


class QualityDimension(Enum):
    """Quality assessment dimensions."""

    COMPLETENESS = "completeness"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    RADIOMETRIC = "radiometric"
    GEOMETRIC = "geometric"
    ATMOSPHERIC = "atmospheric"


class QualityIssue(Enum):
    """Types of quality issues detected."""

    # Completeness issues
    MISSING_DATA = "missing_data"
    INCOMPLETE_COVERAGE = "incomplete_coverage"
    NODATA_CONTAMINATION = "nodata_contamination"

    # Spatial issues
    LOW_RESOLUTION = "low_resolution"
    EDGE_ARTIFACTS = "edge_artifacts"
    MISALIGNMENT = "misalignment"
    TILING_ARTIFACTS = "tiling_artifacts"

    # Temporal issues
    STALE_DATA = "stale_data"
    TEMPORAL_GAP = "temporal_gap"
    TIMESTAMP_MISSING = "timestamp_missing"

    # Radiometric issues
    HIGH_NOISE = "high_noise"
    SATURATION = "saturation"
    STRIPING = "striping"
    BANDING = "banding"
    DETECTOR_ANOMALY = "detector_anomaly"

    # Geometric issues
    PROJECTION_ERROR = "projection_error"
    GEOREFERENCING_ERROR = "georeferencing_error"
    DISTORTION = "distortion"

    # Atmospheric issues
    CLOUD_COVER = "cloud_cover"
    HAZE = "haze"
    SHADOW = "shadow"
    SMOKE = "smoke"
    AEROSOL = "aerosol"


@dataclass
class QualityFlag:
    """A single quality issue or flag."""

    issue: QualityIssue
    severity: float  # 0-1, where 1 is most severe
    affected_area_percent: float  # Percent of data affected
    description: str
    recommendation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue": self.issue.value,
            "severity": round(self.severity, 3),
            "affected_area_percent": round(self.affected_area_percent, 2),
            "description": self.description,
            "recommendation": self.recommendation,
            "metadata": self.metadata,
        }


@dataclass
class DimensionScore:
    """Quality score for a single dimension."""

    dimension: QualityDimension
    score: float  # 0-1
    weight: float  # Relative importance
    flags: List[QualityFlag]
    details: Dict[str, Any]

    @property
    def weighted_score(self) -> float:
        """Score weighted by importance."""
        return self.score * self.weight

    @property
    def level(self) -> QualityLevel:
        """Quality level for this dimension."""
        return QualityLevel.from_score(self.score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 3),
            "weight": round(self.weight, 3),
            "weighted_score": round(self.weighted_score, 3),
            "level": self.level.value,
            "flags": [f.to_dict() for f in self.flags],
            "details": self.details,
        }


@dataclass
class QualityConfig:
    """
    Configuration for quality assessment.

    Attributes:
        dimension_weights: Weights for each quality dimension
        min_valid_percent: Minimum valid data percentage
        max_cloud_cover: Maximum acceptable cloud cover
        max_noise_threshold: Maximum acceptable noise level
        check_geometric: Whether to check geometric quality
        check_atmospheric: Whether to check atmospheric quality
        issue_thresholds: Thresholds for flagging issues
    """

    dimension_weights: Dict[QualityDimension, float] = field(
        default_factory=lambda: {
            QualityDimension.COMPLETENESS: 0.25,
            QualityDimension.SPATIAL: 0.20,
            QualityDimension.TEMPORAL: 0.15,
            QualityDimension.RADIOMETRIC: 0.20,
            QualityDimension.GEOMETRIC: 0.10,
            QualityDimension.ATMOSPHERIC: 0.10,
        }
    )
    min_valid_percent: float = 80.0
    max_cloud_cover: float = 30.0
    max_noise_threshold: float = 0.1
    check_geometric: bool = True
    check_atmospheric: bool = True
    issue_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "saturation_percent": 1.0,
            "striping_score": 0.1,
            "noise_snr_min": 10.0,
            "edge_artifact_pixels": 10,
            "temporal_staleness_days": 30,
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        total_weight = sum(self.dimension_weights.values())
        if total_weight <= 0:
            # If all weights are zero, reset to default weights
            self.dimension_weights = {
                QualityDimension.COMPLETENESS: 0.25,
                QualityDimension.SPATIAL: 0.20,
                QualityDimension.TEMPORAL: 0.15,
                QualityDimension.RADIOMETRIC: 0.20,
                QualityDimension.GEOMETRIC: 0.10,
                QualityDimension.ATMOSPHERIC: 0.10,
            }
        elif abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            for dim in self.dimension_weights:
                self.dimension_weights[dim] /= total_weight


@dataclass
class QualitySummary:
    """
    Comprehensive quality summary for ingested data.

    Combines multiple quality dimensions into an overall assessment
    with detailed flags, scores, and recommendations.
    """

    path: Path
    overall_score: float
    overall_level: QualityLevel
    confidence: float
    dimension_scores: List[DimensionScore]
    flags: List[QualityFlag]
    usable: bool
    recommendations: List[str]
    assessed_at: datetime
    metadata: Dict[str, Any]

    @property
    def critical_issues(self) -> List[QualityFlag]:
        """Get critical issues (severity > 0.7)."""
        return [f for f in self.flags if f.severity > 0.7]

    @property
    def has_critical_issues(self) -> bool:
        """Check if any critical issues exist."""
        return len(self.critical_issues) > 0

    def get_dimension(self, dimension: QualityDimension) -> Optional[DimensionScore]:
        """Get score for a specific dimension."""
        for ds in self.dimension_scores:
            if ds.dimension == dimension:
                return ds
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "overall_score": round(self.overall_score, 3),
            "overall_level": self.overall_level.value,
            "confidence": round(self.confidence, 3),
            "dimension_scores": [ds.to_dict() for ds in self.dimension_scores],
            "flags": [f.to_dict() for f in self.flags],
            "usable": self.usable,
            "recommendations": self.recommendations,
            "assessed_at": self.assessed_at.isoformat(),
            "metadata": self.metadata,
        }


class QualityAssessor:
    """
    Assessor for generating quality summaries.

    Evaluates multiple quality dimensions and combines them into
    a comprehensive quality summary with flags and recommendations.

    Example:
        assessor = QualityAssessor(QualityConfig(
            min_valid_percent=80.0,
            max_cloud_cover=30.0
        ))
        summary = assessor.assess("image.tif")

        if not summary.usable:
            print(f"Data unusable: {summary.recommendations}")
    """

    def __init__(self, config: Optional[QualityConfig] = None):
        """
        Initialize quality assessor.

        Args:
            config: Quality assessment configuration
        """
        self.config = config or QualityConfig()

    def assess(
        self,
        path: Union[str, Path],
        statistics: Optional[Any] = None,
        cloud_mask: Optional[np.ndarray] = None,
        expected_bounds: Optional[Tuple[float, float, float, float]] = None,
        acquisition_time: Optional[datetime] = None,
    ) -> QualitySummary:
        """
        Assess quality of a raster file.

        Args:
            path: Path to raster file
            statistics: Pre-computed statistics (RasterStatistics)
            cloud_mask: Cloud mask array (True = cloud)
            expected_bounds: Expected spatial bounds for coverage check
            acquisition_time: Data acquisition time for temporal assessment

        Returns:
            QualitySummary with comprehensive quality assessment

        Raises:
            FileNotFoundError: If input file doesn't exist
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for quality assessment")

        logger.info(f"Assessing quality for {path}")

        # Read file metadata
        with rasterio.open(path) as src:
            width = src.width
            height = src.height
            band_count = src.count
            nodata = src.nodata
            crs = str(src.crs) if src.crs else None
            bounds = src.bounds
            dtype = src.dtypes[0]

            # Read data for analysis
            data = src.read()

        dimension_scores = []
        all_flags = []

        # 1. Completeness assessment
        completeness_score, completeness_flags, completeness_details = self._assess_completeness(
            data, nodata, expected_bounds, bounds if bounds else None
        )
        dimension_scores.append(
            DimensionScore(
                dimension=QualityDimension.COMPLETENESS,
                score=completeness_score,
                weight=self.config.dimension_weights[QualityDimension.COMPLETENESS],
                flags=completeness_flags,
                details=completeness_details,
            )
        )
        all_flags.extend(completeness_flags)

        # 2. Spatial assessment
        spatial_score, spatial_flags, spatial_details = self._assess_spatial(
            data, width, height
        )
        dimension_scores.append(
            DimensionScore(
                dimension=QualityDimension.SPATIAL,
                score=spatial_score,
                weight=self.config.dimension_weights[QualityDimension.SPATIAL],
                flags=spatial_flags,
                details=spatial_details,
            )
        )
        all_flags.extend(spatial_flags)

        # 3. Temporal assessment
        temporal_score, temporal_flags, temporal_details = self._assess_temporal(
            acquisition_time
        )
        dimension_scores.append(
            DimensionScore(
                dimension=QualityDimension.TEMPORAL,
                score=temporal_score,
                weight=self.config.dimension_weights[QualityDimension.TEMPORAL],
                flags=temporal_flags,
                details=temporal_details,
            )
        )
        all_flags.extend(temporal_flags)

        # 4. Radiometric assessment
        radiometric_score, radiometric_flags, radiometric_details = self._assess_radiometric(
            data, dtype
        )
        dimension_scores.append(
            DimensionScore(
                dimension=QualityDimension.RADIOMETRIC,
                score=radiometric_score,
                weight=self.config.dimension_weights[QualityDimension.RADIOMETRIC],
                flags=radiometric_flags,
                details=radiometric_details,
            )
        )
        all_flags.extend(radiometric_flags)

        # 5. Geometric assessment
        if self.config.check_geometric:
            geometric_score, geometric_flags, geometric_details = self._assess_geometric(
                crs, bounds
            )
            dimension_scores.append(
                DimensionScore(
                    dimension=QualityDimension.GEOMETRIC,
                    score=geometric_score,
                    weight=self.config.dimension_weights[QualityDimension.GEOMETRIC],
                    flags=geometric_flags,
                    details=geometric_details,
                )
            )
            all_flags.extend(geometric_flags)

        # 6. Atmospheric assessment
        if self.config.check_atmospheric and cloud_mask is not None:
            atmospheric_score, atmospheric_flags, atmospheric_details = self._assess_atmospheric(
                cloud_mask
            )
            dimension_scores.append(
                DimensionScore(
                    dimension=QualityDimension.ATMOSPHERIC,
                    score=atmospheric_score,
                    weight=self.config.dimension_weights[QualityDimension.ATMOSPHERIC],
                    flags=atmospheric_flags,
                    details=atmospheric_details,
                )
            )
            all_flags.extend(atmospheric_flags)

        # Calculate overall score
        total_weight = sum(ds.weight for ds in dimension_scores)
        if total_weight > 0:
            overall_score = sum(ds.weighted_score for ds in dimension_scores) / total_weight
        else:
            overall_score = 0.0

        overall_level = QualityLevel.from_score(overall_score)

        # Calculate confidence based on how many dimensions were assessed
        max_dimensions = len(QualityDimension)
        assessed_dimensions = len(dimension_scores)
        confidence = assessed_dimensions / max_dimensions

        # Determine usability
        usable = (
            overall_score >= 0.3
            and not any(f.severity > 0.9 for f in all_flags)
            and completeness_score >= 0.3
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(all_flags, dimension_scores)

        logger.info(
            f"Quality assessment complete: {overall_level.value} "
            f"(score={overall_score:.2f}, usable={usable})"
        )

        return QualitySummary(
            path=path,
            overall_score=overall_score,
            overall_level=overall_level,
            confidence=confidence,
            dimension_scores=dimension_scores,
            flags=all_flags,
            usable=usable,
            recommendations=recommendations,
            assessed_at=datetime.now(timezone.utc),
            metadata={
                "width": width,
                "height": height,
                "band_count": band_count,
                "dtype": dtype,
                "crs": crs,
            },
        )

    def _assess_completeness(
        self,
        data: np.ndarray,
        nodata: Optional[float],
        expected_bounds: Optional[Tuple[float, float, float, float]],
        actual_bounds: Optional[Any],
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess data completeness."""
        flags = []
        details = {}

        # Calculate valid data percentage
        mask = np.zeros(data.shape, dtype=bool)
        if nodata is not None:
            mask |= data == nodata
        mask |= ~np.isfinite(data.astype(float))

        total_pixels = data.size
        valid_pixels = total_pixels - mask.sum()
        valid_percent = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        details["valid_percent"] = valid_percent
        details["valid_pixels"] = int(valid_pixels)
        details["nodata_pixels"] = int(mask.sum())

        # Score based on valid data percentage
        score = valid_percent / 100.0

        if valid_percent < self.config.min_valid_percent:
            severity = 1.0 - (valid_percent / self.config.min_valid_percent)
            flags.append(
                QualityFlag(
                    issue=QualityIssue.MISSING_DATA,
                    severity=min(severity, 1.0),
                    affected_area_percent=100.0 - valid_percent,
                    description=f"Only {valid_percent:.1f}% valid data (minimum: {self.config.min_valid_percent}%)",
                    recommendation="Consider using alternative data source or accepting degraded coverage",
                )
            )

        # Check for nodata contamination patterns
        if data.ndim == 3:
            for band in range(data.shape[0]):
                band_data = data[band]
                band_mask = mask[band] if mask.ndim == 3 else mask[0]
                # Check for large contiguous nodata regions
                nodata_percent = band_mask.sum() / band_mask.size * 100
                if nodata_percent > 20:
                    # Check if it's edge-concentrated
                    edge_nodata = (
                        band_mask[0, :].sum()
                        + band_mask[-1, :].sum()
                        + band_mask[:, 0].sum()
                        + band_mask[:, -1].sum()
                    )
                    total_edge = 2 * (band_mask.shape[0] + band_mask.shape[1])
                    if edge_nodata / total_edge > 0.5:
                        flags.append(
                            QualityFlag(
                                issue=QualityIssue.INCOMPLETE_COVERAGE,
                                severity=0.3,
                                affected_area_percent=nodata_percent,
                                description=f"Band {band + 1} has edge nodata (likely scene boundary)",
                            )
                        )

        return score, flags, details

    def _assess_spatial(
        self, data: np.ndarray, width: int, height: int
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess spatial quality."""
        flags = []
        details = {"width": width, "height": height}

        score = 1.0

        # Check for edge artifacts
        edge_threshold = self.config.issue_thresholds.get("edge_artifact_pixels", 10)

        if data.ndim == 3:
            for band in range(min(data.shape[0], 3)):  # Check first 3 bands
                band_data = data[band].astype(float)

                # Check edge gradients
                if band_data.shape[0] > edge_threshold * 2 and band_data.shape[1] > edge_threshold * 2:
                    top_edge = band_data[:edge_threshold, :]
                    bottom_edge = band_data[-edge_threshold:, :]
                    left_edge = band_data[:, :edge_threshold]
                    right_edge = band_data[:, -edge_threshold:]

                    interior = band_data[edge_threshold:-edge_threshold, edge_threshold:-edge_threshold]

                    if interior.size > 0:
                        interior_std = np.nanstd(interior)
                        if interior_std > 0:
                            edge_deviation = max(
                                abs(np.nanmean(top_edge) - np.nanmean(interior)) / interior_std,
                                abs(np.nanmean(bottom_edge) - np.nanmean(interior)) / interior_std,
                                abs(np.nanmean(left_edge) - np.nanmean(interior)) / interior_std,
                                abs(np.nanmean(right_edge) - np.nanmean(interior)) / interior_std,
                            )

                            if edge_deviation > 3.0:
                                severity = min((edge_deviation - 3.0) / 5.0, 1.0)
                                score *= 1.0 - (severity * 0.2)
                                flags.append(
                                    QualityFlag(
                                        issue=QualityIssue.EDGE_ARTIFACTS,
                                        severity=severity,
                                        affected_area_percent=edge_threshold * 4 * 100 / min(width, height),
                                        description=f"Band {band + 1} shows edge artifacts",
                                        recommendation="Consider masking edge pixels",
                                    )
                                )
                                break

        details["score_factors"] = {"edge_artifacts": score}
        return max(0.0, score), flags, details

    def _assess_temporal(
        self, acquisition_time: Optional[datetime]
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess temporal quality."""
        flags = []
        details = {}

        if acquisition_time is None:
            flags.append(
                QualityFlag(
                    issue=QualityIssue.TIMESTAMP_MISSING,
                    severity=0.3,
                    affected_area_percent=100.0,
                    description="Acquisition timestamp not available",
                )
            )
            return 0.7, flags, {"acquisition_time": None}

        # Ensure both are timezone-aware for comparison
        now = datetime.now(timezone.utc)
        if acquisition_time.tzinfo is None:
            acquisition_time = acquisition_time.replace(tzinfo=timezone.utc)

        age_days = (now - acquisition_time).days
        details["acquisition_time"] = acquisition_time.isoformat()
        details["age_days"] = age_days

        staleness_threshold = self.config.issue_thresholds.get("temporal_staleness_days", 30)

        if age_days > staleness_threshold:
            severity = min((age_days - staleness_threshold) / (staleness_threshold * 2), 1.0)
            score = 1.0 - (severity * 0.5)
            flags.append(
                QualityFlag(
                    issue=QualityIssue.STALE_DATA,
                    severity=severity,
                    affected_area_percent=100.0,
                    description=f"Data is {age_days} days old (threshold: {staleness_threshold} days)",
                    recommendation="Consider acquiring more recent data",
                )
            )
        else:
            score = 1.0

        return score, flags, details

    def _assess_radiometric(
        self, data: np.ndarray, dtype: str
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess radiometric quality."""
        flags = []
        details = {"dtype": dtype}
        score = 1.0

        # Get dtype info for saturation detection
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            min_val, max_val = info.min, info.max
        else:
            min_val, max_val = np.finfo(data.dtype).min, np.finfo(data.dtype).max

        saturation_threshold = self.config.issue_thresholds.get("saturation_percent", 1.0)
        noise_snr_min = self.config.issue_thresholds.get("noise_snr_min", 10.0)

        band_details = []

        for band in range(data.shape[0] if data.ndim == 3 else 1):
            band_data = data[band] if data.ndim == 3 else data
            band_info = {}

            # Check saturation
            if np.issubdtype(data.dtype, np.integer):
                saturated_high = (band_data == max_val).sum()
                saturated_low = (band_data == min_val).sum()
                total_pixels = band_data.size
                saturation_percent = (saturated_high + saturated_low) / total_pixels * 100

                band_info["saturation_percent"] = saturation_percent

                if saturation_percent > saturation_threshold:
                    severity = min(saturation_percent / 10.0, 1.0)
                    score *= 1.0 - (severity * 0.3)
                    flags.append(
                        QualityFlag(
                            issue=QualityIssue.SATURATION,
                            severity=severity,
                            affected_area_percent=saturation_percent,
                            description=f"Band {band + 1}: {saturation_percent:.2f}% saturated pixels",
                            recommendation="Check exposure settings or use alternative data",
                        )
                    )

            # Estimate noise (using local variance method)
            # High-pass filter approximation
            if band_data.size > 100:
                # Simple noise estimation using difference from local mean
                valid_data = band_data[np.isfinite(band_data.astype(float))]
                if len(valid_data) > 10:
                    signal_mean = np.mean(valid_data)
                    noise_std = np.std(valid_data - np.median(valid_data))
                    if noise_std > 0 and signal_mean != 0:
                        snr = abs(signal_mean) / noise_std
                        band_info["snr"] = snr

                        if snr < noise_snr_min:
                            severity = min((noise_snr_min - snr) / noise_snr_min, 1.0)
                            score *= 1.0 - (severity * 0.2)
                            flags.append(
                                QualityFlag(
                                    issue=QualityIssue.HIGH_NOISE,
                                    severity=severity,
                                    affected_area_percent=100.0,
                                    description=f"Band {band + 1}: Low SNR ({snr:.1f})",
                                    recommendation="Apply noise filtering or use alternative data",
                                )
                            )

            # Check for striping (banding artifacts)
            if band_data.shape[0] > 10:
                row_means = np.nanmean(band_data.astype(float), axis=1)
                col_means = np.nanmean(band_data.astype(float), axis=0)

                # Detect periodic patterns
                row_std = np.nanstd(row_means)
                col_std = np.nanstd(col_means)
                overall_std = np.nanstd(valid_data) if len(valid_data) > 0 else 0

                if overall_std > 0:
                    striping_score = max(row_std, col_std) / overall_std
                    band_info["striping_score"] = striping_score

                    striping_threshold = self.config.issue_thresholds.get("striping_score", 0.1)
                    if striping_score > striping_threshold:
                        severity = min((striping_score - striping_threshold) / 0.5, 1.0)
                        score *= 1.0 - (severity * 0.15)
                        flags.append(
                            QualityFlag(
                                issue=QualityIssue.STRIPING,
                                severity=severity,
                                affected_area_percent=100.0,
                                description=f"Band {band + 1}: Striping artifacts detected",
                                recommendation="Apply destriping filter",
                            )
                        )

            band_details.append(band_info)

        details["bands"] = band_details
        return max(0.0, score), flags, details

    def _assess_geometric(
        self, crs: Optional[str], bounds: Optional[Any]
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess geometric quality."""
        flags = []
        details = {"crs": crs}
        score = 1.0

        if crs is None:
            flags.append(
                QualityFlag(
                    issue=QualityIssue.PROJECTION_ERROR,
                    severity=0.5,
                    affected_area_percent=100.0,
                    description="Missing coordinate reference system",
                    recommendation="Define CRS before using in analysis",
                )
            )
            score = 0.5

        if bounds:
            details["bounds"] = {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            }

            # Check for suspicious bounds (e.g., 0,0 origin)
            if bounds.left == 0 and bounds.bottom == 0:
                flags.append(
                    QualityFlag(
                        issue=QualityIssue.GEOREFERENCING_ERROR,
                        severity=0.4,
                        affected_area_percent=100.0,
                        description="Bounds start at 0,0 (possibly missing geotransform)",
                    )
                )
                score *= 0.7

        return score, flags, details

    def _assess_atmospheric(
        self, cloud_mask: np.ndarray
    ) -> Tuple[float, List[QualityFlag], Dict[str, Any]]:
        """Assess atmospheric quality using cloud mask."""
        flags = []
        details = {}

        cloud_percent = cloud_mask.sum() / cloud_mask.size * 100
        details["cloud_cover_percent"] = cloud_percent

        # Score inversely proportional to cloud cover
        score = 1.0 - (cloud_percent / 100.0)

        if cloud_percent > self.config.max_cloud_cover:
            severity = min((cloud_percent - self.config.max_cloud_cover) / 50.0, 1.0)
            flags.append(
                QualityFlag(
                    issue=QualityIssue.CLOUD_COVER,
                    severity=severity,
                    affected_area_percent=cloud_percent,
                    description=f"{cloud_percent:.1f}% cloud cover (max: {self.config.max_cloud_cover}%)",
                    recommendation="Use SAR data or wait for clearer conditions",
                )
            )

        return max(0.0, score), flags, details

    def _generate_recommendations(
        self, flags: List[QualityFlag], dimension_scores: List[DimensionScore]
    ) -> List[str]:
        """Generate actionable recommendations based on flags and scores."""
        recommendations = []

        # Add recommendations from flags
        for flag in flags:
            if flag.recommendation and flag.severity > 0.3:
                if flag.recommendation not in recommendations:
                    recommendations.append(flag.recommendation)

        # Add dimension-specific recommendations
        for ds in dimension_scores:
            if ds.score < 0.5:
                if ds.dimension == QualityDimension.COMPLETENESS:
                    rec = "Consider alternative data source with better coverage"
                elif ds.dimension == QualityDimension.RADIOMETRIC:
                    rec = "Apply radiometric corrections or filtering"
                elif ds.dimension == QualityDimension.TEMPORAL:
                    rec = "Acquire more recent data if timeliness is critical"
                elif ds.dimension == QualityDimension.ATMOSPHERIC:
                    rec = "Consider radar-based alternatives for cloud-free coverage"
                else:
                    rec = None

                if rec and rec not in recommendations:
                    recommendations.append(rec)

        return recommendations


def assess_quality(
    path: Union[str, Path],
    min_valid_percent: float = 80.0,
    max_cloud_cover: float = 30.0,
) -> QualitySummary:
    """
    Convenience function to assess quality of a raster file.

    Args:
        path: Path to raster file
        min_valid_percent: Minimum acceptable valid data percentage
        max_cloud_cover: Maximum acceptable cloud cover percentage

    Returns:
        QualitySummary with comprehensive quality assessment
    """
    config = QualityConfig(
        min_valid_percent=min_valid_percent,
        max_cloud_cover=max_cloud_cover,
    )
    assessor = QualityAssessor(config)
    return assessor.assess(path)
