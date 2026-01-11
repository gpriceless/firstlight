"""
Historical Baseline Validation for Quality Control.

Validates analysis outputs against historical baselines to detect anomalies
and verify consistency with expected patterns, including:
- Comparison to historical statistics (mean, std, extremes)
- Seasonal pattern validation
- Trend deviation analysis
- Return period estimation
- Climatological context assessment

Key Concepts:
- Historical baselines provide expected behavior benchmarks
- Anomalies may indicate real events OR processing errors
- Seasonal context affects what constitutes "normal"
- Long-term trends inform deviation significance
- Return periods quantify event rarity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of historical anomalies."""
    ABOVE_NORMAL = "above_normal"           # Value above historical normal
    BELOW_NORMAL = "below_normal"           # Value below historical normal
    EXTREME_HIGH = "extreme_high"           # Value exceeds historical max
    EXTREME_LOW = "extreme_low"             # Value below historical min
    TREND_DEVIATION = "trend_deviation"     # Deviates from long-term trend
    SEASONAL_ANOMALY = "seasonal_anomaly"   # Unexpected for this season
    PATTERN_BREAK = "pattern_break"         # Breaks established pattern


class AnomalySeverity(Enum):
    """Severity of detected anomalies."""
    CRITICAL = "critical"     # Outside all historical bounds (likely error)
    HIGH = "high"             # Exceeds extreme percentiles
    MODERATE = "moderate"     # Outside normal range but within extremes
    LOW = "low"               # Minor deviation from normal
    NORMAL = "normal"         # Within expected range


class HistoricalMetric(Enum):
    """Metrics for historical comparison."""
    PERCENTILE = "percentile"       # Where current value falls in distribution
    Z_SCORE = "z_score"             # Standard deviations from mean
    RETURN_PERIOD = "return_period" # Expected recurrence interval
    TREND_RESIDUAL = "trend_residual"  # Deviation from trend line


@dataclass
class HistoricalBaseline:
    """
    Historical baseline statistics for comparison.

    Attributes:
        mean: Historical mean value
        std: Historical standard deviation
        min_value: Historical minimum
        max_value: Historical maximum
        percentiles: Dict of percentile values (e.g., {5: x, 25: y, 75: z, 95: w})
        sample_size: Number of historical observations
        period_years: Length of historical record
        seasonal_means: Optional monthly/seasonal means
        trend: Optional trend parameters (slope, intercept)
        metadata: Additional baseline metadata
    """
    mean: float
    std: float
    min_value: float
    max_value: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    sample_size: int = 0
    period_years: float = 0.0
    seasonal_means: Optional[Dict[int, float]] = None  # month -> mean
    seasonal_stds: Optional[Dict[int, float]] = None   # month -> std
    trend: Optional[Dict[str, float]] = None           # slope, intercept
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set default percentiles if not provided."""
        if not self.percentiles:
            self.percentiles = {
                5: self.mean - 1.645 * self.std,
                25: self.mean - 0.674 * self.std,
                50: self.mean,
                75: self.mean + 0.674 * self.std,
                95: self.mean + 1.645 * self.std,
            }

    @classmethod
    def from_data(
        cls,
        data: np.ndarray,
        period_years: float = 1.0,
        timestamps: Optional[List[datetime]] = None,
    ) -> "HistoricalBaseline":
        """Create baseline from historical data array."""
        valid_data = data[~np.isnan(data)] if data.dtype.kind == 'f' else data.flatten()

        if len(valid_data) == 0:
            return cls(
                mean=np.nan,
                std=np.nan,
                min_value=np.nan,
                max_value=np.nan,
                sample_size=0,
                period_years=period_years,
            )

        percentiles = {
            p: float(np.percentile(valid_data, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        }

        baseline = cls(
            mean=float(np.mean(valid_data)),
            std=float(np.std(valid_data)),
            min_value=float(np.min(valid_data)),
            max_value=float(np.max(valid_data)),
            percentiles=percentiles,
            sample_size=len(valid_data),
            period_years=period_years,
        )

        # Calculate seasonal means if timestamps provided
        if timestamps and len(timestamps) == len(data):
            seasonal_data = {}
            for ts, val in zip(timestamps, data.flatten()):
                month = ts.month
                if month not in seasonal_data:
                    seasonal_data[month] = []
                if not np.isnan(val):
                    seasonal_data[month].append(val)

            baseline.seasonal_means = {
                m: float(np.mean(vals)) for m, vals in seasonal_data.items()
                if len(vals) > 0
            }
            baseline.seasonal_stds = {
                m: float(np.std(vals)) for m, vals in seasonal_data.items()
                if len(vals) > 1
            }

        return baseline

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
            "percentiles": self.percentiles,
            "sample_size": self.sample_size,
            "period_years": self.period_years,
            "has_seasonal": self.seasonal_means is not None,
            "has_trend": self.trend is not None,
            "metadata": self.metadata,
        }


@dataclass
class HistoricalAnomaly:
    """
    A detected historical anomaly.

    Attributes:
        anomaly_type: Type of anomaly
        severity: Anomaly severity
        description: Human-readable description
        metric_name: Name of the metric being compared
        current_value: Current observed value
        expected_value: Expected value from baseline
        deviation: Deviation from expected (current - expected)
        z_score: Number of standard deviations from mean
        percentile: Percentile of current value in historical distribution
        return_period: Estimated return period in years
        location: Spatial location if applicable
    """
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    metric_name: str
    current_value: float
    expected_value: float
    deviation: float
    z_score: float
    percentile: float
    return_period: Optional[float] = None
    location: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "metric_name": self.metric_name,
            "current_value": round(self.current_value, 4),
            "expected_value": round(self.expected_value, 4),
            "deviation": round(self.deviation, 4),
            "z_score": round(self.z_score, 2),
            "percentile": round(self.percentile, 1),
            "return_period": round(self.return_period, 1) if self.return_period else None,
            "location": self.location,
        }


@dataclass
class HistoricalConfig:
    """
    Configuration for historical baseline validation.

    Attributes:
        use_seasonal: Use seasonal baselines if available
        use_trend: Adjust for long-term trend
        z_score_thresholds: Thresholds for severity classification
        percentile_thresholds: Percentile thresholds for anomaly detection
        min_sample_size: Minimum historical samples required
        calculate_return_periods: Estimate return periods
    """
    use_seasonal: bool = True
    use_trend: bool = True
    z_score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low": 1.0,
        "moderate": 2.0,
        "high": 3.0,
        "critical": 4.0,
    })
    percentile_thresholds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "normal": (10, 90),
        "moderate": (5, 95),
        "high": (1, 99),
    })
    min_sample_size: int = 10
    calculate_return_periods: bool = True


@dataclass
class HistoricalResult:
    """
    Result from historical baseline validation.

    Attributes:
        anomalies: List of detected anomalies
        summary_stats: Summary statistics
        overall_severity: Overall anomaly severity
        confidence: Confidence in the assessment
        in_historical_bounds: Whether within historical bounds
        context: Contextual information
        diagnostics: Additional diagnostics
    """
    anomalies: List[HistoricalAnomaly]
    summary_stats: Dict[str, float]
    overall_severity: AnomalySeverity
    confidence: float
    in_historical_bounds: bool
    context: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_anomalies": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
            "summary_stats": {k: round(v, 4) for k, v in self.summary_stats.items()},
            "overall_severity": self.overall_severity.value,
            "confidence": round(self.confidence, 4),
            "in_historical_bounds": self.in_historical_bounds,
            "context": self.context,
            "diagnostics": self.diagnostics,
        }


class HistoricalValidator:
    """
    Validates analysis outputs against historical baselines.

    Compares current observations to historical statistics to:
    - Detect anomalies and outliers
    - Assess seasonal appropriateness
    - Evaluate trend consistency
    - Estimate event rarity (return periods)
    """

    def __init__(self, config: Optional[HistoricalConfig] = None):
        """
        Initialize historical validator.

        Args:
            config: Validation configuration
        """
        self.config = config or HistoricalConfig()

    def validate(
        self,
        current_data: np.ndarray,
        baseline: HistoricalBaseline,
        observation_date: Optional[datetime] = None,
        metric_name: str = "value",
    ) -> HistoricalResult:
        """
        Validate current data against historical baseline.

        Args:
            current_data: Current observation data
            baseline: Historical baseline for comparison
            observation_date: Date of current observation (for seasonal adjustment)
            metric_name: Name of the metric being validated

        Returns:
            HistoricalResult with validation details
        """
        # Check baseline validity
        if baseline.sample_size < self.config.min_sample_size:
            return HistoricalResult(
                anomalies=[],
                summary_stats={},
                overall_severity=AnomalySeverity.NORMAL,
                confidence=0.0,
                in_historical_bounds=True,
                diagnostics={"error": "Insufficient historical data"},
            )

        # Compute summary statistics of current data
        valid_data = current_data[~np.isnan(current_data)] if current_data.dtype.kind == 'f' else current_data.flatten()

        if len(valid_data) == 0:
            return HistoricalResult(
                anomalies=[],
                summary_stats={},
                overall_severity=AnomalySeverity.NORMAL,
                confidence=0.0,
                in_historical_bounds=True,
                diagnostics={"error": "No valid current data"},
            )

        current_mean = float(np.mean(valid_data))
        current_max = float(np.max(valid_data))
        current_min = float(np.min(valid_data))

        # Get expected value (with seasonal adjustment if applicable)
        expected, expected_std = self._get_expected_value(
            baseline, observation_date
        )

        # Compute metrics
        summary_stats = self._compute_summary_stats(
            current_mean, current_max, current_min,
            baseline, expected, expected_std
        )

        # Detect anomalies
        anomalies = self._detect_anomalies(
            current_mean, current_max, current_min,
            baseline, expected, expected_std,
            observation_date, metric_name
        )

        # Check bounds
        in_bounds = (
            current_min >= baseline.min_value * 0.9 and
            current_max <= baseline.max_value * 1.1
        )

        # Classify overall severity
        overall_severity = self._classify_overall_severity(anomalies)

        # Calculate confidence
        confidence = self._calculate_confidence(
            baseline, observation_date
        )

        # Build context
        context = self._build_context(
            baseline, observation_date, expected, expected_std
        )

        return HistoricalResult(
            anomalies=anomalies,
            summary_stats=summary_stats,
            overall_severity=overall_severity,
            confidence=confidence,
            in_historical_bounds=in_bounds,
            context=context,
            diagnostics={
                "current_mean": current_mean,
                "current_min": current_min,
                "current_max": current_max,
                "expected_value": expected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def validate_spatial(
        self,
        current_data: np.ndarray,
        baseline_data: np.ndarray,
        metric_name: str = "value",
    ) -> HistoricalResult:
        """
        Validate current spatial data against historical spatial baseline.

        Args:
            current_data: Current observation array
            baseline_data: Historical baseline array (stacked observations)
            metric_name: Name of the metric being validated

        Returns:
            HistoricalResult with pixel-wise validation
        """
        # Calculate per-pixel statistics from baseline
        if baseline_data.ndim == 2:
            # Single baseline image
            baseline_mean = baseline_data
            baseline_std = np.zeros_like(baseline_data)
        else:
            # Stack of historical images
            with np.errstate(invalid='ignore'):
                baseline_mean = np.nanmean(baseline_data, axis=0)
                baseline_std = np.nanstd(baseline_data, axis=0)

        # Calculate z-scores
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = np.where(
                baseline_std > 1e-10,
                (current_data - baseline_mean) / baseline_std,
                0.0
            )
            z_scores = np.where(np.isnan(z_scores), 0.0, z_scores)

        # Find anomalous regions
        anomalies = []
        thresholds = self.config.z_score_thresholds

        for severity_name, threshold in [
            ("critical", thresholds["critical"]),
            ("high", thresholds["high"]),
            ("moderate", thresholds["moderate"]),
        ]:
            high_mask = z_scores > threshold
            low_mask = z_scores < -threshold

            if np.any(high_mask):
                anomalies.append(self._create_spatial_anomaly(
                    z_scores, high_mask, AnomalyType.ABOVE_NORMAL,
                    severity_name, metric_name, baseline_mean, current_data
                ))

            if np.any(low_mask):
                anomalies.append(self._create_spatial_anomaly(
                    z_scores, low_mask, AnomalyType.BELOW_NORMAL,
                    severity_name, metric_name, baseline_mean, current_data
                ))

        # Summary statistics
        summary_stats = {
            "mean_z_score": float(np.nanmean(z_scores)),
            "max_z_score": float(np.nanmax(z_scores)),
            "min_z_score": float(np.nanmin(z_scores)),
            "pct_above_2sigma": float(np.mean(np.abs(z_scores) > 2) * 100),
            "pct_above_3sigma": float(np.mean(np.abs(z_scores) > 3) * 100),
        }

        overall_severity = self._classify_overall_severity(anomalies)
        in_bounds = float(np.mean(np.abs(z_scores) <= thresholds["high"])) > 0.95

        return HistoricalResult(
            anomalies=anomalies,
            summary_stats=summary_stats,
            overall_severity=overall_severity,
            confidence=0.8,  # Default for spatial comparison
            in_historical_bounds=in_bounds,
            diagnostics={
                "z_score_shape": list(z_scores.shape),
                "baseline_samples": baseline_data.shape[0] if baseline_data.ndim > 2 else 1,
            },
        )

    def _get_expected_value(
        self,
        baseline: HistoricalBaseline,
        observation_date: Optional[datetime],
    ) -> Tuple[float, float]:
        """Get expected value and std, with seasonal adjustment."""
        expected = baseline.mean
        expected_std = baseline.std

        # Apply seasonal adjustment
        if (self.config.use_seasonal and
            observation_date and
            baseline.seasonal_means):
            month = observation_date.month
            if month in baseline.seasonal_means:
                expected = baseline.seasonal_means[month]
            if baseline.seasonal_stds and month in baseline.seasonal_stds:
                expected_std = baseline.seasonal_stds[month]

        # Apply trend adjustment
        if (self.config.use_trend and
            observation_date and
            baseline.trend):
            # Adjust for trend (years since baseline start)
            years_offset = baseline.period_years / 2  # assume middle of period
            if "slope" in baseline.trend:
                expected += baseline.trend["slope"] * years_offset

        return expected, expected_std

    def _compute_summary_stats(
        self,
        current_mean: float,
        current_max: float,
        current_min: float,
        baseline: HistoricalBaseline,
        expected: float,
        expected_std: float,
    ) -> Dict[str, float]:
        """Compute summary statistics for comparison."""
        stats = {}

        # Z-scores
        if expected_std > 1e-10:
            stats["z_score_mean"] = (current_mean - expected) / expected_std
            stats["z_score_max"] = (current_max - expected) / expected_std
            stats["z_score_min"] = (current_min - expected) / expected_std
        else:
            stats["z_score_mean"] = 0.0
            stats["z_score_max"] = 0.0
            stats["z_score_min"] = 0.0

        # Percentiles (where current value falls)
        stats["percentile_mean"] = self._calculate_percentile(
            current_mean, baseline
        )
        stats["percentile_max"] = self._calculate_percentile(
            current_max, baseline
        )

        # Deviation from expected
        stats["deviation_mean"] = current_mean - expected
        stats["deviation_pct"] = (
            ((current_mean - expected) / expected * 100)
            if abs(expected) > 1e-10 else 0.0
        )

        return stats

    def _calculate_percentile(
        self,
        value: float,
        baseline: HistoricalBaseline,
    ) -> float:
        """Calculate percentile of value in historical distribution."""
        # Handle NaN/Inf input
        if np.isnan(value) or np.isinf(value):
            return 50.0

        # Use available percentiles to interpolate
        if not baseline.percentiles:
            # Assume normal distribution
            if baseline.std > 1e-10:
                from scipy import stats
                z = (value - baseline.mean) / baseline.std
                return float(np.clip(stats.norm.cdf(z) * 100, 0.0, 100.0))
            return 50.0

        # Linear interpolation between known percentiles
        sorted_pcts = sorted(baseline.percentiles.items())
        for i, (p, v) in enumerate(sorted_pcts):
            if value <= v:
                if i == 0:
                    # Below first percentile - interpolate from 0
                    denom = v - baseline.min_value
                    if abs(denom) < 1e-10:
                        result = float(p) / 2.0 if value <= baseline.min_value else float(p)
                    else:
                        result = float(p) * (value - baseline.min_value) / denom
                    return float(np.clip(result, 0.0, 100.0))
                prev_p, prev_v = sorted_pcts[i - 1]
                denom = v - prev_v
                if abs(denom) < 1e-10:
                    result = float(prev_p + p) / 2.0
                else:
                    result = float(prev_p + (p - prev_p) * (value - prev_v) / denom)
                return float(np.clip(result, 0.0, 100.0))

        # Above highest percentile
        last_p, last_v = sorted_pcts[-1]
        denom = baseline.max_value - last_v
        if abs(denom) < 1e-10:
            result = float(last_p + 100) / 2.0 if value >= baseline.max_value else float(last_p)
        else:
            result = float(last_p + (100 - last_p) * (value - last_v) / denom)
        return float(np.clip(result, 0.0, 100.0))

    def _detect_anomalies(
        self,
        current_mean: float,
        current_max: float,
        current_min: float,
        baseline: HistoricalBaseline,
        expected: float,
        expected_std: float,
        observation_date: Optional[datetime],
        metric_name: str,
    ) -> List[HistoricalAnomaly]:
        """Detect anomalies in current data."""
        anomalies = []
        thresholds = self.config.z_score_thresholds

        # Check mean value
        if expected_std > 1e-10:
            z_score_mean = (current_mean - expected) / expected_std
            pct_mean = self._calculate_percentile(current_mean, baseline)

            anomaly = self._check_value_anomaly(
                current_mean, expected, z_score_mean, pct_mean,
                baseline, metric_name + "_mean", thresholds
            )
            if anomaly:
                anomalies.append(anomaly)

        # Check maximum value
        if current_max > baseline.max_value:
            z_score = (current_max - baseline.mean) / baseline.std if baseline.std > 1e-10 else 0
            return_period = self._estimate_return_period(z_score, baseline) if self.config.calculate_return_periods else None

            anomalies.append(HistoricalAnomaly(
                anomaly_type=AnomalyType.EXTREME_HIGH,
                severity=AnomalySeverity.HIGH,
                description=f"Maximum value {current_max:.4f} exceeds historical maximum {baseline.max_value:.4f}",
                metric_name=metric_name + "_max",
                current_value=current_max,
                expected_value=baseline.max_value,
                deviation=current_max - baseline.max_value,
                z_score=z_score,
                percentile=100.0,
                return_period=return_period,
            ))

        # Check minimum value
        if current_min < baseline.min_value:
            z_score = (current_min - baseline.mean) / baseline.std if baseline.std > 1e-10 else 0

            anomalies.append(HistoricalAnomaly(
                anomaly_type=AnomalyType.EXTREME_LOW,
                severity=AnomalySeverity.HIGH,
                description=f"Minimum value {current_min:.4f} below historical minimum {baseline.min_value:.4f}",
                metric_name=metric_name + "_min",
                current_value=current_min,
                expected_value=baseline.min_value,
                deviation=current_min - baseline.min_value,
                z_score=z_score,
                percentile=0.0,
            ))

        # Check for seasonal anomaly
        if (self.config.use_seasonal and
            observation_date and
            baseline.seasonal_means):
            seasonal_anomaly = self._check_seasonal_anomaly(
                current_mean, baseline, observation_date, metric_name
            )
            if seasonal_anomaly:
                anomalies.append(seasonal_anomaly)

        return anomalies

    def _check_value_anomaly(
        self,
        value: float,
        expected: float,
        z_score: float,
        percentile: float,
        baseline: HistoricalBaseline,
        metric_name: str,
        thresholds: Dict[str, float],
    ) -> Optional[HistoricalAnomaly]:
        """Check if a value is anomalous."""
        abs_z = abs(z_score)

        # Determine severity
        if abs_z >= thresholds["critical"]:
            severity = AnomalySeverity.CRITICAL
        elif abs_z >= thresholds["high"]:
            severity = AnomalySeverity.HIGH
        elif abs_z >= thresholds["moderate"]:
            severity = AnomalySeverity.MODERATE
        elif abs_z >= thresholds["low"]:
            severity = AnomalySeverity.LOW
        else:
            return None  # Not anomalous

        # Determine type
        if z_score > 0:
            if value > baseline.max_value:
                anomaly_type = AnomalyType.EXTREME_HIGH
            else:
                anomaly_type = AnomalyType.ABOVE_NORMAL
        else:
            if value < baseline.min_value:
                anomaly_type = AnomalyType.EXTREME_LOW
            else:
                anomaly_type = AnomalyType.BELOW_NORMAL

        # Return period
        return_period = None
        if self.config.calculate_return_periods:
            return_period = self._estimate_return_period(z_score, baseline)

        return HistoricalAnomaly(
            anomaly_type=anomaly_type,
            severity=severity,
            description=f"Value {value:.4f} is {abs_z:.1f} standard deviations from expected {expected:.4f}",
            metric_name=metric_name,
            current_value=value,
            expected_value=expected,
            deviation=value - expected,
            z_score=z_score,
            percentile=percentile,
            return_period=return_period,
        )

    def _check_seasonal_anomaly(
        self,
        current_mean: float,
        baseline: HistoricalBaseline,
        observation_date: datetime,
        metric_name: str,
    ) -> Optional[HistoricalAnomaly]:
        """Check for seasonal anomaly."""
        month = observation_date.month

        if month not in baseline.seasonal_means:
            return None

        seasonal_mean = baseline.seasonal_means[month]
        seasonal_std = baseline.seasonal_stds.get(month, baseline.std) if baseline.seasonal_stds else baseline.std

        if seasonal_std < 1e-10:
            return None

        z_score = (current_mean - seasonal_mean) / seasonal_std
        abs_z = abs(z_score)

        thresholds = self.config.z_score_thresholds

        # More lenient thresholds for seasonal comparison
        if abs_z < thresholds["moderate"]:
            return None

        severity = AnomalySeverity.MODERATE
        if abs_z >= thresholds["high"]:
            severity = AnomalySeverity.HIGH

        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        return HistoricalAnomaly(
            anomaly_type=AnomalyType.SEASONAL_ANOMALY,
            severity=severity,
            description=f"Value {current_mean:.4f} unusual for {month_names[month]} (expected ~{seasonal_mean:.4f})",
            metric_name=metric_name + "_seasonal",
            current_value=current_mean,
            expected_value=seasonal_mean,
            deviation=current_mean - seasonal_mean,
            z_score=z_score,
            percentile=self._calculate_percentile(current_mean, baseline),
        )

    def _estimate_return_period(
        self,
        z_score: float,
        baseline: HistoricalBaseline,
    ) -> float:
        """Estimate return period in years for a given z-score."""
        # Using normal distribution approximation
        from scipy import stats

        # Probability of exceeding this z-score (one-tailed)
        if z_score > 0:
            prob = 1 - stats.norm.cdf(z_score)
        else:
            prob = stats.norm.cdf(z_score)

        if prob < 1e-10:
            return 10000.0  # Cap at 10000 years

        # Assume observation frequency based on baseline
        obs_per_year = max(1, baseline.sample_size / max(baseline.period_years, 1))

        return_period = 1 / (prob * obs_per_year)
        return min(return_period, 10000.0)  # Cap at 10000 years

    def _create_spatial_anomaly(
        self,
        z_scores: np.ndarray,
        mask: np.ndarray,
        anomaly_type: AnomalyType,
        severity_name: str,
        metric_name: str,
        baseline_mean: np.ndarray,
        current_data: np.ndarray,
    ) -> HistoricalAnomaly:
        """Create anomaly for spatial region."""
        severity_map = {
            "critical": AnomalySeverity.CRITICAL,
            "high": AnomalySeverity.HIGH,
            "moderate": AnomalySeverity.MODERATE,
        }

        affected_pct = np.sum(mask) / mask.size * 100
        mean_z = float(np.mean(z_scores[mask]))

        # Get location of anomalous region
        rows, cols = np.where(mask)
        if len(rows) > 0:
            location = {
                "bbox": {
                    "min_row": int(rows.min()),
                    "max_row": int(rows.max()),
                    "min_col": int(cols.min()),
                    "max_col": int(cols.max()),
                },
                "centroid": {
                    "row": float(np.mean(rows)),
                    "col": float(np.mean(cols)),
                },
                "affected_pixels": int(np.sum(mask)),
                "affected_pct": affected_pct,
            }
        else:
            location = None

        return HistoricalAnomaly(
            anomaly_type=anomaly_type,
            severity=severity_map[severity_name],
            description=f"{affected_pct:.1f}% of pixels show {anomaly_type.value} anomaly (mean z={mean_z:.1f})",
            metric_name=metric_name,
            current_value=float(np.mean(current_data[mask])),
            expected_value=float(np.mean(baseline_mean[mask])),
            deviation=float(np.mean(current_data[mask] - baseline_mean[mask])),
            z_score=mean_z,
            percentile=50.0 + mean_z * 34.0,  # Approximate
            location=location,
        )

    def _classify_overall_severity(
        self,
        anomalies: List[HistoricalAnomaly],
    ) -> AnomalySeverity:
        """Classify overall severity from anomaly list."""
        if not anomalies:
            return AnomalySeverity.NORMAL

        severities = [a.severity for a in anomalies]

        if AnomalySeverity.CRITICAL in severities:
            return AnomalySeverity.CRITICAL
        elif AnomalySeverity.HIGH in severities:
            return AnomalySeverity.HIGH
        elif AnomalySeverity.MODERATE in severities:
            return AnomalySeverity.MODERATE
        elif AnomalySeverity.LOW in severities:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.NORMAL

    def _calculate_confidence(
        self,
        baseline: HistoricalBaseline,
        observation_date: Optional[datetime],
    ) -> float:
        """Calculate confidence in the assessment."""
        confidence = 0.5  # Base confidence

        # Increase for larger sample size
        if baseline.sample_size >= 100:
            confidence += 0.2
        elif baseline.sample_size >= 30:
            confidence += 0.1

        # Increase for longer period
        if baseline.period_years >= 10:
            confidence += 0.15
        elif baseline.period_years >= 5:
            confidence += 0.1

        # Increase for seasonal adjustment
        if (observation_date and
            baseline.seasonal_means and
            observation_date.month in baseline.seasonal_means):
            confidence += 0.1

        return min(confidence, 1.0)

    def _build_context(
        self,
        baseline: HistoricalBaseline,
        observation_date: Optional[datetime],
        expected: float,
        expected_std: float,
    ) -> Dict[str, Any]:
        """Build context information."""
        context = {
            "baseline_period_years": baseline.period_years,
            "baseline_sample_size": baseline.sample_size,
            "expected_value": expected,
            "expected_std": expected_std,
        }

        if observation_date:
            context["observation_date"] = observation_date.isoformat()
            context["observation_month"] = observation_date.month

            if baseline.seasonal_means:
                context["seasonal_adjustment_applied"] = True
                context["seasonal_mean"] = baseline.seasonal_means.get(observation_date.month)

        return context


# Convenience functions

def validate_against_historical(
    current_data: np.ndarray,
    historical_data: np.ndarray,
    observation_date: Optional[datetime] = None,
    metric_name: str = "value",
) -> HistoricalResult:
    """
    Validate current data against historical observations.

    Args:
        current_data: Current observation data
        historical_data: Historical data array (can be 1D series or stack)
        observation_date: Date of current observation
        metric_name: Name of the metric

    Returns:
        HistoricalResult with validation details
    """
    # Create baseline from historical data
    baseline = HistoricalBaseline.from_data(historical_data)

    validator = HistoricalValidator()
    return validator.validate(current_data, baseline, observation_date, metric_name)


def calculate_z_score(
    value: float,
    baseline: HistoricalBaseline,
) -> float:
    """
    Calculate z-score for a value against baseline.

    Args:
        value: Value to check
        baseline: Historical baseline

    Returns:
        Z-score (standard deviations from mean)
    """
    if baseline.std < 1e-10:
        return 0.0
    return (value - baseline.mean) / baseline.std


def estimate_return_period(
    value: float,
    baseline: HistoricalBaseline,
) -> float:
    """
    Estimate return period for a value.

    Args:
        value: Value to check
        baseline: Historical baseline

    Returns:
        Estimated return period in years
    """
    z_score = calculate_z_score(value, baseline)
    validator = HistoricalValidator()
    return validator._estimate_return_period(z_score, baseline)
