"""
Uncertainty Quantification Metrics for Quality Control.

Provides tools for calculating and reporting uncertainty metrics in geospatial
analysis products. This module focuses on quality assessment uncertainty rather
than data fusion uncertainty (handled by core.analysis.fusion.uncertainty).

Key Concepts:
- Uncertainty metrics quantify confidence in quality assessments
- Multiple metrics capture different aspects of uncertainty
- Calibration checks ensure uncertainty estimates are reliable
- Integration with quality control gating decisions

Metrics Provided:
- Coefficient of variation (CV)
- Confidence intervals (parametric and bootstrap)
- Prediction intervals for new observations
- Entropy-based uncertainty
- Ensemble spread metrics
- Calibration metrics (reliability diagrams, Brier skill)

Example:
    from core.quality.uncertainty.quantification import (
        UncertaintyQuantifier,
        calculate_confidence_interval,
        calculate_prediction_interval,
    )

    # Calculate uncertainty metrics
    quantifier = UncertaintyQuantifier()
    metrics = quantifier.calculate_metrics(data, reference)
    print(f"CV: {metrics.coefficient_of_variation}")
    print(f"95% CI: {metrics.confidence_interval_95}")
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class UncertaintyMetricType(Enum):
    """Types of uncertainty metrics."""
    COEFFICIENT_OF_VARIATION = "cv"
    STANDARD_ERROR = "se"
    CONFIDENCE_INTERVAL = "ci"
    PREDICTION_INTERVAL = "pi"
    ENTROPY = "entropy"
    ENSEMBLE_SPREAD = "ensemble_spread"
    INTERQUARTILE_RANGE = "iqr"
    BRIER_SCORE = "brier"
    CALIBRATION_ERROR = "calibration_error"


class CalibrationMethod(Enum):
    """Methods for calibration assessment."""
    RELIABILITY_DIAGRAM = "reliability_diagram"
    HISTOGRAM_BINNING = "histogram_binning"
    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"


class ConfidenceLevel(Enum):
    """Standard confidence levels."""
    CL_68 = 0.68    # 1-sigma
    CL_90 = 0.90
    CL_95 = 0.95
    CL_99 = 0.99
    CL_999 = 0.999


@dataclass
class UncertaintyMetrics:
    """
    Collection of uncertainty metrics for a data product.

    Attributes:
        mean: Mean value of the data
        std: Standard deviation
        standard_error: Standard error of the mean
        coefficient_of_variation: CV (std / mean)
        confidence_interval_68: 68% (1-sigma) confidence interval
        confidence_interval_95: 95% confidence interval
        confidence_interval_99: 99% confidence interval
        prediction_interval_95: 95% prediction interval for new observations
        entropy: Shannon entropy (for classification)
        iqr: Interquartile range
        skewness: Distribution skewness
        kurtosis: Distribution kurtosis
        n_samples: Number of samples
        is_calibrated: Whether uncertainty is well-calibrated
        calibration_error: Expected calibration error
        metadata: Additional metrics metadata
    """
    mean: float
    std: float
    standard_error: float
    coefficient_of_variation: float
    confidence_interval_68: Tuple[float, float]
    confidence_interval_95: Tuple[float, float]
    confidence_interval_99: Tuple[float, float]
    prediction_interval_95: Tuple[float, float]
    entropy: Optional[float] = None
    iqr: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    n_samples: int = 0
    is_calibrated: bool = True
    calibration_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "standard_error": self.standard_error,
            "coefficient_of_variation": self.coefficient_of_variation,
            "confidence_interval_68": list(self.confidence_interval_68),
            "confidence_interval_95": list(self.confidence_interval_95),
            "confidence_interval_99": list(self.confidence_interval_99),
            "prediction_interval_95": list(self.prediction_interval_95),
            "entropy": self.entropy,
            "iqr": self.iqr,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "n_samples": self.n_samples,
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "metadata": self.metadata,
        }


@dataclass
class EnsembleUncertainty:
    """
    Uncertainty from ensemble of predictions.

    Attributes:
        ensemble_mean: Mean of ensemble members
        ensemble_std: Standard deviation across members
        ensemble_spread: Spread metric (max - min)
        member_count: Number of ensemble members
        agreement_fraction: Fraction of members agreeing on classification
        entropy: Entropy of ensemble distribution
        confidence_intervals: CIs at different levels
        outlier_count: Number of outlier ensemble members
    """
    ensemble_mean: Union[float, np.ndarray]
    ensemble_std: Union[float, np.ndarray]
    ensemble_spread: Union[float, np.ndarray]
    member_count: int
    agreement_fraction: Optional[float] = None
    entropy: Optional[Union[float, np.ndarray]] = None
    confidence_intervals: Dict[float, Tuple[float, float]] = field(default_factory=dict)
    outlier_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_mean": float(np.mean(self.ensemble_mean)) if isinstance(self.ensemble_mean, np.ndarray) else self.ensemble_mean,
            "ensemble_std": float(np.mean(self.ensemble_std)) if isinstance(self.ensemble_std, np.ndarray) else self.ensemble_std,
            "ensemble_spread": float(np.mean(self.ensemble_spread)) if isinstance(self.ensemble_spread, np.ndarray) else self.ensemble_spread,
            "member_count": self.member_count,
            "agreement_fraction": self.agreement_fraction,
            "entropy": float(np.mean(self.entropy)) if isinstance(self.entropy, np.ndarray) else self.entropy,
            "confidence_intervals": {str(k): list(v) for k, v in self.confidence_intervals.items()},
            "outlier_count": self.outlier_count,
        }


@dataclass
class CalibrationResult:
    """
    Results from calibration assessment.

    Attributes:
        is_calibrated: Whether predictions are well-calibrated
        expected_calibration_error: ECE metric
        maximum_calibration_error: MCE metric
        brier_score: Brier score for probabilistic predictions
        reliability_bins: Binned reliability data
        sharpness: Prediction sharpness (spread)
        resolution: Resolution component of Brier decomposition
        reliability: Reliability component of Brier decomposition
        refinement_suggestions: Suggestions for improving calibration
    """
    is_calibrated: bool
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: Optional[float] = None
    reliability_bins: Optional[Dict[str, List[float]]] = None
    sharpness: float = 0.0
    resolution: float = 0.0
    reliability: float = 0.0
    refinement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "brier_score": self.brier_score,
            "reliability_bins": self.reliability_bins,
            "sharpness": self.sharpness,
            "resolution": self.resolution,
            "reliability": self.reliability,
            "refinement_suggestions": self.refinement_suggestions,
        }


@dataclass
class QuantificationConfig:
    """
    Configuration for uncertainty quantification.

    Attributes:
        confidence_levels: List of confidence levels to compute
        bootstrap_samples: Number of bootstrap samples for non-parametric CIs
        min_samples: Minimum samples required for reliable metrics
        calibration_bins: Number of bins for calibration assessment
        calibration_threshold: ECE threshold for "well-calibrated"
        outlier_method: Method for outlier detection in ensembles
        outlier_threshold: Z-score threshold for outliers
    """
    confidence_levels: List[float] = field(default_factory=lambda: [0.68, 0.90, 0.95, 0.99])
    bootstrap_samples: int = 1000
    min_samples: int = 30
    calibration_bins: int = 10
    calibration_threshold: float = 0.1
    outlier_method: str = "zscore"
    outlier_threshold: float = 3.0


class UncertaintyQuantifier:
    """
    Calculates uncertainty metrics for quality assessment.

    Provides comprehensive uncertainty quantification including:
    - Standard statistical metrics (mean, std, SE, CV)
    - Confidence and prediction intervals
    - Distribution shape (skewness, kurtosis)
    - Ensemble uncertainty metrics
    - Calibration assessment
    """

    def __init__(self, config: Optional[QuantificationConfig] = None):
        """
        Initialize uncertainty quantifier.

        Args:
            config: Quantification configuration
        """
        self.config = config or QuantificationConfig()

    def calculate_metrics(
        self,
        data: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> UncertaintyMetrics:
        """
        Calculate comprehensive uncertainty metrics.

        Args:
            data: Data array (can be 1D, 2D, or ND)
            weights: Optional weights for weighted statistics

        Returns:
            UncertaintyMetrics with all calculated metrics
        """
        # Flatten for scalar statistics
        flat_data = data.flatten()
        valid_data = flat_data[np.isfinite(flat_data)]

        if len(valid_data) < self.config.min_samples:
            logger.warning(f"Insufficient samples ({len(valid_data)}) for reliable metrics")

        n = len(valid_data)
        if n == 0:
            return UncertaintyMetrics(
                mean=np.nan,
                std=np.nan,
                standard_error=np.nan,
                coefficient_of_variation=np.nan,
                confidence_interval_68=(np.nan, np.nan),
                confidence_interval_95=(np.nan, np.nan),
                confidence_interval_99=(np.nan, np.nan),
                prediction_interval_95=(np.nan, np.nan),
                n_samples=0,
            )

        # Basic statistics
        if weights is not None:
            flat_weights = weights.flatten()[np.isfinite(flat_data)]
            mean = np.average(valid_data, weights=flat_weights)
            variance = np.average((valid_data - mean) ** 2, weights=flat_weights)
            std = np.sqrt(variance)
        else:
            mean = np.mean(valid_data)
            std = np.std(valid_data, ddof=1) if n > 1 else 0.0

        # Standard error
        se = std / np.sqrt(n) if n > 0 else np.nan

        # Coefficient of variation (guard against zero mean)
        cv = std / abs(mean) if abs(mean) > 1e-10 else np.inf

        # Confidence intervals (assuming normal distribution)
        ci_68 = self._calculate_ci(mean, se, 0.68, n)
        ci_95 = self._calculate_ci(mean, se, 0.95, n)
        ci_99 = self._calculate_ci(mean, se, 0.99, n)

        # Prediction interval (for new observations)
        pi_95 = self._calculate_prediction_interval(mean, std, se, 0.95, n)

        # Distribution shape
        skewness = self._calculate_skewness(valid_data) if n > 2 else None
        kurtosis = self._calculate_kurtosis(valid_data) if n > 3 else None

        # IQR
        iqr = np.percentile(valid_data, 75) - np.percentile(valid_data, 25) if n > 0 else None

        return UncertaintyMetrics(
            mean=float(mean),
            std=float(std),
            standard_error=float(se),
            coefficient_of_variation=float(cv),
            confidence_interval_68=ci_68,
            confidence_interval_95=ci_95,
            confidence_interval_99=ci_99,
            prediction_interval_95=pi_95,
            skewness=float(skewness) if skewness is not None else None,
            kurtosis=float(kurtosis) if kurtosis is not None else None,
            iqr=float(iqr) if iqr is not None else None,
            n_samples=n,
        )

    def _calculate_ci(
        self,
        mean: float,
        se: float,
        confidence: float,
        n: int,
    ) -> Tuple[float, float]:
        """Calculate confidence interval for the mean."""
        if n <= 1 or not np.isfinite(se):
            return (np.nan, np.nan)

        # Use t-distribution for small samples
        from scipy import stats

        if n < 30:
            t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        else:
            # Use normal approximation for large n
            t_val = stats.norm.ppf((1 + confidence) / 2)

        margin = t_val * se
        return (float(mean - margin), float(mean + margin))

    def _calculate_prediction_interval(
        self,
        mean: float,
        std: float,
        se: float,
        confidence: float,
        n: int,
    ) -> Tuple[float, float]:
        """Calculate prediction interval for new observations."""
        if n <= 1:
            return (np.nan, np.nan)

        from scipy import stats

        # Prediction interval accounts for both estimation and observation variance
        if n < 30:
            t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        else:
            t_val = stats.norm.ppf((1 + confidence) / 2)

        # SE for prediction = sqrt(se^2 + std^2) = sqrt(std^2/n + std^2) = std * sqrt(1 + 1/n)
        se_pred = std * np.sqrt(1 + 1/n)
        margin = t_val * se_pred

        return (float(mean - margin), float(mean + margin))

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate sample skewness."""
        n = len(data)
        if n < 3:
            return np.nan

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-10:
            return 0.0

        m3 = np.mean((data - mean) ** 3)
        skew = m3 / (std ** 3)

        # Adjust for sample bias
        skew *= np.sqrt(n * (n - 1)) / (n - 2)

        return skew

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        n = len(data)
        if n < 4:
            return np.nan

        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std < 1e-10:
            return 0.0

        m4 = np.mean((data - mean) ** 4)
        kurt = m4 / (std ** 4) - 3  # Excess kurtosis

        # Fisher's adjustment
        kurt = ((n + 1) * kurt + 6) * (n - 1) / ((n - 2) * (n - 3))

        return kurt

    def calculate_ensemble_uncertainty(
        self,
        ensemble: List[np.ndarray],
        classification_threshold: Optional[float] = None,
    ) -> EnsembleUncertainty:
        """
        Calculate uncertainty metrics from ensemble predictions.

        Args:
            ensemble: List of ensemble member predictions
            classification_threshold: Threshold for binary classification (optional)

        Returns:
            EnsembleUncertainty with ensemble metrics
        """
        if not ensemble:
            raise ValueError("Empty ensemble provided")

        # Stack ensemble members
        stack = np.stack(ensemble)
        n_members = len(ensemble)

        # Basic statistics
        ensemble_mean = np.mean(stack, axis=0)
        ensemble_std = np.std(stack, axis=0, ddof=1) if n_members > 1 else np.zeros_like(ensemble_mean)
        ensemble_spread = np.max(stack, axis=0) - np.min(stack, axis=0)

        # Detect outliers
        outlier_count = 0
        if n_members > 3 and self.config.outlier_method == "zscore":
            member_means = np.array([np.nanmean(m) for m in ensemble])
            z_scores = (member_means - np.mean(member_means)) / (np.std(member_means) + 1e-10)
            outlier_count = int(np.sum(np.abs(z_scores) > self.config.outlier_threshold))

        # Confidence intervals
        confidence_intervals = {}
        for level in self.config.confidence_levels:
            alpha = 1 - level
            lower_pct = 100 * alpha / 2
            upper_pct = 100 * (1 - alpha / 2)
            ci_lower = float(np.percentile(stack, lower_pct))
            ci_upper = float(np.percentile(stack, upper_pct))
            confidence_intervals[level] = (ci_lower, ci_upper)

        # Agreement fraction (for classification)
        agreement_fraction = None
        if classification_threshold is not None:
            classifications = (stack > classification_threshold).astype(int)
            mode_class = (np.mean(classifications, axis=0) > 0.5).astype(int)
            agreement = np.mean(classifications == mode_class, axis=0)
            agreement_fraction = float(np.mean(agreement))

        # Entropy (treating ensemble as probability distribution)
        # Approximate with histogram for continuous values
        flat_mean = ensemble_mean.flatten()
        valid_mean = flat_mean[np.isfinite(flat_mean)]
        if len(valid_mean) > 0:
            # Normalize to [0, 1] for entropy calculation
            min_val = np.min(valid_mean)
            max_val = np.max(valid_mean)
            range_val = max_val - min_val
            if range_val > 1e-10:
                normalized = (ensemble_std.flatten()[np.isfinite(flat_mean)]) / range_val
                # Entropy based on relative uncertainty
                p = np.clip(normalized, 1e-10, 1)
                entropy = float(-np.mean(p * np.log(p)))
            else:
                entropy = 0.0
        else:
            entropy = None

        return EnsembleUncertainty(
            ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std,
            ensemble_spread=ensemble_spread,
            member_count=n_members,
            agreement_fraction=agreement_fraction,
            entropy=entropy,
            confidence_intervals=confidence_intervals,
            outlier_count=outlier_count,
        )

    def calculate_bootstrap_ci(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for any statistic.

        Args:
            data: Data array
            statistic_fn: Function that computes the statistic
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        flat_data = data.flatten()
        valid_data = flat_data[np.isfinite(flat_data)]
        n = len(valid_data)

        if n < self.config.min_samples:
            logger.warning(f"Insufficient samples ({n}) for bootstrap")

        # Bootstrap resampling
        bootstrap_stats = []
        for _ in range(self.config.bootstrap_samples):
            sample = np.random.choice(valid_data, size=n, replace=True)
            try:
                stat = statistic_fn(sample)
                if np.isfinite(stat):
                    bootstrap_stats.append(stat)
            except Exception:
                continue

        if len(bootstrap_stats) < 10:
            return (np.nan, np.nan)

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - confidence
        lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

        return (lower, upper)

    def calculate_entropy(
        self,
        probabilities: np.ndarray,
        base: float = 2.0,
    ) -> Union[float, np.ndarray]:
        """
        Calculate Shannon entropy from probability distribution.

        Args:
            probabilities: Probability values (should sum to 1 along axis 0)
            base: Logarithm base (2 for bits, e for nats)

        Returns:
            Entropy value(s)
        """
        # Ensure valid probabilities
        p = np.clip(probabilities, 1e-10, 1.0)

        # Normalize if needed
        if probabilities.ndim > 1:
            p_sum = np.sum(p, axis=0, keepdims=True)
            p_sum = np.where(p_sum > 1e-10, p_sum, 1.0)
            p = p / p_sum

        # Shannon entropy: H = -sum(p * log(p))
        if base == 2.0:
            entropy = -np.sum(p * np.log2(p), axis=0)
        elif base == np.e:
            entropy = -np.sum(p * np.log(p), axis=0)
        else:
            entropy = -np.sum(p * np.log(p), axis=0) / np.log(base)

        return entropy


class CalibrationAssessor:
    """
    Assesses calibration of probabilistic predictions.

    Calibration measures how well predicted probabilities match
    observed frequencies. Well-calibrated predictions have reliable
    uncertainty estimates.
    """

    def __init__(self, config: Optional[QuantificationConfig] = None):
        """
        Initialize calibration assessor.

        Args:
            config: Quantification configuration
        """
        self.config = config or QuantificationConfig()

    def assess_calibration(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
        is_probabilistic: bool = True,
    ) -> CalibrationResult:
        """
        Assess calibration of predictions against observations.

        Args:
            predictions: Predicted values or probabilities
            observations: Observed values (binary for probabilistic)
            is_probabilistic: Whether predictions are probabilities

        Returns:
            CalibrationResult with calibration metrics
        """
        # Flatten
        pred = predictions.flatten()
        obs = observations.flatten()

        # Filter valid pairs
        valid_mask = np.isfinite(pred) & np.isfinite(obs)
        pred = pred[valid_mask]
        obs = obs[valid_mask]

        if len(pred) < self.config.min_samples:
            return CalibrationResult(
                is_calibrated=False,
                expected_calibration_error=np.nan,
                maximum_calibration_error=np.nan,
                refinement_suggestions=["Insufficient samples for calibration assessment"],
            )

        if is_probabilistic:
            # Ensure probabilities are in [0, 1]
            pred = np.clip(pred, 0, 1)
            # Ensure observations are binary
            obs = (obs > 0.5).astype(float)

            ece, mce, reliability = self._compute_calibration_errors(pred, obs)
            brier = self._compute_brier_score(pred, obs)
            sharpness, resolution, reliability_component = self._brier_decomposition(pred, obs)
        else:
            # For continuous predictions, use normalized residuals
            residuals = obs - pred
            pred_std = np.std(pred)
            if pred_std > 1e-10:
                normalized_residuals = residuals / pred_std
                # Check if residuals are normally distributed with expected variance
                ece = abs(np.std(normalized_residuals) - 1.0)
                mce = abs(np.max(np.abs(normalized_residuals)) - 3.0) / 3.0  # Expect ~99.7% within 3 sigma
            else:
                ece = np.nan
                mce = np.nan
            brier = None
            reliability = None
            sharpness = resolution = reliability_component = 0.0

        # Determine if calibrated
        is_calibrated = ece < self.config.calibration_threshold

        # Generate suggestions
        suggestions = []
        if not is_calibrated:
            if is_probabilistic:
                suggestions.append("Consider recalibrating using Platt scaling or isotonic regression")
                if mce > 0.2:
                    suggestions.append("Large calibration errors in some bins - check for systematic bias")
            else:
                suggestions.append("Prediction uncertainty may be over/under-estimated")

        return CalibrationResult(
            is_calibrated=is_calibrated,
            expected_calibration_error=float(ece) if np.isfinite(ece) else np.nan,
            maximum_calibration_error=float(mce) if np.isfinite(mce) else np.nan,
            brier_score=float(brier) if brier is not None else None,
            reliability_bins=reliability,
            sharpness=float(sharpness),
            resolution=float(resolution),
            reliability=float(reliability_component),
            refinement_suggestions=suggestions,
        )

    def _compute_calibration_errors(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
    ) -> Tuple[float, float, Dict[str, List[float]]]:
        """Compute ECE and MCE from reliability diagram."""
        n_bins = self.config.calibration_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        reliability_data = {
            "bin_centers": bin_centers.tolist(),
            "predicted_frequency": [],
            "observed_frequency": [],
            "bin_counts": [],
        }

        ece = 0.0
        mce = 0.0
        total_samples = len(predictions)

        for i in range(n_bins):
            bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == n_bins - 1:  # Include upper bound for last bin
                bin_mask |= (predictions == bin_edges[i + 1])

            bin_count = np.sum(bin_mask)
            if bin_count > 0:
                mean_pred = np.mean(predictions[bin_mask])
                mean_obs = np.mean(observations[bin_mask])
                bin_error = abs(mean_pred - mean_obs)

                ece += (bin_count / total_samples) * bin_error
                mce = max(mce, bin_error)

                reliability_data["predicted_frequency"].append(float(mean_pred))
                reliability_data["observed_frequency"].append(float(mean_obs))
            else:
                reliability_data["predicted_frequency"].append(float(bin_centers[i]))
                reliability_data["observed_frequency"].append(np.nan)

            reliability_data["bin_counts"].append(int(bin_count))

        return ece, mce, reliability_data

    def _compute_brier_score(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
    ) -> float:
        """Compute Brier score."""
        return float(np.mean((predictions - observations) ** 2))

    def _brier_decomposition(
        self,
        predictions: np.ndarray,
        observations: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Decompose Brier score into reliability, resolution, and uncertainty.

        BS = Reliability - Resolution + Uncertainty
        """
        n = len(predictions)
        base_rate = np.mean(observations)
        uncertainty = base_rate * (1 - base_rate)

        # Bin-based decomposition
        n_bins = self.config.calibration_bins
        bin_edges = np.linspace(0, 1, n_bins + 1)

        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
            if i == n_bins - 1:
                bin_mask |= (predictions == bin_edges[i + 1])

            bin_count = np.sum(bin_mask)
            if bin_count > 0:
                mean_pred = np.mean(predictions[bin_mask])
                mean_obs = np.mean(observations[bin_mask])

                reliability += (bin_count / n) * (mean_pred - mean_obs) ** 2
                resolution += (bin_count / n) * (mean_obs - base_rate) ** 2

        sharpness = np.var(predictions)

        return sharpness, resolution, reliability


# Convenience functions

def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = "parametric",
) -> Tuple[float, float]:
    """
    Calculate confidence interval for data mean.

    Args:
        data: Data array
        confidence: Confidence level (0-1)
        method: "parametric" (t-distribution) or "bootstrap"

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    quantifier = UncertaintyQuantifier()

    if method == "bootstrap":
        return quantifier.calculate_bootstrap_ci(
            data,
            statistic_fn=np.mean,
            confidence=confidence,
        )
    else:
        metrics = quantifier.calculate_metrics(data)
        if confidence >= 0.99:
            return metrics.confidence_interval_99
        elif confidence >= 0.95:
            return metrics.confidence_interval_95
        else:
            return metrics.confidence_interval_68


def calculate_prediction_interval(
    data: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Calculate prediction interval for new observations.

    Args:
        data: Historical data
        confidence: Confidence level (0-1)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    quantifier = UncertaintyQuantifier()
    metrics = quantifier.calculate_metrics(data)
    return metrics.prediction_interval_95


def calculate_coefficient_of_variation(data: np.ndarray) -> float:
    """
    Calculate coefficient of variation (CV = std/mean).

    Args:
        data: Data array

    Returns:
        Coefficient of variation
    """
    flat_data = data.flatten()
    valid_data = flat_data[np.isfinite(flat_data)]

    if len(valid_data) == 0:
        return np.nan

    mean = np.mean(valid_data)
    std = np.std(valid_data, ddof=1)

    if abs(mean) < 1e-10:
        return np.inf

    return float(std / abs(mean))


def assess_calibration(
    predictions: np.ndarray,
    observations: np.ndarray,
    is_probabilistic: bool = True,
) -> CalibrationResult:
    """
    Assess calibration of predictions.

    Args:
        predictions: Predicted values/probabilities
        observations: Observed values
        is_probabilistic: Whether predictions are probabilities

    Returns:
        CalibrationResult with calibration metrics
    """
    assessor = CalibrationAssessor()
    return assessor.assess_calibration(predictions, observations, is_probabilistic)


def quantify_ensemble_uncertainty(
    ensemble: List[np.ndarray],
    classification_threshold: Optional[float] = None,
) -> EnsembleUncertainty:
    """
    Quantify uncertainty from ensemble predictions.

    Args:
        ensemble: List of ensemble member predictions
        classification_threshold: Threshold for binary classification

    Returns:
        EnsembleUncertainty with ensemble metrics
    """
    quantifier = UncertaintyQuantifier()
    return quantifier.calculate_ensemble_uncertainty(ensemble, classification_threshold)
