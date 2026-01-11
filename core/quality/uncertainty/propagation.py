"""
Error Propagation for Quality Control Uncertainty.

Provides tools for propagating uncertainty through quality control pipelines
and combining uncertainties from multiple sources in quality assessment.

This module focuses on QC-specific error propagation, complementing the
fusion uncertainty propagation in core.analysis.fusion.uncertainty.

Key Concepts:
- Quality scores have associated uncertainties
- Uncertainties propagate through aggregation operations
- Multi-source validation produces combined uncertainty
- Error budgets track uncertainty sources

Features:
- Linear error propagation (first-order Taylor series)
- Monte Carlo error propagation
- Quality score aggregation with uncertainty
- Multi-source validation uncertainty
- Error budget tracking
- Sensitivity analysis

Example:
    from core.quality.uncertainty.propagation import (
        QualityErrorPropagator,
        propagate_quality_uncertainty,
        compute_error_budget,
    )

    # Propagate uncertainty through quality pipeline
    propagator = QualityErrorPropagator()
    result = propagator.propagate_aggregation(scores, uncertainties)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class PropagationMethod(Enum):
    """Methods for error propagation."""
    LINEAR = "linear"              # First-order Taylor series
    QUADRATIC = "quadratic"        # Second-order Taylor series
    MONTE_CARLO = "monte_carlo"    # Monte Carlo sampling
    ANALYTICAL = "analytical"      # Closed-form (if available)


class AggregationMethod(Enum):
    """Methods for aggregating quality scores."""
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MEDIAN = "median"
    HARMONIC_MEAN = "harmonic_mean"


class CorrelationType(Enum):
    """Types of correlation between sources."""
    INDEPENDENT = "independent"    # No correlation
    FULLY_CORRELATED = "fully_correlated"  # Perfect correlation
    ANTI_CORRELATED = "anti_correlated"    # Negative correlation
    PARTIALLY_CORRELATED = "partially_correlated"


@dataclass
class UncertaintySource:
    """
    A source of uncertainty in quality assessment.

    Attributes:
        name: Source identifier
        uncertainty: Uncertainty value (standard deviation)
        is_systematic: Whether the error is systematic
        correlation_length: Spatial/temporal correlation length
        contribution_fraction: Fraction of total variance contributed
        metadata: Additional source metadata
    """
    name: str
    uncertainty: float
    is_systematic: bool = False
    correlation_length: Optional[float] = None
    contribution_fraction: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "uncertainty": self.uncertainty,
            "is_systematic": self.is_systematic,
            "correlation_length": self.correlation_length,
            "contribution_fraction": self.contribution_fraction,
            "metadata": self.metadata,
        }


@dataclass
class ErrorBudget:
    """
    Complete error budget for quality assessment.

    Attributes:
        sources: List of uncertainty sources
        total_uncertainty: Combined total uncertainty
        dominant_source: Name of dominant source
        systematic_fraction: Fraction from systematic sources
        random_fraction: Fraction from random sources
        is_dominated: Whether one source dominates
        metadata: Budget metadata
    """
    sources: List[UncertaintySource]
    total_uncertainty: float
    dominant_source: Optional[str] = None
    systematic_fraction: float = 0.0
    random_fraction: float = 1.0
    is_dominated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_sources": len(self.sources),
            "total_uncertainty": self.total_uncertainty,
            "dominant_source": self.dominant_source,
            "systematic_fraction": self.systematic_fraction,
            "random_fraction": self.random_fraction,
            "is_dominated": self.is_dominated,
            "sources": [s.to_dict() for s in self.sources],
            "metadata": self.metadata,
        }


@dataclass
class PropagationResult:
    """
    Result from uncertainty propagation.

    Attributes:
        value: Propagated value
        uncertainty: Propagated uncertainty (std)
        confidence_interval: Confidence interval (default 95%)
        sensitivity: Sensitivity coefficients for each input
        method: Method used for propagation
        error_budget: Optional detailed error budget
    """
    value: Union[float, np.ndarray]
    uncertainty: Union[float, np.ndarray]
    confidence_interval: Optional[Tuple[float, float]] = None
    sensitivity: Optional[Dict[str, float]] = None
    method: str = "linear"
    error_budget: Optional[ErrorBudget] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "value": float(np.mean(self.value)) if isinstance(self.value, np.ndarray) else self.value,
            "uncertainty": float(np.mean(self.uncertainty)) if isinstance(self.uncertainty, np.ndarray) else self.uncertainty,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "sensitivity": self.sensitivity,
            "method": self.method,
            "error_budget": self.error_budget.to_dict() if self.error_budget else None,
        }


@dataclass
class SensitivityResult:
    """
    Result from sensitivity analysis.

    Attributes:
        parameter_sensitivities: Sensitivity to each parameter
        most_sensitive: Most sensitive parameter
        least_sensitive: Least sensitive parameter
        total_sensitivity: Sum of absolute sensitivities
        normalized_sensitivities: Normalized to sum to 1
    """
    parameter_sensitivities: Dict[str, float]
    most_sensitive: str
    least_sensitive: str
    total_sensitivity: float
    normalized_sensitivities: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "parameter_sensitivities": self.parameter_sensitivities,
            "most_sensitive": self.most_sensitive,
            "least_sensitive": self.least_sensitive,
            "total_sensitivity": self.total_sensitivity,
            "normalized_sensitivities": self.normalized_sensitivities,
        }


@dataclass
class PropagationConfig:
    """
    Configuration for error propagation.

    Attributes:
        method: Propagation method
        monte_carlo_samples: Number of MC samples
        confidence_level: Confidence level for intervals
        correlation_type: Default correlation between inputs
        correlation_matrix: Optional custom correlation matrix
        track_sources: Whether to track individual sources
        min_uncertainty: Minimum uncertainty floor
    """
    method: PropagationMethod = PropagationMethod.LINEAR
    monte_carlo_samples: int = 10000
    confidence_level: float = 0.95
    correlation_type: CorrelationType = CorrelationType.INDEPENDENT
    correlation_matrix: Optional[np.ndarray] = None
    track_sources: bool = True
    min_uncertainty: float = 1e-10


class QualityErrorPropagator:
    """
    Propagates uncertainty through quality control operations.

    Handles uncertainty propagation for:
    - Quality score aggregation
    - Multi-source validation
    - Threshold-based decisions
    - Weighted combinations
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize error propagator.

        Args:
            config: Propagation configuration
        """
        self.config = config or PropagationConfig()

    def propagate_aggregation(
        self,
        values: List[float],
        uncertainties: List[float],
        method: AggregationMethod = AggregationMethod.MEAN,
        weights: Optional[List[float]] = None,
        names: Optional[List[str]] = None,
    ) -> PropagationResult:
        """
        Propagate uncertainty through aggregation operation.

        Args:
            values: Input values
            uncertainties: Input uncertainties (std)
            method: Aggregation method
            weights: Optional weights (for weighted methods)
            names: Optional names for error budget

        Returns:
            PropagationResult with propagated uncertainty
        """
        n = len(values)
        if n == 0:
            return PropagationResult(
                value=np.nan,
                uncertainty=np.nan,
                method=self.config.method.value,
            )

        values = np.array(values)
        uncertainties = np.array(uncertainties)

        if names is None:
            names = [f"input_{i}" for i in range(n)]

        # Compute aggregated value and uncertainty
        if method == AggregationMethod.MEAN:
            result_value, result_uncertainty, sensitivities = self._propagate_mean(
                values, uncertainties
            )
        elif method == AggregationMethod.WEIGHTED_MEAN:
            if weights is None:
                weights = [1.0 / n] * n
            result_value, result_uncertainty, sensitivities = self._propagate_weighted_mean(
                values, uncertainties, weights
            )
        elif method == AggregationMethod.GEOMETRIC_MEAN:
            result_value, result_uncertainty, sensitivities = self._propagate_geometric_mean(
                values, uncertainties
            )
        elif method == AggregationMethod.MINIMUM:
            result_value, result_uncertainty, sensitivities = self._propagate_minimum(
                values, uncertainties
            )
        elif method == AggregationMethod.MAXIMUM:
            result_value, result_uncertainty, sensitivities = self._propagate_maximum(
                values, uncertainties
            )
        elif method == AggregationMethod.MEDIAN:
            result_value, result_uncertainty, sensitivities = self._propagate_median(
                values, uncertainties
            )
        elif method == AggregationMethod.HARMONIC_MEAN:
            result_value, result_uncertainty, sensitivities = self._propagate_harmonic_mean(
                values, uncertainties
            )
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        # Build error budget
        error_budget = None
        if self.config.track_sources:
            error_budget = self._build_error_budget(
                names, uncertainties, sensitivities, result_uncertainty
            )

        # Compute confidence interval
        ci = self._compute_confidence_interval(result_value, result_uncertainty)

        return PropagationResult(
            value=result_value,
            uncertainty=result_uncertainty,
            confidence_interval=ci,
            sensitivity=dict(zip(names, sensitivities.tolist())),
            method=self.config.method.value,
            error_budget=error_budget,
        )

    def _propagate_mean(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through mean."""
        n = len(values)
        result_value = float(np.mean(values))

        # Sensitivity: d(mean)/d(xi) = 1/n
        sensitivities = np.ones(n) / n

        # Propagated variance
        if self.config.correlation_type == CorrelationType.INDEPENDENT:
            # Uncorrelated: var = sum(si^2 * ui^2)
            result_variance = np.sum((sensitivities * uncertainties) ** 2)
        elif self.config.correlation_type == CorrelationType.FULLY_CORRELATED:
            # Fully correlated: std = sum(|si| * ui)
            result_uncertainty = np.sum(np.abs(sensitivities) * uncertainties)
            return result_value, float(result_uncertainty), sensitivities
        else:
            result_variance = np.sum((sensitivities * uncertainties) ** 2)

        result_uncertainty = float(np.sqrt(max(result_variance, self.config.min_uncertainty**2)))
        return result_value, result_uncertainty, sensitivities

    def _propagate_weighted_mean(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
        weights: List[float],
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through weighted mean."""
        weights = np.array(weights)
        total_weight = np.sum(weights)
        if total_weight < 1e-10:
            return np.nan, np.nan, np.zeros(len(values))

        normalized_weights = weights / total_weight
        result_value = float(np.sum(normalized_weights * values))

        # Sensitivity: d(wmean)/d(xi) = wi / sum(w)
        sensitivities = normalized_weights

        # Propagated variance (assuming independent)
        result_variance = np.sum((sensitivities * uncertainties) ** 2)
        result_uncertainty = float(np.sqrt(max(result_variance, self.config.min_uncertainty**2)))

        return result_value, result_uncertainty, sensitivities

    def _propagate_geometric_mean(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through geometric mean."""
        n = len(values)

        # Handle non-positive values
        if np.any(values <= 0):
            logger.warning("Non-positive values in geometric mean - using absolute values")
            values = np.abs(values) + 1e-10

        log_values = np.log(values)
        log_mean = np.mean(log_values)
        result_value = float(np.exp(log_mean))

        # Sensitivity: d(gmean)/d(xi) = (gmean / n) / xi
        sensitivities = (result_value / n) / values

        # Use relative uncertainties for geometric operations
        relative_uncertainties = uncertainties / (values + 1e-10)
        relative_variance = np.sum(relative_uncertainties ** 2) / (n ** 2)
        result_uncertainty = float(result_value * np.sqrt(max(relative_variance, 0)))

        return result_value, max(result_uncertainty, self.config.min_uncertainty), sensitivities

    def _propagate_minimum(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through minimum."""
        min_idx = np.argmin(values)
        result_value = float(values[min_idx])

        # Sensitivity: 1 for minimum, 0 for others
        sensitivities = np.zeros(len(values))
        sensitivities[min_idx] = 1.0

        result_uncertainty = float(uncertainties[min_idx])
        return result_value, result_uncertainty, sensitivities

    def _propagate_maximum(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through maximum."""
        max_idx = np.argmax(values)
        result_value = float(values[max_idx])

        # Sensitivity: 1 for maximum, 0 for others
        sensitivities = np.zeros(len(values))
        sensitivities[max_idx] = 1.0

        result_uncertainty = float(uncertainties[max_idx])
        return result_value, result_uncertainty, sensitivities

    def _propagate_median(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through median."""
        result_value = float(np.median(values))

        # For median, use Monte Carlo for accurate uncertainty
        if self.config.method == PropagationMethod.MONTE_CARLO:
            result_uncertainty = self._monte_carlo_propagate(
                values, uncertainties, np.median
            )
        else:
            # Approximation: median uncertainty ~ 1.253 * SE of mean for normal distributions
            n = len(values)
            mean_uncertainty = np.sqrt(np.sum(uncertainties ** 2)) / n
            result_uncertainty = 1.253 * mean_uncertainty

        # Approximate sensitivities (uniform contribution near median)
        n = len(values)
        sensitivities = np.ones(n) / n

        return result_value, float(result_uncertainty), sensitivities

    def _propagate_harmonic_mean(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Propagate uncertainty through harmonic mean."""
        n = len(values)

        # Handle near-zero values - use 1e-10 as minimum (np.sign(0) == 0, so we need explicit handling)
        safe_values = np.where(np.abs(values) > 1e-10, values,
                               np.where(values >= 0, 1e-10, -1e-10))

        harmonic_mean = float(n / np.sum(1 / safe_values))

        # Sensitivity: d(hmean)/d(xi) = (hmean / xi)^2 / n
        sensitivities = (harmonic_mean / safe_values) ** 2 / n

        # Propagated variance
        result_variance = np.sum((sensitivities * uncertainties) ** 2)
        result_uncertainty = float(np.sqrt(max(result_variance, self.config.min_uncertainty**2)))

        return harmonic_mean, result_uncertainty, sensitivities

    def _monte_carlo_propagate(
        self,
        values: np.ndarray,
        uncertainties: np.ndarray,
        func: Callable[[np.ndarray], float],
    ) -> float:
        """Use Monte Carlo sampling for propagation."""
        n_samples = self.config.monte_carlo_samples
        n_values = len(values)

        # Generate samples
        samples = np.random.normal(
            loc=values,
            scale=uncertainties,
            size=(n_samples, n_values)
        )

        # Apply function to each sample
        results = np.array([func(s) for s in samples])

        return float(np.std(results))

    def _build_error_budget(
        self,
        names: List[str],
        uncertainties: np.ndarray,
        sensitivities: np.ndarray,
        total_uncertainty: float,
    ) -> ErrorBudget:
        """Build detailed error budget."""
        # Contribution of each source to total variance
        contributions = (sensitivities * uncertainties) ** 2
        total_variance = np.sum(contributions)

        sources = []
        for i, name in enumerate(names):
            contribution_fraction = contributions[i] / total_variance if total_variance > 0 else 0
            sources.append(UncertaintySource(
                name=name,
                uncertainty=float(uncertainties[i]),
                contribution_fraction=float(contribution_fraction),
            ))

        # Find dominant source
        if sources:
            dominant_source = max(sources, key=lambda s: s.contribution_fraction)
            is_dominated = dominant_source.contribution_fraction > 0.5
        else:
            dominant_source = None
            is_dominated = False

        return ErrorBudget(
            sources=sources,
            total_uncertainty=total_uncertainty,
            dominant_source=dominant_source.name if dominant_source else None,
            is_dominated=is_dominated,
        )

    def _compute_confidence_interval(
        self,
        value: float,
        uncertainty: float,
    ) -> Tuple[float, float]:
        """Compute confidence interval."""
        from scipy import stats

        z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        margin = z * uncertainty

        return (float(value - margin), float(value + margin))

    def propagate_threshold_decision(
        self,
        value: float,
        uncertainty: float,
        threshold: float,
        threshold_uncertainty: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Propagate uncertainty through threshold decision.

        Args:
            value: Observed value
            uncertainty: Value uncertainty
            threshold: Decision threshold
            threshold_uncertainty: Threshold uncertainty (if any)

        Returns:
            Dictionary with decision probability and confidence
        """
        # Combined uncertainty for comparison
        combined_uncertainty = np.sqrt(uncertainty**2 + threshold_uncertainty**2)

        if combined_uncertainty < 1e-10:
            # No uncertainty - deterministic decision
            exceeds = value > threshold
            return {
                "decision": "exceeds" if exceeds else "below",
                "probability_exceeds": 1.0 if exceeds else 0.0,
                "decision_confidence": 1.0,
                "margin": float(value - threshold),
            }

        # z-score for the comparison
        z = (value - threshold) / combined_uncertainty

        # Probability of exceeding threshold
        from scipy import stats
        prob_exceeds = float(1 - stats.norm.cdf(0, loc=z, scale=1))

        # Decision and confidence
        if prob_exceeds > 0.5:
            decision = "exceeds"
            confidence = prob_exceeds
        else:
            decision = "below"
            confidence = 1 - prob_exceeds

        return {
            "decision": decision,
            "probability_exceeds": prob_exceeds,
            "decision_confidence": confidence,
            "margin": float(value - threshold),
            "z_score": float(z),
        }

    def combine_validation_uncertainties(
        self,
        validation_results: List[Dict[str, float]],
        method: str = "weighted",
    ) -> PropagationResult:
        """
        Combine uncertainties from multiple validation sources.

        Args:
            validation_results: List of {"value": v, "uncertainty": u, "weight": w}
            method: "weighted" or "optimal"

        Returns:
            PropagationResult with combined uncertainty
        """
        if not validation_results:
            return PropagationResult(
                value=np.nan,
                uncertainty=np.nan,
            )

        values = np.array([r["value"] for r in validation_results])
        uncertainties = np.array([r["uncertainty"] for r in validation_results])

        if method == "optimal":
            # Optimal combination: weights inversely proportional to variance
            variances = uncertainties ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                weights = 1.0 / np.where(variances > 1e-20, variances, 1e-20)
            weights = weights / np.sum(weights)
        else:
            # Use provided weights or equal
            weights = np.array([r.get("weight", 1.0) for r in validation_results])
            weights = weights / np.sum(weights)

        # Combined value (weighted mean)
        combined_value = float(np.sum(weights * values))

        # Combined uncertainty
        combined_variance = np.sum((weights * uncertainties) ** 2)
        combined_uncertainty = float(np.sqrt(max(combined_variance, self.config.min_uncertainty**2)))

        names = [r.get("name", f"source_{i}") for i, r in enumerate(validation_results)]
        error_budget = self._build_error_budget(
            names, uncertainties, weights, combined_uncertainty
        )

        return PropagationResult(
            value=combined_value,
            uncertainty=combined_uncertainty,
            confidence_interval=self._compute_confidence_interval(combined_value, combined_uncertainty),
            method=method,
            error_budget=error_budget,
        )


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis for quality assessment.

    Identifies which input parameters most strongly affect
    output uncertainty.
    """

    def __init__(self, config: Optional[PropagationConfig] = None):
        """
        Initialize sensitivity analyzer.

        Args:
            config: Propagation configuration
        """
        self.config = config or PropagationConfig()

    def analyze_sensitivity(
        self,
        func: Callable[..., float],
        parameters: Dict[str, float],
        uncertainties: Dict[str, float],
        perturbation: float = 0.01,
    ) -> SensitivityResult:
        """
        Analyze sensitivity of output to each parameter.

        Args:
            func: Function to analyze
            parameters: Parameter values
            uncertainties: Parameter uncertainties
            perturbation: Relative perturbation for numerical derivatives

        Returns:
            SensitivityResult with sensitivity analysis
        """
        param_names = list(parameters.keys())
        param_values = list(parameters.values())

        # Compute base value
        base_value = func(**parameters)

        # Compute sensitivity to each parameter
        sensitivities = {}
        for name in param_names:
            value = parameters[name]
            delta = max(abs(value) * perturbation, 1e-10)

            # Perturbed values
            params_plus = parameters.copy()
            params_plus[name] = value + delta

            params_minus = parameters.copy()
            params_minus[name] = value - delta

            # Central difference
            value_plus = func(**params_plus)
            value_minus = func(**params_minus)

            sensitivity = (value_plus - value_minus) / (2 * delta)
            sensitivities[name] = float(sensitivity)

        # Find most/least sensitive
        abs_sensitivities = {k: abs(v) for k, v in sensitivities.items()}
        most_sensitive = max(abs_sensitivities, key=abs_sensitivities.get)
        least_sensitive = min(abs_sensitivities, key=abs_sensitivities.get)

        # Total and normalized
        total_sensitivity = sum(abs_sensitivities.values())
        if total_sensitivity > 0:
            normalized = {k: v / total_sensitivity for k, v in abs_sensitivities.items()}
        else:
            normalized = {k: 0.0 for k in abs_sensitivities}

        return SensitivityResult(
            parameter_sensitivities=sensitivities,
            most_sensitive=most_sensitive,
            least_sensitive=least_sensitive,
            total_sensitivity=total_sensitivity,
            normalized_sensitivities=normalized,
        )

    def compute_uncertainty_contribution(
        self,
        func: Callable[..., float],
        parameters: Dict[str, float],
        uncertainties: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute contribution of each parameter's uncertainty to output uncertainty.

        Args:
            func: Function to analyze
            parameters: Parameter values
            uncertainties: Parameter uncertainties

        Returns:
            Dictionary of parameter contributions to variance
        """
        # Get sensitivities
        sensitivity_result = self.analyze_sensitivity(func, parameters, uncertainties)

        contributions = {}
        total_variance = 0.0

        for name in parameters:
            sensitivity = sensitivity_result.parameter_sensitivities[name]
            uncertainty = uncertainties.get(name, 0.0)
            variance_contribution = (sensitivity * uncertainty) ** 2
            contributions[name] = variance_contribution
            total_variance += variance_contribution

        # Normalize to fractions
        if total_variance > 0:
            contributions = {k: v / total_variance for k, v in contributions.items()}

        return contributions


# Convenience functions

def propagate_quality_uncertainty(
    values: List[float],
    uncertainties: List[float],
    method: str = "mean",
    weights: Optional[List[float]] = None,
) -> PropagationResult:
    """
    Propagate uncertainty through quality aggregation.

    Args:
        values: Input quality values
        uncertainties: Input uncertainties
        method: Aggregation method name
        weights: Optional weights

    Returns:
        PropagationResult
    """
    propagator = QualityErrorPropagator()

    method_map = {
        "mean": AggregationMethod.MEAN,
        "weighted_mean": AggregationMethod.WEIGHTED_MEAN,
        "geometric_mean": AggregationMethod.GEOMETRIC_MEAN,
        "min": AggregationMethod.MINIMUM,
        "max": AggregationMethod.MAXIMUM,
        "median": AggregationMethod.MEDIAN,
        "harmonic_mean": AggregationMethod.HARMONIC_MEAN,
    }

    agg_method = method_map.get(method.lower(), AggregationMethod.MEAN)
    return propagator.propagate_aggregation(values, uncertainties, agg_method, weights)


def compute_error_budget(
    names: List[str],
    uncertainties: List[float],
) -> ErrorBudget:
    """
    Compute error budget from uncertainty sources.

    Args:
        names: Source names
        uncertainties: Source uncertainties

    Returns:
        ErrorBudget
    """
    uncertainties = np.array(uncertainties)
    total_variance = np.sum(uncertainties ** 2)
    total_uncertainty = float(np.sqrt(total_variance))

    sources = []
    for i, name in enumerate(names):
        contribution = uncertainties[i] ** 2 / total_variance if total_variance > 0 else 0
        sources.append(UncertaintySource(
            name=name,
            uncertainty=float(uncertainties[i]),
            contribution_fraction=float(contribution),
        ))

    # Find dominant
    dominant = max(sources, key=lambda s: s.contribution_fraction) if sources else None

    return ErrorBudget(
        sources=sources,
        total_uncertainty=total_uncertainty,
        dominant_source=dominant.name if dominant else None,
        is_dominated=dominant.contribution_fraction > 0.5 if dominant else False,
    )


def combine_independent_uncertainties(
    uncertainties: List[float],
) -> float:
    """
    Combine independent uncertainties (root sum of squares).

    Args:
        uncertainties: List of uncertainties (std)

    Returns:
        Combined uncertainty
    """
    return float(np.sqrt(np.sum(np.array(uncertainties) ** 2)))


def combine_correlated_uncertainties(
    uncertainties: List[float],
    correlation_coefficient: float = 1.0,
) -> float:
    """
    Combine correlated uncertainties.

    Args:
        uncertainties: List of uncertainties (std)
        correlation_coefficient: Correlation between sources (0 to 1)

    Returns:
        Combined uncertainty
    """
    uncertainties = np.array(uncertainties)
    n = len(uncertainties)

    # Build correlation matrix
    corr_matrix = np.full((n, n), correlation_coefficient)
    np.fill_diagonal(corr_matrix, 1.0)

    # Build covariance matrix
    outer = np.outer(uncertainties, uncertainties)
    cov_matrix = outer * corr_matrix

    # Total variance = sum of all covariance elements
    total_variance = np.sum(cov_matrix)

    return float(np.sqrt(max(total_variance, 0)))


def threshold_exceedance_probability(
    value: float,
    uncertainty: float,
    threshold: float,
) -> float:
    """
    Calculate probability that true value exceeds threshold.

    Args:
        value: Observed value
        uncertainty: Value uncertainty (std)
        threshold: Threshold to compare

    Returns:
        Probability of exceeding threshold
    """
    if uncertainty < 1e-10:
        return 1.0 if value > threshold else 0.0

    from scipy import stats
    z = (value - threshold) / uncertainty
    return float(1 - stats.norm.cdf(0, loc=z, scale=1))
