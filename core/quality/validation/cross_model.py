"""
Cross-Model Validation for Quality Control.

Provides tools for comparing results from different algorithms/models
to assess agreement and identify systematic biases, including:
- Per-pixel model agreement assessment
- Statistical comparison metrics (correlation, bias, RMSE)
- Ensemble divergence analysis
- Confidence-weighted model comparison
- Model performance ranking

Key Concepts:
- Multiple algorithms may produce different results for the same input
- Agreement between models increases confidence in results
- Disagreement may indicate difficult regions or model limitations
- Ensemble statistics provide robust uncertainty estimates
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class AgreementLevel(Enum):
    """Level of agreement between models."""
    EXCELLENT = "excellent"    # All models agree within tolerance
    GOOD = "good"              # Most models agree, minor outliers
    MODERATE = "moderate"      # Significant spread but consensus visible
    POOR = "poor"              # Large disagreement, no clear consensus
    CRITICAL = "critical"      # Fundamental disagreement, cannot resolve


class ComparisonMetric(Enum):
    """Metrics for comparing model outputs."""
    CORRELATION = "correlation"          # Pearson correlation coefficient
    RMSE = "rmse"                        # Root mean squared error
    MAE = "mae"                          # Mean absolute error
    BIAS = "bias"                        # Mean bias (signed error)
    MAX_ERROR = "max_error"              # Maximum absolute error
    RELATIVE_RMSE = "relative_rmse"      # RMSE relative to data range
    NASH_SUTCLIFFE = "nash_sutcliffe"    # Nash-Sutcliffe efficiency
    SKILL_SCORE = "skill_score"          # Skill score vs reference


@dataclass
class ModelOutput:
    """
    Output from a single model/algorithm.

    Attributes:
        data: Model output array
        model_id: Unique identifier for this model
        model_name: Human-readable model name
        confidence: Per-pixel confidence (0-1)
        overall_confidence: Aggregate model confidence (0-1)
        parameters: Model parameters used
        metadata: Additional model metadata
    """
    data: np.ndarray
    model_id: str
    model_name: str = ""
    confidence: Optional[np.ndarray] = None
    overall_confidence: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize confidence if not provided."""
        if self.confidence is None:
            # Default to full confidence where data is valid
            if self.data.dtype.kind == 'f':
                self.confidence = (~np.isnan(self.data)).astype(np.float32)
            else:
                self.confidence = np.ones_like(self.data, dtype=np.float32)
        if not self.model_name:
            self.model_name = self.model_id


@dataclass
class PairwiseComparison:
    """
    Comparison between two models.

    Attributes:
        model_a: First model identifier
        model_b: Second model identifier
        correlation: Pearson correlation coefficient
        rmse: Root mean squared error
        mae: Mean absolute error
        bias: Mean bias (model_a - model_b)
        max_error: Maximum absolute difference
        agreement_fraction: Fraction of pixels within tolerance
        metrics: All computed metrics
    """
    model_a: str
    model_b: str
    correlation: float
    rmse: float
    mae: float
    bias: float
    max_error: float
    agreement_fraction: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "correlation": round(self.correlation, 4),
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "bias": round(self.bias, 6),
            "max_error": round(self.max_error, 6),
            "agreement_fraction": round(self.agreement_fraction, 4),
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
        }


@dataclass
class AgreementMap:
    """
    Spatial map of model agreement.

    Attributes:
        level_map: Per-pixel agreement level
        spread_map: Per-pixel spread (std) across models
        num_models_map: Number of valid models per pixel
        consensus_map: Consensus value where agreement is good
        disagreement_regions: Identified regions of high disagreement
    """
    level_map: np.ndarray  # Contains AgreementLevel values
    spread_map: np.ndarray
    num_models_map: np.ndarray
    consensus_map: Optional[np.ndarray] = None
    disagreement_regions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary summary."""
        level_counts = {}
        for level in AgreementLevel:
            level_counts[level.value] = int(np.sum(self.level_map == level.value))

        return {
            "total_pixels": int(self.level_map.size),
            "level_counts": level_counts,
            "mean_spread": float(np.nanmean(self.spread_map)),
            "max_spread": float(np.nanmax(self.spread_map)),
            "mean_num_models": float(np.mean(self.num_models_map)),
            "num_disagreement_regions": len(self.disagreement_regions),
        }


@dataclass
class CrossModelConfig:
    """
    Configuration for cross-model validation.

    Attributes:
        absolute_tolerance: Absolute tolerance for agreement
        relative_tolerance: Relative tolerance for agreement (fraction)
        min_models_for_consensus: Minimum models needed for consensus
        use_confidence_weights: Weight comparisons by confidence
        metrics_to_compute: List of metrics to calculate
        identify_disagreement_regions: Find spatial clusters of disagreement
        min_region_size: Minimum pixels for disagreement region
    """
    absolute_tolerance: float = 0.1
    relative_tolerance: float = 0.05
    min_models_for_consensus: int = 2
    use_confidence_weights: bool = True
    metrics_to_compute: List[ComparisonMetric] = field(
        default_factory=lambda: [
            ComparisonMetric.CORRELATION,
            ComparisonMetric.RMSE,
            ComparisonMetric.MAE,
            ComparisonMetric.BIAS,
        ]
    )
    identify_disagreement_regions: bool = True
    min_region_size: int = 10


@dataclass
class CrossModelResult:
    """
    Result from cross-model validation.

    Attributes:
        pairwise_comparisons: Comparisons between model pairs
        agreement_map: Spatial agreement map
        ensemble_statistics: Ensemble-level statistics
        model_rankings: Models ranked by agreement
        overall_agreement: Overall agreement level
        diagnostics: Additional diagnostic information
    """
    pairwise_comparisons: List[PairwiseComparison]
    agreement_map: AgreementMap
    ensemble_statistics: Dict[str, float]
    model_rankings: List[Tuple[str, float]]  # (model_id, agreement_score)
    overall_agreement: AgreementLevel
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_agreement": self.overall_agreement.value,
            "num_models": len(self.model_rankings),
            "pairwise_comparisons": [p.to_dict() for p in self.pairwise_comparisons],
            "agreement_map_summary": self.agreement_map.to_dict(),
            "ensemble_statistics": {k: round(v, 4) for k, v in self.ensemble_statistics.items()},
            "model_rankings": [
                {"model_id": m, "agreement_score": round(s, 4)}
                for m, s in self.model_rankings
            ],
            "diagnostics": self.diagnostics,
        }


class CrossModelValidator:
    """
    Validates agreement between multiple model outputs.

    Compares outputs from different algorithms/models to assess:
    - Pairwise agreement metrics
    - Ensemble-level statistics
    - Spatial patterns of agreement/disagreement
    - Model ranking by reliability
    """

    def __init__(self, config: Optional[CrossModelConfig] = None):
        """
        Initialize cross-model validator.

        Args:
            config: Validation configuration
        """
        self.config = config or CrossModelConfig()

    def validate(
        self,
        models: List[ModelOutput],
        reference: Optional[ModelOutput] = None,
    ) -> CrossModelResult:
        """
        Validate agreement between model outputs.

        Args:
            models: List of model outputs to compare
            reference: Optional reference model for skill scores

        Returns:
            CrossModelResult with validation details
        """
        if len(models) < 2:
            raise ValueError("At least 2 models required for cross-validation")

        # Compute pairwise comparisons
        pairwise = self._compute_pairwise_comparisons(models)

        # Compute agreement map
        agreement_map = self._compute_agreement_map(models)

        # Compute ensemble statistics
        ensemble_stats = self._compute_ensemble_statistics(models)

        # Rank models by agreement
        rankings = self._rank_models(models, pairwise)

        # If reference provided, compute skill scores
        if reference:
            skill_scores = self._compute_skill_scores(models, reference)
            ensemble_stats.update(skill_scores)

        # Determine overall agreement level
        overall_agreement = self._classify_overall_agreement(
            agreement_map, ensemble_stats
        )

        return CrossModelResult(
            pairwise_comparisons=pairwise,
            agreement_map=agreement_map,
            ensemble_statistics=ensemble_stats,
            model_rankings=rankings,
            overall_agreement=overall_agreement,
            diagnostics={
                "num_models": len(models),
                "model_ids": [m.model_id for m in models],
                "reference_model": reference.model_id if reference else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _compute_pairwise_comparisons(
        self,
        models: List[ModelOutput],
    ) -> List[PairwiseComparison]:
        """Compute pairwise comparison metrics between all model pairs."""
        comparisons = []

        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if j <= i:
                    continue  # Skip self and duplicates

                comparison = self._compare_two_models(model_a, model_b)
                comparisons.append(comparison)

        return comparisons

    def _compare_two_models(
        self,
        model_a: ModelOutput,
        model_b: ModelOutput,
    ) -> PairwiseComparison:
        """Compare two models and compute metrics."""
        data_a = model_a.data.astype(np.float64)
        data_b = model_b.data.astype(np.float64)

        # Create valid mask (both models have valid data)
        if self.config.use_confidence_weights:
            valid_mask = (model_a.confidence > 0) & (model_b.confidence > 0)
            weights = model_a.confidence * model_b.confidence
        else:
            valid_mask = ~(np.isnan(data_a) | np.isnan(data_b))
            weights = np.ones_like(data_a)

        # Extract valid values
        a_valid = data_a[valid_mask]
        b_valid = data_b[valid_mask]
        w_valid = weights[valid_mask]

        if len(a_valid) == 0:
            # No overlapping valid data
            return PairwiseComparison(
                model_a=model_a.model_id,
                model_b=model_b.model_id,
                correlation=np.nan,
                rmse=np.nan,
                mae=np.nan,
                bias=np.nan,
                max_error=np.nan,
                agreement_fraction=0.0,
                metrics={},
            )

        # Compute metrics
        metrics = {}

        # Correlation
        if len(a_valid) > 1:
            # Handle constant arrays to avoid correlation warning
            if np.std(a_valid) > 1e-10 and np.std(b_valid) > 1e-10:
                correlation = float(np.corrcoef(a_valid, b_valid)[0, 1])
            else:
                correlation = 1.0 if np.allclose(a_valid, b_valid) else np.nan
        else:
            correlation = np.nan
        metrics["correlation"] = correlation

        # Differences
        diff = a_valid - b_valid

        # RMSE
        rmse = float(np.sqrt(np.average(diff**2, weights=w_valid)))
        metrics["rmse"] = rmse

        # MAE
        mae = float(np.average(np.abs(diff), weights=w_valid))
        metrics["mae"] = mae

        # Bias
        bias = float(np.average(diff, weights=w_valid))
        metrics["bias"] = bias

        # Max error
        max_error = float(np.max(np.abs(diff)))
        metrics["max_error"] = max_error

        # Agreement fraction
        abs_tol = self.config.absolute_tolerance
        rel_tol = self.config.relative_tolerance
        reference = np.maximum(np.abs(a_valid), np.abs(b_valid))
        reference = np.where(reference > 1e-10, reference, 1.0)
        within_tolerance = (
            (np.abs(diff) <= abs_tol) |
            (np.abs(diff) / reference <= rel_tol)
        )
        agreement_fraction = float(np.mean(within_tolerance))
        metrics["agreement_fraction"] = agreement_fraction

        # Nash-Sutcliffe efficiency
        if ComparisonMetric.NASH_SUTCLIFFE in self.config.metrics_to_compute:
            mean_b = np.mean(b_valid)
            ss_res = np.sum((a_valid - b_valid) ** 2)
            ss_tot = np.sum((b_valid - mean_b) ** 2)
            if ss_tot > 1e-10:
                nse = float(1 - ss_res / ss_tot)
            else:
                nse = 1.0 if ss_res < 1e-10 else np.nan
            metrics["nash_sutcliffe"] = nse

        # Relative RMSE
        if ComparisonMetric.RELATIVE_RMSE in self.config.metrics_to_compute:
            data_range = np.max(b_valid) - np.min(b_valid)
            if data_range > 1e-10:
                rel_rmse = float(rmse / data_range)
            else:
                rel_rmse = 0.0 if rmse < 1e-10 else np.inf
            metrics["relative_rmse"] = rel_rmse

        return PairwiseComparison(
            model_a=model_a.model_id,
            model_b=model_b.model_id,
            correlation=correlation,
            rmse=rmse,
            mae=mae,
            bias=bias,
            max_error=max_error,
            agreement_fraction=agreement_fraction,
            metrics=metrics,
        )

    def _compute_agreement_map(
        self,
        models: List[ModelOutput],
    ) -> AgreementMap:
        """Compute spatial map of model agreement."""
        shape = models[0].data.shape

        # Stack data for analysis
        data_stack = np.stack([m.data.astype(np.float64) for m in models])
        if self.config.use_confidence_weights:
            confidence_stack = np.stack([m.confidence for m in models])
            valid_mask = confidence_stack > 0
        else:
            valid_mask = ~np.isnan(data_stack)

        # Count valid models per pixel
        num_models = np.sum(valid_mask, axis=0).astype(np.int32)

        # Calculate spread (std) across models
        with np.errstate(invalid='ignore'):
            masked_data = np.where(valid_mask, data_stack, np.nan)
            spread = np.nanstd(masked_data, axis=0)
            spread = np.where(np.isnan(spread), 0.0, spread)

        # Calculate consensus where possible
        consensus = np.nanmean(masked_data, axis=0)

        # Classify agreement level per pixel
        level_map = self._classify_agreement_levels(spread, num_models, masked_data)

        # Identify disagreement regions
        disagreement_regions = []
        if self.config.identify_disagreement_regions:
            disagreement_regions = self._find_disagreement_regions(
                level_map, spread, shape
            )

        return AgreementMap(
            level_map=level_map,
            spread_map=spread,
            num_models_map=num_models,
            consensus_map=consensus,
            disagreement_regions=disagreement_regions,
        )

    def _classify_agreement_levels(
        self,
        spread: np.ndarray,
        num_models: np.ndarray,
        masked_data: np.ndarray,
    ) -> np.ndarray:
        """Classify agreement level for each pixel."""
        shape = spread.shape
        level_map = np.full(shape, AgreementLevel.CRITICAL.value, dtype=object)

        # Calculate data range for relative comparison
        with np.errstate(invalid='ignore'):
            data_mean = np.nanmean(masked_data, axis=0)
            data_range = np.nanmax(masked_data, axis=0) - np.nanmin(masked_data, axis=0)

        # Relative spread
        relative_spread = np.where(
            np.abs(data_mean) > 1e-10,
            spread / np.abs(data_mean),
            0.0
        )

        # Single model or no valid models
        single_or_none = num_models <= 1
        level_map[single_or_none] = AgreementLevel.CRITICAL.value

        # Thresholds
        abs_tol = self.config.absolute_tolerance
        rel_tol = self.config.relative_tolerance

        # Excellent: very tight agreement
        excellent = (
            ~single_or_none &
            (num_models >= self.config.min_models_for_consensus) &
            ((spread <= abs_tol * 0.5) | (relative_spread <= rel_tol * 0.5))
        )
        level_map[excellent] = AgreementLevel.EXCELLENT.value

        # Good: within tolerance
        good = (
            ~single_or_none & ~excellent &
            (num_models >= self.config.min_models_for_consensus) &
            ((spread <= abs_tol) | (relative_spread <= rel_tol))
        )
        level_map[good] = AgreementLevel.GOOD.value

        # Moderate: within 2x tolerance
        moderate = (
            ~single_or_none & ~excellent & ~good &
            (num_models >= 2) &
            ((spread <= abs_tol * 2) | (relative_spread <= rel_tol * 2))
        )
        level_map[moderate] = AgreementLevel.MODERATE.value

        # Poor: within 5x tolerance
        poor = (
            ~single_or_none & ~excellent & ~good & ~moderate &
            (num_models >= 2) &
            ((spread <= abs_tol * 5) | (relative_spread <= rel_tol * 5))
        )
        level_map[poor] = AgreementLevel.POOR.value

        return level_map

    def _find_disagreement_regions(
        self,
        level_map: np.ndarray,
        spread: np.ndarray,
        shape: Tuple[int, ...],
    ) -> List[Dict[str, Any]]:
        """Identify connected regions of high disagreement."""
        from scipy import ndimage

        # Create binary mask of poor/critical agreement
        disagreement_mask = np.isin(
            level_map,
            [AgreementLevel.POOR.value, AgreementLevel.CRITICAL.value]
        )

        # Label connected regions
        labeled, num_features = ndimage.label(disagreement_mask)

        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            region_size = int(np.sum(region_mask))

            if region_size < self.config.min_region_size:
                continue

            # Get region statistics
            region_spread = spread[region_mask]

            # Find region bounds
            rows, cols = np.where(region_mask)

            regions.append({
                "region_id": i,
                "size_pixels": region_size,
                "mean_spread": float(np.mean(region_spread)),
                "max_spread": float(np.max(region_spread)),
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
            })

        return regions

    def _compute_ensemble_statistics(
        self,
        models: List[ModelOutput],
    ) -> Dict[str, float]:
        """Compute ensemble-level statistics."""
        # Stack data
        data_stack = np.stack([m.data.astype(np.float64) for m in models])
        if self.config.use_confidence_weights:
            confidence_stack = np.stack([m.confidence for m in models])
            valid_mask = confidence_stack > 0
        else:
            valid_mask = ~np.isnan(data_stack)

        masked_data = np.where(valid_mask, data_stack, np.nan)

        stats = {}

        # Mean spread (std across models)
        with np.errstate(invalid='ignore'):
            spread = np.nanstd(masked_data, axis=0)
            stats["mean_spread"] = float(np.nanmean(spread))
            stats["max_spread"] = float(np.nanmax(spread))

        # Mean range (max - min across models)
        with np.errstate(invalid='ignore'):
            data_range = np.nanmax(masked_data, axis=0) - np.nanmin(masked_data, axis=0)
            stats["mean_range"] = float(np.nanmean(data_range))

        # Mean of ensemble mean
        ensemble_mean = np.nanmean(masked_data, axis=0)
        stats["ensemble_mean"] = float(np.nanmean(ensemble_mean))

        # Coefficient of variation (spatial)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(
                np.abs(ensemble_mean) > 1e-10,
                spread / np.abs(ensemble_mean),
                0.0
            )
            stats["mean_cv"] = float(np.nanmean(cv))

        # Agreement rate (fraction of pixels with excellent/good agreement)
        # We need to compute agreement map here or pass it
        # For now, estimate from spread
        within_tolerance = (spread <= self.config.absolute_tolerance)
        stats["agreement_rate"] = float(np.nanmean(within_tolerance))

        # Ensemble confidence
        if self.config.use_confidence_weights:
            mean_confidence = np.mean([m.overall_confidence for m in models])
            stats["mean_model_confidence"] = float(mean_confidence)

        return stats

    def _rank_models(
        self,
        models: List[ModelOutput],
        pairwise: List[PairwiseComparison],
    ) -> List[Tuple[str, float]]:
        """Rank models by their agreement with other models."""
        # For each model, compute average agreement fraction with all others
        model_scores = {}

        for model in models:
            total_agreement = 0.0
            count = 0

            for comparison in pairwise:
                if comparison.model_a == model.model_id:
                    total_agreement += comparison.agreement_fraction
                    count += 1
                elif comparison.model_b == model.model_id:
                    total_agreement += comparison.agreement_fraction
                    count += 1

            if count > 0:
                avg_agreement = total_agreement / count
            else:
                avg_agreement = 0.0

            # Weight by model confidence
            weighted_score = avg_agreement * model.overall_confidence
            model_scores[model.model_id] = weighted_score

        # Sort by score (descending)
        rankings = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        return rankings

    def _compute_skill_scores(
        self,
        models: List[ModelOutput],
        reference: ModelOutput,
    ) -> Dict[str, float]:
        """Compute skill scores relative to reference model."""
        scores = {}

        for model in models:
            if model.model_id == reference.model_id:
                continue

            comparison = self._compare_two_models(model, reference)

            # Skill score = 1 - (RMSE / reference_std)
            ref_valid = reference.data[~np.isnan(reference.data)]
            if len(ref_valid) > 0:
                ref_std = np.std(ref_valid)
                if ref_std > 1e-10:
                    skill = 1 - (comparison.rmse / ref_std)
                else:
                    skill = 1.0 if comparison.rmse < 1e-10 else 0.0
            else:
                skill = np.nan

            scores[f"skill_{model.model_id}"] = float(skill)

        # Mean skill score
        valid_skills = [s for s in scores.values() if not np.isnan(s)]
        if valid_skills:
            scores["mean_skill"] = float(np.mean(valid_skills))

        return scores

    def _classify_overall_agreement(
        self,
        agreement_map: AgreementMap,
        ensemble_stats: Dict[str, float],
    ) -> AgreementLevel:
        """Determine overall agreement level."""
        agreement_rate = ensemble_stats.get("agreement_rate", 0.0)
        mean_cv = ensemble_stats.get("mean_cv", 1.0)

        # Count agreement levels
        level_map = agreement_map.level_map
        total = level_map.size
        if total == 0:
            return AgreementLevel.CRITICAL

        excellent_pct = np.sum(level_map == AgreementLevel.EXCELLENT.value) / total
        good_pct = np.sum(level_map == AgreementLevel.GOOD.value) / total
        poor_pct = np.sum(level_map == AgreementLevel.POOR.value) / total
        critical_pct = np.sum(level_map == AgreementLevel.CRITICAL.value) / total

        # Classification logic
        if excellent_pct >= 0.8 or (excellent_pct + good_pct >= 0.95):
            return AgreementLevel.EXCELLENT
        elif excellent_pct + good_pct >= 0.8:
            return AgreementLevel.GOOD
        elif excellent_pct + good_pct >= 0.5:
            return AgreementLevel.MODERATE
        elif critical_pct < 0.5:
            return AgreementLevel.POOR
        else:
            return AgreementLevel.CRITICAL


# Convenience functions

def validate_cross_model(
    model_outputs: List[np.ndarray],
    model_ids: Optional[List[str]] = None,
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.05,
) -> CrossModelResult:
    """
    Validate agreement between model outputs.

    Args:
        model_outputs: List of model output arrays
        model_ids: Optional list of model identifiers
        absolute_tolerance: Absolute tolerance for agreement
        relative_tolerance: Relative tolerance for agreement

    Returns:
        CrossModelResult with validation details
    """
    if model_ids is None:
        model_ids = [f"model_{i}" for i in range(len(model_outputs))]

    models = [
        ModelOutput(data=data, model_id=mid)
        for data, mid in zip(model_outputs, model_ids)
    ]

    config = CrossModelConfig(
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )

    validator = CrossModelValidator(config)
    return validator.validate(models)


def compare_two_models(
    model_a: np.ndarray,
    model_b: np.ndarray,
    model_a_id: str = "model_a",
    model_b_id: str = "model_b",
) -> PairwiseComparison:
    """
    Compare two model outputs.

    Args:
        model_a: First model output array
        model_b: Second model output array
        model_a_id: Identifier for first model
        model_b_id: Identifier for second model

    Returns:
        PairwiseComparison with comparison metrics
    """
    validator = CrossModelValidator()
    output_a = ModelOutput(data=model_a, model_id=model_a_id)
    output_b = ModelOutput(data=model_b, model_id=model_b_id)
    return validator._compare_two_models(output_a, output_b)


def get_ensemble_consensus(
    model_outputs: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get weighted consensus from multiple models.

    Args:
        model_outputs: List of model output arrays
        weights: Optional weights for each model

    Returns:
        Tuple of (consensus_data, spread)
    """
    if weights is None:
        weights = [1.0] * len(model_outputs)

    # Normalize weights (guard against all-zero weights)
    total = sum(weights)
    if total < 1e-10:
        # Fall back to equal weights if sum is zero
        weights = [1.0 / len(model_outputs)] * len(model_outputs)
    else:
        weights = [w / total for w in weights]

    # Stack data
    data_stack = np.stack([d.astype(np.float64) for d in model_outputs])

    # Weighted mean
    weighted_sum = sum(w * data_stack[i] for i, w in enumerate(weights))
    consensus = weighted_sum

    # Spread (std)
    spread = np.nanstd(data_stack, axis=0)

    return consensus, spread
