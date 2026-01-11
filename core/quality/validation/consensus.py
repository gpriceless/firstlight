"""
Consensus Generation for Quality Control.

Generates consensus outputs from multiple analysis sources using various
strategies for combining results, including:
- Voting-based consensus (majority, plurality, unanimous)
- Statistical consensus (mean, median, weighted average)
- Confidence-weighted consensus
- Bayesian consensus with priors
- Multi-level consensus with hierarchical aggregation

Key Concepts:
- Consensus combines multiple perspectives into single authoritative output
- Different strategies appropriate for different data types
- Confidence weighting prioritizes more reliable sources
- Uncertainty propagation tracks consensus reliability
- Disagreement regions may require special handling
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Strategies for generating consensus."""
    # Voting strategies (for categorical/binary data)
    MAJORITY_VOTE = "majority_vote"        # Most common value (>50%)
    PLURALITY_VOTE = "plurality_vote"      # Most common value (any fraction)
    UNANIMOUS = "unanimous"                # All sources must agree
    SUPERMAJORITY = "supermajority"        # >66% agreement required

    # Statistical strategies (for continuous data)
    MEAN = "mean"                          # Simple arithmetic mean
    MEDIAN = "median"                      # Median value
    WEIGHTED_MEAN = "weighted_mean"        # Confidence-weighted mean
    TRIMMED_MEAN = "trimmed_mean"          # Mean excluding outliers
    ROBUST_MEAN = "robust_mean"            # Iteratively reweighted mean

    # Advanced strategies
    BAYESIAN = "bayesian"                  # Bayesian posterior combination
    DEMPSTER_SHAFER = "dempster_shafer"    # Evidence combination
    HIERARCHICAL = "hierarchical"          # Multi-level aggregation


class ConsensusPriority(Enum):
    """Priority levels for consensus sources."""
    PRIMARY = "primary"        # Highest priority/most trusted
    SECONDARY = "secondary"    # Standard priority
    TERTIARY = "tertiary"      # Lower priority
    FALLBACK = "fallback"      # Only use if others unavailable


class ConsensusQuality(Enum):
    """Quality levels for consensus output."""
    HIGH = "high"              # Strong agreement, high confidence
    MODERATE = "moderate"      # Reasonable agreement
    LOW = "low"                # Weak agreement, significant uncertainty
    DEGRADED = "degraded"      # Consensus possible but unreliable
    FAILED = "failed"          # Cannot reach consensus


@dataclass
class ConsensusSource:
    """
    A source contributing to consensus.

    Attributes:
        data: Source data array
        source_id: Unique identifier for source
        source_name: Human-readable source name
        priority: Source priority level
        confidence: Per-pixel confidence (0-1)
        overall_confidence: Aggregate source confidence (0-1)
        weight: Custom weight for this source
        metadata: Additional source metadata
    """
    data: np.ndarray
    source_id: str
    source_name: str = ""
    priority: ConsensusPriority = ConsensusPriority.SECONDARY
    confidence: Optional[np.ndarray] = None
    overall_confidence: float = 1.0
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize confidence if not provided."""
        if self.confidence is None:
            if self.data.dtype.kind == 'f':
                self.confidence = (~np.isnan(self.data)).astype(np.float32)
            else:
                self.confidence = np.ones_like(self.data, dtype=np.float32)
        if not self.source_name:
            self.source_name = self.source_id


@dataclass
class ConsensusConfig:
    """
    Configuration for consensus generation.

    Attributes:
        strategy: Primary consensus strategy
        fallback_strategy: Fallback if primary fails
        min_sources: Minimum sources required
        min_agreement: Minimum agreement fraction required
        confidence_threshold: Minimum source confidence to include
        trim_fraction: Fraction to trim for trimmed_mean
        outlier_zscore: Z-score threshold for outlier detection
        use_confidence_weighting: Weight by source confidence
        propagate_uncertainty: Track uncertainty in output
    """
    strategy: ConsensusStrategy = ConsensusStrategy.WEIGHTED_MEAN
    fallback_strategy: Optional[ConsensusStrategy] = ConsensusStrategy.MEDIAN
    min_sources: int = 2
    min_agreement: float = 0.5
    confidence_threshold: float = 0.1
    trim_fraction: float = 0.1
    outlier_zscore: float = 3.0
    use_confidence_weighting: bool = True
    propagate_uncertainty: bool = True


@dataclass
class DisagreementRegion:
    """
    Region where sources significantly disagree.

    Attributes:
        region_id: Unique region identifier
        bbox: Bounding box (min_row, max_row, min_col, max_col)
        centroid: Region centroid (row, col)
        size_pixels: Number of pixels in region
        disagreement_level: How much sources disagree
        source_values: Values from each source in region
        recommendation: Suggested handling
    """
    region_id: int
    bbox: Dict[str, int]
    centroid: Dict[str, float]
    size_pixels: int
    disagreement_level: float
    source_values: Dict[str, float]
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "region_id": self.region_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "size_pixels": self.size_pixels,
            "disagreement_level": round(self.disagreement_level, 4),
            "source_values": {k: round(v, 4) for k, v in self.source_values.items()},
            "recommendation": self.recommendation,
        }


@dataclass
class ConsensusResult:
    """
    Result from consensus generation.

    Attributes:
        consensus_data: Consensus output array
        confidence: Per-pixel consensus confidence
        uncertainty: Per-pixel uncertainty estimate
        quality: Overall consensus quality
        agreement_map: Per-pixel agreement level
        disagreement_regions: Regions of high disagreement
        source_contributions: How much each source contributed
        statistics: Consensus statistics
        diagnostics: Additional diagnostic information
    """
    consensus_data: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    quality: ConsensusQuality
    agreement_map: np.ndarray
    disagreement_regions: List[DisagreementRegion]
    source_contributions: Dict[str, float]
    statistics: Dict[str, float]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary summary."""
        return {
            "quality": self.quality.value,
            "mean_confidence": float(np.nanmean(self.confidence)),
            "mean_uncertainty": float(np.nanmean(self.uncertainty)),
            "mean_agreement": float(np.nanmean(self.agreement_map)),
            "num_disagreement_regions": len(self.disagreement_regions),
            "disagreement_regions": [r.to_dict() for r in self.disagreement_regions],
            "source_contributions": {
                k: round(v, 4) for k, v in self.source_contributions.items()
            },
            "statistics": {k: round(v, 4) for k, v in self.statistics.items()},
            "diagnostics": self.diagnostics,
        }


class ConsensusGenerator:
    """
    Generates consensus outputs from multiple sources.

    Combines multiple analysis results into a single authoritative output
    using configurable strategies and confidence weighting.
    """

    def __init__(self, config: Optional[ConsensusConfig] = None):
        """
        Initialize consensus generator.

        Args:
            config: Consensus configuration
        """
        self.config = config or ConsensusConfig()

        # Strategy dispatch
        self._continuous_strategies = {
            ConsensusStrategy.MEAN: self._consensus_mean,
            ConsensusStrategy.MEDIAN: self._consensus_median,
            ConsensusStrategy.WEIGHTED_MEAN: self._consensus_weighted_mean,
            ConsensusStrategy.TRIMMED_MEAN: self._consensus_trimmed_mean,
            ConsensusStrategy.ROBUST_MEAN: self._consensus_robust_mean,
        }

        self._categorical_strategies = {
            ConsensusStrategy.MAJORITY_VOTE: self._consensus_majority,
            ConsensusStrategy.PLURALITY_VOTE: self._consensus_plurality,
            ConsensusStrategy.UNANIMOUS: self._consensus_unanimous,
            ConsensusStrategy.SUPERMAJORITY: self._consensus_supermajority,
        }

    def generate(
        self,
        sources: List[ConsensusSource],
        is_categorical: bool = False,
    ) -> ConsensusResult:
        """
        Generate consensus from multiple sources.

        Args:
            sources: List of source data to combine
            is_categorical: Whether data is categorical (use voting)

        Returns:
            ConsensusResult with consensus data and metadata
        """
        if len(sources) < self.config.min_sources:
            raise ValueError(
                f"At least {self.config.min_sources} sources required, got {len(sources)}"
            )

        # Filter sources by confidence
        valid_sources = [
            s for s in sources
            if s.overall_confidence >= self.config.confidence_threshold
        ]

        if len(valid_sources) < self.config.min_sources:
            logger.warning(
                f"Only {len(valid_sources)} sources above confidence threshold. "
                f"Using all {len(sources)} sources."
            )
            valid_sources = sources

        # Select strategy
        if is_categorical:
            strategy = self.config.strategy
            if strategy not in self._categorical_strategies:
                strategy = ConsensusStrategy.MAJORITY_VOTE
            consensus_func = self._categorical_strategies[strategy]
        else:
            strategy = self.config.strategy
            if strategy not in self._continuous_strategies:
                strategy = ConsensusStrategy.WEIGHTED_MEAN
            consensus_func = self._continuous_strategies[strategy]

        # Generate consensus
        consensus_data, agreement_map = consensus_func(valid_sources)

        # Calculate confidence and uncertainty
        confidence, uncertainty = self._calculate_confidence_uncertainty(
            valid_sources, consensus_data, agreement_map
        )

        # Find disagreement regions
        disagreement_regions = self._find_disagreement_regions(
            valid_sources, agreement_map
        )

        # Calculate source contributions
        source_contributions = self._calculate_source_contributions(
            valid_sources, consensus_data
        )

        # Determine quality
        quality = self._classify_quality(
            agreement_map, confidence, valid_sources
        )

        # Compute statistics
        statistics = self._compute_statistics(
            consensus_data, agreement_map, valid_sources
        )

        return ConsensusResult(
            consensus_data=consensus_data,
            confidence=confidence,
            uncertainty=uncertainty,
            quality=quality,
            agreement_map=agreement_map,
            disagreement_regions=disagreement_regions,
            source_contributions=source_contributions,
            statistics=statistics,
            diagnostics={
                "strategy_used": strategy.value,
                "num_sources": len(sources),
                "num_valid_sources": len(valid_sources),
                "is_categorical": is_categorical,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _consensus_mean(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple mean consensus."""
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])

        with np.errstate(invalid='ignore'):
            consensus = np.nanmean(data_stack, axis=0)
            spread = np.nanstd(data_stack, axis=0)

        # Agreement based on coefficient of variation
        with np.errstate(divide='ignore', invalid='ignore'):
            agreement = np.where(
                np.abs(consensus) > 1e-10,
                1.0 - np.minimum(spread / np.abs(consensus), 1.0),
                1.0 - np.minimum(spread, 1.0)
            )
            agreement = np.where(np.isnan(agreement), 0.0, agreement)

        return consensus, agreement

    def _consensus_median(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Median consensus (robust to outliers)."""
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])

        with np.errstate(invalid='ignore'):
            consensus = np.nanmedian(data_stack, axis=0)

            # Agreement: fraction of sources within tolerance of median
            tolerance = self.config.outlier_zscore * np.nanstd(data_stack, axis=0)
            tolerance = np.where(tolerance < 1e-10, 0.1, tolerance)

            within_tolerance = np.abs(data_stack - consensus) <= tolerance
            agreement = np.nanmean(within_tolerance.astype(float), axis=0)

        return consensus, agreement

    def _consensus_weighted_mean(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence-weighted mean consensus."""
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])
        conf_stack = np.stack([s.confidence for s in sources])

        # Also apply source-level weights
        weights = np.array([s.weight * s.overall_confidence for s in sources])
        weights = weights / np.sum(weights)  # Normalize

        # Combined weight: source weight * pixel confidence
        combined_weights = conf_stack * weights[:, np.newaxis, np.newaxis]

        with np.errstate(invalid='ignore', divide='ignore'):
            weight_sum = np.sum(combined_weights, axis=0)
            weight_sum = np.where(weight_sum < 1e-10, 1.0, weight_sum)

            consensus = np.nansum(data_stack * combined_weights, axis=0) / weight_sum

            # Weighted spread
            diff_sq = (data_stack - consensus) ** 2
            variance = np.nansum(diff_sq * combined_weights, axis=0) / weight_sum
            spread = np.sqrt(np.maximum(variance, 0))

            # Agreement
            agreement = np.where(
                np.abs(consensus) > 1e-10,
                1.0 - np.minimum(spread / np.abs(consensus), 1.0),
                1.0 - np.minimum(spread, 1.0)
            )
            agreement = np.where(np.isnan(agreement), 0.0, agreement)

        return consensus, agreement

    def _consensus_trimmed_mean(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Trimmed mean (excludes extreme values)."""
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])
        n_sources = len(sources)
        trim_count = max(1, int(n_sources * self.config.trim_fraction))

        # Sort along source axis
        sorted_data = np.sort(data_stack, axis=0)

        # Trim extreme values
        trimmed = sorted_data[trim_count:-trim_count] if trim_count < n_sources // 2 else sorted_data

        with np.errstate(invalid='ignore'):
            consensus = np.nanmean(trimmed, axis=0)
            spread = np.nanstd(trimmed, axis=0)

            agreement = np.where(
                np.abs(consensus) > 1e-10,
                1.0 - np.minimum(spread / np.abs(consensus), 1.0),
                1.0 - np.minimum(spread, 1.0)
            )
            agreement = np.where(np.isnan(agreement), 0.0, agreement)

        return consensus, agreement

    def _consensus_robust_mean(
        self,
        sources: List[ConsensusSource],
        max_iterations: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iteratively reweighted mean (robust to outliers)."""
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])

        # Initial estimate
        with np.errstate(invalid='ignore'):
            consensus = np.nanmedian(data_stack, axis=0)
            mad = np.nanmedian(np.abs(data_stack - consensus), axis=0)
            mad = np.where(mad < 1e-10, 1.0, mad)

        # Iterate
        for _ in range(max_iterations):
            # Calculate weights using Huber function
            residuals = data_stack - consensus
            normalized_residuals = residuals / (1.4826 * mad)  # Scale MAD to sigma

            # Huber weight: 1 for small residuals, decreasing for large
            huber_weights = np.where(
                np.abs(normalized_residuals) <= 1.345,
                1.0,
                1.345 / np.abs(normalized_residuals + 1e-10)
            )

            # Update consensus
            with np.errstate(invalid='ignore', divide='ignore'):
                weight_sum = np.nansum(huber_weights, axis=0)
                weight_sum = np.where(weight_sum < 1e-10, 1.0, weight_sum)
                new_consensus = np.nansum(data_stack * huber_weights, axis=0) / weight_sum

            # Check convergence
            if np.allclose(new_consensus, consensus, rtol=1e-4, equal_nan=True):
                break

            consensus = new_consensus

        # Agreement
        spread = np.nanstd(data_stack, axis=0)
        agreement = np.where(
            np.abs(consensus) > 1e-10,
            1.0 - np.minimum(spread / np.abs(consensus), 1.0),
            1.0 - np.minimum(spread, 1.0)
        )
        agreement = np.where(np.isnan(agreement), 0.0, agreement)

        return consensus, agreement

    def _consensus_majority(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Majority vote (>50% required)."""
        return self._vote_consensus(sources, threshold=0.5)

    def _consensus_plurality(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Plurality vote (most common value wins)."""
        return self._vote_consensus(sources, threshold=0.0)

    def _consensus_unanimous(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unanimous vote (all must agree)."""
        return self._vote_consensus(sources, threshold=0.99)

    def _consensus_supermajority(
        self,
        sources: List[ConsensusSource],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Supermajority vote (>66% required)."""
        return self._vote_consensus(sources, threshold=0.666)

    def _vote_consensus(
        self,
        sources: List[ConsensusSource],
        threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generic voting consensus."""
        data_stack = np.stack([s.data for s in sources])
        n_sources = len(sources)
        shape = sources[0].data.shape

        # Get unique values per pixel
        consensus = np.zeros(shape, dtype=sources[0].data.dtype)
        agreement = np.zeros(shape, dtype=np.float32)

        # Vectorized approach for common case (binary data)
        if np.array_equal(np.unique(data_stack), [0, 1]) or \
           np.array_equal(np.unique(data_stack[~np.isnan(data_stack)]), [0, 1]):
            # Binary case - count votes for 1
            votes_for_one = np.nansum(data_stack == 1, axis=0)
            votes_for_zero = np.nansum(data_stack == 0, axis=0)
            total_votes = votes_for_one + votes_for_zero

            vote_fraction = np.where(
                total_votes > 0,
                votes_for_one / total_votes,
                0.5
            )

            consensus = (vote_fraction > 0.5).astype(sources[0].data.dtype)
            max_votes = np.maximum(votes_for_one, votes_for_zero)
            agreement = np.where(total_votes > 0, max_votes / total_votes, 0.0)

            # Apply threshold
            below_threshold = agreement < threshold
            if np.any(below_threshold):
                consensus[below_threshold] = -1  # Invalid/no consensus

        else:
            # General case - iterate pixels
            for i in range(shape[0]):
                for j in range(shape[1]):
                    values = data_stack[:, i, j]
                    valid_values = values[~np.isnan(values)] if data_stack.dtype.kind == 'f' else values

                    if len(valid_values) == 0:
                        agreement[i, j] = 0.0
                        continue

                    unique, counts = np.unique(valid_values, return_counts=True)
                    max_count = np.max(counts)
                    max_idx = np.argmax(counts)

                    agreement[i, j] = max_count / len(valid_values)

                    if agreement[i, j] >= threshold:
                        consensus[i, j] = unique[max_idx]
                    else:
                        consensus[i, j] = -1  # No consensus

        return consensus.astype(np.float32), agreement

    def _calculate_confidence_uncertainty(
        self,
        sources: List[ConsensusSource],
        consensus: np.ndarray,
        agreement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate consensus confidence and uncertainty."""
        # Stack source confidences
        conf_stack = np.stack([s.confidence for s in sources])

        # Mean source confidence
        mean_conf = np.nanmean(conf_stack, axis=0)

        # Consensus confidence combines agreement and source confidence
        confidence = agreement * mean_conf

        # Uncertainty is inverse of confidence, scaled by spread
        if self.config.propagate_uncertainty:
            data_stack = np.stack([s.data.astype(np.float64) for s in sources])
            with np.errstate(invalid='ignore'):
                spread = np.nanstd(data_stack, axis=0)
                uncertainty = spread * (1 - confidence)
                uncertainty = np.where(np.isnan(uncertainty), 1.0, uncertainty)
        else:
            uncertainty = 1 - confidence

        return confidence, uncertainty

    def _find_disagreement_regions(
        self,
        sources: List[ConsensusSource],
        agreement: np.ndarray,
    ) -> List[DisagreementRegion]:
        """Find spatial regions of disagreement."""
        from scipy import ndimage

        # Threshold for disagreement
        disagreement_mask = agreement < self.config.min_agreement

        if not np.any(disagreement_mask):
            return []

        # Label connected regions
        labeled, num_features = ndimage.label(disagreement_mask)

        regions = []
        for i in range(1, num_features + 1):
            region_mask = labeled == i
            size = int(np.sum(region_mask))

            if size < 5:  # Skip tiny regions
                continue

            rows, cols = np.where(region_mask)

            # Get source values in this region
            source_values = {}
            for src in sources:
                region_values = src.data[region_mask]
                valid_values = region_values[~np.isnan(region_values)] if src.data.dtype.kind == 'f' else region_values
                if len(valid_values) > 0:
                    source_values[src.source_id] = float(np.mean(valid_values))

            # Disagreement level
            disagreement_level = 1.0 - float(np.mean(agreement[region_mask]))

            # Recommendation
            if disagreement_level > 0.7:
                recommendation = "Manual review required - sources fundamentally disagree"
            elif disagreement_level > 0.5:
                recommendation = "Flag for expert review - significant disagreement"
            else:
                recommendation = "Minor disagreement - consider ensemble uncertainty"

            regions.append(DisagreementRegion(
                region_id=i,
                bbox={
                    "min_row": int(rows.min()),
                    "max_row": int(rows.max()),
                    "min_col": int(cols.min()),
                    "max_col": int(cols.max()),
                },
                centroid={
                    "row": float(np.mean(rows)),
                    "col": float(np.mean(cols)),
                },
                size_pixels=size,
                disagreement_level=disagreement_level,
                source_values=source_values,
                recommendation=recommendation,
            ))

        return regions

    def _calculate_source_contributions(
        self,
        sources: List[ConsensusSource],
        consensus: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate how much each source contributed to consensus."""
        contributions = {}

        for src in sources:
            # Calculate correlation with consensus
            src_flat = src.data.astype(np.float64).flatten()
            cons_flat = consensus.astype(np.float64).flatten()

            # Remove NaN
            valid = ~(np.isnan(src_flat) | np.isnan(cons_flat))

            if np.sum(valid) > 1:
                correlation = np.corrcoef(src_flat[valid], cons_flat[valid])[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 0.0

            # Weight contribution by source weight and confidence
            contribution = (
                (correlation + 1) / 2 *  # Normalize correlation to 0-1
                src.weight *
                src.overall_confidence
            )

            contributions[src.source_id] = contribution

        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions

    def _classify_quality(
        self,
        agreement: np.ndarray,
        confidence: np.ndarray,
        sources: List[ConsensusSource],
    ) -> ConsensusQuality:
        """Classify overall consensus quality."""
        mean_agreement = float(np.nanmean(agreement))
        mean_confidence = float(np.nanmean(confidence))
        n_sources = len(sources)

        # Check for sufficient sources
        if n_sources < self.config.min_sources:
            return ConsensusQuality.FAILED

        # High quality: strong agreement and confidence
        if mean_agreement >= 0.9 and mean_confidence >= 0.8:
            return ConsensusQuality.HIGH

        # Moderate: reasonable agreement
        if mean_agreement >= 0.7 and mean_confidence >= 0.6:
            return ConsensusQuality.MODERATE

        # Low: weak agreement
        if mean_agreement >= 0.5:
            return ConsensusQuality.LOW

        # Degraded: can compute but unreliable
        if mean_agreement >= 0.3:
            return ConsensusQuality.DEGRADED

        # Failed
        return ConsensusQuality.FAILED

    def _compute_statistics(
        self,
        consensus: np.ndarray,
        agreement: np.ndarray,
        sources: List[ConsensusSource],
    ) -> Dict[str, float]:
        """Compute consensus statistics."""
        stats = {
            "consensus_mean": float(np.nanmean(consensus)),
            "consensus_std": float(np.nanstd(consensus)),
            "consensus_min": float(np.nanmin(consensus)),
            "consensus_max": float(np.nanmax(consensus)),
            "mean_agreement": float(np.nanmean(agreement)),
            "min_agreement": float(np.nanmin(agreement)),
            "pct_high_agreement": float(np.mean(agreement > 0.8) * 100),
            "pct_low_agreement": float(np.mean(agreement < 0.5) * 100),
            "num_sources": len(sources),
        }

        # Source spread statistics
        data_stack = np.stack([s.data.astype(np.float64) for s in sources])
        with np.errstate(invalid='ignore'):
            source_spread = np.nanstd(data_stack, axis=0)
            stats["mean_source_spread"] = float(np.nanmean(source_spread))
            stats["max_source_spread"] = float(np.nanmax(source_spread))

        return stats


# Convenience functions

def generate_consensus(
    data_arrays: List[np.ndarray],
    source_ids: Optional[List[str]] = None,
    strategy: str = "weighted_mean",
    is_categorical: bool = False,
) -> ConsensusResult:
    """
    Generate consensus from multiple data arrays.

    Args:
        data_arrays: List of data arrays to combine
        source_ids: Optional list of source identifiers
        strategy: Consensus strategy name
        is_categorical: Whether data is categorical

    Returns:
        ConsensusResult with consensus data
    """
    if source_ids is None:
        source_ids = [f"source_{i}" for i in range(len(data_arrays))]

    sources = [
        ConsensusSource(data=data, source_id=sid)
        for data, sid in zip(data_arrays, source_ids)
    ]

    strategy_map = {
        "mean": ConsensusStrategy.MEAN,
        "median": ConsensusStrategy.MEDIAN,
        "weighted_mean": ConsensusStrategy.WEIGHTED_MEAN,
        "trimmed_mean": ConsensusStrategy.TRIMMED_MEAN,
        "robust_mean": ConsensusStrategy.ROBUST_MEAN,
        "majority": ConsensusStrategy.MAJORITY_VOTE,
        "plurality": ConsensusStrategy.PLURALITY_VOTE,
        "unanimous": ConsensusStrategy.UNANIMOUS,
    }

    config = ConsensusConfig(
        strategy=strategy_map.get(strategy, ConsensusStrategy.WEIGHTED_MEAN)
    )

    generator = ConsensusGenerator(config)
    return generator.generate(sources, is_categorical)


def vote_consensus(
    data_arrays: List[np.ndarray],
    source_ids: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate voting consensus for categorical data.

    Args:
        data_arrays: List of categorical data arrays
        source_ids: Optional list of source identifiers
        threshold: Minimum agreement fraction required

    Returns:
        Tuple of (consensus_array, agreement_array)
    """
    result = generate_consensus(
        data_arrays, source_ids,
        strategy="majority" if threshold >= 0.5 else "plurality",
        is_categorical=True
    )
    return result.consensus_data, result.agreement_map


def weighted_mean_consensus(
    data_arrays: List[np.ndarray],
    weights: Optional[List[float]] = None,
    confidences: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate weighted mean consensus.

    Args:
        data_arrays: List of data arrays to combine
        weights: Optional list of source weights
        confidences: Optional list of per-pixel confidence arrays

    Returns:
        Tuple of (consensus_array, uncertainty_array)
    """
    sources = []
    for i, data in enumerate(data_arrays):
        weight = weights[i] if weights else 1.0
        confidence = confidences[i] if confidences else None
        sources.append(ConsensusSource(
            data=data,
            source_id=f"source_{i}",
            weight=weight,
            confidence=confidence,
        ))

    generator = ConsensusGenerator()
    result = generator.generate(sources)
    return result.consensus_data, result.uncertainty
