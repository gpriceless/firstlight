"""
Low Confidence Handler.

Handles situations where individual algorithm confidence is weak by:
- Ensemble methods combining multiple algorithms
- Multi-algorithm voting strategies
- Confidence-weighted combination
- Flagging outputs for manual review when disagreement is high

When no single algorithm provides high confidence results, this module
provides mechanisms to combine multiple weak signals into a more
reliable output while clearly marking areas of uncertainty.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


class VotingMethod(Enum):
    """Methods for combining algorithm outputs through voting."""
    MAJORITY = "majority"            # Simple majority vote
    WEIGHTED_MAJORITY = "weighted_majority"  # Confidence-weighted majority
    UNANIMOUS = "unanimous"          # Require all algorithms to agree
    CONSENSUS = "consensus"          # Agreement above threshold
    CONFIDENCE_MAX = "confidence_max"  # Use highest confidence output
    WEIGHTED_MEAN = "weighted_mean"  # Weighted average of outputs


class CombinationMethod(Enum):
    """Methods for combining continuous algorithm outputs."""
    MEAN = "mean"                    # Simple average
    WEIGHTED_MEAN = "weighted_mean"  # Confidence-weighted average
    MEDIAN = "median"                # Median value
    WEIGHTED_MEDIAN = "weighted_median"  # Weighted median
    MAX_CONFIDENCE = "max_confidence"  # Use highest confidence value
    BAYESIAN = "bayesian"            # Bayesian combination


class DisagreementLevel(Enum):
    """Levels of disagreement between algorithms."""
    NONE = "none"           # Full agreement
    LOW = "low"             # Minor disagreement, acceptable
    MEDIUM = "medium"       # Moderate disagreement, flag for review
    HIGH = "high"           # Major disagreement, requires review
    CRITICAL = "critical"   # Algorithms contradict, manual review required


@dataclass
class AlgorithmResult:
    """
    Result from a single algorithm.

    Attributes:
        algorithm_id: Unique identifier for the algorithm
        algorithm_name: Human-readable algorithm name
        data: Output data array
        confidence: Per-pixel confidence scores (0-1)
        overall_confidence: Overall confidence for this result
        metadata: Additional result metadata
        timestamp: When result was generated
    """
    algorithm_id: str
    algorithm_name: str
    data: np.ndarray
    confidence: np.ndarray
    overall_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get data shape."""
        return self.data.shape

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "overall_confidence": round(self.overall_confidence, 3),
            "data_shape": list(self.data.shape),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DisagreementRegion:
    """
    Region where algorithms disagree.

    Attributes:
        region_id: Unique identifier
        bounds: Bounding box [min_row, min_col, max_row, max_col]
        level: Disagreement level
        mask: Boolean mask of disagreement pixels
        algorithms_disagree: List of algorithm pairs that disagree
        pixel_count: Number of pixels in this region
        mean_disagreement: Mean disagreement score in region
    """
    region_id: str
    bounds: Tuple[int, int, int, int]
    level: DisagreementLevel
    mask: np.ndarray
    algorithms_disagree: List[Tuple[str, str]]
    pixel_count: int
    mean_disagreement: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without mask array)."""
        return {
            "region_id": self.region_id,
            "bounds": self.bounds,
            "level": self.level.value,
            "algorithms_disagree": self.algorithms_disagree,
            "pixel_count": self.pixel_count,
            "mean_disagreement": round(self.mean_disagreement, 3),
        }


@dataclass
class EnsembleResult:
    """
    Result of ensemble combination.

    Attributes:
        data: Combined output data
        confidence: Per-pixel confidence scores
        overall_confidence: Overall ensemble confidence
        method: Method used for combination
        agreement_map: Per-pixel agreement levels
        disagreement_regions: Regions of significant disagreement
        algorithms_used: List of algorithms in the ensemble
        requires_review: Whether result requires manual review
        review_reason: Reason for requiring review (if applicable)
        metadata: Additional result metadata
    """
    data: np.ndarray
    confidence: np.ndarray
    overall_confidence: float
    method: Union[VotingMethod, CombinationMethod]
    agreement_map: np.ndarray
    disagreement_regions: List[DisagreementRegion]
    algorithms_used: List[str]
    requires_review: bool
    review_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def disagreement_count(self) -> int:
        """Get number of disagreement regions."""
        return len(self.disagreement_regions)

    @property
    def high_disagreement_count(self) -> int:
        """Get number of high/critical disagreement regions."""
        return sum(
            1 for r in self.disagreement_regions
            if r.level in (DisagreementLevel.HIGH, DisagreementLevel.CRITICAL)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "method": self.method.value,
            "algorithms_used": self.algorithms_used,
            "requires_review": self.requires_review,
            "review_reason": self.review_reason,
            "disagreement_count": self.disagreement_count,
            "high_disagreement_count": self.high_disagreement_count,
            "disagreement_regions": [r.to_dict() for r in self.disagreement_regions],
            "data_shape": list(self.data.shape),
            "metadata": self.metadata,
        }


@dataclass
class LowConfidenceConfig:
    """
    Configuration for low confidence handling.

    Attributes:
        min_confidence_threshold: Minimum acceptable confidence
        voting_method: Default voting method for binary outputs
        combination_method: Default method for continuous outputs
        agreement_threshold: Threshold for consensus agreement (0-1)
        disagreement_low_threshold: Threshold for low disagreement
        disagreement_medium_threshold: Threshold for medium disagreement
        disagreement_high_threshold: Threshold for high disagreement
        require_review_on_high_disagreement: Flag for review on high disagreement
        min_algorithms_for_ensemble: Minimum algorithms for ensemble
        confidence_weight_power: Power for confidence weighting
    """
    min_confidence_threshold: float = 0.3
    voting_method: VotingMethod = VotingMethod.WEIGHTED_MAJORITY
    combination_method: CombinationMethod = CombinationMethod.WEIGHTED_MEAN
    agreement_threshold: float = 0.7
    disagreement_low_threshold: float = 0.2
    disagreement_medium_threshold: float = 0.4
    disagreement_high_threshold: float = 0.6
    require_review_on_high_disagreement: bool = True
    min_algorithms_for_ensemble: int = 2
    confidence_weight_power: float = 2.0


class LowConfidenceHandler:
    """
    Handles low confidence situations through ensemble methods.

    Combines multiple algorithm outputs to produce more reliable
    results when individual algorithms have low confidence.

    Example:
        handler = LowConfidenceHandler()

        # Combine multiple algorithm results
        result = handler.ensemble_voting([
            AlgorithmResult(algorithm_id="algo1", ...),
            AlgorithmResult(algorithm_id="algo2", ...),
            AlgorithmResult(algorithm_id="algo3", ...),
        ])

        if result.requires_review:
            print(f"Manual review required: {result.review_reason}")
    """

    def __init__(self, config: Optional[LowConfidenceConfig] = None):
        """
        Initialize the low confidence handler.

        Args:
            config: Configuration options
        """
        self.config = config or LowConfidenceConfig()

    def ensemble_voting(
        self,
        results: List[AlgorithmResult],
        method: Optional[VotingMethod] = None,
        threshold: float = 0.5,
    ) -> EnsembleResult:
        """
        Combine binary algorithm outputs through voting.

        Args:
            results: List of algorithm results
            method: Voting method (uses config default if None)
            threshold: Threshold for binary classification

        Returns:
            EnsembleResult with combined output
        """
        if len(results) < self.config.min_algorithms_for_ensemble:
            raise ValueError(
                f"Need at least {self.config.min_algorithms_for_ensemble} "
                f"algorithms for ensemble, got {len(results)}"
            )

        method = method or self.config.voting_method

        # Validate shapes match
        shapes = [r.data.shape for r in results]
        if len(set(shapes)) > 1:
            raise ValueError(f"Algorithm output shapes don't match: {shapes}")

        shape = shapes[0]
        n_algorithms = len(results)

        # Stack binary outputs
        binary_stack = np.stack([r.data > threshold for r in results], axis=0)
        confidence_stack = np.stack([r.confidence for r in results], axis=0)

        # Apply voting method
        if method == VotingMethod.MAJORITY:
            combined, confidence = self._majority_vote(binary_stack, confidence_stack)
        elif method == VotingMethod.WEIGHTED_MAJORITY:
            combined, confidence = self._weighted_majority_vote(binary_stack, confidence_stack)
        elif method == VotingMethod.UNANIMOUS:
            combined, confidence = self._unanimous_vote(binary_stack, confidence_stack)
        elif method == VotingMethod.CONSENSUS:
            combined, confidence = self._consensus_vote(binary_stack, confidence_stack)
        elif method == VotingMethod.CONFIDENCE_MAX:
            combined, confidence = self._confidence_max_vote(binary_stack, confidence_stack)
        else:
            # Default to weighted majority
            combined, confidence = self._weighted_majority_vote(binary_stack, confidence_stack)

        # Calculate agreement map
        agreement_map = self._calculate_agreement(binary_stack)

        # Identify disagreement regions
        disagreement_regions = self._identify_disagreement_regions(
            agreement_map, results
        )

        # Determine if review is needed
        requires_review, review_reason = self._check_review_requirement(
            confidence, disagreement_regions
        )

        # Calculate overall confidence
        overall_confidence = float(np.mean(confidence))

        return EnsembleResult(
            data=combined.astype(np.float32),
            confidence=confidence,
            overall_confidence=overall_confidence,
            method=method,
            agreement_map=agreement_map,
            disagreement_regions=disagreement_regions,
            algorithms_used=[r.algorithm_id for r in results],
            requires_review=requires_review,
            review_reason=review_reason,
            metadata={
                "n_algorithms": n_algorithms,
                "voting_threshold": threshold,
            },
        )

    def ensemble_combination(
        self,
        results: List[AlgorithmResult],
        method: Optional[CombinationMethod] = None,
    ) -> EnsembleResult:
        """
        Combine continuous algorithm outputs through averaging/fusion.

        Args:
            results: List of algorithm results
            method: Combination method (uses config default if None)

        Returns:
            EnsembleResult with combined output
        """
        if len(results) < self.config.min_algorithms_for_ensemble:
            raise ValueError(
                f"Need at least {self.config.min_algorithms_for_ensemble} "
                f"algorithms for ensemble, got {len(results)}"
            )

        method = method or self.config.combination_method

        # Validate shapes match
        shapes = [r.data.shape for r in results]
        if len(set(shapes)) > 1:
            raise ValueError(f"Algorithm output shapes don't match: {shapes}")

        # Stack outputs
        data_stack = np.stack([r.data for r in results], axis=0)
        confidence_stack = np.stack([r.confidence for r in results], axis=0)

        # Apply combination method
        if method == CombinationMethod.MEAN:
            combined, confidence = self._mean_combination(data_stack, confidence_stack)
        elif method == CombinationMethod.WEIGHTED_MEAN:
            combined, confidence = self._weighted_mean_combination(data_stack, confidence_stack)
        elif method == CombinationMethod.MEDIAN:
            combined, confidence = self._median_combination(data_stack, confidence_stack)
        elif method == CombinationMethod.MAX_CONFIDENCE:
            combined, confidence = self._max_confidence_combination(data_stack, confidence_stack)
        else:
            # Default to weighted mean
            combined, confidence = self._weighted_mean_combination(data_stack, confidence_stack)

        # Calculate agreement based on variance
        variance = np.var(data_stack, axis=0)
        max_variance = np.max(variance) if np.max(variance) > 0 else 1.0
        agreement_map = 1.0 - (variance / max_variance)

        # Identify disagreement regions
        disagreement_regions = self._identify_disagreement_regions_continuous(
            variance, results
        )

        # Determine if review is needed
        requires_review, review_reason = self._check_review_requirement(
            confidence, disagreement_regions
        )

        # Calculate overall confidence
        overall_confidence = float(np.mean(confidence))

        return EnsembleResult(
            data=combined,
            confidence=confidence,
            overall_confidence=overall_confidence,
            method=method,
            agreement_map=agreement_map,
            disagreement_regions=disagreement_regions,
            algorithms_used=[r.algorithm_id for r in results],
            requires_review=requires_review,
            review_reason=review_reason,
            metadata={
                "n_algorithms": len(results),
            },
        )

    def flag_for_review(
        self,
        result: EnsembleResult,
        reason: str,
    ) -> EnsembleResult:
        """
        Flag an ensemble result for manual review.

        Args:
            result: Ensemble result to flag
            reason: Reason for flagging

        Returns:
            Updated EnsembleResult with review flag
        """
        result.requires_review = True
        result.review_reason = reason
        return result

    def _majority_vote(
        self,
        binary_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple majority voting."""
        n_algorithms = binary_stack.shape[0]
        vote_sum = np.sum(binary_stack, axis=0)

        # Majority is >50%
        combined = vote_sum > (n_algorithms / 2)

        # Confidence based on agreement
        agreement = vote_sum / n_algorithms
        confidence = np.where(
            combined,
            agreement,
            1.0 - agreement
        ).astype(np.float32)

        return combined, confidence

    def _weighted_majority_vote(
        self,
        binary_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence-weighted majority voting."""
        power = self.config.confidence_weight_power
        weights = confidence_stack ** power

        # Weighted sum
        weighted_yes = np.sum(binary_stack * weights, axis=0)
        total_weight = np.sum(weights, axis=0)

        # Avoid division by zero
        total_weight = np.maximum(total_weight, 1e-10)

        weighted_agreement = weighted_yes / total_weight
        combined = weighted_agreement > 0.5

        # Confidence
        confidence = np.where(
            combined,
            weighted_agreement,
            1.0 - weighted_agreement
        ).astype(np.float32)

        return combined, confidence

    def _unanimous_vote(
        self,
        binary_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unanimous voting - all must agree."""
        all_yes = np.all(binary_stack, axis=0)
        all_no = np.all(~binary_stack, axis=0)

        # Combined result (True only if all agree yes)
        combined = all_yes

        # High confidence only if unanimous
        min_confidence = np.min(confidence_stack, axis=0)
        confidence = np.where(
            all_yes | all_no,
            min_confidence,
            0.3  # Low confidence when not unanimous
        ).astype(np.float32)

        return combined, confidence

    def _consensus_vote(
        self,
        binary_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Consensus voting - agreement above threshold."""
        n_algorithms = binary_stack.shape[0]
        threshold = self.config.agreement_threshold

        vote_sum = np.sum(binary_stack, axis=0)
        agreement = vote_sum / n_algorithms

        # Consensus reached if agreement above threshold
        consensus_yes = agreement >= threshold
        consensus_no = agreement <= (1 - threshold)

        combined = consensus_yes

        # Confidence based on agreement strength
        confidence = np.where(
            consensus_yes | consensus_no,
            np.abs(agreement - 0.5) * 2,  # Scale to 0-1
            0.3  # Low confidence when no consensus
        ).astype(np.float32)

        return combined, confidence

    def _confidence_max_vote(
        self,
        binary_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use result from highest confidence algorithm at each pixel."""
        max_conf_idx = np.argmax(confidence_stack, axis=0)

        # Get result from highest confidence algorithm
        rows, cols = np.meshgrid(
            np.arange(binary_stack.shape[1]),
            np.arange(binary_stack.shape[2]),
            indexing='ij'
        )
        combined = binary_stack[max_conf_idx, rows, cols]
        confidence = np.max(confidence_stack, axis=0)

        return combined, confidence

    def _mean_combination(
        self,
        data_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple mean combination."""
        combined = np.nanmean(data_stack, axis=0)
        confidence = np.nanmean(confidence_stack, axis=0)
        return combined, confidence

    def _weighted_mean_combination(
        self,
        data_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Confidence-weighted mean combination."""
        power = self.config.confidence_weight_power
        weights = confidence_stack ** power

        # Weighted average
        total_weight = np.sum(weights, axis=0)
        total_weight = np.maximum(total_weight, 1e-10)

        combined = np.sum(data_stack * weights, axis=0) / total_weight
        confidence = np.sum(confidence_stack * weights, axis=0) / total_weight

        return combined, confidence

    def _median_combination(
        self,
        data_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Median combination."""
        combined = np.nanmedian(data_stack, axis=0)
        confidence = np.nanmedian(confidence_stack, axis=0)
        return combined, confidence

    def _max_confidence_combination(
        self,
        data_stack: np.ndarray,
        confidence_stack: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use value from highest confidence algorithm."""
        max_conf_idx = np.argmax(confidence_stack, axis=0)

        rows, cols = np.meshgrid(
            np.arange(data_stack.shape[1]),
            np.arange(data_stack.shape[2]),
            indexing='ij'
        )
        combined = data_stack[max_conf_idx, rows, cols]
        confidence = np.max(confidence_stack, axis=0)

        return combined, confidence

    def _calculate_agreement(self, binary_stack: np.ndarray) -> np.ndarray:
        """Calculate agreement map from binary outputs."""
        n_algorithms = binary_stack.shape[0]
        vote_sum = np.sum(binary_stack, axis=0)

        # Agreement is how close to unanimous (0=50/50, 1=unanimous)
        agreement = np.abs(vote_sum - n_algorithms / 2) / (n_algorithms / 2)
        return agreement.astype(np.float32)

    def _identify_disagreement_regions(
        self,
        agreement_map: np.ndarray,
        results: List[AlgorithmResult],
    ) -> List[DisagreementRegion]:
        """Identify regions of significant disagreement."""
        from scipy import ndimage

        # Find low agreement regions
        low_agreement = agreement_map < (1 - self.config.disagreement_medium_threshold)

        if not np.any(low_agreement):
            return []

        # Label connected components
        labeled, num_features = ndimage.label(low_agreement)

        regions = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            pixel_count = np.sum(mask)

            if pixel_count < 10:  # Skip tiny regions
                continue

            # Get bounds
            rows, cols = np.where(mask)
            bounds = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

            # Calculate disagreement level
            mean_agreement = np.mean(agreement_map[mask])
            level = self._get_disagreement_level(1 - mean_agreement)

            # Identify which algorithms disagree
            disagree_pairs = self._find_disagreeing_pairs(results, mask)

            region = DisagreementRegion(
                region_id=f"disagree_{i}",
                bounds=bounds,
                level=level,
                mask=mask,
                algorithms_disagree=disagree_pairs,
                pixel_count=int(pixel_count),
                mean_disagreement=float(1 - mean_agreement),
            )
            regions.append(region)

        return regions

    def _identify_disagreement_regions_continuous(
        self,
        variance: np.ndarray,
        results: List[AlgorithmResult],
    ) -> List[DisagreementRegion]:
        """Identify disagreement regions for continuous outputs."""
        from scipy import ndimage

        # Normalize variance to 0-1
        max_var = np.max(variance)
        if max_var == 0:
            return []

        normalized_var = variance / max_var

        # Find high variance regions
        high_variance = normalized_var > self.config.disagreement_medium_threshold

        if not np.any(high_variance):
            return []

        # Label connected components
        labeled, num_features = ndimage.label(high_variance)

        regions = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            pixel_count = np.sum(mask)

            if pixel_count < 10:
                continue

            rows, cols = np.where(mask)
            bounds = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

            mean_disagreement = float(np.mean(normalized_var[mask]))
            level = self._get_disagreement_level(mean_disagreement)

            disagree_pairs = []  # Can't easily determine pairs for continuous

            region = DisagreementRegion(
                region_id=f"disagree_{i}",
                bounds=bounds,
                level=level,
                mask=mask,
                algorithms_disagree=disagree_pairs,
                pixel_count=int(pixel_count),
                mean_disagreement=mean_disagreement,
            )
            regions.append(region)

        return regions

    def _get_disagreement_level(self, disagreement: float) -> DisagreementLevel:
        """Map disagreement score to level."""
        if disagreement >= self.config.disagreement_high_threshold:
            return DisagreementLevel.CRITICAL
        elif disagreement >= self.config.disagreement_medium_threshold:
            return DisagreementLevel.HIGH
        elif disagreement >= self.config.disagreement_low_threshold:
            return DisagreementLevel.MEDIUM
        elif disagreement > 0:
            return DisagreementLevel.LOW
        else:
            return DisagreementLevel.NONE

    def _find_disagreeing_pairs(
        self,
        results: List[AlgorithmResult],
        mask: np.ndarray,
    ) -> List[Tuple[str, str]]:
        """Find pairs of algorithms that disagree in a region."""
        pairs = []

        for i, r1 in enumerate(results):
            for r2 in results[i + 1:]:
                # Compare binary outputs in masked region
                diff = np.abs(r1.data[mask] - r2.data[mask])
                if np.mean(diff) > 0.5:  # More than 50% different
                    pairs.append((r1.algorithm_id, r2.algorithm_id))

        return pairs

    def _check_review_requirement(
        self,
        confidence: np.ndarray,
        disagreement_regions: List[DisagreementRegion],
    ) -> Tuple[bool, Optional[str]]:
        """Check if result requires manual review."""
        reasons = []

        # Check overall confidence
        mean_confidence = np.mean(confidence)
        if mean_confidence < self.config.min_confidence_threshold:
            reasons.append(f"Low overall confidence ({mean_confidence:.2f})")

        # Check high disagreement regions
        high_disagree = [
            r for r in disagreement_regions
            if r.level in (DisagreementLevel.HIGH, DisagreementLevel.CRITICAL)
        ]

        if high_disagree and self.config.require_review_on_high_disagreement:
            reasons.append(f"{len(high_disagree)} high disagreement regions")

        if reasons:
            return True, "; ".join(reasons)
        else:
            return False, None


def ensemble_voting(
    results: List[AlgorithmResult],
    method: Optional[VotingMethod] = None,
    config: Optional[LowConfidenceConfig] = None,
) -> EnsembleResult:
    """
    Convenience function for ensemble voting.

    Args:
        results: List of algorithm results
        method: Voting method
        config: Configuration options

    Returns:
        EnsembleResult with combined output

    Example:
        result = ensemble_voting([
            AlgorithmResult(...),
            AlgorithmResult(...),
        ])
        print(f"Confidence: {result.overall_confidence:.2f}")
    """
    handler = LowConfidenceHandler(config)
    return handler.ensemble_voting(results, method)
