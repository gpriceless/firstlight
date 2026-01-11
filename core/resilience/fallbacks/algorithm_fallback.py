"""
Algorithm Fallback Manager.

Implements intelligent fallback strategies when algorithms fail or
produce poor quality results:

- No pre-event baseline: Use static water mask + anomaly detection
- Insufficient SAR: Switch to optical-only with cloud masking
- No optical or SAR: Use forecast + terrain analysis
- Algorithm failure: Try alternative algorithm
- Poor results: Adjust parameters

All decisions are logged for reproducibility and analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FallbackTrigger(Enum):
    """Triggers that activate algorithm fallback."""
    NO_BASELINE = "no_baseline"             # No pre-event baseline available
    INSUFFICIENT_SAR = "insufficient_sar"   # SAR quality too low
    NO_OPTICAL = "no_optical"               # No optical data available
    NO_SAR = "no_sar"                       # No SAR data available
    NO_DATA = "no_data"                     # No imagery at all
    ALGORITHM_FAILURE = "algorithm_failure" # Algorithm crashed
    POOR_QUALITY = "poor_quality"           # Results quality below threshold
    TIMEOUT = "timeout"                     # Algorithm timed out
    INVALID_OUTPUT = "invalid_output"       # Output validation failed


class AlternativeAlgorithm(Enum):
    """Alternative algorithms for fallback."""
    # Flood detection
    STATIC_WATER_MASK = "static_water_mask"
    ANOMALY_DETECTION = "anomaly_detection"
    TERRAIN_ANALYSIS = "terrain_analysis"
    FORECAST_INUNDATION = "forecast_inundation"
    THRESHOLD_SIMPLE = "threshold_simple"

    # Generic alternatives
    MANUAL_THRESHOLD = "manual_threshold"
    ENSEMBLE_VOTING = "ensemble_voting"
    SIMPLIFIED_MODEL = "simplified_model"
    REFERENCE_BASED = "reference_based"


class DataRequirement(Enum):
    """Data requirements for algorithms."""
    OPTICAL_PRE = "optical_pre"
    OPTICAL_POST = "optical_post"
    SAR_PRE = "sar_pre"
    SAR_POST = "sar_post"
    DEM = "dem"
    BASELINE = "baseline"
    FORECAST = "forecast"


@dataclass
class AlgorithmFallbackDecision:
    """
    Record of an algorithm fallback decision.

    Attributes:
        timestamp: When decision was made
        trigger: What triggered the fallback
        from_algorithm: Algorithm being abandoned
        to_algorithm: Algorithm being adopted
        reason: Detailed reason
        quality_score: Quality score that triggered fallback
        details: Additional details
    """
    timestamp: datetime
    trigger: FallbackTrigger
    from_algorithm: str
    to_algorithm: str
    reason: str
    quality_score: Optional[float] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger.value,
            "from_algorithm": self.from_algorithm,
            "to_algorithm": self.to_algorithm,
            "reason": self.reason,
            "quality_score": self.quality_score,
            "details": self.details,
        }


@dataclass
class AlgorithmFallbackConfig:
    """
    Configuration for algorithm fallback manager.

    Attributes:
        min_quality_score: Minimum quality score to accept results
        max_retries: Maximum number of fallback attempts
        timeout_seconds: Algorithm timeout in seconds

        enable_static_fallback: Allow static reference fallbacks
        enable_terrain_fallback: Allow terrain-based fallbacks
        enable_forecast_fallback: Allow forecast-based fallbacks
        enable_parameter_adjustment: Allow parameter adjustment retries
    """
    min_quality_score: float = 0.4
    max_retries: int = 3
    timeout_seconds: float = 300.0

    enable_static_fallback: bool = True
    enable_terrain_fallback: bool = True
    enable_forecast_fallback: bool = True
    enable_parameter_adjustment: bool = True

    # Quality thresholds for specific checks
    min_spatial_coherence: float = 0.3
    min_coverage_percent: float = 50.0
    max_anomaly_percent: float = 30.0


@dataclass
class AlgorithmCapabilities:
    """Capabilities and requirements of an algorithm."""
    name: str
    requires: List[DataRequirement]
    produces: List[str]
    quality_typical: float
    processing_time_typical_s: float
    is_fallback: bool = False

    def can_run_with(self, available_data: List[DataRequirement]) -> bool:
        """Check if algorithm can run with available data."""
        return all(req in available_data for req in self.requires)


@dataclass
class AlgorithmFallbackResult:
    """
    Result of algorithm fallback evaluation.

    Attributes:
        success: Whether a usable algorithm was found
        selected_algorithm: Algorithm to use
        is_fallback: Whether this is a fallback algorithm
        expected_quality: Expected quality of results
        confidence_degradation: How much confidence is reduced
        decisions: List of fallback decisions made
        fallback_chain: Sequence of algorithms tried
        recommendation: Action recommendation
        warnings: List of warnings about result quality
        metrics: Additional metrics
    """
    success: bool
    selected_algorithm: str
    is_fallback: bool
    expected_quality: float
    confidence_degradation: float
    decisions: List[AlgorithmFallbackDecision] = field(default_factory=list)
    fallback_chain: List[str] = field(default_factory=list)
    recommendation: str = ""
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "selected_algorithm": self.selected_algorithm,
            "is_fallback": self.is_fallback,
            "expected_quality": round(self.expected_quality, 2),
            "confidence_degradation": round(self.confidence_degradation, 2),
            "num_decisions": len(self.decisions),
            "decisions": [d.to_dict() for d in self.decisions],
            "fallback_chain": self.fallback_chain,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


class AlgorithmFallbackManager:
    """
    Manages algorithm fallback strategies.

    Determines which algorithm to use based on available data,
    handles failures gracefully, and logs all decisions.

    Example:
        manager = AlgorithmFallbackManager()
        result = manager.select_algorithm(
            event_type="flood",
            available_data=[DataRequirement.SAR_POST, DataRequirement.DEM],
            preferred_algorithm="sar_change_detection"
        )
        if result.is_fallback:
            print(f"Using fallback: {result.selected_algorithm}")
    """

    # Algorithm specifications
    ALGORITHMS = {
        # Flood detection
        "sar_change_detection": AlgorithmCapabilities(
            name="sar_change_detection",
            requires=[DataRequirement.SAR_PRE, DataRequirement.SAR_POST],
            produces=["flood_extent", "change_magnitude"],
            quality_typical=0.85,
            processing_time_typical_s=60.0
        ),
        "optical_ndwi": AlgorithmCapabilities(
            name="optical_ndwi",
            requires=[DataRequirement.OPTICAL_POST],
            produces=["water_extent"],
            quality_typical=0.75,
            processing_time_typical_s=30.0
        ),
        "optical_change": AlgorithmCapabilities(
            name="optical_change",
            requires=[DataRequirement.OPTICAL_PRE, DataRequirement.OPTICAL_POST],
            produces=["flood_extent", "change_magnitude"],
            quality_typical=0.80,
            processing_time_typical_s=45.0
        ),
        "hand_model": AlgorithmCapabilities(
            name="hand_model",
            requires=[DataRequirement.DEM],
            produces=["flood_susceptibility", "inundation_depth"],
            quality_typical=0.70,
            processing_time_typical_s=90.0,
            is_fallback=True
        ),
        "static_water_mask": AlgorithmCapabilities(
            name="static_water_mask",
            requires=[],
            produces=["permanent_water"],
            quality_typical=0.50,
            processing_time_typical_s=5.0,
            is_fallback=True
        ),
        "forecast_inundation": AlgorithmCapabilities(
            name="forecast_inundation",
            requires=[DataRequirement.FORECAST, DataRequirement.DEM],
            produces=["predicted_flood_extent"],
            quality_typical=0.55,
            processing_time_typical_s=30.0,
            is_fallback=True
        ),
        # Wildfire detection
        "dnbr_analysis": AlgorithmCapabilities(
            name="dnbr_analysis",
            requires=[DataRequirement.OPTICAL_PRE, DataRequirement.OPTICAL_POST],
            produces=["burn_severity"],
            quality_typical=0.85,
            processing_time_typical_s=40.0
        ),
        "active_fire_thermal": AlgorithmCapabilities(
            name="active_fire_thermal",
            requires=[DataRequirement.OPTICAL_POST],
            produces=["active_fire_locations"],
            quality_typical=0.80,
            processing_time_typical_s=20.0
        ),
    }

    # Fallback chains by event type
    FALLBACK_CHAINS = {
        "flood": [
            "sar_change_detection",
            "optical_change",
            "optical_ndwi",
            "hand_model",
            "forecast_inundation",
            "static_water_mask",
        ],
        "wildfire": [
            "dnbr_analysis",
            "active_fire_thermal",
        ],
        "storm": [
            "sar_change_detection",
            "optical_change",
        ],
    }

    def __init__(self, config: Optional[AlgorithmFallbackConfig] = None):
        """
        Initialize the algorithm fallback manager.

        Args:
            config: Configuration options
        """
        self.config = config or AlgorithmFallbackConfig()
        self.decision_log: List[AlgorithmFallbackDecision] = []

    def select_algorithm(
        self,
        event_type: str,
        available_data: List[DataRequirement],
        preferred_algorithm: Optional[str] = None,
        previous_failures: Optional[List[str]] = None,
    ) -> AlgorithmFallbackResult:
        """
        Select the best algorithm for the given situation.

        Args:
            event_type: Type of event (flood, wildfire, storm)
            available_data: List of available data requirements
            preferred_algorithm: Preferred algorithm to try first
            previous_failures: Algorithms that have already failed

        Returns:
            AlgorithmFallbackResult with selected algorithm
        """
        self.decision_log = []
        previous_failures = previous_failures or []
        fallback_chain_used = []

        # Get fallback chain for event type
        chain = self.FALLBACK_CHAINS.get(event_type, list(self.ALGORITHMS.keys()))

        # Put preferred algorithm first if specified
        if preferred_algorithm and preferred_algorithm in chain:
            chain = [preferred_algorithm] + [a for a in chain if a != preferred_algorithm]

        # Try each algorithm in the chain
        selected = None
        is_fallback = False
        warnings = []

        for i, algo_name in enumerate(chain):
            if algo_name in previous_failures:
                continue

            algo = self.ALGORITHMS.get(algo_name)
            if not algo:
                continue

            fallback_chain_used.append(algo_name)

            # Check if algorithm can run
            if not algo.can_run_with(available_data):
                missing = [r.value for r in algo.requires if r not in available_data]
                self._log_decision(
                    FallbackTrigger.NO_DATA,
                    algo_name,
                    chain[i + 1] if i + 1 < len(chain) else "none",
                    f"Missing data: {', '.join(missing)}",
                    details=f"Algorithm requires {[r.value for r in algo.requires]}"
                )
                continue

            # Algorithm can run
            selected = algo_name
            is_fallback = algo.is_fallback or i > 0

            if is_fallback:
                warnings.append(
                    f"Using fallback algorithm '{algo_name}' - results may be less accurate"
                )

            break

        if selected is None:
            return AlgorithmFallbackResult(
                success=False,
                selected_algorithm="none",
                is_fallback=True,
                expected_quality=0.0,
                confidence_degradation=1.0,
                decisions=list(self.decision_log),
                fallback_chain=fallback_chain_used,
                recommendation="No suitable algorithm available for current data",
                warnings=["Cannot process event with available data"],
            )

        algo = self.ALGORITHMS[selected]
        expected_quality = algo.quality_typical
        confidence_degradation = 1.0 - expected_quality if is_fallback else 0.0

        # Generate recommendation
        recommendation = self._generate_recommendation(
            selected, is_fallback, available_data, event_type
        )

        return AlgorithmFallbackResult(
            success=True,
            selected_algorithm=selected,
            is_fallback=is_fallback,
            expected_quality=expected_quality,
            confidence_degradation=confidence_degradation,
            decisions=list(self.decision_log),
            fallback_chain=fallback_chain_used,
            recommendation=recommendation,
            warnings=warnings,
            metrics={
                "algorithms_tried": len(fallback_chain_used),
                "data_available": [d.value for d in available_data],
                "processing_time_estimate_s": algo.processing_time_typical_s,
            },
        )

    def handle_failure(
        self,
        failed_algorithm: str,
        event_type: str,
        failure_reason: FallbackTrigger,
        available_data: List[DataRequirement],
        quality_score: Optional[float] = None,
    ) -> AlgorithmFallbackResult:
        """
        Handle algorithm failure and select alternative.

        Args:
            failed_algorithm: Algorithm that failed
            event_type: Type of event
            failure_reason: Reason for failure
            available_data: Available data
            quality_score: Quality score if applicable

        Returns:
            AlgorithmFallbackResult with alternative
        """
        self._log_decision(
            failure_reason,
            failed_algorithm,
            "evaluating",
            f"Algorithm {failed_algorithm} failed",
            quality_score=quality_score,
            details=f"Reason: {failure_reason.value}"
        )

        # Get fallback excluding failed algorithm
        return self.select_algorithm(
            event_type=event_type,
            available_data=available_data,
            previous_failures=[failed_algorithm],
        )

    def get_fallback_for_missing_baseline(
        self,
        event_type: str,
        available_data: List[DataRequirement],
    ) -> AlgorithmFallbackResult:
        """
        Get fallback when no pre-event baseline is available.

        Args:
            event_type: Type of event
            available_data: Available data

        Returns:
            AlgorithmFallbackResult for baseline-less processing
        """
        self._log_decision(
            FallbackTrigger.NO_BASELINE,
            "change_detection",
            "single_date",
            "No pre-event baseline available",
            details="Switching to single-date or static reference methods"
        )

        # Remove baseline requirements from available data understanding
        # and try to find single-date algorithms
        if event_type == "flood":
            # For flood without baseline:
            # 1. Try single-date water detection
            # 2. Use static water mask + anomaly
            # 3. Use terrain analysis

            if DataRequirement.SAR_POST in available_data:
                return AlgorithmFallbackResult(
                    success=True,
                    selected_algorithm="threshold_water_sar",
                    is_fallback=True,
                    expected_quality=0.60,
                    confidence_degradation=0.25,
                    decisions=list(self.decision_log),
                    fallback_chain=["threshold_water_sar"],
                    recommendation="Using SAR thresholding without baseline; compare to static water mask",
                    warnings=["No baseline - cannot compute change, only current state"],
                )

            if DataRequirement.OPTICAL_POST in available_data:
                return AlgorithmFallbackResult(
                    success=True,
                    selected_algorithm="optical_ndwi",
                    is_fallback=True,
                    expected_quality=0.65,
                    confidence_degradation=0.20,
                    decisions=list(self.decision_log),
                    fallback_chain=["optical_ndwi"],
                    recommendation="Using NDWI water detection; compare to static reference for change",
                    warnings=["No baseline - comparing to static water mask"],
                )

            if DataRequirement.DEM in available_data:
                return AlgorithmFallbackResult(
                    success=True,
                    selected_algorithm="hand_model",
                    is_fallback=True,
                    expected_quality=0.50,
                    confidence_degradation=0.35,
                    decisions=list(self.decision_log),
                    fallback_chain=["hand_model"],
                    recommendation="Using HAND model for flood susceptibility",
                    warnings=["No imagery - using terrain-based flood modeling only"],
                )

        # Generic fallback
        return AlgorithmFallbackResult(
            success=False,
            selected_algorithm="none",
            is_fallback=True,
            expected_quality=0.0,
            confidence_degradation=1.0,
            decisions=list(self.decision_log),
            fallback_chain=[],
            recommendation="Cannot process without baseline or suitable fallback",
            warnings=["Insufficient data for any algorithm"],
        )

    def _log_decision(
        self,
        trigger: FallbackTrigger,
        from_algo: str,
        to_algo: str,
        reason: str,
        quality_score: Optional[float] = None,
        details: str = "",
    ):
        """Log a fallback decision."""
        decision = AlgorithmFallbackDecision(
            timestamp=datetime.now(),
            trigger=trigger,
            from_algorithm=from_algo,
            to_algorithm=to_algo,
            reason=reason,
            quality_score=quality_score,
            details=details,
        )
        self.decision_log.append(decision)
        logger.debug(f"Algorithm fallback: {trigger.value} - {from_algo} -> {to_algo}")

    def _generate_recommendation(
        self,
        algorithm: str,
        is_fallback: bool,
        available_data: List[DataRequirement],
        event_type: str,
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        if is_fallback:
            parts.append(f"Using fallback algorithm '{algorithm}'")
            parts.append("Results should be validated carefully")
        else:
            parts.append(f"Using primary algorithm '{algorithm}'")

        algo = self.ALGORITHMS.get(algorithm)
        if algo:
            parts.append(f"Expected quality: {algo.quality_typical:.0%}")

        return "; ".join(parts)

    def validate_result(
        self,
        algorithm: str,
        result_data: np.ndarray,
        metadata: Dict[str, Any],
    ) -> Tuple[bool, List[str], float]:
        """
        Validate algorithm result quality.

        Args:
            algorithm: Algorithm that produced result
            result_data: Output data array
            metadata: Result metadata

        Returns:
            Tuple of (is_valid, issues, quality_score)
        """
        issues = []
        quality_score = 1.0

        # Check for empty result
        if result_data.size == 0:
            return False, ["Empty result"], 0.0

        # Check coverage
        valid_pixels = ~np.isnan(result_data)
        coverage = 100.0 * valid_pixels.sum() / result_data.size

        if coverage < self.config.min_coverage_percent:
            issues.append(f"Low coverage: {coverage:.1f}%")
            quality_score *= coverage / self.config.min_coverage_percent

        # Check for anomalous values
        if np.any(np.isinf(result_data)):
            issues.append("Contains infinite values")
            quality_score *= 0.5

        # Check value range (assuming 0-1 for many outputs)
        valid_data = result_data[valid_pixels]
        if len(valid_data) > 0:
            if valid_data.min() < -1 or valid_data.max() > 2:
                issues.append(f"Suspicious value range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
                quality_score *= 0.8

        # Check spatial coherence (simple connectivity check)
        binary = valid_pixels & (result_data > 0.5)
        isolated = self._count_isolated_pixels(binary)
        isolation_percent = 100.0 * isolated / max(1, binary.sum())

        if isolation_percent > self.config.max_anomaly_percent:
            issues.append(f"High noise/fragmentation: {isolation_percent:.1f}% isolated pixels")
            quality_score *= 0.7

        is_valid = quality_score >= self.config.min_quality_score and len(issues) < 3

        return is_valid, issues, float(quality_score)

    def _count_isolated_pixels(self, binary: np.ndarray) -> int:
        """Count isolated (single) pixels in binary array."""
        if binary.sum() == 0:
            return 0

        try:
            from scipy import ndimage
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            neighbors = ndimage.convolve(binary.astype(float), kernel, mode='constant')
            isolated = binary & (neighbors == 0)
            return int(isolated.sum())
        except ImportError:
            return 0  # Skip check if scipy unavailable
