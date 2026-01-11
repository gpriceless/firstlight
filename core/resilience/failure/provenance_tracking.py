"""
Fallback Provenance Tracking Module.

Provides comprehensive audit trail for all fallback decisions:
- Record all fallback decision points
- Track confidence scoring based on fallback depth
- Generate user-facing explanations of limitations
- Ensure reproducibility through decision documentation
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class FallbackReason(Enum):
    """Reasons for triggering a fallback."""
    DATA_UNAVAILABLE = "data_unavailable"
    DATA_LOW_QUALITY = "data_low_quality"
    CLOUD_COVER = "cloud_cover"
    SENSOR_FAILURE = "sensor_failure"
    ALGORITHM_FAILURE = "algorithm_failure"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    VALIDATION_FAILURE = "validation_failure"
    MISSING_BASELINE = "missing_baseline"
    NETWORK_ERROR = "network_error"
    USER_PREFERENCE = "user_preference"


class FallbackType(Enum):
    """Types of fallback mechanisms."""
    SENSOR_ALTERNATIVE = "sensor_alternative"
    ALGORITHM_ALTERNATIVE = "algorithm_alternative"
    RESOLUTION_DEGRADATION = "resolution_degradation"
    TEMPORAL_DEGRADATION = "temporal_degradation"
    SPATIAL_DEGRADATION = "spatial_degradation"
    CACHE_FALLBACK = "cache_fallback"
    DEFAULT_VALUE = "default_value"
    INTERPOLATION = "interpolation"
    EXTRAPOLATION = "extrapolation"


@dataclass
class FallbackDecision:
    """
    Record of a single fallback decision.

    Attributes:
        decision_id: Unique identifier for this decision
        timestamp: When the decision was made
        original_strategy: What was originally planned
        fallback_strategy: What was used instead
        fallback_type: Type of fallback
        reason: Why the fallback was needed
        confidence_before: Confidence before fallback
        confidence_after: Confidence after fallback
        context: Additional context about the decision
        user_message: User-facing explanation
        reversible: Whether this decision can be undone
    """
    decision_id: str
    timestamp: datetime
    original_strategy: str
    fallback_strategy: str
    fallback_type: FallbackType
    reason: FallbackReason
    confidence_before: float
    confidence_after: float
    context: Dict[str, Any] = field(default_factory=dict)
    user_message: str = ""
    reversible: bool = False

    @property
    def confidence_impact(self) -> float:
        """Calculate confidence impact (negative = degradation)."""
        return self.confidence_after - self.confidence_before

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "original_strategy": self.original_strategy,
            "fallback_strategy": self.fallback_strategy,
            "fallback_type": self.fallback_type.value,
            "reason": self.reason.value,
            "confidence_before": round(self.confidence_before, 3),
            "confidence_after": round(self.confidence_after, 3),
            "confidence_impact": round(self.confidence_impact, 3),
            "context": self.context,
            "user_message": self.user_message,
            "reversible": self.reversible,
        }


@dataclass
class FallbackChain:
    """
    Chain of fallback decisions for a single operation.

    Attributes:
        chain_id: Unique identifier for this chain
        operation: What operation this chain is for
        decisions: List of fallback decisions in order
        initial_confidence: Starting confidence
        final_confidence: Ending confidence
        start_time: When the chain started
        end_time: When the chain completed
        success: Whether operation ultimately succeeded
    """
    chain_id: str
    operation: str
    decisions: List[FallbackDecision] = field(default_factory=list)
    initial_confidence: float = 1.0
    final_confidence: float = 1.0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = None
    success: bool = True

    @property
    def fallback_depth(self) -> int:
        """Get number of fallbacks in this chain."""
        return len(self.decisions)

    @property
    def total_confidence_loss(self) -> float:
        """Calculate total confidence lost through fallbacks."""
        return self.initial_confidence - self.final_confidence

    @property
    def fallback_types_used(self) -> List[FallbackType]:
        """Get list of fallback types used."""
        return [d.fallback_type for d in self.decisions]

    @property
    def reasons_summary(self) -> Dict[str, int]:
        """Summarize reasons for fallbacks."""
        summary: Dict[str, int] = {}
        for decision in self.decisions:
            key = decision.reason.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    def add_decision(self, decision: FallbackDecision) -> None:
        """Add a decision to the chain."""
        self.decisions.append(decision)
        self.final_confidence = decision.confidence_after

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chain_id": self.chain_id,
            "operation": self.operation,
            "fallback_depth": self.fallback_depth,
            "initial_confidence": round(self.initial_confidence, 3),
            "final_confidence": round(self.final_confidence, 3),
            "total_confidence_loss": round(self.total_confidence_loss, 3),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "success": self.success,
            "decisions": [d.to_dict() for d in self.decisions],
            "reasons_summary": self.reasons_summary,
        }


@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for an analysis run.

    Attributes:
        record_id: Unique identifier
        event_id: Event being analyzed
        timestamp: When analysis was performed
        fallback_chains: All fallback chains used
        overall_confidence: Final confidence score
        data_sources_used: Data sources actually used
        data_sources_attempted: Data sources attempted but failed
        algorithms_used: Algorithms actually used
        algorithms_attempted: Algorithms attempted but failed
        environment: Environment information
        reproducibility_hash: Hash for reproducibility checking
        user_explanation: User-facing summary
    """
    record_id: str
    event_id: str
    timestamp: datetime
    fallback_chains: List[FallbackChain] = field(default_factory=list)
    overall_confidence: float = 1.0
    data_sources_used: List[str] = field(default_factory=list)
    data_sources_attempted: List[str] = field(default_factory=list)
    algorithms_used: List[str] = field(default_factory=list)
    algorithms_attempted: List[str] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""
    user_explanation: str = ""

    @property
    def total_fallbacks(self) -> int:
        """Get total number of fallbacks across all chains."""
        return sum(chain.fallback_depth for chain in self.fallback_chains)

    @property
    def data_source_success_rate(self) -> float:
        """Calculate data source success rate."""
        attempted = len(self.data_sources_attempted)
        if attempted == 0:
            return 1.0
        return len(self.data_sources_used) / attempted

    @property
    def algorithm_success_rate(self) -> float:
        """Calculate algorithm success rate."""
        attempted = len(self.algorithms_attempted)
        if attempted == 0:
            return 1.0
        return len(self.algorithms_used) / attempted

    def compute_reproducibility_hash(self) -> str:
        """Compute hash for reproducibility checking."""
        hash_input = {
            "event_id": self.event_id,
            "data_sources_used": sorted(self.data_sources_used),
            "algorithms_used": sorted(self.algorithms_used),
            "fallback_decisions": [
                {
                    "original": d.original_strategy,
                    "fallback": d.fallback_strategy,
                    "reason": d.reason.value,
                }
                for chain in self.fallback_chains
                for d in chain.decisions
            ],
        }
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "overall_confidence": round(self.overall_confidence, 3),
            "total_fallbacks": self.total_fallbacks,
            "data_sources_used": self.data_sources_used,
            "data_sources_attempted": self.data_sources_attempted,
            "data_source_success_rate": round(self.data_source_success_rate, 3),
            "algorithms_used": self.algorithms_used,
            "algorithms_attempted": self.algorithms_attempted,
            "algorithm_success_rate": round(self.algorithm_success_rate, 3),
            "fallback_chains": [c.to_dict() for c in self.fallback_chains],
            "environment": self.environment,
            "reproducibility_hash": self.reproducibility_hash,
            "user_explanation": self.user_explanation,
        }


@dataclass
class FallbackProvenanceConfig:
    """
    Configuration for provenance tracking.

    Attributes:
        confidence_decay_per_fallback: Confidence reduction per fallback
        max_fallback_depth: Maximum allowed fallback depth
        require_user_explanation: Always generate user explanation
        track_environment: Include environment in provenance
        enable_reproducibility_hash: Compute reproducibility hash
    """
    confidence_decay_per_fallback: float = 0.1
    max_fallback_depth: int = 5
    require_user_explanation: bool = True
    track_environment: bool = True
    enable_reproducibility_hash: bool = True


class FallbackProvenanceTracker:
    """
    Tracks provenance for all fallback decisions.

    Maintains complete audit trail of what was tried, what failed,
    and what workarounds were used.

    Example:
        tracker = FallbackProvenanceTracker(event_id="flood_miami_2024")

        # Start tracking a chain of fallbacks
        chain_id = tracker.start_chain("data_acquisition")

        # Record fallback decisions
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sentinel2_optical",
            fallback_strategy="landsat8_optical",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
            confidence_before=0.9,
            context={"cloud_cover": 85},
        )

        # Complete the chain
        tracker.complete_chain(chain_id, success=True)

        # Generate provenance record
        record = tracker.generate_provenance_record()
    """

    def __init__(
        self,
        event_id: str,
        config: Optional[FallbackProvenanceConfig] = None,
    ):
        """
        Initialize the provenance tracker.

        Args:
            event_id: Event being analyzed
            config: Configuration options
        """
        self.event_id = event_id
        self.config = config or FallbackProvenanceConfig()
        self._chains: Dict[str, FallbackChain] = {}
        self._data_sources_used: List[str] = []
        self._data_sources_attempted: List[str] = []
        self._algorithms_used: List[str] = []
        self._algorithms_attempted: List[str] = []
        self._current_confidence = 1.0
        self._start_time = datetime.now(timezone.utc)

    def start_chain(
        self,
        operation: str,
        initial_confidence: float = 1.0,
    ) -> str:
        """
        Start a new fallback chain.

        Args:
            operation: What operation this chain is for
            initial_confidence: Starting confidence

        Returns:
            Chain ID for use in subsequent calls
        """
        chain_id = str(uuid.uuid4())
        chain = FallbackChain(
            chain_id=chain_id,
            operation=operation,
            initial_confidence=initial_confidence,
            final_confidence=initial_confidence,
        )
        self._chains[chain_id] = chain
        return chain_id

    def record_fallback(
        self,
        chain_id: str,
        original_strategy: str,
        fallback_strategy: str,
        fallback_type: FallbackType,
        reason: FallbackReason,
        confidence_before: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> FallbackDecision:
        """
        Record a fallback decision.

        Args:
            chain_id: Chain to add decision to
            original_strategy: What was originally planned
            fallback_strategy: What was used instead
            fallback_type: Type of fallback
            reason: Why fallback was needed
            confidence_before: Confidence before fallback
            context: Additional context

        Returns:
            FallbackDecision record
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            chain_id = self.start_chain("unknown")
            chain = self._chains[chain_id]

        if confidence_before is None:
            confidence_before = chain.final_confidence

        # Calculate new confidence
        confidence_after = max(
            0.0,
            confidence_before - self.config.confidence_decay_per_fallback
        )

        # Generate user message
        user_message = self._generate_user_message(
            original_strategy, fallback_strategy, reason, context
        )

        decision = FallbackDecision(
            decision_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            original_strategy=original_strategy,
            fallback_strategy=fallback_strategy,
            fallback_type=fallback_type,
            reason=reason,
            confidence_before=confidence_before,
            confidence_after=confidence_after,
            context=context or {},
            user_message=user_message,
        )

        chain.add_decision(decision)
        self._current_confidence = min(self._current_confidence, confidence_after)

        logger.info(
            f"Fallback recorded: {original_strategy} -> {fallback_strategy} "
            f"(reason={reason.value}, confidence={confidence_after:.2f})"
        )

        return decision

    def complete_chain(
        self,
        chain_id: str,
        success: bool = True,
    ) -> Optional[FallbackChain]:
        """
        Mark a fallback chain as complete.

        Args:
            chain_id: Chain to complete
            success: Whether operation succeeded

        Returns:
            Completed FallbackChain
        """
        chain = self._chains.get(chain_id)
        if chain is None:
            return None

        chain.end_time = datetime.now(timezone.utc)
        chain.success = success
        return chain

    def record_data_source(
        self,
        source: str,
        success: bool,
    ) -> None:
        """
        Record a data source attempt.

        Args:
            source: Data source name
            success: Whether it was successfully used
        """
        if source not in self._data_sources_attempted:
            self._data_sources_attempted.append(source)
        if success and source not in self._data_sources_used:
            self._data_sources_used.append(source)

    def record_algorithm(
        self,
        algorithm: str,
        success: bool,
    ) -> None:
        """
        Record an algorithm attempt.

        Args:
            algorithm: Algorithm name
            success: Whether it was successfully used
        """
        if algorithm not in self._algorithms_attempted:
            self._algorithms_attempted.append(algorithm)
        if success and algorithm not in self._algorithms_used:
            self._algorithms_used.append(algorithm)

    def generate_provenance_record(self) -> ProvenanceRecord:
        """
        Generate complete provenance record.

        Returns:
            ProvenanceRecord with full audit trail
        """
        # Generate user explanation
        user_explanation = self._generate_user_explanation()

        # Capture environment
        environment = self._capture_environment() if self.config.track_environment else {}

        record = ProvenanceRecord(
            record_id=str(uuid.uuid4()),
            event_id=self.event_id,
            timestamp=datetime.now(timezone.utc),
            fallback_chains=list(self._chains.values()),
            overall_confidence=self._current_confidence,
            data_sources_used=self._data_sources_used.copy(),
            data_sources_attempted=self._data_sources_attempted.copy(),
            algorithms_used=self._algorithms_used.copy(),
            algorithms_attempted=self._algorithms_attempted.copy(),
            environment=environment,
            user_explanation=user_explanation,
        )

        # Compute reproducibility hash
        if self.config.enable_reproducibility_hash:
            record.reproducibility_hash = record.compute_reproducibility_hash()

        return record

    def get_confidence_breakdown(self) -> Dict[str, Any]:
        """
        Get breakdown of confidence impacts.

        Returns:
            Dictionary with confidence analysis
        """
        impacts: List[Dict[str, Any]] = []
        for chain in self._chains.values():
            for decision in chain.decisions:
                impacts.append({
                    "operation": chain.operation,
                    "fallback": decision.fallback_strategy,
                    "reason": decision.reason.value,
                    "impact": decision.confidence_impact,
                })

        return {
            "initial_confidence": 1.0,
            "final_confidence": self._current_confidence,
            "total_loss": 1.0 - self._current_confidence,
            "impacts": sorted(impacts, key=lambda x: x["impact"]),
        }

    def _generate_user_message(
        self,
        original: str,
        fallback: str,
        reason: FallbackReason,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Generate user-facing message for a fallback."""
        reason_messages = {
            FallbackReason.DATA_UNAVAILABLE: "data was not available",
            FallbackReason.DATA_LOW_QUALITY: "data quality was insufficient",
            FallbackReason.CLOUD_COVER: f"cloud cover was too high ({context.get('cloud_cover', '?')}%)" if context else "cloud cover was too high",
            FallbackReason.SENSOR_FAILURE: "sensor data could not be retrieved",
            FallbackReason.ALGORITHM_FAILURE: "algorithm execution failed",
            FallbackReason.TIMEOUT: "operation timed out",
            FallbackReason.RESOURCE_LIMIT: "resource limits were exceeded",
            FallbackReason.VALIDATION_FAILURE: "validation checks failed",
            FallbackReason.MISSING_BASELINE: "pre-event baseline was not available",
            FallbackReason.NETWORK_ERROR: "network error occurred",
        }

        reason_text = reason_messages.get(reason, f"{reason.value}")
        return f"Switched from {original} to {fallback} because {reason_text}."

    def _generate_user_explanation(self) -> str:
        """Generate overall user explanation."""
        parts = []

        # Overall confidence
        if self._current_confidence < 0.5:
            parts.append(
                f"NOTICE: Results have low confidence ({self._current_confidence:.0%}) "
                "due to data limitations."
            )
        elif self._current_confidence < 0.8:
            parts.append(
                f"Results have moderate confidence ({self._current_confidence:.0%}) "
                "due to some data limitations."
            )

        # Summarize fallbacks
        total_fallbacks = sum(c.fallback_depth for c in self._chains.values())
        if total_fallbacks > 0:
            parts.append(f"{total_fallbacks} fallback(s) were used during processing:")

            # Group by reason
            reason_counts: Dict[str, int] = {}
            for chain in self._chains.values():
                for summary in chain.reasons_summary.items():
                    reason, count = summary
                    reason_counts[reason] = reason_counts.get(reason, 0) + count

            for reason, count in reason_counts.items():
                parts.append(f"  - {reason}: {count} occurrence(s)")

        # Data source summary
        failed_sources = [
            s for s in self._data_sources_attempted
            if s not in self._data_sources_used
        ]
        if failed_sources:
            parts.append(
                f"Some data sources were unavailable: {', '.join(failed_sources)}"
            )

        return "\n".join(parts) if parts else "Analysis completed without fallbacks."

    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information."""
        import platform
        import sys

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def create_provenance_record(
    event_id: str,
    fallback_decisions: List[Dict[str, Any]],
    data_sources: List[str],
    algorithms: List[str],
) -> ProvenanceRecord:
    """
    Convenience function to create a provenance record.

    Args:
        event_id: Event being analyzed
        fallback_decisions: List of fallback decision dicts
        data_sources: List of data sources used
        algorithms: List of algorithms used

    Returns:
        ProvenanceRecord

    Example:
        record = create_provenance_record(
            event_id="flood_001",
            fallback_decisions=[
                {"original": "sentinel2", "fallback": "landsat8", "reason": "cloud_cover"},
            ],
            data_sources=["landsat8", "sentinel1"],
            algorithms=["sar_threshold"],
        )
    """
    tracker = FallbackProvenanceTracker(event_id)

    chain_id = tracker.start_chain("analysis")
    for decision in fallback_decisions:
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy=decision.get("original", "unknown"),
            fallback_strategy=decision.get("fallback", "unknown"),
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason(decision.get("reason", "data_unavailable")),
        )
    tracker.complete_chain(chain_id)

    for source in data_sources:
        tracker.record_data_source(source, success=True)

    for algo in algorithms:
        tracker.record_algorithm(algo, success=True)

    return tracker.generate_provenance_record()
