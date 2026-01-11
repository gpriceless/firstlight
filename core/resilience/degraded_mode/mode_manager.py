"""
Degraded Mode Manager.

Manages operational mode transitions based on data availability and quality,
providing automatic mode switching and user notifications.

The system operates in four modes:
- FULL: Ideal conditions with high confidence
- PARTIAL: Some limitations but acceptable quality
- MINIMAL: Significant constraints with low confidence
- EMERGENCY: Bare minimum capabilities, manual review required
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import logging
import uuid

import numpy as np

logger = logging.getLogger(__name__)


class DegradedModeLevel(Enum):
    """
    Operational mode levels based on data availability and quality.

    Each level represents a different confidence tier for analysis outputs.
    """
    FULL = "full"          # All data available, high confidence (>0.8)
    PARTIAL = "partial"    # Some data missing, medium confidence (0.5-0.8)
    MINIMAL = "minimal"    # Significant gaps, low confidence (0.3-0.5)
    EMERGENCY = "emergency"  # Bare minimum, very low confidence (<0.3)

    @property
    def confidence_range(self) -> tuple[float, float]:
        """Get confidence range for this mode level."""
        ranges = {
            DegradedModeLevel.FULL: (0.8, 1.0),
            DegradedModeLevel.PARTIAL: (0.5, 0.8),
            DegradedModeLevel.MINIMAL: (0.3, 0.5),
            DegradedModeLevel.EMERGENCY: (0.0, 0.3),
        }
        return ranges[self]

    @property
    def severity(self) -> int:
        """Get severity level (0=FULL, 3=EMERGENCY)."""
        severities = {
            DegradedModeLevel.FULL: 0,
            DegradedModeLevel.PARTIAL: 1,
            DegradedModeLevel.MINIMAL: 2,
            DegradedModeLevel.EMERGENCY: 3,
        }
        return severities[self]

    @classmethod
    def from_confidence(cls, confidence: float) -> "DegradedModeLevel":
        """Determine mode level from confidence score."""
        if confidence >= 0.8:
            return cls.FULL
        elif confidence >= 0.5:
            return cls.PARTIAL
        elif confidence >= 0.3:
            return cls.MINIMAL
        else:
            return cls.EMERGENCY


class DegradedModeTrigger(Enum):
    """Triggers that cause mode degradation."""
    HIGH_CLOUD_COVER = "high_cloud_cover"
    MISSING_BASELINE = "missing_baseline"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    ALGORITHM_FAILURE = "algorithm_failure"
    DATA_QUALITY_LOW = "data_quality_low"
    TEMPORAL_GAP = "temporal_gap"
    SPATIAL_GAP = "spatial_gap"
    NETWORK_TIMEOUT = "network_timeout"
    RESOURCE_LIMIT = "resource_limit"
    VALIDATION_FAILURE = "validation_failure"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class DegradedModeState:
    """
    Current state of degraded mode operation.

    Attributes:
        level: Current operational mode level
        triggers: Active triggers causing degradation
        confidence: Current overall confidence score
        timestamp: When this state was established
        message: Human-readable description
        affected_components: Components impacted by degradation
        fallbacks_used: List of fallback mechanisms in use
        metadata: Additional state information
    """
    level: DegradedModeLevel
    triggers: List[DegradedModeTrigger]
    confidence: float
    timestamp: datetime
    message: str
    affected_components: List[str] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        """Check if operating in any degraded mode."""
        return self.level != DegradedModeLevel.FULL

    @property
    def requires_review(self) -> bool:
        """Check if results require manual review."""
        return self.level in (DegradedModeLevel.MINIMAL, DegradedModeLevel.EMERGENCY)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "triggers": [t.value for t in self.triggers],
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "affected_components": self.affected_components,
            "fallbacks_used": self.fallbacks_used,
            "is_degraded": self.is_degraded,
            "requires_review": self.requires_review,
            "metadata": self.metadata,
        }


@dataclass
class ModeTransition:
    """
    Record of a mode transition event.

    Attributes:
        transition_id: Unique identifier for this transition
        from_level: Previous mode level
        to_level: New mode level
        trigger: Primary trigger for the transition
        timestamp: When the transition occurred
        reason: Detailed reason for the transition
        automatic: Whether transition was automatic or manual
    """
    transition_id: str
    from_level: DegradedModeLevel
    to_level: DegradedModeLevel
    trigger: DegradedModeTrigger
    timestamp: datetime
    reason: str
    automatic: bool = True

    @property
    def is_degradation(self) -> bool:
        """Check if this transition is a degradation (worse mode)."""
        return self.to_level.severity > self.from_level.severity

    @property
    def is_recovery(self) -> bool:
        """Check if this transition is a recovery (better mode)."""
        return self.to_level.severity < self.from_level.severity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transition_id": self.transition_id,
            "from_level": self.from_level.value,
            "to_level": self.to_level.value,
            "trigger": self.trigger.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "automatic": self.automatic,
            "is_degradation": self.is_degradation,
            "is_recovery": self.is_recovery,
        }


@dataclass
class DegradedModeConfig:
    """
    Configuration for degraded mode management.

    Attributes:
        cloud_cover_partial_threshold: Cloud cover % to trigger PARTIAL mode
        cloud_cover_minimal_threshold: Cloud cover % to trigger MINIMAL mode
        cloud_cover_emergency_threshold: Cloud cover % to trigger EMERGENCY mode
        min_spatial_coverage: Minimum spatial coverage % for FULL mode
        min_temporal_coverage: Minimum temporal coverage for FULL mode
        min_sensor_count: Minimum number of sensors for FULL mode
        confidence_decay_per_fallback: Confidence reduction per fallback used
        enable_notifications: Whether to send mode change notifications
        notification_callbacks: List of callbacks for notifications
    """
    cloud_cover_partial_threshold: float = 40.0
    cloud_cover_minimal_threshold: float = 70.0
    cloud_cover_emergency_threshold: float = 90.0
    min_spatial_coverage: float = 80.0
    min_temporal_coverage: float = 70.0
    min_sensor_count: int = 2
    confidence_decay_per_fallback: float = 0.1
    enable_notifications: bool = True
    notification_callbacks: List[Callable[[DegradedModeState], None]] = field(
        default_factory=list
    )


class DegradedModeManager:
    """
    Manages degraded mode operations for the analysis pipeline.

    Tracks current operational state, handles mode transitions,
    and notifies users of degraded operations.

    Example:
        manager = DegradedModeManager()

        # Assess current situation
        state = manager.assess_situation(
            cloud_cover=65.0,
            spatial_coverage=75.0,
            available_sensors=["sentinel2", "landsat8"],
        )

        if state.is_degraded:
            print(f"Operating in {state.level.value} mode: {state.message}")

        # Check transition history
        for transition in manager.get_transition_history():
            print(f"{transition.from_level} -> {transition.to_level}")
    """

    def __init__(self, config: Optional[DegradedModeConfig] = None):
        """
        Initialize the degraded mode manager.

        Args:
            config: Configuration options
        """
        self.config = config or DegradedModeConfig()
        self._current_state: Optional[DegradedModeState] = None
        self._transition_history: List[ModeTransition] = []
        self._active_triggers: Set[DegradedModeTrigger] = set()

    @property
    def current_state(self) -> Optional[DegradedModeState]:
        """Get the current operational state."""
        return self._current_state

    @property
    def current_level(self) -> DegradedModeLevel:
        """Get the current mode level."""
        if self._current_state is None:
            return DegradedModeLevel.FULL
        return self._current_state.level

    def assess_situation(
        self,
        cloud_cover: Optional[float] = None,
        spatial_coverage: Optional[float] = None,
        temporal_coverage: Optional[float] = None,
        available_sensors: Optional[List[str]] = None,
        data_quality_scores: Optional[Dict[str, float]] = None,
        algorithm_status: Optional[Dict[str, bool]] = None,
        baseline_available: bool = True,
    ) -> DegradedModeState:
        """
        Assess the current data situation and determine operational mode.

        Args:
            cloud_cover: Cloud cover percentage (0-100)
            spatial_coverage: Spatial coverage percentage (0-100)
            temporal_coverage: Temporal coverage percentage (0-100)
            available_sensors: List of available sensor names
            data_quality_scores: Quality scores per component (0-1)
            algorithm_status: Algorithm availability (True=available)
            baseline_available: Whether pre-event baseline is available

        Returns:
            DegradedModeState with assessed mode level
        """
        triggers = []
        confidence_factors = []
        affected_components = []
        messages = []

        # Assess cloud cover
        if cloud_cover is not None:
            cloud_cover = max(0.0, min(100.0, cloud_cover))
            if cloud_cover >= self.config.cloud_cover_emergency_threshold:
                triggers.append(DegradedModeTrigger.HIGH_CLOUD_COVER)
                confidence_factors.append(0.1)
                affected_components.append("optical_imagery")
                messages.append(f"Extreme cloud cover ({cloud_cover:.0f}%)")
            elif cloud_cover >= self.config.cloud_cover_minimal_threshold:
                triggers.append(DegradedModeTrigger.HIGH_CLOUD_COVER)
                confidence_factors.append(0.3)
                affected_components.append("optical_imagery")
                messages.append(f"High cloud cover ({cloud_cover:.0f}%)")
            elif cloud_cover >= self.config.cloud_cover_partial_threshold:
                triggers.append(DegradedModeTrigger.HIGH_CLOUD_COVER)
                confidence_factors.append(0.6)
                messages.append(f"Moderate cloud cover ({cloud_cover:.0f}%)")
            else:
                confidence_factors.append(1.0 - cloud_cover / 100.0)

        # Assess spatial coverage
        if spatial_coverage is not None:
            spatial_coverage = max(0.0, min(100.0, spatial_coverage))
            if spatial_coverage < self.config.min_spatial_coverage:
                triggers.append(DegradedModeTrigger.SPATIAL_GAP)
                # Scale confidence based on coverage
                conf = spatial_coverage / 100.0
                confidence_factors.append(conf)
                affected_components.append("spatial_coverage")
                messages.append(f"Incomplete coverage ({spatial_coverage:.0f}%)")
            else:
                confidence_factors.append(spatial_coverage / 100.0)

        # Assess temporal coverage
        if temporal_coverage is not None:
            temporal_coverage = max(0.0, min(100.0, temporal_coverage))
            if temporal_coverage < self.config.min_temporal_coverage:
                triggers.append(DegradedModeTrigger.TEMPORAL_GAP)
                conf = temporal_coverage / 100.0
                confidence_factors.append(conf)
                affected_components.append("temporal_coverage")
                messages.append(f"Temporal gaps ({temporal_coverage:.0f}% coverage)")
            else:
                confidence_factors.append(temporal_coverage / 100.0)

        # Assess sensor availability
        if available_sensors is not None:
            sensor_count = len(available_sensors)
            if sensor_count < self.config.min_sensor_count:
                triggers.append(DegradedModeTrigger.SENSOR_UNAVAILABLE)
                conf = sensor_count / self.config.min_sensor_count
                confidence_factors.append(conf)
                affected_components.append("sensor_availability")
                messages.append(f"Limited sensors ({sensor_count} available)")
            else:
                confidence_factors.append(min(1.0, sensor_count / self.config.min_sensor_count))

        # Assess baseline availability
        if not baseline_available:
            triggers.append(DegradedModeTrigger.MISSING_BASELINE)
            confidence_factors.append(0.5)
            affected_components.append("temporal_baseline")
            messages.append("No pre-event baseline available")

        # Assess data quality
        if data_quality_scores:
            avg_quality = np.mean(list(data_quality_scores.values()))
            if avg_quality < 0.5:
                triggers.append(DegradedModeTrigger.DATA_QUALITY_LOW)
                confidence_factors.append(avg_quality)
                affected_components.append("data_quality")
                messages.append(f"Low data quality (avg={avg_quality:.2f})")
            else:
                confidence_factors.append(avg_quality)

        # Assess algorithm availability
        if algorithm_status:
            failed_algos = [name for name, ok in algorithm_status.items() if not ok]
            if failed_algos:
                triggers.append(DegradedModeTrigger.ALGORITHM_FAILURE)
                ratio = 1.0 - len(failed_algos) / len(algorithm_status)
                confidence_factors.append(ratio)
                affected_components.extend(failed_algos)
                messages.append(f"Algorithm failures: {', '.join(failed_algos)}")

        # Calculate overall confidence
        if confidence_factors:
            # Use geometric mean for overall confidence
            confidence = float(np.prod(confidence_factors) ** (1.0 / len(confidence_factors)))
        else:
            confidence = 1.0

        # Determine mode level
        level = DegradedModeLevel.from_confidence(confidence)

        # Generate message
        if messages:
            message = "; ".join(messages)
        elif level == DegradedModeLevel.FULL:
            message = "Operating at full capacity"
        else:
            message = f"Degraded to {level.value} mode"

        # Create new state
        new_state = DegradedModeState(
            level=level,
            triggers=triggers,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            message=message,
            affected_components=affected_components,
            metadata={
                "cloud_cover": cloud_cover,
                "spatial_coverage": spatial_coverage,
                "temporal_coverage": temporal_coverage,
                "sensor_count": len(available_sensors) if available_sensors else None,
                "baseline_available": baseline_available,
            },
        )

        # Handle transition if level changed
        if self._current_state is None or self._current_state.level != level:
            self._handle_transition(new_state, triggers[0] if triggers else DegradedModeTrigger.MANUAL_OVERRIDE)

        self._current_state = new_state
        self._active_triggers = set(triggers)

        return new_state

    def switch_mode(
        self,
        target_level: DegradedModeLevel,
        trigger: DegradedModeTrigger,
        reason: str,
        automatic: bool = True,
    ) -> DegradedModeState:
        """
        Manually switch to a specific mode level.

        Args:
            target_level: Target mode level
            trigger: Trigger causing the switch
            reason: Detailed reason for the switch
            automatic: Whether this is an automatic switch

        Returns:
            New DegradedModeState
        """
        from_level = self.current_level

        # Record transition
        transition = ModeTransition(
            transition_id=str(uuid.uuid4()),
            from_level=from_level,
            to_level=target_level,
            trigger=trigger,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            automatic=automatic,
        )
        self._transition_history.append(transition)

        # Create new state
        min_conf, max_conf = target_level.confidence_range
        confidence = (min_conf + max_conf) / 2

        new_state = DegradedModeState(
            level=target_level,
            triggers=[trigger],
            confidence=confidence,
            timestamp=datetime.now(timezone.utc),
            message=reason,
            metadata={"manual_switch": not automatic},
        )

        self._current_state = new_state

        # Notify if enabled
        if self.config.enable_notifications:
            self._notify_mode_change(new_state)

        logger.info(
            f"Mode switched: {from_level.value} -> {target_level.value} "
            f"({trigger.value}): {reason}"
        )

        return new_state

    def add_fallback(self, fallback_name: str) -> None:
        """
        Record that a fallback mechanism is being used.

        Args:
            fallback_name: Name of the fallback mechanism
        """
        if self._current_state is None:
            return

        self._current_state.fallbacks_used.append(fallback_name)

        # Reduce confidence for each fallback
        decay = self.config.confidence_decay_per_fallback
        self._current_state.confidence = max(0.0, self._current_state.confidence - decay)

        # Check if mode should degrade further
        new_level = DegradedModeLevel.from_confidence(self._current_state.confidence)
        if new_level.severity > self._current_state.level.severity:
            self.switch_mode(
                new_level,
                DegradedModeTrigger.ALGORITHM_FAILURE,
                f"Fallback {fallback_name} reduced confidence",
            )

        logger.debug(
            f"Fallback added: {fallback_name}, "
            f"confidence now {self._current_state.confidence:.2f}"
        )

    def notify_mode_change(self, state: DegradedModeState) -> None:
        """
        Send notifications about mode changes.

        Args:
            state: Current mode state
        """
        self._notify_mode_change(state)

    def get_transition_history(
        self,
        since: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[ModeTransition]:
        """
        Get mode transition history.

        Args:
            since: Only return transitions after this time
            limit: Maximum number of transitions to return

        Returns:
            List of ModeTransition records
        """
        transitions = self._transition_history

        if since is not None:
            transitions = [t for t in transitions if t.timestamp >= since]

        if limit is not None:
            transitions = transitions[-limit:]

        return transitions

    def get_mode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about mode usage.

        Returns:
            Dictionary with mode statistics
        """
        if not self._transition_history:
            return {
                "total_transitions": 0,
                "degradations": 0,
                "recoveries": 0,
                "mode_counts": {},
                "trigger_counts": {},
            }

        mode_counts: Dict[str, int] = {}
        trigger_counts: Dict[str, int] = {}
        degradations = 0
        recoveries = 0

        for transition in self._transition_history:
            # Count modes
            to_mode = transition.to_level.value
            mode_counts[to_mode] = mode_counts.get(to_mode, 0) + 1

            # Count triggers
            trigger = transition.trigger.value
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1

            # Count degradations/recoveries
            if transition.is_degradation:
                degradations += 1
            elif transition.is_recovery:
                recoveries += 1

        return {
            "total_transitions": len(self._transition_history),
            "degradations": degradations,
            "recoveries": recoveries,
            "mode_counts": mode_counts,
            "trigger_counts": trigger_counts,
        }

    def reset(self) -> None:
        """Reset manager to initial state."""
        self._current_state = None
        self._transition_history = []
        self._active_triggers = set()
        logger.info("Degraded mode manager reset")

    def _handle_transition(
        self,
        new_state: DegradedModeState,
        trigger: DegradedModeTrigger,
    ) -> None:
        """Handle a mode level transition."""
        from_level = self.current_level if self._current_state else DegradedModeLevel.FULL

        if from_level == new_state.level:
            return

        # Record transition
        transition = ModeTransition(
            transition_id=str(uuid.uuid4()),
            from_level=from_level,
            to_level=new_state.level,
            trigger=trigger,
            timestamp=new_state.timestamp,
            reason=new_state.message,
            automatic=True,
        )
        self._transition_history.append(transition)

        # Notify if enabled
        if self.config.enable_notifications:
            self._notify_mode_change(new_state)

        logger.info(
            f"Mode transition: {from_level.value} -> {new_state.level.value} "
            f"(confidence={new_state.confidence:.2f})"
        )

    def _notify_mode_change(self, state: DegradedModeState) -> None:
        """Send mode change notifications."""
        for callback in self.config.notification_callbacks:
            try:
                callback(state)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")


def assess_degraded_mode(
    cloud_cover: Optional[float] = None,
    spatial_coverage: Optional[float] = None,
    available_sensors: Optional[List[str]] = None,
    baseline_available: bool = True,
    config: Optional[DegradedModeConfig] = None,
) -> DegradedModeState:
    """
    Convenience function to assess degraded mode.

    Args:
        cloud_cover: Cloud cover percentage (0-100)
        spatial_coverage: Spatial coverage percentage (0-100)
        available_sensors: List of available sensor names
        baseline_available: Whether pre-event baseline is available
        config: Configuration options

    Returns:
        DegradedModeState with assessed mode level

    Example:
        state = assess_degraded_mode(
            cloud_cover=75.0,
            spatial_coverage=85.0,
            available_sensors=["sentinel1"],
            baseline_available=False,
        )
        print(f"Mode: {state.level.value}, Confidence: {state.confidence:.2f}")
    """
    manager = DegradedModeManager(config)
    return manager.assess_situation(
        cloud_cover=cloud_cover,
        spatial_coverage=spatial_coverage,
        available_sensors=available_sensors,
        baseline_available=baseline_available,
    )
