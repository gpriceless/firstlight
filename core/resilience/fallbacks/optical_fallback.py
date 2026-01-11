"""
Optical Sensor Fallback Chain.

Implements a hierarchical fallback strategy for optical imagery:
- Primary: Sentinel-2 (10m resolution)
- Fallback 1: Landsat-8/9 (30m resolution)
- Fallback 2: MODIS (250m resolution)

Cloud handling strategies:
- >80% cloud cover: Switch to SAR
- 40-80% cloud cover: Use cloud-free pixels only
- <40% cloud cover: Proceed with cloud masking

All decisions are logged for traceability and analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OpticalSource(Enum):
    """Available optical data sources in priority order."""
    SENTINEL2 = "sentinel2"         # 10m, 5-day revisit
    LANDSAT8 = "landsat8"           # 30m, 16-day revisit
    LANDSAT9 = "landsat9"           # 30m, 16-day revisit
    MODIS_TERRA = "modis_terra"     # 250m, daily
    MODIS_AQUA = "modis_aqua"       # 250m, daily
    VIIRS = "viirs"                 # 375m, daily


class CloudHandlingStrategy(Enum):
    """Strategy for handling cloudy imagery."""
    PROCEED = "proceed"                 # Use with standard cloud masking
    CLOUD_FREE_ONLY = "cloud_free_only" # Only use cloud-free pixels
    SWITCH_TO_SAR = "switch_to_sar"     # Abandon optical, use SAR
    TEMPORAL_COMPOSITE = "composite"     # Create cloud-free composite
    HYBRID = "hybrid"                   # Combine with SAR


class FallbackReason(Enum):
    """Reasons for triggering a fallback."""
    HIGH_CLOUD_COVER = "high_cloud_cover"
    NO_DATA_AVAILABLE = "no_data_available"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    RESOLUTION_INADEQUATE = "resolution_inadequate"
    ACQUISITION_TOO_OLD = "acquisition_too_old"
    SENSOR_UNAVAILABLE = "sensor_unavailable"


@dataclass
class OpticalFallbackDecision:
    """
    Record of a fallback decision.

    Attributes:
        timestamp: When decision was made
        from_source: Source being abandoned
        to_source: Source being adopted (or strategy)
        reason: Reason for fallback
        cloud_cover_percent: Cloud cover that triggered decision
        quality_score: Quality score at decision time
        details: Additional details
    """
    timestamp: datetime
    from_source: Optional[OpticalSource]
    to_source: Optional[OpticalSource]
    reason: FallbackReason
    cloud_cover_percent: Optional[float] = None
    quality_score: Optional[float] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_source": self.from_source.value if self.from_source else None,
            "to_source": self.to_source.value if self.to_source else None,
            "reason": self.reason.value,
            "cloud_cover_percent": self.cloud_cover_percent,
            "quality_score": self.quality_score,
            "details": self.details,
        }


@dataclass
class OpticalFallbackConfig:
    """
    Configuration for optical fallback chain.

    Attributes:
        cloud_threshold_sar_switch: Cloud cover % to switch to SAR
        cloud_threshold_partial_use: Cloud cover % to use only clear pixels
        min_quality_score: Minimum acceptable quality score
        max_acquisition_age_hours: Maximum age for usable acquisition
        min_resolution_m: Minimum acceptable resolution

        source_priority: Ordered list of preferred sources
        enable_temporal_composite: Whether to allow temporal compositing
        enable_hybrid_sar: Whether to allow optical+SAR hybrid
    """
    cloud_threshold_sar_switch: float = 80.0
    cloud_threshold_partial_use: float = 40.0
    min_quality_score: float = 0.3
    max_acquisition_age_hours: float = 168.0  # 1 week
    min_resolution_m: float = 100.0

    source_priority: List[OpticalSource] = field(default_factory=lambda: [
        OpticalSource.SENTINEL2,
        OpticalSource.LANDSAT8,
        OpticalSource.LANDSAT9,
        OpticalSource.MODIS_TERRA,
        OpticalSource.MODIS_AQUA,
        OpticalSource.VIIRS,
    ])

    enable_temporal_composite: bool = True
    enable_hybrid_sar: bool = True


@dataclass
class SourceAvailability:
    """Availability status for a data source."""
    source: OpticalSource
    available: bool
    cloud_cover_percent: Optional[float]
    quality_score: Optional[float]
    acquisition_time: Optional[datetime]
    resolution_m: float
    reason_unavailable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "available": self.available,
            "cloud_cover_percent": self.cloud_cover_percent,
            "quality_score": self.quality_score,
            "acquisition_time": self.acquisition_time.isoformat() if self.acquisition_time else None,
            "resolution_m": self.resolution_m,
            "reason_unavailable": self.reason_unavailable,
        }


@dataclass
class OpticalFallbackResult:
    """
    Result of optical fallback chain evaluation.

    Attributes:
        success: Whether a usable source was found
        selected_source: Selected optical source (if any)
        cloud_handling: Cloud handling strategy to use
        resolution_m: Resolution of selected source
        quality_score: Quality score of selected source
        cloud_cover_percent: Cloud cover of selected source
        decisions: List of fallback decisions made
        availability: Source availability details
        recommendation: Action recommendation
        metrics: Additional metrics
    """
    success: bool
    selected_source: Optional[OpticalSource]
    cloud_handling: CloudHandlingStrategy
    resolution_m: float
    quality_score: float
    cloud_cover_percent: float
    decisions: List[OpticalFallbackDecision] = field(default_factory=list)
    availability: List[SourceAvailability] = field(default_factory=list)
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "selected_source": self.selected_source.value if self.selected_source else None,
            "cloud_handling": self.cloud_handling.value,
            "resolution_m": self.resolution_m,
            "quality_score": round(self.quality_score, 3),
            "cloud_cover_percent": round(self.cloud_cover_percent, 1),
            "num_decisions": len(self.decisions),
            "decisions": [d.to_dict() for d in self.decisions],
            "availability": [a.to_dict() for a in self.availability],
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


class OpticalFallbackChain:
    """
    Implements optical sensor fallback with cloud handling.

    Evaluates available optical sources in priority order and selects
    the best option considering cloud cover, quality, and resolution.
    Falls back to SAR when optical is unusable.

    Example:
        chain = OpticalFallbackChain()
        result = chain.evaluate(
            source_status={
                OpticalSource.SENTINEL2: {"cloud_cover": 85, "available": True},
                OpticalSource.LANDSAT8: {"cloud_cover": 45, "available": True},
            },
            required_resolution_m=30.0
        )
        if result.cloud_handling == CloudHandlingStrategy.SWITCH_TO_SAR:
            print("Switching to SAR due to cloud cover")
    """

    # Source resolution specifications
    SOURCE_RESOLUTIONS = {
        OpticalSource.SENTINEL2: 10.0,
        OpticalSource.LANDSAT8: 30.0,
        OpticalSource.LANDSAT9: 30.0,
        OpticalSource.MODIS_TERRA: 250.0,
        OpticalSource.MODIS_AQUA: 250.0,
        OpticalSource.VIIRS: 375.0,
    }

    def __init__(self, config: Optional[OpticalFallbackConfig] = None):
        """
        Initialize the optical fallback chain.

        Args:
            config: Configuration options
        """
        self.config = config or OpticalFallbackConfig()
        self.decision_log: List[OpticalFallbackDecision] = []

    def evaluate(
        self,
        source_status: Dict[OpticalSource, Dict[str, Any]],
        required_resolution_m: Optional[float] = None,
        event_time: Optional[datetime] = None,
        current_time: Optional[datetime] = None,
    ) -> OpticalFallbackResult:
        """
        Evaluate optical sources and determine best option.

        Args:
            source_status: Dict mapping sources to their status
                Expected keys: cloud_cover, available, quality_score, acquisition_time
            required_resolution_m: Required resolution (optional override)
            event_time: Event time for timing relevance
            current_time: Current time (defaults to now)

        Returns:
            OpticalFallbackResult with selected source and strategy
        """
        self.decision_log = []
        current_time = current_time or datetime.now()

        min_resolution = required_resolution_m or self.config.min_resolution_m
        availability: List[SourceAvailability] = []

        # Evaluate each source in priority order
        for source in self.config.source_priority:
            status = source_status.get(source, {})

            avail = self._evaluate_source(
                source, status, min_resolution, event_time, current_time
            )
            availability.append(avail)

        # Find best available source
        selected_source = None
        selected_avail = None

        for avail in availability:
            if avail.available:
                selected_source = avail.source
                selected_avail = avail
                break

        # Determine cloud handling strategy
        if selected_source is None:
            # No optical source available
            cloud_handling = CloudHandlingStrategy.SWITCH_TO_SAR
            recommendation = "No suitable optical source available; switch to SAR"
            self._log_decision(
                None, None, FallbackReason.NO_DATA_AVAILABLE,
                details="All optical sources unavailable or inadequate"
            )
        elif selected_avail.cloud_cover_percent is not None:
            cloud_cover = selected_avail.cloud_cover_percent

            if cloud_cover >= self.config.cloud_threshold_sar_switch:
                # Too cloudy for optical
                cloud_handling = CloudHandlingStrategy.SWITCH_TO_SAR
                recommendation = f"Cloud cover {cloud_cover:.0f}% too high; switch to SAR"
                self._log_decision(
                    selected_source, None, FallbackReason.HIGH_CLOUD_COVER,
                    cloud_cover_percent=cloud_cover,
                    details=f"Exceeds threshold {self.config.cloud_threshold_sar_switch}%"
                )
                selected_source = None

            elif cloud_cover >= self.config.cloud_threshold_partial_use:
                # Moderate clouds - use cloud-free pixels only
                if self.config.enable_temporal_composite:
                    cloud_handling = CloudHandlingStrategy.TEMPORAL_COMPOSITE
                    recommendation = f"Moderate cloud cover ({cloud_cover:.0f}%); create temporal composite"
                else:
                    cloud_handling = CloudHandlingStrategy.CLOUD_FREE_ONLY
                    recommendation = f"Moderate cloud cover ({cloud_cover:.0f}%); use cloud-free pixels only"

            else:
                # Acceptable clouds
                cloud_handling = CloudHandlingStrategy.PROCEED
                recommendation = f"Using {selected_source.value} with standard cloud masking"

        else:
            cloud_handling = CloudHandlingStrategy.PROCEED
            recommendation = f"Using {selected_source.value}"

        # Build result
        result = OpticalFallbackResult(
            success=selected_source is not None,
            selected_source=selected_source,
            cloud_handling=cloud_handling,
            resolution_m=selected_avail.resolution_m if selected_avail else 0.0,
            quality_score=selected_avail.quality_score or 0.0 if selected_avail else 0.0,
            cloud_cover_percent=selected_avail.cloud_cover_percent or 0.0 if selected_avail else 100.0,
            decisions=list(self.decision_log),
            availability=availability,
            recommendation=recommendation,
            metrics=self._compute_metrics(availability),
        )

        logger.info(
            f"Optical fallback: selected={selected_source}, "
            f"strategy={cloud_handling.value}"
        )

        return result

    def _evaluate_source(
        self,
        source: OpticalSource,
        status: Dict[str, Any],
        min_resolution: float,
        event_time: Optional[datetime],
        current_time: datetime,
    ) -> SourceAvailability:
        """Evaluate a single source for suitability."""
        resolution = self.SOURCE_RESOLUTIONS.get(source, 100.0)

        # Check if source is available
        if not status.get("available", False):
            return SourceAvailability(
                source=source,
                available=False,
                cloud_cover_percent=None,
                quality_score=None,
                acquisition_time=None,
                resolution_m=resolution,
                reason_unavailable="Data not available for area/time",
            )

        # Check resolution
        if resolution > min_resolution:
            self._log_decision(
                source, self._get_next_source(source),
                FallbackReason.RESOLUTION_INADEQUATE,
                details=f"Resolution {resolution}m exceeds required {min_resolution}m"
            )
            return SourceAvailability(
                source=source,
                available=False,
                cloud_cover_percent=status.get("cloud_cover"),
                quality_score=status.get("quality_score"),
                acquisition_time=status.get("acquisition_time"),
                resolution_m=resolution,
                reason_unavailable=f"Resolution {resolution}m inadequate",
            )

        # Check acquisition age
        acq_time = status.get("acquisition_time")
        if acq_time:
            age_hours = (current_time - acq_time).total_seconds() / 3600
            if age_hours > self.config.max_acquisition_age_hours:
                self._log_decision(
                    source, self._get_next_source(source),
                    FallbackReason.ACQUISITION_TOO_OLD,
                    details=f"Acquisition {age_hours:.0f}h old, max {self.config.max_acquisition_age_hours}h"
                )
                return SourceAvailability(
                    source=source,
                    available=False,
                    cloud_cover_percent=status.get("cloud_cover"),
                    quality_score=status.get("quality_score"),
                    acquisition_time=acq_time,
                    resolution_m=resolution,
                    reason_unavailable=f"Acquisition too old ({age_hours:.0f}h)",
                )

        # Check quality score
        quality = status.get("quality_score")
        if quality is not None and quality < self.config.min_quality_score:
            self._log_decision(
                source, self._get_next_source(source),
                FallbackReason.QUALITY_BELOW_THRESHOLD,
                quality_score=quality,
                details=f"Quality {quality:.2f} below threshold {self.config.min_quality_score}"
            )
            return SourceAvailability(
                source=source,
                available=False,
                cloud_cover_percent=status.get("cloud_cover"),
                quality_score=quality,
                acquisition_time=acq_time,
                resolution_m=resolution,
                reason_unavailable=f"Quality score {quality:.2f} too low",
            )

        # Source is available
        return SourceAvailability(
            source=source,
            available=True,
            cloud_cover_percent=status.get("cloud_cover"),
            quality_score=quality,
            acquisition_time=acq_time,
            resolution_m=resolution,
        )

    def _get_next_source(self, current: OpticalSource) -> Optional[OpticalSource]:
        """Get next source in priority list."""
        try:
            idx = self.config.source_priority.index(current)
            if idx + 1 < len(self.config.source_priority):
                return self.config.source_priority[idx + 1]
        except ValueError:
            pass
        return None

    def _log_decision(
        self,
        from_source: Optional[OpticalSource],
        to_source: Optional[OpticalSource],
        reason: FallbackReason,
        cloud_cover_percent: Optional[float] = None,
        quality_score: Optional[float] = None,
        details: str = "",
    ):
        """Log a fallback decision."""
        decision = OpticalFallbackDecision(
            timestamp=datetime.now(),
            from_source=from_source,
            to_source=to_source,
            reason=reason,
            cloud_cover_percent=cloud_cover_percent,
            quality_score=quality_score,
            details=details,
        )
        self.decision_log.append(decision)
        logger.debug(f"Fallback decision: {reason.value} - {details}")

    def _compute_metrics(
        self, availability: List[SourceAvailability]
    ) -> Dict[str, Any]:
        """Compute summary metrics."""
        return {
            "sources_evaluated": len(availability),
            "sources_available": sum(1 for a in availability if a.available),
            "best_resolution_m": min(
                (a.resolution_m for a in availability if a.available),
                default=0.0
            ),
            "min_cloud_cover": min(
                (a.cloud_cover_percent for a in availability
                 if a.available and a.cloud_cover_percent is not None),
                default=100.0
            ),
        }

    def get_cloud_handling_recommendation(
        self,
        cloud_cover_percent: float,
        has_sar_backup: bool = True,
    ) -> Tuple[CloudHandlingStrategy, str]:
        """
        Get cloud handling recommendation for given cloud cover.

        Args:
            cloud_cover_percent: Cloud cover percentage
            has_sar_backup: Whether SAR backup is available

        Returns:
            Tuple of (strategy, explanation)
        """
        if cloud_cover_percent >= self.config.cloud_threshold_sar_switch:
            if has_sar_backup:
                return (
                    CloudHandlingStrategy.SWITCH_TO_SAR,
                    f"Cloud cover {cloud_cover_percent:.0f}% exceeds {self.config.cloud_threshold_sar_switch}%; "
                    "SAR recommended for all-weather capability"
                )
            else:
                return (
                    CloudHandlingStrategy.TEMPORAL_COMPOSITE,
                    f"Cloud cover {cloud_cover_percent:.0f}% very high; "
                    "attempt temporal composite from multiple dates"
                )

        elif cloud_cover_percent >= self.config.cloud_threshold_partial_use:
            if self.config.enable_temporal_composite:
                return (
                    CloudHandlingStrategy.TEMPORAL_COMPOSITE,
                    f"Cloud cover {cloud_cover_percent:.0f}% moderate; "
                    "create cloud-free composite from temporal stack"
                )
            else:
                return (
                    CloudHandlingStrategy.CLOUD_FREE_ONLY,
                    f"Cloud cover {cloud_cover_percent:.0f}% moderate; "
                    "mask clouds and use only clear pixels"
                )

        else:
            return (
                CloudHandlingStrategy.PROCEED,
                f"Cloud cover {cloud_cover_percent:.0f}% acceptable; "
                "proceed with standard cloud masking"
            )
