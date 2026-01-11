"""
DEM Fallback Chain.

Implements hierarchical fallback strategy for DEM data:
- Primary: Copernicus DEM (30m)
- Fallback 1: SRTM (30m)
- Fallback 2: ASTER (30m)

Void filling strategies:
- Interpolation for small voids
- Neighboring tiles for edge voids
- Alternative DEM for large voids

Resolution degradation handling for analysis requirements.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DEMSource(Enum):
    """Available DEM sources in priority order."""
    COPERNICUS_30 = "copernicus_30"     # Copernicus GLO-30
    COPERNICUS_90 = "copernicus_90"     # Copernicus GLO-90
    SRTM_30 = "srtm_30"                 # SRTM 1 arc-second
    SRTM_90 = "srtm_90"                 # SRTM 3 arc-second
    ASTER_30 = "aster_30"               # ASTER GDEM
    NASADEM = "nasadem"                 # NASA DEM (SRTM reprocessed)
    ALOS_30 = "alos_30"                 # ALOS World 3D
    LIDAR = "lidar"                     # LiDAR if available


class VoidFillingStrategy(Enum):
    """Strategies for filling DEM voids."""
    NONE = "none"                       # Leave voids as-is
    INTERPOLATION = "interpolation"     # Interpolate small voids
    NEIGHBOR_TILES = "neighbor_tiles"   # Use neighboring tiles
    ALTERNATIVE_DEM = "alternative_dem" # Fill with alternate DEM
    BLEND = "blend"                     # Blend multiple sources
    MORPHOLOGICAL = "morphological"     # Morphological interpolation


class DEMFallbackReason(Enum):
    """Reasons for DEM fallback decisions."""
    EXCESSIVE_VOIDS = "excessive_voids"
    RESOLUTION_INADEQUATE = "resolution_inadequate"
    ARTIFACTS_DETECTED = "artifacts_detected"
    COVERAGE_GAP = "coverage_gap"
    ACCURACY_INSUFFICIENT = "accuracy_insufficient"
    SOURCE_UNAVAILABLE = "source_unavailable"


@dataclass
class DEMFallbackDecision:
    """
    Record of a DEM fallback decision.

    Attributes:
        timestamp: When decision was made
        from_source: Source being abandoned
        to_source: Source being adopted
        reason: Reason for fallback
        void_percent: Void percentage that triggered decision
        details: Additional details
    """
    timestamp: datetime
    from_source: Optional[DEMSource]
    to_source: Optional[DEMSource]
    reason: DEMFallbackReason
    void_percent: Optional[float] = None
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "from_source": self.from_source.value if self.from_source else None,
            "to_source": self.to_source.value if self.to_source else None,
            "reason": self.reason.value,
            "void_percent": self.void_percent,
            "details": self.details,
        }


@dataclass
class DEMFallbackConfig:
    """
    Configuration for DEM fallback chain.

    Attributes:
        max_void_percent: Maximum acceptable void percentage
        max_void_size_pixels: Maximum void size for interpolation
        min_accuracy_m: Minimum acceptable vertical accuracy
        required_resolution_m: Required resolution in meters

        source_priority: Ordered list of preferred sources
        enable_void_filling: Enable void filling strategies
        enable_blending: Enable multi-source blending
    """
    max_void_percent: float = 5.0
    max_void_size_pixels: int = 100
    min_accuracy_m: float = 10.0
    required_resolution_m: float = 30.0

    source_priority: List[DEMSource] = field(default_factory=lambda: [
        DEMSource.COPERNICUS_30,
        DEMSource.NASADEM,
        DEMSource.SRTM_30,
        DEMSource.ASTER_30,
        DEMSource.ALOS_30,
        DEMSource.COPERNICUS_90,
        DEMSource.SRTM_90,
    ])

    enable_void_filling: bool = True
    enable_blending: bool = True

    # Void filling parameters
    interpolation_max_distance: int = 10  # pixels
    blend_buffer_pixels: int = 5


@dataclass
class DEMSourceStatus:
    """Status of a DEM source."""
    source: DEMSource
    available: bool
    void_percent: float
    resolution_m: float
    accuracy_m: float
    coverage_percent: float
    reason_unavailable: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source.value,
            "available": self.available,
            "void_percent": round(self.void_percent, 2),
            "resolution_m": self.resolution_m,
            "accuracy_m": self.accuracy_m,
            "coverage_percent": round(self.coverage_percent, 2),
            "reason_unavailable": self.reason_unavailable,
        }


@dataclass
class DEMFallbackResult:
    """
    Result of DEM fallback chain evaluation.

    Attributes:
        success: Whether a usable DEM was found
        primary_source: Primary DEM source selected
        secondary_source: Secondary source for void filling (if any)
        void_filling: Void filling strategy to use
        effective_resolution_m: Effective resolution after processing
        effective_accuracy_m: Effective accuracy after processing
        void_percent_final: Final void percentage after filling
        decisions: List of fallback decisions made
        source_status: Status of all evaluated sources
        recommendation: Action recommendation
        metrics: Additional metrics
    """
    success: bool
    primary_source: Optional[DEMSource]
    secondary_source: Optional[DEMSource]
    void_filling: VoidFillingStrategy
    effective_resolution_m: float
    effective_accuracy_m: float
    void_percent_final: float
    decisions: List[DEMFallbackDecision] = field(default_factory=list)
    source_status: List[DEMSourceStatus] = field(default_factory=list)
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "primary_source": self.primary_source.value if self.primary_source else None,
            "secondary_source": self.secondary_source.value if self.secondary_source else None,
            "void_filling": self.void_filling.value,
            "effective_resolution_m": self.effective_resolution_m,
            "effective_accuracy_m": round(self.effective_accuracy_m, 2),
            "void_percent_final": round(self.void_percent_final, 2),
            "num_decisions": len(self.decisions),
            "decisions": [d.to_dict() for d in self.decisions],
            "source_status": [s.to_dict() for s in self.source_status],
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


class DEMFallbackChain:
    """
    Implements DEM fallback with void filling strategies.

    Evaluates available DEM sources in priority order and selects
    the best option considering voids, resolution, and accuracy.
    Implements void filling when needed.

    Example:
        chain = DEMFallbackChain()
        result = chain.evaluate(
            source_status={
                DEMSource.COPERNICUS_30: {"available": True, "void_percent": 15},
                DEMSource.SRTM_30: {"available": True, "void_percent": 5},
            },
            required_resolution_m=30.0
        )
        if result.void_filling != VoidFillingStrategy.NONE:
            print(f"Void filling needed: {result.void_filling.value}")
    """

    # Source specifications
    SOURCE_SPECS = {
        DEMSource.COPERNICUS_30: {"resolution_m": 30.0, "accuracy_m": 4.0},
        DEMSource.COPERNICUS_90: {"resolution_m": 90.0, "accuracy_m": 4.0},
        DEMSource.SRTM_30: {"resolution_m": 30.0, "accuracy_m": 16.0},
        DEMSource.SRTM_90: {"resolution_m": 90.0, "accuracy_m": 16.0},
        DEMSource.ASTER_30: {"resolution_m": 30.0, "accuracy_m": 17.0},
        DEMSource.NASADEM: {"resolution_m": 30.0, "accuracy_m": 12.0},
        DEMSource.ALOS_30: {"resolution_m": 30.0, "accuracy_m": 5.0},
        DEMSource.LIDAR: {"resolution_m": 1.0, "accuracy_m": 0.15},
    }

    def __init__(self, config: Optional[DEMFallbackConfig] = None):
        """
        Initialize the DEM fallback chain.

        Args:
            config: Configuration options
        """
        self.config = config or DEMFallbackConfig()
        self.decision_log: List[DEMFallbackDecision] = []

    def evaluate(
        self,
        source_status: Dict[DEMSource, Dict[str, Any]],
        required_resolution_m: Optional[float] = None,
        required_accuracy_m: Optional[float] = None,
        terrain_type: Optional[str] = None,
    ) -> DEMFallbackResult:
        """
        Evaluate DEM sources and determine best option.

        Args:
            source_status: Dict mapping sources to their status
                Expected keys: available, void_percent, coverage_percent
            required_resolution_m: Required resolution (optional override)
            required_accuracy_m: Required accuracy (optional override)
            terrain_type: Type of terrain for priority adjustment

        Returns:
            DEMFallbackResult with selected source and strategy
        """
        self.decision_log = []

        required_res = required_resolution_m or self.config.required_resolution_m
        required_acc = required_accuracy_m or self.config.min_accuracy_m

        # Evaluate all sources
        evaluated_sources: List[DEMSourceStatus] = []

        for source in self.config.source_priority:
            status = source_status.get(source, {})
            specs = self.SOURCE_SPECS.get(source, {})

            evaluated = self._evaluate_source(
                source, status, specs, required_res, required_acc
            )
            evaluated_sources.append(evaluated)

        # Find best primary source
        primary_source = None
        primary_status = None

        for status in evaluated_sources:
            if status.available:
                primary_source = status.source
                primary_status = status
                break

        if primary_source is None:
            return DEMFallbackResult(
                success=False,
                primary_source=None,
                secondary_source=None,
                void_filling=VoidFillingStrategy.NONE,
                effective_resolution_m=0.0,
                effective_accuracy_m=999.0,
                void_percent_final=100.0,
                decisions=list(self.decision_log),
                source_status=evaluated_sources,
                recommendation="No suitable DEM source available",
            )

        # Determine if void filling is needed
        void_percent = primary_status.void_percent
        void_filling = VoidFillingStrategy.NONE
        secondary_source = None
        void_percent_final = void_percent

        if void_percent > self.config.max_void_percent and self.config.enable_void_filling:
            void_filling, secondary_source, void_percent_final = self._determine_void_strategy(
                primary_source,
                primary_status,
                evaluated_sources
            )

        # Calculate effective specs
        effective_resolution = primary_status.resolution_m
        effective_accuracy = primary_status.accuracy_m

        if secondary_source:
            secondary_specs = self.SOURCE_SPECS.get(secondary_source, {})
            # Take worst case for blended data
            effective_resolution = max(
                effective_resolution,
                secondary_specs.get("resolution_m", effective_resolution)
            )
            effective_accuracy = max(
                effective_accuracy,
                secondary_specs.get("accuracy_m", effective_accuracy)
            )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            primary_source,
            secondary_source,
            void_filling,
            void_percent,
            void_percent_final
        )

        return DEMFallbackResult(
            success=True,
            primary_source=primary_source,
            secondary_source=secondary_source,
            void_filling=void_filling,
            effective_resolution_m=effective_resolution,
            effective_accuracy_m=effective_accuracy,
            void_percent_final=void_percent_final,
            decisions=list(self.decision_log),
            source_status=evaluated_sources,
            recommendation=recommendation,
            metrics=self._compute_metrics(evaluated_sources, void_percent, void_percent_final),
        )

    def _evaluate_source(
        self,
        source: DEMSource,
        status: Dict[str, Any],
        specs: Dict[str, Any],
        required_res: float,
        required_acc: float,
    ) -> DEMSourceStatus:
        """Evaluate a single DEM source."""
        resolution = specs.get("resolution_m", 30.0)
        accuracy = specs.get("accuracy_m", 10.0)

        # Check availability
        if not status.get("available", False):
            return DEMSourceStatus(
                source=source,
                available=False,
                void_percent=100.0,
                resolution_m=resolution,
                accuracy_m=accuracy,
                coverage_percent=0.0,
                reason_unavailable="Data not available for area",
            )

        # Get void percentage
        void_percent = status.get("void_percent", 0.0)
        coverage = status.get("coverage_percent", 100.0 - void_percent)

        # Check resolution
        if resolution > required_res:
            self._log_decision(
                source, self._get_next_source(source),
                DEMFallbackReason.RESOLUTION_INADEQUATE,
                details=f"Resolution {resolution}m exceeds required {required_res}m"
            )
            return DEMSourceStatus(
                source=source,
                available=False,
                void_percent=void_percent,
                resolution_m=resolution,
                accuracy_m=accuracy,
                coverage_percent=coverage,
                reason_unavailable=f"Resolution {resolution}m inadequate",
            )

        # Check accuracy
        if accuracy > required_acc:
            self._log_decision(
                source, self._get_next_source(source),
                DEMFallbackReason.ACCURACY_INSUFFICIENT,
                details=f"Accuracy {accuracy}m exceeds required {required_acc}m"
            )
            return DEMSourceStatus(
                source=source,
                available=False,
                void_percent=void_percent,
                resolution_m=resolution,
                accuracy_m=accuracy,
                coverage_percent=coverage,
                reason_unavailable=f"Accuracy {accuracy}m insufficient",
            )

        # Check coverage
        if coverage < 50:
            self._log_decision(
                source, self._get_next_source(source),
                DEMFallbackReason.COVERAGE_GAP,
                details=f"Coverage only {coverage:.0f}%"
            )
            return DEMSourceStatus(
                source=source,
                available=False,
                void_percent=void_percent,
                resolution_m=resolution,
                accuracy_m=accuracy,
                coverage_percent=coverage,
                reason_unavailable=f"Coverage {coverage:.0f}% insufficient",
            )

        return DEMSourceStatus(
            source=source,
            available=True,
            void_percent=void_percent,
            resolution_m=resolution,
            accuracy_m=accuracy,
            coverage_percent=coverage,
        )

    def _determine_void_strategy(
        self,
        primary_source: DEMSource,
        primary_status: DEMSourceStatus,
        all_sources: List[DEMSourceStatus]
    ) -> Tuple[VoidFillingStrategy, Optional[DEMSource], float]:
        """Determine void filling strategy."""
        void_percent = primary_status.void_percent

        # Small voids - interpolation
        if void_percent <= 10:
            self._log_decision(
                primary_source, None,
                DEMFallbackReason.EXCESSIVE_VOIDS,
                void_percent=void_percent,
                details="Small void percentage - using interpolation"
            )
            return (
                VoidFillingStrategy.INTERPOLATION,
                None,
                max(0.0, void_percent - 8.0)  # Estimate remaining voids
            )

        # Larger voids - try alternative DEM
        secondary = None
        for status in all_sources:
            if (status.source != primary_source and
                status.available and
                status.void_percent < void_percent):
                secondary = status.source
                break

        if secondary:
            self._log_decision(
                primary_source, secondary,
                DEMFallbackReason.EXCESSIVE_VOIDS,
                void_percent=void_percent,
                details=f"Using {secondary.value} for void filling"
            )

            if self.config.enable_blending:
                return (
                    VoidFillingStrategy.BLEND,
                    secondary,
                    max(0.0, void_percent * 0.1)  # Significant reduction
                )
            else:
                return (
                    VoidFillingStrategy.ALTERNATIVE_DEM,
                    secondary,
                    max(0.0, void_percent * 0.2)
                )

        # No alternative available - use interpolation anyway
        self._log_decision(
            primary_source, None,
            DEMFallbackReason.EXCESSIVE_VOIDS,
            void_percent=void_percent,
            details="No alternative DEM - using morphological interpolation"
        )
        return (
            VoidFillingStrategy.MORPHOLOGICAL,
            None,
            max(0.0, void_percent * 0.5)  # Partial reduction
        )

    def _get_next_source(self, current: DEMSource) -> Optional[DEMSource]:
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
        from_source: Optional[DEMSource],
        to_source: Optional[DEMSource],
        reason: DEMFallbackReason,
        void_percent: Optional[float] = None,
        details: str = "",
    ):
        """Log a fallback decision."""
        decision = DEMFallbackDecision(
            timestamp=datetime.now(),
            from_source=from_source,
            to_source=to_source,
            reason=reason,
            void_percent=void_percent,
            details=details,
        )
        self.decision_log.append(decision)
        logger.debug(f"DEM fallback decision: {reason.value} - {details}")

    def _generate_recommendation(
        self,
        primary: DEMSource,
        secondary: Optional[DEMSource],
        void_strategy: VoidFillingStrategy,
        void_before: float,
        void_after: float,
    ) -> str:
        """Generate human-readable recommendation."""
        parts = [f"Use {primary.value} as primary DEM"]

        if void_strategy != VoidFillingStrategy.NONE:
            if secondary:
                parts.append(f"fill voids with {secondary.value} using {void_strategy.value}")
            else:
                parts.append(f"apply {void_strategy.value} for void filling")

            parts.append(f"(voids: {void_before:.1f}% -> {void_after:.1f}%)")

        return "; ".join(parts)

    def _compute_metrics(
        self,
        sources: List[DEMSourceStatus],
        void_before: float,
        void_after: float,
    ) -> Dict[str, Any]:
        """Compute summary metrics."""
        return {
            "sources_evaluated": len(sources),
            "sources_available": sum(1 for s in sources if s.available),
            "best_resolution_m": min(
                (s.resolution_m for s in sources if s.available),
                default=0.0
            ),
            "best_accuracy_m": min(
                (s.accuracy_m for s in sources if s.available),
                default=999.0
            ),
            "void_reduction_percent": void_before - void_after,
        }

    def get_void_filling_recommendation(
        self,
        void_percent: float,
        max_void_size_pixels: int,
        has_alternative_dem: bool = False,
    ) -> Tuple[VoidFillingStrategy, str]:
        """
        Get void filling recommendation for given situation.

        Args:
            void_percent: Percentage of voids
            max_void_size_pixels: Size of largest void
            has_alternative_dem: Whether alternative DEM is available

        Returns:
            Tuple of (strategy, explanation)
        """
        if void_percent <= 1.0:
            return (
                VoidFillingStrategy.NONE,
                f"Void coverage {void_percent:.1f}% negligible; no filling needed"
            )

        if max_void_size_pixels <= self.config.max_void_size_pixels:
            return (
                VoidFillingStrategy.INTERPOLATION,
                f"Small voids ({max_void_size_pixels} pixels max); interpolation suitable"
            )

        if has_alternative_dem and self.config.enable_blending:
            return (
                VoidFillingStrategy.BLEND,
                f"Large voids ({void_percent:.1f}%); blending with alternative DEM recommended"
            )

        if has_alternative_dem:
            return (
                VoidFillingStrategy.ALTERNATIVE_DEM,
                f"Large voids ({void_percent:.1f}%); fill with alternative DEM"
            )

        return (
            VoidFillingStrategy.MORPHOLOGICAL,
            f"Large voids ({void_percent:.1f}%), no alternative; morphological interpolation as fallback"
        )
