"""
SAR Sensor Fallback Chain.

Implements fallback strategies for SAR imagery when quality issues arise:
- High noise: Apply enhanced filtering (Lee, Frost, Refined Lee)
- Geometric issues: Try different orbit direction
- C-band inadequate: Switch to L-band for penetration
- Multi-look processing for noise reduction

All decisions are logged for traceability and analysis.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SARBand(Enum):
    """SAR frequency bands."""
    X_BAND = "x_band"       # ~3cm wavelength, high resolution
    C_BAND = "c_band"       # ~5cm wavelength, Sentinel-1
    L_BAND = "l_band"       # ~23cm wavelength, penetration
    P_BAND = "p_band"       # ~70cm wavelength, deep penetration


class SARSensor(Enum):
    """Available SAR sensors."""
    SENTINEL1 = "sentinel1"       # C-band, free
    ALOS2_PALSAR = "alos2"        # L-band
    RADARSAT2 = "radarsat2"       # C-band
    TERRASAR_X = "terrasar_x"     # X-band
    ICEYE = "iceye"               # X-band, rapid revisit
    CAPELLA = "capella"           # X-band, high resolution


class OrbitDirection(Enum):
    """SAR orbit direction."""
    ASCENDING = "ascending"
    DESCENDING = "descending"


class FilterStrategy(Enum):
    """Speckle filtering strategies."""
    NONE = "none"
    LEE = "lee"
    FROST = "frost"
    REFINED_LEE = "refined_lee"
    GAMMA_MAP = "gamma_map"
    MULTI_LOOK = "multi_look"
    TEMPORAL_AVERAGE = "temporal_average"


class SARFallbackReason(Enum):
    """Reasons for SAR fallback decisions."""
    HIGH_SPECKLE = "high_speckle"
    GEOMETRIC_DISTORTION = "geometric_distortion"
    LAYOVER_SHADOW = "layover_shadow"
    PENETRATION_NEEDED = "penetration_needed"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    QUALITY_BELOW_THRESHOLD = "quality_below_threshold"
    CALIBRATION_ERROR = "calibration_error"


@dataclass
class SARFallbackDecision:
    """
    Record of a SAR fallback decision.

    Attributes:
        timestamp: When decision was made
        reason: Reason for fallback
        action: Action taken
        from_config: Original configuration
        to_config: New configuration
        details: Additional details
    """
    timestamp: datetime
    reason: SARFallbackReason
    action: str
    from_config: Dict[str, Any]
    to_config: Dict[str, Any]
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason.value,
            "action": self.action,
            "from_config": self.from_config,
            "to_config": self.to_config,
            "details": self.details,
        }


@dataclass
class SARFallbackConfig:
    """
    Configuration for SAR fallback chain.

    Attributes:
        enl_threshold_filter: ENL below which to apply filtering
        enl_threshold_enhanced: ENL below which to use enhanced filtering
        layover_threshold_percent: Layover % to try different orbit
        shadow_threshold_percent: Shadow % to try different orbit
        calibration_error_threshold_db: Calibration error to trigger correction

        sensor_priority: Ordered list of preferred sensors
        enable_band_switching: Allow switching frequency bands
        enable_orbit_switching: Allow switching orbit direction
        enable_multi_temporal: Allow multi-temporal processing
    """
    enl_threshold_filter: float = 5.0
    enl_threshold_enhanced: float = 3.0
    layover_threshold_percent: float = 20.0
    shadow_threshold_percent: float = 15.0
    calibration_error_threshold_db: float = 2.0

    sensor_priority: List[SARSensor] = field(default_factory=lambda: [
        SARSensor.SENTINEL1,
        SARSensor.ALOS2_PALSAR,
        SARSensor.RADARSAT2,
        SARSensor.TERRASAR_X,
    ])

    enable_band_switching: bool = True
    enable_orbit_switching: bool = True
    enable_multi_temporal: bool = True

    # Filter parameters
    lee_window_size: int = 5
    frost_window_size: int = 5
    frost_damping: float = 2.0
    multi_look_factor: int = 4


@dataclass
class SARFallbackResult:
    """
    Result of SAR fallback chain evaluation.

    Attributes:
        success: Whether a usable configuration was found
        selected_sensor: Selected SAR sensor
        selected_band: Selected frequency band
        selected_orbit: Selected orbit direction
        filter_strategy: Recommended filtering strategy
        filter_parameters: Filter parameters
        processing_chain: Recommended processing chain
        decisions: List of fallback decisions made
        quality_improvement: Expected quality improvement
        recommendation: Action recommendation
        metrics: Additional metrics
    """
    success: bool
    selected_sensor: Optional[SARSensor]
    selected_band: SARBand
    selected_orbit: OrbitDirection
    filter_strategy: FilterStrategy
    filter_parameters: Dict[str, Any]
    processing_chain: List[str]
    decisions: List[SARFallbackDecision] = field(default_factory=list)
    quality_improvement: float = 0.0
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "selected_sensor": self.selected_sensor.value if self.selected_sensor else None,
            "selected_band": self.selected_band.value,
            "selected_orbit": self.selected_orbit.value,
            "filter_strategy": self.filter_strategy.value,
            "filter_parameters": self.filter_parameters,
            "processing_chain": self.processing_chain,
            "num_decisions": len(self.decisions),
            "decisions": [d.to_dict() for d in self.decisions],
            "quality_improvement": round(self.quality_improvement, 2),
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


class SARFallbackChain:
    """
    Implements SAR fallback strategies for quality issues.

    Evaluates SAR data quality and recommends processing adjustments,
    sensor changes, or orbit switching to achieve acceptable results.

    Example:
        chain = SARFallbackChain()
        result = chain.evaluate(
            quality_assessment={
                "enl": 2.5,
                "layover_percent": 25.0,
                "calibration_error_db": 1.5,
            },
            available_sensors=[SARSensor.SENTINEL1, SARSensor.ALOS2_PALSAR],
            current_orbit=OrbitDirection.ASCENDING
        )
        print(f"Recommended filter: {result.filter_strategy.value}")
    """

    # Sensor characteristics
    SENSOR_BANDS = {
        SARSensor.SENTINEL1: SARBand.C_BAND,
        SARSensor.ALOS2_PALSAR: SARBand.L_BAND,
        SARSensor.RADARSAT2: SARBand.C_BAND,
        SARSensor.TERRASAR_X: SARBand.X_BAND,
        SARSensor.ICEYE: SARBand.X_BAND,
        SARSensor.CAPELLA: SARBand.X_BAND,
    }

    SENSOR_RESOLUTIONS = {
        SARSensor.SENTINEL1: 10.0,
        SARSensor.ALOS2_PALSAR: 25.0,
        SARSensor.RADARSAT2: 8.0,
        SARSensor.TERRASAR_X: 1.0,
        SARSensor.ICEYE: 1.0,
        SARSensor.CAPELLA: 0.5,
    }

    def __init__(self, config: Optional[SARFallbackConfig] = None):
        """
        Initialize the SAR fallback chain.

        Args:
            config: Configuration options
        """
        self.config = config or SARFallbackConfig()
        self.decision_log: List[SARFallbackDecision] = []

    def evaluate(
        self,
        quality_assessment: Dict[str, Any],
        available_sensors: List[SARSensor],
        current_sensor: Optional[SARSensor] = None,
        current_orbit: Optional[OrbitDirection] = None,
        terrain_type: Optional[str] = None,
        target_application: Optional[str] = None,
    ) -> SARFallbackResult:
        """
        Evaluate SAR data and determine best processing strategy.

        Args:
            quality_assessment: Quality metrics (enl, layover_percent, etc.)
            available_sensors: List of available SAR sensors
            current_sensor: Currently selected sensor
            current_orbit: Current orbit direction
            terrain_type: Type of terrain (flat, mountainous, urban)
            target_application: Target application (flood, fire, subsidence)

        Returns:
            SARFallbackResult with recommendations
        """
        self.decision_log = []
        current_sensor = current_sensor or (
            available_sensors[0] if available_sensors else None
        )
        current_orbit = current_orbit or OrbitDirection.DESCENDING

        # Initialize result components
        selected_sensor = current_sensor
        selected_band = self.SENSOR_BANDS.get(current_sensor, SARBand.C_BAND) if current_sensor else SARBand.C_BAND
        selected_orbit = current_orbit
        filter_strategy = FilterStrategy.NONE
        filter_parameters: Dict[str, Any] = {}
        processing_chain: List[str] = ["calibration"]

        # Evaluate speckle noise
        enl = quality_assessment.get("enl", 10.0)
        if enl < self.config.enl_threshold_enhanced:
            # Very noisy - use enhanced filtering
            filter_strategy, filter_parameters = self._get_enhanced_filter(enl)
            self._log_decision(
                SARFallbackReason.HIGH_SPECKLE,
                f"Apply {filter_strategy.value} filter",
                {"enl": enl},
                {"filter": filter_strategy.value, "parameters": filter_parameters},
                f"ENL {enl:.1f} below enhanced threshold {self.config.enl_threshold_enhanced}"
            )
            processing_chain.append(f"{filter_strategy.value}_filter")

        elif enl < self.config.enl_threshold_filter:
            # Moderately noisy - standard filtering
            filter_strategy = FilterStrategy.LEE
            filter_parameters = {"window_size": self.config.lee_window_size}
            self._log_decision(
                SARFallbackReason.HIGH_SPECKLE,
                f"Apply {filter_strategy.value} filter",
                {"enl": enl},
                {"filter": filter_strategy.value},
                f"ENL {enl:.1f} below filter threshold {self.config.enl_threshold_filter}"
            )
            processing_chain.append("lee_filter")

        # Evaluate geometric distortion
        layover = quality_assessment.get("layover_percent", 0.0)
        shadow = quality_assessment.get("shadow_percent", 0.0)

        if layover > self.config.layover_threshold_percent or shadow > self.config.shadow_threshold_percent:
            # Try different orbit
            if self.config.enable_orbit_switching:
                new_orbit = (
                    OrbitDirection.DESCENDING
                    if current_orbit == OrbitDirection.ASCENDING
                    else OrbitDirection.ASCENDING
                )
                selected_orbit = new_orbit
                self._log_decision(
                    SARFallbackReason.LAYOVER_SHADOW,
                    f"Switch to {new_orbit.value} orbit",
                    {"layover": layover, "shadow": shadow, "orbit": current_orbit.value},
                    {"orbit": new_orbit.value},
                    f"Layover {layover:.0f}% or shadow {shadow:.0f}% exceeds threshold"
                )

            processing_chain.append("terrain_correction")

        # Check if band switching is needed
        if target_application and self.config.enable_band_switching:
            recommended_band = self._recommend_band(target_application, terrain_type)
            if recommended_band != selected_band:
                # Find sensor with recommended band
                for sensor in available_sensors:
                    if self.SENSOR_BANDS.get(sensor) == recommended_band:
                        selected_sensor = sensor
                        selected_band = recommended_band
                        self._log_decision(
                            SARFallbackReason.PENETRATION_NEEDED,
                            f"Switch to {recommended_band.value} ({sensor.value})",
                            {"band": selected_band.value},
                            {"band": recommended_band.value, "sensor": sensor.value},
                            f"Application {target_application} benefits from {recommended_band.value}"
                        )
                        break

        # Check calibration
        cal_error = quality_assessment.get("calibration_error_db", 0.0)
        if abs(cal_error) > self.config.calibration_error_threshold_db:
            self._log_decision(
                SARFallbackReason.CALIBRATION_ERROR,
                "Apply radiometric terrain flattening",
                {"calibration_error_db": cal_error},
                {"correction": "terrain_flattening"},
                f"Calibration error {cal_error:.1f}dB exceeds threshold"
            )
            processing_chain.append("terrain_flattening")

        # Calculate expected quality improvement
        quality_improvement = self._estimate_quality_improvement(
            enl, filter_strategy, selected_orbit != current_orbit
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            filter_strategy,
            selected_sensor,
            selected_orbit,
            processing_chain
        )

        return SARFallbackResult(
            success=True,
            selected_sensor=selected_sensor,
            selected_band=selected_band,
            selected_orbit=selected_orbit,
            filter_strategy=filter_strategy,
            filter_parameters=filter_parameters,
            processing_chain=processing_chain,
            decisions=list(self.decision_log),
            quality_improvement=quality_improvement,
            recommendation=recommendation,
            metrics={
                "input_enl": enl,
                "expected_enl": self._estimate_filtered_enl(enl, filter_strategy),
                "layover_percent": layover,
                "shadow_percent": shadow,
            },
        )

    def _get_enhanced_filter(
        self, enl: float
    ) -> Tuple[FilterStrategy, Dict[str, Any]]:
        """Determine enhanced filtering strategy for very noisy data."""
        if enl < 1.5:
            # Extremely noisy - use multi-temporal if available
            if self.config.enable_multi_temporal:
                return (
                    FilterStrategy.TEMPORAL_AVERAGE,
                    {"num_looks": 4, "window_size": 5}
                )
            else:
                return (
                    FilterStrategy.REFINED_LEE,
                    {"window_size": 7, "num_looks": 4}
                )
        else:
            return (
                FilterStrategy.FROST,
                {"window_size": self.config.frost_window_size,
                 "damping": self.config.frost_damping}
            )

    def _recommend_band(
        self,
        application: str,
        terrain_type: Optional[str]
    ) -> SARBand:
        """Recommend optimal SAR band for application."""
        band_recommendations = {
            "flood": SARBand.C_BAND,       # Good for water detection
            "wildfire": SARBand.C_BAND,    # Burn scar detection
            "subsidence": SARBand.C_BAND,  # InSAR applications
            "forestry": SARBand.L_BAND,    # Penetrates canopy
            "agriculture": SARBand.C_BAND, # Crop monitoring
            "urban": SARBand.X_BAND,       # High resolution
            "glacier": SARBand.C_BAND,     # Ice dynamics
            "oil_spill": SARBand.C_BAND,   # Surface detection
        }

        # Override for mountainous terrain (L-band handles slopes better)
        if terrain_type == "mountainous":
            return SARBand.L_BAND

        return band_recommendations.get(application, SARBand.C_BAND)

    def _estimate_quality_improvement(
        self,
        current_enl: float,
        filter_strategy: FilterStrategy,
        orbit_change: bool
    ) -> float:
        """Estimate expected quality improvement (0-1 scale)."""
        improvement = 0.0

        # Filtering improvement
        filter_improvements = {
            FilterStrategy.NONE: 0.0,
            FilterStrategy.LEE: 0.15,
            FilterStrategy.FROST: 0.20,
            FilterStrategy.REFINED_LEE: 0.25,
            FilterStrategy.GAMMA_MAP: 0.22,
            FilterStrategy.MULTI_LOOK: 0.30,
            FilterStrategy.TEMPORAL_AVERAGE: 0.40,
        }
        improvement += filter_improvements.get(filter_strategy, 0.0)

        # Orbit change improvement for geometric issues
        if orbit_change:
            improvement += 0.15

        return min(1.0, improvement)

    def _estimate_filtered_enl(
        self,
        current_enl: float,
        filter_strategy: FilterStrategy
    ) -> float:
        """Estimate ENL after filtering."""
        enl_multipliers = {
            FilterStrategy.NONE: 1.0,
            FilterStrategy.LEE: 2.5,
            FilterStrategy.FROST: 2.0,
            FilterStrategy.REFINED_LEE: 3.0,
            FilterStrategy.GAMMA_MAP: 2.5,
            FilterStrategy.MULTI_LOOK: 4.0,
            FilterStrategy.TEMPORAL_AVERAGE: 4.0,
        }
        multiplier = enl_multipliers.get(filter_strategy, 1.0)
        return current_enl * multiplier

    def _generate_recommendation(
        self,
        filter_strategy: FilterStrategy,
        sensor: Optional[SARSensor],
        orbit: OrbitDirection,
        processing_chain: List[str]
    ) -> str:
        """Generate human-readable recommendation."""
        parts = []

        if sensor:
            parts.append(f"Use {sensor.value}")

        if orbit:
            parts.append(f"{orbit.value} orbit")

        if filter_strategy != FilterStrategy.NONE:
            parts.append(f"with {filter_strategy.value} filtering")

        if len(processing_chain) > 2:
            parts.append(f"Processing: {' -> '.join(processing_chain)}")

        return "; ".join(parts) if parts else "No changes needed"

    def _log_decision(
        self,
        reason: SARFallbackReason,
        action: str,
        from_config: Dict[str, Any],
        to_config: Dict[str, Any],
        details: str = "",
    ):
        """Log a fallback decision."""
        decision = SARFallbackDecision(
            timestamp=datetime.now(),
            reason=reason,
            action=action,
            from_config=from_config,
            to_config=to_config,
            details=details,
        )
        self.decision_log.append(decision)
        logger.debug(f"SAR fallback decision: {reason.value} - {action}")

    def get_filter_recommendation(
        self,
        enl: float,
        has_temporal_stack: bool = False
    ) -> Tuple[FilterStrategy, Dict[str, Any], str]:
        """
        Get filter recommendation for given ENL.

        Args:
            enl: Equivalent number of looks
            has_temporal_stack: Whether multi-temporal data is available

        Returns:
            Tuple of (strategy, parameters, explanation)
        """
        if enl >= self.config.enl_threshold_filter:
            return (
                FilterStrategy.NONE,
                {},
                f"ENL {enl:.1f} adequate; filtering optional"
            )

        if enl < self.config.enl_threshold_enhanced:
            if has_temporal_stack:
                return (
                    FilterStrategy.TEMPORAL_AVERAGE,
                    {"num_images": 4},
                    f"Very noisy (ENL={enl:.1f}); temporal averaging recommended"
                )
            else:
                return (
                    FilterStrategy.REFINED_LEE,
                    {"window_size": 7, "num_looks": 4},
                    f"Very noisy (ENL={enl:.1f}); refined Lee filter recommended"
                )

        return (
            FilterStrategy.LEE,
            {"window_size": 5},
            f"Moderate noise (ENL={enl:.1f}); Lee filter recommended"
        )
