"""
Cross-Sensor Validation for Quality Control.

Validates agreement between observations from different sensor types to assess
multi-source reliability and identify sensor-specific biases, including:
- Sensor-to-sensor comparison metrics
- Expected differences by sensor physics
- Calibration verification across sensors
- Complementary vs redundant sensor validation
- Temporal coincidence analysis

Key Concepts:
- Different sensors measure different physical properties
- Expected relationships exist (e.g., SAR/optical for water detection)
- Complementary sensors provide independent validation
- Redundant sensors provide direct comparison
- Sensor physics determines expected agreement levels
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Types of sensors for cross-validation."""
    OPTICAL = "optical"               # Multispectral optical (Sentinel-2, Landsat)
    SAR = "sar"                       # Synthetic Aperture Radar (Sentinel-1)
    THERMAL = "thermal"               # Thermal infrared (MODIS thermal, Landsat TIR)
    LIDAR = "lidar"                   # Light Detection and Ranging
    DEM = "dem"                       # Digital Elevation Models
    WEATHER = "weather"               # Weather/forecast data
    ANCILLARY = "ancillary"           # Reference/ancillary data (land cover, etc.)


class SensorPairType(Enum):
    """Relationship type between two sensors."""
    REDUNDANT = "redundant"           # Same observable, direct comparison
    COMPLEMENTARY = "complementary"    # Different observables, physics relationship
    INDEPENDENT = "independent"        # No expected relationship


class ValidationOutcome(Enum):
    """Outcome of cross-sensor validation."""
    CONSISTENT = "consistent"          # Sensors agree as expected
    MINOR_DISCREPANCY = "minor"        # Small differences, within tolerance
    SIGNIFICANT_DISCREPANCY = "significant"  # Differences outside tolerance
    CONFLICT = "conflict"              # Fundamental disagreement
    INSUFFICIENT_DATA = "insufficient" # Cannot validate due to data gaps


@dataclass
class SensorObservation:
    """
    Observation from a single sensor.

    Attributes:
        data: Observation data array
        sensor_type: Type of sensor
        sensor_id: Specific sensor identifier (e.g., "sentinel2_msi")
        timestamp: Observation timestamp
        confidence: Per-pixel confidence (0-1)
        metadata: Additional sensor metadata
        observable: What is being observed (e.g., "water_extent", "burn_severity")
    """
    data: np.ndarray
    sensor_type: SensorType
    sensor_id: str
    timestamp: Optional[datetime] = None
    confidence: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    observable: str = "unknown"

    def __post_init__(self):
        """Initialize confidence if not provided."""
        if self.confidence is None:
            if self.data.dtype.kind == 'f':
                self.confidence = (~np.isnan(self.data)).astype(np.float32)
            else:
                self.confidence = np.ones_like(self.data, dtype=np.float32)


@dataclass
class SensorPhysicsRule:
    """
    Expected relationship between two sensors based on physics.

    Attributes:
        sensor_a: First sensor type
        sensor_b: Second sensor type
        observable: The phenomenon being observed
        pair_type: Relationship type (redundant/complementary/independent)
        expected_correlation: Expected correlation range
        expected_bias: Expected bias between sensors
        tolerance: Acceptable disagreement threshold
        description: Human-readable explanation
    """
    sensor_a: SensorType
    sensor_b: SensorType
    observable: str
    pair_type: SensorPairType
    expected_correlation: Tuple[float, float] = (0.0, 1.0)  # min, max
    expected_bias: Tuple[float, float] = (-0.1, 0.1)  # min, max
    tolerance: float = 0.2
    description: str = ""


@dataclass
class SensorComparisonResult:
    """
    Result of comparing two sensors.

    Attributes:
        sensor_a_id: First sensor identifier
        sensor_b_id: Second sensor identifier
        pair_type: Relationship type
        correlation: Actual correlation
        bias: Actual bias
        rmse: Root mean squared error
        agreement_fraction: Fraction of pixels agreeing
        outcome: Validation outcome
        physics_rule: Applied physics rule
        spatial_agreement_map: Per-pixel agreement map
        diagnostics: Additional diagnostic info
    """
    sensor_a_id: str
    sensor_b_id: str
    pair_type: SensorPairType
    correlation: float
    bias: float
    rmse: float
    agreement_fraction: float
    outcome: ValidationOutcome
    physics_rule: Optional[SensorPhysicsRule] = None
    spatial_agreement_map: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensor_a_id": self.sensor_a_id,
            "sensor_b_id": self.sensor_b_id,
            "pair_type": self.pair_type.value,
            "correlation": round(self.correlation, 4) if not np.isnan(self.correlation) else None,
            "bias": round(self.bias, 4) if not np.isnan(self.bias) else None,
            "rmse": round(self.rmse, 4) if not np.isnan(self.rmse) else None,
            "agreement_fraction": round(self.agreement_fraction, 4),
            "outcome": self.outcome.value,
            "physics_rule_used": self.physics_rule.description if self.physics_rule else None,
            "diagnostics": self.diagnostics,
        }


@dataclass
class CrossSensorConfig:
    """
    Configuration for cross-sensor validation.

    Attributes:
        temporal_tolerance: Max time difference for coincident observations (hours)
        spatial_tolerance: Max spatial misalignment tolerance (pixels)
        min_overlap_fraction: Minimum spatial overlap required
        use_physics_rules: Apply sensor physics rules
        check_temporal_coincidence: Validate observation timing
        tolerance_factor: Multiplier for default tolerance
    """
    temporal_tolerance: float = 24.0  # hours
    spatial_tolerance: float = 2.0    # pixels
    min_overlap_fraction: float = 0.3
    use_physics_rules: bool = True
    check_temporal_coincidence: bool = True
    tolerance_factor: float = 1.0


@dataclass
class CrossSensorResult:
    """
    Result from cross-sensor validation.

    Attributes:
        comparisons: Individual sensor comparisons
        overall_outcome: Overall validation outcome
        confidence: Overall confidence (0-1)
        sensor_rankings: Sensors ranked by agreement
        temporal_analysis: Temporal coincidence analysis
        recommendations: Suggested actions
        diagnostics: Additional diagnostic info
    """
    comparisons: List[SensorComparisonResult]
    overall_outcome: ValidationOutcome
    confidence: float
    sensor_rankings: List[Tuple[str, float]]
    temporal_analysis: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_outcome": self.overall_outcome.value,
            "confidence": round(self.confidence, 4),
            "num_comparisons": len(self.comparisons),
            "comparisons": [c.to_dict() for c in self.comparisons],
            "sensor_rankings": [
                {"sensor_id": s, "agreement_score": round(sc, 4)}
                for s, sc in self.sensor_rankings
            ],
            "temporal_analysis": self.temporal_analysis,
            "recommendations": self.recommendations,
            "diagnostics": self.diagnostics,
        }


class SensorPhysicsLibrary:
    """
    Library of expected sensor relationships based on physics.

    Encodes domain knowledge about how different sensors should agree
    when observing the same phenomenon.
    """

    def __init__(self):
        """Initialize physics rules library."""
        self.rules: List[SensorPhysicsRule] = []
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default physics rules."""
        # Water detection rules
        self.rules.extend([
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.SAR,
                observable="water_extent",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.5, 0.95),
                expected_bias=(-0.15, 0.15),
                tolerance=0.25,
                description="Optical NDWI vs SAR backscatter for water detection",
            ),
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.OPTICAL,
                observable="water_extent",
                pair_type=SensorPairType.REDUNDANT,
                expected_correlation=(0.85, 1.0),
                expected_bias=(-0.05, 0.05),
                tolerance=0.1,
                description="Inter-optical sensor water detection",
            ),
            SensorPhysicsRule(
                sensor_a=SensorType.SAR,
                sensor_b=SensorType.SAR,
                observable="water_extent",
                pair_type=SensorPairType.REDUNDANT,
                expected_correlation=(0.8, 1.0),
                expected_bias=(-0.1, 0.1),
                tolerance=0.15,
                description="Inter-SAR sensor water detection",
            ),
        ])

        # Burn severity rules
        self.rules.extend([
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.THERMAL,
                observable="burn_severity",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.6, 0.9),
                expected_bias=(-0.2, 0.2),
                tolerance=0.3,
                description="Optical dNBR vs thermal anomaly for burn detection",
            ),
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.SAR,
                observable="burn_severity",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.4, 0.8),
                expected_bias=(-0.25, 0.25),
                tolerance=0.35,
                description="Optical dNBR vs SAR coherence for burn mapping",
            ),
        ])

        # Vegetation damage rules
        self.rules.extend([
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.SAR,
                observable="vegetation_damage",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.5, 0.85),
                expected_bias=(-0.2, 0.2),
                tolerance=0.3,
                description="Optical NDVI change vs SAR backscatter change",
            ),
            SensorPhysicsRule(
                sensor_a=SensorType.OPTICAL,
                sensor_b=SensorType.LIDAR,
                observable="vegetation_damage",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.6, 0.95),
                expected_bias=(-0.15, 0.15),
                tolerance=0.2,
                description="Optical damage vs LIDAR canopy height change",
            ),
        ])

        # DEM-related rules
        self.rules.extend([
            SensorPhysicsRule(
                sensor_a=SensorType.DEM,
                sensor_b=SensorType.OPTICAL,
                observable="flood_susceptibility",
                pair_type=SensorPairType.COMPLEMENTARY,
                expected_correlation=(0.3, 0.7),
                expected_bias=(-0.3, 0.3),
                tolerance=0.4,
                description="DEM-based HAND vs optical flood extent",
            ),
        ])

    def get_rule(
        self,
        sensor_a: SensorType,
        sensor_b: SensorType,
        observable: str,
    ) -> Optional[SensorPhysicsRule]:
        """Get applicable physics rule for sensor pair and observable."""
        for rule in self.rules:
            # Check both orderings
            if (
                (rule.sensor_a == sensor_a and rule.sensor_b == sensor_b) or
                (rule.sensor_a == sensor_b and rule.sensor_b == sensor_a)
            ):
                if rule.observable == observable or observable == "unknown":
                    return rule
        return None

    def get_pair_type(
        self,
        sensor_a: SensorType,
        sensor_b: SensorType,
    ) -> SensorPairType:
        """Determine relationship type between two sensors."""
        if sensor_a == sensor_b:
            return SensorPairType.REDUNDANT

        # Check if any rule exists
        for rule in self.rules:
            if (
                (rule.sensor_a == sensor_a and rule.sensor_b == sensor_b) or
                (rule.sensor_a == sensor_b and rule.sensor_b == sensor_a)
            ):
                return rule.pair_type

        return SensorPairType.INDEPENDENT


class CrossSensorValidator:
    """
    Validates agreement between observations from different sensors.

    Compares sensor outputs based on physics-based expectations to assess:
    - Consistency across sensor types
    - Sensor-specific biases
    - Temporal coincidence
    - Spatial agreement patterns
    """

    def __init__(
        self,
        config: Optional[CrossSensorConfig] = None,
        physics_library: Optional[SensorPhysicsLibrary] = None,
    ):
        """
        Initialize cross-sensor validator.

        Args:
            config: Validation configuration
            physics_library: Library of sensor physics rules
        """
        self.config = config or CrossSensorConfig()
        self.physics = physics_library or SensorPhysicsLibrary()

    def validate(
        self,
        observations: List[SensorObservation],
        observable: str = "unknown",
    ) -> CrossSensorResult:
        """
        Validate agreement between sensor observations.

        Args:
            observations: List of sensor observations to compare
            observable: What is being observed (e.g., "water_extent")

        Returns:
            CrossSensorResult with validation details
        """
        if len(observations) < 2:
            raise ValueError("At least 2 sensor observations required")

        # Perform pairwise comparisons
        comparisons = self._compare_all_pairs(observations, observable)

        # Analyze temporal coincidence
        temporal_analysis = {}
        if self.config.check_temporal_coincidence:
            temporal_analysis = self._analyze_temporal_coincidence(observations)

        # Rank sensors by agreement
        rankings = self._rank_sensors(observations, comparisons)

        # Determine overall outcome
        overall_outcome, confidence = self._classify_overall_outcome(
            comparisons, temporal_analysis
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            comparisons, temporal_analysis, overall_outcome
        )

        return CrossSensorResult(
            comparisons=comparisons,
            overall_outcome=overall_outcome,
            confidence=confidence,
            sensor_rankings=rankings,
            temporal_analysis=temporal_analysis,
            recommendations=recommendations,
            diagnostics={
                "num_sensors": len(observations),
                "sensor_types": list(set(o.sensor_type.value for o in observations)),
                "observable": observable,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _compare_all_pairs(
        self,
        observations: List[SensorObservation],
        observable: str,
    ) -> List[SensorComparisonResult]:
        """Compare all pairs of sensor observations."""
        comparisons = []

        for i, obs_a in enumerate(observations):
            for j, obs_b in enumerate(observations):
                if j <= i:
                    continue

                comparison = self._compare_two_sensors(obs_a, obs_b, observable)
                comparisons.append(comparison)

        return comparisons

    def _compare_two_sensors(
        self,
        obs_a: SensorObservation,
        obs_b: SensorObservation,
        observable: str,
    ) -> SensorComparisonResult:
        """Compare two sensor observations."""
        # Get applicable physics rule
        physics_rule = None
        pair_type = self.physics.get_pair_type(obs_a.sensor_type, obs_b.sensor_type)

        if self.config.use_physics_rules:
            physics_rule = self.physics.get_rule(
                obs_a.sensor_type, obs_b.sensor_type, observable
            )
            if physics_rule:
                pair_type = physics_rule.pair_type

        # Prepare data for comparison
        data_a = obs_a.data.astype(np.float64)
        data_b = obs_b.data.astype(np.float64)

        # Create valid mask
        valid_mask = (obs_a.confidence > 0) & (obs_b.confidence > 0)
        if data_a.dtype.kind == 'f':
            valid_mask &= ~np.isnan(data_a)
        if data_b.dtype.kind == 'f':
            valid_mask &= ~np.isnan(data_b)

        # Check minimum overlap
        overlap_fraction = np.sum(valid_mask) / valid_mask.size
        if overlap_fraction < self.config.min_overlap_fraction:
            return SensorComparisonResult(
                sensor_a_id=obs_a.sensor_id,
                sensor_b_id=obs_b.sensor_id,
                pair_type=pair_type,
                correlation=np.nan,
                bias=np.nan,
                rmse=np.nan,
                agreement_fraction=0.0,
                outcome=ValidationOutcome.INSUFFICIENT_DATA,
                physics_rule=physics_rule,
                diagnostics={"overlap_fraction": overlap_fraction},
            )

        # Extract valid values
        a_valid = data_a[valid_mask]
        b_valid = data_b[valid_mask]

        # Calculate metrics
        correlation = self._calculate_correlation(a_valid, b_valid)
        bias = float(np.mean(a_valid - b_valid))
        rmse = float(np.sqrt(np.mean((a_valid - b_valid) ** 2)))

        # Calculate agreement fraction
        tolerance = self._get_tolerance(physics_rule)
        agreement_mask = np.abs(a_valid - b_valid) <= tolerance
        agreement_fraction = float(np.mean(agreement_mask))

        # Create spatial agreement map
        spatial_agreement = np.full_like(data_a, np.nan)
        spatial_agreement[valid_mask] = np.where(
            np.abs(data_a[valid_mask] - data_b[valid_mask]) <= tolerance,
            1.0,
            0.0
        )

        # Classify outcome
        outcome = self._classify_comparison_outcome(
            correlation, bias, rmse, agreement_fraction,
            physics_rule, pair_type
        )

        return SensorComparisonResult(
            sensor_a_id=obs_a.sensor_id,
            sensor_b_id=obs_b.sensor_id,
            pair_type=pair_type,
            correlation=correlation,
            bias=bias,
            rmse=rmse,
            agreement_fraction=agreement_fraction,
            outcome=outcome,
            physics_rule=physics_rule,
            spatial_agreement_map=spatial_agreement,
            diagnostics={
                "overlap_fraction": overlap_fraction,
                "n_valid_pixels": int(np.sum(valid_mask)),
                "tolerance_used": tolerance,
            },
        )

    def _calculate_correlation(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """Calculate correlation handling edge cases."""
        if len(a) < 2:
            return np.nan

        std_a = np.std(a)
        std_b = np.std(b)

        if std_a < 1e-10 or std_b < 1e-10:
            # Constant arrays
            return 1.0 if np.allclose(a, b) else np.nan

        return float(np.corrcoef(a, b)[0, 1])

    def _get_tolerance(
        self,
        physics_rule: Optional[SensorPhysicsRule],
    ) -> float:
        """Get comparison tolerance."""
        base_tolerance = 0.1  # default

        if physics_rule:
            base_tolerance = physics_rule.tolerance

        return base_tolerance * self.config.tolerance_factor

    def _classify_comparison_outcome(
        self,
        correlation: float,
        bias: float,
        rmse: float,
        agreement_fraction: float,
        physics_rule: Optional[SensorPhysicsRule],
        pair_type: SensorPairType,
    ) -> ValidationOutcome:
        """Classify the outcome of a sensor comparison."""
        # Check against physics rule if available
        if physics_rule:
            corr_min, corr_max = physics_rule.expected_correlation
            bias_min, bias_max = physics_rule.expected_bias

            # Check if correlation within expected range
            corr_ok = np.isnan(correlation) or (corr_min <= correlation <= corr_max)

            # Check if bias within expected range
            bias_ok = np.isnan(bias) or (bias_min <= bias <= bias_max)

            if agreement_fraction >= 0.9 and corr_ok and bias_ok:
                return ValidationOutcome.CONSISTENT
            elif agreement_fraction >= 0.7 and (corr_ok or bias_ok):
                return ValidationOutcome.MINOR_DISCREPANCY
            elif agreement_fraction >= 0.5:
                return ValidationOutcome.SIGNIFICANT_DISCREPANCY
            else:
                return ValidationOutcome.CONFLICT

        # Default classification without physics rule
        if pair_type == SensorPairType.REDUNDANT:
            # Stricter thresholds for redundant sensors
            if agreement_fraction >= 0.9:
                return ValidationOutcome.CONSISTENT
            elif agreement_fraction >= 0.75:
                return ValidationOutcome.MINOR_DISCREPANCY
            elif agreement_fraction >= 0.5:
                return ValidationOutcome.SIGNIFICANT_DISCREPANCY
            else:
                return ValidationOutcome.CONFLICT
        else:
            # More lenient for complementary/independent
            if agreement_fraction >= 0.8:
                return ValidationOutcome.CONSISTENT
            elif agreement_fraction >= 0.6:
                return ValidationOutcome.MINOR_DISCREPANCY
            elif agreement_fraction >= 0.4:
                return ValidationOutcome.SIGNIFICANT_DISCREPANCY
            else:
                return ValidationOutcome.CONFLICT

    def _analyze_temporal_coincidence(
        self,
        observations: List[SensorObservation],
    ) -> Dict[str, Any]:
        """Analyze temporal coincidence of observations."""
        timestamps = []
        for obs in observations:
            if obs.timestamp:
                timestamps.append((obs.sensor_id, obs.timestamp))

        if not timestamps:
            return {"has_timestamps": False}

        # Sort by time
        timestamps.sort(key=lambda x: x[1])

        # Calculate time span
        earliest = timestamps[0][1]
        latest = timestamps[-1][1]
        time_span = (latest - earliest).total_seconds() / 3600  # hours

        # Check pairwise time differences
        time_differences = []
        for i, (id_a, ts_a) in enumerate(timestamps):
            for j, (id_b, ts_b) in enumerate(timestamps):
                if j <= i:
                    continue
                diff = abs((ts_b - ts_a).total_seconds() / 3600)
                time_differences.append({
                    "sensor_a": id_a,
                    "sensor_b": id_b,
                    "hours_apart": round(diff, 2),
                    "within_tolerance": diff <= self.config.temporal_tolerance,
                })

        # Check if all within tolerance
        all_coincident = all(td["within_tolerance"] for td in time_differences)

        return {
            "has_timestamps": True,
            "num_observations": len(timestamps),
            "time_span_hours": round(time_span, 2),
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
            "all_temporally_coincident": all_coincident,
            "time_differences": time_differences,
        }

    def _rank_sensors(
        self,
        observations: List[SensorObservation],
        comparisons: List[SensorComparisonResult],
    ) -> List[Tuple[str, float]]:
        """Rank sensors by their agreement with other sensors."""
        scores = {}

        for obs in observations:
            total_agreement = 0.0
            count = 0

            for comp in comparisons:
                if comp.outcome == ValidationOutcome.INSUFFICIENT_DATA:
                    continue

                if comp.sensor_a_id == obs.sensor_id:
                    total_agreement += comp.agreement_fraction
                    count += 1
                elif comp.sensor_b_id == obs.sensor_id:
                    total_agreement += comp.agreement_fraction
                    count += 1

            if count > 0:
                scores[obs.sensor_id] = total_agreement / count
            else:
                scores[obs.sensor_id] = 0.0

        # Sort descending
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def _classify_overall_outcome(
        self,
        comparisons: List[SensorComparisonResult],
        temporal_analysis: Dict[str, Any],
    ) -> Tuple[ValidationOutcome, float]:
        """Classify overall validation outcome and confidence."""
        if not comparisons:
            return ValidationOutcome.INSUFFICIENT_DATA, 0.0

        # Filter out insufficient data
        valid_comparisons = [
            c for c in comparisons
            if c.outcome != ValidationOutcome.INSUFFICIENT_DATA
        ]

        if not valid_comparisons:
            return ValidationOutcome.INSUFFICIENT_DATA, 0.0

        # Count outcomes
        consistent = sum(1 for c in valid_comparisons if c.outcome == ValidationOutcome.CONSISTENT)
        minor = sum(1 for c in valid_comparisons if c.outcome == ValidationOutcome.MINOR_DISCREPANCY)
        significant = sum(1 for c in valid_comparisons if c.outcome == ValidationOutcome.SIGNIFICANT_DISCREPANCY)
        conflict = sum(1 for c in valid_comparisons if c.outcome == ValidationOutcome.CONFLICT)

        total = len(valid_comparisons)

        # Calculate confidence
        mean_agreement = np.mean([c.agreement_fraction for c in valid_comparisons])
        confidence = float(mean_agreement)

        # Apply temporal penalty if not coincident
        if not temporal_analysis.get("all_temporally_coincident", True):
            confidence *= 0.8  # 20% penalty for temporal mismatch

        # Classify
        if consistent / total >= 0.8:
            return ValidationOutcome.CONSISTENT, confidence
        elif (consistent + minor) / total >= 0.7:
            return ValidationOutcome.MINOR_DISCREPANCY, confidence * 0.9
        elif conflict / total >= 0.5:
            return ValidationOutcome.CONFLICT, confidence * 0.5
        else:
            return ValidationOutcome.SIGNIFICANT_DISCREPANCY, confidence * 0.7

    def _generate_recommendations(
        self,
        comparisons: List[SensorComparisonResult],
        temporal_analysis: Dict[str, Any],
        overall_outcome: ValidationOutcome,
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Temporal recommendations
        if temporal_analysis.get("has_timestamps"):
            if not temporal_analysis.get("all_temporally_coincident", True):
                recommendations.append(
                    f"Observations span {temporal_analysis.get('time_span_hours', 0):.1f} hours. "
                    "Consider limiting temporal window for more reliable comparison."
                )

        # Conflict recommendations
        conflict_pairs = [
            c for c in comparisons
            if c.outcome == ValidationOutcome.CONFLICT
        ]
        if conflict_pairs:
            for c in conflict_pairs:
                recommendations.append(
                    f"Conflict between {c.sensor_a_id} and {c.sensor_b_id}: "
                    f"agreement only {c.agreement_fraction:.1%}. "
                    "Manual review recommended."
                )

        # Bias recommendations
        biased_pairs = [
            c for c in comparisons
            if not np.isnan(c.bias) and abs(c.bias) > 0.2
        ]
        if biased_pairs:
            for c in biased_pairs:
                direction = "higher" if c.bias > 0 else "lower"
                recommendations.append(
                    f"Systematic bias detected: {c.sensor_a_id} reads {direction} "
                    f"than {c.sensor_b_id} by {abs(c.bias):.2f}. "
                    "Consider calibration adjustment."
                )

        # Overall recommendations
        if overall_outcome == ValidationOutcome.CONFLICT:
            recommendations.append(
                "Multiple sensors show fundamental disagreement. "
                "Results should be flagged for expert review."
            )
        elif overall_outcome == ValidationOutcome.SIGNIFICANT_DISCREPANCY:
            recommendations.append(
                "Significant discrepancies detected. "
                "Consider ensemble approach with increased uncertainty."
            )

        return recommendations


# Convenience functions

def validate_cross_sensor(
    sensor_data: List[np.ndarray],
    sensor_types: List[str],
    sensor_ids: Optional[List[str]] = None,
    observable: str = "unknown",
) -> CrossSensorResult:
    """
    Validate agreement between sensor observations.

    Args:
        sensor_data: List of sensor data arrays
        sensor_types: List of sensor types (optical, sar, thermal, etc.)
        sensor_ids: Optional list of sensor identifiers
        observable: What is being observed

    Returns:
        CrossSensorResult with validation details
    """
    if sensor_ids is None:
        sensor_ids = [f"sensor_{i}" for i in range(len(sensor_data))]

    # Map sensor types
    type_map = {
        "optical": SensorType.OPTICAL,
        "sar": SensorType.SAR,
        "thermal": SensorType.THERMAL,
        "lidar": SensorType.LIDAR,
        "dem": SensorType.DEM,
        "weather": SensorType.WEATHER,
        "ancillary": SensorType.ANCILLARY,
    }

    observations = []
    for data, stype, sid in zip(sensor_data, sensor_types, sensor_ids):
        observations.append(SensorObservation(
            data=data,
            sensor_type=type_map.get(stype.lower(), SensorType.ANCILLARY),
            sensor_id=sid,
            observable=observable,
        ))

    validator = CrossSensorValidator()
    return validator.validate(observations, observable)


def compare_optical_sar(
    optical_data: np.ndarray,
    sar_data: np.ndarray,
    observable: str = "water_extent",
    optical_id: str = "optical",
    sar_id: str = "sar",
) -> SensorComparisonResult:
    """
    Compare optical and SAR sensor outputs.

    Args:
        optical_data: Optical sensor data array
        sar_data: SAR sensor data array
        observable: What is being observed
        optical_id: Identifier for optical sensor
        sar_id: Identifier for SAR sensor

    Returns:
        SensorComparisonResult with comparison metrics
    """
    obs_optical = SensorObservation(
        data=optical_data,
        sensor_type=SensorType.OPTICAL,
        sensor_id=optical_id,
        observable=observable,
    )
    obs_sar = SensorObservation(
        data=sar_data,
        sensor_type=SensorType.SAR,
        sensor_id=sar_id,
        observable=observable,
    )

    validator = CrossSensorValidator()
    return validator._compare_two_sensors(obs_optical, obs_sar, observable)
