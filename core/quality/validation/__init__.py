"""
Validation Module for Quality Control.

Provides comprehensive validation tools for assessing analysis output quality
through multiple validation strategies:

- **Cross-Model Validation**: Compare results from different algorithms
- **Cross-Sensor Validation**: Compare results from different sensors
- **Historical Validation**: Compare against historical baselines
- **Consensus Generation**: Combine multiple sources into authoritative output

Example:
    from core.quality.validation import (
        CrossModelValidator,
        CrossSensorValidator,
        HistoricalValidator,
        ConsensusGenerator,
    )

    # Cross-model validation
    model_validator = CrossModelValidator()
    result = model_validator.validate(model_outputs)

    # Cross-sensor validation
    sensor_validator = CrossSensorValidator()
    result = sensor_validator.validate(sensor_observations, observable="water_extent")

    # Historical validation
    hist_validator = HistoricalValidator()
    result = hist_validator.validate(current_data, baseline)

    # Consensus generation
    generator = ConsensusGenerator()
    result = generator.generate(sources)
"""

# Cross-model validation
from .cross_model import (
    AgreementLevel,
    AgreementMap,
    ComparisonMetric,
    CrossModelConfig,
    CrossModelResult,
    CrossModelValidator,
    ModelOutput,
    PairwiseComparison,
    compare_two_models,
    get_ensemble_consensus,
    validate_cross_model,
)

# Cross-sensor validation
from .cross_sensor import (
    CrossSensorConfig,
    CrossSensorResult,
    CrossSensorValidator,
    SensorComparisonResult,
    SensorObservation,
    SensorPairType,
    SensorPhysicsLibrary,
    SensorPhysicsRule,
    SensorType,
    ValidationOutcome,
    compare_optical_sar,
    validate_cross_sensor,
)

# Historical baseline validation
from .historical import (
    AnomalySeverity,
    AnomalyType,
    HistoricalAnomaly,
    HistoricalBaseline,
    HistoricalConfig,
    HistoricalMetric,
    HistoricalResult,
    HistoricalValidator,
    calculate_z_score,
    estimate_return_period,
    validate_against_historical,
)

# Consensus generation
from .consensus import (
    ConsensusConfig,
    ConsensusGenerator,
    ConsensusPriority,
    ConsensusQuality,
    ConsensusResult,
    ConsensusSource,
    ConsensusStrategy,
    DisagreementRegion,
    generate_consensus,
    vote_consensus,
    weighted_mean_consensus,
)

__all__ = [
    # Cross-model validation
    "AgreementLevel",
    "AgreementMap",
    "ComparisonMetric",
    "CrossModelConfig",
    "CrossModelResult",
    "CrossModelValidator",
    "ModelOutput",
    "PairwiseComparison",
    "compare_two_models",
    "get_ensemble_consensus",
    "validate_cross_model",
    # Cross-sensor validation
    "CrossSensorConfig",
    "CrossSensorResult",
    "CrossSensorValidator",
    "SensorComparisonResult",
    "SensorObservation",
    "SensorPairType",
    "SensorPhysicsLibrary",
    "SensorPhysicsRule",
    "SensorType",
    "ValidationOutcome",
    "compare_optical_sar",
    "validate_cross_sensor",
    # Historical validation
    "AnomalySeverity",
    "AnomalyType",
    "HistoricalAnomaly",
    "HistoricalBaseline",
    "HistoricalConfig",
    "HistoricalMetric",
    "HistoricalResult",
    "HistoricalValidator",
    "calculate_z_score",
    "estimate_return_period",
    "validate_against_historical",
    # Consensus generation
    "ConsensusConfig",
    "ConsensusGenerator",
    "ConsensusPriority",
    "ConsensusQuality",
    "ConsensusResult",
    "ConsensusSource",
    "ConsensusStrategy",
    "DisagreementRegion",
    "generate_consensus",
    "vote_consensus",
    "weighted_mean_consensus",
]
