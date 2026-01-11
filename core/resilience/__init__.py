"""
Resilience module for data quality assessment and fallback handling.

This module provides comprehensive data quality assessment and automatic
fallback mechanisms for sensor and algorithm failures:

- Assessment: Quality scoring for optical, SAR, DEM, and temporal data
- Fallbacks: Sensor and algorithm fallback chains with decision logging
- Degraded Mode: Mode management, partial coverage, low confidence handling
- Failure Handling: Failure logging, recovery strategies, provenance tracking

Typical usage:
    from core.resilience import (
        # Assessment
        OpticalQualityAssessor,
        SARQualityAssessor,
        DEMQualityAssessor,
        TemporalQualityAssessor,
        # Fallbacks
        OpticalFallbackChain,
        SARFallbackChain,
        DEMFallbackChain,
        AlgorithmFallbackManager,
        AdaptiveParameterTuner,
        # Degraded mode
        DegradedModeLevel,
        DegradedModeManager,
        PartialCoverageHandler,
        LowConfidenceHandler,
        # Failure handling
        FailureLogger,
        RecoveryOrchestrator,
        FallbackProvenanceTracker,
    )

    # Assess data quality
    optical_assessor = OpticalQualityAssessor()
    quality = optical_assessor.assess(image_data, metadata)

    # Use fallback chains
    optical_chain = OpticalFallbackChain()
    result = optical_chain.get_best_available(aoi, time_range, cloud_cover=85)

    # Manage degraded mode
    manager = DegradedModeManager()
    state = manager.assess_situation(cloud_cover=85.0)
"""

from core.resilience.assessment import (
    # Optical quality
    OpticalQualityAssessor,
    OpticalQualityConfig,
    OpticalQualityResult,
    CloudDetectionMethod,
    # SAR quality
    SARQualityAssessor,
    SARQualityConfig,
    SARQualityResult,
    SpeckleFilterType,
    # DEM quality
    DEMQualityAssessor,
    DEMQualityConfig,
    DEMQualityResult,
    DEMArtifactType,
    # Temporal quality
    TemporalQualityAssessor,
    TemporalQualityConfig,
    TemporalQualityResult,
)

from core.resilience.fallbacks import (
    # Optical fallback
    OpticalFallbackChain,
    OpticalFallbackConfig,
    OpticalFallbackResult,
    # SAR fallback
    SARFallbackChain,
    SARFallbackConfig,
    SARFallbackResult,
    # DEM fallback
    DEMFallbackChain,
    DEMFallbackConfig,
    DEMFallbackResult,
    # Algorithm fallback
    AlgorithmFallbackManager,
    AlgorithmFallbackConfig,
    AlgorithmFallbackResult,
    FallbackTrigger,
    # Parameter tuning
    AdaptiveParameterTuner,
    ParameterTuningConfig,
    ParameterTuningResult,
    TuningMethod,
)

from core.resilience.degraded_mode import (
    # Mode manager
    DegradedModeLevel,
    DegradedModeTrigger,
    DegradedModeState,
    ModeTransition,
    DegradedModeConfig,
    DegradedModeManager,
    assess_degraded_mode,
    # Partial coverage
    CoverageGap,
    CoverageRegion,
    PartialAcquisition,
    CoverageMosaicResult,
    PartialCoverageConfig,
    PartialCoverageHandler,
    mosaic_partial_coverage,
    # Low confidence
    AlgorithmResult,
    VotingMethod,
    EnsembleResult,
    DisagreementRegion,
    LowConfidenceConfig,
    LowConfidenceHandler,
    ensemble_voting,
)

from core.resilience.failure import (
    # Failure logging
    FailureComponent,
    FailureSeverity,
    FailureLog,
    FailureQuery,
    FailureStats,
    FailureLogger,
    log_failure,
    # Recovery strategies
    RecoveryStrategy,
    RecoveryAttempt,
    RecoveryResult,
    RecoveryConfig,
    RecoveryOrchestrator,
    retry_with_backoff,
    # Provenance tracking
    FallbackDecision,
    FallbackChain,
    ProvenanceRecord,
    FallbackProvenanceConfig,
    FallbackProvenanceTracker,
    create_provenance_record,
)

__all__ = [
    # Assessment - Optical
    "OpticalQualityAssessor",
    "OpticalQualityConfig",
    "OpticalQualityResult",
    "CloudDetectionMethod",
    # Assessment - SAR
    "SARQualityAssessor",
    "SARQualityConfig",
    "SARQualityResult",
    "SpeckleFilterType",
    # Assessment - DEM
    "DEMQualityAssessor",
    "DEMQualityConfig",
    "DEMQualityResult",
    "DEMArtifactType",
    # Assessment - Temporal
    "TemporalQualityAssessor",
    "TemporalQualityConfig",
    "TemporalQualityResult",
    # Fallbacks - Optical
    "OpticalFallbackChain",
    "OpticalFallbackConfig",
    "OpticalFallbackResult",
    # Fallbacks - SAR
    "SARFallbackChain",
    "SARFallbackConfig",
    "SARFallbackResult",
    # Fallbacks - DEM
    "DEMFallbackChain",
    "DEMFallbackConfig",
    "DEMFallbackResult",
    # Fallbacks - Algorithm
    "AlgorithmFallbackManager",
    "AlgorithmFallbackConfig",
    "AlgorithmFallbackResult",
    "FallbackTrigger",
    # Parameter tuning
    "AdaptiveParameterTuner",
    "ParameterTuningConfig",
    "ParameterTuningResult",
    "TuningMethod",
    # Degraded mode - Mode manager
    "DegradedModeLevel",
    "DegradedModeTrigger",
    "DegradedModeState",
    "ModeTransition",
    "DegradedModeConfig",
    "DegradedModeManager",
    "assess_degraded_mode",
    # Degraded mode - Partial coverage
    "CoverageGap",
    "CoverageRegion",
    "PartialAcquisition",
    "CoverageMosaicResult",
    "PartialCoverageConfig",
    "PartialCoverageHandler",
    "mosaic_partial_coverage",
    # Degraded mode - Low confidence
    "AlgorithmResult",
    "VotingMethod",
    "EnsembleResult",
    "DisagreementRegion",
    "LowConfidenceConfig",
    "LowConfidenceHandler",
    "ensemble_voting",
    # Failure - Logging
    "FailureComponent",
    "FailureSeverity",
    "FailureLog",
    "FailureQuery",
    "FailureStats",
    "FailureLogger",
    "log_failure",
    # Failure - Recovery strategies
    "RecoveryStrategy",
    "RecoveryAttempt",
    "RecoveryResult",
    "RecoveryConfig",
    "RecoveryOrchestrator",
    "retry_with_backoff",
    # Failure - Provenance tracking
    "FallbackDecision",
    "FallbackChain",
    "ProvenanceRecord",
    "FallbackProvenanceConfig",
    "FallbackProvenanceTracker",
    "create_provenance_record",
]
