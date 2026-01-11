"""
Degraded Mode Operations Module.

Provides graceful degradation capabilities when ideal data or processing
conditions are unavailable:

- Mode Management: Track and switch between operational modes based on data quality
- Partial Coverage: Handle incomplete spatial coverage with mosaics and interpolation
- Low Confidence: Ensemble methods and voting when single algorithm confidence is weak

Degraded Mode Levels:
- FULL: All data available, high confidence results
- PARTIAL: Some data missing, medium confidence with uncertainty markers
- MINIMAL: Significant gaps, low confidence requiring careful interpretation
- EMERGENCY: Bare minimum data, very low confidence, flagged for manual review

Example:
    from core.resilience.degraded_mode import (
        DegradedModeLevel,
        DegradedModeManager,
        PartialCoverageHandler,
        LowConfidenceHandler,
    )

    # Manage degraded modes
    manager = DegradedModeManager()
    mode = manager.assess_situation(data_availability)
    if mode.level != DegradedModeLevel.FULL:
        manager.notify_mode_change(mode)

    # Handle partial coverage
    coverage_handler = PartialCoverageHandler()
    result = coverage_handler.mosaic_partial_acquisitions(acquisitions)

    # Handle low confidence
    confidence_handler = LowConfidenceHandler()
    result = confidence_handler.ensemble_voting(algorithm_outputs)
"""

from core.resilience.degraded_mode.mode_manager import (
    DegradedModeLevel,
    DegradedModeTrigger,
    DegradedModeState,
    ModeTransition,
    DegradedModeConfig,
    DegradedModeManager,
    assess_degraded_mode,
)

from core.resilience.degraded_mode.partial_coverage import (
    CoverageGap,
    CoverageRegion,
    PartialAcquisition,
    CoverageMosaicResult,
    PartialCoverageConfig,
    PartialCoverageHandler,
    mosaic_partial_coverage,
)

from core.resilience.degraded_mode.low_confidence import (
    AlgorithmResult,
    VotingMethod,
    EnsembleResult,
    DisagreementRegion,
    LowConfidenceConfig,
    LowConfidenceHandler,
    ensemble_voting,
)

__all__ = [
    # Mode manager
    "DegradedModeLevel",
    "DegradedModeTrigger",
    "DegradedModeState",
    "ModeTransition",
    "DegradedModeConfig",
    "DegradedModeManager",
    "assess_degraded_mode",
    # Partial coverage
    "CoverageGap",
    "CoverageRegion",
    "PartialAcquisition",
    "CoverageMosaicResult",
    "PartialCoverageConfig",
    "PartialCoverageHandler",
    "mosaic_partial_coverage",
    # Low confidence
    "AlgorithmResult",
    "VotingMethod",
    "EnsembleResult",
    "DisagreementRegion",
    "LowConfidenceConfig",
    "LowConfidenceHandler",
    "ensemble_voting",
]
