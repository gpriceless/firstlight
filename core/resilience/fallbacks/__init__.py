"""
Fallback Chains Module.

Provides intelligent fallback mechanisms for sensors and algorithms:
- Optical fallback: Sentinel-2 -> Landsat-8 -> MODIS
- SAR fallback: Enhanced filtering, different orbits, band switching
- DEM fallback: Copernicus -> SRTM -> ASTER with void filling
- Algorithm fallback: Alternative algorithms and parameter tuning

Each fallback chain implements decision logic with comprehensive logging.
"""

from core.resilience.fallbacks.optical_fallback import (
    OpticalFallbackChain,
    OpticalFallbackConfig,
    OpticalFallbackResult,
    OpticalFallbackDecision,
    OpticalSource,
)

from core.resilience.fallbacks.sar_fallback import (
    SARFallbackChain,
    SARFallbackConfig,
    SARFallbackResult,
    SARFallbackDecision,
    SARBand,
    FilterStrategy,
)

from core.resilience.fallbacks.dem_fallback import (
    DEMFallbackChain,
    DEMFallbackConfig,
    DEMFallbackResult,
    DEMFallbackDecision,
    DEMSource,
    VoidFillingStrategy,
)

from core.resilience.fallbacks.algorithm_fallback import (
    AlgorithmFallbackManager,
    AlgorithmFallbackConfig,
    AlgorithmFallbackResult,
    AlgorithmFallbackDecision,
    FallbackTrigger,
    AlternativeAlgorithm,
)

from core.resilience.fallbacks.parameter_tuning import (
    AdaptiveParameterTuner,
    ParameterTuningConfig,
    ParameterTuningResult,
    TuningMethod,
    ParameterSpace,
    TuningHistory,
)

__all__ = [
    # Optical fallback
    "OpticalFallbackChain",
    "OpticalFallbackConfig",
    "OpticalFallbackResult",
    "OpticalFallbackDecision",
    "OpticalSource",
    # SAR fallback
    "SARFallbackChain",
    "SARFallbackConfig",
    "SARFallbackResult",
    "SARFallbackDecision",
    "SARBand",
    "FilterStrategy",
    # DEM fallback
    "DEMFallbackChain",
    "DEMFallbackConfig",
    "DEMFallbackResult",
    "DEMFallbackDecision",
    "DEMSource",
    "VoidFillingStrategy",
    # Algorithm fallback
    "AlgorithmFallbackManager",
    "AlgorithmFallbackConfig",
    "AlgorithmFallbackResult",
    "AlgorithmFallbackDecision",
    "FallbackTrigger",
    "AlternativeAlgorithm",
    # Parameter tuning
    "AdaptiveParameterTuner",
    "ParameterTuningConfig",
    "ParameterTuningResult",
    "TuningMethod",
    "ParameterSpace",
    "TuningHistory",
]
