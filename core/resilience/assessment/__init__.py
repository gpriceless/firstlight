"""
Data Quality Assessment Module.

Provides comprehensive quality assessment for different data types:
- Optical imagery: Cloud cover, shadows, haze, sun glint, saturation
- SAR imagery: Speckle noise, geometric distortion, radiometric calibration
- DEM data: Voids, artifacts, resolution adequacy, vertical accuracy
- Temporal data: Baseline availability, temporal gaps, acquisition timing

Each assessor returns a quality score (0-1) along with detailed metrics.
"""

from core.resilience.assessment.optical_quality import (
    OpticalQualityAssessor,
    OpticalQualityConfig,
    OpticalQualityResult,
    OpticalQualityIssue,
    CloudDetectionMethod,
    assess_optical_quality,
)

from core.resilience.assessment.sar_quality import (
    SARQualityAssessor,
    SARQualityConfig,
    SARQualityResult,
    SARQualityIssue,
    SpeckleFilterType,
    assess_sar_quality,
)

from core.resilience.assessment.dem_quality import (
    DEMQualityAssessor,
    DEMQualityConfig,
    DEMQualityResult,
    DEMQualityIssue,
    DEMArtifactType,
    assess_dem_quality,
)

from core.resilience.assessment.temporal_quality import (
    TemporalQualityAssessor,
    TemporalQualityConfig,
    TemporalQualityResult,
    TemporalQualityIssue,
    assess_temporal_quality,
)

__all__ = [
    # Optical
    "OpticalQualityAssessor",
    "OpticalQualityConfig",
    "OpticalQualityResult",
    "OpticalQualityIssue",
    "CloudDetectionMethod",
    "assess_optical_quality",
    # SAR
    "SARQualityAssessor",
    "SARQualityConfig",
    "SARQualityResult",
    "SARQualityIssue",
    "SpeckleFilterType",
    "assess_sar_quality",
    # DEM
    "DEMQualityAssessor",
    "DEMQualityConfig",
    "DEMQualityResult",
    "DEMQualityIssue",
    "DEMArtifactType",
    "assess_dem_quality",
    # Temporal
    "TemporalQualityAssessor",
    "TemporalQualityConfig",
    "TemporalQualityResult",
    "TemporalQualityIssue",
    "assess_temporal_quality",
]
