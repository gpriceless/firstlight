"""
Uncertainty Quantification Module for Quality Control.

Provides comprehensive tools for measuring, mapping, and propagating
uncertainty in quality control pipelines.

Submodules:
- quantification: Uncertainty metrics and calibration assessment
- spatial_uncertainty: Spatial uncertainty mapping and hotspot detection
- propagation: Error propagation through QC operations

This module focuses on quality assessment uncertainty (Track 3 of Group I),
complementing the fusion uncertainty in core.analysis.fusion.uncertainty.

Example Usage:
    from core.quality.uncertainty import (
        # Metrics calculation
        UncertaintyQuantifier,
        calculate_confidence_interval,
        calculate_prediction_interval,
        assess_calibration,

        # Spatial mapping
        SpatialUncertaintyMapper,
        HotspotDetector,
        compute_local_uncertainty,
        detect_uncertainty_hotspots,

        # Error propagation
        QualityErrorPropagator,
        propagate_quality_uncertainty,
        compute_error_budget,
    )

    # Calculate uncertainty metrics
    quantifier = UncertaintyQuantifier()
    metrics = quantifier.calculate_metrics(data)

    # Map spatial uncertainty
    mapper = SpatialUncertaintyMapper()
    surface = mapper.compute_uncertainty_surface(data)

    # Detect hotspots
    hotspots = detect_uncertainty_hotspots(surface.uncertainty)

    # Propagate through aggregation
    result = propagate_quality_uncertainty(
        values=[0.8, 0.9, 0.85],
        uncertainties=[0.05, 0.03, 0.04],
        method="mean"
    )
"""

# Quantification module exports
from core.quality.uncertainty.quantification import (
    # Enums
    UncertaintyMetricType,
    CalibrationMethod,
    ConfidenceLevel,
    # Config dataclasses
    QuantificationConfig,
    # Result dataclasses
    UncertaintyMetrics,
    EnsembleUncertainty,
    CalibrationResult,
    # Core classes
    UncertaintyQuantifier,
    CalibrationAssessor,
    # Convenience functions
    calculate_confidence_interval,
    calculate_prediction_interval,
    calculate_coefficient_of_variation,
    assess_calibration,
    quantify_ensemble_uncertainty,
)

# Spatial uncertainty module exports
from core.quality.uncertainty.spatial_uncertainty import (
    # Enums
    SpatialUncertaintyMethod,
    SmoothingMethod,
    HotspotMethod,
    # Config dataclasses
    SpatialUncertaintyConfig,
    # Result dataclasses
    UncertaintySurface,
    UncertaintyHotspot,
    HotspotAnalysis,
    LocalStatistics,
    # Core classes
    SpatialUncertaintyMapper,
    HotspotDetector,
    SpatialUncertaintyAnalyzer,
    # Convenience functions
    compute_local_uncertainty,
    compute_ensemble_uncertainty,
    detect_uncertainty_hotspots,
    compute_spatial_autocorrelation,
)

# Propagation module exports
from core.quality.uncertainty.propagation import (
    # Enums
    PropagationMethod,
    AggregationMethod,
    CorrelationType,
    # Config dataclasses
    PropagationConfig,
    # Result dataclasses
    UncertaintySource,
    ErrorBudget,
    PropagationResult,
    SensitivityResult,
    # Core classes
    QualityErrorPropagator,
    SensitivityAnalyzer,
    # Convenience functions
    propagate_quality_uncertainty,
    compute_error_budget,
    combine_independent_uncertainties,
    combine_correlated_uncertainties,
    threshold_exceedance_probability,
)

__all__ = [
    # Quantification - Enums
    "UncertaintyMetricType",
    "CalibrationMethod",
    "ConfidenceLevel",
    # Quantification - Config
    "QuantificationConfig",
    # Quantification - Results
    "UncertaintyMetrics",
    "EnsembleUncertainty",
    "CalibrationResult",
    # Quantification - Classes
    "UncertaintyQuantifier",
    "CalibrationAssessor",
    # Quantification - Functions
    "calculate_confidence_interval",
    "calculate_prediction_interval",
    "calculate_coefficient_of_variation",
    "assess_calibration",
    "quantify_ensemble_uncertainty",
    # Spatial - Enums
    "SpatialUncertaintyMethod",
    "SmoothingMethod",
    "HotspotMethod",
    # Spatial - Config
    "SpatialUncertaintyConfig",
    # Spatial - Results
    "UncertaintySurface",
    "UncertaintyHotspot",
    "HotspotAnalysis",
    "LocalStatistics",
    # Spatial - Classes
    "SpatialUncertaintyMapper",
    "HotspotDetector",
    "SpatialUncertaintyAnalyzer",
    # Spatial - Functions
    "compute_local_uncertainty",
    "compute_ensemble_uncertainty",
    "detect_uncertainty_hotspots",
    "compute_spatial_autocorrelation",
    # Propagation - Enums
    "PropagationMethod",
    "AggregationMethod",
    "CorrelationType",
    # Propagation - Config
    "PropagationConfig",
    # Propagation - Results
    "UncertaintySource",
    "ErrorBudget",
    "PropagationResult",
    "SensitivityResult",
    # Propagation - Classes
    "QualityErrorPropagator",
    "SensitivityAnalyzer",
    # Propagation - Functions
    "propagate_quality_uncertainty",
    "compute_error_budget",
    "combine_independent_uncertainties",
    "combine_correlated_uncertainties",
    "threshold_exceedance_probability",
]
