"""
Comprehensive Quality Control Integration Tests (Group I, Track 6).

End-to-end tests that validate the complete quality control pipeline:
- Sanity checks (spatial, value, temporal, artifacts)
- Uncertainty quantification

These tests ensure all quality modules work correctly together as a unified system.
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

import numpy as np

# Mark all tests in this module
pytestmark = pytest.mark.quality


# ============================================================================
# EDGE CASE PIPELINE TESTS
# ============================================================================

class TestPipelineEdgeCases:
    """Edge case tests for quality pipeline."""

    def test_empty_data_pipeline(self):
        """Test pipeline handles empty data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import UncertaintyQuantifier

        data = np.array([]).reshape(0, 0)

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Should handle gracefully
        assert result is not None

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.n_samples == 0

    def test_constant_data_pipeline(self):
        """Test pipeline with constant data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
        )

        data = np.full((50, 50), 0.5)

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Constant data should pass value checks (use .values not .value)
        assert result.values.is_plausible

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.std == 0.0
        assert metrics.coefficient_of_variation == 0.0

        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)
        # Constant data has zero local variance
        assert np.allclose(surface.uncertainty, 0, atol=1e-10) or np.all(np.isnan(surface.uncertainty))

    def test_single_pixel_pipeline(self):
        """Test pipeline with single pixel data."""
        from core.quality.sanity import (
            check_spatial_coherence,
            check_value_plausibility,
        )
        from core.quality.uncertainty import UncertaintyQuantifier

        data = np.array([[0.5]])

        spatial = check_spatial_coherence(data)
        assert spatial is not None

        values = check_value_plausibility(data)
        assert values is not None

        quantifier = UncertaintyQuantifier()
        metrics = quantifier.calculate_metrics(data.flatten())
        assert metrics.n_samples == 1

    def test_mixed_valid_invalid_data(self):
        """Test pipeline with mixed valid and invalid data."""
        from core.quality.sanity import SanitySuite, SanitySuiteConfig
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
        )

        np.random.seed(42)
        data = np.random.rand(100, 100)
        data[0:20, :] = np.nan  # 20% NaN
        data[80:100, :] = np.inf  # 20% Inf

        suite = SanitySuite(SanitySuiteConfig(run_temporal=False))
        result = suite.check(data)
        # Should detect issues (use .values not .value)
        assert len(result.values.issues) > 0

        # Uncertainty should handle NaN/Inf
        quantifier = UncertaintyQuantifier()
        valid_data = data[np.isfinite(data)]
        metrics = quantifier.calculate_metrics(valid_data)
        assert metrics.n_samples < 10000  # Less than full array

        # Spatial uncertainty should handle
        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)
        assert surface.uncertainty.shape == data.shape


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================


class TestModuleImports:
    """Tests to verify all module imports work correctly."""

    def test_quality_module_public_api(self):
        """Test that core.quality exposes expected API."""
        import core.quality as quality

        assert hasattr(quality, 'sanity')
        assert hasattr(quality, 'uncertainty')

    def test_sanity_module_imports(self):
        """Test sanity module imports."""
        from core.quality.sanity import (
            SanitySuite,
            SanitySuiteConfig,
            SanitySuiteResult,
            # Spatial
            SpatialCoherenceChecker,
            check_spatial_coherence,
            # Values
            ValuePlausibilityChecker,
            check_value_plausibility,
            ValueType,
            # Temporal
            TemporalConsistencyChecker,
            check_temporal_consistency,
            # Artifacts
            ArtifactDetector,
            detect_artifacts,
        )
        assert SanitySuite is not None

    def test_uncertainty_module_imports(self):
        """Test uncertainty module imports."""
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
            HotspotDetector,
            QualityErrorPropagator,
            CalibrationAssessor,
            SensitivityAnalyzer,
            # Convenience functions
            propagate_quality_uncertainty,
            calculate_confidence_interval,
            calculate_prediction_interval,
            combine_independent_uncertainties,
            threshold_exceedance_probability,
        )
        assert UncertaintyQuantifier is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
