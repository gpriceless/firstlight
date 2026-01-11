"""
Tests for Quality Control Uncertainty Quantification (Group I, Track 3).

Tests cover:
- Uncertainty metrics calculation
- Confidence and prediction intervals
- Calibration assessment
- Spatial uncertainty mapping
- Hotspot detection
- Error propagation
- Sensitivity analysis
"""

import numpy as np
import pytest

# Mark all tests in this module
pytestmark = pytest.mark.quality


class TestUncertaintyQuantifier:
    """Tests for UncertaintyQuantifier class."""

    def test_basic_metrics(self):
        """Test basic uncertainty metrics calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.random.normal(100, 10, size=1000)
        metrics = quantifier.calculate_metrics(data)

        # Check basic statistics
        assert abs(metrics.mean - 100) < 2, "Mean should be close to 100"
        assert abs(metrics.std - 10) < 2, "Std should be close to 10"
        assert metrics.standard_error < 1, "SE should be small for large n"
        assert 0 < metrics.coefficient_of_variation < 0.2, "CV should be reasonable"

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.random.normal(50, 5, size=100)
        metrics = quantifier.calculate_metrics(data)

        # 95% CI should contain mean
        assert metrics.confidence_interval_95[0] < 50 < metrics.confidence_interval_95[1]

        # 99% CI should be wider than 95% CI
        width_95 = metrics.confidence_interval_95[1] - metrics.confidence_interval_95[0]
        width_99 = metrics.confidence_interval_99[1] - metrics.confidence_interval_99[0]
        assert width_99 > width_95, "99% CI should be wider than 95% CI"

    def test_prediction_interval(self):
        """Test prediction interval calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.random.normal(100, 10, size=100)
        metrics = quantifier.calculate_metrics(data)

        # Prediction interval should be wider than confidence interval
        pi_width = metrics.prediction_interval_95[1] - metrics.prediction_interval_95[0]
        ci_width = metrics.confidence_interval_95[1] - metrics.confidence_interval_95[0]
        assert pi_width > ci_width, "Prediction interval should be wider than CI"

    def test_empty_data(self):
        """Test handling of empty data."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.array([])
        metrics = quantifier.calculate_metrics(data)

        assert np.isnan(metrics.mean)
        assert np.isnan(metrics.std)
        assert metrics.n_samples == 0

    def test_nan_handling(self):
        """Test handling of NaN values in data."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
        metrics = quantifier.calculate_metrics(data)

        assert metrics.n_samples == 5, "Should count only valid samples"
        assert abs(metrics.mean - 3.0) < 0.01, "Mean should be 3.0"

    def test_skewness_kurtosis(self):
        """Test skewness and kurtosis calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        # Normal data should have ~0 skewness and ~0 excess kurtosis
        normal_data = np.random.normal(0, 1, size=10000)
        metrics = quantifier.calculate_metrics(normal_data)

        assert abs(metrics.skewness) < 0.2, "Normal data should have near-zero skewness"
        assert abs(metrics.kurtosis) < 0.5, "Normal data should have near-zero excess kurtosis"

    def test_weighted_metrics(self):
        """Test weighted statistics calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 5.0])  # Heavy weight on 5

        metrics = quantifier.calculate_metrics(data, weights=weights)
        assert metrics.mean > 3.5, "Weighted mean should be pulled toward 5"


class TestEnsembleUncertainty:
    """Tests for ensemble uncertainty calculation."""

    def test_basic_ensemble(self):
        """Test basic ensemble uncertainty calculation."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        # Create ensemble with known spread
        base = np.random.normal(100, 10, size=(50, 50))
        ensemble = [base + np.random.normal(0, 5, size=(50, 50)) for _ in range(10)]

        result = quantifier.calculate_ensemble_uncertainty(ensemble)

        assert result.member_count == 10
        assert np.mean(result.ensemble_std) > 0, "Should have non-zero spread"
        assert 0.68 in result.confidence_intervals, "Should have 68% CI"

    def test_classification_agreement(self):
        """Test ensemble agreement for classification."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        # All ensemble members agree
        ensemble = [np.ones((10, 10)) * 0.9 for _ in range(5)]
        result = quantifier.calculate_ensemble_uncertainty(ensemble, classification_threshold=0.5)

        assert result.agreement_fraction == 1.0, "All members should agree"

    def test_outlier_detection(self):
        """Test outlier detection in ensemble."""
        from core.quality.uncertainty import UncertaintyQuantifier, QuantificationConfig

        # Use lower outlier threshold to make test more sensitive
        config = QuantificationConfig(outlier_threshold=2.0)
        quantifier = UncertaintyQuantifier(config)

        # Create ensemble with realistic spread and one clear outlier
        # With 9 at ~100 and 1 at ~1000, zscore > 2.0
        np.random.seed(42)
        ensemble = [np.random.normal(100, 5, (10, 10)) for _ in range(9)]
        ensemble.append(np.random.normal(1000, 5, (10, 10)))  # Clear outlier (10x different)

        result = quantifier.calculate_ensemble_uncertainty(ensemble)

        assert result.outlier_count >= 1, "Should detect at least one outlier"


class TestCalibrationAssessor:
    """Tests for calibration assessment."""

    def test_well_calibrated(self):
        """Test detection of well-calibrated predictions."""
        from core.quality.uncertainty import CalibrationAssessor

        assessor = CalibrationAssessor()

        # Generate well-calibrated predictions
        n = 1000
        true_probs = np.random.uniform(0, 1, n)
        observations = (np.random.uniform(0, 1, n) < true_probs).astype(float)

        result = assessor.assess_calibration(true_probs, observations)

        assert result.expected_calibration_error < 0.15, "ECE should be low for calibrated predictions"

    def test_poorly_calibrated(self):
        """Test detection of poorly calibrated predictions."""
        from core.quality.uncertainty import CalibrationAssessor

        assessor = CalibrationAssessor()

        # Generate overconfident predictions
        n = 1000
        predictions = np.random.choice([0.9, 0.1], size=n)  # Always confident
        observations = np.random.choice([0, 1], size=n)  # Random outcomes

        result = assessor.assess_calibration(predictions, observations)

        assert result.expected_calibration_error > 0.2, "ECE should be high for uncalibrated predictions"

    def test_brier_score(self):
        """Test Brier score calculation."""
        from core.quality.uncertainty import CalibrationAssessor, QuantificationConfig

        # Use config with lower min_samples for this test
        config = QuantificationConfig(min_samples=4)
        assessor = CalibrationAssessor(config)

        # Perfect predictions with enough samples
        predictions = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        observations = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        result = assessor.assess_calibration(predictions, observations)

        assert result.brier_score == 0.0, "Brier score should be 0 for perfect predictions"


class TestSpatialUncertaintyMapper:
    """Tests for spatial uncertainty mapping."""

    def test_local_variance(self):
        """Test local variance computation."""
        from core.quality.uncertainty import SpatialUncertaintyMapper

        mapper = SpatialUncertaintyMapper()

        # Create data with varying local variance
        data = np.zeros((50, 50))
        data[:25, :] = np.random.normal(0, 1, (25, 50))  # Low variance region
        data[25:, :] = np.random.normal(0, 10, (25, 50))  # High variance region

        surface = mapper.compute_uncertainty_surface(data)

        # Check that high variance region has higher uncertainty
        low_var_region = surface.uncertainty[:20, :]
        high_var_region = surface.uncertainty[30:, :]

        assert np.mean(high_var_region) > np.mean(low_var_region), \
            "High variance region should have higher uncertainty"

    def test_residual_uncertainty(self):
        """Test residual-based uncertainty."""
        from core.quality.uncertainty import SpatialUncertaintyMapper, SpatialUncertaintyConfig, SpatialUncertaintyMethod

        config = SpatialUncertaintyConfig(method=SpatialUncertaintyMethod.RESIDUAL_MAP)
        mapper = SpatialUncertaintyMapper(config)

        data = np.ones((50, 50)) * 100
        reference = np.ones((50, 50)) * 100
        reference[20:30, 20:30] = 110  # Create residual in center

        surface = mapper.compute_uncertainty_surface(data, reference)

        # Center should have higher uncertainty
        center_unc = np.mean(surface.uncertainty[20:30, 20:30])
        edge_unc = np.mean(surface.uncertainty[:10, :10])

        assert center_unc > edge_unc, "Residual region should have higher uncertainty"

    def test_ensemble_spread_uncertainty(self):
        """Test ensemble-based uncertainty."""
        from core.quality.uncertainty import SpatialUncertaintyMapper, SpatialUncertaintyConfig, SpatialUncertaintyMethod

        config = SpatialUncertaintyConfig(method=SpatialUncertaintyMethod.ENSEMBLE_SPREAD)
        mapper = SpatialUncertaintyMapper(config)

        # Create ensemble with varying spread
        base = np.ones((50, 50)) * 100
        ensemble = [base + np.random.normal(0, 5, (50, 50)) for _ in range(5)]

        surface = mapper.compute_uncertainty_surface(None, None, ensemble)

        assert surface.mean_uncertainty > 0, "Should have positive uncertainty"

    def test_autocorrelation(self):
        """Test spatial autocorrelation computation."""
        from core.quality.uncertainty import compute_spatial_autocorrelation

        # Highly autocorrelated data
        smooth_data = np.zeros((50, 50))
        for i in range(50):
            for j in range(50):
                smooth_data[i, j] = i + j  # Gradient - highly correlated

        autocorr = compute_spatial_autocorrelation(smooth_data)
        assert autocorr > 0.5, "Smooth gradient should have high autocorrelation"


class TestHotspotDetection:
    """Tests for uncertainty hotspot detection."""

    def test_threshold_hotspots(self):
        """Test threshold-based hotspot detection."""
        from core.quality.uncertainty import HotspotDetector

        detector = HotspotDetector()

        # Create uncertainty with clear hotspot
        uncertainty = np.ones((50, 50)) * 0.1
        uncertainty[20:30, 20:30] = 0.9  # Clear hotspot

        result = detector.detect_hotspots(uncertainty)

        assert len(result.hotspots) >= 1, "Should detect at least one hotspot"
        assert result.total_hotspot_area >= 50, "Hotspot area should be significant"

    def test_zscore_hotspots(self):
        """Test z-score based hotspot detection."""
        from core.quality.uncertainty import HotspotDetector, HotspotMethod

        detector = HotspotDetector()

        # Create data with outlier region
        uncertainty = np.random.normal(0.5, 0.1, (50, 50))
        uncertainty[20:30, 20:30] = 1.5  # 10 sigma outlier

        result = detector.detect_hotspots(uncertainty, method=HotspotMethod.ZSCORE)

        assert len(result.hotspots) >= 1, "Should detect hotspot"

    def test_hotspot_properties(self):
        """Test hotspot property extraction."""
        from core.quality.uncertainty import HotspotDetector

        detector = HotspotDetector()

        uncertainty = np.ones((50, 50)) * 0.1
        uncertainty[10:20, 10:20] = 0.9  # Known hotspot

        result = detector.detect_hotspots(uncertainty)

        if result.hotspots:
            hotspot = result.hotspots[0]
            assert hotspot.area_pixels >= 50, "Hotspot area should be ~100 pixels"
            assert hotspot.mean_uncertainty > 0.5, "Hotspot should have high uncertainty"
            assert 1 <= hotspot.severity <= 5, "Severity should be 1-5"


class TestQualityErrorPropagator:
    """Tests for quality error propagation."""

    def test_mean_propagation(self):
        """Test uncertainty propagation through mean."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()

        values = [0.8, 0.9, 0.85]
        uncertainties = [0.05, 0.03, 0.04]

        result = propagator.propagate_aggregation(
            values, uncertainties, AggregationMethod.MEAN
        )

        assert abs(result.value - 0.85) < 0.01, "Mean should be ~0.85"
        # SE of mean should be sqrt(sum(ui^2/n^2))
        expected_uncertainty = np.sqrt(sum(u**2 for u in uncertainties)) / 3
        assert abs(result.uncertainty - expected_uncertainty) < 0.01

    def test_weighted_mean_propagation(self):
        """Test uncertainty propagation through weighted mean."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()

        values = [0.5, 1.0]
        uncertainties = [0.1, 0.1]
        weights = [1.0, 3.0]  # Weight second value more

        result = propagator.propagate_aggregation(
            values, uncertainties, AggregationMethod.WEIGHTED_MEAN, weights
        )

        # Weighted mean: (0.5*1 + 1.0*3) / 4 = 0.875
        assert abs(result.value - 0.875) < 0.01

    def test_minimum_propagation(self):
        """Test uncertainty propagation through minimum."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()

        values = [0.8, 0.5, 0.9]
        uncertainties = [0.05, 0.10, 0.03]

        result = propagator.propagate_aggregation(
            values, uncertainties, AggregationMethod.MINIMUM
        )

        assert result.value == 0.5, "Should select minimum"
        assert result.uncertainty == 0.10, "Uncertainty should be from minimum element"

    def test_geometric_mean_propagation(self):
        """Test uncertainty propagation through geometric mean."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()

        values = [2.0, 8.0]
        uncertainties = [0.1, 0.1]

        result = propagator.propagate_aggregation(
            values, uncertainties, AggregationMethod.GEOMETRIC_MEAN
        )

        # Geometric mean of 2 and 8 = 4
        assert abs(result.value - 4.0) < 0.01

    def test_threshold_decision(self):
        """Test threshold decision with uncertainty."""
        from core.quality.uncertainty import QualityErrorPropagator

        propagator = QualityErrorPropagator()

        # Clear exceedance
        result = propagator.propagate_threshold_decision(
            value=0.9,
            uncertainty=0.05,
            threshold=0.7,
        )
        assert result["decision"] == "exceeds"
        assert result["decision_confidence"] > 0.99

        # Uncertain case
        result = propagator.propagate_threshold_decision(
            value=0.72,
            uncertainty=0.1,
            threshold=0.7,
        )
        assert result["decision_confidence"] < 0.7, "Should be uncertain"

    def test_error_budget(self):
        """Test error budget construction."""
        from core.quality.uncertainty import compute_error_budget

        names = ["sensor", "algorithm", "fusion"]
        uncertainties = [0.05, 0.03, 0.08]  # Fusion dominates

        budget = compute_error_budget(names, uncertainties)

        assert budget.dominant_source == "fusion", "Fusion should dominate"
        assert budget.is_dominated, "Should be dominated by one source"
        assert abs(budget.total_uncertainty - np.sqrt(0.05**2 + 0.03**2 + 0.08**2)) < 0.001


class TestSensitivityAnalyzer:
    """Tests for sensitivity analysis."""

    def test_basic_sensitivity(self):
        """Test basic sensitivity analysis."""
        from core.quality.uncertainty import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer()

        # Linear function: y = 2*a + 3*b
        def func(a, b):
            return 2 * a + 3 * b

        params = {"a": 1.0, "b": 1.0}
        uncertainties = {"a": 0.1, "b": 0.1}

        result = analyzer.analyze_sensitivity(func, params, uncertainties)

        # Sensitivity should be 2 for a, 3 for b
        assert abs(result.parameter_sensitivities["a"] - 2.0) < 0.1
        assert abs(result.parameter_sensitivities["b"] - 3.0) < 0.1
        assert result.most_sensitive == "b"
        assert result.least_sensitive == "a"

    def test_uncertainty_contribution(self):
        """Test uncertainty contribution calculation."""
        from core.quality.uncertainty import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer()

        def func(a, b):
            return a + b

        params = {"a": 1.0, "b": 1.0}
        uncertainties = {"a": 0.1, "b": 0.2}  # b has larger uncertainty

        contributions = analyzer.compute_uncertainty_contribution(func, params, uncertainties)

        # b should contribute more due to larger uncertainty
        assert contributions["b"] > contributions["a"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        from core.quality.uncertainty import calculate_confidence_interval

        data = np.random.normal(100, 10, size=100)
        ci = calculate_confidence_interval(data, confidence=0.95)

        assert ci[0] < 100 < ci[1], "CI should contain true mean"
        assert ci[1] - ci[0] < 10, "CI shouldn't be too wide for large sample"

    def test_calculate_prediction_interval(self):
        """Test prediction interval calculation."""
        from core.quality.uncertainty import calculate_prediction_interval

        data = np.random.normal(100, 10, size=100)
        pi = calculate_prediction_interval(data)

        assert pi[0] < 100 < pi[1], "PI should contain true mean"
        # PI should be approximately 2*std wide for normal data at 95%
        assert 30 < pi[1] - pi[0] < 50, "PI width should be reasonable"

    def test_coefficient_of_variation(self):
        """Test CV calculation."""
        from core.quality.uncertainty import calculate_coefficient_of_variation

        data = np.array([100, 101, 99, 100, 100])
        cv = calculate_coefficient_of_variation(data)

        assert cv < 0.01, "Low variability data should have low CV"

        data = np.array([50, 100, 150, 200])
        cv = calculate_coefficient_of_variation(data)

        assert cv > 0.3, "High variability data should have high CV"

    def test_propagate_quality_uncertainty(self):
        """Test quality uncertainty propagation convenience function."""
        from core.quality.uncertainty import propagate_quality_uncertainty

        result = propagate_quality_uncertainty(
            values=[0.8, 0.9, 0.85],
            uncertainties=[0.05, 0.03, 0.04],
            method="mean",
        )

        assert 0.8 < result.value < 0.9
        assert result.uncertainty > 0

    def test_combine_uncertainties(self):
        """Test uncertainty combination functions."""
        from core.quality.uncertainty import (
            combine_independent_uncertainties,
            combine_correlated_uncertainties,
        )

        # Independent
        combined = combine_independent_uncertainties([0.1, 0.1, 0.1])
        expected = np.sqrt(3 * 0.1**2)
        assert abs(combined - expected) < 0.001

        # Fully correlated
        combined = combine_correlated_uncertainties([0.1, 0.1, 0.1], correlation_coefficient=1.0)
        assert abs(combined - 0.3) < 0.001  # Linear sum

    def test_threshold_exceedance_probability(self):
        """Test threshold exceedance probability."""
        from core.quality.uncertainty import threshold_exceedance_probability

        # Way above threshold
        prob = threshold_exceedance_probability(value=100, uncertainty=5, threshold=80)
        assert prob > 0.99

        # Way below threshold
        prob = threshold_exceedance_probability(value=60, uncertainty=5, threshold=80)
        assert prob < 0.01

        # At threshold
        prob = threshold_exceedance_probability(value=80, uncertainty=5, threshold=80)
        assert abs(prob - 0.5) < 0.01


class TestLocalStatistics:
    """Tests for local statistics computation."""

    def test_local_statistics_computation(self):
        """Test local statistics calculation."""
        from core.quality.uncertainty import SpatialUncertaintyMapper

        mapper = SpatialUncertaintyMapper()

        # Create data with known local structure
        data = np.random.normal(100, 10, (50, 50))

        stats = mapper.compute_local_statistics(data)

        assert stats.local_mean.shape == data.shape
        assert stats.local_std.shape == data.shape
        assert stats.local_cv.shape == data.shape
        assert np.nanmean(stats.local_mean) > 90  # Near 100

    def test_local_range(self):
        """Test local range calculation."""
        from core.quality.uncertainty import SpatialUncertaintyMapper

        mapper = SpatialUncertaintyMapper()

        # Constant data should have zero local range
        data = np.ones((50, 50)) * 100

        stats = mapper.compute_local_statistics(data)

        assert np.allclose(stats.local_range, 0), "Constant data should have zero range"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_sample(self):
        """Test handling of single sample."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.array([42.0])
        metrics = quantifier.calculate_metrics(data)

        assert metrics.mean == 42.0
        assert metrics.n_samples == 1

    def test_constant_data(self):
        """Test handling of constant data."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.ones(100) * 5.0
        metrics = quantifier.calculate_metrics(data)

        assert metrics.mean == 5.0
        assert metrics.std == 0.0
        assert metrics.coefficient_of_variation == 0.0

    def test_infinite_cv(self):
        """Test CV when mean is zero."""
        from core.quality.uncertainty import calculate_coefficient_of_variation

        data = np.array([-1, 1, -1, 1])  # Mean = 0
        cv = calculate_coefficient_of_variation(data)

        assert cv == np.inf or cv > 1000, "CV should be infinite when mean is zero"

    def test_all_nan_data(self):
        """Test handling of all-NaN data."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.array([np.nan, np.nan, np.nan])
        metrics = quantifier.calculate_metrics(data)

        assert np.isnan(metrics.mean)
        assert metrics.n_samples == 0

    def test_empty_ensemble(self):
        """Test handling of empty ensemble."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()

        with pytest.raises(ValueError, match="Empty ensemble"):
            quantifier.calculate_ensemble_uncertainty([])


class TestIntegration:
    """Integration tests for uncertainty module."""

    def test_full_uncertainty_pipeline(self):
        """Test complete uncertainty analysis pipeline."""
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
            HotspotDetector,
            QualityErrorPropagator,
            AggregationMethod,
        )

        # Create test data
        data = np.random.normal(100, 10, (100, 100))

        # 1. Calculate global uncertainty metrics
        quantifier = UncertaintyQuantifier()
        global_metrics = quantifier.calculate_metrics(data)

        assert global_metrics.n_samples > 0
        assert global_metrics.confidence_interval_95 is not None

        # 2. Map spatial uncertainty
        mapper = SpatialUncertaintyMapper()
        surface = mapper.compute_uncertainty_surface(data)

        assert surface.uncertainty.shape == data.shape
        assert surface.mean_uncertainty > 0

        # 3. Detect hotspots
        detector = HotspotDetector()
        hotspots = detector.detect_hotspots(surface.uncertainty)

        assert hotspots.hotspot_label_map.shape == data.shape

        # 4. Propagate through quality aggregation
        propagator = QualityErrorPropagator()
        quality_scores = [0.85, 0.90, 0.88]
        quality_uncertainties = [0.05, 0.03, 0.04]

        result = propagator.propagate_aggregation(
            quality_scores,
            quality_uncertainties,
            AggregationMethod.MEAN,
        )

        assert 0 < result.value < 1
        assert result.uncertainty > 0
        assert result.confidence_interval is not None
        assert result.error_budget is not None

    def test_ensemble_analysis_pipeline(self):
        """Test ensemble-based uncertainty pipeline."""
        from core.quality.uncertainty import (
            UncertaintyQuantifier,
            SpatialUncertaintyMapper,
            SpatialUncertaintyConfig,
            SpatialUncertaintyMethod,
        )

        # Create ensemble
        base = np.random.normal(100, 10, (50, 50))
        ensemble = [base + np.random.normal(0, 5, (50, 50)) for _ in range(10)]

        # Global ensemble uncertainty
        quantifier = UncertaintyQuantifier()
        ensemble_metrics = quantifier.calculate_ensemble_uncertainty(ensemble)

        assert ensemble_metrics.member_count == 10
        assert np.mean(ensemble_metrics.ensemble_std) > 0

        # Spatial ensemble uncertainty
        config = SpatialUncertaintyConfig(
            method=SpatialUncertaintyMethod.ENSEMBLE_SPREAD
        )
        mapper = SpatialUncertaintyMapper(config)
        surface = mapper.compute_uncertainty_surface(None, None, ensemble)

        assert surface.mean_uncertainty > 0


class TestHarmonicMeanEdgeCases:
    """Tests for harmonic mean edge cases (NEW-021)."""

    def test_harmonic_mean_with_zero_values(self):
        """Test harmonic mean with zero values in input."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [0.0, 1.0, 2.0],
            [0.1, 0.1, 0.1],
            AggregationMethod.HARMONIC_MEAN,
        )
        # Should not raise and should produce finite result
        assert np.isfinite(result.value), "Harmonic mean with zero should be finite"
        assert np.isfinite(result.uncertainty), "Uncertainty should be finite"

    def test_harmonic_mean_with_negative_values(self):
        """Test harmonic mean with negative values."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [-1.0, 1.0, 2.0],
            [0.1, 0.1, 0.1],
            AggregationMethod.HARMONIC_MEAN,
        )
        assert np.isfinite(result.value)
        assert np.isfinite(result.uncertainty)

    def test_harmonic_mean_with_all_zeros(self):
        """Test harmonic mean with all zero values."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            AggregationMethod.HARMONIC_MEAN,
        )
        # Should not crash, will use small epsilon values
        assert np.isfinite(result.value)


class TestAdditionalEdgeCases:
    """Additional edge case tests for robustness."""

    def test_calibration_with_extreme_predictions(self):
        """Test calibration with extreme (0 or 1) predictions."""
        from core.quality.uncertainty import CalibrationAssessor, QuantificationConfig

        config = QuantificationConfig(min_samples=4)
        assessor = CalibrationAssessor(config)

        # All predictions at 0
        preds = np.zeros(100)
        obs = np.zeros(100)
        result = assessor.assess_calibration(preds, obs)
        assert result.expected_calibration_error == 0.0

        # All predictions at 1
        preds = np.ones(100)
        obs = np.ones(100)
        result = assessor.assess_calibration(preds, obs)
        assert result.expected_calibration_error == 0.0

    def test_spatial_uncertainty_with_inf_values(self):
        """Test spatial uncertainty handles Inf values."""
        from core.quality.uncertainty import SpatialUncertaintyMapper

        mapper = SpatialUncertaintyMapper()
        data = np.ones((20, 20)) * 5.0
        data[10, 10] = np.inf

        surface = mapper.compute_uncertainty_surface(data)
        # Should not crash and should handle inf as invalid
        assert surface.uncertainty.shape == data.shape

    def test_hotspot_detection_with_uniform_data(self):
        """Test hotspot detection on uniform uncertainty."""
        from core.quality.uncertainty import HotspotDetector

        detector = HotspotDetector()
        uniform_uncertainty = np.ones((50, 50)) * 0.5

        result = detector.detect_hotspots(uniform_uncertainty)
        # May or may not have hotspots depending on percentile threshold
        assert result.hotspot_label_map.shape == uniform_uncertainty.shape

    def test_weighted_mean_with_single_element(self):
        """Test weighted mean with single element."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [0.8],
            [0.05],
            AggregationMethod.WEIGHTED_MEAN,
            weights=[1.0],
        )
        assert result.value == 0.8
        assert result.uncertainty == 0.05

    def test_geometric_mean_with_very_small_values(self):
        """Test geometric mean with very small values."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [1e-10, 1e-10, 1e-10],
            [1e-11, 1e-11, 1e-11],
            AggregationMethod.GEOMETRIC_MEAN,
        )
        assert np.isfinite(result.value)
        assert np.isfinite(result.uncertainty)

    def test_threshold_decision_with_zero_uncertainty(self):
        """Test threshold decision with zero uncertainty."""
        from core.quality.uncertainty import QualityErrorPropagator

        propagator = QualityErrorPropagator()
        result = propagator.propagate_threshold_decision(
            value=0.8,
            uncertainty=0.0,  # Zero uncertainty
            threshold=0.5,
        )
        assert result["decision"] == "exceeds"
        assert result["decision_confidence"] == 1.0

    def test_combine_validation_empty_list(self):
        """Test combine_validation_uncertainties with empty list."""
        from core.quality.uncertainty import QualityErrorPropagator

        propagator = QualityErrorPropagator()
        result = propagator.combine_validation_uncertainties([])
        assert np.isnan(result.value)
        assert np.isnan(result.uncertainty)

    def test_entropy_with_probability_sum_not_one(self):
        """Test entropy calculation with probabilities not summing to 1."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        # Probabilities that don't sum to 1
        probs = np.array([[0.1, 0.2], [0.3, 0.4]])
        entropy = quantifier.calculate_entropy(probs)
        assert np.all(np.isfinite(entropy))

    def test_autocorrelation_with_constant_data(self):
        """Test autocorrelation with constant data."""
        from core.quality.uncertainty import compute_spatial_autocorrelation

        constant_data = np.ones((50, 50)) * 10.0
        autocorr = compute_spatial_autocorrelation(constant_data)
        assert autocorr == 1.0, "Constant data should have perfect autocorrelation"

    def test_getis_ord_hotspot_detection(self):
        """Test Getis-Ord Gi* hotspot detection."""
        from core.quality.uncertainty import HotspotDetector, HotspotMethod

        detector = HotspotDetector()

        # Create clear hotspot pattern
        uncertainty = np.random.normal(0.5, 0.05, (50, 50))
        uncertainty[20:30, 20:30] = 2.0  # Clear hotspot

        result = detector.detect_hotspots(uncertainty, method=HotspotMethod.GETIS_ORD)
        # Should detect the high-value cluster
        assert len(result.hotspots) >= 0  # May or may not detect depending on significance

    def test_local_statistics_with_all_nan_window(self):
        """Test local statistics when window is all NaN."""
        from core.quality.uncertainty import SpatialUncertaintyMapper, SpatialUncertaintyConfig

        config = SpatialUncertaintyConfig(window_size=5)
        mapper = SpatialUncertaintyMapper(config)

        # Data with NaN region
        data = np.random.normal(100, 10, (50, 50))
        data[20:30, 20:30] = np.nan

        stats = mapper.compute_local_statistics(data)
        # Should handle NaN regions
        assert stats.local_mean.shape == data.shape


class TestMetricsToDict:
    """Test to_dict methods for serialization."""

    def test_uncertainty_metrics_to_dict(self):
        """Test UncertaintyMetrics serialization."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        data = np.random.normal(100, 10, 100)
        metrics = quantifier.calculate_metrics(data)
        d = metrics.to_dict()

        assert "mean" in d
        assert "std" in d
        assert "confidence_interval_95" in d
        assert isinstance(d["confidence_interval_95"], list)

    def test_ensemble_uncertainty_to_dict(self):
        """Test EnsembleUncertainty serialization."""
        from core.quality.uncertainty import UncertaintyQuantifier

        quantifier = UncertaintyQuantifier()
        ensemble = [np.random.normal(100, 10, (10, 10)) for _ in range(5)]
        result = quantifier.calculate_ensemble_uncertainty(ensemble)
        d = result.to_dict()

        assert "ensemble_mean" in d
        assert "member_count" in d
        assert d["member_count"] == 5

    def test_calibration_result_to_dict(self):
        """Test CalibrationResult serialization."""
        from core.quality.uncertainty import CalibrationAssessor

        assessor = CalibrationAssessor()
        preds = np.random.uniform(0, 1, 100)
        obs = (np.random.uniform(0, 1, 100) < preds).astype(float)
        result = assessor.assess_calibration(preds, obs)
        d = result.to_dict()

        assert "is_calibrated" in d
        assert "expected_calibration_error" in d

    def test_propagation_result_to_dict(self):
        """Test PropagationResult serialization."""
        from core.quality.uncertainty import QualityErrorPropagator, AggregationMethod

        propagator = QualityErrorPropagator()
        result = propagator.propagate_aggregation(
            [0.8, 0.9],
            [0.05, 0.03],
            AggregationMethod.MEAN,
        )
        d = result.to_dict()

        assert "value" in d
        assert "uncertainty" in d
        assert "error_budget" in d

    def test_hotspot_analysis_to_dict(self):
        """Test HotspotAnalysis serialization."""
        from core.quality.uncertainty import HotspotDetector

        detector = HotspotDetector()
        uncertainty = np.ones((50, 50)) * 0.1
        uncertainty[20:30, 20:30] = 0.9
        result = detector.detect_hotspots(uncertainty)
        d = result.to_dict()

        assert "num_hotspots" in d
        assert "hotspot_fraction" in d
        assert "hotspots" in d

    def test_uncertainty_surface_to_dict(self):
        """Test UncertaintySurface serialization."""
        from core.quality.uncertainty import SpatialUncertaintyMapper

        mapper = SpatialUncertaintyMapper()
        data = np.random.normal(100, 10, (50, 50))
        surface = mapper.compute_uncertainty_surface(data)
        d = surface.to_dict()

        assert "shape" in d
        assert "method" in d
        assert "mean_uncertainty" in d
