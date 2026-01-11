"""
Tests for Quality Control Validation Module (Group I Track 2).

Tests for:
- Cross-Model Validation
- Cross-Sensor Validation
- Historical Baseline Validation
- Consensus Generation
"""

import pytest
from datetime import datetime, timedelta, timezone

import numpy as np


# ============================================================================
# CROSS-MODEL VALIDATION TESTS
# ============================================================================

class TestCrossModelDataStructures:
    """Test cross-model validation data structures."""

    def test_model_output_creation(self):
        """Test ModelOutput dataclass."""
        from core.quality.validation import ModelOutput

        data = np.random.rand(100, 100).astype(np.float32)
        output = ModelOutput(
            data=data,
            model_id="test_model",
            model_name="Test Model",
            overall_confidence=0.9,
        )

        assert output.model_id == "test_model"
        assert output.model_name == "Test Model"
        assert output.overall_confidence == 0.9
        assert output.confidence is not None
        assert output.confidence.shape == data.shape

    def test_model_output_auto_confidence(self):
        """Test ModelOutput auto-generates confidence from NaN mask."""
        from core.quality.validation import ModelOutput

        data = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
        output = ModelOutput(data=data, model_id="test")

        assert output.confidence[0, 0] == 1.0
        assert output.confidence[0, 1] == 0.0  # NaN has zero confidence
        assert output.confidence[1, 0] == 1.0
        assert output.confidence[1, 1] == 1.0

    def test_pairwise_comparison_to_dict(self):
        """Test PairwiseComparison serialization."""
        from core.quality.validation import PairwiseComparison

        comparison = PairwiseComparison(
            model_a="model_1",
            model_b="model_2",
            correlation=0.95,
            rmse=0.1,
            mae=0.08,
            bias=-0.02,
            max_error=0.5,
            agreement_fraction=0.92,
        )

        d = comparison.to_dict()
        assert d["model_a"] == "model_1"
        assert d["model_b"] == "model_2"
        assert d["correlation"] == 0.95
        assert d["agreement_fraction"] == 0.92

    def test_cross_model_config_defaults(self):
        """Test CrossModelConfig default values."""
        from core.quality.validation import CrossModelConfig

        config = CrossModelConfig()
        assert config.absolute_tolerance == 0.1
        assert config.relative_tolerance == 0.05
        assert config.min_models_for_consensus == 2
        assert config.use_confidence_weights is True


class TestCrossModelValidator:
    """Test CrossModelValidator class."""

    def test_validate_two_similar_models(self):
        """Test validation with two similar model outputs."""
        from core.quality.validation import (
            CrossModelValidator,
            ModelOutput,
            AgreementLevel,
        )

        # Create two similar outputs
        base = np.random.rand(50, 50).astype(np.float32)
        data1 = base + np.random.randn(50, 50) * 0.01
        data2 = base + np.random.randn(50, 50) * 0.01

        models = [
            ModelOutput(data=data1, model_id="model_1"),
            ModelOutput(data=data2, model_id="model_2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        assert len(result.pairwise_comparisons) == 1
        assert result.pairwise_comparisons[0].correlation > 0.95
        assert result.overall_agreement in [AgreementLevel.EXCELLENT, AgreementLevel.GOOD]

    def test_validate_three_models(self):
        """Test validation with three model outputs."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        base = np.random.rand(50, 50).astype(np.float32)
        data1 = base + np.random.randn(50, 50) * 0.02
        data2 = base + np.random.randn(50, 50) * 0.02
        data3 = base + np.random.randn(50, 50) * 0.02

        models = [
            ModelOutput(data=data1, model_id="model_1"),
            ModelOutput(data=data2, model_id="model_2"),
            ModelOutput(data=data3, model_id="model_3"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should have C(3,2) = 3 pairwise comparisons
        assert len(result.pairwise_comparisons) == 3
        assert len(result.model_rankings) == 3

    def test_validate_with_disagreement(self):
        """Test validation with significant disagreement."""
        from core.quality.validation import (
            CrossModelValidator,
            ModelOutput,
            AgreementLevel,
        )

        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32) + 2.0  # Offset

        models = [
            ModelOutput(data=data1, model_id="model_1"),
            ModelOutput(data=data2, model_id="model_2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should detect poor agreement
        assert result.overall_agreement in [
            AgreementLevel.POOR,
            AgreementLevel.CRITICAL,
        ]

    def test_validate_requires_minimum_models(self):
        """Test that validation requires at least 2 models."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        data = np.random.rand(50, 50).astype(np.float32)
        models = [ModelOutput(data=data, model_id="model_1")]

        validator = CrossModelValidator()
        with pytest.raises(ValueError, match="At least 2 models required"):
            validator.validate(models)

    def test_disagreement_regions_detected(self):
        """Test that disagreement regions are identified."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        # Create data with a region of disagreement
        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.5

        # Add disagreement in a corner
        data1[40:50, 40:50] = 1.0
        data2[40:50, 40:50] = 0.0

        models = [
            ModelOutput(data=data1, model_id="model_1"),
            ModelOutput(data=data2, model_id="model_2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should have at least one disagreement region
        assert len(result.agreement_map.disagreement_regions) >= 1


class TestCrossModelConvenienceFunctions:
    """Test cross-model convenience functions."""

    def test_validate_cross_model(self):
        """Test validate_cross_model function."""
        from core.quality.validation import validate_cross_model

        data1 = np.random.rand(50, 50)
        data2 = np.random.rand(50, 50)

        result = validate_cross_model(
            [data1, data2],
            model_ids=["model_a", "model_b"],
        )

        assert result.diagnostics["num_models"] == 2
        assert "model_a" in result.diagnostics["model_ids"]

    def test_compare_two_models(self):
        """Test compare_two_models function."""
        from core.quality.validation import compare_two_models

        data1 = np.random.rand(50, 50)
        data2 = data1 + np.random.randn(50, 50) * 0.05

        comparison = compare_two_models(data1, data2)

        assert comparison.model_a == "model_a"
        assert comparison.model_b == "model_b"
        assert comparison.correlation > 0.9

    def test_get_ensemble_consensus(self):
        """Test get_ensemble_consensus function."""
        from core.quality.validation import get_ensemble_consensus

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0
        data3 = np.ones((50, 50)) * 3.0

        consensus, spread = get_ensemble_consensus([data1, data2, data3])

        assert np.allclose(consensus, 2.0)
        assert np.all(spread > 0)


# ============================================================================
# CROSS-SENSOR VALIDATION TESTS
# ============================================================================

class TestCrossSensorDataStructures:
    """Test cross-sensor validation data structures."""

    def test_sensor_observation_creation(self):
        """Test SensorObservation dataclass."""
        from core.quality.validation import SensorObservation, SensorType

        data = np.random.rand(100, 100).astype(np.float32)
        obs = SensorObservation(
            data=data,
            sensor_type=SensorType.OPTICAL,
            sensor_id="sentinel2_msi",
            observable="water_extent",
        )

        assert obs.sensor_type == SensorType.OPTICAL
        assert obs.sensor_id == "sentinel2_msi"
        assert obs.observable == "water_extent"
        assert obs.confidence is not None

    def test_sensor_physics_rule(self):
        """Test SensorPhysicsRule dataclass."""
        from core.quality.validation import (
            SensorPhysicsRule,
            SensorType,
            SensorPairType,
        )

        rule = SensorPhysicsRule(
            sensor_a=SensorType.OPTICAL,
            sensor_b=SensorType.SAR,
            observable="water_extent",
            pair_type=SensorPairType.COMPLEMENTARY,
            expected_correlation=(0.5, 0.95),
            tolerance=0.25,
        )

        assert rule.pair_type == SensorPairType.COMPLEMENTARY
        assert rule.tolerance == 0.25

    def test_sensor_comparison_result_to_dict(self):
        """Test SensorComparisonResult serialization."""
        from core.quality.validation import (
            SensorComparisonResult,
            SensorPairType,
            ValidationOutcome,
        )

        result = SensorComparisonResult(
            sensor_a_id="sentinel2",
            sensor_b_id="sentinel1",
            pair_type=SensorPairType.COMPLEMENTARY,
            correlation=0.75,
            bias=0.05,
            rmse=0.15,
            agreement_fraction=0.82,
            outcome=ValidationOutcome.MINOR_DISCREPANCY,
        )

        d = result.to_dict()
        assert d["sensor_a_id"] == "sentinel2"
        assert d["pair_type"] == "complementary"
        assert d["outcome"] == "minor"


class TestSensorPhysicsLibrary:
    """Test SensorPhysicsLibrary class."""

    def test_default_rules_loaded(self):
        """Test that default physics rules are loaded."""
        from core.quality.validation import SensorPhysicsLibrary

        library = SensorPhysicsLibrary()
        assert len(library.rules) > 0

    def test_get_rule_optical_sar_water(self):
        """Test getting rule for optical-SAR water detection."""
        from core.quality.validation import SensorPhysicsLibrary, SensorType

        library = SensorPhysicsLibrary()
        rule = library.get_rule(
            SensorType.OPTICAL,
            SensorType.SAR,
            "water_extent",
        )

        assert rule is not None
        assert rule.observable == "water_extent"

    def test_get_pair_type_same_sensor(self):
        """Test pair type for same sensor type."""
        from core.quality.validation import (
            SensorPhysicsLibrary,
            SensorType,
            SensorPairType,
        )

        library = SensorPhysicsLibrary()
        pair_type = library.get_pair_type(SensorType.OPTICAL, SensorType.OPTICAL)

        assert pair_type == SensorPairType.REDUNDANT


class TestCrossSensorValidator:
    """Test CrossSensorValidator class."""

    def test_validate_two_sensors(self):
        """Test validation with two sensor observations."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
        )

        # Create similar observations
        base = np.random.rand(50, 50).astype(np.float32)
        data1 = base + np.random.randn(50, 50) * 0.05
        data2 = base + np.random.randn(50, 50) * 0.05

        observations = [
            SensorObservation(
                data=data1,
                sensor_type=SensorType.OPTICAL,
                sensor_id="sentinel2",
            ),
            SensorObservation(
                data=data2,
                sensor_type=SensorType.SAR,
                sensor_id="sentinel1",
            ),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations, observable="water_extent")

        assert len(result.comparisons) == 1
        assert result.confidence > 0

    def test_validate_with_timestamps(self):
        """Test validation with temporal coincidence analysis."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
        )

        now = datetime.now(timezone.utc)
        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32)

        observations = [
            SensorObservation(
                data=data1,
                sensor_type=SensorType.OPTICAL,
                sensor_id="sensor1",
                timestamp=now,
            ),
            SensorObservation(
                data=data2,
                sensor_type=SensorType.SAR,
                sensor_id="sensor2",
                timestamp=now + timedelta(hours=2),
            ),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations)

        assert result.temporal_analysis["has_timestamps"] is True
        assert "time_differences" in result.temporal_analysis

    def test_validate_insufficient_overlap(self):
        """Test validation with insufficient spatial overlap."""
        from core.quality.validation import (
            CrossSensorValidator,
            CrossSensorConfig,
            SensorObservation,
            SensorType,
            ValidationOutcome,
        )

        # Create data with mostly NaN (low overlap)
        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data1[:10, :10] = np.random.rand(10, 10)

        data2 = np.full((50, 50), np.nan, dtype=np.float32)
        data2[40:50, 40:50] = np.random.rand(10, 10)  # No overlap

        observations = [
            SensorObservation(data=data1, sensor_type=SensorType.OPTICAL, sensor_id="s1"),
            SensorObservation(data=data2, sensor_type=SensorType.SAR, sensor_id="s2"),
        ]

        config = CrossSensorConfig(min_overlap_fraction=0.3)
        validator = CrossSensorValidator(config)
        result = validator.validate(observations)

        # Should detect insufficient data
        assert result.comparisons[0].outcome == ValidationOutcome.INSUFFICIENT_DATA


class TestCrossSensorConvenienceFunctions:
    """Test cross-sensor convenience functions."""

    def test_validate_cross_sensor(self):
        """Test validate_cross_sensor function."""
        from core.quality.validation import validate_cross_sensor

        data1 = np.random.rand(50, 50)
        data2 = np.random.rand(50, 50)

        result = validate_cross_sensor(
            [data1, data2],
            sensor_types=["optical", "sar"],
            sensor_ids=["sensor_a", "sensor_b"],
            observable="water_extent",
        )

        assert len(result.comparisons) == 1

    def test_compare_optical_sar(self):
        """Test compare_optical_sar function."""
        from core.quality.validation import compare_optical_sar, SensorPairType

        optical_data = np.random.rand(50, 50)
        sar_data = optical_data + np.random.randn(50, 50) * 0.1

        comparison = compare_optical_sar(
            optical_data,
            sar_data,
            observable="water_extent",
        )

        assert comparison.pair_type == SensorPairType.COMPLEMENTARY


# ============================================================================
# HISTORICAL BASELINE VALIDATION TESTS
# ============================================================================

class TestHistoricalDataStructures:
    """Test historical validation data structures."""

    def test_historical_baseline_creation(self):
        """Test HistoricalBaseline dataclass."""
        from core.quality.validation import HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.1,
            max_value=0.9,
            sample_size=100,
            period_years=10.0,
        )

        assert baseline.mean == 0.5
        assert baseline.std == 0.1
        assert len(baseline.percentiles) > 0  # Auto-generated

    def test_historical_baseline_from_data(self):
        """Test HistoricalBaseline.from_data class method."""
        from core.quality.validation import HistoricalBaseline

        data = np.random.randn(1000) * 0.1 + 0.5
        baseline = HistoricalBaseline.from_data(data, period_years=5.0)

        assert abs(baseline.mean - 0.5) < 0.02
        assert abs(baseline.std - 0.1) < 0.02
        assert baseline.sample_size == 1000
        assert baseline.period_years == 5.0

    def test_historical_anomaly_to_dict(self):
        """Test HistoricalAnomaly serialization."""
        from core.quality.validation import (
            HistoricalAnomaly,
            AnomalyType,
            AnomalySeverity,
        )

        anomaly = HistoricalAnomaly(
            anomaly_type=AnomalyType.ABOVE_NORMAL,
            severity=AnomalySeverity.MODERATE,
            description="Value above historical mean",
            metric_name="flood_extent",
            current_value=0.8,
            expected_value=0.5,
            deviation=0.3,
            z_score=3.0,
            percentile=99.5,
            return_period=50.0,
        )

        d = anomaly.to_dict()
        assert d["anomaly_type"] == "above_normal"
        assert d["severity"] == "moderate"
        assert d["z_score"] == 3.0
        assert d["return_period"] == 50.0


class TestHistoricalValidator:
    """Test HistoricalValidator class."""

    def test_validate_normal_value(self):
        """Test validation with value within normal range."""
        from core.quality.validation import (
            HistoricalValidator,
            HistoricalBaseline,
            AnomalySeverity,
        )

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
            period_years=10.0,
        )

        current_data = np.ones((50, 50)) * 0.52  # Close to mean

        validator = HistoricalValidator()
        result = validator.validate(current_data, baseline, metric_name="test")

        assert result.in_historical_bounds is True
        assert result.overall_severity == AnomalySeverity.NORMAL or \
               result.overall_severity == AnomalySeverity.LOW

    def test_validate_extreme_value(self):
        """Test validation with extreme value."""
        from core.quality.validation import (
            HistoricalValidator,
            HistoricalBaseline,
            AnomalySeverity,
        )

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
            period_years=10.0,
        )

        current_data = np.ones((50, 50)) * 0.95  # Above historical max

        validator = HistoricalValidator()
        result = validator.validate(current_data, baseline, metric_name="test")

        # Should detect extreme anomaly
        assert len(result.anomalies) > 0
        assert result.overall_severity in [
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        ]

    def test_validate_with_seasonal_adjustment(self):
        """Test validation with seasonal baseline."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
            period_years=10.0,
            seasonal_means={1: 0.3, 7: 0.7},  # Jan vs Jul
            seasonal_stds={1: 0.05, 7: 0.08},
        )

        # Value of 0.7 normal in July but extreme in January
        current_data = np.ones((50, 50)) * 0.7

        validator = HistoricalValidator()

        # January observation
        jan_result = validator.validate(
            current_data,
            baseline,
            observation_date=datetime(2024, 1, 15),
        )

        # July observation
        jul_result = validator.validate(
            current_data,
            baseline,
            observation_date=datetime(2024, 7, 15),
        )

        # January should show more anomalies
        assert len(jan_result.anomalies) >= len(jul_result.anomalies)

    def test_validate_spatial(self):
        """Test spatial validation against historical baseline."""
        from core.quality.validation import HistoricalValidator

        # Create historical baseline stack
        baseline_stack = np.random.randn(10, 50, 50) * 0.1 + 0.5

        # Current with anomalous region
        current = np.ones((50, 50)) * 0.5
        current[20:30, 20:30] = 1.5  # Anomaly

        validator = HistoricalValidator()
        result = validator.validate_spatial(current, baseline_stack)

        assert result.summary_stats["pct_above_3sigma"] > 0


class TestHistoricalConvenienceFunctions:
    """Test historical validation convenience functions."""

    def test_validate_against_historical(self):
        """Test validate_against_historical function."""
        from core.quality.validation import validate_against_historical

        historical_data = np.random.randn(1000) * 0.1 + 0.5
        current_data = np.ones((50, 50)) * 0.55

        result = validate_against_historical(current_data, historical_data)

        assert result.confidence > 0
        assert "expected_value" in result.diagnostics

    def test_calculate_z_score(self):
        """Test calculate_z_score function."""
        from core.quality.validation import calculate_z_score, HistoricalBaseline

        baseline = HistoricalBaseline(mean=0.5, std=0.1, min_value=0.2, max_value=0.8)

        z = calculate_z_score(0.7, baseline)
        assert abs(z - 2.0) < 0.01  # Should be 2 std above mean

    def test_estimate_return_period(self):
        """Test estimate_return_period function."""
        from core.quality.validation import estimate_return_period, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
            period_years=10.0,
        )

        # Extreme value should have long return period
        rp = estimate_return_period(0.9, baseline)
        assert rp > 10  # Should be rare


# ============================================================================
# CONSENSUS GENERATION TESTS
# ============================================================================

class TestConsensusDataStructures:
    """Test consensus generation data structures."""

    def test_consensus_source_creation(self):
        """Test ConsensusSource dataclass."""
        from core.quality.validation import ConsensusSource, ConsensusPriority

        data = np.random.rand(50, 50).astype(np.float32)
        source = ConsensusSource(
            data=data,
            source_id="source_1",
            source_name="Test Source",
            priority=ConsensusPriority.PRIMARY,
            weight=1.5,
        )

        assert source.source_id == "source_1"
        assert source.priority == ConsensusPriority.PRIMARY
        assert source.weight == 1.5
        assert source.confidence is not None

    def test_disagreement_region_to_dict(self):
        """Test DisagreementRegion serialization."""
        from core.quality.validation import DisagreementRegion

        region = DisagreementRegion(
            region_id=1,
            bbox={"min_row": 10, "max_row": 20, "min_col": 30, "max_col": 40},
            centroid={"row": 15.0, "col": 35.0},
            size_pixels=100,
            disagreement_level=0.7,
            source_values={"source_a": 0.5, "source_b": 0.9},
            recommendation="Manual review required",
        )

        d = region.to_dict()
        assert d["region_id"] == 1
        assert d["size_pixels"] == 100
        assert d["disagreement_level"] == 0.7

    def test_consensus_config_defaults(self):
        """Test ConsensusConfig default values."""
        from core.quality.validation import ConsensusConfig, ConsensusStrategy

        config = ConsensusConfig()
        assert config.strategy == ConsensusStrategy.WEIGHTED_MEAN
        assert config.min_sources == 2
        assert config.min_agreement == 0.5


class TestConsensusGenerator:
    """Test ConsensusGenerator class."""

    def test_generate_mean_consensus(self):
        """Test mean consensus generation."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0
        data3 = np.ones((50, 50)) * 3.0

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
            ConsensusSource(data=data3, source_id="s3"),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.MEAN)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        # Mean should be 2.0
        assert np.allclose(result.consensus_data, 2.0)

    def test_generate_median_consensus(self):
        """Test median consensus generation."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0
        data3 = np.ones((50, 50)) * 10.0  # Outlier

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
            ConsensusSource(data=data3, source_id="s3"),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.MEDIAN)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        # Median should be 2.0 (robust to outlier)
        assert np.allclose(result.consensus_data, 2.0)

    def test_generate_weighted_mean_consensus(self):
        """Test weighted mean consensus generation."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0

        sources = [
            ConsensusSource(data=data1, source_id="s1", weight=3.0),  # Higher weight
            ConsensusSource(data=data2, source_id="s2", weight=1.0),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.WEIGHTED_MEAN)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        # Weighted mean should be closer to 1.0
        assert np.mean(result.consensus_data) < 1.5

    def test_generate_majority_vote(self):
        """Test majority vote consensus for categorical data."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.ones((50, 50))
        data2 = np.ones((50, 50))
        data3 = np.zeros((50, 50))

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
            ConsensusSource(data=data3, source_id="s3"),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.MAJORITY_VOTE)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources, is_categorical=True)

        # Majority should be 1.0
        assert np.allclose(result.consensus_data, 1.0)

    def test_generate_requires_minimum_sources(self):
        """Test that consensus requires minimum sources."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
        )

        data = np.random.rand(50, 50)
        sources = [ConsensusSource(data=data, source_id="s1")]

        config = ConsensusConfig(min_sources=2)
        generator = ConsensusGenerator(config)

        with pytest.raises(ValueError, match="At least 2 sources required"):
            generator.generate(sources)

    def test_disagreement_regions_detected(self):
        """Test that disagreement regions are identified."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
        )

        # Create data with disagreement region
        data1 = np.ones((50, 50)) * 0.5
        data2 = np.ones((50, 50)) * 0.5

        data1[20:30, 20:30] = 1.0
        data2[20:30, 20:30] = 0.0  # Opposite

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        config = ConsensusConfig(min_agreement=0.8)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        assert len(result.disagreement_regions) >= 1


class TestConsensusConvenienceFunctions:
    """Test consensus generation convenience functions."""

    def test_generate_consensus(self):
        """Test generate_consensus function."""
        from core.quality.validation import generate_consensus

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0

        result = generate_consensus(
            [data1, data2],
            source_ids=["src_a", "src_b"],
            strategy="mean",
        )

        assert np.allclose(result.consensus_data, 1.5)

    def test_vote_consensus(self):
        """Test vote_consensus function."""
        from core.quality.validation import vote_consensus

        data1 = np.ones((50, 50))
        data2 = np.ones((50, 50))
        data3 = np.zeros((50, 50))

        consensus, agreement = vote_consensus([data1, data2, data3])

        assert np.allclose(consensus, 1.0)
        assert np.all(agreement >= 0.66)

    def test_weighted_mean_consensus(self):
        """Test weighted_mean_consensus function."""
        from core.quality.validation import weighted_mean_consensus

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0

        consensus, uncertainty = weighted_mean_consensus(
            [data1, data2],
            weights=[2.0, 1.0],
        )

        # Weighted average: (1*2 + 2*1) / 3 = 1.33
        expected = (1.0 * 2 + 2.0 * 1) / 3
        assert np.allclose(consensus, expected, rtol=0.1)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestValidationIntegration:
    """Integration tests combining multiple validation approaches."""

    def test_full_validation_workflow(self):
        """Test complete validation workflow."""
        from core.quality.validation import (
            CrossModelValidator,
            CrossSensorValidator,
            HistoricalValidator,
            ConsensusGenerator,
            ModelOutput,
            SensorObservation,
            SensorType,
            HistoricalBaseline,
            ConsensusSource,
        )

        # Generate test data
        np.random.seed(42)
        base = np.random.rand(50, 50).astype(np.float32)

        # Model outputs
        model_outputs = [
            ModelOutput(data=base + np.random.randn(50, 50) * 0.02, model_id=f"model_{i}")
            for i in range(3)
        ]

        # Sensor observations
        sensor_obs = [
            SensorObservation(
                data=base + np.random.randn(50, 50) * 0.03,
                sensor_type=SensorType.OPTICAL,
                sensor_id="optical_1",
            ),
            SensorObservation(
                data=base + np.random.randn(50, 50) * 0.03,
                sensor_type=SensorType.SAR,
                sensor_id="sar_1",
            ),
        ]

        # Historical baseline
        historical = HistoricalBaseline.from_data(base.flatten(), period_years=5.0)

        # Run all validations
        cross_model_result = CrossModelValidator().validate(model_outputs)
        cross_sensor_result = CrossSensorValidator().validate(sensor_obs)
        historical_result = HistoricalValidator().validate(base, historical)

        # Generate consensus
        sources = [ConsensusSource(data=m.data, source_id=m.model_id) for m in model_outputs]
        consensus_result = ConsensusGenerator().generate(sources)

        # All should complete successfully
        assert cross_model_result.overall_agreement is not None
        assert cross_sensor_result.confidence > 0
        assert historical_result.overall_severity is not None
        assert consensus_result.quality is not None

    def test_validation_with_degraded_data(self):
        """Test validation handles degraded/partial data gracefully."""
        from core.quality.validation import (
            CrossModelValidator,
            ModelOutput,
            validate_cross_model,
        )

        # Create data with significant NaN regions
        data1 = np.random.rand(50, 50).astype(np.float32)
        data1[:25, :] = np.nan

        data2 = np.random.rand(50, 50).astype(np.float32)
        data2[25:, :] = np.nan

        # Should handle partial overlap gracefully
        models = [
            ModelOutput(data=data1, model_id="m1"),
            ModelOutput(data=data2, model_id="m2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should complete and report on valid overlap
        assert result is not None


# ============================================================================
# EDGE CASE TESTS FOR TRACK 2 BUG FIXES
# ============================================================================

class TestHistoricalPercentileEdgeCases:
    """Edge case tests for historical percentile calculation (FIX-I2-001)."""

    def test_percentile_value_below_min(self):
        """Test percentile for value below historical minimum."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            percentiles={5: 0.25, 25: 0.4, 50: 0.5, 75: 0.6, 95: 0.75},
            sample_size=100,
            period_years=10.0,
        )

        validator = HistoricalValidator()
        # Value way below minimum - should return 0, not negative
        percentile = validator._calculate_percentile(0.0, baseline)
        assert 0.0 <= percentile <= 100.0

    def test_percentile_value_above_max(self):
        """Test percentile for value above historical maximum."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            percentiles={5: 0.25, 25: 0.4, 50: 0.5, 75: 0.6, 95: 0.75},
            sample_size=100,
            period_years=10.0,
        )

        validator = HistoricalValidator()
        # Value way above maximum - should return <= 100, not > 100
        percentile = validator._calculate_percentile(100.0, baseline)
        assert 0.0 <= percentile <= 100.0

    def test_percentile_nan_value(self):
        """Test percentile for NaN input value."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
        )

        validator = HistoricalValidator()
        percentile = validator._calculate_percentile(float('nan'), baseline)
        # Should return default value, not propagate NaN
        assert percentile == 50.0

    def test_percentile_inf_value(self):
        """Test percentile for infinite input value."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
        )

        validator = HistoricalValidator()
        percentile_pos = validator._calculate_percentile(float('inf'), baseline)
        percentile_neg = validator._calculate_percentile(float('-inf'), baseline)
        # Should return default value, not crash
        assert percentile_pos == 50.0
        assert percentile_neg == 50.0

    def test_percentile_identical_min_max(self):
        """Test percentile when min_value equals max_value."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.0,  # Zero std means constant values
            min_value=0.5,
            max_value=0.5,  # Same as min
            percentiles={50: 0.5},  # Only one percentile
            sample_size=100,
            period_years=10.0,
        )

        validator = HistoricalValidator()
        # Should not divide by zero
        percentile = validator._calculate_percentile(0.5, baseline)
        assert 0.0 <= percentile <= 100.0

    def test_percentile_adjacent_equal_percentile_values(self):
        """Test percentile with adjacent percentiles having same value."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            percentiles={25: 0.5, 50: 0.5, 75: 0.5},  # All same value
            sample_size=100,
            period_years=10.0,
        )

        validator = HistoricalValidator()
        # Should handle without division by zero
        percentile = validator._calculate_percentile(0.5, baseline)
        assert 0.0 <= percentile <= 100.0


class TestCrossSensorEdgeCases:
    """Edge case tests for cross-sensor validation."""

    def test_all_nan_data(self):
        """Test validation with all NaN data."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
            ValidationOutcome,
        )

        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data2 = np.full((50, 50), np.nan, dtype=np.float32)

        observations = [
            SensorObservation(data=data1, sensor_type=SensorType.OPTICAL, sensor_id="s1"),
            SensorObservation(data=data2, sensor_type=SensorType.SAR, sensor_id="s2"),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations)

        # Should detect insufficient data, not crash
        assert result.comparisons[0].outcome == ValidationOutcome.INSUFFICIENT_DATA

    def test_constant_arrays(self):
        """Test validation with constant value arrays."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
        )

        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.5

        observations = [
            SensorObservation(data=data1, sensor_type=SensorType.OPTICAL, sensor_id="s1"),
            SensorObservation(data=data2, sensor_type=SensorType.SAR, sensor_id="s2"),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations)

        # Should handle constant arrays (std=0) gracefully
        assert result.comparisons[0].correlation == 1.0


class TestConsensusEdgeCases:
    """Edge case tests for consensus generation."""

    def test_all_nan_sources(self):
        """Test consensus with all NaN sources."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusSource,
        )

        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data2 = np.full((50, 50), np.nan, dtype=np.float32)

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        generator = ConsensusGenerator()
        result = generator.generate(sources)

        # Should complete without crash
        assert result is not None

    def test_empty_unique_values_voting(self):
        """Test voting consensus when all values are NaN."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.full((10, 10), np.nan, dtype=np.float32)
        data2 = np.full((10, 10), np.nan, dtype=np.float32)

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.MAJORITY_VOTE)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources, is_categorical=True)

        # Should complete without crash
        assert result is not None

    def test_single_valid_pixel(self):
        """Test consensus with only single valid pixel."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusSource,
        )

        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data1[25, 25] = 0.5
        data2 = np.full((50, 50), np.nan, dtype=np.float32)
        data2[25, 25] = 0.6

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        generator = ConsensusGenerator()
        result = generator.generate(sources)

        # Should handle single valid pixel
        assert result is not None
        # The one valid pixel should have consensus
        assert not np.isnan(result.consensus_data[25, 25])


class TestCrossModelEdgeCases:
    """Edge case tests for cross-model validation."""

    def test_get_ensemble_consensus_zero_weights(self):
        """Test get_ensemble_consensus with all-zero weights (NEW-024)."""
        from core.quality.validation import get_ensemble_consensus

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0
        data3 = np.ones((50, 50)) * 3.0

        # All zero weights should fall back to equal weights
        consensus, spread = get_ensemble_consensus(
            [data1, data2, data3],
            weights=[0.0, 0.0, 0.0]
        )

        # Should produce valid consensus (mean with equal weights)
        assert np.allclose(consensus, 2.0)
        assert not np.any(np.isnan(consensus))

    def test_compare_models_empty_overlap(self):
        """Test model comparison with no overlapping valid data."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        data1 = np.full((50, 50), np.nan, dtype=np.float32)
        data1[:25, :] = np.random.rand(25, 50)

        data2 = np.full((50, 50), np.nan, dtype=np.float32)
        data2[25:, :] = np.random.rand(25, 50)

        models = [
            ModelOutput(data=data1, model_id="m1"),
            ModelOutput(data=data2, model_id="m2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should handle no overlap gracefully
        assert result is not None
        # Comparison metrics should be NaN for no overlap
        assert np.isnan(result.pairwise_comparisons[0].correlation)

    def test_constant_arrays_model_comparison(self):
        """Test model comparison with constant value arrays."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        # Two constant arrays with same value
        data1 = np.ones((50, 50), dtype=np.float32) * 0.5
        data2 = np.ones((50, 50), dtype=np.float32) * 0.5

        models = [
            ModelOutput(data=data1, model_id="m1"),
            ModelOutput(data=data2, model_id="m2"),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Perfect agreement on constant data
        assert result.pairwise_comparisons[0].correlation == 1.0
        assert result.pairwise_comparisons[0].rmse < 1e-10

    def test_model_ranking_single_comparison(self):
        """Test model ranking when only one pairwise comparison exists."""
        from core.quality.validation import CrossModelValidator, ModelOutput

        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = data1 + np.random.randn(50, 50).astype(np.float32) * 0.01

        models = [
            ModelOutput(data=data1, model_id="m1", overall_confidence=0.9),
            ModelOutput(data=data2, model_id="m2", overall_confidence=0.8),
        ]

        validator = CrossModelValidator()
        result = validator.validate(models)

        # Should rank both models
        assert len(result.model_rankings) == 2


class TestCrossSensorAdvancedEdgeCases:
    """Advanced edge case tests for cross-sensor validation."""

    def test_sensor_pair_no_physics_rule(self):
        """Test sensor comparison when no physics rule exists."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
            SensorPairType,
        )

        # Use types that may not have a rule
        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32)

        observations = [
            SensorObservation(
                data=data1,
                sensor_type=SensorType.LIDAR,
                sensor_id="lidar1",
                observable="custom_metric",
            ),
            SensorObservation(
                data=data2,
                sensor_type=SensorType.WEATHER,
                sensor_id="weather1",
                observable="custom_metric",
            ),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations, observable="custom_metric")

        # Should default to INDEPENDENT pair type
        assert result.comparisons[0].pair_type == SensorPairType.INDEPENDENT

    def test_temporal_analysis_no_timestamps(self):
        """Test temporal analysis when no timestamps provided."""
        from core.quality.validation import (
            CrossSensorValidator,
            SensorObservation,
            SensorType,
        )

        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32)

        observations = [
            SensorObservation(data=data1, sensor_type=SensorType.OPTICAL, sensor_id="s1"),
            SensorObservation(data=data2, sensor_type=SensorType.SAR, sensor_id="s2"),
        ]

        validator = CrossSensorValidator()
        result = validator.validate(observations)

        # Should indicate no timestamps
        assert result.temporal_analysis.get("has_timestamps") is False


class TestHistoricalAdvancedEdgeCases:
    """Advanced edge case tests for historical validation."""

    def test_validate_insufficient_baseline_samples(self):
        """Test validation with insufficient historical samples."""
        from core.quality.validation import HistoricalValidator, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=5,  # Below minimum
            period_years=1.0,
        )

        current_data = np.ones((50, 50)) * 0.6

        validator = HistoricalValidator()
        result = validator.validate(current_data, baseline)

        # Should report insufficient data
        assert result.confidence == 0.0
        assert "error" in result.diagnostics

    def test_return_period_extreme_zscore(self):
        """Test return period estimation for extreme z-scores."""
        from core.quality.validation import estimate_return_period, HistoricalBaseline

        baseline = HistoricalBaseline(
            mean=0.5,
            std=0.1,
            min_value=0.2,
            max_value=0.8,
            sample_size=100,
            period_years=10.0,
        )

        # Very extreme value
        rp = estimate_return_period(1.5, baseline)  # z-score = 10

        # Should be capped at reasonable value
        assert rp <= 10000.0

    def test_spatial_validation_3d_baseline(self):
        """Test spatial validation with 3D baseline stack."""
        from core.quality.validation import HistoricalValidator

        # Create historical baseline stack (10 observations)
        baseline_stack = np.random.randn(10, 50, 50) * 0.1 + 0.5

        # Current with anomalous region
        current = np.ones((50, 50)) * 0.5
        current[20:30, 20:30] = 2.0  # Very high anomaly

        validator = HistoricalValidator()
        result = validator.validate_spatial(current, baseline_stack)

        # Should detect the anomaly
        assert result.summary_stats["pct_above_3sigma"] > 0
        assert len(result.anomalies) > 0


class TestConsensusAdvancedEdgeCases:
    """Advanced edge case tests for consensus generation."""

    def test_robust_mean_convergence(self):
        """Test robust mean converges with outliers."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        # Create data with one outlier source
        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 1.1
        data3 = np.ones((50, 50)) * 100.0  # Extreme outlier

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
            ConsensusSource(data=data3, source_id="s3"),
        ]

        config = ConsensusConfig(strategy=ConsensusStrategy.ROBUST_MEAN)
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        # Robust mean should be closer to 1.0 than to 100.0
        mean_val = np.mean(result.consensus_data)
        assert mean_val < 10.0

    def test_trimmed_mean_with_few_sources(self):
        """Test trimmed mean when trim count equals number of sources."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusConfig,
            ConsensusSource,
            ConsensusStrategy,
        )

        data1 = np.ones((50, 50)) * 1.0
        data2 = np.ones((50, 50)) * 2.0

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        config = ConsensusConfig(
            strategy=ConsensusStrategy.TRIMMED_MEAN,
            trim_fraction=0.5  # Would trim both if not guarded
        )
        generator = ConsensusGenerator(config)
        result = generator.generate(sources)

        # Should still produce valid consensus
        assert result is not None
        assert not np.any(np.isnan(result.consensus_data))

    def test_quality_classification_edge_thresholds(self):
        """Test quality classification at edge of thresholds."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusSource,
            ConsensusQuality,
        )

        # Perfect agreement
        data1 = np.ones((50, 50)) * 0.5
        data2 = np.ones((50, 50)) * 0.5
        data3 = np.ones((50, 50)) * 0.5

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
            ConsensusSource(data=data3, source_id="s3"),
        ]

        generator = ConsensusGenerator()
        result = generator.generate(sources)

        # Perfect agreement should yield high quality
        assert result.quality == ConsensusQuality.HIGH

    def test_source_contributions_sum_to_one(self):
        """Test that source contributions are normalized."""
        from core.quality.validation import (
            ConsensusGenerator,
            ConsensusSource,
        )

        data1 = np.random.rand(50, 50).astype(np.float32)
        data2 = np.random.rand(50, 50).astype(np.float32)

        sources = [
            ConsensusSource(data=data1, source_id="s1"),
            ConsensusSource(data=data2, source_id="s2"),
        ]

        generator = ConsensusGenerator()
        result = generator.generate(sources)

        # Contributions should sum to approximately 1
        total = sum(result.source_contributions.values())
        assert abs(total - 1.0) < 1e-6
