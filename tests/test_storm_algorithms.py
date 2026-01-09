"""
Comprehensive Storm Algorithm Test Suite (Group E, Track 4)

This module provides integrated tests for all storm damage algorithms.
Tests cover:
- WindDamageDetection algorithm execution and edge cases
- StructuralDamageAssessment algorithm execution and edge cases
- Configuration validation
- Reproducibility validation
- Parameter range testing
- Empty input handling
- Module exports and registry

Following the Agent Code Review Checklist:
- Division operations guarded against zero
- Array indexing validated for bounds
- NaN/Inf handling with np.isnan(), np.isinf()
- Edge cases tested (empty arrays, single elements, all-same values)
"""

import pytest
import numpy as np
from typing import Tuple, Optional

from core.analysis.library.baseline.storm import (
    WindDamageDetection,
    WindDamageConfig,
    WindDamageResult,
    StructuralDamageAssessment,
    StructuralDamageConfig,
    StructuralDamageResult,
    STORM_ALGORITHMS,
    get_algorithm,
    list_algorithms,
)


# ============================================================================
# Synthetic Data Generators
# ============================================================================

class StormSyntheticDataGenerator:
    """Generate synthetic test data for storm damage algorithms."""

    @staticmethod
    def create_vegetation_data(
        shape: Tuple[int, int] = (100, 100),
        with_damage_region: bool = True,
        damage_region: Optional[Tuple[int, int, int, int]] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create synthetic optical data for vegetation damage detection.

        Args:
            shape: Image dimensions (height, width)
            with_damage_region: Whether to include a simulated damage region
            damage_region: (row_start, row_end, col_start, col_end) for damage area
            seed: Random seed for reproducibility

        Returns:
            Tuple of (red_pre, nir_pre, red_post, nir_post) reflectance values (0-1)
        """
        if seed is not None:
            np.random.seed(seed)

        h, w = shape

        # Pre-event: healthy vegetation
        # Vegetation has low red reflectance, high NIR reflectance
        red_pre = np.random.uniform(0.03, 0.08, shape).astype(np.float32)
        nir_pre = np.random.uniform(0.35, 0.55, shape).astype(np.float32)

        # Post-event: copy of pre-event initially
        red_post = red_pre.copy()
        nir_post = nir_pre.copy()

        if with_damage_region:
            if damage_region is None:
                damage_region = (h // 4, h // 2, w // 4, w // 2)
            r1, r2, c1, c2 = damage_region

            # Damaged vegetation: higher red, lower NIR (browning/defoliation)
            red_post[r1:r2, c1:c2] = np.random.uniform(0.10, 0.18, (r2 - r1, c2 - c1))
            nir_post[r1:r2, c1:c2] = np.random.uniform(0.15, 0.25, (r2 - r1, c2 - c1))

        return red_pre, nir_pre, red_post, nir_post

    @staticmethod
    def create_structural_data(
        shape: Tuple[int, int] = (100, 100),
        with_damage_region: bool = True,
        damage_region: Optional[Tuple[int, int, int, int]] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create synthetic optical data for structural damage assessment.

        Args:
            shape: Image dimensions (height, width)
            with_damage_region: Whether to include a simulated damage region
            damage_region: (row_start, row_end, col_start, col_end) for damage area
            seed: Random seed for reproducibility

        Returns:
            Tuple of (optical_pre, optical_post) grayscale imagery (0-1)
        """
        if seed is not None:
            np.random.seed(seed)

        h, w = shape

        # Pre-event: urban area with some texture variation
        optical_pre = np.random.uniform(0.3, 0.6, shape).astype(np.float32)

        # Post-event: copy of pre-event initially
        optical_post = optical_pre.copy()

        if with_damage_region:
            if damage_region is None:
                damage_region = (h // 4, h // 2, w // 4, w // 2)
            r1, r2, c1, c2 = damage_region

            # Damaged area: higher variance (debris), brightness changes
            optical_post[r1:r2, c1:c2] = np.random.uniform(0.1, 0.9, (r2 - r1, c2 - c1))

        return optical_pre, optical_post


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_vegetation_data():
    """Fixture providing synthetic vegetation data."""
    return StormSyntheticDataGenerator.create_vegetation_data(seed=42)


@pytest.fixture
def synthetic_structural_data():
    """Fixture providing synthetic structural data."""
    return StormSyntheticDataGenerator.create_structural_data(seed=42)


@pytest.fixture
def deterministic_vegetation_data():
    """Fixture providing deterministic vegetation data for reproducibility tests."""
    return StormSyntheticDataGenerator.create_vegetation_data(seed=123)


@pytest.fixture
def deterministic_structural_data():
    """Fixture providing deterministic structural data for reproducibility tests."""
    return StormSyntheticDataGenerator.create_structural_data(seed=123)


# ============================================================================
# WindDamageDetection Tests
# ============================================================================

class TestWindDamageConfig:
    """Test WindDamageConfig validation."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = WindDamageConfig()
        assert config.ndvi_change_threshold == -0.2
        assert config.min_damage_area_ha == 0.5
        assert config.vegetation_index == "ndvi"
        assert config.pre_event_ndvi_min == 0.3
        assert config.use_severity_classification is True
        assert config.cloud_mask_threshold == 0.2

    def test_config_threshold_boundary_valid(self):
        """Test that boundary threshold values are accepted."""
        config_zero = WindDamageConfig(ndvi_change_threshold=0.0)
        assert config_zero.ndvi_change_threshold == 0.0

        config_min = WindDamageConfig(ndvi_change_threshold=-1.0)
        assert config_min.ndvi_change_threshold == -1.0

    def test_config_threshold_invalid(self):
        """Test that invalid threshold values raise errors."""
        with pytest.raises(ValueError, match="ndvi_change_threshold must be in"):
            WindDamageConfig(ndvi_change_threshold=0.5)

        with pytest.raises(ValueError, match="ndvi_change_threshold must be in"):
            WindDamageConfig(ndvi_change_threshold=-1.5)

    def test_config_min_area_invalid(self):
        """Test that negative min_damage_area_ha raises error."""
        with pytest.raises(ValueError, match="min_damage_area_ha must be non-negative"):
            WindDamageConfig(min_damage_area_ha=-1.0)

    def test_config_vegetation_index_invalid(self):
        """Test that invalid vegetation_index raises error."""
        with pytest.raises(ValueError, match="vegetation_index must be"):
            WindDamageConfig(vegetation_index="invalid")

    def test_config_pre_event_ndvi_invalid(self):
        """Test that invalid pre_event_ndvi_min raises error."""
        with pytest.raises(ValueError, match="pre_event_ndvi_min must be in"):
            WindDamageConfig(pre_event_ndvi_min=1.5)

        with pytest.raises(ValueError, match="pre_event_ndvi_min must be in"):
            WindDamageConfig(pre_event_ndvi_min=-0.5)


class TestWindDamageDetection:
    """Test WindDamageDetection algorithm execution."""

    def test_basic_execution(self, synthetic_vegetation_data):
        """Test basic algorithm execution."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        assert result is not None
        assert isinstance(result, WindDamageResult)
        assert result.damage_extent.shape == red_pre.shape
        assert result.damage_extent.dtype == bool
        assert result.damage_severity.dtype == np.uint8
        assert result.confidence_raster.dtype == np.float32
        assert result.statistics['damage_area_ha'] >= 0

    def test_detects_damage_region(self, synthetic_vegetation_data):
        """Test that damage region is detected."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Check that some damage is detected
        assert result.damage_extent.any()
        assert result.statistics['damage_pixels'] > 0

    def test_severity_classification(self, synthetic_vegetation_data):
        """Test damage severity classification."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Check severity values are in expected range (0-3)
        assert result.damage_severity.min() >= 0
        assert result.damage_severity.max() <= 3

        # Check statistics include severity breakdown
        assert 'severity_counts' in result.statistics
        assert 'minor' in result.statistics['severity_counts']
        assert 'moderate' in result.statistics['severity_counts']
        assert 'severe' in result.statistics['severity_counts']

    def test_without_severity_classification(self, synthetic_vegetation_data):
        """Test execution without severity classification."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        config = WindDamageConfig(use_severity_classification=False)
        algo = WindDamageDetection(config)
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Without classification, severity should be binary (0 or 1)
        unique_values = np.unique(result.damage_severity)
        assert len(unique_values) <= 2
        assert all(v in [0, 1] for v in unique_values)

    def test_with_evi_index(self, synthetic_vegetation_data):
        """Test execution with EVI vegetation index."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        # Create blue band data
        np.random.seed(42)
        blue_pre = np.random.uniform(0.02, 0.06, red_pre.shape).astype(np.float32)
        blue_post = np.random.uniform(0.03, 0.08, red_pre.shape).astype(np.float32)

        config = WindDamageConfig(vegetation_index="evi")
        algo = WindDamageDetection(config)
        result = algo.execute(
            red_pre, nir_pre, red_post, nir_post,
            blue_pre=blue_pre, blue_post=blue_post,
            pixel_size_m=10.0
        )

        assert result is not None
        assert result.metadata['execution']['index_used'] == 'EVI'

    def test_with_cloud_mask(self, synthetic_vegetation_data):
        """Test execution with cloud mask."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        # Create cloud mask (True = cloudy)
        cloud_mask = np.zeros(red_pre.shape, dtype=bool)
        cloud_mask[0:20, 0:20] = True  # Top-left corner is cloudy

        algo = WindDamageDetection()
        result = algo.execute(
            red_pre, nir_pre, red_post, nir_post,
            cloud_mask=cloud_mask,
            pixel_size_m=10.0
        )

        # Cloudy area should not be detected as damage
        assert not result.damage_extent[0:20, 0:20].any()

    def test_with_nodata_value(self, synthetic_vegetation_data):
        """Test execution with nodata masking."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        # Set some pixels to nodata
        nodata = -9999.0
        red_pre[0:10, 0:10] = nodata

        algo = WindDamageDetection()
        result = algo.execute(
            red_pre, nir_pre, red_post, nir_post,
            pixel_size_m=10.0,
            nodata_value=nodata
        )

        # Nodata area should not be detected as damage
        assert not result.damage_extent[0:10, 0:10].any()

    def test_confidence_range(self, synthetic_vegetation_data):
        """Test that confidence values are in valid range."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        assert result.confidence_raster.min() >= 0.0
        assert result.confidence_raster.max() <= 1.0

    def test_reproducibility(self, deterministic_vegetation_data):
        """Test that algorithm produces identical results with same input."""
        red_pre, nir_pre, red_post, nir_post = deterministic_vegetation_data

        algo1 = WindDamageDetection()
        result1 = algo1.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        algo2 = WindDamageDetection()
        result2 = algo2.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        np.testing.assert_array_equal(result1.damage_extent, result2.damage_extent)
        np.testing.assert_array_equal(result1.damage_severity, result2.damage_severity)
        np.testing.assert_array_almost_equal(
            result1.confidence_raster, result2.confidence_raster
        )

    def test_metadata(self, synthetic_vegetation_data):
        """Test that metadata is correctly populated."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        assert result.metadata['id'] == 'storm.baseline.wind_damage'
        assert result.metadata['version'] == '1.0.0'
        assert result.metadata['deterministic'] is True
        assert 'parameters' in result.metadata
        assert 'execution' in result.metadata

    def test_statistics_calculation(self, synthetic_vegetation_data):
        """Test that statistics are correctly calculated."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        stats = result.statistics
        assert 'total_pixels' in stats
        assert 'valid_pixels' in stats
        assert 'vegetated_pixels' in stats
        assert 'damage_pixels' in stats
        assert 'damage_area_ha' in stats
        assert 'damage_percent_of_vegetated' in stats
        assert 'mean_vi_change' in stats
        assert 'mean_damage_confidence' in stats

        # Verify consistency
        assert stats['damage_pixels'] <= stats['vegetated_pixels']
        assert stats['vegetated_pixels'] <= stats['valid_pixels']
        assert stats['valid_pixels'] <= stats['total_pixels']

    def test_get_metadata(self):
        """Test static get_metadata method."""
        metadata = WindDamageDetection.get_metadata()
        assert metadata['id'] == 'storm.baseline.wind_damage'
        assert 'storm.*' in metadata['event_types']

    def test_create_from_dict(self):
        """Test creating algorithm from parameter dictionary."""
        params = {
            'ndvi_change_threshold': -0.3,
            'min_damage_area_ha': 1.0,
            'vegetation_index': 'ndvi'
        }
        algo = WindDamageDetection.create_from_dict(params)
        assert algo.config.ndvi_change_threshold == -0.3
        assert algo.config.min_damage_area_ha == 1.0

    def test_to_dict(self, synthetic_vegetation_data):
        """Test result to_dict method."""
        red_pre, nir_pre, red_post, nir_post = synthetic_vegetation_data

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        result_dict = result.to_dict()
        assert 'damage_extent' in result_dict
        assert 'damage_severity' in result_dict
        assert 'confidence_raster' in result_dict
        assert 'metadata' in result_dict
        assert 'statistics' in result_dict


class TestWindDamageEdgeCases:
    """Test edge cases for WindDamageDetection."""

    def test_no_vegetation(self):
        """Test with no vegetation (very low NDVI everywhere)."""
        np.random.seed(42)
        shape = (50, 50)

        # Non-vegetated area (similar red and NIR, low NDVI)
        red_pre = np.random.uniform(0.2, 0.3, shape).astype(np.float32)
        nir_pre = np.random.uniform(0.2, 0.3, shape).astype(np.float32)
        red_post = red_pre.copy()
        nir_post = nir_pre.copy()

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Should detect minimal or no damage (no vegetation to damage)
        assert result.statistics['vegetated_pixels'] == 0 or result.statistics['damage_pixels'] == 0

    def test_no_change(self):
        """Test with identical pre and post images."""
        np.random.seed(42)
        shape = (50, 50)

        # Healthy vegetation, no change
        red = np.random.uniform(0.03, 0.08, shape).astype(np.float32)
        nir = np.random.uniform(0.35, 0.55, shape).astype(np.float32)

        algo = WindDamageDetection()
        result = algo.execute(red, nir, red.copy(), nir.copy(), pixel_size_m=10.0)

        # Should detect no damage
        assert result.statistics['damage_pixels'] == 0

    def test_small_image(self):
        """Test with very small image (2x2)."""
        red_pre = np.array([[0.05, 0.05], [0.05, 0.05]], dtype=np.float32)
        nir_pre = np.array([[0.4, 0.4], [0.4, 0.4]], dtype=np.float32)
        red_post = np.array([[0.05, 0.15], [0.05, 0.05]], dtype=np.float32)
        nir_post = np.array([[0.4, 0.2], [0.4, 0.4]], dtype=np.float32)

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        assert result.damage_extent.shape == (2, 2)

    def test_single_pixel(self):
        """Test with single pixel image."""
        red_pre = np.array([[0.05]], dtype=np.float32)
        nir_pre = np.array([[0.4]], dtype=np.float32)
        red_post = np.array([[0.15]], dtype=np.float32)
        nir_post = np.array([[0.2]], dtype=np.float32)

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        assert result.damage_extent.shape == (1, 1)

    def test_all_nan(self):
        """Test with all NaN input."""
        shape = (10, 10)
        red_pre = np.full(shape, np.nan, dtype=np.float32)
        nir_pre = np.full(shape, np.nan, dtype=np.float32)
        red_post = np.full(shape, np.nan, dtype=np.float32)
        nir_post = np.full(shape, np.nan, dtype=np.float32)

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Should have no valid pixels
        assert result.statistics['valid_pixels'] == 0
        assert result.statistics['damage_pixels'] == 0

    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid pixels."""
        np.random.seed(42)
        shape = (20, 20)

        red_pre = np.random.uniform(0.03, 0.08, shape).astype(np.float32)
        nir_pre = np.random.uniform(0.35, 0.55, shape).astype(np.float32)
        red_post = red_pre.copy()
        nir_post = nir_pre.copy()

        # Add some NaN pixels
        red_pre[0:5, 0:5] = np.nan
        # Add some Inf pixels
        nir_post[15:20, 15:20] = np.inf

        algo = WindDamageDetection()
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Should exclude invalid pixels
        assert result.statistics['valid_pixels'] < shape[0] * shape[1]
        # Invalid areas should not be marked as damage
        assert not result.damage_extent[0:5, 0:5].any()
        assert not result.damage_extent[15:20, 15:20].any()

    def test_input_shape_mismatch(self):
        """Test that mismatched input shapes raise error."""
        red_pre = np.zeros((10, 10), dtype=np.float32)
        nir_pre = np.zeros((10, 10), dtype=np.float32)
        red_post = np.zeros((10, 10), dtype=np.float32)
        nir_post = np.zeros((20, 20), dtype=np.float32)  # Wrong shape

        algo = WindDamageDetection()
        with pytest.raises(ValueError, match="shape .* doesn't match"):
            algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

    def test_3d_input_raises_error(self):
        """Test that 3D input raises error."""
        data_3d = np.zeros((10, 10, 3), dtype=np.float32)

        algo = WindDamageDetection()
        with pytest.raises(ValueError, match="Expected 2D"):
            algo.execute(data_3d, data_3d, data_3d, data_3d, pixel_size_m=10.0)

    def test_zero_threshold(self):
        """Test with zero threshold (edge case for confidence calculation)."""
        np.random.seed(42)
        shape = (20, 20)

        red_pre = np.random.uniform(0.03, 0.08, shape).astype(np.float32)
        nir_pre = np.random.uniform(0.35, 0.55, shape).astype(np.float32)
        red_post = red_pre.copy()
        nir_post = nir_pre.copy()

        # Add some damage
        red_post[5:15, 5:15] = np.random.uniform(0.12, 0.18, (10, 10))
        nir_post[5:15, 5:15] = np.random.uniform(0.18, 0.25, (10, 10))

        config = WindDamageConfig(ndvi_change_threshold=0.0)
        algo = WindDamageDetection(config)
        result = algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Should not raise division by zero
        assert np.all(np.isfinite(result.confidence_raster))


# ============================================================================
# StructuralDamageAssessment Tests
# ============================================================================

class TestStructuralDamageConfig:
    """Test StructuralDamageConfig validation."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = StructuralDamageConfig()
        assert config.texture_analysis is True
        assert config.building_mask_required is False
        assert config.damage_categories == 4
        assert config.texture_kernel_size == 5
        assert config.brightness_weight == 0.4
        assert config.texture_weight == 0.6
        assert config.min_damage_area_m2 == 50.0
        assert config.use_sar_coherence is False

    def test_config_damage_categories_valid(self):
        """Test valid damage_categories values."""
        for n in [1, 2, 3, 4, 5]:
            config = StructuralDamageConfig(damage_categories=n)
            assert config.damage_categories == n

    def test_config_damage_categories_invalid(self):
        """Test invalid damage_categories values."""
        with pytest.raises(ValueError, match="damage_categories must be in"):
            StructuralDamageConfig(damage_categories=0)

        with pytest.raises(ValueError, match="damage_categories must be in"):
            StructuralDamageConfig(damage_categories=6)

    def test_config_kernel_size_valid(self):
        """Test valid texture_kernel_size values."""
        for size in [3, 5, 7, 9]:
            config = StructuralDamageConfig(texture_kernel_size=size)
            assert config.texture_kernel_size == size

    def test_config_kernel_size_invalid(self):
        """Test invalid texture_kernel_size values."""
        with pytest.raises(ValueError, match="texture_kernel_size must be"):
            StructuralDamageConfig(texture_kernel_size=4)

    def test_config_weights_must_sum_to_one(self):
        """Test that brightness_weight + texture_weight must equal 1.0."""
        # Valid combination
        config = StructuralDamageConfig(brightness_weight=0.3, texture_weight=0.7)
        assert config.brightness_weight + config.texture_weight == 1.0

        # Invalid combination
        with pytest.raises(ValueError, match="must equal 1.0"):
            StructuralDamageConfig(brightness_weight=0.5, texture_weight=0.3)

    def test_config_min_area_invalid(self):
        """Test that negative min_damage_area_m2 raises error."""
        with pytest.raises(ValueError, match="min_damage_area_m2 must be non-negative"):
            StructuralDamageConfig(min_damage_area_m2=-10.0)


class TestStructuralDamageAssessment:
    """Test StructuralDamageAssessment algorithm execution."""

    def test_basic_execution(self, synthetic_structural_data):
        """Test basic algorithm execution."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        assert result is not None
        assert isinstance(result, StructuralDamageResult)
        assert result.damage_map.shape == optical_pre.shape
        assert result.damage_map.dtype == bool
        assert result.damage_grade.dtype == np.uint8
        assert result.confidence_map.dtype == np.float32
        assert result.statistics['damage_area_m2'] >= 0

    def test_detects_damage_region(self, synthetic_structural_data):
        """Test that damage region is detected."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Check that some damage is detected
        assert result.damage_map.any()
        assert result.statistics['damage_pixels'] > 0

    def test_damage_grades(self, synthetic_structural_data):
        """Test damage grade classification."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Check grades are in valid range
        assert result.damage_grade.min() >= 0
        assert result.damage_grade.max() <= 4

        # Check grade distribution in statistics
        assert 'grade_distribution' in result.statistics
        assert 'grade_areas' in result.statistics

    def test_with_building_mask(self, synthetic_structural_data):
        """Test execution with building footprint mask."""
        optical_pre, optical_post = synthetic_structural_data

        # Create building mask (True = building)
        building_mask = np.zeros(optical_pre.shape, dtype=bool)
        building_mask[20:80, 20:80] = True

        algo = StructuralDamageAssessment()
        result = algo.execute(
            optical_pre, optical_post,
            building_mask=building_mask,
            pixel_size_m=1.0
        )

        # Analysis should be restricted to building areas
        assert result.metadata['execution']['building_mask_used'] is True

    def test_building_mask_required(self):
        """Test that missing building mask raises error when required."""
        optical_pre = np.random.uniform(0.3, 0.6, (50, 50)).astype(np.float32)
        optical_post = optical_pre.copy()

        config = StructuralDamageConfig(building_mask_required=True)
        algo = StructuralDamageAssessment(config)

        with pytest.raises(ValueError, match="building_mask_required=True but no building_mask"):
            algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

    def test_without_texture_analysis(self, synthetic_structural_data):
        """Test execution without texture analysis."""
        optical_pre, optical_post = synthetic_structural_data

        config = StructuralDamageConfig(
            texture_analysis=False,
            brightness_weight=1.0,
            texture_weight=0.0
        )
        algo = StructuralDamageAssessment(config)
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Texture change should be all zeros
        assert np.all(result.texture_change == 0.0)

    def test_with_sar_coherence(self, synthetic_structural_data):
        """Test execution with SAR coherence data."""
        optical_pre, optical_post = synthetic_structural_data
        shape = optical_pre.shape

        # Create coherence data
        np.random.seed(42)
        coherence_pre = np.random.uniform(0.7, 0.95, shape).astype(np.float32)
        coherence_post = coherence_pre.copy()
        # Simulate coherence loss in damage area
        coherence_post[25:50, 25:50] = np.random.uniform(0.2, 0.4, (25, 25))

        config = StructuralDamageConfig(use_sar_coherence=True)
        algo = StructuralDamageAssessment(config)
        result = algo.execute(
            optical_pre, optical_post,
            sar_coherence_pre=coherence_pre,
            sar_coherence_post=coherence_post,
            pixel_size_m=1.0
        )

        assert result.metadata['execution']['sar_coherence_used'] is True

    def test_multiband_input(self):
        """Test execution with multi-band (RGB) input."""
        np.random.seed(42)
        shape = (50, 50, 3)

        optical_pre = np.random.uniform(0.3, 0.6, shape).astype(np.float32)
        optical_post = optical_pre.copy()
        # Add damage in one area
        optical_post[10:30, 10:30] = np.random.uniform(0.1, 0.9, (20, 20, 3))

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Should process and return 2D output
        assert result.damage_map.ndim == 2

    def test_confidence_range(self, synthetic_structural_data):
        """Test that confidence values are in valid range."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        assert result.confidence_map.min() >= 0.0
        assert result.confidence_map.max() <= 1.0

    def test_reproducibility(self, deterministic_structural_data):
        """Test that algorithm produces identical results with same input."""
        optical_pre, optical_post = deterministic_structural_data

        algo1 = StructuralDamageAssessment()
        result1 = algo1.execute(optical_pre, optical_post, pixel_size_m=1.0)

        algo2 = StructuralDamageAssessment()
        result2 = algo2.execute(optical_pre, optical_post, pixel_size_m=1.0)

        np.testing.assert_array_equal(result1.damage_map, result2.damage_map)
        np.testing.assert_array_equal(result1.damage_grade, result2.damage_grade)
        np.testing.assert_array_almost_equal(
            result1.confidence_map, result2.confidence_map
        )

    def test_metadata(self, synthetic_structural_data):
        """Test that metadata is correctly populated."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        assert result.metadata['id'] == 'storm.baseline.structural_damage'
        assert result.metadata['version'] == '1.0.0'
        assert result.metadata['deterministic'] is True
        assert 'parameters' in result.metadata
        assert 'execution' in result.metadata

    def test_get_metadata(self):
        """Test static get_metadata method."""
        metadata = StructuralDamageAssessment.get_metadata()
        assert metadata['id'] == 'storm.baseline.structural_damage'
        assert 'storm.*' in metadata['event_types']

    def test_create_from_dict(self):
        """Test creating algorithm from parameter dictionary."""
        params = {
            'texture_analysis': False,
            'damage_categories': 3,
            'brightness_weight': 1.0,
            'texture_weight': 0.0
        }
        algo = StructuralDamageAssessment.create_from_dict(params)
        assert algo.config.texture_analysis is False
        assert algo.config.damage_categories == 3

    def test_to_dict(self, synthetic_structural_data):
        """Test result to_dict method."""
        optical_pre, optical_post = synthetic_structural_data

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        result_dict = result.to_dict()
        assert 'damage_map' in result_dict
        assert 'damage_grade' in result_dict
        assert 'confidence_map' in result_dict
        assert 'metadata' in result_dict
        assert 'statistics' in result_dict


class TestStructuralDamageEdgeCases:
    """Test edge cases for StructuralDamageAssessment."""

    def test_no_change(self):
        """Test with identical pre and post images."""
        np.random.seed(42)
        shape = (50, 50)

        optical = np.random.uniform(0.3, 0.6, shape).astype(np.float32)

        algo = StructuralDamageAssessment()
        result = algo.execute(optical, optical.copy(), pixel_size_m=1.0)

        # Should detect minimal or no damage
        assert result.statistics['damage_pixels'] == 0 or result.statistics['damage_percent'] < 5.0

    def test_small_image(self):
        """Test with very small image (5x5)."""
        np.random.seed(42)
        optical_pre = np.random.uniform(0.3, 0.6, (5, 5)).astype(np.float32)
        optical_post = optical_pre.copy()
        optical_post[2:4, 2:4] = np.random.uniform(0.1, 0.9, (2, 2))

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        assert result.damage_map.shape == (5, 5)

    def test_all_nan(self):
        """Test with all NaN input."""
        shape = (10, 10)
        optical_pre = np.full(shape, np.nan, dtype=np.float32)
        optical_post = np.full(shape, np.nan, dtype=np.float32)

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Should have no damage detected
        assert result.statistics['damage_pixels'] == 0

    def test_mixed_valid_invalid(self):
        """Test with mix of valid and invalid pixels."""
        np.random.seed(42)
        shape = (20, 20)

        optical_pre = np.random.uniform(0.3, 0.6, shape).astype(np.float32)
        optical_post = optical_pre.copy()

        # Add some NaN pixels
        optical_pre[0:5, 0:5] = np.nan
        # Add some Inf pixels
        optical_post[15:20, 15:20] = np.inf

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Invalid areas should not be marked as damage
        assert not result.damage_map[0:5, 0:5].any()
        assert not result.damage_map[15:20, 15:20].any()

    def test_input_shape_mismatch(self):
        """Test that mismatched input shapes raise error."""
        optical_pre = np.zeros((10, 10), dtype=np.float32)
        optical_post = np.zeros((20, 20), dtype=np.float32)

        algo = StructuralDamageAssessment()
        with pytest.raises(ValueError, match="shape mismatch"):
            algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

    def test_high_value_normalization(self):
        """Test that high pixel values are normalized."""
        np.random.seed(42)
        shape = (20, 20)

        # Values in 0-255 range (typical 8-bit)
        optical_pre = np.random.uniform(75, 150, shape).astype(np.float32)
        optical_post = optical_pre.copy()
        optical_post[5:15, 5:15] = np.random.uniform(25, 225, (10, 10))

        algo = StructuralDamageAssessment()
        result = algo.execute(optical_pre, optical_post, pixel_size_m=1.0)

        # Should not raise errors and should handle normalization
        assert result is not None


# ============================================================================
# Module Export Tests
# ============================================================================

class TestModuleExports:
    """Test module exports and registry."""

    def test_storm_algorithms_dict(self):
        """Test STORM_ALGORITHMS dictionary."""
        assert 'storm.baseline.wind_damage' in STORM_ALGORITHMS
        assert 'storm.baseline.structural_damage' in STORM_ALGORITHMS
        assert STORM_ALGORITHMS['storm.baseline.wind_damage'] == WindDamageDetection
        assert STORM_ALGORITHMS['storm.baseline.structural_damage'] == StructuralDamageAssessment

    def test_get_algorithm(self):
        """Test get_algorithm function."""
        algo_cls = get_algorithm('storm.baseline.wind_damage')
        assert algo_cls == WindDamageDetection

        algo_cls = get_algorithm('storm.baseline.structural_damage')
        assert algo_cls == StructuralDamageAssessment

    def test_get_algorithm_invalid(self):
        """Test get_algorithm with invalid ID."""
        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm('invalid.algorithm.id')

    def test_list_algorithms(self):
        """Test list_algorithms function."""
        algos = list_algorithms()
        assert len(algos) == 2

        ids = [id for id, cls in algos]
        assert 'storm.baseline.wind_damage' in ids
        assert 'storm.baseline.structural_damage' in ids


# ============================================================================
# Integration Tests
# ============================================================================

class TestStormAlgorithmIntegration:
    """Integration tests for storm algorithms."""

    def test_workflow_vegetation_to_structural(self):
        """Test running both algorithms on same area."""
        np.random.seed(42)
        shape = (100, 100)

        # Create optical data suitable for both algorithms
        red_pre = np.random.uniform(0.05, 0.15, shape).astype(np.float32)
        nir_pre = np.random.uniform(0.30, 0.50, shape).astype(np.float32)
        red_post = red_pre.copy()
        nir_post = nir_pre.copy()

        # Add damage region
        damage_region = (30, 70, 30, 70)
        r1, r2, c1, c2 = damage_region
        red_post[r1:r2, c1:c2] = np.random.uniform(0.15, 0.25, (r2 - r1, c2 - c1))
        nir_post[r1:r2, c1:c2] = np.random.uniform(0.15, 0.25, (r2 - r1, c2 - c1))

        # Create grayscale for structural analysis
        optical_pre = (red_pre + nir_pre) / 2
        optical_post = (red_post + nir_post) / 2

        # Run wind damage detection
        wind_algo = WindDamageDetection()
        wind_result = wind_algo.execute(red_pre, nir_pre, red_post, nir_post, pixel_size_m=10.0)

        # Run structural damage assessment
        struct_algo = StructuralDamageAssessment()
        struct_result = struct_algo.execute(optical_pre, optical_post, pixel_size_m=10.0)

        # Both should detect damage
        assert wind_result.damage_extent.any()
        assert struct_result.damage_map.any()

    def test_algorithm_metadata_consistency(self):
        """Test that algorithm metadata is consistent between YAML and code."""
        # Wind damage
        wind_meta = WindDamageDetection.get_metadata()
        assert wind_meta['id'] == 'storm.baseline.wind_damage'
        assert wind_meta['deterministic'] is True
        assert 'storm.*' in wind_meta['event_types']

        # Structural damage
        struct_meta = StructuralDamageAssessment.get_metadata()
        assert struct_meta['id'] == 'storm.baseline.structural_damage'
        assert struct_meta['deterministic'] is True
        assert 'storm.*' in struct_meta['event_types']
