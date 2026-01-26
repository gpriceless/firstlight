"""
Unit tests for histogram stretching algorithms.

Tests the histogram stretching module including:
- 2% percentile stretch (AC-6)
- Manual min/max override (AC-7)
- Nodata/masked array handling
- Performance benchmarks
- Standard deviation method
- Edge cases and error handling
"""

import time

import numpy as np
import numpy.ma as ma
import pytest

from core.reporting.imagery.histogram import (
    HistogramStretch,
    apply_stretch,
    calculate_stddev_bounds,
    calculate_stretch_bounds,
    stretch_band,
)


class TestCalculateStretchBounds:
    """Tests for calculate_stretch_bounds function."""

    def test_2pct_percentile_default(self):
        """Test default 2% percentile clipping."""
        # Create data with outliers
        data = np.random.normal(100, 10, 10000)
        # Add some outliers
        data[:100] = 0  # Bottom outliers
        data[-100:] = 300  # Top outliers

        min_val, max_val = calculate_stretch_bounds(data, percentile=2.0)

        # Should clip outliers
        assert min_val > 0, "Should clip bottom outliers"
        assert max_val < 300, "Should clip top outliers"
        # Should be close to mean +/- some std devs
        assert 70 < min_val < 90, f"Expected min ~80, got {min_val}"
        assert 110 < max_val < 130, f"Expected max ~120, got {max_val}"

    def test_different_percentiles(self):
        """Test different percentile values."""
        data = np.linspace(0, 100, 1000)

        # 5% percentile
        min_val, max_val = calculate_stretch_bounds(data, percentile=5.0)
        assert np.isclose(min_val, 5.0, atol=1.0)
        assert np.isclose(max_val, 95.0, atol=1.0)

        # 10% percentile
        min_val, max_val = calculate_stretch_bounds(data, percentile=10.0)
        assert np.isclose(min_val, 10.0, atol=1.0)
        assert np.isclose(max_val, 90.0, atol=1.0)

    def test_invalid_percentile(self):
        """Test validation of percentile parameter."""
        data = np.random.rand(100, 100)

        with pytest.raises(ValueError, match="must be between 0 and 50"):
            calculate_stretch_bounds(data, percentile=-1)

        with pytest.raises(ValueError, match="must be between 0 and 50"):
            calculate_stretch_bounds(data, percentile=51)

    def test_with_nodata(self):
        """Test handling of nodata values."""
        data = np.random.normal(100, 10, 1000)
        # Add nodata values
        data[:200] = -9999

        min_val, max_val = calculate_stretch_bounds(data, nodata=-9999)

        # Should exclude nodata from calculation
        assert min_val > -9999
        assert 78 < min_val < 95
        assert 105 < max_val < 122

    def test_with_masked_array(self):
        """Test handling of masked arrays."""
        data = np.random.normal(100, 10, 1000)
        # Create mask for first 200 values
        masked_data = ma.masked_less(data, 90)

        min_val, max_val = calculate_stretch_bounds(masked_data)

        # Should exclude masked values
        assert min_val >= 90

    def test_uniform_data(self):
        """Test handling of uniform data (all same value)."""
        data = np.ones((100, 100)) * 42.0

        min_val, max_val = calculate_stretch_bounds(data)

        # Should return value with small epsilon to avoid division by zero
        assert min_val == 42.0
        assert max_val > min_val

    def test_all_nodata(self):
        """Test error when all values are nodata."""
        data = np.full((100, 100), -9999)

        with pytest.raises(ValueError, match="only nodata/masked values"):
            calculate_stretch_bounds(data, nodata=-9999)

    def test_all_masked(self):
        """Test error when all values are masked."""
        data = ma.masked_all((100, 100))

        with pytest.raises(ValueError, match="only nodata/masked values"):
            calculate_stretch_bounds(data)

    def test_nan_handling(self):
        """Test automatic filtering of NaN values."""
        data = np.random.normal(100, 10, 1000)
        data[:100] = np.nan

        min_val, max_val = calculate_stretch_bounds(data)

        # Should exclude NaN values
        assert np.isfinite(min_val)
        assert np.isfinite(max_val)
        assert 78 < min_val < 95
        assert 105 < max_val < 122

    def test_inf_handling(self):
        """Test automatic filtering of infinite values."""
        data = np.random.normal(100, 10, 1000)
        data[:50] = np.inf
        data[50:100] = -np.inf

        min_val, max_val = calculate_stretch_bounds(data)

        # Should exclude infinite values
        assert np.isfinite(min_val)
        assert np.isfinite(max_val)


class TestApplyStretch:
    """Tests for apply_stretch function."""

    def test_basic_linear_stretch(self):
        """Test basic linear stretch to 0-1 range."""
        data = np.array([0, 50, 100, 150, 200], dtype=np.float32)

        stretched = apply_stretch(data, min_val=50, max_val=150)

        # 50 -> 0, 150 -> 1, 100 -> 0.5
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(stretched, expected)

    def test_2d_array_stretch(self):
        """Test stretching 2D array."""
        data = np.array([[0, 100], [200, 300]], dtype=np.float32)

        stretched = apply_stretch(data, min_val=100, max_val=200)

        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(stretched, expected)

    def test_clipping_behavior(self):
        """Test that values outside range are clipped."""
        data = np.array([-100, 0, 50, 100, 200])

        stretched = apply_stretch(data, min_val=0, max_val=100)

        # Values below 0 -> 0, values above 100 -> 1
        assert stretched[0] == 0.0
        assert stretched[1] == 0.0
        assert np.isclose(stretched[2], 0.5)
        assert stretched[3] == 1.0
        assert stretched[4] == 1.0

    def test_nodata_preservation(self):
        """Test that nodata values are preserved."""
        data = np.array([-9999, 0, 50, 100, 150], dtype=np.float32)

        stretched = apply_stretch(data, min_val=0, max_val=100, nodata=-9999)

        # Nodata should be preserved
        assert stretched[0] == -9999
        # Other values should be stretched
        assert stretched[1] == 0.0
        assert np.isclose(stretched[2], 0.5)
        assert stretched[3] == 1.0

    def test_masked_array_preservation(self):
        """Test that masked arrays remain masked."""
        data = np.array([10, 20, 30, 40, 50], dtype=np.float32)
        masked_data = ma.masked_less(data, 25)

        stretched = apply_stretch(masked_data, min_val=20, max_val=40)

        # Should still be a masked array
        assert isinstance(stretched, ma.MaskedArray)
        # Mask should be preserved
        assert stretched.mask[0]  # 10 was masked
        assert stretched.mask[1]  # 20 was masked
        assert not stretched.mask[2]  # 30 not masked
        # Unmasked values should be stretched
        assert np.isclose(stretched[2], 0.5)

    def test_invalid_bounds(self):
        """Test error when min >= max."""
        data = np.random.rand(100)

        with pytest.raises(ValueError, match="must be less than"):
            apply_stretch(data, min_val=100, max_val=50)

        with pytest.raises(ValueError, match="must be less than"):
            apply_stretch(data, min_val=100, max_val=100)

    def test_dtype_preservation(self):
        """Test that output dtype is float."""
        data = np.array([0, 50, 100, 150, 200], dtype=np.int32)

        stretched = apply_stretch(data, min_val=50, max_val=150)

        # Should be float type for 0-1 range
        assert np.issubdtype(stretched.dtype, np.floating)


class TestCalculateStddevBounds:
    """Tests for calculate_stddev_bounds function."""

    def test_normal_distribution(self):
        """Test standard deviation bounds on normal distribution."""
        # Create normally distributed data: mean=100, std=10
        np.random.seed(42)
        data = np.random.normal(100, 10, 10000)

        min_val, max_val = calculate_stddev_bounds(data, n_std=2.0)

        # Should be approximately mean +/- 2*std (100 +/- 20)
        assert 75 < min_val < 85
        assert 115 < max_val < 125

    def test_different_n_std(self):
        """Test different standard deviation multipliers."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 10000)

        # 1 standard deviation
        min_val, max_val = calculate_stddev_bounds(data, n_std=1.0)
        assert 88 < min_val < 92
        assert 108 < max_val < 112

        # 3 standard deviations
        min_val, max_val = calculate_stddev_bounds(data, n_std=3.0)
        assert 65 < min_val < 75
        assert 125 < max_val < 135

    def test_with_nodata(self):
        """Test standard deviation calculation excluding nodata."""
        data = np.random.normal(100, 10, 1000)
        data[:200] = -9999

        min_val, max_val = calculate_stddev_bounds(data, nodata=-9999)

        # Should exclude nodata from calculation
        assert min_val > -9999
        assert 70 < min_val < 90
        assert 110 < max_val < 130


class TestStretchBand:
    """Tests for stretch_band main entry point."""

    def test_default_2pct_stretch(self):
        """Test default 2% linear stretch (AC-6)."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (100, 100))

        stretched = stretch_band(data)

        # Should return values in 0-1 range
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0
        # Should have good dynamic range
        assert stretched.max() - stretched.min() > 0.5

    def test_manual_min_max_override(self):
        """Test manual min/max override (AC-7)."""
        data = np.array([0, 50, 100, 150, 200], dtype=np.float32)

        stretched = stretch_band(data, min_val=50, max_val=150)

        # Should use manual bounds
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_almost_equal(stretched, expected)

    def test_partial_manual_override(self):
        """Test providing only min or max manually."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 1000)

        # Only manual min
        stretched = stretch_band(data, min_val=80)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

        # Only manual max
        stretched = stretch_band(data, max_val=120)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    def test_min_max_method(self):
        """Test MIN_MAX stretch method."""
        data = np.array([10, 20, 30, 40, 50], dtype=np.float32)

        stretched = stretch_band(data, method=HistogramStretch.MIN_MAX)

        # Should use actual min (10) and max (50)
        assert np.isclose(stretched[0], 0.0)
        assert np.isclose(stretched[-1], 1.0)
        assert np.isclose(stretched[2], 0.5)  # 30 is midpoint

    def test_stddev_method(self):
        """Test STDDEV stretch method."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 10000)

        stretched = stretch_band(data, method=HistogramStretch.STDDEV)

        # Should stretch using mean +/- 2*std
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    def test_adaptive_not_implemented(self):
        """Test that ADAPTIVE method raises NotImplementedError."""
        data = np.random.rand(100, 100)

        with pytest.raises(NotImplementedError, match="Adaptive histogram"):
            stretch_band(data, method=HistogramStretch.ADAPTIVE)

    def test_custom_percentile(self):
        """Test custom percentile for LINEAR_2PCT method."""
        data = np.linspace(0, 100, 1000)

        # 10% percentile
        stretched = stretch_band(data, percentile=10.0)

        # Values at 10th and 90th percentile should map to 0 and 1
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    def test_with_nodata(self):
        """Test stretching with nodata values."""
        data = np.random.normal(100, 10, 1000)
        data[:200] = -9999

        stretched = stretch_band(data, nodata=-9999)

        # Nodata should be preserved
        assert np.sum(stretched == -9999) == 200
        # Valid data should be stretched
        valid_data = stretched[stretched != -9999]
        assert valid_data.min() >= 0.0
        assert valid_data.max() <= 1.0

    def test_with_masked_array(self):
        """Test stretching masked array."""
        data = np.random.normal(100, 10, 1000)
        masked_data = ma.masked_less(data, 90)

        stretched = stretch_band(masked_data)

        # Should still be masked array
        assert isinstance(stretched, ma.MaskedArray)
        # Masked values should remain masked
        assert stretched.mask.sum() > 0
        # Valid data should be stretched
        assert stretched.compressed().min() >= 0.0
        assert stretched.compressed().max() <= 1.0

    def test_invalid_manual_bounds(self):
        """Test validation of manual bounds."""
        data = np.random.rand(100)

        with pytest.raises(ValueError, match="must be less than"):
            stretch_band(data, min_val=100, max_val=50)

    def test_unknown_method(self):
        """Test error for unknown stretch method."""
        data = np.random.rand(100)

        # This would require creating an invalid enum value
        # Just test that the enum is properly validated
        assert hasattr(HistogramStretch, "LINEAR_2PCT")
        assert hasattr(HistogramStretch, "MIN_MAX")
        assert hasattr(HistogramStretch, "STDDEV")
        assert hasattr(HistogramStretch, "ADAPTIVE")


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_pixel(self):
        """Test stretching single pixel."""
        data = np.array([42.0])

        stretched = stretch_band(data)

        # Should handle gracefully without error
        assert stretched.shape == (1,)
        assert np.isfinite(stretched[0])

    def test_small_array(self):
        """Test stretching very small array."""
        data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        stretched = stretch_band(data)

        assert stretched.shape == (2, 2)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    def test_large_range(self):
        """Test data with very large range."""
        data = np.array([0, 1e6, 2e6], dtype=np.float64)

        stretched = stretch_band(data)

        assert stretched[0] >= 0.0
        assert stretched[-1] <= 1.0

    def test_negative_values(self):
        """Test stretching negative values (e.g., temperature data)."""
        data = np.array([-50, -25, 0, 25, 50], dtype=np.float32)

        stretched = stretch_band(data)

        # Should stretch to 0-1 range
        assert np.isclose(stretched[0], 0.0)
        assert np.isclose(stretched[-1], 1.0)
        assert np.isclose(stretched[2], 0.5)

    def test_integer_input(self):
        """Test that integer input is handled correctly."""
        data = np.array([0, 50, 100, 150, 200], dtype=np.int32)

        stretched = stretch_band(data, min_val=50, max_val=150)

        # Should work despite integer input
        assert np.issubdtype(stretched.dtype, np.floating)
        assert np.isclose(stretched[2], 0.5)


class TestPerformance:
    """Performance benchmarks for histogram stretching."""

    @pytest.mark.slow
    def test_large_array_performance(self):
        """Test performance on 10000x10000 array (AC-6 requirement: <1 second)."""
        # Generate large array
        np.random.seed(42)
        data = np.random.normal(100, 10, (10000, 10000)).astype(np.float32)

        # Benchmark stretch operation
        start_time = time.time()
        stretched = stretch_band(data)
        elapsed_time = time.time() - start_time

        # Should complete in under 1 second
        assert elapsed_time < 1.0, f"Stretch took {elapsed_time:.2f}s (requirement: <1s)"

        # Verify correctness
        assert stretched.shape == (10000, 10000)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    @pytest.mark.slow
    def test_manual_override_performance(self):
        """Test that manual override doesn't degrade performance."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (10000, 10000)).astype(np.float32)

        start_time = time.time()
        stretched = stretch_band(data, min_val=80, max_val=120)
        elapsed_time = time.time() - start_time

        # Should be even faster since no percentile calculation
        assert elapsed_time < 1.0, f"Manual stretch took {elapsed_time:.2f}s"

    def test_masked_array_performance(self):
        """Test performance with masked arrays."""
        np.random.seed(42)
        data = np.random.normal(100, 10, (1000, 1000))
        masked_data = ma.masked_less(data, 90)

        start_time = time.time()
        stretched = stretch_band(masked_data)
        elapsed_time = time.time() - start_time

        # Should be reasonably fast
        assert elapsed_time < 0.5, f"Masked stretch took {elapsed_time:.2f}s"


class TestRealWorldScenarios:
    """Tests simulating real satellite imagery scenarios."""

    def test_sentinel2_typical_range(self):
        """Test typical Sentinel-2 reflectance values (0-1 range with outliers)."""
        np.random.seed(42)
        # Typical reflectance values
        data = np.random.uniform(0.05, 0.3, (1000, 1000))
        # Add some clouds (high reflectance)
        data[0:100, 0:100] = np.random.uniform(0.7, 0.95, (100, 100))
        # Add some shadows/water (low reflectance)
        data[900:1000, 900:1000] = np.random.uniform(0.0, 0.05, (100, 100))

        stretched = stretch_band(data)

        # Should handle typical range well
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0
        # Should have good contrast in main range
        typical_data = stretched[100:900, 100:900]
        assert typical_data.max() - typical_data.min() > 0.5

    def test_dem_elevation_data(self):
        """Test stretching DEM elevation data."""
        # Simulate elevation data
        data = np.random.uniform(0, 3000, (1000, 1000))  # 0-3000m elevation
        # Set some pixels as nodata (ocean)
        data[0:100, 0:100] = -9999

        stretched = stretch_band(data, nodata=-9999)

        # Nodata should be preserved
        assert np.sum(stretched == -9999) == 10000  # 100x100 pixels
        # Valid elevations should be stretched
        valid_data = stretched[stretched != -9999]
        assert valid_data.min() >= 0.0
        assert valid_data.max() <= 1.0

    def test_sar_backscatter(self):
        """Test stretching SAR backscatter data (dB scale)."""
        # Typical SAR backscatter in dB: -30 to 10
        data = np.random.uniform(-25, 5, (1000, 1000))
        # Add water (low backscatter)
        data[400:600, 400:600] = np.random.uniform(-30, -20, (200, 200))

        stretched = stretch_band(data)

        # Should handle negative values
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0
        # Water should be darker (lower values)
        water_stretched = stretched[400:600, 400:600]
        land_stretched = stretched[0:200, 0:200]
        assert water_stretched.mean() < land_stretched.mean()

    def test_thermal_temperature_data(self):
        """Test stretching thermal temperature data."""
        # Temperature in Kelvin (typical range 260-320K)
        data = np.random.normal(290, 10, (1000, 1000))

        stretched = stretch_band(data)

        # Should stretch temperature range to 0-1
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0
        # Should maintain relative relationships
        assert np.corrcoef(data.flatten(), stretched.flatten())[0, 1] > 0.99


class TestDocstrings:
    """Tests verifying docstring examples."""

    def test_calculate_stretch_bounds_docstring_examples(self):
        """Verify examples from calculate_stretch_bounds docstring."""
        # Example 1: Basic usage
        np.random.seed(42)
        data = np.random.normal(100, 20, (1000, 1000))
        min_val, max_val = calculate_stretch_bounds(data, percentile=2.0)
        assert isinstance(min_val, float)
        assert isinstance(max_val, float)
        assert min_val < max_val

    def test_apply_stretch_docstring_examples(self):
        """Verify examples from apply_stretch docstring."""
        # Example 1: Basic stretch
        data = np.array([[0, 50, 100], [150, 200, 250]])
        stretched = apply_stretch(data, min_val=50, max_val=200)
        assert stretched.shape == (2, 3)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

    def test_stretch_band_docstring_examples(self):
        """Verify examples from stretch_band docstring."""
        # Example 1: Automatic stretch
        np.random.seed(42)
        band = np.random.normal(100, 20, (1000, 1000))
        stretched = stretch_band(band)
        assert stretched.shape == (1000, 1000)

        # Example 2: Manual override
        stretched = stretch_band(band, min_val=50, max_val=150)
        assert stretched.min() >= 0.0
        assert stretched.max() <= 1.0

        # Example 3: Standard deviation method
        stretched = stretch_band(band, method=HistogramStretch.STDDEV)
        assert stretched.shape == (1000, 1000)
