"""
Unit Tests for Imagery Processing Utilities (VIS-1.1 Task 1.3).

Tests utility functions for:
- normalize_to_uint8: Float to uint8 conversion
- stack_bands_to_rgb: Band combination
- handle_nodata: NoData detection and masking
- detect_partial_coverage: Coverage percentage calculation
- get_valid_data_bounds: Valid data bounding box

Includes edge cases: empty arrays, all nodata, NaN values, masked arrays.
"""

import numpy as np
import pytest

from core.reporting.imagery.utils import (
    detect_partial_coverage,
    get_valid_data_bounds,
    handle_nodata,
    normalize_to_uint8,
    stack_bands_to_rgb,
)


class TestNormalizeToUint8:
    """Tests for normalize_to_uint8 function."""

    def test_standard_normalization(self):
        """Test standard 0-1 to 0-255 conversion."""
        data = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = normalize_to_uint8(data, 0.0, 1.0)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, [0, 63, 127, 191, 255])

    def test_custom_range(self):
        """Test normalization with custom min/max range."""
        data = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        result = normalize_to_uint8(data, 0.0, 2000.0)

        assert result.dtype == np.uint8
        expected = [0, 63, 127, 191, 255]
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_clipping_out_of_range(self):
        """Test that values outside range are clipped."""
        data = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
        result = normalize_to_uint8(data, 0.0, 1.0)

        assert result.dtype == np.uint8
        assert result[0] == 0  # Clipped to minimum
        assert result[-1] == 255  # Clipped to maximum

    def test_constant_array(self):
        """Test handling of constant array (min_val == max_val)."""
        data = np.full((5, 5), 42.0)
        result = normalize_to_uint8(data, 42.0, 42.0)

        assert result.dtype == np.uint8
        # Should return middle gray when all values are the same
        np.testing.assert_array_equal(result, 127)

    def test_empty_array(self):
        """Test handling of empty array."""
        data = np.array([])
        result = normalize_to_uint8(data, 0.0, 1.0)

        assert result.dtype == np.uint8
        assert result.size == 0

    def test_2d_array(self):
        """Test normalization of 2D array."""
        data = np.array([[0.0, 0.5], [1.0, 0.25]])
        result = normalize_to_uint8(data, 0.0, 1.0)

        assert result.dtype == np.uint8
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, [[0, 127], [255, 63]])

    def test_preserves_shape(self):
        """Test that output shape matches input shape."""
        for shape in [(10,), (5, 5), (3, 4, 5)]:
            data = np.random.rand(*shape)
            result = normalize_to_uint8(data, 0.0, 1.0)
            assert result.shape == shape


class TestStackBandsToRGB:
    """Tests for stack_bands_to_rgb function."""

    def test_simple_stack(self):
        """Test basic band stacking."""
        red = np.array([[255, 0], [128, 64]], dtype=np.uint8)
        green = np.array([[0, 255], [128, 64]], dtype=np.uint8)
        blue = np.array([[0, 0], [255, 64]], dtype=np.uint8)

        result = stack_bands_to_rgb(red, green, blue)

        assert result.shape == (2, 2, 3)
        np.testing.assert_array_equal(result[0, 0, :], [255, 0, 0])  # Red pixel
        np.testing.assert_array_equal(result[0, 1, :], [0, 255, 0])  # Green pixel
        np.testing.assert_array_equal(result[1, 0, :], [128, 128, 255])  # Purple-ish

    def test_float_bands(self):
        """Test stacking float bands (e.g., reflectance values)."""
        red = np.array([[1.0, 0.0], [0.5, 0.25]])
        green = np.array([[0.0, 1.0], [0.5, 0.25]])
        blue = np.array([[0.0, 0.0], [1.0, 0.25]])

        result = stack_bands_to_rgb(red, green, blue)

        assert result.shape == (2, 2, 3)
        assert result.dtype == red.dtype
        np.testing.assert_array_equal(result[0, 0, :], [1.0, 0.0, 0.0])

    def test_mismatched_shapes_raise_error(self):
        """Test that mismatched band shapes raise ValueError."""
        red = np.zeros((10, 10))
        green = np.zeros((10, 10))
        blue = np.zeros((10, 12))  # Different size

        with pytest.raises(ValueError, match="same shape"):
            stack_bands_to_rgb(red, green, blue)

    def test_non_2d_bands_raise_error(self):
        """Test that non-2D bands raise ValueError."""
        red = np.zeros((10, 10, 3))  # 3D array
        green = np.zeros((10, 10))
        blue = np.zeros((10, 10))

        with pytest.raises(ValueError, match="2D"):
            stack_bands_to_rgb(red, green, blue)

    def test_empty_bands(self):
        """Test stacking empty bands."""
        red = np.array([]).reshape(0, 0)
        green = np.array([]).reshape(0, 0)
        blue = np.array([]).reshape(0, 0)

        result = stack_bands_to_rgb(red, green, blue)

        assert result.shape == (0, 0, 3)

    def test_large_bands(self):
        """Test stacking realistic-sized bands."""
        shape = (1000, 1000)
        red = np.random.rand(*shape)
        green = np.random.rand(*shape)
        blue = np.random.rand(*shape)

        result = stack_bands_to_rgb(red, green, blue)

        assert result.shape == (1000, 1000, 3)
        # Verify each channel preserved
        np.testing.assert_array_equal(result[:, :, 0], red)
        np.testing.assert_array_equal(result[:, :, 1], green)
        np.testing.assert_array_equal(result[:, :, 2], blue)


class TestHandleNodata:
    """Tests for handle_nodata function."""

    def test_explicit_nodata_value(self):
        """Test handling of explicit nodata value."""
        data = np.array([1.0, 2.0, -9999.0, 4.0, -9999.0])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        np.testing.assert_array_equal(filled, [1.0, 2.0, 0.0, 4.0, 0.0])
        np.testing.assert_array_equal(mask, [True, True, False, True, False])

    def test_nan_values(self):
        """Test detection of NaN as nodata."""
        data = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        filled, mask = handle_nodata(data, fill=0.0)

        np.testing.assert_array_equal(filled, [1.0, 2.0, 0.0, 4.0, 0.0])
        np.testing.assert_array_equal(mask, [True, True, False, True, False])

    def test_masked_array(self):
        """Test handling of numpy masked arrays."""
        data = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0], mask=[0, 0, 1, 0, 1])
        filled, mask = handle_nodata(data, fill=0.0)

        np.testing.assert_array_equal(filled, [1.0, 2.0, 0.0, 4.0, 0.0])
        np.testing.assert_array_equal(mask, [True, True, False, True, False])

    def test_combined_nodata_sources(self):
        """Test combining explicit nodata, NaN, and masked array."""
        base_data = np.array([1.0, 2.0, -9999.0, 4.0, np.nan, 6.0])
        data = np.ma.array(base_data, mask=[0, 0, 0, 0, 0, 1])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        expected_filled = [1.0, 2.0, 0.0, 4.0, 0.0, 0.0]
        expected_mask = [True, True, False, True, False, False]
        np.testing.assert_array_equal(filled, expected_filled)
        np.testing.assert_array_equal(mask, expected_mask)

    def test_all_valid_data(self):
        """Test array with no nodata values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        np.testing.assert_array_equal(filled, data)
        np.testing.assert_array_equal(mask, [True, True, True, True, True])

    def test_all_nodata(self):
        """Test array with all nodata values."""
        data = np.array([-9999.0, -9999.0, -9999.0])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        np.testing.assert_array_equal(filled, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(mask, [False, False, False])

    def test_empty_array(self):
        """Test empty array handling."""
        data = np.array([])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        assert filled.size == 0
        assert mask.size == 0

    def test_2d_array(self):
        """Test 2D array handling."""
        data = np.array([[1.0, -9999.0], [3.0, 4.0]])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=0.0)

        np.testing.assert_array_equal(filled, [[1.0, 0.0], [3.0, 4.0]])
        np.testing.assert_array_equal(mask, [[True, False], [True, True]])

    def test_custom_fill_value(self):
        """Test using custom fill value."""
        data = np.array([1.0, -9999.0, 3.0])
        filled, mask = handle_nodata(data, nodata_value=-9999.0, fill=-1.0)

        np.testing.assert_array_equal(filled, [1.0, -1.0, 3.0])

    def test_nan_as_nodata_value(self):
        """Test explicit NaN as nodata value parameter."""
        data = np.array([1.0, np.nan, 3.0, np.nan])
        filled, mask = handle_nodata(data, nodata_value=np.nan, fill=0.0)

        np.testing.assert_array_equal(filled, [1.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_integer_array(self):
        """Test handling integer arrays (no NaN detection)."""
        data = np.array([1, 2, -9999, 4, -9999], dtype=np.int32)
        filled, mask = handle_nodata(data, nodata_value=-9999, fill=0)

        np.testing.assert_array_equal(filled, [1, 2, 0, 4, 0])
        np.testing.assert_array_equal(mask, [True, True, False, True, False])


class TestDetectPartialCoverage:
    """Tests for detect_partial_coverage function."""

    def test_full_coverage(self):
        """Test array with 100% valid data."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)

        assert coverage == 100.0

    def test_no_coverage(self):
        """Test array with 0% valid data."""
        data = np.array([-9999.0, -9999.0, -9999.0])
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)

        assert coverage == 0.0

    def test_partial_coverage(self):
        """Test array with partial coverage."""
        data = np.array([1.0, 2.0, -9999.0, -9999.0, 5.0])
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)

        assert coverage == 60.0  # 3 out of 5

    def test_nan_coverage(self):
        """Test coverage calculation with NaN values."""
        data = np.array([1.0, np.nan, 3.0, np.nan, np.nan])
        coverage = detect_partial_coverage(data)

        assert coverage == 40.0  # 2 out of 5

    def test_2d_array_coverage(self):
        """Test coverage calculation on 2D array."""
        data = np.array([[1.0, 2.0, -9999.0], [4.0, -9999.0, -9999.0]])
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)

        assert coverage == pytest.approx(50.0)  # 3 out of 6

    def test_empty_array(self):
        """Test empty array returns 0% coverage."""
        data = np.array([])
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)

        assert coverage == 0.0

    def test_masked_array_coverage(self):
        """Test coverage with masked array."""
        data = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 1, 1])
        coverage = detect_partial_coverage(data)

        assert coverage == 50.0  # 2 out of 4


class TestGetValidDataBounds:
    """Tests for get_valid_data_bounds function."""

    def test_full_valid_bounds(self):
        """Test bounds when all data is valid."""
        data = np.ones((10, 10))
        bounds = get_valid_data_bounds(data)

        assert bounds == (0, 10, 0, 10)

    def test_centered_valid_region(self):
        """Test bounds with valid data in center."""
        data = np.full((10, 10), -9999.0)
        data[3:7, 4:8] = 1.0  # Valid region in center

        bounds = get_valid_data_bounds(data, nodata_value=-9999.0)

        assert bounds == (3, 7, 4, 8)

    def test_corner_valid_region(self):
        """Test bounds with valid data in corner."""
        data = np.full((10, 10), -9999.0)
        data[0:3, 0:3] = 1.0  # Valid region in top-left

        bounds = get_valid_data_bounds(data, nodata_value=-9999.0)

        assert bounds == (0, 3, 0, 3)

    def test_single_pixel(self):
        """Test bounds with single valid pixel."""
        data = np.full((10, 10), -9999.0)
        data[5, 7] = 1.0  # Single valid pixel

        bounds = get_valid_data_bounds(data, nodata_value=-9999.0)

        assert bounds == (5, 6, 7, 8)  # Single pixel bounds (exclusive end)

    def test_no_valid_data(self):
        """Test bounds when no valid data exists."""
        data = np.full((10, 10), -9999.0)
        bounds = get_valid_data_bounds(data, nodata_value=-9999.0)

        assert bounds == (0, 0, 0, 0)

    def test_nan_bounds(self):
        """Test bounds with NaN values."""
        data = np.full((10, 10), np.nan)
        data[2:5, 3:6] = 1.0  # Valid region

        bounds = get_valid_data_bounds(data)

        assert bounds == (2, 5, 3, 6)

    def test_irregular_valid_region(self):
        """Test bounds with scattered valid pixels."""
        data = np.full((10, 10), -9999.0)
        data[2, 3] = 1.0
        data[7, 8] = 2.0
        data[4, 5] = 3.0

        bounds = get_valid_data_bounds(data, nodata_value=-9999.0)

        # Should encompass all valid pixels
        assert bounds == (2, 8, 3, 9)

    def test_empty_array(self):
        """Test bounds on empty array."""
        data = np.array([]).reshape(0, 0)
        bounds = get_valid_data_bounds(data)

        assert bounds == (0, 0, 0, 0)

    def test_non_2d_array_raises_error(self):
        """Test that non-2D array raises ValueError."""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="2D"):
            get_valid_data_bounds(data)

    def test_bounds_with_cropping(self):
        """Test that returned bounds can be used for array slicing."""
        data = np.full((10, 10), -9999.0)
        data[3:7, 4:8] = np.arange(16).reshape(4, 4)

        r0, r1, c0, c1 = get_valid_data_bounds(data, nodata_value=-9999.0)

        # Verify slicing produces expected result
        cropped = data[r0:r1, c0:c1]
        assert cropped.shape == (4, 4)
        assert np.all(cropped != -9999.0)


class TestIntegration:
    """Integration tests combining multiple utility functions."""

    def test_full_workflow_rgb_normalization(self):
        """Test complete workflow: handle nodata, normalize, stack."""
        # Create sample bands with nodata
        red = np.array([[0.0, 0.5, -9999.0], [1.0, 0.25, 0.75]])
        green = np.array([[0.5, 0.0, -9999.0], [0.75, 1.0, 0.25]])
        blue = np.array([[1.0, 0.25, -9999.0], [0.5, 0.0, 0.75]])

        # Handle nodata
        red_filled, red_mask = handle_nodata(red, nodata_value=-9999.0, fill=0.0)
        green_filled, green_mask = handle_nodata(green, nodata_value=-9999.0, fill=0.0)
        blue_filled, blue_mask = handle_nodata(blue, nodata_value=-9999.0, fill=0.0)

        # Normalize to uint8
        red_uint8 = normalize_to_uint8(red_filled, 0.0, 1.0)
        green_uint8 = normalize_to_uint8(green_filled, 0.0, 1.0)
        blue_uint8 = normalize_to_uint8(blue_filled, 0.0, 1.0)

        # Stack to RGB
        rgb = stack_bands_to_rgb(red_uint8, green_uint8, blue_uint8)

        assert rgb.shape == (2, 3, 3)
        assert rgb.dtype == np.uint8

        # Verify nodata pixel is black (0, 0, 0)
        np.testing.assert_array_equal(rgb[0, 2, :], [0, 0, 0])

    def test_coverage_and_bounds_workflow(self):
        """Test workflow: detect coverage, get bounds, crop."""
        # Create data with partial coverage
        data = np.full((20, 20), -9999.0)
        data[5:15, 5:15] = np.random.rand(10, 10)

        # Check coverage
        coverage = detect_partial_coverage(data, nodata_value=-9999.0)
        assert 0 < coverage < 100

        # Get valid bounds
        r0, r1, c0, c1 = get_valid_data_bounds(data, nodata_value=-9999.0)

        # Crop to valid region
        cropped = data[r0:r1, c0:c1]

        # Verify cropped region has 100% coverage
        cropped_coverage = detect_partial_coverage(cropped, nodata_value=-9999.0)
        assert cropped_coverage == 100.0
