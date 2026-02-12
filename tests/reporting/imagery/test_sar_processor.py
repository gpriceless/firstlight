"""
Unit tests for SAR imagery processing.

Tests the SAR processor module including:
- VV single polarization visualization
- VH single polarization visualization
- Dual polarization RGB composite
- Flood detection visualization
- dB scaling and stretching
- Speckle filtering
- Colormap application
- Error handling and edge cases
"""

import numpy as np
import pytest

from core.reporting.imagery.sar_processor import (
    SARConfig,
    SARProcessor,
    SARResult,
)


class TestSARConfig:
    """Tests for SARConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SARConfig()

        assert config.visualization == "vv"
        assert config.db_min == -20.0
        assert config.db_max == 5.0
        assert config.apply_speckle_filter is False
        assert config.colormap == "grayscale"

    def test_custom_config(self):
        """Test custom configuration."""
        config = SARConfig(
            visualization="dual_pol",
            db_min=-25.0,
            db_max=0.0,
            apply_speckle_filter=True,
            colormap="viridis"
        )

        assert config.visualization == "dual_pol"
        assert config.db_min == -25.0
        assert config.db_max == 0.0
        assert config.apply_speckle_filter is True
        assert config.colormap == "viridis"

    def test_invalid_visualization(self):
        """Test validation of visualization parameter."""
        with pytest.raises(ValueError, match="Invalid visualization"):
            SARConfig(visualization="invalid_mode")

    def test_invalid_colormap(self):
        """Test validation of colormap parameter."""
        with pytest.raises(ValueError, match="Invalid colormap"):
            SARConfig(colormap="rainbow")

    def test_invalid_db_range(self):
        """Test validation of dB range."""
        with pytest.raises(ValueError, match="db_min.*must be less than"):
            SARConfig(db_min=5.0, db_max=-20.0)

        with pytest.raises(ValueError, match="db_min.*must be less than"):
            SARConfig(db_min=0.0, db_max=0.0)

    def test_all_visualizations(self):
        """Test all valid visualization modes."""
        for viz in ["vv", "vh", "dual_pol", "flood_detection"]:
            config = SARConfig(visualization=viz)
            assert config.visualization == viz

    def test_all_colormaps(self):
        """Test all valid colormaps."""
        for cmap in ["grayscale", "viridis", "water"]:
            config = SARConfig(colormap=cmap)
            assert config.colormap == cmap


class TestSARProcessorBasic:
    """Basic tests for SARProcessor."""

    def test_initialization_default(self):
        """Test processor initialization with default config."""
        processor = SARProcessor()
        assert processor.config is not None
        assert processor.config.visualization == "vv"

    def test_initialization_custom(self):
        """Test processor initialization with custom config."""
        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)
        assert processor.config.visualization == "dual_pol"

    def test_vv_single_polarization(self):
        """Test VV single polarization visualization."""
        # Create synthetic SAR data (linear backscatter)
        # Typical range: 0.001 to 1.0 (linear)
        # Converts to approximately -30 to 0 dB
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        config = SARConfig(visualization="vv")
        processor = SARProcessor(config)
        result = processor.process(vv)

        # Check result structure
        assert isinstance(result, SARResult)
        assert result.rgb_array.shape == (100, 100)
        assert result.rgb_array.dtype == np.float32
        assert np.all((result.rgb_array >= 0) & (result.rgb_array <= 1))

        # Check metadata
        assert result.metadata["visualization"] == "vv"
        assert result.metadata["db_range"] == (-20.0, 5.0)
        assert result.metadata["speckle_filtered"] is False

        # Check valid mask
        assert result.valid_mask.shape == (100, 100)
        assert result.valid_mask.dtype == bool

    def test_vh_single_polarization(self):
        """Test VH single polarization visualization."""
        vh = np.random.uniform(0.001, 0.05, (100, 100))

        config = SARConfig(visualization="vh")
        processor = SARProcessor(config)
        result = processor.process(vh)

        assert result.rgb_array.shape == (100, 100)
        assert result.metadata["visualization"] == "vh"

    def test_dual_pol_rgb(self):
        """Test dual polarization RGB composite."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))
        vh = np.random.uniform(0.001, 0.05, (100, 100))

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Should return RGB array
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.rgb_array.dtype == np.float32
        assert np.all((result.rgb_array >= 0) & (result.rgb_array <= 1))

    def test_flood_detection_rgb(self):
        """Test flood detection RGB composite."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))
        vh = np.random.uniform(0.001, 0.05, (100, 100))

        config = SARConfig(visualization="flood_detection")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Should return RGB array
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.metadata["visualization"] == "flood_detection"


class TestDBConversion:
    """Tests for linear to dB conversion."""

    def test_to_db_conversion(self):
        """Test linear to dB conversion formula: dB = 10 * log10(linear)."""
        processor = SARProcessor()

        # Test known conversions
        linear = np.array([0.001, 0.01, 0.1, 1.0, 10.0])
        expected_db = np.array([-30.0, -20.0, -10.0, 0.0, 10.0])

        db = processor._to_db(linear)

        np.testing.assert_array_almost_equal(db, expected_db, decimal=5)

    def test_to_db_invalid_values(self):
        """Test that invalid values (<=0, NaN, inf) become NaN."""
        processor = SARProcessor()

        linear = np.array([0.0, -1.0, np.nan, np.inf, -np.inf, 1.0])
        db = processor._to_db(linear)

        # First 5 should be NaN
        assert np.isnan(db[0])
        assert np.isnan(db[1])
        assert np.isnan(db[2])
        assert np.isnan(db[3])
        assert np.isnan(db[4])
        # Last should be valid (0 dB)
        assert np.isclose(db[5], 0.0)

    def test_db_stretch_typical_range(self):
        """Test dB stretching with typical SAR range."""
        config = SARConfig(db_min=-20.0, db_max=5.0)
        processor = SARProcessor(config)

        # Test values in typical range
        db = np.array([-25.0, -20.0, -10.0, 0.0, 5.0, 10.0])
        expected = np.array([0.0, 0.0, 0.4, 0.8, 1.0, 1.0])

        stretched = processor._apply_db_stretch(db)

        np.testing.assert_array_almost_equal(stretched, expected, decimal=5)

    def test_db_stretch_custom_range(self):
        """Test dB stretching with custom range."""
        config = SARConfig(db_min=-25.0, db_max=0.0)
        processor = SARProcessor(config)

        db = np.array([-30.0, -25.0, -12.5, 0.0, 5.0])
        expected = np.array([0.0, 0.0, 0.5, 1.0, 1.0])

        stretched = processor._apply_db_stretch(db)

        np.testing.assert_array_almost_equal(stretched, expected, decimal=5)


class TestSpeckleFiltering:
    """Tests for speckle noise reduction."""

    def test_speckle_filter_applied(self):
        """Test that speckle filter is applied when enabled."""
        # Create noisy data
        np.random.seed(42)
        vv = np.random.uniform(0.01, 0.1, (100, 100))
        # Add speckle noise
        vv *= np.random.uniform(0.5, 1.5, (100, 100))

        config = SARConfig(visualization="vv", apply_speckle_filter=True)
        processor = SARProcessor(config)
        result = processor.process(vv)

        assert result.metadata["speckle_filtered"] is True

    def test_speckle_filter_not_applied(self):
        """Test that speckle filter is not applied by default."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        config = SARConfig(visualization="vv", apply_speckle_filter=False)
        processor = SARProcessor(config)
        result = processor.process(vv)

        assert result.metadata["speckle_filtered"] is False

    def test_speckle_filter_preserves_nan(self):
        """Test that speckle filter preserves NaN values."""
        processor = SARProcessor()

        data = np.random.uniform(-20, 5, (10, 10))
        data[2:4, 2:4] = np.nan

        filtered = processor._speckle_filter(data)

        # NaN values should be preserved
        assert np.isnan(filtered[2, 2])
        assert np.isnan(filtered[3, 3])
        # Other values should be filtered
        assert np.isfinite(filtered[0, 0])

    def test_speckle_filter_reduces_variance(self):
        """Test that speckle filter reduces variance."""
        processor = SARProcessor()

        # Create noisy data
        np.random.seed(42)
        data = np.random.uniform(-20, 5, (100, 100))

        filtered = processor._speckle_filter(data)

        # Filtered data should have lower variance
        original_var = np.var(data)
        filtered_var = np.var(filtered)

        assert filtered_var < original_var


class TestColormaps:
    """Tests for colormap application."""

    def test_grayscale_no_colormap(self):
        """Test that grayscale mode doesn't apply colormap."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        config = SARConfig(visualization="vv", colormap="grayscale")
        processor = SARProcessor(config)
        result = processor.process(vv)

        # Should remain 2D grayscale
        assert result.rgb_array.ndim == 2

    def test_viridis_colormap(self):
        """Test viridis colormap application."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        config = SARConfig(visualization="vv", colormap="viridis")
        processor = SARProcessor(config)
        result = processor.process(vv)

        # Should be RGB
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.metadata["colormap"] == "viridis"

    def test_water_colormap(self):
        """Test water colormap application."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        config = SARConfig(visualization="vv", colormap="water")
        processor = SARProcessor(config)
        result = processor.process(vv)

        # Should be RGB
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.metadata["colormap"] == "water"

    def test_colormap_range(self):
        """Test that colormap output is in valid range."""
        processor = SARProcessor()

        # Test with known values
        gray = np.linspace(0, 1, 100)

        for colormap in ["viridis", "water"]:
            rgb = processor._apply_colormap(gray, colormap)

            assert rgb.shape == (100, 3)
            assert np.all((rgb >= 0) & (rgb <= 1))

    def test_colormap_not_applied_to_rgb(self):
        """Test that colormap is not applied to RGB composites."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))
        vh = np.random.uniform(0.001, 0.05, (100, 100))

        config = SARConfig(visualization="dual_pol", colormap="viridis")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Should be RGB but colormap metadata should be None (not applied)
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.metadata["colormap"] is None


class TestValidationAndErrors:
    """Tests for input validation and error handling."""

    def test_vv_not_2d(self):
        """Test error when VV is not 2D."""
        vv = np.random.rand(10, 10, 3)  # 3D array

        processor = SARProcessor()

        with pytest.raises(ValueError, match="must be 2D"):
            processor.process(vv)

    def test_vh_not_2d(self):
        """Test error when VH is not 2D."""
        vv = np.random.rand(10, 10)
        vh = np.random.rand(10, 10, 3)  # 3D array

        processor = SARProcessor()

        with pytest.raises(ValueError, match="must be 2D"):
            processor.process(vv, vh)

    def test_vv_vh_shape_mismatch(self):
        """Test error when VV and VH have different shapes."""
        vv = np.random.rand(100, 100)
        vh = np.random.rand(50, 50)  # Different shape

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)

        with pytest.raises(ValueError, match="must have same shape"):
            processor.process(vv, vh)

    def test_dual_pol_requires_vh(self):
        """Test error when dual_pol visualization is used without VH."""
        vv = np.random.rand(100, 100)

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)

        with pytest.raises(ValueError, match="requires VH polarization"):
            processor.process(vv)

    def test_flood_detection_requires_vh(self):
        """Test error when flood_detection visualization is used without VH."""
        vv = np.random.rand(100, 100)

        config = SARConfig(visualization="flood_detection")
        processor = SARProcessor(config)

        with pytest.raises(ValueError, match="requires VH polarization"):
            processor.process(vv)


class TestValidDataMask:
    """Tests for valid data masking."""

    def test_nan_values_masked(self):
        """Test that NaN values are masked."""
        vv = np.random.uniform(0.01, 0.1, (10, 10))
        vv[2:4, 2:4] = np.nan

        processor = SARProcessor()
        result = processor.process(vv)

        # NaN region should be invalid
        assert not result.valid_mask[2, 2]
        assert not result.valid_mask[3, 3]
        # Other regions should be valid
        assert result.valid_mask[0, 0]
        assert result.valid_mask[9, 9]

    def test_inf_values_masked(self):
        """Test that infinite values are masked."""
        vv = np.random.uniform(0.01, 0.1, (10, 10))
        vv[2, 2] = np.inf
        vv[3, 3] = -np.inf

        processor = SARProcessor()
        result = processor.process(vv)

        # Inf values should be invalid
        assert not result.valid_mask[2, 2]
        assert not result.valid_mask[3, 3]

    def test_negative_values_masked(self):
        """Test that negative/zero values are masked (invalid backscatter)."""
        vv = np.random.uniform(0.01, 0.1, (10, 10))
        vv[2, 2] = 0.0
        vv[3, 3] = -0.1

        processor = SARProcessor()
        result = processor.process(vv)

        # Non-positive values should be invalid
        assert not result.valid_mask[2, 2]
        assert not result.valid_mask[3, 3]

    def test_masked_pixels_zero(self):
        """Test that masked pixels are set to zero in output."""
        vv = np.random.uniform(0.01, 0.1, (10, 10))
        vv[2:4, 2:4] = np.nan

        processor = SARProcessor()
        result = processor.process(vv)

        # Invalid pixels should be zero
        assert result.rgb_array[2, 2] == 0.0
        assert result.rgb_array[3, 3] == 0.0


class TestRealWorldScenarios:
    """Tests simulating real SAR imagery scenarios."""

    def test_water_detection(self):
        """Test that water appears dark in VV polarization."""
        # Create synthetic scene: land and water
        vv = np.ones((100, 100)) * 0.05  # Land (moderate backscatter)
        vv[40:60, 40:60] = 0.005  # Water region (low backscatter)

        processor = SARProcessor()
        result = processor.process(vv)

        # Water should be darker than land
        water_mean = result.rgb_array[40:60, 40:60].mean()
        land_mean = result.rgb_array[0:20, 0:20].mean()

        assert water_mean < land_mean

    def test_urban_vs_vegetation(self):
        """Test dual pol distinguishes urban from vegetation."""
        # Urban: high VV, moderate VH
        # Vegetation: moderate VV, high VH
        vv = np.ones((100, 100)) * 0.05
        vh = np.ones((100, 100)) * 0.02

        # Urban area - high VV
        vv[0:50, 0:50] = 0.1
        vh[0:50, 0:50] = 0.03

        # Vegetation area - high VH
        vv[50:100, 50:100] = 0.04
        vh[50:100, 50:100] = 0.04

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Urban should have more red (channel 0 = VV)
        urban_red = result.rgb_array[25, 25, 0]
        veg_red = result.rgb_array[75, 75, 0]
        assert urban_red > veg_red

        # Vegetation should have more green (channel 1 = VH)
        urban_green = result.rgb_array[25, 25, 1]
        veg_green = result.rgb_array[75, 75, 1]
        assert veg_green > urban_green

    def test_flood_detection_scenario(self):
        """Test flood detection visualization on flooded area."""
        # Create flooded landscape
        vv = np.ones((100, 100)) * 0.05  # Normal land
        vh = np.ones((100, 100)) * 0.02

        # Flooded area - very low VV and VH
        vv[30:70, 30:70] = 0.001
        vh[30:70, 30:70] = 0.0005

        config = SARConfig(visualization="flood_detection")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Flooded area should be very dark (low in all channels)
        flood_brightness = result.rgb_array[50, 50].mean()
        land_brightness = result.rgb_array[10, 10].mean()

        assert flood_brightness < land_brightness * 0.5

    def test_typical_sentinel1_values(self):
        """Test with typical Sentinel-1 backscatter values."""
        # Sentinel-1 GRD typical range: 0.0001 to 1.0 (linear)
        # Corresponds to approximately -40 to 0 dB
        np.random.seed(42)

        # Mixed land cover
        vv = np.random.lognormal(mean=-3, sigma=0.5, size=(100, 100))
        vh = np.random.lognormal(mean=-4, sigma=0.5, size=(100, 100))

        # Clip to realistic range
        vv = np.clip(vv, 0.0001, 1.0)
        vh = np.clip(vh, 0.0001, 0.5)

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Should process without errors
        assert result.rgb_array.shape == (100, 100, 3)
        assert np.all((result.rgb_array >= 0) & (result.rgb_array <= 1))


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_pixel(self):
        """Test processing single pixel."""
        vv = np.array([[0.05]])

        processor = SARProcessor()
        result = processor.process(vv)

        assert result.rgb_array.shape == (1, 1)
        assert result.valid_mask.shape == (1, 1)

    def test_small_array(self):
        """Test processing very small array."""
        vv = np.array([[0.05, 0.06], [0.07, 0.08]])

        processor = SARProcessor()
        result = processor.process(vv)

        assert result.rgb_array.shape == (2, 2)

    def test_uniform_values(self):
        """Test processing uniform values."""
        vv = np.ones((100, 100)) * 0.05

        processor = SARProcessor()
        result = processor.process(vv)

        # Should handle gracefully
        assert result.rgb_array.shape == (100, 100)

    def test_all_invalid_values(self):
        """Test processing array with all invalid values."""
        vv = np.full((10, 10), np.nan)

        processor = SARProcessor()
        result = processor.process(vv)

        # All pixels should be invalid
        assert not np.any(result.valid_mask)
        # All output should be zero
        assert np.all(result.rgb_array == 0.0)

    def test_extreme_db_values(self):
        """Test with extreme dB values outside typical range."""
        processor = SARProcessor()

        # Extreme values: -100 to 50 dB
        db = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])

        stretched = processor._apply_db_stretch(db)

        # Should clip properly
        assert stretched[0] == 0.0  # Below min
        assert stretched[-1] == 1.0  # Above max
        assert 0.0 <= stretched[2] <= 1.0  # Within range


class TestMetadata:
    """Tests for result metadata."""

    def test_metadata_completeness(self):
        """Test that all metadata fields are present."""
        vv = np.random.uniform(0.01, 0.1, (100, 100))

        processor = SARProcessor()
        result = processor.process(vv)

        # Check all required fields
        assert "db_range" in result.metadata
        assert "visualization" in result.metadata
        assert "speckle_filtered" in result.metadata
        assert "colormap" in result.metadata

    def test_metadata_db_range(self):
        """Test dB range in metadata."""
        config = SARConfig(db_min=-25.0, db_max=0.0)
        processor = SARProcessor(config)

        vv = np.random.uniform(0.01, 0.1, (100, 100))
        result = processor.process(vv)

        assert result.metadata["db_range"] == (-25.0, 0.0)

    def test_metadata_visualization_type(self):
        """Test visualization type in metadata."""
        for viz in ["vv", "vh", "dual_pol", "flood_detection"]:
            config = SARConfig(visualization=viz)
            processor = SARProcessor(config)

            vv = np.random.uniform(0.01, 0.1, (10, 10))
            vh = np.random.uniform(0.001, 0.05, (10, 10))

            if viz in ["dual_pol", "flood_detection"]:
                result = processor.process(vv, vh)
            else:
                result = processor.process(vv)

            assert result.metadata["visualization"] == viz
