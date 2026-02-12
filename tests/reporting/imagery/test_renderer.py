"""
Unit Tests for ImageryRenderer (VIS-1.1 Task 2.1).

Tests the main ImageryRenderer class that orchestrates:
- Band selection from composites
- Histogram stretching
- RGB composition
- Missing band handling with graceful degradation
- Multiple input formats (dict, file path, array)

Includes edge cases: missing bands, partial coverage, invalid inputs.
"""

import numpy as np
import pytest

from core.reporting.imagery.histogram import HistogramStretch
from core.reporting.imagery.renderer import (
    ImageryRenderer,
    RenderedImage,
    RendererConfig,
)


class TestRendererConfig:
    """Tests for RendererConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RendererConfig()

        assert config.composite_name == "true_color"
        assert config.stretch_method == HistogramStretch.LINEAR_2PCT
        assert config.output_format == "uint8"
        assert config.apply_cloud_mask is False
        assert config.custom_min is None
        assert config.custom_max is None

    def test_custom_config(self):
        """Test custom configuration."""
        config = RendererConfig(
            composite_name="false_color_ir",
            stretch_method=HistogramStretch.STDDEV,
            output_format="float",
            apply_cloud_mask=True,
            custom_min=0.0,
            custom_max=0.5
        )

        assert config.composite_name == "false_color_ir"
        assert config.stretch_method == HistogramStretch.STDDEV
        assert config.output_format == "float"
        assert config.apply_cloud_mask is True
        assert config.custom_min == 0.0
        assert config.custom_max == 0.5

    def test_invalid_output_format(self):
        """Test that invalid output format raises error."""
        with pytest.raises(ValueError, match="output_format must be"):
            RendererConfig(output_format="invalid")

    def test_invalid_custom_range(self):
        """Test that invalid custom min/max raises error."""
        with pytest.raises(ValueError, match="custom_min.*must be less than"):
            RendererConfig(custom_min=1.0, custom_max=0.5)

        with pytest.raises(ValueError, match="custom_min.*must be less than"):
            RendererConfig(custom_min=0.5, custom_max=0.5)


class TestRenderedImage:
    """Tests for RenderedImage dataclass."""

    def test_valid_rendered_image(self):
        """Test creating valid RenderedImage."""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        metadata = {"sensor": "sentinel2", "composite": "true_color"}
        mask = np.ones((100, 100), dtype=bool)

        result = RenderedImage(
            rgb_array=rgb,
            metadata=metadata,
            valid_mask=mask
        )

        assert result.rgb_array.shape == (100, 100, 3)
        assert result.metadata["sensor"] == "sentinel2"
        assert result.valid_mask.shape == (100, 100)

    def test_invalid_rgb_shape(self):
        """Test that invalid RGB shape raises error."""
        # 2D array instead of 3D
        with pytest.raises(ValueError, match="must have shape.*W, 3"):
            RenderedImage(rgb_array=np.zeros((100, 100)))

        # Wrong number of channels
        with pytest.raises(ValueError, match="must have shape.*W, 3"):
            RenderedImage(rgb_array=np.zeros((100, 100, 4)))

    def test_default_values(self):
        """Test default values for optional fields."""
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        result = RenderedImage(rgb_array=rgb)

        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0
        assert result.valid_mask.size == 0


class TestImageryRenderer:
    """Tests for ImageryRenderer main class."""

    def create_sentinel2_bands(self, shape=(100, 100)):
        """Helper to create synthetic Sentinel-2 band data."""
        # Create bands with different reflectance patterns
        rng = np.random.default_rng(42)

        return {
            "B02": rng.normal(0.1, 0.02, shape),  # Blue
            "B03": rng.normal(0.15, 0.03, shape),  # Green
            "B04": rng.normal(0.2, 0.04, shape),  # Red
            "B08": rng.normal(0.3, 0.05, shape),  # NIR
        }

    def test_initialization_default(self):
        """Test renderer initialization with defaults."""
        renderer = ImageryRenderer()

        assert renderer.config is not None
        assert renderer.config.composite_name == "true_color"

    def test_initialization_custom_config(self):
        """Test renderer initialization with custom config."""
        config = RendererConfig(
            composite_name="false_color_ir",
            stretch_method=HistogramStretch.STDDEV
        )
        renderer = ImageryRenderer(config)

        assert renderer.config.composite_name == "false_color_ir"
        assert renderer.config.stretch_method == HistogramStretch.STDDEV

    def test_render_sentinel2_true_color(self):
        """Test rendering Sentinel-2 to true color RGB."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        result = renderer.render(bands, sensor="sentinel2")

        # Check output structure
        assert isinstance(result, RenderedImage)
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.rgb_array.dtype == np.uint8

        # Check metadata
        assert result.metadata["sensor"] == "sentinel2"
        assert result.metadata["composite_name"] == "true_color"
        assert result.metadata["bands_used"] == ("B04", "B03", "B02")
        assert "coverage_percent" in result.metadata
        assert result.metadata["coverage_percent"] > 99.0  # Should be ~100%

        # Check valid mask
        assert result.valid_mask.shape == (100, 100)
        assert result.valid_mask.dtype == bool

    def test_render_sentinel2_false_color_ir(self):
        """Test rendering Sentinel-2 to false color infrared."""
        bands = self.create_sentinel2_bands()
        config = RendererConfig(composite_name="false_color_ir")
        renderer = ImageryRenderer(config)

        result = renderer.render(bands, sensor="sentinel2")

        # Check composite used
        assert result.metadata["composite_name"] == "false_color_ir"
        assert result.metadata["bands_used"] == ("B08", "B04", "B03")

        # Output should still be valid RGB
        assert result.rgb_array.shape == (100, 100, 3)
        assert result.rgb_array.dtype == np.uint8

    def test_render_with_composite_override(self):
        """Test overriding composite name at render time."""
        bands = self.create_sentinel2_bands()

        # Config says true_color, but we override to false_color_ir
        config = RendererConfig(composite_name="true_color")
        renderer = ImageryRenderer(config)

        result = renderer.render(
            bands,
            sensor="sentinel2",
            composite_name="false_color_ir"
        )

        # Should use the override
        assert result.metadata["composite_name"] == "false_color_ir"
        assert result.metadata["bands_used"] == ("B08", "B04", "B03")

    def test_render_float_output(self):
        """Test rendering with float output format."""
        bands = self.create_sentinel2_bands()
        config = RendererConfig(output_format="float")
        renderer = ImageryRenderer(config)

        result = renderer.render(bands, sensor="sentinel2")

        # Output should be float in [0, 1]
        assert result.rgb_array.dtype in (np.float32, np.float64)
        assert result.rgb_array.min() >= 0.0
        assert result.rgb_array.max() <= 1.0

    def test_render_with_custom_stretch(self):
        """Test rendering with custom min/max stretch."""
        bands = self.create_sentinel2_bands()
        config = RendererConfig(
            custom_min=0.05,
            custom_max=0.25,
            output_format="float"
        )
        renderer = ImageryRenderer(config)

        result = renderer.render(bands, sensor="sentinel2")

        # Should record custom stretch in metadata
        assert result.metadata["custom_min"] == 0.05
        assert result.metadata["custom_max"] == 0.25

    def test_render_with_stddev_stretch(self):
        """Test rendering with standard deviation stretch method."""
        bands = self.create_sentinel2_bands()
        config = RendererConfig(stretch_method=HistogramStretch.STDDEV)
        renderer = ImageryRenderer(config)

        result = renderer.render(bands, sensor="sentinel2")

        assert result.metadata["stretch_method"] == "stddev"

    def test_render_with_nodata_values(self):
        """Test rendering with nodata values in bands."""
        bands = self.create_sentinel2_bands()

        # Add nodata values (0.0 as nodata)
        bands["B02"][0:10, 0:10] = 0.0
        bands["B03"][0:10, 0:10] = 0.0
        bands["B04"][0:10, 0:10] = 0.0

        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2", nodata_value=0.0)

        # Coverage should be less than 100%
        assert result.metadata["coverage_percent"] < 100.0
        assert result.metadata["coverage_percent"] > 90.0  # 100 pixels out of 10000

        # Valid mask should be False in nodata region
        assert not result.valid_mask[5, 5]
        assert result.valid_mask[50, 50]  # Valid elsewhere

    def test_render_with_partial_coverage(self):
        """Test rendering with significant missing data."""
        bands = self.create_sentinel2_bands()

        # Set half the data to nodata
        bands["B02"][50:, :] = -9999
        bands["B03"][50:, :] = -9999
        bands["B04"][50:, :] = -9999

        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2", nodata_value=-9999)

        # Coverage should be approximately 50%
        assert 45.0 <= result.metadata["coverage_percent"] <= 55.0

    def test_missing_bands_raises_error(self):
        """Test that missing required bands raises error."""
        # Only provide red and green, no blue
        bands = {"B04": np.random.rand(100, 100), "B03": np.random.rand(100, 100)}

        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Cannot render composite.*missing bands"):
            renderer.render(bands, sensor="sentinel2")

    def test_graceful_degradation_fallback(self):
        """Test graceful degradation when SWIR bands missing."""
        # Provide only basic RGB and NIR bands (no SWIR)
        bands = self.create_sentinel2_bands()

        # Try to render SWIR composite
        config = RendererConfig(composite_name="swir")
        renderer = ImageryRenderer(config)

        # Should fall back to true_color since SWIR bands (B12, B8A) are missing
        result = renderer.render(bands, sensor="sentinel2")

        # Should have fallen back to true_color
        assert result.metadata["composite_name"] == "true_color"
        assert result.metadata["bands_used"] == ("B04", "B03", "B02")

    def test_no_fallback_available_raises_error(self):
        """Test that error is raised when no fallback is possible."""
        # Provide only NIR band, missing all RGB bands
        bands = {"B08": np.random.rand(100, 100)}

        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Cannot render composite.*missing bands"):
            renderer.render(bands, sensor="sentinel2")

    def test_mismatched_band_shapes_raises_error(self):
        """Test that mismatched band shapes raise error."""
        bands = {
            "B02": np.random.rand(100, 100),
            "B03": np.random.rand(100, 100),
            "B04": np.random.rand(50, 50),  # Different shape!
        }

        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Band shapes must match"):
            renderer.render(bands, sensor="sentinel2")

    def test_unsupported_sensor_raises_error(self):
        """Test that unsupported sensor raises error."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Unsupported sensor"):
            renderer.render(bands, sensor="invalid_sensor")

    def test_invalid_composite_raises_error(self):
        """Test that invalid composite name raises error."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Composite.*not available"):
            renderer.render(bands, sensor="sentinel2", composite_name="nonexistent")

    def test_load_from_dict(self):
        """Test _load_bands with dictionary input."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        # Get a composite to pass to _load_bands
        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        loaded = renderer._load_bands(bands, "sentinel2", composite)

        assert loaded is bands  # Should return same dict
        assert "B02" in loaded
        assert "B03" in loaded
        assert "B04" in loaded

    def test_load_from_2d_array(self):
        """Test _load_bands with 2D numpy array (grayscale)."""
        array = np.random.rand(100, 100)
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        loaded = renderer._load_bands(array, "sentinel2", composite)

        # Should replicate for all three bands
        assert "B04" in loaded
        assert "B03" in loaded
        assert "B02" in loaded
        np.testing.assert_array_equal(loaded["B04"], array)
        np.testing.assert_array_equal(loaded["B03"], array)
        np.testing.assert_array_equal(loaded["B02"], array)

    def test_load_from_3d_array_not_implemented(self):
        """Test that 3D array raises NotImplementedError."""
        array = np.random.rand(100, 100, 4)  # 4 bands
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        with pytest.raises(NotImplementedError, match="multi-band numpy array"):
            renderer._load_bands(array, "sentinel2", composite)

    def test_load_from_invalid_type(self):
        """Test that invalid source type raises error."""
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        with pytest.raises(ValueError, match="Unsupported data source type"):
            renderer._load_bands([1, 2, 3], "sentinel2", composite)

    def test_select_bands_success(self):
        """Test _select_bands with all bands present."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        r, g, b = renderer._select_bands(bands, composite)

        assert r.shape == (100, 100)
        assert g.shape == (100, 100)
        assert b.shape == (100, 100)
        np.testing.assert_array_equal(r, bands["B04"])
        np.testing.assert_array_equal(g, bands["B03"])
        np.testing.assert_array_equal(b, bands["B02"])

    def test_select_bands_missing_raises_error(self):
        """Test _select_bands with missing bands."""
        bands = {"B04": np.random.rand(100, 100)}  # Only red
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        with pytest.raises(ValueError, match="Missing required bands"):
            renderer._select_bands(bands, composite)

    def test_apply_stretch(self):
        """Test _apply_stretch applies to all three bands."""
        rng = np.random.default_rng(42)
        r_band = rng.normal(100, 20, (100, 100))
        g_band = rng.normal(150, 30, (100, 100))
        b_band = rng.normal(80, 15, (100, 100))

        renderer = ImageryRenderer()
        r_str, g_str, b_str = renderer._apply_stretch((r_band, g_band, b_band))

        # Should be stretched to [0, 1]
        assert 0.0 <= r_str.min() <= 0.1
        assert 0.9 <= r_str.max() <= 1.0
        assert 0.0 <= g_str.min() <= 0.1
        assert 0.9 <= g_str.max() <= 1.0
        assert 0.0 <= b_str.min() <= 0.1
        assert 0.9 <= b_str.max() <= 1.0

    def test_to_rgb_uint8(self):
        """Test _to_rgb with uint8 output."""
        r = np.full((100, 100), 0.0)
        g = np.full((100, 100), 0.5)
        b = np.full((100, 100), 1.0)

        config = RendererConfig(output_format="uint8")
        renderer = ImageryRenderer(config)

        rgb = renderer._to_rgb(r, g, b)

        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype == np.uint8
        assert rgb[0, 0, 0] == 0  # Red = 0
        assert rgb[0, 0, 1] == 127  # Green = 0.5
        assert rgb[0, 0, 2] == 255  # Blue = 1.0

    def test_to_rgb_float(self):
        """Test _to_rgb with float output."""
        r = np.full((100, 100), 0.0)
        g = np.full((100, 100), 0.5)
        b = np.full((100, 100), 1.0)

        config = RendererConfig(output_format="float")
        renderer = ImageryRenderer(config)

        rgb = renderer._to_rgb(r, g, b)

        assert rgb.shape == (100, 100, 3)
        assert rgb.dtype in (np.float32, np.float64)
        assert np.isclose(rgb[0, 0, 0], 0.0)
        assert np.isclose(rgb[0, 0, 1], 0.5)
        assert np.isclose(rgb[0, 0, 2], 1.0)

    def test_compute_valid_mask(self):
        """Test _compute_valid_mask combines masks from all bands."""
        r = np.ones((100, 100))
        g = np.ones((100, 100))
        b = np.ones((100, 100))

        # Set nodata in different locations
        r[0:10, :] = -9999
        g[:, 0:10] = -9999
        b[90:, 90:] = -9999

        renderer = ImageryRenderer()
        mask = renderer._compute_valid_mask(r, g, b, nodata_value=-9999)

        # Pixels should be valid only where all three bands are valid
        assert not mask[5, 5]  # R is nodata
        assert not mask[50, 5]  # G is nodata
        assert not mask[95, 95]  # B is nodata
        assert mask[50, 50]  # All valid

    def test_handle_missing_bands_no_fallback_needed(self):
        """Test _handle_missing_bands when all bands present."""
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        result_composite = renderer._handle_missing_bands(bands, composite)

        # Should return same composite
        assert result_composite.name == "true_color"
        assert result_composite.bands == composite.bands

    def test_handle_missing_bands_with_fallback(self):
        """Test _handle_missing_bands falls back when bands missing."""
        # Only RGB and NIR bands (no SWIR)
        bands = self.create_sentinel2_bands()
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "swir")  # Needs B12, B8A

        result_composite = renderer._handle_missing_bands(bands, composite)

        # Should fall back to true_color
        assert result_composite.name == "true_color"
        assert result_composite.bands == ("B04", "B03", "B02")

    def test_handle_missing_bands_no_suitable_fallback(self):
        """Test _handle_missing_bands raises when no fallback available."""
        # Only NIR band
        bands = {"B08": np.random.rand(100, 100)}
        renderer = ImageryRenderer()

        from core.reporting.imagery.band_combinations import get_bands_for_composite
        composite = get_bands_for_composite("sentinel2", "true_color")

        with pytest.raises(ValueError, match="No suitable fallback composite"):
            renderer._handle_missing_bands(bands, composite)

    def test_end_to_end_workflow(self):
        """Test complete workflow from bands to rendered image."""
        # Create realistic Sentinel-2 data
        bands = self.create_sentinel2_bands(shape=(200, 200))

        # Add some nodata
        bands["B02"][0:20, 0:20] = np.nan
        bands["B03"][0:20, 0:20] = np.nan
        bands["B04"][0:20, 0:20] = np.nan

        # Configure renderer
        config = RendererConfig(
            composite_name="true_color",
            stretch_method=HistogramStretch.LINEAR_2PCT,
            output_format="uint8"
        )
        renderer = ImageryRenderer(config)

        # Render
        result = renderer.render(bands, sensor="sentinel2")

        # Validate result
        assert isinstance(result, RenderedImage)
        assert result.rgb_array.shape == (200, 200, 3)
        assert result.rgb_array.dtype == np.uint8
        assert result.valid_mask.shape == (200, 200)
        assert result.metadata["sensor"] == "sentinel2"
        assert result.metadata["composite_name"] == "true_color"
        assert result.metadata["coverage_percent"] < 100.0  # Due to NaN region
        assert result.metadata["coverage_percent"] > 90.0  # Should be high
