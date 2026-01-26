"""
Unit tests for legend generation module (VIS-1.3 Task 1.2).
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.reporting.imagery.color_scales import (
    ColorPalette,
    get_confidence_palette,
    get_fire_palette,
    get_flood_palette,
)
from core.reporting.imagery.legend import LegendConfig, LegendRenderer


class TestLegendConfig:
    """Test LegendConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = LegendConfig()

        assert config.title == "Legend"
        assert config.position == "top-right"
        assert config.width == 200
        assert config.height is None
        assert config.font_size == 12
        assert config.background_alpha == 0.8
        assert config.show_title is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LegendConfig(
            title="Custom Legend",
            position="bottom-left",
            width=300,
            height=400,
            font_size=14,
            background_alpha=0.5,
            show_title=False,
        )

        assert config.title == "Custom Legend"
        assert config.position == "bottom-left"
        assert config.width == 300
        assert config.height == 400
        assert config.font_size == 14
        assert config.background_alpha == 0.5
        assert config.show_title is False

    def test_invalid_position(self):
        """Test validation fails with invalid position."""
        with pytest.raises(ValueError, match="Invalid position"):
            LegendConfig(position="middle")

    def test_negative_width(self):
        """Test validation fails with negative width."""
        with pytest.raises(ValueError, match="Width must be positive"):
            LegendConfig(width=-10)

    def test_zero_width(self):
        """Test validation fails with zero width."""
        with pytest.raises(ValueError, match="Width must be positive"):
            LegendConfig(width=0)

    def test_negative_height(self):
        """Test validation fails with negative height."""
        with pytest.raises(ValueError, match="Height must be positive"):
            LegendConfig(height=-10)

    def test_invalid_alpha_low(self):
        """Test validation fails with alpha < 0."""
        with pytest.raises(ValueError, match="Background alpha must be between 0 and 1"):
            LegendConfig(background_alpha=-0.1)

    def test_invalid_alpha_high(self):
        """Test validation fails with alpha > 1."""
        with pytest.raises(ValueError, match="Background alpha must be between 0 and 1"):
            LegendConfig(background_alpha=1.5)

    def test_negative_font_size(self):
        """Test validation fails with negative font size."""
        with pytest.raises(ValueError, match="Font size must be positive"):
            LegendConfig(font_size=-5)

    def test_zero_font_size(self):
        """Test validation fails with zero font size."""
        with pytest.raises(ValueError, match="Font size must be positive"):
            LegendConfig(font_size=0)

    def test_valid_positions(self):
        """Test all valid position values."""
        valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

        for position in valid_positions:
            config = LegendConfig(position=position)
            assert config.position == position

    def test_boundary_alpha_values(self):
        """Test boundary alpha values (0 and 1)."""
        config_transparent = LegendConfig(background_alpha=0.0)
        assert config_transparent.background_alpha == 0.0

        config_opaque = LegendConfig(background_alpha=1.0)
        assert config_opaque.background_alpha == 1.0


class TestLegendRenderer:
    """Test LegendRenderer class."""

    def test_initialization_default(self):
        """Test renderer initialization with default config."""
        renderer = LegendRenderer()

        assert renderer.config.title == "Legend"
        assert renderer.config.width == 200
        assert renderer.font is not None
        assert renderer.title_font is not None

    def test_initialization_custom_config(self):
        """Test renderer initialization with custom config."""
        config = LegendConfig(title="Custom", width=250)
        renderer = LegendRenderer(config)

        assert renderer.config.title == "Custom"
        assert renderer.config.width == 250

    def test_generate_discrete_flood(self):
        """Test discrete legend generation for flood palette."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Check result is RGBA array
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 4
        assert result.dtype == np.uint8

        # Check dimensions
        assert result.shape[1] == 200  # Width matches default config
        assert result.shape[0] > 0  # Height is auto-calculated

    def test_generate_discrete_fire(self):
        """Test discrete legend generation for fire palette."""
        renderer = LegendRenderer()
        palette = get_fire_palette()

        result = renderer.generate_discrete(palette)

        # Check result structure
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 4
        assert result.dtype == np.uint8

        # Fire palette has 6 colors, should be taller than flood (5 colors)
        flood_palette = get_flood_palette()
        flood_result = renderer.generate_discrete(flood_palette)
        assert result.shape[0] > flood_result.shape[0]

    def test_generate_discrete_confidence(self):
        """Test discrete legend generation for confidence palette."""
        renderer = LegendRenderer()
        palette = get_confidence_palette()

        result = renderer.generate_discrete(palette)

        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 4
        assert result.dtype == np.uint8

    def test_generate_discrete_custom_width(self):
        """Test discrete legend with custom width."""
        config = LegendConfig(width=300)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        assert result.shape[1] == 300

    def test_generate_discrete_no_title(self):
        """Test discrete legend without title."""
        config = LegendConfig(show_title=False)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result_no_title = renderer.generate_discrete(palette)

        # Compare to legend with title
        config_with_title = LegendConfig(show_title=True)
        renderer_with_title = LegendRenderer(config_with_title)
        result_with_title = renderer_with_title.generate_discrete(palette)

        # Legend without title should be shorter
        assert result_no_title.shape[0] < result_with_title.shape[0]

    def test_generate_discrete_save_to_file(self):
        """Test saving discrete legend to file."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "legend.png"

            result = renderer.generate_discrete(palette, output_path=output_path)

            # Check file was created
            assert output_path.exists()

            # Check file is valid PNG
            img = Image.open(output_path)
            assert img.format == "PNG"
            assert img.mode == "RGBA"

            # Check dimensions match returned array
            img_array = np.array(img)
            assert img_array.shape == result.shape

    def test_generate_continuous_flood(self):
        """Test continuous legend generation for flood palette."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        result = renderer.generate_continuous(palette, 0.0, 1.0)

        # Check result is RGBA array
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[2] == 4
        assert result.dtype == np.uint8

        # Check dimensions
        assert result.shape[1] == 200  # Width matches default config
        assert result.shape[0] > 0  # Height is auto-calculated

    def test_generate_continuous_fire(self):
        """Test continuous legend generation for fire palette."""
        renderer = LegendRenderer()
        palette = get_fire_palette()

        result = renderer.generate_continuous(palette, -0.5, 1.3)

        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 4
        assert result.dtype == np.uint8

    def test_generate_continuous_custom_range(self):
        """Test continuous legend with custom value range."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        # Test with different value ranges
        result1 = renderer.generate_continuous(palette, 0.0, 100.0)
        result2 = renderer.generate_continuous(palette, -50.0, 50.0)

        # Both should have same dimensions (range only affects labels)
        assert result1.shape == result2.shape

    def test_generate_continuous_custom_width(self):
        """Test continuous legend with custom width."""
        config = LegendConfig(width=300)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_continuous(palette, 0.0, 1.0)

        assert result.shape[1] == 300

    def test_generate_continuous_no_title(self):
        """Test continuous legend without title."""
        config = LegendConfig(show_title=False)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result_no_title = renderer.generate_continuous(palette, 0.0, 1.0)

        # Compare to legend with title
        config_with_title = LegendConfig(show_title=True)
        renderer_with_title = LegendRenderer(config_with_title)
        result_with_title = renderer_with_title.generate_continuous(palette, 0.0, 1.0)

        # Legend without title should be shorter
        assert result_no_title.shape[0] < result_with_title.shape[0]

    def test_generate_continuous_save_to_file(self):
        """Test saving continuous legend to file."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "legend_continuous.png"

            result = renderer.generate_continuous(
                palette, 0.0, 1.0, output_path=output_path
            )

            # Check file was created
            assert output_path.exists()

            # Check file is valid PNG
            img = Image.open(output_path)
            assert img.format == "PNG"
            assert img.mode == "RGBA"

            # Check dimensions match returned array
            img_array = np.array(img)
            assert img_array.shape == result.shape

    def test_discrete_has_transparency(self):
        """Test discrete legend has transparent areas."""
        config = LegendConfig(background_alpha=0.5)  # Semi-transparent background
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Check that alpha channel has values < 255 (some transparency)
        alpha_channel = result[:, :, 3]
        assert np.any(alpha_channel < 255)

    def test_continuous_has_transparency(self):
        """Test continuous legend has transparent areas."""
        config = LegendConfig(background_alpha=0.5)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_continuous(palette, 0.0, 1.0)

        # Check that alpha channel has values < 255 (some transparency)
        alpha_channel = result[:, :, 3]
        assert np.any(alpha_channel < 255)

    def test_discrete_colors_present(self):
        """Test discrete legend contains colors from palette."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Check that some pixels have non-zero colors (not all white/black/transparent)
        # We expect to see blue colors from the flood palette
        has_color = (result[:, :, 0] > 0) | (result[:, :, 1] > 0) | (result[:, :, 2] > 0)
        assert np.any(has_color)

    def test_continuous_gradient_present(self):
        """Test continuous legend contains gradient colors."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        result = renderer.generate_continuous(palette, 0.0, 1.0)

        # Check that RGB channels have variation (gradient effect)
        for channel in range(3):
            channel_data = result[:, :, channel]
            # Should have multiple distinct values in each channel
            unique_values = len(np.unique(channel_data))
            assert unique_values > 2  # More than just background and one color

    def test_custom_title(self):
        """Test legend with custom title."""
        config = LegendConfig(title="Water Depth (meters)")
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Result should be valid
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 4

    def test_output_path_creates_directory(self):
        """Test that output path creates parent directories."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "dir" / "legend.png"

            renderer.generate_discrete(palette, output_path=output_path)

            # Check parent directories were created
            assert output_path.parent.exists()
            assert output_path.exists()

    def test_discrete_height_scales_with_colors(self):
        """Test that discrete legend height increases with more colors."""
        renderer = LegendRenderer()

        # Create palettes with different numbers of colors
        palette_2 = ColorPalette(
            name="Test 2",
            colors=[(255, 0, 0, 200), (0, 255, 0, 200)],
            labels=["Red", "Green"],
            value_ranges=[(0.0, 0.5), (0.5, 1.0)],
        )

        palette_5 = get_flood_palette()  # 5 colors

        result_2 = renderer.generate_discrete(palette_2)
        result_5 = renderer.generate_discrete(palette_5)

        # More colors should result in taller legend
        assert result_5.shape[0] > result_2.shape[0]

    def test_background_alpha_zero(self):
        """Test legend with fully transparent background."""
        config = LegendConfig(background_alpha=0.0)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # With alpha=0, background should be mostly transparent
        # (though legend content itself will have some opacity)
        assert isinstance(result, np.ndarray)
        assert result.shape[2] == 4

    def test_background_alpha_one(self):
        """Test legend with fully opaque background."""
        config = LegendConfig(background_alpha=1.0)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # With alpha=1, background should be opaque
        # Most pixels should have alpha=255
        alpha_channel = result[:, :, 3]
        opaque_pixels = np.sum(alpha_channel == 255)
        total_pixels = alpha_channel.size

        # At least 50% of pixels should be fully opaque
        assert opaque_pixels > total_pixels * 0.5

    def test_large_font_size(self):
        """Test legend with large font size."""
        config = LegendConfig(font_size=20)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Larger font should result in taller legend
        config_small = LegendConfig(font_size=10)
        renderer_small = LegendRenderer(config_small)
        result_small = renderer_small.generate_discrete(palette)

        assert result.shape[0] > result_small.shape[0]

    def test_continuous_value_formatting(self):
        """Test continuous legend formats values correctly."""
        renderer = LegendRenderer()
        palette = get_flood_palette()

        # Test with different value ranges
        result1 = renderer.generate_continuous(palette, 0.0, 1.0)
        result2 = renderer.generate_continuous(palette, 0.123456, 9.876543)

        # Both should be valid arrays
        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)

    def test_fixed_height_respected(self):
        """Test that fixed height configuration is respected."""
        fixed_height = 150
        config = LegendConfig(height=fixed_height)
        renderer = LegendRenderer(config)
        palette = get_flood_palette()

        result = renderer.generate_discrete(palette)

        # Height should match configured value
        assert result.shape[0] == fixed_height

    def test_all_palettes_discrete(self):
        """Test discrete legend generation for all standard palettes."""
        renderer = LegendRenderer()

        palettes = [
            get_flood_palette(),
            get_fire_palette(),
            get_confidence_palette(),
        ]

        for palette in palettes:
            result = renderer.generate_discrete(palette)

            assert isinstance(result, np.ndarray)
            assert result.ndim == 3
            assert result.shape[2] == 4
            assert result.dtype == np.uint8
            assert result.shape[0] > 0
            assert result.shape[1] == 200

    def test_all_palettes_continuous(self):
        """Test continuous legend generation for all standard palettes."""
        renderer = LegendRenderer()

        test_cases = [
            (get_flood_palette(), 0.0, 1.0),
            (get_fire_palette(), -0.5, 1.3),
            (get_confidence_palette(), 0.0, 1.0),
        ]

        for palette, min_val, max_val in test_cases:
            result = renderer.generate_continuous(palette, min_val, max_val)

            assert isinstance(result, np.ndarray)
            assert result.ndim == 3
            assert result.shape[2] == 4
            assert result.dtype == np.uint8
            assert result.shape[0] > 0
            assert result.shape[1] == 200
