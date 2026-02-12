"""
Unit tests for color scales module (VIS-1.3 Task 1.1).
"""

import numpy as np
import pytest

from core.reporting.imagery.color_scales import (
    ColorPalette,
    SeverityLevel,
    apply_colormap,
    get_confidence_palette,
    get_fire_palette,
    get_flood_palette,
    get_palette_by_name,
    hex_to_rgba,
)


class TestHexToRgba:
    """Test hex color conversion."""

    def test_hex_with_hash(self):
        """Test conversion with # prefix."""
        rgba = hex_to_rgba("#E3F2FD")
        assert rgba == (227, 242, 253, 255)

    def test_hex_without_hash(self):
        """Test conversion without # prefix."""
        rgba = hex_to_rgba("E3F2FD")
        assert rgba == (227, 242, 253, 255)

    def test_custom_alpha(self):
        """Test conversion with custom alpha."""
        rgba = hex_to_rgba("#E3F2FD", alpha=128)
        assert rgba == (227, 242, 253, 128)

    def test_black(self):
        """Test black color."""
        rgba = hex_to_rgba("#000000")
        assert rgba == (0, 0, 0, 255)

    def test_white(self):
        """Test white color."""
        rgba = hex_to_rgba("#FFFFFF")
        assert rgba == (255, 255, 255, 255)

    def test_zero_alpha(self):
        """Test transparent alpha."""
        rgba = hex_to_rgba("#FF0000", alpha=0)
        assert rgba == (255, 0, 0, 0)


class TestColorPalette:
    """Test ColorPalette dataclass."""

    def test_valid_palette(self):
        """Test creating valid palette."""
        palette = ColorPalette(
            name="Test",
            colors=[(255, 0, 0, 255), (0, 255, 0, 255)],
            labels=["Red", "Green"],
            value_ranges=[(0.0, 0.5), (0.5, 1.0)],
        )
        assert palette.name == "Test"
        assert len(palette.colors) == 2
        assert len(palette.labels) == 2
        assert len(palette.value_ranges) == 2

    def test_mismatched_labels(self):
        """Test validation fails with mismatched labels."""
        with pytest.raises(ValueError, match="Number of labels"):
            ColorPalette(
                name="Test",
                colors=[(255, 0, 0, 255), (0, 255, 0, 255)],
                labels=["Red"],  # Only 1 label for 2 colors
                value_ranges=[(0.0, 0.5), (0.5, 1.0)],
            )

    def test_mismatched_ranges(self):
        """Test validation fails with mismatched ranges."""
        with pytest.raises(ValueError, match="Number of value ranges"):
            ColorPalette(
                name="Test",
                colors=[(255, 0, 0, 255), (0, 255, 0, 255)],
                labels=["Red", "Green"],
                value_ranges=[(0.0, 1.0)],  # Only 1 range for 2 colors
            )


class TestFloodPalette:
    """Test flood severity palette."""

    def test_palette_structure(self):
        """Test flood palette has correct structure."""
        palette = get_flood_palette()

        assert palette.name == "Flood Severity"
        assert len(palette.colors) == 5
        assert len(palette.labels) == 5
        assert len(palette.value_ranges) == 5

    def test_color_count(self):
        """Test flood palette has 5 levels."""
        palette = get_flood_palette()
        assert len(palette.colors) == 5

    def test_labels(self):
        """Test flood palette has correct labels."""
        palette = get_flood_palette()
        expected_labels = ["Minimal", "Low", "Moderate", "High", "Severe"]
        assert palette.labels == expected_labels

    def test_value_ranges(self):
        """Test flood palette value ranges cover 0-1."""
        palette = get_flood_palette()

        # First range should start at 0
        assert palette.value_ranges[0][0] == 0.0

        # Last range should end at 1
        assert palette.value_ranges[-1][1] == 1.0

        # Ranges should be continuous
        for i in range(len(palette.value_ranges) - 1):
            assert palette.value_ranges[i][1] == palette.value_ranges[i + 1][0]

    def test_rgba_format(self):
        """Test all colors are valid RGBA tuples."""
        palette = get_flood_palette()

        for color in palette.colors:
            assert len(color) == 4
            assert all(0 <= c <= 255 for c in color)

    def test_blue_gradient(self):
        """Test colors form blue gradient (increasing blue intensity)."""
        palette = get_flood_palette()

        # First color should be lightest (high RGB values)
        # Last color should be darkest (low RGB values)
        first_brightness = sum(palette.colors[0][:3])
        last_brightness = sum(palette.colors[-1][:3])

        assert first_brightness > last_brightness


class TestFirePalette:
    """Test wildfire severity palette."""

    def test_palette_structure(self):
        """Test fire palette has correct structure."""
        palette = get_fire_palette()

        assert palette.name == "Fire Burn Severity"
        assert len(palette.colors) == 6
        assert len(palette.labels) == 6
        assert len(palette.value_ranges) == 6

    def test_color_count(self):
        """Test fire palette has 6 levels."""
        palette = get_fire_palette()
        assert len(palette.colors) == 6

    def test_labels(self):
        """Test fire palette has correct labels."""
        palette = get_fire_palette()
        expected_labels = [
            "Unburned",
            "Unburned/Regrowth",
            "Low Severity",
            "Moderate-Low",
            "Moderate-High",
            "High Severity",
        ]
        assert palette.labels == expected_labels

    def test_value_ranges_dnbr(self):
        """Test fire palette value ranges match dNBR classification."""
        palette = get_fire_palette()

        # Check key dNBR thresholds
        assert palette.value_ranges[0][0] == -0.5  # Unburned start
        assert palette.value_ranges[2][0] == 0.1  # Low severity start
        assert palette.value_ranges[-1][1] == 1.3  # High severity end

    def test_rgba_format(self):
        """Test all colors are valid RGBA tuples."""
        palette = get_fire_palette()

        for color in palette.colors:
            assert len(color) == 4
            assert all(0 <= c <= 255 for c in color)

    def test_color_progression(self):
        """Test colors progress from green to red."""
        palette = get_fire_palette()

        # First color (unburned) should be greenish
        first_color = palette.colors[0]
        assert first_color[1] > first_color[0]  # Green > Red

        # Last color (high severity) should be reddish
        last_color = palette.colors[-1]
        assert last_color[0] > last_color[1]  # Red > Green


class TestConfidencePalette:
    """Test confidence level palette."""

    def test_palette_structure(self):
        """Test confidence palette has correct structure."""
        palette = get_confidence_palette()

        assert palette.name == "Confidence Level"
        assert len(palette.colors) == 4
        assert len(palette.labels) == 4
        assert len(palette.value_ranges) == 4

    def test_color_count(self):
        """Test confidence palette has 4 levels."""
        palette = get_confidence_palette()
        assert len(palette.colors) == 4

    def test_labels(self):
        """Test confidence palette has correct labels."""
        palette = get_confidence_palette()
        expected_labels = ["Very Low", "Low", "Medium", "High"]
        assert palette.labels == expected_labels

    def test_value_ranges(self):
        """Test confidence palette value ranges cover 0-1."""
        palette = get_confidence_palette()

        # First range should start at 0
        assert palette.value_ranges[0][0] == 0.0

        # Last range should end at 1
        assert palette.value_ranges[-1][1] == 1.0

        # Ranges should be continuous
        for i in range(len(palette.value_ranges) - 1):
            assert palette.value_ranges[i][1] == palette.value_ranges[i + 1][0]

    def test_rgba_format(self):
        """Test all colors are valid RGBA tuples."""
        palette = get_confidence_palette()

        for color in palette.colors:
            assert len(color) == 4
            assert all(0 <= c <= 255 for c in color)

    def test_grayscale(self):
        """Test colors are grayscale (R=G=B)."""
        palette = get_confidence_palette()

        for color in palette.colors:
            r, g, b, a = color
            assert r == g == b  # Grayscale

    def test_brightness_progression(self):
        """Test colors progress from light to dark."""
        palette = get_confidence_palette()

        # Each color should be darker than the previous
        for i in range(len(palette.colors) - 1):
            current_brightness = palette.colors[i][0]  # R value
            next_brightness = palette.colors[i + 1][0]
            assert current_brightness > next_brightness


class TestApplyColormap:
    """Test colormap application."""

    def test_output_shape(self):
        """Test output has correct shape."""
        values = np.array([[0.1, 0.5], [0.7, 0.9]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        assert rgba.shape == (2, 2, 4)
        assert rgba.dtype == np.uint8

    def test_output_dtype(self):
        """Test output is uint8."""
        values = np.random.rand(10, 10)
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        assert rgba.dtype == np.uint8

    def test_value_mapping(self):
        """Test values map to correct colors."""
        # Create simple test case
        values = np.array([[0.1, 0.3], [0.5, 0.9]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        # 0.1 should map to first color (minimal)
        expected_color_0 = palette.colors[0]
        assert tuple(rgba[0, 0]) == expected_color_0

        # 0.3 should map to second color (low)
        expected_color_1 = palette.colors[1]
        assert tuple(rgba[0, 1]) == expected_color_1

        # 0.5 should map to third color (moderate)
        expected_color_2 = palette.colors[2]
        assert tuple(rgba[1, 0]) == expected_color_2

        # 0.9 should map to fifth color (severe)
        expected_color_4 = palette.colors[4]
        assert tuple(rgba[1, 1]) == expected_color_4

    def test_nodata_handling(self):
        """Test nodata values become transparent."""
        values = np.array([[0.5, np.nan], [0.7, np.nan]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        # Valid values should have color
        assert rgba[0, 0, 3] > 0  # Alpha > 0

        # NaN values should be transparent
        assert rgba[0, 1, 3] == 0  # Alpha = 0
        assert rgba[1, 1, 3] == 0

    def test_custom_nodata(self):
        """Test custom nodata value."""
        values = np.array([[0.5, -999], [0.7, -999]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette, nodata_value=-999)

        # Valid values should have color
        assert rgba[0, 0, 3] > 0

        # -999 values should be transparent
        assert rgba[0, 1, 3] == 0
        assert rgba[1, 1, 3] == 0

    def test_edge_values(self):
        """Test edge values (min and max of ranges)."""
        values = np.array([[0.0, 0.2], [0.8, 1.0]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        # All pixels should have color
        assert all(rgba[:, :, 3].flatten() > 0)

    def test_out_of_range_values(self):
        """Test values outside palette range."""
        values = np.array([[-0.5, 1.5], [0.5, 0.5]])
        palette = get_flood_palette()

        rgba = apply_colormap(values, palette)

        # Out of range values should be transparent
        assert rgba[0, 0, 3] == 0  # -0.5
        assert rgba[0, 1, 3] == 0  # 1.5

        # In-range values should have color
        assert rgba[1, 0, 3] > 0
        assert rgba[1, 1, 3] > 0

    def test_1d_array_fails(self):
        """Test 1D array raises ValueError."""
        values = np.array([0.1, 0.5, 0.9])
        palette = get_flood_palette()

        with pytest.raises(ValueError, match="must be 2D array"):
            apply_colormap(values, palette)

    def test_3d_array_fails(self):
        """Test 3D array raises ValueError."""
        values = np.random.rand(5, 5, 3)
        palette = get_flood_palette()

        with pytest.raises(ValueError, match="must be 2D array"):
            apply_colormap(values, palette)

    def test_fire_palette_dnbr_values(self):
        """Test fire palette with realistic dNBR values."""
        # dNBR values ranging from unburned to high severity
        values = np.array([[-0.3, 0.0], [0.2, 0.5], [0.7, np.nan]])
        palette = get_fire_palette()

        rgba = apply_colormap(values, palette)

        # Check output shape
        assert rgba.shape == (3, 2, 4)

        # Unburned (-0.3) should get first color
        assert tuple(rgba[0, 0]) == palette.colors[0]

        # Unburned/regrowth (0.0) should get second color
        assert tuple(rgba[0, 1]) == palette.colors[1]

        # Low severity (0.2) should get third color
        assert tuple(rgba[1, 0]) == palette.colors[2]

        # Moderate-high (0.5) should get fifth color
        assert tuple(rgba[1, 1]) == palette.colors[4]

        # High severity (0.7) should get sixth color
        assert tuple(rgba[2, 0]) == palette.colors[5]

        # NaN should be transparent
        assert rgba[2, 1, 3] == 0

    def test_large_array(self):
        """Test colormap on larger realistic array."""
        values = np.random.rand(100, 100)
        palette = get_confidence_palette()

        rgba = apply_colormap(values, palette)

        assert rgba.shape == (100, 100, 4)
        assert rgba.dtype == np.uint8

        # All valid pixels should have some color
        assert np.all(rgba[:, :, 3] > 0)


class TestGetPaletteByName:
    """Test palette retrieval by name."""

    def test_get_flood(self):
        """Test getting flood palette by name."""
        palette = get_palette_by_name("flood")
        assert palette.name == "Flood Severity"

    def test_get_fire(self):
        """Test getting fire palette by name."""
        palette = get_palette_by_name("fire")
        assert palette.name == "Fire Burn Severity"

    def test_get_confidence(self):
        """Test getting confidence palette by name."""
        palette = get_palette_by_name("confidence")
        assert palette.name == "Confidence Level"

    def test_case_insensitive(self):
        """Test name matching is case insensitive."""
        palette1 = get_palette_by_name("FLOOD")
        palette2 = get_palette_by_name("Flood")
        palette3 = get_palette_by_name("flood")

        assert palette1.name == palette2.name == palette3.name

    def test_unknown_name(self):
        """Test unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown palette name"):
            get_palette_by_name("unknown")


class TestSeverityLevel:
    """Test SeverityLevel enum."""

    def test_flood_levels(self):
        """Test flood severity levels exist."""
        assert SeverityLevel.FLOOD_MINIMAL.value == "minimal"
        assert SeverityLevel.FLOOD_LOW.value == "low"
        assert SeverityLevel.FLOOD_MODERATE.value == "moderate"
        assert SeverityLevel.FLOOD_HIGH.value == "high"
        assert SeverityLevel.FLOOD_SEVERE.value == "severe"

    def test_fire_levels(self):
        """Test fire severity levels exist."""
        assert SeverityLevel.FIRE_UNBURNED.value == "unburned"
        assert SeverityLevel.FIRE_UNBURNED_REGROWTH.value == "unburned_regrowth"
        assert SeverityLevel.FIRE_LOW.value == "low"
        assert SeverityLevel.FIRE_MODERATE_LOW.value == "moderate_low"
        assert SeverityLevel.FIRE_MODERATE_HIGH.value == "moderate_high"
        assert SeverityLevel.FIRE_HIGH.value == "high"

    def test_confidence_levels(self):
        """Test confidence levels exist."""
        assert SeverityLevel.CONFIDENCE_VERY_LOW.value == "very_low"
        assert SeverityLevel.CONFIDENCE_LOW.value == "low"
        assert SeverityLevel.CONFIDENCE_MEDIUM.value == "medium"
        assert SeverityLevel.CONFIDENCE_HIGH.value == "high"


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_flood_workflow(self):
        """Test complete flood overlay workflow."""
        # Simulate flood severity map
        values = np.random.rand(50, 50)

        # Apply flood palette
        palette = get_flood_palette()
        rgba = apply_colormap(values, palette)

        # Verify output
        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8
        assert np.all(rgba[:, :, 3] > 0)  # All pixels have color

    def test_fire_workflow(self):
        """Test complete fire overlay workflow."""
        # Simulate dNBR map with realistic values
        values = np.random.uniform(-0.5, 1.3, size=(50, 50))

        # Apply fire palette
        palette = get_fire_palette()
        rgba = apply_colormap(values, palette)

        # Verify output
        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8

    def test_confidence_workflow(self):
        """Test complete confidence overlay workflow."""
        # Simulate confidence map
        values = np.random.rand(50, 50)

        # Apply confidence palette
        palette = get_confidence_palette()
        rgba = apply_colormap(values, palette)

        # Verify output
        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8
        assert np.all(rgba[:, :, 3] > 0)

    def test_mixed_palettes(self):
        """Test using different palettes on same data."""
        values = np.random.rand(20, 20)

        flood_rgba = apply_colormap(values, get_flood_palette())
        fire_rgba = apply_colormap(values, get_fire_palette())
        confidence_rgba = apply_colormap(values, get_confidence_palette())

        # All should produce valid output
        assert flood_rgba.shape == fire_rgba.shape == confidence_rgba.shape
        assert flood_rgba.dtype == fire_rgba.dtype == confidence_rgba.dtype

        # Colors should differ (palettes are different)
        assert not np.array_equal(flood_rgba, fire_rgba)
        assert not np.array_equal(flood_rgba, confidence_rgba)
