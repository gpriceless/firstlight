"""
Unit Tests for Area Metrics Module (VIS-1.4 Task 3.1).

Tests factual area measurement calculations:
- AreaMetrics dataclass
- calculate_area_metrics: Pixel mask to area conversion
- format_area: Human-readable formatting

NO arbitrary comparisons tested - this is factual measurement only.
"""

import numpy as np
import pytest

from core.reporting.imagery.area_metrics import (
    AreaMetrics,
    calculate_area_metrics,
    format_area,
)


class TestAreaMetrics:
    """Tests for AreaMetrics dataclass."""

    def test_create_full_metrics(self):
        """Test creating AreaMetrics with all fields."""
        metrics = AreaMetrics(
            hectares=100.0,
            acres=247.1,
            square_kilometers=1.0,
            square_miles=0.386,
            percent_of_aoi=25.0,
        )

        assert metrics.hectares == 100.0
        assert metrics.acres == 247.1
        assert metrics.square_kilometers == 1.0
        assert metrics.square_miles == 0.386
        assert metrics.percent_of_aoi == 25.0

    def test_create_metrics_without_percentage(self):
        """Test creating AreaMetrics without percentage."""
        metrics = AreaMetrics(
            hectares=50.0,
            acres=123.6,
            square_kilometers=0.5,
            square_miles=0.193,
        )

        assert metrics.hectares == 50.0
        assert metrics.percent_of_aoi is None


class TestCalculateAreaMetrics:
    """Tests for calculate_area_metrics function."""

    def test_simple_calculation(self):
        """Test basic area calculation from mask."""
        # 100x100 pixels, each 10m x 10m = 100m² per pixel
        # Total: 10,000 pixels * 100m² = 1,000,000 m² = 100 hectares
        mask = np.ones((100, 100), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == pytest.approx(100.0)
        assert metrics.acres == pytest.approx(247.105, rel=0.01)
        assert metrics.square_kilometers == pytest.approx(1.0)
        assert metrics.square_miles == pytest.approx(0.386, rel=0.01)
        assert metrics.percent_of_aoi is None

    def test_partial_detection(self):
        """Test area calculation with partial detection."""
        # 100x100 mask, but only 25% detected
        mask = np.zeros((100, 100), dtype=bool)
        mask[:50, :50] = True  # 2,500 pixels

        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        # 2,500 pixels * 100m² = 250,000 m² = 25 hectares
        assert metrics.hectares == pytest.approx(25.0)
        assert metrics.acres == pytest.approx(61.776, rel=0.01)

    def test_with_percentage_calculation(self):
        """Test area calculation with percentage of AOI."""
        # 100x100 pixels, all detected
        mask = np.ones((100, 100), dtype=bool)

        # Total AOI is 4,000,000 m² (400 hectares)
        aoi_area_sqm = 4_000_000.0

        metrics = calculate_area_metrics(
            mask, pixel_size_meters=10.0, aoi_area_sqm=aoi_area_sqm
        )

        # Detected area is 1,000,000 m² out of 4,000,000 m² = 25%
        assert metrics.percent_of_aoi == pytest.approx(25.0)

    def test_small_area(self):
        """Test calculation of small area (< 1 hectare)."""
        # 10x10 pixels, each 10m x 10m
        # Total: 100 pixels * 100m² = 10,000 m² = 1 hectare
        mask = np.ones((10, 10), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == pytest.approx(1.0)
        assert metrics.acres == pytest.approx(2.471, rel=0.01)

    def test_large_area(self):
        """Test calculation of large area (> 1000 hectares)."""
        # 1000x1000 pixels, each 30m x 30m (Landsat scale)
        # Total: 1,000,000 pixels * 900m² = 900,000,000 m² = 90,000 hectares
        mask = np.ones((1000, 1000), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=30.0)

        assert metrics.hectares == pytest.approx(90_000.0)
        assert metrics.square_kilometers == pytest.approx(900.0)
        assert metrics.square_miles == pytest.approx(347.5, rel=0.01)

    def test_high_resolution_pixels(self):
        """Test with high-resolution imagery (small pixels)."""
        # 1000x1000 pixels, each 2m x 2m (high-res optical)
        # Total: 1,000,000 pixels * 4m² = 4,000,000 m² = 400 hectares
        mask = np.ones((1000, 1000), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=2.0)

        assert metrics.hectares == pytest.approx(400.0)
        assert metrics.square_kilometers == pytest.approx(4.0)

    def test_zero_detection(self):
        """Test area calculation with no detected pixels."""
        mask = np.zeros((100, 100), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == 0.0
        assert metrics.acres == 0.0
        assert metrics.square_kilometers == 0.0
        assert metrics.square_miles == 0.0

    def test_single_pixel_detection(self):
        """Test area calculation with single detected pixel."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50, 50] = True

        # Single 10m x 10m pixel = 100 m² = 0.01 hectares
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == pytest.approx(0.01)
        assert metrics.acres == pytest.approx(0.02471, rel=0.01)

    def test_scattered_detection(self):
        """Test area calculation with scattered detection pattern."""
        mask = np.zeros((100, 100), dtype=bool)
        # Create scattered pattern: every other row and column
        mask[::2, ::2] = True  # 50x50 = 2,500 pixels

        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == pytest.approx(25.0)

    def test_empty_mask(self):
        """Test area calculation with empty mask."""
        mask = np.array([], dtype=bool).reshape(0, 0)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.hectares == 0.0

    def test_invalid_pixel_size_raises_error(self):
        """Test that invalid pixel size raises ValueError."""
        mask = np.ones((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="pixel_size_meters must be positive"):
            calculate_area_metrics(mask, pixel_size_meters=0.0)

        with pytest.raises(ValueError, match="pixel_size_meters must be positive"):
            calculate_area_metrics(mask, pixel_size_meters=-10.0)

    def test_invalid_aoi_area_raises_error(self):
        """Test that invalid AOI area raises ValueError."""
        mask = np.ones((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="aoi_area_sqm must be positive"):
            calculate_area_metrics(mask, pixel_size_meters=10.0, aoi_area_sqm=0.0)

        with pytest.raises(ValueError, match="aoi_area_sqm must be positive"):
            calculate_area_metrics(mask, pixel_size_meters=10.0, aoi_area_sqm=-100.0)

    def test_percentage_edge_cases(self):
        """Test percentage calculation edge cases."""
        # 100% detection
        mask = np.ones((100, 100), dtype=bool)
        aoi_area_sqm = 100 * 100 * 100  # Exactly the mask area

        metrics = calculate_area_metrics(
            mask, pixel_size_meters=10.0, aoi_area_sqm=aoi_area_sqm
        )
        assert metrics.percent_of_aoi == pytest.approx(100.0)

        # Less than 1% detection
        mask = np.zeros((1000, 1000), dtype=bool)
        mask[:10, :10] = True  # 100 pixels out of 1,000,000
        aoi_area_sqm = 1000 * 1000 * 100

        metrics = calculate_area_metrics(
            mask, pixel_size_meters=10.0, aoi_area_sqm=aoi_area_sqm
        )
        assert metrics.percent_of_aoi == pytest.approx(0.01)

    def test_real_world_sentinel2_scale(self):
        """Test with realistic Sentinel-2 data (10m pixels)."""
        # 500x500 pixels at 10m resolution = 25 km²
        mask = np.ones((500, 500), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)

        assert metrics.square_kilometers == pytest.approx(25.0)
        assert metrics.hectares == pytest.approx(2500.0)

    def test_real_world_landsat_scale(self):
        """Test with realistic Landsat data (30m pixels)."""
        # 1000x1000 pixels at 30m resolution = 900 km²
        mask = np.ones((1000, 1000), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=30.0)

        assert metrics.square_kilometers == pytest.approx(900.0)
        assert metrics.hectares == pytest.approx(90_000.0)


class TestFormatArea:
    """Tests for format_area function."""

    def test_small_area_formatting(self):
        """Test formatting of area < 100 hectares (uses hectares/acres)."""
        metrics = AreaMetrics(
            hectares=24.5,
            acres=60.5,
            square_kilometers=0.245,
            square_miles=0.095,
        )

        result = format_area(metrics, include_percentage=False)

        assert "24.5 hectares" in result
        assert "60.5 acres" in result
        assert "km²" not in result  # Should use hectares for small areas

    def test_large_area_formatting(self):
        """Test formatting of area >= 100 hectares (uses km²/mi²)."""
        metrics = AreaMetrics(
            hectares=245.0,
            acres=605.0,
            square_kilometers=2.45,
            square_miles=0.95,
        )

        result = format_area(metrics, include_percentage=False)

        assert "2.5 km²" in result or "2.4 km²" in result  # Rounding
        assert "0.9 mi²" in result or "1.0 mi²" in result
        assert "hectares" not in result  # Should use km² for large areas

    def test_formatting_with_percentage(self):
        """Test formatting with percentage of AOI."""
        metrics = AreaMetrics(
            hectares=245.0,
            acres=605.0,
            square_kilometers=2.45,
            square_miles=0.95,
            percent_of_aoi=18.3,
        )

        result = format_area(metrics, include_percentage=True)

        assert "18% of analysis area" in result

    def test_formatting_without_percentage(self):
        """Test formatting without percentage."""
        metrics = AreaMetrics(
            hectares=245.0,
            acres=605.0,
            square_kilometers=2.45,
            square_miles=0.95,
            percent_of_aoi=18.3,
        )

        result = format_area(metrics, include_percentage=False)

        assert "analysis area" not in result

    def test_formatting_no_percentage_available(self):
        """Test formatting when no percentage is available."""
        metrics = AreaMetrics(
            hectares=50.0,
            acres=123.6,
            square_kilometers=0.5,
            square_miles=0.193,
            percent_of_aoi=None,
        )

        result = format_area(metrics, include_percentage=True)

        assert "analysis area" not in result

    def test_thousands_separator(self):
        """Test that large numbers have thousands separator."""
        metrics = AreaMetrics(
            hectares=2450.0,
            acres=6050.0,
            square_kilometers=24.5,
            square_miles=9.5,
        )

        result = format_area(metrics, include_percentage=False)

        # Should format with comma separator (24.5 km², not 2,450 hectares for >100)
        assert "24.5" in result or "24.4" in result

    def test_boundary_at_100_hectares(self):
        """Test formatting at the 100 hectare boundary."""
        # Exactly 100 hectares - should use km²
        metrics = AreaMetrics(
            hectares=100.0,
            acres=247.1,
            square_kilometers=1.0,
            square_miles=0.386,
        )

        result = format_area(metrics, include_percentage=False)

        assert "km²" in result
        assert "hectares" not in result

        # Just under 100 hectares - should use hectares
        metrics = AreaMetrics(
            hectares=99.9,
            acres=246.9,
            square_kilometers=0.999,
            square_miles=0.386,
        )

        result = format_area(metrics, include_percentage=False)

        assert "hectares" in result
        assert "km²" not in result

    def test_zero_area_formatting(self):
        """Test formatting of zero area."""
        metrics = AreaMetrics(
            hectares=0.0,
            acres=0.0,
            square_kilometers=0.0,
            square_miles=0.0,
        )

        result = format_area(metrics, include_percentage=False)

        assert "0.0 hectares" in result
        assert "0.0 acres" in result

    def test_very_small_area_formatting(self):
        """Test formatting of very small area (< 1 hectare)."""
        metrics = AreaMetrics(
            hectares=0.5,
            acres=1.24,
            square_kilometers=0.005,
            square_miles=0.002,
        )

        result = format_area(metrics, include_percentage=False)

        assert "0.5 hectares" in result
        assert "1.2 acres" in result or "1.3 acres" in result

    def test_very_large_area_formatting(self):
        """Test formatting of very large area (thousands of km²)."""
        metrics = AreaMetrics(
            hectares=250_000.0,
            acres=617_763.0,
            square_kilometers=2_500.0,
            square_miles=965.3,
        )

        result = format_area(metrics, include_percentage=False)

        assert "2,500.0 km²" in result
        assert "965.3 mi²" in result

    def test_percentage_rounding(self):
        """Test that percentages are rounded to whole numbers."""
        metrics = AreaMetrics(
            hectares=50.0,
            acres=123.6,
            square_kilometers=0.5,
            square_miles=0.193,
            percent_of_aoi=18.7,
        )

        result = format_area(metrics, include_percentage=True)

        assert "19% of analysis area" in result

        metrics.percent_of_aoi = 18.3
        result = format_area(metrics, include_percentage=True)

        assert "18% of analysis area" in result

    def test_no_arbitrary_comparisons(self):
        """Test that output contains NO arbitrary comparisons."""
        metrics = AreaMetrics(
            hectares=50.0,
            acres=123.6,
            square_kilometers=0.5,
            square_miles=0.193,
        )

        result = format_area(metrics, include_percentage=False)

        # Verify NO arbitrary comparison language
        forbidden_terms = [
            "football",
            "field",
            "soccer",
            "stadium",
            "park",
            "city block",
            "basketball",
            "tennis",
            "olympic",
        ]

        result_lower = result.lower()
        for term in forbidden_terms:
            assert term not in result_lower, f"Found forbidden term: {term}"


class TestIntegration:
    """Integration tests combining calculation and formatting."""

    def test_full_workflow_small_area(self):
        """Test complete workflow for small detected area."""
        # Simulate flood detection: 50x50 pixels at 10m resolution
        mask = np.ones((50, 50), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)
        formatted = format_area(metrics, include_percentage=False)

        assert "25.0 hectares" in formatted
        assert "hectares" in formatted
        assert "km²" not in formatted

    def test_full_workflow_large_area(self):
        """Test complete workflow for large detected area."""
        # Simulate wildfire detection: 500x500 pixels at 30m resolution
        mask = np.ones((500, 500), dtype=bool)
        metrics = calculate_area_metrics(mask, pixel_size_meters=30.0)
        formatted = format_area(metrics, include_percentage=False)

        assert "km²" in formatted
        assert "mi²" in formatted
        assert "hectares" not in formatted

    def test_full_workflow_with_percentage(self):
        """Test complete workflow including percentage calculation."""
        # 100x100 detection within 200x200 AOI
        mask = np.zeros((200, 200), dtype=bool)
        mask[:100, :100] = True

        aoi_area_sqm = 200 * 200 * 100  # 200x200 pixels at 10m = 4 km²

        metrics = calculate_area_metrics(
            mask, pixel_size_meters=10.0, aoi_area_sqm=aoi_area_sqm
        )
        formatted = format_area(metrics, include_percentage=True)

        # Should be 100 hectares (1 km²), 25% of 400 hectare AOI
        # Since >= 100 hectares, will use km² format
        assert "1.0 km²" in formatted
        assert "25% of analysis area" in formatted

    def test_realistic_sentinel2_flood_detection(self):
        """Test realistic Sentinel-2 flood detection scenario."""
        # 1000x800 image, 15% flooded areas
        mask = np.random.rand(1000, 800) > 0.85  # ~15% True

        # Sentinel-2 at 10m resolution
        metrics = calculate_area_metrics(mask, pixel_size_meters=10.0)
        formatted = format_area(metrics, include_percentage=False)

        # Should detect ~120,000 pixels = ~1,200 hectares = ~12 km²
        assert metrics.hectares > 1000
        assert "km²" in formatted

    def test_realistic_landsat_wildfire_detection(self):
        """Test realistic Landsat wildfire detection scenario."""
        # 500x400 image, 30% burned areas
        mask = np.random.rand(500, 400) > 0.70  # ~30% True

        # Landsat at 30m resolution
        aoi_area_sqm = 500 * 400 * 900  # Total AOI area

        metrics = calculate_area_metrics(
            mask, pixel_size_meters=30.0, aoi_area_sqm=aoi_area_sqm
        )
        formatted = format_area(metrics, include_percentage=True)

        # Should detect ~60,000 pixels = ~5,400 hectares = ~54 km²
        assert metrics.square_kilometers > 40
        assert "% of analysis area" in formatted
