"""
Unit tests for visual enhancements (VIS-1.4).

Tests cover:
- Locator map generation
- Attribution block creation
- Infrastructure label rendering
- Edge cases and validation
"""

from datetime import datetime

import numpy as np
import pytest

from core.reporting.imagery.attribution import (
    DataSource,
    add_license_info,
    create_attribution_block,
    format_attribution_text,
    landsat8_source,
    osm_source,
    sentinel2_source,
)
from core.reporting.imagery.locator_map import create_locator_map
from core.reporting.imagery.overlay import DetectionOverlay


# === Fixtures ===


@pytest.fixture
def sample_bounds():
    """Sample geographic bounds for Miami area."""
    return (-80.5, 25.5, -80.0, 26.0)


@pytest.fixture
def sample_sources():
    """Sample data sources for testing."""
    return [
        DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25)),
        DataSource("OpenStreetMap", "OSM Contributors"),
    ]


# === Locator Map Tests ===


def test_create_locator_map_basic(sample_bounds):
    """Test basic locator map creation."""
    locator = create_locator_map(sample_bounds)

    # Check output shape
    assert locator.shape == (200, 200, 4)
    assert locator.dtype == np.uint8

    # Check it's not all the same color (should have content)
    assert not np.all(locator == locator[0, 0])


def test_create_locator_map_custom_size(sample_bounds):
    """Test locator map with custom size."""
    locator = create_locator_map(sample_bounds, size=(300, 150))

    assert locator.shape == (150, 300, 4)
    assert locator.dtype == np.uint8


def test_create_locator_map_context_zoom(sample_bounds):
    """Test locator map with different context zoom levels."""
    # Tight context
    locator_tight = create_locator_map(sample_bounds, context_zoom=1)
    assert locator_tight.shape == (200, 200, 4)

    # Wide context
    locator_wide = create_locator_map(sample_bounds, context_zoom=5)
    assert locator_wide.shape == (200, 200, 4)

    # Maps should be different (different zoom levels)
    assert not np.array_equal(locator_tight, locator_wide)


def test_create_locator_map_custom_colors(sample_bounds):
    """Test locator map with custom colors."""
    locator = create_locator_map(
        sample_bounds,
        background_color=(200, 200, 255),
        detection_color=(0, 255, 0),
    )

    assert locator.shape == (200, 200, 4)


def test_create_locator_map_invalid_bounds():
    """Test that invalid bounds raise errors."""
    # Wrong number of elements
    with pytest.raises(ValueError, match="must have 4 elements"):
        create_locator_map((-80.5, 25.5, -80.0))

    # Invalid bounds (minx >= maxx)
    with pytest.raises(ValueError, match="Invalid bounds"):
        create_locator_map((-80.0, 25.5, -80.5, 26.0))

    # Invalid bounds (miny >= maxy)
    with pytest.raises(ValueError, match="Invalid bounds"):
        create_locator_map((-80.5, 26.0, -80.0, 25.5))


def test_create_locator_map_invalid_zoom():
    """Test that invalid context_zoom raises error."""
    bounds = (-80.5, 25.5, -80.0, 26.0)

    with pytest.raises(ValueError, match="context_zoom must be 1-5"):
        create_locator_map(bounds, context_zoom=0)

    with pytest.raises(ValueError, match="context_zoom must be 1-5"):
        create_locator_map(bounds, context_zoom=6)


def test_create_locator_map_has_detection_highlight(sample_bounds):
    """Test that locator map highlights detection area."""
    locator = create_locator_map(sample_bounds, detection_color=(255, 0, 0))

    # Should contain red pixels (detection highlight)
    red_pixels = np.any((locator[:, :, 0] > 200) & (locator[:, :, 1] < 50))
    assert red_pixels, "Locator map should contain red detection highlight"


# === Attribution Tests ===


def test_data_source_creation():
    """Test DataSource creation and validation."""
    source = DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25))

    assert source.name == "Sentinel-2"
    assert source.provider == "ESA"
    assert source.acquisition_date == datetime(2026, 1, 25)
    assert source.license is None


def test_data_source_with_license():
    """Test DataSource with license information."""
    source = DataSource(
        "OpenStreetMap",
        "OSM Contributors",
        license="ODbL"
    )

    assert source.name == "OpenStreetMap"
    assert source.provider == "OSM Contributors"
    assert source.license == "ODbL"


def test_data_source_validation():
    """Test that invalid DataSource raises error."""
    # Empty name
    with pytest.raises(ValueError, match="name and provider are required"):
        DataSource("", "Provider")

    # Empty provider
    with pytest.raises(ValueError, match="name and provider are required"):
        DataSource("Source", "")


def test_format_attribution_text_single_source():
    """Test formatting single data source."""
    sources = [DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25))]

    text = format_attribution_text(sources)
    assert text == "Data: Sentinel-2 (ESA, 2026-01-25)"


def test_format_attribution_text_multiple_sources(sample_sources):
    """Test formatting multiple data sources."""
    text = format_attribution_text(sample_sources)

    assert "Sentinel-2 (ESA, 2026-01-25)" in text
    assert "OpenStreetMap (OSM Contributors)" in text
    assert text.startswith("Data: ")


def test_format_attribution_text_no_sources():
    """Test formatting with no sources."""
    text = format_attribution_text([])
    assert text == "Data: No sources specified"


def test_format_attribution_text_no_date():
    """Test formatting source without acquisition date."""
    sources = [DataSource("OpenStreetMap", "OSM Contributors")]

    text = format_attribution_text(sources)
    assert text == "Data: OpenStreetMap (OSM Contributors)"


def test_create_attribution_block_basic(sample_sources):
    """Test basic attribution block creation."""
    attribution = create_attribution_block(sample_sources)

    # Check output is RGBA image
    assert attribution.ndim == 3
    assert attribution.shape[2] == 4
    assert attribution.dtype == np.uint8

    # Should have some content (not empty)
    assert attribution.shape[0] > 0
    assert attribution.shape[1] > 0


def test_create_attribution_block_positions(sample_sources):
    """Test attribution block with different positions."""
    positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

    for position in positions:
        attribution = create_attribution_block(sample_sources, position=position)
        assert attribution.shape[2] == 4
        assert attribution.dtype == np.uint8


def test_create_attribution_block_invalid_position(sample_sources):
    """Test that invalid position raises error."""
    with pytest.raises(ValueError, match="Invalid position"):
        create_attribution_block(sample_sources, position="middle")


def test_create_attribution_block_custom_colors(sample_sources):
    """Test attribution block with custom colors."""
    attribution = create_attribution_block(
        sample_sources,
        background_color=(200, 200, 200, 255),
        text_color=(255, 0, 0, 255),
    )

    assert attribution.shape[2] == 4


def test_create_attribution_block_custom_width(sample_sources):
    """Test attribution block with fixed width."""
    attribution = create_attribution_block(sample_sources, width=400)

    assert attribution.shape[1] == 400


def test_create_attribution_block_font_size(sample_sources):
    """Test attribution block with different font sizes."""
    small = create_attribution_block(sample_sources, font_size=8)
    large = create_attribution_block(sample_sources, font_size=14)

    # Larger font should produce taller block
    assert large.shape[0] > small.shape[0]


def test_add_license_info(sample_sources):
    """Test license information generation."""
    # Add license to first source
    sample_sources[0].license = "CC-BY-SA 3.0 IGO"
    sample_sources[1].license = "ODbL"

    license_text = add_license_info(sample_sources)

    assert "Sentinel-2" in license_text
    assert "CC-BY-SA 3.0 IGO" in license_text
    assert "OpenStreetMap" in license_text
    assert "ODbL" in license_text


def test_add_license_info_no_license():
    """Test license info without license field."""
    sources = [DataSource("Test", "Provider")]

    license_text = add_license_info(sources)
    assert "Test: Copyright Provider" in license_text


def test_add_license_info_empty():
    """Test license info with empty sources."""
    license_text = add_license_info([])
    assert license_text == "No license information available"


# === Convenience Functions Tests ===


def test_sentinel2_source():
    """Test Sentinel-2 convenience function."""
    date = datetime(2026, 1, 25)
    source = sentinel2_source(date)

    assert source.name == "Sentinel-2"
    assert source.provider == "ESA"
    assert source.acquisition_date == date
    assert source.license == "CC-BY-SA 3.0 IGO"


def test_landsat8_source():
    """Test Landsat-8 convenience function."""
    date = datetime(2026, 1, 20)
    source = landsat8_source(date)

    assert source.name == "Landsat-8"
    assert source.provider == "NASA/USGS"
    assert source.acquisition_date == date
    assert source.license == "Public Domain"


def test_osm_source():
    """Test OpenStreetMap convenience function."""
    source = osm_source()

    assert source.name == "OpenStreetMap"
    assert source.provider == "OSM Contributors"
    assert source.license == "ODbL"
    assert source.acquisition_date is None


# === Infrastructure Labels Tests ===


def test_add_infrastructure_labels_no_client():
    """Test infrastructure labeling without client returns unchanged image."""
    overlay = DetectionOverlay()
    image = np.full((100, 100, 4), 128, dtype=np.uint8)
    bounds = (-80.5, 25.5, -80.0, 26.0)

    result = overlay.add_infrastructure_labels(image, bounds, infrastructure_client=None)

    # Should return image unchanged
    assert np.array_equal(result, image)


def test_add_infrastructure_labels_with_types():
    """Test infrastructure labeling with specific types."""
    overlay = DetectionOverlay()
    image = np.full((100, 100, 4), 128, dtype=np.uint8)
    bounds = (-80.5, 25.5, -80.0, 26.0)

    # Without client, should return unchanged
    result = overlay.add_infrastructure_labels(
        image,
        bounds,
        infrastructure_client=None,
        label_types=["hospital", "fire_station"]
    )

    assert np.array_equal(result, image)


# === Integration Tests ===


def test_locator_and_attribution_integration():
    """Test that locator map and attribution can be composited together."""
    bounds = (-80.5, 25.5, -80.0, 26.0)
    sources = [
        sentinel2_source(datetime(2026, 1, 25)),
        osm_source(),
    ]

    # Create locator map
    locator = create_locator_map(bounds, size=(200, 200))

    # Create attribution block
    attribution = create_attribution_block(sources)

    # Both should be valid RGBA images
    assert locator.shape[2] == 4
    assert attribution.shape[2] == 4

    # Should be different sizes
    assert locator.shape != attribution.shape


def test_attribution_text_formatting_consistency():
    """Test that attribution text is formatted consistently."""
    sources = [
        DataSource("Source1", "Provider1", datetime(2026, 1, 1)),
        DataSource("Source2", "Provider2", datetime(2026, 1, 2)),
        DataSource("Source3", "Provider3"),
    ]

    text = format_attribution_text(sources)

    # All sources should be mentioned
    assert "Source1" in text
    assert "Source2" in text
    assert "Source3" in text

    # Dates should be formatted consistently
    assert "2026-01-01" in text
    assert "2026-01-02" in text

    # Should be comma-separated
    assert text.count(",") >= 2


def test_locator_map_bounds_validation():
    """Test comprehensive bounds validation."""
    valid_bounds = (-80.5, 25.5, -80.0, 26.0)

    # Valid bounds should work
    locator = create_locator_map(valid_bounds)
    assert locator.shape[2] == 4

    # Test various invalid bounds
    invalid_cases = [
        ((-80.0, 25.5, -80.5, 26.0), "minx >= maxx"),  # Inverted X
        ((-80.5, 26.0, -80.0, 25.5), "miny >= maxy"),  # Inverted Y
        ((-80.5, 25.5, -80.5, 26.0), "minx >= maxx"),  # Equal X
        ((-80.5, 25.5, -80.0, 25.5), "miny >= maxy"),  # Equal Y
    ]

    for bounds, error_msg in invalid_cases:
        with pytest.raises(ValueError, match="Invalid bounds"):
            create_locator_map(bounds)


def test_attribution_block_empty_sources():
    """Test attribution block with empty sources list."""
    attribution = create_attribution_block([])

    # Should still create a valid image
    assert attribution.shape[2] == 4
    assert attribution.dtype == np.uint8

    # Should contain "No sources" text
    # We can't easily check text content, but image should be non-empty
    assert attribution.shape[0] > 0
    assert attribution.shape[1] > 0
