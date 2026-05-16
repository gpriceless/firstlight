"""
Unit tests for static map generation — V4.1 Geographic Context Layer.

Tests cover:
- Basemap tile fetching with mocked contextily (correct provider passed)
- CRS auto-detection from mock raster
- Geographic axes: lat/lon gridlines present when cartopy available
- Title block rendering with event metadata
- Attribution block rendering with source info
- DPI presets (web: 144 DPI 1200x800, print: 300 DPI 3600x2400)
- Graceful degradation when contextily unavailable
"""

import io
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from dataclasses import dataclass

from core.reporting.maps.base import MapConfig, MapBounds, MapOutputPreset
from core.reporting.maps.static_map import StaticMapGenerator


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def small_bounds():
    """Small local extent: ~55 km x 55 km (triggers UTM selection)."""
    return MapBounds(
        min_lon=-82.2,
        min_lat=26.3,
        max_lon=-81.7,
        max_lat=26.8,
    )


@pytest.fixture
def large_bounds():
    """Large extent: ~1100 km x 1100 km (triggers Web Mercator selection)."""
    return MapBounds(
        min_lon=-90.0,
        min_lat=20.0,
        max_lon=-80.0,
        max_lat=30.0,
    )


@pytest.fixture
def flood_array_small():
    """2x2 boolean flood extent array."""
    return np.array([[False, True], [True, True]])


@pytest.fixture
def flood_array_10x10():
    """10x10 boolean flood extent."""
    arr = np.zeros((10, 10), dtype=bool)
    arr[3:7, 3:7] = True
    return arr


@pytest.fixture
def minimal_config():
    """Minimal MapConfig — no optional metadata."""
    return MapConfig(
        width=400,
        height=300,
        dpi=72,
        show_scale_bar=False,
        show_north_arrow=False,
        show_legend=False,
    )


@pytest.fixture
def full_event_config():
    """MapConfig with full event and attribution metadata."""
    return MapConfig(
        width=400,
        height=300,
        dpi=72,
        event_name="Hurricane Ian",
        location="Fort Myers, FL",
        event_date="2022-09-28",
        data_source="Sentinel-1 SAR",
        satellite_platform="ESA Sentinel-1A",
        acquisition_date="2022-09-29",
        processing_level="GRD",
        show_scale_bar=False,
        show_north_arrow=False,
        show_legend=False,
    )


def make_generator(config=None, *, has_contextily=True, has_cartopy=True, has_rasterio=True):
    """
    Build a StaticMapGenerator with optional dependencies mocked.

    Patches the matplotlib pyplot import so no display is needed, and
    optionally sets has_contextily/has_cartopy/has_rasterio flags.
    """
    mock_plt = MagicMock()
    # make subplots return a real-ish (fig, ax) pair
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.Blues = MagicMock()

    with patch("core.reporting.maps.static_map.StaticMapGenerator.__init__", lambda self, cfg=None: None):
        gen = StaticMapGenerator.__new__(StaticMapGenerator)

    gen.config = config or MapConfig(
        width=400, height=300, dpi=72,
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )
    gen.plt = mock_plt
    gen.has_contextily = has_contextily
    gen.has_cartopy = has_cartopy
    gen.has_rasterio = has_rasterio
    gen.has_scalebar = False

    if has_contextily:
        gen.ctx = MagicMock()
        gen.ctx.providers.CartoDB.Positron = "CartoDB.Positron"
        gen.ctx.providers.OpenStreetMap.Mapnik = "OSM.Mapnik"

    if has_cartopy:
        gen.ccrs = MagicMock()
        gen.cfeature = MagicMock()

    if has_rasterio:
        gen.rasterio = MagicMock()

    return gen, mock_plt, mock_ax, mock_fig


# ===========================================================================
# DPI Presets (Task 3)
# ===========================================================================


def test_web_preset_dimensions():
    """web() preset returns 1200x800 at 144 DPI."""
    preset = MapOutputPreset.web()
    assert preset.width == 1200
    assert preset.height == 800
    assert preset.dpi == 144


def test_print_preset_dimensions():
    """print() preset returns 3600x2400 at 300 DPI."""
    preset = MapOutputPreset.print()
    assert preset.width == 3600
    assert preset.height == 2400
    assert preset.dpi == 300


def test_default_dpi_unchanged():
    """Default MapConfig DPI remains 150 — no regression."""
    config = MapConfig()
    assert config.dpi == 150


def test_from_preset_wires_dimensions():
    """MapConfig.from_preset() copies width/height/dpi from preset."""
    preset = MapOutputPreset.web()
    config = MapConfig.from_preset(preset, title="Test Map")
    assert config.width == 1200
    assert config.height == 800
    assert config.dpi == 144
    assert config.title == "Test Map"


def test_from_preset_print_wires_dimensions():
    """MapConfig.from_preset(print()) copies 3600x2400@300."""
    config = MapConfig.from_preset(MapOutputPreset.print())
    assert config.width == 3600
    assert config.height == 2400
    assert config.dpi == 300


def test_savefig_uses_config_dpi(tmp_path, small_bounds, flood_array_small):
    """generate_flood_map() saves figure using config.dpi."""
    gen, mock_plt, mock_ax, mock_fig = make_generator(
        config=MapConfig(width=400, height=300, dpi=144,
                         show_scale_bar=False, show_north_arrow=False, show_legend=False),
        has_contextily=False, has_cartopy=False,
    )
    out = tmp_path / "test.png"
    gen.generate_flood_map(flood_array_small, small_bounds, out)
    mock_plt.savefig.assert_called_once()
    _, save_kwargs = mock_plt.savefig.call_args
    assert save_kwargs.get("dpi") == 144


# ===========================================================================
# Basemap Tile Fetching (Task 1)
# ===========================================================================


def test_get_basemap_tiles_uses_primary_provider(small_bounds):
    """_get_basemap_tiles() calls add_basemap with CartoDB.Positron first."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()

    gen._get_basemap_tiles(small_bounds, ax=mock_ax)

    gen.ctx.add_basemap.assert_called_once()
    call_kwargs = gen.ctx.add_basemap.call_args[1]
    assert call_kwargs["source"] == "CartoDB.Positron"


def test_get_basemap_tiles_fallback_on_primary_failure(small_bounds):
    """_get_basemap_tiles() falls back to OpenStreetMap.Mapnik if primary fails."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()

    # Primary raises; fallback should succeed
    gen.ctx.add_basemap.side_effect = [Exception("tile server error"), None]

    gen._get_basemap_tiles(small_bounds, ax=mock_ax)

    assert gen.ctx.add_basemap.call_count == 2
    fallback_call = gen.ctx.add_basemap.call_args_list[1]
    assert fallback_call[1]["source"] == "OSM.Mapnik"


def test_get_basemap_tiles_standalone_uses_bounds2img(small_bounds):
    """_get_basemap_tiles() without ax calls bounds2img with correct coords."""
    gen, *_ = make_generator()
    gen.ctx.bounds2img.return_value = (np.zeros((10, 10, 3)), (-82.2, 26.3, -81.7, 26.8))

    result = gen._get_basemap_tiles(small_bounds, ax=None)

    gen.ctx.bounds2img.assert_called_once()
    call_args = gen.ctx.bounds2img.call_args
    positional = call_args[0]
    assert positional[0] == small_bounds.min_lon
    assert positional[1] == small_bounds.min_lat
    assert positional[2] == small_bounds.max_lon
    assert positional[3] == small_bounds.max_lat
    assert result is not None


def test_get_basemap_tiles_standalone_fallback(small_bounds):
    """_get_basemap_tiles() standalone falls back to OSM if primary bounds2img fails."""
    gen, *_ = make_generator()
    gen.ctx.bounds2img.side_effect = [
        Exception("primary failed"),
        (np.zeros((10, 10, 3)), (-82.2, 26.3, -81.7, 26.8)),
    ]

    result = gen._get_basemap_tiles(small_bounds, ax=None)

    assert gen.ctx.bounds2img.call_count == 2
    # Second call should use fallback source
    fallback_call = gen.ctx.bounds2img.call_args_list[1]
    assert fallback_call[1]["source"] == "OSM.Mapnik"


def test_get_basemap_tiles_returns_none_without_contextily(small_bounds):
    """_get_basemap_tiles() returns None gracefully when contextily not installed."""
    gen, *_ = make_generator(has_contextily=False)

    result = gen._get_basemap_tiles(small_bounds, ax=None)

    assert result is None


# ===========================================================================
# CRS Auto-Detection (Task 1)
# ===========================================================================


def test_detect_crs_from_raster_returns_crs_string():
    """_detect_crs_from_raster() returns CRS string from mock rasterio dataset."""
    gen, *_ = make_generator()

    mock_crs = MagicMock()
    mock_crs.to_string.return_value = "EPSG:4326"
    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: mock_ds
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = mock_crs
    gen.rasterio.open.return_value = mock_ds

    result = gen._detect_crs_from_raster("/fake/path.tif")

    assert result == "EPSG:4326"
    gen.rasterio.open.assert_called_once_with("/fake/path.tif")


def test_detect_crs_from_raster_handles_none_crs():
    """_detect_crs_from_raster() returns None when raster has no CRS."""
    gen, *_ = make_generator()

    mock_ds = MagicMock()
    mock_ds.__enter__ = lambda s: mock_ds
    mock_ds.__exit__ = MagicMock(return_value=False)
    mock_ds.crs = None
    gen.rasterio.open.return_value = mock_ds

    result = gen._detect_crs_from_raster("/fake/path.tif")

    assert result is None


def test_detect_crs_from_raster_handles_open_error():
    """_detect_crs_from_raster() returns None when rasterio.open raises."""
    gen, *_ = make_generator()

    gen.rasterio.open.side_effect = Exception("file not found")

    result = gen._detect_crs_from_raster("/missing/file.tif")

    assert result is None


def test_detect_crs_returns_none_without_rasterio():
    """_detect_crs_from_raster() returns None when rasterio not installed."""
    gen, *_ = make_generator(has_rasterio=False)

    result = gen._detect_crs_from_raster("/fake/path.tif")

    assert result is None


def test_source_crs_kwarg_passed_to_basemap(small_bounds, flood_array_small, tmp_path):
    """generate_flood_map() passes source_crs kwarg into _add_basemap_with_projection."""
    gen, mock_plt, mock_ax, _ = make_generator(has_contextily=True, has_cartopy=True)
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    recorded_crs = {}

    def capture_basemap(ax, bounds, source_crs=None, raster_path=None):
        recorded_crs["source_crs"] = source_crs

    gen._add_basemap_with_projection = capture_basemap

    out = tmp_path / "test.png"
    gen.generate_flood_map(flood_array_small, small_bounds, out, source_crs="EPSG:32617")

    assert recorded_crs.get("source_crs") == "EPSG:32617"


def test_config_source_crs_used_as_fallback(small_bounds, flood_array_small, tmp_path):
    """generate_flood_map() uses config.source_crs when kwarg not given."""
    config = MapConfig(
        width=400, height=300, dpi=72,
        source_crs="EPSG:32617",
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )
    gen, mock_plt, mock_ax, _ = make_generator(config=config, has_contextily=True, has_cartopy=True)
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    recorded_crs = {}

    def capture_basemap(ax, bounds, source_crs=None, raster_path=None):
        recorded_crs["source_crs"] = source_crs

    gen._add_basemap_with_projection = capture_basemap

    out = tmp_path / "test.png"
    gen.generate_flood_map(flood_array_small, small_bounds, out)

    assert recorded_crs.get("source_crs") == "EPSG:32617"


# ===========================================================================
# Geographic Axes — Gridlines (Task 1)
# ===========================================================================


def test_add_basemap_with_projection_calls_gridlines(small_bounds):
    """_add_basemap_with_projection() enables lat/lon gridlines on axes."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()
    mock_gl = MagicMock()
    mock_ax.gridlines.return_value = mock_gl

    gen._add_basemap_with_projection(mock_ax, small_bounds)

    mock_ax.gridlines.assert_called_once()
    # draw_labels should be True so axis labels appear
    call_kwargs = mock_ax.gridlines.call_args[1]
    assert call_kwargs.get("draw_labels") is True


def test_add_basemap_with_projection_sets_extent(small_bounds):
    """_add_basemap_with_projection() sets axes extent to bounds."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()

    gen._add_basemap_with_projection(mock_ax, small_bounds)

    mock_ax.set_extent.assert_called_once()
    extent_args = mock_ax.set_extent.call_args[0][0]
    assert extent_args == small_bounds.bbox


def test_add_basemap_with_projection_no_cartopy(small_bounds):
    """_add_basemap_with_projection() is a no-op when cartopy unavailable."""
    gen, *_ = make_generator(has_cartopy=False)
    mock_ax = MagicMock()

    # Should not raise
    gen._add_basemap_with_projection(mock_ax, small_bounds)

    mock_ax.set_extent.assert_not_called()


def test_gridlines_failure_does_not_propagate(small_bounds):
    """_add_basemap_with_projection() survives gridlines() exception."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()
    mock_ax.gridlines.side_effect = Exception("gridlines not supported")

    # Should not raise
    gen._add_basemap_with_projection(mock_ax, small_bounds)

    mock_ax.set_extent.assert_called_once()


# ===========================================================================
# Title Block (Task 2)
# ===========================================================================


def test_title_block_renders_event_name(full_event_config):
    """_add_title_block() calls set_title with event_name when set."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax)

    mock_ax.set_title.assert_called_once()
    call_args = mock_ax.set_title.call_args[0]
    assert "Hurricane Ian" in call_args[0]


def test_title_block_renders_location_and_date(full_event_config):
    """_add_title_block() includes location and event_date in subtitle."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax)

    # Subtitle is rendered via ax.text
    mock_ax.text.assert_called()
    text_content = mock_ax.text.call_args[0][2]  # third positional arg is the string
    assert "Fort Myers, FL" in text_content
    assert "2022-09-28" in text_content


def test_title_block_override_title_takes_precedence(full_event_config):
    """_add_title_block() uses override_title when provided."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax, override_title="Custom Override")

    mock_ax.set_title.assert_called_once()
    assert "Custom Override" in mock_ax.set_title.call_args[0][0]


def test_title_block_falls_back_to_config_title():
    """_add_title_block() falls back to config.title when event_name not set."""
    config = MapConfig(title="Fallback Title", show_scale_bar=False,
                       show_north_arrow=False, show_legend=False)
    gen, *_ = make_generator(config=config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax)

    mock_ax.set_title.assert_called_once()
    assert "Fallback Title" in mock_ax.set_title.call_args[0][0]


def test_title_block_no_title_no_call():
    """_add_title_block() does not call set_title when no title configured."""
    config = MapConfig(show_scale_bar=False, show_north_arrow=False, show_legend=False)
    gen, *_ = make_generator(config=config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax)

    mock_ax.set_title.assert_not_called()


def test_title_block_single_line_no_metadata():
    """_add_title_block() renders single line when only title (no location/date)."""
    config = MapConfig(
        title="Simple Title",
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )
    gen, *_ = make_generator(config=config)
    mock_ax = MagicMock()

    gen._add_title_block(mock_ax)

    # set_title called; ax.text should NOT be called (no subtitle)
    mock_ax.set_title.assert_called_once()
    mock_ax.text.assert_not_called()


# ===========================================================================
# Attribution Block (Task 2)
# ===========================================================================


def test_attribution_block_renders_satellite_platform(full_event_config):
    """_add_attribution_block() includes satellite_platform in text."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    mock_ax.text.assert_called_once()
    text_content = mock_ax.text.call_args[0][2]
    assert "Sentinel-1A" in text_content


def test_attribution_block_renders_acquisition_date(full_event_config):
    """_add_attribution_block() includes acquisition_date in text."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    text_content = mock_ax.text.call_args[0][2]
    assert "2022-09-29" in text_content


def test_attribution_block_renders_processing_level(full_event_config):
    """_add_attribution_block() includes processing_level in text."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    text_content = mock_ax.text.call_args[0][2]
    assert "GRD" in text_content


def test_attribution_block_renders_data_source(full_event_config):
    """_add_attribution_block() includes data_source in text."""
    gen, *_ = make_generator(config=full_event_config)
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    text_content = mock_ax.text.call_args[0][2]
    assert "Sentinel-1 SAR" in text_content


def test_attribution_block_falls_back_to_legacy_string():
    """_add_attribution_block() uses attribution string when no metadata set."""
    config = MapConfig(
        attribution="© OpenStreetMap contributors",
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )
    gen, *_ = make_generator(config=config)
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    text_content = mock_ax.text.call_args[0][2]
    assert "OpenStreetMap" in text_content


def test_attribution_block_placed_bottom_right():
    """_add_attribution_block() places text at bottom-right (x=0.99, y=0.01)."""
    gen, *_ = make_generator()
    mock_ax = MagicMock()

    gen._add_attribution_block(mock_ax)

    args = mock_ax.text.call_args[0]
    assert args[0] == pytest.approx(0.99)
    assert args[1] == pytest.approx(0.01)


# ===========================================================================
# Graceful Degradation (Task 1)
# ===========================================================================


def test_generate_flood_map_without_contextily(tmp_path, small_bounds, flood_array_small):
    """generate_flood_map() runs without contextily — falls back to bbox plot."""
    gen, mock_plt, mock_ax, _ = make_generator(has_contextily=False, has_cartopy=False)
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    out = tmp_path / "test.png"
    result = gen.generate_flood_map(flood_array_small, small_bounds, out)

    assert result == out
    mock_plt.savefig.assert_called_once()
    # Fallback path sets xlim/ylim, not set_extent
    mock_ax.set_xlim.assert_called_once_with(small_bounds.min_lon, small_bounds.max_lon)
    mock_ax.set_ylim.assert_called_once_with(small_bounds.min_lat, small_bounds.max_lat)


def test_generate_flood_map_contextily_exception_does_not_crash(tmp_path, small_bounds, flood_array_small):
    """generate_flood_map() completes even when basemap tile fetch raises."""
    gen, mock_plt, mock_ax, _ = make_generator(has_contextily=True, has_cartopy=True)
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    # Both primary and fallback fail
    gen.ctx.add_basemap.side_effect = Exception("network error")

    out = tmp_path / "test.png"
    result = gen.generate_flood_map(flood_array_small, small_bounds, out)

    assert result == out
    mock_plt.savefig.assert_called_once()


def test_generate_infrastructure_map_without_contextily(tmp_path, small_bounds):
    """generate_infrastructure_map() runs without contextily."""
    gen, mock_plt, mock_ax, _ = make_generator(has_contextily=False, has_cartopy=False)
    mock_plt.subplots.return_value = (MagicMock(), mock_ax)

    infrastructure = [{"lon": -82.0, "lat": 26.5, "type": "hospital"}]
    out = tmp_path / "infra.png"
    result = gen.generate_infrastructure_map(infrastructure, small_bounds, output_path=out)

    assert result == out
    mock_plt.savefig.assert_called_once()


# ===========================================================================
# MapConfig Backward Compatibility
# ===========================================================================


def test_mapconfig_defaults_unchanged():
    """MapConfig default values have not regressed."""
    config = MapConfig()
    assert config.width == 800
    assert config.height == 600
    assert config.dpi == 150
    assert config.show_scale_bar is True
    assert config.show_north_arrow is True
    assert config.show_legend is True
    assert config.attribution == "© OpenStreetMap contributors"


def test_mapconfig_new_fields_default_to_none():
    """New optional MapConfig fields all default to None."""
    config = MapConfig()
    assert config.event_name is None
    assert config.location is None
    assert config.event_date is None
    assert config.data_source is None
    assert config.satellite_platform is None
    assert config.acquisition_date is None
    assert config.processing_level is None
    assert config.source_crs is None


def test_mapbounds_is_unchanged():
    """MapBounds has not gained any new fields (read-only contract preserved)."""
    b = MapBounds(min_lon=-82.0, min_lat=26.0, max_lon=-81.0, max_lat=27.0)
    assert b.bbox == (-82.0, 26.0, -81.0, 27.0)
    assert b.center == (-81.5, 26.5)
    assert b.width_degrees == pytest.approx(1.0)
    assert b.height_degrees == pytest.approx(1.0)
