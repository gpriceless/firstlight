"""
Smoke test: Phase 4a end-to-end exercise.

Exercises the complete Phase 4a feature set in a single integrated pass:
  - MapConfig with all new title-block and attribution-block fields
  - MapOutputPreset.web() and MapOutputPreset.print() factory presets
  - StaticMapGenerator with mocked contextily (_get_basemap_tiles flow)
  - _render_perimeter_overlay with a real GeoDataFrame and containment metric
  - generate_flood_map() with perimeter GeoDataFrame
  - generate_fire_map() with perimeter GeoDataFrame
  - ReportVisualPipeline.generate_static_map() pipeline entry point
  - Graceful degradation: contextily disabled → no crash
  - Graceful degradation: perimeter=None → no crash

All external calls (contextily tile fetches, matplotlib display) are mocked.
No real network or filesystem access is required beyond tmp_path.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.reporting.maps.base import MapConfig, MapBounds, MapOutputPreset
from core.reporting.maps.static_map import StaticMapGenerator
from core.reporting.data.perimeter_loader import PerimeterLoader


# ===========================================================================
# Shared helpers
# ===========================================================================


def _make_generator(config=None, *, has_contextily=True, has_cartopy=False):
    """
    Construct a StaticMapGenerator with mocked plt and optional contextily.

    Bypasses __init__ so no real matplotlib display is needed. Mirrors the
    pattern used across test_static_map.py and test_perimeter_overlay.py.
    """
    mock_plt = MagicMock()
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.Blues = MagicMock()
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    with patch(
        "core.reporting.maps.static_map.StaticMapGenerator.__init__",
        lambda self, cfg=None: None,
    ):
        gen = StaticMapGenerator.__new__(StaticMapGenerator)

    gen.config = config or MapConfig(
        width=400, height=300, dpi=72,
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )
    gen.plt = mock_plt
    gen.has_contextily = has_contextily
    gen.has_cartopy = has_cartopy
    gen.has_rasterio = False
    gen.has_scalebar = False

    if has_contextily:
        gen.ctx = MagicMock()
        gen.ctx.providers.CartoDB.Positron = "CartoDB.Positron"
        gen.ctx.providers.OpenStreetMap.Mapnik = "OSM.Mapnik"

    if has_cartopy:
        gen.ccrs = MagicMock()
        gen.cfeature = MagicMock()

    return gen, mock_plt, mock_ax, mock_fig


def _make_perimeter_gdf(minx=-82.0, miny=26.3, maxx=-81.5, maxy=26.7):
    """Return a real GeoDataFrame with a single polygon in WGS84."""
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoDataFrame(
        geometry=[box(minx, miny, maxx, maxy)],
        crs="EPSG:4326",
    )


FLOOD_BOUNDS = MapBounds(min_lon=-82.2, min_lat=26.3, max_lon=-81.7, max_lat=26.8)
FLOOD_ARRAY = np.array([[False, True], [True, True]])


def _full_event_config(**overrides):
    """MapConfig with all Phase 4a fields populated."""
    defaults = dict(
        width=400, height=300, dpi=72,
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
    defaults.update(overrides)
    return MapConfig(**defaults)


# ===========================================================================
# Smoke: MapOutputPreset factory presets
# ===========================================================================


class TestMapOutputPresetSmoke:
    """Verify both factory presets produce the documented dimensions and DPI."""

    def test_web_preset(self):
        """web() → 1200×800 at 144 DPI."""
        p = MapOutputPreset.web()
        assert p.width == 1200
        assert p.height == 800
        assert p.dpi == 144

    def test_print_preset(self):
        """print() → 3600×2400 at 300 DPI."""
        p = MapOutputPreset.print()
        assert p.width == 3600
        assert p.height == 2400
        assert p.dpi == 300

    def test_mapconfig_from_web_preset(self):
        """MapConfig.from_preset(web()) wires width/height/dpi correctly."""
        config = MapConfig.from_preset(MapOutputPreset.web(), event_name="Test Event")
        assert config.width == 1200
        assert config.dpi == 144
        assert config.event_name == "Test Event"

    def test_mapconfig_from_print_preset(self):
        """MapConfig.from_preset(print()) wires 3600×2400@300."""
        config = MapConfig.from_preset(MapOutputPreset.print())
        assert config.width == 3600
        assert config.dpi == 300


# ===========================================================================
# Smoke: MapConfig all Phase 4a fields
# ===========================================================================


class TestMapConfigPhase4aFields:
    """All new Phase 4a MapConfig fields are stored and accessible."""

    def test_all_new_fields_stored(self):
        """Creating MapConfig with all Phase 4a fields stores them correctly."""
        config = _full_event_config()
        assert config.event_name == "Hurricane Ian"
        assert config.location == "Fort Myers, FL"
        assert config.event_date == "2022-09-28"
        assert config.data_source == "Sentinel-1 SAR"
        assert config.satellite_platform == "ESA Sentinel-1A"
        assert config.acquisition_date == "2022-09-29"
        assert config.processing_level == "GRD"

    def test_defaults_are_none(self):
        """New fields default to None when not provided."""
        config = MapConfig()
        assert config.event_name is None
        assert config.location is None
        assert config.event_date is None
        assert config.data_source is None
        assert config.satellite_platform is None
        assert config.acquisition_date is None
        assert config.processing_level is None


# ===========================================================================
# Smoke: _add_title_block with full metadata
# ===========================================================================


class TestTitleBlockSmoke:
    """_add_title_block renders event name, location and date from MapConfig."""

    def test_title_block_uses_event_name(self):
        """_add_title_block sets title from event_name when no override given."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_title_block(mock_ax)
        # ax.set_title must have been called with the event name
        mock_ax.set_title.assert_called()
        call_args = mock_ax.set_title.call_args[0]
        assert call_args[0] == "Hurricane Ian"

    def test_title_block_subtitle_contains_location_and_date(self):
        """Subtitle text (ax.text) includes location and event_date."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_title_block(mock_ax)
        # The subtitle is placed via ax.text
        mock_ax.text.assert_called()
        text_call = mock_ax.text.call_args[0]
        subtitle = text_call[2]  # third positional arg is the text string
        assert "Fort Myers, FL" in subtitle
        assert "2022-09-28" in subtitle

    def test_title_block_override_takes_precedence(self):
        """override_title kwarg overrides event_name."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_title_block(mock_ax, override_title="Custom Title")
        call_args = mock_ax.set_title.call_args[0]
        assert call_args[0] == "Custom Title"

    def test_title_block_empty_config_no_call(self):
        """_add_title_block makes no ax.set_title call when all fields are None."""
        gen, _, mock_ax, _ = _make_generator(config=MapConfig(title=None))
        gen._add_title_block(mock_ax)
        mock_ax.set_title.assert_not_called()


# ===========================================================================
# Smoke: _add_attribution_block with full metadata
# ===========================================================================


class TestAttributionBlockSmoke:
    """_add_attribution_block renders satellite platform, acquisition date, etc."""

    def test_attribution_block_contains_platform(self):
        """ax.text call contains the satellite platform name."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_attribution_block(mock_ax)
        mock_ax.text.assert_called()
        text_arg = mock_ax.text.call_args[0][2]
        assert "ESA Sentinel-1A" in text_arg

    def test_attribution_block_contains_acquisition_date(self):
        """ax.text call includes the acquisition date."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_attribution_block(mock_ax)
        text_arg = mock_ax.text.call_args[0][2]
        assert "2022-09-29" in text_arg

    def test_attribution_block_contains_processing_level(self):
        """ax.text call includes the processing level."""
        gen, _, mock_ax, _ = _make_generator(config=_full_event_config())
        gen._add_attribution_block(mock_ax)
        text_arg = mock_ax.text.call_args[0][2]
        assert "GRD" in text_arg

    def test_attribution_block_falls_back_to_legacy_string(self):
        """When no metadata fields are set, falls back to config.attribution."""
        config = MapConfig(attribution="Custom Attribution Text")
        gen, _, mock_ax, _ = _make_generator(config=config)
        gen._add_attribution_block(mock_ax)
        text_arg = mock_ax.text.call_args[0][2]
        assert "Custom Attribution Text" in text_arg


# ===========================================================================
# Smoke: _get_basemap_tiles (mocked contextily)
# ===========================================================================


class TestBasemapTilesSmoke:
    """Verify _get_basemap_tiles dispatches to contextily with correct provider."""

    def test_tiles_added_to_axes_via_add_basemap(self):
        """When ax is provided, ctx.add_basemap is called."""
        gen, _, mock_ax, _ = _make_generator(has_contextily=True, has_cartopy=False)
        gen._get_basemap_tiles(FLOOD_BOUNDS, ax=mock_ax)
        gen.ctx.add_basemap.assert_called_once()

    def test_primary_provider_is_cartodb_positron(self):
        """Primary tile source is CartoDB.Positron."""
        gen, _, mock_ax, _ = _make_generator(has_contextily=True, has_cartopy=False)
        gen._get_basemap_tiles(FLOOD_BOUNDS, ax=mock_ax)
        call_kwargs = gen.ctx.add_basemap.call_args[1]
        assert call_kwargs["source"] == "CartoDB.Positron"

    def test_no_crash_when_contextily_disabled(self):
        """_get_basemap_tiles returns None gracefully when has_contextily=False."""
        gen, _, mock_ax, _ = _make_generator(has_contextily=False, has_cartopy=False)
        result = gen._get_basemap_tiles(FLOOD_BOUNDS, ax=mock_ax)
        assert result is None

    def test_fallback_used_when_primary_fails(self):
        """If primary provider raises, fallback OpenStreetMap.Mapnik is tried."""
        gen, _, mock_ax, _ = _make_generator(has_contextily=True, has_cartopy=False)
        # First call (primary) raises; second call (fallback) succeeds
        gen.ctx.add_basemap.side_effect = [Exception("tile fetch failed"), None]
        # Should not raise
        gen._get_basemap_tiles(FLOOD_BOUNDS, ax=mock_ax)
        assert gen.ctx.add_basemap.call_count == 2


# ===========================================================================
# Smoke: _render_perimeter_overlay with real GeoDataFrame
# ===========================================================================


class TestPerimeterOverlaySmoke:
    """Verify _render_perimeter_overlay works with a real GeoDataFrame."""

    def test_overlay_adds_legend(self):
        """_render_perimeter_overlay always calls ax.legend."""
        gen, _, mock_ax, _ = _make_generator()
        perimeter = _make_perimeter_gdf()
        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="fire")
        mock_ax.legend.assert_called_once()

    def test_flood_overlay_uses_blue(self):
        """event_type='flood' results in a blue legend handle."""
        gen, _, mock_ax, _ = _make_generator()
        perimeter = _make_perimeter_gdf()
        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="flood")
        handles = mock_ax.legend.call_args[1]["handles"]
        assert handles[-1].get_color() == "blue"

    def test_fire_overlay_uses_red(self):
        """event_type='fire' results in a red legend handle."""
        gen, _, mock_ax, _ = _make_generator()
        perimeter = _make_perimeter_gdf()
        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="fire")
        handles = mock_ax.legend.call_args[1]["handles"]
        assert handles[-1].get_color() == "red"

    def test_containment_label_present_when_analysis_provided(self):
        """When analysis_extent is given, containment % appears in legend label."""
        import geopandas as gpd
        from shapely.geometry import box

        gen, _, mock_ax, _ = _make_generator()
        # perimeter and analysis are the same square → 100 % containment
        perimeter = _make_perimeter_gdf(0.0, 0.0, 1.0, 1.0)
        analysis = gpd.GeoDataFrame(
            geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
        )
        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=analysis)
        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "100.0%" in combined


# ===========================================================================
# Smoke: generate_flood_map with perimeter
# ===========================================================================


class TestGenerateFloodMapSmoke:
    """End-to-end smoke for generate_flood_map() with Phase 4a perimeter support."""

    def test_flood_map_with_perimeter_calls_savefig(self, tmp_path):
        """generate_flood_map() with perimeter reaches plt.savefig without error."""
        config = _full_event_config()
        gen, mock_plt, mock_ax, _ = _make_generator(
            config=config, has_contextily=False, has_cartopy=False
        )
        out = tmp_path / "flood.png"
        perimeter = _make_perimeter_gdf()
        result = gen.generate_flood_map(FLOOD_ARRAY, FLOOD_BOUNDS, out, perimeter=perimeter)
        mock_plt.savefig.assert_called_once()
        assert result == out

    def test_flood_map_without_perimeter_no_crash(self, tmp_path):
        """generate_flood_map(perimeter=None) completes without error."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "flood_no_perim.png"
        result = gen.generate_flood_map(FLOOD_ARRAY, FLOOD_BOUNDS, out, perimeter=None)
        mock_plt.savefig.assert_called_once()
        assert result == out

    def test_flood_map_uses_config_dpi(self, tmp_path):
        """generate_flood_map() passes config.dpi to savefig."""
        config = MapConfig(width=400, height=300, dpi=144,
                           show_scale_bar=False, show_north_arrow=False, show_legend=False)
        gen, mock_plt, _, _ = _make_generator(config=config,
                                               has_contextily=False, has_cartopy=False)
        out = tmp_path / "dpi_test.png"
        gen.generate_flood_map(FLOOD_ARRAY, FLOOD_BOUNDS, out)
        _, save_kwargs = mock_plt.savefig.call_args
        assert save_kwargs.get("dpi") == 144


# ===========================================================================
# Smoke: generate_fire_map with perimeter
# ===========================================================================


class TestGenerateFireMapSmoke:
    """End-to-end smoke for generate_fire_map() with Phase 4a perimeter support."""

    def test_fire_map_with_perimeter_calls_savefig(self, tmp_path):
        """generate_fire_map() with perimeter reaches plt.savefig without error."""
        config = _full_event_config()
        gen, mock_plt, _, _ = _make_generator(
            config=config, has_contextily=False, has_cartopy=False
        )
        out = tmp_path / "fire.png"
        perimeter = _make_perimeter_gdf()
        result = gen.generate_fire_map(FLOOD_BOUNDS, out, perimeter=perimeter)
        mock_plt.savefig.assert_called_once()
        assert result == out

    def test_fire_map_without_perimeter_no_crash(self, tmp_path):
        """generate_fire_map(perimeter=None) completes without error."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "fire_no_perim.png"
        result = gen.generate_fire_map(FLOOD_BOUNDS, out, perimeter=None)
        mock_plt.savefig.assert_called_once()
        assert result == out

    def test_fire_map_with_full_event_config(self, tmp_path):
        """generate_fire_map() works with all Phase 4a MapConfig fields set."""
        config = _full_event_config()
        gen, mock_plt, _, _ = _make_generator(
            config=config, has_contextily=False, has_cartopy=False
        )
        out = tmp_path / "fire_full.png"
        gen.generate_fire_map(FLOOD_BOUNDS, out, title="Park Fire 2024")
        mock_plt.savefig.assert_called_once()


# ===========================================================================
# Smoke: ReportVisualPipeline.generate_static_map entry point
# ===========================================================================


class TestReportVisualPipelineSmoke:
    """Verify the pipeline entry point is callable and delegates correctly."""

    def test_generate_static_map_is_callable(self):
        """ReportVisualPipeline.generate_static_map is importable and callable."""
        from core.reporting.imagery.pipeline import ReportVisualPipeline
        assert callable(getattr(ReportVisualPipeline, "generate_static_map", None))

    def test_pipeline_generate_static_map_flood(self, tmp_path):
        """Pipeline generate_static_map delegates to StaticMapGenerator.generate_flood_map."""
        from core.reporting.imagery.pipeline import ReportVisualPipeline

        mock_gen = MagicMock()
        mock_gen.generate_flood_map.return_value = tmp_path / "flood.png"

        # StaticMapGenerator is imported locally inside generate_static_map,
        # so we patch it at its source module path.
        with patch(
            "core.reporting.maps.static_map.StaticMapGenerator",
            return_value=mock_gen,
        ):
            pipeline = ReportVisualPipeline.__new__(ReportVisualPipeline)
            result = pipeline.generate_static_map(
                bounds=FLOOD_BOUNDS,
                output_path=tmp_path / "flood.png",
                event_type="flood",
                flood_extent=FLOOD_ARRAY,
            )

        mock_gen.generate_flood_map.assert_called_once()
        assert result == tmp_path / "flood.png"

    def test_pipeline_generate_static_map_fire(self, tmp_path):
        """Pipeline generate_static_map delegates to StaticMapGenerator.generate_fire_map."""
        from core.reporting.imagery.pipeline import ReportVisualPipeline

        mock_gen = MagicMock()
        mock_gen.generate_fire_map.return_value = tmp_path / "fire.png"

        with patch(
            "core.reporting.maps.static_map.StaticMapGenerator",
            return_value=mock_gen,
        ):
            pipeline = ReportVisualPipeline.__new__(ReportVisualPipeline)
            result = pipeline.generate_static_map(
                bounds=FLOOD_BOUNDS,
                output_path=tmp_path / "fire.png",
                event_type="fire",
            )

        mock_gen.generate_fire_map.assert_called_once()
        assert result == tmp_path / "fire.png"

    def test_pipeline_generate_static_map_with_perimeter(self, tmp_path):
        """Pipeline passes perimeter GeoDataFrame through to the generator."""
        from core.reporting.imagery.pipeline import ReportVisualPipeline

        mock_gen = MagicMock()
        mock_gen.generate_fire_map.return_value = tmp_path / "fire_perim.png"
        perimeter = _make_perimeter_gdf()

        with patch(
            "core.reporting.maps.static_map.StaticMapGenerator",
            return_value=mock_gen,
        ):
            pipeline = ReportVisualPipeline.__new__(ReportVisualPipeline)
            pipeline.generate_static_map(
                bounds=FLOOD_BOUNDS,
                output_path=tmp_path / "fire_perim.png",
                event_type="fire",
                perimeter=perimeter,
            )

        call_kwargs = mock_gen.generate_fire_map.call_args[1]
        assert call_kwargs.get("perimeter") is perimeter


# ===========================================================================
# Smoke: PerimeterLoader import and API surface
# ===========================================================================


class TestPerimeterLoaderSmoke:
    """Verify PerimeterLoader is importable and its public API exists."""

    def test_perimeter_loader_is_importable(self):
        """PerimeterLoader class is importable from the expected path."""
        assert PerimeterLoader is not None

    def test_perimeter_loader_has_load_nifc_perimeter(self):
        """PerimeterLoader exposes load_nifc_perimeter method."""
        assert callable(getattr(PerimeterLoader, "load_nifc_perimeter", None))

    def test_perimeter_loader_has_load_nws_flood_polygon(self):
        """PerimeterLoader exposes load_nws_flood_polygon method."""
        assert callable(getattr(PerimeterLoader, "load_nws_flood_polygon", None))

    def test_perimeter_loader_has_load_from_file(self):
        """PerimeterLoader exposes load_from_file method."""
        assert callable(getattr(PerimeterLoader, "load_from_file", None))

    def test_nifc_network_failure_returns_none(self):
        """load_nifc_perimeter returns None and does not raise on network error."""
        import requests
        loader = PerimeterLoader()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("offline")):
            result = loader.load_nifc_perimeter("Park Fire", "2024-07-23")
        assert result is None

    def test_nws_network_failure_returns_none(self):
        """load_nws_flood_polygon returns None and does not raise on network error."""
        import requests
        loader = PerimeterLoader()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("offline")):
            result = loader.load_nws_flood_polygon("FLZ052")
        assert result is None

    def test_load_from_file_missing_path_returns_none(self):
        """load_from_file returns None when the file does not exist."""
        loader = PerimeterLoader()
        result = loader.load_from_file(Path("/nonexistent/perimeter.geojson"))
        assert result is None


# ===========================================================================
# Smoke: Graceful degradation — contextily disabled
# ===========================================================================


class TestGracefulDegradationSmoke:
    """No crash when optional dependencies are unavailable."""

    def test_no_contextily_flood_map_still_saves(self, tmp_path):
        """generate_flood_map() saves successfully when contextily is disabled."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "no_ctx.png"
        gen.generate_flood_map(FLOOD_ARRAY, FLOOD_BOUNDS, out)
        mock_plt.savefig.assert_called_once()

    def test_no_contextily_fire_map_still_saves(self, tmp_path):
        """generate_fire_map() saves successfully when contextily is disabled."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "no_ctx_fire.png"
        gen.generate_fire_map(FLOOD_BOUNDS, out)
        mock_plt.savefig.assert_called_once()

    def test_perimeter_none_flood_map_no_crash(self, tmp_path):
        """generate_flood_map(perimeter=None) does not raise any exception."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "none_perim.png"
        gen.generate_flood_map(FLOOD_ARRAY, FLOOD_BOUNDS, out, perimeter=None)
        mock_plt.savefig.assert_called_once()

    def test_perimeter_none_fire_map_no_crash(self, tmp_path):
        """generate_fire_map(perimeter=None) does not raise any exception."""
        gen, mock_plt, _, _ = _make_generator(has_contextily=False, has_cartopy=False)
        out = tmp_path / "none_perim_fire.png"
        gen.generate_fire_map(FLOOD_BOUNDS, out, perimeter=None)
        mock_plt.savefig.assert_called_once()

    def test_get_basemap_tiles_returns_none_when_no_contextily(self):
        """_get_basemap_tiles returns None immediately when has_contextily=False."""
        gen, _, mock_ax, _ = _make_generator(has_contextily=False)
        result = gen._get_basemap_tiles(FLOOD_BOUNDS, ax=mock_ax)
        assert result is None
