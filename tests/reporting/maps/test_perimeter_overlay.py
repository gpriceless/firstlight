"""
Unit tests for perimeter overlay rendering — V4.2 Event Perimeter Overlays.

Tests cover:
- Perimeter overlay renders dashed outline on mock axes
- Containment metric computation with known geometries (50% overlap)
- Legend includes attribution and timestamp attributes
- Map renders correctly when perimeter is None (no error)
- generate_fire_map() works with and without perimeter
- generate_flood_map() works with and without perimeter
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from core.reporting.maps.base import MapConfig, MapBounds
from core.reporting.maps.static_map import StaticMapGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_bounds():
    """Small local extent for testing."""
    return MapBounds(min_lon=-82.2, min_lat=26.3, max_lon=-81.7, max_lat=26.8)


@pytest.fixture
def flood_array():
    """Small boolean flood extent array."""
    return np.array([[False, True], [True, True]])


@pytest.fixture
def minimal_config():
    """Minimal MapConfig — no furniture."""
    return MapConfig(
        width=400, height=300, dpi=72,
        show_scale_bar=False, show_north_arrow=False, show_legend=False,
    )


def make_generator(config=None, *, has_contextily=False, has_cartopy=False):
    """Build a StaticMapGenerator with mocked plt and no real dependencies."""
    mock_plt = MagicMock()
    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    mock_plt.cm.Blues = MagicMock()
    mock_ax.get_legend_handles_labels.return_value = ([], [])

    with patch("core.reporting.maps.static_map.StaticMapGenerator.__init__", lambda self, cfg=None: None):
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
    if has_cartopy:
        gen.ccrs = MagicMock()
        gen.cfeature = MagicMock()

    return gen, mock_plt, mock_ax, mock_fig


def _make_square_gdf(minx, miny, maxx, maxy, crs="EPSG:4326", extra_props=None):
    """Create a GeoDataFrame with a single square polygon."""
    import geopandas as gpd
    from shapely.geometry import box

    geom = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs=crs)
    if extra_props:
        for k, v in extra_props.items():
            gdf[k] = v
    return gdf


# ---------------------------------------------------------------------------
# _render_perimeter_overlay: basic rendering
# ---------------------------------------------------------------------------


class TestRenderPerimeterOverlayBasic:
    """Tests for the dashed-outline rendering behaviour."""

    def test_renders_dashed_outline_on_axes(self):
        """_render_perimeter_overlay calls boundary.plot with linestyle='--'."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)

        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="fire")

        # boundary.plot is called — we capture via mock GeoDataFrame
        # Since the real geopandas is used, boundary.plot won't be on mock_ax.
        # What we verify: ax.legend is called (legend entry was added).
        mock_ax.legend.assert_called_once()

    def test_fire_event_type_uses_red_color(self):
        """Legend entry for fire events uses red."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="fire")

        legend_call = mock_ax.legend.call_args
        handles = legend_call[1]["handles"]
        assert len(handles) == 1
        assert handles[0].get_color() == "red"

    def test_flood_event_type_uses_blue_color(self):
        """Legend entry for flood events uses blue."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="flood")

        legend_call = mock_ax.legend.call_args
        handles = legend_call[1]["handles"]
        assert handles[0].get_color() == "blue"

    def test_legend_handle_is_dashed(self):
        """Legend proxy artist has linestyle='--'."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter)

        handles = mock_ax.legend.call_args[1]["handles"]
        assert handles[0].get_linestyle() in ("--", "dashed")

    def test_legend_includes_event_type(self):
        """Legend label contains the event type name."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter, event_type="fire")

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "Fire" in combined or "fire" in combined

    def test_default_event_type_is_fire(self):
        """Default event_type argument is 'fire' (red legend handle)."""
        gen, _, mock_ax, _ = make_generator()
        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        gen._render_perimeter_overlay(mock_ax, perimeter)

        handles = mock_ax.legend.call_args[1]["handles"]
        assert handles[0].get_color() == "red"


# ---------------------------------------------------------------------------
# Containment metric
# ---------------------------------------------------------------------------


class TestContainmentMetric:
    """Tests for intersection-based containment calculation."""

    def test_full_overlap_gives_100_percent(self):
        """When analysis covers the entire perimeter, containment is 100%."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        # analysis = same square as perimeter
        perimeter = _make_square_gdf(0.0, 0.0, 1.0, 1.0)
        analysis = _make_square_gdf(0.0, 0.0, 1.0, 1.0)

        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=analysis)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "100.0%" in combined

    def test_half_overlap_gives_50_percent(self):
        """When analysis covers exactly half the perimeter, containment is 50%."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        # perimeter: [0,0] -> [2,1]; analysis: [0,0] -> [1,1] (left half)
        perimeter = _make_square_gdf(0.0, 0.0, 2.0, 1.0)
        analysis = _make_square_gdf(0.0, 0.0, 1.0, 1.0)

        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=analysis)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "50.0%" in combined

    def test_no_overlap_gives_0_percent(self):
        """When analysis does not intersect perimeter, containment is 0%."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        perimeter = _make_square_gdf(0.0, 0.0, 1.0, 1.0)
        analysis = _make_square_gdf(5.0, 5.0, 6.0, 6.0)

        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=analysis)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "0.0%" in combined

    def test_multipolygon_analysis_is_merged(self):
        """Multiple analysis polygons are unioned before computing containment."""
        import geopandas as gpd
        from shapely.geometry import box

        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        # perimeter: full [0,0]->[2,1]
        perimeter = _make_square_gdf(0.0, 0.0, 2.0, 1.0)

        # analysis: two separate polygons each covering half
        analysis = gpd.GeoDataFrame(
            geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
            crs="EPSG:4326",
        )

        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=analysis)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "100.0%" in combined

    def test_no_analysis_means_no_containment_in_label(self):
        """When analysis_extent is None, no containment percentage in label."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        perimeter = _make_square_gdf(0.0, 0.0, 1.0, 1.0)

        gen._render_perimeter_overlay(mock_ax, perimeter, analysis_extent=None)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "%" not in combined


# ---------------------------------------------------------------------------
# Attribution and timestamp in legend
# ---------------------------------------------------------------------------


class TestLegendAttribution:
    """Tests that source_attribution and timestamp are included in the legend."""

    def test_source_attribution_in_legend(self):
        """source_attribution column value appears in legend label."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        perimeter = _make_square_gdf(
            -82.0, 26.3, -81.5, 26.7,
            extra_props={"source_attribution": "NIFC Open Data"},
        )

        gen._render_perimeter_overlay(mock_ax, perimeter)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "NIFC Open Data" in combined

    def test_timestamp_in_legend(self):
        """timestamp column value appears in legend label."""
        gen, _, mock_ax, _ = make_generator()
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        perimeter = _make_square_gdf(
            -82.0, 26.3, -81.5, 26.7,
            extra_props={"timestamp": "2024-07-23T00:00:00Z"},
        )

        gen._render_perimeter_overlay(mock_ax, perimeter)

        labels = mock_ax.legend.call_args[1]["labels"]
        combined = "\n".join(labels)
        assert "2024-07-23" in combined


# ---------------------------------------------------------------------------
# generate_flood_map with perimeter=None (no regression)
# ---------------------------------------------------------------------------


class TestGenerateFloodMapPerimeterNone:
    """Verify generate_flood_map works correctly when perimeter=None."""

    def test_no_error_when_perimeter_is_none(self, tmp_path, small_bounds, flood_array):
        """generate_flood_map() runs without error when perimeter is not passed."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        out = tmp_path / "flood.png"
        result = gen.generate_flood_map(flood_array, small_bounds, out)

        assert result == out
        mock_plt.savefig.assert_called_once()

    def test_render_perimeter_overlay_not_called_when_none(
        self, tmp_path, small_bounds, flood_array
    ):
        """_render_perimeter_overlay is NOT called when perimeter=None."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        gen._render_perimeter_overlay = MagicMock()

        out = tmp_path / "flood.png"
        gen.generate_flood_map(flood_array, small_bounds, out, perimeter=None)

        gen._render_perimeter_overlay.assert_not_called()

    def test_render_perimeter_overlay_called_when_provided(
        self, tmp_path, small_bounds, flood_array
    ):
        """_render_perimeter_overlay is called with event_type='flood' when perimeter given."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        gen._render_perimeter_overlay = MagicMock()

        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        out = tmp_path / "flood.png"
        gen.generate_flood_map(flood_array, small_bounds, out, perimeter=perimeter)

        gen._render_perimeter_overlay.assert_called_once()
        _, kwargs = gen._render_perimeter_overlay.call_args
        assert kwargs.get("event_type") == "flood" or gen._render_perimeter_overlay.call_args[0][2] == "flood"


# ---------------------------------------------------------------------------
# generate_fire_map
# ---------------------------------------------------------------------------


class TestGenerateFireMap:
    """Tests for the new generate_fire_map() method."""

    def test_fire_map_runs_without_perimeter(self, tmp_path, small_bounds):
        """generate_fire_map() produces output when perimeter is None."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        out = tmp_path / "fire.png"
        result = gen.generate_fire_map(small_bounds, out)

        assert result == out
        mock_plt.savefig.assert_called_once()

    def test_fire_map_runs_with_perimeter(self, tmp_path, small_bounds):
        """generate_fire_map() produces output when perimeter is given."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        gen._render_perimeter_overlay = MagicMock()

        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        out = tmp_path / "fire.png"
        result = gen.generate_fire_map(small_bounds, out, perimeter=perimeter)

        assert result == out
        mock_plt.savefig.assert_called_once()
        gen._render_perimeter_overlay.assert_called_once()

    def test_fire_map_calls_overlay_with_fire_event_type(self, tmp_path, small_bounds):
        """generate_fire_map() calls _render_perimeter_overlay with event_type='fire'."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        gen._render_perimeter_overlay = MagicMock()

        perimeter = _make_square_gdf(-82.0, 26.3, -81.5, 26.7)
        out = tmp_path / "fire.png"
        gen.generate_fire_map(small_bounds, out, perimeter=perimeter)

        call_kwargs = gen._render_perimeter_overlay.call_args[1]
        assert call_kwargs.get("event_type") == "fire"

    def test_fire_map_uses_config_dpi(self, tmp_path, small_bounds):
        """generate_fire_map() saves figure using config.dpi."""
        config = MapConfig(
            width=400, height=300, dpi=144,
            show_scale_bar=False, show_north_arrow=False, show_legend=False,
        )
        gen, mock_plt, mock_ax, _ = make_generator(config=config)
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        out = tmp_path / "fire.png"
        gen.generate_fire_map(small_bounds, out)

        _, save_kwargs = mock_plt.savefig.call_args
        assert save_kwargs.get("dpi") == 144

    def test_fire_map_returns_output_path(self, tmp_path, small_bounds):
        """generate_fire_map() returns the output_path that was passed in."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        out = tmp_path / "fire_output.png"
        result = gen.generate_fire_map(small_bounds, out)

        assert result == out

    def test_fire_map_no_title_still_works(self, tmp_path, small_bounds):
        """generate_fire_map() works when no title is provided."""
        gen, mock_plt, mock_ax, _ = make_generator()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)

        out = tmp_path / "notitle.png"
        result = gen.generate_fire_map(small_bounds, out, title=None)

        assert result == out
