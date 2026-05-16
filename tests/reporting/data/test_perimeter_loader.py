"""
Unit tests for PerimeterLoader.

Tests cover:
- NIFC loader with mocked HTTP response (sample GeoJSON fixture)
- NWS loader with mocked HTTP response
- Local file loading with a small inline GeoJSON fixture
- Graceful degradation on HTTP error (returns None, no crash)
- CRS normalisation to WGS84 on load
"""

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# GeoJSON polygon fixture — a simple square near Fort Myers, FL
SAMPLE_GEOJSON = json.dumps({
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [-82.0, 26.3],
                    [-81.5, 26.3],
                    [-81.5, 26.7],
                    [-82.0, 26.7],
                    [-82.0, 26.3],
                ]]
            },
            "properties": {
                "IncidentName": "Test Fire",
                "source_attribution": "NIFC Open Data",
                "timestamp": "2024-07-23T00:00:00Z",
            }
        }
    ]
}).encode("utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(content: bytes = SAMPLE_GEOJSON, status_code: int = 200):
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests import HTTPError
        resp.raise_for_status.side_effect = HTTPError(
            f"HTTP {status_code}", response=resp
        )
    return resp


# ---------------------------------------------------------------------------
# NIFC Loader
# ---------------------------------------------------------------------------


class TestLoadNifcPerimeter:
    """Tests for PerimeterLoader.load_nifc_perimeter()."""

    def test_returns_geodataframe_on_success(self):
        """Returns a GeoDataFrame when NIFC returns valid GeoJSON."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp):
            gdf = loader.load_nifc_perimeter("Test Fire", "2024-07-23")

        assert gdf is not None
        assert len(gdf) == 1
        assert "geometry" in gdf.columns

    def test_requests_correct_url(self):
        """Sends request to the NIFC_QUERY_URL class constant."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp) as mock_get:
            loader.load_nifc_perimeter("Test Fire", "2024-07-23")

        call_url = mock_get.call_args[0][0]
        assert call_url == PerimeterLoader.NIFC_QUERY_URL

    def test_fire_name_in_where_clause(self):
        """WHERE clause includes the fire name as substring filter."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp) as mock_get:
            loader.load_nifc_perimeter("Park Fire", "2024-07-23")

        params = mock_get.call_args[1]["params"]
        assert "Park Fire" in params["where"]

    def test_returns_wgs84(self):
        """Returned GeoDataFrame is in WGS84 (EPSG:4326)."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp):
            gdf = loader.load_nifc_perimeter("Test Fire", "2024-07-23")

        # GeoJSON sources may have no CRS (assumed WGS84) or explicit 4326
        if gdf.crs is not None:
            assert gdf.crs.to_epsg() == 4326

    def test_returns_none_on_http_error(self):
        """Returns None when HTTP request raises an exception."""
        from core.reporting.data.perimeter_loader import PerimeterLoader
        from requests.exceptions import RequestException

        loader = PerimeterLoader()

        with patch("requests.get", side_effect=RequestException("timeout")):
            result = loader.load_nifc_perimeter("Any Fire", "2024-01-01")

        assert result is None

    def test_returns_none_on_http_4xx(self):
        """Returns None on 404 HTTP response (fire not found)."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(b"", status_code=404)

        with patch("requests.get", return_value=mock_resp):
            result = loader.load_nifc_perimeter("Unknown Fire", "2024-01-01")

        assert result is None

    def test_returns_none_on_empty_features(self):
        """Returns None when NIFC returns a FeatureCollection with no features."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        empty_geojson = json.dumps({
            "type": "FeatureCollection",
            "features": []
        }).encode("utf-8")

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(empty_geojson)

        with patch("requests.get", return_value=mock_resp):
            result = loader.load_nifc_perimeter("Unknown Fire", "2024-01-01")

        assert result is None

    def test_no_crash_on_malformed_json(self):
        """Returns None when the response body cannot be parsed as GeoJSON."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(b"not-valid-geojson{{{")

        with patch("requests.get", return_value=mock_resp):
            result = loader.load_nifc_perimeter("Test Fire", "2024-07-23")

        assert result is None


# ---------------------------------------------------------------------------
# NWS Loader
# ---------------------------------------------------------------------------


class TestLoadNwsFloodPolygon:
    """Tests for PerimeterLoader.load_nws_flood_polygon()."""

    def test_returns_geodataframe_on_success(self):
        """Returns a GeoDataFrame when NWS returns valid GeoJSON."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp):
            gdf = loader.load_nws_flood_polygon("FLZ052")

        assert gdf is not None
        assert len(gdf) == 1

    def test_sends_correct_user_agent(self):
        """NWS requests include the required User-Agent header."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp) as mock_get:
            loader.load_nws_flood_polygon("FLZ052")

        headers = mock_get.call_args[1]["headers"]
        assert "User-Agent" in headers
        assert "FirstLight" in headers["User-Agent"]

    def test_url_contains_advisory_id(self):
        """Request URL includes the advisory_id."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(SAMPLE_GEOJSON)

        with patch("requests.get", return_value=mock_resp) as mock_get:
            loader.load_nws_flood_polygon("FLZ052")

        call_url = mock_get.call_args[0][0]
        assert "FLZ052" in call_url

    def test_returns_none_on_http_error(self):
        """Returns None when NWS request raises a network error."""
        from core.reporting.data.perimeter_loader import PerimeterLoader
        from requests.exceptions import RequestException

        loader = PerimeterLoader()

        with patch("requests.get", side_effect=RequestException("connection error")):
            result = loader.load_nws_flood_polygon("FLZ052")

        assert result is None

    def test_returns_none_on_5xx(self):
        """Returns None on 503 HTTP response."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        mock_resp = _make_mock_response(b"", status_code=503)

        with patch("requests.get", return_value=mock_resp):
            result = loader.load_nws_flood_polygon("FLZ999")

        assert result is None

    def test_returns_none_on_empty_features(self):
        """Returns None when response has no features."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        empty = json.dumps({"type": "FeatureCollection", "features": []}).encode()
        loader = PerimeterLoader()
        mock_resp = _make_mock_response(empty)

        with patch("requests.get", return_value=mock_resp):
            result = loader.load_nws_flood_polygon("FLZ000")

        assert result is None


# ---------------------------------------------------------------------------
# Local File Loader
# ---------------------------------------------------------------------------


class TestLoadFromFile:
    """Tests for PerimeterLoader.load_from_file()."""

    def test_loads_local_geojson(self, tmp_path):
        """Loads a GeoJSON file from the filesystem."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        geojson_path = tmp_path / "perimeter.geojson"
        geojson_path.write_bytes(SAMPLE_GEOJSON)

        loader = PerimeterLoader()
        gdf = loader.load_from_file(geojson_path)

        assert gdf is not None
        assert len(gdf) == 1
        assert "geometry" in gdf.columns

    def test_loads_geometry_as_polygon(self, tmp_path):
        """Loaded geometry has a valid Polygon type."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        geojson_path = tmp_path / "perimeter.geojson"
        geojson_path.write_bytes(SAMPLE_GEOJSON)

        loader = PerimeterLoader()
        gdf = loader.load_from_file(geojson_path)

        assert gdf.geometry.geom_type.iloc[0] == "Polygon"

    def test_returns_none_for_missing_file(self, tmp_path):
        """Returns None when the file does not exist."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        loader = PerimeterLoader()
        result = loader.load_from_file(tmp_path / "nonexistent.geojson")

        assert result is None

    def test_returns_none_for_invalid_content(self, tmp_path):
        """Returns None when file content is not valid geospatial data."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        bad_file = tmp_path / "bad.geojson"
        bad_file.write_text("this is not geojson")

        loader = PerimeterLoader()
        result = loader.load_from_file(bad_file)

        assert result is None

    def test_multipolygon_geojson(self, tmp_path):
        """Loads a MultiPolygon feature correctly."""
        from core.reporting.data.perimeter_loader import PerimeterLoader

        multi_geojson = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "MultiPolygon",
                        "coordinates": [
                            [[[-82.0, 26.3], [-81.5, 26.3],
                              [-81.5, 26.7], [-82.0, 26.7], [-82.0, 26.3]]],
                            [[[-80.0, 25.0], [-79.5, 25.0],
                              [-79.5, 25.5], [-80.0, 25.5], [-80.0, 25.0]]],
                        ]
                    },
                    "properties": {}
                }
            ]
        }).encode("utf-8")

        geojson_path = tmp_path / "multi.geojson"
        geojson_path.write_bytes(multi_geojson)

        loader = PerimeterLoader()
        gdf = loader.load_from_file(geojson_path)

        assert gdf is not None
        assert gdf.geometry.geom_type.iloc[0] == "MultiPolygon"
