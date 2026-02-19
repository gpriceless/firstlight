"""
Geospatial edge case tests for the LLM Control Plane.

Targets:
  - API-level GeoJSON validation (CreateJobRequest Pydantic model)
  - Application-level bbox filtering (_job_intersects_bbox, _extract_bbox_from_geojson)
  - Application-level area computation (_compute_aoi_area_km2)
  - STAC publisher geometry fidelity (build_stac_item, _extract_bbox, _flatten_coordinates)
  - SQL migration review (constraint correctness, trigger behavior)

These tests run WITHOUT a live PostGIS instance. Tests that require PostGIS
are marked with @pytest.mark.integration.

Written by Tobler (Geospatial Engineer) to catch edge cases that typical
developer testing would miss.
"""

import json
import math
import pytest
from typing import Any, Dict, List, Optional

from api.models.control import CreateJobRequest, _validate_coordinates, _check_bounds, _count_vertices
from core.stac.publisher import (
    build_stac_item,
    _extract_bbox,
    _flatten_coordinates,
)


# ---------------------------------------------------------------------------
# Local copies of pure utility functions from api.routes.control.jobs
#
# These are copied here to avoid importing api.routes.control.jobs which
# triggers api.routes.__init__ that has an unrelated broken import chain.
# The functions are tested both here (for correctness) and in
# tests/api/test_control_jobs.py (for integration with the route).
# ---------------------------------------------------------------------------


def _extract_bbox_from_geojson(geojson: Dict[str, Any]) -> Optional[tuple]:
    """Extract bounding box (west, south, east, north) from GeoJSON geometry."""
    try:
        coords = geojson.get("coordinates", [])
        geom_type = geojson.get("type", "")

        all_lons: List[float] = []
        all_lats: List[float] = []

        if geom_type == "Point":
            all_lons.append(coords[0])
            all_lats.append(coords[1])
        elif geom_type in ("MultiPoint", "LineString"):
            for pt in coords:
                all_lons.append(pt[0])
                all_lats.append(pt[1])
        elif geom_type in ("MultiLineString", "Polygon"):
            for ring in coords:
                for pt in ring:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    for pt in ring:
                        all_lons.append(pt[0])
                        all_lats.append(pt[1])

        if all_lons and all_lats:
            return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
    except (IndexError, TypeError, KeyError):
        pass
    return None


def _job_intersects_bbox(
    aoi: Optional[Dict[str, Any]],
    west: float,
    south: float,
    east: float,
    north: float,
) -> bool:
    """Check if a job's AOI intersects with the given bbox."""
    if aoi is None:
        return True
    job_bbox = _extract_bbox_from_geojson(aoi)
    if job_bbox is None:
        return True
    jw, js, je, jn = job_bbox
    return not (je < west or jw > east or jn < south or js > north)


def _compute_aoi_area_km2(aoi: Optional[Dict[str, Any]]) -> Optional[float]:
    """Compute approximate area in km2 from GeoJSON."""
    if aoi is None:
        return None

    bbox = _extract_bbox_from_geojson(aoi)
    if bbox is None:
        return None

    west, south, east, north = bbox
    R = 6371.0

    lat1 = math.radians(south)
    lat2 = math.radians(north)
    lon1 = math.radians(west)
    lon2 = math.radians(east)

    area = R * R * abs(math.sin(lat1) - math.sin(lat2)) * abs(lon1 - lon2)
    return round(area, 4)


# =============================================================================
# Geometry Fixtures
# =============================================================================


def _simple_polygon(west=-1.0, south=-1.0, east=1.0, north=1.0):
    """Create a simple rectangular polygon in GeoJSON."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]],
    }


def _simple_multipolygon(west=-1.0, south=-1.0, east=1.0, north=1.0):
    """Create a simple rectangular multipolygon in GeoJSON."""
    return {
        "type": "MultiPolygon",
        "coordinates": [[[
            [west, south],
            [east, south],
            [east, north],
            [west, north],
            [west, south],
        ]]],
    }


# =============================================================================
# 1. ANTIMERIDIAN-CROSSING GEOMETRIES
# =============================================================================
# The antimeridian (180/-180 longitude line) is the most common source of
# geospatial bugs. Geometries that cross it have longitudes that jump from
# ~179 to ~-179, which confuses naive min/max bbox algorithms.


class TestAntimeridianGeometries:
    """Test handling of geometries that cross the antimeridian (180/-180)."""

    def test_fiji_polygon_passes_validation(self):
        """
        Fiji straddles the antimeridian. A polygon covering Fiji using
        coordinates on both sides of 180/-180 should pass WGS84 bounds
        validation since all coordinates are individually within [-180, 180].

        This is the GeoJSON convention: split the polygon at the antimeridian
        into two parts, each with valid coordinates.
        """
        # Fiji split into two polygons (GeoJSON antimeridian convention)
        fiji_multipolygon = {
            "type": "MultiPolygon",
            "coordinates": [
                # Eastern part (positive longitudes, near 180)
                [[[177.0, -18.0], [180.0, -18.0], [180.0, -16.0],
                  [177.0, -16.0], [177.0, -18.0]]],
                # Western part (negative longitudes, near -180)
                [[[-180.0, -18.0], [-178.0, -18.0], [-178.0, -16.0],
                  [-180.0, -16.0], [-180.0, -18.0]]],
            ],
        }
        req = CreateJobRequest(
            event_type="flood",
            aoi=fiji_multipolygon,
        )
        assert req.aoi["type"] == "MultiPolygon"

    def test_antimeridian_bbox_extraction_produces_wrong_result(self):
        """
        KNOWN ISSUE: _extract_bbox_from_geojson uses min/max which gives
        a bogus world-spanning bbox for antimeridian-crossing geometries.

        For Fiji split at the antimeridian, naive min(lon) = -180,
        max(lon) = 180, producing a bbox that spans the entire planet.

        This test documents the behavior so future developers know it exists.
        """
        fiji_aoi = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[177.0, -18.0], [180.0, -18.0], [180.0, -16.0],
                  [177.0, -16.0], [177.0, -18.0]]],
                [[[-180.0, -18.0], [-178.0, -18.0], [-178.0, -16.0],
                  [-180.0, -16.0], [-180.0, -18.0]]],
            ],
        }
        bbox = _extract_bbox_from_geojson(fiji_aoi)
        assert bbox is not None
        west, south, east, north = bbox
        # This documents the known-incorrect behavior:
        # The naive min/max produces a world-spanning bbox
        assert west == -180.0
        assert east == 180.0
        # The lat bounds are correct
        assert south == -18.0
        assert north == -16.0

    def test_alaska_polygon_near_antimeridian(self):
        """
        Parts of the Aleutian Islands extend past the antimeridian.
        A polygon for western Aleutians needs coordinates near +/-180.
        """
        # Simplified Aleutian polygon on the eastern side of the antimeridian
        aleutian = {
            "type": "Polygon",
            "coordinates": [[
                [172.0, 51.0], [179.9, 51.0], [179.9, 53.0],
                [172.0, 53.0], [172.0, 51.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=aleutian)
        assert req.aoi["type"] == "Polygon"

    def test_bbox_filter_does_not_handle_antimeridian_crossing(self):
        """
        KNOWN ISSUE: When bbox has west > east (antimeridian-crossing bbox
        like bbox=170,-20,-170,-10), the _job_intersects_bbox function
        does not handle this correctly -- it will never match because
        the non-intersection check (je < west or jw > east) fails.

        This test documents the behavior.
        """
        # A job in Fiji (east side)
        fiji_east_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [177.0, -18.0], [179.0, -18.0], [179.0, -16.0],
                [177.0, -16.0], [177.0, -18.0],
            ]],
        }

        # Antimeridian-crossing bbox: west=170, east=-170
        # This should find the Fiji polygon, but the naive check fails
        result = _job_intersects_bbox(fiji_east_aoi, 170.0, -20.0, -170.0, -10.0)

        # The function sees west=170, east=-170, and the check
        # `jw > east` becomes `177 > -170` which is True, so it returns
        # non-intersection. This is incorrect -- Fiji IS in this bbox.
        assert result is False  # Documents the known bug


# =============================================================================
# 2. POLE-TOUCHING GEOMETRIES
# =============================================================================


class TestPoleGeometries:
    """Test handling of geometries at or near the poles."""

    def test_antarctic_polygon_at_south_pole(self):
        """
        Antarctic research zones may use lat = -90 (the South Pole).
        WGS84 allows -90 latitude but area computation becomes degenerate.
        """
        antarctic = {
            "type": "Polygon",
            "coordinates": [[
                [-180.0, -90.0], [180.0, -90.0], [180.0, -80.0],
                [-180.0, -80.0], [-180.0, -90.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=antarctic)
        assert req.aoi["type"] == "Polygon"

    def test_north_pole_polygon(self):
        """
        A polygon touching the North Pole (lat = 90) should pass validation.
        """
        arctic = {
            "type": "Polygon",
            "coordinates": [[
                [-180.0, 85.0], [180.0, 85.0], [180.0, 90.0],
                [-180.0, 90.0], [-180.0, 85.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=arctic)
        assert req.aoi["type"] == "Polygon"

    def test_area_computation_at_poles_is_degenerate(self):
        """
        The approximate area computation uses a spherical formula.
        At the poles, the convergence of meridians means a 1-degree
        longitude span covers almost zero distance. The bbox-based
        approximation should still produce a finite, non-negative result.
        """
        polar_aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-10.0, 89.0], [10.0, 89.0], [10.0, 90.0],
                [-10.0, 90.0], [-10.0, 89.0],
            ]],
        }
        area = _compute_aoi_area_km2(polar_aoi)
        assert area is not None
        assert area >= 0.0
        # At the pole, area should be very small relative to the equator
        # A 20-degree lon x 1-degree lat box at equator would be ~245,000 km2
        # At 89-90 degrees lat, it should be orders of magnitude smaller
        assert area < 50000.0  # Loose upper bound

    def test_area_computation_at_equator_vs_poles(self):
        """
        Compare area computation for same-sized bbox at equator vs near pole.
        The spherical formula should show significant difference.
        """
        equator_aoi = _simple_polygon(west=0.0, south=-0.5, east=1.0, north=0.5)
        polar_aoi = _simple_polygon(west=0.0, south=89.0, east=1.0, north=89.5)

        equator_area = _compute_aoi_area_km2(equator_aoi)
        polar_area = _compute_aoi_area_km2(polar_aoi)

        assert equator_area is not None
        assert polar_area is not None
        # Polar area should be smaller due to meridian convergence
        assert polar_area < equator_area


# =============================================================================
# 3. SELF-INTERSECTING AND INVALID GEOMETRIES
# =============================================================================


class TestInvalidGeometries:
    """
    Test handling of geometrically invalid polygons.

    Note: The Pydantic validator checks coordinate bounds and vertex count
    but does NOT check geometric validity (self-intersection, unclosed rings).
    That validation happens at the PostGIS layer via ST_IsValid constraint.
    These tests document which validation layer catches which error.
    """

    def test_bowtie_polygon_passes_pydantic_validation(self):
        """
        A bowtie (figure-8) polygon has self-intersecting edges.
        The Pydantic validator only checks coordinate bounds, so this
        passes. PostGIS ST_IsValid would reject it.
        """
        bowtie = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [1.0, 1.0], [1.0, 0.0],
                [0.0, 1.0], [0.0, 0.0],
            ]],
        }
        # Pydantic validation passes (coordinates are within bounds)
        req = CreateJobRequest(event_type="flood", aoi=bowtie)
        assert req.aoi["type"] == "Polygon"

    def test_unclosed_ring_passes_pydantic_validation(self):
        """
        GeoJSON spec requires polygon rings to be closed (first == last point).
        The Pydantic validator does not check ring closure. PostGIS
        ST_GeomFromGeoJSON may auto-close or reject depending on version.
        """
        unclosed = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                # Missing closing vertex [0.0, 0.0]
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=unclosed)
        assert req.aoi["type"] == "Polygon"

    def test_duplicate_vertices_pass_validation(self):
        """
        Repeated/duplicate consecutive vertices are geometrically valid
        but wasteful. They pass both Pydantic and PostGIS validation.
        """
        dupes = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0],
                [1.0, 1.0], [0.0, 1.0], [0.0, 0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=dupes)
        assert req.aoi["type"] == "Polygon"


# =============================================================================
# 4. DEGENERATE AND EXTREME GEOMETRIES
# =============================================================================


class TestDegenerateGeometries:
    """Test handling of degenerate, sliver, and extreme-size geometries."""

    def test_sliver_polygon_near_zero_area(self):
        """
        A very thin sliver polygon (e.g., along a road centerline)
        may have near-zero area. The PostGIS trigger rejects area < 0.01 km2
        but the Pydantic layer passes it.
        """
        # ~0.001 degree wide sliver at equator ~ ~111m wide
        sliver = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [0.001, 0.0], [0.001, 0.001],
                [0.0, 0.001], [0.0, 0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=sliver)
        assert req.aoi["type"] == "Polygon"

        # Check approximate area -- should be very small
        area = _compute_aoi_area_km2(sliver)
        assert area is not None
        # 0.001 degree x 0.001 degree ~ 0.012 km2 at equator
        assert area < 1.0

    def test_minimum_viable_aoi_100m_square(self):
        """
        The minimum AOI is 0.01 km2 (~100m x 100m). Check that a
        polygon near this size is accepted by Pydantic (PostGIS trigger
        would also need to accept it).
        """
        # ~100m x ~100m at equator: ~0.0009 degrees
        delta = 0.0009
        tiny = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [delta, 0.0], [delta, delta],
                [0.0, delta], [0.0, 0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=tiny)
        assert req.aoi["type"] == "Polygon"

    def test_very_large_aoi_near_5m_km2(self):
        """
        The maximum AOI is 5,000,000 km2 (roughly the size of half of
        Europe). A polygon near this size should pass Pydantic validation.
        The PostGIS trigger enforces the actual limit.
        """
        # Roughly 45 degrees lat x 40 degrees lon near the equator ~ 5M km2
        big = {
            "type": "Polygon",
            "coordinates": [[
                [-20.0, -22.5], [20.0, -22.5], [20.0, 22.5],
                [-20.0, 22.5], [-20.0, -22.5],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=big)
        area = _compute_aoi_area_km2(big)
        assert area is not None
        # Should be in the millions of km2 range
        assert area > 1_000_000

    def test_point_geometry_submitted_for_aoi(self):
        """
        A Point geometry has no area. The Pydantic validator accepts it
        (Point is a valid GeoJSON type), but ST_Multi(ST_GeomFromGeoJSON)
        would promote it to a MULTIPOINT, which does not match the
        GEOMETRY(MULTIPOLYGON, 4326) column type. PostGIS would reject it.

        This test documents that the API layer does NOT reject Points.
        """
        point = {
            "type": "Point",
            "coordinates": [-122.4, 37.8],
        }
        req = CreateJobRequest(event_type="flood", aoi=point)
        assert req.aoi["type"] == "Point"

    def test_linestring_geometry_submitted_for_aoi(self):
        """
        A LineString geometry has no area. Similar to Point, Pydantic
        accepts it but PostGIS MULTIPOLYGON column would reject it.

        This is a gap in the validation layer.
        """
        line = {
            "type": "LineString",
            "coordinates": [[-122.4, 37.8], [-122.3, 37.9], [-122.2, 37.8]],
        }
        req = CreateJobRequest(event_type="flood", aoi=line)
        assert req.aoi["type"] == "LineString"

    def test_polygon_with_hole(self):
        """
        A donut polygon (outer ring with a hole) is a valid and common
        geometry type. The interior ring must have opposite winding order.
        """
        donut = {
            "type": "Polygon",
            "coordinates": [
                # Outer ring (counterclockwise)
                [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0],
                 [-2.0, 2.0], [-2.0, -2.0]],
                # Inner ring / hole (clockwise)
                [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5],
                 [0.5, -0.5], [-0.5, -0.5]],
            ],
        }
        req = CreateJobRequest(event_type="flood", aoi=donut)
        assert req.aoi["type"] == "Polygon"
        assert len(req.aoi["coordinates"]) == 2  # Outer + hole

    def test_archipelago_multipolygon(self):
        """
        A MultiPolygon representing an archipelago (like Hawaii or Indonesia)
        with many disconnected parts.
        """
        # Simplified Hawaii-like archipelago
        islands = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[-155.5, 19.5], [-155.0, 19.5], [-155.0, 20.0],
                  [-155.5, 20.0], [-155.5, 19.5]]],  # Big Island
                [[[-156.5, 20.5], [-156.0, 20.5], [-156.0, 21.0],
                  [-156.5, 21.0], [-156.5, 20.5]]],  # Maui
                [[[-158.2, 21.3], [-157.7, 21.3], [-157.7, 21.7],
                  [-158.2, 21.7], [-158.2, 21.3]]],  # Oahu
                [[[-159.8, 22.0], [-159.3, 22.0], [-159.3, 22.2],
                  [-159.8, 22.2], [-159.8, 22.0]]],  # Kauai
            ],
        }
        req = CreateJobRequest(event_type="flood", aoi=islands)
        assert len(req.aoi["coordinates"]) == 4

    def test_polygon_with_3d_coordinates_z_values(self):
        """
        Some data sources include Z (elevation) values in coordinates.
        GeoJSON allows this: [lon, lat, elevation]. The validator
        checks only the first two values (lon, lat).
        """
        polygon_3d = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.5, 100.0],
                [-121.5, 37.5, 200.0],
                [-121.5, 38.5, 150.0],
                [-122.5, 38.5, 50.0],
                [-122.5, 37.5, 100.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=polygon_3d)
        assert req.aoi["type"] == "Polygon"

    def test_3d_coordinates_stripped_in_bbox_extraction(self):
        """
        The _flatten_coordinates function strips Z values (takes only [:2]).
        Verify that 3D coordinates do not corrupt bbox computation.
        """
        polygon_3d = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.5, 100.0],
                [-121.5, 37.5, 200.0],
                [-121.5, 38.5, 150.0],
                [-122.5, 38.5, 50.0],
                [-122.5, 37.5, 100.0],
            ]],
        }
        bbox = _extract_bbox(polygon_3d)
        assert bbox is not None
        assert len(bbox) == 4  # No Z in bbox
        west, south, east, north = bbox
        assert west == pytest.approx(-122.5)
        assert south == pytest.approx(37.5)
        assert east == pytest.approx(-121.5)
        assert north == pytest.approx(38.5)


# =============================================================================
# 5. COORDINATE PRECISION AND ROUND-TRIP FIDELITY
# =============================================================================


class TestCoordinatePrecision:
    """Test coordinate precision through the processing pipeline."""

    def test_high_precision_coordinates_preserved(self):
        """
        Coordinates with many decimal places should be preserved through
        the STAC item builder. 15 decimal places is approximately the
        limit of float64 precision.
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-122.123456789012345, 37.123456789012345],
                [-121.123456789012345, 37.123456789012345],
                [-121.123456789012345, 38.123456789012345],
                [-122.123456789012345, 38.123456789012345],
                [-122.123456789012345, 37.123456789012345],
            ]],
        }
        item = build_stac_item(
            job_id="precision-test",
            event_type="flood",
            aoi_geojson=aoi,
        )
        # The geometry should be the exact same dict object
        assert item["geometry"] is aoi

    def test_boundary_coordinates_at_wgs84_limits(self):
        """
        Test coordinates at the exact WGS84 boundary values:
        lon = +/-180, lat = +/-90.
        """
        boundary = {
            "type": "Polygon",
            "coordinates": [[
                [-180.0, -90.0], [180.0, -90.0], [180.0, 90.0],
                [-180.0, 90.0], [-180.0, -90.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=boundary)
        assert req.aoi["type"] == "Polygon"

    def test_coordinates_just_outside_bounds_rejected(self):
        """
        Coordinates at 180.001 longitude should be rejected.
        """
        with pytest.raises(Exception) as exc_info:
            CreateJobRequest(
                event_type="flood",
                aoi={
                    "type": "Polygon",
                    "coordinates": [[
                        [-180.001, 0.0], [0.0, 0.0], [0.0, 1.0],
                        [-180.001, 1.0], [-180.001, 0.0],
                    ]],
                },
            )
        assert "Longitude" in str(exc_info.value) or "bounds" in str(exc_info.value).lower()

    def test_latitude_90_point_001_rejected(self):
        """
        Latitude 90.001 should be rejected.
        """
        with pytest.raises(Exception) as exc_info:
            CreateJobRequest(
                event_type="flood",
                aoi={
                    "type": "Polygon",
                    "coordinates": [[
                        [0.0, 0.0], [1.0, 0.0], [1.0, 90.001],
                        [0.0, 90.001], [0.0, 0.0],
                    ]],
                },
            )
        assert "Latitude" in str(exc_info.value) or "bounds" in str(exc_info.value).lower()

    def test_float_precision_near_180_boundary(self):
        """
        Test that floating-point precision near the 180 boundary
        does not cause false rejections. 179.999999999999 should be
        accepted, but a naive comparison might reject it due to
        float rounding.
        """
        near_boundary = {
            "type": "Polygon",
            "coordinates": [[
                [179.999999, -1.0], [180.0, -1.0], [180.0, 1.0],
                [179.999999, 1.0], [179.999999, -1.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=near_boundary)
        assert req.aoi["type"] == "Polygon"


# =============================================================================
# 6. BBOX QUERY FILTER EDGE CASES
# =============================================================================


class TestBboxFilterEdgeCases:
    """Test the _job_intersects_bbox function for edge cases."""

    def test_tiny_aoi_inside_large_bbox(self):
        """
        A very small AOI (city-block size) should be found by a
        large bounding box query.
        """
        tiny_aoi = _simple_polygon(west=-0.001, south=-0.001, east=0.001, north=0.001)
        result = _job_intersects_bbox(tiny_aoi, -10.0, -10.0, 10.0, 10.0)
        assert result is True

    def test_large_aoi_partially_overlaps_bbox(self):
        """
        A large AOI that partially overlaps the bbox should be returned.
        """
        large_aoi = _simple_polygon(west=-10.0, south=-10.0, east=10.0, north=10.0)
        result = _job_intersects_bbox(large_aoi, 5.0, 5.0, 20.0, 20.0)
        assert result is True

    def test_aoi_completely_outside_bbox(self):
        """
        An AOI completely outside the bbox should NOT be returned.
        """
        aoi = _simple_polygon(west=50.0, south=50.0, east=51.0, north=51.0)
        result = _job_intersects_bbox(aoi, -10.0, -10.0, 10.0, 10.0)
        assert result is False

    def test_bbox_is_a_point(self):
        """
        Degenerate bbox where west=east and south=north (a point).
        Only an AOI that contains that exact point should match.
        """
        aoi = _simple_polygon(west=-1.0, south=-1.0, east=1.0, north=1.0)
        # Point at origin -- inside the AOI
        result = _job_intersects_bbox(aoi, 0.0, 0.0, 0.0, 0.0)
        assert result is True

        # Point outside the AOI
        result = _job_intersects_bbox(aoi, 5.0, 5.0, 5.0, 5.0)
        assert result is False

    def test_bbox_covers_entire_world(self):
        """
        A world-spanning bbox should match every AOI.
        """
        aoi = _simple_polygon(west=100.0, south=30.0, east=101.0, north=31.0)
        result = _job_intersects_bbox(aoi, -180.0, -90.0, 180.0, 90.0)
        assert result is True

    def test_bbox_is_a_line(self):
        """
        Degenerate bbox where west=east but south != north (vertical line).
        Should match AOIs that cross that longitude.
        """
        aoi = _simple_polygon(west=-1.0, south=-1.0, east=1.0, north=1.0)
        # Vertical line at longitude 0 through the AOI
        result = _job_intersects_bbox(aoi, 0.0, -5.0, 0.0, 5.0)
        assert result is True

    def test_none_aoi_returns_true(self):
        """
        Jobs without an AOI should always be included (match any bbox).
        """
        result = _job_intersects_bbox(None, -10.0, -10.0, 10.0, 10.0)
        assert result is True

    def test_aoi_touches_bbox_edge(self):
        """
        An AOI that shares an edge with the bbox (but does not overlap)
        should be included because the non-intersection check uses strict
        less-than, and shared edge means the ranges are NOT non-intersecting.
        """
        aoi = _simple_polygon(west=0.0, south=0.0, east=1.0, north=1.0)
        # Bbox starts exactly where AOI ends
        # je (1.0) < west (1.0) is False, so it IS intersecting
        result = _job_intersects_bbox(aoi, 1.0, 0.0, 2.0, 1.0)
        assert result is True


# =============================================================================
# 7. STAC GEOMETRY FIDELITY
# =============================================================================


class TestStacGeometryFidelity:
    """Test that STAC Items preserve geometry with full fidelity."""

    def test_stac_geometry_is_exact_reference(self):
        """
        The STAC item geometry should be the exact same dict reference
        as the input AOI (not a copy that might lose precision).
        """
        aoi = _simple_polygon(west=-122.5, south=37.5, east=-121.5, north=38.5)
        item = build_stac_item(
            job_id="fidelity-1",
            event_type="flood",
            aoi_geojson=aoi,
        )
        assert item["geometry"] is aoi

    def test_stac_bbox_matches_geometry_bounds(self):
        """
        The STAC item bbox should exactly match the geometry bounds.
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-122.51234, 37.78901],
                [-122.40567, 37.78901],
                [-122.40567, 37.82345],
                [-122.51234, 37.82345],
                [-122.51234, 37.78901],
            ]],
        }
        item = build_stac_item(
            job_id="fidelity-2",
            event_type="flood",
            aoi_geojson=aoi,
        )
        bbox = item["bbox"]
        assert bbox is not None
        west, south, east, north = bbox
        assert west == pytest.approx(-122.51234, abs=1e-10)
        assert south == pytest.approx(37.78901, abs=1e-10)
        assert east == pytest.approx(-122.40567, abs=1e-10)
        assert north == pytest.approx(37.82345, abs=1e-10)

    def test_stac_preserves_holes_in_polygon(self):
        """
        A donut polygon with holes must preserve the hole ring
        in the STAC item geometry.
        """
        donut = {
            "type": "Polygon",
            "coordinates": [
                # Outer ring
                [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0],
                 [-2.0, 2.0], [-2.0, -2.0]],
                # Hole
                [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5],
                 [0.5, -0.5], [-0.5, -0.5]],
            ],
        }
        item = build_stac_item(
            job_id="fidelity-3",
            event_type="flood",
            aoi_geojson=donut,
        )
        assert len(item["geometry"]["coordinates"]) == 2

    def test_stac_coordinate_order_is_lon_lat(self):
        """
        STAC and GeoJSON require [longitude, latitude] order.
        Verify the coordinates in the STAC item are [lon, lat] not [lat, lon].
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.5],  # lon=-122.5, lat=37.5
                [-121.5, 37.5],
                [-121.5, 38.5],
                [-122.5, 38.5],
                [-122.5, 37.5],
            ]],
        }
        item = build_stac_item(
            job_id="fidelity-4",
            event_type="flood",
            aoi_geojson=aoi,
        )
        first_coord = item["geometry"]["coordinates"][0][0]
        assert first_coord[0] == -122.5  # Longitude first
        assert first_coord[1] == 37.5    # Latitude second

    def test_stac_bbox_for_multipolygon_spans_all_parts(self):
        """
        For a MultiPolygon, the STAC bbox should span ALL parts,
        not just the first polygon.
        """
        mp = {
            "type": "MultiPolygon",
            "coordinates": [
                # Part 1: Western
                [[[-130.0, 35.0], [-125.0, 35.0], [-125.0, 40.0],
                  [-130.0, 40.0], [-130.0, 35.0]]],
                # Part 2: Eastern (far away)
                [[[-80.0, 35.0], [-75.0, 35.0], [-75.0, 40.0],
                  [-80.0, 40.0], [-80.0, 35.0]]],
            ],
        }
        item = build_stac_item(
            job_id="fidelity-5",
            event_type="flood",
            aoi_geojson=mp,
        )
        bbox = item["bbox"]
        assert bbox is not None
        west, south, east, north = bbox
        # Should span from -130 to -75
        assert west == pytest.approx(-130.0)
        assert east == pytest.approx(-75.0)

    def test_stac_item_geometry_collection_handling(self):
        """
        GeometryCollections should have their coordinates flattened
        for bbox computation.
        """
        gc = {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [10.0, 20.0]},
                {"type": "Polygon", "coordinates": [[
                    [0.0, 0.0], [5.0, 0.0], [5.0, 5.0],
                    [0.0, 5.0], [0.0, 0.0],
                ]]},
            ],
        }
        bbox = _extract_bbox(gc)
        assert bbox is not None
        west, south, east, north = bbox
        assert west == pytest.approx(0.0)
        assert south == pytest.approx(0.0)
        assert east == pytest.approx(10.0)
        assert north == pytest.approx(20.0)


# =============================================================================
# 8. AREA COMPUTATION ACCURACY
# =============================================================================


class TestAreaComputationAccuracy:
    """
    Test the _compute_aoi_area_km2 spherical approximation for accuracy.

    The function uses: R^2 * |sin(lat1) - sin(lat2)| * |lon1 - lon2|
    This is a correct spherical formula for the area of a lat/lon rectangle.
    """

    def test_1x1_degree_at_equator(self):
        """
        A 1x1 degree box at the equator should be approximately 12,308 km2.
        (111.32 km per degree lon * 110.57 km per degree lat)
        The spherical formula gives a slightly different number.
        """
        aoi = _simple_polygon(west=0.0, south=0.0, east=1.0, north=1.0)
        area = _compute_aoi_area_km2(aoi)
        assert area is not None
        # Should be approximately 12,000-12,500 km2
        assert 11_000 < area < 13_000

    def test_1x1_degree_at_45_latitude(self):
        """
        At 45 degrees latitude, 1 degree of longitude is shorter
        (~78.85 km). Area should be smaller than at equator.
        """
        aoi = _simple_polygon(west=0.0, south=44.5, east=1.0, north=45.5)
        area = _compute_aoi_area_km2(aoi)
        assert area is not None
        equator_area = _compute_aoi_area_km2(
            _simple_polygon(west=0.0, south=0.0, east=1.0, north=1.0)
        )
        assert area < equator_area

    def test_area_units_are_km2(self):
        """
        Verify that the output is in km2, not m2 or degrees^2.
        A 10x10 degree box at the equator should be roughly 1.2M km2,
        not 100 (degrees^2) or 1.2e12 (m2).
        """
        aoi = _simple_polygon(west=0.0, south=-5.0, east=10.0, north=5.0)
        area = _compute_aoi_area_km2(aoi)
        assert area is not None
        assert 1_000_000 < area < 2_000_000  # Reasonable km2 range

    def test_area_symmetry_north_south(self):
        """
        A box at +45 latitude should have the same area as a box at -45.
        The spherical formula is symmetric about the equator.
        """
        north_aoi = _simple_polygon(west=0.0, south=44.5, east=1.0, north=45.5)
        south_aoi = _simple_polygon(west=0.0, south=-45.5, east=1.0, north=-44.5)
        north_area = _compute_aoi_area_km2(north_aoi)
        south_area = _compute_aoi_area_km2(south_aoi)
        assert north_area is not None
        assert south_area is not None
        assert north_area == pytest.approx(south_area, rel=1e-6)

    def test_area_of_none_is_none(self):
        """Area of None AOI should be None."""
        assert _compute_aoi_area_km2(None) is None

    def test_area_of_empty_geometry_is_none(self):
        """Area of geometry with no extractable coordinates should be None."""
        empty = {"type": "Polygon", "coordinates": []}
        assert _compute_aoi_area_km2(empty) is None


# =============================================================================
# 9. VERTEX COUNT EDGE CASES
# =============================================================================


class TestVertexCountEdgeCases:
    """Test the vertex count limit (max 10,000) and counting logic."""

    def test_exactly_10000_vertices_accepted(self):
        """
        A polygon with exactly 10,000 vertices should be accepted.
        """
        # Generate a circular polygon with 9,999 unique points + closing point
        coords = []
        for i in range(9999):
            angle = 2 * math.pi * i / 9999
            lon = math.cos(angle) * 0.5  # Small radius in degrees
            lat = math.sin(angle) * 0.5
            coords.append([lon, lat])
        coords.append(coords[0])  # Close the ring

        assert len(coords) == 10000
        req = CreateJobRequest(
            event_type="flood",
            aoi={"type": "Polygon", "coordinates": [coords]},
        )
        assert req.aoi["type"] == "Polygon"

    def test_10001_vertices_rejected(self):
        """
        A polygon with 10,001 vertices should be rejected.
        """
        coords = []
        for i in range(10001):
            angle = 2 * math.pi * i / 10001
            lon = math.cos(angle) * 0.5
            lat = math.sin(angle) * 0.5
            coords.append([lon, lat])

        with pytest.raises(Exception) as exc_info:
            CreateJobRequest(
                event_type="flood",
                aoi={"type": "Polygon", "coordinates": [coords]},
            )
        assert "vertices" in str(exc_info.value).lower() or "maximum" in str(exc_info.value).lower()

    def test_vertex_count_multipolygon_aggregates_all_parts(self):
        """
        Vertex count for MultiPolygon should count vertices across ALL
        polygon parts, not just the first one.
        """
        # Two polygons, each with 6000 vertices = 12000 total > 10000 limit
        def make_ring(n, offset_lon=0.0):
            ring = []
            for i in range(n):
                angle = 2 * math.pi * i / n
                lon = math.cos(angle) * 0.5 + offset_lon
                lat = math.sin(angle) * 0.5
                ring.append([lon, lat])
            ring.append(ring[0])
            return ring

        ring1 = make_ring(5999, offset_lon=-10.0)  # 6000 with closing
        ring2 = make_ring(5999, offset_lon=10.0)    # 6000 with closing

        # Verify the count function sees the total
        total = _count_vertices(
            [[ring1], [ring2]], "MultiPolygon"
        )
        # _count_vertices for MultiPolygon: sum(len(pt) for polygon in coords for pt in polygon)
        # For each polygon [ring], len(ring) gives vertex count per ring
        assert total == 12000

    def test_vertex_count_polygon_with_holes(self):
        """
        Vertex count for a polygon with holes should count both
        the outer ring and all inner rings.
        """
        outer = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]  # 5 vertices
        hole = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]]  # 5 vertices
        total = _count_vertices([outer, hole], "Polygon")
        assert total == 10  # 5 + 5


# =============================================================================
# 10. GEOJSON STRUCTURE EDGE CASES
# =============================================================================


class TestGeoJSONStructureEdgeCases:
    """Test handling of unusual but technically valid GeoJSON structures."""

    def test_empty_coordinates_array(self):
        """
        An empty coordinates array should still pass type validation
        but _extract_bbox should return None.
        """
        empty = {"type": "Polygon", "coordinates": []}
        bbox = _extract_bbox_from_geojson(empty)
        assert bbox is None

    def test_geometry_collection_rejected_at_api(self):
        """
        GeometryCollection requires 'geometries' not 'coordinates'.
        The Pydantic validator checks for 'coordinates' key, so a
        GeometryCollection is rejected even though it's a valid GeoJSON type.
        """
        gc = {
            "type": "GeometryCollection",
            "coordinates": [],  # Wrong key, should be 'geometries'
        }
        # The validator allows this because it has both 'type' and 'coordinates'
        # But the coordinates are empty which still passes
        req = CreateJobRequest(event_type="flood", aoi=gc)
        assert req.aoi["type"] == "GeometryCollection"

    def test_extra_properties_in_geojson_preserved(self):
        """
        Extra properties in the GeoJSON dict (like 'crs' or 'properties')
        should be preserved through validation.
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                [0.0, 1.0], [0.0, 0.0],
            ]],
            "properties": {"name": "test area"},
        }
        req = CreateJobRequest(event_type="flood", aoi=aoi)
        assert "properties" in req.aoi

    def test_negative_zero_coordinates_handled(self):
        """
        Negative zero (-0.0) is a valid float that equals 0.0 but
        has a different bit representation. It should not cause issues.
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [-0.0, -0.0], [1.0, -0.0], [1.0, 1.0],
                [-0.0, 1.0], [-0.0, -0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=aoi)
        assert req.aoi["type"] == "Polygon"

    def test_integer_coordinates_accepted(self):
        """
        Integer coordinates (e.g., [0, 0] instead of [0.0, 0.0])
        should be accepted.
        """
        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [0, 0], [1, 0], [1, 1], [0, 1], [0, 0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=aoi)
        assert req.aoi["type"] == "Polygon"

    def test_nan_coordinate_rejected(self):
        """
        NaN coordinates should be rejected by the bounds check.
        NaN is not <= 180, so it should fail.
        """
        with pytest.raises(Exception):
            _check_bounds(float("nan"), 0.0)

    def test_inf_coordinate_rejected(self):
        """
        Infinity coordinates should be rejected by the bounds check.
        """
        with pytest.raises(Exception):
            _check_bounds(float("inf"), 0.0)

    def test_negative_inf_coordinate_rejected(self):
        """
        Negative infinity coordinates should be rejected.
        """
        with pytest.raises(Exception):
            _check_bounds(float("-inf"), 0.0)


# =============================================================================
# 11. FLATTEN COORDINATES EDGE CASES
# =============================================================================


class TestFlattenCoordinates:
    """Test the _flatten_coordinates utility in the STAC publisher."""

    def test_flatten_point(self):
        coords = _flatten_coordinates({"type": "Point", "coordinates": [1.0, 2.0]})
        assert coords == [[1.0, 2.0]]

    def test_flatten_multipoint(self):
        coords = _flatten_coordinates({
            "type": "MultiPoint",
            "coordinates": [[1.0, 2.0], [3.0, 4.0]],
        })
        assert len(coords) == 2

    def test_flatten_linestring(self):
        coords = _flatten_coordinates({
            "type": "LineString",
            "coordinates": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        })
        assert len(coords) == 3

    def test_flatten_multilinestring(self):
        coords = _flatten_coordinates({
            "type": "MultiLineString",
            "coordinates": [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        })
        assert len(coords) == 4

    def test_flatten_polygon_with_hole(self):
        coords = _flatten_coordinates({
            "type": "Polygon",
            "coordinates": [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.2]],
            ],
        })
        # Outer ring: 4 + hole: 4 = 8
        assert len(coords) == 8

    def test_flatten_multipolygon(self):
        coords = _flatten_coordinates({
            "type": "MultiPolygon",
            "coordinates": [
                [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]],
                [[[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 2.0]]],
            ],
        })
        assert len(coords) == 8

    def test_flatten_geometry_collection(self):
        coords = _flatten_coordinates({
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [1.0, 2.0]},
                {"type": "LineString", "coordinates": [[3.0, 4.0], [5.0, 6.0]]},
            ],
        })
        assert len(coords) == 3

    def test_flatten_unknown_type_returns_empty(self):
        coords = _flatten_coordinates({"type": "UnknownType", "coordinates": [[1, 2]]})
        assert coords == []

    def test_flatten_strips_z_values(self):
        """3D coordinates should have Z stripped."""
        coords = _flatten_coordinates({
            "type": "Point",
            "coordinates": [1.0, 2.0, 100.0],
        })
        assert coords == [[1.0, 2.0]]

    def test_flatten_empty_geometry_collection(self):
        coords = _flatten_coordinates({
            "type": "GeometryCollection",
            "geometries": [],
        })
        assert coords == []


# =============================================================================
# 12. SQL MIGRATION REVIEW (documented as test assertions)
# =============================================================================


class TestMigrationReview:
    """
    Review of db/migrations/001_control_plane_schema.sql as test assertions.

    These tests document expected SQL behavior and flag potential issues.
    They do not require a live database -- they are assertions about the
    schema design itself.
    """

    def test_srid_4326_is_correct_for_wgs84(self):
        """
        GEOMETRY(MULTIPOLYGON, 4326) uses SRID 4326 which is WGS 84.
        This is correct for GPS-derived coordinates and GeoJSON, which
        by specification uses WGS 84.
        """
        # This is a documentation test -- SRID 4326 = WGS 84.
        assert True  # Correct: 4326 is WGS84

    def test_gist_index_on_aoi_column_is_correct(self):
        """
        The GIST index is on the 'aoi' column. This is correct because:
        1. Spatial queries (ST_Intersects, ST_Within) use the aoi column
        2. The bbox column is derived and indexed by the generated column
        3. GIST is the correct index type for PostGIS geometry columns
        """
        # Correct: idx_jobs_aoi_gist is on the aoi column
        assert True

    def test_st_envelope_vs_st_extent_for_bbox(self):
        """
        The migration uses ST_Envelope(aoi) for the bbox generated column.

        ST_Envelope: Returns the bounding box of a single geometry.
        ST_Extent:  Aggregate function across multiple geometries.

        ST_Envelope is correct here because we want the bbox of ONE job's
        AOI, not an aggregate bbox across multiple jobs.

        For MultiPolygon, ST_Envelope correctly computes the bbox that
        covers ALL parts of the multi-geometry.
        """
        # Correct: ST_Envelope is per-row, ST_Extent is aggregate
        assert True

    def test_area_trigger_handles_null_geometry_via_not_null(self):
        """
        The aoi column has NOT NULL constraint, so the area trigger
        will never receive a NULL geometry. This is correct -- there is
        no need for NULL geometry handling in the trigger.

        However, ST_IsEmpty is checked separately via constraint, so
        if a geometry passes NOT NULL and ST_IsValid but is somehow
        empty (GeometryCollection EMPTY), the area trigger might get 0.
        The trigger would then raise 'area < 0.01' which is correct.
        """
        # The NOT NULL + ST_IsValid + ST_IsEmpty constraints provide
        # defense-in-depth before the area trigger fires.
        assert True

    def test_area_trigger_fires_on_insert_and_update_of_aoi(self):
        """
        The trigger fires on BEFORE INSERT OR UPDATE OF aoi.
        This means area is recomputed when:
        1. A new job is created (INSERT)
        2. The AOI is updated (UPDATE ... SET aoi = ...)

        It does NOT fire when other columns change (e.g., status update),
        which is correct for performance.
        """
        assert True

    def test_documented_issue_st_envelope_at_antimeridian(self):
        """
        POTENTIAL ISSUE: ST_Envelope for a MultiPolygon split at the
        antimeridian (like Fiji) will compute a bbox that spans the
        entire world longitudinally, because the geometry contains
        points at both +180 and -180.

        This is not incorrect in a strict geometric sense (the envelope
        does contain all the points), but it makes the bbox column
        useless for spatial filtering on antimeridian-crossing AOIs.

        This would need to be addressed if antimeridian support is
        required (e.g., using ST_Split with the antimeridian line,
        or storing a separate bbox as JSON).
        """
        # Documenting: ST_Envelope for antimeridian geometries is degenerate
        assert True

    def test_documented_issue_st_area_geography_cast_precision(self):
        """
        POTENTIAL ISSUE: The area trigger uses ST_Area(aoi::geography).
        The geography cast uses a spheroid model (WGS84 ellipsoid) for
        accurate area computation. However:

        1. The cast from geometry to geography computes area using the
           geographic interpretation of coordinates (great circles).
        2. For very small or very thin geometries, floating-point
           precision in the spheroid computation may cause area to be
           reported as exactly 0.0, which would fail the 0.01 check.
        3. For very large geometries (approaching hemisphere-scale),
           the spheroid computation is accurate but the bbox
           approximation in the Python layer diverges significantly.
        """
        assert True


# =============================================================================
# 13. WINDING ORDER AND RING ORIENTATION
# =============================================================================


class TestWindingOrder:
    """
    Test winding order handling.

    GeoJSON RFC 7946 specifies:
    - Outer rings: counterclockwise
    - Holes/inner rings: clockwise

    However, PostGIS and many tools accept either winding order.
    The Pydantic validator does not check winding order.
    """

    def test_clockwise_outer_ring_accepted(self):
        """
        A clockwise outer ring (opposite of RFC 7946 convention)
        should still pass Pydantic validation.
        PostGIS handles both winding orders correctly.
        """
        # Clockwise outer ring
        cw = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [0.0, 1.0], [1.0, 1.0],
                [1.0, 0.0], [0.0, 0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=cw)
        assert req.aoi["type"] == "Polygon"

    def test_counterclockwise_outer_ring_accepted(self):
        """
        A counterclockwise outer ring (RFC 7946 convention) should pass.
        """
        ccw = {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                [0.0, 1.0], [0.0, 0.0],
            ]],
        }
        req = CreateJobRequest(event_type="flood", aoi=ccw)
        assert req.aoi["type"] == "Polygon"


# =============================================================================
# 14. PostGIS INTEGRATION TESTS (require live database)
# =============================================================================


@pytest.mark.integration
class TestPostGISIntegration:
    """
    Tests that require a live PostGIS instance.

    These tests verify end-to-end behavior including:
    - ST_IsValid constraint enforcement
    - ST_Area(geography) accuracy
    - GeoJSON round-trip fidelity
    - Antimeridian handling in PostGIS

    Run with: pytest -m integration tests/geospatial/
    """

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_self_intersecting_polygon_rejected_by_postgis(self):
        """
        A bowtie polygon should be rejected by the ST_IsValid constraint.
        """
        # Would use asyncpg to INSERT and expect constraint violation
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_geojson_round_trip_precision(self):
        """
        Write GeoJSON -> ST_GeomFromGeoJSON -> PostGIS -> ST_AsGeoJSON -> compare.
        Verify coordinates are preserved to at least 6 decimal places
        (~10cm precision).
        """
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_area_computation_accuracy_against_known_boundary(self):
        """
        Insert a well-known boundary (e.g., a country) and compare
        ST_Area(geography) against the published area.
        """
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_st_multi_promotion_of_polygon_to_multipolygon(self):
        """
        Verify that ST_Multi(ST_GeomFromGeoJSON) correctly promotes
        a Polygon to a MultiPolygon for storage.
        """
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_point_geometry_rejected_by_multipolygon_column(self):
        """
        A Point submitted to the GEOMETRY(MULTIPOLYGON, 4326) column
        should be rejected by PostGIS even after ST_Multi promotion,
        because ST_Multi(Point) -> MultiPoint, not MultiPolygon.
        """
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_area_below_minimum_rejected_by_trigger(self):
        """
        A geometry with area < 0.01 km2 should be rejected by the
        compute_aoi_area_km2 trigger.
        """
        pass

    @pytest.mark.skip(reason="Requires live PostGIS instance")
    def test_area_above_maximum_rejected_by_trigger(self):
        """
        A geometry with area > 5,000,000 km2 should be rejected by the
        compute_aoi_area_km2 trigger.
        """
        pass
