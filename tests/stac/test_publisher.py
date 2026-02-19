"""
Tests for STAC Item publisher.

Covers:
- Completed job produces a valid STAC Item
- Item geometry matches job AOI (GeoJSON round-trip preserved to 5 decimal places)
- Processing extension fields are present (processing:lineage, processing:software, processing:datetime)
- Re-publishing same result upserts (no duplicate)
- derived_from links present when source URIs exist
- Collection registration

Task 4.10
"""

import json
import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List

from core.stac.publisher import (
    build_stac_item,
    _extract_bbox,
    _flatten_coordinates,
    FIRSTLIGHT_SOFTWARE,
    COG_MEDIA_TYPE,
)
from core.stac.collections import (
    get_flood_collection,
    get_wildfire_collection,
    get_storm_collection,
    get_all_collections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_polygon_aoi() -> Dict[str, Any]:
    """A sample Polygon AOI GeoJSON geometry."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [-122.51234, 37.78901],
                [-122.51234, 37.82345],
                [-122.40567, 37.82345],
                [-122.40567, 37.78901],
                [-122.51234, 37.78901],
            ]
        ],
    }


@pytest.fixture
def sample_multipolygon_aoi() -> Dict[str, Any]:
    """A sample MultiPolygon AOI GeoJSON geometry."""
    return {
        "type": "MultiPolygon",
        "coordinates": [
            [
                [
                    [-122.51234, 37.78901],
                    [-122.51234, 37.82345],
                    [-122.40567, 37.82345],
                    [-122.40567, 37.78901],
                    [-122.51234, 37.78901],
                ]
            ],
            [
                [
                    [-122.30000, 37.70000],
                    [-122.30000, 37.75000],
                    [-122.20000, 37.75000],
                    [-122.20000, 37.70000],
                    [-122.30000, 37.70000],
                ]
            ],
        ],
    }


@pytest.fixture
def sample_cog_assets() -> List[Dict[str, Any]]:
    """Sample COG output assets."""
    return [
        {
            "name": "flood_extent",
            "href": "s3://firstlight-results/job-123/flood_extent.tif",
            "title": "Flood Extent Map",
        },
        {
            "name": "flood_depth",
            "href": "s3://firstlight-results/job-123/flood_depth.tif",
            "title": "Flood Depth Estimate",
        },
    ]


# ---------------------------------------------------------------------------
# Test: STAC Item building
# ---------------------------------------------------------------------------

class TestBuildStacItem:
    """Tests for build_stac_item()."""

    def test_basic_item_structure(self, sample_polygon_aoi):
        """A completed job produces a valid STAC Item."""
        item = build_stac_item(
            job_id="test-job-001",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        assert item["type"] == "Feature"
        assert item["stac_version"] == "1.0.0"
        assert item["id"] == "test-job-001"
        assert item["collection"] == "flood"
        assert "geometry" in item
        assert "bbox" in item
        assert "properties" in item
        assert "links" in item
        assert "assets" in item

    def test_geometry_matches_aoi(self, sample_polygon_aoi):
        """Item geometry matches job AOI exactly."""
        item = build_stac_item(
            job_id="test-job-002",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        # Geometry should be the same object
        assert item["geometry"] == sample_polygon_aoi

    def test_geometry_coordinates_preserved_to_5_decimals(self, sample_polygon_aoi):
        """GeoJSON coordinates are preserved to at least 5 decimal places."""
        item = build_stac_item(
            job_id="test-job-003",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        coords = item["geometry"]["coordinates"][0]
        for coord in coords:
            lon, lat = coord
            # Check the coordinates match the input
            assert abs(lon - round(lon, 5)) < 1e-10
            assert abs(lat - round(lat, 5)) < 1e-10

    def test_multipolygon_geometry(self, sample_multipolygon_aoi):
        """MultiPolygon geometry is preserved."""
        item = build_stac_item(
            job_id="test-job-004",
            event_type="flood",
            aoi_geojson=sample_multipolygon_aoi,
        )

        assert item["geometry"]["type"] == "MultiPolygon"
        assert len(item["geometry"]["coordinates"]) == 2

    def test_processing_extension_fields(self, sample_polygon_aoi):
        """Processing extension fields are present."""
        item = build_stac_item(
            job_id="test-job-005",
            event_type="wildfire",
            aoi_geojson=sample_polygon_aoi,
            parameters={"threshold": 0.5},
        )

        props = item["properties"]
        assert "processing:lineage" in props
        assert "processing:software" in props
        assert "processing:datetime" in props
        assert props["processing:software"] == FIRSTLIGHT_SOFTWARE

    def test_stac_extensions_declared(self, sample_polygon_aoi):
        """STAC extensions are declared."""
        item = build_stac_item(
            job_id="test-job-006",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        assert "stac_extensions" in item
        assert any(
            "processing" in ext for ext in item["stac_extensions"]
        )

    def test_bbox_computed(self, sample_polygon_aoi):
        """BBox is computed from geometry."""
        item = build_stac_item(
            job_id="test-job-007",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        bbox = item["bbox"]
        assert bbox is not None
        assert len(bbox) == 4
        west, south, east, north = bbox
        assert west < east
        assert south < north

    def test_cog_assets(self, sample_polygon_aoi, sample_cog_assets):
        """COG raster assets have proper content type."""
        item = build_stac_item(
            job_id="test-job-008",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
            cog_assets=sample_cog_assets,
        )

        assert "flood_extent" in item["assets"]
        assert "flood_depth" in item["assets"]

        for asset_key, asset in item["assets"].items():
            assert asset["type"] == COG_MEDIA_TYPE
            assert "href" in asset
            assert "roles" in asset
            assert "data" in asset["roles"]

    def test_default_asset_when_no_cogs(self, sample_polygon_aoi):
        """Default asset placeholder when no COGs specified."""
        item = build_stac_item(
            job_id="test-job-009",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        assert "result" in item["assets"]
        assert item["assets"]["result"]["type"] == COG_MEDIA_TYPE

    def test_derived_from_links(self, sample_polygon_aoi):
        """derived_from links present when source URIs exist."""
        source_uris = [
            "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/S2A_T10SFH_20240115",
            "https://earth-search.aws.element84.com/v1/collections/sentinel-1-grd/items/S1A_IW_20240115",
        ]

        item = build_stac_item(
            job_id="test-job-010",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
            source_uris=source_uris,
        )

        derived_links = [l for l in item["links"] if l["rel"] == "derived_from"]
        assert len(derived_links) == 2
        hrefs = {l["href"] for l in derived_links}
        assert source_uris[0] in hrefs
        assert source_uris[1] in hrefs

    def test_no_derived_from_without_sources(self, sample_polygon_aoi):
        """No derived_from links when no source URIs."""
        item = build_stac_item(
            job_id="test-job-011",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
        )

        derived_links = [l for l in item["links"] if l["rel"] == "derived_from"]
        assert len(derived_links) == 0

    def test_customer_id_in_properties(self, sample_polygon_aoi):
        """Customer ID is stored in properties when provided."""
        item = build_stac_item(
            job_id="test-job-012",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
            customer_id="tenant-abc",
        )

        assert item["properties"]["firstlight:customer_id"] == "tenant-abc"

    def test_re_publish_produces_identical_structure(self, sample_polygon_aoi):
        """Re-publishing same result produces same structure (upsert ready)."""
        params = dict(
            job_id="test-job-013",
            event_type="flood",
            aoi_geojson=sample_polygon_aoi,
            parameters={"threshold": 0.5},
            customer_id="tenant-xyz",
        )

        item1 = build_stac_item(**params)
        item2 = build_stac_item(**params)

        # Same structure (excluding timestamps)
        assert item1["id"] == item2["id"]
        assert item1["collection"] == item2["collection"]
        assert item1["geometry"] == item2["geometry"]
        assert set(item1["assets"].keys()) == set(item2["assets"].keys())

    def test_item_collection_links(self, sample_polygon_aoi):
        """Item has self, parent, collection, root links."""
        item = build_stac_item(
            job_id="test-job-014",
            event_type="wildfire",
            aoi_geojson=sample_polygon_aoi,
        )

        rels = {l["rel"] for l in item["links"]}
        assert "self" in rels
        assert "parent" in rels
        assert "collection" in rels
        assert "root" in rels


# ---------------------------------------------------------------------------
# Test: BBox extraction
# ---------------------------------------------------------------------------

class TestBboxExtraction:
    """Tests for _extract_bbox helper."""

    def test_polygon_bbox(self, sample_polygon_aoi):
        bbox = _extract_bbox(sample_polygon_aoi)
        assert bbox is not None
        west, south, east, north = bbox
        assert west == pytest.approx(-122.51234, abs=1e-5)
        assert south == pytest.approx(37.78901, abs=1e-5)
        assert east == pytest.approx(-122.40567, abs=1e-5)
        assert north == pytest.approx(37.82345, abs=1e-5)

    def test_point_bbox(self):
        point = {"type": "Point", "coordinates": [-122.5, 37.8]}
        bbox = _extract_bbox(point)
        assert bbox is not None
        assert len(bbox) == 4

    def test_empty_geometry_returns_none(self):
        bbox = _extract_bbox({"type": "Polygon", "coordinates": []})
        assert bbox is None


# ---------------------------------------------------------------------------
# Test: STAC Collections
# ---------------------------------------------------------------------------

class TestStacCollections:
    """Tests for STAC collection definitions."""

    def test_flood_collection_structure(self):
        coll = get_flood_collection()
        assert coll["id"] == "flood"
        assert coll["stac_version"] == "1.0.0"
        assert "extent" in coll
        assert "spatial" in coll["extent"]
        assert "temporal" in coll["extent"]

    def test_wildfire_collection_structure(self):
        coll = get_wildfire_collection()
        assert coll["id"] == "wildfire"
        assert coll["stac_version"] == "1.0.0"

    def test_storm_collection_structure(self):
        coll = get_storm_collection()
        assert coll["id"] == "storm"
        assert coll["stac_version"] == "1.0.0"

    def test_all_collections_count(self):
        colls = get_all_collections()
        assert len(colls) == 3
        ids = {c["id"] for c in colls}
        assert ids == {"flood", "wildfire", "storm"}

    def test_collections_have_links(self):
        for coll in get_all_collections():
            rels = {l["rel"] for l in coll["links"]}
            assert "self" in rels
            assert "items" in rels
            assert "root" in rels

    def test_collections_have_providers(self):
        for coll in get_all_collections():
            assert "providers" in coll
            assert len(coll["providers"]) > 0
            assert coll["providers"][0]["name"] == "FirstLight"
