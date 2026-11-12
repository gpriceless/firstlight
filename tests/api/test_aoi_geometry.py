"""An AOI is an area: the request model refuses points and lines."""

import pytest
from pydantic import ValidationError

from api.models.requests import SpatialSpec

SQUARE = [[[-80.5, 25.5], [-80.0, 25.5], [-80.0, 26.0], [-80.5, 26.0], [-80.5, 25.5]]]


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Point", "coordinates": [-80.2, 25.8]},
        {"type": "LineString", "coordinates": [[-80.5, 25.5], [-80.0, 26.0]]},
        {"type": "MultiPoint", "coordinates": [[-80.5, 25.5], [-80.0, 26.0]]},
        {"type": "MultiLineString", "coordinates": [[[-80.5, 25.5], [-80.0, 26.0]]]},
    ],
)
def test_spatial_spec_rejects_non_area_geometry(geometry):
    with pytest.raises(ValidationError, match="Polygon or MultiPolygon"):
        SpatialSpec(geometry=geometry)


@pytest.mark.parametrize(
    "geometry",
    [
        {"type": "Polygon", "coordinates": SQUARE},
        {"type": "MultiPolygon", "coordinates": [SQUARE]},
    ],
)
def test_spatial_spec_accepts_area_geometry(geometry):
    spec = SpatialSpec(geometry=geometry)
    assert spec.geometry.type.value == geometry["type"]


def test_spatial_spec_accepts_bbox_without_geometry():
    spec = SpatialSpec(bbox={"west": -80.5, "south": 25.5, "east": -80.0, "north": 26.0})
    assert spec.geometry is None
