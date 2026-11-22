"""The ingest range check judges a band in reflectance, from the item's own scale.

It used to warn when any one pixel passed DN 10000. Real Sentinel-2 L2A has
cloud tops and glints above reflectance 1, so it warned on most real scenes.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.data.ingestion.validation.band_validator import BandValidator
from core.data.ingestion.validation.config import ValidationConfig

FIXTURES = Path(__file__).parent / "fixtures" / "s2_cloud"
# Earth Search L2A: raster:bands scale 0.0001, offset -0.1 with
# earthsearch:boa_offset_applied true, so the offset is not applied again.
EARTH_SEARCH = {"scale": 0.0001, "offset": 0.0}


def _range_warnings(path: Path, scaling):
    validator = BandValidator(ValidationConfig())
    with rasterio.open(path) as ds:
        result = validator.validate_band_content(ds, 1, reflectance_scaling=scaling)
    assert result.statistics is not None and result.errors == []
    return [w for w in result.warnings if "expected range" in w]


def _write(path: Path, values: np.ndarray) -> Path:
    with rasterio.open(
        path, "w", driver="GTiff", height=values.shape[0], width=values.shape[1],
        count=1, dtype=values.dtype, crs="EPSG:32617",
        transform=from_origin(500000, 2900000, 10, 10), nodata=0,
    ) as ds:
        ds.write(values, 1)
    return path


def test_real_cloudy_l2a_band_with_pixels_over_one_passes():
    path = FIXTURES / "S2C_17RNJ_20260509_0_L2A" / "red.tif"
    with rasterio.open(path) as ds:
        assert ds.read(1).max() > 10000  # cloud tops above reflectance 1

    assert _range_warnings(path, EARTH_SEARCH) == []


def test_band_whose_values_are_out_of_range_still_warns(tmp_path):
    # Made up: a band at DN 25000 (reflectance 2.5) everywhere.
    values = np.full((20, 20), 25000, dtype="uint16")
    values[::2, ::2] = 24000
    warnings = _range_warnings(_write(tmp_path / "bad.tif", values), EARTH_SEARCH)

    assert len(warnings) == 1
    assert "mean reflectance 2.475 above expected range" in warnings[0]


def test_real_band_read_with_the_wrong_scale_warns():
    # The same real band, but its DN taken as reflectance (scale 1).
    path = FIXTURES / "S2C_17RNJ_20260509_0_L2A" / "red.tif"
    warnings = _range_warnings(path, {"scale": 1.0, "offset": 0.0})

    assert len(warnings) == 1
    assert "above expected range 0.0 to 1.0" in warnings[0]
