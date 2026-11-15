"""The wildfire path masks cloud and shadow from the Sentinel-2 SCL layer.

The fixture is a real 1 km block cut from two Earth Search Sentinel-2 L2A
scenes over the Max Road fire box (tile 17RNJ): pre S2C 2026-05-09 and post
S2A 2026-05-31, bands B08 (10 m), B12 (20 m) and SCL (20 m), unmodified.
In this block SCL marks 48.3% of the pre scene and 31.8% of the post scene as
cloud or shadow. A third scene, S2A 2026-05-11, is the same block and is 100%
cloud: the one a nearest-date pick chose as post-fire for a 10 May fire. Cloud pixels are bright in NIR (class 9 median 0.78) and
shadow pixels dark (class 3 median 0.10), as they should be.

Before the mask, dNBR had no no-data pixels here and every cloud counted as
ground.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from cli.commands.analyze import load_algorithm
from cli.commands.run import (
    _analyze_wildfire,
    _find_sentinel2_pre_post_scenes,
    _load_sentinel2_pre_post,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "s2_cloud"
PRE = "S2C_17RNJ_20260509_0_L2A"
CLOUDED = "S2A_17RNJ_20260511_0_L2A"
POST = "S2A_17RNJ_20260531_0_L2A"
EVENT = datetime(2026, 5, 20)
# SCL classes that mean the ground was not seen: no data, saturated, shadow,
# cloud medium, cloud high, thin cirrus.
NOT_SEEN = [0, 1, 3, 8, 9, 10]


def _tree(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "sentinel2")
    return tmp_path


def _expected_mask_10m() -> np.ndarray:
    """Union of not-seen SCL pixels, read straight from the files, on the 10 m grid."""
    masks = []
    for sid in (PRE, POST):
        with rasterio.open(FIXTURE / sid / "scl.tif") as src:
            scl = src.read(1)
        masks.append(np.isin(scl, NOT_SEEN))
    union = masks[0] | masks[1]
    return np.repeat(np.repeat(union, 2, axis=0), 2, axis=1)


def test_loader_mask_is_the_scl_cloud_and_shadow(tmp_path):
    *_, grid, cloud_mask = _load_sentinel2_pre_post(_tree(tmp_path), event_date=EVENT)

    expected = _expected_mask_10m()
    assert cloud_mask.shape == (100, 100)
    np.testing.assert_array_equal(cloud_mask, expected)
    # Known cloud in this block: 1,207 + 795 SCL pixels, 1,438 in the union.
    assert cloud_mask.mean() == pytest.approx(1438 / 2500)


def test_dnbr_reports_cloud_as_no_data(tmp_path):
    data_dir = _tree(tmp_path / "data")
    results = tmp_path / "results"
    results.mkdir()

    out = _analyze_wildfire(
        load_algorithm("dnbr", None), data_dir, results,
        raster_files=[], npy_files=[], event_date=EVENT,
    )

    dnbr = np.load(results / "dnbr.npy")
    severity = np.load(results / "burn_severity.npy")
    extent = np.load(results / "burn_extent.npy")
    cloud = _expected_mask_10m()

    np.testing.assert_array_equal(np.isnan(dnbr), cloud)
    assert np.all(severity[cloud] == 255)
    assert not extent[cloud].any()
    assert out["statistics"]["cloud_masked_percent"] == pytest.approx(57.52)
    assert out["statistics"]["valid_pixels"] == int((~cloud).sum())


def test_without_the_mask_cloud_counts_as_ground(tmp_path):
    """Positive control: the same data unmasked reports no no-data at all."""
    pre_nir, pre_swir, post_nir, post_swir, px, _grid, _mask = _load_sentinel2_pre_post(
        _tree(tmp_path), event_date=EVENT
    )
    result = load_algorithm("dnbr", None).execute(
        nir_pre=pre_nir, swir_pre=pre_swir, nir_post=post_nir, swir_post=post_swir,
        pixel_size_m=px,
    )
    assert np.isnan(result.dnbr_map).sum() == 0


def test_scene_without_scl_is_refused(tmp_path):
    root = _tree(tmp_path)
    for sid in (PRE, CLOUDED, POST):
        (root / "sentinel2" / sid / "scl.tif").unlink()

    with pytest.raises(FileNotFoundError, match="SCL"):
        _find_sentinel2_pre_post_scenes(root, event_date=EVENT)


def test_exported_geotiffs_carry_no_data(tmp_path):
    from cli.commands.run import _export_geotiff

    data_dir = _tree(tmp_path / "data")
    results = tmp_path / "results"
    products = tmp_path / "products"
    results.mkdir()
    products.mkdir()
    _analyze_wildfire(
        load_algorithm("dnbr", None), data_dir, results,
        raster_files=[], npy_files=[], event_date=EVENT,
    )
    arrays = {n: np.load(results / f"{n}.npy") for n in ("dnbr", "burn_severity")}
    grid = json.loads((results / "grid.json").read_text())

    _export_geotiff(arrays, products, [-80.55, 25.97, -80.35, 26.10], "real", grid=grid)

    with rasterio.open(products / "dnbr.tif") as src:
        assert np.isnan(src.nodata)
        assert src.read(1, masked=True).mask.mean() == pytest.approx(0.5752)
    with rasterio.open(products / "burn_severity.tif") as src:
        assert src.nodata == 255


def test_a_fully_clouded_scene_loses_to_a_clearer_one(tmp_path):
    """Fire on 10 May: 11 May is nearest but all cloud, so 31 May is the post scene."""
    scenes = _find_sentinel2_pre_post_scenes(_tree(tmp_path), event_date=datetime(2026, 5, 10))
    assert scenes["pre"]["item_id"] == PRE
    assert scenes["post"]["item_id"] == POST
    assert scenes["post"]["clear_share"] == pytest.approx(1 - 0.318)
    assert scenes["pre"]["clear_share"] == pytest.approx(1 - 0.4828)


def test_a_clouded_pre_scene_loses_too(tmp_path):
    """Event on 20 May: 11 May is the latest pre scene but 9 May is clearer."""
    scenes = _find_sentinel2_pre_post_scenes(_tree(tmp_path), event_date=EVENT)
    assert scenes["pre"]["item_id"] == PRE


def _block_lonlat():
    from rasterio.warp import transform_bounds

    with rasterio.open(FIXTURE / POST / "scl.tif") as src:
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def test_clear_share_counts_box_outside_the_scene_as_not_seen():
    from cli.commands.run import _scl_clear_share

    west, south, east, north = _block_lonlat()
    inside = f"{west},{south},{east},{north}"
    # Twice as wide: the east half lies beyond the cut and has no data.
    wider = f"{west},{south},{east + (east - west)},{north}"

    whole = _scl_clear_share(FIXTURE / POST / "scl.tif")
    boxed = _scl_clear_share(FIXTURE / POST / "scl.tif", inside)
    assert whole == pytest.approx(1 - 0.318)
    # The lon/lat box around a UTM block is slightly larger than the block,
    # and its corners beyond the cut count as not seen.
    assert whole - 0.05 < boxed <= whole
    assert _scl_clear_share(FIXTURE / POST / "scl.tif", wider) == pytest.approx(boxed / 2, abs=0.02)
    assert _scl_clear_share(FIXTURE / CLOUDED / "scl.tif", inside) == 0.0
