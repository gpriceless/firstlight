"""A box across a tile edge is read from both tiles of the day.

The real 1 km Max Road block (tile 17RNJ) is cut into two overlapping halves,
named as two tiles of the same day, the way two neighbouring Sentinel-2 tiles
cover one box. The mosaic over the box must equal the block read whole.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from cli.commands.run import (
    _find_sentinel2_pre_post_scenes,
    _load_sentinel2_pre_post,
    _load_sentinel2_storm_pre_post,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "s2_cloud"
PRE = "S2C_17RNJ_20260509_0_L2A"
POST = "S2A_17RNJ_20260531_0_L2A"
EVENT = datetime(2026, 5, 10)


def _box_inside_block() -> str:
    """A lon/lat box well inside the block, so the whole read has no gaps."""
    with rasterio.open(FIXTURE / POST / "nir.tif") as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    dx, dy = (e - w) * 0.15, (n - s) * 0.15
    return f"{w + dx},{s + dy},{e - dx},{n - dy}"


def _cut(src_path: Path, dst_path: Path, first: float, last: float) -> None:
    """Write columns [first, last) of the raster, as fractions of its width."""
    with rasterio.open(src_path) as src:
        window = Window(int(src.width * first), 0, int(src.width * (last - first)), src.height)
        profile = src.profile | {
            "width": int(window.width),
            "transform": src.window_transform(window),
        }
        data = src.read(window=window)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data)


def _whole_tree(tmp_path: Path) -> Path:
    root = tmp_path / "whole"
    for sid in (PRE, POST):
        shutil.copytree(FIXTURE / sid, root / "sentinel2" / sid)
    return root


def _split_tree(tmp_path: Path) -> Path:
    """Each scene as tile 17RNJ (west 60%) and tile 17RNK (east 60%)."""
    root = tmp_path / "split"
    for sid in (PRE, POST):
        halves = {
            sid: (0.0, 0.6),
            sid.replace("17RNJ", "17RNK"): (0.4, 1.0),
        }
        for tile, (first, last) in halves.items():
            out = root / "sentinel2" / tile
            out.mkdir(parents=True)
            for band in FIXTURE.joinpath(sid).glob("*.tif"):
                _cut(band, out / band.name, first, last)
    return root


def test_same_day_tiles_are_one_scene(tmp_path):
    scenes = _find_sentinel2_pre_post_scenes(
        _split_tree(tmp_path), event_date=EVENT, bbox=_box_inside_block(),
        bands=("red", "nir"), hazard="storm",
    )

    post = scenes["post"]
    assert len(post["tiles"]) == 2
    assert len(post["red"]) == len(post["scl"]) == 2
    whole = _find_sentinel2_pre_post_scenes(
        _whole_tree(tmp_path), event_date=EVENT, bbox=_box_inside_block(),
        bands=("red", "nir"), hazard="storm",
    )
    assert post["clear_share"] == pytest.approx(whole["post"]["clear_share"])


def test_storm_mosaic_equals_the_block_read_whole(tmp_path):
    bbox = _box_inside_block()
    whole, _, whole_grid, whole_cloud = _load_sentinel2_storm_pre_post(
        _whole_tree(tmp_path), event_date=EVENT, bbox=bbox
    )
    split, _, split_grid, split_cloud = _load_sentinel2_storm_pre_post(
        _split_tree(tmp_path), event_date=EVENT, bbox=bbox
    )

    assert split_grid["transform"] == whole_grid["transform"]
    for key in ("red_pre", "red_post", "nir_pre", "nir_post"):
        assert not np.isnan(split[key]).any(), key
        np.testing.assert_array_equal(split[key], whole[key], err_msg=key)
    np.testing.assert_array_equal(split_cloud, whole_cloud)


def test_wildfire_mosaic_fills_the_box_from_both_tiles(tmp_path):
    bbox = _box_inside_block()
    whole = _load_sentinel2_pre_post(_whole_tree(tmp_path), event_date=EVENT, bbox=bbox)
    split = _load_sentinel2_pre_post(_split_tree(tmp_path), event_date=EVENT, bbox=bbox)

    for name, w, s in zip(("pre_nir", "pre_swir", "post_nir", "post_swir"), whole[:4], split[:4]):
        assert not np.isnan(s).any(), name
        # 20 m SWIR is resampled bilinearly. A tile's edge pixel has no
        # neighbour to blend with and was 0.026 off before the interior of the
        # other tile took precedence there.
        np.testing.assert_allclose(s, w, atol=1e-6, err_msg=name)
    np.testing.assert_array_equal(split[6], whole[6])


def test_part_of_the_box_no_tile_covers_is_masked(tmp_path):
    root = _split_tree(tmp_path)
    for sid in (PRE, POST):
        shutil.rmtree(root / "sentinel2" / sid.replace("17RNJ", "17RNK"))

    bands, _, _, cloud = _load_sentinel2_storm_pre_post(
        root, event_date=EVENT, bbox=_box_inside_block()
    )

    east = np.isnan(bands["nir_post"])
    assert east.any()
    assert cloud[east].all()


def test_part_of_the_box_with_no_pre_scene_is_said_in_words(tmp_path):
    from cli.commands.run import _no_data_lines, _no_data_statistics

    root = _split_tree(tmp_path)
    shutil.rmtree(root / "sentinel2" / PRE.replace("17RNJ", "17RNK"))

    _, _, grid, _ = _load_sentinel2_storm_pre_post(
        root, event_date=EVENT, bbox=_box_inside_block()
    )

    stats = _no_data_statistics(grid)
    assert stats["post_no_data_percent"] == 0
    assert 20 < stats["pre_no_data_percent"] < 50
    [line] = _no_data_lines(stats, "storm")
    assert line.startswith(f"No pre-storm data over {stats['pre_no_data_percent']:.1f}% of the box")
    assert "2026-05-09" in line
