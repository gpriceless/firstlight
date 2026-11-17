"""Normalise rewrites each band in place instead of writing a copy beside it.

Uses one real scene from the Max Road cloud fixture (tile 17RNJ, 9 May 2026).
Before, every band gained a ``.cog`` twin that nothing read, doubling disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")

from cli.commands.ingest import normalize_item

SCENE = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "s2_cloud" / "S2C_17RNJ_20260509_0_L2A"
)


def test_normalise_leaves_one_file_per_band_with_the_same_pixels(tmp_path):
    item_dir = tmp_path / SCENE.name
    shutil.copytree(SCENE, item_dir)
    before = {}
    for f in item_dir.glob("*.tif"):
        with rasterio.open(f) as src:
            before[f.name] = (src.read(1), src.crs, src.transform)

    assert normalize_item(item_dir, "cog", None, None) is True

    assert sorted(p.name for p in item_dir.iterdir()) == sorted(before)
    for name, (data, crs, transform) in before.items():
        with rasterio.open(item_dir / name) as src:
            np.testing.assert_array_equal(src.read(1), data)
            assert src.crs == crs
            assert src.transform == transform
            assert src.profile.get("tiled") is True
