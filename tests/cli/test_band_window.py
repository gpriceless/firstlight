"""Ingest reads only the part of a band that covers the box.

The source is a real 1 km block of Sentinel-2 B08 (tile 17RNJ, 31 May 2026).
GDAL reads a local file and a remote COG through the same window, so the
pixels and grid checked here are what a range read over HTTP returns.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.warp import transform_bounds

from cli.commands.ingest import (
    NotATiffError,
    _WINDOW_PAD_PX,
    _read_band_window,
    download_bands,
)

NIR = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "s2_cloud" / "S2A_17RNJ_20260531_0_L2A" / "nir.tif"
)


def _sub_box(frac_x0, frac_y0, frac_x1, frac_y1):
    """A lon/lat box over part of the fixture, by fractions of its extent."""
    with rasterio.open(NIR) as src:
        b = src.bounds
        w = b.right - b.left
        h = b.top - b.bottom
        utm = (b.left + frac_x0 * w, b.bottom + frac_y0 * h,
               b.left + frac_x1 * w, b.bottom + frac_y1 * h)
        return transform_bounds(src.crs, "EPSG:4326", *utm), src.read(1), src.transform


def test_window_holds_the_box_and_the_same_pixels(tmp_path):
    box, full, full_transform = _sub_box(0.25, 0.25, 0.5, 0.5)
    out = tmp_path / "nir.tif"

    _read_band_window(str(NIR), out, box)

    with rasterio.open(out) as dst:
        part = dst.read(1)
        # A quarter of the block on a side, widened by the lon/lat box around
        # a UTM square and the padding, but far short of the whole block.
        assert 25 + 2 * _WINDOW_PAD_PX <= part.shape[0] < 40
        col = round((dst.transform.c - full_transform.c) / 10)
        row = round((dst.transform.f - full_transform.f) / -10)
        np.testing.assert_array_equal(part, full[row:row + part.shape[0], col:col + part.shape[1]])
        assert dst.crs.to_epsg() == 32617
    assert out.stat().st_size < NIR.stat().st_size / 4


def test_window_is_clipped_to_the_raster(tmp_path):
    box, full, _ = _sub_box(0.5, 0.5, 1.5, 1.5)
    out = tmp_path / "nir.tif"

    _read_band_window(str(NIR), out, box)

    with rasterio.open(out) as dst:
        assert dst.height <= 50 + _WINDOW_PAD_PX + 1
        assert dst.width <= 50 + _WINDOW_PAD_PX + 1


def test_box_that_misses_the_raster_is_refused(tmp_path):
    box, _, _ = _sub_box(3, 3, 4, 4)
    with pytest.raises(ValueError, match="does not overlap"):
        _read_band_window(str(NIR), tmp_path / "nir.tif", box)
    assert not (tmp_path / "nir.tif").exists()


def test_a_source_that_is_not_a_raster_falls_back_and_is_still_refused(tmp_path):
    """A login page cannot be range-read; the whole-file path names it and saves nothing."""
    box, _, _ = _sub_box(0.25, 0.25, 0.5, 0.5)
    page = tmp_path / "login.html"
    page.write_bytes(b"<!DOCTYPE html><html><head><title>Login</title></head></html>")

    with patch(
        "cli.commands.ingest._download_band_file",
        side_effect=NotATiffError("not a TIFF"),
    ) as whole:
        results = download_bands({"blue": str(page)}, tmp_path / "out", "x", bbox=box)

    assert whole.call_count == 1
    assert not results["blue"].success
    assert not (tmp_path / "out" / "blue.tif").exists()
