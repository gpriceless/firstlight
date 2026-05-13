"""LGT-417 regression tests: analyze -> export preserves CRS + transform.

Before LGT-417, the run pipeline's analyze stage wrote .npy arrays without
persisting the source raster's CRS / transform, so the export stage fell
back to a degenerate (-180,-90,180,90) world transform. This test round-trips
a tiny synthetic 16x16 Sentinel-2 NIR/SWIR pair through the real-mode
wildfire analyze stage and verifies the exported GeoTIFFs land at the
source scene's real-world bounds.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import Affine

from cli.commands.analyze import load_algorithm
from cli.commands.run import (
    _analyze_wildfire,
    _load_sentinel2_pre_post,
    run_export,
)


# UTM 17N, somewhere over south Florida — pixel size 10 m to match Sentinel-2 NIR.
SRC_CRS = CRS.from_epsg(32617)
ORIGIN_X = 500000.0
ORIGIN_Y = 2900000.0
PIXEL = 10.0
SRC_TRANSFORM = Affine(PIXEL, 0.0, ORIGIN_X, 0.0, -PIXEL, ORIGIN_Y)
SIZE = 16


def _write_s2_band(
    path: Path,
    arr: np.ndarray,
    transform: Affine = SRC_TRANSFORM,
    crs: CRS = SRC_CRS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(arr.astype(np.float32), 1)


def _make_sentinel2_pre_post(root: Path) -> tuple[Path, Path]:
    """Build a minimal pre/post Sentinel-2 tree under root/sentinel2/."""
    pre_dir = root / "sentinel2" / "S2A_17RNJ_20260101_0_L2A"
    post_dir = root / "sentinel2" / "S2A_17RNJ_20260201_0_L2A"

    # Healthy vegetation pre-fire: high NIR, low SWIR.
    pre_nir = np.full((SIZE, SIZE), 0.45, dtype=np.float32)
    pre_swir = np.full((SIZE, SIZE), 0.12, dtype=np.float32)

    # Post-fire: stripe of severe burn in the middle (low NIR, high SWIR).
    post_nir = pre_nir.copy()
    post_swir = pre_swir.copy()
    burn_rows = slice(4, 12)
    burn_cols = slice(2, 14)
    post_nir[burn_rows, burn_cols] = 0.08
    post_swir[burn_rows, burn_cols] = 0.45

    _write_s2_band(pre_dir / "nir.tif", pre_nir)
    _write_s2_band(pre_dir / "swir22.tif", pre_swir)
    _write_s2_band(post_dir / "nir.tif", post_nir)
    _write_s2_band(post_dir / "swir22.tif", post_swir)

    return pre_dir, post_dir


def test_load_sentinel2_pre_post_returns_grid(tmp_path: Path) -> None:
    _make_sentinel2_pre_post(tmp_path)

    pre_nir, pre_swir, post_nir, post_swir, pixel_size_m, grid = (
        _load_sentinel2_pre_post(tmp_path, event_date=datetime(2026, 1, 15))
    )

    assert pre_nir.shape == (SIZE, SIZE)
    assert post_swir.shape == (SIZE, SIZE)
    assert pixel_size_m == pytest.approx(PIXEL)
    assert grid is not None
    assert grid["crs"] == SRC_CRS.to_string()
    assert grid["shape"] == [SIZE, SIZE]
    assert grid["pixel_size_m"] == pytest.approx(PIXEL)
    assert grid["transform"][0] == pytest.approx(PIXEL)
    assert grid["transform"][2] == pytest.approx(ORIGIN_X)
    assert grid["transform"][5] == pytest.approx(ORIGIN_Y)


def test_analyze_writes_grid_sidecar(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _make_sentinel2_pre_post(data_dir)

    algorithm = load_algorithm("dnbr", None)
    out = _analyze_wildfire(
        algorithm,
        data_dir,
        results_dir,
        raster_files=[],
        npy_files=[],
        event_date=datetime(2026, 1, 15),
    )

    assert out["success"] is True
    assert (results_dir / "dnbr.npy").exists()
    grid_path = results_dir / "grid.json"
    assert grid_path.exists(), "analyze stage must write grid.json sidecar"

    grid = json.loads(grid_path.read_text())
    assert grid["crs"] == SRC_CRS.to_string()
    assert grid["shape"] == [SIZE, SIZE]
    assert grid["transform"][2] == pytest.approx(ORIGIN_X)
    assert grid["transform"][5] == pytest.approx(ORIGIN_Y)


def test_export_preserves_source_bounds(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    products_dir = tmp_path / "products"
    results_dir.mkdir()
    _make_sentinel2_pre_post(data_dir)

    algorithm = load_algorithm("dnbr", None)
    _analyze_wildfire(
        algorithm,
        data_dir,
        results_dir,
        raster_files=[],
        npy_files=[],
        event_date=datetime(2026, 1, 15),
    )

    run_export(
        input_path=results_dir,
        output_path=products_dir,
        formats="geotiff,geojson",
        data_mode="real",
        bbox=None,
    )

    expected_bounds = (
        ORIGIN_X,
        ORIGIN_Y - SIZE * PIXEL,
        ORIGIN_X + SIZE * PIXEL,
        ORIGIN_Y,
    )

    for name in ("dnbr.tif", "burn_severity.tif", "burn_extent.tif", "confidence.tif"):
        path = products_dir / name
        assert path.exists(), f"missing exported product: {name}"
        with rasterio.open(path) as src:
            assert src.crs == SRC_CRS, f"{name} CRS got {src.crs}"
            assert src.bounds.left == pytest.approx(expected_bounds[0])
            assert src.bounds.bottom == pytest.approx(expected_bounds[1])
            assert src.bounds.right == pytest.approx(expected_bounds[2])
            assert src.bounds.top == pytest.approx(expected_bounds[3])
            assert src.shape == (SIZE, SIZE)


def test_export_geojson_uses_grid_for_real_world_coords(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    products_dir = tmp_path / "products"
    results_dir.mkdir()
    _make_sentinel2_pre_post(data_dir)

    algorithm = load_algorithm("dnbr", None)
    _analyze_wildfire(
        algorithm,
        data_dir,
        results_dir,
        raster_files=[],
        npy_files=[],
        event_date=datetime(2026, 1, 15),
    )

    run_export(
        input_path=results_dir,
        output_path=products_dir,
        formats="geojson",
        data_mode="real",
        bbox=None,
    )

    geojson_path = products_dir / "result.geojson"
    assert geojson_path.exists()
    gj = json.loads(geojson_path.read_text())
    assert gj["crs"]["properties"]["name"] == "EPSG:4326"
    features = gj["features"]
    assert features, "expected at least one burn polygon"

    # All coordinates must be on-planet — i.e. not (-180, -90) world placeholders.
    for feature in features:
        coords = feature["geometry"]["coordinates"]
        for ring in coords:
            for lon, lat, *_ in ring:
                assert -90.0 < lon < 0.0, (
                    f"unexpected longitude {lon}; LGT-417 placeholder regression"
                )
                assert 0.0 < lat < 90.0, (
                    f"unexpected latitude {lat}; LGT-417 placeholder regression"
                )
                # AOI in test fixture is south Florida UTM 17N.
                assert -82.0 < lon < -79.0
                assert 25.0 < lat < 28.0


def test_export_falls_back_when_no_grid(tmp_path: Path) -> None:
    """Synthetic / legacy runs without grid.json must still export (placeholder)."""
    results_dir = tmp_path / "results"
    products_dir = tmp_path / "products"
    results_dir.mkdir()

    # Mimic a synthetic run: .npy arrays only, no grid.json.
    extent = np.zeros((8, 8), dtype=np.uint8)
    extent[3:5, 3:5] = 1
    np.save(results_dir / "burn_extent.npy", extent)
    np.save(results_dir / "dnbr.npy", extent.astype(np.float32))

    run_export(
        input_path=results_dir,
        output_path=products_dir,
        formats="geotiff",
        data_mode="synthetic",
        bbox="-82.2,26.3,-81.7,26.8",
    )

    with rasterio.open(products_dir / "burn_extent.tif") as src:
        assert src.bounds.left == pytest.approx(-82.2)
        assert src.bounds.right == pytest.approx(-81.7)
        assert src.bounds.bottom == pytest.approx(26.3)
        assert src.bounds.top == pytest.approx(26.8)
