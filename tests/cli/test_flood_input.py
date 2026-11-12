"""Real-mode flood analysis reads only SAR rasters, never an optical band."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_tif(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32",
        crs="EPSG:32617", transform=from_origin(500000, 2900000, 10, 10),
    ) as dst:
        dst.write(np.full((4, 4), value, dtype="float32"), 1)


def test_flood_refuses_sentinel2_band_as_sar(tmp_path):
    from cli.commands.run import _analyze_flood

    nir = tmp_path / "sentinel2" / "S2A_17RNJ_20260511_0_L2A" / "nir.tif"
    _write_tif(nir, 0.3)
    algorithm = MagicMock()

    with pytest.raises(FileNotFoundError, match="requires SAR data"):
        _analyze_flood(algorithm, tmp_path, tmp_path / "out", [nir], [])
    algorithm.execute.assert_not_called()


def test_flood_reads_raster_named_as_sar(tmp_path):
    from cli.commands.run import _analyze_flood

    nir = tmp_path / "sentinel2" / "scene" / "nir.tif"
    vv = tmp_path / "sentinel1" / "scene" / "vv.tif"
    _write_tif(nir, 0.3)
    _write_tif(vv, -18.0)
    algorithm = MagicMock()
    algorithm.execute.side_effect = RuntimeError("stop after input selection")

    with pytest.raises(RuntimeError, match="stop after input selection"):
        _analyze_flood(algorithm, tmp_path, tmp_path / "out", [nir, vv], [])
    sar_input = algorithm.execute.call_args.args[0]
    assert np.all(sar_input == -18.0)
