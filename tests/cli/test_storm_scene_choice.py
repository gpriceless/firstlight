"""Storm picks its scenes by how much of the box is clear, like wildfire.

Same real 1 km block as ``test_wildfire_cloud_mask.py`` (Max Road box, tile
17RNJ), with B04 red cut from the same Earth Search scenes onto the NIR grid.
The 11 May scene is all cloud over the block (red never below 2140 DN); a
nearest-date pick takes it as the post scene for a 10 May event.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pytest

rasterio = pytest.importorskip("rasterio")

from cli.commands.analyze import load_algorithm
from cli.commands.run import _analyze_storm, _find_sentinel2_pre_post_scenes

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "s2_cloud"
PRE = "S2C_17RNJ_20260509_0_L2A"
CLOUDED = "S2A_17RNJ_20260511_0_L2A"
POST = "S2A_17RNJ_20260531_0_L2A"
STORM_BANDS = dict(bands=("red", "nir"), optional=("blue",), hazard="storm")


def _tree(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE, tmp_path / "sentinel2")
    return tmp_path


def test_storm_skips_the_clouded_scene_nearest_the_event(tmp_path):
    scenes = _find_sentinel2_pre_post_scenes(
        _tree(tmp_path), event_date=datetime(2026, 5, 10), **STORM_BANDS
    )
    assert scenes["pre"]["item_id"] == PRE
    assert scenes["post"]["item_id"] == POST
    assert scenes["post"]["clear_share"] == pytest.approx(1 - 0.318)
    assert scenes["post"]["red"].name == "red.tif"
    assert scenes["post"]["blue"] is None


def test_storm_run_records_the_chosen_scenes_and_clear_shares(tmp_path):
    data_dir = _tree(tmp_path / "data")
    results = tmp_path / "results"
    results.mkdir()

    _analyze_storm(
        load_algorithm("wind_damage", None), data_dir, results,
        raster_files=[], npy_files=[], event_date=datetime(2026, 5, 10),
    )

    grid = json.loads((results / "grid.json").read_text())
    assert grid["post_item_id"] == POST
    assert grid["post_clear_share"] == pytest.approx(1 - 0.318, abs=1e-4)
    assert grid["pre_item_id"] == PRE


def test_storm_refuses_scenes_without_scl(tmp_path):
    root = _tree(tmp_path)
    for sid in (PRE, CLOUDED, POST):
        (root / "sentinel2" / sid / "scl.tif").unlink()

    with pytest.raises(FileNotFoundError, match="storm.*SCL"):
        _find_sentinel2_pre_post_scenes(
            root, event_date=datetime(2026, 5, 10), **STORM_BANDS
        )
