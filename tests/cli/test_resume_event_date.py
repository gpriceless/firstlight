"""
Regression tests for LGT-419: `flight resume` must forward event_date,
bbox, and data_mode into the analyze stage. Previously the analyze
branch called `run_analyze` without those kwargs, causing the Sentinel-2
pre/post loader to fall back to a midpoint heuristic that picked the
wrong pre/post pair on three-scene cases (Max Road Miramar: Apr 04 +
Apr 21 instead of Apr 21 + May 11).
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_workflow_state(workdir: Path, completed_stages, config_overrides=None) -> None:
    """Seed a `.workflow_state.json` mimicking what `flight run` writes."""
    workdir.mkdir(parents=True, exist_ok=True)
    config = {
        "area_path": None,
        "bbox": "-80.50,25.95,-80.40,26.05",
        "analyze_bbox": "-80.50,25.95,-80.40,26.05",
        "max_cloud": 90.0,
        "start_date": "2026-04-01T00:00:00",
        "end_date": "2026-05-13T00:00:00",
        "event_type": "wildfire",
        "profile": "workstation",
        "algorithm": "dnbr",
        "formats": "geotiff,geojson",
        "data_mode": "synthetic",  # safe default for tests; avoids real network
        "skip_validate": False,
    }
    if config_overrides:
        config.update(config_overrides)

    state = {
        "completed_stages": list(completed_stages),
        "current_stage": None,
        "stage_results": {s: {"status": "completed"} for s in completed_stages},
        "config": config,
    }
    with open(workdir / ".workflow_state.json", "w") as f:
        json.dump(state, f)


class TestResumeForwardsAnalyzeContext:
    """LGT-419: analyze on resume forwards event_date, bbox, data_mode."""

    def test_resume_analyze_forwards_event_date_from_window_midpoint(self, tmp_path):
        from click.testing import CliRunner

        from cli.commands.resume import resume as resume_cmd

        workdir = tmp_path / "out"
        _write_workflow_state(workdir, completed_stages=["discover", "ingest"])

        captured: dict = {}

        def fake_analyze(**kwargs):
            captured.update(kwargs)
            return {"algorithm": "dnbr", "path": "/tmp/out", "statistics": {}}

        with patch("cli.commands.run.run_analyze", side_effect=fake_analyze), \
             patch("cli.commands.run.run_validate", return_value={"score": 1.0, "passed": True}), \
             patch("cli.commands.run.run_export", return_value={"formats": ["geotiff"]}):
            result = CliRunner().invoke(
                resume_cmd,
                ["--workdir", str(workdir), "--from-stage", "analyze"],
                obj={},
            )

        assert result.exit_code == 0, f"resume failed:\n{result.output}"
        # `event_date` is the window midpoint (Apr 01 + (May 13 - Apr 01)/2 = ~Apr 22).
        event_date = captured.get("event_date")
        assert event_date is not None, "resume analyze must forward event_date"
        assert isinstance(event_date, datetime)
        assert event_date.date() == datetime(2026, 4, 22).date()

    def test_resume_analyze_forwards_bbox_for_aoi_clipping(self, tmp_path):
        from click.testing import CliRunner

        from cli.commands.resume import resume as resume_cmd

        workdir = tmp_path / "out"
        _write_workflow_state(workdir, completed_stages=["discover", "ingest"])

        captured: dict = {}

        def fake_analyze(**kwargs):
            captured.update(kwargs)
            return {"algorithm": "dnbr", "path": "/tmp/out", "statistics": {}}

        with patch("cli.commands.run.run_analyze", side_effect=fake_analyze), \
             patch("cli.commands.run.run_validate", return_value={"score": 1.0, "passed": True}), \
             patch("cli.commands.run.run_export", return_value={"formats": ["geotiff"]}):
            CliRunner().invoke(
                resume_cmd,
                ["--workdir", str(workdir), "--from-stage", "analyze"],
                obj={},
            )

        assert captured.get("bbox") == "-80.50,25.95,-80.40,26.05"

    def test_resume_analyze_forwards_data_mode(self, tmp_path):
        from click.testing import CliRunner

        from cli.commands.resume import resume as resume_cmd

        workdir = tmp_path / "out"
        _write_workflow_state(workdir, completed_stages=["discover", "ingest"])

        captured: dict = {}

        def fake_analyze(**kwargs):
            captured.update(kwargs)
            return {"algorithm": "dnbr", "path": "/tmp/out", "statistics": {}}

        with patch("cli.commands.run.run_analyze", side_effect=fake_analyze), \
             patch("cli.commands.run.run_validate", return_value={"score": 1.0, "passed": True}), \
             patch("cli.commands.run.run_export", return_value={"formats": ["geotiff"]}):
            CliRunner().invoke(
                resume_cmd,
                ["--workdir", str(workdir), "--from-stage", "analyze"],
                obj={},
            )

        assert captured.get("data_mode") == "synthetic"

    def test_resume_falls_back_to_bbox_when_analyze_bbox_absent(self, tmp_path):
        """Old state.json (pre-LGT-420) lacks analyze_bbox; resume must use bbox."""
        from click.testing import CliRunner

        from cli.commands.resume import resume as resume_cmd

        workdir = tmp_path / "out"
        _write_workflow_state(
            workdir,
            completed_stages=["discover", "ingest"],
            config_overrides={"analyze_bbox": None},
        )

        captured: dict = {}

        def fake_analyze(**kwargs):
            captured.update(kwargs)
            return {"algorithm": "dnbr", "path": "/tmp/out", "statistics": {}}

        with patch("cli.commands.run.run_analyze", side_effect=fake_analyze), \
             patch("cli.commands.run.run_validate", return_value={"score": 1.0, "passed": True}), \
             patch("cli.commands.run.run_export", return_value={"formats": ["geotiff"]}):
            CliRunner().invoke(
                resume_cmd,
                ["--workdir", str(workdir), "--from-stage", "analyze"],
                obj={},
            )

        assert captured.get("bbox") == "-80.50,25.95,-80.40,26.05"


class TestResumePicksCorrectPrePostScenesByEventDate:
    """End-to-end-ish: resume's analyze stage feeds event_date into
    `_find_sentinel2_pre_post_scenes` so a 3-scene case (one post-event)
    picks the latest-pre + earliest-post, not the midpoint pair."""

    def test_three_scene_resume_picks_pre_post_around_event_date(self, tmp_path):
        pytest.importorskip("rasterio")
        from cli.commands.run import _find_sentinel2_pre_post_scenes

        # Recreate the Max Road Miramar scene set: Apr 04, Apr 21, May 11.
        for item_id in (
            "S2B_17RNJ_20260404_0_L2A",
            "S2A_17RNJ_20260421_0_L2A",
            "S2A_17RNJ_20260511_0_L2A",
        ):
            scene_dir = tmp_path / "sentinel2" / item_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            # `_find_..._scenes` only checks for nir.tif + swir22.tif existence;
            # content doesn't matter for this assertion.
            (scene_dir / "nir.tif").write_bytes(b"")
            (scene_dir / "swir22.tif").write_bytes(b"")

        # Without an event_date (the bug): midpoint heuristic splits at
        # the second scene and the post window collapses to Apr 21 alone.
        pair_no_event = _find_sentinel2_pre_post_scenes(tmp_path, event_date=None)
        assert pair_no_event["pre"]["item_id"] == "S2B_17RNJ_20260404_0_L2A"
        assert pair_no_event["post"]["item_id"] == "S2A_17RNJ_20260421_0_L2A"

        # With an event_date (resume now forwards this): pre = Apr 21,
        # post = May 11 — the correct bracket of the 2026-05-10 fire.
        pair_with_event = _find_sentinel2_pre_post_scenes(
            tmp_path, event_date=datetime(2026, 5, 10)
        )
        assert pair_with_event["pre"]["item_id"] == "S2A_17RNJ_20260421_0_L2A"
        assert pair_with_event["post"]["item_id"] == "S2A_17RNJ_20260511_0_L2A"
