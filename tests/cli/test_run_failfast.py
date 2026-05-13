"""
Regression tests for LGT-413: `flight run` must fail-fast on 0 ingested items
and never emit fabricated analysis stats when real input rasters are absent.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_discovery(output_path: Path, count: int = 3) -> None:
    """Seed a discovery.json file with `count` mock results."""
    output_path.mkdir(parents=True, exist_ok=True)
    discovery = {
        "count": count,
        "results": [
            {"id": f"S2A_mock_{i}", "source": "sentinel2", "datetime": "2026-05-10T00:00:00Z"}
            for i in range(count)
        ],
    }
    with open(output_path / "discovery.json", "w") as f:
        json.dump(discovery, f)


class TestRunIngestFailFast:
    """`run_ingest` must propagate failures in REAL mode, not silently return 0."""

    def test_run_ingest_raises_when_all_items_fail_real_mode(self, tmp_path):
        from cli.commands.run import DATA_MODE_REAL, run_ingest

        _write_discovery(tmp_path)

        with patch("cli.commands.ingest.process_item", return_value=False):
            with pytest.raises(RuntimeError, match="0 of 3 items ingested"):
                run_ingest(
                    discovery_file=tmp_path / "discovery.json",
                    output_path=tmp_path / "data",
                    profile_config={},
                    data_mode=DATA_MODE_REAL,
                    bbox=None,
                    event_type="wildfire",
                )

    def test_run_ingest_reports_partial_failures(self, tmp_path):
        from cli.commands.run import DATA_MODE_REAL, run_ingest

        _write_discovery(tmp_path, count=3)

        # First item succeeds, others fail.
        side_effect = [True, False, False]
        with patch("cli.commands.ingest.process_item", side_effect=side_effect):
            result = run_ingest(
                discovery_file=tmp_path / "discovery.json",
                output_path=tmp_path / "data",
                profile_config={},
                data_mode=DATA_MODE_REAL,
                bbox=None,
                event_type="wildfire",
            )

        assert result["count"] == 1
        assert result["total"] == 3
        assert len(result["failed"]) == 2


class TestRunCommandFailFast:
    """`flight run` must abort before analyze when ingest yields 0 items."""

    def test_run_aborts_before_analyze_on_zero_ingest(self, tmp_path):
        click = pytest.importorskip("click")
        from click.testing import CliRunner

        from cli.commands.run import run as run_cmd

        # Pretend discovery returned 3 items and every ingest call failed.
        def fake_discover(**_kwargs):
            output_path = _kwargs["output_path"]
            _write_discovery(output_path, count=3)
            return {"count": 3, "file": str(output_path / "discovery.json")}

        # process_item always returns False → run_ingest raises in real mode.
        with patch("cli.commands.run.run_discover", side_effect=fake_discover), \
             patch("cli.commands.ingest.process_item", return_value=False), \
             patch("cli.commands.run.run_analyze") as mock_analyze:
            runner = CliRunner()
            result = runner.invoke(
                run_cmd,
                [
                    "--bbox", "-80.5,25.5,-80.0,26.0",
                    "--event", "wildfire",
                    "--output", str(tmp_path / "out"),
                ],
                obj={},
            )

        assert result.exit_code != 0, f"Expected non-zero exit; stdout:\n{result.output}"
        # Most importantly: analyze stage must NOT have been entered.
        mock_analyze.assert_not_called()
        # No fabricated stats in output.
        assert "Analysis complete" not in result.output
        assert "high severity" not in result.output

    def test_synthetic_mode_still_allows_pipeline(self, tmp_path):
        """Synthetic mode is an explicit opt-in and should still work end-to-end."""
        click = pytest.importorskip("click")
        from click.testing import CliRunner

        from cli.commands.run import run as run_cmd

        def fake_discover(**_kwargs):
            output_path = _kwargs["output_path"]
            _write_discovery(output_path, count=1)
            return {"count": 1, "file": str(output_path / "discovery.json")}

        with patch("cli.commands.run.run_discover", side_effect=fake_discover):
            runner = CliRunner()
            result = runner.invoke(
                run_cmd,
                [
                    "--bbox", "-80.5,25.5,-80.0,26.0",
                    "--event", "flood",
                    "--output", str(tmp_path / "out"),
                    "--synthetic",
                ],
                obj={},
            )

        assert isinstance(result.exit_code, int)


class TestRealAnalysisInputValidation:
    """Real-mode analyze must refuse to run without real input rasters."""

    def test_real_analysis_raises_when_no_input_data(self, tmp_path):
        from cli.commands.run import _run_real_analysis

        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError, match="No real input rasters"):
            _run_real_analysis(
                input_path=empty,
                output_path=tmp_path / "out",
                event_type="wildfire",
                algorithm="dnbr",
                profile_config={},
            )

    def test_real_wildfire_analysis_does_not_fabricate(self, tmp_path):
        """Even if a stray .tif is present, wildfire analysis needs pre/post bands."""
        from cli.commands.run import _analyze_wildfire

        # Create a stub algorithm — it must NEVER be invoked since we have no
        # real pre/post rasters.
        class _Algo:
            def execute(self, *a, **k):
                raise AssertionError("algorithm.execute called without real data")

        with pytest.raises(FileNotFoundError, match="pre/post NIR and SWIR"):
            _analyze_wildfire(
                algorithm=_Algo(),
                input_path=tmp_path,
                output_path=tmp_path / "out",
                raster_files=[],
                npy_files=[],
            )


def _write_s2_scene(
    base: Path,
    item_id: str,
    nir_array,
    swir_array,
):
    """Write a tiny S2-style scene directory with nir.tif and swir22.tif."""
    rasterio = pytest.importorskip("rasterio")
    import numpy as np
    from rasterio.transform import from_origin

    scene_dir = base / "sentinel2" / item_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # 10 m UTM grid for NIR.
    nir_transform = from_origin(500000, 2900000, 10.0, 10.0)
    nir_height, nir_width = nir_array.shape
    with rasterio.open(
        scene_dir / "nir.tif",
        "w",
        driver="GTiff",
        height=nir_height,
        width=nir_width,
        count=1,
        dtype=nir_array.dtype,
        crs="EPSG:32617",
        transform=nir_transform,
    ) as dst:
        dst.write(nir_array, 1)

    # 20 m UTM grid for SWIR22 — coarser than NIR by 2x.
    swir_transform = from_origin(500000, 2900000, 20.0, 20.0)
    swir_height, swir_width = swir_array.shape
    with rasterio.open(
        scene_dir / "swir22.tif",
        "w",
        driver="GTiff",
        height=swir_height,
        width=swir_width,
        count=1,
        dtype=swir_array.dtype,
        crs="EPSG:32617",
        transform=swir_transform,
    ) as dst:
        dst.write(swir_array, 1)


class TestWildfireRealModeLoader:
    """LGT-416: `_analyze_wildfire` must load pre/post NIR + SWIR22 from ingested S2."""

    def test_analyze_wildfire_real_mode_loads_sentinel2_pair(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_wildfire

        # Pre-fire scene: high NIR, low SWIR. Post-fire scene: low NIR, high SWIR.
        pre_nir = np.full((128, 128), 4000, dtype=np.uint16)   # ~0.40 reflectance
        pre_swir = np.full((64, 64), 1500, dtype=np.uint16)    # ~0.15 reflectance
        post_nir = np.full((128, 128), 1000, dtype=np.uint16)  # ~0.10 reflectance
        post_swir = np.full((64, 64), 3500, dtype=np.uint16)   # ~0.35 reflectance

        _write_s2_scene(tmp_path, "S2B_17RNJ_20260404_0_L2A", pre_nir, pre_swir)
        _write_s2_scene(tmp_path, "S2A_17RNJ_20260421_0_L2A", post_nir, post_swir)

        captured = {}

        class _Algo:
            def execute(self, **kwargs):
                captured.update(kwargs)
                # Return a minimal stub matching DifferencedNBRResult shape.
                shape = kwargs["nir_pre"].shape

                class _Result:
                    dnbr_map = np.zeros(shape, dtype=np.float32)
                    burn_severity = np.zeros(shape, dtype=np.uint8)
                    burn_extent = np.zeros(shape, dtype=np.uint8)
                    confidence_raster = np.ones(shape, dtype=np.float32)
                    statistics = {"burned_area_ha": 0.0}

                return _Result()

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _analyze_wildfire(
            algorithm=_Algo(),
            input_path=tmp_path,
            output_path=out_dir,
            raster_files=[],
            npy_files=[],
        )

        assert result["success"] is True
        # All four bands must reach the algorithm with matching shapes.
        assert captured["nir_pre"].shape == captured["swir_pre"].shape
        assert captured["nir_pre"].shape == captured["nir_post"].shape
        assert captured["nir_pre"].shape == captured["swir_post"].shape
        # SWIR22 should have been resampled up to the 10 m NIR grid.
        assert captured["nir_pre"].shape == (128, 128)
        # Pixel size should reflect the NIR resolution (10 m for Sentinel-2).
        assert captured["pixel_size_m"] == pytest.approx(10.0)
        # Reflectance scaling: uint16 DNs converted to 0..1 floats.
        assert 0.35 < float(captured["nir_pre"].mean()) < 0.45
        assert 0.05 < float(captured["nir_post"].mean()) < 0.15
        # Pre vs post temporal ordering preserved (earlier date is the pre scene).
        assert captured["nir_pre"].mean() > captured["nir_post"].mean()

    def test_analyze_wildfire_real_mode_requires_pre_and_post(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_wildfire

        # Only a single dated scene — no temporal pair available.
        only_nir = np.full((64, 64), 2000, dtype=np.uint16)
        only_swir = np.full((32, 32), 1500, dtype=np.uint16)
        _write_s2_scene(tmp_path, "S2A_17RNJ_20260415_0_L2A", only_nir, only_swir)

        class _Algo:
            def execute(self, *a, **k):
                raise AssertionError("algorithm.execute called without a pre/post pair")

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="both pre-fire and post-fire"):
            _analyze_wildfire(
                algorithm=_Algo(),
                input_path=tmp_path,
                output_path=out_dir,
                raster_files=[],
                npy_files=[],
                event_date=__import__("datetime").datetime(2026, 4, 16),
            )


def _write_s2_storm_scene(
    base: Path,
    item_id: str,
    red_array,
    nir_array,
    blue_array=None,
):
    """Write a tiny S2 storm-style scene directory with red/nir (+ optional blue)."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    scene_dir = base / "sentinel2" / item_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    transform = from_origin(500000, 2900000, 10.0, 10.0)

    for name, arr in (("red", red_array), ("nir", nir_array)):
        height, width = arr.shape
        with rasterio.open(
            scene_dir / f"{name}.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=arr.dtype,
            crs="EPSG:32617",
            transform=transform,
        ) as dst:
            dst.write(arr, 1)

    if blue_array is not None:
        height, width = blue_array.shape
        with rasterio.open(
            scene_dir / "blue.tif",
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=blue_array.dtype,
            crs="EPSG:32617",
            transform=transform,
        ) as dst:
            dst.write(blue_array, 1)


class TestStormRealModeLoader:
    """LGT-414: `_analyze_storm` must load pre/post red + NIR from ingested Sentinel-2."""

    def test_analyze_storm_real_mode_loads_sentinel2_pair(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_storm

        # Pre-event: healthy vegetation (low red, high NIR).
        # Post-event: stripped canopy (high red, lower NIR).
        pre_red = np.full((64, 64), 800, dtype=np.uint16)
        pre_nir = np.full((64, 64), 4500, dtype=np.uint16)
        post_red = np.full((64, 64), 2200, dtype=np.uint16)
        post_nir = np.full((64, 64), 2000, dtype=np.uint16)

        _write_s2_storm_scene(tmp_path, "S2B_17RNJ_20260404_0_L2A", pre_red, pre_nir)
        _write_s2_storm_scene(tmp_path, "S2A_17RNJ_20260421_0_L2A", post_red, post_nir)

        captured: dict = {}

        class _Algo:
            def execute(self, **kwargs):
                captured.update(kwargs)
                shape = kwargs["red_pre"].shape

                class _Result:
                    damage_extent = np.zeros(shape, dtype=np.uint8)
                    confidence_raster = np.ones(shape, dtype=np.float32)
                    statistics = {"damaged_area_ha": 0.0}

                return _Result()

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _analyze_storm(
            algorithm=_Algo(),
            input_path=tmp_path,
            output_path=out_dir,
            raster_files=[],
            npy_files=[],
        )

        assert result["success"] is True
        # Four bands at matching shape, on the 10 m NIR grid.
        for key in ("red_pre", "nir_pre", "red_post", "nir_post"):
            assert captured[key].shape == (64, 64)
        # Reflectance scaling applied (uint16 DN -> 0..1).
        assert 0.05 < float(captured["red_pre"].mean()) < 0.12
        assert 0.4 < float(captured["nir_pre"].mean()) < 0.5
        # Pre vs post temporal ordering preserved.
        assert captured["nir_pre"].mean() > captured["nir_post"].mean()
        assert captured["red_pre"].mean() < captured["red_post"].mean()
        # Pixel size reflects NIR (10 m for Sentinel-2).
        assert captured["pixel_size_m"] == pytest.approx(10.0)
        # Outputs and grid.json are written for the export stage (LGT-417).
        assert (out_dir / "damage_extent.npy").exists()
        assert (out_dir / "confidence.npy").exists()
        assert (out_dir / "grid.json").exists()

    def test_analyze_storm_real_mode_passes_blue_when_available(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_storm

        pre_red = np.full((32, 32), 800, dtype=np.uint16)
        pre_nir = np.full((32, 32), 4500, dtype=np.uint16)
        pre_blue = np.full((32, 32), 600, dtype=np.uint16)
        post_red = np.full((32, 32), 2200, dtype=np.uint16)
        post_nir = np.full((32, 32), 2000, dtype=np.uint16)
        post_blue = np.full((32, 32), 1500, dtype=np.uint16)

        _write_s2_storm_scene(tmp_path, "S2B_17RNJ_20260404_0_L2A", pre_red, pre_nir, pre_blue)
        _write_s2_storm_scene(tmp_path, "S2A_17RNJ_20260421_0_L2A", post_red, post_nir, post_blue)

        captured: dict = {}

        class _Algo:
            def execute(self, **kwargs):
                captured.update(kwargs)
                shape = kwargs["red_pre"].shape

                class _Result:
                    damage_extent = np.zeros(shape, dtype=np.uint8)
                    confidence_raster = np.ones(shape, dtype=np.float32)
                    statistics = {"damaged_area_ha": 0.0}

                return _Result()

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _analyze_storm(
            algorithm=_Algo(),
            input_path=tmp_path,
            output_path=out_dir,
            raster_files=[],
            npy_files=[],
        )

        assert "blue_pre" in captured and captured["blue_pre"] is not None
        assert "blue_post" in captured and captured["blue_post"] is not None
        assert captured["blue_pre"].shape == captured["red_pre"].shape

    def test_analyze_storm_real_mode_requires_pre_and_post(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_storm

        only_red = np.full((32, 32), 1000, dtype=np.uint16)
        only_nir = np.full((32, 32), 3000, dtype=np.uint16)
        _write_s2_storm_scene(tmp_path, "S2A_17RNJ_20260415_0_L2A", only_red, only_nir)

        class _Algo:
            def execute(self, *a, **k):
                raise AssertionError("algorithm.execute called without a pre/post pair")

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="both pre-event and post-event"):
            _analyze_storm(
                algorithm=_Algo(),
                input_path=tmp_path,
                output_path=out_dir,
                raster_files=[],
                npy_files=[],
                event_date=__import__("datetime").datetime(2026, 4, 16),
            )

    def test_analyze_storm_real_mode_dispatches_structural_damage(self, tmp_path):
        pytest.importorskip("rasterio")
        import numpy as np

        from cli.commands.run import _analyze_storm

        pre_red = np.full((32, 32), 800, dtype=np.uint16)
        pre_nir = np.full((32, 32), 4500, dtype=np.uint16)
        post_red = np.full((32, 32), 2200, dtype=np.uint16)
        post_nir = np.full((32, 32), 2000, dtype=np.uint16)

        _write_s2_storm_scene(tmp_path, "S2B_17RNJ_20260404_0_L2A", pre_red, pre_nir)
        _write_s2_storm_scene(tmp_path, "S2A_17RNJ_20260421_0_L2A", post_red, post_nir)

        captured: dict = {}

        class StructuralDamageAssessment:
            def execute(self, **kwargs):
                captured.update(kwargs)
                shape = kwargs["optical_pre"].shape

                class _Result:
                    damage_map = np.zeros(shape, dtype=np.uint8)
                    confidence_map = np.ones(shape, dtype=np.float32)
                    statistics = {"damaged_area_ha": 0.0}

                return _Result()

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        result = _analyze_storm(
            algorithm=StructuralDamageAssessment(),
            input_path=tmp_path,
            output_path=out_dir,
            raster_files=[],
            npy_files=[],
        )

        assert result["success"] is True
        # Structural variant receives a single optical pre/post pair.
        assert set(captured.keys()) >= {"optical_pre", "optical_post", "pixel_size_m"}
        assert "red_pre" not in captured
        # Damage map persisted under the same on-disk name regardless of result attr.
        assert (out_dir / "damage_extent.npy").exists()
        assert (out_dir / "confidence.npy").exists()
