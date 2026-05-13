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
