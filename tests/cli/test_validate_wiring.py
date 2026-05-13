"""
Tests for CLI validate command wiring to SanitySuite.

Verifies that the validate command calls real quality checks instead of mocks.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from cli.commands.validate import (
    validate,
    _load_raster_data,
    _run_spatial_coherence_check,
    _run_value_range_check,
    _run_artifact_check,
    _run_coverage_check,
    QUALITY_CHECKS,
)


@pytest.fixture
def sample_raster_data():
    """Create sample raster data for testing."""
    # Create a 100x100 array with some spatial structure
    np.random.seed(42)
    data = np.random.randn(100, 100) * 0.1
    # Add some coherent features
    data[30:50, 30:50] = 0.8
    data[60:80, 60:80] = 0.9
    return data


@pytest.fixture
def temp_raster_file(tmp_path, sample_raster_data):
    """Create a temporary GeoTIFF file."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not available")

    tif_path = tmp_path / "test.tif"

    # Create GeoTIFF
    transform = from_bounds(0, 0, 100, 100, 100, 100)
    with rasterio.open(
        tif_path,
        'w',
        driver='GTiff',
        height=100,
        width=100,
        count=1,
        dtype=sample_raster_data.dtype,
        transform=transform,
    ) as dst:
        dst.write(sample_raster_data, 1)

    return tif_path


class TestLoadRasterData:
    """Test raster data loading."""

    def test_load_from_file(self, temp_raster_file):
        """Test loading raster from a single file."""
        data = _load_raster_data(temp_raster_file)
        assert data is not None
        assert isinstance(data, np.ndarray)
        assert data.shape == (100, 100)

    def test_load_from_directory(self, temp_raster_file):
        """Test loading raster from directory."""
        directory = temp_raster_file.parent
        data = _load_raster_data(directory)
        assert data is not None
        assert isinstance(data, np.ndarray)

    def test_load_nonexistent(self, tmp_path):
        """Test loading from nonexistent path."""
        fake_path = tmp_path / "nonexistent.tif"
        data = _load_raster_data(fake_path)
        assert data is None

    def test_load_npy_file(self, tmp_path, sample_raster_data):
        """Loading a single .npy file should return the array (LGT-418)."""
        npy_path = tmp_path / "dnbr.npy"
        np.save(npy_path, sample_raster_data)

        data = _load_raster_data(npy_path)
        assert data is not None
        assert data.shape == sample_raster_data.shape
        np.testing.assert_array_equal(data, sample_raster_data)

    def test_load_npy_from_directory(self, tmp_path, sample_raster_data):
        """A directory with only .npy outputs (analyze stage) must load (LGT-418)."""
        # Mirror what _analyze_wildfire writes to results/.
        for name in ("dnbr.npy", "burn_severity.npy", "burn_extent.npy", "confidence.npy"):
            np.save(tmp_path / name, sample_raster_data)

        data = _load_raster_data(tmp_path)
        assert data is not None
        assert data.ndim == 2

    def test_geotiff_preferred_over_npy(self, tmp_path, sample_raster_data, temp_raster_file):
        """When both .tif and .npy live in a directory, prefer the .tif."""
        # temp_raster_file already wrote a .tif into tmp_path. Add a different .npy.
        directory = temp_raster_file.parent
        decoy = np.zeros_like(sample_raster_data)
        np.save(directory / "decoy.npy", decoy)

        data = _load_raster_data(directory)
        assert data is not None
        # Should come from the .tif (sample_raster_data), not the all-zeros .npy.
        assert data.shape == sample_raster_data.shape
        assert not np.array_equal(data, decoy)

    def test_load_3d_npy_returns_first_band(self, tmp_path):
        """3D arrays (bands, h, w) should be reduced to band 1."""
        cube = np.stack([np.ones((10, 10)) * i for i in range(3)])
        npy_path = tmp_path / "cube.npy"
        np.save(npy_path, cube)

        data = _load_raster_data(npy_path)
        assert data is not None
        assert data.shape == (10, 10)
        np.testing.assert_array_equal(data, cube[0])


class TestSpatialCoherenceCheck:
    """Test spatial coherence check wiring."""

    def test_run_spatial_check(self, sample_raster_data):
        """Test that spatial coherence check runs and returns proper format."""
        check_info = QUALITY_CHECKS["spatial_coherence"]
        result = _run_spatial_coherence_check(sample_raster_data, check_info)

        # Verify result structure
        assert "check_id" in result
        assert result["check_id"] == "spatial_coherence"
        assert "passed" in result
        assert "score" in result
        assert isinstance(result["score"], float)
        assert 0.0 <= result["score"] <= 1.0
        assert "details" in result
        assert "metrics" in result["details"]

    def test_spatial_check_no_random(self, sample_raster_data):
        """Verify spatial check returns consistent results (not random)."""
        check_info = QUALITY_CHECKS["spatial_coherence"]
        result1 = _run_spatial_coherence_check(sample_raster_data, check_info)
        result2 = _run_spatial_coherence_check(sample_raster_data, check_info)

        # Results should be identical (not random)
        assert result1["score"] == result2["score"]
        assert result1["passed"] == result2["passed"]


class TestValueRangeCheck:
    """Test value range check wiring."""

    def test_run_value_check(self, sample_raster_data):
        """Test that value range check runs properly."""
        check_info = QUALITY_CHECKS["value_range"]
        result = _run_value_range_check(sample_raster_data, check_info)

        assert "check_id" in result
        assert result["check_id"] == "value_range"
        assert "passed" in result
        assert "score" in result
        assert isinstance(result["score"], float)
        assert "statistics" in result["details"]

    def test_value_check_confidence_range(self):
        """Test value check with confidence scores (0-1)."""
        data = np.random.uniform(0.0, 1.0, size=(100, 100))
        check_info = QUALITY_CHECKS["value_range"]
        result = _run_value_range_check(data, check_info)

        # Should pass for valid confidence range
        assert result["passed"]
        assert result["details"]["value_type"] == "confidence"

    def test_value_check_invalid_range(self):
        """Test value check with out-of-range values."""
        data = np.random.uniform(-100.0, 100.0, size=(100, 100))
        check_info = QUALITY_CHECKS["value_range"]
        result = _run_value_range_check(data, check_info)

        # May or may not pass, but should have statistics
        assert "statistics" in result["details"]


class TestArtifactCheck:
    """Test artifact detection wiring."""

    def test_run_artifact_check(self, sample_raster_data):
        """Test that artifact detection runs properly."""
        check_info = QUALITY_CHECKS["artifacts"]
        result = _run_artifact_check(sample_raster_data, check_info)

        assert "check_id" in result
        assert result["check_id"] == "artifacts"
        assert "passed" in result
        assert "score" in result
        assert "artifact_count" in result["details"]

    def test_artifact_check_with_stripes(self):
        """Test artifact detection with stripe pattern."""
        # Create data with horizontal stripes
        data = np.zeros((100, 100))
        data[10, :] = 1.0  # Horizontal stripe
        data[50, :] = 1.0  # Another stripe

        check_info = QUALITY_CHECKS["artifacts"]
        result = _run_artifact_check(data, check_info)

        # Should detect stripes
        assert "artifact_count" in result["details"]
        # May or may not fail depending on threshold, but should detect something


class TestCoverageCheck:
    """Test coverage completeness check."""

    def test_run_coverage_check(self, sample_raster_data):
        """Test coverage check with full data."""
        check_info = QUALITY_CHECKS["coverage"]
        result = _run_coverage_check(sample_raster_data, check_info)

        assert result["check_id"] == "coverage"
        assert result["passed"]
        assert result["details"]["coverage_percentage"] >= 99.0

    def test_coverage_check_with_gaps(self):
        """Test coverage check with data gaps."""
        data = np.ones((100, 100))
        # Add NaN gaps
        data[40:60, 40:60] = np.nan

        check_info = QUALITY_CHECKS["coverage"]
        result = _run_coverage_check(data, check_info)

        assert result["details"]["coverage_percentage"] < 100.0
        assert result["details"]["nodata_percentage"] > 0.0


class TestValidateCommandIntegration:
    """Integration tests for validate command."""

    def test_validate_command_runs(self, temp_raster_file):
        """Test that validate command runs without errors."""
        runner = CliRunner()
        result = runner.invoke(
            validate,
            ['--input', str(temp_raster_file), '--checks', 'spatial_coherence'],
            obj={},
        )

        # Should complete without crashing
        assert result.exit_code in [0, 1]  # 0 = passed, 1 = failed checks
        assert "Quality Validation" in result.output

    def test_validate_all_checks(self, temp_raster_file):
        """Test validate with all checks enabled."""
        runner = CliRunner()
        result = runner.invoke(
            validate,
            ['--input', str(temp_raster_file), '--checks', 'all'],
            obj={},
        )

        assert result.exit_code in [0, 1]
        assert "Overall Score" in result.output

    def test_validate_output_formats(self, temp_raster_file, tmp_path):
        """Test different output formats."""
        runner = CliRunner()

        # JSON format
        json_output = tmp_path / "report.json"
        result = runner.invoke(
            validate,
            [
                '--input', str(temp_raster_file),
                '--output', str(json_output),
                '--format', 'json',
            ],
            obj={},
        )
        assert json_output.exists()

        # Markdown format
        md_output = tmp_path / "report.md"
        result = runner.invoke(
            validate,
            [
                '--input', str(temp_raster_file),
                '--output', str(md_output),
                '--format', 'markdown',
            ],
            obj={},
        )
        assert md_output.exists()


class TestValidateAcceptsAnalyzeNpyOutputs:
    """Regression: validate must accept the .npy outputs analyze emits (LGT-418).

    Before the fix, `_load_raster_data` only looked at `.tif` files, so every
    real-mode wildfire/flood/storm run failed validate with:
        ERROR - No valid raster data found in .../results
    The synthetic wildfire fast path hit the same wall whenever
    `--skip-validate` was omitted.
    """

    def _write_wildfire_results(self, results_dir: Path) -> None:
        """Write the four .npy files _analyze_wildfire produces."""
        results_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        dnbr = rng.uniform(-0.2, 0.8, size=(64, 64)).astype(np.float32)
        np.save(results_dir / "dnbr.npy", dnbr)
        np.save(results_dir / "burn_severity.npy", (dnbr > 0.27).astype(np.uint8))
        np.save(results_dir / "burn_extent.npy", (dnbr > 0.1).astype(np.uint8))
        np.save(
            results_dir / "confidence.npy",
            rng.uniform(0.0, 1.0, size=(64, 64)).astype(np.float32),
        )

    def test_run_quality_checks_does_not_raise_on_npy_only(self, tmp_path):
        """run_quality_checks must succeed on a results dir containing only .npy."""
        from cli.commands.validate import run_quality_checks

        results_dir = tmp_path / "results"
        self._write_wildfire_results(results_dir)

        # Pre-fix behaviour: FileNotFoundError("No valid raster data found ...").
        results = run_quality_checks(results_dir, ["spatial_coherence", "value_range", "coverage"])
        assert len(results) == 3
        # No check should be a hard error stemming from "no data loaded".
        for r in results:
            assert "No valid raster data" not in r.get("message", "")

    def test_validate_cli_does_not_raise_on_npy_only(self, tmp_path):
        """Invoking `flight validate` against an analyze results/ dir must not crash."""
        results_dir = tmp_path / "results"
        self._write_wildfire_results(results_dir)

        runner = CliRunner()
        result = runner.invoke(
            validate,
            [
                "--input", str(results_dir),
                "--checks", "coverage,value_range",
                "--threshold", "0.0",
            ],
            obj={},
        )

        # Should not raise FileNotFoundError from _load_raster_data.
        if result.exception is not None and not isinstance(result.exception, SystemExit):
            raise AssertionError(
                f"validate raised {type(result.exception).__name__}: {result.exception}\n"
                f"Output:\n{result.output}"
            )
        # The pre-fix error message must be gone.
        assert "No valid raster data found" not in result.output
        # Exit 0 (passed) or 1 (failed-by-score) are both fine — what matters is
        # that validate actually ran the checks instead of bailing on load.
        assert result.exit_code in (0, 1)


class TestNoRandomScores:
    """Verify no random scores are used."""

    def test_no_random_import_in_results(self, sample_raster_data):
        """Verify results don't use random module."""
        from cli.commands import validate as validate_module

        # Check that random is not imported at module level for score generation
        # (It may be imported for other reasons, but not for QC scores)
        check_info = QUALITY_CHECKS["spatial_coherence"]

        with patch('random.uniform', side_effect=Exception("random.uniform should not be called")):
            # This should NOT raise an exception because we're using real checks
            result = _run_spatial_coherence_check(sample_raster_data, check_info)
            assert "score" in result

    def test_deterministic_results(self, sample_raster_data):
        """Verify all checks return deterministic (non-random) results."""
        check_info = QUALITY_CHECKS["spatial_coherence"]

        # Run check multiple times
        results = [
            _run_spatial_coherence_check(sample_raster_data, check_info)
            for _ in range(3)
        ]

        # All scores should be identical
        scores = [r["score"] for r in results]
        assert len(set(scores)) == 1, "Scores should be deterministic, not random"
