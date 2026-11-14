"""
Tests for CLI Analyze Command Wiring to AlgorithmRegistry

Verifies that:
1. Algorithm registry lookup works correctly
2. Real algorithm classes are instantiated
3. Missing algorithms raise descriptive errors
4. Missing input files raise descriptive errors
5. No MockAlgorithm class exists
6. No random fallback output
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from cli.commands.analyze import load_algorithm, run_analysis, ALGORITHMS


class TestAlgorithmLoading:
    """Test algorithm loading and instantiation."""

    def test_load_sar_threshold_algorithm(self):
        """Test loading SAR threshold algorithm with default parameters."""
        algorithm = load_algorithm("sar_threshold", {})

        # Verify it's the real class, not a mock
        assert algorithm.__class__.__name__ == "ThresholdSARAlgorithm"
        assert hasattr(algorithm, "execute")
        assert hasattr(algorithm, "config")

    def test_load_sar_threshold_with_params(self):
        """Test loading SAR threshold algorithm with custom parameters."""
        params = {
            "threshold_db": -16.0,
            "min_area_ha": 1.0,
        }
        algorithm = load_algorithm("sar_threshold", params)

        assert algorithm.__class__.__name__ == "ThresholdSARAlgorithm"
        assert algorithm.config.threshold_db == -16.0
        assert algorithm.config.min_area_ha == 1.0

    def test_load_ndwi_algorithm(self):
        """Test loading NDWI optical algorithm."""
        algorithm = load_algorithm("ndwi", {})

        assert algorithm.__class__.__name__ == "NDWIOpticalAlgorithm"
        assert hasattr(algorithm, "execute")

    def test_load_dnbr_algorithm(self):
        """Test loading dNBR wildfire algorithm."""
        algorithm = load_algorithm("dnbr", {})

        assert algorithm.__class__.__name__ == "DifferencedNBRAlgorithm"
        assert hasattr(algorithm, "execute")

    def test_load_thermal_anomaly_algorithm(self):
        """Test loading thermal anomaly algorithm."""
        algorithm = load_algorithm("thermal_anomaly", {})

        assert algorithm.__class__.__name__ == "ThermalAnomalyAlgorithm"
        assert hasattr(algorithm, "execute")

    def test_load_wind_damage_algorithm(self):
        """Test loading wind damage algorithm."""
        algorithm = load_algorithm("wind_damage", {})

        assert algorithm.__class__.__name__ == "WindDamageDetection"
        assert hasattr(algorithm, "execute")

    def test_missing_algorithm_raises_error(self):
        """Test that missing algorithm raises ImportError."""
        # Temporarily break the module path
        original_module = ALGORITHMS["sar_threshold"]["module"]
        ALGORITHMS["sar_threshold"]["module"] = "nonexistent.module"

        with pytest.raises(ImportError) as excinfo:
            load_algorithm("sar_threshold", {})

        assert "not available" in str(excinfo.value)

        # Restore
        ALGORITHMS["sar_threshold"]["module"] = original_module

    def test_algorithm_metadata_correct(self):
        """Test that algorithms have correct metadata."""
        algo_info = ALGORITHMS["sar_threshold"]

        assert algo_info["name"] == "SAR Backscatter Threshold"
        assert algo_info["module"] == "core.analysis.library.baseline.flood.threshold_sar"
        assert algo_info["class"] == "ThresholdSARAlgorithm"
        assert "flood" in algo_info["event_types"]
        assert algo_info["supports_tiled"] is True


class TestRunAnalysisWiring:
    """Test the run_analysis function wiring."""

    def test_run_analysis_with_real_algorithm(self, tmp_path):
        """Test run_analysis with a real algorithm instance."""
        # Create mock input/output paths
        input_path = tmp_path / "input"
        output_path = tmp_path / "output"
        input_path.mkdir()
        output_path.mkdir()

        # A small real GeoTIFF of SAR-like backscatter in dB.
        import rasterio
        from rasterio.transform import from_origin

        raster_file = input_path / "vv.tif"
        with rasterio.open(
            raster_file, "w", driver="GTiff", height=64, width=64, count=1,
            dtype="float32", crs="EPSG:32617",
            transform=from_origin(500000, 2900000, 10, 10),
        ) as dst:
            dst.write(np.full((64, 64), -18.0, dtype=np.float32), 1)

        algorithm = load_algorithm("sar_threshold", {})

        results = run_analysis(
            input_path=input_path,
            output_path=output_path,
            algorithm=algorithm,
            algo_info=ALGORITHMS["sar_threshold"],
            tiles=[0],
            tile_size=512,
            parallel=1,
            supports_tiled=False,
        )

        assert results["tiles_completed"] == 1
        assert results["tiles_failed"] == 0
        # -18 dB is below the water threshold, so every pixel is flagged.
        assert results["statistics"]["flood_percentage"] == 100.0
        with rasterio.open(output_path / "flood_extent.tif") as out:
            assert out.crs.to_epsg() == 32617

    def test_missing_input_files_raises_error(self, tmp_path):
        """Test that missing input files raise FileNotFoundError."""
        input_path = tmp_path / "empty_input"
        output_path = tmp_path / "output"
        input_path.mkdir()
        output_path.mkdir()

        algorithm = load_algorithm("sar_threshold", {})
        algo_info = ALGORITHMS["sar_threshold"]

        # Should raise FileNotFoundError when no .tif files exist
        with pytest.raises(FileNotFoundError) as excinfo:
            run_analysis(
                input_path=input_path,
                output_path=output_path,
                algorithm=algorithm,
                algo_info=algo_info,
                tiles=[0],
                tile_size=512,
                parallel=1,
                supports_tiled=False
            )

        assert "No input raster files" in str(excinfo.value)
        assert str(input_path) in str(excinfo.value)


def _write_band(path, value, size=16):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:32617",
        transform=from_origin(500000, 2900000, 10, 10),
    ) as dst:
        dst.write(np.full((size, size), value, dtype=np.float32), 1)


class TestAnalyzeInputSelection:
    """flight analyze reads the raster the algorithm declares, not the first one found."""

    def _run(self, input_path, output_path, name):
        output_path.mkdir(exist_ok=True)
        return run_analysis(
            input_path=input_path,
            output_path=output_path,
            algorithm=load_algorithm(name, {}),
            algo_info=ALGORITHMS[name],
            tiles=[0],
            tile_size=512,
            parallel=1,
            supports_tiled=False,
        )

    def test_sar_threshold_reads_vv_not_an_optical_band_found_first(self, tmp_path, monkeypatch):
        input_path = tmp_path / "input"
        nir = input_path / "a_scene" / "nir.tif"
        vv = input_path / "z_scene" / "vv.tif"
        _write_band(nir, 0.3)
        _write_band(vv, -18.0)
        # rglob order is up to the filesystem; put the optical band first.
        monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter([nir, vv]))

        results = self._run(input_path, tmp_path / "out", "sar_threshold")

        # Reflectance 0.3 read as dB would be "no water"; -18 dB is all water.
        assert results["tiles_completed"] == 1
        assert results["statistics"]["flood_percentage"] == 100.0

    def test_refuses_and_names_the_input_it_wanted(self, tmp_path):
        input_path = tmp_path / "input"
        _write_band(input_path / "scene" / "nir.tif", 0.3)
        _write_band(input_path / "scene" / "red.tif", 0.1)

        with pytest.raises(FileNotFoundError) as excinfo:
            self._run(input_path, tmp_path / "out", "sar_threshold")

        message = str(excinfo.value)
        assert "sar raster" in message
        assert "vv, vh" in message
        assert "nir.tif, red.tif" in message
        assert not (tmp_path / "out" / "flood_extent.tif").exists()

    def test_refuses_an_algorithm_that_needs_more_than_one_input(self, tmp_path):
        input_path = tmp_path / "input"
        _write_band(input_path / "scene" / "nir.tif", 0.3)
        _write_band(input_path / "scene" / "swir16.tif", 0.2)

        with pytest.raises(ValueError) as excinfo:
            self._run(input_path, tmp_path / "out", "dnbr")

        message = str(excinfo.value)
        assert "needs 2 inputs" in message
        assert "nir, swir" in message
        assert "flight run" in message

    def test_tile_estimate_counts_the_declared_raster(self, tmp_path, monkeypatch):
        from cli.commands.analyze import estimate_tiles

        input_path = tmp_path / "input"
        nir = input_path / "a_scene" / "nir.tif"
        vv = input_path / "z_scene" / "vv.tif"
        _write_band(nir, 0.3, size=1100)  # 9 tiles of 512
        _write_band(vv, -18.0, size=16)  # 1 tile
        monkeypatch.setattr(Path, "rglob", lambda self, pattern: iter([nir, vv]))

        algorithm = load_algorithm("sar_threshold", {})
        assert estimate_tiles(input_path, algorithm, "sar_threshold", 512) == 1

    def test_tile_estimate_refuses_without_the_declared_raster(self, tmp_path):
        from cli.commands.analyze import estimate_tiles

        input_path = tmp_path / "input"
        _write_band(input_path / "scene" / "nir.tif", 0.3)

        algorithm = load_algorithm("sar_threshold", {})
        with pytest.raises(FileNotFoundError):
            estimate_tiles(input_path, algorithm, "sar_threshold", 512)

    def test_token_match_does_not_take_a_substring(self):
        from cli.commands.analyze import select_input_raster

        algorithm = load_algorithm("sar_threshold", {})
        with pytest.raises(FileNotFoundError):
            select_input_raster([Path("navvy.tif")], algorithm, "sar_threshold")
        assert select_input_raster(
            [Path("S1A_IW_VV_db.tif")], algorithm, "sar_threshold"
        ) == Path("S1A_IW_VV_db.tif")


class TestNoMockFallbacks:
    """Verify that no mock/fallback code exists."""

    def test_no_mock_algorithm_class(self):
        """Verify MockAlgorithm class does not exist."""
        from cli.commands import analyze

        assert not hasattr(analyze, "MockAlgorithm")

    def test_no_random_output_in_module(self):
        """Verify no random.randint usage in analyze module."""
        import inspect
        from cli.commands import analyze

        source = inspect.getsource(analyze)

        # Check that random.randint is not in the source
        assert "random.randint" not in source
        assert "mock_result" not in source


class TestAlgorithmEventTypeMapping:
    """Test algorithm selection by event type."""

    def test_flood_algorithms(self):
        """Test that flood algorithms are correctly mapped."""
        flood_algos = [
            name for name, info in ALGORITHMS.items()
            if "flood" in info["event_types"]
        ]

        assert "sar_threshold" in flood_algos
        assert "ndwi" in flood_algos
        assert "change_detection" in flood_algos
        assert "hand_model" in flood_algos

    def test_wildfire_algorithms(self):
        """Test that wildfire algorithms are correctly mapped."""
        wildfire_algos = [
            name for name, info in ALGORITHMS.items()
            if "wildfire" in info["event_types"]
        ]

        assert "dnbr" in wildfire_algos
        assert "thermal_anomaly" in wildfire_algos

    def test_storm_algorithms(self):
        """Test that storm algorithms are correctly mapped."""
        storm_algos = [
            name for name, info in ALGORITHMS.items()
            if "storm" in info["event_types"]
        ]

        assert "wind_damage" in storm_algos
        assert "structural_damage" in storm_algos


class TestParameterPassing:
    """Test parameter validation and passing to algorithms."""

    def test_invalid_threshold_raises_error(self):
        """Test that invalid parameters raise errors."""
        params = {
            "threshold_db": -50.0,  # Outside valid range [-20, -10]
        }

        with pytest.raises((ValueError, RuntimeError)) as excinfo:
            load_algorithm("sar_threshold", params)

        # Should get validation error from config
        assert "threshold_db" in str(excinfo.value) or "Failed to initialize" in str(excinfo.value)

    def test_valid_parameters_accepted(self):
        """Test that valid parameters are accepted."""
        params = {
            "threshold_db": -15.0,
            "min_area_ha": 0.5,
            "polarization": "VV",
        }

        algorithm = load_algorithm("sar_threshold", params)
        assert algorithm.config.threshold_db == -15.0
        assert algorithm.config.min_area_ha == 0.5
        assert algorithm.config.polarization == "VV"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
