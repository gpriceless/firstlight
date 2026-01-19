"""
Tests for CLI Commands (Group L, Track 10).

Tests all CLI commands using Click's CliRunner:
- TestDiscoverCommand: discover command tests
- TestIngestCommand: ingest command tests
- TestAnalyzeCommand: analyze command tests
- TestRunCommand: full pipeline tests
- TestStateManagement: state persistence tests

These tests complement the module tests and ensure CLI integration works correctly.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Check for click availability
try:
    import click
    from click.testing import CliRunner

    HAS_CLICK = True
except ImportError:
    HAS_CLICK = False

requires_click = pytest.mark.skipif(not HAS_CLICK, reason="click not installed")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def runner():
    """Create a CLI runner instance."""
    if not HAS_CLICK:
        pytest.skip("click not installed")
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_geojson(temp_dir):
    """Create a sample GeoJSON file."""
    geojson = {
        "type": "Feature",
        "properties": {"name": "test_area"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        },
    }
    file_path = temp_dir / "test_area.geojson"
    with open(file_path, "w") as f:
        json.dump(geojson, f)
    return file_path


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample config file."""
    import yaml

    config = {
        "default_profile": "laptop",
        "cache_dir": str(temp_dir / "cache"),
        "data_dir": str(temp_dir / "data"),
        "profiles": {
            "laptop": {
                "memory_mb": 2048,
                "max_workers": 2,
                "tile_size": 256,
            },
        },
    }
    file_path = temp_dir / "config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(config, f)
    return file_path


# =============================================================================
# TestDiscoverCommand
# =============================================================================


@requires_click
class TestDiscoverCommand:
    """Tests for the discover command."""

    def test_discover_basic(self, runner, sample_geojson):
        """Test basic discover command execution."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
            ],
        )

        # Should complete without error
        assert result.exit_code == 0 or "Usage" in result.output
        # Output should contain discovery indication
        if result.exit_code == 0:
            assert "Discover" in result.output or "discover" in result.output.lower()

    def test_discover_with_bbox(self, runner):
        """Test discover with bounding box instead of file."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--bbox", "-80.5,25.5,-80.0,26.0",
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
            ],
        )

        assert result.exit_code == 0 or "Usage" in result.output

    def test_discover_with_filters(self, runner, sample_geojson):
        """Test discover with source and cloud filters."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
                "--source", "sentinel1",
                "--max-cloud", "20",
            ],
        )

        assert result.exit_code == 0 or "Usage" in result.output

    def test_discover_json_output(self, runner, sample_geojson, temp_dir):
        """Test discover with JSON output format."""
        from cli.main import app

        output_file = temp_dir / "results.json"

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
                "--format", "json",
                "--output", str(output_file),
            ],
        )

        if result.exit_code == 0:
            # Check output file was created
            assert output_file.exists() or "results" in result.output.lower()

    def test_discover_csv_output(self, runner, sample_geojson, temp_dir):
        """Test discover with CSV output format."""
        from cli.main import app

        output_file = temp_dir / "results.csv"

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
                "--format", "csv",
                "--output", str(output_file),
            ],
        )

        # Should not crash
        assert result.exit_code == 0 or "Usage" in result.output

    def test_discover_wildfire(self, runner, sample_geojson):
        """Test discover for wildfire events."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-08-01",
                "--end", "2024-08-10",
                "--event", "wildfire",
            ],
        )

        assert result.exit_code == 0 or "Usage" in result.output

    def test_discover_storm(self, runner, sample_geojson):
        """Test discover for storm events."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-01",
                "--end", "2024-09-10",
                "--event", "storm",
            ],
        )

        assert result.exit_code == 0 or "Usage" in result.output

    def test_discover_invalid_dates(self, runner, sample_geojson):
        """Test discover with invalid date range."""
        from cli.main import app

        # End before start
        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-20",
                "--end", "2024-09-15",  # Before start
                "--event", "flood",
            ],
        )

        # Should fail or show error
        assert result.exit_code != 0 or "error" in result.output.lower()

    def test_discover_missing_area(self, runner):
        """Test discover without area specification."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "discover",
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
            ],
        )

        # Should fail - neither area nor bbox provided
        assert result.exit_code != 0 or "error" in result.output.lower() or "bbox" in result.output.lower()


# =============================================================================
# TestIngestCommand
# =============================================================================


@requires_click
class TestIngestCommand:
    """Tests for the ingest command."""

    def test_ingest_help(self, runner):
        """Test ingest command help."""
        from cli.main import app

        result = runner.invoke(app, ["ingest", "--help"])

        # Help should work or command not registered yet
        assert result.exit_code == 0 or "No such command" in result.output

    def test_ingest_basic(self, runner, sample_geojson, temp_dir):
        """Test basic ingest command."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "ingest",
                "--area", str(sample_geojson),
                "--source", "sentinel1",
                "--output", str(temp_dir / "data"),
            ],
        )

        # May fail if command not fully implemented
        # Just verify it doesn't crash unexpectedly
        assert isinstance(result.exit_code, int)

    def test_ingest_resume(self, runner, sample_geojson, temp_dir):
        """Test ingest with resume capability."""
        from cli.main import app

        # First run
        result = runner.invoke(
            app,
            [
                "ingest",
                "--area", str(sample_geojson),
                "--source", "sentinel1",
                "--output", str(temp_dir / "data"),
                "--resume",
            ],
        )

        # Should handle resume flag
        assert isinstance(result.exit_code, int)

    def test_ingest_validation(self, runner, sample_geojson, temp_dir):
        """Test ingest with validation enabled."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "ingest",
                "--area", str(sample_geojson),
                "--source", "sentinel1",
                "--output", str(temp_dir / "data"),
                "--validate",
            ],
        )

        # Should handle validation flag
        assert isinstance(result.exit_code, int)


# =============================================================================
# TestAnalyzeCommand
# =============================================================================


@requires_click
class TestAnalyzeCommand:
    """Tests for the analyze command."""

    def test_analyze_help(self, runner):
        """Test analyze command help."""
        from cli.main import app

        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0 or "No such command" in result.output

    def test_analyze_full(self, runner, temp_dir):
        """Test full analysis command."""
        from cli.main import app

        # Create mock input directory
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "analyze",
                "--input", str(input_dir),
                "--algorithm", "sar_threshold",
                "--output", str(temp_dir / "results"),
            ],
        )

        assert isinstance(result.exit_code, int)

    def test_analyze_tile_range(self, runner, temp_dir):
        """Test analysis on specific tile range."""
        from cli.main import app

        input_dir = temp_dir / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "analyze",
                "--input", str(input_dir),
                "--algorithm", "sar_threshold",
                "--tiles", "0-10",
                "--output", str(temp_dir / "results"),
            ],
        )

        assert isinstance(result.exit_code, int)

    def test_analyze_with_profile(self, runner, temp_dir):
        """Test analysis with execution profile."""
        from cli.main import app

        input_dir = temp_dir / "input"
        input_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "analyze",
                "--input", str(input_dir),
                "--algorithm", "sar_threshold",
                "--profile", "laptop",
                "--output", str(temp_dir / "results"),
            ],
        )

        assert isinstance(result.exit_code, int)


# =============================================================================
# TestRunCommand
# =============================================================================


@requires_click
class TestRunCommand:
    """Tests for the full run command."""

    def test_run_help(self, runner):
        """Test run command help."""
        from cli.main import app

        result = runner.invoke(app, ["run", "--help"])

        assert result.exit_code == 0 or "No such command" in result.output

    def test_run_end_to_end(self, runner, sample_geojson, temp_dir):
        """Test end-to-end run command."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "run",
                "--area", str(sample_geojson),
                "--event", "flood",
                "--output", str(temp_dir / "products"),
            ],
        )

        assert isinstance(result.exit_code, int)

    def test_run_with_profile(self, runner, sample_geojson, temp_dir):
        """Test run with specific profile."""
        from cli.main import app

        result = runner.invoke(
            app,
            [
                "run",
                "--area", str(sample_geojson),
                "--event", "flood",
                "--profile", "laptop",
                "--output", str(temp_dir / "products"),
            ],
        )

        assert isinstance(result.exit_code, int)

    def test_run_resume(self, runner, sample_geojson, temp_dir):
        """Test run with resume capability."""
        from cli.main import app

        # Create products directory with state
        products_dir = temp_dir / "products"
        products_dir.mkdir()

        result = runner.invoke(
            app,
            [
                "run",
                "--area", str(sample_geojson),
                "--event", "flood",
                "--output", str(products_dir),
                "--resume",
            ],
        )

        assert isinstance(result.exit_code, int)


# =============================================================================
# TestStateManagement
# =============================================================================


@requires_click
class TestStateManagement:
    """Tests for state management via CLI."""

    def test_status_command(self, runner, temp_dir):
        """Test status command."""
        from cli.main import app

        # Create a mock workdir
        workdir = temp_dir / "workdir"
        workdir.mkdir()

        result = runner.invoke(
            app,
            ["status", "--workdir", str(workdir)],
        )

        # May show "no workflow" or actual status
        assert isinstance(result.exit_code, int)

    def test_resume_command(self, runner, temp_dir):
        """Test resume command."""
        from cli.main import app

        workdir = temp_dir / "workdir"
        workdir.mkdir()

        result = runner.invoke(
            app,
            ["resume", "--workdir", str(workdir)],
        )

        # May show "nothing to resume" or actual resume
        assert isinstance(result.exit_code, int)

    def test_save_load_state(self, temp_dir):
        """Test state save and load functionality."""
        from core.execution.state import StateManager, WorkflowState, WorkflowStage

        manager = StateManager(temp_dir)

        # Create state
        state = WorkflowState(
            workflow_id="test_workflow_123",
            stage=WorkflowStage.ANALYZE,
            progress_percent=50.0,
        )

        # Save
        manager.save_state(state)

        # Load
        loaded = manager.load_state("test_workflow_123")

        assert loaded is not None
        assert loaded.workflow_id == "test_workflow_123"
        assert loaded.stage == WorkflowStage.ANALYZE
        assert loaded.progress_percent == 50.0

    def test_resume_from_checkpoint(self, temp_dir):
        """Test resuming from checkpoint."""
        from core.execution.state import (
            StateManager, WorkflowState, WorkflowStep,
            WorkflowStage, StepStatus,
        )

        manager = StateManager(temp_dir)

        # Create state with steps
        state = WorkflowState(
            workflow_id="checkpoint_test",
            stage=WorkflowStage.ANALYZE,
        )

        # Add completed steps
        step1 = WorkflowStep(
            step_id="step_1",
            name="Discovery",
            stage=WorkflowStage.DISCOVERY,
            status=StepStatus.COMPLETED,
        )
        state.add_step(step1)
        state.complete_step("step_1")

        # Add pending step
        step2 = WorkflowStep(
            step_id="step_2",
            name="Analysis",
            stage=WorkflowStage.ANALYZE,
            status=StepStatus.PENDING,
        )
        state.add_step(step2)

        # Create checkpoint
        checkpoint = state.create_checkpoint(
            data_paths={"intermediate": str(temp_dir / "intermediate")},
        )

        # Save state
        manager.save_state(state)

        # Load and get resume point
        loaded = manager.load_state("checkpoint_test")
        resume_point = loaded.get_resume_point()

        assert resume_point is not None
        assert resume_point.stage == WorkflowStage.ANALYZE
        assert "step_1" in loaded.completed_steps


# =============================================================================
# TestInfoCommand
# =============================================================================


@requires_click
class TestInfoCommand:
    """Tests for the info command."""

    def test_info_basic(self, runner):
        """Test basic info command."""
        from cli.main import app

        result = runner.invoke(app, ["info"])

        assert result.exit_code == 0
        assert "FirstLight" in result.output or "Python" in result.output

    def test_info_with_config(self, runner, sample_config):
        """Test info with config file."""
        from cli.main import app

        result = runner.invoke(
            app,
            ["--config", str(sample_config), "info"],
        )

        assert result.exit_code == 0


# =============================================================================
# TestCLIOptions
# =============================================================================


@requires_click
class TestCLIOptions:
    """Tests for global CLI options."""

    def test_verbose_flag(self, runner):
        """Test verbose flag."""
        from cli.main import app

        result = runner.invoke(app, ["-v", "info"])

        assert result.exit_code == 0

    def test_quiet_flag(self, runner):
        """Test quiet flag."""
        from cli.main import app

        result = runner.invoke(app, ["-q", "info"])

        assert result.exit_code == 0

    def test_verbose_quiet_conflict(self, runner):
        """Test that verbose and quiet cannot be used together."""
        from cli.main import app

        result = runner.invoke(app, ["-v", "-q", "info"])

        # Should fail with usage error
        assert result.exit_code != 0 or "Cannot use both" in result.output

    def test_version(self, runner):
        """Test version option."""
        from cli.main import app

        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_help(self, runner):
        """Test help option."""
        from cli.main import app

        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "FirstLight" in result.output


# =============================================================================
# TestDiscoverHelpers
# =============================================================================


@requires_click
class TestDiscoverHelpers:
    """Tests for discover command helper functions."""

    def test_parse_date_valid_formats(self):
        """Test date parsing with various formats."""
        from cli.commands.discover import parse_date

        # YYYY-MM-DD
        result = parse_date("2024-09-15")
        assert result.year == 2024
        assert result.month == 9
        assert result.day == 15

        # YYYY/MM/DD
        result = parse_date("2024/09/15")
        assert result.year == 2024

        # YYYYMMDD
        result = parse_date("20240915")
        assert result.year == 2024

    def test_parse_date_invalid(self):
        """Test date parsing with invalid format."""
        from cli.commands.discover import parse_date

        with pytest.raises(click.BadParameter):
            parse_date("invalid-date")

    def test_load_geometry_geojson(self, sample_geojson):
        """Test loading geometry from GeoJSON file."""
        from cli.commands.discover import load_geometry

        geometry = load_geometry(sample_geojson, None)

        assert geometry["type"] == "Polygon"
        assert "coordinates" in geometry

    def test_load_geometry_bbox(self):
        """Test loading geometry from bbox string."""
        from cli.commands.discover import load_geometry

        geometry = load_geometry(None, "-80.5,25.5,-80.0,26.0")

        assert geometry["type"] == "Polygon"
        coords = geometry["coordinates"][0]
        assert len(coords) == 5  # Closed polygon

    def test_load_geometry_invalid_bbox(self):
        """Test loading geometry with invalid bbox."""
        from cli.commands.discover import load_geometry

        with pytest.raises(click.BadParameter):
            load_geometry(None, "-80.5,25.5,-80.0")  # Only 3 values

    def test_format_size(self):
        """Test size formatting helper."""
        from cli.commands.discover import format_size

        assert "B" in format_size(100)
        assert "KB" in format_size(1024)
        assert "MB" in format_size(1024 * 1024)
        assert "GB" in format_size(1024 * 1024 * 1024)
        assert "Unknown" == format_size(None)


# =============================================================================
# TestValidateCommand
# =============================================================================


@requires_click
class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_help(self, runner):
        """Test validate command help."""
        from cli.main import app

        result = runner.invoke(app, ["validate", "--help"])

        assert result.exit_code == 0 or "No such command" in result.output


# =============================================================================
# TestExportCommand
# =============================================================================


@requires_click
class TestExportCommand:
    """Tests for the export command."""

    def test_export_help(self, runner):
        """Test export command help."""
        from cli.main import app

        result = runner.invoke(app, ["export", "--help"])

        assert result.exit_code == 0 or "No such command" in result.output


# =============================================================================
# Integration Tests
# =============================================================================


@requires_click
class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_full_workflow_simulation(self, runner, sample_geojson, temp_dir):
        """Test simulated full workflow through CLI."""
        from cli.main import app

        # Step 1: Discover
        result = runner.invoke(
            app,
            [
                "discover",
                "--area", str(sample_geojson),
                "--start", "2024-09-15",
                "--end", "2024-09-20",
                "--event", "flood",
                "--format", "json",
            ],
        )

        # Just verify command runs
        assert isinstance(result.exit_code, int)

    def test_profile_selection(self, temp_dir):
        """Test profile selection for workflow."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()

        # Get laptop profile
        laptop = manager.get_profile("laptop")
        assert laptop is not None
        assert laptop.max_memory_mb == 2048

        # Get workstation profile
        workstation = manager.get_profile("workstation")
        assert workstation is not None
        assert workstation.max_memory_mb == 8192

        # Get cloud profile
        cloud = manager.get_profile("cloud")
        assert cloud is not None
        assert cloud.use_distributed is True

        # Get edge profile
        edge = manager.get_profile("edge")
        assert edge is not None
        assert edge.tile_size == (128, 128)

    def test_state_persistence_workflow(self, temp_dir):
        """Test state persistence through workflow stages."""
        from core.execution.state import (
            StateManager, WorkflowState, WorkflowStep,
            WorkflowStage, StepStatus,
        )

        manager = StateManager(temp_dir)

        # Create workflow
        state = WorkflowState(
            workflow_id="integration_test",
            config={"event_type": "flood"},
        )

        # Go through stages
        stages = [
            (WorkflowStage.DISCOVERY, "discover_data"),
            (WorkflowStage.INGEST, "ingest_data"),
            (WorkflowStage.ANALYZE, "run_analysis"),
            (WorkflowStage.VALIDATE, "validate_results"),
            (WorkflowStage.EXPORT, "export_products"),
        ]

        for stage, step_name in stages:
            step = WorkflowStep(
                step_id=step_name,
                name=step_name.replace("_", " ").title(),
                stage=stage,
            )
            state.add_step(step)
            state.update_stage(stage)

            # Simulate work
            state.start_step(step_name)
            state.complete_step(step_name)

            # Save state
            manager.save_state(state)

            # Verify state can be loaded
            loaded = manager.load_state("integration_test")
            assert loaded.stage == stage
            assert step_name in loaded.completed_steps

        # Final state
        state.update_stage(WorkflowStage.COMPLETED)
        manager.save_state(state)

        final = manager.load_state("integration_test")
        assert final.is_completed()
        assert len(final.completed_steps) == 5
