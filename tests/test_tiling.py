"""
Tests for Tiling Infrastructure and Execution Profiles (Group L, Track 10).

Tests tile grid generation, tiled execution, and execution profiles:
- TestTileGrid: tile grid generation tests
- TestTileBounds: tile bounds handling tests
- TestTiledRunner: tiled execution tests
- TestProfiles: execution profile tests
- TestStateManager: state persistence tests
- TestLocalStorage: local storage backend tests

These tests ensure the lightweight execution infrastructure works correctly
for resource-constrained environments.
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def sample_raster():
    """Create sample raster data."""
    np.random.seed(42)
    return np.random.rand(1000, 1000).astype(np.float32)


@pytest.fixture
def small_raster():
    """Create smaller raster for quick tests."""
    np.random.seed(42)
    return np.random.rand(100, 100).astype(np.float32)


# =============================================================================
# TestTileGrid
# =============================================================================


class TestTileGrid:
    """Tests for tile grid generation."""

    def test_grid_generation_basic(self):
        """Test basic tile grid generation."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme,
        )

        config = TileGridConfig(
            tile_size=(256, 256),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, 1000, 1000),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)

        # Generate grid for 1000x1000 area
        tiles = list(grid.iterate_tiles())

        # Should have 4x4 = 16 tiles
        assert len(tiles) == 16

    def test_grid_generation_with_overlap(self):
        """Test tile grid generation with overlap."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme,
        )

        config = TileGridConfig(
            tile_size=(256, 256),
            overlap=32,
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, 1000, 1000),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiles = list(grid.iterate_tiles())

        # Same number of tiles, but each has overlap bounds
        assert len(tiles) == 16
        for tile in tiles:
            assert tile.overlap_bounds is not None

    def test_grid_generation_non_divisible(self):
        """Test grid generation when dimensions don't divide evenly."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme,
        )

        config = TileGridConfig(
            tile_size=(256, 256),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, 500, 500),  # Doesn't divide evenly
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiles = list(grid.iterate_tiles())

        # Should cover entire area with 2x2 tiles
        assert len(tiles) == 4

    def test_tile_bounds_calculation(self):
        """Test tile bounds are calculated correctly."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme, TileIndex,
        )

        config = TileGridConfig(
            tile_size=(256, 256),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, 512, 512),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiles = list(grid.iterate_tiles())

        # First tile should be at index (0, 0)
        first_tile = tiles[0]
        assert first_tile.index.x == 0
        assert first_tile.index.y == 0

        # Last tile should be at index (1, 1)
        last_tile = tiles[-1]
        assert last_tile.index.x == 1
        assert last_tile.index.y == 1


class TestTileBounds:
    """Tests for tile bounds handling."""

    def test_tile_bounds_intersection(self):
        """Test tile bounds intersection calculation."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds1 = TileBounds(minx=0, miny=0, maxx=100, maxy=100)
        bounds2 = TileBounds(minx=50, miny=50, maxx=150, maxy=150)

        # Should intersect
        assert bounds1.intersects(bounds2)

        # Get intersection
        intersection = bounds1.intersection(bounds2)
        assert intersection is not None
        assert intersection.minx == 50
        assert intersection.miny == 50
        assert intersection.maxx == 100
        assert intersection.maxy == 100

    def test_tile_bounds_no_intersection(self):
        """Test non-intersecting tile bounds."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds1 = TileBounds(minx=0, miny=0, maxx=100, maxy=100)
        bounds2 = TileBounds(minx=200, miny=200, maxx=300, maxy=300)

        # Should not intersect
        assert not bounds1.intersects(bounds2)
        assert bounds1.intersection(bounds2) is None

    def test_tile_bounds_properties(self):
        """Test tile bounds properties."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(minx=10, miny=20, maxx=110, maxy=220)

        assert bounds.width == 100
        assert bounds.height == 200
        assert bounds.center == (60, 120)
        assert bounds.as_tuple() == (10, 20, 110, 220)

    def test_overlap_handling(self):
        """Test overlap between tiles is handled correctly."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme,
        )

        config = TileGridConfig(
            tile_size=(100, 100),
            overlap=10,
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, 200, 100),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiles = list(grid.iterate_tiles())

        # Two tiles horizontally
        assert len(tiles) == 2

        # Each tile should have overlap bounds
        tile0 = tiles[0]
        tile1 = tiles[1]

        # Verify overlap bounds exist
        assert tile0.overlap_bounds is not None
        assert tile1.overlap_bounds is not None

        # Overlap bounds should be larger than regular bounds
        assert tile0.overlap_bounds.width > tile0.bounds.width


# =============================================================================
# TestTiledRunner (Conceptual - uses tiling infrastructure)
# =============================================================================


class TestTiledRunner:
    """Tests for tiled execution."""

    def test_tiled_execution_basic(self, small_raster, temp_dir):
        """Test basic tiled execution."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme, TileBounds, RasterTiler,
        )

        tile_size = 32
        config = TileGridConfig(
            tile_size=(tile_size, tile_size),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, small_raster.shape[1], small_raster.shape[0]),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiler = RasterTiler(grid)
        data_bounds = TileBounds(0, 0, small_raster.shape[1], small_raster.shape[0])

        # Generate tiles
        tiles = list(grid.iterate_tiles())

        # Process each tile
        results = []
        for tile in tiles:
            # Extract tile data using tiler
            tile_data = tiler.extract_tile(small_raster, data_bounds, tile)

            # Simple processing: compute mean
            result = {
                "tile_id": f"{tile.index.x}_{tile.index.y}",
                "mean": float(np.mean(tile_data)),
                "shape": tile_data.shape,
            }
            results.append(result)

        # Should have processed all tiles
        assert len(results) == len(tiles)
        assert all("mean" in r for r in results)

    def test_result_stitching(self, small_raster, temp_dir):
        """Test stitching results from multiple tiles."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme, TileBounds, RasterTiler,
        )

        tile_size = 32
        config = TileGridConfig(
            tile_size=(tile_size, tile_size),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, small_raster.shape[1], small_raster.shape[0]),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiler = RasterTiler(grid)
        data_bounds = TileBounds(0, 0, small_raster.shape[1], small_raster.shape[0])

        # Process tiles
        processed_tiles = []
        for tile in grid.iterate_tiles():
            tile_data = tiler.extract_tile(small_raster, data_bounds, tile)
            # Simple transform
            tile.data = tile_data * 2.0
            processed_tiles.append(tile)

        # Stitch results
        output = tiler.stitch_tiles(processed_tiles, small_raster.shape)

        # Verify stitching
        expected = small_raster * 2.0
        np.testing.assert_array_almost_equal(output, expected, decimal=5)


# =============================================================================
# TestProfiles
# =============================================================================


class TestProfiles:
    """Tests for execution profiles."""

    def test_profile_loading(self):
        """Test loading predefined profiles."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()

        # Check all predefined profiles exist
        profiles = manager.list_profiles()
        assert "laptop" in profiles
        assert "workstation" in profiles
        assert "cloud" in profiles
        assert "edge" in profiles

    def test_profile_parameters(self):
        """Test profile parameters are correct."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()

        # Laptop profile
        laptop = manager.get_profile("laptop")
        assert laptop is not None
        assert laptop.max_memory_mb == 2048
        assert laptop.max_concurrent_tiles == 1
        assert laptop.tile_size == (256, 256)
        assert laptop.use_distributed is False

        # Workstation profile
        workstation = manager.get_profile("workstation")
        assert workstation is not None
        assert workstation.max_memory_mb == 8192
        assert workstation.max_concurrent_tiles == 4
        assert workstation.tile_size == (512, 512)

        # Cloud profile
        cloud = manager.get_profile("cloud")
        assert cloud is not None
        assert cloud.max_memory_mb == 32768
        assert cloud.use_distributed is True
        assert cloud.tile_size == (1024, 1024)

        # Edge profile
        edge = manager.get_profile("edge")
        assert edge is not None
        assert edge.max_memory_mb == 1024
        assert edge.tile_size == (128, 128)

    def test_resource_detection(self):
        """Test system resource detection."""
        from core.execution.profiles import SystemResources

        resources = SystemResources.detect()

        # Should have valid values
        assert resources.total_memory_mb > 0
        assert resources.available_memory_mb > 0
        assert resources.cpu_cores >= 1
        assert resources.platform_name != ""

    def test_auto_select_profile(self):
        """Test automatic profile selection."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()
        profile = manager.auto_select_profile()

        # Should return a valid profile
        assert profile is not None
        assert profile.name in ["laptop", "workstation", "cloud", "edge"]

    def test_custom_profile_creation(self):
        """Test creating custom profiles."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()

        profile = manager.create_custom_profile(
            name="custom_test",
            max_memory_mb=4096,
            max_concurrent_tiles=2,
            tile_size=(384, 384),
            timeout_seconds=600,
            description="Test custom profile",
        )

        assert profile.name == "custom_test"
        assert profile.max_memory_mb == 4096
        assert profile.tile_size == (384, 384)

        # Should be retrievable
        retrieved = manager.get_profile("custom_test")
        assert retrieved is not None
        assert retrieved.max_memory_mb == 4096

    def test_profile_validation(self):
        """Test profile validation against system."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()

        # Get a valid profile
        profile = manager.auto_select_profile()

        # Validate
        is_valid, issues = manager.validate_profile(profile)

        # Auto-selected profile should be valid
        # (may have warnings but not be invalid)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)

    def test_profile_effective_memory(self):
        """Test effective memory calculation."""
        from core.execution.profiles import ExecutionProfile

        profile = ExecutionProfile(
            name="test",
            max_memory_mb=1000,
            max_concurrent_tiles=4,
            tile_size=(256, 256),
            use_distributed=False,
            timeout_seconds=300,
            memory_buffer_factor=0.2,
        )

        # Effective = 1000 * (1 - 0.2) = 800
        assert profile.effective_memory_mb == 800

        # Per tile = 800 / 4 = 200
        assert profile.memory_per_tile_mb == 200

    def test_profile_tile_estimation(self):
        """Test tile count estimation."""
        from core.execution.profiles import ExecutionProfile

        profile = ExecutionProfile(
            name="test",
            max_memory_mb=4096,
            max_concurrent_tiles=4,
            tile_size=(256, 256),
            use_distributed=False,
            timeout_seconds=300,
        )

        # 1000x1000 with 256x256 tiles, no overlap
        tile_count = profile.estimate_total_tiles(1000, 1000, overlap=0)
        assert tile_count == 16  # 4x4

        # With overlap
        tile_count = profile.estimate_total_tiles(1000, 1000, overlap=32)
        assert tile_count > 16  # More tiles needed

    def test_processing_time_estimate(self):
        """Test processing time estimation."""
        from core.execution.profiles import ProfileManager

        manager = ProfileManager()
        laptop = manager.get_profile("laptop")

        min_time, max_time = manager.estimate_processing_time(
            profile=laptop,
            tile_count=16,
            time_per_tile_seconds=10.0,
        )

        # Laptop processes sequentially
        assert min_time >= 160.0  # 16 tiles * 10 seconds
        assert max_time > min_time


class TestProfileEdgeCases:
    """Edge case tests for execution profiles."""

    def test_profile_min_memory(self):
        """Test profile with minimum memory."""
        from core.execution.profiles import ExecutionProfile

        # Should work with 256 MB minimum
        profile = ExecutionProfile(
            name="minimal",
            max_memory_mb=256,
            max_concurrent_tiles=1,
            tile_size=(64, 64),
            use_distributed=False,
            timeout_seconds=60,
        )

        assert profile.max_memory_mb == 256

    def test_profile_invalid_memory(self):
        """Test profile with invalid memory raises error."""
        from core.execution.profiles import ExecutionProfile

        with pytest.raises(ValueError):
            ExecutionProfile(
                name="invalid",
                max_memory_mb=100,  # Below 256 minimum
                max_concurrent_tiles=1,
                tile_size=(256, 256),
                use_distributed=False,
                timeout_seconds=300,
            )

    def test_profile_invalid_tile_size(self):
        """Test profile with invalid tile size."""
        from core.execution.profiles import ExecutionProfile

        with pytest.raises(ValueError):
            ExecutionProfile(
                name="invalid",
                max_memory_mb=2048,
                max_concurrent_tiles=1,
                tile_size=(16, 16),  # Below 32 minimum
                use_distributed=False,
                timeout_seconds=300,
            )

    def test_profile_serialization(self):
        """Test profile serialization and deserialization."""
        from core.execution.profiles import ExecutionProfile

        original = ExecutionProfile(
            name="serialization_test",
            max_memory_mb=4096,
            max_concurrent_tiles=4,
            tile_size=(512, 512),
            use_distributed=False,
            timeout_seconds=600,
            description="Test profile",
            gpu_enabled=True,
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = ExecutionProfile.from_dict(data)

        assert restored.name == original.name
        assert restored.max_memory_mb == original.max_memory_mb
        assert restored.tile_size == original.tile_size
        assert restored.gpu_enabled == original.gpu_enabled


# =============================================================================
# TestStateManager
# =============================================================================


class TestStateManager:
    """Tests for state persistence."""

    def test_create_workflow_state(self, temp_dir):
        """Test creating workflow state."""
        from core.execution.state import StateManager, WorkflowState, WorkflowStage

        manager = StateManager(temp_dir)

        state = WorkflowState(
            workflow_id="test_workflow",
            stage=WorkflowStage.INITIALIZED,
        )

        manager.save_state(state)

        # Verify file was created
        loaded = manager.load_state("test_workflow")
        assert loaded is not None
        assert loaded.workflow_id == "test_workflow"

    def test_state_stage_progression(self, temp_dir):
        """Test state progression through stages."""
        from core.execution.state import StateManager, WorkflowState, WorkflowStage

        manager = StateManager(temp_dir)

        state = WorkflowState(workflow_id="progress_test")

        stages = [
            WorkflowStage.DISCOVERY,
            WorkflowStage.INGEST,
            WorkflowStage.ANALYZE,
            WorkflowStage.VALIDATE,
            WorkflowStage.EXPORT,
            WorkflowStage.COMPLETED,
        ]

        for stage in stages:
            state.update_stage(stage)
            manager.save_state(state)

            loaded = manager.load_state("progress_test")
            assert loaded.stage == stage

    def test_step_management(self, temp_dir):
        """Test step management in workflow."""
        from core.execution.state import (
            StateManager, WorkflowState, WorkflowStep,
            WorkflowStage, StepStatus,
        )

        manager = StateManager(temp_dir)

        state = WorkflowState(workflow_id="step_test")

        # Add step
        step = WorkflowStep(
            step_id="step_1",
            name="First Step",
            stage=WorkflowStage.DISCOVERY,
        )
        state.add_step(step)

        # Start step
        state.start_step("step_1")
        manager.save_state(state)

        loaded = manager.load_state("step_test")
        assert loaded.steps["step_1"].status == StepStatus.RUNNING

        # Complete step
        state.complete_step("step_1", output_path="/tmp/output")
        manager.save_state(state)

        loaded = manager.load_state("step_test")
        assert loaded.steps["step_1"].status == StepStatus.COMPLETED
        assert "step_1" in loaded.completed_steps

    def test_checkpoint_creation(self, temp_dir):
        """Test checkpoint creation and retrieval."""
        from core.execution.state import StateManager, WorkflowState, WorkflowStage

        manager = StateManager(temp_dir)

        state = WorkflowState(
            workflow_id="checkpoint_test",
            stage=WorkflowStage.ANALYZE,
        )

        # Create checkpoint
        checkpoint = state.create_checkpoint(
            data_paths={"intermediate": "/tmp/data"},
        )

        assert checkpoint is not None
        assert checkpoint.stage == WorkflowStage.ANALYZE

        manager.save_state(state)

        # Get resume point
        loaded = manager.load_state("checkpoint_test")
        resume = loaded.get_resume_point()

        assert resume is not None
        assert resume.checkpoint_id == checkpoint.checkpoint_id

    def test_sqlite_backend(self, temp_dir):
        """Test SQLite storage backend."""
        from core.execution.state import StateManager, WorkflowState, WorkflowStage

        manager = StateManager(temp_dir, use_sqlite=True)

        state = WorkflowState(
            workflow_id="sqlite_test",
            stage=WorkflowStage.INGEST,
        )

        manager.save_state(state)

        # Verify can load
        loaded = manager.load_state("sqlite_test")
        assert loaded is not None
        assert loaded.workflow_id == "sqlite_test"

    def test_state_cleanup(self, temp_dir):
        """Test state cleanup."""
        from core.execution.state import StateManager, WorkflowState

        manager = StateManager(temp_dir)

        # Create multiple states
        for i in range(5):
            state = WorkflowState(workflow_id=f"cleanup_test_{i}")
            manager.save_state(state)

        # List workflows
        workflows = manager.list_workflows()
        assert len(workflows) >= 5

        # Delete one
        deleted = manager.delete_state("cleanup_test_0")
        assert deleted

        # Verify deleted
        loaded = manager.load_state("cleanup_test_0")
        assert loaded is None


# =============================================================================
# TestLocalStorage
# =============================================================================


class TestLocalStorage:
    """Tests for local storage backend."""

    def test_workspace_creation(self, temp_dir):
        """Test workspace creation with subdirectories."""
        from core.data.storage.local import LocalWorkspace, WorkspaceConfig

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="storage_test",
        )

        workspace = LocalWorkspace(config)

        # Verify directories exist
        assert workspace.raw_path.exists()
        assert workspace.normalized_path.exists()
        assert workspace.intermediate_path.exists()
        assert workspace.results_path.exists()
        assert workspace.products_path.exists()
        assert workspace.state_path.exists()

    def test_file_tracking(self, temp_dir):
        """Test file tracking in workspace."""
        from core.data.storage.local import LocalWorkspace, WorkspaceConfig, WorkspaceSubdir

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="tracking_test",
        )

        workspace = LocalWorkspace(config)

        # Create a test file
        test_file = workspace.raw_path / "test_data.bin"
        test_file.write_bytes(b"test data content")

        # Track file
        entry = workspace.add_file(test_file, WorkspaceSubdir.RAW)

        assert entry.size_bytes > 0
        assert entry.category == WorkspaceSubdir.RAW

        # Retrieve
        retrieved = workspace.get_file(test_file)
        assert retrieved is not None

    def test_disk_usage(self, temp_dir):
        """Test disk usage calculation."""
        from core.data.storage.local import LocalWorkspace, WorkspaceConfig

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="usage_test",
        )

        workspace = LocalWorkspace(config)

        # Create some files
        for i in range(5):
            test_file = workspace.intermediate_path / f"file_{i}.bin"
            test_file.write_bytes(b"x" * 1000)

        # Check usage
        usage = workspace.calculate_disk_usage()

        assert usage.workspace_bytes >= 5000
        assert usage.total_bytes > 0
        assert usage.free_bytes > 0

    def test_cleanup_temporary(self, temp_dir):
        """Test cleanup of temporary files."""
        from core.data.storage.local import (
            LocalWorkspace, WorkspaceConfig, WorkspaceSubdir,
        )

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="cleanup_test",
            preserve_raw=True,
            preserve_products=True,
        )

        workspace = LocalWorkspace(config)

        # Create files in different categories
        raw_file = workspace.raw_path / "raw.bin"
        raw_file.write_bytes(b"raw data")

        intermediate_file = workspace.intermediate_path / "temp.bin"
        intermediate_file.write_bytes(b"temp data")

        product_file = workspace.products_path / "product.bin"
        product_file.write_bytes(b"product data")

        # Cleanup
        freed = workspace.cleanup_temporary()

        # Intermediate should be cleaned
        assert not intermediate_file.exists()

        # Raw and products should be preserved
        assert raw_file.exists()
        assert product_file.exists()

    def test_space_estimation(self, temp_dir):
        """Test space estimation."""
        from core.data.storage.local import LocalWorkspace, WorkspaceConfig

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="estimation_test",
        )

        workspace = LocalWorkspace(config)

        # Estimate for 100MB input
        estimated = workspace.estimate_space_needed(
            input_size_mb=100.0,
            processing_factor=3.0,
        )

        # Should be approximately 100 * (1 + 1 + 3 + 0.5) = 550 MB
        assert estimated > 500
        assert estimated < 600

    def test_storage_manager(self, temp_dir):
        """Test storage manager operations."""
        from core.data.storage.local import StorageManager

        manager = StorageManager(
            base_storage_path=temp_dir / "storage",
            default_max_size_gb=10.0,
        )

        # Create workspace
        workspace = manager.create_workspace("test_workflow")
        assert workspace is not None

        # List workspaces
        workspaces = manager.list_workspaces()
        assert "test_workflow" in workspaces

        # Get workspace
        retrieved = manager.get_workspace("test_workflow")
        assert retrieved is not None

        # Delete workspace
        deleted = manager.delete_workspace("test_workflow", force=True)
        assert deleted

        # Verify deleted
        workspaces = manager.list_workspaces()
        assert "test_workflow" not in workspaces

    def test_integrity_verification(self, temp_dir):
        """Test file integrity verification."""
        from core.data.storage.local import LocalWorkspace, WorkspaceConfig, WorkspaceSubdir

        config = WorkspaceConfig(
            base_path=temp_dir / "workspace",
            workflow_id="integrity_test",
        )

        workspace = LocalWorkspace(config)

        # Create and track file with checksum
        test_file = workspace.raw_path / "data.bin"
        test_file.write_bytes(b"test content")

        workspace.add_file(test_file, WorkspaceSubdir.RAW, compute_checksum=True)

        # Verify integrity - should be clean
        issues = workspace.verify_integrity()
        assert len(issues) == 0

        # Modify file
        test_file.write_bytes(b"modified content")

        # Verify integrity - should detect issue
        issues = workspace.verify_integrity()
        assert len(issues) > 0
        assert "checksum" in issues[0].lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestTilingIntegration:
    """Integration tests for tiling with profiles and state."""

    def test_full_tiled_workflow(self, small_raster, temp_dir):
        """Test complete tiled workflow with state tracking."""
        from core.execution.profiles import ProfileManager
        from core.execution.state import (
            StateManager, WorkflowState, WorkflowStep,
            WorkflowStage, StepStatus,
        )
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme, TileBounds, RasterTiler,
        )

        # Get profile
        profile_mgr = ProfileManager()
        profile = profile_mgr.get_profile("laptop")

        # Setup state
        state_mgr = StateManager(temp_dir / "state")
        state = WorkflowState(
            workflow_id="tiled_workflow",
            config={"profile": profile.name},
        )

        # Setup tiling using profile
        tile_size = profile.tile_size[0]
        config = TileGridConfig(
            tile_size=(tile_size, tile_size),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, small_raster.shape[1], small_raster.shape[0]),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiler = RasterTiler(grid)
        data_bounds = TileBounds(0, 0, small_raster.shape[1], small_raster.shape[0])

        # Generate tiles
        tiles = list(grid.iterate_tiles())

        # Add step for each tile
        for i, tile in enumerate(tiles):
            step = WorkflowStep(
                step_id=f"tile_{i}",
                name=f"Process tile {tile.index.x},{tile.index.y}",
                stage=WorkflowStage.ANALYZE,
            )
            state.add_step(step)

        state_mgr.save_state(state)

        # Process tiles
        for i, tile in enumerate(tiles):
            step_id = f"tile_{i}"
            state.start_step(step_id)

            # Extract and process tile
            tile_data = tiler.extract_tile(small_raster, data_bounds, tile)
            result = np.mean(tile_data)

            state.complete_step(step_id, metadata={"mean": float(result)})
            state_mgr.save_state(state)

        # Verify completion
        state.update_stage(WorkflowStage.COMPLETED)
        state_mgr.save_state(state)

        final = state_mgr.load_state("tiled_workflow")
        assert final.is_completed()
        assert len(final.completed_steps) == len(tiles)

    def test_resume_tiled_workflow(self, small_raster, temp_dir):
        """Test resuming interrupted tiled workflow."""
        from core.execution.state import (
            StateManager, WorkflowState, WorkflowStep,
            WorkflowStage, StepStatus,
        )
        from core.data.ingestion.normalization.tiling import (
            TileGrid, TileGridConfig, TileScheme,
        )

        state_mgr = StateManager(temp_dir / "state")

        # Create initial state with tiles
        state = WorkflowState(workflow_id="resume_test")

        tile_size = 32
        config = TileGridConfig(
            tile_size=(tile_size, tile_size),
            scheme=TileScheme.CUSTOM,
            bounds=(0, 0, small_raster.shape[1], small_raster.shape[0]),
            resolution=(1.0, 1.0),
        )

        grid = TileGrid(config)
        tiles = list(grid.iterate_tiles())

        for i, tile in enumerate(tiles):
            step = WorkflowStep(
                step_id=f"tile_{i}",
                name=f"Tile {i}",
                stage=WorkflowStage.ANALYZE,
            )
            state.add_step(step)

        # Complete only half
        for i in range(len(tiles) // 2):
            state.start_step(f"tile_{i}")
            state.complete_step(f"tile_{i}")

        # Create checkpoint
        state.create_checkpoint()
        state_mgr.save_state(state)

        # Simulate restart - load state
        loaded = state_mgr.load_state("resume_test")
        assert loaded.is_resumable()

        # Get pending tiles (make a copy since we'll modify the list)
        pending = list(loaded.pending_steps)
        assert len(pending) == len(tiles) // 2

        # Complete remaining
        for step_id in pending:
            loaded.start_step(step_id)
            loaded.complete_step(step_id)

        loaded.update_stage(WorkflowStage.COMPLETED)
        state_mgr.save_state(loaded)

        # Verify all tiles are now complete
        final = state_mgr.load_state("resume_test")
        assert final.is_completed()
        assert len(final.completed_steps) == len(tiles)
