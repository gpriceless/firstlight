"""
Tests for the ingestion tile grid (core/data/ingestion/normalization/tiling.py).

- TestTileGrid: tile grid generation tests
- TestTileBounds: tile bounds handling tests
- TestTiledRunner: process a raster tile by tile and stitch the result
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
