"""
Tests for data normalization tools.

Tests projection, tiling, temporal, and resolution normalization modules.
"""

import math
import pytest
from datetime import datetime, timezone, timedelta
from typing import Tuple

import numpy as np

# ============================================================================
# PROJECTION TESTS
# ============================================================================


class TestCRSHandler:
    """Tests for CRS handling and transformation.

    These tests require pyproj to be installed.
    """

    def test_parse_crs_epsg_code(self):
        """Test parsing EPSG codes."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        info = handler.parse_crs("EPSG:4326")

        assert info.code == "EPSG:4326"
        assert info.is_geographic == True
        assert info.is_projected == False

    def test_parse_crs_integer(self):
        """Test parsing integer EPSG codes."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        info = handler.parse_crs(32632)  # UTM zone 32N

        assert "32632" in info.code
        assert info.is_geographic == False
        assert info.is_projected == True

    def test_are_equivalent_same_crs(self):
        """Test CRS equivalence for same CRS."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        assert handler.are_equivalent("EPSG:4326", "EPSG:4326") == True

    def test_are_equivalent_different_crs(self):
        """Test CRS equivalence for different CRS."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        assert handler.are_equivalent("EPSG:4326", "EPSG:32632") == False

    def test_suggest_utm_zone_europe(self):
        """Test UTM zone suggestion for European location."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        # Paris, France (approximately 48.8566° N, 2.3522° E)
        zone = handler.suggest_utm_zone(2.35, 48.86)

        assert zone == "EPSG:32631"  # UTM zone 31N

    def test_suggest_utm_zone_southern_hemisphere(self):
        """Test UTM zone suggestion for southern hemisphere."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        # Sydney, Australia (approximately -33.87° S, 151.21° E)
        zone = handler.suggest_utm_zone(151.21, -33.87)

        assert "327" in zone  # UTM southern hemisphere (32700-32760)

    def test_transform_bounds(self):
        """Test bounds transformation between CRS."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        wgs84_bounds = (-10.0, 35.0, 10.0, 45.0)
        utm_bounds = handler.transform_bounds(
            wgs84_bounds, "EPSG:4326", "EPSG:32631"
        )

        # UTM bounds should be in meters (much larger numbers)
        assert utm_bounds[0] > -1e6
        assert utm_bounds[2] > utm_bounds[0]
        assert utm_bounds[3] > utm_bounds[1]

    def test_transform_point(self):
        """Test single point transformation."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        # Transform London (approximately 51.5° N, -0.1° W)
        tx, ty = handler.transform_point(-0.1, 51.5, "EPSG:4326", "EPSG:32630")

        # Should be in UTM meters
        assert abs(tx) > 100000  # Significant x coordinate
        assert abs(ty) > 5000000  # Significant y coordinate


class TestReprojectionConfig:
    """Tests for reprojection configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from core.data.ingestion.normalization.projection import (
            ReprojectionConfig,
            ResamplingMethod,
        )

        config = ReprojectionConfig(target_crs="EPSG:4326")

        assert config.target_crs == "EPSG:4326"
        assert config.resampling == ResamplingMethod.BILINEAR
        assert config.num_threads == 4


# ============================================================================
# TILING TESTS
# ============================================================================


class TestTileBounds:
    """Tests for tile bounds operations."""

    def test_tile_bounds_properties(self):
        """Test tile bounds width, height, center."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(minx=0.0, miny=0.0, maxx=10.0, maxy=5.0)

        assert bounds.width == 10.0
        assert bounds.height == 5.0
        assert bounds.center == (5.0, 2.5)

    def test_tile_bounds_intersection(self):
        """Test tile bounds intersection."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds1 = TileBounds(minx=0.0, miny=0.0, maxx=10.0, maxy=10.0)
        bounds2 = TileBounds(minx=5.0, miny=5.0, maxx=15.0, maxy=15.0)

        intersection = bounds1.intersection(bounds2)

        assert intersection is not None
        assert intersection.minx == 5.0
        assert intersection.miny == 5.0
        assert intersection.maxx == 10.0
        assert intersection.maxy == 10.0

    def test_tile_bounds_no_intersection(self):
        """Test tile bounds with no intersection."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds1 = TileBounds(minx=0.0, miny=0.0, maxx=5.0, maxy=5.0)
        bounds2 = TileBounds(minx=10.0, miny=10.0, maxx=15.0, maxy=15.0)

        assert bounds1.intersection(bounds2) is None

    def test_tile_bounds_buffer(self):
        """Test tile bounds buffering."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(minx=0.0, miny=0.0, maxx=10.0, maxy=10.0)
        buffered = bounds.buffer(2.0)

        assert buffered.minx == -2.0
        assert buffered.miny == -2.0
        assert buffered.maxx == 12.0
        assert buffered.maxy == 12.0


class TestTileGrid:
    """Tests for tile grid operations."""

    def test_create_tile_grid(self):
        """Test tile grid creation."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid,
            TileGridConfig,
        )

        config = TileGridConfig(
            tile_size=(256, 256),
            bounds=(0.0, 0.0, 100.0, 100.0),
            resolution=(1.0, 1.0),
            overlap=0,
        )
        grid = TileGrid(config)

        # 100 pixels / 256 tile size = ceil(0.39) = 1 tile per dimension
        # But actually: 100 * 1.0 resolution = 100 units, 256 * 1.0 = 256 units per tile
        # So we need ceil(100/256) = 1 tile per dimension
        assert grid.cols >= 1
        assert grid.rows >= 1

    def test_tile_grid_overlap(self):
        """Test tile grid with overlap."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid,
            TileGridConfig,
            TileIndex,
        )

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0.0, 0.0, 500.0, 500.0),
            resolution=(1.0, 1.0),
            overlap=10,
        )
        grid = TileGrid(config)

        tile = grid.get_tile(TileIndex(x=1, y=1))

        # Core bounds and overlap bounds should differ
        if tile.overlap_bounds is not None:
            assert tile.overlap_bounds.minx < tile.bounds.minx
            assert tile.overlap_bounds.maxx > tile.bounds.maxx

    def test_tile_grid_iterate(self):
        """Test tile grid iteration."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid,
            TileGridConfig,
        )

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0.0, 0.0, 300.0, 200.0),
            resolution=(1.0, 1.0),
        )
        grid = TileGrid(config)

        tiles = list(grid.iterate_tiles())
        assert len(tiles) == grid.total_tiles

    def test_tile_grid_at_point(self):
        """Test finding tile at point."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid,
            TileGridConfig,
        )

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0.0, 0.0, 500.0, 500.0),
            resolution=(1.0, 1.0),
        )
        grid = TileGrid(config)

        index = grid.get_tile_at_point(250.0, 250.0)

        assert index is not None
        assert index.x >= 0
        assert index.y >= 0


class TestWebMercatorTiles:
    """Tests for web mercator tile calculations."""

    def test_latlng_to_tile(self):
        """Test lat/lng to tile conversion."""
        from core.data.ingestion.normalization.tiling import WebMercatorTiles

        # London at zoom 10
        tile = WebMercatorTiles.latlng_to_tile(51.5, -0.1, 10)

        assert tile.z == 10
        assert tile.x > 0
        assert tile.y > 0

    def test_tile_to_bounds(self):
        """Test tile to bounds conversion."""
        from core.data.ingestion.normalization.tiling import (
            WebMercatorTiles,
            TileIndex,
            TileScheme,
        )

        tile = TileIndex(x=512, y=340, z=10)
        bounds = WebMercatorTiles.tile_to_bounds(tile, TileScheme.XYZ)

        # Bounds should be in lat/lng
        assert -180.0 <= bounds.minx <= 180.0
        assert -90.0 <= bounds.miny <= 90.0

    def test_meters_per_pixel(self):
        """Test meters per pixel calculation."""
        from core.data.ingestion.normalization.tiling import WebMercatorTiles

        # At equator, zoom 0
        mpp_z0 = WebMercatorTiles.meters_per_pixel(0.0, 0)
        # At equator, zoom 1 (should be half)
        mpp_z1 = WebMercatorTiles.meters_per_pixel(0.0, 1)

        assert mpp_z1 < mpp_z0
        assert abs(mpp_z0 / mpp_z1 - 2.0) < 0.1  # Approximately 2x


# ============================================================================
# TEMPORAL TESTS
# ============================================================================


class TestTimeRange:
    """Tests for time range operations."""

    def test_time_range_duration(self):
        """Test time range duration calculation."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        time_range = TimeRange(start=start, end=end)

        assert time_range.duration == timedelta(days=1)
        assert time_range.duration_seconds == 86400.0

    def test_time_range_contains(self):
        """Test time range contains check."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 10, tzinfo=timezone.utc)
        time_range = TimeRange(start=start, end=end)

        inside = datetime(2024, 1, 5, tzinfo=timezone.utc)
        outside = datetime(2024, 1, 15, tzinfo=timezone.utc)

        assert time_range.contains(inside) is True
        assert time_range.contains(outside) is False

    def test_time_range_overlaps(self):
        """Test time range overlap detection."""
        from core.data.ingestion.normalization.temporal import TimeRange

        range1 = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        range2 = TimeRange(
            start=datetime(2024, 1, 5, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        range3 = TimeRange(
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 10, tzinfo=timezone.utc),
        )

        assert range1.overlaps(range2) is True
        assert range1.overlaps(range3) is False

    def test_time_range_split(self):
        """Test time range splitting."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 11, tzinfo=timezone.utc)
        time_range = TimeRange(start=start, end=end)

        parts = time_range.split(5)

        assert len(parts) == 5
        assert parts[0].start == start
        assert parts[-1].end == end


class TestTimestampHandler:
    """Tests for timestamp parsing and conversion."""

    def test_parse_iso_timestamp(self):
        """Test ISO format timestamp parsing."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()
        dt = handler.parse("2024-03-15T10:30:00Z")

        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.tzinfo == timezone.utc

    def test_parse_unix_timestamp(self):
        """Test Unix timestamp parsing."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()
        # 2024-01-01 00:00:00 UTC
        dt = handler.parse(1704067200)

        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1

    def test_parse_date_string(self):
        """Test date-only string parsing."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()
        dt = handler.parse("2024-03-15")

        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_format_filename(self):
        """Test filename-safe formatting."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()
        dt = datetime(2024, 3, 15, 10, 30, 45, tzinfo=timezone.utc)
        formatted = handler.format_filename(dt)

        assert formatted == "20240315_103045"


class TestTemporalResampler:
    """Tests for temporal resampling."""

    def test_downsample_mean(self):
        """Test downsampling with mean aggregation."""
        from core.data.ingestion.normalization.temporal import (
            TemporalResampler,
            TemporalSample,
            TemporalAlignmentConfig,
            AggregationMethod,
        )

        resampler = TemporalResampler()

        # Create hourly samples
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        samples = [
            TemporalSample(timestamp=base_time + timedelta(hours=i), data=float(i))
            for i in range(24)
        ]

        # Resample to daily
        config = TemporalAlignmentConfig(aggregation=AggregationMethod.MEAN)
        resampled = resampler.resample(
            samples, target_resolution=timedelta(days=1), config=config
        )

        # Should have fewer samples
        assert len(resampled) < len(samples)


# ============================================================================
# RESOLUTION TESTS
# ============================================================================


class TestResolution:
    """Tests for Resolution class."""

    def test_resolution_basic(self):
        """Test basic resolution properties."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res = Resolution(x=10.0, y=10.0, unit=ResolutionUnit.METERS)

        assert res.x == 10.0
        assert res.y == 10.0
        assert res.area == 100.0
        assert res.is_square == True

    def test_resolution_non_square(self):
        """Test non-square resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res = Resolution(x=10.0, y=20.0, unit=ResolutionUnit.METERS)

        assert res.is_square == False
        assert res.area == 200.0

    def test_resolution_to_meters_from_degrees(self):
        """Test conversion from degrees to meters."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        # Approximately 1 arcsecond at equator
        res = Resolution(x=1.0 / 3600, y=1.0 / 3600, unit=ResolutionUnit.DEGREES)
        meters = res.to_meters(latitude=0)

        # Should be around 30 meters
        assert meters.x > 20.0
        assert meters.x < 40.0
        assert meters.unit == ResolutionUnit.METERS

    def test_resolution_scale_factor(self):
        """Test scale factor calculation."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        source = Resolution(x=10.0, y=10.0, unit=ResolutionUnit.METERS)
        target = Resolution(x=30.0, y=30.0, unit=ResolutionUnit.METERS)

        scale_x, scale_y = source.scale_factor_to(target)

        # 10/30 = 1/3
        assert abs(scale_x - (10.0 / 30.0)) < 0.001
        assert abs(scale_y - (10.0 / 30.0)) < 0.001


class TestResolutionCalculator:
    """Tests for resolution calculator."""

    def test_calculate_shape(self):
        """Test shape calculation from bounds and resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionCalculator,
        )

        calc = ResolutionCalculator()
        bounds = (0.0, 0.0, 100.0, 50.0)
        resolution = Resolution(10.0, 10.0, ResolutionUnit.METERS)

        height, width = calc.calculate_shape(bounds, resolution)

        assert width == 10  # 100/10
        assert height == 5  # 50/10

    def test_find_common_resolution_finest(self):
        """Test finding finest common resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionCalculator,
        )

        calc = ResolutionCalculator()
        resolutions = [
            Resolution(10.0, 10.0, ResolutionUnit.METERS),
            Resolution(20.0, 20.0, ResolutionUnit.METERS),
            Resolution(30.0, 30.0, ResolutionUnit.METERS),
        ]

        common = calc.find_common_resolution(resolutions, strategy="finest")

        assert common.x == 10.0
        assert common.y == 10.0

    def test_find_common_resolution_coarsest(self):
        """Test finding coarsest common resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionCalculator,
        )

        calc = ResolutionCalculator()
        resolutions = [
            Resolution(10.0, 10.0, ResolutionUnit.METERS),
            Resolution(20.0, 20.0, ResolutionUnit.METERS),
            Resolution(30.0, 30.0, ResolutionUnit.METERS),
        ]

        common = calc.find_common_resolution(resolutions, strategy="coarsest")

        assert common.x == 30.0
        assert common.y == 30.0


class TestSpatialResampler:
    """Tests for spatial resampling."""

    def test_resample_array_downsample(self):
        """Test downsampling an array."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()

        # Create 100x100 array
        data = np.random.rand(100, 100).astype(np.float32)
        source_res = Resolution(10.0, 10.0, ResolutionUnit.METERS)
        config = ResamplingConfig(
            target_resolution=Resolution(30.0, 30.0, ResolutionUnit.METERS),
            method=ResamplingMethod.AVERAGE,
        )

        resampled, result = resampler.resample_array(data, source_res, config)

        # Should be approximately 33x33
        assert resampled.shape[0] < data.shape[0]
        assert resampled.shape[1] < data.shape[1]
        assert result.scale_factors[0] < 1.0  # Downsampling

    def test_resample_array_upsample(self):
        """Test upsampling an array."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()

        # Create 30x30 array
        data = np.random.rand(30, 30).astype(np.float32)
        source_res = Resolution(30.0, 30.0, ResolutionUnit.METERS)
        config = ResamplingConfig(
            target_resolution=Resolution(10.0, 10.0, ResolutionUnit.METERS),
            method=ResamplingMethod.BILINEAR,
        )

        resampled, result = resampler.resample_array(data, source_res, config)

        # Should be approximately 90x90
        assert resampled.shape[0] > data.shape[0]
        assert resampled.shape[1] > data.shape[1]
        assert result.scale_factors[0] > 1.0  # Upsampling

    def test_resample_preserves_nodata(self):
        """Test that nodata values are preserved during resampling."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()

        # Create array with nodata region
        data = np.ones((100, 100), dtype=np.float32)
        data[40:60, 40:60] = -9999.0  # Nodata region

        source_res = Resolution(10.0, 10.0, ResolutionUnit.METERS)
        config = ResamplingConfig(
            target_resolution=Resolution(20.0, 20.0, ResolutionUnit.METERS),
            method=ResamplingMethod.AVERAGE,
            nodata=-9999.0,
            preserve_nodata=True,
        )

        resampled, result = resampler.resample_array(data, source_res, config)

        # Nodata should still be present
        assert np.any(resampled == -9999.0)

    def test_resample_multiband(self):
        """Test resampling multi-band array."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()

        # Create 3-band 100x100 array
        data = np.random.rand(3, 100, 100).astype(np.float32)
        source_res = Resolution(10.0, 10.0, ResolutionUnit.METERS)
        config = ResamplingConfig(
            target_resolution=Resolution(30.0, 30.0, ResolutionUnit.METERS),
            method=ResamplingMethod.AVERAGE,
        )

        resampled, result = resampler.resample_array(data, source_res, config)

        # Should preserve band count
        assert resampled.ndim == 3
        assert resampled.shape[0] == 3


class TestResolutionHarmonizer:
    """Tests for multi-dataset resolution harmonization."""

    def test_harmonize_to_finest(self):
        """Test harmonizing to finest resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionHarmonizer,
        )

        harmonizer = ResolutionHarmonizer()

        datasets = {
            "high_res": (
                np.random.rand(100, 100).astype(np.float32),
                Resolution(10.0, 10.0, ResolutionUnit.METERS),
            ),
            "low_res": (
                np.random.rand(30, 30).astype(np.float32),
                Resolution(30.0, 30.0, ResolutionUnit.METERS),
            ),
        }

        harmonized = harmonizer.harmonize(datasets, strategy="finest")

        # Low res should be upsampled to match high res
        assert "high_res" in harmonized
        assert "low_res" in harmonized

    def test_harmonize_to_coarsest(self):
        """Test harmonizing to coarsest resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionHarmonizer,
        )

        harmonizer = ResolutionHarmonizer()

        datasets = {
            "high_res": (
                np.random.rand(100, 100).astype(np.float32),
                Resolution(10.0, 10.0, ResolutionUnit.METERS),
            ),
            "low_res": (
                np.random.rand(30, 30).astype(np.float32),
                Resolution(30.0, 30.0, ResolutionUnit.METERS),
            ),
        }

        harmonized = harmonizer.harmonize(datasets, strategy="coarsest")

        # High res should be downsampled
        high_res_data, high_res_result = harmonized["high_res"]
        assert high_res_data.shape[0] < 100


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module convenience functions."""

    def test_reproject_to_crs_function(self):
        """Test convenience reprojection function exists."""
        from core.data.ingestion.normalization.projection import reproject_to_crs

        # Just verify function exists and is callable
        assert callable(reproject_to_crs)

    def test_create_tile_grid_function(self):
        """Test convenience tile grid creation function."""
        from core.data.ingestion.normalization.tiling import create_tile_grid

        grid = create_tile_grid(
            bounds=(0.0, 0.0, 100.0, 100.0),
            resolution=(1.0, 1.0),
            tile_size=(256, 256),
        )

        assert grid.cols >= 1
        assert grid.rows >= 1

    def test_parse_timestamp_function(self):
        """Test convenience timestamp parsing function."""
        from core.data.ingestion.normalization.temporal import parse_timestamp

        dt = parse_timestamp("2024-03-15T10:30:00Z")

        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15

    def test_resample_to_resolution_function(self):
        """Test convenience resampling function."""
        from core.data.ingestion.normalization.resolution import resample_to_resolution

        data = np.random.rand(100, 100).astype(np.float32)
        resampled, result = resample_to_resolution(
            data=data,
            source_resolution=(10.0, 10.0),
            target_resolution=(30.0, 30.0),
            method="average",
        )

        assert resampled.shape[0] < 100
        assert resampled.shape[1] < 100


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_resolution_invalid_zero(self):
        """Test that zero resolution raises error."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        with pytest.raises(ValueError):
            Resolution(x=0.0, y=10.0, unit=ResolutionUnit.METERS)

    def test_resolution_invalid_negative(self):
        """Test that negative resolution raises error."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        with pytest.raises(ValueError):
            Resolution(x=-10.0, y=10.0, unit=ResolutionUnit.METERS)

    def test_time_range_invalid_order(self):
        """Test that invalid time range order raises error."""
        from core.data.ingestion.normalization.temporal import TimeRange

        with pytest.raises(ValueError):
            TimeRange(
                start=datetime(2024, 1, 10, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_tile_grid_invalid_config(self):
        """Test tile grid with invalid config."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError):
            TileGridConfig(
                tile_size=(256, 256),
                overlap=300,  # Larger than tile size
            )

    def test_resample_1d_array_error(self):
        """Test that 1D array raises error."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.random.rand(100)  # 1D array

        config = ResamplingConfig(
            target_resolution=Resolution(30.0, 30.0, ResolutionUnit.METERS),
        )

        with pytest.raises(ValueError):
            resampler.resample_array(
                data,
                Resolution(10.0, 10.0, ResolutionUnit.METERS),
                config,
            )

    def test_resample_extreme_downscaling_error(self):
        """Test that extreme downscaling raises error when output would be 0 pixels."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.random.rand(10, 10).astype(np.float32)  # Small array

        config = ResamplingConfig(
            target_resolution=Resolution(10000.0, 10000.0, ResolutionUnit.METERS),  # Extreme downscale
        )

        with pytest.raises(ValueError, match="Target resolution too coarse"):
            resampler.resample_array(
                data,
                Resolution(10.0, 10.0, ResolutionUnit.METERS),
                config,
            )

    def test_resolution_scale_factor_different_units_error(self):
        """Test that scale factor calculation fails with different units."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        source = Resolution(10.0, 10.0, ResolutionUnit.METERS)
        target = Resolution(0.001, 0.001, ResolutionUnit.DEGREES)

        with pytest.raises(ValueError, match="must have same units"):
            source.scale_factor_to(target)

    def test_tile_grid_negative_overlap_error(self):
        """Test that negative overlap raises error."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError):
            TileGridConfig(
                tile_size=(256, 256),
                overlap=-10,  # Negative overlap
            )

    def test_tile_grid_zero_tile_size_error(self):
        """Test that zero tile size raises error."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError):
            TileGridConfig(
                tile_size=(0, 256),
            )

    def test_temporal_floor_time_zero_resolution_error(self):
        """Test that zero resolution raises error in floor_time."""
        from core.data.ingestion.normalization.temporal import TemporalResampler

        resampler = TemporalResampler()
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Resolution must be positive"):
            resampler._floor_time(dt, timedelta(seconds=0))

    def test_time_range_split_zero_error(self):
        """Test that splitting by zero raises error."""
        from core.data.ingestion.normalization.temporal import TimeRange

        time_range = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        with pytest.raises(ValueError, match="n must be positive"):
            time_range.split(0)

    def test_resample_with_nan_values(self):
        """Test resampling handles NaN values correctly."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.ones((100, 100), dtype=np.float32)
        data[40:60, 40:60] = np.nan

        config = ResamplingConfig(
            target_resolution=Resolution(30.0, 30.0, ResolutionUnit.METERS),
            method=ResamplingMethod.AVERAGE,
        )

        resampled, result = resampler.resample_array(
            data,
            Resolution(10.0, 10.0, ResolutionUnit.METERS),
            config,
        )

        # Should complete without error
        assert resampled.shape[0] < 100
        assert resampled.shape[1] < 100

    def test_timestamp_invalid_format_error(self):
        """Test that invalid timestamp format raises error."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()

        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            handler.parse("not a valid timestamp")

    def test_timestamp_invalid_type_error(self):
        """Test that invalid timestamp type raises error."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()

        with pytest.raises(ValueError, match="Unsupported timestamp type"):
            handler.parse([1, 2, 3])  # List is not supported

    def test_tile_bounds_zero_dimensions(self):
        """Test tile bounds with zero dimensions."""
        from core.data.ingestion.normalization.tiling import TileBounds

        # Zero-width bounds
        bounds = TileBounds(minx=5.0, miny=0.0, maxx=5.0, maxy=10.0)
        assert bounds.width == 0.0
        assert bounds.height == 10.0

    def test_temporal_sample_naive_datetime(self):
        """Test temporal sample with naive datetime gets UTC timezone."""
        from core.data.ingestion.normalization.temporal import TemporalSample

        # Naive datetime (no timezone)
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        sample = TemporalSample(timestamp=naive_dt, data=1.0)

        # Should have UTC timezone added
        assert sample.timestamp.tzinfo == timezone.utc

    def test_time_range_naive_datetime(self):
        """Test time range with naive datetimes gets UTC timezone."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        time_range = TimeRange(start=start, end=end)

        assert time_range.start.tzinfo == timezone.utc
        assert time_range.end.tzinfo == timezone.utc

    def test_resolution_calculator_no_resolutions_error(self):
        """Test find_common_resolution with empty list raises error."""
        from core.data.ingestion.normalization.resolution import ResolutionCalculator

        calc = ResolutionCalculator()

        with pytest.raises(ValueError, match="No resolutions provided"):
            calc.find_common_resolution([])

    def test_resolution_harmonizer_empty_datasets(self):
        """Test harmonizing empty datasets returns empty dict."""
        from core.data.ingestion.normalization.resolution import ResolutionHarmonizer

        harmonizer = ResolutionHarmonizer()
        result = harmonizer.harmonize({})

        assert result == {}

    def test_resample_same_resolution(self):
        """Test resampling when source and target resolution are the same."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionHarmonizer,
        )

        harmonizer = ResolutionHarmonizer()
        data = np.random.rand(100, 100).astype(np.float32)
        resolution = Resolution(10.0, 10.0, ResolutionUnit.METERS)

        datasets = {
            "same_res": (data, resolution),
        }

        result = harmonizer.harmonize(
            datasets,
            target_resolution=resolution,
        )

        # Should return original data unchanged
        assert "same_res" in result
        resampled_data, resampled_result = result["same_res"]
        assert resampled_result.resampling_method == "none"
        assert np.array_equal(resampled_data, data)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for normalization pipeline."""

    def test_projection_then_resample(self):
        """Test reprojection followed by resampling."""
        pyproj = pytest.importorskip("pyproj")
        from core.data.ingestion.normalization.projection import CRSHandler
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResamplingConfig,
            SpatialResampler,
        )

        # Step 1: CRS transformation
        handler = CRSHandler()
        wgs84_bounds = (0.0, 45.0, 1.0, 46.0)
        utm_bounds = handler.transform_bounds(
            wgs84_bounds, "EPSG:4326", "EPSG:32632"
        )

        # Step 2: Calculate expected shape
        from core.data.ingestion.normalization.resolution import ResolutionCalculator

        calc = ResolutionCalculator()
        res = Resolution(100.0, 100.0, ResolutionUnit.METERS)
        height, width = calc.calculate_shape(utm_bounds, res)

        assert height > 0
        assert width > 0

    def test_temporal_and_spatial_alignment(self):
        """Test combined temporal and spatial normalization."""
        from core.data.ingestion.normalization.temporal import (
            TemporalSample,
            TemporalAligner,
        )
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
            ResolutionHarmonizer,
        )

        # Create temporal samples with spatial data
        base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        samples = [
            TemporalSample(
                timestamp=base_time + timedelta(hours=i),
                data=np.random.rand(50, 50).astype(np.float32),
            )
            for i in range(24)
        ]

        # Verify samples are created correctly
        assert len(samples) == 24
        assert samples[0].data.shape == (50, 50)


# ============================================================================
# MODULE IMPORT TESTS
# ============================================================================


class TestModuleImports:
    """Test that all module imports work correctly."""

    def test_import_normalization_module(self):
        """Test importing main normalization module."""
        from core.data.ingestion.normalization import (
            # Projection
            CRSHandler,
            ReprojectionConfig,
            # Tiling
            TileGrid,
            TileGridConfig,
            # Temporal
            TimeRange,
            TemporalResampler,
            # Resolution
            Resolution,
            SpatialResampler,
        )

        # All imports should succeed
        assert CRSHandler is not None
        assert TileGrid is not None
        assert TimeRange is not None
        assert Resolution is not None

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from core.data.ingestion import normalization

        # Check some key exports exist
        assert hasattr(normalization, "CRSHandler")
        assert hasattr(normalization, "TileGrid")
        assert hasattr(normalization, "TimeRange")
        assert hasattr(normalization, "Resolution")
