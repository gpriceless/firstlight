"""
Tests for normalization tools in the ingestion pipeline.

Tests cover:
- Projection and CRS handling
- Tiling operations
- Temporal alignment
- Resolution and resampling
"""

import math
import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

# Check for optional dependencies
try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

requires_pyproj = pytest.mark.skipif(not HAS_PYPROJ, reason="pyproj not installed")
requires_rasterio = pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")


# ============================================================================
# Projection Tests
# ============================================================================

@requires_pyproj
class TestCRSHandler:
    """Tests for CRS handling operations."""

    def test_parse_epsg_code(self):
        """Test parsing EPSG code."""
        from core.data.ingestion.normalization.projection import CRSHandler, CRSType

        handler = CRSHandler()
        info = handler.parse_crs("EPSG:4326")

        assert info.is_geographic is True
        assert info.is_projected is False
        assert info.crs_type == CRSType.GEOGRAPHIC

    def test_parse_epsg_integer(self):
        """Test parsing EPSG code as integer."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        info = handler.parse_crs(4326)

        assert info.is_geographic is True

    def test_parse_projected_crs(self):
        """Test parsing projected CRS (UTM)."""
        from core.data.ingestion.normalization.projection import CRSHandler, CRSType

        handler = CRSHandler()
        info = handler.parse_crs("EPSG:32632")  # UTM zone 32N

        assert info.is_projected is True
        assert info.is_geographic is False
        assert info.crs_type == CRSType.PROJECTED

    def test_parse_invalid_crs_raises(self):
        """Test that invalid CRS raises ValueError."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        with pytest.raises(ValueError, match="Cannot parse CRS"):
            handler.parse_crs("INVALID:9999")

    def test_are_equivalent_same_crs(self):
        """Test CRS equivalence for same CRS."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        assert handler.are_equivalent("EPSG:4326", "EPSG:4326") is True

    def test_are_equivalent_different_crs(self):
        """Test CRS equivalence for different CRS."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        assert handler.are_equivalent("EPSG:4326", "EPSG:32632") is False

    def test_are_equivalent_invalid_crs(self):
        """Test CRS equivalence returns False for invalid CRS."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        assert handler.are_equivalent("INVALID", "EPSG:4326") is False

    def test_suggest_utm_zone_northern_hemisphere(self):
        """Test UTM zone suggestion for northern hemisphere."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        # Paris, France (longitude ~2.35)
        utm = handler.suggest_utm_zone(2.35, 48.85)
        assert utm == "EPSG:32631"  # Zone 31N

    def test_suggest_utm_zone_southern_hemisphere(self):
        """Test UTM zone suggestion for southern hemisphere."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        # Sydney, Australia
        utm = handler.suggest_utm_zone(151.2, -33.87)
        assert utm == "EPSG:32756"  # Zone 56S

    def test_transform_bounds(self):
        """Test bounds transformation between CRS."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        wgs84_bounds = (-1, 51, 1, 52)  # Near London

        utm_bounds = handler.transform_bounds(
            wgs84_bounds, "EPSG:4326", "EPSG:32631"
        )

        # Check bounds are valid
        assert utm_bounds[0] < utm_bounds[2]  # minx < maxx
        assert utm_bounds[1] < utm_bounds[3]  # miny < maxy

    def test_transform_point(self):
        """Test point transformation."""
        from core.data.ingestion.normalization.projection import CRSHandler

        handler = CRSHandler()
        x, y = handler.transform_point(0, 51, "EPSG:4326", "EPSG:32631")

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert math.isfinite(x)
        assert math.isfinite(y)


@requires_pyproj
class TestVectorReprojector:
    """Tests for vector geometry reprojection."""

    def test_transform_point_geometry(self):
        """Test transforming a Point geometry."""
        from core.data.ingestion.normalization.projection import VectorReprojector

        reprojector = VectorReprojector()
        geom = {"type": "Point", "coordinates": [0, 51]}

        transformed = reprojector.transform_geometry(
            geom, "EPSG:4326", "EPSG:32631"
        )

        assert transformed["type"] == "Point"
        assert len(transformed["coordinates"]) == 2

    def test_transform_polygon_geometry(self):
        """Test transforming a Polygon geometry."""
        from core.data.ingestion.normalization.projection import VectorReprojector

        reprojector = VectorReprojector()
        geom = {
            "type": "Polygon",
            "coordinates": [[[0, 51], [1, 51], [1, 52], [0, 52], [0, 51]]],
        }

        transformed = reprojector.transform_geometry(
            geom, "EPSG:4326", "EPSG:32631"
        )

        assert transformed["type"] == "Polygon"
        assert len(transformed["coordinates"]) == 1
        assert len(transformed["coordinates"][0]) == 5


@requires_pyproj
class TestConvenienceFunctions:
    """Tests for projection convenience functions."""

    def test_get_crs_info(self):
        """Test get_crs_info convenience function."""
        from core.data.ingestion.normalization.projection import get_crs_info

        info = get_crs_info("EPSG:4326")
        assert info.is_geographic is True

    def test_suggest_target_crs(self):
        """Test suggest_target_crs convenience function."""
        from core.data.ingestion.normalization.projection import suggest_target_crs

        bounds = (0, 50, 2, 52)
        suggested = suggest_target_crs(bounds)

        assert suggested.startswith("EPSG:")


# ============================================================================
# Tiling Tests
# ============================================================================

class TestTileBounds:
    """Tests for TileBounds operations."""

    def test_tile_bounds_creation(self):
        """Test TileBounds basic creation."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(0, 0, 10, 10)

        assert bounds.width == 10
        assert bounds.height == 10
        assert bounds.center == (5, 5)

    def test_tile_bounds_intersects(self):
        """Test TileBounds intersection check."""
        from core.data.ingestion.normalization.tiling import TileBounds

        b1 = TileBounds(0, 0, 10, 10)
        b2 = TileBounds(5, 5, 15, 15)
        b3 = TileBounds(20, 20, 30, 30)

        assert b1.intersects(b2) is True
        assert b1.intersects(b3) is False

    def test_tile_bounds_intersection(self):
        """Test TileBounds intersection computation."""
        from core.data.ingestion.normalization.tiling import TileBounds

        b1 = TileBounds(0, 0, 10, 10)
        b2 = TileBounds(5, 5, 15, 15)

        intersection = b1.intersection(b2)

        assert intersection is not None
        assert intersection.minx == 5
        assert intersection.miny == 5
        assert intersection.maxx == 10
        assert intersection.maxy == 10

    def test_tile_bounds_no_intersection(self):
        """Test TileBounds non-intersection returns None."""
        from core.data.ingestion.normalization.tiling import TileBounds

        b1 = TileBounds(0, 0, 10, 10)
        b2 = TileBounds(20, 20, 30, 30)

        assert b1.intersection(b2) is None

    def test_tile_bounds_buffer(self):
        """Test TileBounds buffer expansion."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(10, 10, 20, 20)
        buffered = bounds.buffer(5)

        assert buffered.minx == 5
        assert buffered.miny == 5
        assert buffered.maxx == 25
        assert buffered.maxy == 25


class TestTileIndex:
    """Tests for TileIndex."""

    def test_tile_index_creation(self):
        """Test TileIndex basic creation."""
        from core.data.ingestion.normalization.tiling import TileIndex

        idx = TileIndex(x=1, y=2, z=3)

        assert idx.x == 1
        assert idx.y == 2
        assert idx.z == 3

    def test_tile_index_hash(self):
        """Test TileIndex is hashable."""
        from core.data.ingestion.normalization.tiling import TileIndex

        idx1 = TileIndex(1, 2, 3)
        idx2 = TileIndex(1, 2, 3)
        idx3 = TileIndex(1, 2, 4)

        assert hash(idx1) == hash(idx2)
        assert idx1 == idx2
        assert idx1 != idx3


class TestTileGridConfig:
    """Tests for TileGridConfig validation."""

    def test_valid_config(self):
        """Test valid config creation."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        config = TileGridConfig(tile_size=(256, 256), overlap=16)
        assert config.tile_size == (256, 256)
        assert config.overlap == 16

    def test_invalid_tile_size_raises(self):
        """Test invalid tile size raises ValueError."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError, match="positive"):
            TileGridConfig(tile_size=(0, 256))

    def test_negative_overlap_raises(self):
        """Test negative overlap raises ValueError."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError, match="non-negative"):
            TileGridConfig(tile_size=(256, 256), overlap=-1)

    def test_overlap_too_large_raises(self):
        """Test overlap >= tile size raises ValueError."""
        from core.data.ingestion.normalization.tiling import TileGridConfig

        with pytest.raises(ValueError, match="less than tile size"):
            TileGridConfig(tile_size=(256, 256), overlap=256)


class TestTileGrid:
    """Tests for TileGrid operations."""

    def test_grid_creation(self):
        """Test TileGrid creation."""
        from core.data.ingestion.normalization.tiling import TileGrid, TileGridConfig

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0, 0, 1000, 500),
            resolution=(1, 1),
        )
        grid = TileGrid(config)

        assert grid.cols == 10
        assert grid.rows == 5
        assert grid.total_tiles == 50

    def test_grid_missing_bounds_raises(self):
        """Test TileGrid without bounds raises."""
        from core.data.ingestion.normalization.tiling import TileGrid, TileGridConfig

        config = TileGridConfig(tile_size=(100, 100), resolution=(1, 1))

        with pytest.raises(ValueError, match="Bounds"):
            TileGrid(config)

    def test_get_tile_at_point(self):
        """Test getting tile index at point."""
        from core.data.ingestion.normalization.tiling import TileGrid, TileGridConfig

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0, 0, 1000, 500),
            resolution=(1, 1),
        )
        grid = TileGrid(config)

        idx = grid.get_tile_at_point(150, 450)
        assert idx is not None
        assert idx.x == 1  # Second column

    def test_get_tile_at_point_outside_bounds(self):
        """Test getting tile at point outside bounds returns None."""
        from core.data.ingestion.normalization.tiling import TileGrid, TileGridConfig

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0, 0, 1000, 500),
            resolution=(1, 1),
        )
        grid = TileGrid(config)

        idx = grid.get_tile_at_point(2000, 2000)
        assert idx is None

    def test_iterate_tiles(self):
        """Test iterating over all tiles."""
        from core.data.ingestion.normalization.tiling import TileGrid, TileGridConfig

        config = TileGridConfig(
            tile_size=(100, 100),
            bounds=(0, 0, 300, 200),
            resolution=(1, 1),
        )
        grid = TileGrid(config)

        tiles = list(grid.iterate_tiles())
        assert len(tiles) == grid.total_tiles


class TestWebMercatorTiles:
    """Tests for Web Mercator tile calculations."""

    def test_latlng_to_tile(self):
        """Test lat/lng to tile index conversion."""
        from core.data.ingestion.normalization.tiling import WebMercatorTiles

        # London at zoom 10
        tile = WebMercatorTiles.latlng_to_tile(51.5, -0.1, 10)

        assert tile.z == 10
        assert 500 <= tile.x <= 520
        assert 330 <= tile.y <= 350

    def test_tile_to_bounds(self):
        """Test tile index to bounds conversion."""
        from core.data.ingestion.normalization.tiling import (
            WebMercatorTiles,
            TileIndex,
            TileScheme,
        )

        tile = TileIndex(0, 0, 0)
        bounds = WebMercatorTiles.tile_to_bounds(tile, TileScheme.XYZ)

        assert bounds.minx == pytest.approx(-180, rel=0.01)
        assert bounds.maxy == pytest.approx(85.05, rel=0.01)

    def test_meters_per_pixel(self):
        """Test meters per pixel calculation."""
        from core.data.ingestion.normalization.tiling import WebMercatorTiles

        mpp = WebMercatorTiles.meters_per_pixel(0, 0)
        assert mpp > 100000

        mpp_18 = WebMercatorTiles.meters_per_pixel(0, 18)
        assert 0.1 < mpp_18 < 2


# ============================================================================
# Temporal Tests
# ============================================================================

class TestTimeRange:
    """Tests for TimeRange operations."""

    def test_time_range_creation(self):
        """Test TimeRange basic creation."""
        from core.data.ingestion.normalization.temporal import TimeRange

        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)

        tr = TimeRange(start=start, end=end)

        assert tr.duration == timedelta(days=1)
        assert tr.duration_seconds == 86400

    def test_time_range_invalid_raises(self):
        """Test invalid time range (start >= end) raises."""
        from core.data.ingestion.normalization.temporal import TimeRange

        with pytest.raises(ValueError, match="Start must be before end"):
            TimeRange(
                start=datetime(2024, 1, 2, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )

    def test_time_range_contains(self):
        """Test TimeRange contains check."""
        from core.data.ingestion.normalization.temporal import TimeRange

        tr = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert tr.contains(datetime(2024, 1, 5, tzinfo=timezone.utc)) is True
        assert tr.contains(datetime(2024, 1, 15, tzinfo=timezone.utc)) is False

    def test_time_range_overlaps(self):
        """Test TimeRange overlaps check."""
        from core.data.ingestion.normalization.temporal import TimeRange

        tr1 = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )
        tr2 = TimeRange(
            start=datetime(2024, 1, 5, tzinfo=timezone.utc),
            end=datetime(2024, 1, 15, tzinfo=timezone.utc),
        )
        tr3 = TimeRange(
            start=datetime(2024, 2, 1, tzinfo=timezone.utc),
            end=datetime(2024, 2, 10, tzinfo=timezone.utc),
        )

        assert tr1.overlaps(tr2) is True
        assert tr1.overlaps(tr3) is False

    def test_time_range_split(self):
        """Test TimeRange split into parts."""
        from core.data.ingestion.normalization.temporal import TimeRange

        tr = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 11, tzinfo=timezone.utc),
        )

        parts = tr.split(10)
        assert len(parts) == 10


class TestTimestampHandler:
    """Tests for timestamp parsing and handling."""

    def test_parse_iso_format(self):
        """Test parsing ISO format timestamp."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()

        dt = handler.parse("2024-01-15T10:30:00Z")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.tzinfo == timezone.utc

    def test_parse_unix_timestamp(self):
        """Test parsing Unix timestamp."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()

        dt = handler.parse(1710499800)
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2024

    def test_parse_invalid_raises(self):
        """Test parsing invalid timestamp raises."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()

        with pytest.raises(ValueError, match="Cannot parse"):
            handler.parse("not-a-timestamp")

    def test_format_filename(self):
        """Test formatting for filename."""
        from core.data.ingestion.normalization.temporal import TimestampHandler

        handler = TimestampHandler()
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)

        filename = handler.format_filename(dt)
        assert filename == "20240115_103045"


class TestTemporalResampler:
    """Tests for temporal resampling."""

    def test_resample_downsampling(self):
        """Test downsampling (aggregation)."""
        from core.data.ingestion.normalization.temporal import (
            TemporalResampler,
            TemporalSample,
        )

        resampler = TemporalResampler()

        samples = []
        for i in range(24):
            samples.append(
                TemporalSample(
                    timestamp=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
                    data=float(i),
                )
            )

        resampled = resampler.resample(samples, timedelta(hours=6))

        assert len(resampled) == 4

    def test_resample_empty_input(self):
        """Test resampling empty input returns empty."""
        from core.data.ingestion.normalization.temporal import TemporalResampler

        resampler = TemporalResampler()
        resampled = resampler.resample([], timedelta(hours=1))

        assert resampled == []

    def test_floor_time_zero_resolution_raises(self):
        """Test _floor_time with zero resolution raises."""
        from core.data.ingestion.normalization.temporal import TemporalResampler

        resampler = TemporalResampler()
        dt = datetime(2024, 1, 15, 10, 35, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="positive"):
            resampler._floor_time(dt, timedelta(0))


class TestTemporalConvenienceFunctions:
    """Tests for temporal convenience functions."""

    def test_parse_timestamp(self):
        """Test parse_timestamp function."""
        from core.data.ingestion.normalization.temporal import parse_timestamp

        dt = parse_timestamp("2024-01-15T10:30:00Z")
        assert dt.tzinfo == timezone.utc

    def test_generate_time_range(self):
        """Test generate_time_range function."""
        from core.data.ingestion.normalization.temporal import generate_time_range

        times = list(
            generate_time_range(
                "2024-01-01T00:00:00Z",
                "2024-01-01T05:00:00Z",
                timedelta(hours=1),
            )
        )

        assert len(times) == 6


# ============================================================================
# Resolution Tests
# ============================================================================

class TestResolution:
    """Tests for Resolution dataclass."""

    def test_resolution_creation(self):
        """Test Resolution creation."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res = Resolution(10, 10, ResolutionUnit.METERS)

        assert res.x == 10
        assert res.y == 10
        assert res.area == 100
        assert res.is_square == True  # np.isclose returns np.bool_

    def test_resolution_invalid_raises(self):
        """Test non-positive resolution raises."""
        from core.data.ingestion.normalization.resolution import Resolution

        with pytest.raises(ValueError, match="positive"):
            Resolution(0, 10)

        with pytest.raises(ValueError, match="positive"):
            Resolution(-10, 10)

    def test_resolution_to_meters(self):
        """Test converting resolution to meters."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res = Resolution(1, 1, ResolutionUnit.ARCSECONDS)
        meters = res.to_meters(latitude=0)

        assert meters.unit == ResolutionUnit.METERS
        assert 25 < meters.x < 35

    def test_scale_factor_to(self):
        """Test calculating scale factor."""
        from core.data.ingestion.normalization.resolution import Resolution

        res1 = Resolution(10, 10)
        res2 = Resolution(30, 30)

        scale = res1.scale_factor_to(res2)

        assert scale[0] == pytest.approx(10 / 30)

    def test_scale_factor_to_different_units_raises(self):
        """Test scale factor with different units raises."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionUnit,
        )

        res1 = Resolution(10, 10, ResolutionUnit.METERS)
        res2 = Resolution(0.001, 0.001, ResolutionUnit.DEGREES)

        with pytest.raises(ValueError, match="same units"):
            res1.scale_factor_to(res2)


class TestResolutionCalculator:
    """Tests for ResolutionCalculator."""

    def test_calculate_shape(self):
        """Test calculating raster shape from bounds and resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionCalculator,
        )

        calc = ResolutionCalculator()
        bounds = (0, 0, 1000, 500)
        res = Resolution(10, 10)

        shape = calc.calculate_shape(bounds, res)

        assert shape == (50, 100)

    def test_find_common_resolution_finest(self):
        """Test finding finest common resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionCalculator,
        )

        calc = ResolutionCalculator()
        resolutions = [
            Resolution(10, 10),
            Resolution(20, 20),
            Resolution(30, 30),
        ]

        common = calc.find_common_resolution(resolutions, "finest")

        assert common.x == 10

    def test_find_common_resolution_empty_raises(self):
        """Test empty resolutions list raises."""
        from core.data.ingestion.normalization.resolution import ResolutionCalculator

        calc = ResolutionCalculator()

        with pytest.raises(ValueError, match="No resolutions"):
            calc.find_common_resolution([], "finest")


class TestSpatialResampler:
    """Tests for spatial resampling."""

    def test_resample_array_downsampling(self):
        """Test downsampling an array."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.ones((100, 100), dtype=np.float32)

        config = ResamplingConfig(
            target_resolution=Resolution(2, 2),
            method=ResamplingMethod.AVERAGE,
        )

        result_data, result = resampler.resample_array(
            data, Resolution(1, 1), config
        )

        # Downsampling from 1m to 2m: scale = 1/2 = 0.5, size = 100 * 0.5 = 50
        assert result_data.shape == (50, 50)

    def test_resample_array_upsampling(self):
        """Test upsampling an array."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.ones((50, 50), dtype=np.float32)

        config = ResamplingConfig(
            target_resolution=Resolution(1, 1),
            method=ResamplingMethod.BILINEAR,
        )

        result_data, result = resampler.resample_array(
            data, Resolution(2, 2), config
        )

        # Upsampling from 2m to 1m: scale = 2/1 = 2.0, size = 50 * 2 = 100
        assert result_data.shape == (100, 100)

    def test_resample_array_3d(self):
        """Test resampling 3D array (multi-band)."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResamplingConfig,
            SpatialResampler,
        )

        resampler = SpatialResampler()
        data = np.ones((3, 100, 100), dtype=np.float32)

        config = ResamplingConfig(target_resolution=Resolution(2, 2))

        result_data, result = resampler.resample_array(
            data, Resolution(1, 1), config
        )

        assert result_data.shape == (3, 50, 50)


class TestResolutionHarmonizer:
    """Tests for resolution harmonization."""

    def test_harmonize_datasets(self):
        """Test harmonizing multiple datasets."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionHarmonizer,
        )

        harmonizer = ResolutionHarmonizer()

        datasets = {
            "fine": (np.ones((100, 100), dtype=np.float32), Resolution(10, 10)),
            "coarse": (np.ones((50, 50), dtype=np.float32), Resolution(20, 20)),
        }

        harmonized = harmonizer.harmonize(
            datasets, target_resolution=Resolution(20, 20)
        )

        assert harmonized["fine"][0].shape == (50, 50)
        assert harmonized["coarse"][0].shape == (50, 50)

    def test_harmonize_already_at_target(self):
        """Test harmonizing dataset already at target resolution."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResolutionHarmonizer,
        )

        harmonizer = ResolutionHarmonizer()

        datasets = {
            "data": (np.ones((100, 100), dtype=np.float32), Resolution(10, 10)),
        }

        harmonized = harmonizer.harmonize(
            datasets, target_resolution=Resolution(10, 10)
        )

        assert harmonized["data"][0].shape == (100, 100)
        assert harmonized["data"][1].resampling_method == "none"


class TestResolutionConvenienceFunctions:
    """Tests for resolution convenience functions."""

    def test_resample_to_resolution(self):
        """Test resample_to_resolution function."""
        from core.data.ingestion.normalization.resolution import resample_to_resolution

        data = np.ones((100, 100), dtype=np.float32)

        result_data, result = resample_to_resolution(
            data,
            source_resolution=(10, 10),
            target_resolution=(20, 20),
        )

        assert result_data.shape == (50, 50)

    def test_calculate_resolution(self):
        """Test calculate_resolution function."""
        from core.data.ingestion.normalization.resolution import calculate_resolution

        bounds = (0, 0, 1000, 500)
        shape = (50, 100)

        res = calculate_resolution(bounds, shape)

        assert res.x == 10
        assert res.y == 10


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tile_bounds_zero_size(self):
        """Test TileBounds with zero size."""
        from core.data.ingestion.normalization.tiling import TileBounds

        bounds = TileBounds(0, 0, 0, 0)
        assert bounds.width == 0
        assert bounds.height == 0

    def test_time_range_minimal_duration(self):
        """Test TimeRange with minimal duration."""
        from core.data.ingestion.normalization.temporal import TimeRange

        tr = TimeRange(
            start=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )

        assert tr.duration == timedelta(seconds=1)

    def test_temporal_sample_with_array_data(self):
        """Test TemporalSample with numpy array data."""
        from core.data.ingestion.normalization.temporal import TemporalSample

        arr = np.array([[1, 2], [3, 4]])
        sample = TemporalSample(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            data=arr,
        )

        assert np.array_equal(sample.data, arr)

    def test_resolution_very_large_values(self):
        """Test Resolution with very large values."""
        from core.data.ingestion.normalization.resolution import Resolution

        res = Resolution(1e9, 1e9)
        assert res.area == 1e18

    def test_resolution_very_small_values(self):
        """Test Resolution with very small values."""
        from core.data.ingestion.normalization.resolution import Resolution

        res = Resolution(1e-9, 1e-9)
        assert res.area == pytest.approx(1e-18, rel=1e-10)


# ============================================================================
# Integration Tests
# ============================================================================

class TestNormalizationIntegration:
    """Integration tests for normalization pipeline."""

    @requires_pyproj
    def test_full_projection_workflow(self):
        """Test full projection workflow."""
        from core.data.ingestion.normalization.projection import (
            CRSHandler,
            VectorReprojector,
        )

        handler = CRSHandler()
        vector_reprojector = VectorReprojector()

        info = handler.parse_crs("EPSG:4326")
        assert info.is_geographic

        utm = handler.suggest_utm_zone(0, 51)
        assert "EPSG:326" in utm

        bounds = (-1, 50, 1, 52)
        transformed = handler.transform_bounds(bounds, "EPSG:4326", utm)
        assert all(math.isfinite(v) for v in transformed)

        geom = {"type": "Point", "coordinates": [0, 51]}
        transformed_geom = vector_reprojector.transform_geometry(
            geom, "EPSG:4326", utm
        )
        assert transformed_geom["type"] == "Point"

    def test_full_tiling_workflow(self):
        """Test full tiling workflow."""
        from core.data.ingestion.normalization.tiling import (
            TileGrid,
            TileGridConfig,
            RasterTiler,
            TileBounds,
        )

        config = TileGridConfig(
            tile_size=(50, 50),
            bounds=(0, 0, 500, 300),
            resolution=(1, 1),
            overlap=5,
        )
        grid = TileGrid(config)

        assert grid.total_tiles == 60

        tiler = RasterTiler(grid)

        data = np.arange(300 * 500).reshape(300, 500).astype(np.float32)
        data_bounds = TileBounds(0, 0, 500, 300)

        tile = grid.get_tile(grid.get_tile_at_point(25, 275))
        extracted = tiler.extract_tile(data, data_bounds, tile)

        assert extracted.shape[0] == 60
        assert extracted.shape[1] == 60

    def test_full_temporal_workflow(self):
        """Test full temporal workflow."""
        from core.data.ingestion.normalization.temporal import (
            TemporalResampler,
            TemporalAligner,
            TemporalSample,
            TimeRange,
        )

        tr = TimeRange(
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )

        samples = []
        for i in range(24):
            samples.append(
                TemporalSample(
                    timestamp=tr.start + timedelta(hours=i),
                    data=float(i),
                )
            )

        resampler = TemporalResampler()
        resampled = resampler.resample(samples, timedelta(hours=4))

        assert len(resampled) == 6

        aligner = TemporalAligner()
        target_times = [tr.start + timedelta(hours=i * 6) for i in range(4)]
        aligned = aligner.align_to_times(samples, target_times)

        assert len(aligned) == 4

    def test_full_resolution_workflow(self):
        """Test full resolution workflow."""
        from core.data.ingestion.normalization.resolution import (
            Resolution,
            ResamplingConfig,
            ResamplingMethod,
            SpatialResampler,
            ResolutionHarmonizer,
        )

        fine_data = np.random.rand(100, 100).astype(np.float32)
        coarse_data = np.random.rand(50, 50).astype(np.float32)

        resampler = SpatialResampler()
        config = ResamplingConfig(
            target_resolution=Resolution(20, 20),
            method=ResamplingMethod.AVERAGE,
        )

        resampled, result = resampler.resample_array(
            fine_data, Resolution(10, 10), config
        )

        assert resampled.shape == (50, 50)

        harmonizer = ResolutionHarmonizer()
        datasets = {
            "fine": (fine_data, Resolution(10, 10)),
            "coarse": (coarse_data, Resolution(20, 20)),
        }

        harmonized = harmonizer.harmonize(datasets, strategy="coarsest")

        assert harmonized["fine"][0].shape == (50, 50)
        assert harmonized["coarse"][0].shape == (50, 50)
