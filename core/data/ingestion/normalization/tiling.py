"""
Tile Scheme and Tiling Operations for Data Normalization.

Provides tools for generating tile grids, partitioning raster data into tiles,
and managing tile coordinate systems. Supports standard schemes like XYZ/TMS
and custom grids with configurable overlap.

Key Capabilities:
- XYZ/TMS tile scheme generation
- Custom tile grids with arbitrary origins and sizes
- Overlap handling for edge effects
- Tile index calculations and bounds lookups
- Efficient tile iteration for large datasets
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TileScheme(Enum):
    """Standard tile schemes."""

    XYZ = "xyz"  # Standard web mercator (origin top-left)
    TMS = "tms"  # Tile Map Service (origin bottom-left)
    CUSTOM = "custom"  # Custom grid with explicit parameters


class TileOrigin(Enum):
    """Tile grid origin position."""

    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"
    TOP_RIGHT = "top_right"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class TileBounds:
    """
    Bounds of a tile in geographic or projected coordinates.

    Attributes:
        minx: Minimum x coordinate
        miny: Minimum y coordinate
        maxx: Maximum x coordinate
        maxy: Maximum y coordinate
    """

    minx: float
    miny: float
    maxx: float
    maxy: float

    @property
    def width(self) -> float:
        """Width of bounds."""
        return self.maxx - self.minx

    @property
    def height(self) -> float:
        """Height of bounds."""
        return self.maxy - self.miny

    @property
    def center(self) -> Tuple[float, float]:
        """Center point of bounds."""
        return (
            (self.minx + self.maxx) / 2,
            (self.miny + self.maxy) / 2,
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Return bounds as tuple."""
        return (self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other: "TileBounds") -> bool:
        """Check if bounds intersect another."""
        return not (
            self.maxx < other.minx
            or self.minx > other.maxx
            or self.maxy < other.miny
            or self.miny > other.maxy
        )

    def intersection(self, other: "TileBounds") -> Optional["TileBounds"]:
        """Get intersection with another bounds, or None if no intersection."""
        if not self.intersects(other):
            return None
        return TileBounds(
            minx=max(self.minx, other.minx),
            miny=max(self.miny, other.miny),
            maxx=min(self.maxx, other.maxx),
            maxy=min(self.maxy, other.maxy),
        )

    def buffer(self, amount: float) -> "TileBounds":
        """Expand bounds by given amount in all directions."""
        return TileBounds(
            minx=self.minx - amount,
            miny=self.miny - amount,
            maxx=self.maxx + amount,
            maxy=self.maxy + amount,
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "minx": self.minx,
            "miny": self.miny,
            "maxx": self.maxx,
            "maxy": self.maxy,
        }


@dataclass
class TileIndex:
    """
    Index of a tile within a grid.

    Attributes:
        x: Column index
        y: Row index
        z: Zoom level (for web mercator schemes)
    """

    x: int
    y: int
    z: int = 0

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __eq__(self, other):
        if not isinstance(other, TileIndex):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def to_tuple(self) -> Tuple[int, int, int]:
        """Return as tuple (x, y, z)."""
        return (self.x, self.y, self.z)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Tile:
    """
    A single tile with index and bounds.

    Attributes:
        index: Tile index in grid
        bounds: Geographic/projected bounds
        overlap_bounds: Bounds including overlap region
        data: Optional numpy array of tile data
        metadata: Additional tile metadata
    """

    index: TileIndex
    bounds: TileBounds
    overlap_bounds: Optional[TileBounds] = None
    data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_overlap(self) -> bool:
        """Check if tile has overlap region."""
        return self.overlap_bounds is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without data)."""
        return {
            "index": self.index.to_dict(),
            "bounds": self.bounds.to_dict(),
            "overlap_bounds": self.overlap_bounds.to_dict() if self.overlap_bounds else None,
            "has_data": self.data is not None,
            "metadata": self.metadata,
        }


@dataclass
class TileGridConfig:
    """
    Configuration for a tile grid.

    Attributes:
        tile_size: Size of tiles in pixels (width, height)
        scheme: Tile scheme type
        origin: Grid origin position
        crs: Coordinate reference system
        bounds: Total grid bounds in CRS units
        resolution: Pixel resolution (x, y) in CRS units
        overlap: Overlap in pixels between adjacent tiles
        min_zoom: Minimum zoom level (for XYZ/TMS)
        max_zoom: Maximum zoom level (for XYZ/TMS)
    """

    tile_size: Tuple[int, int] = (256, 256)
    scheme: TileScheme = TileScheme.CUSTOM
    origin: TileOrigin = TileOrigin.TOP_LEFT
    crs: str = "EPSG:4326"
    bounds: Optional[Tuple[float, float, float, float]] = None
    resolution: Optional[Tuple[float, float]] = None
    overlap: int = 0
    min_zoom: int = 0
    max_zoom: int = 18

    def __post_init__(self):
        """Validate configuration."""
        if self.tile_size[0] < 1 or self.tile_size[1] < 1:
            raise ValueError("Tile size must be positive")
        if self.overlap < 0:
            raise ValueError("Overlap must be non-negative")
        if self.overlap >= min(self.tile_size):
            raise ValueError("Overlap must be less than tile size")


@dataclass
class TileGridInfo:
    """
    Information about a generated tile grid.

    Attributes:
        config: Grid configuration
        total_tiles: Total number of tiles
        cols: Number of columns
        rows: Number of rows
        tile_bounds_crs: Tile size in CRS units
    """

    config: TileGridConfig
    total_tiles: int
    cols: int
    rows: int
    tile_bounds_crs: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tiles": self.total_tiles,
            "cols": self.cols,
            "rows": self.rows,
            "tile_size_pixels": self.config.tile_size,
            "tile_size_crs": self.tile_bounds_crs,
            "overlap_pixels": self.config.overlap,
            "crs": self.config.crs,
        }


class TileGrid:
    """
    Manages a grid of tiles over a geographic extent.

    Supports creating tile grids, looking up tiles by coordinate,
    and iterating over tiles for processing.

    Example:
        config = TileGridConfig(
            tile_size=(512, 512),
            bounds=(-180, -90, 180, 90),
            resolution=(0.1, 0.1),
            overlap=32
        )
        grid = TileGrid(config)

        # Get all tiles
        for tile in grid.iterate_tiles():
            process_tile(tile)

        # Get tiles intersecting a region
        aoi = TileBounds(-10, 40, 10, 50)
        for tile in grid.tiles_in_bounds(aoi):
            process_tile(tile)
    """

    def __init__(self, config: TileGridConfig):
        """
        Initialize tile grid.

        Args:
            config: Grid configuration
        """
        self.config = config
        self._calculate_grid()

    def _calculate_grid(self) -> None:
        """Calculate grid dimensions."""
        if self.config.bounds is None:
            raise ValueError("Bounds must be specified for grid calculation")

        if self.config.resolution is None:
            raise ValueError("Resolution must be specified for grid calculation")

        minx, miny, maxx, maxy = self.config.bounds
        res_x, res_y = self.config.resolution
        tile_w, tile_h = self.config.tile_size

        # Calculate tile size in CRS units
        self.tile_width_crs = tile_w * abs(res_x)
        self.tile_height_crs = tile_h * abs(res_y)

        # Calculate grid dimensions
        extent_x = maxx - minx
        extent_y = maxy - miny

        self.cols = math.ceil(extent_x / self.tile_width_crs)
        self.rows = math.ceil(extent_y / self.tile_height_crs)
        self.total_tiles = self.cols * self.rows

        # Calculate overlap in CRS units
        self.overlap_crs_x = self.config.overlap * abs(res_x)
        self.overlap_crs_y = self.config.overlap * abs(res_y)

        logger.debug(
            f"Grid initialized: {self.cols}x{self.rows} = {self.total_tiles} tiles"
        )

    @property
    def info(self) -> TileGridInfo:
        """Get grid information."""
        return TileGridInfo(
            config=self.config,
            total_tiles=self.total_tiles,
            cols=self.cols,
            rows=self.rows,
            tile_bounds_crs=(self.tile_width_crs, self.tile_height_crs),
        )

    def get_tile_bounds(self, index: TileIndex) -> TileBounds:
        """
        Get bounds of a tile by index.

        Args:
            index: Tile index

        Returns:
            TileBounds for the tile
        """
        if self.config.bounds is None:
            raise ValueError("Bounds not set")

        minx, miny, maxx, maxy = self.config.bounds

        # Calculate tile origin based on origin setting
        if self.config.origin in [TileOrigin.TOP_LEFT, TileOrigin.TOP_RIGHT]:
            # Y increases downward
            tile_miny = maxy - (index.y + 1) * self.tile_height_crs
            tile_maxy = maxy - index.y * self.tile_height_crs
        else:
            # Y increases upward
            tile_miny = miny + index.y * self.tile_height_crs
            tile_maxy = miny + (index.y + 1) * self.tile_height_crs

        if self.config.origin in [TileOrigin.TOP_LEFT, TileOrigin.BOTTOM_LEFT]:
            # X increases rightward
            tile_minx = minx + index.x * self.tile_width_crs
            tile_maxx = minx + (index.x + 1) * self.tile_width_crs
        else:
            # X increases leftward
            tile_minx = maxx - (index.x + 1) * self.tile_width_crs
            tile_maxx = maxx - index.x * self.tile_width_crs

        # Clamp to grid bounds
        tile_minx = max(tile_minx, minx)
        tile_miny = max(tile_miny, miny)
        tile_maxx = min(tile_maxx, maxx)
        tile_maxy = min(tile_maxy, maxy)

        return TileBounds(
            minx=tile_minx,
            miny=tile_miny,
            maxx=tile_maxx,
            maxy=tile_maxy,
        )

    def get_tile_overlap_bounds(self, index: TileIndex) -> TileBounds:
        """
        Get bounds of a tile including overlap region.

        Args:
            index: Tile index

        Returns:
            TileBounds including overlap
        """
        bounds = self.get_tile_bounds(index)

        if self.config.overlap == 0:
            return bounds

        return TileBounds(
            minx=bounds.minx - self.overlap_crs_x,
            miny=bounds.miny - self.overlap_crs_y,
            maxx=bounds.maxx + self.overlap_crs_x,
            maxy=bounds.maxy + self.overlap_crs_y,
        )

    def get_tile(self, index: TileIndex) -> Tile:
        """
        Get a Tile object by index.

        Args:
            index: Tile index

        Returns:
            Tile with bounds
        """
        bounds = self.get_tile_bounds(index)
        overlap_bounds = None
        if self.config.overlap > 0:
            overlap_bounds = self.get_tile_overlap_bounds(index)

        return Tile(
            index=index,
            bounds=bounds,
            overlap_bounds=overlap_bounds,
        )

    def get_tile_at_point(self, x: float, y: float) -> Optional[TileIndex]:
        """
        Get tile index containing a point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            TileIndex or None if outside grid
        """
        if self.config.bounds is None:
            return None

        minx, miny, maxx, maxy = self.config.bounds

        # Check if point is in bounds
        if not (minx <= x <= maxx and miny <= y <= maxy):
            return None

        # Calculate tile indices
        if self.config.origin in [TileOrigin.TOP_LEFT, TileOrigin.BOTTOM_LEFT]:
            col = int((x - minx) / self.tile_width_crs)
        else:
            col = int((maxx - x) / self.tile_width_crs)

        if self.config.origin in [TileOrigin.TOP_LEFT, TileOrigin.TOP_RIGHT]:
            row = int((maxy - y) / self.tile_height_crs)
        else:
            row = int((y - miny) / self.tile_height_crs)

        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        return TileIndex(x=col, y=row)

    def iterate_tiles(self) -> Generator[Tile, None, None]:
        """
        Iterate over all tiles in the grid.

        Yields:
            Tile objects in row-major order
        """
        for row in range(self.rows):
            for col in range(self.cols):
                yield self.get_tile(TileIndex(x=col, y=row))

    def tiles_in_bounds(
        self, query_bounds: TileBounds
    ) -> Generator[Tile, None, None]:
        """
        Get tiles that intersect given bounds.

        Args:
            query_bounds: Bounds to query

        Yields:
            Tiles that intersect the query bounds
        """
        if self.config.bounds is None:
            return

        minx, miny, maxx, maxy = self.config.bounds

        # Calculate tile range that could intersect
        # Conservative approach - check all potentially overlapping tiles
        start_x = max(
            0, int((query_bounds.minx - minx) / self.tile_width_crs) - 1
        )
        end_x = min(
            self.cols, int((query_bounds.maxx - minx) / self.tile_width_crs) + 2
        )
        start_y = max(
            0, int((miny - query_bounds.maxy) / self.tile_height_crs) - 1
        )
        end_y = min(
            self.rows, int((maxy - query_bounds.miny) / self.tile_height_crs) + 2
        )

        # Handle origin-specific indexing
        if self.config.origin in [TileOrigin.TOP_LEFT, TileOrigin.TOP_RIGHT]:
            start_y = max(0, int((maxy - query_bounds.maxy) / self.tile_height_crs) - 1)
            end_y = min(self.rows, int((maxy - query_bounds.miny) / self.tile_height_crs) + 2)

        for row in range(start_y, end_y):
            for col in range(start_x, end_x):
                tile = self.get_tile(TileIndex(x=col, y=row))
                if tile.bounds.intersects(query_bounds):
                    yield tile


class WebMercatorTiles:
    """
    Web Mercator (EPSG:3857) tile calculations.

    Implements standard XYZ tile scheme used by web maps.
    """

    # Web Mercator constants
    EARTH_RADIUS = 6378137.0
    MAX_LATITUDE = 85.051128779806589
    ORIGIN_SHIFT = 2 * math.pi * EARTH_RADIUS / 2.0

    @classmethod
    def latlng_to_tile(cls, lat: float, lng: float, zoom: int) -> TileIndex:
        """
        Convert lat/lng to tile index at given zoom.

        Args:
            lat: Latitude in degrees
            lng: Longitude in degrees
            zoom: Zoom level

        Returns:
            TileIndex at the given zoom level
        """
        # Clamp latitude to valid range
        lat = max(-cls.MAX_LATITUDE, min(cls.MAX_LATITUDE, lat))

        n = 2 ** zoom
        x = int((lng + 180.0) / 360.0 * n)
        lat_rad = math.radians(lat)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)

        # Clamp to valid range
        x = max(0, min(x, n - 1))
        y = max(0, min(y, n - 1))

        return TileIndex(x=x, y=y, z=zoom)

    @classmethod
    def tile_to_bounds(cls, tile: TileIndex, scheme: TileScheme = TileScheme.XYZ) -> TileBounds:
        """
        Get geographic bounds of a tile.

        Args:
            tile: Tile index
            scheme: Tile scheme (XYZ or TMS)

        Returns:
            TileBounds in EPSG:4326
        """
        n = 2 ** tile.z

        # Handle TMS y-flip
        y = tile.y
        if scheme == TileScheme.TMS:
            y = n - 1 - tile.y

        lng_min = tile.x / n * 360.0 - 180.0
        lng_max = (tile.x + 1) / n * 360.0 - 180.0

        lat_max_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_min_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))

        lat_max = math.degrees(lat_max_rad)
        lat_min = math.degrees(lat_min_rad)

        return TileBounds(
            minx=lng_min,
            miny=lat_min,
            maxx=lng_max,
            maxy=lat_max,
        )

    @classmethod
    def tiles_in_bounds(
        cls,
        bounds: TileBounds,
        zoom: int,
        scheme: TileScheme = TileScheme.XYZ,
    ) -> Generator[TileIndex, None, None]:
        """
        Get all tiles within bounds at given zoom.

        Args:
            bounds: Geographic bounds
            zoom: Zoom level
            scheme: Tile scheme

        Yields:
            TileIndex for each tile
        """
        min_tile = cls.latlng_to_tile(bounds.maxy, bounds.minx, zoom)
        max_tile = cls.latlng_to_tile(bounds.miny, bounds.maxx, zoom)

        for y in range(min_tile.y, max_tile.y + 1):
            for x in range(min_tile.x, max_tile.x + 1):
                yield TileIndex(x=x, y=y, z=zoom)

    @classmethod
    def meters_per_pixel(cls, lat: float, zoom: int, tile_size: int = 256) -> float:
        """
        Calculate meters per pixel at given location and zoom.

        Args:
            lat: Latitude in degrees
            zoom: Zoom level
            tile_size: Tile size in pixels

        Returns:
            Meters per pixel
        """
        return (
            cls.EARTH_RADIUS
            * 2
            * math.pi
            * math.cos(math.radians(lat))
            / (tile_size * 2 ** zoom)
        )

    @classmethod
    def zoom_for_resolution(
        cls,
        resolution: float,
        lat: float = 0.0,
        tile_size: int = 256,
    ) -> int:
        """
        Find zoom level that matches target resolution.

        Args:
            resolution: Target resolution in meters per pixel
            lat: Reference latitude
            tile_size: Tile size in pixels

        Returns:
            Best matching zoom level
        """
        for zoom in range(25):
            mpp = cls.meters_per_pixel(lat, zoom, tile_size)
            if mpp <= resolution:
                return zoom
        return 24


class RasterTiler:
    """
    Tiles raster data according to a tile grid.

    Extracts tile data from rasters with optional overlap
    and handles edge cases at grid boundaries.
    """

    def __init__(self, grid: TileGrid):
        """
        Initialize tiler with grid.

        Args:
            grid: Tile grid to use
        """
        self.grid = grid

    def extract_tile(
        self,
        data: np.ndarray,
        data_bounds: TileBounds,
        tile: Tile,
        fill_value: float = 0,
    ) -> np.ndarray:
        """
        Extract tile data from a larger raster.

        Args:
            data: Source raster data (bands, height, width) or (height, width)
            data_bounds: Bounds of the source data
            tile: Tile to extract
            fill_value: Value for out-of-bounds areas

        Returns:
            Tile data array
        """
        # Handle 2D arrays
        if data.ndim == 2:
            data = data[np.newaxis, :, :]
        elif data.ndim != 3:
            raise ValueError(f"Expected 2D or 3D array, got {data.ndim}D")

        bands, src_height, src_width = data.shape
        tile_w, tile_h = self.grid.config.tile_size
        overlap = self.grid.config.overlap

        # Calculate pixel resolution
        res_x = data_bounds.width / src_width
        res_y = data_bounds.height / src_height

        # Get effective tile bounds (with overlap if present)
        if tile.overlap_bounds:
            extract_bounds = tile.overlap_bounds
            output_w = tile_w + 2 * overlap
            output_h = tile_h + 2 * overlap
        else:
            extract_bounds = tile.bounds
            output_w = tile_w
            output_h = tile_h

        # Calculate source pixel coordinates
        src_x_start = int((extract_bounds.minx - data_bounds.minx) / res_x)
        src_y_start = int((data_bounds.maxy - extract_bounds.maxy) / res_y)
        src_x_end = int((extract_bounds.maxx - data_bounds.minx) / res_x)
        src_y_end = int((data_bounds.maxy - extract_bounds.miny) / res_y)

        # Initialize output with fill value
        output = np.full((bands, output_h, output_w), fill_value, dtype=data.dtype)

        # Calculate valid source region
        valid_src_x_start = max(0, src_x_start)
        valid_src_y_start = max(0, src_y_start)
        valid_src_x_end = min(src_width, src_x_end)
        valid_src_y_end = min(src_height, src_y_end)

        # Calculate corresponding output region
        out_x_start = valid_src_x_start - src_x_start
        out_y_start = valid_src_y_start - src_y_start
        out_x_end = out_x_start + (valid_src_x_end - valid_src_x_start)
        out_y_end = out_y_start + (valid_src_y_end - valid_src_y_start)

        # Copy valid data
        if valid_src_x_end > valid_src_x_start and valid_src_y_end > valid_src_y_start:
            output[
                :,
                out_y_start:out_y_end,
                out_x_start:out_x_end,
            ] = data[
                :,
                valid_src_y_start:valid_src_y_end,
                valid_src_x_start:valid_src_x_end,
            ]

        return output.squeeze()

    def stitch_tiles(
        self,
        tiles: List[Tile],
        output_shape: Tuple[int, int],
        blend_overlap: bool = True,
    ) -> np.ndarray:
        """
        Stitch multiple tiles back into a single raster.

        Args:
            tiles: List of tiles with data
            output_shape: Shape of output (height, width)
            blend_overlap: Whether to blend overlap regions

        Returns:
            Stitched raster array
        """
        if not tiles:
            raise ValueError("No tiles to stitch")

        # Get reference tile for dtype
        ref_tile = next(t for t in tiles if t.data is not None)
        if ref_tile.data is None:
            raise ValueError("No tiles have data")

        # Determine output bands
        if ref_tile.data.ndim == 2:
            bands = 1
        else:
            bands = ref_tile.data.shape[0]

        output = np.zeros((bands, *output_shape), dtype=ref_tile.data.dtype)
        weight = np.zeros(output_shape, dtype=np.float32)

        tile_w, tile_h = self.grid.config.tile_size
        overlap = self.grid.config.overlap

        for tile in tiles:
            if tile.data is None:
                continue

            tile_data = tile.data
            if tile_data.ndim == 2:
                tile_data = tile_data[np.newaxis, :, :]

            # Calculate output position
            out_x = tile.index.x * tile_w
            out_y = tile.index.y * tile_h

            # Handle overlap trimming
            if overlap > 0 and tile.overlap_bounds:
                # Extract core region without overlap
                tile_data = tile_data[:, overlap:-overlap, overlap:-overlap]

            # Calculate valid region
            h, w = tile_data.shape[1], tile_data.shape[2]
            end_x = min(out_x + w, output_shape[1])
            end_y = min(out_y + h, output_shape[0])
            valid_w = end_x - out_x
            valid_h = end_y - out_y

            if valid_w > 0 and valid_h > 0:
                output[:, out_y:end_y, out_x:end_x] += tile_data[:, :valid_h, :valid_w]
                weight[out_y:end_y, out_x:end_x] += 1

        # Normalize by weight
        weight = np.maximum(weight, 1)
        for b in range(bands):
            output[b] = output[b] / weight

        return output.squeeze()


def create_tile_grid(
    bounds: Tuple[float, float, float, float],
    resolution: Tuple[float, float],
    tile_size: Tuple[int, int] = (512, 512),
    overlap: int = 0,
    crs: str = "EPSG:4326",
) -> TileGrid:
    """
    Convenience function to create a tile grid.

    Args:
        bounds: Grid bounds (minx, miny, maxx, maxy)
        resolution: Pixel resolution (x, y)
        tile_size: Tile size in pixels (width, height)
        overlap: Overlap in pixels
        crs: Coordinate reference system

    Returns:
        TileGrid instance
    """
    config = TileGridConfig(
        tile_size=tile_size,
        bounds=bounds,
        resolution=resolution,
        overlap=overlap,
        crs=crs,
    )
    return TileGrid(config)


def get_web_mercator_tiles(
    bounds: Tuple[float, float, float, float],
    zoom: int,
) -> List[TileIndex]:
    """
    Get list of web mercator tiles covering bounds.

    Args:
        bounds: Geographic bounds (minx, miny, maxx, maxy)
        zoom: Zoom level

    Returns:
        List of TileIndex
    """
    tile_bounds = TileBounds(*bounds)
    return list(WebMercatorTiles.tiles_in_bounds(tile_bounds, zoom))
