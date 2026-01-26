"""
Base configuration and data structures for map visualization.

Defines common types, bounds, and configurations used by both
static and interactive map generators.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
from enum import Enum


class MapType(Enum):
    """Types of maps that can be generated."""
    FLOOD_EXTENT = "flood_extent"
    INFRASTRUCTURE = "infrastructure"
    BEFORE_AFTER = "before_after"


@dataclass
class MapConfig:
    """
    Configuration for map generation.

    Attributes:
        width: Map width in pixels
        height: Map height in pixels
        dpi: Dots per inch for print output (150 for screen, 300 for print)
        title: Optional map title
        show_scale_bar: Include scale bar
        show_north_arrow: Include north arrow
        show_legend: Include legend
        attribution: Map data attribution text

    Examples:
        # Screen display
        config = MapConfig(width=800, height=600, dpi=150)

        # Print quality
        config = MapConfig(width=3300, height=2550, dpi=300)
    """
    width: int = 800
    height: int = 600
    dpi: int = 150
    title: Optional[str] = None
    show_scale_bar: bool = True
    show_north_arrow: bool = True
    show_legend: bool = True
    attribution: str = "© OpenStreetMap contributors"


@dataclass
class MapBounds:
    """
    Geographic bounds for a map.

    Uses WGS 84 (EPSG:4326) coordinate system.

    Attributes:
        min_lon: Western edge longitude
        min_lat: Southern edge latitude
        max_lon: Eastern edge longitude
        max_lat: Northern edge latitude

    Examples:
        # Fort Myers, Florida area
        bounds = MapBounds(
            min_lon=-82.2,
            min_lat=26.3,
            max_lon=-81.7,
            max_lat=26.8
        )

        # Get bounding box tuple
        bbox = bounds.bbox  # (-82.2, 26.3, -81.7, 26.8)

        # Get center point
        center = bounds.center  # (-81.95, 26.55)
    """
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        Return bounding box as (min_lon, min_lat, max_lon, max_lat).

        This format is compatible with most GIS libraries.
        """
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    @property
    def center(self) -> Tuple[float, float]:
        """
        Return center point as (lon, lat).

        Calculated as simple midpoint. For large areas, consider
        using a proper centroid calculation.
        """
        center_lon = (self.min_lon + self.max_lon) / 2
        center_lat = (self.min_lat + self.max_lat) / 2
        return (center_lon, center_lat)

    @property
    def width_degrees(self) -> float:
        """Return east-west extent in degrees."""
        return self.max_lon - self.min_lon

    @property
    def height_degrees(self) -> float:
        """Return north-south extent in degrees."""
        return self.max_lat - self.min_lat

    def to_geojson_bbox(self) -> List[float]:
        """
        Return bounding box in GeoJSON format.

        Returns:
            List of [west, south, east, north] coordinates
        """
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]
