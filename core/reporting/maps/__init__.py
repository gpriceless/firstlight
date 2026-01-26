"""
Map visualization module for FirstLight reporting.

Provides static and interactive map generation for flood extent,
infrastructure overlays, and before/after comparisons.

Components:
- base.py: Configuration and bounds dataclasses
- static_map.py: Static map generation using matplotlib/cartopy
- folium_map.py: Interactive web maps using Folium

Examples:
    # Static map for print reports
    from core.reporting.maps.static_map import StaticMapGenerator

    generator = StaticMapGenerator()
    generator.generate_flood_map(
        flood_extent=flood_array,
        bounds=MapBounds(-82.0, 26.3, -81.5, 26.7),
        output_path=Path("flood_map.png"),
        title="Hurricane Ian Flood Extent"
    )

    # Interactive map for web reports
    from core.reporting.maps.folium_map import InteractiveMapGenerator

    generator = InteractiveMapGenerator()
    html = generator.generate_flood_map(
        flood_geojson=flood_geojson,
        bounds=MapBounds(-82.0, 26.3, -81.5, 26.7),
        title="Interactive Flood Map"
    )
"""

from core.reporting.maps.base import (
    MapType,
    MapConfig,
    MapBounds,
)

from core.reporting.maps.static_map import StaticMapGenerator
from core.reporting.maps.folium_map import InteractiveMapGenerator

__all__ = [
    'MapType',
    'MapConfig',
    'MapBounds',
    'StaticMapGenerator',
    'InteractiveMapGenerator',
]
