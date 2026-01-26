"""
Interactive map generation using Folium.

Generates web-based interactive maps with zoom, pan, layer controls,
and infrastructure tooltips. Outputs standalone HTML or embeddable HTML.

Dependencies:
    folium: Required for interactive maps
"""

from typing import Optional, List, Dict
import json

from core.reporting.maps.base import MapBounds
from core.reporting.utils.color_utils import FLOOD_SEVERITY


class InteractiveMapGenerator:
    """
    Generate interactive web maps using Folium.

    Creates HTML-based maps with zoom/pan controls, layer toggles,
    and interactive tooltips. Suitable for web reports and
    embedded dashboards.

    Examples:
        # Basic interactive flood map
        generator = InteractiveMapGenerator()
        html = generator.generate_flood_map(
            flood_geojson={'type': 'FeatureCollection', 'features': [...]},
            bounds=MapBounds(-82.0, 26.3, -81.5, 26.7)
        )

        # Save to file
        Path("map.html").write_text(html)
    """

    def __init__(self):
        """Initialize interactive map generator."""
        # Check for required dependency
        try:
            import folium
            self.folium = folium
        except ImportError:
            raise RuntimeError(
                "folium is required for interactive maps: pip install folium"
            )

        # Check for optional plugins
        try:
            from folium import plugins
            self.has_plugins = True
            self.plugins = plugins
        except ImportError:
            self.has_plugins = False

    def generate_flood_map(
        self,
        flood_geojson: dict,
        bounds: MapBounds,
        infrastructure: Optional[List[dict]] = None,
        title: str = "Flood Extent",
    ) -> str:
        """
        Generate interactive flood map.

        Args:
            flood_geojson: Flood extent as GeoJSON FeatureCollection
            bounds: Geographic bounds for map centering
            infrastructure: Optional list of infrastructure features with
                          {'lon', 'lat', 'type', 'name', 'address'} keys
            title: Map title

        Returns:
            HTML string containing the interactive map

        Examples:
            flood_geojson = {
                'type': 'FeatureCollection',
                'features': [{
                    'type': 'Feature',
                    'geometry': {'type': 'Polygon', 'coordinates': [...]},
                    'properties': {'severity': 'moderate'}
                }]
            }

            html = generator.generate_flood_map(
                flood_geojson=flood_geojson,
                bounds=MapBounds(-82.0, 26.3, -81.5, 26.7),
                title="Hurricane Ian Flooding"
            )
        """
        # Calculate center
        center = bounds.center

        # Create map with CartoDB Positron base
        m = self.folium.Map(
            location=[center[1], center[0]],  # Folium uses [lat, lon]
            zoom_start=12,
            tiles='CartoDB positron',
            attr='CartoDB'
        )

        # Define severity colors
        severity_colors = {
            'minor': FLOOD_SEVERITY['minor'],
            'moderate': FLOOD_SEVERITY['moderate'],
            'significant': FLOOD_SEVERITY['significant'],
            'severe': FLOOD_SEVERITY['severe'],
            'extreme': FLOOD_SEVERITY['extreme'],
        }

        # Add flood extent layer
        if flood_geojson and 'features' in flood_geojson:
            for feature in flood_geojson['features']:
                severity = feature.get('properties', {}).get('severity', 'moderate')
                color = severity_colors.get(severity, FLOOD_SEVERITY['moderate'])

                self.folium.GeoJson(
                    feature,
                    name="Flood Extent",
                    style_function=lambda x, c=color: {
                        'fillColor': c,
                        'fillOpacity': 0.7,
                        'color': '#2D3748',
                        'weight': 1
                    },
                    tooltip=self.folium.Tooltip(f"Severity: {severity}")
                ).add_to(m)

        # Add infrastructure markers if provided
        if infrastructure:
            self._add_infrastructure_markers(m, infrastructure)

        # Add layer control
        self.folium.LayerControl().add_to(m)

        # Add mouse position display if plugins available
        if self.has_plugins:
            self.plugins.MousePosition().add_to(m)

        # Return HTML
        return m._repr_html_()

    def generate_before_after_slider(
        self,
        before_tile_url: str,
        after_tile_url: str,
        bounds: MapBounds,
    ) -> str:
        """
        Generate before/after comparison with slider.

        Args:
            before_tile_url: URL template for before tiles
            after_tile_url: URL template for after tiles
            bounds: Geographic bounds

        Returns:
            HTML string with slider control

        Note:
            Requires folium.plugins.SideBySideLayers plugin.
            Falls back to simple dual-layer map if not available.
        """
        center = bounds.center

        # Create map
        m = self.folium.Map(
            location=[center[1], center[0]],
            zoom_start=12
        )

        # Add before layer
        before_layer = self.folium.TileLayer(
            tiles=before_tile_url,
            name='Before',
            overlay=True,
            control=True
        )
        before_layer.add_to(m)

        # Add after layer
        after_layer = self.folium.TileLayer(
            tiles=after_tile_url,
            name='After',
            overlay=True,
            control=True
        )
        after_layer.add_to(m)

        # Add side-by-side comparison if plugin available
        if self.has_plugins and hasattr(self.plugins, 'SideBySideLayers'):
            self.plugins.SideBySideLayers(
                layer_left=before_layer,
                layer_right=after_layer
            ).add_to(m)

        # Add layer control
        self.folium.LayerControl().add_to(m)

        return m._repr_html_()

    def _add_infrastructure_markers(
        self,
        map_object,
        infrastructure: List[dict]
    ):
        """
        Add infrastructure markers to map.

        Args:
            map_object: Folium map instance
            infrastructure: List of infrastructure features
        """
        # Icon mapping
        icon_map = {
            'hospital': ('plus', 'red'),
            'school': ('graduation-cap', 'blue'),
            'shelter': ('home', 'green'),
            'fire_station': ('fire-extinguisher', 'orange'),
            'police': ('shield', 'darkblue'),
            'power': ('bolt', 'yellow'),
        }

        for feature in infrastructure:
            lon = feature.get('lon')
            lat = feature.get('lat')

            if lon is None or lat is None:
                continue

            fac_type = feature.get('type', 'other')
            name = feature.get('name', 'Facility')
            address = feature.get('address', '')

            # Get icon config
            icon_name, color = icon_map.get(fac_type, ('info', 'gray'))

            # Create popup content
            popup_html = f"<b>{name}</b>"
            if address:
                popup_html += f"<br>{address}"
            if fac_type:
                popup_html += f"<br><i>{fac_type.replace('_', ' ').title()}</i>"

            # Add marker
            self.folium.Marker(
                location=[lat, lon],  # Folium uses [lat, lon]
                popup=self.folium.Popup(popup_html, max_width=200),
                tooltip=name,
                icon=self.folium.Icon(
                    icon=icon_name,
                    prefix='fa',
                    color=color
                )
            ).add_to(map_object)
