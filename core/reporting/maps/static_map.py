"""
Static map generation using matplotlib and contextily.

Generates print-quality PNG/PDF maps with base layers, flood extent,
infrastructure overlays, and map furniture (scale bar, north arrow, legend).

Dependencies:
    matplotlib: Required
    cartopy: Required for projection support
    contextily: Optional, for base map tiles
    matplotlib-scalebar: Optional, for scale bars
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from core.reporting.maps.base import MapConfig, MapBounds, MapType, MapOutputPreset
from core.reporting.utils.color_utils import FLOOD_SEVERITY


class StaticMapGenerator:
    """
    Generate static map images for reports.

    Creates print-quality maps with proper projections, base layers,
    and cartographic elements. Supports flood extent visualization,
    infrastructure overlays, and before/after comparisons.

    Examples:
        # Basic flood map
        generator = StaticMapGenerator()
        generator.generate_flood_map(
            flood_extent=flood_array,
            bounds=MapBounds(-82.0, 26.3, -81.5, 26.7),
            output_path=Path("flood_map.png")
        )

        # High-resolution print map
        config = MapConfig(width=3300, height=2550, dpi=300)
        generator = StaticMapGenerator(config)
    """

    def __init__(self, config: MapConfig = None):
        """
        Initialize map generator.

        Args:
            config: Map configuration. Defaults to standard screen resolution.
        """
        self.config = config or MapConfig()

        # Check for required dependencies
        try:
            import matplotlib
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError(
                "matplotlib is required for static maps: pip install matplotlib"
            )

        self.plt = plt
        self._check_optional_dependencies()

    def _check_optional_dependencies(self):
        """Check for optional dependencies and set availability flags."""
        # Check for cartopy (projection support)
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            self.has_cartopy = True
            self.ccrs = ccrs
            self.cfeature = cfeature
        except ImportError:
            self.has_cartopy = False
            print("Warning: cartopy not available. Install for projection support: pip install cartopy")

        # Check for contextily (base maps)
        try:
            import contextily as ctx
            self.has_contextily = True
            self.ctx = ctx
        except ImportError:
            self.has_contextily = False
            print("Warning: contextily not available. Install for base maps: pip install contextily")

        # Check for matplotlib-scalebar
        try:
            from matplotlib_scalebar.scalebar import ScaleBar
            self.has_scalebar = True
            self.ScaleBar = ScaleBar
        except ImportError:
            self.has_scalebar = False
            print("Warning: matplotlib-scalebar not available. Install for scale bars: pip install matplotlib-scalebar")

        # Check for rasterio (CRS reading)
        try:
            import rasterio
            self.has_rasterio = True
            self.rasterio = rasterio
        except ImportError:
            self.has_rasterio = False

    def generate_flood_map(
        self,
        flood_extent: np.ndarray,
        bounds: MapBounds,
        output_path: Path,
        title: Optional[str] = None,
        infrastructure: Optional[List[Dict]] = None,
        source_crs: Optional[str] = None,
    ) -> Path:
        """
        Generate flood extent map with base layer.

        Args:
            flood_extent: Boolean array of flood extent (True = flooded)
            bounds: Geographic bounds of the map
            output_path: Where to save the map image
            title: Optional map title (overrides config.title)
            infrastructure: Optional list of infrastructure features with
                           {'lon', 'lat', 'type', 'name'} keys
            source_crs: Optional CRS string for the source data. Overrides
                        config.source_crs. Used for projection selection.

        Returns:
            Path to generated map image

        Examples:
            flood_extent = np.array([[False, True], [True, True]])
            bounds = MapBounds(-82.0, 26.3, -81.5, 26.7)

            generator.generate_flood_map(
                flood_extent=flood_extent,
                bounds=bounds,
                output_path=Path("flood.png"),
                title="Hurricane Ian Flood Extent"
            )
        """
        # Resolve CRS: kwarg > config field
        effective_crs = source_crs or self.config.source_crs

        # Set up figure
        fig_width = self.config.width / self.config.dpi
        fig_height = self.config.height / self.config.dpi

        fig, ax = self.plt.subplots(
            figsize=(fig_width, fig_height),
            dpi=self.config.dpi
        )

        # Add base map if available
        if self.has_contextily and self.has_cartopy:
            self._add_basemap_with_projection(ax, bounds, source_crs=effective_crs)
        else:
            # Simple plot without base map
            ax.set_xlim(bounds.min_lon, bounds.max_lon)
            ax.set_ylim(bounds.min_lat, bounds.max_lat)
            ax.set_aspect('equal')

        # Plot flood extent
        self._plot_flood_extent(ax, flood_extent, bounds)

        # Add infrastructure if provided
        if infrastructure:
            self._plot_infrastructure(ax, infrastructure)

        # Add map furniture
        if self.config.show_scale_bar:
            self._add_scale_bar(ax, bounds)

        if self.config.show_north_arrow:
            self._add_north_arrow(ax)

        if self.config.show_legend:
            self._add_legend(ax, [
                {'label': 'Flooded Area', 'color': FLOOD_SEVERITY['moderate']}
            ])

        # Add title block (replaces simple ax.set_title)
        self._add_title_block(ax, override_title=title)

        # Add attribution block (replaces single-line text)
        self._add_attribution_block(ax)

        # Save
        self.plt.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        self.plt.close()

        return output_path

    def generate_infrastructure_map(
        self,
        infrastructure: List[Dict],
        bounds: MapBounds,
        flood_extent: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        source_crs: Optional[str] = None,
    ) -> Path:
        """
        Generate infrastructure overlay map.

        Args:
            infrastructure: List of infrastructure features
            bounds: Geographic bounds
            flood_extent: Optional flood extent to show context
            output_path: Where to save (required)
            source_crs: Optional CRS string for the source data. Overrides
                        config.source_crs. Used for projection selection.

        Returns:
            Path to generated map
        """
        if output_path is None:
            raise ValueError("output_path is required")

        # Resolve CRS: kwarg > config field
        effective_crs = source_crs or self.config.source_crs

        # Set up figure
        fig_width = self.config.width / self.config.dpi
        fig_height = self.config.height / self.config.dpi

        fig, ax = self.plt.subplots(
            figsize=(fig_width, fig_height),
            dpi=self.config.dpi
        )

        # Add base map
        if self.has_contextily and self.has_cartopy:
            self._add_basemap_with_projection(ax, bounds, source_crs=effective_crs)
        else:
            ax.set_xlim(bounds.min_lon, bounds.max_lon)
            ax.set_ylim(bounds.min_lat, bounds.max_lat)
            ax.set_aspect('equal')

        # Add flood extent with transparency if provided
        if flood_extent is not None:
            self._plot_flood_extent(ax, flood_extent, bounds, alpha=0.3)

        # Plot infrastructure
        self._plot_infrastructure(ax, infrastructure)

        # Add map furniture
        if self.config.show_scale_bar:
            self._add_scale_bar(ax, bounds)

        if self.config.show_north_arrow:
            self._add_north_arrow(ax)

        # Add title block
        self._add_title_block(ax)

        # Add attribution block
        self._add_attribution_block(ax)

        # Save
        self.plt.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        self.plt.close()

        return output_path

    def generate_before_after(
        self,
        before_image: np.ndarray,
        after_image: np.ndarray,
        bounds: MapBounds,
        output_path: Path,
        before_date: str,
        after_date: str,
    ) -> Path:
        """
        Generate side-by-side before/after comparison.

        Args:
            before_image: Pre-event imagery (H x W x 3 RGB)
            after_image: Post-event imagery (H x W x 3 RGB)
            bounds: Geographic bounds (same for both)
            output_path: Where to save
            before_date: Date string for before image
            after_date: Date string for after image

        Returns:
            Path to generated comparison map
        """
        # Create side-by-side figure
        fig_width = (self.config.width * 2 + 32) / self.config.dpi
        fig_height = self.config.height / self.config.dpi

        fig, (ax1, ax2) = self.plt.subplots(
            1, 2,
            figsize=(fig_width, fig_height),
            dpi=self.config.dpi
        )

        # Plot before image
        ax1.imshow(before_image, extent=bounds.bbox, aspect='auto')
        ax1.set_title(f"BEFORE\n{before_date}", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot after image
        ax2.imshow(after_image, extent=bounds.bbox, aspect='auto')
        ax2.set_title(f"AFTER\n{after_date}", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Add shared scale bar to first plot
        if self.config.show_scale_bar:
            self._add_scale_bar(ax1, bounds)

        # Save
        self.plt.tight_layout()
        self.plt.savefig(
            output_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            facecolor='white'
        )
        self.plt.close()

        return output_path

    def _plot_flood_extent(
        self,
        ax,
        flood_extent: np.ndarray,
        bounds: MapBounds,
        alpha: float = 0.7
    ):
        """Plot flood extent on axis."""
        # Create masked array where True = flooded
        masked_flood = np.ma.masked_where(~flood_extent, flood_extent)

        # Plot with flood color
        ax.imshow(
            masked_flood,
            extent=bounds.bbox,
            cmap=self.plt.cm.Blues,
            alpha=alpha,
            interpolation='nearest'
        )

    def _plot_infrastructure(self, ax, infrastructure: List[Dict]):
        """Plot infrastructure points on axis."""
        # Infrastructure colors by type
        colors = {
            'hospital': '#E53E3E',
            'school': '#3182CE',
            'shelter': '#38A169',
            'fire_station': '#F97316',
            'default': '#718096'
        }

        for feature in infrastructure:
            lon = feature.get('lon')
            lat = feature.get('lat')
            fac_type = feature.get('type', 'default')

            if lon is None or lat is None:
                continue

            color = colors.get(fac_type, colors['default'])

            ax.plot(
                lon, lat,
                marker='o',
                markersize=8,
                color=color,
                markeredgecolor='white',
                markeredgewidth=2,
                zorder=10
            )

    def _detect_crs_from_raster(self, raster_path: str) -> Optional[str]:
        """
        Read CRS from a GeoTIFF file using rasterio.

        Args:
            raster_path: Path to GeoTIFF file

        Returns:
            CRS string (e.g. 'EPSG:4326') or None if unreadable
        """
        if not self.has_rasterio:
            return None
        try:
            with self.rasterio.open(raster_path) as ds:
                crs = ds.crs
                if crs is not None:
                    return crs.to_string()
        except Exception as e:
            print(f"Warning: Could not read CRS from {raster_path}: {e}")
        return None

    def _choose_projection(self, bounds: MapBounds, source_crs: Optional[str] = None):
        """
        Choose an appropriate display projection.

        Uses UTM for small extents (<100 km) and Web Mercator for display.
        Always returns a cartopy CRS object for use with contextily.

        Args:
            bounds: Geographic bounds of the area
            source_crs: Optional CRS hint from source data (currently used as
                        metadata; display projection is based on extent size)

        Returns:
            cartopy CRS object for the display projection, or None if cartopy
            is not available
        """
        if not self.has_cartopy:
            return None

        # Estimate spatial extent in km using a rough conversion
        # 1 degree latitude ≈ 111 km; longitude varies with latitude
        center_lat = (bounds.min_lat + bounds.max_lat) / 2
        lat_km = abs(bounds.max_lat - bounds.min_lat) * 111.0
        lon_km = abs(bounds.max_lon - bounds.min_lon) * 111.0 * abs(np.cos(np.radians(center_lat)))
        max_extent_km = max(lat_km, lon_km)

        if max_extent_km < 100:
            # Small local extent: use UTM for accurate distance representation
            # Calculate UTM zone from center longitude
            utm_zone = int((bounds.center[0] + 180) / 6) + 1
            hemisphere = "north" if center_lat >= 0 else "south"
            try:
                if hemisphere == "north":
                    return self.ccrs.UTM(zone=utm_zone, southern_hemisphere=False)
                else:
                    return self.ccrs.UTM(zone=utm_zone, southern_hemisphere=True)
            except Exception:
                # Fall back to Mercator if UTM fails
                pass

        # Default: Web Mercator for display compatibility with tile providers
        return self.ccrs.epsg(3857)

    def _add_basemap_with_projection(
        self,
        ax,
        bounds: MapBounds,
        source_crs: Optional[str] = None,
        raster_path: Optional[str] = None,
    ):
        """
        Add base map tiles with CRS auto-detection and geographic axes.

        Reads CRS from a raster file if provided. Falls back gracefully if
        contextily or cartopy are unavailable.

        Args:
            ax: matplotlib axes (GeoAxes when cartopy is available)
            bounds: WGS84 geographic bounds
            source_crs: Optional pre-resolved CRS string. Skips raster detection.
            raster_path: Optional path to GeoTIFF for CRS auto-detection.
        """
        if not (self.has_contextily and self.has_cartopy):
            return

        # Auto-detect CRS from raster if not already supplied
        if source_crs is None and raster_path is not None:
            source_crs = self._detect_crs_from_raster(raster_path)

        try:
            # Set extent in WGS84
            ax.set_extent(bounds.bbox, crs=self.ccrs.PlateCarree())

            # Add geographic gridlines (lat/lon labels)
            try:
                gl = ax.gridlines(
                    crs=self.ccrs.PlateCarree(),
                    draw_labels=True,
                    linewidth=0.5,
                    color='gray',
                    alpha=0.5,
                    linestyle='--',
                )
                gl.top_labels = False
                gl.right_labels = False
            except Exception:
                # gridlines may fail for some projections; non-fatal
                pass

            # Add base map tiles via contextily
            self._get_basemap_tiles(bounds, ax=ax)

        except Exception as e:
            print(f"Warning: Could not add basemap: {e}")

    def _get_basemap_tiles(
        self,
        bounds: MapBounds,
        ax=None,
    ) -> Optional[Tuple]:
        """
        Fetch and apply base map tiles using contextily.

        Primary provider: CartoDB.Positron (clean, light background).
        Fallback provider: OpenStreetMap.Mapnik (if primary fetch fails).

        Args:
            bounds: Geographic bounds (WGS84). Used only when ax is None.
            ax: matplotlib/cartopy Axes to add the basemap to. When provided,
                contextily.add_basemap() is called directly on the axes.

        Returns:
            Tuple of (image_array, extent) when fetching tiles without axes,
            or None when adding directly to axes.
        """
        if not self.has_contextily:
            return None

        primary_source = self.ctx.providers.CartoDB.Positron
        fallback_source = self.ctx.providers.OpenStreetMap.Mapnik

        if ax is not None:
            # Add tiles directly onto existing axes
            for source in (primary_source, fallback_source):
                try:
                    self.ctx.add_basemap(
                        ax,
                        crs=self.ccrs.PlateCarree() if self.has_cartopy else "EPSG:4326",
                        source=source,
                        zoom='auto',
                    )
                    return None  # success
                except Exception as e:
                    print(f"Warning: Basemap tile fetch failed ({source.name if hasattr(source, 'name') else source}): {e}")
            return None

        # Standalone tile fetch (no axes)
        try:
            img, extent = self.ctx.bounds2img(
                bounds.min_lon, bounds.min_lat,
                bounds.max_lon, bounds.max_lat,
                zoom='auto',
                source=primary_source,
                ll=True,  # input is lon/lat (WGS84)
            )
            return (img, extent)
        except Exception as e:
            print(f"Warning: Primary tile fetch failed: {e}. Trying fallback.")
            try:
                img, extent = self.ctx.bounds2img(
                    bounds.min_lon, bounds.min_lat,
                    bounds.max_lon, bounds.max_lat,
                    zoom='auto',
                    source=fallback_source,
                    ll=True,
                )
                return (img, extent)
            except Exception as e2:
                print(f"Warning: Fallback tile fetch also failed: {e2}")
                return None

    def _add_scale_bar(self, ax, bounds: MapBounds):
        """Add scale bar to map."""
        if not self.has_scalebar:
            # Simple text-based scale bar as fallback
            scale_text = "Scale: varies by latitude"
            ax.text(
                0.05, 0.05,
                scale_text,
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            return

        # Calculate approximate meters per pixel at center latitude
        center_lat = (bounds.min_lat + bounds.max_lat) / 2
        meters_per_degree = 111320 * np.cos(np.radians(center_lat))

        # Add scale bar
        scalebar = self.ScaleBar(
            meters_per_degree,
            location='lower left',
            length_fraction=0.2,
            height_fraction=0.01,
            pad=0.5,
            color='#4A5568',
            box_alpha=0.8,
            font_properties={'size': 10, 'weight': 500}
        )
        ax.add_artist(scalebar)

    def _add_north_arrow(self, ax):
        """Add north arrow to map."""
        # Simple north arrow using text
        ax.annotate(
            'N',
            xy=(0.95, 0.95),
            xycoords='axes fraction',
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round',
                facecolor='white',
                alpha=0.8,
                edgecolor='#4A5568'
            )
        )

        # Add arrow
        ax.annotate(
            '',
            xy=(0.95, 0.98),
            xycoords='axes fraction',
            xytext=(0.95, 0.92),
            arrowprops=dict(
                arrowstyle='->',
                lw=2,
                color='#4A5568'
            )
        )

    def _add_legend(self, ax, items: List[Dict]):
        """
        Add legend to map.

        Args:
            items: List of dicts with 'label' and 'color' keys
        """
        from matplotlib.patches import Patch

        patches = [
            Patch(facecolor=item['color'], label=item['label'])
            for item in items
        ]

        ax.legend(
            handles=patches,
            loc='lower right',
            framealpha=0.9,
            fontsize=9
        )

    def _add_title_block(self, ax, override_title: Optional[str] = None):
        """
        Add a structured title block to the map.

        Renders event name, location, and date from MapConfig as a multi-line
        title. Falls back to a simple title if only config.title is set.

        Args:
            ax: matplotlib axes
            override_title: Optional title string that overrides config fields.
        """
        # Build title lines from config metadata
        title_lines = []

        # Determine main title: override kwarg > event_name > config.title
        main_title = override_title or self.config.event_name or self.config.title
        if main_title:
            title_lines.append(main_title)

        # Location and date as subtitle
        subtitle_parts = []
        if self.config.location:
            subtitle_parts.append(self.config.location)
        if self.config.event_date:
            subtitle_parts.append(self.config.event_date)

        if subtitle_parts:
            title_lines.append(" | ".join(subtitle_parts))

        if not title_lines:
            return

        # Render: first line is the primary title, second is subtitle
        if len(title_lines) >= 2:
            # Two-line title block
            ax.set_title(
                title_lines[0],
                fontsize=14,
                fontweight='bold',
                pad=4,
                loc='center',
            )
            # Subtitle placed just above the axes via suptitle-style text
            ax.text(
                0.5, 1.01,
                title_lines[1],
                transform=ax.transAxes,
                fontsize=10,
                color='#4A5568',
                ha='center',
                va='bottom',
            )
        else:
            ax.set_title(
                title_lines[0],
                fontsize=14,
                fontweight='bold',
                pad=10,
            )

    def _add_attribution_block(self, ax):
        """
        Add a structured attribution block to the map.

        Renders satellite platform, acquisition date, processing level, and
        data source from MapConfig. Falls back to the legacy attribution string
        if no structured fields are set.

        Args:
            ax: matplotlib axes
        """
        parts = []

        if self.config.satellite_platform:
            parts.append(self.config.satellite_platform)
        if self.config.acquisition_date:
            parts.append(f"Acquired: {self.config.acquisition_date}")
        if self.config.processing_level:
            parts.append(f"Level: {self.config.processing_level}")
        if self.config.data_source:
            parts.append(self.config.data_source)

        if not parts:
            # Fall back to legacy attribution string
            attribution_text = self.config.attribution
        else:
            attribution_text = "  |  ".join(parts)
            # Append base attribution if it's not the default and we have metadata
            if self.config.attribution and self.config.attribution != "© OpenStreetMap contributors":
                attribution_text = f"{attribution_text}  |  {self.config.attribution}"

        ax.text(
            0.99, 0.01,
            attribution_text,
            transform=ax.transAxes,
            fontsize=8,
            color='#A0AEC0',
            ha='right',
            va='bottom',
        )
