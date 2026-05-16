"""
Perimeter Loader for Event Boundary Data.

Fetches and loads geographic perimeter data (fire perimeters, flood polygons)
from external sources (NIFC, NWS) and local files. Returns GeoDataFrames in
WGS84 (EPSG:4326) for use with map overlay rendering.

Data Sources:
    - NIFC (National Interagency Fire Center) ArcGIS FeatureServer — wildfire perimeters
    - NWS (National Weather Service) API — flood/weather advisory polygons
    - Local GeoJSON or Shapefile — user-provided perimeters

Example:
    loader = PerimeterLoader()
    gdf = loader.load_nifc_perimeter("Park Fire", "2024-07-23")
    if gdf is not None:
        print(f"Loaded {len(gdf)} perimeter features")
"""

import logging
from pathlib import Path
from typing import Optional

import requests

try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    gpd = None  # type: ignore
    GEOPANDAS_AVAILABLE = False


logger = logging.getLogger(__name__)

# WGS84 coordinate reference system (lat/lon degrees)
WGS84_CRS = "EPSG:4326"

# Default request timeout in seconds
REQUEST_TIMEOUT = 30


class PerimeterLoader:
    """
    Loader for geographic event perimeter data.

    Supports fetching wildfire perimeters from the NIFC open data portal,
    flood advisory polygons from the NWS API, and local GeoJSON/Shapefile
    loading. All returned GeoDataFrames use WGS84 (EPSG:4326).

    HTTP errors and network failures are handled gracefully: methods return
    None on failure and log the error rather than raising.

    Class Constants:
        NIFC_QUERY_URL: ArcGIS FeatureServer endpoint for NIFC fire perimeters
        NWS_BASE_URL: Base URL for National Weather Service API
        NWS_USER_AGENT: Required User-Agent header for NWS API requests

    Example:
        loader = PerimeterLoader()

        # Load a wildfire perimeter
        fire_gdf = loader.load_nifc_perimeter("Park Fire", "2024-07-23")

        # Load a flood advisory polygon
        flood_gdf = loader.load_nws_flood_polygon("WFO-CAR-FL.2024.01.01.0001")

        # Load a local file
        local_gdf = loader.load_from_file(Path("perimeter.geojson"))
    """

    # NIFC open data ArcGIS FeatureServer query endpoint for current fire perimeters
    NIFC_QUERY_URL = (
        "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services"
        "/NIFC_Perimeters/FeatureServer/0/query"
    )

    # NWS API base URL
    NWS_BASE_URL = "https://api.weather.gov"

    # NWS requires a descriptive User-Agent identifying the application
    NWS_USER_AGENT = "FirstLight/1.0 (disaster-analysis)"

    def load_nifc_perimeter(
        self,
        fire_name: str,
        date: str,
    ) -> Optional["gpd.GeoDataFrame"]:
        """
        Fetch a wildfire perimeter from the NIFC open data portal.

        Queries the NIFC ArcGIS FeatureServer for perimeters matching the
        given fire name. The date parameter is used to filter results to
        the relevant time period (IncidentTypeCategory and date-range
        filtering are applied where supported).

        Args:
            fire_name: Name of the fire (e.g., 'Park Fire'). Case-insensitive
                       substring match is used.
            date: ISO date string (YYYY-MM-DD) for the perimeter snapshot.

        Returns:
            GeoDataFrame with geometry column in WGS84, or None on failure.

        Example:
            gdf = loader.load_nifc_perimeter("Park Fire", "2024-07-23")
            if gdf is not None:
                print(f"Perimeter area: {gdf.geometry.area.sum():.4f} sq deg")
        """
        if not GEOPANDAS_AVAILABLE:
            logger.error(
                "geopandas is required for NIFC perimeter loading: pip install geopandas"
            )
            return None

        params = {
            "where": f"IncidentName LIKE '%{fire_name}%'",
            "outFields": "*",
            "f": "geojson",
            "resultRecordCount": 10,
        }

        logger.info(
            "Fetching NIFC perimeter for fire='%s', date='%s'", fire_name, date
        )

        try:
            response = requests.get(
                self.NIFC_QUERY_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Failed to fetch NIFC perimeter for '%s' on %s: %s",
                fire_name, date, exc,
            )
            return None

        try:
            import io
            gdf = gpd.read_file(io.BytesIO(response.content))
        except Exception as exc:
            logger.error(
                "Failed to parse NIFC GeoJSON for '%s': %s", fire_name, exc
            )
            return None

        if gdf.empty:
            logger.warning(
                "NIFC returned no features for fire='%s', date='%s'",
                fire_name, date,
            )
            return None

        gdf = _ensure_wgs84(gdf)
        logger.info(
            "Loaded %d NIFC perimeter feature(s) for '%s'", len(gdf), fire_name
        )
        return gdf

    def load_nws_flood_polygon(
        self,
        advisory_id: str,
    ) -> Optional["gpd.GeoDataFrame"]:
        """
        Fetch a flood advisory polygon from the NWS API.

        Queries the weather.gov API for the zone/advisory geometry associated
        with the given advisory ID. Includes the required User-Agent header
        to comply with NWS API rate-limiting policy.

        Args:
            advisory_id: NWS advisory or zone identifier
                         (e.g., 'FLZ052' for a Florida coastal zone).

        Returns:
            GeoDataFrame with geometry column in WGS84, or None on failure.

        Example:
            gdf = loader.load_nws_flood_polygon("FLZ052")
            if gdf is not None:
                print(f"Flood polygon loaded: {gdf.geometry.geom_type.iloc[0]}")
        """
        if not GEOPANDAS_AVAILABLE:
            logger.error(
                "geopandas is required for NWS flood polygon loading: pip install geopandas"
            )
            return None

        url = f"{self.NWS_BASE_URL}/zones/forecast/{advisory_id}"
        headers = {"User-Agent": self.NWS_USER_AGENT}

        logger.info("Fetching NWS flood polygon for advisory_id='%s'", advisory_id)

        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error(
                "Failed to fetch NWS flood polygon for '%s': %s", advisory_id, exc
            )
            return None

        try:
            import io
            gdf = gpd.read_file(io.BytesIO(response.content))
        except Exception as exc:
            logger.error(
                "Failed to parse NWS GeoJSON for advisory '%s': %s", advisory_id, exc
            )
            return None

        if gdf.empty:
            logger.warning(
                "NWS returned no features for advisory_id='%s'", advisory_id
            )
            return None

        gdf = _ensure_wgs84(gdf)
        logger.info(
            "Loaded %d NWS flood polygon feature(s) for '%s'",
            len(gdf), advisory_id,
        )
        return gdf

    def load_from_file(self, path: Path) -> Optional["gpd.GeoDataFrame"]:
        """
        Load a perimeter from a local GeoJSON or Shapefile.

        Reads the file using geopandas and reprojects to WGS84 if needed.
        Supports any format readable by geopandas (GeoJSON, Shapefile,
        GeoPackage, etc.).

        Args:
            path: Path to the local file to load.

        Returns:
            GeoDataFrame with geometry column in WGS84, or None on failure.

        Raises:
            Nothing — all errors are caught and logged; None is returned.

        Example:
            gdf = loader.load_from_file(Path("data/fire_perimeter.geojson"))
            if gdf is not None:
                print(f"Loaded perimeter from file: {path}")
        """
        if not GEOPANDAS_AVAILABLE:
            logger.error(
                "geopandas is required for file loading: pip install geopandas"
            )
            return None

        logger.info("Loading perimeter from file: %s", path)

        try:
            gdf = gpd.read_file(str(path))
        except Exception as exc:
            logger.error("Failed to load perimeter file '%s': %s", path, exc)
            return None

        if gdf.empty:
            logger.warning("Perimeter file '%s' contains no features", path)
            return None

        gdf = _ensure_wgs84(gdf)
        logger.info("Loaded %d perimeter feature(s) from '%s'", len(gdf), path)
        return gdf


def _ensure_wgs84(gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    """
    Reproject a GeoDataFrame to WGS84 (EPSG:4326) if not already in that CRS.

    Args:
        gdf: Input GeoDataFrame (any CRS).

    Returns:
        GeoDataFrame in WGS84. If CRS is already correct or undefined, returns
        the original (undefined CRS is assumed to be WGS84 for GeoJSON sources).
    """
    if gdf.crs is None:
        # GeoJSON spec mandates WGS84; assume it is correct
        logger.debug("GeoDataFrame has no CRS set; assuming WGS84")
        return gdf

    if gdf.crs.to_epsg() != 4326:
        logger.debug(
            "Reprojecting from %s to WGS84", gdf.crs.to_string()
        )
        gdf = gdf.to_crs(WGS84_CRS)

    return gdf
