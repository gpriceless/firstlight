"""
STAC Client - Simple, direct STAC catalog access.

This module provides a straightforward interface to STAC catalogs,
bypassing the complex DataBroker architecture for direct data access.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging

from core.data.discovery.band_config import (
    get_band_config,
    get_canonical_order,
    get_qa_layers,
)

logger = logging.getLogger(__name__)


@dataclass
class STACItem:
    """Represents a STAC item (satellite scene)."""

    id: str
    collection: str
    datetime: datetime
    bbox: List[float]
    geometry: Dict[str, Any]
    cloud_cover: Optional[float]
    assets: Dict[str, str]  # asset_key -> URL
    properties: Dict[str, Any]
    # asset_key -> (scale, offset) from the asset's own raster:bands
    asset_scaling: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_pystac(cls, item) -> "STACItem":
        """Create from pystac Item object."""
        # Extract datetime
        dt = item.datetime
        if dt is None:
            dt_str = item.properties.get("datetime") or item.properties.get("start_datetime")
            if dt_str:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            else:
                dt = datetime.now()

        # Extract cloud cover
        cloud_cover = item.properties.get("eo:cloud_cover")

        # Extract asset URLs, preferring a downloadable HTTPS href over a raw
        # s3:// URI. The STAC `alternate-assets` extension exposes both, and
        # Element84's Landsat C2L2 collection only sets s3:// as the primary
        # href (`requests`/`httpx` cannot speak s3://).
        assets = {}
        asset_scaling = {}
        for key, asset in item.assets.items():
            href = _pick_downloadable_href(asset)
            if href:
                assets[key] = href
            bands = (getattr(asset, "extra_fields", None) or {}).get("raster:bands") or []
            if len(bands) == 1 and "scale" in bands[0]:
                asset_scaling[key] = (
                    float(bands[0]["scale"]), float(bands[0].get("offset", 0.0))
                )

        return cls(
            id=item.id,
            collection=item.collection_id,
            datetime=dt,
            bbox=list(item.bbox) if item.bbox else [],
            geometry=item.geometry or {},
            cloud_cover=cloud_cover,
            assets=assets,
            properties=dict(item.properties),
            asset_scaling=asset_scaling,
        )


class STACClient:
    """
    Simple STAC catalog client for satellite data discovery.

    Provides direct access to STAC catalogs without the complexity
    of the DataBroker architecture.

    Example:
        client = STACClient()
        items = client.search_sentinel2(
            bbox=[-121.7, 39.7, -121.5, 39.85],
            start_date="2018-10-01",
            end_date="2018-11-07",
            max_cloud_cover=20
        )
    """

    # Known STAC endpoints
    CATALOGS = {
        "earth_search": "https://earth-search.aws.element84.com/v1",
        "planetary_computer": "https://planetarycomputer.microsoft.com/api/stac/v1",
    }

    # Common collection mappings
    COLLECTIONS = {
        "sentinel2_l2a": {
            "earth_search": "sentinel-2-l2a",
            "planetary_computer": "sentinel-2-l2a",
        },
        "landsat_c2_l2": {
            "earth_search": "landsat-c2-l2",
            "planetary_computer": "landsat-c2-l2",
        },
    }

    def __init__(self, catalog: str = "earth_search"):
        """
        Initialize STAC client.

        Args:
            catalog: Catalog name ("earth_search" or "planetary_computer")
                    or full URL to STAC API endpoint.
        """
        if catalog in self.CATALOGS:
            self.endpoint = self.CATALOGS[catalog]
            self.catalog_name = catalog
        else:
            self.endpoint = catalog
            self.catalog_name = "custom"

        self._client = None
        logger.info(f"STACClient initialized with endpoint: {self.endpoint}")

    def _get_client(self):
        """Lazily initialize pystac_client."""
        if self._client is None:
            try:
                from pystac_client import Client

                self._client = Client.open(self.endpoint)
                logger.info(f"Connected to STAC catalog: {self._client.title}")
            except ImportError:
                raise ImportError(
                    "pystac_client is required for STAC access. "
                    "Install with: pip install pystac-client"
                )
        return self._client

    def search(
        self,
        collections: List[str],
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: Optional[float] = None,
        max_items: int = 100,
    ) -> List[STACItem]:
        """
        Search STAC catalog for items matching criteria.

        Args:
            collections: List of collection IDs to search
            bbox: Bounding box [west, south, east, north]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            max_items: Maximum number of items to return

        Returns:
            List of STACItem objects
        """
        client = self._get_client()

        # Build query parameters
        query = {}
        if max_cloud_cover is not None:
            query["eo:cloud_cover"] = {"lt": max_cloud_cover}

        # Execute search
        search = client.search(
            collections=collections,
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query=query if query else None,
            max_items=max_items,
        )

        # Convert to STACItem objects
        items = []
        for item in search.items():
            try:
                stac_item = STACItem.from_pystac(item)
                items.append(stac_item)
            except Exception as e:
                logger.warning(f"Error processing STAC item {item.id}: {e}")
                continue

        logger.info(f"Found {len(items)} items")
        return items

    def search_sentinel2(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: Optional[float] = 20.0,
        max_items: int = 50,
    ) -> List[STACItem]:
        """
        Search for Sentinel-2 L2A imagery.

        Args:
            bbox: Bounding box [west, south, east, north]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover (default 20%)
            max_items: Maximum items to return

        Returns:
            List of STACItem objects sorted by cloud cover
        """
        collection = self.COLLECTIONS["sentinel2_l2a"].get(
            self.catalog_name, "sentinel-2-l2a"
        )

        items = self.search(
            collections=[collection],
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
        )

        # Sort by cloud cover (ascending)
        items.sort(key=lambda x: x.cloud_cover if x.cloud_cover is not None else 100)

        return items

    def search_landsat(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: float = 20.0,
        max_items: int = 50,
    ) -> List[STACItem]:
        """
        Search for Landsat Collection 2 Level-2 imagery.

        Args:
            bbox: Bounding box [west, south, east, north]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_cloud_cover: Maximum cloud cover (default 20%)
            max_items: Maximum items to return

        Returns:
            List of STACItem objects sorted by cloud cover
        """
        collection = self.COLLECTIONS["landsat_c2_l2"].get(
            self.catalog_name, "landsat-c2-l2"
        )

        items = self.search(
            collections=[collection],
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_cloud_cover=max_cloud_cover,
            max_items=max_items,
        )

        # Sort by cloud cover (ascending)
        items.sort(key=lambda x: x.cloud_cover if x.cloud_cover is not None else 100)

        return items

    def get_best_pair(
        self,
        bbox: List[float],
        event_date: str,
        pre_window_days: int = 30,
        post_window_days: int = 30,
        max_cloud_cover: float = 20.0,
        sensor: str = "sentinel2",
    ) -> Tuple[Optional[STACItem], Optional[STACItem]]:
        """
        Find best pre-event and post-event image pair.

        Args:
            bbox: Bounding box [west, south, east, north]
            event_date: Event date (YYYY-MM-DD)
            pre_window_days: Days before event to search
            post_window_days: Days after event to search
            max_cloud_cover: Maximum cloud cover
            sensor: Sensor type ("sentinel2", "landsat")

        Returns:
            Tuple of (pre_event_item, post_event_item)
        """
        from datetime import timedelta

        event_dt = datetime.strptime(event_date, "%Y-%m-%d")
        pre_start = (event_dt - timedelta(days=pre_window_days)).strftime("%Y-%m-%d")
        pre_end = (event_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        post_start = (event_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        post_end = (event_dt + timedelta(days=post_window_days)).strftime("%Y-%m-%d")

        # Search for pre-event imagery
        if sensor == "sentinel2":
            pre_items = self.search_sentinel2(
                bbox=bbox,
                start_date=pre_start,
                end_date=pre_end,
                max_cloud_cover=max_cloud_cover,
            )
            post_items = self.search_sentinel2(
                bbox=bbox,
                start_date=post_start,
                end_date=post_end,
                max_cloud_cover=max_cloud_cover,
            )
        elif sensor == "landsat":
            pre_items = self.search_landsat(
                bbox=bbox,
                start_date=pre_start,
                end_date=pre_end,
                max_cloud_cover=max_cloud_cover,
            )
            post_items = self.search_landsat(
                bbox=bbox,
                start_date=post_start,
                end_date=post_end,
                max_cloud_cover=max_cloud_cover,
            )
        else:
            raise ValueError(f"Unknown sensor: {sensor}")

        pre_item = pre_items[0] if pre_items else None
        post_item = post_items[0] if post_items else None

        return pre_item, post_item


def _get_asset_href(asset: Optional[Any]) -> str:
    """Safely extract href from a STAC asset."""
    if asset is None:
        return ""
    if hasattr(asset, 'href'):
        return asset.href
    if isinstance(asset, dict) and 'href' in asset:
        return asset['href']
    return ""


_S3_BUCKET_TO_HTTPS = {
    # USGS Landsat C2L2 — public HTTPS mirror of `s3://usgs-landsat/...`. Note
    # the underlying S3 bucket is requester-pays, but the landsatlook HTTPS
    # mirror serves the same objects for unauthenticated reads.
    "usgs-landsat": "https://landsatlook.usgs.gov/data",
}


def _translate_s3_uri(uri: str) -> Optional[str]:
    """Translate a known-public ``s3://bucket/key`` URI to an HTTPS URL.

    Returns ``None`` if no mapping is known — callers should keep the original
    href so it surfaces as a clear, actionable error downstream rather than a
    silent miss.
    """
    if not uri or not uri.startswith("s3://"):
        return None
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    base = _S3_BUCKET_TO_HTTPS.get(bucket)
    if not base or not key:
        return None
    return f"{base}/{key}"


def _pick_downloadable_href(asset: Any) -> str:
    """Pick an HTTPS href for a STAC asset, falling back through alternates.

    Lookup order:
    1. The asset's primary ``href`` if it's already HTTP(S).
    2. ``alternate.https.href`` from the STAC alternate-assets extension —
       Element84's ``landsat-c2-l2`` collection sets this even when the
       primary ``href`` is ``s3://usgs-landsat/...``.
    3. A known ``s3://bucket → HTTPS`` mapping (see ``_S3_BUCKET_TO_HTTPS``).
    4. The original href as a last resort (signals the bug clearly downstream
       instead of swallowing the asset silently).
    """
    primary = _get_asset_href(asset)
    if primary and primary.startswith(("http://", "https://")):
        return primary

    extra_fields: Dict[str, Any] = {}
    if hasattr(asset, "extra_fields") and isinstance(asset.extra_fields, dict):
        extra_fields = asset.extra_fields
    elif isinstance(asset, dict):
        extra_fields = asset

    alternates = extra_fields.get("alternate") if isinstance(extra_fields, dict) else None
    if isinstance(alternates, dict):
        for key in ("https", "http"):
            alt = alternates.get(key)
            if isinstance(alt, dict):
                alt_href = alt.get("href")
                if isinstance(alt_href, str) and alt_href.startswith(("http://", "https://")):
                    return alt_href
        for alt in alternates.values():
            if isinstance(alt, dict):
                alt_href = alt.get("href")
                if isinstance(alt_href, str) and alt_href.startswith(("http://", "https://")):
                    return alt_href

    translated = _translate_s3_uri(primary)
    if translated:
        return translated

    return primary


def _extract_band_urls(item: STACItem, source: str) -> Dict[str, str]:
    """Extract band and quality-layer URLs from STAC item assets."""
    band_config = {**get_band_config(source), **get_qa_layers(source)}
    band_urls = {}

    for band_name, asset_key in band_config.items():
        if asset_key in item.assets:
            band_urls[band_name] = item.assets[asset_key]

    return band_urls


def _reflectance_scaling(item: STACItem, source: str) -> Optional[Dict[str, float]]:
    """Scale and offset that turn the item's band values into reflectance.

    Read from the band assets' own ``raster:bands``. Earth Search Sentinel-2
    L2A declares offset -0.1 but sets ``earthsearch:boa_offset_applied``: the
    pixels already have it removed, so DN * scale is reflectance and the
    offset is not applied again. None when the bands disagree or say nothing.
    """
    scalings = {
        item.asset_scaling[key]
        for key in get_band_config(source).values()
        if key in item.asset_scaling
    }
    if len(scalings) != 1:
        return None
    scale, offset = scalings.pop()
    if item.properties.get("earthsearch:boa_offset_applied"):
        offset = 0.0
    return {"scale": scale, "offset": offset}


def _get_missing_bands(item: STACItem, source: str) -> List[str]:
    """Get list of expected bands that are missing from the item."""
    band_config = get_band_config(source)
    canonical_order = get_canonical_order(source)

    missing = []
    for band_name in canonical_order:
        asset_key = band_config.get(band_name)
        if asset_key and asset_key not in item.assets:
            missing.append(band_name)

    return missing


def _get_primary_url(item: STACItem, source: str) -> str:
    """Get primary URL for backward compatibility (blue band or visual)."""
    # Try to get the blue band first (primary for optical sensors)
    band_config = get_band_config(source)

    if "blue" in band_config:
        blue_asset = band_config["blue"]
        if blue_asset in item.assets:
            return item.assets[blue_asset]

    # Fallback to visual asset
    if "visual" in item.assets:
        return item.assets["visual"]

    # Final fallback to first available asset
    if item.assets:
        return next(iter(item.assets.values()))

    return ""


def _with_same_day_tiles(
    client: "STACClient", items: List[STACItem], bbox: List[float]
) -> List[STACItem]:
    """Add every Sentinel-2 tile over the box from each day already kept.

    Cloud cover is a whole-tile figure, so the filter can keep one tile of a
    day and drop its neighbour even when the box is clear in both. Analysis
    mosaics a day's tiles into one scene and ranks scenes by the clear share
    of the box itself, so a day is taken whole or not at all.
    """
    seen = {item.id for item in items}
    extra: List[STACItem] = []
    for day in sorted({item.datetime.date().isoformat() for item in items}):
        for item in client.search_sentinel2(
            bbox=bbox, start_date=day, end_date=day, max_cloud_cover=None
        ):
            if item.id not in seen and item.datetime.date().isoformat() == day:
                seen.add(item.id)
                extra.append(item)
    if extra:
        logger.info(
            "Added %d same-day Sentinel-2 tile(s) over the cloud limit: %s",
            len(extra), [item.id for item in extra],
        )
    return items + extra


class DiscoveryResults(list):
    """Discovery results that also say what happened to each source.

    ``sources`` maps a source name to ``{"status", "count", "error"}``, where
    status is ``"found"``, ``"empty"`` (the search answered with nothing) or
    ``"failed"`` (the search raised, so the source was never really asked).
    """

    def __init__(self, items=(), sources: Optional[Dict[str, Dict[str, Any]]] = None):
        super().__init__(items)
        self.sources: Dict[str, Dict[str, Any]] = dict(sources or {})


def describe_sources(sources: Dict[str, Dict[str, Any]]) -> List[str]:
    """Plain sentences about each source, failed ones first."""
    lines = []
    for name, info in sources.items():
        if info.get("status") == "failed":
            lines.append(
                f"{name} was not searched: the search failed ({info.get('error')}). "
                f"Nothing from {name} is in this result."
            )
    for name, info in sources.items():
        if info.get("status") == "empty":
            lines.append(f"{name} was searched and has no scenes for this box and window.")
        elif info.get("status") == "found":
            lines.append(f"{name} was searched and returned {info.get('count')} scene(s).")
    return lines


def discover_data(
    bbox: List[float],
    start_date: str,
    end_date: str,
    event_type: str = "flood",
    max_cloud_cover: float = 30.0,
    catalog: str = "earth_search",
) -> List[Dict[str, Any]]:
    """
    Discover available satellite data for an area and time window.

    This is the main entry point for CLI integration.

    Args:
        bbox: Bounding box [west, south, east, north]
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        event_type: Event type (flood, wildfire, storm)
        max_cloud_cover: Maximum cloud cover for optical data
        catalog: STAC catalog to use

    Returns:
        DiscoveryResults: the result dictionaries, with ``sources`` saying
        which sources answered, which answered with nothing and which failed.

    Raises:
        The first source's error when every source failed, so a caller can
        retry or stop rather than read an empty list as "no scenes".
    """
    client = STACClient(catalog=catalog)
    results = []
    report: Dict[str, Dict[str, Any]] = {}
    errors: List[Exception] = []

    # Event-specific data sources
    event_sources = {
        "flood": ["sentinel2"],
        "wildfire": ["sentinel2", "landsat"],
        "storm": ["sentinel2"],
    }

    sources = event_sources.get(event_type.lower(), ["sentinel2"])

    # Search each source
    for source in sources:
        try:
            if source == "sentinel2":
                items = client.search_sentinel2(
                    bbox=bbox,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover,
                )
                items = _with_same_day_tiles(client, items, bbox)
            elif source == "landsat":
                items = client.search_landsat(
                    bbox=bbox,
                    start_date=start_date,
                    end_date=end_date,
                    max_cloud_cover=max_cloud_cover,
                )
            else:
                continue

            # Convert to result dictionaries
            found = []
            for item in items:
                found.append({
                    "id": item.id,
                    "source": source,
                    "datetime": item.datetime.isoformat(),
                    "cloud_cover": item.cloud_cover,
                    "resolution_m": 10.0 if source == "sentinel2" else 30.0,
                    # Unknown: Earth Search assets carry no file:size, and
                    # ingest reads only the part of each band the box needs.
                    "size_bytes": None,
                    # Backward compatibility: primary band or fallback to visual
                    "url": _get_primary_url(item, source),
                    # NEW: Individual band URLs
                    "band_urls": _extract_band_urls(item, source),
                    # NEW: Visualization URL (TCI)
                    "visualization_url": item.assets.get("visual", ""),
                    # NEW: Expected but missing bands
                    "missing_bands": _get_missing_bands(item, source),
                    "reflectance_scaling": _reflectance_scaling(item, source),
                    "priority": "primary" if source == "sentinel2" else "secondary",
                    "bbox": item.bbox,
                    "assets": item.assets,
                })
            results.extend(found)
            report[source] = {
                "status": "found" if found else "empty",
                "count": len(found),
                "error": None,
            }

        except Exception as e:
            logger.warning(f"Error searching {source}: {e}")
            report[source] = {"status": "failed", "count": 0, "error": str(e)}
            errors.append(e)
            continue

    if errors and len(errors) == len(report):
        raise errors[0]

    # Sort by date
    results.sort(key=lambda x: x.get("datetime", ""))

    return DiscoveryResults(results, sources=report)
