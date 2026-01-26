"""
OpenStreetMap Infrastructure Client.

Provides access to infrastructure data from OpenStreetMap via the Overpass API.
Enables querying of critical facilities like hospitals, schools, fire stations,
police stations, shelters, and utilities within disaster-affected areas.

The Overpass API is a read-only API that allows querying OpenStreetMap data
with custom filters and spatial constraints.

Overpass API Documentation:
    https://wiki.openstreetmap.org/wiki/Overpass_API
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False

try:
    import shapely.geometry
    from shapely.geometry import Point, box, shape
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    shapely = None  # type: ignore
    SHAPELY_AVAILABLE = False


logger = logging.getLogger(__name__)


class InfrastructureType(Enum):
    """Types of infrastructure facilities."""

    HOSPITAL = "hospital"
    SCHOOL = "school"
    FIRE_STATION = "fire_station"
    POLICE = "police"
    SHELTER = "shelter"
    POWER_STATION = "power"
    WATER_TREATMENT = "water"


# Mapping from InfrastructureType to OSM query tags
OSM_TAGS = {
    InfrastructureType.HOSPITAL: [
        ('amenity', 'hospital'),
        ('amenity', 'clinic'),
    ],
    InfrastructureType.SCHOOL: [
        ('amenity', 'school'),
        ('amenity', 'university'),
        ('amenity', 'college'),
    ],
    InfrastructureType.FIRE_STATION: [
        ('amenity', 'fire_station'),
    ],
    InfrastructureType.POLICE: [
        ('amenity', 'police'),
    ],
    InfrastructureType.SHELTER: [
        ('amenity', 'shelter'),
        ('emergency', 'assembly_point'),
    ],
    InfrastructureType.POWER_STATION: [
        ('power', 'plant'),
        ('power', 'station'),
        ('power', 'substation'),
    ],
    InfrastructureType.WATER_TREATMENT: [
        ('man_made', 'water_works'),
        ('man_made', 'wastewater_plant'),
    ],
}


@dataclass
class InfrastructureFeature:
    """A single infrastructure feature from OpenStreetMap."""

    id: str
    type: InfrastructureType
    name: str
    lat: float
    lon: float
    address: Optional[str] = None
    phone: Optional[str] = None
    capacity: Optional[int] = None

    def to_geojson_feature(self) -> Dict:
        """
        Convert to GeoJSON Feature.

        Returns:
            GeoJSON Feature dict with Point geometry and properties
        """
        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [self.lon, self.lat]
            },
            "properties": {
                "id": self.id,
                "type": self.type.value,
                "name": self.name,
                "address": self.address,
                "phone": self.phone,
                "capacity": self.capacity,
            }
        }


class InfrastructureClient:
    """
    Client for querying infrastructure from OpenStreetMap.

    Uses the Overpass API to retrieve infrastructure data for disaster
    response and impact assessment. Supports querying by bounding box
    and finding facilities within flood zones or other hazard areas.

    Example:
        async with InfrastructureClient() as client:
            # Query by bounding box
            bbox = (-80.87, 25.13, -80.12, 25.97)
            hospitals = await client.query_by_bbox(
                bbox,
                types=[InfrastructureType.HOSPITAL]
            )

            # Find facilities in flood zone
            result = await client.find_in_flood_zone(
                flood_geojson={"type": "Polygon", "coordinates": [...]},
                types=[InfrastructureType.HOSPITAL, InfrastructureType.SHELTER]
            )
            print(f"Found {len(result['in_flood_zone'])} facilities in flood zone")

    Rate Limits:
        The Overpass API has rate limits and can be slow for large queries.
        This client implements caching and exponential backoff to handle
        rate limits gracefully.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    CACHE_DIR = Path.home() / ".cache" / "firstlight" / "infrastructure"

    # Retry configuration for rate limiting
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 2.0  # seconds
    MAX_BACKOFF = 60.0  # seconds

    def __init__(self, cache_ttl: int = 7776000):
        """
        Initialize Infrastructure client.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 90 days)
        """
        self.cache_ttl = cache_ttl
        self._session: Optional[aiohttp.ClientSession] = None

        # Create cache directory
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "InfrastructureClient initialized (cache_ttl=%ds, cache_dir=%s)",
            cache_ttl, self.CACHE_DIR
        )

    async def __aenter__(self):
        """Async context manager entry."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required but not installed")
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
            self._session = None

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.CACHE_DIR / f"{key_hash}.json"

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if valid."""
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)

            # Check if cache is still valid
            cached_time = cached.get("cached_at", 0)
            if time.time() - cached_time > self.cache_ttl:
                logger.debug("Cache expired: %s", cache_key)
                cache_path.unlink()
                return None

            logger.debug("Cache hit: %s", cache_key)
            return cached.get("data")

        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read cache %s: %s", cache_key, e)
            return None

    def _cache_response(self, cache_key: str, data: Dict) -> None:
        """Cache response to disk."""
        cache_path = self._get_cache_path(cache_key)

        try:
            cached = {
                "cached_at": time.time(),
                "cache_key": cache_key,
                "data": data
            }
            with open(cache_path, "w") as f:
                json.dump(cached, f)
            logger.debug("Cached response: %s", cache_key)

        except OSError as e:
            logger.warning("Failed to cache response %s: %s", cache_key, e)

    def _build_overpass_query(
        self,
        bbox: Tuple[float, float, float, float],
        types: Optional[List[InfrastructureType]] = None
    ) -> str:
        """
        Build Overpass QL query for infrastructure types.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            types: List of infrastructure types (default: all)

        Returns:
            Overpass QL query string
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

        # Default to all types
        if types is None:
            types = list(InfrastructureType)

        # Build query parts for each type
        query_parts = []
        for infra_type in types:
            tags = OSM_TAGS.get(infra_type, [])
            for key, value in tags:
                # Query both nodes and ways
                query_parts.append(f'  node["{key}"="{value}"]({bbox_str});')
                query_parts.append(f'  way["{key}"="{value}"]({bbox_str});')

        # Combine into complete query
        query = "[out:json][timeout:60];\n(\n"
        query += "\n".join(query_parts)
        query += "\n);\nout center;"

        return query

    async def _fetch_overpass(self, query: str) -> Dict:
        """
        Fetch data from Overpass API with retry logic.

        Args:
            query: Overpass QL query string

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If request fails after all retries
        """
        # Create cache key from query
        cache_key = f"overpass:{query}"

        # Check cache first
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        # Ensure we have a session
        if self._session is None:
            if not AIOHTTP_AVAILABLE:
                raise RuntimeError("aiohttp is required but not installed")
            self._session = aiohttp.ClientSession()

        # Retry loop with exponential backoff
        backoff = self.INITIAL_BACKOFF
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with self._session.post(
                    self.OVERPASS_URL,
                    data={"data": query},
                    timeout=timeout
                ) as response:

                    # Handle rate limiting
                    if response.status == 429:
                        logger.warning(
                            "Overpass API rate limited (attempt %d/%d), waiting %ds",
                            attempt + 1, self.MAX_RETRIES, backoff
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, self.MAX_BACKOFF)
                        continue

                    # Handle other errors
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(
                            "Overpass API error: HTTP %d - %s",
                            response.status, error_text
                        )
                        raise RuntimeError(
                            f"Overpass API request failed: HTTP {response.status}"
                        )

                    data = await response.json()

                    # Cache successful response
                    self._cache_response(cache_key, data)

                    return data

            except asyncio.TimeoutError:
                last_error = "Overpass API request timed out"
                logger.warning("%s (attempt %d/%d)", last_error, attempt + 1, self.MAX_RETRIES)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)

            except aiohttp.ClientError as e:
                last_error = f"Overpass API client error: {e}"
                logger.warning("%s (attempt %d/%d)", last_error, attempt + 1, self.MAX_RETRIES)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)

        # All retries failed
        raise RuntimeError(f"Overpass API request failed after {self.MAX_RETRIES} attempts: {last_error}")

    def _parse_overpass_element(self, element: Dict) -> Optional[InfrastructureFeature]:
        """
        Parse a single Overpass API element into InfrastructureFeature.

        Args:
            element: OSM element from Overpass response

        Returns:
            InfrastructureFeature or None if element can't be parsed
        """
        try:
            # Get element ID
            osm_id = str(element.get("id", "unknown"))

            # Get coordinates (use 'center' for ways, direct lat/lon for nodes)
            if "center" in element:
                lat = element["center"]["lat"]
                lon = element["center"]["lon"]
            else:
                lat = element.get("lat")
                lon = element.get("lon")

            if lat is None or lon is None:
                return None

            # Get tags
            tags = element.get("tags", {})

            # Determine infrastructure type
            infra_type = None
            for itype, tag_list in OSM_TAGS.items():
                for key, value in tag_list:
                    if tags.get(key) == value:
                        infra_type = itype
                        break
                if infra_type:
                    break

            if infra_type is None:
                return None

            # Extract properties
            name = tags.get("name", tags.get("operator", "Unnamed"))
            address = self._format_address(tags)
            phone = tags.get("phone")

            # Try to extract capacity
            capacity = None
            capacity_str = tags.get("capacity") or tags.get("beds")
            if capacity_str:
                try:
                    capacity = int(capacity_str)
                except (ValueError, TypeError):
                    pass

            return InfrastructureFeature(
                id=osm_id,
                type=infra_type,
                name=name,
                lat=float(lat),
                lon=float(lon),
                address=address,
                phone=phone,
                capacity=capacity
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse OSM element: %s", e)
            return None

    def _format_address(self, tags: Dict) -> Optional[str]:
        """Format address from OSM tags."""
        # Try to build address from components
        addr_parts = []

        if "addr:housenumber" in tags:
            addr_parts.append(tags["addr:housenumber"])
        if "addr:street" in tags:
            addr_parts.append(tags["addr:street"])
        if "addr:city" in tags:
            addr_parts.append(tags["addr:city"])
        if "addr:postcode" in tags:
            addr_parts.append(tags["addr:postcode"])

        if addr_parts:
            return ", ".join(addr_parts)

        # Fallback to full address if available
        return tags.get("addr:full")

    async def query_by_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        types: Optional[List[InfrastructureType]] = None
    ) -> List[InfrastructureFeature]:
        """
        Query infrastructure within bounding box.

        Args:
            bbox: Bounding box as (min_lon, min_lat, max_lon, max_lat)
            types: List of infrastructure types to query (default: all)

        Returns:
            List of InfrastructureFeature objects

        Example:
            # Query hospitals in Miami area
            bbox = (-80.87, 25.13, -80.12, 25.97)
            hospitals = await client.query_by_bbox(
                bbox,
                types=[InfrastructureType.HOSPITAL]
            )
        """
        logger.info(
            "Querying infrastructure in bbox %s (types=%s)",
            bbox, [t.value for t in types] if types else "all"
        )

        try:
            # Build and execute query
            query = self._build_overpass_query(bbox, types)
            response = await self._fetch_overpass(query)

            # Parse elements
            features = []
            for element in response.get("elements", []):
                feature = self._parse_overpass_element(element)
                if feature:
                    features.append(feature)

            logger.info("Found %d infrastructure features", len(features))
            return features

        except Exception as e:
            logger.error("Failed to query infrastructure: %s", e)
            return []

    async def find_in_flood_zone(
        self,
        flood_geojson: Dict,
        types: Optional[List[InfrastructureType]] = None
    ) -> Dict:
        """
        Find infrastructure within flood zone.

        Args:
            flood_geojson: GeoJSON geometry of flood extent
            types: List of infrastructure types to query (default: all)

        Returns:
            Dictionary with:
                - in_flood_zone: List of features inside flood area
                - nearby: List of features within 1km of flood area
                - by_type: Dict mapping type to count

        Example:
            result = await client.find_in_flood_zone(
                flood_geojson={"type": "Polygon", "coordinates": [...]},
                types=[InfrastructureType.HOSPITAL, InfrastructureType.SHELTER]
            )

            print(f"{len(result['in_flood_zone'])} facilities flooded")
            print(f"{len(result['nearby'])} facilities nearby")
            print(f"By type: {result['by_type']}")
        """
        if not SHAPELY_AVAILABLE:
            logger.error("shapely is required for flood zone analysis")
            return {
                "in_flood_zone": [],
                "nearby": [],
                "by_type": {}
            }

        try:
            # Parse flood geometry
            if "geometry" in flood_geojson:
                flood_geom = shape(flood_geojson["geometry"])
            elif "type" in flood_geojson and "coordinates" in flood_geojson:
                flood_geom = shape(flood_geojson)
            else:
                raise ValueError("Invalid GeoJSON structure")

            # Calculate bbox with buffer for nearby search (1km = ~0.009 degrees)
            bounds = flood_geom.bounds  # (minx, miny, maxx, maxy)
            buffer_deg = 0.009
            bbox = (
                bounds[0] - buffer_deg,
                bounds[1] - buffer_deg,
                bounds[2] + buffer_deg,
                bounds[3] + buffer_deg
            )

            # Query all infrastructure in expanded bbox
            all_features = await self.query_by_bbox(bbox, types)

            # Categorize by location
            in_flood_zone = []
            nearby = []

            for feature in all_features:
                point = Point(feature.lon, feature.lat)

                if flood_geom.contains(point):
                    in_flood_zone.append(feature)
                else:
                    # Check if within 1km (~0.009 degrees)
                    distance = flood_geom.distance(point)
                    if distance <= 0.009:
                        nearby.append(feature)

            # Count by type
            by_type = {}
            for feature in in_flood_zone:
                type_name = feature.type.value
                by_type[type_name] = by_type.get(type_name, 0) + 1

            logger.info(
                "Found %d facilities in flood zone, %d nearby",
                len(in_flood_zone), len(nearby)
            )

            return {
                "in_flood_zone": in_flood_zone,
                "nearby": nearby,
                "by_type": by_type
            }

        except Exception as e:
            logger.error("Failed to find infrastructure in flood zone: %s", e)
            return {
                "in_flood_zone": [],
                "nearby": [],
                "by_type": {}
            }

    def to_geojson(self, features: List[InfrastructureFeature]) -> Dict:
        """
        Convert list of features to GeoJSON FeatureCollection.

        Args:
            features: List of InfrastructureFeature objects

        Returns:
            GeoJSON FeatureCollection dict

        Example:
            geojson = client.to_geojson(hospitals)
            with open("hospitals.geojson", "w") as f:
                json.dump(geojson, f)
        """
        return {
            "type": "FeatureCollection",
            "features": [f.to_geojson_feature() for f in features]
        }
