"""
Census API Client for Population Data.

Provides access to US Census Bureau data for population estimation and
demographic analysis in disaster scenarios. Uses the Census API to retrieve
population counts and housing unit data for affected areas.

The Census Bureau provides free API access to American Community Survey (ACS)
data without requiring an API key for basic queries.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
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
    from shapely.geometry import box, shape
    SHAPELY_AVAILABLE = True
except ImportError:
    shapely = None  # type: ignore
    SHAPELY_AVAILABLE = False


logger = logging.getLogger(__name__)


# State FIPS codes for quick lookup
STATE_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16",
    "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22",
    "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
    "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40",
    "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56"
}

# County FIPS to bounding box mapping (sample for key states)
# In production, this would be loaded from a complete dataset
COUNTY_BOUNDS = {
    "12": {  # Florida
        "086": (-80.87, 25.13, -80.12, 25.97),  # Miami-Dade
        "011": (-80.93, 26.00, -80.05, 26.94),  # Broward
    }
}


@dataclass
class PopulationData:
    """Population data for an area."""

    total_population: int
    housing_units: int
    occupied_housing: int
    area_name: str
    state_fips: str
    county_fips: str
    tract: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_population": self.total_population,
            "housing_units": self.housing_units,
            "occupied_housing": self.occupied_housing,
            "area_name": self.area_name,
            "state_fips": self.state_fips,
            "county_fips": self.county_fips,
            "tract": self.tract,
        }


class CensusClient:
    """
    Client for US Census Bureau API.

    Provides methods to retrieve population and housing data from the
    American Community Survey (ACS) 5-year estimates. Does not require
    an API key for basic queries.

    Example:
        client = CensusClient()
        data = await client.get_population_by_county("12", "086")
        print(f"Miami-Dade County: {data.total_population:,} people")

    API Documentation:
        https://www.census.gov/data/developers/data-sets/acs-5year.html
    """

    BASE_URL = "https://api.census.gov/data"
    CACHE_DIR = Path.home() / ".cache" / "firstlight" / "census"

    # Census API variable codes
    VARIABLES = {
        "population": "B01003_001E",      # Total population
        "housing_units": "B25001_001E",   # Total housing units
        "occupied_housing": "B25002_002E" # Occupied housing units
    }

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 86400):
        """
        Initialize Census client.

        Args:
            api_key: Optional Census API key (not required for basic queries)
            cache_ttl: Cache time-to-live in seconds (default 24 hours)
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self._session: Optional[aiohttp.ClientSession] = None

        # Create cache directory
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "CensusClient initialized (cache_ttl=%ds, cache_dir=%s)",
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
        # Use hash to avoid filesystem issues with special characters
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

    async def _fetch_json(self, url: str, params: Dict) -> Dict:
        """
        Fetch JSON from Census API with caching.

        Args:
            url: API endpoint URL
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If request fails
        """
        # Create cache key from URL and params
        cache_key = f"{url}?{json.dumps(params, sort_keys=True)}"

        # Check cache first
        cached = self._get_cached_response(cache_key)
        if cached is not None:
            return cached

        # Add API key if provided
        if self.api_key:
            params["key"] = self.api_key

        # Ensure we have a session
        if self._session is None:
            if not AIOHTTP_AVAILABLE:
                raise RuntimeError("aiohttp is required but not installed")
            self._session = aiohttp.ClientSession()

        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with self._session.get(url, params=params, timeout=timeout) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        "Census API error: HTTP %d - %s",
                        response.status, error_text
                    )
                    raise RuntimeError(
                        f"Census API request failed: HTTP {response.status}"
                    )

                data = await response.json()

                # Cache successful response
                self._cache_response(cache_key, data)

                return data

        except asyncio.TimeoutError:
            logger.error("Census API timeout: %s", url)
            raise RuntimeError("Census API request timed out")

        except aiohttp.ClientError as e:
            logger.error("Census API client error: %s", e)
            raise RuntimeError(f"Census API request failed: {e}")

    def _parse_census_response(
        self,
        data: List[List],
        state_fips: str,
        county_fips: str,
        tract: Optional[str] = None
    ) -> PopulationData:
        """
        Parse Census API response.

        The API returns data as a list of lists, where the first row
        is headers and subsequent rows are data.

        Args:
            data: Raw API response
            state_fips: State FIPS code
            county_fips: County FIPS code
            tract: Optional tract code

        Returns:
            PopulationData object
        """
        if len(data) < 2:
            raise ValueError("Census API returned no data")

        headers = data[0]
        values = data[1]

        # Create mapping from headers to values
        result = dict(zip(headers, values))

        # Extract values (handle both string and int responses)
        total_pop = int(result.get(self.VARIABLES["population"], 0))
        housing = int(result.get(self.VARIABLES["housing_units"], 0))
        occupied = int(result.get(self.VARIABLES["occupied_housing"], 0))

        # Get area name
        area_name = result.get("NAME", "Unknown")

        return PopulationData(
            total_population=total_pop,
            housing_units=housing,
            occupied_housing=occupied,
            area_name=area_name,
            state_fips=state_fips,
            county_fips=county_fips,
            tract=tract
        )

    async def get_population_by_county(
        self,
        state_fips: str,
        county_fips: str
    ) -> PopulationData:
        """
        Get population data for a specific county.

        Args:
            state_fips: Two-digit state FIPS code (e.g., "12" for Florida)
            county_fips: Three-digit county FIPS code (e.g., "086" for Miami-Dade)

        Returns:
            PopulationData for the county

        Example:
            data = await client.get_population_by_county("12", "086")
            print(f"{data.area_name}: {data.total_population:,}")
        """
        # Use latest ACS 5-year data (2022)
        url = f"{self.BASE_URL}/2022/acs/acs5"

        # Build query parameters
        variables = ",".join(self.VARIABLES.values())
        params = {
            "get": f"{variables},NAME",
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}"
        }

        logger.info(
            "Fetching census data for state=%s, county=%s",
            state_fips, county_fips
        )

        try:
            data = await self._fetch_json(url, params)
            return self._parse_census_response(data, state_fips, county_fips)

        except Exception as e:
            logger.error("Failed to fetch census data: %s", e)
            # Return sensible default on failure
            return PopulationData(
                total_population=0,
                housing_units=0,
                occupied_housing=0,
                area_name="Unknown County",
                state_fips=state_fips,
                county_fips=county_fips
            )

    async def get_population_by_bbox(
        self,
        bbox: Tuple[float, float, float, float]
    ) -> List[PopulationData]:
        """
        Get population data for all counties intersecting a bounding box.

        This is a simplified implementation that checks county centroids.
        A production version would use actual county boundary geometries.

        Args:
            bbox: Bounding box as (minx, miny, maxx, maxy) in WGS84

        Returns:
            List of PopulationData for intersecting counties

        Example:
            # Miami area
            bbox = (-80.87, 25.13, -80.12, 25.97)
            counties = await client.get_population_by_bbox(bbox)
        """
        if not SHAPELY_AVAILABLE:
            logger.warning("shapely not available, returning empty list")
            return []

        minx, miny, maxx, maxy = bbox
        bbox_geom = box(minx, miny, maxx, maxy)

        results = []

        # In a production system, we'd query a county boundary database
        # For now, check known county bounds
        for state_fips, counties in COUNTY_BOUNDS.items():
            for county_fips, county_bbox in counties.items():
                c_minx, c_miny, c_maxx, c_maxy = county_bbox
                county_geom = box(c_minx, c_miny, c_maxx, c_maxy)

                if bbox_geom.intersects(county_geom):
                    logger.debug(
                        "County %s-%s intersects bbox",
                        state_fips, county_fips
                    )
                    try:
                        data = await self.get_population_by_county(
                            state_fips, county_fips
                        )
                        results.append(data)
                    except Exception as e:
                        logger.warning(
                            "Failed to get data for %s-%s: %s",
                            state_fips, county_fips, e
                        )

        return results

    def _calculate_bbox_from_geojson(self, geojson: Dict) -> Tuple[float, float, float, float]:
        """Calculate bounding box from GeoJSON geometry."""
        if not SHAPELY_AVAILABLE:
            raise RuntimeError("shapely is required for GeoJSON processing")

        # Handle different GeoJSON structures
        if "geometry" in geojson:
            geom = shape(geojson["geometry"])
        elif "type" in geojson and "coordinates" in geojson:
            geom = shape(geojson)
        else:
            raise ValueError("Invalid GeoJSON structure")

        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        return bounds

    async def estimate_affected_population(
        self,
        flood_geojson: Dict,
        total_flood_area_km2: float
    ) -> Dict:
        """
        Estimate population affected by flood event.

        This uses a simple area-based approximation:
        1. Find counties intersecting the flood zone
        2. Calculate overlap ratio for each county
        3. Estimate affected population as (county_pop * overlap_ratio)

        Args:
            flood_geojson: GeoJSON geometry of flood extent
            total_flood_area_km2: Total flooded area in square kilometers

        Returns:
            Dictionary with:
                - estimated_population: Estimated affected population
                - estimated_housing_units: Estimated affected housing units
                - methodology: Description of estimation method
                - confidence: Confidence level (high/medium/low)
                - counties_affected: List of affected county names

        Example:
            estimate = await client.estimate_affected_population(
                flood_geojson={"type": "Polygon", "coordinates": [...]},
                total_flood_area_km2=45.2
            )
            print(f"Estimated {estimate['estimated_population']:,} affected")
        """
        try:
            # Calculate bounding box
            bbox = self._calculate_bbox_from_geojson(flood_geojson)

            # Get counties in bbox
            counties = await self.get_population_by_bbox(bbox)

            if not counties:
                logger.warning("No counties found for flood area")
                return {
                    "estimated_population": 0,
                    "estimated_housing_units": 0,
                    "methodology": "bbox_intersection",
                    "confidence": "low",
                    "counties_affected": []
                }

            # Simple estimation: assume flood covers fraction of counties
            # In production, we'd use actual geometry intersection
            total_pop = sum(c.total_population for c in counties)
            total_housing = sum(c.housing_units for c in counties)

            # Rough approximation based on area
            # Assume average county is ~1000 km2
            avg_county_area_km2 = 1000.0
            total_county_area_km2 = len(counties) * avg_county_area_km2

            # Estimate affected population as proportional to flooded area
            if total_county_area_km2 > 0:
                affected_ratio = min(1.0, total_flood_area_km2 / total_county_area_km2)
            else:
                affected_ratio = 0.1  # Conservative default

            estimated_pop = int(total_pop * affected_ratio)
            estimated_housing = int(total_housing * affected_ratio)

            # Determine confidence
            if len(counties) == 1 and affected_ratio < 0.5:
                confidence = "medium"
            elif len(counties) > 3:
                confidence = "low"
            else:
                confidence = "medium"

            county_names = [c.area_name for c in counties]

            logger.info(
                "Population estimate: %d people, %d housing units (confidence: %s)",
                estimated_pop, estimated_housing, confidence
            )

            return {
                "estimated_population": estimated_pop,
                "estimated_housing_units": estimated_housing,
                "methodology": "bbox_intersection_area_weighted",
                "confidence": confidence,
                "counties_affected": county_names,
                "total_flood_area_km2": total_flood_area_km2,
                "counties_data": [c.to_dict() for c in counties]
            }

        except Exception as e:
            logger.error("Failed to estimate affected population: %s", e)
            return {
                "estimated_population": 0,
                "estimated_housing_units": 0,
                "methodology": "estimation_failed",
                "confidence": "none",
                "counties_affected": [],
                "error": str(e)
            }
