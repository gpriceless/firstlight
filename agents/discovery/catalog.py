"""
Catalog Query Manager for parallel STAC catalog queries.

Manages:
- Parallel queries to multiple STAC catalogs
- Query result caching with TTL
- Timeout and retry handling per catalog
- Aggregation of results from multiple sources
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False

from core.data.discovery.base import DiscoveryResult, DiscoveryError
from core.data.providers.registry import Provider


logger = logging.getLogger(__name__)


class CatalogQueryStatus(Enum):
    """Status of a catalog query."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    ERROR = "error"
    CACHED = "cached"


@dataclass
class CatalogQueryResult:
    """
    Result from a single catalog query.

    Attributes:
        catalog_id: Identifier of the catalog queried
        provider_id: Provider ID associated with catalog
        status: Query status
        results: List of discovery results
        items_found: Total items found
        items_returned: Items returned (may be limited)
        query_time_ms: Query execution time
        error_message: Error message if failed
        cached: Whether result came from cache
        cache_age_seconds: Age of cached result
        metadata: Additional query metadata
    """

    catalog_id: str
    provider_id: str
    status: CatalogQueryStatus
    results: List[DiscoveryResult] = field(default_factory=list)
    items_found: int = 0
    items_returned: int = 0
    query_time_ms: float = 0.0
    error_message: Optional[str] = None
    cached: bool = False
    cache_age_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "catalog_id": self.catalog_id,
            "provider_id": self.provider_id,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "items_found": self.items_found,
            "items_returned": self.items_returned,
            "query_time_ms": self.query_time_ms,
            "error_message": self.error_message,
            "cached": self.cached,
            "cache_age_seconds": self.cache_age_seconds,
            "metadata": self.metadata,
        }


@dataclass
class CacheEntry:
    """Cache entry for query results."""

    key: str
    results: List[CatalogQueryResult]
    created_at: datetime
    hits: int = 0

    @property
    def age_seconds(self) -> float:
        """Get cache entry age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()


class CatalogQueryManager:
    """
    Manages parallel queries to multiple STAC catalogs.

    Features:
    - Concurrent queries with configurable parallelism
    - Per-catalog timeout handling
    - Query result caching with TTL
    - Retry with exponential backoff
    - Aggregation of results from multiple sources
    """

    def __init__(
        self,
        max_concurrent_queries: int = 10,
        query_timeout_seconds: float = 30.0,
        cache_enabled: bool = True,
        cache_ttl_seconds: int = 300,
        max_retries: int = 2,
        retry_delay_seconds: float = 1.0
    ):
        """
        Initialize CatalogQueryManager.

        Args:
            max_concurrent_queries: Maximum parallel queries
            query_timeout_seconds: Timeout per catalog query
            cache_enabled: Whether to cache results
            cache_ttl_seconds: Cache TTL in seconds
            max_retries: Maximum retry attempts per catalog
            retry_delay_seconds: Initial retry delay
        """
        self.max_concurrent_queries = max_concurrent_queries
        self.query_timeout_seconds = query_timeout_seconds
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Semaphore for limiting concurrent queries
        self._query_semaphore = asyncio.Semaphore(max_concurrent_queries)

        # Query result cache
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = asyncio.Lock()

        # HTTP session management
        self._session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self._stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "query_errors": 0,
            "query_timeouts": 0,
        }

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create HTTP session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for catalog queries. Install with: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.query_timeout_seconds)
            )
        return self._session

    async def close(self) -> None:
        """Close HTTP session and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        # Clear cache
        async with self._cache_lock:
            self._cache.clear()

    def _generate_cache_key(
        self,
        catalog_id: str,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> str:
        """Generate cache key from query parameters."""
        key_data = {
            "catalog_id": catalog_id,
            "spatial": spatial,
            "temporal": temporal,
            "constraints": constraints
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    async def _get_from_cache(self, cache_key: str) -> Optional[CacheEntry]:
        """Get entry from cache if valid."""
        async with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry is None:
                return None

            # Check TTL
            if entry.age_seconds > self.cache_ttl_seconds:
                del self._cache[cache_key]
                return None

            entry.hits += 1
            return entry

    async def _store_in_cache(
        self,
        cache_key: str,
        results: List[CatalogQueryResult]
    ) -> None:
        """Store results in cache."""
        async with self._cache_lock:
            self._cache[cache_key] = CacheEntry(
                key=cache_key,
                results=results,
                created_at=datetime.now(timezone.utc)
            )

            # Simple cache eviction - remove oldest entries if too many
            max_cache_size = 1000
            if len(self._cache) > max_cache_size:
                # Sort by creation time and remove oldest
                sorted_entries = sorted(
                    self._cache.items(),
                    key=lambda x: x[1].created_at
                )
                for key, _ in sorted_entries[:len(self._cache) - max_cache_size]:
                    del self._cache[key]

    async def query_catalogs(
        self,
        providers: List[Provider],
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[CatalogQueryResult]:
        """
        Query multiple catalogs in parallel.

        Args:
            providers: List of providers to query
            spatial: GeoJSON geometry or bbox
            temporal: Temporal extent with start/end
            constraints: Per-data-type constraints

        Returns:
            List of CatalogQueryResult from all catalogs
        """
        constraints = constraints or {}

        logger.info(f"Querying {len(providers)} catalogs in parallel")

        # Create query tasks for each provider
        tasks = []
        for provider in providers:
            # Get data-type specific constraints
            provider_constraints = constraints.get(provider.type, {})

            task = self._query_single_catalog(
                provider=provider,
                spatial=spatial,
                temporal=temporal,
                constraints=provider_constraints
            )
            tasks.append(task)

        # Execute all queries in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        catalog_results: List[CatalogQueryResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                provider = providers[i]
                catalog_results.append(CatalogQueryResult(
                    catalog_id=provider.access.get("endpoint", provider.id),
                    provider_id=provider.id,
                    status=CatalogQueryStatus.ERROR,
                    error_message=str(result)
                ))
                self._stats["query_errors"] += 1
            else:
                catalog_results.append(result)

        # Log summary
        successful = sum(1 for r in catalog_results if r.status == CatalogQueryStatus.COMPLETED)
        cached = sum(1 for r in catalog_results if r.cached)
        total_items = sum(r.items_returned for r in catalog_results)

        logger.info(
            f"Catalog queries complete: {successful}/{len(catalog_results)} successful, "
            f"{cached} from cache, {total_items} total items"
        )

        return catalog_results

    async def _query_single_catalog(
        self,
        provider: Provider,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> CatalogQueryResult:
        """
        Query a single catalog with retry and caching.

        Args:
            provider: Provider to query
            spatial: GeoJSON geometry
            temporal: Temporal extent
            constraints: Query constraints

        Returns:
            CatalogQueryResult
        """
        catalog_id = provider.access.get("endpoint", provider.id)
        cache_key = self._generate_cache_key(
            catalog_id, spatial, temporal, constraints
        )

        # Check cache
        if self.cache_enabled:
            cached_entry = await self._get_from_cache(cache_key)
            if cached_entry:
                self._stats["cache_hits"] += 1
                # Return first result (should only be one per cache key)
                if cached_entry.results:
                    result = cached_entry.results[0]
                    result.cached = True
                    result.cache_age_seconds = cached_entry.age_seconds
                    return result

        self._stats["cache_misses"] += 1

        # Acquire semaphore for rate limiting
        async with self._query_semaphore:
            return await self._execute_query_with_retry(
                provider=provider,
                spatial=spatial,
                temporal=temporal,
                constraints=constraints,
                cache_key=cache_key
            )

    async def _execute_query_with_retry(
        self,
        provider: Provider,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any],
        cache_key: str
    ) -> CatalogQueryResult:
        """Execute query with retry logic."""
        catalog_id = provider.access.get("endpoint", provider.id)
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_catalog_query(
                    provider=provider,
                    spatial=spatial,
                    temporal=temporal,
                    constraints=constraints
                )

                # Cache successful results
                if self.cache_enabled and result.status == CatalogQueryStatus.COMPLETED:
                    await self._store_in_cache(cache_key, [result])

                return result

            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(f"Timeout querying {catalog_id}")
                self._stats["query_timeouts"] += 1

                if attempt < self.max_retries:
                    delay = self.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Query timeout for {catalog_id}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)

            except Exception as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = self.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"Query error for {catalog_id}: {e}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        status = (
            CatalogQueryStatus.TIMEOUT
            if isinstance(last_error, asyncio.TimeoutError)
            else CatalogQueryStatus.ERROR
        )

        return CatalogQueryResult(
            catalog_id=catalog_id,
            provider_id=provider.id,
            status=status,
            error_message=str(last_error) if last_error else "Unknown error"
        )

    async def _execute_catalog_query(
        self,
        provider: Provider,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> CatalogQueryResult:
        """Execute the actual catalog query."""
        catalog_id = provider.access.get("endpoint", provider.id)
        protocol = provider.access.get("protocol", "stac")

        start_time = datetime.now(timezone.utc)
        self._stats["queries_executed"] += 1

        if protocol == "stac":
            return await self._query_stac_catalog(
                provider=provider,
                spatial=spatial,
                temporal=temporal,
                constraints=constraints,
                start_time=start_time
            )
        else:
            # Unsupported protocol
            return CatalogQueryResult(
                catalog_id=catalog_id,
                provider_id=provider.id,
                status=CatalogQueryStatus.ERROR,
                error_message=f"Unsupported protocol: {protocol}"
            )

    async def _query_stac_catalog(
        self,
        provider: Provider,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any],
        start_time: datetime
    ) -> CatalogQueryResult:
        """Query a STAC catalog."""
        endpoint = provider.access.get("endpoint", "")
        collection = provider.metadata.get("stac_collection")
        catalog_id = endpoint

        if not endpoint:
            return CatalogQueryResult(
                catalog_id=catalog_id,
                provider_id=provider.id,
                status=CatalogQueryStatus.ERROR,
                error_message="No endpoint configured"
            )

        # Build search URL
        search_url = endpoint.rstrip("/")
        if not search_url.endswith("/search"):
            search_url = f"{search_url}/search"

        # Build search parameters
        search_params = self._build_stac_search_params(
            spatial=spatial,
            temporal=temporal,
            collection=collection,
            constraints=constraints
        )

        # Execute search
        session = await self._get_session()

        async with session.post(
            search_url,
            json=search_params,
            timeout=aiohttp.ClientTimeout(total=self.query_timeout_seconds)
        ) as response:
            response.raise_for_status()
            data = await response.json()

        # Parse results
        items = data.get("features", [])
        results = self._parse_stac_items(items, provider, spatial)

        query_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return CatalogQueryResult(
            catalog_id=catalog_id,
            provider_id=provider.id,
            status=CatalogQueryStatus.COMPLETED,
            results=results,
            items_found=data.get("numberMatched", len(items)),
            items_returned=len(items),
            query_time_ms=query_time_ms,
            metadata={
                "collection": collection,
                "search_params": search_params
            }
        )

    def _build_stac_search_params(
        self,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        collection: Optional[str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build STAC search parameters."""
        params: Dict[str, Any] = {}

        # Collections
        if collection:
            params["collections"] = [collection]

        # Spatial (intersects)
        params["intersects"] = spatial

        # Temporal
        start = temporal.get("start", "")
        end = temporal.get("end", "")
        if start and end:
            params["datetime"] = f"{start}/{end}"

        # Query parameters for constraints
        query = {}

        # Cloud cover constraint
        max_cloud = constraints.get("max_cloud_cover")
        if max_cloud is not None:
            query["eo:cloud_cover"] = {"lte": max_cloud * 100}

        if query:
            params["query"] = query

        # Result limit
        params["limit"] = constraints.get("max_results", 100)

        return params

    def _parse_stac_items(
        self,
        items: List[Dict[str, Any]],
        provider: Provider,
        spatial: Dict[str, Any]
    ) -> List[DiscoveryResult]:
        """Parse STAC items into DiscoveryResult objects."""
        results = []
        query_bbox = self._extract_bbox(spatial)

        for item in items:
            try:
                result = self._stac_item_to_result(item, provider, query_bbox)
                results.append(result)
            except Exception as e:
                logger.warning(f"Error parsing STAC item {item.get('id')}: {e}")
                continue

        return results

    def _stac_item_to_result(
        self,
        item: Dict[str, Any],
        provider: Provider,
        query_bbox: List[float]
    ) -> DiscoveryResult:
        """Convert STAC item to DiscoveryResult."""
        properties = item.get("properties", {})
        bbox = item.get("bbox", [])

        # Acquisition time
        datetime_str = properties.get("datetime")
        if datetime_str:
            acquisition_time = datetime.fromisoformat(
                datetime_str.replace("Z", "+00:00")
            )
        else:
            acquisition_time = datetime.now(timezone.utc)

        # Find primary asset
        assets = item.get("assets", {})
        source_uri, data_format = self._select_primary_asset(assets)

        # Spatial coverage
        if bbox:
            spatial_coverage = self._calculate_coverage_percent(bbox, query_bbox)
        else:
            spatial_coverage = 50.0

        # Resolution
        resolution_m = properties.get(
            "gsd",
            provider.capabilities.get("resolution_m", 10.0)
        )

        # Cloud cover
        cloud_cover = None
        if provider.type == "optical":
            cloud_cover = properties.get("eo:cloud_cover")

        return DiscoveryResult(
            dataset_id=item["id"],
            provider=provider.id,
            data_type=provider.type,
            source_uri=source_uri,
            format=data_format,
            acquisition_time=acquisition_time,
            spatial_coverage_percent=spatial_coverage,
            resolution_m=resolution_m,
            cloud_cover_percent=cloud_cover,
            quality_flag=self._determine_quality_flag(properties),
            cost_tier=provider.cost.get("tier", "open"),
            metadata={"stac_item": item["id"], "properties": properties}
        )

    def _select_primary_asset(
        self,
        assets: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Select primary data asset from STAC assets."""
        priority_keys = ["data", "visual", "B04", "VV", "thumbnail"]

        for key in priority_keys:
            if key in assets:
                asset = assets[key]
                href = asset.get("href", "")
                if href:
                    data_format = self._infer_format(asset)
                    return href, data_format

        # Fallback to first asset
        if assets:
            first_asset = next(iter(assets.values()))
            return first_asset.get("href", ""), self._infer_format(first_asset)

        return "", "unknown"

    def _infer_format(self, asset: Dict[str, Any]) -> str:
        """Infer data format from asset."""
        media_type = asset.get("type", "")
        href = asset.get("href", "")

        if "geotiff" in media_type or "tiff" in media_type:
            return "cog" if "cloud-optimized" in str(asset.get("roles", [])) else "geotiff"
        elif "netcdf" in media_type:
            return "netcdf"
        elif "zarr" in media_type:
            return "zarr"
        elif href.endswith(".tif") or href.endswith(".tiff"):
            return "cog"
        elif href.endswith(".nc"):
            return "netcdf"

        return "geotiff"

    def _determine_quality_flag(self, properties: Dict[str, Any]) -> str:
        """Determine quality flag from properties."""
        cloud_cover = properties.get("eo:cloud_cover")
        if cloud_cover is not None:
            if cloud_cover <= 10:
                return "excellent"
            elif cloud_cover <= 30:
                return "good"
            elif cloud_cover <= 50:
                return "fair"
            else:
                return "poor"
        return "good"

    def _extract_bbox(self, spatial: Dict[str, Any]) -> List[float]:
        """Extract bounding box from GeoJSON."""
        if "bbox" in spatial:
            return spatial["bbox"]

        geom_type = spatial.get("type")
        coords = spatial.get("coordinates", [])

        if geom_type == "Point":
            lon, lat = coords
            return [lon, lat, lon, lat]
        elif geom_type == "Polygon":
            lons = [p[0] for p in coords[0]]
            lats = [p[1] for p in coords[0]]
            return [min(lons), min(lats), max(lons), max(lats)]
        elif geom_type == "MultiPolygon":
            all_lons = []
            all_lats = []
            for polygon in coords:
                for p in polygon[0]:
                    all_lons.append(p[0])
                    all_lats.append(p[1])
            return [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]

        return [-180, -90, 180, 90]

    def _calculate_coverage_percent(
        self,
        dataset_bbox: List[float],
        query_bbox: List[float]
    ) -> float:
        """Calculate spatial coverage percentage."""
        int_west = max(dataset_bbox[0], query_bbox[0])
        int_south = max(dataset_bbox[1], query_bbox[1])
        int_east = min(dataset_bbox[2], query_bbox[2])
        int_north = min(dataset_bbox[3], query_bbox[3])

        if int_west >= int_east or int_south >= int_north:
            return 0.0

        int_area = (int_east - int_west) * (int_north - int_south)
        query_area = (query_bbox[2] - query_bbox[0]) * (query_bbox[3] - query_bbox[1])

        if query_area == 0:
            return 0.0

        return min((int_area / query_area) * 100.0, 100.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get query statistics."""
        return {
            **self._stats,
            "cache_entries": len(self._cache),
            "cache_hit_rate": (
                self._stats["cache_hits"] /
                max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)
            )
        }
