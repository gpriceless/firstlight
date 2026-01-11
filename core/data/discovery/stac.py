"""
STAC (SpatioTemporal Asset Catalog) discovery adapter.

Queries STAC-compliant catalogs for geospatial data discovery.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
import aiohttp

from core.data.discovery.base import (
    DiscoveryAdapter,
    DiscoveryResult,
    DiscoveryError
)


class STACAdapter(DiscoveryAdapter):
    """
    STAC catalog discovery adapter.

    Implements STAC API search protocol for discovering geospatial datasets.
    Compatible with Element84 Earth Search, Microsoft Planetary Computer, etc.
    """

    def __init__(self):
        super().__init__("stac")
        self._session: Optional[aiohttp.ClientSession] = None

    async def discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[DiscoveryResult]:
        """
        Discover datasets via STAC API search.

        Args:
            provider: Provider configuration with access.endpoint
            spatial: GeoJSON geometry
            temporal: Temporal extent
            constraints: Optional data-type-specific constraints

        Returns:
            List of DiscoveryResult objects
        """
        constraints = constraints or {}

        # Extract provider access configuration
        access = provider.access
        endpoint = access["endpoint"]
        collection = provider.metadata.get("stac_collection")

        # Build STAC search parameters
        search_params = self._build_search_params(
            spatial=spatial,
            temporal=temporal,
            collection=collection,
            constraints=constraints
        )

        # Execute STAC search
        try:
            items = await self._search_stac(endpoint, search_params)
        except Exception as e:
            raise DiscoveryError(
                f"STAC search failed: {str(e)}",
                provider=provider.id
            )

        # Convert STAC items to DiscoveryResults
        results = []
        query_bbox = self._extract_bbox(spatial)

        for item in items:
            try:
                result = self._stac_item_to_result(
                    item=item,
                    provider=provider,
                    query_bbox=query_bbox
                )
                results.append(result)
            except Exception as e:
                # Log error but continue with other items
                print(f"Error processing STAC item {item.get('id')}: {e}")
                continue

        return results

    def supports_provider(self, provider: Any) -> bool:
        """Check if provider uses STAC protocol."""
        return provider.access.get("protocol") == "stac"

    def _build_search_params(
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

        # Temporal (datetime range)
        start_dt, end_dt = self._parse_temporal_extent(temporal)
        params["datetime"] = f"{start_dt.isoformat()}/{end_dt.isoformat()}"

        # Query parameters for constraints
        query = {}

        # Cloud cover constraint (for optical)
        if "max_cloud_cover" in constraints:
            max_cloud = constraints["max_cloud_cover"]
            query["eo:cloud_cover"] = {"lte": max_cloud * 100}

        # Add query if not empty
        if query:
            params["query"] = query

        # Limit results
        params["limit"] = constraints.get("max_results", 100)

        return params

    async def _search_stac(
        self,
        endpoint: str,
        search_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Execute STAC API search.

        Args:
            endpoint: STAC catalog endpoint URL
            search_params: Search parameters

        Returns:
            List of STAC items
        """
        # Ensure endpoint has /search
        if not endpoint.endswith("/search"):
            search_url = f"{endpoint.rstrip('/')}/search"
        else:
            search_url = endpoint

        # Create session if needed
        if self._session is None:
            self._session = aiohttp.ClientSession()

        items = []
        next_link = None

        try:
            # Initial search
            async with self._session.post(
                search_url,
                json=search_params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Extract items
                items.extend(data.get("features", []))

                # Check for pagination
                links = data.get("links", [])
                next_link = next((link["href"] for link in links if link["rel"] == "next"), None)

            # Follow pagination (limit to 5 pages)
            page_count = 1
            while next_link and page_count < 5:
                async with self._session.get(
                    next_link,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                    items.extend(data.get("features", []))

                    links = data.get("links", [])
                    next_link = next((link["href"] for link in links if link["rel"] == "next"), None)
                    page_count += 1

        except aiohttp.ClientError as e:
            raise DiscoveryError(f"STAC API request failed: {str(e)}")

        return items

    def _stac_item_to_result(
        self,
        item: Dict[str, Any],
        provider: Any,
        query_bbox: List[float]
    ) -> DiscoveryResult:
        """
        Convert STAC item to DiscoveryResult.

        Args:
            item: STAC item dictionary
            provider: Provider configuration
            query_bbox: Query bounding box

        Returns:
            DiscoveryResult object
        """
        # Extract item metadata
        item_id = item["id"]
        properties = item.get("properties", {})
        geometry = item.get("geometry")
        bbox = item.get("bbox", [])

        # Acquisition time
        datetime_str = properties.get("datetime")
        if datetime_str:
            acquisition_time = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        else:
            # Fallback to start_datetime
            acquisition_time = datetime.fromisoformat(
                properties.get("start_datetime", datetime.now().isoformat()).replace('Z', '+00:00')
            )

        # Assets (find primary data asset)
        assets = item.get("assets", {})
        source_uri, data_format = self._select_primary_asset(assets, provider)

        # Spatial coverage
        if bbox:
            spatial_coverage = self._calculate_coverage_percent(bbox, query_bbox)
        else:
            spatial_coverage = 50.0  # Default estimate

        # Resolution
        resolution_m = properties.get("gsd", provider.capabilities.get("resolution_m", 10.0))

        # Cloud cover (for optical)
        cloud_cover = None
        if provider.type == "optical":
            cloud_cover = properties.get("eo:cloud_cover")

        # Quality flag
        quality_flag = self._determine_quality_flag(properties)

        # Cost tier
        cost_tier = provider.cost.get("tier", "open")

        # Checksum (if available)
        checksum = None
        if "checksum:multihash" in properties:
            checksum = {
                "algorithm": "multihash",
                "value": properties["checksum:multihash"]
            }

        return DiscoveryResult(
            dataset_id=item_id,
            provider=provider.id,
            data_type=provider.type,
            source_uri=source_uri,
            format=data_format,
            acquisition_time=acquisition_time,
            spatial_coverage_percent=spatial_coverage,
            resolution_m=resolution_m,
            cloud_cover_percent=cloud_cover,
            quality_flag=quality_flag,
            cost_tier=cost_tier,
            checksum=checksum,
            metadata={
                "stac_item": item_id,
                "properties": properties
            }
        )

    def _select_primary_asset(
        self,
        assets: Dict[str, Any],
        provider: Any
    ) -> tuple[str, str]:
        """
        Select primary data asset from STAC item assets.

        Returns:
            (source_uri, format)
        """
        # Priority order for asset keys
        asset_priorities = [
            "data",
            "visual",
            "thumbnail",
            "B04",  # Sentinel-2 red band
            "VV",  # Sentinel-1 VV polarization
        ]

        # Try priority assets first
        for key in asset_priorities:
            if key in assets:
                asset = assets[key]
                href = asset.get("href")
                if href:
                    asset_format = self._infer_format(asset)
                    return href, asset_format

        # Fallback to first available asset
        if assets:
            first_asset = next(iter(assets.values()))
            href = first_asset.get("href", "")
            asset_format = self._infer_format(first_asset)
            return href, asset_format

        # No assets found
        return "", "unknown"

    def _infer_format(self, asset: Dict[str, Any]) -> str:
        """Infer data format from STAC asset."""
        # Check explicit format fields
        if "type" in asset:
            media_type = asset["type"]
            if "geotiff" in media_type or "tiff" in media_type:
                # Check if it's a COG
                if asset.get("roles") and "cloud-optimized" in asset["roles"]:
                    return "cog"
                return "geotiff"
            elif "netcdf" in media_type:
                return "netcdf"
            elif "zarr" in media_type:
                return "zarr"
            elif "json" in media_type:
                return "geojson"

        # Check href extension
        href = asset.get("href", "")
        if href.endswith(".tif") or href.endswith(".tiff"):
            return "cog"  # Assume COG for cloud catalogs
        elif href.endswith(".nc"):
            return "netcdf"
        elif href.endswith(".zarr"):
            return "zarr"

        return "geotiff"  # Default

    def _determine_quality_flag(self, properties: Dict[str, Any]) -> str:
        """Determine quality flag from STAC properties."""
        # Check for quality indicators
        quality_score = properties.get("quality_score")
        if quality_score is not None:
            if quality_score >= 0.9:
                return "excellent"
            elif quality_score >= 0.7:
                return "good"
            elif quality_score >= 0.5:
                return "fair"
            else:
                return "poor"

        # Check cloud cover (for optical)
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

        # Default to good
        return "good"

    async def close(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None
