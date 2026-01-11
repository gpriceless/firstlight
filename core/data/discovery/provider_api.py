"""
Provider-specific API discovery adapter.

Generic adapter for custom provider APIs that don't conform to STAC or OGC standards.
Providers can implement custom discovery logic here.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any
import aiohttp

from core.data.discovery.base import (
    DiscoveryAdapter,
    DiscoveryResult,
    DiscoveryError
)


class ProviderAPIAdapter(DiscoveryAdapter):
    """
    Generic provider API discovery adapter.

    Supports custom provider APIs with flexible query mechanisms.
    Provider-specific logic can be added via strategy pattern.
    """

    def __init__(self):
        super().__init__("provider_api")
        self._session: Optional[aiohttp.ClientSession] = None
        self._provider_strategies: Dict[str, Any] = {}

    def register_strategy(self, provider_id: str, strategy: Any):
        """
        Register a custom strategy for a specific provider.

        Args:
            provider_id: Provider identifier
            strategy: Strategy object with discover() method
        """
        self._provider_strategies[provider_id] = strategy

    async def discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[DiscoveryResult]:
        """
        Discover datasets via custom provider API.

        Args:
            provider: Provider configuration
            spatial: GeoJSON geometry
            temporal: Temporal extent
            constraints: Optional constraints

        Returns:
            List of DiscoveryResult objects
        """
        constraints = constraints or {}

        # Check if custom strategy exists for this provider
        if provider.id in self._provider_strategies:
            strategy = self._provider_strategies[provider.id]
            return await strategy.discover(
                provider=provider,
                spatial=spatial,
                temporal=temporal,
                constraints=constraints
            )

        # Fallback to generic REST API query
        return await self._generic_api_discover(
            provider=provider,
            spatial=spatial,
            temporal=temporal,
            constraints=constraints
        )

    def supports_provider(self, provider: Any) -> bool:
        """Check if provider uses custom API protocol."""
        protocol = provider.access.get("protocol", "")
        return protocol in ["api", "http", "https"]

    async def _generic_api_discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> List[DiscoveryResult]:
        """
        Generic REST API discovery.

        Assumes a simple REST API that accepts JSON query parameters.
        """
        access = provider.access
        endpoint = access["endpoint"]

        # Build query parameters
        query_params = self._build_query_params(
            spatial=spatial,
            temporal=temporal,
            constraints=constraints
        )

        # Create session if needed
        if self._session is None:
            self._session = aiohttp.ClientSession()

        try:
            # Make API request
            async with self._session.post(
                endpoint,
                json=query_params,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                data = await response.json()

            # Parse response (assumes response has 'results' array)
            results = []
            items = data.get("results", data.get("items", []))

            query_bbox = self._extract_bbox(spatial)

            for item in items:
                result = self._parse_api_result(
                    item=item,
                    provider=provider,
                    query_bbox=query_bbox
                )
                if result:
                    results.append(result)

            return results

        except aiohttp.ClientError as e:
            raise DiscoveryError(
                f"Provider API request failed: {str(e)}",
                provider=provider.id
            )
        except Exception as e:
            # Log error but return empty results
            print(f"Error parsing provider API response: {e}")
            return []

    def _build_query_params(
        self,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build generic query parameters."""
        bbox = self._extract_bbox(spatial)
        start_dt, end_dt = self._parse_temporal_extent(temporal)

        return {
            "bbox": bbox,
            "start_date": start_dt.isoformat(),
            "end_date": end_dt.isoformat(),
            "constraints": constraints
        }

    def _parse_api_result(
        self,
        item: Dict[str, Any],
        provider: Any,
        query_bbox: List[float]
    ) -> Optional[DiscoveryResult]:
        """
        Parse generic API result into DiscoveryResult.

        Attempts to extract common fields from various API response formats.
        """
        try:
            # Extract common fields (with fallbacks)
            dataset_id = item.get("id", item.get("dataset_id", f"unknown_{id(item)}"))

            # Acquisition time
            date_field = item.get("date", item.get("acquisition_date", item.get("datetime")))
            if date_field:
                if isinstance(date_field, str):
                    acquisition_time = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                else:
                    acquisition_time = datetime.now()
            else:
                acquisition_time = datetime.now()

            # Source URI
            source_uri = item.get("url", item.get("href", item.get("download_url", "")))

            # Format
            data_format = item.get("format", "geotiff")

            # Spatial coverage
            bbox = item.get("bbox", item.get("bounding_box"))
            if bbox:
                spatial_coverage = self._calculate_coverage_percent(bbox, query_bbox)
            else:
                spatial_coverage = 50.0  # Default estimate

            # Resolution
            resolution_m = item.get("resolution", provider.capabilities.get("resolution_m", 10.0))

            # Cloud cover (optional)
            cloud_cover = item.get("cloud_cover")

            # Quality
            quality_flag = item.get("quality", "good")

            # Cost tier
            cost_tier = provider.cost.get("tier", "open")

            return DiscoveryResult(
                dataset_id=dataset_id,
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
                metadata=item
            )

        except Exception as e:
            print(f"Error parsing API result item: {e}")
            return None

    async def close(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None


class ProviderStrategy(ABC):
    """
    Abstract base class for provider-specific discovery strategies.

    Providers with unique APIs can implement this interface for
    custom discovery logic.
    """

    @abstractmethod
    async def discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Dict[str, Any]
    ) -> List[DiscoveryResult]:
        """
        Implement provider-specific discovery logic.

        Returns:
            List of DiscoveryResult objects
        """
        pass
