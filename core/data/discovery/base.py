"""
Base discovery adapter interface.

Defines abstract interface for all discovery adapters (STAC, WMS/WCS, custom APIs).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class DiscoveryResult:
    """
    Standardized discovery result.

    Represents a discovered dataset candidate with all metadata
    needed for evaluation and selection.
    """

    # Identity
    dataset_id: str
    provider: str
    data_type: str  # optical, sar, dem, weather, ancillary

    # Access
    source_uri: str
    format: str  # geotiff, cog, netcdf, zarr, etc.

    # Temporal
    acquisition_time: datetime

    # Spatial
    spatial_coverage_percent: float  # 0-100
    resolution_m: float

    # Optional metadata
    cloud_cover_percent: Optional[float] = None
    quality_flag: Optional[str] = None
    cost_tier: Optional[str] = None
    checksum: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset_id": self.dataset_id,
            "provider": self.provider,
            "data_type": self.data_type,
            "source_uri": self.source_uri,
            "format": self.format,
            "acquisition_time": self.acquisition_time.isoformat(),
            "spatial_coverage_percent": self.spatial_coverage_percent,
            "resolution_m": self.resolution_m,
            "cloud_cover_percent": self.cloud_cover_percent,
            "quality_flag": self.quality_flag,
            "cost_tier": self.cost_tier,
            "checksum": self.checksum,
            "metadata": self.metadata or {}
        }


class DiscoveryAdapter(ABC):
    """
    Abstract base class for discovery adapters.

    Subclasses implement specific discovery protocols:
    - STAC: SpatioTemporal Asset Catalogs
    - WMS/WCS: OGC Web Services
    - Provider APIs: Custom provider-specific APIs
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def discover(
        self,
        provider: Any,
        spatial: Dict[str, Any],
        temporal: Dict[str, str],
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[DiscoveryResult]:
        """
        Discover datasets matching spatial/temporal criteria.

        Args:
            provider: Provider configuration object
            spatial: GeoJSON geometry or bbox
            temporal: Temporal extent with start/end/reference_time
            constraints: Data-type-specific constraints

        Returns:
            List of DiscoveryResult objects
        """
        pass

    @abstractmethod
    def supports_provider(self, provider: Any) -> bool:
        """
        Check if this adapter supports the given provider.

        Args:
            provider: Provider configuration object

        Returns:
            True if adapter can query this provider
        """
        pass

    def _parse_temporal_extent(self, temporal: Dict[str, str]) -> tuple[datetime, datetime]:
        """Parse temporal extent into datetime objects."""
        start = datetime.fromisoformat(temporal["start"].replace('Z', '+00:00'))
        end = datetime.fromisoformat(temporal["end"].replace('Z', '+00:00'))
        return start, end

    def _extract_bbox(self, spatial: Dict[str, Any]) -> List[float]:
        """
        Extract bounding box from GeoJSON geometry.

        Returns:
            [west, south, east, north]
        """
        if "bbox" in spatial:
            return spatial["bbox"]

        # Calculate bbox from coordinates
        geom_type = spatial.get("type")
        coords = spatial.get("coordinates", [])

        if geom_type == "Point":
            lon, lat = coords
            return [lon, lat, lon, lat]

        elif geom_type == "Polygon":
            # First ring is outer boundary
            lons = [point[0] for point in coords[0]]
            lats = [point[1] for point in coords[0]]
            return [min(lons), min(lats), max(lons), max(lats)]

        elif geom_type == "MultiPolygon":
            all_lons = []
            all_lats = []
            for polygon in coords:
                for point in polygon[0]:  # First ring of each polygon
                    all_lons.append(point[0])
                    all_lats.append(point[1])
            return [min(all_lons), min(all_lats), max(all_lons), max(all_lats)]

        # Default to global if can't parse
        return [-180, -90, 180, 90]

    def _calculate_coverage_percent(
        self,
        dataset_bbox: List[float],
        query_bbox: List[float]
    ) -> float:
        """
        Calculate approximate spatial coverage percentage.

        Args:
            dataset_bbox: [west, south, east, north] of dataset
            query_bbox: [west, south, east, north] of query AOI

        Returns:
            Coverage percentage (0-100)
        """
        # Calculate intersection
        int_west = max(dataset_bbox[0], query_bbox[0])
        int_south = max(dataset_bbox[1], query_bbox[1])
        int_east = min(dataset_bbox[2], query_bbox[2])
        int_north = min(dataset_bbox[3], query_bbox[3])

        # Check if there's overlap
        if int_west >= int_east or int_south >= int_north:
            return 0.0

        # Calculate areas (simplified - not accounting for projection)
        int_area = (int_east - int_west) * (int_north - int_south)
        query_area = (query_bbox[2] - query_bbox[0]) * (query_bbox[3] - query_bbox[1])

        if query_area == 0:
            return 0.0

        coverage = (int_area / query_area) * 100.0
        return min(coverage, 100.0)


class DiscoveryError(Exception):
    """Exception raised during discovery operations."""

    def __init__(self, message: str, provider: Optional[str] = None):
        self.message = message
        self.provider = provider
        super().__init__(self.message)
