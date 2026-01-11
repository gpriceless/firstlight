"""
Catalog API Routes.

Provides endpoints for discovering available data sources, algorithms,
and supported event types in the platform.
"""

import logging
from typing import Annotated, List, Optional

from fastapi import APIRouter, Depends, Query

from api.dependencies import (
    AlgorithmRegistryDep,
    ProviderRegistryDep,
    SettingsDep,
)
from api.models.requests import CatalogQueryParams, DataTypeCategory
from api.models.responses import (
    AlgorithmInfo,
    CatalogAlgorithmsResponse,
    CatalogEventTypesResponse,
    CatalogSourcesResponse,
    DataSourceInfo,
    EventTypeInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["Catalog"])


def _provider_to_data_source_info(provider: dict) -> DataSourceInfo:
    """Convert provider dictionary to DataSourceInfo model."""
    return DataSourceInfo(
        id=provider.get("id", "unknown"),
        name=provider.get("name", "Unknown"),
        provider=provider.get("provider", "unknown"),
        data_type=DataTypeCategory(provider.get("data_type", "ancillary")),
        description=provider.get("description"),
        resolution_m=provider.get("resolution_m"),
        revisit_days=provider.get("revisit_days"),
        bands=provider.get("bands"),
        coverage=provider.get("coverage"),
        license=provider.get("license"),
        open_data=provider.get("open_data", False),
        applicable_event_classes=provider.get("applicable_event_classes"),
    )


def _algorithm_to_info(algorithm) -> AlgorithmInfo:
    """Convert AlgorithmMetadata to AlgorithmInfo model."""
    return AlgorithmInfo(
        id=algorithm.id,
        name=algorithm.name,
        category=algorithm.category.value if hasattr(algorithm.category, "value") else str(algorithm.category),
        version=algorithm.version,
        description=algorithm.description,
        event_types=algorithm.event_types,
        required_data_types=[
            DataTypeCategory(dt.value if hasattr(dt, "value") else str(dt))
            for dt in algorithm.required_data_types
        ],
        outputs=algorithm.outputs if algorithm.outputs else None,
        deterministic=algorithm.deterministic,
    )


# Predefined event types from OpenSpec definitions
# This would be loaded from YAML files in production
_EVENT_TYPES: List[EventTypeInfo] = [
    EventTypeInfo(
        class_name="flood",
        name="Flood Events",
        description="Flooding events including riverine, coastal, flash, and urban flooding",
        parent=None,
        children=["flood.riverine", "flood.coastal", "flood.flash", "flood.urban"],
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.DEM],
        optional_data_types=[DataTypeCategory.OPTICAL, DataTypeCategory.WEATHER],
        keywords=["flood", "flooding", "inundation"],
    ),
    EventTypeInfo(
        class_name="flood.riverine",
        name="Riverine Flooding",
        description="River flooding from prolonged precipitation or snowmelt",
        parent="flood",
        children=None,
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.DEM],
        optional_data_types=[DataTypeCategory.OPTICAL],
        keywords=["river flood", "fluvial flood", "overflow"],
    ),
    EventTypeInfo(
        class_name="flood.coastal",
        name="Coastal Flooding",
        description="Coastal flooding events",
        parent="flood",
        children=["flood.coastal.storm_surge", "flood.coastal.tidal"],
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.DEM],
        optional_data_types=[DataTypeCategory.WEATHER],
        keywords=["coastal flood", "coastal flooding"],
    ),
    EventTypeInfo(
        class_name="flood.coastal.storm_surge",
        name="Storm Surge Flooding",
        description="Storm surge flooding from tropical cyclones",
        parent="flood.coastal",
        children=None,
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.DEM, DataTypeCategory.WEATHER],
        optional_data_types=[],
        keywords=["storm surge", "hurricane flooding", "typhoon flooding"],
    ),
    EventTypeInfo(
        class_name="flood.flash",
        name="Flash Flooding",
        description="Flash flooding from intense precipitation",
        parent="flood",
        children=None,
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.WEATHER],
        optional_data_types=[DataTypeCategory.DEM],
        keywords=["flash flood", "sudden flood", "rapid onset flood"],
    ),
    EventTypeInfo(
        class_name="flood.urban",
        name="Urban Flooding",
        description="Urban flooding from overwhelmed drainage",
        parent="flood",
        children=None,
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.ANCILLARY],
        optional_data_types=[DataTypeCategory.OPTICAL],
        keywords=["urban flood", "pluvial flood", "street flooding"],
    ),
    EventTypeInfo(
        class_name="wildfire",
        name="Wildfire Events",
        description="Wildfire detection and burn area mapping",
        parent=None,
        children=["wildfire.forest", "wildfire.grassland", "wildfire.interface"],
        required_data_types=[DataTypeCategory.OPTICAL],
        optional_data_types=[DataTypeCategory.SAR, DataTypeCategory.WEATHER],
        keywords=["wildfire", "fire", "burn"],
    ),
    EventTypeInfo(
        class_name="wildfire.forest",
        name="Forest Wildfire",
        description="Wildfires in forested areas",
        parent="wildfire",
        children=None,
        required_data_types=[DataTypeCategory.OPTICAL],
        optional_data_types=[DataTypeCategory.SAR],
        keywords=["forest fire", "timber fire"],
    ),
    EventTypeInfo(
        class_name="wildfire.interface",
        name="Wildland-Urban Interface Fire",
        description="Fires at the wildland-urban interface",
        parent="wildfire",
        children=None,
        required_data_types=[DataTypeCategory.OPTICAL, DataTypeCategory.ANCILLARY],
        optional_data_types=[],
        keywords=["WUI fire", "interface fire", "urban wildfire"],
    ),
    EventTypeInfo(
        class_name="storm",
        name="Storm Events",
        description="Severe storm damage assessment",
        parent=None,
        children=["storm.tropical_cyclone", "storm.severe_convective", "storm.winter"],
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.OPTICAL],
        optional_data_types=[DataTypeCategory.WEATHER],
        keywords=["storm", "severe weather"],
    ),
    EventTypeInfo(
        class_name="storm.tropical_cyclone",
        name="Tropical Cyclone",
        description="Hurricane, typhoon, and cyclone damage",
        parent="storm",
        children=None,
        required_data_types=[DataTypeCategory.SAR, DataTypeCategory.OPTICAL, DataTypeCategory.WEATHER],
        optional_data_types=[DataTypeCategory.DEM],
        keywords=["hurricane", "typhoon", "cyclone", "tropical storm"],
    ),
]


@router.get(
    "/sources",
    response_model=CatalogSourcesResponse,
    summary="List available data sources",
    description=(
        "Returns a list of all available data sources in the platform, "
        "including satellite imagery, DEMs, weather data, and ancillary sources."
    ),
)
async def list_data_sources(
    provider_registry: ProviderRegistryDep,
    settings: SettingsDep,
    data_type: Annotated[
        Optional[List[DataTypeCategory]],
        Query(description="Filter by data type category"),
    ] = None,
    event_class: Annotated[
        Optional[str],
        Query(description="Filter by compatible event class"),
    ] = None,
    available_only: Annotated[
        bool,
        Query(description="Only return currently available sources"),
    ] = False,
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Maximum number of results"),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of results to skip"),
    ] = 0,
) -> CatalogSourcesResponse:
    """
    List all available data sources.

    Supports filtering by:
    - Data type category (optical, SAR, DEM, weather, ancillary)
    - Compatible event class
    - Current availability
    """
    # Get all providers from registry
    providers = provider_registry.list_all()

    # Convert to source info objects
    sources: List[DataSourceInfo] = []
    for provider in providers:
        # Handle both dict and object providers
        if hasattr(provider, "to_dict"):
            provider_dict = provider.to_dict()
        elif isinstance(provider, dict):
            provider_dict = provider
        else:
            # Try to extract attributes
            provider_dict = {
                "id": getattr(provider, "id", "unknown"),
                "name": getattr(provider, "name", "Unknown"),
                "provider": getattr(provider, "provider_name", "unknown"),
                "data_type": getattr(provider, "data_type", "ancillary"),
                "description": getattr(provider, "description", None),
                "resolution_m": getattr(provider, "resolution_m", None),
                "revisit_days": getattr(provider, "revisit_days", None),
                "bands": getattr(provider, "bands", None),
                "open_data": getattr(provider, "open_data", True),
            }

        source = _provider_to_data_source_info(provider_dict)

        # Apply filters
        if data_type and source.data_type not in data_type:
            continue

        if event_class and source.applicable_event_classes:
            # Check if any pattern matches
            matches = False
            for pattern in source.applicable_event_classes:
                if pattern.endswith(".*"):
                    prefix = pattern[:-2]
                    if event_class.startswith(prefix):
                        matches = True
                        break
                elif pattern == event_class:
                    matches = True
                    break
            if not matches:
                continue

        sources.append(source)

    # Apply pagination
    total = len(sources)
    sources = sources[offset : offset + limit]

    return CatalogSourcesResponse(
        items=sources,
        total=total,
    )


@router.get(
    "/algorithms",
    response_model=CatalogAlgorithmsResponse,
    summary="List available algorithms",
    description=(
        "Returns a list of all available analysis algorithms, "
        "including baseline, advanced, and experimental algorithms."
    ),
)
async def list_algorithms(
    algorithm_registry: AlgorithmRegistryDep,
    settings: SettingsDep,
    category: Annotated[
        Optional[str],
        Query(description="Filter by category (baseline, advanced, experimental)"),
    ] = None,
    event_class: Annotated[
        Optional[str],
        Query(description="Filter by compatible event class"),
    ] = None,
    data_type: Annotated[
        Optional[List[DataTypeCategory]],
        Query(description="Filter by required data type"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Maximum number of results"),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of results to skip"),
    ] = 0,
) -> CatalogAlgorithmsResponse:
    """
    List all available algorithms.

    Supports filtering by:
    - Category (baseline, advanced, experimental)
    - Compatible event class
    - Required data types
    """
    # Get algorithms from registry
    if event_class:
        algorithms = algorithm_registry.search_by_event_type(event_class)
    else:
        algorithms = algorithm_registry.list_all()

    # Convert to info objects and apply filters
    algorithm_infos: List[AlgorithmInfo] = []
    for algo in algorithms:
        info = _algorithm_to_info(algo)

        # Apply category filter
        if category and info.category != category:
            continue

        # Apply data type filter
        if data_type:
            # Check if algorithm requires any of the specified data types
            has_required_type = any(dt in info.required_data_types for dt in data_type)
            if not has_required_type:
                continue

        algorithm_infos.append(info)

    # Apply pagination
    total = len(algorithm_infos)
    algorithm_infos = algorithm_infos[offset : offset + limit]

    return CatalogAlgorithmsResponse(
        items=algorithm_infos,
        total=total,
    )


@router.get(
    "/event-types",
    response_model=CatalogEventTypesResponse,
    summary="List supported event types",
    description=(
        "Returns the hierarchical taxonomy of supported event types, "
        "including required and optional data types for each."
    ),
)
async def list_event_types(
    settings: SettingsDep,
    parent: Annotated[
        Optional[str],
        Query(description="Filter by parent event class"),
    ] = None,
    data_type: Annotated[
        Optional[List[DataTypeCategory]],
        Query(description="Filter by required data type"),
    ] = None,
    limit: Annotated[
        int,
        Query(ge=1, le=200, description="Maximum number of results"),
    ] = 50,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of results to skip"),
    ] = 0,
) -> CatalogEventTypesResponse:
    """
    List all supported event types.

    Returns the hierarchical taxonomy of event types with:
    - Parent/child relationships
    - Required and optional data types
    - Keywords for NLP matching

    Supports filtering by:
    - Parent event class
    - Required data types
    """
    # Filter event types
    filtered: List[EventTypeInfo] = []
    for event_type in _EVENT_TYPES:
        # Apply parent filter
        if parent is not None:
            if event_type.parent != parent:
                continue

        # Apply data type filter
        if data_type:
            # Check if event type requires any of the specified data types
            has_required_type = any(
                dt in event_type.required_data_types for dt in data_type
            )
            if not has_required_type:
                continue

        filtered.append(event_type)

    # Apply pagination
    total = len(filtered)
    filtered = filtered[offset : offset + limit]

    return CatalogEventTypesResponse(
        items=filtered,
        total=total,
    )
