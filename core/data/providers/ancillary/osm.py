"""
OpenStreetMap data provider.

OSM provides vector data for buildings, roads, and infrastructure
useful for impact assessment.
"""

from core.data.providers.registry import Provider


def create_osm_provider() -> Provider:
    """
    Create OpenStreetMap provider configuration.

    OSM provides crowdsourced vector data for buildings, roads,
    and other infrastructure features.
    """
    return Provider(
        id="osm",
        provider="openstreetmap",
        type="ancillary",
        capabilities={
            "data_types": ["buildings", "roads", "waterways", "landuse"],
            "format": "vector",
            "temporal_coverage": {
                "start": "2004-08-09",
                "end": None  # Continuously updated
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://overpass-api.de/api/interpreter",
            "authentication": "none",
            "rate_limit": {
                "requests_per_second": 0.5
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": False
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "OpenStreetMap",
            "description": "Crowdsourced vector data for infrastructure and features",
            "documentation_url": "https://wiki.openstreetmap.org/",
            "license": "ODbL",
            "citation": "OpenStreetMap Contributors",
            "preference_score": 0.9
        }
    )
