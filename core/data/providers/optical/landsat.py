"""
Landsat optical data provider.

USGS/NASA Landsat program provides 30m multispectral imagery dating back to 1972.
"""

from core.data.providers.registry import Provider


def create_landsat8_provider() -> Provider:
    """
    Create Landsat-8 Collection 2 Level-2 provider configuration.

    Landsat-8 provides 30m multispectral imagery with 16-day revisit.
    Access via Element84 Earth Search STAC catalog.
    """
    return Provider(
        id="landsat8_c2l2",
        provider="usgs",
        type="optical",
        capabilities={
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "resolution_m": 30,
            "revisit_days": 16,
            "temporal_coverage": {
                "start": "2013-04-11",
                "end": None  # Ongoing
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "stac",
            "endpoint": "https://earth-search.aws.element84.com/v1",
            "authentication": "none",
            "rate_limit": {
                "requests_per_second": 10
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": True,
                "max_cloud_cover": 0.3
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Landsat-8 Collection 2 Level-2",
            "description": "Surface reflectance from USGS Landsat-8 OLI",
            "documentation_url": "https://www.usgs.gov/landsat-missions/landsat-8",
            "license": "Public Domain",
            "citation": "USGS Landsat-8 Collection 2",
            "stac_collection": "landsat-c2-l2",
            "preference_score": 0.8
        }
    )


def create_landsat9_provider() -> Provider:
    """Create Landsat-9 Collection 2 Level-2 provider configuration."""
    return Provider(
        id="landsat9_c2l2",
        provider="usgs",
        type="optical",
        capabilities={
            "bands": ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
            "resolution_m": 30,
            "revisit_days": 16,
            "temporal_coverage": {
                "start": "2021-10-31",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "stac",
            "endpoint": "https://earth-search.aws.element84.com/v1",
            "authentication": "none",
            "rate_limit": {
                "requests_per_second": 10
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": True,
                "max_cloud_cover": 0.3
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Landsat-9 Collection 2 Level-2",
            "description": "Surface reflectance from USGS Landsat-9 OLI-2",
            "documentation_url": "https://www.usgs.gov/landsat-missions/landsat-9",
            "license": "Public Domain",
            "citation": "USGS Landsat-9 Collection 2",
            "stac_collection": "landsat-c2-l2",
            "preference_score": 0.85
        }
    )
