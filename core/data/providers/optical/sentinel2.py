"""
Sentinel-2 optical data provider.

ESA's Copernicus Sentinel-2 mission provides high-resolution multispectral imagery.
"""

from core.data.providers.registry import Provider


def create_sentinel2_provider() -> Provider:
    """
    Create Sentinel-2 L2A provider configuration.

    Sentinel-2 provides 10-20m multispectral imagery with 5-day revisit.
    Access via Element84 Earth Search STAC catalog.
    """
    return Provider(
        id="sentinel2_l2a",
        provider="copernicus",
        type="optical",
        capabilities={
            "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
            "resolution_m": 10,
            "revisit_days": 5,
            "temporal_coverage": {
                "start": "2015-06-23",
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
            "name": "Sentinel-2 Level-2A",
            "description": "Bottom-of-atmosphere reflectance from ESA Sentinel-2",
            "documentation_url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-2",
            "license": "CC-BY-4.0",
            "citation": "ESA Copernicus Sentinel-2",
            "stac_collection": "sentinel-2-l2a",
            "preference_score": 0.9
        }
    )
