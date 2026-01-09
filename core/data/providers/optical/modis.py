"""
MODIS optical data provider.

NASA's MODIS (Moderate Resolution Imaging Spectroradiometer) provides
250m-1km multispectral imagery with daily global coverage.
"""

from core.data.providers.registry import Provider


def create_modis_terra_provider() -> Provider:
    """
    Create MODIS Terra provider configuration.

    MODIS Terra provides daily 250m-1km imagery, useful for rapid
    response and large-area monitoring.
    """
    return Provider(
        id="modis_terra",
        provider="nasa",
        type="optical",
        capabilities={
            "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
            "resolution_m": 250,  # 250m for bands 1-2, 500m for 3-7
            "revisit_days": 1,
            "temporal_coverage": {
                "start": "2000-02-24",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 5,
                "requests_per_day": 10000
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": True,
                "max_cloud_cover": 0.5
            }
        },
        cost={
            "tier": "open_restricted",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "MODIS Terra",
            "description": "Daily moderate resolution imagery from NASA Terra satellite",
            "documentation_url": "https://modis.gsfc.nasa.gov/",
            "license": "NASA Open Data",
            "citation": "NASA MODIS Terra",
            "preference_score": 0.7
        }
    )


def create_modis_aqua_provider() -> Provider:
    """Create MODIS Aqua provider configuration."""
    return Provider(
        id="modis_aqua",
        provider="nasa",
        type="optical",
        capabilities={
            "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07"],
            "resolution_m": 250,
            "revisit_days": 1,
            "temporal_coverage": {
                "start": "2002-07-04",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://ladsweb.modaps.eosdis.nasa.gov/api/v2/content",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 5,
                "requests_per_day": 10000
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": True,
                "max_cloud_cover": 0.5
            }
        },
        cost={
            "tier": "open_restricted",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "MODIS Aqua",
            "description": "Daily moderate resolution imagery from NASA Aqua satellite",
            "documentation_url": "https://modis.gsfc.nasa.gov/",
            "license": "NASA Open Data",
            "citation": "NASA MODIS Aqua",
            "preference_score": 0.7
        }
    )
