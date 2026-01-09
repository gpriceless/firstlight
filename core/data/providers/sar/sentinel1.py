"""
Sentinel-1 SAR data provider.

ESA's Copernicus Sentinel-1 provides C-band SAR imagery,
all-weather day/night capability for flood and change detection.
"""

from core.data.providers.registry import Provider


def create_sentinel1_grd_provider() -> Provider:
    """
    Create Sentinel-1 GRD (Ground Range Detected) provider configuration.

    Sentinel-1 provides 10m SAR imagery, ideal for flood mapping and
    change detection in all weather conditions.
    """
    return Provider(
        id="sentinel1_grd",
        provider="copernicus",
        type="sar",
        capabilities={
            "polarizations": ["VV", "VH", "HH", "HV"],
            "resolution_m": 10,
            "revisit_days": 6,  # 6 days with both satellites
            "temporal_coverage": {
                "start": "2014-10-03",
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
            "event_classes": ["flood.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": False  # SAR works through clouds
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Sentinel-1 GRD",
            "description": "C-band SAR imagery from ESA Sentinel-1",
            "documentation_url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
            "license": "CC-BY-4.0",
            "citation": "ESA Copernicus Sentinel-1",
            "stac_collection": "sentinel-1-grd",
            "preference_score": 0.95
        }
    )


def create_sentinel1_slc_provider() -> Provider:
    """
    Create Sentinel-1 SLC (Single Look Complex) provider configuration.

    SLC products provide phase information for interferometric applications.
    """
    return Provider(
        id="sentinel1_slc",
        provider="copernicus",
        type="sar",
        capabilities={
            "polarizations": ["VV", "VH", "HH", "HV"],
            "resolution_m": 5,
            "revisit_days": 6,
            "temporal_coverage": {
                "start": "2014-10-03",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://scihub.copernicus.eu/dhus/search",
            "authentication": "oauth2",
            "rate_limit": {
                "requests_per_second": 2
            }
        },
        applicability={
            "event_classes": ["flood.*", "storm.*"],
            "constraints": {
                "requires_clear_sky": False
            }
        },
        cost={
            "tier": "open_restricted",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Sentinel-1 SLC",
            "description": "Complex SAR data for interferometry from ESA Sentinel-1",
            "documentation_url": "https://sentinel.esa.int/web/sentinel/missions/sentinel-1",
            "license": "CC-BY-4.0",
            "citation": "ESA Copernicus Sentinel-1 SLC",
            "preference_score": 0.85
        }
    )
