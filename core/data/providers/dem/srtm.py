"""
SRTM DEM provider.

NASA's Shuttle Radar Topography Mission provides 30m and 90m
global elevation data.
"""

from core.data.providers.registry import Provider


def create_srtm_30_provider() -> Provider:
    """
    Create SRTM 30m provider configuration.

    SRTM provides 30m elevation data for most of the globe
    (between 60°N and 56°S).
    """
    return Provider(
        id="srtm_30",
        provider="nasa",
        type="dem",
        capabilities={
            "resolution_m": 30,
            "vertical_accuracy_m": 16,
            "temporal_coverage": {
                "start": "2000-02-11",
                "end": "2000-02-22"  # 11-day mission
            },
            "spatial_coverage": "regional"  # 60°N to 56°S
        },
        access={
            "protocol": "api",
            "endpoint": "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 5
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
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
            "name": "SRTM 30m",
            "description": "Shuttle Radar Topography Mission elevation data at 30m",
            "documentation_url": "https://lpdaac.usgs.gov/products/srtmgl1v003/",
            "license": "Public Domain",
            "citation": "NASA SRTM v3.0",
            "preference_score": 0.8
        }
    )


def create_srtm_90_provider() -> Provider:
    """Create SRTM 90m provider configuration."""
    return Provider(
        id="srtm_90",
        provider="nasa",
        type="dem",
        capabilities={
            "resolution_m": 90,
            "vertical_accuracy_m": 16,
            "temporal_coverage": {
                "start": "2000-02-11",
                "end": "2000-02-22"
            },
            "spatial_coverage": "regional"
        },
        access={
            "protocol": "api",
            "endpoint": "https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL3.003/2000.02.11/",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 5
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*", "storm.*"],
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
            "name": "SRTM 90m",
            "description": "Shuttle Radar Topography Mission elevation data at 90m",
            "documentation_url": "https://lpdaac.usgs.gov/products/srtmgl3v003/",
            "license": "Public Domain",
            "citation": "NASA SRTM v3.0",
            "preference_score": 0.75
        }
    )
