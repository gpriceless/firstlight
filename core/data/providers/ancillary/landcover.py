"""
Land cover data providers.

Global land cover datasets useful for context and analysis refinement.
"""

from core.data.providers.registry import Provider


def create_esa_worldcover_provider() -> Provider:
    """
    Create ESA WorldCover provider configuration.

    WorldCover provides 10m global land cover classification
    based on Sentinel-1 and Sentinel-2.
    """
    return Provider(
        id="esa_worldcover",
        provider="esa",
        type="ancillary",
        capabilities={
            "resolution_m": 10,
            "data_type": "land_cover",
            "classes": [
                "tree_cover", "shrubland", "grassland", "cropland",
                "built_up", "bare", "snow_ice", "water", "herbaceous_wetland",
                "mangroves", "moss_lichen"
            ],
            "temporal_coverage": {
                "start": "2020-01-01",
                "end": "2020-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "wms",
            "endpoint": "https://services.terrascope.be/wms/v2",
            "authentication": "none",
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
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "ESA WorldCover",
            "description": "Global 10m land cover classification from ESA",
            "documentation_url": "https://esa-worldcover.org/",
            "license": "CC-BY-4.0",
            "citation": "ESA WorldCover 2020",
            "preference_score": 0.9
        }
    )


def create_modis_landcover_provider() -> Provider:
    """
    Create MODIS Land Cover provider configuration.

    Annual global land cover at 500m resolution.
    """
    return Provider(
        id="modis_lc",
        provider="nasa",
        type="ancillary",
        capabilities={
            "resolution_m": 500,
            "data_type": "land_cover",
            "temporal_resolution": "annual",
            "temporal_coverage": {
                "start": "2001-01-01",
                "end": "2020-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://lpdaac.usgs.gov/products/mcd12q1v006/",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 2
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
            "name": "MODIS Land Cover",
            "description": "Annual global land cover from MODIS",
            "documentation_url": "https://lpdaac.usgs.gov/products/mcd12q1v006/",
            "license": "Public Domain",
            "citation": "NASA MODIS Land Cover MCD12Q1",
            "preference_score": 0.75
        }
    )


def create_copernicus_landcover_provider() -> Provider:
    """
    Create Copernicus Global Land Cover provider configuration.

    Annual global land cover at 100m resolution.
    """
    return Provider(
        id="copernicus_lc",
        provider="copernicus",
        type="ancillary",
        capabilities={
            "resolution_m": 100,
            "data_type": "land_cover",
            "temporal_resolution": "annual",
            "temporal_coverage": {
                "start": "2015-01-01",
                "end": "2019-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "wcs",
            "endpoint": "https://land.copernicus.eu/global/products/lc",
            "authentication": "none",
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
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Copernicus Global Land Cover",
            "description": "Annual global land cover at 100m from Copernicus",
            "documentation_url": "https://land.copernicus.eu/global/products/lc",
            "license": "CC-BY-4.0",
            "citation": "Copernicus Global Land Service",
            "preference_score": 0.8
        }
    )
