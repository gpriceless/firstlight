"""
Copernicus DEM provider.

ESA's Copernicus Digital Elevation Model provides global elevation data
at 30m and 90m resolution.
"""

from core.data.providers.registry import Provider


def create_copernicus_dem_30_provider() -> Provider:
    """
    Create Copernicus DEM 30m provider configuration.

    High-resolution global DEM at 30m, derived from TanDEM-X.
    """
    return Provider(
        id="copernicus_dem_30",
        provider="copernicus",
        type="dem",
        capabilities={
            "resolution_m": 30,
            "vertical_accuracy_m": 4,
            "temporal_coverage": {
                "start": "2011-01-01",
                "end": "2015-12-31"  # Acquisition period
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
                "requires_clear_sky": False
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Copernicus DEM 30m",
            "description": "Global elevation model at 30m resolution from ESA",
            "documentation_url": "https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model",
            "license": "CC-BY-4.0",
            "citation": "ESA Copernicus DEM",
            "stac_collection": "cop-dem-glo-30",
            "preference_score": 0.9
        }
    )


def create_copernicus_dem_90_provider() -> Provider:
    """Create Copernicus DEM 90m provider configuration."""
    return Provider(
        id="copernicus_dem_90",
        provider="copernicus",
        type="dem",
        capabilities={
            "resolution_m": 90,
            "vertical_accuracy_m": 4,
            "temporal_coverage": {
                "start": "2011-01-01",
                "end": "2015-12-31"
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
                "requires_clear_sky": False
            }
        },
        cost={
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "Copernicus DEM 90m",
            "description": "Global elevation model at 90m resolution from ESA",
            "documentation_url": "https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model",
            "license": "CC-BY-4.0",
            "citation": "ESA Copernicus DEM",
            "stac_collection": "cop-dem-glo-90",
            "preference_score": 0.85
        }
    )
