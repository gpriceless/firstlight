"""
ERA5 weather data provider.

ECMWF's ERA5 reanalysis provides comprehensive atmospheric data
at 30km resolution with hourly temporal resolution.
"""

from core.data.providers.registry import Provider


def create_era5_provider() -> Provider:
    """
    Create ERA5 reanalysis provider configuration.

    ERA5 provides historical weather data including precipitation,
    temperature, wind, and many other variables.
    """
    return Provider(
        id="era5",
        provider="ecmwf",
        type="weather",
        capabilities={
            "resolution_m": 30000,  # ~30km
            "temporal_resolution": "hourly",
            "variables": [
                "temperature",
                "precipitation",
                "wind_u", "wind_v",
                "pressure",
                "humidity",
                "cloud_cover"
            ],
            "temporal_coverage": {
                "start": "1940-01-01",
                "end": None  # Updated with ~5 day delay
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://cds.climate.copernicus.eu/api/v2",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 0.5,
                "requests_per_day": 100
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
            "name": "ERA5 Reanalysis",
            "description": "ECMWF global reanalysis with hourly atmospheric data",
            "documentation_url": "https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5",
            "license": "Copernicus License",
            "citation": "ECMWF ERA5",
            "preference_score": 0.9
        }
    )


def create_era5_land_provider() -> Provider:
    """
    Create ERA5-Land provider configuration.

    Higher resolution (9km) land-focused reanalysis.
    """
    return Provider(
        id="era5_land",
        provider="ecmwf",
        type="weather",
        capabilities={
            "resolution_m": 9000,  # ~9km
            "temporal_resolution": "hourly",
            "variables": [
                "temperature",
                "precipitation",
                "soil_moisture",
                "snow_cover",
                "evaporation",
                "runoff"
            ],
            "temporal_coverage": {
                "start": "1950-01-01",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://cds.climate.copernicus.eu/api/v2",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 0.5,
                "requests_per_day": 100
            }
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*"],
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
            "name": "ERA5-Land",
            "description": "High-resolution land reanalysis from ECMWF",
            "documentation_url": "https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land",
            "license": "Copernicus License",
            "citation": "ECMWF ERA5-Land",
            "preference_score": 0.85
        }
    )
