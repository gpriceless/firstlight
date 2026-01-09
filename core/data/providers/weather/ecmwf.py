"""
ECMWF weather data provider.

European Centre for Medium-Range Weather Forecasts provides
high-quality weather forecasts.
"""

from core.data.providers.registry import Provider


def create_ecmwf_hres_provider() -> Provider:
    """
    Create ECMWF HRES (High Resolution) forecast provider configuration.

    ECMWF provides operational weather forecasts at ~9km resolution
    out to 10 days.
    """
    return Provider(
        id="ecmwf_hres",
        provider="ecmwf",
        type="weather",
        capabilities={
            "resolution_m": 9000,  # ~9km
            "temporal_resolution": "hourly",
            "forecast_horizon_days": 10,
            "variables": [
                "temperature",
                "precipitation",
                "wind_u", "wind_v",
                "pressure",
                "humidity",
                "cloud_cover"
            ],
            "temporal_coverage": {
                "start": "2016-01-01",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://api.ecmwf.int/v1",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 0.5,
                "requests_per_day": 50
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
            "name": "ECMWF HRES",
            "description": "ECMWF high-resolution operational weather forecasts",
            "documentation_url": "https://www.ecmwf.int/en/forecasts/datasets",
            "license": "ECMWF License",
            "citation": "ECMWF HRES",
            "preference_score": 0.85
        }
    )


def create_ecmwf_ensemble_provider() -> Provider:
    """
    Create ECMWF ENS (Ensemble) forecast provider configuration.

    Probabilistic forecasts using 51-member ensemble.
    """
    return Provider(
        id="ecmwf_ens",
        provider="ecmwf",
        type="weather",
        capabilities={
            "resolution_m": 18000,  # ~18km
            "temporal_resolution": "3-hourly",
            "forecast_horizon_days": 15,
            "ensemble_members": 51,
            "variables": [
                "temperature",
                "precipitation",
                "wind_u", "wind_v",
                "pressure"
            ],
            "temporal_coverage": {
                "start": "2016-01-01",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://api.ecmwf.int/v1",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 0.5,
                "requests_per_day": 50
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
            "name": "ECMWF ENS",
            "description": "ECMWF ensemble probabilistic weather forecasts",
            "documentation_url": "https://www.ecmwf.int/en/forecasts/datasets",
            "license": "ECMWF License",
            "citation": "ECMWF ENS",
            "preference_score": 0.75
        }
    )
