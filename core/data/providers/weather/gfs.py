"""
GFS weather data provider.

NOAA's Global Forecast System provides weather forecasts
at ~13km resolution out to 16 days.
"""

from core.data.providers.registry import Provider


def create_gfs_provider() -> Provider:
    """
    Create GFS forecast provider configuration.

    GFS provides operational weather forecasts updated every 6 hours.
    """
    return Provider(
        id="gfs",
        provider="noaa",
        type="weather",
        capabilities={
            "resolution_m": 13000,  # ~13km (0.25 degree)
            "temporal_resolution": "3-hourly",
            "forecast_horizon_days": 16,
            "variables": [
                "temperature",
                "precipitation",
                "wind_u", "wind_v",
                "pressure",
                "humidity",
                "cloud_cover"
            ],
            "temporal_coverage": {
                "start": "2015-01-15",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "http",
            "endpoint": "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl",
            "authentication": "none",
            "rate_limit": {
                "requests_per_second": 1
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
            "name": "GFS",
            "description": "NOAA Global Forecast System operational weather forecasts",
            "documentation_url": "https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast",
            "license": "Public Domain",
            "citation": "NOAA GFS",
            "preference_score": 0.8
        }
    )
