"""
FABDEM provider.

Forest And Buildings removed Copernicus DEM - removes vegetation and
building height bias from Copernicus DEM for better bare-earth elevation.
"""

from core.data.providers.registry import Provider


def create_fabdem_provider() -> Provider:
    """
    Create FABDEM provider configuration.

    FABDEM provides bias-corrected elevation by removing vegetation
    and building heights from Copernicus DEM.
    """
    return Provider(
        id="fabdem",
        provider="bristol",
        type="dem",
        capabilities={
            "resolution_m": 30,
            "vertical_accuracy_m": 5,
            "temporal_coverage": {
                "start": "2011-01-01",
                "end": "2015-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "http",
            "endpoint": "https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn",
            "authentication": "none",
            "rate_limit": {
                "requests_per_second": 2
            }
        },
        applicability={
            "event_classes": ["flood.*"],  # Especially useful for flood modeling
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
            "name": "FABDEM",
            "description": "Forest And Buildings removed Copernicus DEM",
            "documentation_url": "https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn",
            "license": "CC-BY-4.0",
            "citation": "Hawker et al. (2022) FABDEM",
            "preference_score": 0.95  # Preferred for flood modeling
        }
    )
