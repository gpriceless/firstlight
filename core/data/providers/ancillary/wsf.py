"""
World Settlement Footprint (WSF) data provider.

DLR's WSF provides global settlement extent data derived from
Sentinel-1 and Landsat imagery.
"""

from core.data.providers.registry import Provider


def create_wsf_2019_provider() -> Provider:
    """
    Create World Settlement Footprint 2019 provider configuration.

    WSF2019 provides 10m resolution global settlement mapping.
    """
    return Provider(
        id="wsf_2019",
        provider="dlr",
        type="ancillary",
        capabilities={
            "resolution_m": 10,
            "data_type": "settlement_mask",
            "temporal_coverage": {
                "start": "2019-01-01",
                "end": "2019-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "wms",
            "endpoint": "https://geoservice.dlr.de/eoc/land/wms",
            "authentication": "none",
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
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "World Settlement Footprint 2019",
            "description": "Global 10m settlement extent from DLR",
            "documentation_url": "https://geoservice.dlr.de/web/dataguide/wsf2019/",
            "license": "CC-BY-4.0",
            "citation": "DLR WSF 2019",
            "preference_score": 0.85
        }
    )


def create_wsf_evolution_provider() -> Provider:
    """
    Create World Settlement Footprint Evolution provider configuration.

    Multi-temporal settlement growth from 1985-2015.
    """
    return Provider(
        id="wsf_evolution",
        provider="dlr",
        type="ancillary",
        capabilities={
            "resolution_m": 30,
            "data_type": "settlement_evolution",
            "temporal_coverage": {
                "start": "1985-01-01",
                "end": "2015-12-31"
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "wms",
            "endpoint": "https://geoservice.dlr.de/eoc/land/wms",
            "authentication": "none",
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
            "tier": "open",
            "unit_cost": 0.0,
            "currency": "USD"
        },
        metadata={
            "name": "World Settlement Footprint Evolution",
            "description": "Global settlement growth 1985-2015 from DLR",
            "documentation_url": "https://geoservice.dlr.de/web/dataguide/wsf-evolution/",
            "license": "CC-BY-4.0",
            "citation": "DLR WSF Evolution",
            "preference_score": 0.75
        }
    )
