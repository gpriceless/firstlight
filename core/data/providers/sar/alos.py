"""
ALOS PALSAR SAR data provider.

JAXA's ALOS-2 PALSAR-2 provides L-band SAR imagery with
deeper penetration for forest and terrain applications.
"""

from core.data.providers.registry import Provider


def create_alos2_palsar_provider() -> Provider:
    """
    Create ALOS-2 PALSAR-2 provider configuration.

    L-band SAR with better penetration through vegetation,
    useful for forest monitoring and terrain mapping.
    """
    return Provider(
        id="alos2_palsar",
        provider="jaxa",
        type="sar",
        capabilities={
            "polarizations": ["HH", "HV", "VV", "VH"],
            "resolution_m": 10,
            "revisit_days": 14,
            "temporal_coverage": {
                "start": "2014-08-04",
                "end": None
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "api",
            "endpoint": "https://gportal.jaxa.jp/gpr/search",
            "authentication": "api_key",
            "rate_limit": {
                "requests_per_second": 1,
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
            "name": "ALOS-2 PALSAR-2",
            "description": "L-band SAR imagery from JAXA ALOS-2 satellite",
            "documentation_url": "https://www.eorc.jaxa.jp/ALOS-2/en/about/overview.htm",
            "license": "JAXA Terms",
            "citation": "JAXA ALOS-2 PALSAR-2",
            "preference_score": 0.75
        }
    )


def create_alos_palsar_mosaic_provider() -> Provider:
    """
    Create ALOS PALSAR global mosaic provider configuration.

    Annual global mosaics useful for baseline terrain characterization.
    """
    return Provider(
        id="alos_palsar_mosaic",
        provider="jaxa",
        type="sar",
        capabilities={
            "polarizations": ["HH", "HV"],
            "resolution_m": 25,
            "revisit_days": 365,  # Annual mosaics
            "temporal_coverage": {
                "start": "2007-01-01",
                "end": "2010-12-31"  # ALOS-1 period
            },
            "spatial_coverage": "global"
        },
        access={
            "protocol": "wms",
            "endpoint": "https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm",
            "authentication": "none"
        },
        applicability={
            "event_classes": ["flood.*", "wildfire.*"],
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
            "name": "ALOS PALSAR Global Mosaic",
            "description": "Annual L-band SAR mosaics from ALOS PALSAR",
            "documentation_url": "https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm",
            "license": "JAXA Terms",
            "citation": "JAXA ALOS PALSAR Global Mosaic",
            "preference_score": 0.65
        }
    )
