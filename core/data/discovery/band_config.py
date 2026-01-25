"""
Band configuration for satellite sensors.

Defines which bands to extract from STAC assets for each sensor type.
"""

from typing import Dict, List

# Sentinel-2 L2A band configuration (Earth Search STAC asset keys)
SENTINEL2_BANDS = {
    "blue": "blue",       # B02, 490nm, 10m
    "green": "green",     # B03, 560nm, 10m
    "red": "red",         # B04, 665nm, 10m
    "nir": "nir",         # B08, 842nm, 10m
    "swir16": "swir16",   # B11, 1610nm, 20m
    "swir22": "swir22",   # B12, 2190nm, 20m
}

SENTINEL2_CANONICAL_ORDER = ["blue", "green", "red", "nir", "swir16", "swir22"]

# Landsat 8/9 band configuration
LANDSAT_BANDS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
}

LANDSAT_CANONICAL_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Sentinel-1 (SAR) configuration
SENTINEL1_BANDS = {
    "vv": "vv",
    "vh": "vh",
}

SENTINEL1_CANONICAL_ORDER = ["vv", "vh"]


def get_band_config(source: str) -> Dict[str, str]:
    """Get band configuration for a sensor type."""
    configs = {
        "sentinel2": SENTINEL2_BANDS,
        "landsat": LANDSAT_BANDS,
        "sentinel1": SENTINEL1_BANDS,
    }
    return configs.get(source, {})


def get_canonical_order(source: str) -> List[str]:
    """Get canonical band order for a sensor type."""
    orders = {
        "sentinel2": SENTINEL2_CANONICAL_ORDER,
        "landsat": LANDSAT_CANONICAL_ORDER,
        "sentinel1": SENTINEL1_CANONICAL_ORDER,
    }
    return orders.get(source, [])
