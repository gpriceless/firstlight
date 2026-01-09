"""
Provider loader utility.

Loads and registers all provider implementations into the registry.
"""

from core.data.providers.registry import ProviderRegistry

# Import all provider factory functions
from core.data.providers.optical.sentinel2 import create_sentinel2_provider
from core.data.providers.optical.landsat import (
    create_landsat8_provider,
    create_landsat9_provider
)
from core.data.providers.optical.modis import (
    create_modis_terra_provider,
    create_modis_aqua_provider
)

from core.data.providers.sar.sentinel1 import (
    create_sentinel1_grd_provider,
    create_sentinel1_slc_provider
)
from core.data.providers.sar.alos import (
    create_alos2_palsar_provider,
    create_alos_palsar_mosaic_provider
)

from core.data.providers.dem.copernicus import (
    create_copernicus_dem_30_provider,
    create_copernicus_dem_90_provider
)
from core.data.providers.dem.srtm import (
    create_srtm_30_provider,
    create_srtm_90_provider
)
from core.data.providers.dem.fabdem import create_fabdem_provider

from core.data.providers.weather.era5 import (
    create_era5_provider,
    create_era5_land_provider
)
from core.data.providers.weather.gfs import create_gfs_provider
from core.data.providers.weather.ecmwf import (
    create_ecmwf_hres_provider,
    create_ecmwf_ensemble_provider
)

from core.data.providers.ancillary.osm import create_osm_provider
from core.data.providers.ancillary.wsf import (
    create_wsf_2019_provider,
    create_wsf_evolution_provider
)
from core.data.providers.ancillary.landcover import (
    create_esa_worldcover_provider,
    create_modis_landcover_provider,
    create_copernicus_landcover_provider
)


def load_all_providers(registry: ProviderRegistry) -> None:
    """
    Load all provider definitions into the registry.

    Args:
        registry: ProviderRegistry instance to populate
    """
    # Optical providers
    registry.register(create_sentinel2_provider())
    registry.register(create_landsat8_provider())
    registry.register(create_landsat9_provider())
    registry.register(create_modis_terra_provider())
    registry.register(create_modis_aqua_provider())

    # SAR providers
    registry.register(create_sentinel1_grd_provider())
    registry.register(create_sentinel1_slc_provider())
    registry.register(create_alos2_palsar_provider())
    registry.register(create_alos_palsar_mosaic_provider())

    # DEM providers
    registry.register(create_copernicus_dem_30_provider())
    registry.register(create_copernicus_dem_90_provider())
    registry.register(create_srtm_30_provider())
    registry.register(create_srtm_90_provider())
    registry.register(create_fabdem_provider())

    # Weather providers
    registry.register(create_era5_provider())
    registry.register(create_era5_land_provider())
    registry.register(create_gfs_provider())
    registry.register(create_ecmwf_hres_provider())
    registry.register(create_ecmwf_ensemble_provider())

    # Ancillary providers
    registry.register(create_osm_provider())
    registry.register(create_wsf_2019_provider())
    registry.register(create_wsf_evolution_provider())
    registry.register(create_esa_worldcover_provider())
    registry.register(create_modis_landcover_provider())
    registry.register(create_copernicus_landcover_provider())


def create_default_registry() -> ProviderRegistry:
    """
    Create and populate a ProviderRegistry with all default providers.

    Returns:
        ProviderRegistry with all providers loaded
    """
    registry = ProviderRegistry()
    load_all_providers(registry)
    return registry
