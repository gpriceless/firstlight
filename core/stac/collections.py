"""
STAC Collection definitions for FirstLight event types.

Each supported event type (flood, wildfire, storm) gets its own STAC
Collection. Collections define the spatial and temporal extent of
analysis results for that event type.

These collections are registered in pgSTAC on application startup.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Default global spatial extent (WGS84)
DEFAULT_SPATIAL_EXTENT = {
    "bbox": [[-180, -90, 180, 90]],
}

# Default temporal extent (open-ended from 2024)
DEFAULT_TEMPORAL_EXTENT = {
    "interval": [["2024-01-01T00:00:00Z", None]],
}


def get_flood_collection() -> Dict[str, Any]:
    """Build the STAC Collection for flood event analyses."""
    return {
        "type": "Collection",
        "id": "flood",
        "stac_version": "1.0.0",
        "title": "FirstLight Flood Analysis Results",
        "description": (
            "Geospatial analysis results for flood events processed by "
            "FirstLight. Includes flood extent maps, depth estimates, "
            "and impact assessments derived from satellite imagery."
        ),
        "license": "proprietary",
        "extent": {
            "spatial": DEFAULT_SPATIAL_EXTENT,
            "temporal": DEFAULT_TEMPORAL_EXTENT,
        },
        "keywords": [
            "flood",
            "inundation",
            "SAR",
            "optical",
            "disaster",
            "firstlight",
        ],
        "providers": [
            {
                "name": "FirstLight",
                "description": "Geospatial Event Intelligence Platform",
                "roles": ["producer", "processor"],
                "url": "https://firstlight.example.com",
            }
        ],
        "summaries": {
            "platform": ["sentinel-1", "sentinel-2", "landsat-8", "landsat-9"],
            "instruments": ["c-sar", "msi", "oli", "tirs"],
        },
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": "/stac/collections/flood",
            },
            {
                "rel": "items",
                "type": "application/geo+json",
                "href": "/stac/collections/flood/items",
            },
            {
                "rel": "root",
                "type": "application/json",
                "href": "/stac/",
            },
        ],
    }


def get_wildfire_collection() -> Dict[str, Any]:
    """Build the STAC Collection for wildfire event analyses."""
    return {
        "type": "Collection",
        "id": "wildfire",
        "stac_version": "1.0.0",
        "title": "FirstLight Wildfire Analysis Results",
        "description": (
            "Geospatial analysis results for wildfire events processed by "
            "FirstLight. Includes burn severity maps, active fire detections, "
            "and progression tracking derived from satellite imagery."
        ),
        "license": "proprietary",
        "extent": {
            "spatial": DEFAULT_SPATIAL_EXTENT,
            "temporal": DEFAULT_TEMPORAL_EXTENT,
        },
        "keywords": [
            "wildfire",
            "burn",
            "fire",
            "thermal",
            "NBR",
            "dNBR",
            "firstlight",
        ],
        "providers": [
            {
                "name": "FirstLight",
                "description": "Geospatial Event Intelligence Platform",
                "roles": ["producer", "processor"],
                "url": "https://firstlight.example.com",
            }
        ],
        "summaries": {
            "platform": ["sentinel-2", "landsat-8", "landsat-9", "viirs"],
            "instruments": ["msi", "oli", "tirs", "viirs-i"],
        },
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": "/stac/collections/wildfire",
            },
            {
                "rel": "items",
                "type": "application/geo+json",
                "href": "/stac/collections/wildfire/items",
            },
            {
                "rel": "root",
                "type": "application/json",
                "href": "/stac/",
            },
        ],
    }


def get_storm_collection() -> Dict[str, Any]:
    """Build the STAC Collection for storm event analyses."""
    return {
        "type": "Collection",
        "id": "storm",
        "stac_version": "1.0.0",
        "title": "FirstLight Storm Analysis Results",
        "description": (
            "Geospatial analysis results for storm events processed by "
            "FirstLight. Includes wind damage assessments, storm surge "
            "extent, and infrastructure impact maps."
        ),
        "license": "proprietary",
        "extent": {
            "spatial": DEFAULT_SPATIAL_EXTENT,
            "temporal": DEFAULT_TEMPORAL_EXTENT,
        },
        "keywords": [
            "storm",
            "hurricane",
            "cyclone",
            "wind",
            "surge",
            "firstlight",
        ],
        "providers": [
            {
                "name": "FirstLight",
                "description": "Geospatial Event Intelligence Platform",
                "roles": ["producer", "processor"],
                "url": "https://firstlight.example.com",
            }
        ],
        "summaries": {
            "platform": ["sentinel-1", "sentinel-2", "landsat-8"],
            "instruments": ["c-sar", "msi", "oli"],
        },
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": "/stac/collections/storm",
            },
            {
                "rel": "items",
                "type": "application/geo+json",
                "href": "/stac/collections/storm/items",
            },
            {
                "rel": "root",
                "type": "application/json",
                "href": "/stac/",
            },
        ],
    }


def get_all_collections() -> List[Dict[str, Any]]:
    """Return all STAC collections for FirstLight event types."""
    return [
        get_flood_collection(),
        get_wildfire_collection(),
        get_storm_collection(),
    ]


async def register_collections_in_pgstac(
    dsn: str,
) -> int:
    """
    Register all FirstLight STAC collections in pgSTAC.

    Uses pypgstac's load functionality to upsert collections.
    Falls back to direct SQL insert if pypgstac is not available.

    Args:
        dsn: PostgreSQL connection string.

    Returns:
        Number of collections registered.
    """
    import json

    collections = get_all_collections()
    registered = 0

    try:
        import asyncpg

        conn = await asyncpg.connect(dsn)
        try:
            for collection in collections:
                await conn.execute(
                    """
                    INSERT INTO pgstac.collections (id, content)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        content = EXCLUDED.content,
                        updated_at = now()
                    """,
                    collection["id"],
                    json.dumps(collection),
                )
                registered += 1
                logger.info(
                    "Registered STAC collection: %s", collection["id"]
                )
        finally:
            await conn.close()
    except ImportError:
        logger.warning("asyncpg not available, cannot register STAC collections")
    except Exception as e:
        logger.warning("Failed to register STAC collections: %s", e)

    return registered
