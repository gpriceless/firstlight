"""
STAC Item publisher for completed FirstLight analysis jobs.

Converts a completed job's results into a STAC Item with:
- processing:lineage, processing:software, processing:datetime extension fields
- AOI as Item geometry (from ST_AsGeoJSON(aoi))
- derived_from links for source imagery STAC Items
- COG raster assets with proper content type

The publisher reads job data from the jobs and job_events tables,
builds a valid STAC Item, and upserts it into pgSTAC.

Task 4.7
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# STAC Item content type for COG assets
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"

# FirstLight software version for processing extension
FIRSTLIGHT_SOFTWARE = {"firstlight": "0.1.0"}


def build_stac_item(
    job_id: str,
    event_type: str,
    aoi_geojson: Dict[str, Any],
    parameters: Optional[Dict[str, Any]] = None,
    created_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    cog_assets: Optional[List[Dict[str, Any]]] = None,
    source_uris: Optional[List[str]] = None,
    lineage: Optional[str] = None,
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a STAC Item from a completed job.

    Args:
        job_id: The job UUID.
        event_type: Event type (flood, wildfire, storm).
        aoi_geojson: The job AOI as GeoJSON geometry.
        parameters: Job parameters dict.
        created_at: When the job was created.
        completed_at: When the job completed.
        cog_assets: List of COG output dicts with keys: name, href, title.
        source_uris: List of source imagery STAC Item URIs for derived_from.
        lineage: Processing lineage description.
        customer_id: Tenant identifier.

    Returns:
        A STAC Item dict ready for pgSTAC insertion.
    """
    now = datetime.now(timezone.utc)
    item_datetime = completed_at or now
    start_datetime = created_at or now
    end_datetime = completed_at or now

    # Build bbox from geometry
    bbox = _extract_bbox(aoi_geojson)

    # Build assets from COG outputs
    assets: Dict[str, Any] = {}
    if cog_assets:
        for i, cog in enumerate(cog_assets):
            asset_key = cog.get("name", f"result_{i}")
            assets[asset_key] = {
                "href": cog.get("href", f"s3://firstlight-results/{job_id}/{asset_key}.tif"),
                "type": COG_MEDIA_TYPE,
                "title": cog.get("title", asset_key),
                "roles": ["data"],
            }
    else:
        # Default asset placeholder
        assets["result"] = {
            "href": f"s3://firstlight-results/{job_id}/result.tif",
            "type": COG_MEDIA_TYPE,
            "title": "Analysis Result",
            "roles": ["data"],
        }

    # Build links
    links: List[Dict[str, Any]] = [
        {
            "rel": "self",
            "type": "application/geo+json",
            "href": f"/stac/collections/{event_type}/items/{job_id}",
        },
        {
            "rel": "parent",
            "type": "application/json",
            "href": f"/stac/collections/{event_type}",
        },
        {
            "rel": "collection",
            "type": "application/json",
            "href": f"/stac/collections/{event_type}",
        },
        {
            "rel": "root",
            "type": "application/json",
            "href": "/stac/",
        },
    ]

    # Add derived_from links for source imagery
    if source_uris:
        for uri in source_uris:
            links.append({
                "rel": "derived_from",
                "type": "application/geo+json",
                "href": uri,
                "title": "Source imagery",
            })

    # Build the STAC Item
    item: Dict[str, Any] = {
        "type": "Feature",
        "stac_version": "1.0.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
        ],
        "id": job_id,
        "collection": event_type,
        "geometry": aoi_geojson,
        "bbox": bbox,
        "properties": {
            "datetime": item_datetime.isoformat(),
            "start_datetime": start_datetime.isoformat(),
            "end_datetime": end_datetime.isoformat(),
            "created": now.isoformat(),
            "updated": now.isoformat(),
            # Processing extension fields
            "processing:lineage": lineage or (
                f"FirstLight {event_type} analysis pipeline. "
                f"Parameters: {json.dumps(parameters or {})}"
            ),
            "processing:software": FIRSTLIGHT_SOFTWARE,
            "processing:datetime": item_datetime.isoformat(),
            # FirstLight custom properties
            "firstlight:event_type": event_type,
            "firstlight:job_id": job_id,
        },
        "links": links,
        "assets": assets,
    }

    if customer_id:
        item["properties"]["firstlight:customer_id"] = customer_id

    return item


def _extract_bbox(
    geojson: Dict[str, Any],
) -> Optional[List[float]]:
    """
    Extract a bounding box from a GeoJSON geometry.

    Returns [west, south, east, north] or None.
    """
    try:
        coords = _flatten_coordinates(geojson)
        if not coords:
            return None

        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [min(lons), min(lats), max(lons), max(lats)]
    except Exception:
        return None


def _flatten_coordinates(
    geojson: Dict[str, Any],
) -> List[List[float]]:
    """Recursively flatten all coordinates from a GeoJSON geometry."""
    geom_type = geojson.get("type", "")
    coordinates = geojson.get("coordinates", [])

    if geom_type == "Point":
        return [coordinates[:2]]
    elif geom_type == "MultiPoint":
        return [c[:2] for c in coordinates]
    elif geom_type == "LineString":
        return [c[:2] for c in coordinates]
    elif geom_type == "MultiLineString":
        return [c[:2] for line in coordinates for c in line]
    elif geom_type == "Polygon":
        return [c[:2] for ring in coordinates for c in ring]
    elif geom_type == "MultiPolygon":
        return [c[:2] for poly in coordinates for ring in poly for c in ring]
    elif geom_type == "GeometryCollection":
        result = []
        for geom in geojson.get("geometries", []):
            result.extend(_flatten_coordinates(geom))
        return result
    return []


async def publish_result_as_stac_item(
    job_id: str,
    dsn: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Publish a completed job's results as a STAC Item.

    Reads the job record from PostGIS, builds a STAC Item, and upserts
    it into pgSTAC.

    This function is called from the orchestrator after a job transitions
    to the COMPLETE phase.

    Args:
        job_id: The job UUID.
        dsn: PostgreSQL connection string. If None, builds from settings.

    Returns:
        The published STAC Item dict, or None on failure.
    """
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg not available, cannot publish STAC item")
        return None

    if dsn is None:
        try:
            from api.config import get_settings
            settings = get_settings()
            db = settings.database
            dsn = (
                f"postgresql://{db.user}:{db.password}"
                f"@{db.host}:{db.port}/{db.name}"
            )
        except Exception as e:
            logger.warning("Cannot determine database DSN: %s", e)
            return None

    try:
        conn = await asyncpg.connect(dsn)
        try:
            # Read job data
            row = await conn.fetchrow(
                """
                SELECT
                    job_id,
                    customer_id,
                    event_type,
                    ST_AsGeoJSON(aoi)::text AS aoi_geojson,
                    phase,
                    status,
                    parameters,
                    created_at
                FROM jobs
                WHERE job_id = $1
                """,
                uuid.UUID(job_id) if isinstance(job_id, str) else job_id,
            )

            if row is None:
                logger.warning("Job %s not found, cannot publish STAC item", job_id)
                return None

            if row["phase"] != "COMPLETE":
                logger.warning(
                    "Job %s is in phase %s, not COMPLETE. Skipping STAC publish.",
                    job_id,
                    row["phase"],
                )
                return None

            # Parse AOI GeoJSON
            aoi_geojson = json.loads(row["aoi_geojson"])

            # Parse parameters
            parameters = {}
            if row["parameters"]:
                parameters = (
                    json.loads(row["parameters"])
                    if isinstance(row["parameters"], str)
                    else row["parameters"]
                )

            # Look for source imagery URIs in job_events
            source_uris = []
            source_rows = await conn.fetch(
                """
                SELECT payload
                FROM job_events
                WHERE job_id = $1
                  AND event_type = 'SOURCE_RECORDED'
                ORDER BY event_seq
                """,
                uuid.UUID(job_id) if isinstance(job_id, str) else job_id,
            )
            for source_row in source_rows:
                payload = source_row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                uri = payload.get("stac_item_uri") or payload.get("uri")
                if uri:
                    source_uris.append(uri)

            # Build the STAC Item
            item = build_stac_item(
                job_id=str(row["job_id"]),
                event_type=row["event_type"],
                aoi_geojson=aoi_geojson,
                parameters=parameters,
                created_at=row["created_at"],
                completed_at=datetime.now(timezone.utc),
                source_uris=source_uris if source_uris else None,
                customer_id=row["customer_id"],
            )

            # Upsert into pgSTAC
            await conn.execute(
                """
                INSERT INTO pgstac.items (id, collection, content, geometry, datetime)
                VALUES ($1, $2, $3::jsonb, ST_GeomFromGeoJSON($4), $5)
                ON CONFLICT (id, collection) DO UPDATE SET
                    content = EXCLUDED.content,
                    geometry = EXCLUDED.geometry,
                    datetime = EXCLUDED.datetime
                """,
                str(row["job_id"]),
                row["event_type"],
                json.dumps(item),
                row["aoi_geojson"],
                datetime.now(timezone.utc),
            )

            logger.info(
                "Published STAC item: collection=%s id=%s",
                row["event_type"],
                job_id,
            )
            return item

        finally:
            await conn.close()

    except Exception as e:
        logger.error("Failed to publish STAC item for job %s: %s", job_id, e)
        return None
