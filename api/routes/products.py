"""
Products API Routes.

Provides endpoints for listing, retrieving, and downloading
products generated from event processing.
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Dict, List, Optional

from fastapi import APIRouter, Depends, Path, Query, status
from fastapi.responses import FileResponse, StreamingResponse

from api.dependencies import AuthDep, DBSessionDep, SettingsDep
from api.models.errors import EventNotFoundError, NotFoundError, ProductNotFoundError
from api.models.requests import BoundingBox, ProductDownloadParams, ProductQueryParams
from api.models.responses import (
    EventStatus,
    EventResponse,
    PaginatedResponse,
    PaginationMeta,
    ProductFormat,
    ProductMetadata,
    ProductQuality,
    ProductResponse,
    ProductStatus,
    QualityFlag,
)
from api.routes.events import deserialize_event_from_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/events", tags=["Products"])


# In-memory products store (would be replaced with database in production)
_products_store: Dict[str, Dict[str, ProductResponse]] = {}


def _generate_mock_products(event_id: str, event: EventResponse) -> List[ProductResponse]:
    """Generate mock products for completed events."""
    if event.status != EventStatus.COMPLETED:
        return []

    now = datetime.now(timezone.utc)
    products = []

    # Flood extent product
    products.append(
        ProductResponse(
            id=f"prod_{uuid.uuid4().hex[:8]}",
            event_id=event_id,
            product_type="flood_extent",
            name="Flood Extent Map",
            status=ProductStatus.READY,
            format=ProductFormat.COG,
            metadata=ProductMetadata(
                format=ProductFormat.COG,
                crs="EPSG:4326",
                resolution_m=10.0,
                bbox=event.spatial.bbox,
                file_size_bytes=15_234_567,
                checksum="sha256:abc123...",
                bands=["flood_mask", "water_probability"],
                statistics={
                    "flooded_area_km2": 125.6,
                    "max_depth_m": 2.3,
                    "water_pixels": 1562890,
                },
            ),
            quality=ProductQuality(
                overall_confidence=0.87,
                flags=[QualityFlag.HIGH_CONFIDENCE],
                validation_status="validated",
                uncertainty_metrics={
                    "commission_error": 0.05,
                    "omission_error": 0.08,
                },
            ),
            download_url=f"/api/v1/events/{event_id}/products/flood_extent/download",
            expires_at=now + timedelta(hours=24),
            created_at=event.completed_at or now,
            ready_at=event.completed_at or now,
        )
    )

    # Vector product
    products.append(
        ProductResponse(
            id=f"prod_{uuid.uuid4().hex[:8]}",
            event_id=event_id,
            product_type="flood_extent_vector",
            name="Flood Extent Polygons",
            status=ProductStatus.READY,
            format=ProductFormat.GEOJSON,
            metadata=ProductMetadata(
                format=ProductFormat.GEOJSON,
                crs="EPSG:4326",
                resolution_m=None,
                bbox=event.spatial.bbox,
                file_size_bytes=2_345_678,
                checksum="sha256:def456...",
                bands=None,
                statistics={
                    "polygon_count": 47,
                    "total_area_km2": 125.6,
                },
            ),
            quality=ProductQuality(
                overall_confidence=0.85,
                flags=[QualityFlag.HIGH_CONFIDENCE],
                validation_status="validated",
                uncertainty_metrics=None,
            ),
            download_url=f"/api/v1/events/{event_id}/products/flood_extent_vector/download",
            expires_at=now + timedelta(hours=24),
            created_at=event.completed_at or now,
            ready_at=event.completed_at or now,
        )
    )

    # Report product
    products.append(
        ProductResponse(
            id=f"prod_{uuid.uuid4().hex[:8]}",
            event_id=event_id,
            product_type="summary_report",
            name="Event Summary Report",
            status=ProductStatus.READY,
            format=ProductFormat.PDF,
            metadata=ProductMetadata(
                format=ProductFormat.PDF,
                crs="EPSG:4326",
                resolution_m=None,
                bbox=event.spatial.bbox,
                file_size_bytes=1_234_567,
                checksum="sha256:ghi789...",
                bands=None,
                statistics=None,
            ),
            quality=None,
            download_url=f"/api/v1/events/{event_id}/products/summary_report/download",
            expires_at=now + timedelta(hours=24),
            created_at=event.completed_at or now,
            ready_at=event.completed_at or now,
        )
    )

    return products


@router.get(
    "/{event_id}/products",
    response_model=PaginatedResponse[ProductResponse],
    summary="List event products",
    description="List all products generated for an event.",
    responses={
        200: {"description": "List of products"},
        404: {"description": "Event not found"},
    },
)
async def list_event_products(
    event_id: Annotated[str, Path(description="Event identifier")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
    # Filters
    product_type: Annotated[
        Optional[List[str]],
        Query(description="Filter by product type"),
    ] = None,
    format: Annotated[
        Optional[List[str]],
        Query(description="Filter by output format"),
    ] = None,
    ready_only: Annotated[
        bool,
        Query(description="Only return products that are ready"),
    ] = False,
    # Pagination
    limit: Annotated[
        int,
        Query(ge=1, le=100, description="Maximum number of results"),
    ] = 20,
    offset: Annotated[
        int,
        Query(ge=0, description="Number of results to skip"),
    ] = 0,
) -> PaginatedResponse[ProductResponse]:
    """
    List all products generated for an event.

    Products are generated during event processing and include:
    - Raster products (flood extent, burn severity, etc.)
    - Vector products (polygons, points of interest)
    - Reports (PDF summaries, data quality reports)

    Supports filtering by product type, format, and readiness status.
    """
    row = await db_session.get_event(event_id)
    if not row:
        raise EventNotFoundError(event_id)

    event = deserialize_event_from_db(row)

    # Get or generate products
    if event_id not in _products_store:
        products = _generate_mock_products(event_id, event)
        _products_store[event_id] = {p.product_type: p for p in products}

    products = list(_products_store.get(event_id, {}).values())

    # Apply filters
    if product_type:
        products = [p for p in products if p.product_type in product_type]

    if format:
        products = [p for p in products if p.format.value in format]

    if ready_only:
        products = [p for p in products if p.status == ProductStatus.READY]

    # Apply pagination
    total = len(products)
    products = products[offset : offset + limit]

    return PaginatedResponse(
        items=products,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_more=(offset + limit) < total,
        ),
    )


@router.get(
    "/{event_id}/products/{product_id}",
    response_model=ProductResponse,
    summary="Get product details",
    description="Get details for a specific product, including download URL.",
    responses={
        200: {"description": "Product details"},
        404: {"description": "Event or product not found"},
    },
)
async def get_product(
    event_id: Annotated[str, Path(description="Event identifier")],
    product_id: Annotated[str, Path(description="Product identifier or type")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> ProductResponse:
    """
    Get details for a specific product.

    The product_id can be either:
    - The unique product ID (prod_xxx)
    - The product type (e.g., 'flood_extent', 'summary_report')

    Returns full product details including:
    - Metadata (format, resolution, size)
    - Quality information (confidence, validation status)
    - Download URL (if ready)
    """
    row = await db_session.get_event(event_id)
    if not row:
        raise EventNotFoundError(event_id)

    event = deserialize_event_from_db(row)

    # Get or generate products
    if event_id not in _products_store:
        products = _generate_mock_products(event_id, event)
        _products_store[event_id] = {p.product_type: p for p in products}

    products = _products_store.get(event_id, {})

    # Search by product ID or type
    for product in products.values():
        if product.id == product_id or product.product_type == product_id:
            return product

    raise ProductNotFoundError(product_id, event_id)


@router.get(
    "/{event_id}/products/{product_id}/metadata",
    response_model=ProductMetadata,
    summary="Get product metadata",
    description="Get metadata for a specific product.",
    responses={
        200: {"description": "Product metadata"},
        404: {"description": "Event or product not found"},
    },
)
async def get_product_metadata(
    event_id: Annotated[str, Path(description="Event identifier")],
    product_id: Annotated[str, Path(description="Product identifier or type")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> ProductMetadata:
    """
    Get metadata for a specific product.

    Returns detailed metadata including:
    - Format and CRS
    - Resolution and bounding box
    - File size and checksum
    - Band information (for raster products)
    - Statistics
    """
    product = await get_product(event_id, product_id, settings, db_session, auth)

    if not product.metadata:
        raise NotFoundError(
            message=f"No metadata available for product '{product_id}'",
        )

    return product.metadata


@router.get(
    "/{event_id}/products/{product_id}/download",
    summary="Download product",
    description="Download a product file.",
    responses={
        200: {"description": "Product file"},
        404: {"description": "Event or product not found"},
        409: {"description": "Product not ready for download"},
    },
)
async def download_product(
    event_id: Annotated[str, Path(description="Event identifier")],
    product_id: Annotated[str, Path(description="Product identifier or type")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
    # Download options
    format: Annotated[
        Optional[str],
        Query(description="Convert to this format (if supported)"),
    ] = None,
    crs: Annotated[
        Optional[str],
        Query(pattern=r"^EPSG:[0-9]+$", description="Reproject to this CRS"),
    ] = None,
    resolution: Annotated[
        Optional[float],
        Query(gt=0, description="Resample to this resolution (meters)"),
    ] = None,
    clip_to_aoi: Annotated[
        bool,
        Query(description="Clip to original area of interest"),
    ] = True,
) -> StreamingResponse:
    """
    Download a product file.

    Supports optional transformations:
    - Format conversion (GeoTIFF to COG, etc.)
    - Reprojection to different CRS
    - Resampling to different resolution
    - Clipping to area of interest

    Note: This is a placeholder that returns mock data.
    In production, this would stream the actual product file
    from object storage.
    """
    product = await get_product(event_id, product_id, settings, db_session, auth)

    if product.status != ProductStatus.READY:
        from api.models.errors import ConflictError
        raise ConflictError(
            message=f"Product is not ready for download. Current status: {product.status.value}",
        )

    # Determine content type based on format
    content_types = {
        ProductFormat.COG: "image/tiff",
        ProductFormat.GEOTIFF: "image/tiff",
        ProductFormat.GEOJSON: "application/geo+json",
        ProductFormat.GEOPARQUET: "application/vnd.apache.parquet",
        ProductFormat.NETCDF: "application/x-netcdf",
        ProductFormat.ZARR: "application/zip",
        ProductFormat.PDF: "application/pdf",
        ProductFormat.PNG: "image/png",
    }
    content_type = content_types.get(product.format, "application/octet-stream")

    # Determine filename
    extensions = {
        ProductFormat.COG: ".tif",
        ProductFormat.GEOTIFF: ".tif",
        ProductFormat.GEOJSON: ".geojson",
        ProductFormat.GEOPARQUET: ".parquet",
        ProductFormat.NETCDF: ".nc",
        ProductFormat.ZARR: ".zarr.zip",
        ProductFormat.PDF: ".pdf",
        ProductFormat.PNG: ".png",
    }
    extension = extensions.get(product.format, "")
    filename = f"{event_id}_{product.product_type}{extension}"

    # In production, this would stream from object storage
    # For now, return a placeholder response
    async def generate_mock_content():
        """Generate mock file content for demonstration."""
        if product.format == ProductFormat.GEOJSON:
            content = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
                        },
                        "properties": {
                            "event_id": event_id,
                            "product_type": product.product_type,
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }
                ],
            }
            import json
            yield json.dumps(content, indent=2).encode()
        else:
            # Return placeholder bytes for binary formats
            yield f"# Placeholder for {product.product_type}\n".encode()
            yield f"# Event: {event_id}\n".encode()
            yield f"# Format: {product.format.value}\n".encode()
            yield f"# Generated: {datetime.now(timezone.utc).isoformat()}\n".encode()

    return StreamingResponse(
        generate_mock_content(),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Product-ID": product.id,
            "X-Product-Type": product.product_type,
        },
    )


@router.get(
    "/{event_id}/products/{product_id}/quality",
    response_model=ProductQuality,
    summary="Get product quality information",
    description="Get quality and validation information for a product.",
    responses={
        200: {"description": "Product quality information"},
        404: {"description": "Event, product, or quality info not found"},
    },
)
async def get_product_quality(
    event_id: Annotated[str, Path(description="Event identifier")],
    product_id: Annotated[str, Path(description="Product identifier or type")],
    settings: SettingsDep,
    db_session: DBSessionDep,
    auth: AuthDep,
) -> ProductQuality:
    """
    Get quality and validation information for a product.

    Returns:
    - Overall confidence score
    - Quality flags (degraded resolution, single sensor mode, etc.)
    - Validation status
    - Uncertainty metrics
    """
    product = await get_product(event_id, product_id, settings, db_session, auth)

    if not product.quality:
        raise NotFoundError(
            message=f"No quality information available for product '{product_id}'",
        )

    return product.quality
