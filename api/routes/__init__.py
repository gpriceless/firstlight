"""
API Routes Package.

Aggregates all API routers for inclusion in the main FastAPI application.
"""

from fastapi import APIRouter

from api.routes.catalog import router as catalog_router
from api.routes.events import router as events_router
from api.routes.health import router as health_router
from api.routes.products import router as products_router
from api.routes.status import router as status_router

# Create main API router with version prefix
api_router = APIRouter(prefix="/api/v1")

# Include all route modules
api_router.include_router(health_router)
api_router.include_router(catalog_router)
api_router.include_router(events_router)
api_router.include_router(status_router)
api_router.include_router(products_router)

# Export all routers for direct access if needed
__all__ = [
    "api_router",
    "catalog_router",
    "events_router",
    "health_router",
    "products_router",
    "status_router",
]
