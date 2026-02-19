"""
Partner Integration API Routes (Internal).

Provides endpoints for partner systems (e.g., MAIA Analytics) to:
- Stream structured events in real time via SSE
- Register and manage webhook subscriptions
- Inspect pipeline health metrics and queue summaries
- View open escalations

All endpoints are under /internal/v1 and require the internal:read permission.
"""

from fastapi import APIRouter, Depends

from api.routes.internal.events import router as events_router
from api.routes.internal.webhooks import router as webhooks_router
from api.routes.internal.metrics import router as metrics_router
from api.routes.internal.escalations import router as escalations_router


internal_router = APIRouter(
    prefix="/internal/v1",
    tags=["Partner Integration"],
)

# Include sub-routers
internal_router.include_router(events_router)
internal_router.include_router(webhooks_router)
internal_router.include_router(metrics_router)
internal_router.include_router(escalations_router)

__all__ = [
    "internal_router",
]
