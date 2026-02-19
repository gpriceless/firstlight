"""
LLM Control Plane API Routes.

Provides endpoints for LLM agents to read job state, trigger transitions,
adjust parameters, submit reasoning, manage escalations, and discover tools.

All endpoints are under /control/v1 and are tenant-scoped via customer_id
from TenantMiddleware.
"""

from fastapi import APIRouter, Depends, Request

from api.routes.control.jobs import router as jobs_router
from api.routes.control.escalations import router as escalations_router
from api.routes.control.tools import router as tools_router


def get_current_customer(request: Request) -> str:
    """
    Dependency that extracts the current customer_id from request state.

    The customer_id is set by TenantMiddleware from the resolved API key.
    Returns the customer_id string for use in tenant-scoped queries.

    Raises:
        HTTPException 401 if customer_id is not available.
    """
    from fastapi import HTTPException, status

    customer_id = getattr(request.state, "customer_id", None)
    if not customer_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: no tenant context available",
        )
    return customer_id


# Create the control router with /control/v1 prefix
# Include rate limiting dependency for all control plane endpoints
from api.rate_limit import check_control_plane_rate_limit

control_router = APIRouter(
    prefix="/control/v1",
    tags=["LLM Control"],
    dependencies=[Depends(check_control_plane_rate_limit)],
)

# Include sub-routers
control_router.include_router(jobs_router)
control_router.include_router(escalations_router)
control_router.include_router(tools_router)

__all__ = [
    "control_router",
    "get_current_customer",
]
