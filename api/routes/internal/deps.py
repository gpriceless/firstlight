"""
Dependencies for internal/partner API routes.

Provides authentication and authorization dependencies that check for
the internal:read permission required by all partner integration endpoints.
"""

from fastapi import Depends, Request

from api.auth import Permission, UserContext, authenticate, require_permissions


# Dependency that checks for internal:read permission
require_internal_scope = require_permissions(Permission.INTERNAL_READ)


def get_current_customer(request: Request) -> str:
    """
    Extract the current customer_id from request state.

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
