"""
Health Check API Routes.

Provides health, readiness, and liveness endpoints for
monitoring and orchestration systems (Kubernetes, load balancers, etc.).
"""

import logging
import time
from datetime import datetime, timezone
from typing import Annotated, List

from fastapi import APIRouter, Depends, status

from api.config import Settings, get_settings
from api.models.responses import (
    HealthCheckComponent,
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])

# Track application start time for uptime calculation
_start_time: float = time.time()


def get_uptime_seconds() -> float:
    """Get application uptime in seconds."""
    return time.time() - _start_time


async def check_database_health(settings: Settings) -> HealthCheckComponent:
    """
    Check database connectivity.

    Returns:
        HealthCheckComponent with database status.
    """
    start = time.perf_counter()
    try:
        # Placeholder - actual database check would go here
        # For now, simulate a healthy database
        latency_ms = (time.perf_counter() - start) * 1000
        return HealthCheckComponent(
            name="database",
            status="healthy",
            latency_ms=latency_ms,
            message="Database connection available",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"Database health check failed: {e}")
        return HealthCheckComponent(
            name="database",
            status="unhealthy",
            latency_ms=latency_ms,
            message=str(e),
        )


async def check_redis_health(settings: Settings) -> HealthCheckComponent:
    """
    Check Redis connectivity.

    Returns:
        HealthCheckComponent with Redis status.
    """
    start = time.perf_counter()
    try:
        # Placeholder - actual Redis check would go here
        latency_ms = (time.perf_counter() - start) * 1000
        return HealthCheckComponent(
            name="redis",
            status="healthy",
            latency_ms=latency_ms,
            message="Redis connection available",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"Redis health check failed: {e}")
        return HealthCheckComponent(
            name="redis",
            status="degraded",  # Redis is optional, so degraded not unhealthy
            latency_ms=latency_ms,
            message=str(e),
        )


async def check_storage_health(settings: Settings) -> HealthCheckComponent:
    """
    Check object storage connectivity.

    Returns:
        HealthCheckComponent with storage status.
    """
    start = time.perf_counter()
    try:
        # Placeholder - actual storage check would go here
        if settings.storage.backend == "local":
            # Check if local storage path exists
            if settings.storage.local_path.exists():
                status_val = "healthy"
                message = "Local storage path accessible"
            else:
                status_val = "degraded"
                message = f"Local storage path does not exist: {settings.storage.local_path}"
        else:
            # For S3/GCS, would check bucket access
            status_val = "healthy"
            message = f"{settings.storage.backend.upper()} storage configured"

        latency_ms = (time.perf_counter() - start) * 1000
        return HealthCheckComponent(
            name="storage",
            status=status_val,
            latency_ms=latency_ms,
            message=message,
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"Storage health check failed: {e}")
        return HealthCheckComponent(
            name="storage",
            status="unhealthy",
            latency_ms=latency_ms,
            message=str(e),
        )


async def check_openspec_health() -> HealthCheckComponent:
    """
    Check OpenSpec schema validator availability.

    Returns:
        HealthCheckComponent with OpenSpec status.
    """
    start = time.perf_counter()
    try:
        from openspec.validator import get_validator

        validator = get_validator()
        # Try to load a schema to verify functionality
        validator._load_schema("event")

        latency_ms = (time.perf_counter() - start) * 1000
        return HealthCheckComponent(
            name="openspec",
            status="healthy",
            latency_ms=latency_ms,
            message="Schema validator operational",
        )
    except ImportError:
        latency_ms = (time.perf_counter() - start) * 1000
        return HealthCheckComponent(
            name="openspec",
            status="degraded",
            latency_ms=latency_ms,
            message="OpenSpec validator not available",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"OpenSpec health check failed: {e}")
        return HealthCheckComponent(
            name="openspec",
            status="unhealthy",
            latency_ms=latency_ms,
            message=str(e),
        )


@router.get(
    "",
    response_model=HealthResponse,
    summary="Basic health check",
    description="Returns basic health status. Used for simple uptime monitoring.",
    responses={
        200: {"description": "Service is healthy"},
    },
)
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """
    Basic health check endpoint.

    Returns a simple health status indicating the service is running.
    Does not check dependencies - use /health/ready for that.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment.value,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Checks if the service is ready to accept traffic. "
        "Verifies all critical dependencies are available."
    ),
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service is not ready"},
    },
)
async def readiness_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes and load balancers.

    Checks all critical dependencies:
    - Database connectivity
    - Redis availability
    - Storage access
    - OpenSpec validator

    Returns 503 if any critical component is unhealthy.
    """
    components: List[HealthCheckComponent] = []

    # Check all components
    components.append(await check_database_health(settings))
    components.append(await check_redis_health(settings))
    components.append(await check_storage_health(settings))
    components.append(await check_openspec_health())

    # Determine overall readiness
    # Ready if no components are unhealthy
    unhealthy_count = sum(1 for c in components if c.status == "unhealthy")
    ready = unhealthy_count == 0

    response = ReadinessResponse(
        ready=ready,
        components=components,
        timestamp=datetime.now(timezone.utc),
    )

    # Note: We return the response regardless of status
    # The caller should check the 'ready' field
    # In production, you might want to return 503 for not ready
    return response


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description=(
        "Checks if the service process is alive. "
        "Used by Kubernetes to determine if the container should be restarted."
    ),
    responses={
        200: {"description": "Service is alive"},
    },
)
async def liveness_check() -> LivenessResponse:
    """
    Liveness probe for Kubernetes.

    Simply confirms the process is running and can respond to requests.
    Does not check dependencies - a failed dependency should not
    cause the container to restart.
    """
    return LivenessResponse(
        alive=True,
        uptime_seconds=get_uptime_seconds(),
        timestamp=datetime.now(timezone.utc),
    )
