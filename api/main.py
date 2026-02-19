"""
FastAPI Application Entry Point.

Initializes the FastAPI application with all routers, middleware,
exception handlers, and startup/shutdown events.

FirstLight API - Geospatial Event Intelligence Platform.

This FastAPI application provides RESTful endpoints for:
- Event submission and retrieval
- Status monitoring and progress tracking
- Product download and metadata
- Data catalog browsing
- Health checks and readiness probes

See /api/docs for interactive Swagger documentation.
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import Settings, get_settings
from api.middleware import setup_middleware
from api.models.errors import ErrorCode, ErrorResponse, register_exception_handlers
from api.routes import api_router
from api.routes.control import control_router
from api.routes.internal import internal_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# OpenAPI metadata
TAGS_METADATA = [
    {
        "name": "Events",
        "description": (
            "Submit and manage event specifications. Events define the area, "
            "time window, and type of hazard to analyze."
        ),
    },
    {
        "name": "Status",
        "description": "Monitor event processing status and progress through pipeline phases.",
    },
    {
        "name": "Products",
        "description": "Access analysis outputs including raster data, vector files, and reports.",
    },
    {
        "name": "Catalog",
        "description": "Browse available data sources, algorithms, and supported event types.",
    },
    {
        "name": "Health",
        "description": "Health checks, readiness, and liveness probes for monitoring.",
    },
    {
        "name": "LLM Control",
        "description": (
            "LLM Control Plane endpoints for job management, state transitions, "
            "escalations, and tool discovery. Tenant-scoped."
        ),
    },
    {
        "name": "Partner Integration",
        "description": (
            "Partner API endpoints for SSE event streaming, webhook subscriptions, "
            "pipeline health metrics, and queue summaries. Requires internal:read permission."
        ),
    },
]


async def _refresh_materialized_views_loop(settings) -> None:
    """
    Background task that refreshes materialized views every 30 seconds.

    Refreshes pipeline_health_metrics and queue_summary views using
    REFRESH MATERIALIZED VIEW CONCURRENTLY for non-blocking updates.
    """
    refresh_interval = 30  # seconds

    # Wait a bit for the database to be ready
    await asyncio.sleep(5)

    while True:
        try:
            import asyncpg

            db = settings.database
            dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"
            conn = await asyncpg.connect(dsn)
            try:
                await conn.execute(
                    "REFRESH MATERIALIZED VIEW CONCURRENTLY pipeline_health_metrics"
                )
                await conn.execute(
                    "REFRESH MATERIALIZED VIEW CONCURRENTLY queue_summary"
                )
                logger.debug("Materialized views refreshed")
            finally:
                await conn.close()
        except ImportError:
            logger.debug("asyncpg not available, skipping view refresh")
        except Exception as e:
            logger.warning(f"Materialized view refresh failed: {e}")

        await asyncio.sleep(refresh_interval)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler for startup and shutdown events.

    Startup:
    - Initialize database connections
    - Load registries (algorithms, providers)
    - Set up background tasks

    Shutdown:
    - Close database connections
    - Clean up resources
    - Cancel background tasks
    """
    settings = get_settings()

    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize registries
    try:
        from api.dependencies import (
            get_algorithm_registry,
            get_provider_registry,
            get_schema_validator,
        )

        # Warm up caches
        schema_validator = get_schema_validator()
        if schema_validator.validator:
            logger.info("OpenSpec schema validator initialized")

        algorithm_registry = get_algorithm_registry()
        if algorithm_registry.registry:
            logger.info(
                f"Algorithm registry initialized with "
                f"{len(algorithm_registry.list_all())} algorithms"
            )

        provider_registry = get_provider_registry()
        if provider_registry.registry:
            logger.info(
                f"Provider registry initialized with "
                f"{len(provider_registry.list_all())} providers"
            )

    except Exception as e:
        logger.warning(f"Registry initialization warning: {e}")

    # Create storage directory if using local storage
    if settings.storage.backend == "local":
        settings.storage.local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local storage path: {settings.storage.local_path}")

    # Initialize Taskiq broker (if control-plane extras are installed)
    _taskiq_broker = None
    try:
        from workers.taskiq_app import broker as taskiq_broker

        if not taskiq_broker.is_worker_process:
            await taskiq_broker.startup()
            _taskiq_broker = taskiq_broker
            logger.info("Taskiq broker started (API-side task kickoff enabled)")
    except ImportError:
        logger.info("Taskiq not available — background tasks disabled")
    except Exception as e:
        logger.warning(f"Taskiq startup warning: {e}")

    # Start materialized view refresh background task
    _refresh_task = None
    try:
        _refresh_task = asyncio.create_task(
            _refresh_materialized_views_loop(settings)
        )
        logger.info("Materialized view refresh task started (30s interval)")
    except Exception as e:
        logger.warning(f"Materialized view refresh startup warning: {e}")

    logger.info("Application startup complete")

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down application...")

    # Cancel materialized view refresh task
    if _refresh_task is not None:
        _refresh_task.cancel()
        try:
            await _refresh_task
        except asyncio.CancelledError:
            pass
        logger.info("Materialized view refresh task stopped")

    # Shutdown Taskiq broker
    if _taskiq_broker is not None:
        try:
            await _taskiq_broker.shutdown()
            logger.info("Taskiq broker shut down")
        except Exception as e:
            logger.warning(f"Taskiq shutdown warning: {e}")

    # Close database connections
    # (Would close actual connections here)

    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()

    # Create FastAPI app with OpenAPI configuration
    openapi_config = settings.get_openapi_config()

    app = FastAPI(
        title=openapi_config["title"],
        version=openapi_config["version"],
        description=openapi_config["description"],
        contact=openapi_config.get("contact"),
        license_info=openapi_config.get("license_info"),
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        openapi_url="/api/openapi.json" if settings.debug else None,
    )

    # Configure logging level
    logging.getLogger().setLevel(settings.log_level.value)

    # Add CORS middleware
    if settings.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors.allow_origins,
            allow_credentials=settings.cors.allow_credentials,
            allow_methods=settings.cors.allow_methods,
            allow_headers=settings.cors.allow_headers,
            max_age=settings.cors.max_age,
        )
        logger.info("CORS middleware configured")

    # Set up custom middleware
    setup_middleware(app, settings)

    # Register exception handlers
    register_exception_handlers(app)

    # Add custom 404 handler
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle 404 Not Found errors."""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=ErrorResponse(
                code=ErrorCode.NOT_FOUND,
                message=f"Path not found: {request.url.path}",
            ).model_dump(exclude_none=True),
        )

    # Add custom validation error handler
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        from api.models.errors import ErrorDetail

        details = []
        for error in exc.errors():
            loc = ".".join(str(x) for x in error.get("loc", []))
            details.append(
                ErrorDetail(
                    field=loc if loc else None,
                    message=error.get("msg", "Invalid value"),
                    value=error.get("input"),
                )
            )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ErrorResponse(
                code=ErrorCode.VALIDATION_ERROR,
                message="Request validation failed",
                details=[d.model_dump(exclude_none=True) for d in details],
            ).model_dump(exclude_none=True),
        )

    # Include API routes
    app.include_router(api_router)

    # Include LLM Control Plane routes (separate /control/v1 prefix)
    app.include_router(control_router)

    # Include Partner Integration routes (/internal/v1 prefix)
    app.include_router(internal_router)

    # Mount pygeoapi OGC API Processes at /oapi (if installed)
    try:
        from core.ogc.config import create_pygeoapi_app

        pygeoapi_app = create_pygeoapi_app()
        if pygeoapi_app is not None:
            app.mount("/oapi", pygeoapi_app)
            logger.info("pygeoapi mounted at /oapi")
        else:
            logger.info("pygeoapi not available, /oapi endpoint disabled")
    except Exception as e:
        logger.warning("Failed to mount pygeoapi at /oapi: %s", e)

    # Mount stac-fastapi-pgstac at /stac (if installed)
    try:
        from core.stac.mount import create_stac_app

        stac_app = create_stac_app(settings)
        if stac_app is not None:
            app.mount("/stac", stac_app)
            logger.info("stac-fastapi-pgstac mounted at /stac")
        else:
            logger.info("stac-fastapi not available, /stac endpoint disabled")
    except Exception as e:
        logger.warning("Failed to mount stac-fastapi at /stac: %s", e)

    # Add root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Root endpoint with API information."""
        return JSONResponse(
            content={
                "name": settings.app_name,
                "version": settings.app_version,
                "environment": settings.environment.value,
                "docs_url": "/api/docs" if settings.debug else None,
                "health_url": "/api/v1/health",
                "description": (
                    "Geospatial Event Intelligence Platform. "
                    "Transforms (area, time window, event type) into decision products."
                ),
            }
        )

    logger.info(f"Application configured with {len(app.routes)} routes")

    return app


# Create the application instance
app = create_application()


# CLI entry point for development
if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.value.lower(),
    )
