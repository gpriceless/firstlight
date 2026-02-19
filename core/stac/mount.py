"""
STAC FastAPI application factory.

Creates and configures a stac-fastapi-pgstac application for mounting
at /stac in the main FastAPI app. Also handles STAC collection
registration on startup.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def create_stac_app(settings: Any = None) -> Optional[Any]:
    """
    Create a stac-fastapi-pgstac ASGI application.

    Attempts to import and configure stac-fastapi-pgstac. Returns None
    if the library is not installed or configuration fails.

    Args:
        settings: Application settings (api.config.Settings instance).

    Returns:
        A stac-fastapi ASGI application, or None.
    """
    try:
        from stac_fastapi.pgstac.app import create_app as create_pgstac_app
        from stac_fastapi.pgstac.config import Settings as PgstacSettings
    except ImportError:
        logger.info(
            "stac-fastapi-pgstac not installed. STAC catalog disabled. "
            "Install with: pip install firstlight[control-plane]"
        )
        return None

    try:
        # Build the pgSTAC database DSN
        if settings is not None:
            db = settings.database
            postgres_dsn = (
                f"postgresql://{db.user}:{db.password}"
                f"@{db.host}:{db.port}/{db.name}"
            )
        else:
            import os
            postgres_dsn = os.getenv(
                "DATABASE_URL",
                "postgresql://firstlight:password@localhost:5432/firstlight",
            )

        # Configure stac-fastapi-pgstac
        pgstac_settings = PgstacSettings(
            postgres_user=settings.database.user if settings else "firstlight",
            postgres_pass=settings.database.password if settings else "password",
            postgres_host_reader=settings.database.host if settings else "localhost",
            postgres_host_writer=settings.database.host if settings else "localhost",
            postgres_port=str(settings.database.port) if settings else "5432",
            postgres_dbname=settings.database.name if settings else "firstlight",
        )

        stac_app = create_pgstac_app(settings=pgstac_settings)
        logger.info("stac-fastapi-pgstac application created")
        return stac_app

    except Exception as e:
        logger.warning("Failed to create stac-fastapi app: %s", e)
        return None


async def register_stac_collections(settings: Any = None) -> int:
    """
    Register STAC collections in pgSTAC on startup.

    This is called during the application lifespan startup to ensure
    all FirstLight event type collections exist in pgSTAC.

    Args:
        settings: Application settings.

    Returns:
        Number of collections registered.
    """
    from core.stac.collections import register_collections_in_pgstac

    try:
        if settings is not None:
            db = settings.database
            dsn = (
                f"postgresql://{db.user}:{db.password}"
                f"@{db.host}:{db.port}/{db.name}"
            )
        else:
            import os
            dsn = os.getenv(
                "DATABASE_URL",
                "postgresql://firstlight:password@localhost:5432/firstlight",
            )

        count = await register_collections_in_pgstac(dsn)
        logger.info("Registered %d STAC collections", count)
        return count

    except Exception as e:
        logger.warning("Failed to register STAC collections: %s", e)
        return 0
