"""
CORS Configuration for FastAPI.

Provides Cross-Origin Resource Sharing (CORS) configuration with
environment-based allowed origins, credentials handling, and preflight caching.
"""

import logging
import os
from typing import List, Optional, Sequence, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class CORSConfig(BaseModel):
    """CORS configuration settings."""

    # Allowed origins
    allow_origins: List[str] = Field(
        default_factory=list,
        description="List of allowed origins (use ['*'] for all)",
    )

    # Origin patterns (regex-like patterns)
    allow_origin_regex: Optional[str] = Field(
        default=None,
        description="Regex pattern for matching allowed origins",
    )

    # Allowed methods
    allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        description="List of allowed HTTP methods",
    )

    # Allowed headers
    allow_headers: List[str] = Field(
        default=["*"],
        description="List of allowed headers",
    )

    # Exposed headers (headers that browsers can access)
    expose_headers: List[str] = Field(
        default=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        description="Headers exposed to the browser",
    )

    # Credentials
    allow_credentials: bool = Field(
        default=True,
        description="Allow credentials (cookies, auth headers)",
    )

    # Preflight cache
    max_age: int = Field(
        default=600,
        description="Max age for preflight response caching (seconds)",
    )

    class Config:
        env_prefix = "CORS_"


# =============================================================================
# Environment Presets
# =============================================================================


def get_development_origins() -> List[str]:
    """Get allowed origins for development environment."""
    return [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # Vue dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "http://localhost:8000",  # Same-origin API
        "http://127.0.0.1:8000",
    ]


def get_production_origins() -> List[str]:
    """
    Get allowed origins for production environment.

    Reads from CORS_ALLOWED_ORIGINS environment variable.
    Format: comma-separated list of origins.
    """
    origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if not origins_str:
        return []

    return [origin.strip() for origin in origins_str.split(",") if origin.strip()]


def get_cors_config() -> CORSConfig:
    """
    Get CORS configuration based on environment.

    Environment variables:
        - ENVIRONMENT: 'development' or 'production'
        - CORS_ALLOWED_ORIGINS: Comma-separated list of allowed origins
        - CORS_ALLOW_CREDENTIALS: 'true' or 'false'
        - CORS_MAX_AGE: Preflight cache duration in seconds
        - CORS_ALLOW_METHODS: Comma-separated list of allowed methods
        - CORS_ALLOW_HEADERS: Comma-separated list of allowed headers
        - CORS_EXPOSE_HEADERS: Comma-separated list of exposed headers
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()

    # Determine allowed origins based on environment
    if environment == "development":
        origins = get_development_origins()
        # Also add any custom origins from env
        custom_origins = get_production_origins()
        origins.extend(o for o in custom_origins if o not in origins)
    else:
        origins = get_production_origins()

    # Check for wildcard
    if os.getenv("CORS_ALLOW_ALL_ORIGINS", "").lower() == "true":
        logger.warning("CORS configured to allow all origins - use only in development!")
        origins = ["*"]

    # Parse methods
    methods_str = os.getenv("CORS_ALLOW_METHODS", "")
    if methods_str:
        methods = [m.strip().upper() for m in methods_str.split(",")]
    else:
        methods = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]

    # Parse headers
    headers_str = os.getenv("CORS_ALLOW_HEADERS", "")
    if headers_str:
        headers = [h.strip() for h in headers_str.split(",")]
    else:
        headers = ["*"]

    # Parse exposed headers
    expose_str = os.getenv("CORS_EXPOSE_HEADERS", "")
    if expose_str:
        expose = [h.strip() for h in expose_str.split(",")]
    else:
        expose = [
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]

    return CORSConfig(
        allow_origins=origins,
        allow_origin_regex=os.getenv("CORS_ALLOW_ORIGIN_REGEX"),
        allow_methods=methods,
        allow_headers=headers,
        expose_headers=expose,
        allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true",
        max_age=int(os.getenv("CORS_MAX_AGE", "600")),
    )


# =============================================================================
# Middleware Setup
# =============================================================================


def setup_cors(
    app: FastAPI,
    config: Optional[CORSConfig] = None,
) -> None:
    """
    Configure CORS middleware on a FastAPI application.

    Args:
        app: The FastAPI application instance
        config: Optional CORSConfig, defaults to environment-based config
    """
    if config is None:
        config = get_cors_config()

    # Validate configuration
    if config.allow_credentials and "*" in config.allow_origins:
        logger.warning(
            "CORS: Cannot use credentials with wildcard origins. "
            "Disabling credentials or use specific origins."
        )
        # In strict mode, we would disable credentials
        # For flexibility, we keep the warning and let FastAPI handle it

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allow_origins,
        allow_origin_regex=config.allow_origin_regex,
        allow_credentials=config.allow_credentials,
        allow_methods=config.allow_methods,
        allow_headers=config.allow_headers,
        expose_headers=config.expose_headers,
        max_age=config.max_age,
    )

    logger.info(
        f"CORS configured: origins={len(config.allow_origins)}, "
        f"credentials={config.allow_credentials}, "
        f"max_age={config.max_age}s"
    )


def create_cors_middleware_config(config: Optional[CORSConfig] = None) -> dict:
    """
    Create a configuration dict suitable for CORSMiddleware.

    Useful when you need to apply CORS middleware manually or in tests.

    Args:
        config: Optional CORSConfig

    Returns:
        Dict with CORSMiddleware constructor arguments
    """
    if config is None:
        config = get_cors_config()

    return {
        "allow_origins": config.allow_origins,
        "allow_origin_regex": config.allow_origin_regex,
        "allow_credentials": config.allow_credentials,
        "allow_methods": config.allow_methods,
        "allow_headers": config.allow_headers,
        "expose_headers": config.expose_headers,
        "max_age": config.max_age,
    }


# =============================================================================
# Utility Functions
# =============================================================================


def is_origin_allowed(origin: str, config: Optional[CORSConfig] = None) -> bool:
    """
    Check if an origin is allowed by the CORS configuration.

    Args:
        origin: The origin to check
        config: Optional CORSConfig

    Returns:
        True if origin is allowed
    """
    if config is None:
        config = get_cors_config()

    # Wildcard allows all
    if "*" in config.allow_origins:
        return True

    # Direct match
    if origin in config.allow_origins:
        return True

    # Regex match
    if config.allow_origin_regex:
        import re

        if re.match(config.allow_origin_regex, origin):
            return True

    return False


def validate_cors_config(config: CORSConfig) -> List[str]:
    """
    Validate CORS configuration and return list of warnings.

    Args:
        config: The CORSConfig to validate

    Returns:
        List of warning messages
    """
    warnings = []

    # Check for wildcard with credentials
    if config.allow_credentials and "*" in config.allow_origins:
        warnings.append(
            "Wildcard origins with credentials is insecure. "
            "Use specific origins in production."
        )

    # Check for empty origins
    if not config.allow_origins and not config.allow_origin_regex:
        warnings.append("No allowed origins configured. CORS requests will be rejected.")

    # Check for very long max_age
    if config.max_age > 86400:  # 24 hours
        warnings.append(
            f"Preflight cache duration ({config.max_age}s) is very long. "
            "Consider reducing for better security."
        )

    # Check for broad headers
    if "*" in config.allow_headers:
        warnings.append(
            "Allowing all headers may expose sensitive information. "
            "Consider specifying explicit headers."
        )

    return warnings
