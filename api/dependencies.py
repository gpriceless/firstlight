"""
FastAPI Dependency Injection Setup.

Provides reusable dependencies for database sessions, agent registry access,
authentication, and other common functionality across API routes.
"""

import logging
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated, Any, AsyncGenerator, Dict, Generator, Optional

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from api.config import Settings, get_settings
from api.models.errors import AuthenticationError, AuthorizationError, RateLimitError

logger = logging.getLogger(__name__)


# =============================================================================
# Settings Dependency
# =============================================================================


def get_app_settings() -> Settings:
    """
    Get application settings dependency.

    Returns:
        Cached Settings instance.
    """
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


# =============================================================================
# Database Session Dependencies
# =============================================================================


class DatabaseSession:
    """
    Async database session wrapper.

    This is a placeholder that will be replaced with actual database
    session management when the database layer is implemented.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._connection: Optional[Any] = None

    async def connect(self) -> None:
        """Establish database connection."""
        # Placeholder - will connect to actual database
        logger.debug("Database session connected")

    async def disconnect(self) -> None:
        """Close database connection."""
        # Placeholder - will close actual connection
        logger.debug("Database session disconnected")

    async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute a database query."""
        # Placeholder for query execution
        raise NotImplementedError("Database layer not yet implemented")


async def get_db_session(
    settings: SettingsDep,
) -> AsyncGenerator[DatabaseSession, None]:
    """
    Get database session dependency.

    Yields:
        DatabaseSession instance that is automatically closed.
    """
    session = DatabaseSession(settings)
    try:
        await session.connect()
        yield session
    finally:
        await session.disconnect()


DBSessionDep = Annotated[DatabaseSession, Depends(get_db_session)]


# =============================================================================
# Agent Registry Dependencies
# =============================================================================


class AgentRegistry:
    """
    Agent registry for accessing orchestrator and specialized agents.

    This is a placeholder that will be replaced with actual agent
    implementations when the agent layer is built.
    """

    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}

    def get_orchestrator(self) -> Any:
        """Get the main orchestrator agent."""
        return self._agents.get("orchestrator")

    def get_discovery_agent(self) -> Any:
        """Get the data discovery agent."""
        return self._agents.get("discovery")

    def get_pipeline_agent(self) -> Any:
        """Get the pipeline assembly agent."""
        return self._agents.get("pipeline")

    def get_quality_agent(self) -> Any:
        """Get the quality control agent."""
        return self._agents.get("quality")

    def get_reporting_agent(self) -> Any:
        """Get the product reporting agent."""
        return self._agents.get("reporting")


@lru_cache()
def get_agent_registry() -> AgentRegistry:
    """
    Get cached agent registry instance.

    Returns:
        AgentRegistry singleton instance.
    """
    return AgentRegistry()


AgentRegistryDep = Annotated[AgentRegistry, Depends(get_agent_registry)]


# =============================================================================
# Schema Validator Dependencies
# =============================================================================


class SchemaValidatorWrapper:
    """
    Wrapper for OpenSpec schema validator.

    Provides lazy loading of the actual validator to avoid
    circular imports during startup.
    """

    def __init__(self) -> None:
        self._validator: Optional[Any] = None

    @property
    def validator(self) -> Any:
        """Get or create the schema validator instance."""
        if self._validator is None:
            try:
                from openspec.validator import get_validator

                self._validator = get_validator()
            except ImportError:
                logger.warning("OpenSpec validator not available")
                self._validator = None
        return self._validator

    def validate_event(self, data: Dict) -> tuple[bool, list[str]]:
        """Validate event specification."""
        if self.validator is None:
            return True, []  # Skip validation if validator unavailable
        return self.validator.validate_event(data)

    def validate_intent(self, data: Dict) -> tuple[bool, list[str]]:
        """Validate intent specification."""
        if self.validator is None:
            return True, []
        return self.validator.validate_intent(data)


@lru_cache()
def get_schema_validator() -> SchemaValidatorWrapper:
    """Get cached schema validator instance."""
    return SchemaValidatorWrapper()


SchemaValidatorDep = Annotated[SchemaValidatorWrapper, Depends(get_schema_validator)]


# =============================================================================
# Algorithm Registry Dependencies
# =============================================================================


class AlgorithmRegistryWrapper:
    """
    Wrapper for algorithm registry.

    Provides lazy loading to avoid circular imports.
    """

    def __init__(self) -> None:
        self._registry: Optional[Any] = None

    @property
    def registry(self) -> Any:
        """Get or create the algorithm registry instance."""
        if self._registry is None:
            try:
                from core.analysis.library.registry import get_global_registry

                self._registry = get_global_registry()
            except ImportError:
                logger.warning("Algorithm registry not available")
                self._registry = None
        return self._registry

    def list_all(self) -> list:
        """List all algorithms."""
        if self.registry is None:
            return []
        return self.registry.list_all()

    def search_by_event_type(self, event_type: str) -> list:
        """Search algorithms by event type."""
        if self.registry is None:
            return []
        return self.registry.search_by_event_type(event_type)


@lru_cache()
def get_algorithm_registry() -> AlgorithmRegistryWrapper:
    """Get cached algorithm registry instance."""
    return AlgorithmRegistryWrapper()


AlgorithmRegistryDep = Annotated[
    AlgorithmRegistryWrapper, Depends(get_algorithm_registry)
]


# =============================================================================
# Provider Registry Dependencies
# =============================================================================


class ProviderRegistryWrapper:
    """
    Wrapper for data provider registry.

    Provides lazy loading to avoid circular imports.
    """

    def __init__(self) -> None:
        self._registry: Optional[Any] = None

    @property
    def registry(self) -> Any:
        """Get or create the provider registry instance."""
        if self._registry is None:
            try:
                from core.data.providers.loader import create_default_registry

                self._registry = create_default_registry()
            except ImportError:
                logger.warning("Provider registry not available")
                self._registry = None
        return self._registry

    def list_all(self) -> list:
        """List all providers."""
        if self.registry is None:
            return []
        return self.registry.list_all()


@lru_cache()
def get_provider_registry() -> ProviderRegistryWrapper:
    """Get cached provider registry instance."""
    return ProviderRegistryWrapper()


ProviderRegistryDep = Annotated[ProviderRegistryWrapper, Depends(get_provider_registry)]


# =============================================================================
# Authentication Dependencies
# =============================================================================

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Bearer token authentication
bearer_scheme = HTTPBearer(auto_error=False)


async def get_api_key(
    api_key: Annotated[Optional[str], Depends(api_key_header)],
    settings: SettingsDep,
) -> Optional[str]:
    """
    Validate API key if authentication is enabled.

    Args:
        api_key: API key from header
        settings: Application settings

    Returns:
        Validated API key or None if auth disabled

    Raises:
        AuthenticationError: If authentication is required but key is invalid
    """
    if not settings.auth.enabled:
        return None

    if not api_key:
        raise AuthenticationError(message="API key is required")

    if api_key not in settings.auth.allowed_api_keys:
        raise AuthenticationError(message="Invalid API key")

    return api_key


async def get_bearer_token(
    credentials: Annotated[
        Optional[HTTPAuthorizationCredentials], Depends(bearer_scheme)
    ],
    settings: SettingsDep,
) -> Optional[str]:
    """
    Validate bearer token if authentication is enabled.

    Args:
        credentials: Bearer credentials
        settings: Application settings

    Returns:
        Validated token or None if auth disabled

    Raises:
        AuthenticationError: If authentication is required but token is invalid
    """
    if not settings.auth.enabled:
        return None

    if not credentials:
        raise AuthenticationError(message="Bearer token is required")

    # TODO: Implement actual JWT validation
    # For now, just return the token
    return credentials.credentials


async def require_auth(
    api_key: Annotated[Optional[str], Depends(get_api_key)],
    bearer_token: Annotated[Optional[str], Depends(get_bearer_token)],
    settings: SettingsDep,
) -> Optional[str]:
    """
    Require authentication (API key or bearer token).

    Returns:
        Authenticated credential or None if auth disabled
    """
    if not settings.auth.enabled:
        return None

    if api_key:
        return api_key
    if bearer_token:
        return bearer_token

    raise AuthenticationError(message="Authentication required")


AuthDep = Annotated[Optional[str], Depends(require_auth)]


# =============================================================================
# Rate Limiting Dependencies
# =============================================================================


class RateLimiter:
    """
    Simple in-memory rate limiter.

    For production, this should be replaced with a Redis-backed implementation.
    """

    def __init__(self) -> None:
        self._requests: Dict[str, list] = {}

    def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """
        Check if request is within rate limit.

        Args:
            key: Rate limit key (e.g., IP address or API key)
            max_requests: Maximum requests per window
            window_seconds: Window duration in seconds

        Returns:
            True if within limit, False otherwise
        """
        import time

        now = time.time()
        window_start = now - window_seconds

        if key not in self._requests:
            self._requests[key] = []

        # Remove old requests outside the window
        self._requests[key] = [
            ts for ts in self._requests[key] if ts > window_start
        ]

        if len(self._requests[key]) >= max_requests:
            return False

        self._requests[key].append(now)
        return True


@lru_cache()
def get_rate_limiter() -> RateLimiter:
    """Get cached rate limiter instance."""
    return RateLimiter()


async def check_rate_limit(
    request: Request,
    settings: SettingsDep,
    rate_limiter: Annotated[RateLimiter, Depends(get_rate_limiter)],
) -> None:
    """
    Check rate limit for the current request.

    Raises:
        RateLimitError: If rate limit exceeded
    """
    if not settings.rate_limit_enabled:
        return

    # Use client IP as rate limit key
    client_ip = request.client.host if request.client else "unknown"
    key = f"rate_limit:{client_ip}"

    if not rate_limiter.check_rate_limit(
        key=key,
        max_requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window,
    ):
        raise RateLimitError(retry_after=settings.rate_limit_window)


RateLimitDep = Annotated[None, Depends(check_rate_limit)]


# =============================================================================
# Request Context Dependencies
# =============================================================================


async def get_correlation_id(
    request: Request,
    x_correlation_id: Annotated[Optional[str], Header()] = None,
) -> str:
    """
    Get or generate correlation ID for request tracing.

    Args:
        request: FastAPI request
        x_correlation_id: Correlation ID from header

    Returns:
        Correlation ID for the request
    """
    if x_correlation_id:
        return x_correlation_id

    # Check if already set in request state
    if hasattr(request.state, "correlation_id"):
        return request.state.correlation_id

    # Generate new correlation ID
    import uuid

    return f"req_{uuid.uuid4().hex[:12]}"


CorrelationIdDep = Annotated[str, Depends(get_correlation_id)]


# =============================================================================
# Common Dependency Bundles
# =============================================================================


class CommonDependencies:
    """Bundle of common dependencies for route handlers."""

    def __init__(
        self,
        settings: Settings,
        db_session: DatabaseSession,
        correlation_id: str,
    ) -> None:
        self.settings = settings
        self.db = db_session
        self.correlation_id = correlation_id


async def get_common_deps(
    settings: SettingsDep,
    db_session: DBSessionDep,
    correlation_id: CorrelationIdDep,
) -> CommonDependencies:
    """Get common dependencies bundle."""
    return CommonDependencies(
        settings=settings,
        db_session=db_session,
        correlation_id=correlation_id,
    )


CommonDep = Annotated[CommonDependencies, Depends(get_common_deps)]
