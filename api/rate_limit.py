"""
Rate Limiting Middleware for FastAPI.

Provides rate limiting with Redis-backed counters (with in-memory fallback),
per-endpoint and per-user limits, and rate limit headers in responses.
"""

import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    # Default limits
    default_requests_per_minute: int = Field(default=60, description="Default RPM limit")
    default_requests_per_hour: int = Field(default=1000, description="Default RPH limit")
    default_requests_per_day: int = Field(default=10000, description="Default RPD limit")

    # Burst allowance
    burst_multiplier: float = Field(default=1.5, description="Burst multiplier for short windows")

    # Backend settings
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    use_memory_fallback: bool = Field(default=True, description="Fall back to memory if Redis unavailable")

    # Header settings
    include_headers: bool = Field(default=True, description="Include rate limit headers in response")
    header_prefix: str = Field(default="X-RateLimit", description="Prefix for rate limit headers")

    # Behavior settings
    exempt_internal: bool = Field(default=True, description="Exempt internal/health endpoints")
    log_exceeded: bool = Field(default=True, description="Log rate limit exceeded events")

    class Config:
        env_prefix = "RATE_LIMIT_"


def get_rate_limit_config() -> RateLimitConfig:
    """Get rate limiting configuration from environment."""
    return RateLimitConfig(
        default_requests_per_minute=int(os.getenv("RATE_LIMIT_RPM", "60")),
        default_requests_per_hour=int(os.getenv("RATE_LIMIT_RPH", "1000")),
        default_requests_per_day=int(os.getenv("RATE_LIMIT_RPD", "10000")),
        burst_multiplier=float(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "1.5")),
        redis_url=os.getenv("REDIS_URL"),
        use_memory_fallback=os.getenv("RATE_LIMIT_MEMORY_FALLBACK", "true").lower() == "true",
        include_headers=os.getenv("RATE_LIMIT_HEADERS", "true").lower() == "true",
        header_prefix=os.getenv("RATE_LIMIT_HEADER_PREFIX", "X-RateLimit"),
        exempt_internal=os.getenv("RATE_LIMIT_EXEMPT_INTERNAL", "true").lower() == "true",
        log_exceeded=os.getenv("RATE_LIMIT_LOG_EXCEEDED", "true").lower() == "true",
    )


# =============================================================================
# Rate Limit Rules
# =============================================================================


class TimeWindow(Enum):
    """Time windows for rate limiting."""

    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400


@dataclass
class RateLimitRule:
    """A single rate limit rule."""

    requests: int
    window: TimeWindow
    scope: str = "global"  # "global", "user", "ip", "endpoint"
    burst_allowed: bool = True

    @property
    def window_seconds(self) -> int:
        """Get window duration in seconds."""
        return self.window.value

    def key_suffix(self) -> str:
        """Get suffix for cache key based on window."""
        return f":{self.window.name.lower()}"


@dataclass
class EndpointRateLimit:
    """Rate limit configuration for a specific endpoint."""

    path_pattern: str
    methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    rules: List[RateLimitRule] = field(default_factory=list)
    exempt: bool = False
    custom_key_func: Optional[Callable] = None

    def matches(self, path: str, method: str) -> bool:
        """Check if this config matches a request."""
        if self.exempt:
            return False

        # Simple pattern matching (supports * wildcard)
        if self.path_pattern == "*":
            return method.upper() in self.methods

        if self.path_pattern.endswith("*"):
            prefix = self.path_pattern[:-1]
            return path.startswith(prefix) and method.upper() in self.methods

        return path == self.path_pattern and method.upper() in self.methods


# Default rate limits for different endpoint patterns
DEFAULT_ENDPOINT_LIMITS: List[EndpointRateLimit] = [
    # Health/status endpoints - exempt
    EndpointRateLimit(path_pattern="/health", methods=["GET"], exempt=True),
    EndpointRateLimit(path_pattern="/health/*", methods=["GET"], exempt=True),
    EndpointRateLimit(path_pattern="/metrics", methods=["GET"], exempt=True),

    # Event creation - stricter limits
    EndpointRateLimit(
        path_pattern="/events",
        methods=["POST"],
        rules=[
            RateLimitRule(requests=10, window=TimeWindow.MINUTE, scope="user"),
            RateLimitRule(requests=100, window=TimeWindow.HOUR, scope="user"),
        ],
    ),

    # Product downloads - stricter limits
    EndpointRateLimit(
        path_pattern="/products/*/download",
        methods=["GET"],
        rules=[
            RateLimitRule(requests=30, window=TimeWindow.MINUTE, scope="user"),
            RateLimitRule(requests=500, window=TimeWindow.HOUR, scope="user"),
        ],
    ),

    # Admin endpoints - relaxed limits for admins
    EndpointRateLimit(
        path_pattern="/admin/*",
        methods=["GET", "POST", "PUT", "DELETE"],
        rules=[
            RateLimitRule(requests=120, window=TimeWindow.MINUTE, scope="user"),
        ],
    ),

    # Default for all other endpoints
    EndpointRateLimit(
        path_pattern="*",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        rules=[
            RateLimitRule(requests=60, window=TimeWindow.MINUTE, scope="user"),
            RateLimitRule(requests=1000, window=TimeWindow.HOUR, scope="user"),
        ],
    ),
]


# =============================================================================
# Rate Limit Backends
# =============================================================================


class RateLimitBackend:
    """Base class for rate limit storage backends."""

    async def increment(self, key: str, window_seconds: int) -> Tuple[int, int]:
        """
        Increment counter and return (current_count, ttl_seconds).

        Args:
            key: The rate limit key
            window_seconds: The window duration

        Returns:
            Tuple of (current request count, seconds until window reset)
        """
        raise NotImplementedError

    async def get_count(self, key: str) -> int:
        """Get current count for a key."""
        raise NotImplementedError

    async def reset(self, key: str) -> None:
        """Reset a rate limit key."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close any connections."""
        pass


class MemoryRateLimitBackend(RateLimitBackend):
    """In-memory rate limit backend using sliding window."""

    def __init__(self):
        # key -> list of timestamps
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background cleanup task."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup(self) -> None:
        """Remove expired timestamps."""
        now = time.time()
        max_window = TimeWindow.DAY.value

        async with self._lock:
            keys_to_delete = []
            for key, timestamps in self._requests.items():
                # Remove timestamps older than max window
                self._requests[key] = [t for t in timestamps if now - t < max_window]
                if not self._requests[key]:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del self._requests[key]

    async def increment(self, key: str, window_seconds: int) -> Tuple[int, int]:
        """Increment counter using sliding window."""
        now = time.time()
        window_start = now - window_seconds

        async with self._lock:
            # Remove old timestamps
            self._requests[key] = [t for t in self._requests[key] if t > window_start]

            # Add current timestamp
            self._requests[key].append(now)

            count = len(self._requests[key])

            # Calculate TTL (time until oldest request expires)
            if self._requests[key]:
                oldest = min(self._requests[key])
                ttl = int(window_seconds - (now - oldest))
            else:
                ttl = window_seconds

            return count, max(0, ttl)

    async def get_count(self, key: str) -> int:
        """Get current count for a key."""
        async with self._lock:
            return len(self._requests.get(key, []))

    async def reset(self, key: str) -> None:
        """Reset a rate limit key."""
        async with self._lock:
            if key in self._requests:
                del self._requests[key]

    async def close(self) -> None:
        """Stop background task."""
        await self.stop()


class RedisRateLimitBackend(RateLimitBackend):
    """Redis-backed rate limit backend using sorted sets."""

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None
        self._available = False

    async def connect(self) -> bool:
        """Attempt to connect to Redis."""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            self._available = True
            logger.info("Connected to Redis for rate limiting")
            return True
        except ImportError:
            logger.warning("redis-py not installed, using memory backend")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available

    async def increment(self, key: str, window_seconds: int) -> Tuple[int, int]:
        """Increment counter using Redis sorted set with sliding window."""
        if not self._available:
            raise RuntimeError("Redis not available")

        now = time.time()
        window_start = now - window_seconds

        # Use pipeline for atomicity
        pipe = self._redis.pipeline()

        # Remove expired entries
        pipe.zremrangebyscore(key, 0, window_start)

        # Add current timestamp
        pipe.zadd(key, {str(now): now})

        # Get count
        pipe.zcard(key)

        # Set expiry on key
        pipe.expire(key, window_seconds)

        results = await pipe.execute()
        count = results[2]

        # Calculate TTL
        ttl = await self._redis.ttl(key)

        return count, max(0, ttl)

    async def get_count(self, key: str) -> int:
        """Get current count for a key."""
        if not self._available:
            return 0

        now = time.time()
        # We need to know the window, but we'll just count all entries
        return await self._redis.zcard(key)

    async def reset(self, key: str) -> None:
        """Reset a rate limit key."""
        if self._available:
            await self._redis.delete(key)

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    limit: int
    remaining: int
    reset_seconds: int
    rule: Optional[RateLimitRule] = None
    key: Optional[str] = None

    @property
    def reset_timestamp(self) -> int:
        """Get reset time as Unix timestamp."""
        return int(time.time()) + self.reset_seconds


class RateLimiter:
    """
    Rate limiter with configurable backends and rules.

    Supports both Redis and in-memory backends, with automatic
    fallback to memory if Redis is unavailable.
    """

    def __init__(
        self,
        config: Optional[RateLimitConfig] = None,
        endpoint_limits: Optional[List[EndpointRateLimit]] = None,
    ):
        self.config = config or get_rate_limit_config()
        self.endpoint_limits = endpoint_limits or DEFAULT_ENDPOINT_LIMITS

        self._backend: Optional[RateLimitBackend] = None
        self._memory_backend: Optional[MemoryRateLimitBackend] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the rate limiter backend."""
        if self._initialized:
            return

        # Try Redis first
        if self.config.redis_url:
            redis_backend = RedisRateLimitBackend(self.config.redis_url)
            if await redis_backend.connect():
                self._backend = redis_backend
                self._initialized = True
                return

        # Fall back to memory
        if self.config.use_memory_fallback:
            self._memory_backend = MemoryRateLimitBackend()
            await self._memory_backend.start()
            self._backend = self._memory_backend
            logger.info("Using in-memory rate limiting backend")

        self._initialized = True

    async def close(self) -> None:
        """Close the rate limiter."""
        if self._backend:
            await self._backend.close()

    def _get_client_identifier(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Try to get user ID from state (set by auth middleware)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Try API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            # Hash the API key for privacy
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        return f"ip:{ip}"

    def _build_key(
        self,
        client_id: str,
        path: str,
        method: str,
        rule: RateLimitRule,
    ) -> str:
        """Build a rate limit key."""
        parts = ["ratelimit"]

        if rule.scope == "global":
            parts.append("global")
        elif rule.scope == "user":
            parts.append(client_id)
        elif rule.scope == "ip":
            parts.append(client_id if client_id.startswith("ip:") else "global")
        elif rule.scope == "endpoint":
            parts.append(f"{method}:{path}")

        parts.append(rule.window.name.lower())

        return ":".join(parts)

    def _find_endpoint_config(self, path: str, method: str) -> Optional[EndpointRateLimit]:
        """Find the matching endpoint configuration."""
        for endpoint_limit in self.endpoint_limits:
            if endpoint_limit.matches(path, method):
                return endpoint_limit
        return None

    async def check(self, request: Request) -> RateLimitResult:
        """
        Check if request should be rate limited.

        Args:
            request: The FastAPI request

        Returns:
            RateLimitResult indicating if request is allowed
        """
        if not self._initialized:
            await self.initialize()

        path = request.url.path
        method = request.method

        # Check if endpoint is exempt
        if self.config.exempt_internal:
            if path.startswith("/health") or path == "/metrics":
                return RateLimitResult(
                    allowed=True,
                    limit=0,
                    remaining=0,
                    reset_seconds=0,
                )

        # Find endpoint config
        endpoint_config = self._find_endpoint_config(path, method)
        if endpoint_config is None or endpoint_config.exempt:
            return RateLimitResult(
                allowed=True,
                limit=0,
                remaining=0,
                reset_seconds=0,
            )

        # Get client identifier
        client_id = self._get_client_identifier(request)

        # Check each rule
        rules = endpoint_config.rules or [
            RateLimitRule(
                requests=self.config.default_requests_per_minute,
                window=TimeWindow.MINUTE,
                scope="user",
            )
        ]

        # Track the most restrictive result
        most_restrictive: Optional[RateLimitResult] = None

        for rule in rules:
            key = self._build_key(client_id, path, method, rule)

            # Apply burst multiplier for short windows
            effective_limit = rule.requests
            if rule.burst_allowed and rule.window in (TimeWindow.SECOND, TimeWindow.MINUTE):
                effective_limit = int(rule.requests * self.config.burst_multiplier)

            # Get current count
            count, ttl = await self._backend.increment(key, rule.window_seconds)

            result = RateLimitResult(
                allowed=count <= effective_limit,
                limit=effective_limit,
                remaining=max(0, effective_limit - count),
                reset_seconds=ttl,
                rule=rule,
                key=key,
            )

            # Track most restrictive
            if most_restrictive is None or (
                not result.allowed
                or result.remaining < most_restrictive.remaining
            ):
                most_restrictive = result

            # If any rule is exceeded, we can stop
            if not result.allowed:
                if self.config.log_exceeded:
                    logger.warning(
                        f"Rate limit exceeded: {client_id} on {method} {path} "
                        f"({count}/{effective_limit} per {rule.window.name.lower()})"
                    )
                break

        return most_restrictive or RateLimitResult(
            allowed=True,
            limit=0,
            remaining=0,
            reset_seconds=0,
        )

    def add_headers(self, response: Response, result: RateLimitResult) -> None:
        """Add rate limit headers to response."""
        if not self.config.include_headers:
            return

        prefix = self.config.header_prefix

        response.headers[f"{prefix}-Limit"] = str(result.limit)
        response.headers[f"{prefix}-Remaining"] = str(result.remaining)
        response.headers[f"{prefix}-Reset"] = str(result.reset_timestamp)

        if not result.allowed:
            response.headers["Retry-After"] = str(result.reset_seconds)


# =============================================================================
# Middleware
# =============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request through rate limiter."""
        # Initialize if needed
        if not self.rate_limiter._initialized:
            await self.rate_limiter.initialize()

        # Check rate limit
        result = await self.rate_limiter.check(request)

        if not result.allowed:
            # Rate limit exceeded
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": result.reset_seconds,
                },
            )
            self.rate_limiter.add_headers(response, result)
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers to successful response
        self.rate_limiter.add_headers(response, result)

        return response


# =============================================================================
# Dependencies
# =============================================================================


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def check_rate_limit(request: Request) -> RateLimitResult:
    """
    Dependency for checking rate limits in routes.

    Usage:
        @app.get("/endpoint")
        async def endpoint(rate_limit: RateLimitResult = Depends(check_rate_limit)):
            ...
    """
    limiter = get_rate_limiter()
    if not limiter._initialized:
        await limiter.initialize()
    return await limiter.check(request)


def rate_limit(
    requests: int,
    window: TimeWindow = TimeWindow.MINUTE,
    scope: str = "user",
) -> Callable:
    """
    Decorator/dependency for applying custom rate limits to specific endpoints.

    Usage:
        @app.get("/endpoint")
        async def endpoint(
            _: None = Depends(rate_limit(10, TimeWindow.MINUTE, "user"))
        ):
            ...
    """

    async def rate_limit_check(request: Request) -> None:
        limiter = get_rate_limiter()
        if not limiter._initialized:
            await limiter.initialize()

        # Create custom rule
        rule = RateLimitRule(requests=requests, window=window, scope=scope)

        # Get client identifier
        client_id = limiter._get_client_identifier(request)

        # Build key
        key = limiter._build_key(client_id, request.url.path, request.method, rule)

        # Check limit
        count, ttl = await limiter._backend.increment(key, rule.window_seconds)

        if count > requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded: {requests} per {window.name.lower()}",
                    "retry_after": ttl,
                },
                headers={"Retry-After": str(ttl)},
            )

    return rate_limit_check


# =============================================================================
# Utilities
# =============================================================================


async def reset_rate_limit(client_id: str, window: Optional[TimeWindow] = None) -> None:
    """
    Reset rate limits for a specific client.

    Args:
        client_id: The client identifier (user ID, IP, or API key hash)
        window: Optional specific window to reset (resets all if None)
    """
    limiter = get_rate_limiter()
    if not limiter._initialized:
        await limiter.initialize()

    windows = [window] if window else list(TimeWindow)

    for w in windows:
        key = f"ratelimit:{client_id}:{w.name.lower()}"
        await limiter._backend.reset(key)


def configure_rate_limits(endpoint_limits: List[EndpointRateLimit]) -> None:
    """
    Configure custom endpoint rate limits.

    Should be called before the application starts.
    """
    limiter = get_rate_limiter()
    limiter.endpoint_limits = endpoint_limits
