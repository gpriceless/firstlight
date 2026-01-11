"""
FastAPI Middleware Components.

Provides request logging, timing, error handling, and correlation ID
middleware for consistent request processing across the API.
"""

import logging
import time
import uuid
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from api.config import Settings
from api.models.errors import APIError, ErrorCode, ErrorResponse

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add correlation ID to each request for tracing.

    Generates a unique ID for each request or uses one from the
    X-Correlation-ID header if provided.
    """

    def __init__(self, app: ASGIApp, header_name: str = "X-Correlation-ID") -> None:
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and add correlation ID."""
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get(self.header_name)
        if not correlation_id:
            correlation_id = f"req_{uuid.uuid4().hex[:12]}"

        # Store in request state for access by route handlers
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log incoming requests and responses.

    Logs request method, path, status code, and timing information.
    """

    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        exclude_paths: Optional[list[str]] = None,
    ) -> None:
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.exclude_paths = exclude_paths or ["/health", "/health/live", "/health/ready"]

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and log details."""
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Extract request info
        method = request.method
        path = request.url.path
        query = str(request.query_params) if request.query_params else ""
        client_ip = request.client.host if request.client else "unknown"
        correlation_id = getattr(request.state, "correlation_id", "unknown")

        # Log request
        log_msg = f"Request: {method} {path}"
        if query:
            log_msg += f"?{query}"
        log_msg += f" | Client: {client_ip} | Correlation: {correlation_id}"
        logger.info(log_msg)

        # Optionally log request body
        if self.log_request_body and method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    logger.debug(f"Request body: {body.decode('utf-8')[:1000]}")
            except Exception:
                pass

        # Process request
        response = await call_next(request)

        # Log response
        logger.info(
            f"Response: {method} {path} | "
            f"Status: {response.status_code} | "
            f"Correlation: {correlation_id}"
        )

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to measure and report request processing time.

    Adds X-Response-Time header with milliseconds taken to process.
    """

    def __init__(
        self,
        app: ASGIApp,
        header_name: str = "X-Response-Time",
        log_slow_requests: bool = True,
        slow_request_threshold_ms: float = 1000.0,
    ) -> None:
        super().__init__(app)
        self.header_name = header_name
        self.log_slow_requests = log_slow_requests
        self.slow_request_threshold_ms = slow_request_threshold_ms

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and measure timing."""
        start_time = time.perf_counter()

        response = await call_next(request)

        # Calculate processing time
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Add timing header
        response.headers[self.header_name] = f"{process_time_ms:.2f}ms"

        # Log slow requests
        if self.log_slow_requests and process_time_ms > self.slow_request_threshold_ms:
            correlation_id = getattr(request.state, "correlation_id", "unknown")
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time_ms:.2f}ms | "
                f"Correlation: {correlation_id}"
            )

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle uncaught exceptions and return consistent error responses.

    Catches exceptions that escape route handlers and converts them
    to standardized ErrorResponse format.
    """

    def __init__(
        self,
        app: ASGIApp,
        debug: bool = False,
        include_stack_trace: bool = False,
    ) -> None:
        super().__init__(app)
        self.debug = debug
        self.include_stack_trace = include_stack_trace

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and handle exceptions."""
        try:
            return await call_next(request)
        except APIError as e:
            # Handle our custom API errors
            correlation_id = getattr(request.state, "correlation_id", None)
            response = e.to_response(request_id=correlation_id)

            logger.warning(
                f"API error: {e.code.value} - {e.message} | "
                f"Correlation: {correlation_id}"
            )

            return JSONResponse(
                status_code=e.status_code,
                content=response.model_dump(exclude_none=True),
                headers=e.headers,
            )
        except Exception as e:
            # Handle unexpected exceptions
            correlation_id = getattr(request.state, "correlation_id", None)

            logger.exception(
                f"Unhandled exception: {type(e).__name__}: {e} | "
                f"Correlation: {correlation_id}"
            )

            error_response = ErrorResponse(
                code=ErrorCode.INTERNAL_ERROR,
                message="An unexpected error occurred" if not self.debug else str(e),
                request_id=correlation_id,
            )

            return JSONResponse(
                status_code=500,
                content=error_response.model_dump(exclude_none=True),
            )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to responses.

    Adds headers for XSS protection, content type options, etc.
    """

    def __init__(
        self,
        app: ASGIApp,
        content_security_policy: Optional[str] = None,
    ) -> None:
        super().__init__(app)
        self.content_security_policy = content_security_policy

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process request and add security headers."""
        response = await call_next(request)

        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Clickjacking protection
        response.headers["X-Frame-Options"] = "DENY"

        # XSS protection (for older browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Referrer policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Content Security Policy (if configured)
        if self.content_security_policy:
            response.headers["Content-Security-Policy"] = self.content_security_policy

        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce maximum request size limits.

    Rejects requests that exceed the configured maximum size.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_size_bytes: int = 100 * 1024 * 1024,  # 100MB default
    ) -> None:
        super().__init__(app)
        self.max_size_bytes = max_size_bytes

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check request size and reject if too large."""
        content_length = request.headers.get("content-length")

        if content_length:
            try:
                size = int(content_length)
                if size > self.max_size_bytes:
                    correlation_id = getattr(request.state, "correlation_id", None)
                    response = ErrorResponse(
                        code=ErrorCode.VALIDATION_ERROR,
                        message=f"Request body too large. Maximum size is {self.max_size_bytes} bytes.",
                        request_id=correlation_id,
                    )
                    return JSONResponse(
                        status_code=413,
                        content=response.model_dump(exclude_none=True),
                    )
            except ValueError:
                pass

        return await call_next(request)


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """
    Configure all middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Order matters - middleware are executed in reverse order of addition
    # So add in order: last to execute first, first to execute last

    # Security headers (executes last, after response is generated)
    app.add_middleware(SecurityHeadersMiddleware)

    # Error handling (catches exceptions from all inner middleware and routes)
    app.add_middleware(
        ErrorHandlingMiddleware,
        debug=settings.debug,
        include_stack_trace=settings.debug,
    )

    # Request size limiting
    app.add_middleware(
        RequestSizeMiddleware,
        max_size_bytes=settings.max_request_size,
    )

    # Timing (measures total request time)
    app.add_middleware(
        TimingMiddleware,
        log_slow_requests=True,
        slow_request_threshold_ms=1000.0 if settings.is_production else 5000.0,
    )

    # Request logging
    app.add_middleware(
        RequestLoggingMiddleware,
        log_request_body=settings.debug,
        log_response_body=False,
    )

    # Correlation ID (executes first, sets up tracing context)
    app.add_middleware(CorrelationIdMiddleware)

    logger.info("Middleware configured successfully")
