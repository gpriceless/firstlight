"""
API Error Classes and Exception Handlers.

Provides a consistent error handling framework with typed exceptions,
error response models, and FastAPI exception handlers.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Standardized error codes for API responses."""

    # General errors (1xxx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    ALREADY_EXISTS = "ALREADY_EXISTS"
    RATE_LIMITED = "RATE_LIMITED"

    # Authentication/Authorization errors (2xxx)
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    INVALID_TOKEN = "INVALID_TOKEN"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"
    INVALID_API_KEY = "INVALID_API_KEY"

    # Event errors (3xxx)
    EVENT_NOT_FOUND = "EVENT_NOT_FOUND"
    EVENT_INVALID = "EVENT_INVALID"
    EVENT_ALREADY_EXISTS = "EVENT_ALREADY_EXISTS"
    EVENT_CANCELLED = "EVENT_CANCELLED"
    EVENT_PROCESSING_FAILED = "EVENT_PROCESSING_FAILED"

    # Data errors (4xxx)
    DATA_SOURCE_NOT_FOUND = "DATA_SOURCE_NOT_FOUND"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    DATA_ACQUISITION_FAILED = "DATA_ACQUISITION_FAILED"
    DATA_VALIDATION_FAILED = "DATA_VALIDATION_FAILED"

    # Pipeline errors (5xxx)
    PIPELINE_NOT_FOUND = "PIPELINE_NOT_FOUND"
    PIPELINE_INVALID = "PIPELINE_INVALID"
    PIPELINE_EXECUTION_FAILED = "PIPELINE_EXECUTION_FAILED"
    ALGORITHM_NOT_FOUND = "ALGORITHM_NOT_FOUND"

    # Product errors (6xxx)
    PRODUCT_NOT_FOUND = "PRODUCT_NOT_FOUND"
    PRODUCT_NOT_READY = "PRODUCT_NOT_READY"
    PRODUCT_EXPIRED = "PRODUCT_EXPIRED"
    PRODUCT_GENERATION_FAILED = "PRODUCT_GENERATION_FAILED"

    # Schema/Spec errors (7xxx)
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    INTENT_RESOLUTION_FAILED = "INTENT_RESOLUTION_FAILED"
    INVALID_EVENT_CLASS = "INVALID_EVENT_CLASS"


class ErrorDetail(BaseModel):
    """Detailed error information for a specific field or issue."""

    field: Optional[str] = Field(
        default=None, description="Field path where error occurred"
    )
    message: str = Field(..., description="Human-readable error message")
    value: Optional[Any] = Field(
        default=None, description="The invalid value (if applicable)"
    )


class ErrorResponse(BaseModel):
    """Standardized API error response model."""

    code: ErrorCode = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[List[ErrorDetail]] = Field(
        default=None, description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None, description="Request correlation ID for debugging"
    )
    documentation_url: Optional[str] = Field(
        default=None, description="URL to relevant documentation"
    )

    model_config = {"json_schema_extra": {"example": {
        "code": "EVENT_NOT_FOUND",
        "message": "Event with ID 'evt_123' was not found",
        "details": None,
        "request_id": "req_abc123",
        "documentation_url": "https://docs.firstlight.io/errors#EVENT_NOT_FOUND"
    }}}


class APIError(Exception):
    """Base exception class for all API errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize API error.

        Args:
            message: Human-readable error message
            code: Error code for programmatic handling
            status_code: HTTP status code
            details: Additional error details
            headers: Optional response headers
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        self.headers = headers

    def to_response(self, request_id: Optional[str] = None) -> ErrorResponse:
        """Convert exception to ErrorResponse model."""
        return ErrorResponse(
            code=self.code,
            message=self.message,
            details=self.details,
            request_id=request_id,
        )


class NotFoundError(APIError):
    """Resource not found error."""

    def __init__(
        self,
        message: str = "Resource not found",
        code: ErrorCode = ErrorCode.NOT_FOUND,
        details: Optional[List[ErrorDetail]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_404_NOT_FOUND,
            details=details,
        )


class EventNotFoundError(NotFoundError):
    """Event not found error."""

    def __init__(self, event_id: str) -> None:
        super().__init__(
            message=f"Event with ID '{event_id}' was not found",
            code=ErrorCode.EVENT_NOT_FOUND,
        )


class ProductNotFoundError(NotFoundError):
    """Product not found error."""

    def __init__(self, product_id: str, event_id: Optional[str] = None) -> None:
        msg = f"Product with ID '{product_id}' was not found"
        if event_id:
            msg = f"Product with ID '{product_id}' for event '{event_id}' was not found"
        super().__init__(
            message=msg,
            code=ErrorCode.PRODUCT_NOT_FOUND,
        )


class ValidationError(APIError):
    """Request validation error."""

    def __init__(
        self,
        message: str = "Validation error",
        details: Optional[List[ErrorDetail]] = None,
        code: ErrorCode = ErrorCode.VALIDATION_ERROR,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
        )

    @classmethod
    def from_pydantic_errors(
        cls, errors: List[Dict[str, Any]]
    ) -> "ValidationError":
        """Create ValidationError from Pydantic validation errors."""
        details = []
        for error in errors:
            loc = ".".join(str(x) for x in error.get("loc", []))
            details.append(
                ErrorDetail(
                    field=loc if loc else None,
                    message=error.get("msg", "Invalid value"),
                    value=error.get("input"),
                )
            )
        return cls(
            message="Request validation failed",
            details=details,
        )


class SchemaValidationError(ValidationError):
    """OpenSpec schema validation error."""

    def __init__(
        self,
        message: str = "Schema validation failed",
        schema_name: Optional[str] = None,
        errors: Optional[List[str]] = None,
    ) -> None:
        details = None
        if errors:
            details = [ErrorDetail(message=err) for err in errors]
        super().__init__(
            message=f"{message}" + (f" (schema: {schema_name})" if schema_name else ""),
            details=details,
            code=ErrorCode.SCHEMA_VALIDATION_ERROR,
        )


class ProcessingError(APIError):
    """Processing/execution error."""

    def __init__(
        self,
        message: str = "Processing failed",
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: Optional[List[ErrorDetail]] = None,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
        )


class EventProcessingError(ProcessingError):
    """Event processing failed error."""

    def __init__(
        self,
        event_id: str,
        reason: Optional[str] = None,
    ) -> None:
        msg = f"Processing failed for event '{event_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            message=msg,
            code=ErrorCode.EVENT_PROCESSING_FAILED,
        )


class PipelineExecutionError(ProcessingError):
    """Pipeline execution failed error."""

    def __init__(
        self,
        pipeline_id: str,
        step: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        msg = f"Pipeline '{pipeline_id}' execution failed"
        if step:
            msg += f" at step '{step}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            message=msg,
            code=ErrorCode.PIPELINE_EXECUTION_FAILED,
        )


class DataAcquisitionError(ProcessingError):
    """Data acquisition failed error."""

    def __init__(
        self,
        source_id: str,
        reason: Optional[str] = None,
    ) -> None:
        msg = f"Data acquisition failed for source '{source_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(
            message=msg,
            code=ErrorCode.DATA_ACQUISITION_FAILED,
        )


class AuthenticationError(APIError):
    """Authentication error."""

    def __init__(
        self,
        message: str = "Authentication required",
        code: ErrorCode = ErrorCode.UNAUTHORIZED,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(APIError):
    """Authorization/permission error."""

    def __init__(
        self,
        message: str = "Permission denied",
    ) -> None:
        super().__init__(
            message=message,
            code=ErrorCode.FORBIDDEN,
            status_code=status.HTTP_403_FORBIDDEN,
        )


class RateLimitError(APIError):
    """Rate limit exceeded error."""

    def __init__(
        self,
        retry_after: int = 60,
    ) -> None:
        super().__init__(
            message=f"Rate limit exceeded. Please retry after {retry_after} seconds.",
            code=ErrorCode.RATE_LIMITED,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            headers={"Retry-After": str(retry_after)},
        )


class ConflictError(APIError):
    """Resource conflict error."""

    def __init__(
        self,
        message: str = "Resource already exists",
        code: ErrorCode = ErrorCode.ALREADY_EXISTS,
    ) -> None:
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_409_CONFLICT,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle APIError exceptions."""
    request_id = getattr(request.state, "correlation_id", None)
    response = exc.to_response(request_id=request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=response.model_dump(exclude_none=True),
        headers=exc.headers,
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "correlation_id", None)
    response = ErrorResponse(
        code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred",
        request_id=request_id,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(exclude_none=True),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI application."""
    app.add_exception_handler(APIError, api_error_handler)
    # Note: Don't override all Exception handling in production
    # as it may hide useful debugging information
