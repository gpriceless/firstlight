"""
Control Plane Pydantic Models.

Request and response models for the LLM Control Plane API endpoints.
All models follow the project's existing Pydantic conventions from
api/models/responses.py and api/models/requests.py.
"""

import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Shared / Base Models
# =============================================================================


class JobSummary(BaseModel):
    """Summary job information for list responses."""

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    phase: str = Field(..., description="Current job phase")
    status: str = Field(..., description="Current job status within the phase")
    event_type: str = Field(..., description="Type of event being analyzed")
    aoi_area_km2: Optional[float] = Field(
        default=None, description="Area of interest in square kilometers"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class PaginatedJobsResponse(BaseModel):
    """Paginated response for job listing."""

    items: List[JobSummary] = Field(..., description="List of job summaries")
    total: int = Field(..., description="Total number of matching jobs")
    page: int = Field(..., description="Current page number (1-based)")
    page_size: int = Field(..., description="Number of items per page")


# =============================================================================
# Escalation Models (Task 2.8)
# =============================================================================


class EscalationSeverity(str, Enum):
    """Escalation severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EscalationRequest(BaseModel):
    """Request body for creating an escalation."""

    severity: EscalationSeverity = Field(
        ..., description="Escalation severity"
    )
    reason: str = Field(
        ...,
        description="Reason for the escalation",
        min_length=1,
        max_length=65536,
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional context JSON (max 16KB)",
    )

    @field_validator("context")
    @classmethod
    def validate_context_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate context JSON size does not exceed 16KB."""
        if v is not None:
            serialized = json.dumps(v)
            if len(serialized.encode("utf-8")) > 16384:
                raise ValueError("Context JSON exceeds maximum size of 16KB")
        return v


class EscalationResolveRequest(BaseModel):
    """Request body for resolving an escalation."""

    resolution: str = Field(
        ...,
        description="Resolution text",
        min_length=1,
        max_length=65536,
    )


class EscalationResponse(BaseModel):
    """Response for an escalation."""

    escalation_id: str = Field(..., description="Escalation identifier (UUID)")
    job_id: str = Field(..., description="Job identifier (UUID)")
    customer_id: str = Field(..., description="Customer identifier")
    severity: str = Field(..., description="Escalation severity")
    reason: str = Field(..., description="Escalation reason")
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional context"
    )
    created_at: datetime = Field(..., description="When the escalation was created")
    resolved_at: Optional[datetime] = Field(
        default=None, description="When the escalation was resolved"
    )
    resolution: Optional[str] = Field(
        default=None, description="Resolution text"
    )
    resolved_by: Optional[str] = Field(
        default=None, description="Who resolved the escalation"
    )


class PaginatedEscalationsResponse(BaseModel):
    """Paginated response for escalation listing."""

    items: List[EscalationResponse] = Field(..., description="List of escalations")
    total: int = Field(..., description="Total matching escalations")


# =============================================================================
# Tool Schema Models (Task 2.9)
# =============================================================================


class ToolSchema(BaseModel):
    """OpenAI-compatible function-calling tool schema."""

    name: str = Field(..., description="Tool/function name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(
        ..., description="JSON Schema for the function parameters"
    )


class ToolsResponse(BaseModel):
    """Response containing available tool schemas."""

    tools: List[ToolSchema] = Field(
        ..., description="Available tool schemas in OpenAI function-calling format"
    )
