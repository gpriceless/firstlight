"""
Control Plane Pydantic Models.

Request and response models for the LLM Control Plane API endpoints.
All models follow the project's existing Pydantic conventions from
api/models/responses.py and api/models/requests.py.
"""

import json
import re
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
# Job Create (Task 2.3)
# =============================================================================


class CreateJobRequest(BaseModel):
    """Request body for creating a new job."""

    event_type: str = Field(
        ...,
        description="Type of event to analyze (e.g., 'flood', 'wildfire')",
        min_length=1,
        max_length=256,
    )
    aoi: Dict[str, Any] = Field(
        ...,
        description="Area of interest as GeoJSON geometry (any type, will be promoted to MultiPolygon)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job parameters",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional reasoning for job creation",
        max_length=65536,
    )

    @field_validator("aoi")
    @classmethod
    def validate_aoi_geojson(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GeoJSON structure and coordinate bounds."""
        if "type" not in v:
            raise ValueError("GeoJSON must include a 'type' field")
        if "coordinates" not in v:
            raise ValueError("GeoJSON must include a 'coordinates' field")

        valid_types = {
            "Point", "MultiPoint", "LineString", "MultiLineString",
            "Polygon", "MultiPolygon", "GeometryCollection",
        }
        if v["type"] not in valid_types:
            raise ValueError(
                f"Invalid GeoJSON type: {v['type']}. Must be one of: {valid_types}"
            )

        # Validate coordinate bounds (WGS84: lon [-180, 180], lat [-90, 90])
        _validate_coordinates(v["coordinates"], v["type"])

        # Check vertex count (configurable, default max 10000)
        vertex_count = _count_vertices(v["coordinates"], v["type"])
        max_vertices = 10000
        if vertex_count > max_vertices:
            raise ValueError(
                f"GeoJSON has {vertex_count} vertices, exceeding maximum of {max_vertices}"
            )

        return v

    @field_validator("reasoning")
    @classmethod
    def reject_null_bytes(cls, v: Optional[str]) -> Optional[str]:
        """Reject null bytes per security spec."""
        if v is not None and "\x00" in v:
            raise ValueError("Reasoning text must not contain null bytes")
        return v


def _validate_coordinates(coords: Any, geom_type: str) -> None:
    """Recursively validate coordinate bounds for WGS84."""
    if geom_type == "Point":
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            raise ValueError("Point coordinates must have at least [lon, lat]")
        _check_bounds(coords[0], coords[1])
    elif geom_type in ("MultiPoint", "LineString"):
        for pt in coords:
            if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                raise ValueError("Coordinate must have at least [lon, lat]")
            _check_bounds(pt[0], pt[1])
    elif geom_type in ("MultiLineString", "Polygon"):
        for ring in coords:
            for pt in ring:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    raise ValueError("Coordinate must have at least [lon, lat]")
                _check_bounds(pt[0], pt[1])
    elif geom_type == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for pt in ring:
                    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                        raise ValueError("Coordinate must have at least [lon, lat]")
                    _check_bounds(pt[0], pt[1])


def _check_bounds(lon: float, lat: float) -> None:
    """Check that coordinates are within WGS84 bounds."""
    if not (-180 <= lon <= 180):
        raise ValueError(f"Longitude {lon} out of WGS84 bounds [-180, 180]")
    if not (-90 <= lat <= 90):
        raise ValueError(f"Latitude {lat} out of WGS84 bounds [-90, 90]")


def _count_vertices(coords: Any, geom_type: str) -> int:
    """Count the total number of vertices in a GeoJSON geometry."""
    if geom_type == "Point":
        return 1
    elif geom_type in ("MultiPoint", "LineString"):
        return len(coords)
    elif geom_type in ("MultiLineString", "Polygon"):
        return sum(len(ring) for ring in coords)
    elif geom_type == "MultiPolygon":
        return sum(len(pt) for polygon in coords for pt in polygon)
    return 0


class JobResponse(BaseModel):
    """Response for job creation."""

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    phase: str = Field(..., description="Initial job phase")
    status: str = Field(..., description="Initial job status")
    event_type: str = Field(..., description="Type of event")
    created_at: datetime = Field(..., description="Creation timestamp")


# =============================================================================
# Job Detail (Task 2.4)
# =============================================================================


class JobDetailLinks(BaseModel):
    """HATEOAS links for job detail."""

    self: str = Field(..., description="Link to this job")
    events: str = Field(..., description="Link to job events")
    checkpoints: str = Field(..., description="Link to job checkpoints")


class JobDetailResponse(BaseModel):
    """Full job detail response."""

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    phase: str = Field(..., description="Current job phase")
    status: str = Field(..., description="Current job status")
    event_type: str = Field(..., description="Type of event being analyzed")
    aoi: Optional[Dict[str, Any]] = Field(
        default=None, description="Area of interest as GeoJSON"
    )
    aoi_area_km2: Optional[float] = Field(
        default=None, description="Area of interest in square kilometers"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    orchestrator_id: Optional[str] = Field(
        default=None, description="ID of the orchestrator handling this job"
    )
    created_at: datetime = Field(..., description="Job creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    _links: Optional[JobDetailLinks] = Field(
        default=None,
        description="HATEOAS links",
        alias="_links",
    )

    model_config = {"populate_by_name": True}


# =============================================================================
# Transitions (Task 2.5)
# =============================================================================


class TransitionRequest(BaseModel):
    """Request body for state transitions with TOCTOU guard."""

    expected_phase: str = Field(
        ..., description="Expected current phase (TOCTOU guard)"
    )
    expected_status: str = Field(
        ..., description="Expected current status (TOCTOU guard)"
    )
    target_phase: str = Field(..., description="Target phase to transition to")
    target_status: str = Field(..., description="Target status to transition to")
    reason: Optional[str] = Field(
        default=None,
        description="Reason for the transition",
        max_length=65536,
    )


class TransitionResponse(BaseModel):
    """Response after a successful state transition."""

    job_id: str = Field(..., description="Job identifier")
    phase: str = Field(..., description="New phase after transition")
    status: str = Field(..., description="New status after transition")
    updated_at: datetime = Field(..., description="Timestamp of the transition")


# =============================================================================
# Reasoning (Task 2.7)
# =============================================================================


class ReasoningRequest(BaseModel):
    """Request body for appending a reasoning entry."""

    reasoning: str = Field(
        ...,
        description="Reasoning text from the LLM agent",
        max_length=65536,
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score [0.0, 1.0]",
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured payload",
    )

    @field_validator("reasoning")
    @classmethod
    def reject_null_bytes_reasoning(cls, v: str) -> str:
        """Reject null bytes per security spec."""
        if "\x00" in v:
            raise ValueError("Reasoning text must not contain null bytes")
        return v


class ReasoningResponse(BaseModel):
    """Response after appending a reasoning entry."""

    event_seq: int = Field(..., description="Sequence number of the inserted event")


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
