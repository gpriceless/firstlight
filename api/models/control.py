"""
Control Plane Pydantic Models.

Request and response models for the LLM Control Plane API endpoints.
All models follow the project's existing Pydantic conventions from
api/models/responses.py and api/models/requests.py.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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
