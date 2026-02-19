"""
Control Plane Job Endpoints.

Provides CRUD operations and state transitions for jobs
accessible to LLM agents via the Control API.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/jobs", tags=["LLM Control - Jobs"])
