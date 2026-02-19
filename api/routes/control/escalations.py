"""
Control Plane Escalation Endpoints.

Provides endpoints for creating, resolving, and listing
escalations scoped to jobs.
"""

from fastapi import APIRouter

router = APIRouter(tags=["LLM Control - Escalations"])
