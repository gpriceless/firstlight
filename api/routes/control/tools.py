"""
Control Plane Tool Schema Endpoints.

Provides endpoints for discovering available tool schemas
(OpenAI-compatible function-calling format) for LLM agents.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/tools", tags=["LLM Control - Tools"])
