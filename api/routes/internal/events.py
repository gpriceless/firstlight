"""
Internal SSE Event Streaming Endpoint (stub).

Provides GET /internal/v1/events/stream for real-time event streaming.
Full implementation in Task 3.3.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Partner Integration - Events"])
