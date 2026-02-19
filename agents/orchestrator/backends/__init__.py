"""
State backend implementations for the Orchestrator Agent.

Provides pluggable state storage backends:
- StateBackend: Abstract base class defining the interface
- SQLiteStateBackend: Wraps existing StateManager for backward compatibility
- PostGISStateBackend: PostGIS-backed state storage with geospatial support
- DualWriteBackend: Writes to both PostGIS (primary) and SQLite (fallback)
"""

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)

__all__ = [
    "JobState",
    "StateBackend",
    "StateConflictError",
]
