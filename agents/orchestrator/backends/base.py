"""
StateBackend ABC and core types for the LLM Control Plane.

Defines the simplified state interface that all backend implementations
must satisfy. This is NOT a 1:1 mirror of the existing StateManager class
(which has 11+ methods). The ABC provides a clean, async-first interface
for external consumers (LLM agents, Control API) while the adapter layer
bridges to the existing StateManager internals.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class StateConflictError(Exception):
    """
    Raised when a state transition fails due to a concurrent modification.

    The TOCTOU guard: the caller's expected (phase, status) does not match
    the current (phase, status) in the store.

    Attributes:
        job_id: The job that was being transitioned.
        expected_phase: The phase the caller expected.
        expected_status: The status the caller expected.
        actual_phase: The actual current phase.
        actual_status: The actual current status.
    """

    def __init__(
        self,
        job_id: str,
        expected_phase: str,
        expected_status: str,
        actual_phase: str,
        actual_status: str,
    ):
        self.job_id = job_id
        self.expected_phase = expected_phase
        self.expected_status = expected_status
        self.actual_phase = actual_phase
        self.actual_status = actual_status
        super().__init__(
            f"State conflict for job {job_id}: "
            f"expected ({expected_phase}, {expected_status}), "
            f"got ({actual_phase}, {actual_status})"
        )


@dataclass
class JobState:
    """
    Represents the current state of a job.

    This is the canonical return type for all StateBackend read operations.
    It contains the minimal set of fields needed by external consumers
    (LLM agents, Control API) to understand and act on a job's state.

    Attributes:
        job_id: Unique job identifier (UUID as string).
        customer_id: Tenant identifier for multi-tenant isolation.
        event_type: Type of event being analyzed (e.g., 'flood', 'wildfire').
        phase: Current job phase (from JobPhase enum value).
        status: Current job status within the phase (from JobStatus enum value).
        aoi: Area of Interest as GeoJSON dict, or None if not set.
        parameters: Job parameters as a dict.
        orchestrator_id: ID of the orchestrator handling this job, or None.
        created_at: When the job was created.
        updated_at: When the job was last updated.
    """

    job_id: str
    customer_id: str
    event_type: str
    phase: str
    status: str
    aoi: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    orchestrator_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "job_id": self.job_id,
            "customer_id": self.customer_id,
            "event_type": self.event_type,
            "phase": self.phase,
            "status": self.status,
            "aoi": self.aoi,
            "parameters": self.parameters,
            "orchestrator_id": self.orchestrator_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class StateBackend(ABC):
    """
    Abstract base class for state storage backends.

    All methods are async to support both synchronous backends (wrapped in
    asyncio.to_thread) and natively async backends (asyncpg).

    Implementations:
    - SQLiteStateBackend: Adapter wrapping the existing StateManager.
    - PostGISStateBackend: Native PostGIS backend with asyncpg.
    - DualWriteBackend: Writes to PostGIS primary + SQLite fallback.
    """

    @abstractmethod
    async def get_state(self, job_id: str) -> Optional[JobState]:
        """
        Retrieve the current state for a job.

        Args:
            job_id: Unique job identifier.

        Returns:
            JobState if the job exists, None otherwise.
        """
        ...

    @abstractmethod
    async def set_state(
        self,
        job_id: str,
        phase: str,
        status: str,
        **kwargs: Any,
    ) -> JobState:
        """
        Set the phase and status for a job.

        This is a direct write without TOCTOU protection. Use `transition()`
        for safe concurrent updates.

        Args:
            job_id: Unique job identifier.
            phase: New phase value.
            status: New status value.
            **kwargs: Additional fields to update (e.g., parameters, orchestrator_id).

        Returns:
            The updated JobState.

        Raises:
            KeyError: If the job does not exist.
        """
        ...

    @abstractmethod
    async def transition(
        self,
        job_id: str,
        expected_phase: str,
        expected_status: str,
        new_phase: str,
        new_status: str,
        *,
        reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> JobState:
        """
        Atomically transition a job from one (phase, status) to another.

        This is the TOCTOU-safe state transition method. The transition only
        succeeds if the current (phase, status) matches (expected_phase,
        expected_status). If not, a StateConflictError is raised with the
        actual current state.

        Args:
            job_id: Unique job identifier.
            expected_phase: The phase the caller expects the job to be in.
            expected_status: The status the caller expects.
            new_phase: The target phase.
            new_status: The target status.
            reason: Optional human-readable reason for the transition.
            actor: Optional identifier of who/what triggered the transition.

        Returns:
            The updated JobState after the transition.

        Raises:
            StateConflictError: If current state does not match expected state.
            KeyError: If the job does not exist.
        """
        ...

    @abstractmethod
    async def list_jobs(
        self,
        *,
        customer_id: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobState]:
        """
        List jobs with optional filtering.

        Args:
            customer_id: Filter by tenant.
            phase: Filter by current phase.
            status: Filter by current status.
            event_type: Filter by event type.
            limit: Maximum number of results.
            offset: Pagination offset.

        Returns:
            List of matching JobState instances.
        """
        ...

    @abstractmethod
    async def checkpoint(
        self,
        job_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Save a checkpoint snapshot for a job.

        Checkpoints enable recovery from failures by storing the full
        job state at a point in time.

        Args:
            job_id: Unique job identifier.
            payload: Checkpoint data (typically a serialized state snapshot).

        Raises:
            KeyError: If the job does not exist.
        """
        ...
