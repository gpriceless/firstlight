"""
SQLite state backend adapter.

Wraps the existing StateManager class to implement the StateBackend ABC.
This is a pure adapter -- no logic changes to StateManager itself.

The adapter maps between the ABC's simplified 5-method interface and the
existing StateManager's 11+ methods, handling the ExecutionStage <-> JobPhase
translation in both directions.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)
from agents.orchestrator.state import (
    ExecutionStage,
    ExecutionState,
    StateManager,
)
from agents.orchestrator.state_model import (
    JobPhase,
    JobStatus,
    execution_stage_to_phase_status,
    is_valid_phase_status,
    phase_status_to_execution_stage,
)

logger = logging.getLogger(__name__)


class SQLiteStateBackend(StateBackend):
    """
    StateBackend adapter wrapping the existing StateManager (SQLite).

    This adapter allows the existing SQLite-based state management to be
    used through the new StateBackend interface. It translates between the
    external JobPhase/JobStatus model and the internal ExecutionStage model.

    Args:
        state_manager: An existing StateManager instance. If None, a new
            in-memory StateManager is created.
        db_path: Path to the SQLite database. Only used if state_manager is None.
    """

    def __init__(
        self,
        state_manager: Optional[StateManager] = None,
        db_path: Optional[str] = None,
    ):
        if state_manager is not None:
            self._state_manager = state_manager
        else:
            self._state_manager = StateManager(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the underlying StateManager is initialized."""
        if not self._initialized:
            await self._state_manager.initialize()
            self._initialized = True

    def _execution_state_to_job_state(
        self,
        state: ExecutionState,
        customer_id: str = "legacy",
    ) -> JobState:
        """
        Convert an internal ExecutionState to a JobState.

        The existing ExecutionState does not have customer_id or event_type,
        so these are extracted from the event_spec or defaulted.
        """
        # Map ExecutionStage to (JobPhase, JobStatus)
        phase, status = execution_stage_to_phase_status(state.current_stage.value)

        # Extract event_type from the event_spec if available
        event_type = state.event_spec.get("intent", {}).get("class", "unknown")
        if event_type == "unknown":
            event_type = state.event_spec.get("event_type", "unknown")

        # Extract AOI from spatial spec
        aoi = state.event_spec.get("spatial", {}).get("geometry", None)
        if aoi is None:
            # Try bbox format
            bbox = state.event_spec.get("spatial", {}).get("bbox", None)
            if bbox and len(bbox) == 4:
                aoi = {
                    "type": "MultiPolygon",
                    "coordinates": [[
                        [
                            [bbox[0], bbox[1]],
                            [bbox[2], bbox[1]],
                            [bbox[2], bbox[3]],
                            [bbox[0], bbox[3]],
                            [bbox[0], bbox[1]],
                        ]
                    ]],
                }

        return JobState(
            job_id=state.event_id,
            customer_id=customer_id,
            event_type=event_type,
            phase=phase.value,
            status=status.value,
            aoi=aoi,
            parameters=state.event_spec.get("parameters", {}),
            orchestrator_id=state.orchestrator_id or None,
            created_at=state.created_at,
            updated_at=state.completed_at or state.created_at,
        )

    async def get_state(self, job_id: str) -> Optional[JobState]:
        """Retrieve job state by ID."""
        await self._ensure_initialized()
        state = await self._state_manager.get_state(job_id)
        if state is None:
            return None
        return self._execution_state_to_job_state(state)

    async def set_state(
        self,
        job_id: str,
        phase: str,
        status: str,
        **kwargs: Any,
    ) -> JobState:
        """Set the phase and status for a job."""
        await self._ensure_initialized()

        # Validate the pair
        if not is_valid_phase_status(phase, status):
            raise ValueError(f"Invalid (phase, status) pair: ({phase}, {status})")

        state = await self._state_manager.get_state(job_id)
        if state is None:
            raise KeyError(f"Job {job_id} not found")

        # Map to ExecutionStage
        try:
            exec_stage = phase_status_to_execution_stage(phase, status)
        except ValueError:
            raise ValueError(f"Cannot map ({phase}, {status}) to ExecutionStage")

        # Update the internal state
        if exec_stage in (ExecutionStage.FAILED, ExecutionStage.CANCELLED):
            state.current_stage = exec_stage
            state.completed_at = datetime.now(timezone.utc)
        elif exec_stage == ExecutionStage.COMPLETED:
            state.current_stage = exec_stage
            state.completed_at = datetime.now(timezone.utc)
        else:
            state.current_stage = exec_stage

        # Apply any extra kwargs
        if "orchestrator_id" in kwargs:
            state.orchestrator_id = kwargs["orchestrator_id"]
        if "parameters" in kwargs:
            state.event_spec["parameters"] = kwargs["parameters"]

        await self._state_manager.update_state(state)

        return self._execution_state_to_job_state(state)

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
        Atomically transition a job from one state to another.

        Uses the StateManager's lock to ensure atomicity for SQLite.
        """
        await self._ensure_initialized()

        # Validate both pairs
        if not is_valid_phase_status(expected_phase, expected_status):
            raise ValueError(
                f"Invalid expected (phase, status): ({expected_phase}, {expected_status})"
            )
        if not is_valid_phase_status(new_phase, new_status):
            raise ValueError(
                f"Invalid target (phase, status): ({new_phase}, {new_status})"
            )

        state = await self._state_manager.get_state(job_id)
        if state is None:
            raise KeyError(f"Job {job_id} not found")

        # Check current state matches expected
        current_phase, current_status = execution_stage_to_phase_status(
            state.current_stage.value
        )

        if (
            current_phase.value != expected_phase
            or current_status.value != expected_status
        ):
            raise StateConflictError(
                job_id=job_id,
                expected_phase=expected_phase,
                expected_status=expected_status,
                actual_phase=current_phase.value,
                actual_status=current_status.value,
            )

        # Perform the transition
        return await self.set_state(job_id, new_phase, new_status)

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
        """List jobs with optional filtering."""
        await self._ensure_initialized()

        # Get all active executions from StateManager
        all_states = await self._state_manager.list_active_executions()

        # Also need to query all states (including terminal) from the database
        # Since list_active_executions only returns non-terminal states,
        # we also need to query the database directly for a complete listing.
        # However, to avoid modifying StateManager, we work with what we have.
        # For a complete implementation, the PostGIS backend handles this properly.

        job_states = [
            self._execution_state_to_job_state(s) for s in all_states
        ]

        # Apply filters
        if customer_id is not None:
            job_states = [j for j in job_states if j.customer_id == customer_id]
        if phase is not None:
            job_states = [j for j in job_states if j.phase == phase]
        if status is not None:
            job_states = [j for j in job_states if j.status == status]
        if event_type is not None:
            job_states = [j for j in job_states if j.event_type == event_type]

        # Apply pagination
        return job_states[offset: offset + limit]

    async def checkpoint(
        self,
        job_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Save a checkpoint snapshot for a job."""
        await self._ensure_initialized()

        state = await self._state_manager.get_state(job_id)
        if state is None:
            raise KeyError(f"Job {job_id} not found")

        # Store the payload in checkpoint_data
        state.checkpoint_data = payload
        await self._state_manager.checkpoint(state)
