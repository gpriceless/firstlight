"""
Job state model for the LLM Control Plane.

Defines the canonical phase/status enums, valid phase-status pairs,
transition validation, and the mapping between the existing ExecutionStage
enum and the new JobPhase/JobStatus enums.

The existing ExecutionStage (in agents/orchestrator/state.py) is an internal
implementation detail of the orchestrator. The new JobPhase/JobStatus are the
externally-visible state model exposed through the Control API.
"""

from enum import Enum
from typing import Dict, FrozenSet, Optional, Set, Tuple


class JobPhase(str, Enum):
    """
    Top-level phases of a job lifecycle.

    These are the externally-visible phases exposed through the Control API.
    Terminal states FAILED and CANCELLED are represented as JobStatus values,
    not phases -- a job can fail or be cancelled in any phase.
    """

    QUEUED = "QUEUED"
    DISCOVERING = "DISCOVERING"
    INGESTING = "INGESTING"
    NORMALIZING = "NORMALIZING"
    ANALYZING = "ANALYZING"
    REPORTING = "REPORTING"
    COMPLETE = "COMPLETE"


class JobStatus(str, Enum):
    """
    Per-phase substatus values.

    Each phase has a defined set of valid substatuses. Terminal statuses
    FAILED and CANCELLED are valid in any phase.
    """

    # QUEUED phase statuses
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"

    # DISCOVERING phase statuses
    DISCOVERING = "DISCOVERING"
    DISCOVERED = "DISCOVERED"
    DISCOVERY_FAILED = "DISCOVERY_FAILED"

    # INGESTING phase statuses
    INGESTING = "INGESTING"
    INGESTED = "INGESTED"
    INGESTION_FAILED = "INGESTION_FAILED"

    # NORMALIZING phase statuses
    NORMALIZING = "NORMALIZING"
    NORMALIZED = "NORMALIZED"
    NORMALIZATION_FAILED = "NORMALIZATION_FAILED"

    # ANALYZING phase statuses
    ANALYZING = "ANALYZING"
    QUALITY_CHECK = "QUALITY_CHECK"
    ANALYZED = "ANALYZED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"

    # REPORTING phase statuses
    REPORTING = "REPORTING"
    ASSEMBLING = "ASSEMBLING"
    REPORTED = "REPORTED"
    REPORTING_FAILED = "REPORTING_FAILED"

    # COMPLETE phase statuses
    COMPLETE = "COMPLETE"

    # Universal terminal statuses (valid in any phase)
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# ============================================================================
# Valid (phase, status) pairs
# ============================================================================

# Terminal statuses that are valid in any phase
_TERMINAL_STATUSES: FrozenSet[JobStatus] = frozenset({
    JobStatus.FAILED,
    JobStatus.CANCELLED,
})

# Per-phase valid statuses (excluding universal terminal statuses)
_PHASE_STATUSES: Dict[JobPhase, FrozenSet[JobStatus]] = {
    JobPhase.QUEUED: frozenset({
        JobStatus.PENDING,
        JobStatus.VALIDATING,
        JobStatus.VALIDATED,
        JobStatus.VALIDATION_FAILED,
    }),
    JobPhase.DISCOVERING: frozenset({
        JobStatus.DISCOVERING,
        JobStatus.DISCOVERED,
        JobStatus.DISCOVERY_FAILED,
    }),
    JobPhase.INGESTING: frozenset({
        JobStatus.INGESTING,
        JobStatus.INGESTED,
        JobStatus.INGESTION_FAILED,
    }),
    JobPhase.NORMALIZING: frozenset({
        JobStatus.NORMALIZING,
        JobStatus.NORMALIZED,
        JobStatus.NORMALIZATION_FAILED,
    }),
    JobPhase.ANALYZING: frozenset({
        JobStatus.ANALYZING,
        JobStatus.QUALITY_CHECK,
        JobStatus.ANALYZED,
        JobStatus.ANALYSIS_FAILED,
    }),
    JobPhase.REPORTING: frozenset({
        JobStatus.REPORTING,
        JobStatus.ASSEMBLING,
        JobStatus.REPORTED,
        JobStatus.REPORTING_FAILED,
    }),
    JobPhase.COMPLETE: frozenset({
        JobStatus.COMPLETE,
    }),
}

# Full valid pairs dict: phase -> set of valid statuses (including terminal)
VALID_PHASE_STATUS_PAIRS: Dict[JobPhase, FrozenSet[JobStatus]] = {
    phase: statuses | _TERMINAL_STATUSES
    for phase, statuses in _PHASE_STATUSES.items()
}


def is_valid_phase_status(phase: str, status: str) -> bool:
    """
    Check whether a (phase, status) pair is valid.

    Args:
        phase: Phase string (must match a JobPhase value).
        status: Status string (must match a JobStatus value).

    Returns:
        True if the pair is valid, False otherwise.
    """
    try:
        p = JobPhase(phase)
        s = JobStatus(status)
    except ValueError:
        return False

    return s in VALID_PHASE_STATUS_PAIRS.get(p, frozenset())


def is_terminal_status(status: str) -> bool:
    """Check if a status is terminal (FAILED or CANCELLED)."""
    try:
        return JobStatus(status) in _TERMINAL_STATUSES
    except ValueError:
        return False


# ============================================================================
# Phase ordering for transition validation
# ============================================================================

_PHASE_ORDER: Dict[JobPhase, int] = {
    JobPhase.QUEUED: 0,
    JobPhase.DISCOVERING: 1,
    JobPhase.INGESTING: 2,
    JobPhase.NORMALIZING: 3,
    JobPhase.ANALYZING: 4,
    JobPhase.REPORTING: 5,
    JobPhase.COMPLETE: 6,
}


def validate_transition(
    from_phase: str,
    from_status: str,
    to_phase: str,
    to_status: str,
) -> bool:
    """
    Validate whether a state transition is allowed.

    Rules:
    1. Both (from_phase, from_status) and (to_phase, to_status) must be valid pairs.
    2. Cannot transition FROM a terminal status (FAILED, CANCELLED) or
       from the COMPLETE phase.
    3. A job can move to a terminal status from any non-terminal state.
    4. Forward phase transitions are allowed (phase order increases).
    5. Same-phase transitions are allowed (status changes within a phase).
    6. Backward phase transitions are NOT allowed (no going back).

    Args:
        from_phase: Current phase.
        from_status: Current status.
        to_phase: Target phase.
        to_status: Target status.

    Returns:
        True if the transition is valid, False otherwise.
    """
    # Both pairs must be valid
    if not is_valid_phase_status(from_phase, from_status):
        return False
    if not is_valid_phase_status(to_phase, to_status):
        return False

    from_p = JobPhase(from_phase)
    from_s = JobStatus(from_status)
    to_p = JobPhase(to_phase)
    to_s = JobStatus(to_status)

    # Cannot transition from a terminal status
    if from_s in _TERMINAL_STATUSES:
        return False

    # Cannot transition from COMPLETE phase
    if from_p == JobPhase.COMPLETE:
        return False

    # Transitioning to a terminal status is always valid (from non-terminal)
    if to_s in _TERMINAL_STATUSES:
        return True

    # Phase ordering check: no backward transitions
    from_order = _PHASE_ORDER[from_p]
    to_order = _PHASE_ORDER[to_p]

    if to_order < from_order:
        return False

    return True


# ============================================================================
# ExecutionStage <-> JobPhase mapping
# ============================================================================

# Import ExecutionStage lazily to avoid circular imports at module level.
# The mapping functions import it on first call.

_EXECUTION_STAGE_TO_PHASE_STATUS: Optional[Dict] = None


def _build_stage_mapping() -> Dict:
    """Build the ExecutionStage -> (JobPhase, JobStatus) mapping lazily."""
    from agents.orchestrator.state import ExecutionStage

    return {
        ExecutionStage.PENDING: (JobPhase.QUEUED, JobStatus.PENDING),
        ExecutionStage.VALIDATING: (JobPhase.QUEUED, JobStatus.VALIDATING),
        ExecutionStage.DISCOVERY: (JobPhase.DISCOVERING, JobStatus.DISCOVERING),
        ExecutionStage.PIPELINE: (JobPhase.INGESTING, JobStatus.INGESTING),
        ExecutionStage.QUALITY: (JobPhase.ANALYZING, JobStatus.QUALITY_CHECK),
        ExecutionStage.REPORTING: (JobPhase.REPORTING, JobStatus.REPORTING),
        ExecutionStage.ASSEMBLY: (JobPhase.REPORTING, JobStatus.ASSEMBLING),
        ExecutionStage.COMPLETED: (JobPhase.COMPLETE, JobStatus.COMPLETE),
        ExecutionStage.FAILED: (JobPhase.QUEUED, JobStatus.FAILED),  # phase is contextual
        ExecutionStage.CANCELLED: (JobPhase.QUEUED, JobStatus.CANCELLED),  # phase is contextual
    }


def execution_stage_to_phase_status(
    stage_value: str,
) -> Tuple[JobPhase, JobStatus]:
    """
    Map an ExecutionStage value to (JobPhase, JobStatus).

    For FAILED and CANCELLED, the returned phase is a placeholder (QUEUED);
    callers should use the job's actual current phase instead.

    Args:
        stage_value: The string value of an ExecutionStage enum member.

    Returns:
        Tuple of (JobPhase, JobStatus).

    Raises:
        ValueError: If the stage_value is not a valid ExecutionStage.
    """
    global _EXECUTION_STAGE_TO_PHASE_STATUS
    if _EXECUTION_STAGE_TO_PHASE_STATUS is None:
        _EXECUTION_STAGE_TO_PHASE_STATUS = _build_stage_mapping()

    from agents.orchestrator.state import ExecutionStage

    stage = ExecutionStage(stage_value)
    return _EXECUTION_STAGE_TO_PHASE_STATUS[stage]


_PHASE_STATUS_TO_EXECUTION_STAGE: Optional[Dict] = None


def _build_reverse_mapping() -> Dict[Tuple[JobPhase, JobStatus], "ExecutionStage"]:
    """Build the (JobPhase, JobStatus) -> ExecutionStage reverse mapping lazily."""
    from agents.orchestrator.state import ExecutionStage

    return {
        (JobPhase.QUEUED, JobStatus.PENDING): ExecutionStage.PENDING,
        (JobPhase.QUEUED, JobStatus.VALIDATING): ExecutionStage.VALIDATING,
        (JobPhase.DISCOVERING, JobStatus.DISCOVERING): ExecutionStage.DISCOVERY,
        (JobPhase.INGESTING, JobStatus.INGESTING): ExecutionStage.PIPELINE,
        (JobPhase.NORMALIZING, JobStatus.NORMALIZING): ExecutionStage.PIPELINE,
        (JobPhase.ANALYZING, JobStatus.ANALYZING): ExecutionStage.PIPELINE,
        (JobPhase.ANALYZING, JobStatus.QUALITY_CHECK): ExecutionStage.QUALITY,
        (JobPhase.REPORTING, JobStatus.REPORTING): ExecutionStage.REPORTING,
        (JobPhase.REPORTING, JobStatus.ASSEMBLING): ExecutionStage.ASSEMBLY,
        (JobPhase.COMPLETE, JobStatus.COMPLETE): ExecutionStage.COMPLETED,
    }


def phase_status_to_execution_stage(
    phase: str,
    status: str,
) -> "ExecutionStage":
    """
    Map a (JobPhase, JobStatus) pair to the closest ExecutionStage.

    This is a best-effort reverse mapping. Some (phase, status) pairs
    map to the same ExecutionStage (e.g., INGESTING and NORMALIZING both
    map to PIPELINE). Terminal statuses map to their respective
    ExecutionStage (FAILED -> FAILED, CANCELLED -> CANCELLED).

    Args:
        phase: JobPhase value string.
        status: JobStatus value string.

    Returns:
        The closest ExecutionStage.

    Raises:
        ValueError: If no mapping exists for the given pair.
    """
    global _PHASE_STATUS_TO_EXECUTION_STAGE
    if _PHASE_STATUS_TO_EXECUTION_STAGE is None:
        _PHASE_STATUS_TO_EXECUTION_STAGE = _build_reverse_mapping()

    from agents.orchestrator.state import ExecutionStage

    p = JobPhase(phase)
    s = JobStatus(status)

    # Terminal statuses have direct mappings regardless of phase
    if s == JobStatus.FAILED:
        return ExecutionStage.FAILED
    if s == JobStatus.CANCELLED:
        return ExecutionStage.CANCELLED

    key = (p, s)
    if key in _PHASE_STATUS_TO_EXECUTION_STAGE:
        return _PHASE_STATUS_TO_EXECUTION_STAGE[key]

    # For statuses within a phase that don't have exact mappings,
    # fall back to the phase's primary ExecutionStage
    _PHASE_TO_DEFAULT_STAGE = {
        JobPhase.QUEUED: ExecutionStage.PENDING,
        JobPhase.DISCOVERING: ExecutionStage.DISCOVERY,
        JobPhase.INGESTING: ExecutionStage.PIPELINE,
        JobPhase.NORMALIZING: ExecutionStage.PIPELINE,
        JobPhase.ANALYZING: ExecutionStage.QUALITY,
        JobPhase.REPORTING: ExecutionStage.REPORTING,
        JobPhase.COMPLETE: ExecutionStage.COMPLETED,
    }

    if p in _PHASE_TO_DEFAULT_STAGE:
        return _PHASE_TO_DEFAULT_STAGE[p]

    raise ValueError(f"No ExecutionStage mapping for ({phase}, {status})")
