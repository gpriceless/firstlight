"""
Execution Infrastructure for Multiverse Dive.

Provides execution profiles, state persistence, and tiled processing
for resource-constrained environments.

Key Components (Group L, Tracks 7-8):
- ExecutionProfile: Resource constraint definitions
- ProfileManager: Profile selection and validation
- WorkflowState: Workflow state tracking
- StateManager: State persistence and recovery
"""

from core.execution.profiles import (
    ExecutionProfile,
    ProcessingMode,
    ProfileManager,
    ProfileName,
    SystemResources,
    auto_select_profile,
    detect_resources,
    get_profile,
)
from core.execution.state import (
    Checkpoint,
    StateManager,
    StepStatus,
    WorkflowStage,
    WorkflowState,
    WorkflowStep,
    create_workflow_state,
    get_state_manager,
)

__all__ = [
    # Profiles (Track 7)
    "ExecutionProfile",
    "ProcessingMode",
    "ProfileManager",
    "ProfileName",
    "SystemResources",
    "auto_select_profile",
    "detect_resources",
    "get_profile",
    # State (Track 8)
    "Checkpoint",
    "StateManager",
    "StepStatus",
    "WorkflowStage",
    "WorkflowState",
    "WorkflowStep",
    "create_workflow_state",
    "get_state_manager",
]
