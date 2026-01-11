"""
Algorithm selection module.

Provides intelligent algorithm selection based on:
- Event type matching
- Data availability
- Compute constraints
- Algorithm validation status
- Reproducible, deterministic selection
"""

# Deterministic selection (Track 7)
from core.analysis.selection.deterministic import (
    DeterministicSelector,
    SelectionPlan,
    AlgorithmSelection,
    SelectionQuality,
    SelectionReason,
    SelectionConstraints,
    DataAvailability,
    create_deterministic_selector,
    SELECTOR_VERSION,
)

# Algorithm selector (Track 6)
from core.analysis.selection.selector import (
    AlgorithmSelector,
    SelectionCriteria,
    SelectionResult,
    SelectionContext,
    ComputeProfile,
    ComputeConstraints,
    RejectionReason,
)

__all__ = [
    # Deterministic selection (Track 7)
    "DeterministicSelector",
    "SelectionPlan",
    "AlgorithmSelection",
    "SelectionQuality",
    "SelectionReason",
    "SelectionConstraints",
    "DataAvailability",
    "create_deterministic_selector",
    "SELECTOR_VERSION",
    # Algorithm selector (Track 6)
    "AlgorithmSelector",
    "SelectionCriteria",
    "SelectionResult",
    "SelectionContext",
    "ComputeProfile",
    "ComputeConstraints",
    "RejectionReason",
]
