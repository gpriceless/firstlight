"""
Orchestrator Agent Module for Event Intelligence Platform.

The Orchestrator is the main conductor that coordinates the entire workflow
from event specification to final products. It provides:

- OrchestratorAgent: Main orchestrator that coordinates Discovery, Pipeline,
  Quality, and Reporting agents
- DelegationStrategy: Task routing and load balancing across agents
- StateManager: Execution state persistence and checkpoint/restore

Example:
    from agents.orchestrator import OrchestratorAgent, OrchestratorConfig

    # Create and configure
    config = OrchestratorConfig(
        state_db_path="./state.db",
        output_dir="./output",
    )
    orchestrator = OrchestratorAgent(config)

    # Initialize and start
    await orchestrator.initialize()
    await orchestrator.start()

    # Process an event
    result = await orchestrator.process_event(event_spec)

    # Shutdown
    await orchestrator.shutdown()
"""

# Main orchestrator
from agents.orchestrator.main import (
    OrchestratorAgent,
    OrchestratorConfig,
)

# State management
from agents.orchestrator.state import (
    ExecutionState,
    ExecutionStage,
    DegradedModeLevel,
    DegradedModeInfo,
    StageProgress,
    StateManager,
)

# Delegation
from agents.orchestrator.delegation import (
    DelegationStrategy,
    DelegationTask,
    DelegationStatus,
    DelegationError,
    AgentType,
    AgentInfo,
    RetryPolicy,
    TimeoutPolicy,
    LoadBalancingStrategy,
    TaskRouter,
)

# Assembly
from agents.orchestrator.assembly import (
    ProductAssembler,
    ProductMetadata,
    ProductFormat,
    ProductType,
    ProvenanceRecord,
    ExecutionSummary,
    AssemblyResult,
    ProductPackager,
)

__all__ = [
    # Main
    "OrchestratorAgent",
    "OrchestratorConfig",
    # State
    "ExecutionState",
    "ExecutionStage",
    "DegradedModeLevel",
    "DegradedModeInfo",
    "StageProgress",
    "StateManager",
    # Delegation
    "DelegationStrategy",
    "DelegationTask",
    "DelegationStatus",
    "DelegationError",
    "AgentType",
    "AgentInfo",
    "RetryPolicy",
    "TimeoutPolicy",
    "LoadBalancingStrategy",
    "TaskRouter",
    # Assembly
    "ProductAssembler",
    "ProductMetadata",
    "ProductFormat",
    "ProductType",
    "ProvenanceRecord",
    "ExecutionSummary",
    "AssemblyResult",
    "ProductPackager",
]
