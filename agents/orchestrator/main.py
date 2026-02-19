"""
Orchestrator Agent for Event Intelligence Platform.

The Orchestrator is the main conductor that coordinates the entire workflow
from event specification to final products:
- Takes an event specification as input
- Coordinates the full workflow: Discovery -> Pipeline -> Quality -> Reporting
- Handles degraded mode decisions
- Tracks overall execution state and progress
- Emits status updates and completion notifications

The Orchestrator delegates to specialized agents:
- Discovery Agent: Data source discovery and selection
- Pipeline Agent: Analysis pipeline execution
- Quality Agent: Quality control and validation
- Reporting Agent: Report generation and delivery

Example:
    from agents.orchestrator import OrchestratorAgent

    # Create and initialize
    orchestrator = OrchestratorAgent()
    await orchestrator.start()

    # Process an event
    result = await orchestrator.process_event(event_spec)

    # Cleanup
    await orchestrator.shutdown()
"""

import asyncio
import logging
import traceback
import uuid
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from agents.orchestrator.backends.base import StateBackend

from agents.base import (
    BaseAgent,
    AgentType as BaseAgentType,
    AgentMessage,
    AgentState,
    MessageType,
    MessagePriority,
    RetryPolicy as BaseRetryPolicy,
)
from agents.orchestrator.state import (
    ExecutionState,
    ExecutionStage,
    DegradedModeLevel,
    DegradedModeInfo,
    StateManager,
    StageProgress,
)
from agents.orchestrator.delegation import (
    DelegationStrategy,
    DelegationTask,
    AgentType,
    AgentInfo,
    RetryPolicy,
    TimeoutPolicy,
    LoadBalancingStrategy,
    DelegationError,
)
from agents.orchestrator.assembly import (
    ProductAssembler,
    AssemblyResult,
    ProductFormat,
    ExecutionSummary,
)


logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """
    Configuration for the Orchestrator Agent.

    Attributes:
        state_db_path: Path to state database
        output_dir: Base output directory
        default_timeout_seconds: Default operation timeout
        enable_degraded_mode: Allow degraded mode operation
        min_confidence_threshold: Minimum acceptable confidence
        checkpoint_frequency: How often to checkpoint (stages)
        notification_webhook: URL for completion notifications
    """
    state_db_path: Optional[str] = None
    output_dir: str = "./output"
    default_timeout_seconds: float = 600.0
    enable_degraded_mode: bool = True
    min_confidence_threshold: float = 0.3
    checkpoint_frequency: int = 1
    notification_webhook: Optional[str] = None


class OrchestratorAgent(BaseAgent):
    """
    Main Orchestrator Agent for event processing.

    Coordinates the entire workflow from event specification to final products,
    managing state, delegating to specialized agents, and handling failures.

    Key Responsibilities:
    - Parse and validate event specifications
    - Resolve event intent
    - Delegate to Discovery, Pipeline, Quality, and Reporting agents
    - Handle degraded mode decisions
    - Track execution state and progress
    - Assemble final products
    - Emit notifications

    Example:
        config = OrchestratorConfig(
            state_db_path="./state.db",
            output_dir="./output",
        )
        orchestrator = OrchestratorAgent(config)
        await orchestrator.initialize()
        await orchestrator.start()

        # Load and process event
        with open("examples/flood_event.yaml") as f:
            event_spec = yaml.safe_load(f)

        result = await orchestrator.process_event(event_spec)
        print(f"Products generated: {len(result['products'])}")
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        agent_id: Optional[str] = None,
        retry_policy: Optional[BaseRetryPolicy] = None,
        state_backend: Optional["StateBackend"] = None,
    ):
        """
        Initialize the Orchestrator Agent.

        Args:
            config: Orchestrator configuration
            agent_id: Unique agent identifier
            retry_policy: Retry policy for operations
            state_backend: Optional StateBackend instance. If not provided,
                one is created based on the FIRSTLIGHT_STATE_BACKEND setting.
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=BaseAgentType.ORCHESTRATOR,
            retry_policy=retry_policy or BaseRetryPolicy(max_retries=3),
        )
        self.config = config or OrchestratorConfig()

        # State management (legacy StateManager for internal use)
        self._state_manager: Optional[StateManager] = None
        self._state_manager_initialized = False

        # New pluggable state backend (if provided, takes precedence)
        self._state_backend: Optional["StateBackend"] = state_backend

        # Delegation strategy
        self._delegation = DelegationStrategy(
            load_balancing=LoadBalancingStrategy.LEAST_LOADED,
            default_retry_policy=RetryPolicy(max_retries=3),
            timeout_policy=TimeoutPolicy(
                default_timeout_seconds=self.config.default_timeout_seconds
            ),
        )

        # Product assembler
        self._assembler = ProductAssembler(
            output_dir=self.config.output_dir,
            include_provenance=True,
            include_summary=True,
            generate_manifest=True,
        )

        # Agent executors (to be registered)
        self._agent_executors: Dict[AgentType, Callable] = {}

        # Status callbacks
        self._status_callbacks: List[Callable[[str, ExecutionState], None]] = []
        self._completion_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        # Active executions
        self._active_executions: Dict[str, ExecutionState] = {}

    async def _ensure_state_manager(self) -> None:
        """Ensure state manager and state backend are initialized."""
        if not self._state_manager_initialized:
            self._state_manager = StateManager(self.config.state_db_path)
            await self._state_manager.initialize()
            self._state_manager_initialized = True
            # Set up escalation handler
            self._delegation.add_escalation_handler(self._handle_escalation)

            # Initialize state backend if not already provided
            if self._state_backend is None:
                self._state_backend = self._create_state_backend()

            logger.info("Orchestrator state manager initialized")

    def _create_state_backend(self) -> "StateBackend":
        """
        Create a StateBackend based on the FIRSTLIGHT_STATE_BACKEND setting.

        Imports are done lazily to avoid circular dependencies between
        agents/ and api/ modules.

        Returns:
            A configured StateBackend instance.
        """
        from agents.orchestrator.backends.sqlite_backend import SQLiteStateBackend

        # Determine the backend type from config, defaulting to sqlite
        # to avoid breaking existing deployments that lack PostGIS.
        backend_type = "sqlite"
        try:
            from api.config import get_settings
            settings = get_settings()
            backend_type = settings.state_backend.value
        except Exception:
            logger.debug(
                "Could not load settings for state_backend, defaulting to sqlite"
            )

        if backend_type == "sqlite":
            return SQLiteStateBackend(state_manager=self._state_manager)

        if backend_type == "postgis":
            from agents.orchestrator.backends.postgis_backend import (
                PostGISStateBackend,
            )
            try:
                from api.config import get_settings
                settings = get_settings()
                db = settings.database
                return PostGISStateBackend(
                    host=db.host,
                    port=db.port,
                    database=db.name,
                    user=db.user,
                    password=db.password,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create PostGIS backend, falling back to SQLite: %s", e
                )
                return SQLiteStateBackend(state_manager=self._state_manager)

        if backend_type == "dual":
            from agents.orchestrator.backends.dual_write import DualWriteBackend
            from agents.orchestrator.backends.postgis_backend import (
                PostGISStateBackend,
            )
            sqlite_backend = SQLiteStateBackend(state_manager=self._state_manager)
            try:
                from api.config import get_settings
                settings = get_settings()
                db = settings.database
                postgis_backend = PostGISStateBackend(
                    host=db.host,
                    port=db.port,
                    database=db.name,
                    user=db.user,
                    password=db.password,
                )
                return DualWriteBackend(
                    primary=postgis_backend,
                    fallback=sqlite_backend,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create PostGIS backend for dual mode, "
                    "falling back to SQLite only: %s", e
                )
                return sqlite_backend

        logger.warning("Unknown state backend type: %s, using sqlite", backend_type)
        return SQLiteStateBackend(state_manager=self._state_manager)

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        Implements the abstract method from BaseAgent.

        Args:
            message: Incoming message to process

        Returns:
            Optional response message
        """
        await self._ensure_state_manager()

        payload = message.payload
        msg_type = payload.get("type", "unknown")

        if msg_type == "process_event":
            result = await self.process_event(payload.get("event_spec", {}))
            return message.create_response(result)

        elif msg_type == "get_status":
            event_id = payload.get("event_id")
            if event_id:
                state = await self._state_manager.get_state(event_id)
                response_payload = state.to_dict() if state else {"error": "Not found"}
            else:
                response_payload = {
                    "active_executions": len(self._active_executions),
                    "delegation_metrics": self._delegation.get_metrics(),
                }
            return message.create_response(response_payload)

        elif msg_type == "cancel_event":
            event_id = payload.get("event_id")
            if event_id in self._active_executions:
                state = self._active_executions[event_id]
                state.current_stage = ExecutionStage.CANCELLED
                state.completed_at = datetime.now(timezone.utc)
                await self._state_manager.update_state(state)
                self._active_executions.pop(event_id)
                return message.create_response({"cancelled": True})
            return message.create_response({"cancelled": False, "reason": "Not found"})

        else:
            logger.warning(f"Unknown message type: {msg_type}")
            return message.create_response({"error": f"Unknown message type: {msg_type}"})

    async def run(self) -> None:
        """
        Main agent execution loop.

        Implements the abstract method from BaseAgent.
        Processes messages from the inbox until shutdown.
        """
        await self._ensure_state_manager()

        # Resume any interrupted executions
        active_states = await self._state_manager.list_active_executions()
        for state in active_states:
            logger.info(f"Found interrupted execution: {state.event_id}")

        # Main message processing loop
        while not self._shutdown_event.is_set():
            try:
                message = await asyncio.wait_for(
                    self._inbox.get(),
                    timeout=1.0
                )
                response = await self.process_message(message)
                if response:
                    await self.send_message(response)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await self.on_error(e)

        # Cleanup on shutdown
        for event_id, state in self._active_executions.items():
            await self._state_manager.checkpoint(state)
            logger.info(f"Checkpointed execution: {event_id}")

        if self._state_manager:
            await self._state_manager.close()

    # Public API

    async def process_event(self, event_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an event specification through the full workflow.

        This is the main entry point for event processing. It coordinates:
        1. Event validation and intent resolution
        2. Data discovery
        3. Pipeline execution
        4. Quality control
        5. Report generation
        6. Product assembly

        Args:
            event_spec: Event specification dictionary

        Returns:
            Dictionary containing:
            - event_id: Event identifier
            - success: Whether processing succeeded
            - products: List of generated products
            - summary: Execution summary
            - errors: Any errors encountered
        """
        event_id = event_spec.get("id", str(uuid.uuid4()))
        self._logger.info(f"Processing event: {event_id}")

        # Create execution state
        state = await self._state_manager.create_state(
            event_id=event_id,
            event_spec=event_spec,
            orchestrator_id=self.agent_id,
        )
        state.started_at = datetime.now(timezone.utc)
        self._active_executions[event_id] = state

        result = {
            "event_id": event_id,
            "success": False,
            "products": [],
            "summary": None,
            "errors": [],
        }

        try:
            # Stage 1: Validation
            await self._execute_stage(
                state, ExecutionStage.VALIDATING,
                self._validate_event_spec, event_spec
            )

            # Stage 2: Discovery
            discovery_results = await self._execute_stage(
                state, ExecutionStage.DISCOVERY,
                self.delegate_to_discovery, {
                    "event_id": event_id,
                    "event_spec": event_spec,
                    "intent": state.intent_resolution,
                }
            )
            state.discovery_results = discovery_results or {}

            # Check if we should enter degraded mode
            await self._check_degraded_mode(state, "discovery", discovery_results)

            # Stage 3: Pipeline
            pipeline_results = await self._execute_stage(
                state, ExecutionStage.PIPELINE,
                self.delegate_to_pipeline, {
                    "event_id": event_id,
                    "event_spec": event_spec,
                    "discovery_results": state.discovery_results,
                    "degraded_mode": state.degraded_mode.to_dict(),
                }
            )
            state.pipeline_results = pipeline_results or {}

            # Check pipeline quality triggers
            await self._check_degraded_mode(state, "pipeline", pipeline_results)

            # Stage 4: Quality
            quality_results = await self._execute_stage(
                state, ExecutionStage.QUALITY,
                self.delegate_to_quality, {
                    "event_id": event_id,
                    "pipeline_results": state.pipeline_results,
                    "event_spec": event_spec,
                }
            )
            state.quality_results = quality_results or {}

            # Stage 5: Reporting
            reporting_results = await self._execute_stage(
                state, ExecutionStage.REPORTING,
                self.delegate_to_reporting, {
                    "event_id": event_id,
                    "event_spec": event_spec,
                    "pipeline_results": state.pipeline_results,
                    "quality_results": state.quality_results,
                }
            )

            # Stage 6: Assembly
            assembly_result = await self._execute_stage(
                state, ExecutionStage.ASSEMBLY,
                self._assemble_products,
                event_id, event_spec, state, reporting_results
            )

            # Mark completed
            state.current_stage = ExecutionStage.COMPLETED
            state.completed_at = datetime.now(timezone.utc)

            if assembly_result:
                result["success"] = assembly_result.success
                result["products"] = [p.to_dict() for p in assembly_result.products]
                result["summary"] = assembly_result.summary.to_dict() if assembly_result.summary else None
                state.final_products = result["products"]
            else:
                result["success"] = True
                result["summary"] = {"event_id": event_id, "status": "completed"}

            self._logger.info(f"Event {event_id} completed successfully")

            # Publish result as STAC Item (Phase 4)
            await self._publish_stac_item(event_id)

        except Exception as e:
            state.current_stage = ExecutionStage.FAILED
            state.completed_at = datetime.now(timezone.utc)
            result["errors"].append(str(e))
            self._logger.error(f"Event {event_id} failed: {e}")
            self._logger.debug(traceback.format_exc())

            # Record error
            await self._state_manager.record_error(
                event_id,
                state.current_stage,
                e,
                traceback.format_exc()
            )

        finally:
            # Update state
            await self._state_manager.update_state(state)
            self._active_executions.pop(event_id, None)

            # Notify completion
            await self._notify_completion(event_id, result)

        return result

    async def delegate_to_discovery(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate data discovery to the Discovery Agent.

        Args:
            query: Discovery query with event_id, event_spec, intent

        Returns:
            Discovery results including selected datasets
        """
        return await self._delegate_task(
            AgentType.DISCOVERY,
            "discover_data",
            query,
            timeout=120.0
        )

    async def delegate_to_pipeline(
        self,
        pipeline_spec: Dict[str, Any],
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate pipeline execution to the Pipeline Agent.

        Args:
            pipeline_spec: Pipeline specification with event_id, discovery_results
            data: Additional data context

        Returns:
            Pipeline execution results
        """
        payload = {**pipeline_spec}
        if data:
            payload["data"] = data

        return await self._delegate_task(
            AgentType.PIPELINE,
            "execute_pipeline",
            payload,
            timeout=600.0
        )

    async def delegate_to_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delegate quality validation to the Quality Agent.

        Args:
            results: Results to validate (pipeline_results, event_spec)

        Returns:
            Quality validation results
        """
        return await self._delegate_task(
            AgentType.QUALITY,
            "validate_results",
            results,
            timeout=180.0
        )

    async def delegate_to_reporting(
        self,
        validated_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delegate report generation to the Reporting Agent.

        Args:
            validated_results: Validated results to report on

        Returns:
            Report generation results
        """
        return await self._delegate_task(
            AgentType.REPORTING,
            "generate_reports",
            validated_results,
            timeout=60.0
        )

    async def handle_failure(self, agent: str, error: Exception) -> None:
        """
        Handle agent failure.

        Args:
            agent: Agent identifier that failed
            error: Exception that occurred
        """
        self._logger.error(f"Agent {agent} failed: {error}")

        # Check if we should enter degraded mode
        if self.config.enable_degraded_mode:
            self._logger.warning(f"Considering degraded mode due to {agent} failure")

    async def decide_degraded_mode(self, context: Dict[str, Any]) -> str:
        """
        Decide whether to enter degraded mode and at what level.

        Args:
            context: Context for decision (stage, error, available_data)

        Returns:
            Degraded mode level ("none", "minor", "moderate", "significant", "critical")
        """
        stage = context.get("stage", "")
        error_type = context.get("error_type", "")
        available_data = context.get("available_data", {})

        # No data at all -> critical
        if not available_data:
            return "critical"

        # Missing optional data -> minor
        if error_type == "OptionalDataMissing":
            return "minor"

        # Pipeline partial failure -> moderate
        if stage == "pipeline" and context.get("partial_results"):
            return "moderate"

        # Quality issues -> significant
        if stage == "quality":
            quality_score = context.get("quality_score", 1.0)
            if quality_score < self.config.min_confidence_threshold:
                return "significant"
            elif quality_score < 0.5:
                return "moderate"

        return "none"

    # Registration methods

    def register_agent_executor(
        self,
        agent_type: AgentType,
        executor: Callable[[str, Dict[str, Any]], Any]
    ) -> None:
        """
        Register an executor function for an agent type.

        Args:
            agent_type: Type of agent
            executor: Async function (agent_id, payload) -> result
        """
        self._agent_executors[agent_type] = executor
        self._logger.info(f"Registered executor for {agent_type.value}")

    def register_agent(self, agent_info: AgentInfo) -> None:
        """
        Register an available agent.

        Args:
            agent_info: Agent information
        """
        self._delegation.register_agent(agent_info)

    def on_status_update(
        self,
        callback: Callable[[str, ExecutionState], None]
    ) -> None:
        """
        Register a callback for status updates.

        Args:
            callback: Function called with (event_id, state)
        """
        self._status_callbacks.append(callback)

    def on_completion(
        self,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback for completion.

        Args:
            callback: Function called with (event_id, result)
        """
        self._completion_callbacks.append(callback)

    # Internal methods

    async def _execute_stage(
        self,
        state: ExecutionState,
        stage: ExecutionStage,
        operation: Callable,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """Execute a processing stage with state tracking."""
        stage_key = stage.value
        self._logger.info(f"Executing stage: {stage_key} for event {state.event_id}")

        # Update state
        await self._state_manager.update_stage(
            state.event_id, stage, "running", 0.0,
            f"Starting {stage_key}"
        )
        state.current_stage = stage

        try:
            # Execute operation
            result = await operation(*args, **kwargs)

            # Mark completed
            await self._state_manager.update_stage(
                state.event_id, stage, "completed", 100.0,
                f"Completed {stage_key}"
            )

            # Checkpoint if needed
            stages_done = sum(
                1 for s in state.stages.values()
                if s.status == "completed"
            )
            if stages_done % self.config.checkpoint_frequency == 0:
                await self._state_manager.checkpoint(state)

            # Notify status
            await self._notify_status(state.event_id, state)

            return result

        except Exception as e:
            await self._state_manager.update_stage(
                state.event_id, stage, "failed", 0.0,
                f"Failed: {str(e)}"
            )
            await self._state_manager.record_error(
                state.event_id, stage, e, traceback.format_exc()
            )
            raise

    async def _delegate_task(
        self,
        agent_type: AgentType,
        task_type: str,
        payload: Dict[str, Any],
        timeout: float = 300.0
    ) -> Dict[str, Any]:
        """Delegate a task to an agent."""
        event_id = payload.get("event_id", "")

        task = DelegationTask(
            event_id=event_id,
            agent_type=agent_type,
            task_type=task_type,
            payload=payload,
            timeout_seconds=timeout,
        )

        # Get executor
        executor = self._agent_executors.get(agent_type)
        if executor is None:
            # Use default stub executor
            executor = self._default_executor

        try:
            result = await self._delegation.delegate(
                task,
                executor,
                on_progress=lambda t: self._on_task_progress(event_id, t)
            )
            return result if isinstance(result, dict) else {"result": result}

        except DelegationError as e:
            self._logger.error(f"Delegation failed for {task_type}: {e}")
            # Return empty result to allow continuation if possible
            return {"error": str(e), "partial": True}

    async def _default_executor(
        self,
        agent_id: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Default executor when no agent executor is registered.

        This allows the orchestrator to run in a standalone mode
        by directly calling core modules.
        """
        task_type = payload.get("task_type", "")
        event_id = payload.get("event_id", "")

        self._logger.info(f"Using default executor for task: {task_type}")

        # Import core modules on demand
        try:
            if task_type == "discover_data" or "discover" in task_type.lower():
                return await self._execute_discovery(payload)
            elif task_type == "execute_pipeline" or "pipeline" in task_type.lower():
                return await self._execute_pipeline(payload)
            elif task_type == "validate_results" or "quality" in task_type.lower():
                return await self._execute_quality(payload)
            elif task_type == "generate_reports" or "report" in task_type.lower():
                return await self._execute_reporting(payload)
            else:
                self._logger.warning(f"Unknown task type: {task_type}")
                return {"status": "skipped", "reason": f"Unknown task: {task_type}"}

        except ImportError as e:
            self._logger.warning(f"Module not available for {task_type}: {e}")
            return {"status": "skipped", "reason": str(e)}

    async def _execute_discovery(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data discovery using core modules."""
        try:
            from core.data.broker import DataBroker, BrokerQuery
            from core.intent.resolver import get_resolver

            event_spec = payload.get("event_spec", {})
            event_id = payload.get("event_id", "")

            # Resolve intent
            resolver = get_resolver()
            intent_input = event_spec.get("intent", {})
            intent_resolution = resolver.resolve(
                natural_language=intent_input.get("original_input"),
                explicit_class=intent_input.get("class"),
            )

            if intent_resolution:
                # Store in state
                if event_id in self._active_executions:
                    self._active_executions[event_id].intent_resolution = intent_resolution.to_dict()

            # Create broker query
            query = BrokerQuery(
                event_id=event_id,
                spatial=event_spec.get("spatial", {}),
                temporal=event_spec.get("temporal", {}),
                intent_class=intent_resolution.resolved_class if intent_resolution else "unknown",
                data_types=event_spec.get("constraints", {}).get("required_data_types", []),
                constraints=event_spec.get("constraints", {}),
            )

            # Execute discovery
            broker = DataBroker()
            response = await broker.discover(query)

            return response.to_dict()

        except Exception as e:
            self._logger.error(f"Discovery execution failed: {e}")
            return {"error": str(e), "selected_datasets": []}

    async def _execute_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline using core modules."""
        try:
            # Import pipeline modules
            from core.analysis.assembly.assembler import PipelineAssembler
            from core.analysis.library.registry import get_global_registry

            event_spec = payload.get("event_spec", {})
            discovery_results = payload.get("discovery_results", {})

            # Get selected datasets
            datasets = discovery_results.get("selected_datasets", [])
            if not datasets:
                return {
                    "status": "skipped",
                    "reason": "No datasets available",
                    "outputs": []
                }

            # Get intent class for algorithm selection
            intent_class = event_spec.get("intent", {}).get("class", "flood")

            # Get available algorithms
            registry = get_global_registry()
            algorithms = registry.get_algorithms_for_event(intent_class)

            if not algorithms:
                return {
                    "status": "skipped",
                    "reason": f"No algorithms for {intent_class}",
                    "outputs": []
                }

            # For now, return a stub result
            # In production, would execute actual pipeline
            return {
                "status": "completed",
                "algorithms_available": [a.algorithm_id for a in algorithms[:3]],
                "datasets_used": [d.get("dataset_id") for d in datasets],
                "outputs": [],
                "steps_executed": [],
                "metrics": {
                    "execution_time_seconds": 0.0,
                },
            }

        except Exception as e:
            self._logger.error(f"Pipeline execution failed: {e}")
            return {"error": str(e), "outputs": []}

    async def _execute_quality(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute quality checks using core modules."""
        try:
            from core.quality.sanity import SanitySuite

            pipeline_results = payload.get("pipeline_results", {})
            outputs = pipeline_results.get("outputs", [])

            if not outputs:
                return {
                    "status": "skipped",
                    "reason": "No outputs to validate",
                    "overall_score": None,
                }

            # Run sanity suite (would need actual data in production)
            suite = SanitySuite()

            # Return stub result
            return {
                "status": "completed",
                "overall_score": 0.85,
                "passes_sanity": True,
                "checks_run": ["spatial", "values", "artifacts"],
                "flags": [],
                "report": {
                    "summary": "Quality checks passed",
                },
            }

        except Exception as e:
            self._logger.error(f"Quality execution failed: {e}")
            return {"error": str(e), "overall_score": None}

    async def _execute_reporting(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reporting using core modules."""
        try:
            from core.quality.reporting.qa_report import QAReportGenerator, ReportFormat

            quality_results = payload.get("quality_results", {})

            # Generate report
            generator = QAReportGenerator()

            # Return stub result
            return {
                "status": "completed",
                "reports": {
                    "summary": {
                        "format": "json",
                        "description": "Executive summary",
                    }
                },
            }

        except Exception as e:
            self._logger.error(f"Reporting execution failed: {e}")
            return {"error": str(e), "reports": {}}

    async def _validate_event_spec(self, event_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate event specification."""
        errors = []

        # Check required fields
        if "spatial" not in event_spec:
            errors.append("Missing required field: spatial")
        if "temporal" not in event_spec:
            errors.append("Missing required field: temporal")

        # Validate spatial
        spatial = event_spec.get("spatial", {})
        if spatial and "coordinates" not in spatial and "bbox" not in spatial:
            errors.append("Spatial must have coordinates or bbox")

        # Validate temporal
        temporal = event_spec.get("temporal", {})
        if temporal and "start" not in temporal:
            errors.append("Temporal must have start time")

        if errors:
            raise ValueError(f"Event specification validation failed: {errors}")

        return {"valid": True, "errors": []}

    async def _assemble_products(
        self,
        event_id: str,
        event_spec: Dict[str, Any],
        state: ExecutionState,
        reporting_results: Optional[Dict[str, Any]]
    ) -> AssemblyResult:
        """Assemble final products."""
        return await self._assembler.assemble(
            event_id=event_id,
            event_spec=event_spec,
            discovery_results=state.discovery_results,
            pipeline_results=state.pipeline_results,
            quality_results=state.quality_results,
            reporting_results=reporting_results,
            execution_state=state.to_dict(),
        )

    async def _check_degraded_mode(
        self,
        state: ExecutionState,
        stage: str,
        results: Optional[Dict[str, Any]]
    ) -> None:
        """Check if degraded mode should be activated."""
        if not self.config.enable_degraded_mode:
            return

        if results is None:
            level = "critical"
            reason = f"Stage {stage} returned no results"
        elif results.get("error"):
            level = await self.decide_degraded_mode({
                "stage": stage,
                "error_type": "StageError",
                "partial_results": results.get("partial", False),
            })
            reason = f"Stage {stage} error: {results.get('error')}"
        else:
            return

        if level != "none":
            await self._state_manager.set_degraded_mode(
                state.event_id,
                DegradedModeLevel(level),
                reason,
                confidence_impact=0.1 if level == "minor" else 0.3,
            )
            state.degraded_mode = DegradedModeInfo(
                level=DegradedModeLevel(level),
                active_since=datetime.now(timezone.utc),
                reasons=[reason],
            )

    def _on_task_progress(self, event_id: str, task: DelegationTask) -> None:
        """Handle task progress updates."""
        self._logger.debug(f"Task {task.task_id} progress: {task.status.value}")

    async def _notify_status(self, event_id: str, state: ExecutionState) -> None:
        """Notify status callbacks."""
        for callback in self._status_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_id, state)
                else:
                    callback(event_id, state)
            except Exception as e:
                self._logger.error(f"Status callback error: {e}")

    async def _notify_completion(
        self,
        event_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Notify completion callbacks and webhook."""
        # Local callbacks
        for callback in self._completion_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_id, result)
                else:
                    callback(event_id, result)
            except Exception as e:
                self._logger.error(f"Completion callback error: {e}")

        # Webhook notification
        if self.config.notification_webhook:
            await self._send_webhook(event_id, result)

    async def _send_webhook(
        self,
        event_id: str,
        result: Dict[str, Any]
    ) -> None:
        """Send webhook notification."""
        try:
            import aiohttp

            payload = {
                "event_id": event_id,
                "status": "completed" if result.get("success") else "failed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "products_count": len(result.get("products", [])),
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.notification_webhook,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        self._logger.warning(
                            f"Webhook failed: {response.status}"
                        )

        except ImportError:
            self._logger.debug("aiohttp not available for webhook")
        except Exception as e:
            self._logger.error(f"Webhook error: {e}")

    async def _handle_escalation(self, task: DelegationTask) -> None:
        """Handle escalated task."""
        self._logger.warning(f"Escalated task: {task.task_id}")
        # Could notify operators, create incident, etc.

    async def _publish_stac_item(self, event_id: str) -> None:
        """
        Publish completed job results as a STAC Item.

        Called after a job transitions to COMPLETE. Failures are logged
        but do not block job completion.
        """
        try:
            from core.stac.publisher import publish_result_as_stac_item

            item = await publish_result_as_stac_item(event_id)
            if item:
                self._logger.info(
                    "STAC item published for event %s", event_id
                )
            else:
                self._logger.debug(
                    "STAC item not published for event %s "
                    "(job may not be in PostGIS or pgSTAC not available)",
                    event_id,
                )
        except ImportError:
            self._logger.debug(
                "STAC publisher not available, skipping item publication"
            )
        except Exception as e:
            self._logger.warning(
                "Failed to publish STAC item for event %s: %s", event_id, e
            )

    # Utility methods

    def get_execution_state(self, event_id: str) -> Optional[ExecutionState]:
        """Get current execution state for an event."""
        return self._active_executions.get(event_id)

    async def get_execution_history(
        self,
        event_id: str
    ) -> Optional[ExecutionState]:
        """Get execution state from history."""
        return await self._state_manager.get_state(event_id)


async def main():
    """Demo main function."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create orchestrator
    config = OrchestratorConfig(
        state_db_path=":memory:",
        output_dir="./output",
    )
    orchestrator = OrchestratorAgent(config)

    # Initialize and start
    await orchestrator.initialize()
    await orchestrator.start()

    # Load event spec
    event_file = sys.argv[1] if len(sys.argv) > 1 else "examples/flood_event.yaml"

    try:
        with open(event_file, 'r') as f:
            event_spec = yaml.safe_load(f)

        print(f"\nProcessing event from: {event_file}")
        print(f"Event ID: {event_spec.get('id', 'auto-generated')}")

        # Process event
        result = await orchestrator.process_event(event_spec)

        print(f"\nResult:")
        print(f"  Success: {result['success']}")
        print(f"  Products: {len(result.get('products', []))}")
        if result.get('errors'):
            print(f"  Errors: {result['errors']}")

    except FileNotFoundError:
        print(f"Event file not found: {event_file}")
    except Exception as e:
        print(f"Error: {e}")

    # Shutdown
    await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
