"""
Task Delegation Logic for Orchestrator Agent.

Provides intelligent routing of tasks to specialized agents:
- DelegationStrategy class for routing tasks to appropriate agents
- Load balancing across multiple agents of the same type
- Timeout handling with configurable limits
- Retry logic with exponential backoff and escalation

The delegation system ensures:
- Tasks reach the right agent based on type and capabilities
- Load is distributed efficiently across available agents
- Failures are handled gracefully with retries and fallbacks
- Escalation paths for critical failures
"""

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of specialized agents."""
    DISCOVERY = "discovery"
    PIPELINE = "pipeline"
    QUALITY = "quality"
    REPORTING = "reporting"


class DelegationStatus(Enum):
    """Status of a delegation request."""
    PENDING = "pending"
    DISPATCHED = "dispatched"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    ESCALATED = "escalated"
    CANCELLED = "cancelled"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies for agent selection."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RANDOM = "random"
    AFFINITY = "affinity"  # Prefer same agent for related tasks


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay_seconds: Initial delay before first retry
        max_delay_seconds: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delay (0-1)
        retry_on_timeout: Whether to retry on timeout
        non_retriable_errors: Error types that should not be retried
    """
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retry_on_timeout: bool = True
    non_retriable_errors: Set[str] = field(default_factory=lambda: {
        "InvalidInputError",
        "ValidationError",
        "ConfigurationError",
    })

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt."""
        delay = self.initial_delay_seconds * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay_seconds)

        # Add jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0.1, delay)  # Minimum 100ms


@dataclass
class TimeoutPolicy:
    """
    Configuration for timeout behavior.

    Attributes:
        default_timeout_seconds: Default timeout for all tasks
        agent_timeouts: Timeout overrides per agent type
        task_timeouts: Timeout overrides per task type
    """
    default_timeout_seconds: float = 300.0  # 5 minutes
    agent_timeouts: Dict[str, float] = field(default_factory=lambda: {
        "discovery": 120.0,   # 2 minutes
        "pipeline": 600.0,    # 10 minutes
        "quality": 180.0,     # 3 minutes
        "reporting": 60.0,    # 1 minute
    })
    task_timeouts: Dict[str, float] = field(default_factory=dict)

    def get_timeout(
        self,
        agent_type: Optional[AgentType] = None,
        task_type: Optional[str] = None
    ) -> float:
        """Get appropriate timeout for a task."""
        # Task-specific timeout takes priority
        if task_type and task_type in self.task_timeouts:
            return self.task_timeouts[task_type]

        # Then agent type timeout
        if agent_type and agent_type.value in self.agent_timeouts:
            return self.agent_timeouts[agent_type.value]

        return self.default_timeout_seconds


@dataclass
class DelegationTask:
    """
    A task to be delegated to an agent.

    Attributes:
        task_id: Unique task identifier
        event_id: Associated event ID
        agent_type: Target agent type
        task_type: Type of task
        payload: Task payload data
        priority: Task priority (higher = more important)
        timeout_seconds: Task timeout
        retry_policy: Retry configuration
        created_at: Task creation time
        dispatched_at: When task was dispatched
        completed_at: When task completed
        status: Current task status
        assigned_agent: ID of assigned agent
        attempt_count: Number of attempts made
        result: Task result (if completed)
        error: Error info (if failed)
        metadata: Additional task metadata
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    agent_type: AgentType = AgentType.DISCOVERY
    task_type: str = "generic"
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    timeout_seconds: float = 300.0
    retry_policy: Optional[RetryPolicy] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dispatched_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: DelegationStatus = DelegationStatus.PENDING
    assigned_agent: Optional[str] = None
    attempt_count: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "event_id": self.event_id,
            "agent_type": self.agent_type.value,
            "task_type": self.task_type,
            "payload": self.payload,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at.isoformat(),
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "attempt_count": self.attempt_count,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task execution duration."""
        if self.dispatched_at is None:
            return None
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.dispatched_at).total_seconds()


@dataclass
class AgentInfo:
    """
    Information about an available agent.

    Attributes:
        agent_id: Agent identifier
        agent_type: Type of agent
        status: Current agent status
        current_load: Number of active tasks
        max_capacity: Maximum concurrent tasks
        capabilities: Agent capabilities/features
        last_heartbeat: Time of last heartbeat
        metrics: Performance metrics
    """
    agent_id: str
    agent_type: AgentType
    status: str = "ready"
    current_load: int = 0
    max_capacity: int = 10
    capabilities: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def available_capacity(self) -> int:
        """Get remaining capacity."""
        return max(0, self.max_capacity - self.current_load)

    @property
    def load_percent(self) -> float:
        """Get load percentage."""
        if self.max_capacity == 0:
            return 100.0
        return (self.current_load / self.max_capacity) * 100.0

    @property
    def is_available(self) -> bool:
        """Check if agent can accept tasks."""
        return (
            self.status == "ready" and
            self.current_load < self.max_capacity
        )


class DelegationStrategy:
    """
    Strategy for delegating tasks to agents.

    Handles agent selection, load balancing, retries, and escalation.

    Example:
        strategy = DelegationStrategy(timeout_policy=TimeoutPolicy())

        # Register available agents
        strategy.register_agent(AgentInfo(
            agent_id="discovery_001",
            agent_type=AgentType.DISCOVERY,
        ))

        # Delegate a task
        task = DelegationTask(
            event_id="event_001",
            agent_type=AgentType.DISCOVERY,
            task_type="discover_data",
            payload={"query": {...}},
        )

        result = await strategy.delegate(task, execute_fn)
    """

    def __init__(
        self,
        load_balancing: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        default_retry_policy: Optional[RetryPolicy] = None,
        timeout_policy: Optional[TimeoutPolicy] = None,
    ):
        """
        Initialize delegation strategy.

        Args:
            load_balancing: Load balancing strategy
            default_retry_policy: Default retry configuration
            timeout_policy: Timeout configuration
        """
        self.load_balancing = load_balancing
        self.default_retry_policy = default_retry_policy or RetryPolicy()
        self.timeout_policy = timeout_policy or TimeoutPolicy()

        # Agent registry
        self._agents: Dict[str, AgentInfo] = {}
        self._agents_by_type: Dict[AgentType, List[str]] = {}

        # Round-robin state
        self._round_robin_indices: Dict[AgentType, int] = {}

        # Affinity mapping (event_id -> agent_id)
        self._affinity_map: Dict[str, str] = {}

        # Task tracking
        self._pending_tasks: Dict[str, DelegationTask] = {}
        self._task_history: List[Dict[str, Any]] = []

        # Escalation handlers
        self._escalation_handlers: List[Callable] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "tasks_delegated": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "tasks_escalated": 0,
            "total_retries": 0,
            "avg_duration_seconds": 0.0,
        }

    # Agent management

    def register_agent(self, agent_info: AgentInfo) -> None:
        """
        Register an agent as available for tasks.

        Args:
            agent_info: Agent information
        """
        self._agents[agent_info.agent_id] = agent_info

        if agent_info.agent_type not in self._agents_by_type:
            self._agents_by_type[agent_info.agent_type] = []

        if agent_info.agent_id not in self._agents_by_type[agent_info.agent_type]:
            self._agents_by_type[agent_info.agent_type].append(agent_info.agent_id)

        logger.info(f"Registered agent: {agent_info.agent_id} ({agent_info.agent_type.value})")

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agents:
            agent = self._agents.pop(agent_id)
            if agent.agent_type in self._agents_by_type:
                agent_ids = self._agents_by_type[agent.agent_type]
                if agent_id in agent_ids:
                    agent_ids.remove(agent_id)
            logger.info(f"Unregistered agent: {agent_id}")

    def update_agent_load(self, agent_id: str, load: int) -> None:
        """
        Update agent's current load.

        Args:
            agent_id: Agent identifier
            load: Current task count
        """
        if agent_id in self._agents:
            self._agents[agent_id].current_load = load

    def update_agent_status(self, agent_id: str, status: str) -> None:
        """
        Update agent status.

        Args:
            agent_id: Agent identifier
            status: New status
        """
        if agent_id in self._agents:
            self._agents[agent_id].status = status

    def update_agent_heartbeat(self, agent_id: str) -> None:
        """
        Update agent heartbeat timestamp.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = datetime.now(timezone.utc)

    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """Get agent information."""
        return self._agents.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentInfo]:
        """Get all agents of a specific type."""
        agent_ids = self._agents_by_type.get(agent_type, [])
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]

    # Agent selection

    def select_agent(
        self,
        agent_type: AgentType,
        event_id: Optional[str] = None,
        required_capabilities: Optional[Set[str]] = None
    ) -> Optional[AgentInfo]:
        """
        Select an agent for a task.

        Args:
            agent_type: Type of agent needed
            event_id: Event ID for affinity-based selection
            required_capabilities: Required agent capabilities

        Returns:
            Selected agent or None if none available
        """
        candidates = self.get_agents_by_type(agent_type)

        # Filter by status and capacity
        available = [a for a in candidates if a.is_available]

        # Filter by capabilities
        if required_capabilities:
            available = [
                a for a in available
                if required_capabilities.issubset(a.capabilities)
            ]

        if not available:
            return None

        # Apply selection strategy
        if self.load_balancing == LoadBalancingStrategy.AFFINITY and event_id:
            # Check affinity map
            affinity_agent_id = self._affinity_map.get(event_id)
            if affinity_agent_id:
                affinity_agent = self._agents.get(affinity_agent_id)
                if affinity_agent and affinity_agent.is_available:
                    return affinity_agent

        if self.load_balancing == LoadBalancingStrategy.LEAST_LOADED:
            return min(available, key=lambda a: a.current_load)

        elif self.load_balancing == LoadBalancingStrategy.ROUND_ROBIN:
            idx = self._round_robin_indices.get(agent_type, 0)
            selected = available[idx % len(available)]
            self._round_robin_indices[agent_type] = (idx + 1) % len(available)
            return selected

        elif self.load_balancing == LoadBalancingStrategy.RANDOM:
            return random.choice(available)

        # Default to least loaded
        return min(available, key=lambda a: a.current_load)

    # Task delegation

    async def delegate(
        self,
        task: DelegationTask,
        execute_fn: Callable[[str, Dict[str, Any]], Any],
        on_progress: Optional[Callable[[DelegationTask], None]] = None
    ) -> Dict[str, Any]:
        """
        Delegate a task to an appropriate agent.

        Args:
            task: Task to delegate
            execute_fn: Function to execute task (agent_id, payload) -> result
            on_progress: Optional callback for progress updates

        Returns:
            Task result

        Raises:
            DelegationError: If task cannot be completed after retries
        """
        retry_policy = task.retry_policy or self.default_retry_policy
        timeout = task.timeout_seconds or self.timeout_policy.get_timeout(
            task.agent_type,
            task.task_type
        )

        task.status = DelegationStatus.PENDING
        self._pending_tasks[task.task_id] = task

        async with self._lock:
            self._metrics["tasks_delegated"] += 1

        last_error: Optional[Exception] = None

        for attempt in range(retry_policy.max_retries + 1):
            task.attempt_count = attempt + 1

            # Select an agent
            agent = self.select_agent(
                task.agent_type,
                event_id=task.event_id
            )

            if agent is None:
                logger.warning(f"No available agent for task {task.task_id}")
                if attempt < retry_policy.max_retries:
                    delay = retry_policy.calculate_delay(attempt)
                    task.status = DelegationStatus.RETRYING
                    if on_progress:
                        on_progress(task)
                    await asyncio.sleep(delay)
                    continue
                else:
                    task.status = DelegationStatus.FAILED
                    task.error = {
                        "type": "NoAgentAvailable",
                        "message": f"No available {task.agent_type.value} agent",
                    }
                    await self._handle_failure(task)
                    raise DelegationError(f"No available agent for {task.agent_type.value}")

            task.assigned_agent = agent.agent_id
            task.dispatched_at = datetime.now(timezone.utc)
            task.status = DelegationStatus.DISPATCHED

            # Update affinity
            if task.event_id:
                self._affinity_map[task.event_id] = agent.agent_id

            # Update agent load
            agent.current_load += 1

            if on_progress:
                on_progress(task)

            try:
                task.status = DelegationStatus.IN_PROGRESS

                # Execute with timeout
                result = await asyncio.wait_for(
                    execute_fn(agent.agent_id, task.payload),
                    timeout=timeout
                )

                task.status = DelegationStatus.COMPLETED
                task.completed_at = datetime.now(timezone.utc)
                task.result = result

                async with self._lock:
                    self._metrics["tasks_completed"] += 1
                    self._update_avg_duration(task)

                if on_progress:
                    on_progress(task)

                # Record in history
                self._task_history.append(task.to_dict())

                # Cleanup
                self._pending_tasks.pop(task.task_id, None)

                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Task {task.task_id} timed out after {timeout}s")
                task.status = DelegationStatus.TIMEOUT
                task.error = {
                    "type": "TimeoutError",
                    "message": str(last_error),
                    "timeout_seconds": timeout,
                }
                logger.warning(f"Task {task.task_id} timed out (attempt {attempt + 1})")

                if not retry_policy.retry_on_timeout:
                    break

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                task.error = {
                    "type": error_type,
                    "message": str(e),
                }
                logger.error(f"Task {task.task_id} failed: {e}")

                # Check if error is non-retriable
                if error_type in retry_policy.non_retriable_errors:
                    task.status = DelegationStatus.FAILED
                    break

            finally:
                # Update agent load
                agent.current_load = max(0, agent.current_load - 1)

            # Retry if attempts remain
            if attempt < retry_policy.max_retries:
                async with self._lock:
                    self._metrics["total_retries"] += 1
                    self._metrics["tasks_retried"] += 1

                delay = retry_policy.calculate_delay(attempt)
                task.status = DelegationStatus.RETRYING

                if on_progress:
                    on_progress(task)

                logger.info(f"Retrying task {task.task_id} in {delay:.1f}s (attempt {attempt + 2})")
                await asyncio.sleep(delay)

        # All retries exhausted
        task.status = DelegationStatus.FAILED
        task.completed_at = datetime.now(timezone.utc)

        await self._handle_failure(task)

        error_msg = str(last_error) if last_error else "Unknown error"
        raise DelegationError(f"Task {task.task_id} failed after {task.attempt_count} attempts: {error_msg}")

    async def _handle_failure(self, task: DelegationTask) -> None:
        """Handle task failure and potential escalation."""
        async with self._lock:
            self._metrics["tasks_failed"] += 1

        # Record in history
        self._task_history.append(task.to_dict())

        # Cleanup
        self._pending_tasks.pop(task.task_id, None)

        # Check if escalation is needed
        if task.priority >= 8:  # High priority tasks escalate
            await self._escalate(task)

    async def _escalate(self, task: DelegationTask) -> None:
        """Escalate a failed task."""
        task.status = DelegationStatus.ESCALATED

        async with self._lock:
            self._metrics["tasks_escalated"] += 1

        logger.warning(f"Escalating failed task: {task.task_id}")

        # Call escalation handlers
        for handler in self._escalation_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(task)
                else:
                    handler(task)
            except Exception as e:
                logger.error(f"Escalation handler error: {e}")

    def _update_avg_duration(self, task: DelegationTask) -> None:
        """Update average task duration metric."""
        duration = task.duration_seconds
        if duration is not None:
            current_avg = self._metrics["avg_duration_seconds"]
            completed = self._metrics["tasks_completed"]
            # Running average
            self._metrics["avg_duration_seconds"] = (
                (current_avg * (completed - 1) + duration) / completed
            )

    # Escalation management

    def add_escalation_handler(self, handler: Callable[[DelegationTask], Any]) -> None:
        """
        Add an escalation handler.

        Args:
            handler: Callable that handles escalated tasks
        """
        self._escalation_handlers.append(handler)

    def remove_escalation_handler(self, handler: Callable) -> None:
        """Remove an escalation handler."""
        if handler in self._escalation_handlers:
            self._escalation_handlers.remove(handler)

    # Task management

    def get_pending_tasks(self) -> List[DelegationTask]:
        """Get all pending tasks."""
        return list(self._pending_tasks.values())

    def get_task(self, task_id: str) -> Optional[DelegationTask]:
        """Get task by ID."""
        return self._pending_tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending task.

        Args:
            task_id: Task identifier

        Returns:
            True if cancelled, False if not found or already completed
        """
        task = self._pending_tasks.get(task_id)
        if task and task.status in (DelegationStatus.PENDING, DelegationStatus.RETRYING):
            task.status = DelegationStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            self._pending_tasks.pop(task_id, None)
            self._task_history.append(task.to_dict())
            logger.info(f"Cancelled task: {task_id}")
            return True
        return False

    # Metrics and monitoring

    def get_metrics(self) -> Dict[str, Any]:
        """Get delegation metrics."""
        return {
            **self._metrics,
            "pending_tasks": len(self._pending_tasks),
            "registered_agents": len(self._agents),
            "agents_by_type": {
                agent_type.value: len(agent_ids)
                for agent_type, agent_ids in self._agents_by_type.items()
            },
        }

    def get_task_history(
        self,
        limit: int = 100,
        event_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get task history.

        Args:
            limit: Maximum number of records
            event_id: Filter by event ID

        Returns:
            List of task records
        """
        history = self._task_history

        if event_id:
            history = [t for t in history if t.get("event_id") == event_id]

        return history[-limit:]


class DelegationError(Exception):
    """Error during task delegation."""
    pass


class TaskRouter:
    """
    Routes tasks to appropriate handlers based on type.

    Provides a simple way to register handlers for different task types
    and route incoming tasks accordingly.

    Example:
        router = TaskRouter()

        @router.handler("discover_data")
        async def handle_discovery(payload):
            # Process discovery task
            return results

        result = await router.route("discover_data", {"query": ...})
    """

    def __init__(self):
        """Initialize task router."""
        self._handlers: Dict[str, Callable] = {}
        self._default_handler: Optional[Callable] = None

    def handler(self, task_type: str) -> Callable:
        """
        Decorator for registering a task handler.

        Args:
            task_type: Type of task to handle

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            self._handlers[task_type] = func
            return func
        return decorator

    def register(self, task_type: str, handler: Callable) -> None:
        """
        Register a handler for a task type.

        Args:
            task_type: Type of task
            handler: Handler function
        """
        self._handlers[task_type] = handler

    def set_default_handler(self, handler: Callable) -> None:
        """
        Set the default handler for unregistered task types.

        Args:
            handler: Default handler function
        """
        self._default_handler = handler

    async def route(
        self,
        task_type: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Route a task to its handler.

        Args:
            task_type: Type of task
            payload: Task payload

        Returns:
            Handler result

        Raises:
            ValueError: If no handler found
        """
        handler = self._handlers.get(task_type, self._default_handler)

        if handler is None:
            raise ValueError(f"No handler registered for task type: {task_type}")

        if asyncio.iscoroutinefunction(handler):
            return await handler(payload)
        return handler(payload)

    def get_registered_types(self) -> List[str]:
        """Get list of registered task types."""
        return list(self._handlers.keys())
