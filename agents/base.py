"""
Agent Foundation Module for Multiverse Dive.

Provides the base infrastructure for all agents in the orchestration system:
- AgentState enum for lifecycle management
- AgentMessage dataclass for inter-agent communication
- BaseAgent abstract class with lifecycle, messaging, and state persistence
- AgentRegistry for agent registration and message routing
- MessageBus for async message passing with priority queue
- RetryPolicy for configurable error handling

All specialized agents (orchestrator, discovery, pipeline, quality, reporting)
inherit from BaseAgent and use AgentRegistry/MessageBus for coordination.
"""

import asyncio
import functools
import json
import logging
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class AgentState(Enum):
    """
    Lifecycle states for agents.

    State transitions:
        pending -> running -> completed
        pending -> running -> failed
        pending -> running -> paused -> running -> completed
        pending -> cancelled
        running -> cancelled
    """
    PENDING = "pending"          # Created but not started
    RUNNING = "running"          # Actively processing
    PAUSED = "paused"            # Temporarily suspended
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"            # Terminated due to error
    CANCELLED = "cancelled"      # Terminated by request


class AgentType(Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"   # Main coordination agent
    DISCOVERY = "discovery"          # Data discovery agent
    PIPELINE = "pipeline"            # Pipeline execution agent
    QUALITY = "quality"              # Quality control agent
    REPORTING = "reporting"          # Report generation agent


class MessageType(Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"              # Request for action
    RESPONSE = "response"            # Response to a request
    EVENT = "event"                  # Event notification
    ERROR = "error"                  # Error notification
    STATUS_UPDATE = "status_update"  # Status/progress update


class MessagePriority(Enum):
    """Priority levels for messages."""
    CRITICAL = 0    # Highest priority - immediate handling
    HIGH = 1        # High priority - handle soon
    NORMAL = 2      # Normal priority - standard queue
    LOW = 3         # Low priority - process when available

    def __lt__(self, other: "MessagePriority") -> bool:
        """Enable comparison for priority queue ordering."""
        return self.value < other.value


class BackoffStrategy(Enum):
    """Backoff strategies for retry policies."""
    NONE = "none"                          # No backoff (immediate retry)
    LINEAR = "linear"                      # Linear delay increase
    EXPONENTIAL = "exponential"            # Exponential delay increase
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"  # Exponential + random


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MessageContext:
    """
    Contextual information attached to messages.

    Provides shared state and tracking information that flows
    through the agent system during event processing.

    Attributes:
        event_id: The event being processed
        execution_id: Current execution/run identifier
        degraded_mode: Whether operating in degraded mode
        cumulative_confidence: Running confidence score
        trace_id: Distributed trace identifier
        parent_span_id: Parent span for tracing
        metadata: Additional context data
    """
    event_id: Optional[str] = None
    execution_id: Optional[str] = None
    degraded_mode: bool = False
    cumulative_confidence: float = 1.0
    trace_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "execution_id": self.execution_id,
            "degraded_mode": self.degraded_mode,
            "cumulative_confidence": self.cumulative_confidence,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageContext":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id"),
            execution_id=data.get("execution_id"),
            degraded_mode=data.get("degraded_mode", False),
            cumulative_confidence=data.get("cumulative_confidence", 1.0),
            trace_id=data.get("trace_id"),
            parent_span_id=data.get("parent_span_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentMessage:
    """
    Message for inter-agent communication.

    Represents a single message passed between agents, with full
    tracking, routing, and payload information.

    Attributes:
        message_id: Unique message identifier
        correlation_id: Links request-response pairs
        timestamp: When the message was created
        from_agent: Source agent type
        to_agent: Destination agent type
        message_type: Type of message (request, response, etc.)
        priority: Message priority level
        payload: Message data/content
        context: Execution context
        ttl_seconds: Time-to-live in seconds (None = no expiry)
        retry_count: Number of delivery attempts
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    from_agent: AgentType = AgentType.ORCHESTRATOR
    to_agent: AgentType = AgentType.ORCHESTRATOR
    message_type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    context: MessageContext = field(default_factory=MessageContext)
    ttl_seconds: Optional[int] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat(),
            "from_agent": self.from_agent.value,
            "to_agent": self.to_agent.value,
            "message_type": self.message_type.value,
            "priority": self.priority.name,
            "payload": self.payload,
            "context": self.context.to_dict(),
            "ttl_seconds": self.ttl_seconds,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Create from dictionary."""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            correlation_id=data.get("correlation_id"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            from_agent=AgentType(data.get("from_agent", "orchestrator")),
            to_agent=AgentType(data.get("to_agent", "orchestrator")),
            message_type=MessageType(data.get("message_type", "request")),
            priority=MessagePriority[data.get("priority", "NORMAL")],
            payload=data.get("payload", {}),
            context=MessageContext.from_dict(data.get("context", {})),
            ttl_seconds=data.get("ttl_seconds"),
            retry_count=data.get("retry_count", 0),
        )

    def is_expired(self) -> bool:
        """Check if the message has expired."""
        if self.ttl_seconds is None:
            return False
        age = (datetime.now(timezone.utc) - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def create_response(
        self,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.RESPONSE,
    ) -> "AgentMessage":
        """Create a response message to this message."""
        return AgentMessage(
            correlation_id=self.message_id,
            from_agent=self.to_agent,
            to_agent=self.from_agent,
            message_type=message_type,
            priority=self.priority,
            payload=payload,
            context=self.context,
        )


@dataclass
class RetryPolicy:
    """
    Configuration for retry behavior.

    Defines how failed operations should be retried, with configurable
    backoff strategies and error filtering.

    Attributes:
        max_retries: Maximum number of retry attempts
        backoff_strategy: How to increase delay between retries
        initial_delay_seconds: First retry delay
        max_delay_seconds: Maximum delay cap
        retry_on_exceptions: Exception types to retry (None = all)
        non_retryable_exceptions: Exceptions that should not be retried
    """
    max_retries: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    retry_on_exceptions: Optional[Tuple[Type[Exception], ...]] = None
    non_retryable_exceptions: Tuple[Type[Exception], ...] = field(
        default_factory=lambda: (KeyboardInterrupt, SystemExit, GeneratorExit)
    )

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an operation should be retried."""
        if attempt >= self.max_retries:
            return False

        # Check non-retryable exceptions
        if isinstance(exception, self.non_retryable_exceptions):
            return False

        # If specific exceptions are defined, check against them
        if self.retry_on_exceptions is not None:
            return isinstance(exception, self.retry_on_exceptions)

        return True

    def get_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt."""
        if self.backoff_strategy == BackoffStrategy.NONE:
            return 0.0

        if self.backoff_strategy == BackoffStrategy.LINEAR:
            delay = self.initial_delay_seconds * (attempt + 1)
        elif self.backoff_strategy in (BackoffStrategy.EXPONENTIAL, BackoffStrategy.EXPONENTIAL_WITH_JITTER):
            delay = self.initial_delay_seconds * (2 ** attempt)
            if self.backoff_strategy == BackoffStrategy.EXPONENTIAL_WITH_JITTER:
                import random
                delay = delay * (0.5 + random.random())
        else:
            delay = self.initial_delay_seconds

        return min(delay, self.max_delay_seconds)


@dataclass
class AgentCheckpoint:
    """
    Checkpoint data for agent state persistence.

    Captures the full state of an agent at a point in time for
    recovery after failures or restarts.

    Attributes:
        agent_id: Agent identifier
        agent_type: Type of agent
        state: Current state
        checkpoint_id: Unique checkpoint identifier
        created_at: When checkpoint was created
        state_data: Serialized agent state
        pending_messages: Unprocessed messages
        metadata: Additional checkpoint info
    """
    agent_id: str
    agent_type: AgentType
    state: AgentState
    checkpoint_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state_data: Dict[str, Any] = field(default_factory=dict)
    pending_messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "state_data": self.state_data,
            "pending_messages": self.pending_messages,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentCheckpoint":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            agent_type=AgentType(data["agent_type"]),
            state=AgentState(data["state"]),
            checkpoint_id=data.get("checkpoint_id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            state_data=data.get("state_data", {}),
            pending_messages=data.get("pending_messages", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Retry Decorator
# =============================================================================


def with_retry(
    policy: Optional[RetryPolicy] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """
    Decorator for adding retry behavior to functions.

    Supports both sync and async functions with configurable retry
    policies and optional callback on each retry.

    Args:
        policy: Retry policy configuration (uses default if None)
        on_retry: Optional callback(exception, attempt) on each retry

    Returns:
        Decorated function with retry behavior

    Example:
        @with_retry(RetryPolicy(max_retries=5))
        async def fetch_data():
            ...
    """
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                for attempt in range(policy.max_retries + 1):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if not policy.should_retry(e, attempt):
                            raise
                        delay = policy.get_delay(attempt)
                        if on_retry:
                            on_retry(e, attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{policy.max_retries} for {func.__name__}: {e}. "
                            f"Waiting {delay:.2f}s"
                        )
                        await asyncio.sleep(delay)
                raise last_exception  # type: ignore
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                for attempt in range(policy.max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if not policy.should_retry(e, attempt):
                            raise
                        delay = policy.get_delay(attempt)
                        if on_retry:
                            on_retry(e, attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{policy.max_retries} for {func.__name__}: {e}. "
                            f"Waiting {delay:.2f}s"
                        )
                        time.sleep(delay)
                raise last_exception  # type: ignore
            return sync_wrapper
    return decorator


# =============================================================================
# Base Agent Abstract Class
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common infrastructure for lifecycle management, message
    handling, and state persistence. All specialized agents inherit
    from this class.

    Lifecycle:
        1. Create agent (PENDING state)
        2. start() -> RUNNING
        3. Process messages via run() loop
        4. stop() -> COMPLETED or shutdown() -> CANCELLED

    Subclasses must implement:
        - process_message(): Handle incoming messages
        - run(): Main execution loop

    Attributes:
        agent_id: Unique agent identifier
        agent_type: Type of agent
        state: Current lifecycle state
        started_at: When agent started running
        updated_at: Last state change time
        retry_policy: Default retry policy
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: AgentType = AgentType.ORCHESTRATOR,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier (auto-generated if None)
            agent_type: Type of agent
            retry_policy: Default retry configuration
        """
        self._agent_id = agent_id or f"{agent_type.value}-{str(uuid.uuid4())[:8]}"
        self._agent_type = agent_type
        self._state = AgentState.PENDING
        self._started_at: Optional[datetime] = None
        self._updated_at = datetime.now(timezone.utc)
        self._retry_policy = retry_policy or RetryPolicy()

        # Message handling
        self._inbox: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self._pending_responses: Dict[str, asyncio.Future[AgentMessage]] = {}

        # Registry reference (set when registered)
        self._registry: Optional["AgentRegistry"] = None
        self._message_bus: Optional["MessageBus"] = None

        # State persistence
        self._state_data: Dict[str, Any] = {}
        self._checkpoint_path: Optional[Path] = None

        # Shutdown flag
        self._shutdown_event = asyncio.Event()

        # Hooks storage
        self._on_start_hooks: List[Callable[[], Coroutine[Any, Any, None]]] = []
        self._on_stop_hooks: List[Callable[[], Coroutine[Any, Any, None]]] = []
        self._on_error_hooks: List[Callable[[Exception], Coroutine[Any, Any, None]]] = []
        self._on_message_hooks: List[Callable[[AgentMessage], Coroutine[Any, Any, None]]] = []

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def agent_id(self) -> str:
        """Get agent identifier."""
        return self._agent_id

    @property
    def agent_type(self) -> AgentType:
        """Get agent type."""
        return self._agent_type

    @property
    def state(self) -> AgentState:
        """Get current state."""
        return self._state

    @property
    def started_at(self) -> Optional[datetime]:
        """Get start time."""
        return self._started_at

    @property
    def updated_at(self) -> datetime:
        """Get last update time."""
        return self._updated_at

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._state == AgentState.RUNNING

    @property
    def retry_policy(self) -> RetryPolicy:
        """Get retry policy."""
        return self._retry_policy

    # -------------------------------------------------------------------------
    # Abstract Methods (must be implemented by subclasses)
    # -------------------------------------------------------------------------

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        This is the main message handler that subclasses implement to
        define their behavior.

        Args:
            message: Incoming message to process

        Returns:
            Optional response message (auto-sent if returned)
        """
        pass

    @abstractmethod
    async def run(self) -> None:
        """
        Main agent execution loop.

        Called when the agent starts. Should process messages from
        the inbox until shutdown is requested.

        Example implementation:
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
                except Exception as e:
                    await self.on_error(e)
        """
        pass

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the agent.

        Transitions from PENDING to RUNNING and triggers on_start hooks.
        """
        if self._state != AgentState.PENDING:
            raise RuntimeError(f"Cannot start agent in {self._state} state")

        logger.info(f"Starting agent {self._agent_id}")
        self._state = AgentState.RUNNING
        self._started_at = datetime.now(timezone.utc)
        self._updated_at = datetime.now(timezone.utc)

        # Run hooks
        await self.on_start()

    async def pause(self) -> None:
        """
        Pause the agent.

        Transitions from RUNNING to PAUSED. Agent stops processing
        but retains state.
        """
        if self._state != AgentState.RUNNING:
            raise RuntimeError(f"Cannot pause agent in {self._state} state")

        logger.info(f"Pausing agent {self._agent_id}")
        self._state = AgentState.PAUSED
        self._updated_at = datetime.now(timezone.utc)

    async def resume(self) -> None:
        """
        Resume a paused agent.

        Transitions from PAUSED to RUNNING.
        """
        if self._state != AgentState.PAUSED:
            raise RuntimeError(f"Cannot resume agent in {self._state} state")

        logger.info(f"Resuming agent {self._agent_id}")
        self._state = AgentState.RUNNING
        self._updated_at = datetime.now(timezone.utc)

    async def stop(self) -> None:
        """
        Stop the agent normally.

        Transitions to COMPLETED state and triggers on_stop hooks.
        """
        if self._state not in (AgentState.RUNNING, AgentState.PAUSED):
            raise RuntimeError(f"Cannot stop agent in {self._state} state")

        logger.info(f"Stopping agent {self._agent_id}")
        self._shutdown_event.set()
        self._state = AgentState.COMPLETED
        self._updated_at = datetime.now(timezone.utc)

        # Run hooks
        await self.on_stop()

    async def shutdown(self) -> None:
        """
        Force shutdown the agent.

        Cancels current operations and transitions to CANCELLED state.
        """
        logger.info(f"Shutting down agent {self._agent_id}")
        self._shutdown_event.set()
        self._state = AgentState.CANCELLED
        self._updated_at = datetime.now(timezone.utc)

        # Cancel pending response futures
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()

        # Run hooks
        await self.on_stop()

    async def fail(self, error: Exception) -> None:
        """
        Transition to failed state due to error.

        Args:
            error: The exception that caused failure
        """
        logger.error(f"Agent {self._agent_id} failed: {error}")
        self._shutdown_event.set()
        self._state = AgentState.FAILED
        self._updated_at = datetime.now(timezone.utc)
        self._state_data["failure_reason"] = str(error)

        # Run error hooks
        await self.on_error(error)

    # -------------------------------------------------------------------------
    # Message Methods
    # -------------------------------------------------------------------------

    async def send_message(self, message: AgentMessage) -> None:
        """
        Send a message to another agent.

        Uses the message bus if available, otherwise sends directly
        to the target agent via registry.

        Args:
            message: Message to send
        """
        if self._message_bus:
            await self._message_bus.publish(message)
        elif self._registry:
            target = self._registry.get_agent_by_type(message.to_agent)
            if target:
                await target.receive_message(message)
            else:
                logger.warning(f"No agent found for type {message.to_agent}")
        else:
            logger.warning(f"Agent {self._agent_id} has no message bus or registry")

    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message into the inbox.

        Triggers on_message_received hooks and handles response
        correlation.

        Args:
            message: Incoming message
        """
        # Check for response to pending request
        if message.correlation_id and message.correlation_id in self._pending_responses:
            future = self._pending_responses.pop(message.correlation_id)
            if not future.done():
                future.set_result(message)
            return

        # Run hooks
        await self.on_message_received(message)

        # Add to inbox
        await self._inbox.put(message)

    async def request(
        self,
        to_agent: AgentType,
        payload: Dict[str, Any],
        timeout: float = 30.0,
        priority: MessagePriority = MessagePriority.NORMAL,
        context: Optional[MessageContext] = None,
    ) -> AgentMessage:
        """
        Send a request and wait for response.

        Creates a request message, sends it, and waits for a
        correlated response.

        Args:
            to_agent: Target agent type
            payload: Request data
            timeout: Max wait time in seconds
            priority: Message priority
            context: Execution context

        Returns:
            Response message

        Raises:
            asyncio.TimeoutError: If no response within timeout
        """
        message = AgentMessage(
            from_agent=self._agent_type,
            to_agent=to_agent,
            message_type=MessageType.REQUEST,
            priority=priority,
            payload=payload,
            context=context or MessageContext(),
        )

        # Create future for response
        loop = asyncio.get_running_loop()
        future: asyncio.Future[AgentMessage] = loop.create_future()
        self._pending_responses[message.message_id] = future

        try:
            await self.send_message(message)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_responses.pop(message.message_id, None)
            raise

    async def broadcast(
        self,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.EVENT,
        exclude: Optional[Set[AgentType]] = None,
    ) -> None:
        """
        Broadcast a message to all agents.

        Args:
            payload: Message data
            message_type: Type of message
            exclude: Agent types to exclude
        """
        if self._message_bus:
            message = AgentMessage(
                from_agent=self._agent_type,
                to_agent=self._agent_type,  # Overridden by broadcast
                message_type=message_type,
                payload=payload,
            )
            await self._message_bus.broadcast(message, exclude=exclude)
        elif self._registry:
            for agent_type in AgentType:
                if exclude and agent_type in exclude:
                    continue
                if agent_type == self._agent_type:
                    continue
                message = AgentMessage(
                    from_agent=self._agent_type,
                    to_agent=agent_type,
                    message_type=message_type,
                    payload=payload,
                )
                await self.send_message(message)

    # -------------------------------------------------------------------------
    # State Persistence Methods
    # -------------------------------------------------------------------------

    def set_state_data(self, key: str, value: Any) -> None:
        """Store data in agent state."""
        self._state_data[key] = value

    def get_state_data(self, key: str, default: Any = None) -> Any:
        """Retrieve data from agent state."""
        return self._state_data.get(key, default)

    async def save_state(self, path: Optional[Path] = None) -> Path:
        """
        Save agent state to file.

        Args:
            path: Output path (uses default if None)

        Returns:
            Path where state was saved
        """
        path = path or self._checkpoint_path or Path(f".agent_state/{self._agent_id}.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = AgentCheckpoint(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            state=self._state,
            state_data=self._state_data,
            pending_messages=[],  # Drain inbox for checkpoint
            metadata={
                "started_at": self._started_at.isoformat() if self._started_at else None,
                "updated_at": self._updated_at.isoformat(),
            },
        )

        # Drain pending messages
        while not self._inbox.empty():
            try:
                msg = self._inbox.get_nowait()
                checkpoint.pending_messages.append(msg.to_dict())
            except asyncio.QueueEmpty:
                break

        with open(path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        logger.info(f"Saved state for agent {self._agent_id} to {path}")
        return path

    async def load_state(self, path: Optional[Path] = None) -> bool:
        """
        Load agent state from file.

        Args:
            path: Input path (uses default if None)

        Returns:
            True if state was loaded successfully
        """
        path = path or self._checkpoint_path or Path(f".agent_state/{self._agent_id}.json")

        if not path.exists():
            logger.warning(f"No state file found at {path}")
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            checkpoint = AgentCheckpoint.from_dict(data)

            self._state = checkpoint.state
            self._state_data = checkpoint.state_data

            if checkpoint.metadata.get("started_at"):
                self._started_at = datetime.fromisoformat(checkpoint.metadata["started_at"])
            if checkpoint.metadata.get("updated_at"):
                self._updated_at = datetime.fromisoformat(checkpoint.metadata["updated_at"])

            # Restore pending messages
            for msg_data in checkpoint.pending_messages:
                msg = AgentMessage.from_dict(msg_data)
                await self._inbox.put(msg)

            logger.info(f"Loaded state for agent {self._agent_id} from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load state from {path}: {e}")
            return False

    async def checkpoint(self) -> str:
        """
        Create a checkpoint of current state.

        Returns:
            Checkpoint ID
        """
        checkpoint = AgentCheckpoint(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            state=self._state,
            state_data=self._state_data.copy(),
        )

        await self.save_state()
        logger.info(f"Created checkpoint {checkpoint.checkpoint_id} for agent {self._agent_id}")
        return checkpoint.checkpoint_id

    # -------------------------------------------------------------------------
    # Hooks (override in subclasses)
    # -------------------------------------------------------------------------

    async def on_start(self) -> None:
        """Hook called when agent starts."""
        for hook in self._on_start_hooks:
            await hook()

    async def on_stop(self) -> None:
        """Hook called when agent stops."""
        for hook in self._on_stop_hooks:
            await hook()

    async def on_error(self, error: Exception) -> None:
        """Hook called when an error occurs."""
        for hook in self._on_error_hooks:
            await hook(error)

    async def on_message_received(self, message: AgentMessage) -> None:
        """Hook called when a message is received."""
        for hook in self._on_message_hooks:
            await hook(message)

    def add_start_hook(self, hook: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register a start hook."""
        self._on_start_hooks.append(hook)

    def add_stop_hook(self, hook: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Register a stop hook."""
        self._on_stop_hooks.append(hook)

    def add_error_hook(self, hook: Callable[[Exception], Coroutine[Any, Any, None]]) -> None:
        """Register an error hook."""
        self._on_error_hooks.append(hook)

    def add_message_hook(self, hook: Callable[[AgentMessage], Coroutine[Any, Any, None]]) -> None:
        """Register a message received hook."""
        self._on_message_hooks.append(hook)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._agent_id}, type={self._agent_type.value}, state={self._state.value})"


# =============================================================================
# Agent Registry
# =============================================================================


class AgentRegistry:
    """
    Registry for managing agents and routing messages.

    Provides centralized agent registration, lookup by type or ID,
    and message routing between agents.

    Thread-safe for concurrent access.

    Example:
        registry = AgentRegistry()
        registry.register(orchestrator)
        registry.register(discovery_agent)

        # Route message
        await registry.route_message(message)

        # Get agent
        agent = registry.get_agent_by_type(AgentType.DISCOVERY)
    """

    def __init__(self):
        """Initialize the registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._agents_by_type: Dict[AgentType, List[BaseAgent]] = {
            agent_type: [] for agent_type in AgentType
        }
        self._lock = threading.RLock()
        self._message_bus: Optional["MessageBus"] = None

    def set_message_bus(self, bus: "MessageBus") -> None:
        """Associate a message bus with this registry."""
        self._message_bus = bus

    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            agent: Agent to register

        Raises:
            ValueError: If agent ID is already registered
        """
        with self._lock:
            if agent.agent_id in self._agents:
                raise ValueError(f"Agent {agent.agent_id} already registered")

            self._agents[agent.agent_id] = agent
            self._agents_by_type[agent.agent_type].append(agent)
            agent._registry = self

            if self._message_bus:
                agent._message_bus = self._message_bus

            logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type.value}")

    def unregister(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Unregister an agent.

        Args:
            agent_id: ID of agent to unregister

        Returns:
            The unregistered agent, or None if not found
        """
        with self._lock:
            agent = self._agents.pop(agent_id, None)
            if agent:
                self._agents_by_type[agent.agent_type].remove(agent)
                agent._registry = None
                agent._message_bus = None
                logger.info(f"Unregistered agent {agent_id}")
            return agent

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID."""
        with self._lock:
            return self._agents.get(agent_id)

    def get_agent_by_type(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        Get an agent by type.

        Returns the first registered agent of the given type.
        For load balancing, use get_agents_by_type() instead.

        Args:
            agent_type: Type of agent to find

        Returns:
            An agent of the specified type, or None
        """
        with self._lock:
            agents = self._agents_by_type.get(agent_type, [])
            return agents[0] if agents else None

    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a type."""
        with self._lock:
            return list(self._agents_by_type.get(agent_type, []))

    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents."""
        with self._lock:
            return list(self._agents.values())

    async def route_message(self, message: AgentMessage) -> bool:
        """
        Route a message to the target agent.

        Args:
            message: Message to route

        Returns:
            True if message was delivered
        """
        target = self.get_agent_by_type(message.to_agent)
        if target:
            await target.receive_message(message)
            return True

        logger.warning(f"No agent found for type {message.to_agent}")
        return False

    async def broadcast(
        self,
        message: AgentMessage,
        exclude: Optional[Set[AgentType]] = None,
    ) -> int:
        """
        Broadcast a message to all agents.

        Args:
            message: Message to broadcast
            exclude: Agent types to exclude

        Returns:
            Number of agents that received the message
        """
        count = 0
        for agent in self.get_all_agents():
            if exclude and agent.agent_type in exclude:
                continue
            if agent.agent_type == message.from_agent:
                continue

            msg_copy = AgentMessage(
                message_id=str(uuid.uuid4()),
                correlation_id=message.correlation_id,
                timestamp=message.timestamp,
                from_agent=message.from_agent,
                to_agent=agent.agent_type,
                message_type=message.message_type,
                priority=message.priority,
                payload=message.payload.copy(),
                context=message.context,
            )
            await agent.receive_message(msg_copy)
            count += 1

        return count

    async def shutdown_all(self) -> None:
        """Shutdown all registered agents."""
        for agent in self.get_all_agents():
            try:
                await agent.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down agent {agent.agent_id}: {e}")


# =============================================================================
# Message Bus
# =============================================================================


class MessageBus:
    """
    Async message bus for inter-agent communication.

    Provides publish-subscribe messaging with priority queue handling.
    Messages are delivered based on priority (CRITICAL first, LOW last).

    Features:
        - Priority-based message ordering
        - Subscribe/unsubscribe to specific message types
        - Broadcast support
        - Message TTL handling

    Example:
        bus = MessageBus()
        bus.set_registry(registry)

        # Subscribe to discovery messages
        async def handler(msg):
            print(f"Got: {msg}")
        bus.subscribe(AgentType.DISCOVERY, handler)

        # Publish
        await bus.publish(message)

        # Start processing
        await bus.start()
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize the message bus.

        Args:
            max_queue_size: Maximum number of queued messages
        """
        self._max_queue_size = max_queue_size
        self._queue: asyncio.PriorityQueue[Tuple[int, float, AgentMessage]] = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._registry: Optional[AgentRegistry] = None
        self._subscribers: Dict[AgentType, List[Callable[[AgentMessage], Coroutine[Any, Any, None]]]] = {
            agent_type: [] for agent_type in AgentType
        }
        self._global_subscribers: List[Callable[[AgentMessage], Coroutine[Any, Any, None]]] = []
        self._running = False
        self._message_count = 0
        self._lock = threading.RLock()

    def set_registry(self, registry: AgentRegistry) -> None:
        """Associate a registry with this bus."""
        self._registry = registry
        registry.set_message_bus(self)

    def subscribe(
        self,
        agent_type: AgentType,
        handler: Callable[[AgentMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Subscribe to messages for a specific agent type.

        Args:
            agent_type: Type of agent to receive messages for
            handler: Async handler function
        """
        with self._lock:
            self._subscribers[agent_type].append(handler)

    def subscribe_all(
        self,
        handler: Callable[[AgentMessage], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Subscribe to all messages.

        Args:
            handler: Async handler function
        """
        with self._lock:
            self._global_subscribers.append(handler)

    def unsubscribe(
        self,
        agent_type: AgentType,
        handler: Callable[[AgentMessage], Coroutine[Any, Any, None]],
    ) -> bool:
        """
        Unsubscribe from messages.

        Args:
            agent_type: Agent type to unsubscribe from
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        with self._lock:
            try:
                self._subscribers[agent_type].remove(handler)
                return True
            except ValueError:
                return False

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the bus.

        Messages are queued by priority for processing.

        Args:
            message: Message to publish
        """
        if message.is_expired():
            logger.warning(f"Dropping expired message {message.message_id}")
            return

        # Priority tuple: (priority_value, timestamp_for_ordering, message)
        priority_item = (
            message.priority.value,
            time.time(),
            message,
        )

        try:
            await self._queue.put(priority_item)
            self._message_count += 1
        except asyncio.QueueFull:
            logger.error(f"Message bus queue full, dropping message {message.message_id}")

    async def broadcast(
        self,
        message: AgentMessage,
        exclude: Optional[Set[AgentType]] = None,
    ) -> None:
        """
        Broadcast a message to all agent types.

        Args:
            message: Message to broadcast
            exclude: Agent types to exclude
        """
        for agent_type in AgentType:
            if exclude and agent_type in exclude:
                continue
            if agent_type == message.from_agent:
                continue

            msg_copy = AgentMessage(
                message_id=str(uuid.uuid4()),
                correlation_id=message.correlation_id,
                timestamp=message.timestamp,
                from_agent=message.from_agent,
                to_agent=agent_type,
                message_type=message.message_type,
                priority=message.priority,
                payload=message.payload.copy(),
                context=message.context,
            )
            await self.publish(msg_copy)

    async def start(self) -> None:
        """Start processing messages from the queue."""
        self._running = True
        logger.info("Message bus started")

        while self._running:
            try:
                # Wait for message with timeout
                priority_item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )
                _, _, message = priority_item

                # Skip expired messages
                if message.is_expired():
                    logger.debug(f"Skipping expired message {message.message_id}")
                    continue

                # Deliver to target via registry
                if self._registry:
                    await self._registry.route_message(message)

                # Notify subscribers
                with self._lock:
                    handlers = list(self._subscribers[message.to_agent])
                    global_handlers = list(self._global_subscribers)

                for handler in handlers + global_handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message bus error: {e}")

    async def stop(self) -> None:
        """Stop processing messages."""
        self._running = False
        logger.info("Message bus stopped")

    @property
    def pending_count(self) -> int:
        """Get number of pending messages."""
        return self._queue.qsize()

    @property
    def total_messages(self) -> int:
        """Get total number of messages processed."""
        return self._message_count


# =============================================================================
# State Persistence (SQLite Backend)
# =============================================================================


class AgentStateStore:
    """
    SQLite-backed state store for agent checkpoints.

    Provides persistent storage for agent state, enabling recovery
    after failures or restarts.

    Example:
        store = AgentStateStore("agents.db")

        # Save checkpoint
        await store.save_checkpoint(checkpoint)

        # Load latest
        checkpoint = await store.load_latest_checkpoint("agent-123")
    """

    def __init__(self, db_path: Union[str, Path] = ":memory:"):
        """
        Initialize state store.

        Args:
            db_path: SQLite database path (":memory:" for in-memory)
        """
        self._db_path = str(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._is_memory = self._db_path == ":memory:"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        with self._lock:
            if self._connection is None or (not self._is_memory):
                self._connection = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False
                )
                self._connection.row_factory = sqlite3.Row
            return self._connection

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                state_data TEXT,
                pending_messages TEXT,
                metadata TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_agent_id
            ON checkpoints(agent_id);

            CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at
            ON checkpoints(created_at DESC);
        """)
        conn.commit()

    async def save_checkpoint(self, checkpoint: AgentCheckpoint) -> None:
        """
        Save a checkpoint to the database.

        Args:
            checkpoint: Checkpoint to save
        """
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints
                (checkpoint_id, agent_id, agent_type, state, created_at,
                 state_data, pending_messages, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.agent_id,
                    checkpoint.agent_type.value,
                    checkpoint.state.value,
                    checkpoint.created_at.isoformat(),
                    json.dumps(checkpoint.state_data),
                    json.dumps(checkpoint.pending_messages),
                    json.dumps(checkpoint.metadata),
                )
            )
            conn.commit()
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id}")

    async def load_checkpoint(self, checkpoint_id: str) -> Optional[AgentCheckpoint]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to load

        Returns:
            Checkpoint if found, None otherwise
        """
        conn = self._get_connection()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,)
            ).fetchone()

        if not row:
            return None

        return AgentCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            agent_id=row["agent_id"],
            agent_type=AgentType(row["agent_type"]),
            state=AgentState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            state_data=json.loads(row["state_data"]) if row["state_data"] else {},
            pending_messages=json.loads(row["pending_messages"]) if row["pending_messages"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def load_latest_checkpoint(self, agent_id: str) -> Optional[AgentCheckpoint]:
        """
        Load the most recent checkpoint for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Latest checkpoint if found, None otherwise
        """
        conn = self._get_connection()
        with self._lock:
            row = conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (agent_id,)
            ).fetchone()

        if not row:
            return None

        return AgentCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            agent_id=row["agent_id"],
            agent_type=AgentType(row["agent_type"]),
            state=AgentState(row["state"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            state_data=json.loads(row["state_data"]) if row["state_data"] else {},
            pending_messages=json.loads(row["pending_messages"]) if row["pending_messages"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def list_checkpoints(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AgentCheckpoint]:
        """
        List checkpoints, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID filter
            limit: Maximum number to return

        Returns:
            List of checkpoints
        """
        conn = self._get_connection()
        with self._lock:
            if agent_id:
                rows = conn.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE agent_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (agent_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM checkpoints
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,)
                ).fetchall()

        return [
            AgentCheckpoint(
                checkpoint_id=row["checkpoint_id"],
                agent_id=row["agent_id"],
                agent_type=AgentType(row["agent_type"]),
                state=AgentState(row["state"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                state_data=json.loads(row["state_data"]) if row["state_data"] else {},
                pending_messages=json.loads(row["pending_messages"]) if row["pending_messages"] else [],
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            for row in rows
        ]

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: Checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_connection()
        with self._lock:
            cursor = conn.execute(
                "DELETE FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    async def cleanup_old_checkpoints(
        self,
        agent_id: str,
        keep_count: int = 5,
    ) -> int:
        """
        Remove old checkpoints, keeping only the most recent.

        Args:
            agent_id: Agent to clean up
            keep_count: Number of checkpoints to keep

        Returns:
            Number of deleted checkpoints
        """
        conn = self._get_connection()
        with self._lock:
            # Get IDs to keep
            keep_ids = conn.execute(
                """
                SELECT checkpoint_id FROM checkpoints
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (agent_id, keep_count)
            ).fetchall()
            keep_ids = [row[0] for row in keep_ids]

            if not keep_ids:
                return 0

            # Delete others
            placeholders = ",".join("?" * len(keep_ids))
            cursor = conn.execute(
                f"""
                DELETE FROM checkpoints
                WHERE agent_id = ? AND checkpoint_id NOT IN ({placeholders})
                """,
                [agent_id] + keep_ids
            )
            conn.commit()
            return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            if self._connection:
                self._connection.close()
                self._connection = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "AgentState",
    "AgentType",
    "MessageType",
    "MessagePriority",
    "BackoffStrategy",
    # Data classes
    "MessageContext",
    "AgentMessage",
    "RetryPolicy",
    "AgentCheckpoint",
    # Base classes
    "BaseAgent",
    # Registry and bus
    "AgentRegistry",
    "MessageBus",
    # State persistence
    "AgentStateStore",
    # Decorators
    "with_retry",
]
