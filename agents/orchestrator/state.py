"""
State Management for Orchestrator Agent.

Provides state tracking, persistence, and recovery for workflow execution:
- ExecutionState dataclass tracking event_id, stage, progress, timestamps, errors
- StateManager class with SQLite backend for persistence
- Checkpoint/restore functionality for fault tolerance
- Progress tracking per processing stage

The state management system enables:
- Resume interrupted workflows from last checkpoint
- Track progress across Discovery, Pipeline, Quality, Reporting stages
- Record errors and degraded mode decisions
- Maintain full execution history for provenance
"""

import asyncio
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class ExecutionStage(Enum):
    """Processing stages in the orchestrator workflow."""
    PENDING = "pending"
    VALIDATING = "validating"
    DISCOVERY = "discovery"
    PIPELINE = "pipeline"
    QUALITY = "quality"
    REPORTING = "reporting"
    ASSEMBLY = "assembly"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DegradedModeLevel(Enum):
    """Levels of degraded operation."""
    NONE = "none"           # Full quality operation
    MINOR = "minor"         # Some optional data missing
    MODERATE = "moderate"   # Fallback algorithms used
    SIGNIFICANT = "significant"  # Reduced confidence
    CRITICAL = "critical"   # Minimal viable output


@dataclass
class StageProgress:
    """
    Progress information for a processing stage.

    Attributes:
        stage: Stage identifier
        status: Current status (pending, running, completed, failed)
        started_at: Stage start time
        completed_at: Stage completion time
        progress_percent: Progress percentage (0-100)
        message: Current status message
        metrics: Stage-specific metrics
        errors: Any errors encountered
    """
    stage: ExecutionStage
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float = 0.0
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage": self.stage.value,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress_percent": self.progress_percent,
            "message": self.message,
            "metrics": self.metrics,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageProgress':
        """Create from dictionary."""
        return cls(
            stage=ExecutionStage(data["stage"]),
            status=data.get("status", "pending"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress_percent=data.get("progress_percent", 0.0),
            message=data.get("message", ""),
            metrics=data.get("metrics", {}),
            errors=data.get("errors", []),
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get stage duration in seconds."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()


@dataclass
class DegradedModeInfo:
    """
    Information about degraded mode operation.

    Attributes:
        level: Degradation level
        active_since: When degraded mode was activated
        reasons: List of reasons for degradation
        fallbacks_used: Fallback strategies employed
        confidence_impact: Impact on confidence scores
    """
    level: DegradedModeLevel = DegradedModeLevel.NONE
    active_since: Optional[datetime] = None
    reasons: List[str] = field(default_factory=list)
    fallbacks_used: List[str] = field(default_factory=list)
    confidence_impact: float = 0.0  # Reduction in confidence (0-1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "active_since": self.active_since.isoformat() if self.active_since else None,
            "reasons": self.reasons,
            "fallbacks_used": self.fallbacks_used,
            "confidence_impact": self.confidence_impact,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DegradedModeInfo':
        """Create from dictionary."""
        return cls(
            level=DegradedModeLevel(data.get("level", "none")),
            active_since=datetime.fromisoformat(data["active_since"]) if data.get("active_since") else None,
            reasons=data.get("reasons", []),
            fallbacks_used=data.get("fallbacks_used", []),
            confidence_impact=data.get("confidence_impact", 0.0),
        )


@dataclass
class ExecutionState:
    """
    Complete execution state for an event workflow.

    Attributes:
        event_id: Event identifier being processed
        orchestrator_id: ID of orchestrator agent handling this event
        current_stage: Current processing stage
        stages: Progress for each stage
        created_at: Workflow creation time
        started_at: Workflow start time
        completed_at: Workflow completion time
        event_spec: Original event specification
        intent_resolution: Resolved intent from specification
        discovery_results: Data discovery results
        pipeline_results: Pipeline execution results
        quality_results: Quality check results
        final_products: Generated products
        degraded_mode: Degraded mode information
        error_summary: Summary of all errors
        checkpoint_data: Last checkpoint for recovery
        metadata: Additional metadata
    """
    event_id: str
    orchestrator_id: str = ""
    current_stage: ExecutionStage = ExecutionStage.PENDING
    stages: Dict[str, StageProgress] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    event_spec: Dict[str, Any] = field(default_factory=dict)
    intent_resolution: Dict[str, Any] = field(default_factory=dict)
    discovery_results: Dict[str, Any] = field(default_factory=dict)
    pipeline_results: Dict[str, Any] = field(default_factory=dict)
    quality_results: Dict[str, Any] = field(default_factory=dict)
    final_products: List[Dict[str, Any]] = field(default_factory=list)
    degraded_mode: DegradedModeInfo = field(default_factory=DegradedModeInfo)
    error_summary: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize stage progress tracking."""
        if not self.stages:
            for stage in ExecutionStage:
                if stage not in (ExecutionStage.COMPLETED, ExecutionStage.FAILED, ExecutionStage.CANCELLED):
                    self.stages[stage.value] = StageProgress(stage=stage)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "orchestrator_id": self.orchestrator_id,
            "current_stage": self.current_stage.value,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "event_spec": self.event_spec,
            "intent_resolution": self.intent_resolution,
            "discovery_results": self.discovery_results,
            "pipeline_results": self.pipeline_results,
            "quality_results": self.quality_results,
            "final_products": self.final_products,
            "degraded_mode": self.degraded_mode.to_dict(),
            "error_summary": self.error_summary,
            "checkpoint_data": self.checkpoint_data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionState':
        """Create from dictionary."""
        stages = {}
        for k, v in data.get("stages", {}).items():
            stages[k] = StageProgress.from_dict(v)

        state = cls(
            event_id=data["event_id"],
            orchestrator_id=data.get("orchestrator_id", ""),
            current_stage=ExecutionStage(data.get("current_stage", "pending")),
            stages=stages,
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            event_spec=data.get("event_spec", {}),
            intent_resolution=data.get("intent_resolution", {}),
            discovery_results=data.get("discovery_results", {}),
            pipeline_results=data.get("pipeline_results", {}),
            quality_results=data.get("quality_results", {}),
            final_products=data.get("final_products", []),
            degraded_mode=DegradedModeInfo.from_dict(data.get("degraded_mode", {})),
            error_summary=data.get("error_summary", []),
            checkpoint_data=data.get("checkpoint_data", {}),
            metadata=data.get("metadata", {}),
        )
        return state

    @property
    def is_terminal(self) -> bool:
        """Check if execution is in a terminal state."""
        return self.current_stage in (
            ExecutionStage.COMPLETED,
            ExecutionStage.FAILED,
            ExecutionStage.CANCELLED
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get total execution duration."""
        if self.started_at is None:
            return None
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        stage_weights = {
            ExecutionStage.PENDING: 0.0,
            ExecutionStage.VALIDATING: 0.05,
            ExecutionStage.DISCOVERY: 0.20,
            ExecutionStage.PIPELINE: 0.45,
            ExecutionStage.QUALITY: 0.20,
            ExecutionStage.REPORTING: 0.05,
            ExecutionStage.ASSEMBLY: 0.05,
        }

        total_progress = 0.0
        for stage_name, progress in self.stages.items():
            try:
                stage = ExecutionStage(stage_name)
                weight = stage_weights.get(stage, 0.0)
                if progress.status == "completed":
                    total_progress += weight * 100
                elif progress.status == "running":
                    total_progress += weight * progress.progress_percent
            except (ValueError, KeyError):
                continue

        return min(total_progress, 100.0)


class StateManager:
    """
    Manager for execution state persistence.

    Uses SQLite for durable storage with checkpoint/restore capabilities.

    Example:
        manager = StateManager("/path/to/state.db")
        await manager.initialize()

        # Create new execution state
        state = await manager.create_state("event_001", event_spec)

        # Update stage progress
        await manager.update_stage(state.event_id, ExecutionStage.DISCOVERY, "running", 50.0)

        # Checkpoint for recovery
        await manager.checkpoint(state)

        # Later: restore from checkpoint
        restored = await manager.restore_checkpoint("event_001")
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database. Uses in-memory if None.
        """
        self.db_path = db_path or ":memory:"
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @contextmanager
    def _get_connection(self):
        """Get database connection with context management."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._connection.row_factory = sqlite3.Row
        yield self._connection

    async def initialize(self) -> None:
        """Initialize the database schema."""
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Main execution state table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS execution_states (
                        event_id TEXT PRIMARY KEY,
                        orchestrator_id TEXT,
                        current_stage TEXT,
                        created_at TEXT,
                        started_at TEXT,
                        completed_at TEXT,
                        state_json TEXT,
                        updated_at TEXT
                    )
                """)

                # Checkpoints table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT,
                        stage TEXT,
                        checkpoint_data TEXT,
                        created_at TEXT,
                        FOREIGN KEY (event_id) REFERENCES execution_states(event_id)
                    )
                """)

                # Stage history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stage_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT,
                        stage TEXT,
                        status TEXT,
                        progress_percent REAL,
                        message TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (event_id) REFERENCES execution_states(event_id)
                    )
                """)

                # Error log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS error_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT,
                        stage TEXT,
                        error_type TEXT,
                        error_message TEXT,
                        stack_trace TEXT,
                        timestamp TEXT,
                        FOREIGN KEY (event_id) REFERENCES execution_states(event_id)
                    )
                """)

                # Indices for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_execution_stage
                    ON execution_states(current_stage)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_checkpoints_event
                    ON checkpoints(event_id)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_stage_history_event
                    ON stage_history(event_id)
                """)

                conn.commit()
                self._initialized = True
                logger.info(f"StateManager initialized with database: {self.db_path}")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None

    async def create_state(
        self,
        event_id: str,
        event_spec: Dict[str, Any],
        orchestrator_id: str = ""
    ) -> ExecutionState:
        """
        Create a new execution state.

        Args:
            event_id: Event identifier
            event_spec: Event specification
            orchestrator_id: Orchestrator agent ID

        Returns:
            New ExecutionState instance
        """
        state = ExecutionState(
            event_id=event_id,
            orchestrator_id=orchestrator_id,
            event_spec=event_spec,
        )

        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()

                cursor.execute("""
                    INSERT INTO execution_states
                    (event_id, orchestrator_id, current_stage, created_at, state_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    orchestrator_id,
                    state.current_stage.value,
                    now,
                    json.dumps(state.to_dict()),
                    now,
                ))
                conn.commit()

        logger.info(f"Created execution state for event: {event_id}")
        return state

    async def get_state(self, event_id: str) -> Optional[ExecutionState]:
        """
        Get execution state by event ID.

        Args:
            event_id: Event identifier

        Returns:
            ExecutionState if found, None otherwise
        """
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT state_json FROM execution_states WHERE event_id = ?",
                    (event_id,)
                )
                row = cursor.fetchone()

                if row:
                    return ExecutionState.from_dict(json.loads(row["state_json"]))
                return None

    async def update_state(self, state: ExecutionState) -> None:
        """
        Update execution state in database.

        Args:
            state: ExecutionState to update
        """
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now(timezone.utc).isoformat()

                cursor.execute("""
                    UPDATE execution_states
                    SET orchestrator_id = ?,
                        current_stage = ?,
                        started_at = ?,
                        completed_at = ?,
                        state_json = ?,
                        updated_at = ?
                    WHERE event_id = ?
                """, (
                    state.orchestrator_id,
                    state.current_stage.value,
                    state.started_at.isoformat() if state.started_at else None,
                    state.completed_at.isoformat() if state.completed_at else None,
                    json.dumps(state.to_dict()),
                    now,
                    state.event_id,
                ))
                conn.commit()

    async def update_stage(
        self,
        event_id: str,
        stage: ExecutionStage,
        status: str,
        progress_percent: float = 0.0,
        message: str = "",
        metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[ExecutionState]:
        """
        Update progress for a specific stage.

        Args:
            event_id: Event identifier
            stage: Stage to update
            status: New status (pending, running, completed, failed)
            progress_percent: Progress percentage
            message: Status message
            metrics: Stage metrics

        Returns:
            Updated ExecutionState
        """
        state = await self.get_state(event_id)
        if not state:
            logger.warning(f"No state found for event: {event_id}")
            return None

        stage_key = stage.value
        if stage_key not in state.stages:
            state.stages[stage_key] = StageProgress(stage=stage)

        stage_progress = state.stages[stage_key]
        old_status = stage_progress.status
        stage_progress.status = status
        stage_progress.progress_percent = progress_percent
        stage_progress.message = message

        if metrics:
            stage_progress.metrics.update(metrics)

        now = datetime.now(timezone.utc)

        # Track timing
        if status == "running" and stage_progress.started_at is None:
            stage_progress.started_at = now
        elif status in ("completed", "failed"):
            stage_progress.completed_at = now

        # Update current stage if moving forward
        if status == "running":
            state.current_stage = stage
        elif status == "completed":
            # Move to next stage
            stage_order = [
                ExecutionStage.PENDING,
                ExecutionStage.VALIDATING,
                ExecutionStage.DISCOVERY,
                ExecutionStage.PIPELINE,
                ExecutionStage.QUALITY,
                ExecutionStage.REPORTING,
                ExecutionStage.ASSEMBLY,
                ExecutionStage.COMPLETED,
            ]
            try:
                current_idx = stage_order.index(stage)
                if current_idx < len(stage_order) - 1:
                    # Only update current_stage if completing the current stage
                    if state.current_stage == stage:
                        state.current_stage = stage_order[current_idx + 1]
            except ValueError:
                pass
        elif status == "failed":
            state.current_stage = ExecutionStage.FAILED

        await self.update_state(state)

        # Record in stage history
        await self._record_stage_history(event_id, stage, status, progress_percent, message)

        logger.debug(f"Updated stage {stage.value} for {event_id}: {old_status} -> {status}")
        return state

    async def _record_stage_history(
        self,
        event_id: str,
        stage: ExecutionStage,
        status: str,
        progress_percent: float,
        message: str
    ) -> None:
        """Record stage transition in history."""
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO stage_history
                    (event_id, stage, status, progress_percent, message, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    stage.value,
                    status,
                    progress_percent,
                    message,
                    datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()

    async def checkpoint(self, state: ExecutionState) -> None:
        """
        Save a checkpoint for recovery.

        Args:
            state: Current execution state
        """
        state.checkpoint_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": state.current_stage.value,
            "progress": state.overall_progress,
        }

        await self.update_state(state)

        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO checkpoints
                    (event_id, stage, checkpoint_data, created_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    state.event_id,
                    state.current_stage.value,
                    json.dumps(state.to_dict()),
                    datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()

        logger.info(f"Created checkpoint for {state.event_id} at stage {state.current_stage.value}")

    async def restore_checkpoint(self, event_id: str) -> Optional[ExecutionState]:
        """
        Restore execution state from latest checkpoint.

        Args:
            event_id: Event identifier

        Returns:
            Restored ExecutionState if checkpoint exists
        """
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT checkpoint_data FROM checkpoints
                    WHERE event_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (event_id,))
                row = cursor.fetchone()

                if row:
                    state = ExecutionState.from_dict(json.loads(row["checkpoint_data"]))
                    logger.info(f"Restored checkpoint for {event_id}")
                    return state
                return None

    async def record_error(
        self,
        event_id: str,
        stage: ExecutionStage,
        error: Exception,
        stack_trace: Optional[str] = None
    ) -> None:
        """
        Record an error in the error log.

        Args:
            event_id: Event identifier
            stage: Stage where error occurred
            error: Exception that occurred
            stack_trace: Optional stack trace
        """
        state = await self.get_state(event_id)
        if state:
            error_info = {
                "stage": stage.value,
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state.error_summary.append(error_info)

            # Update stage errors
            if stage.value in state.stages:
                state.stages[stage.value].errors.append(str(error))

            await self.update_state(state)

        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO error_log
                    (event_id, stage, error_type, error_message, stack_trace, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    event_id,
                    stage.value,
                    type(error).__name__,
                    str(error),
                    stack_trace,
                    datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()

        logger.error(f"Recorded error for {event_id} at {stage.value}: {error}")

    async def set_degraded_mode(
        self,
        event_id: str,
        level: DegradedModeLevel,
        reason: str,
        fallback: Optional[str] = None,
        confidence_impact: float = 0.0
    ) -> Optional[ExecutionState]:
        """
        Set degraded mode for an execution.

        Args:
            event_id: Event identifier
            level: Degradation level
            reason: Reason for degradation
            fallback: Fallback strategy used (if any)
            confidence_impact: Impact on confidence (0-1)

        Returns:
            Updated ExecutionState
        """
        state = await self.get_state(event_id)
        if not state:
            return None

        # Only upgrade degradation level, never downgrade
        if level.value > state.degraded_mode.level.value or state.degraded_mode.level == DegradedModeLevel.NONE:
            state.degraded_mode.level = level
            state.degraded_mode.active_since = datetime.now(timezone.utc)

        state.degraded_mode.reasons.append(reason)
        state.degraded_mode.confidence_impact = max(
            state.degraded_mode.confidence_impact,
            confidence_impact
        )

        if fallback:
            state.degraded_mode.fallbacks_used.append(fallback)

        await self.update_state(state)

        logger.warning(f"Degraded mode set for {event_id}: {level.value} - {reason}")
        return state

    async def list_active_executions(self) -> List[ExecutionState]:
        """Get all non-terminal execution states."""
        terminal_stages = ["completed", "failed", "cancelled"]

        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT state_json FROM execution_states
                    WHERE current_stage NOT IN (?, ?, ?)
                """, tuple(terminal_stages))

                results = []
                for row in cursor.fetchall():
                    results.append(ExecutionState.from_dict(json.loads(row["state_json"])))
                return results

    async def get_stage_history(
        self,
        event_id: str,
        stage: Optional[ExecutionStage] = None
    ) -> List[Dict[str, Any]]:
        """Get stage transition history for an event."""
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                if stage:
                    cursor.execute("""
                        SELECT * FROM stage_history
                        WHERE event_id = ? AND stage = ?
                        ORDER BY timestamp
                    """, (event_id, stage.value))
                else:
                    cursor.execute("""
                        SELECT * FROM stage_history
                        WHERE event_id = ?
                        ORDER BY timestamp
                    """, (event_id,))

                return [dict(row) for row in cursor.fetchall()]

    async def get_error_history(self, event_id: str) -> List[Dict[str, Any]]:
        """Get error history for an event."""
        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM error_log
                    WHERE event_id = ?
                    ORDER BY timestamp
                """, (event_id,))
                return [dict(row) for row in cursor.fetchall()]

    async def cleanup_old_states(self, max_age_days: int = 30) -> int:
        """
        Remove old completed execution states.

        Args:
            max_age_days: Maximum age in days for completed states

        Returns:
            Number of states removed
        """
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()

        async with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get events to delete
                cursor.execute("""
                    SELECT event_id FROM execution_states
                    WHERE current_stage IN ('completed', 'failed', 'cancelled')
                    AND completed_at < ?
                """, (cutoff,))

                event_ids = [row["event_id"] for row in cursor.fetchall()]

                if event_ids:
                    placeholders = ",".join("?" * len(event_ids))

                    # Delete from all tables
                    cursor.execute(f"DELETE FROM error_log WHERE event_id IN ({placeholders})", event_ids)
                    cursor.execute(f"DELETE FROM stage_history WHERE event_id IN ({placeholders})", event_ids)
                    cursor.execute(f"DELETE FROM checkpoints WHERE event_id IN ({placeholders})", event_ids)
                    cursor.execute(f"DELETE FROM execution_states WHERE event_id IN ({placeholders})", event_ids)

                    conn.commit()

                logger.info(f"Cleaned up {len(event_ids)} old execution states")
                return len(event_ids)
