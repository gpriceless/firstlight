"""
Workflow State Persistence for Resumable Processing.

Provides state management for interruptible and resumable workflows:
- Save state to JSON or SQLite
- Load and resume from checkpoints
- Track progress through stages
- Handle intermediate results locations

Supports offline operation and crash recovery.
"""

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Workflow processing stages."""

    INITIALIZED = "initialized"  # Workflow created but not started
    DISCOVERY = "discovery"  # Finding available data sources
    INGEST = "ingest"  # Downloading and normalizing data
    ANALYZE = "analyze"  # Running analysis algorithms
    VALIDATE = "validate"  # Quality control and validation
    EXPORT = "export"  # Generating output products
    COMPLETED = "completed"  # Workflow finished successfully
    FAILED = "failed"  # Workflow failed


class StepStatus(Enum):
    """Status of individual workflow steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """
    Individual step within a workflow.

    Attributes:
        step_id: Unique identifier for the step
        name: Human-readable step name
        stage: Workflow stage this step belongs to
        status: Current status of the step
        started_at: When the step started
        completed_at: When the step completed
        error: Error message if failed
        output_path: Path to step outputs
        metadata: Additional step metadata
    """

    step_id: str
    name: str
    stage: WorkflowStage
    status: StepStatus = StepStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "output_path": self.output_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowStep":
        """Create from dictionary."""
        started_at = None
        completed_at = None

        if data.get("started_at"):
            started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            completed_at = datetime.fromisoformat(data["completed_at"])

        return cls(
            step_id=data["step_id"],
            name=data["name"],
            stage=WorkflowStage(data["stage"]),
            status=StepStatus(data.get("status", "pending")),
            started_at=started_at,
            completed_at=completed_at,
            error=data.get("error"),
            output_path=data.get("output_path"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Checkpoint:
    """
    Checkpoint for workflow recovery.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        workflow_id: Associated workflow
        stage: Stage at checkpoint
        step_id: Last completed step
        created_at: When checkpoint was created
        data_paths: Paths to intermediate data
        state_hash: Hash of state at checkpoint
    """

    checkpoint_id: str
    workflow_id: str
    stage: WorkflowStage
    step_id: Optional[str]
    created_at: datetime
    data_paths: Dict[str, str] = field(default_factory=dict)
    state_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "workflow_id": self.workflow_id,
            "stage": self.stage.value,
            "step_id": self.step_id,
            "created_at": self.created_at.isoformat(),
            "data_paths": self.data_paths,
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            workflow_id=data["workflow_id"],
            stage=WorkflowStage(data["stage"]),
            step_id=data.get("step_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            data_paths=data.get("data_paths", {}),
            state_hash=data.get("state_hash", ""),
        )


@dataclass
class WorkflowState:
    """
    Complete workflow state for persistence and recovery.

    Attributes:
        workflow_id: Unique workflow identifier
        stage: Current workflow stage
        progress_percent: Overall progress (0-100)
        completed_steps: List of completed step IDs
        pending_steps: List of pending step IDs
        last_checkpoint: ID of last checkpoint
        error: Error message if failed
        created_at: When workflow was created
        updated_at: Last state update time
        steps: All workflow steps
        checkpoints: All checkpoints
        config: Workflow configuration
        results: Step results/outputs
    """

    workflow_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZED
    progress_percent: float = 0.0
    completed_steps: List[str] = field(default_factory=list)
    pending_steps: List[str] = field(default_factory=list)
    last_checkpoint: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    steps: Dict[str, WorkflowStep] = field(default_factory=dict)
    checkpoints: Dict[str, Checkpoint] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize state tracking."""
        self._lock = threading.RLock()

    def add_step(self, step: WorkflowStep) -> None:
        """
        Add a step to the workflow.

        Args:
            step: WorkflowStep to add
        """
        with self._lock:
            self.steps[step.step_id] = step
            if step.status == StepStatus.PENDING:
                if step.step_id not in self.pending_steps:
                    self.pending_steps.append(step.step_id)
            self.updated_at = datetime.now(timezone.utc)

    def get_step(self, step_id: str) -> Optional[WorkflowStep]:
        """
        Get a step by ID.

        Args:
            step_id: Step identifier

        Returns:
            WorkflowStep if found
        """
        return self.steps.get(step_id)

    def start_step(self, step_id: str) -> bool:
        """
        Mark a step as started.

        Args:
            step_id: Step to start

        Returns:
            True if step was started
        """
        with self._lock:
            step = self.steps.get(step_id)
            if not step:
                return False

            step.status = StepStatus.RUNNING
            step.started_at = datetime.now(timezone.utc)
            self.updated_at = datetime.now(timezone.utc)
            return True

    def complete_step(
        self,
        step_id: str,
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a step as completed.

        Args:
            step_id: Step to complete
            output_path: Path to step outputs
            metadata: Additional metadata

        Returns:
            True if step was completed
        """
        with self._lock:
            step = self.steps.get(step_id)
            if not step:
                return False

            step.status = StepStatus.COMPLETED
            step.completed_at = datetime.now(timezone.utc)
            if output_path:
                step.output_path = output_path
            if metadata:
                step.metadata.update(metadata)

            # Update tracking lists
            if step_id in self.pending_steps:
                self.pending_steps.remove(step_id)
            if step_id not in self.completed_steps:
                self.completed_steps.append(step_id)

            self._update_progress()
            self.updated_at = datetime.now(timezone.utc)
            return True

    def fail_step(self, step_id: str, error: str) -> bool:
        """
        Mark a step as failed.

        Args:
            step_id: Step that failed
            error: Error message

        Returns:
            True if step was marked failed
        """
        with self._lock:
            step = self.steps.get(step_id)
            if not step:
                return False

            step.status = StepStatus.FAILED
            step.error = error
            step.completed_at = datetime.now(timezone.utc)

            if step_id in self.pending_steps:
                self.pending_steps.remove(step_id)

            self.error = f"Step {step_id} failed: {error}"
            self.updated_at = datetime.now(timezone.utc)
            return True

    def _update_progress(self) -> None:
        """Update progress percentage based on completed steps."""
        total = len(self.steps)
        if total == 0:
            self.progress_percent = 0.0
            return

        completed = len(self.completed_steps)
        self.progress_percent = (completed / total) * 100.0

    def update_stage(self, stage: WorkflowStage) -> None:
        """
        Update the workflow stage.

        Args:
            stage: New workflow stage
        """
        with self._lock:
            self.stage = stage
            self.updated_at = datetime.now(timezone.utc)
            logger.info(f"Workflow {self.workflow_id} moved to stage: {stage.value}")

    def create_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        data_paths: Optional[Dict[str, str]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint of current state.

        Args:
            checkpoint_id: Optional checkpoint ID (auto-generated if None)
            data_paths: Paths to intermediate data

        Returns:
            Created Checkpoint
        """
        with self._lock:
            if checkpoint_id is None:
                checkpoint_id = f"ckpt_{self.workflow_id}_{len(self.checkpoints)}"

            # Get last completed step
            last_step_id = self.completed_steps[-1] if self.completed_steps else None

            # Calculate state hash
            state_hash = self._calculate_state_hash()

            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                workflow_id=self.workflow_id,
                stage=self.stage,
                step_id=last_step_id,
                created_at=datetime.now(timezone.utc),
                data_paths=data_paths or {},
                state_hash=state_hash,
            )

            self.checkpoints[checkpoint_id] = checkpoint
            self.last_checkpoint = checkpoint_id
            self.updated_at = datetime.now(timezone.utc)

            logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint

    def _calculate_state_hash(self) -> str:
        """Calculate hash of current state."""
        state_data = {
            "workflow_id": self.workflow_id,
            "stage": self.stage.value,
            "completed_steps": sorted(self.completed_steps),
            "results": json.dumps(self.results, sort_keys=True, default=str),
        }
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    def get_resume_point(self) -> Optional[Checkpoint]:
        """
        Get the checkpoint to resume from.

        Returns:
            Last checkpoint if available
        """
        if self.last_checkpoint and self.last_checkpoint in self.checkpoints:
            return self.checkpoints[self.last_checkpoint]
        return None

    def is_completed(self) -> bool:
        """Check if workflow is completed."""
        return self.stage == WorkflowStage.COMPLETED

    def is_failed(self) -> bool:
        """Check if workflow has failed."""
        return self.stage == WorkflowStage.FAILED

    def is_resumable(self) -> bool:
        """Check if workflow can be resumed."""
        return (
            self.last_checkpoint is not None
            and self.stage not in (WorkflowStage.COMPLETED, WorkflowStage.FAILED)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "workflow_id": self.workflow_id,
            "stage": self.stage.value,
            "progress_percent": self.progress_percent,
            "completed_steps": self.completed_steps,
            "pending_steps": self.pending_steps,
            "last_checkpoint": self.last_checkpoint,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "checkpoints": {k: v.to_dict() for k, v in self.checkpoints.items()},
            "config": self.config,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowState":
        """Create from dictionary."""
        state = cls(
            workflow_id=data["workflow_id"],
            stage=WorkflowStage(data.get("stage", "initialized")),
            progress_percent=data.get("progress_percent", 0.0),
            completed_steps=data.get("completed_steps", []),
            pending_steps=data.get("pending_steps", []),
            last_checkpoint=data.get("last_checkpoint"),
            error=data.get("error"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            config=data.get("config", {}),
            results=data.get("results", {}),
        )

        # Load steps
        for step_data in data.get("steps", {}).values():
            step = WorkflowStep.from_dict(step_data)
            state.steps[step.step_id] = step

        # Load checkpoints
        for ckpt_data in data.get("checkpoints", {}).values():
            ckpt = Checkpoint.from_dict(ckpt_data)
            state.checkpoints[ckpt.checkpoint_id] = ckpt

        return state


class StateManager:
    """
    Manages workflow state persistence.

    Supports both JSON file and SQLite storage backends.
    """

    def __init__(
        self,
        storage_path: Union[str, Path],
        use_sqlite: bool = False,
    ) -> None:
        """
        Initialize state manager.

        Args:
            storage_path: Directory for state storage
            use_sqlite: Use SQLite instead of JSON files
        """
        self.storage_path = Path(storage_path)
        self.use_sqlite = use_sqlite
        self._states: Dict[str, WorkflowState] = {}
        self._lock = threading.RLock()

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite if needed
        if use_sqlite:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        db_path = self.storage_path / "workflow_state.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                workflow_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (workflow_id) REFERENCES workflows(workflow_id)
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoints_workflow
            ON checkpoints(workflow_id)
        """
        )

        conn.commit()
        conn.close()

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """Get SQLite connection."""
        db_path = self.storage_path / "workflow_state.db"
        return sqlite3.connect(str(db_path))

    def _get_state_file_path(self, workflow_id: str) -> Path:
        """Get JSON state file path for workflow."""
        return self.storage_path / f"{workflow_id}.state.json"

    def save_state(self, state: WorkflowState) -> None:
        """
        Save workflow state.

        Args:
            state: WorkflowState to save
        """
        with self._lock:
            self._states[state.workflow_id] = state

            if self.use_sqlite:
                self._save_state_sqlite(state)
            else:
                self._save_state_json(state)

            logger.debug(f"Saved state for workflow: {state.workflow_id}")

    def _save_state_json(self, state: WorkflowState) -> None:
        """Save state to JSON file."""
        file_path = self._get_state_file_path(state.workflow_id)
        state_dict = state.to_dict()

        # Write to temp file first for atomicity
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(state_dict, f, indent=2, default=str)

        # Atomic rename
        temp_path.rename(file_path)

    def _save_state_sqlite(self, state: WorkflowState) -> None:
        """Save state to SQLite."""
        conn = self._get_sqlite_connection()
        cursor = conn.cursor()

        state_json = json.dumps(state.to_dict(), default=str)
        updated_at = state.updated_at.isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO workflows (workflow_id, state_json, updated_at)
            VALUES (?, ?, ?)
        """,
            (state.workflow_id, state_json, updated_at),
        )

        conn.commit()
        conn.close()

    def load_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """
        Load workflow state.

        Args:
            workflow_id: Workflow to load

        Returns:
            WorkflowState if found
        """
        with self._lock:
            # Check memory cache first
            if workflow_id in self._states:
                return self._states[workflow_id]

            if self.use_sqlite:
                state = self._load_state_sqlite(workflow_id)
            else:
                state = self._load_state_json(workflow_id)

            if state:
                self._states[workflow_id] = state

            return state

    def _load_state_json(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load state from JSON file."""
        file_path = self._get_state_file_path(workflow_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                state_dict = json.load(f)
            return WorkflowState.from_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to load state from {file_path}: {e}")
            return None

    def _load_state_sqlite(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load state from SQLite."""
        conn = self._get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT state_json FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        try:
            state_dict = json.loads(row[0])
            return WorkflowState.from_dict(state_dict)
        except Exception as e:
            logger.error(f"Failed to parse state for {workflow_id}: {e}")
            return None

    def update_progress(
        self,
        workflow_id: str,
        progress_percent: float,
    ) -> bool:
        """
        Update workflow progress.

        Args:
            workflow_id: Workflow to update
            progress_percent: New progress percentage

        Returns:
            True if updated successfully
        """
        state = self.load_state(workflow_id)
        if not state:
            return False

        with state._lock:
            state.progress_percent = progress_percent
            state.updated_at = datetime.now(timezone.utc)

        self.save_state(state)
        return True

    def mark_step_complete(
        self,
        workflow_id: str,
        step_id: str,
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a step as complete.

        Args:
            workflow_id: Workflow ID
            step_id: Step to complete
            output_path: Path to outputs
            metadata: Additional metadata

        Returns:
            True if step was marked complete
        """
        state = self.load_state(workflow_id)
        if not state:
            return False

        result = state.complete_step(step_id, output_path, metadata)
        if result:
            self.save_state(state)

        return result

    def get_resume_point(self, workflow_id: str) -> Optional[Checkpoint]:
        """
        Get resume point for a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Checkpoint to resume from
        """
        state = self.load_state(workflow_id)
        if not state:
            return None

        return state.get_resume_point()

    def create_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: Optional[str] = None,
        data_paths: Optional[Dict[str, str]] = None,
    ) -> Optional[Checkpoint]:
        """
        Create a checkpoint for a workflow.

        Args:
            workflow_id: Workflow ID
            checkpoint_id: Optional checkpoint ID
            data_paths: Paths to intermediate data

        Returns:
            Created Checkpoint
        """
        state = self.load_state(workflow_id)
        if not state:
            return None

        checkpoint = state.create_checkpoint(checkpoint_id, data_paths)
        self.save_state(state)

        # Also save checkpoint separately for quick access
        if self.use_sqlite:
            self._save_checkpoint_sqlite(checkpoint)
        else:
            self._save_checkpoint_json(checkpoint)

        return checkpoint

    def _save_checkpoint_json(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to JSON file."""
        ckpt_dir = self.storage_path / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        file_path = ckpt_dir / f"{checkpoint.checkpoint_id}.json"
        with open(file_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, default=str)

    def _save_checkpoint_sqlite(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to SQLite."""
        conn = self._get_sqlite_connection()
        cursor = conn.cursor()

        checkpoint_json = json.dumps(checkpoint.to_dict(), default=str)
        created_at = checkpoint.created_at.isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO checkpoints
            (checkpoint_id, workflow_id, checkpoint_json, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (
                checkpoint.checkpoint_id,
                checkpoint.workflow_id,
                checkpoint_json,
                created_at,
            ),
        )

        conn.commit()
        conn.close()

    def list_workflows(self) -> List[str]:
        """
        List all workflow IDs.

        Returns:
            List of workflow IDs
        """
        if self.use_sqlite:
            return self._list_workflows_sqlite()
        return self._list_workflows_json()

    def _list_workflows_json(self) -> List[str]:
        """List workflows from JSON files."""
        workflow_ids = []
        for file_path in self.storage_path.glob("*.state.json"):
            workflow_id = file_path.stem.replace(".state", "")
            workflow_ids.append(workflow_id)
        return workflow_ids

    def _list_workflows_sqlite(self) -> List[str]:
        """List workflows from SQLite."""
        conn = self._get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT workflow_id FROM workflows")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def delete_state(self, workflow_id: str) -> bool:
        """
        Delete workflow state.

        Args:
            workflow_id: Workflow to delete

        Returns:
            True if deleted
        """
        with self._lock:
            if workflow_id in self._states:
                del self._states[workflow_id]

            if self.use_sqlite:
                return self._delete_state_sqlite(workflow_id)
            return self._delete_state_json(workflow_id)

    def _delete_state_json(self, workflow_id: str) -> bool:
        """Delete JSON state file."""
        file_path = self._get_state_file_path(workflow_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def _delete_state_sqlite(self, workflow_id: str) -> bool:
        """Delete state from SQLite."""
        conn = self._get_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM checkpoints WHERE workflow_id = ?",
            (workflow_id,),
        )
        cursor.execute(
            "DELETE FROM workflows WHERE workflow_id = ?",
            (workflow_id,),
        )

        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return deleted

    def cleanup_old_states(
        self,
        max_age_days: int = 30,
        keep_completed: bool = True,
    ) -> int:
        """
        Clean up old workflow states.

        Args:
            max_age_days: Maximum age in days
            keep_completed: Keep completed workflows

        Returns:
            Number of states deleted
        """
        deleted = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)

        for workflow_id in self.list_workflows():
            state = self.load_state(workflow_id)
            if not state:
                continue

            if keep_completed and state.is_completed():
                continue

            if state.updated_at.timestamp() < cutoff:
                if self.delete_state(workflow_id):
                    deleted += 1
                    logger.info(f"Deleted old state: {workflow_id}")

        return deleted

    def get_state_summary(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of workflow state.

        Args:
            workflow_id: Workflow ID

        Returns:
            Summary dictionary
        """
        state = self.load_state(workflow_id)
        if not state:
            return None

        return {
            "workflow_id": state.workflow_id,
            "stage": state.stage.value,
            "progress_percent": state.progress_percent,
            "completed_steps": len(state.completed_steps),
            "pending_steps": len(state.pending_steps),
            "total_steps": len(state.steps),
            "has_checkpoint": state.last_checkpoint is not None,
            "is_resumable": state.is_resumable(),
            "error": state.error,
            "created_at": state.created_at.isoformat(),
            "updated_at": state.updated_at.isoformat(),
        }


# Module-level convenience functions


def create_workflow_state(
    workflow_id: str,
    config: Optional[Dict[str, Any]] = None,
) -> WorkflowState:
    """
    Create a new workflow state.

    Args:
        workflow_id: Unique workflow ID
        config: Workflow configuration

    Returns:
        New WorkflowState
    """
    return WorkflowState(
        workflow_id=workflow_id,
        config=config or {},
    )


def get_state_manager(
    storage_path: Union[str, Path],
    use_sqlite: bool = False,
) -> StateManager:
    """
    Get a state manager instance.

    Args:
        storage_path: Directory for state storage
        use_sqlite: Use SQLite backend

    Returns:
        StateManager instance
    """
    return StateManager(storage_path, use_sqlite)
