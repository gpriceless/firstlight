"""
Local Filesystem Storage Backend.

Provides organized workspace management for local processing:
- Structured directory layouts
- Disk space estimation and monitoring
- Temporary file cleanup
- Offline operation support

Directory Structure:
    ./raw/           - Downloaded raw data
    ./normalized/    - Ingested/normalized data
    ./intermediate/  - Processing artifacts
    ./results/       - Analysis results
    ./products/      - Final output products
    ./state/         - Workflow state files
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class WorkspaceSubdir(Enum):
    """Standard workspace subdirectories."""

    RAW = "raw"
    NORMALIZED = "normalized"
    INTERMEDIATE = "intermediate"
    RESULTS = "results"
    PRODUCTS = "products"
    STATE = "state"


@dataclass
class WorkspaceConfig:
    """
    Configuration for workspace creation.

    Attributes:
        base_path: Root directory for workspace
        workflow_id: Optional workflow identifier
        create_subdirs: Whether to create standard subdirectories
        max_size_gb: Maximum workspace size in GB (0 = unlimited)
        cleanup_on_exit: Clean up temporary files on exit
        preserve_raw: Preserve raw data on cleanup
        preserve_products: Preserve products on cleanup
    """

    base_path: Union[str, Path]
    workflow_id: Optional[str] = None
    create_subdirs: bool = True
    max_size_gb: float = 0.0
    cleanup_on_exit: bool = False
    preserve_raw: bool = True
    preserve_products: bool = True

    def __post_init__(self) -> None:
        """Convert base_path to Path."""
        self.base_path = Path(self.base_path)


@dataclass
class DiskUsage:
    """
    Disk usage statistics.

    Attributes:
        total_bytes: Total disk space
        used_bytes: Used disk space
        free_bytes: Free disk space
        workspace_bytes: Space used by workspace
    """

    total_bytes: int
    used_bytes: int
    free_bytes: int
    workspace_bytes: int = 0

    @property
    def total_gb(self) -> float:
        """Total space in GB."""
        return self.total_bytes / (1024**3)

    @property
    def used_gb(self) -> float:
        """Used space in GB."""
        return self.used_bytes / (1024**3)

    @property
    def free_gb(self) -> float:
        """Free space in GB."""
        return self.free_bytes / (1024**3)

    @property
    def workspace_gb(self) -> float:
        """Workspace size in GB."""
        return self.workspace_bytes / (1024**3)

    @property
    def usage_percent(self) -> float:
        """Usage percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "workspace_gb": round(self.workspace_gb, 2),
            "usage_percent": round(self.usage_percent, 1),
        }


@dataclass
class FileEntry:
    """
    Entry for a file in the workspace.

    Attributes:
        path: File path
        size_bytes: File size in bytes
        modified_at: Last modification time
        checksum: Optional file checksum
        category: Workspace subdirectory category
    """

    path: Path
    size_bytes: int
    modified_at: datetime
    checksum: Optional[str] = None
    category: Optional[WorkspaceSubdir] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "size_mb": round(self.size_bytes / (1024 * 1024), 2),
            "modified_at": self.modified_at.isoformat(),
            "checksum": self.checksum,
            "category": self.category.value if self.category else None,
        }


class LocalWorkspace:
    """
    Manages a local workspace directory for processing.

    Provides organized storage with automatic directory structure,
    disk usage tracking, and cleanup utilities.
    """

    def __init__(self, config: WorkspaceConfig) -> None:
        """
        Initialize workspace.

        Args:
            config: Workspace configuration
        """
        self.config = config
        self.base_path = config.base_path
        self.workflow_id = config.workflow_id

        # Initialize directory paths
        self._subdirs: Dict[WorkspaceSubdir, Path] = {}
        for subdir in WorkspaceSubdir:
            self._subdirs[subdir] = self.base_path / subdir.value

        # File tracking
        self._tracked_files: Dict[str, FileEntry] = {}
        self._manifest_path = self.base_path / "manifest.json"

        # Create workspace
        if config.create_subdirs:
            self._create_directories()
            self._load_manifest()

    def _create_directories(self) -> None:
        """Create workspace directory structure."""
        self.base_path.mkdir(parents=True, exist_ok=True)

        for subdir_path in self._subdirs.values():
            subdir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created workspace at: {self.base_path}")

    def _load_manifest(self) -> None:
        """Load workspace manifest if exists."""
        if self._manifest_path.exists():
            try:
                with open(self._manifest_path, "r") as f:
                    data = json.load(f)
                    for entry_data in data.get("files", []):
                        entry = FileEntry(
                            path=Path(entry_data["path"]),
                            size_bytes=entry_data["size_bytes"],
                            modified_at=datetime.fromisoformat(
                                entry_data["modified_at"]
                            ),
                            checksum=entry_data.get("checksum"),
                            category=WorkspaceSubdir(entry_data["category"])
                            if entry_data.get("category")
                            else None,
                        )
                        self._tracked_files[str(entry.path)] = entry
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")

    def _save_manifest(self) -> None:
        """Save workspace manifest."""
        data = {
            "workflow_id": self.workflow_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": [entry.to_dict() for entry in self._tracked_files.values()],
        }
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_path(self, category: WorkspaceSubdir) -> Path:
        """
        Get path for a workspace category.

        Args:
            category: Workspace subdirectory

        Returns:
            Path to the subdirectory
        """
        return self._subdirs[category]

    @property
    def raw_path(self) -> Path:
        """Path to raw data directory."""
        return self._subdirs[WorkspaceSubdir.RAW]

    @property
    def normalized_path(self) -> Path:
        """Path to normalized data directory."""
        return self._subdirs[WorkspaceSubdir.NORMALIZED]

    @property
    def intermediate_path(self) -> Path:
        """Path to intermediate artifacts directory."""
        return self._subdirs[WorkspaceSubdir.INTERMEDIATE]

    @property
    def results_path(self) -> Path:
        """Path to results directory."""
        return self._subdirs[WorkspaceSubdir.RESULTS]

    @property
    def products_path(self) -> Path:
        """Path to final products directory."""
        return self._subdirs[WorkspaceSubdir.PRODUCTS]

    @property
    def state_path(self) -> Path:
        """Path to state files directory."""
        return self._subdirs[WorkspaceSubdir.STATE]

    def add_file(
        self,
        file_path: Union[str, Path],
        category: WorkspaceSubdir,
        compute_checksum: bool = False,
    ) -> FileEntry:
        """
        Add/register a file in the workspace.

        Args:
            file_path: Path to file
            category: Workspace category
            compute_checksum: Compute SHA-256 checksum

        Returns:
            FileEntry for the file
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stat = file_path.stat()
        checksum = None

        if compute_checksum:
            checksum = self._compute_checksum(file_path)

        entry = FileEntry(
            path=file_path,
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
            checksum=checksum,
            category=category,
        )

        self._tracked_files[str(file_path)] = entry
        self._save_manifest()

        return entry

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get_file(self, file_path: Union[str, Path]) -> Optional[FileEntry]:
        """
        Get file entry by path.

        Args:
            file_path: Path to file

        Returns:
            FileEntry if tracked
        """
        return self._tracked_files.get(str(file_path))

    def list_files(
        self,
        category: Optional[WorkspaceSubdir] = None,
    ) -> List[FileEntry]:
        """
        List files in workspace.

        Args:
            category: Filter by category

        Returns:
            List of FileEntry objects
        """
        entries = list(self._tracked_files.values())

        if category:
            entries = [e for e in entries if e.category == category]

        return entries

    def iter_files(
        self,
        category: Optional[WorkspaceSubdir] = None,
        pattern: str = "*",
    ) -> Generator[Path, None, None]:
        """
        Iterate over files in workspace.

        Args:
            category: Filter by category
            pattern: Glob pattern

        Yields:
            File paths matching criteria
        """
        if category:
            search_path = self._subdirs[category]
            for path in search_path.rglob(pattern):
                if path.is_file():
                    yield path
        else:
            for subdir_path in self._subdirs.values():
                for path in subdir_path.rglob(pattern):
                    if path.is_file():
                        yield path

    def calculate_disk_usage(self) -> DiskUsage:
        """
        Calculate current disk usage.

        Returns:
            DiskUsage statistics
        """
        # Get filesystem usage
        disk = shutil.disk_usage(self.base_path)

        # Calculate workspace size
        workspace_bytes = 0
        for subdir_path in self._subdirs.values():
            workspace_bytes += self._get_directory_size(subdir_path)

        return DiskUsage(
            total_bytes=disk.total,
            used_bytes=disk.used,
            free_bytes=disk.free,
            workspace_bytes=workspace_bytes,
        )

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory contents."""
        total = 0
        if path.exists():
            for entry in path.rglob("*"):
                if entry.is_file():
                    total += entry.stat().st_size
        return total

    def estimate_space_needed(
        self,
        input_size_mb: float,
        processing_factor: float = 3.0,
    ) -> float:
        """
        Estimate disk space needed for processing.

        Args:
            input_size_mb: Size of input data in MB
            processing_factor: Multiplier for intermediate data

        Returns:
            Estimated space needed in MB
        """
        # Raw data + normalized + intermediate + results
        # Intermediate can be 2-3x input size during processing
        estimated_mb = input_size_mb * (1 + 1 + processing_factor + 0.5)
        return estimated_mb

    def has_space(self, needed_mb: float) -> bool:
        """
        Check if workspace has sufficient free space.

        Args:
            needed_mb: Space needed in MB

        Returns:
            True if sufficient space available
        """
        usage = self.calculate_disk_usage()
        free_mb = usage.free_bytes / (1024 * 1024)

        # Also check workspace size limit
        if self.config.max_size_gb > 0:
            max_bytes = self.config.max_size_gb * (1024**3)
            remaining = max_bytes - usage.workspace_bytes
            return remaining >= (needed_mb * 1024 * 1024)

        return free_mb >= needed_mb

    def cleanup_temporary(
        self,
        preserve_categories: Optional[Set[WorkspaceSubdir]] = None,
    ) -> int:
        """
        Clean up temporary/intermediate files.

        Args:
            preserve_categories: Categories to preserve

        Returns:
            Number of bytes freed
        """
        if preserve_categories is None:
            preserve_categories = set()

        # Always preserve what config says
        if self.config.preserve_raw:
            preserve_categories.add(WorkspaceSubdir.RAW)
        if self.config.preserve_products:
            preserve_categories.add(WorkspaceSubdir.PRODUCTS)

        # Also preserve state
        preserve_categories.add(WorkspaceSubdir.STATE)

        freed = 0
        for category, path in self._subdirs.items():
            if category in preserve_categories:
                continue

            category_size = self._get_directory_size(path)
            if category_size > 0:
                for item in path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                freed += category_size
                logger.info(
                    f"Cleaned {category.value}: {category_size / (1024*1024):.1f} MB"
                )

        # Update manifest
        self._tracked_files = {
            k: v
            for k, v in self._tracked_files.items()
            if v.category in preserve_categories
        }
        self._save_manifest()

        return freed

    def cleanup_old_files(
        self,
        max_age_hours: int = 24,
        categories: Optional[List[WorkspaceSubdir]] = None,
    ) -> int:
        """
        Clean up files older than specified age.

        Args:
            max_age_hours: Maximum file age in hours
            categories: Categories to clean (default: intermediate only)

        Returns:
            Number of bytes freed
        """
        if categories is None:
            categories = [WorkspaceSubdir.INTERMEDIATE]

        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        freed = 0

        for category in categories:
            path = self._subdirs[category]
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff:
                    size = file_path.stat().st_size
                    file_path.unlink()
                    freed += size

                    # Remove from tracking
                    key = str(file_path)
                    if key in self._tracked_files:
                        del self._tracked_files[key]

        if freed > 0:
            self._save_manifest()

        return freed

    def get_usage_by_category(self) -> Dict[str, float]:
        """
        Get disk usage breakdown by category.

        Returns:
            Dictionary of category -> size in MB
        """
        usage = {}
        for category, path in self._subdirs.items():
            size_bytes = self._get_directory_size(path)
            usage[category.value] = size_bytes / (1024 * 1024)
        return usage

    def verify_integrity(self) -> List[str]:
        """
        Verify integrity of tracked files.

        Returns:
            List of issues found
        """
        issues = []

        for key, entry in list(self._tracked_files.items()):
            if not entry.path.exists():
                issues.append(f"Missing file: {entry.path}")
                del self._tracked_files[key]
                continue

            if entry.checksum:
                current_checksum = self._compute_checksum(entry.path)
                if current_checksum != entry.checksum:
                    issues.append(f"Checksum mismatch: {entry.path}")

        if issues:
            self._save_manifest()

        return issues

    def close(self) -> None:
        """
        Close workspace and optionally cleanup.
        """
        if self.config.cleanup_on_exit:
            self.cleanup_temporary()

        self._save_manifest()
        logger.info(f"Closed workspace: {self.base_path}")


class StorageManager:
    """
    Manager for multiple workspaces and storage operations.

    Provides high-level storage management including:
    - Creating and managing workspaces
    - Disk usage monitoring
    - Space estimation and allocation
    - Cleanup coordination
    """

    def __init__(
        self,
        base_storage_path: Union[str, Path],
        default_max_size_gb: float = 50.0,
    ) -> None:
        """
        Initialize storage manager.

        Args:
            base_storage_path: Root path for all workspaces
            default_max_size_gb: Default max size per workspace
        """
        self.base_path = Path(base_storage_path)
        self.default_max_size_gb = default_max_size_gb
        self._workspaces: Dict[str, LocalWorkspace] = {}

        # Create base directory
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_workspace(
        self,
        workflow_id: str,
        max_size_gb: Optional[float] = None,
        cleanup_on_exit: bool = False,
    ) -> LocalWorkspace:
        """
        Create a new workspace for a workflow.

        Args:
            workflow_id: Unique workflow identifier
            max_size_gb: Maximum workspace size
            cleanup_on_exit: Clean up on close

        Returns:
            Created LocalWorkspace
        """
        workspace_path = self.base_path / workflow_id

        config = WorkspaceConfig(
            base_path=workspace_path,
            workflow_id=workflow_id,
            max_size_gb=max_size_gb or self.default_max_size_gb,
            cleanup_on_exit=cleanup_on_exit,
        )

        workspace = LocalWorkspace(config)
        self._workspaces[workflow_id] = workspace

        logger.info(f"Created workspace for workflow: {workflow_id}")
        return workspace

    def get_workspace(self, workflow_id: str) -> Optional[LocalWorkspace]:
        """
        Get existing workspace.

        Args:
            workflow_id: Workflow identifier

        Returns:
            LocalWorkspace if exists
        """
        if workflow_id in self._workspaces:
            return self._workspaces[workflow_id]

        # Try to load from disk
        workspace_path = self.base_path / workflow_id
        if workspace_path.exists():
            config = WorkspaceConfig(
                base_path=workspace_path,
                workflow_id=workflow_id,
                create_subdirs=False,  # Already exists
            )
            workspace = LocalWorkspace(config)
            self._workspaces[workflow_id] = workspace
            return workspace

        return None

    def list_workspaces(self) -> List[str]:
        """
        List all workspace IDs.

        Returns:
            List of workflow IDs with workspaces
        """
        workspace_ids = []
        for path in self.base_path.iterdir():
            if path.is_dir() and not path.name.startswith("."):
                workspace_ids.append(path.name)
        return workspace_ids

    def delete_workspace(self, workflow_id: str, force: bool = False) -> bool:
        """
        Delete a workspace.

        Args:
            workflow_id: Workflow identifier
            force: Force deletion even with products

        Returns:
            True if deleted
        """
        workspace_path = self.base_path / workflow_id

        if not workspace_path.exists():
            return False

        # Check for products
        products_path = workspace_path / "products"
        if products_path.exists() and any(products_path.iterdir()) and not force:
            logger.warning(
                f"Workspace {workflow_id} has products. Use force=True to delete."
            )
            return False

        # Remove from cache
        if workflow_id in self._workspaces:
            del self._workspaces[workflow_id]

        # Delete directory
        shutil.rmtree(workspace_path)
        logger.info(f"Deleted workspace: {workflow_id}")
        return True

    def get_total_disk_usage(self) -> DiskUsage:
        """
        Get total disk usage across all workspaces.

        Returns:
            DiskUsage statistics
        """
        disk = shutil.disk_usage(self.base_path)

        total_workspace_bytes = 0
        for workspace_id in self.list_workspaces():
            workspace = self.get_workspace(workspace_id)
            if workspace:
                usage = workspace.calculate_disk_usage()
                total_workspace_bytes += usage.workspace_bytes

        return DiskUsage(
            total_bytes=disk.total,
            used_bytes=disk.used,
            free_bytes=disk.free,
            workspace_bytes=total_workspace_bytes,
        )

    def estimate_space_needed(
        self,
        input_files: List[Union[str, Path]],
        processing_factor: float = 3.0,
    ) -> float:
        """
        Estimate space needed for input files.

        Args:
            input_files: List of input file paths
            processing_factor: Multiplier for processing

        Returns:
            Estimated space in MB
        """
        total_input_mb = 0.0
        for file_path in input_files:
            path = Path(file_path)
            if path.exists():
                total_input_mb += path.stat().st_size / (1024 * 1024)

        return total_input_mb * (1 + 1 + processing_factor + 0.5)

    def allocate_space(
        self,
        workflow_id: str,
        needed_mb: float,
    ) -> Tuple[bool, Optional[LocalWorkspace]]:
        """
        Allocate space for a workflow.

        Args:
            workflow_id: Workflow identifier
            needed_mb: Space needed in MB

        Returns:
            Tuple of (success, workspace)
        """
        usage = self.get_total_disk_usage()
        free_mb = usage.free_bytes / (1024 * 1024)

        # Check if we have space
        if free_mb < needed_mb:
            logger.warning(
                f"Insufficient space: need {needed_mb:.1f}MB, "
                f"have {free_mb:.1f}MB free"
            )
            return False, None

        # Create workspace with size limit
        workspace = self.create_workspace(
            workflow_id,
            max_size_gb=needed_mb / 1024 * 1.5,  # 50% buffer
        )

        return True, workspace

    def cleanup_all_temporary(self) -> int:
        """
        Clean up temporary files across all workspaces.

        Returns:
            Total bytes freed
        """
        total_freed = 0

        for workspace_id in self.list_workspaces():
            workspace = self.get_workspace(workspace_id)
            if workspace:
                freed = workspace.cleanup_temporary()
                total_freed += freed

        logger.info(f"Total cleanup: {total_freed / (1024*1024):.1f} MB freed")
        return total_freed

    def cleanup_old_workspaces(
        self,
        max_age_days: int = 7,
        preserve_with_products: bool = True,
    ) -> int:
        """
        Clean up old workspaces.

        Args:
            max_age_days: Maximum workspace age
            preserve_with_products: Keep workspaces with products

        Returns:
            Number of workspaces deleted
        """
        deleted = 0
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)

        for workspace_id in self.list_workspaces():
            workspace_path = self.base_path / workspace_id
            manifest_path = workspace_path / "manifest.json"

            # Check age from manifest or directory mtime
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r") as f:
                        data = json.load(f)
                        created_str = data.get("created_at", "")
                        if created_str:
                            created = datetime.fromisoformat(created_str)
                            if created.timestamp() < cutoff:
                                if self.delete_workspace(
                                    workspace_id, force=not preserve_with_products
                                ):
                                    deleted += 1
                                    continue
                except Exception:
                    pass

            # Fall back to directory mtime
            if workspace_path.stat().st_mtime < cutoff:
                if self.delete_workspace(
                    workspace_id, force=not preserve_with_products
                ):
                    deleted += 1

        return deleted

    def get_storage_summary(self) -> Dict[str, Any]:
        """
        Get storage summary.

        Returns:
            Summary dictionary
        """
        usage = self.get_total_disk_usage()
        workspaces = self.list_workspaces()

        workspace_sizes = {}
        for wid in workspaces:
            workspace = self.get_workspace(wid)
            if workspace:
                ws_usage = workspace.calculate_disk_usage()
                workspace_sizes[wid] = round(ws_usage.workspace_gb, 2)

        return {
            "base_path": str(self.base_path),
            "total_gb": round(usage.total_gb, 2),
            "used_gb": round(usage.used_gb, 2),
            "free_gb": round(usage.free_gb, 2),
            "workspace_total_gb": round(usage.workspace_gb, 2),
            "workspace_count": len(workspaces),
            "workspaces": workspace_sizes,
        }


# Module-level convenience functions


def create_workspace(
    base_path: Union[str, Path],
    workflow_id: Optional[str] = None,
) -> LocalWorkspace:
    """
    Create a new workspace.

    Args:
        base_path: Base directory path
        workflow_id: Optional workflow ID

    Returns:
        LocalWorkspace instance
    """
    config = WorkspaceConfig(
        base_path=base_path,
        workflow_id=workflow_id,
    )
    return LocalWorkspace(config)


def get_storage_manager(
    base_path: Union[str, Path],
    default_max_size_gb: float = 50.0,
) -> StorageManager:
    """
    Get storage manager instance.

    Args:
        base_path: Base storage path
        default_max_size_gb: Default max workspace size

    Returns:
        StorageManager instance
    """
    return StorageManager(base_path, default_max_size_gb)
