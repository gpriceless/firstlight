"""
Local Storage Backend for Multiverse Dive.

Provides filesystem-based storage for offline operation:
- Organized workspace directories
- Disk usage tracking and estimation
- Temporary file cleanup utilities
"""

from core.data.storage.local import (
    LocalWorkspace,
    StorageManager,
    WorkspaceConfig,
    create_workspace,
    get_storage_manager,
)

__all__ = [
    "LocalWorkspace",
    "StorageManager",
    "WorkspaceConfig",
    "create_workspace",
    "get_storage_manager",
]
