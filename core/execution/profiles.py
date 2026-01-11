"""
Execution Profiles for Resource-Constrained Processing.

Provides execution profiles for different hardware configurations:
- Laptop: 2GB RAM, sequential processing, small tiles
- Workstation: 8GB RAM, parallel tiles, medium tiles
- Cloud: 32GB+ RAM, distributed processing, large tiles
- Edge: 1GB RAM, sequential processing, tiny tiles

Includes automatic resource detection and custom profile creation.
"""

import logging
import os
import platform
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProfileName(Enum):
    """Predefined execution profile names."""

    LAPTOP = "laptop"
    WORKSTATION = "workstation"
    CLOUD = "cloud"
    EDGE = "edge"
    CUSTOM = "custom"


class ProcessingMode(Enum):
    """Processing execution modes."""

    SEQUENTIAL = "sequential"  # Process tiles one at a time
    PARALLEL = "parallel"  # Process tiles in parallel threads/processes
    DISTRIBUTED = "distributed"  # Use Dask/Ray for distributed processing


@dataclass
class ExecutionProfile:
    """
    Configuration for resource-constrained execution.

    Defines memory limits, concurrency, tile sizes, and timeout settings
    for running analysis pipelines on different hardware configurations.

    Attributes:
        name: Profile identifier (laptop, workstation, cloud, edge, or custom)
        max_memory_mb: Maximum memory usage in megabytes
        max_concurrent_tiles: Maximum number of tiles to process concurrently
        tile_size: Tile dimensions (width, height) in pixels
        use_distributed: Whether to use distributed processing (Dask/Ray)
        timeout_seconds: Maximum execution time per tile in seconds
        description: Human-readable description of the profile
        gpu_enabled: Whether GPU acceleration is allowed
        memory_buffer_factor: Factor to reserve memory for overhead (0.0-1.0)
        disk_cache_mb: Maximum disk cache size in megabytes
    """

    name: str
    max_memory_mb: int
    max_concurrent_tiles: int
    tile_size: Tuple[int, int]
    use_distributed: bool
    timeout_seconds: int
    description: str = ""
    gpu_enabled: bool = False
    memory_buffer_factor: float = 0.2
    disk_cache_mb: int = 1024

    def __post_init__(self) -> None:
        """Validate profile parameters."""
        if self.max_memory_mb < 256:
            raise ValueError("max_memory_mb must be at least 256 MB")
        if self.max_concurrent_tiles < 1:
            raise ValueError("max_concurrent_tiles must be at least 1")
        if self.tile_size[0] < 32 or self.tile_size[1] < 32:
            raise ValueError("tile_size dimensions must be at least 32 pixels")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be at least 1")
        if not 0.0 <= self.memory_buffer_factor <= 1.0:
            raise ValueError("memory_buffer_factor must be between 0.0 and 1.0")

    @property
    def effective_memory_mb(self) -> int:
        """
        Calculate effective memory available after buffer.

        Returns:
            Available memory in MB after accounting for buffer
        """
        return int(self.max_memory_mb * (1.0 - self.memory_buffer_factor))

    @property
    def memory_per_tile_mb(self) -> int:
        """
        Calculate memory allocation per concurrent tile.

        Returns:
            Memory in MB available per tile
        """
        if self.max_concurrent_tiles == 0:
            return self.effective_memory_mb
        return self.effective_memory_mb // self.max_concurrent_tiles

    @property
    def processing_mode(self) -> ProcessingMode:
        """
        Determine processing mode from profile settings.

        Returns:
            Processing mode (sequential, parallel, or distributed)
        """
        if self.use_distributed:
            return ProcessingMode.DISTRIBUTED
        elif self.max_concurrent_tiles > 1:
            return ProcessingMode.PARALLEL
        return ProcessingMode.SEQUENTIAL

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert profile to dictionary.

        Returns:
            Dictionary representation of profile
        """
        return {
            "name": self.name,
            "max_memory_mb": self.max_memory_mb,
            "max_concurrent_tiles": self.max_concurrent_tiles,
            "tile_size": list(self.tile_size),
            "use_distributed": self.use_distributed,
            "timeout_seconds": self.timeout_seconds,
            "description": self.description,
            "gpu_enabled": self.gpu_enabled,
            "memory_buffer_factor": self.memory_buffer_factor,
            "disk_cache_mb": self.disk_cache_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionProfile":
        """
        Create profile from dictionary.

        Args:
            data: Dictionary with profile parameters

        Returns:
            ExecutionProfile instance
        """
        # Handle tile_size as list or tuple
        tile_size = data.get("tile_size", (256, 256))
        if isinstance(tile_size, list):
            tile_size = tuple(tile_size)

        return cls(
            name=data.get("name", "custom"),
            max_memory_mb=data.get("max_memory_mb", 2048),
            max_concurrent_tiles=data.get("max_concurrent_tiles", 1),
            tile_size=tile_size,
            use_distributed=data.get("use_distributed", False),
            timeout_seconds=data.get("timeout_seconds", 300),
            description=data.get("description", ""),
            gpu_enabled=data.get("gpu_enabled", False),
            memory_buffer_factor=data.get("memory_buffer_factor", 0.2),
            disk_cache_mb=data.get("disk_cache_mb", 1024),
        )

    def can_process_tile(self, tile_memory_mb: int) -> bool:
        """
        Check if a tile can be processed within memory constraints.

        Args:
            tile_memory_mb: Estimated memory requirement for tile

        Returns:
            True if tile can be processed
        """
        return tile_memory_mb <= self.memory_per_tile_mb

    def estimate_total_tiles(
        self,
        total_width: int,
        total_height: int,
        overlap: int = 0,
    ) -> int:
        """
        Estimate total number of tiles for an area.

        Args:
            total_width: Total width in pixels
            total_height: Total height in pixels
            overlap: Overlap in pixels between tiles

        Returns:
            Estimated number of tiles
        """
        tile_w, tile_h = self.tile_size
        step_w = tile_w - overlap
        step_h = tile_h - overlap

        if step_w <= 0 or step_h <= 0:
            raise ValueError("Overlap cannot exceed tile size")

        num_cols = max(1, (total_width + step_w - 1) // step_w)
        num_rows = max(1, (total_height + step_h - 1) // step_h)

        return num_cols * num_rows


@dataclass
class SystemResources:
    """
    Detected system resources.

    Attributes:
        total_memory_mb: Total system RAM in MB
        available_memory_mb: Available RAM in MB
        cpu_cores: Number of CPU cores
        gpu_available: Whether GPU is detected
        gpu_memory_mb: GPU memory if available
        disk_free_mb: Free disk space in MB
        platform_name: Operating system name
    """

    total_memory_mb: int
    available_memory_mb: int
    cpu_cores: int
    gpu_available: bool = False
    gpu_memory_mb: int = 0
    disk_free_mb: int = 0
    platform_name: str = ""

    @classmethod
    def detect(cls) -> "SystemResources":
        """
        Detect current system resources.

        Returns:
            SystemResources instance with detected values
        """
        total_memory_mb = 4096  # Default fallback
        available_memory_mb = 2048
        cpu_cores = 1
        gpu_available = False
        gpu_memory_mb = 0
        disk_free_mb = 0
        platform_name = platform.system()

        # Try to detect memory
        try:
            import psutil

            mem = psutil.virtual_memory()
            total_memory_mb = mem.total // (1024 * 1024)
            available_memory_mb = mem.available // (1024 * 1024)
        except ImportError:
            # psutil not available, try platform-specific detection
            try:
                if platform_name == "Linux":
                    with open("/proc/meminfo", "r") as f:
                        for line in f:
                            if line.startswith("MemTotal:"):
                                total_memory_mb = int(line.split()[1]) // 1024
                            elif line.startswith("MemAvailable:"):
                                available_memory_mb = int(line.split()[1]) // 1024
                elif platform_name == "Darwin":  # macOS
                    import subprocess

                    result = subprocess.run(
                        ["sysctl", "-n", "hw.memsize"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        total_memory_mb = int(result.stdout.strip()) // (1024 * 1024)
                        available_memory_mb = total_memory_mb // 2
            except Exception:
                pass

        # Detect CPU cores
        try:
            cpu_cores = os.cpu_count() or 1
        except Exception:
            cpu_cores = 1

        # Detect GPU (CUDA)
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                gpu_available = True
                gpu_memory_mb = int(result.stdout.strip().split("\n")[0])
        except Exception:
            pass

        # Detect disk space
        try:
            disk_usage = shutil.disk_usage(Path.home())
            disk_free_mb = disk_usage.free // (1024 * 1024)
        except Exception:
            pass

        return cls(
            total_memory_mb=total_memory_mb,
            available_memory_mb=available_memory_mb,
            cpu_cores=cpu_cores,
            gpu_available=gpu_available,
            gpu_memory_mb=gpu_memory_mb,
            disk_free_mb=disk_free_mb,
            platform_name=platform_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_memory_mb": self.total_memory_mb,
            "available_memory_mb": self.available_memory_mb,
            "cpu_cores": self.cpu_cores,
            "gpu_available": self.gpu_available,
            "gpu_memory_mb": self.gpu_memory_mb,
            "disk_free_mb": self.disk_free_mb,
            "platform_name": self.platform_name,
        }


class ProfileManager:
    """
    Manager for execution profiles.

    Handles loading predefined profiles, auto-detection of system resources,
    creating custom profiles, and validating profiles against available resources.
    """

    # Predefined profile configurations
    PREDEFINED_PROFILES: Dict[str, Dict[str, Any]] = {
        ProfileName.LAPTOP.value: {
            "name": ProfileName.LAPTOP.value,
            "max_memory_mb": 2048,
            "max_concurrent_tiles": 1,
            "tile_size": (256, 256),
            "use_distributed": False,
            "timeout_seconds": 300,
            "description": "Conservative profile for laptops with 4-8GB RAM",
            "gpu_enabled": False,
            "memory_buffer_factor": 0.3,
            "disk_cache_mb": 512,
        },
        ProfileName.WORKSTATION.value: {
            "name": ProfileName.WORKSTATION.value,
            "max_memory_mb": 8192,
            "max_concurrent_tiles": 4,
            "tile_size": (512, 512),
            "use_distributed": False,
            "timeout_seconds": 600,
            "description": "Standard profile for workstations with 16-32GB RAM",
            "gpu_enabled": False,
            "memory_buffer_factor": 0.2,
            "disk_cache_mb": 2048,
        },
        ProfileName.CLOUD.value: {
            "name": ProfileName.CLOUD.value,
            "max_memory_mb": 32768,
            "max_concurrent_tiles": 16,
            "tile_size": (1024, 1024),
            "use_distributed": True,
            "timeout_seconds": 1800,
            "description": "High-performance profile for cloud/HPC with 64GB+ RAM",
            "gpu_enabled": True,
            "memory_buffer_factor": 0.15,
            "disk_cache_mb": 8192,
        },
        ProfileName.EDGE.value: {
            "name": ProfileName.EDGE.value,
            "max_memory_mb": 1024,
            "max_concurrent_tiles": 1,
            "tile_size": (128, 128),
            "use_distributed": False,
            "timeout_seconds": 600,
            "description": "Minimal profile for edge devices with 1-2GB RAM",
            "gpu_enabled": False,
            "memory_buffer_factor": 0.4,
            "disk_cache_mb": 256,
        },
    }

    def __init__(self) -> None:
        """Initialize profile manager."""
        self._profiles: Dict[str, ExecutionProfile] = {}
        self._system_resources: Optional[SystemResources] = None
        self._load_predefined_profiles()

    def _load_predefined_profiles(self) -> None:
        """Load predefined profiles into memory."""
        for name, config in self.PREDEFINED_PROFILES.items():
            self._profiles[name] = ExecutionProfile.from_dict(config)

    def get_profile(self, name: str) -> Optional[ExecutionProfile]:
        """
        Get a profile by name.

        Args:
            name: Profile name (laptop, workstation, cloud, edge, or custom)

        Returns:
            ExecutionProfile if found, None otherwise
        """
        return self._profiles.get(name)

    def list_profiles(self) -> List[str]:
        """
        List all available profile names.

        Returns:
            List of profile names
        """
        return list(self._profiles.keys())

    def get_all_profiles(self) -> Dict[str, ExecutionProfile]:
        """
        Get all profiles.

        Returns:
            Dictionary of all profiles
        """
        return dict(self._profiles)

    def add_profile(self, profile: ExecutionProfile) -> None:
        """
        Add a custom profile.

        Args:
            profile: Profile to add
        """
        self._profiles[profile.name] = profile
        logger.info(f"Added custom profile: {profile.name}")

    def remove_profile(self, name: str) -> bool:
        """
        Remove a profile (only custom profiles can be removed).

        Args:
            name: Profile name to remove

        Returns:
            True if removed, False if not found or predefined
        """
        if name in self.PREDEFINED_PROFILES:
            logger.warning(f"Cannot remove predefined profile: {name}")
            return False

        if name in self._profiles:
            del self._profiles[name]
            logger.info(f"Removed profile: {name}")
            return True

        return False

    def detect_system_resources(self) -> SystemResources:
        """
        Detect and cache system resources.

        Returns:
            SystemResources instance
        """
        if self._system_resources is None:
            self._system_resources = SystemResources.detect()
            logger.info(
                f"Detected system: {self._system_resources.total_memory_mb}MB RAM, "
                f"{self._system_resources.cpu_cores} cores, "
                f"GPU: {self._system_resources.gpu_available}"
            )
        return self._system_resources

    def auto_select_profile(self) -> ExecutionProfile:
        """
        Automatically select the best profile for current system.

        Uses detected system resources to choose appropriate profile.

        Returns:
            Best matching ExecutionProfile
        """
        resources = self.detect_system_resources()

        # Selection based on available memory
        available_mb = resources.available_memory_mb

        if available_mb >= 16384:  # 16GB+ available
            selected = ProfileName.CLOUD.value
        elif available_mb >= 6144:  # 6GB+ available
            selected = ProfileName.WORKSTATION.value
        elif available_mb >= 2048:  # 2GB+ available
            selected = ProfileName.LAPTOP.value
        else:
            selected = ProfileName.EDGE.value

        profile = self._profiles[selected]
        logger.info(f"Auto-selected profile: {selected} (available memory: {available_mb}MB)")

        return profile

    def create_custom_profile(
        self,
        name: str,
        max_memory_mb: Optional[int] = None,
        max_concurrent_tiles: Optional[int] = None,
        tile_size: Optional[Tuple[int, int]] = None,
        use_distributed: bool = False,
        timeout_seconds: int = 300,
        description: str = "",
        gpu_enabled: bool = False,
    ) -> ExecutionProfile:
        """
        Create a custom profile with specified or auto-detected parameters.

        Args:
            name: Profile name
            max_memory_mb: Max memory (auto-detect if None)
            max_concurrent_tiles: Max concurrent tiles (auto-detect if None)
            tile_size: Tile dimensions (auto-detect if None)
            use_distributed: Enable distributed processing
            timeout_seconds: Timeout per tile
            description: Profile description
            gpu_enabled: Enable GPU acceleration

        Returns:
            Created ExecutionProfile
        """
        resources = self.detect_system_resources()

        # Auto-detect memory if not specified
        if max_memory_mb is None:
            # Use 50% of available memory by default
            max_memory_mb = resources.available_memory_mb // 2
            max_memory_mb = max(256, max_memory_mb)  # Minimum 256MB

        # Auto-detect concurrency if not specified
        if max_concurrent_tiles is None:
            # Use up to half the CPU cores, minimum 1
            max_concurrent_tiles = max(1, resources.cpu_cores // 2)
            # Limit by memory - assume 512MB per tile minimum
            memory_limit = max_memory_mb // 512
            max_concurrent_tiles = min(max_concurrent_tiles, max(1, memory_limit))

        # Auto-detect tile size if not specified
        if tile_size is None:
            # Scale tile size based on available memory per tile
            memory_per_tile = max_memory_mb // max_concurrent_tiles
            if memory_per_tile >= 2048:
                tile_size = (1024, 1024)
            elif memory_per_tile >= 512:
                tile_size = (512, 512)
            elif memory_per_tile >= 256:
                tile_size = (256, 256)
            else:
                tile_size = (128, 128)

        profile = ExecutionProfile(
            name=name,
            max_memory_mb=max_memory_mb,
            max_concurrent_tiles=max_concurrent_tiles,
            tile_size=tile_size,
            use_distributed=use_distributed,
            timeout_seconds=timeout_seconds,
            description=description or f"Custom profile: {name}",
            gpu_enabled=gpu_enabled,
        )

        self._profiles[name] = profile
        logger.info(f"Created custom profile: {name}")

        return profile

    def validate_profile(
        self,
        profile: ExecutionProfile,
    ) -> Tuple[bool, List[str]]:
        """
        Validate profile against current system resources.

        Args:
            profile: Profile to validate

        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        resources = self.detect_system_resources()
        issues: List[str] = []
        is_valid = True

        # Check memory
        if profile.max_memory_mb > resources.available_memory_mb:
            issues.append(
                f"Profile memory ({profile.max_memory_mb}MB) exceeds "
                f"available ({resources.available_memory_mb}MB)"
            )
            is_valid = False

        # Check concurrency vs CPU cores
        if profile.max_concurrent_tiles > resources.cpu_cores * 2:
            issues.append(
                f"High concurrency ({profile.max_concurrent_tiles}) "
                f"may exceed CPU capacity ({resources.cpu_cores} cores)"
            )
            # Warning, not invalid

        # Check GPU requirement
        if profile.gpu_enabled and not resources.gpu_available:
            issues.append("Profile requires GPU but none detected")
            # Warning, will fall back to CPU

        # Check distributed requirement
        if profile.use_distributed:
            try:
                import dask  # noqa: F401

                has_dask = True
            except ImportError:
                has_dask = False

            try:
                import ray  # noqa: F401

                has_ray = True
            except ImportError:
                has_ray = False

            if not has_dask and not has_ray:
                issues.append(
                    "Profile uses distributed processing but "
                    "neither Dask nor Ray is installed"
                )
                # Warning, will fall back to parallel

        # Check disk cache
        if profile.disk_cache_mb > resources.disk_free_mb:
            issues.append(
                f"Disk cache ({profile.disk_cache_mb}MB) exceeds "
                f"free space ({resources.disk_free_mb}MB)"
            )
            # Warning

        return is_valid, issues

    def get_recommended_profile(
        self,
        memory_estimate_mb: int,
        tile_count: int,
        require_gpu: bool = False,
    ) -> ExecutionProfile:
        """
        Get recommended profile based on job requirements.

        Args:
            memory_estimate_mb: Estimated memory needed per tile
            tile_count: Total number of tiles to process
            require_gpu: Whether job requires GPU

        Returns:
            Recommended ExecutionProfile
        """
        resources = self.detect_system_resources()

        # Start with auto-selected profile
        profile = self.auto_select_profile()

        # Check if we need to adjust based on requirements
        if memory_estimate_mb > profile.memory_per_tile_mb:
            # Need more memory per tile - reduce concurrency or use bigger profile
            for name in [ProfileName.CLOUD.value, ProfileName.WORKSTATION.value]:
                candidate = self._profiles[name]
                if candidate.memory_per_tile_mb >= memory_estimate_mb:
                    profile = candidate
                    break

        # Check GPU requirement
        if require_gpu:
            if resources.gpu_available:
                # Use cloud profile for GPU jobs
                profile = self._profiles[ProfileName.CLOUD.value]
            else:
                logger.warning("GPU required but not available, using CPU profile")

        logger.info(
            f"Recommended profile for job: {profile.name} "
            f"(memory/tile: {memory_estimate_mb}MB, tiles: {tile_count})"
        )

        return profile

    def estimate_processing_time(
        self,
        profile: ExecutionProfile,
        tile_count: int,
        time_per_tile_seconds: float = 10.0,
    ) -> Tuple[float, float]:
        """
        Estimate processing time for a job.

        Args:
            profile: Execution profile to use
            tile_count: Total tiles to process
            time_per_tile_seconds: Estimated time per tile

        Returns:
            Tuple of (min_time_seconds, max_time_seconds)
        """
        concurrency = profile.max_concurrent_tiles

        # Best case: perfect parallelization
        batches = (tile_count + concurrency - 1) // concurrency
        min_time = batches * time_per_tile_seconds

        # Worst case: overhead and variability
        overhead_factor = 1.3  # 30% overhead for context switching, etc.
        variability = 1.5  # 50% variability in tile processing time
        max_time = min_time * overhead_factor * variability

        return min_time, max_time


# Module-level convenience functions


def get_profile(name: str) -> Optional[ExecutionProfile]:
    """
    Get a profile by name using default manager.

    Args:
        name: Profile name

    Returns:
        ExecutionProfile if found
    """
    manager = ProfileManager()
    return manager.get_profile(name)


def auto_select_profile() -> ExecutionProfile:
    """
    Auto-select best profile for current system.

    Returns:
        Best matching ExecutionProfile
    """
    manager = ProfileManager()
    return manager.auto_select_profile()


def detect_resources() -> SystemResources:
    """
    Detect current system resources.

    Returns:
        SystemResources instance
    """
    return SystemResources.detect()
