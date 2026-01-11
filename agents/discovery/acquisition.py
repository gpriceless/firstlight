"""
Data Acquisition Manager for Discovery Agent.

Provides:
- AcquisitionManager class for coordinating downloads
- Download progress tracking
- Data integrity verification
- Partial failure handling
- Integration with core/data/ingestion/
"""

import asyncio
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore
    AIOHTTP_AVAILABLE = False


logger = logging.getLogger(__name__)


class AcquisitionStatus(Enum):
    """Status of an acquisition operation."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some downloads succeeded, others failed


class DownloadStatus(Enum):
    """Status of individual download."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Already exists or cached


@dataclass
class DatasetInfo:
    """
    Information about a dataset to acquire.

    Attributes:
        dataset_id: Unique dataset identifier
        source_uri: URL to download from
        provider: Data provider ID
        data_type: Type of data (optical, sar, dem, etc.)
        format: Data format (cog, geotiff, netcdf, etc.)
        expected_size_bytes: Expected file size
        checksum: Expected checksum for verification
        metadata: Additional dataset metadata
    """

    dataset_id: str
    source_uri: str
    provider: str
    data_type: str
    format: str = "geotiff"
    expected_size_bytes: Optional[int] = None
    checksum: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            dataset_id=data.get("dataset_id", data.get("id", "")),
            source_uri=data.get("source_uri", data.get("uri", "")),
            provider=data.get("provider", ""),
            data_type=data.get("data_type", ""),
            format=data.get("format", data.get("source_format", "geotiff")),
            expected_size_bytes=data.get("expected_size_bytes"),
            checksum=data.get("checksum"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DownloadProgress:
    """
    Progress information for a download.

    Attributes:
        dataset_id: Dataset being downloaded
        status: Current download status
        bytes_downloaded: Bytes downloaded so far
        total_bytes: Total bytes to download
        started_at: Download start time
        completed_at: Download completion time
        speed_bytes_per_sec: Current download speed
        error_message: Error if failed
    """

    dataset_id: str
    status: DownloadStatus = DownloadStatus.PENDING
    bytes_downloaded: int = 0
    total_bytes: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    speed_bytes_per_sec: float = 0.0
    error_message: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        """Get download progress as percentage."""
        if self.total_bytes is None or self.total_bytes == 0:
            return 0.0
        return (self.bytes_downloaded / self.total_bytes) * 100.0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed download time."""
        if self.started_at is None:
            return 0.0
        end_time = self.completed_at or datetime.now(timezone.utc)
        return (end_time - self.started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "status": self.status.value,
            "bytes_downloaded": self.bytes_downloaded,
            "total_bytes": self.total_bytes,
            "progress_percent": self.progress_percent,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "elapsed_seconds": self.elapsed_seconds,
            "speed_bytes_per_sec": self.speed_bytes_per_sec,
            "error_message": self.error_message,
        }


@dataclass
class AcquisitionRequest:
    """
    Request for data acquisition.

    Attributes:
        request_id: Unique request identifier
        datasets: Datasets to acquire
        output_path: Directory to save downloaded files
        format_options: Format conversion options
        verify_checksums: Whether to verify checksums
        max_concurrent_downloads: Max parallel downloads
        timeout_seconds: Download timeout per file
        retry_count: Number of retry attempts
        skip_existing: Skip files that already exist
    """

    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    output_path: str = ""
    format_options: Dict[str, Any] = field(default_factory=dict)
    verify_checksums: bool = True
    max_concurrent_downloads: int = 5
    timeout_seconds: float = 300.0
    retry_count: int = 3
    skip_existing: bool = True

    def get_dataset_infos(self) -> List[DatasetInfo]:
        """Convert dataset dicts to DatasetInfo objects."""
        return [DatasetInfo.from_dict(d) for d in self.datasets]


@dataclass
class DownloadedFile:
    """
    Information about a downloaded file.

    Attributes:
        dataset_id: Dataset that was downloaded
        local_path: Path to downloaded file
        file_size_bytes: Size of downloaded file
        checksum_verified: Whether checksum was verified
        download_time_seconds: Time to download
        format: File format
    """

    dataset_id: str
    local_path: str
    file_size_bytes: int
    checksum_verified: bool = False
    download_time_seconds: float = 0.0
    format: str = "geotiff"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "local_path": self.local_path,
            "file_size_bytes": self.file_size_bytes,
            "checksum_verified": self.checksum_verified,
            "download_time_seconds": self.download_time_seconds,
            "format": self.format,
        }


@dataclass
class AcquisitionResult:
    """
    Result of acquisition operation.

    Attributes:
        request_id: Original request ID
        status: Overall acquisition status
        downloaded_files: Successfully downloaded files
        failed_downloads: Failed download information
        total_bytes: Total bytes downloaded
        elapsed_seconds: Total acquisition time
        errors: Error messages
    """

    request_id: str
    status: AcquisitionStatus
    downloaded_files: List[DownloadedFile] = field(default_factory=list)
    failed_downloads: List[Dict[str, Any]] = field(default_factory=list)
    total_bytes: int = 0
    elapsed_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "downloaded_files": [f.to_dict() for f in self.downloaded_files],
            "failed_downloads": self.failed_downloads,
            "total_bytes": self.total_bytes,
            "total_bytes_mb": self.total_bytes / (1024 * 1024),
            "elapsed_seconds": self.elapsed_seconds,
            "success_count": len(self.downloaded_files),
            "failure_count": len(self.failed_downloads),
            "errors": self.errors,
        }


class AcquisitionManager:
    """
    Manages data acquisition from multiple sources.

    Features:
    - Parallel downloads with configurable concurrency
    - Progress tracking
    - Checksum verification
    - Retry handling for failed downloads
    - Integration with core ingestion pipeline
    """

    def __init__(
        self,
        default_output_path: Optional[str] = None,
        max_concurrent_downloads: int = 5,
        chunk_size: int = 8192,
        verify_checksums: bool = True
    ):
        """
        Initialize AcquisitionManager.

        Args:
            default_output_path: Default directory for downloads
            max_concurrent_downloads: Default max parallel downloads
            chunk_size: Chunk size for streaming downloads
            verify_checksums: Default checksum verification setting
        """
        self._default_output_path = default_output_path or "/tmp/multiverse_dive/downloads"
        self._max_concurrent_downloads = max_concurrent_downloads
        self._chunk_size = chunk_size
        self._verify_checksums = verify_checksums

        # Active downloads tracking
        self._active_downloads: Dict[str, DownloadProgress] = {}
        self._download_semaphore: Optional[asyncio.Semaphore] = None

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self._stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes_downloaded": 0,
        }

    async def _get_session(self) -> "aiohttp.ClientSession":
        """Get or create HTTP session."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for data acquisition. Install with: pip install aiohttp")
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close HTTP session and cleanup."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        self._active_downloads.clear()

    async def acquire(self, request: AcquisitionRequest) -> AcquisitionResult:
        """
        Acquire datasets based on request.

        Args:
            request: Acquisition request with datasets and options

        Returns:
            AcquisitionResult with download results
        """
        logger.info(
            f"Starting acquisition of {len(request.datasets)} datasets "
            f"to {request.output_path or self._default_output_path}"
        )

        start_time = datetime.now(timezone.utc)
        output_path = request.output_path or self._default_output_path

        # Ensure output directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Get dataset infos
        datasets = request.get_dataset_infos()

        # Create semaphore for concurrent downloads
        max_concurrent = request.max_concurrent_downloads or self._max_concurrent_downloads
        self._download_semaphore = asyncio.Semaphore(max_concurrent)

        # Initialize progress tracking
        for dataset in datasets:
            self._active_downloads[dataset.dataset_id] = DownloadProgress(
                dataset_id=dataset.dataset_id,
                status=DownloadStatus.PENDING,
                total_bytes=dataset.expected_size_bytes
            )

        # Create download tasks
        tasks = [
            self._download_dataset(
                dataset=dataset,
                output_path=output_path,
                verify_checksum=request.verify_checksums,
                timeout=request.timeout_seconds,
                retry_count=request.retry_count,
                skip_existing=request.skip_existing
            )
            for dataset in datasets
        ]

        # Execute downloads in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        downloaded_files: List[DownloadedFile] = []
        failed_downloads: List[Dict[str, Any]] = []
        errors: List[str] = []
        total_bytes = 0

        for i, result in enumerate(results):
            dataset = datasets[i]

            if isinstance(result, Exception):
                failed_downloads.append({
                    "dataset_id": dataset.dataset_id,
                    "error": str(result)
                })
                errors.append(f"Failed to download {dataset.dataset_id}: {result}")
                self._stats["failed_downloads"] += 1
            elif isinstance(result, DownloadedFile):
                downloaded_files.append(result)
                total_bytes += result.file_size_bytes
                self._stats["successful_downloads"] += 1
            elif result is None:
                # Skipped (already exists)
                logger.debug(f"Skipped {dataset.dataset_id} (already exists)")

        self._stats["total_downloads"] += len(datasets)
        self._stats["total_bytes_downloaded"] += total_bytes

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Determine status
        if len(downloaded_files) == len(datasets):
            status = AcquisitionStatus.COMPLETED
        elif downloaded_files:
            status = AcquisitionStatus.PARTIAL
        else:
            status = AcquisitionStatus.FAILED

        logger.info(
            f"Acquisition complete: {len(downloaded_files)}/{len(datasets)} succeeded, "
            f"{total_bytes / (1024*1024):.1f} MB in {elapsed:.1f}s"
        )

        return AcquisitionResult(
            request_id=request.request_id,
            status=status,
            downloaded_files=downloaded_files,
            failed_downloads=failed_downloads,
            total_bytes=total_bytes,
            elapsed_seconds=elapsed,
            errors=errors
        )

    async def _download_dataset(
        self,
        dataset: DatasetInfo,
        output_path: str,
        verify_checksum: bool,
        timeout: float,
        retry_count: int,
        skip_existing: bool
    ) -> Optional[DownloadedFile]:
        """
        Download a single dataset with retry logic.

        Args:
            dataset: Dataset to download
            output_path: Directory to save file
            verify_checksum: Whether to verify checksum
            timeout: Download timeout
            retry_count: Number of retries
            skip_existing: Skip if file exists

        Returns:
            DownloadedFile if successful, None if skipped
        """
        # Determine output file path
        filename = self._generate_filename(dataset)
        local_path = os.path.join(output_path, filename)

        # Check if file exists
        if skip_existing and os.path.exists(local_path):
            progress = self._active_downloads.get(dataset.dataset_id)
            if progress:
                progress.status = DownloadStatus.SKIPPED
            return None

        # Acquire semaphore
        async with self._download_semaphore:
            last_error: Optional[Exception] = None

            for attempt in range(retry_count + 1):
                try:
                    result = await self._execute_download(
                        dataset=dataset,
                        local_path=local_path,
                        timeout=timeout,
                        verify_checksum=verify_checksum
                    )
                    return result

                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Download attempt {attempt + 1}/{retry_count + 1} failed for "
                        f"{dataset.dataset_id}: {e}"
                    )

                    if attempt < retry_count:
                        delay = 2 ** attempt  # Exponential backoff
                        await asyncio.sleep(delay)

            # All retries exhausted
            progress = self._active_downloads.get(dataset.dataset_id)
            if progress:
                progress.status = DownloadStatus.FAILED
                progress.error_message = str(last_error)

            raise last_error or RuntimeError(f"Failed to download {dataset.dataset_id}")

    async def _execute_download(
        self,
        dataset: DatasetInfo,
        local_path: str,
        timeout: float,
        verify_checksum: bool
    ) -> DownloadedFile:
        """Execute the actual download."""
        progress = self._active_downloads.get(dataset.dataset_id)
        if progress:
            progress.status = DownloadStatus.IN_PROGRESS
            progress.started_at = datetime.now(timezone.utc)

        session = await self._get_session()
        start_time = datetime.now(timezone.utc)

        try:
            async with session.get(
                dataset.source_uri,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                response.raise_for_status()

                # Get content length if available
                total_size = response.content_length
                if progress:
                    progress.total_bytes = total_size

                # Stream download
                bytes_downloaded = 0
                last_update_time = start_time

                with open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self._chunk_size):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)

                        # Update progress
                        if progress:
                            progress.bytes_downloaded = bytes_downloaded
                            now = datetime.now(timezone.utc)
                            elapsed = (now - last_update_time).total_seconds()
                            if elapsed > 0:
                                progress.speed_bytes_per_sec = len(chunk) / elapsed
                            last_update_time = now

            # Get file size
            file_size = os.path.getsize(local_path)

            # Update progress
            if progress:
                progress.status = DownloadStatus.COMPLETED
                progress.completed_at = datetime.now(timezone.utc)
                progress.bytes_downloaded = file_size

            # Verify checksum if requested
            checksum_verified = False
            if verify_checksum and dataset.checksum:
                checksum_verified = await self._verify_checksum(
                    local_path,
                    dataset.checksum
                )
                if not checksum_verified:
                    raise ValueError(f"Checksum verification failed for {dataset.dataset_id}")

            download_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            return DownloadedFile(
                dataset_id=dataset.dataset_id,
                local_path=local_path,
                file_size_bytes=file_size,
                checksum_verified=checksum_verified,
                download_time_seconds=download_time,
                format=dataset.format
            )

        except Exception as e:
            # Clean up partial download
            if os.path.exists(local_path):
                try:
                    os.remove(local_path)
                except OSError:
                    pass
            raise

    def _generate_filename(self, dataset: DatasetInfo) -> str:
        """Generate output filename for dataset."""
        # Use dataset ID as base
        base_name = dataset.dataset_id

        # Clean up filename
        base_name = base_name.replace("/", "_").replace(":", "_")

        # Add extension based on format
        format_extensions = {
            "cog": ".tif",
            "geotiff": ".tif",
            "netcdf": ".nc",
            "zarr": ".zarr",
            "geojson": ".geojson",
            "parquet": ".parquet"
        }

        ext = format_extensions.get(dataset.format.lower(), ".dat")

        return f"{base_name}{ext}"

    async def _verify_checksum(
        self,
        file_path: str,
        expected_checksum: Dict[str, str]
    ) -> bool:
        """
        Verify file checksum.

        Args:
            file_path: Path to file
            expected_checksum: Dict with algorithm and value

        Returns:
            True if checksum matches
        """
        algorithm = expected_checksum.get("algorithm", "sha256").lower()
        expected_value = expected_checksum.get("value", "")

        if not expected_value:
            return True  # No checksum to verify

        # Calculate checksum
        if algorithm == "sha256":
            hash_obj = hashlib.sha256()
        elif algorithm == "md5":
            hash_obj = hashlib.md5()
        elif algorithm == "sha1":
            hash_obj = hashlib.sha1()
        else:
            logger.warning(f"Unknown checksum algorithm: {algorithm}")
            return False

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)

        calculated = hash_obj.hexdigest()

        if calculated.lower() == expected_value.lower():
            logger.debug(f"Checksum verified for {file_path}")
            return True
        else:
            logger.error(
                f"Checksum mismatch for {file_path}: "
                f"expected {expected_value}, got {calculated}"
            )
            return False

    def get_progress(self, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get download progress.

        Args:
            dataset_id: Optional specific dataset ID

        Returns:
            Progress information
        """
        if dataset_id:
            progress = self._active_downloads.get(dataset_id)
            if progress:
                return progress.to_dict()
            return {"error": f"Unknown dataset: {dataset_id}"}

        return {
            "active_downloads": {
                k: v.to_dict() for k, v in self._active_downloads.items()
            },
            "statistics": self._stats.copy()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get acquisition statistics."""
        return self._stats.copy()


async def batch_acquire(
    datasets: List[Dict[str, Any]],
    output_path: str,
    max_concurrent: int = 5
) -> AcquisitionResult:
    """
    Convenience function for batch acquisition.

    Args:
        datasets: List of datasets to acquire
        output_path: Output directory
        max_concurrent: Max parallel downloads

    Returns:
        AcquisitionResult
    """
    manager = AcquisitionManager(max_concurrent_downloads=max_concurrent)

    try:
        request = AcquisitionRequest(
            datasets=datasets,
            output_path=output_path,
            max_concurrent_downloads=max_concurrent
        )

        return await manager.acquire(request)

    finally:
        await manager.close()
