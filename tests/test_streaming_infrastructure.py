"""
Tests for the streaming ingestion infrastructure.

- StreamingDownloader, WindowedReader, StreamingIngester (core/data/ingestion/streaming.py)
"""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.data.ingestion.streaming import (
    ChunkInfo,
    DownloadProgress,
    DownloadStatus,
    MemoryMappedReader,
    StreamingConfig,
    StreamingDownloader,
    StreamingIngester,
    WindowInfo,
    estimate_chunk_count,
)

class TestStreamingConfig:
    """Tests for StreamingConfig class."""

    def test_create_default_config(self):
        """Test default config creation."""
        config = StreamingConfig()
        assert config.chunk_size_bytes == 10 * 1024 * 1024
        assert config.max_retries == 3

    def test_create_config_from_mb(self):
        """Test creating config with MB values."""
        config = StreamingConfig.from_mb(
            chunk_size_mb=50,
            buffer_size_mb=512,
            bandwidth_limit_mbps=100,
        )

        assert config.chunk_size_bytes == 50 * 1024 * 1024
        assert config.buffer_size_bytes == 512 * 1024 * 1024


class TestDownloadProgress:
    """Tests for DownloadProgress class."""

    def test_progress_properties(self):
        """Test progress computed properties."""
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
        )

        assert progress.progress_percent == 50.0
        assert progress.remaining_bytes == 500

    def test_progress_zero_total(self):
        """Test progress with zero total."""
        progress = DownloadProgress(total_bytes=0, downloaded_bytes=0)
        assert progress.progress_percent == 0.0


class TestWindowInfo:
    """Tests for WindowInfo class."""

    def test_window_info_properties(self):
        """Test WindowInfo properties."""
        window = WindowInfo(col_off=100, row_off=200, width=256, height=256)
        assert window.shape == (256, 256)

    def test_window_to_dict(self):
        """Test serialization."""
        window = WindowInfo(col_off=100, row_off=200, width=256, height=256)
        d = window.to_dict()
        assert d["col_off"] == 100
        assert d["row_off"] == 200


class TestStreamingDownloader:
    """Tests for StreamingDownloader class."""

    def test_create_downloader(self):
        """Test downloader creation."""
        downloader = StreamingDownloader()
        assert downloader.config is not None

    def test_cancel_download(self):
        """Test cancelling download."""
        downloader = StreamingDownloader()
        downloader.cancel()
        assert downloader._cancelled.is_set()


class TestMemoryMappedReader:
    """Tests for MemoryMappedReader class."""

    def test_create_and_read(self):
        """Test memory-mapped file access."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            # Write test data
            data = np.arange(100 * 100, dtype=np.float32).reshape(100, 100)
            data.tofile(f)
            path = f.name

        try:
            reader = MemoryMappedReader(path, shape=(100, 100), dtype=np.float32)
            with reader as mmap:
                assert mmap.shape == (100, 100)
                assert mmap[50, 50] == 50 * 100 + 50

            # Test region read
            reader = MemoryMappedReader(path, shape=(100, 100), dtype=np.float32)
            reader.open()
            region = reader.read_region(10, 10, 20, 20)
            assert region.shape == (20, 20)
            reader.close()
        finally:
            Path(path).unlink()


class TestStreamingUtilities:
    """Tests for streaming utility functions."""

    def test_estimate_chunk_count(self):
        """Test chunk count estimation."""
        count = estimate_chunk_count(
            file_size_bytes=100 * 1024 * 1024,  # 100 MB
            chunk_size_mb=10,
        )
        assert count == 10
