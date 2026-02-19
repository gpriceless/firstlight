"""
Tests for the DualWrite state backend.

These tests use mock backends to verify the dual-write behavior
without requiring a live PostGIS instance.
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)
from agents.orchestrator.backends.dual_write import DualWriteBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    job_id: str = "job-1",
    phase: str = "QUEUED",
    status: str = "PENDING",
    customer_id: str = "tenant-1",
) -> JobState:
    """Create a test JobState."""
    from datetime import datetime, timezone

    return JobState(
        job_id=job_id,
        customer_id=customer_id,
        event_type="flood",
        phase=phase,
        status=status,
        parameters={"threshold": 0.5},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


def _mock_backend() -> AsyncMock:
    """Create a mock StateBackend with sensible defaults."""
    mock = AsyncMock(spec=StateBackend)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDualWriteReads:
    """Reads should always go to the primary (PostGIS) backend."""

    @pytest.mark.asyncio
    async def test_get_state_reads_from_primary(self):
        """get_state should query the primary backend only."""
        primary = _mock_backend()
        fallback = _mock_backend()
        job = _make_job()
        primary.get_state.return_value = job

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.get_state("job-1")

        assert result is job
        primary.get_state.assert_called_once_with("job-1")
        fallback.get_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_jobs_reads_from_primary(self):
        """list_jobs should query the primary backend only."""
        primary = _mock_backend()
        fallback = _mock_backend()
        jobs = [_make_job("j1"), _make_job("j2")]
        primary.list_jobs.return_value = jobs

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.list_jobs(phase="QUEUED")

        assert result == jobs
        primary.list_jobs.assert_called_once_with(
            customer_id=None, phase="QUEUED", status=None,
            event_type=None, limit=100, offset=0,
        )
        fallback.list_jobs.assert_not_called()


class TestDualWriteWrites:
    """Writes should go to primary first, then shadow-write to fallback."""

    @pytest.mark.asyncio
    async def test_set_state_writes_to_both(self):
        """set_state should write to primary and fallback."""
        primary = _mock_backend()
        fallback = _mock_backend()
        job = _make_job(phase="DISCOVERING", status="DISCOVERING")
        primary.set_state.return_value = job
        fallback.set_state.return_value = job

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.set_state("job-1", "DISCOVERING", "DISCOVERING")

        assert result is job
        primary.set_state.assert_called_once()
        fallback.set_state.assert_called_once()

    @pytest.mark.asyncio
    async def test_transition_writes_to_both(self):
        """transition should transition on primary, then set_state on fallback."""
        primary = _mock_backend()
        fallback = _mock_backend()
        job = _make_job(phase="DISCOVERING", status="DISCOVERING")
        primary.transition.return_value = job
        fallback.set_state.return_value = job

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.transition(
            "job-1",
            "QUEUED", "PENDING",
            "DISCOVERING", "DISCOVERING",
        )

        assert result is job
        primary.transition.assert_called_once()
        # Fallback gets set_state (not transition) since it may be out of sync
        fallback.set_state.assert_called_once_with(
            "job-1", "DISCOVERING", "DISCOVERING"
        )

    @pytest.mark.asyncio
    async def test_checkpoint_writes_to_both(self):
        """checkpoint should write to primary and fallback."""
        primary = _mock_backend()
        fallback = _mock_backend()

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        payload = {"stage": "test"}
        await dual.checkpoint("job-1", payload)

        primary.checkpoint.assert_called_once_with("job-1", payload)
        fallback.checkpoint.assert_called_once_with("job-1", payload)


class TestDualWriteFallbackFailure:
    """Fallback (SQLite) failures should not propagate to the caller."""

    @pytest.mark.asyncio
    async def test_set_state_fallback_failure_swallowed(self):
        """set_state should succeed even if fallback fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        job = _make_job()
        primary.set_state.return_value = job
        fallback.set_state.side_effect = Exception("SQLite disk full")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.set_state("job-1", "QUEUED", "PENDING")

        assert result is job  # Should succeed despite fallback failure
        assert dual.primary_healthy is True

    @pytest.mark.asyncio
    async def test_transition_fallback_failure_swallowed(self):
        """transition should succeed even if fallback shadow write fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        job = _make_job()
        primary.transition.return_value = job
        fallback.set_state.side_effect = Exception("SQLite locked")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        result = await dual.transition(
            "job-1", "QUEUED", "PENDING", "DISCOVERING", "DISCOVERING",
        )

        assert result is job  # Should succeed

    @pytest.mark.asyncio
    async def test_checkpoint_fallback_failure_swallowed(self):
        """checkpoint should succeed even if fallback fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        fallback.checkpoint.side_effect = Exception("SQLite error")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        # Should not raise
        await dual.checkpoint("job-1", {"data": "test"})
        primary.checkpoint.assert_called_once()


class TestDualWritePrimaryFailure:
    """Primary (PostGIS) failures should propagate to the caller."""

    @pytest.mark.asyncio
    async def test_get_state_primary_failure_propagates(self):
        """get_state should raise if primary fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.get_state.side_effect = ConnectionError("PostGIS down")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        with pytest.raises(ConnectionError):
            await dual.get_state("job-1")
        assert dual.primary_healthy is False

    @pytest.mark.asyncio
    async def test_set_state_primary_failure_propagates(self):
        """set_state should raise if primary fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.set_state.side_effect = ConnectionError("PostGIS timeout")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        with pytest.raises(ConnectionError):
            await dual.set_state("job-1", "QUEUED", "PENDING")
        assert dual.primary_healthy is False
        fallback.set_state.assert_not_called()  # Should NOT write to fallback

    @pytest.mark.asyncio
    async def test_list_jobs_primary_failure_propagates(self):
        """list_jobs should raise if primary fails."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.list_jobs.side_effect = ConnectionError("PostGIS down")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        with pytest.raises(ConnectionError):
            await dual.list_jobs()
        assert dual.primary_healthy is False


class TestDualWriteHealthTracking:
    """primary_healthy should reflect the actual state of the primary."""

    @pytest.mark.asyncio
    async def test_healthy_after_success(self):
        """primary_healthy should be True after a successful operation."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.get_state.return_value = _make_job()

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        await dual.get_state("job-1")
        assert dual.primary_healthy is True

    @pytest.mark.asyncio
    async def test_unhealthy_after_failure(self):
        """primary_healthy should be False after a primary failure."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.get_state.side_effect = ConnectionError("down")

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        with pytest.raises(ConnectionError):
            await dual.get_state("job-1")
        assert dual.primary_healthy is False

    @pytest.mark.asyncio
    async def test_recovers_after_success(self):
        """primary_healthy should recover to True after the primary heals."""
        primary = _mock_backend()
        fallback = _mock_backend()

        dual = DualWriteBackend(primary=primary, fallback=fallback)

        # First call fails
        primary.get_state.side_effect = ConnectionError("down")
        with pytest.raises(ConnectionError):
            await dual.get_state("job-1")
        assert dual.primary_healthy is False

        # Second call succeeds
        primary.get_state.side_effect = None
        primary.get_state.return_value = _make_job()
        await dual.get_state("job-1")
        assert dual.primary_healthy is True

    @pytest.mark.asyncio
    async def test_conflict_does_not_mark_unhealthy(self):
        """StateConflictError is a business error, not a health issue."""
        primary = _mock_backend()
        fallback = _mock_backend()
        primary.transition.side_effect = StateConflictError(
            "job-1", "QUEUED", "PENDING", "DISCOVERING", "DISCOVERING"
        )

        dual = DualWriteBackend(primary=primary, fallback=fallback)
        with pytest.raises(StateConflictError):
            await dual.transition(
                "job-1", "QUEUED", "PENDING", "INGESTING", "INGESTING"
            )
        assert dual.primary_healthy is True  # Conflict is not a health issue


class TestDualWriteProperties:
    """Tests for backend access properties."""

    def test_primary_property(self):
        """primary property should return the primary backend."""
        primary = _mock_backend()
        fallback = _mock_backend()
        dual = DualWriteBackend(primary=primary, fallback=fallback)
        assert dual.primary is primary

    def test_fallback_property(self):
        """fallback property should return the fallback backend."""
        primary = _mock_backend()
        fallback = _mock_backend()
        dual = DualWriteBackend(primary=primary, fallback=fallback)
        assert dual.fallback is fallback
