"""
Dual-write state backend.

Writes to PostGIS as the canonical primary and SQLite as a fallback.
Reads always go to PostGIS. SQLite write failures are logged as warnings
but never propagated to the caller.

This backend enables a safe migration path from SQLite-only to PostGIS-only:
1. Deploy with STATE_BACKEND=dual (writes to both, reads from PostGIS)
2. Validate PostGIS data matches SQLite
3. Switch to STATE_BACKEND=postgis (PostGIS only)
"""

import logging
from typing import Any, Dict, List, Optional

from agents.orchestrator.backends.base import (
    JobState,
    StateBackend,
    StateConflictError,
)

logger = logging.getLogger(__name__)


class DualWriteBackend(StateBackend):
    """
    Dual-write backend: PostGIS primary + SQLite fallback.

    Write operations:
    - PostGIS is the canonical write target. If PostGIS fails, the error
      propagates to the caller.
    - SQLite write is attempted after PostGIS succeeds. If SQLite fails,
      a warning is logged but the operation is considered successful.

    Read operations:
    - Always read from PostGIS (the primary).

    Attributes:
        primary: The PostGIS backend (canonical).
        fallback: The SQLite backend (best-effort shadow).
        primary_healthy: Whether the primary backend is responding.
    """

    def __init__(
        self,
        primary: StateBackend,
        fallback: StateBackend,
    ):
        """
        Initialize the dual-write backend.

        Args:
            primary: The PostGIS backend (canonical writes and reads).
            fallback: The SQLite backend (shadow writes only).
        """
        self._primary = primary
        self._fallback = fallback
        self._primary_healthy = True

    @property
    def primary_healthy(self) -> bool:
        """Whether the primary backend last responded successfully."""
        return self._primary_healthy

    @property
    def primary(self) -> StateBackend:
        """The primary (PostGIS) backend."""
        return self._primary

    @property
    def fallback(self) -> StateBackend:
        """The fallback (SQLite) backend."""
        return self._fallback

    async def _shadow_write(self, operation: str, coro) -> None:
        """
        Execute a shadow write to the fallback backend.

        Failures are logged as warnings and swallowed.

        Args:
            operation: Description of the operation (for logging).
            coro: The awaitable to execute.
        """
        try:
            await coro
        except Exception as e:
            logger.warning(
                "Fallback %s failed (non-fatal): %s: %s",
                operation,
                type(e).__name__,
                e,
            )

    async def get_state(self, job_id: str) -> Optional[JobState]:
        """Read from primary (PostGIS)."""
        try:
            result = await self._primary.get_state(job_id)
            self._primary_healthy = True
            return result
        except Exception:
            self._primary_healthy = False
            raise

    async def set_state(
        self,
        job_id: str,
        phase: str,
        status: str,
        **kwargs: Any,
    ) -> JobState:
        """Write to primary, then shadow-write to fallback."""
        try:
            result = await self._primary.set_state(job_id, phase, status, **kwargs)
            self._primary_healthy = True
        except Exception:
            self._primary_healthy = False
            raise

        # Shadow write to fallback (best effort)
        await self._shadow_write(
            f"set_state({job_id})",
            self._fallback.set_state(job_id, phase, status, **kwargs),
        )

        return result

    async def transition(
        self,
        job_id: str,
        expected_phase: str,
        expected_status: str,
        new_phase: str,
        new_status: str,
        *,
        reason: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> JobState:
        """
        Transition on primary, then shadow-write to fallback.

        The TOCTOU guard runs against the primary. If the primary transition
        succeeds, a direct set_state is used on the fallback (not a transition,
        since the fallback may be out of sync).
        """
        try:
            result = await self._primary.transition(
                job_id,
                expected_phase,
                expected_status,
                new_phase,
                new_status,
                reason=reason,
                actor=actor,
            )
            self._primary_healthy = True
        except StateConflictError:
            # Conflict is a business-logic error, not a health issue
            raise
        except Exception:
            self._primary_healthy = False
            raise

        # Shadow write: use set_state on fallback since it may be at a
        # different state than expected. We trust the primary's result.
        await self._shadow_write(
            f"transition({job_id})",
            self._fallback.set_state(job_id, new_phase, new_status),
        )

        return result

    async def list_jobs(
        self,
        *,
        customer_id: Optional[str] = None,
        phase: Optional[str] = None,
        status: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[JobState]:
        """Read from primary (PostGIS)."""
        try:
            result = await self._primary.list_jobs(
                customer_id=customer_id,
                phase=phase,
                status=status,
                event_type=event_type,
                limit=limit,
                offset=offset,
            )
            self._primary_healthy = True
            return result
        except Exception:
            self._primary_healthy = False
            raise

    async def checkpoint(
        self,
        job_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """Write checkpoint to primary, shadow-write to fallback."""
        try:
            await self._primary.checkpoint(job_id, payload)
            self._primary_healthy = True
        except Exception:
            self._primary_healthy = False
            raise

        await self._shadow_write(
            f"checkpoint({job_id})",
            self._fallback.checkpoint(job_id, payload),
        )
