"""
Recovery Strategies Module.

Provides systematic recovery mechanisms for handling failures:
- Retry with exponential backoff
- Alternative data source discovery
- Graceful degradation paths
- User communication templates

When operations fail, this module orchestrates recovery attempts
while tracking all attempts for debugging and analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union
import logging
import random
import time
import uuid

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"                  # Retry the same operation
    RETRY_BACKOFF = "retry_backoff"  # Retry with exponential backoff
    ALTERNATIVE = "alternative"      # Try alternative approach
    DEGRADE = "degrade"              # Accept degraded result
    CACHE = "cache"                  # Use cached result
    PARTIAL = "partial"              # Return partial result
    SKIP = "skip"                    # Skip this operation
    FAIL = "fail"                    # Fail completely


class RecoveryOutcome(Enum):
    """Outcome of recovery attempt."""
    SUCCESS = "success"              # Fully recovered
    PARTIAL_SUCCESS = "partial_success"  # Partially recovered
    DEGRADED = "degraded"            # Degraded result accepted
    EXHAUSTED = "exhausted"          # All strategies tried
    TIMEOUT = "timeout"              # Recovery timed out
    ABORTED = "aborted"              # Recovery aborted


@dataclass
class RecoveryAttempt:
    """
    Record of a recovery attempt.

    Attributes:
        attempt_id: Unique identifier
        strategy: Strategy used
        start_time: When attempt started
        end_time: When attempt ended
        success: Whether attempt succeeded
        error: Error message if failed
        result: Result if successful
        metadata: Additional attempt information
    """
    attempt_id: str
    strategy: RecoveryStrategy
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    error: Optional[str] = None
    result: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Get attempt duration in milliseconds."""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attempt_id": self.attempt_id,
            "strategy": self.strategy.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class RecoveryResult(Generic[T]):
    """
    Result of recovery orchestration.

    Attributes:
        outcome: Final outcome of recovery
        result: Final result (if any)
        attempts: List of recovery attempts
        total_duration_ms: Total time spent on recovery
        strategy_used: Successful strategy (if any)
        message: Human-readable summary
        confidence: Confidence in result (0-1)
    """
    outcome: RecoveryOutcome
    result: Optional[T] = None
    attempts: List[RecoveryAttempt] = field(default_factory=list)
    total_duration_ms: float = 0.0
    strategy_used: Optional[RecoveryStrategy] = None
    message: str = ""
    confidence: float = 1.0

    @property
    def success(self) -> bool:
        """Check if recovery was successful."""
        return self.outcome in (
            RecoveryOutcome.SUCCESS,
            RecoveryOutcome.PARTIAL_SUCCESS,
            RecoveryOutcome.DEGRADED,
        )

    @property
    def attempt_count(self) -> int:
        """Get number of recovery attempts."""
        return len(self.attempts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outcome": self.outcome.value,
            "success": self.success,
            "attempt_count": self.attempt_count,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "message": self.message,
            "confidence": round(self.confidence, 3),
            "attempts": [a.to_dict() for a in self.attempts],
        }


@dataclass
class RecoveryConfig:
    """
    Configuration for recovery orchestration.

    Attributes:
        max_retries: Maximum number of retry attempts
        initial_delay_ms: Initial delay for backoff (milliseconds)
        max_delay_ms: Maximum delay for backoff (milliseconds)
        backoff_multiplier: Multiplier for exponential backoff
        jitter: Add random jitter to delays
        timeout_ms: Overall timeout for recovery (milliseconds)
        enable_alternatives: Enable alternative source discovery
        enable_degradation: Enable graceful degradation
        enable_caching: Enable cached result fallback
        min_acceptable_confidence: Minimum confidence for degraded results
    """
    max_retries: int = 3
    initial_delay_ms: float = 100.0
    max_delay_ms: float = 10000.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    timeout_ms: float = 60000.0
    enable_alternatives: bool = True
    enable_degradation: bool = True
    enable_caching: bool = True
    min_acceptable_confidence: float = 0.3


class RecoveryOrchestrator(Generic[T]):
    """
    Orchestrates recovery from failures.

    Executes recovery strategies in order, tracking all attempts
    and producing detailed recovery results.

    Example:
        orchestrator = RecoveryOrchestrator()

        # Define operation and fallbacks
        def fetch_data():
            return api_client.get_data()

        def get_cached():
            return cache.get("data")

        # Attempt recovery
        result = orchestrator.attempt_recovery(
            operation=fetch_data,
            fallbacks=[
                (RecoveryStrategy.CACHE, get_cached),
                (RecoveryStrategy.DEGRADE, lambda: default_data),
            ],
        )

        if result.success:
            data = result.result
        else:
            print(f"Recovery failed: {result.message}")
    """

    def __init__(self, config: Optional[RecoveryConfig] = None):
        """
        Initialize the recovery orchestrator.

        Args:
            config: Configuration options
        """
        self.config = config or RecoveryConfig()

    def attempt_recovery(
        self,
        operation: Callable[[], T],
        fallbacks: Optional[List[tuple[RecoveryStrategy, Callable[[], T]]]] = None,
        on_attempt: Optional[Callable[[RecoveryAttempt], None]] = None,
    ) -> RecoveryResult[T]:
        """
        Attempt to recover from a failed operation.

        Args:
            operation: Primary operation to attempt
            fallbacks: List of (strategy, callable) fallback options
            on_attempt: Callback for each attempt

        Returns:
            RecoveryResult with outcome and result
        """
        start_time = time.time()
        attempts: List[RecoveryAttempt] = []

        # Try primary operation with retries
        for retry in range(self.config.max_retries + 1):
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.config.timeout_ms:
                return RecoveryResult(
                    outcome=RecoveryOutcome.TIMEOUT,
                    attempts=attempts,
                    total_duration_ms=elapsed_ms,
                    message=f"Recovery timed out after {elapsed_ms:.0f}ms",
                )

            attempt = self._execute_attempt(
                RecoveryStrategy.RETRY if retry > 0 else RecoveryStrategy.RETRY,
                operation,
            )
            attempts.append(attempt)

            if on_attempt:
                on_attempt(attempt)

            if attempt.success:
                return RecoveryResult(
                    outcome=RecoveryOutcome.SUCCESS,
                    result=attempt.result,
                    attempts=attempts,
                    total_duration_ms=(time.time() - start_time) * 1000,
                    strategy_used=RecoveryStrategy.RETRY,
                    message="Operation succeeded" + (f" after {retry + 1} attempts" if retry > 0 else ""),
                    confidence=1.0,
                )

            # Apply backoff before next retry
            if retry < self.config.max_retries:
                delay = self._calculate_delay(retry)
                logger.debug(f"Retry {retry + 1}/{self.config.max_retries}, waiting {delay:.0f}ms")
                time.sleep(delay / 1000)

        # Try fallbacks
        if fallbacks:
            for strategy, fallback_fn in fallbacks:
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.timeout_ms:
                    break

                # Skip disabled strategies
                if strategy == RecoveryStrategy.ALTERNATIVE and not self.config.enable_alternatives:
                    continue
                if strategy == RecoveryStrategy.DEGRADE and not self.config.enable_degradation:
                    continue
                if strategy == RecoveryStrategy.CACHE and not self.config.enable_caching:
                    continue

                attempt = self._execute_attempt(strategy, fallback_fn)
                attempts.append(attempt)

                if on_attempt:
                    on_attempt(attempt)

                if attempt.success:
                    outcome = (
                        RecoveryOutcome.SUCCESS if strategy == RecoveryStrategy.ALTERNATIVE
                        else RecoveryOutcome.DEGRADED if strategy == RecoveryStrategy.DEGRADE
                        else RecoveryOutcome.PARTIAL_SUCCESS
                    )
                    confidence = (
                        0.9 if strategy == RecoveryStrategy.ALTERNATIVE
                        else 0.5 if strategy == RecoveryStrategy.DEGRADE
                        else 0.7 if strategy == RecoveryStrategy.CACHE
                        else 0.8
                    )

                    return RecoveryResult(
                        outcome=outcome,
                        result=attempt.result,
                        attempts=attempts,
                        total_duration_ms=(time.time() - start_time) * 1000,
                        strategy_used=strategy,
                        message=f"Recovered using {strategy.value} strategy",
                        confidence=confidence,
                    )

        # All strategies exhausted
        return RecoveryResult(
            outcome=RecoveryOutcome.EXHAUSTED,
            attempts=attempts,
            total_duration_ms=(time.time() - start_time) * 1000,
            message=f"All recovery strategies exhausted after {len(attempts)} attempts",
            confidence=0.0,
        )

    def retry_with_backoff(
        self,
        operation: Callable[[], T],
        max_retries: Optional[int] = None,
    ) -> RecoveryResult[T]:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: Operation to retry
            max_retries: Override max retries from config

        Returns:
            RecoveryResult with outcome
        """
        original_max = self.config.max_retries
        if max_retries is not None:
            self.config.max_retries = max_retries

        try:
            result = self.attempt_recovery(operation, fallbacks=None)
            return result
        finally:
            self.config.max_retries = original_max

    def with_alternatives(
        self,
        primary: Callable[[], T],
        alternatives: List[Callable[[], T]],
    ) -> RecoveryResult[T]:
        """
        Try primary operation, then alternatives.

        Args:
            primary: Primary operation
            alternatives: List of alternative operations

        Returns:
            RecoveryResult with first successful result
        """
        fallbacks = [
            (RecoveryStrategy.ALTERNATIVE, alt)
            for alt in alternatives
        ]
        return self.attempt_recovery(primary, fallbacks)

    def with_graceful_degradation(
        self,
        operation: Callable[[], T],
        degraded_result: T,
        degraded_confidence: float = 0.5,
    ) -> RecoveryResult[T]:
        """
        Try operation with graceful degradation fallback.

        Args:
            operation: Primary operation
            degraded_result: Result to use if operation fails
            degraded_confidence: Confidence level for degraded result

        Returns:
            RecoveryResult with result
        """
        fallbacks = [
            (RecoveryStrategy.DEGRADE, lambda: degraded_result),
        ]

        result = self.attempt_recovery(operation, fallbacks)
        if result.strategy_used == RecoveryStrategy.DEGRADE:
            result.confidence = degraded_confidence

        return result

    def _execute_attempt(
        self,
        strategy: RecoveryStrategy,
        operation: Callable[[], T],
    ) -> RecoveryAttempt:
        """Execute a single recovery attempt."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            strategy=strategy,
            start_time=datetime.now(timezone.utc),
        )

        try:
            result = operation()
            attempt.success = True
            attempt.result = result
        except Exception as e:
            attempt.success = False
            attempt.error = str(e)
            logger.debug(f"Recovery attempt failed ({strategy.value}): {e}")

        attempt.end_time = datetime.now(timezone.utc)
        return attempt

    def _calculate_delay(self, retry_number: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = self.config.initial_delay_ms * (
            self.config.backoff_multiplier ** retry_number
        )
        delay = min(delay, self.config.max_delay_ms)

        if self.config.jitter:
            # Add up to 25% random jitter
            jitter_amount = delay * 0.25 * random.random()
            delay += jitter_amount

        return delay


class UserCommunicationTemplates:
    """
    Templates for communicating recovery status to users.

    Provides consistent, informative messages about degraded operations.
    """

    @staticmethod
    def recovery_in_progress(strategy: RecoveryStrategy, attempt: int) -> str:
        """Message while recovery is in progress."""
        messages = {
            RecoveryStrategy.RETRY: f"Retrying operation (attempt {attempt})...",
            RecoveryStrategy.RETRY_BACKOFF: f"Waiting before retry {attempt}...",
            RecoveryStrategy.ALTERNATIVE: "Trying alternative data source...",
            RecoveryStrategy.DEGRADE: "Preparing degraded result...",
            RecoveryStrategy.CACHE: "Checking cached results...",
        }
        return messages.get(strategy, f"Attempting recovery ({strategy.value})...")

    @staticmethod
    def recovery_succeeded(result: RecoveryResult[Any]) -> str:
        """Message when recovery succeeds."""
        if result.outcome == RecoveryOutcome.SUCCESS:
            if result.attempt_count == 1:
                return "Operation completed successfully."
            return f"Operation succeeded after {result.attempt_count} attempts."

        if result.outcome == RecoveryOutcome.DEGRADED:
            return (
                f"Using degraded result. "
                f"Confidence: {result.confidence:.0%}. "
                f"Reason: {result.message}"
            )

        if result.outcome == RecoveryOutcome.PARTIAL_SUCCESS:
            return (
                f"Partial result available. "
                f"Confidence: {result.confidence:.0%}."
            )

        return result.message

    @staticmethod
    def recovery_failed(result: RecoveryResult[Any]) -> str:
        """Message when recovery fails."""
        if result.outcome == RecoveryOutcome.TIMEOUT:
            return (
                f"Operation timed out after {result.total_duration_ms / 1000:.1f} seconds. "
                f"Please try again later."
            )

        if result.outcome == RecoveryOutcome.EXHAUSTED:
            return (
                f"Unable to complete operation after {result.attempt_count} attempts. "
                f"All recovery strategies exhausted. "
                f"Please check data availability and try again."
            )

        return f"Operation failed: {result.message}"

    @staticmethod
    def degraded_mode_warning(
        reason: str,
        confidence: float,
        limitations: List[str],
    ) -> str:
        """Warning about degraded mode operation."""
        warning = (
            f"NOTICE: Operating in degraded mode.\n"
            f"Reason: {reason}\n"
            f"Confidence level: {confidence:.0%}\n"
        )

        if limitations:
            warning += "Limitations:\n"
            for limitation in limitations:
                warning += f"  - {limitation}\n"

        return warning


def retry_with_backoff(
    operation: Callable[[], T],
    max_retries: int = 3,
    initial_delay_ms: float = 100.0,
    max_delay_ms: float = 10000.0,
    backoff_multiplier: float = 2.0,
) -> RecoveryResult[T]:
    """
    Convenience function for retry with exponential backoff.

    Args:
        operation: Operation to retry
        max_retries: Maximum retry attempts
        initial_delay_ms: Initial delay between retries
        max_delay_ms: Maximum delay between retries
        backoff_multiplier: Multiplier for backoff

    Returns:
        RecoveryResult with outcome

    Example:
        result = retry_with_backoff(
            lambda: api.fetch_data(),
            max_retries=5,
        )
        if result.success:
            data = result.result
    """
    config = RecoveryConfig(
        max_retries=max_retries,
        initial_delay_ms=initial_delay_ms,
        max_delay_ms=max_delay_ms,
        backoff_multiplier=backoff_multiplier,
    )
    orchestrator: RecoveryOrchestrator[T] = RecoveryOrchestrator(config)
    return orchestrator.retry_with_backoff(operation)
