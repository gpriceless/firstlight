"""
Failure Logging Module.

Provides structured failure logging with full context for debugging,
analysis, and system improvement. All failures are documented with:
- What failed (component, operation)
- Why it failed (error type, message)
- Context (what was being processed)
- Outcome (fallback used, recovery attempted)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


class FailureComponent(Enum):
    """System components that can fail."""
    DISCOVERY = "discovery"          # Data discovery operations
    INGESTION = "ingestion"          # Data ingestion pipeline
    PIPELINE = "pipeline"            # Analysis pipeline execution
    ALGORITHM = "algorithm"          # Individual algorithm execution
    QUALITY = "quality"              # Quality control checks
    VALIDATION = "validation"        # Validation operations
    NETWORK = "network"              # Network/API operations
    STORAGE = "storage"              # Storage operations
    FUSION = "fusion"                # Data fusion operations
    AGENT = "agent"                  # Agent orchestration
    UNKNOWN = "unknown"


class FailureSeverity(Enum):
    """Severity levels for failures."""
    DEBUG = "debug"          # Not actually a failure, informational
    INFO = "info"            # Minor issue, handled automatically
    WARNING = "warning"      # Degraded operation, fallback used
    ERROR = "error"          # Operation failed, partial results
    CRITICAL = "critical"    # System failure, no results possible

    @property
    def level(self) -> int:
        """Numeric severity level."""
        levels = {
            FailureSeverity.DEBUG: 0,
            FailureSeverity.INFO: 1,
            FailureSeverity.WARNING: 2,
            FailureSeverity.ERROR: 3,
            FailureSeverity.CRITICAL: 4,
        }
        return levels[self]


@dataclass
class FailureLog:
    """
    Structured log entry for a failure.

    Attributes:
        failure_id: Unique identifier for this failure
        timestamp: When the failure occurred
        component: Which system component failed
        error_type: Classification of the error
        error_message: Human-readable error description
        severity: Failure severity level
        context: What was being processed when failure occurred
        stack_trace: Optional stack trace
        fallback_used: Fallback mechanism used (if any)
        outcome: Final outcome (recovered, degraded, failed)
        recovery_attempted: Whether recovery was attempted
        duration_ms: How long the operation ran before failure
        metadata: Additional failure information
    """
    failure_id: str
    timestamp: datetime
    component: FailureComponent
    error_type: str
    error_message: str
    severity: FailureSeverity = FailureSeverity.ERROR
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    fallback_used: Optional[str] = None
    outcome: str = "failed"
    recovery_attempted: bool = False
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_recoverable(self) -> bool:
        """Check if failure was recoverable."""
        return self.outcome in ("recovered", "degraded")

    @property
    def context_hash(self) -> str:
        """Generate hash of context for deduplication."""
        context_str = str(sorted(self.context.items()))
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component.value,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "severity": self.severity.value,
            "context": self.context,
            "stack_trace": self.stack_trace,
            "fallback_used": self.fallback_used,
            "outcome": self.outcome,
            "recovery_attempted": self.recovery_attempted,
            "duration_ms": self.duration_ms,
            "is_recoverable": self.is_recoverable,
            "metadata": self.metadata,
        }

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        component: FailureComponent,
        context: Optional[Dict[str, Any]] = None,
        severity: FailureSeverity = FailureSeverity.ERROR,
    ) -> "FailureLog":
        """Create failure log from an exception."""
        import traceback

        return cls(
            failure_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            component=component,
            error_type=type(exception).__name__,
            error_message=str(exception),
            severity=severity,
            context=context or {},
            stack_trace=traceback.format_exc(),
        )


@dataclass
class FailureQuery:
    """
    Query parameters for searching failure logs.

    Attributes:
        component: Filter by component
        severity_min: Minimum severity level
        error_type: Filter by error type
        since: Only failures after this time
        until: Only failures before this time
        outcome: Filter by outcome
        limit: Maximum results to return
    """
    component: Optional[FailureComponent] = None
    severity_min: Optional[FailureSeverity] = None
    error_type: Optional[str] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    outcome: Optional[str] = None
    limit: int = 100


@dataclass
class FailureStats:
    """
    Statistics about failures.

    Attributes:
        total_count: Total number of failures
        by_component: Counts per component
        by_severity: Counts per severity level
        by_error_type: Counts per error type
        by_outcome: Counts per outcome
        recovery_rate: Percentage of recovered failures
        mean_duration_ms: Mean failure duration
        period_start: Start of analysis period
        period_end: End of analysis period
    """
    total_count: int
    by_component: Dict[str, int]
    by_severity: Dict[str, int]
    by_error_type: Dict[str, int]
    by_outcome: Dict[str, int]
    recovery_rate: float
    mean_duration_ms: Optional[float]
    period_start: datetime
    period_end: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "by_component": self.by_component,
            "by_severity": self.by_severity,
            "by_error_type": self.by_error_type,
            "by_outcome": self.by_outcome,
            "recovery_rate": round(self.recovery_rate, 3),
            "mean_duration_ms": round(self.mean_duration_ms, 2) if self.mean_duration_ms else None,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
        }


class FailureLogger:
    """
    Structured failure logging system.

    Logs failures with full context, supports querying failure history,
    and provides analytics for identifying patterns.

    Example:
        logger = FailureLogger()

        # Log a failure
        failure = logger.log_failure(
            component=FailureComponent.DISCOVERY,
            error_type="timeout",
            error_message="STAC query timed out after 30s",
            context={"catalog": "earth-search", "bbox": [...]}
        )

        # Query recent failures
        recent = logger.query(FailureQuery(
            component=FailureComponent.DISCOVERY,
            since=datetime.now() - timedelta(hours=1)
        ))

        # Get statistics
        stats = logger.get_statistics()
        print(f"Recovery rate: {stats.recovery_rate:.1%}")
    """

    def __init__(self, max_history: int = 10000):
        """
        Initialize the failure logger.

        Args:
            max_history: Maximum number of failures to keep in memory
        """
        self._failures: List[FailureLog] = []
        self._max_history = max_history
        self._callbacks: List[Callable[[FailureLog], None]] = []

    def log_failure(
        self,
        component: FailureComponent,
        error_type: str,
        error_message: str,
        severity: FailureSeverity = FailureSeverity.ERROR,
        context: Optional[Dict[str, Any]] = None,
        stack_trace: Optional[str] = None,
        fallback_used: Optional[str] = None,
        outcome: str = "failed",
        recovery_attempted: bool = False,
        duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureLog:
        """
        Log a failure with full context.

        Args:
            component: Which component failed
            error_type: Classification of the error
            error_message: Human-readable description
            severity: Failure severity
            context: What was being processed
            stack_trace: Optional stack trace
            fallback_used: Fallback mechanism used
            outcome: Final outcome
            recovery_attempted: Whether recovery was tried
            duration_ms: Operation duration before failure
            metadata: Additional information

        Returns:
            FailureLog record
        """
        failure = FailureLog(
            failure_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            component=component,
            error_type=error_type,
            error_message=error_message,
            severity=severity,
            context=context or {},
            stack_trace=stack_trace,
            fallback_used=fallback_used,
            outcome=outcome,
            recovery_attempted=recovery_attempted,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        self._add_failure(failure)

        # Log to standard logger
        log_level = {
            FailureSeverity.DEBUG: logging.DEBUG,
            FailureSeverity.INFO: logging.INFO,
            FailureSeverity.WARNING: logging.WARNING,
            FailureSeverity.ERROR: logging.ERROR,
            FailureSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.ERROR)

        logger.log(
            log_level,
            f"[{component.value}] {error_type}: {error_message} "
            f"(outcome={outcome}, fallback={fallback_used})"
        )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(failure)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")

        return failure

    def log_exception(
        self,
        exception: Exception,
        component: FailureComponent,
        context: Optional[Dict[str, Any]] = None,
        severity: FailureSeverity = FailureSeverity.ERROR,
    ) -> FailureLog:
        """
        Log a failure from an exception.

        Args:
            exception: The exception that occurred
            component: Which component failed
            context: What was being processed
            severity: Failure severity

        Returns:
            FailureLog record
        """
        failure = FailureLog.from_exception(
            exception, component, context, severity
        )
        self._add_failure(failure)

        logger.exception(
            f"[{component.value}] {type(exception).__name__}: {exception}"
        )

        return failure

    def update_outcome(
        self,
        failure_id: str,
        outcome: str,
        fallback_used: Optional[str] = None,
    ) -> Optional[FailureLog]:
        """
        Update the outcome of a previously logged failure.

        Args:
            failure_id: ID of the failure to update
            outcome: New outcome status
            fallback_used: Fallback that was used (if any)

        Returns:
            Updated FailureLog or None if not found
        """
        for failure in self._failures:
            if failure.failure_id == failure_id:
                failure.outcome = outcome
                failure.recovery_attempted = True
                if fallback_used:
                    failure.fallback_used = fallback_used
                return failure
        return None

    def query(self, query: FailureQuery) -> List[FailureLog]:
        """
        Query failure history.

        Args:
            query: Query parameters

        Returns:
            List of matching FailureLog entries
        """
        results = []

        for failure in reversed(self._failures):
            # Apply filters
            if query.component and failure.component != query.component:
                continue
            if query.severity_min and failure.severity.level < query.severity_min.level:
                continue
            if query.error_type and failure.error_type != query.error_type:
                continue
            if query.since and failure.timestamp < query.since:
                continue
            if query.until and failure.timestamp > query.until:
                continue
            if query.outcome and failure.outcome != query.outcome:
                continue

            results.append(failure)

            if len(results) >= query.limit:
                break

        return results

    def get_recent(
        self,
        count: int = 10,
        component: Optional[FailureComponent] = None,
    ) -> List[FailureLog]:
        """
        Get most recent failures.

        Args:
            count: Number of failures to return
            component: Optional component filter

        Returns:
            List of recent FailureLog entries
        """
        return self.query(FailureQuery(component=component, limit=count))

    def get_statistics(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> FailureStats:
        """
        Get failure statistics for a time period.

        Args:
            since: Start of analysis period (default: 24 hours ago)
            until: End of analysis period (default: now)

        Returns:
            FailureStats with aggregated statistics
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)
        if until is None:
            until = datetime.now(timezone.utc)

        # Filter failures to period
        period_failures = [
            f for f in self._failures
            if since <= f.timestamp <= until
        ]

        if not period_failures:
            return FailureStats(
                total_count=0,
                by_component={},
                by_severity={},
                by_error_type={},
                by_outcome={},
                recovery_rate=0.0,
                mean_duration_ms=None,
                period_start=since,
                period_end=until,
            )

        # Aggregate counts
        by_component: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_error_type: Dict[str, int] = {}
        by_outcome: Dict[str, int] = {}
        recovered_count = 0
        durations = []

        for failure in period_failures:
            # Count by component
            comp = failure.component.value
            by_component[comp] = by_component.get(comp, 0) + 1

            # Count by severity
            sev = failure.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1

            # Count by error type
            by_error_type[failure.error_type] = by_error_type.get(failure.error_type, 0) + 1

            # Count by outcome
            by_outcome[failure.outcome] = by_outcome.get(failure.outcome, 0) + 1

            # Track recovery
            if failure.is_recoverable:
                recovered_count += 1

            # Track durations
            if failure.duration_ms is not None:
                durations.append(failure.duration_ms)

        total_count = len(period_failures)
        recovery_rate = recovered_count / total_count if total_count > 0 else 0.0
        mean_duration = sum(durations) / len(durations) if durations else None

        return FailureStats(
            total_count=total_count,
            by_component=by_component,
            by_severity=by_severity,
            by_error_type=by_error_type,
            by_outcome=by_outcome,
            recovery_rate=recovery_rate,
            mean_duration_ms=mean_duration,
            period_start=since,
            period_end=until,
        )

    def find_patterns(
        self,
        min_occurrences: int = 3,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find recurring failure patterns.

        Args:
            min_occurrences: Minimum occurrences for a pattern
            since: Only analyze failures after this time

        Returns:
            List of identified patterns
        """
        if since is None:
            since = datetime.now(timezone.utc) - timedelta(hours=24)

        # Group by component and error type
        pattern_counts: Dict[Tuple[str, str], List[FailureLog]] = {}

        for failure in self._failures:
            if failure.timestamp < since:
                continue

            key = (failure.component.value, failure.error_type)
            if key not in pattern_counts:
                pattern_counts[key] = []
            pattern_counts[key].append(failure)

        # Find patterns above threshold
        patterns = []
        for (component, error_type), failures in pattern_counts.items():
            if len(failures) >= min_occurrences:
                patterns.append({
                    "component": component,
                    "error_type": error_type,
                    "occurrence_count": len(failures),
                    "first_occurrence": failures[0].timestamp.isoformat(),
                    "last_occurrence": failures[-1].timestamp.isoformat(),
                    "recovery_rate": sum(1 for f in failures if f.is_recoverable) / len(failures),
                    "sample_contexts": [f.context for f in failures[:3]],
                })

        return sorted(patterns, key=lambda p: p["occurrence_count"], reverse=True)

    def export_report(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Export a comprehensive failure report.

        Args:
            since: Start of report period
            until: End of report period

        Returns:
            Dictionary with full report
        """
        stats = self.get_statistics(since, until)
        patterns = self.find_patterns(since=since or (datetime.now(timezone.utc) - timedelta(hours=24)))
        recent = self.get_recent(20)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": stats.to_dict(),
            "patterns": patterns,
            "recent_failures": [f.to_dict() for f in recent],
        }

    def add_callback(self, callback: Callable[[FailureLog], None]) -> None:
        """
        Add a callback to be notified of new failures.

        Args:
            callback: Function to call with new FailureLog entries
        """
        self._callbacks.append(callback)

    def clear_history(self) -> int:
        """
        Clear all failure history.

        Returns:
            Number of entries cleared
        """
        count = len(self._failures)
        self._failures = []
        return count

    def _add_failure(self, failure: FailureLog) -> None:
        """Add failure to history, pruning if necessary."""
        self._failures.append(failure)

        # Prune oldest entries if over limit
        if len(self._failures) > self._max_history:
            self._failures = self._failures[-self._max_history:]


# Global failure logger instance
_global_logger: Optional[FailureLogger] = None


def get_failure_logger() -> FailureLogger:
    """Get or create the global failure logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = FailureLogger()
    return _global_logger


def log_failure(
    component: FailureComponent,
    error_type: str,
    error_message: str,
    **kwargs: Any,
) -> FailureLog:
    """
    Convenience function to log a failure.

    Args:
        component: Which component failed
        error_type: Classification of the error
        error_message: Human-readable description
        **kwargs: Additional arguments for FailureLogger.log_failure

    Returns:
        FailureLog record

    Example:
        failure = log_failure(
            FailureComponent.DISCOVERY,
            "timeout",
            "STAC query timed out",
            context={"catalog": "earth-search"},
        )
    """
    return get_failure_logger().log_failure(
        component, error_type, error_message, **kwargs
    )
