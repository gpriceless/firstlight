"""
Failure Documentation & Recovery Module.

Provides comprehensive failure handling capabilities:

- Failure Logging: Structured logging with full context
- Recovery Strategies: Retry, alternative sources, graceful degradation
- Provenance Tracking: Audit trail for all fallback decisions

When failures occur, this module ensures they are properly documented,
recovery is attempted systematically, and the full decision history
is available for reproducibility and debugging.

Example:
    from core.resilience.failure import (
        FailureLog,
        FailureLogger,
        RecoveryStrategy,
        RecoveryOrchestrator,
        FallbackProvenanceTracker,
    )

    # Log failures
    logger = FailureLogger()
    logger.log_failure(
        component="discovery",
        error_type="timeout",
        error_message="STAC catalog query timed out",
        context={"catalog": "earth-search", "timeout": 30},
    )

    # Orchestrate recovery
    orchestrator = RecoveryOrchestrator()
    result = orchestrator.attempt_recovery(
        operation=lambda: fetch_data(),
        fallbacks=[alternative_source, cached_data],
    )

    # Track provenance
    tracker = FallbackProvenanceTracker()
    tracker.record_fallback(
        original_strategy="sentinel2_optical",
        fallback_strategy="landsat8_optical",
        reason="Cloud cover exceeded threshold",
    )
"""

from core.resilience.failure.failure_log import (
    FailureComponent,
    FailureSeverity,
    FailureLog,
    FailureQuery,
    FailureStats,
    FailureLogger,
    log_failure,
)

from core.resilience.failure.recovery_strategies import (
    RecoveryStrategy,
    RecoveryAttempt,
    RecoveryResult,
    RecoveryConfig,
    RecoveryOrchestrator,
    retry_with_backoff,
)

from core.resilience.failure.provenance_tracking import (
    FallbackDecision,
    FallbackChain,
    ProvenanceRecord,
    FallbackProvenanceConfig,
    FallbackProvenanceTracker,
    create_provenance_record,
)

__all__ = [
    # Failure logging
    "FailureComponent",
    "FailureSeverity",
    "FailureLog",
    "FailureQuery",
    "FailureStats",
    "FailureLogger",
    "log_failure",
    # Recovery strategies
    "RecoveryStrategy",
    "RecoveryAttempt",
    "RecoveryResult",
    "RecoveryConfig",
    "RecoveryOrchestrator",
    "retry_with_backoff",
    # Provenance tracking
    "FallbackDecision",
    "FallbackChain",
    "ProvenanceRecord",
    "FallbackProvenanceConfig",
    "FallbackProvenanceTracker",
    "create_provenance_record",
]
