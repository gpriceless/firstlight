"""
Quality Action Management.

This module provides tools for acting on quality control results:
- **Gating**: Pass/fail/review decisions based on QC checks
- **Flagging**: Quality flag management and summarization
- **Routing**: Expert review routing and workflow management

Usage:
    from core.quality.actions import (
        QualityGate,
        QualityFlagger,
        ReviewRouter,
        GateStatus,
        StandardFlag,
        ReviewPriority,
    )

    # Gating example
    gate = QualityGate()
    decision = gate.evaluate(checks, context)
    if decision.status == GateStatus.REVIEW_REQUIRED:
        # Route to expert review
        ...

    # Flagging example
    flagger = QualityFlagger()
    flagger.apply_standard_flag(product_id, StandardFlag.LOW_CONFIDENCE)
    summary = flagger.summarize(product_id, event_id)

    # Routing example
    router = ReviewRouter()
    router.register_expert(expert)
    request = router.create_request(
        ReviewType.QUALITY_VALIDATION,
        ExpertDomain.FLOOD,
        ReviewPriority.NORMAL,
        context,
    )
"""

# Gating exports
from core.quality.actions.gating import (
    CheckCategory,
    CheckStatus,
    GateStatus,
    GatingContext,
    GatingDecision,
    GatingRule,
    GatingThresholds,
    QCCheck,
    QualityGate,
    RuleSeverity,
    create_emergency_gate,
    create_operational_gate,
    create_research_gate,
    quick_gate,
)

# Flagging exports
from core.quality.actions.flagging import (
    AppliedFlag,
    FlagDefinition,
    FlagLevel,
    FlagRegistry,
    FlagSeverity,
    FlagSummary,
    QualityFlagger,
    StandardFlag,
    create_confidence_flag,
    flag_from_conditions,
)

# Routing exports
from core.quality.actions.routing import (
    Expert,
    ExpertDomain,
    ReviewContext,
    ReviewOutcome,
    ReviewPriority,
    ReviewRequest,
    ReviewRouter,
    ReviewStatus,
    ReviewType,
    RoutingStrategy,
    create_review_from_gating,
    quick_route,
)

__all__ = [
    # Gating
    "CheckCategory",
    "CheckStatus",
    "GateStatus",
    "GatingContext",
    "GatingDecision",
    "GatingRule",
    "GatingThresholds",
    "QCCheck",
    "QualityGate",
    "RuleSeverity",
    "create_emergency_gate",
    "create_operational_gate",
    "create_research_gate",
    "quick_gate",
    # Flagging
    "AppliedFlag",
    "FlagDefinition",
    "FlagLevel",
    "FlagRegistry",
    "FlagSeverity",
    "FlagSummary",
    "QualityFlagger",
    "StandardFlag",
    "create_confidence_flag",
    "flag_from_conditions",
    # Routing
    "Expert",
    "ExpertDomain",
    "ReviewContext",
    "ReviewOutcome",
    "ReviewPriority",
    "ReviewRequest",
    "ReviewRouter",
    "ReviewStatus",
    "ReviewType",
    "RoutingStrategy",
    "create_review_from_gating",
    "quick_route",
]
