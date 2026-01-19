"""
Quality Agent Module.

The Quality Agent handles all validation, QC, and quality gating decisions
for the FirstLight platform.

Components:
- QualityAgent: Main agent class orchestrating all QC operations
- ValidationRunner: Executes validation checks (sanity, cross-validation)
- ReviewManager: Manages human review workflows
- QAReportGenerator: Generates quality reports in multiple formats

The Quality Agent integrates core quality modules:
- core/quality/sanity/ - Spatial, temporal, value, artifact checks
- core/quality/validation/ - Cross-model, cross-sensor, historical, consensus
- core/quality/uncertainty/ - Quantification, spatial, propagation
- core/quality/actions/ - Gating, flagging, routing
- core/quality/reporting/ - QA reports, diagnostics

Example:
    from agents.quality import (
        QualityAgent,
        ValidationRunner,
        ReviewManager,
        QAReportGenerator,
    )

    # Create and start Quality Agent
    agent = QualityAgent()
    await agent.initialize()
    await agent.start()

    # Validate a step result
    decision = await agent.validate_step("step_001", result_data)

    # Generate final validation report
    report = await agent.validate_final(final_results)

    # Use validation runner directly
    runner = ValidationRunner()
    sanity_result = await runner.run_sanity_checks(data)

    # Manage reviews
    manager = ReviewManager()
    review = await manager.create_review_request(
        event_id="evt_001",
        product_id="prod_001",
        review_type=ReviewType.QUALITY_VALIDATION,
    )

    # Generate reports
    generator = QAReportGenerator()
    report = await generator.generate_report(
        event_id="evt_001",
        product_id="prod_001",
        qa_results=results,
    )
"""

# Main Quality Agent
from agents.quality.main import (
    QualityAgent,
    QualityAgentConfig,
    QAGateDecision,
    SanityResult,
    ValidationResult,
    UncertaintyMap,
    GateDecision,
    QAReport,
)

# Validation Runner
from agents.quality.validation import (
    ValidationRunner,
    ValidationRunnerConfig,
    ValidationMode,
    SanityCheckResult,
    ModelValidationResult,
    SensorValidationResult,
    HistoricalValidationResult,
)

# Review Manager
from agents.quality.review import (
    ReviewManager,
    ReviewManagerConfig,
    ActiveReview,
    ReviewOutcomeResult,
    EscalationTrigger,
    EscalationRule,
)

# Report Generator
from agents.quality.reporting import (
    QAReportGenerator,
    ReportGeneratorConfig,
    ReportType,
    GeneratedReport,
    ReportSummary,
)

# Re-export commonly used types from core modules
from core.quality.actions import (
    GateStatus,
    ReviewPriority,
    ReviewType,
    ReviewStatus,
    ExpertDomain,
    Expert,
)

from core.quality.reporting import (
    ReportFormat,
    ReportLevel,
)

__all__ = [
    # Main Agent
    "QualityAgent",
    "QualityAgentConfig",
    "QAGateDecision",
    "SanityResult",
    "ValidationResult",
    "UncertaintyMap",
    "GateDecision",
    "QAReport",
    # Validation Runner
    "ValidationRunner",
    "ValidationRunnerConfig",
    "ValidationMode",
    "SanityCheckResult",
    "ModelValidationResult",
    "SensorValidationResult",
    "HistoricalValidationResult",
    # Review Manager
    "ReviewManager",
    "ReviewManagerConfig",
    "ActiveReview",
    "ReviewOutcomeResult",
    "EscalationTrigger",
    "EscalationRule",
    # Report Generator
    "QAReportGenerator",
    "ReportGeneratorConfig",
    "ReportType",
    "GeneratedReport",
    "ReportSummary",
    # Core types
    "GateStatus",
    "ReviewPriority",
    "ReviewType",
    "ReviewStatus",
    "ExpertDomain",
    "Expert",
    "ReportFormat",
    "ReportLevel",
]
