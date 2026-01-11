"""
Quality Gating - Pass/Fail/Review Decision Logic.

Provides a comprehensive gating system for quality control decisions,
determining whether analysis outputs should be:
- PASS: Released immediately
- PASS_WITH_WARNINGS: Released with caveats documented
- REVIEW_REQUIRED: Held for expert review
- BLOCKED: Not released due to quality issues

Key Concepts:
- Gates are composed of multiple rules evaluated against QC results
- Rules can be mandatory (fail = block) or advisory (fail = review)
- Confidence thresholds adapt to degraded mode operation
- All decisions are logged with full rationale
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GateStatus(Enum):
    """Possible outcomes of quality gating."""
    PASS = "PASS"                           # All checks passed, release immediately
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"  # Passed with documented caveats
    REVIEW_REQUIRED = "REVIEW_REQUIRED"     # Held for expert review
    BLOCKED = "BLOCKED"                     # Quality too low, do not release


class RuleSeverity(Enum):
    """Severity of a gating rule."""
    MANDATORY = "mandatory"    # Failure blocks release
    CRITICAL = "critical"      # Failure requires review
    ADVISORY = "advisory"      # Failure adds warning only
    INFORMATIONAL = "informational"  # Logged but no impact


class CheckCategory(Enum):
    """Categories of quality checks."""
    SPATIAL = "spatial"             # Spatial coherence checks
    VALUE = "value"                 # Value plausibility checks
    TEMPORAL = "temporal"           # Temporal consistency checks
    ARTIFACT = "artifact"           # Artifact detection
    CROSS_VALIDATION = "cross_validation"  # Cross-validation metrics
    HISTORICAL = "historical"       # Historical baseline comparisons
    UNCERTAINTY = "uncertainty"     # Uncertainty quantification


class CheckStatus(Enum):
    """Status of an individual check."""
    PASS = "pass"
    WARNING = "warning"
    SOFT_FAIL = "soft_fail"
    HARD_FAIL = "hard_fail"


@dataclass
class QCCheck:
    """
    Result of a quality check.

    Attributes:
        check_name: Unique identifier for the check
        category: Category of this check
        status: Check result status
        metric_value: Numeric metric value if applicable
        threshold: Threshold used for evaluation
        details: Human-readable description
        spatial_extent: GeoJSON geometry of affected area (optional)
    """
    check_name: str
    category: CheckCategory
    status: CheckStatus
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    details: str = ""
    spatial_extent: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "check_name": self.check_name,
            "category": self.category.value,
            "status": self.status.value,
            "details": self.details,
        }
        if self.metric_value is not None:
            result["metric_value"] = self.metric_value
        if self.threshold is not None:
            result["threshold"] = self.threshold
        if self.spatial_extent is not None:
            result["spatial_extent"] = self.spatial_extent
        return result


@dataclass
class GatingRule:
    """
    A rule for gating decisions.

    Attributes:
        rule_id: Unique identifier
        name: Human-readable name
        description: What this rule checks
        severity: How failures are treated
        check_categories: Which check categories this rule applies to
        check_function: Function to evaluate the rule
        enabled: Whether this rule is active
    """
    rule_id: str
    name: str
    description: str
    severity: RuleSeverity
    check_categories: List[CheckCategory]
    check_function: Callable[["GatingContext", List[QCCheck]], Tuple[bool, str]]
    enabled: bool = True

    def applies_to(self, check: QCCheck) -> bool:
        """Check if this rule applies to a given check."""
        if not self.check_categories:
            return True
        return check.category in self.check_categories

    def evaluate(self, context: "GatingContext", checks: List[QCCheck]) -> Tuple[bool, str]:
        """
        Evaluate this rule.

        Args:
            context: Gating context with thresholds and configuration
            checks: QC checks relevant to this rule

        Returns:
            (passed, reason) tuple
        """
        if not self.enabled:
            return True, f"{self.name}: disabled"

        try:
            return self.check_function(context, checks)
        except Exception as e:
            logger.error(f"Rule {self.rule_id} evaluation error: {e}")
            # Fail safe for mandatory rules
            if self.severity == RuleSeverity.MANDATORY:
                return False, f"{self.name}: evaluation error - {str(e)}"
            return True, f"{self.name}: evaluation error (ignored) - {str(e)}"


@dataclass
class GatingThresholds:
    """
    Configurable thresholds for gating decisions.

    Attributes:
        min_confidence: Minimum confidence score to pass
        max_hard_fails: Maximum hard failures before blocking
        max_soft_fails: Maximum soft failures before review
        max_warnings: Maximum warnings before review
        min_coverage_percent: Minimum spatial coverage
        max_uncertainty_mean: Maximum mean uncertainty
        min_agreement_score: Minimum cross-validation agreement
        degraded_mode_confidence_factor: Multiplier for degraded mode
    """
    min_confidence: float = 0.6
    max_hard_fails: int = 0
    max_soft_fails: int = 2
    max_warnings: int = 5
    min_coverage_percent: float = 80.0
    max_uncertainty_mean: float = 0.3
    min_agreement_score: float = 0.7
    degraded_mode_confidence_factor: float = 0.8

    def adjust_for_degraded_mode(self, level: int) -> "GatingThresholds":
        """
        Return adjusted thresholds for degraded mode operation.

        Args:
            level: Degraded mode level (0=normal, 1-4=increasing degradation)

        Returns:
            Adjusted thresholds
        """
        if level == 0:
            return self

        # Each level relaxes thresholds
        factor = self.degraded_mode_confidence_factor ** level

        return GatingThresholds(
            min_confidence=max(0.3, self.min_confidence * factor),
            max_hard_fails=self.max_hard_fails,  # Never relax hard fail limit
            max_soft_fails=self.max_soft_fails + level,
            max_warnings=self.max_warnings + level * 2,
            min_coverage_percent=max(50.0, self.min_coverage_percent - level * 10),
            max_uncertainty_mean=min(0.6, self.max_uncertainty_mean + level * 0.1),
            min_agreement_score=max(0.4, self.min_agreement_score - level * 0.1),
            degraded_mode_confidence_factor=self.degraded_mode_confidence_factor,
        )


@dataclass
class GatingContext:
    """
    Context for gating decisions.

    Attributes:
        event_id: Event being processed
        product_id: Product being evaluated
        thresholds: Gating thresholds
        degraded_mode_level: Current degraded mode level (0=normal)
        confidence_score: Overall confidence from analysis
        uncertainty_summary: Uncertainty metrics
        cross_validation: Cross-validation results
        metadata: Additional context
    """
    event_id: str
    product_id: str
    thresholds: GatingThresholds = field(default_factory=GatingThresholds)
    degraded_mode_level: int = 0
    confidence_score: float = 1.0
    uncertainty_summary: Optional[Dict[str, float]] = None
    cross_validation: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_thresholds(self) -> GatingThresholds:
        """Get thresholds adjusted for degraded mode."""
        return self.thresholds.adjust_for_degraded_mode(self.degraded_mode_level)


@dataclass
class GatingDecision:
    """
    Result of a gating evaluation.

    Attributes:
        status: Final gate status
        rule_results: Results from each evaluated rule
        failed_mandatory: Mandatory rules that failed
        failed_critical: Critical rules that failed
        warnings: Warning messages
        confidence_modifier: Adjustment to confidence based on issues
        rationale: Human-readable explanation of decision
        timestamp: When decision was made
    """
    status: GateStatus
    rule_results: Dict[str, Tuple[bool, str]] = field(default_factory=dict)
    failed_mandatory: List[str] = field(default_factory=list)
    failed_critical: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    confidence_modifier: float = 1.0
    rationale: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "rule_results": {
                rule_id: {"passed": passed, "reason": reason}
                for rule_id, (passed, reason) in self.rule_results.items()
            },
            "failed_mandatory": self.failed_mandatory,
            "failed_critical": self.failed_critical,
            "warnings": self.warnings,
            "confidence_modifier": self.confidence_modifier,
            "rationale": self.rationale,
            "timestamp": self.timestamp.isoformat(),
        }


class QualityGate:
    """
    Main quality gating engine.

    Evaluates QC checks against configurable rules and thresholds
    to produce pass/fail/review decisions.
    """

    def __init__(self, thresholds: Optional[GatingThresholds] = None):
        """
        Initialize quality gate.

        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or GatingThresholds()
        self.rules: List[GatingRule] = []
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default gating rules."""
        # Mandatory: No hard failures
        self.register_rule(GatingRule(
            rule_id="no_hard_failures",
            name="No Hard Failures",
            description="Blocks if any check has hard failure status",
            severity=RuleSeverity.MANDATORY,
            check_categories=[],  # All categories
            check_function=self._rule_no_hard_failures,
        ))

        # Mandatory: Minimum confidence
        self.register_rule(GatingRule(
            rule_id="min_confidence",
            name="Minimum Confidence",
            description="Blocks if confidence below threshold",
            severity=RuleSeverity.MANDATORY,
            check_categories=[],
            check_function=self._rule_min_confidence,
        ))

        # Critical: Soft failure limit
        self.register_rule(GatingRule(
            rule_id="soft_failure_limit",
            name="Soft Failure Limit",
            description="Requires review if too many soft failures",
            severity=RuleSeverity.CRITICAL,
            check_categories=[],
            check_function=self._rule_soft_failure_limit,
        ))

        # Critical: Cross-validation agreement
        self.register_rule(GatingRule(
            rule_id="cross_validation_agreement",
            name="Cross-Validation Agreement",
            description="Requires review if methods disagree significantly",
            severity=RuleSeverity.CRITICAL,
            check_categories=[CheckCategory.CROSS_VALIDATION],
            check_function=self._rule_cross_validation_agreement,
        ))

        # Critical: Uncertainty threshold
        self.register_rule(GatingRule(
            rule_id="uncertainty_threshold",
            name="Uncertainty Threshold",
            description="Requires review if mean uncertainty is too high",
            severity=RuleSeverity.CRITICAL,
            check_categories=[CheckCategory.UNCERTAINTY],
            check_function=self._rule_uncertainty_threshold,
        ))

        # Advisory: Spatial coherence
        self.register_rule(GatingRule(
            rule_id="spatial_coherence",
            name="Spatial Coherence",
            description="Warns if spatial checks have issues",
            severity=RuleSeverity.ADVISORY,
            check_categories=[CheckCategory.SPATIAL],
            check_function=self._rule_spatial_coherence,
        ))

        # Advisory: Value plausibility
        self.register_rule(GatingRule(
            rule_id="value_plausibility",
            name="Value Plausibility",
            description="Warns if values are questionable",
            severity=RuleSeverity.ADVISORY,
            check_categories=[CheckCategory.VALUE],
            check_function=self._rule_value_plausibility,
        ))

        # Advisory: Warning limit
        self.register_rule(GatingRule(
            rule_id="warning_limit",
            name="Warning Limit",
            description="Notes excessive warnings",
            severity=RuleSeverity.ADVISORY,
            check_categories=[],
            check_function=self._rule_warning_limit,
        ))

        # Informational: Historical deviation
        self.register_rule(GatingRule(
            rule_id="historical_deviation",
            name="Historical Deviation",
            description="Logs deviation from historical baselines",
            severity=RuleSeverity.INFORMATIONAL,
            check_categories=[CheckCategory.HISTORICAL],
            check_function=self._rule_historical_deviation,
        ))

    def register_rule(self, rule: GatingRule) -> None:
        """Register a gating rule."""
        # Remove existing rule with same ID if present
        self.rules = [r for r in self.rules if r.rule_id != rule.rule_id]
        self.rules.append(rule)
        logger.debug(f"Registered gating rule: {rule.rule_id}")

    def unregister_rule(self, rule_id: str) -> bool:
        """
        Unregister a gating rule.

        Args:
            rule_id: ID of rule to remove

        Returns:
            True if rule was found and removed
        """
        original_count = len(self.rules)
        self.rules = [r for r in self.rules if r.rule_id != rule_id]
        return len(self.rules) < original_count

    def evaluate(
        self,
        checks: List[QCCheck],
        context: Optional[GatingContext] = None,
        event_id: str = "",
        product_id: str = "",
    ) -> GatingDecision:
        """
        Evaluate QC checks and produce a gating decision.

        Args:
            checks: List of QC check results
            context: Optional gating context (created from params if None)
            event_id: Event ID (used if context is None)
            product_id: Product ID (used if context is None)

        Returns:
            Gating decision with status and rationale
        """
        # Create context if not provided
        if context is None:
            context = GatingContext(
                event_id=event_id,
                product_id=product_id,
                thresholds=self.thresholds,
            )

        # Evaluate all rules
        rule_results: Dict[str, Tuple[bool, str]] = {}
        failed_mandatory: List[str] = []
        failed_critical: List[str] = []
        warnings: List[str] = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            # Filter checks relevant to this rule
            relevant_checks = [c for c in checks if rule.applies_to(c)]

            # Evaluate rule
            passed, reason = rule.evaluate(context, relevant_checks)
            rule_results[rule.rule_id] = (passed, reason)

            if not passed:
                if rule.severity == RuleSeverity.MANDATORY:
                    failed_mandatory.append(f"{rule.name}: {reason}")
                elif rule.severity == RuleSeverity.CRITICAL:
                    failed_critical.append(f"{rule.name}: {reason}")
                elif rule.severity == RuleSeverity.ADVISORY:
                    warnings.append(f"{rule.name}: {reason}")
                # Informational failures are just logged

        # Determine final status
        status, rationale, confidence_modifier = self._determine_status(
            failed_mandatory,
            failed_critical,
            warnings,
            context,
        )

        return GatingDecision(
            status=status,
            rule_results=rule_results,
            failed_mandatory=failed_mandatory,
            failed_critical=failed_critical,
            warnings=warnings,
            confidence_modifier=confidence_modifier,
            rationale=rationale,
        )

    def _determine_status(
        self,
        failed_mandatory: List[str],
        failed_critical: List[str],
        warnings: List[str],
        context: GatingContext,
    ) -> Tuple[GateStatus, str, float]:
        """
        Determine final gating status from rule results.

        Returns:
            (status, rationale, confidence_modifier)
        """
        # Mandatory failures = BLOCKED
        if failed_mandatory:
            rationale = f"BLOCKED: {len(failed_mandatory)} mandatory rule(s) failed. " + \
                        "; ".join(failed_mandatory[:3])
            if len(failed_mandatory) > 3:
                rationale += f" (and {len(failed_mandatory) - 3} more)"
            return GateStatus.BLOCKED, rationale, 0.0

        # Critical failures = REVIEW_REQUIRED
        if failed_critical:
            rationale = f"REVIEW_REQUIRED: {len(failed_critical)} critical issue(s). " + \
                        "; ".join(failed_critical[:3])
            if len(failed_critical) > 3:
                rationale += f" (and {len(failed_critical) - 3} more)"
            # Apply confidence modifier based on severity
            modifier = max(0.5, 1.0 - 0.1 * len(failed_critical))
            return GateStatus.REVIEW_REQUIRED, rationale, modifier

        # Warnings = PASS_WITH_WARNINGS
        if warnings:
            rationale = f"PASS_WITH_WARNINGS: {len(warnings)} warning(s) documented. " + \
                        "; ".join(warnings[:3])
            if len(warnings) > 3:
                rationale += f" (and {len(warnings) - 3} more)"
            # Small confidence adjustment for warnings
            modifier = max(0.8, 1.0 - 0.02 * len(warnings))
            return GateStatus.PASS_WITH_WARNINGS, rationale, modifier

        # No issues = PASS
        if context.degraded_mode_level > 0:
            rationale = f"PASS: All checks passed (degraded mode level {context.degraded_mode_level})"
        else:
            rationale = "PASS: All quality checks passed"
        return GateStatus.PASS, rationale, 1.0

    # ==================== Rule Implementations ====================

    def _rule_no_hard_failures(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: No hard failures allowed."""
        hard_fails = [c for c in checks if c.status == CheckStatus.HARD_FAIL]
        threshold = context.effective_thresholds.max_hard_fails

        if len(hard_fails) > threshold:
            fail_names = [c.check_name for c in hard_fails[:3]]
            return False, f"{len(hard_fails)} hard failure(s): {', '.join(fail_names)}"
        return True, "No hard failures"

    def _rule_min_confidence(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Minimum confidence score."""
        threshold = context.effective_thresholds.min_confidence

        if context.confidence_score < threshold:
            return False, f"Confidence {context.confidence_score:.2f} < {threshold:.2f}"
        return True, f"Confidence {context.confidence_score:.2f} >= {threshold:.2f}"

    def _rule_soft_failure_limit(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Soft failure count limit."""
        soft_fails = [c for c in checks if c.status == CheckStatus.SOFT_FAIL]
        threshold = context.effective_thresholds.max_soft_fails

        if len(soft_fails) > threshold:
            return False, f"{len(soft_fails)} soft failures exceeds limit of {threshold}"
        return True, f"{len(soft_fails)} soft failure(s) within limit"

    def _rule_cross_validation_agreement(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Cross-validation agreement threshold."""
        threshold = context.effective_thresholds.min_agreement_score

        # Check for cross-validation metrics in context
        if context.cross_validation:
            agreement = context.cross_validation.get("agreement_metrics", {})
            iou = agreement.get("iou")
            kappa = agreement.get("kappa")

            if iou is not None and iou < threshold:
                return False, f"IoU {iou:.2f} < {threshold:.2f}"
            if kappa is not None and kappa < threshold:
                return False, f"Kappa {kappa:.2f} < {threshold:.2f}"

        # Also check individual checks
        cv_checks = [c for c in checks if c.category == CheckCategory.CROSS_VALIDATION]
        failed = [c for c in cv_checks if c.status in (CheckStatus.SOFT_FAIL, CheckStatus.HARD_FAIL)]

        if failed:
            return False, f"{len(failed)} cross-validation check(s) failed"
        return True, "Cross-validation agreement acceptable"

    def _rule_uncertainty_threshold(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Mean uncertainty threshold."""
        threshold = context.effective_thresholds.max_uncertainty_mean

        if context.uncertainty_summary:
            mean_unc = context.uncertainty_summary.get("mean_uncertainty")
            if mean_unc is not None and mean_unc > threshold:
                return False, f"Mean uncertainty {mean_unc:.2f} > {threshold:.2f}"

        # Check uncertainty-related QC checks
        unc_checks = [c for c in checks if c.category == CheckCategory.UNCERTAINTY]
        failed = [c for c in unc_checks if c.status in (CheckStatus.SOFT_FAIL, CheckStatus.HARD_FAIL)]

        if failed:
            return False, f"{len(failed)} uncertainty check(s) failed"
        return True, "Uncertainty within acceptable range"

    def _rule_spatial_coherence(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Spatial coherence checks."""
        spatial_checks = [c for c in checks if c.category == CheckCategory.SPATIAL]

        if not spatial_checks:
            return True, "No spatial checks performed"

        failed = [c for c in spatial_checks if c.status in (CheckStatus.SOFT_FAIL, CheckStatus.HARD_FAIL)]
        warnings = [c for c in spatial_checks if c.status == CheckStatus.WARNING]

        if failed:
            return False, f"{len(failed)} spatial coherence issue(s)"
        if warnings:
            return True, f"{len(warnings)} spatial warning(s)"
        return True, "Spatial coherence verified"

    def _rule_value_plausibility(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Value plausibility checks."""
        value_checks = [c for c in checks if c.category == CheckCategory.VALUE]

        if not value_checks:
            return True, "No value checks performed"

        failed = [c for c in value_checks if c.status in (CheckStatus.SOFT_FAIL, CheckStatus.HARD_FAIL)]
        warnings = [c for c in value_checks if c.status == CheckStatus.WARNING]

        if failed:
            return False, f"{len(failed)} implausible value(s) detected"
        if warnings:
            return True, f"{len(warnings)} questionable value(s)"
        return True, "Values plausible"

    def _rule_warning_limit(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Total warning count limit."""
        warnings = [c for c in checks if c.status == CheckStatus.WARNING]
        threshold = context.effective_thresholds.max_warnings

        if len(warnings) > threshold:
            return False, f"{len(warnings)} warnings exceeds limit of {threshold}"
        return True, f"{len(warnings)} warning(s) within limit"

    def _rule_historical_deviation(
        self,
        context: GatingContext,
        checks: List[QCCheck],
    ) -> Tuple[bool, str]:
        """Rule: Historical baseline deviation (informational)."""
        hist_checks = [c for c in checks if c.category == CheckCategory.HISTORICAL]

        if not hist_checks:
            return True, "No historical baseline available"

        failed = [c for c in hist_checks if c.status in (CheckStatus.SOFT_FAIL, CheckStatus.HARD_FAIL)]

        if failed:
            # Informational only - always passes but logs the deviation
            return True, f"Historical deviation detected in {len(failed)} check(s)"
        return True, "Consistent with historical baseline"


# Convenience functions for common use cases

def create_emergency_gate() -> QualityGate:
    """
    Create a gate with relaxed thresholds for emergency/rapid response.

    Returns:
        QualityGate configured for emergency mode
    """
    emergency_thresholds = GatingThresholds(
        min_confidence=0.4,
        max_hard_fails=0,
        max_soft_fails=5,
        max_warnings=10,
        min_coverage_percent=60.0,
        max_uncertainty_mean=0.5,
        min_agreement_score=0.5,
    )
    return QualityGate(thresholds=emergency_thresholds)


def create_operational_gate() -> QualityGate:
    """
    Create a gate with standard operational thresholds.

    Returns:
        QualityGate configured for standard operation
    """
    return QualityGate()


def create_research_gate() -> QualityGate:
    """
    Create a gate with strict thresholds for research/archival use.

    Returns:
        QualityGate configured for research quality
    """
    research_thresholds = GatingThresholds(
        min_confidence=0.8,
        max_hard_fails=0,
        max_soft_fails=1,
        max_warnings=3,
        min_coverage_percent=90.0,
        max_uncertainty_mean=0.2,
        min_agreement_score=0.85,
    )
    return QualityGate(thresholds=research_thresholds)


def quick_gate(
    checks: List[QCCheck],
    confidence: float = 1.0,
    degraded_level: int = 0,
) -> GatingDecision:
    """
    Quick gating evaluation with default settings.

    Args:
        checks: QC checks to evaluate
        confidence: Overall confidence score
        degraded_level: Degraded mode level (0=normal)

    Returns:
        Gating decision
    """
    gate = QualityGate()
    context = GatingContext(
        event_id="",
        product_id="",
        degraded_mode_level=degraded_level,
        confidence_score=confidence,
    )
    return gate.evaluate(checks, context)
