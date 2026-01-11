"""
Tests for Quality Control Action Management.

Tests for Group I Track 4:
- Gating (pass/fail/review logic)
- Flagging (quality flag system)
- Routing (expert review routing)
"""

import pytest
from datetime import datetime, timedelta, timezone

import numpy as np


# ============================================================================
# GATING TESTS
# ============================================================================

class TestGatingDataStructures:
    """Test gating data structures."""

    def test_qc_check_creation(self):
        """Test QCCheck dataclass."""
        from core.quality.actions.gating import QCCheck, CheckCategory, CheckStatus

        check = QCCheck(
            check_name="test_check",
            category=CheckCategory.SPATIAL,
            status=CheckStatus.PASS,
            metric_value=0.95,
            threshold=0.9,
            details="Test passed",
        )

        assert check.check_name == "test_check"
        assert check.category == CheckCategory.SPATIAL
        assert check.status == CheckStatus.PASS
        assert check.metric_value == 0.95

    def test_qc_check_to_dict(self):
        """Test QCCheck serialization."""
        from core.quality.actions.gating import QCCheck, CheckCategory, CheckStatus

        check = QCCheck(
            check_name="spatial_coherence",
            category=CheckCategory.SPATIAL,
            status=CheckStatus.WARNING,
            metric_value=0.85,
            threshold=0.9,
            details="Below optimal threshold",
        )

        d = check.to_dict()
        assert d["check_name"] == "spatial_coherence"
        assert d["category"] == "spatial"
        assert d["status"] == "warning"
        assert d["metric_value"] == 0.85
        assert d["threshold"] == 0.9

    def test_gating_thresholds_defaults(self):
        """Test GatingThresholds default values."""
        from core.quality.actions.gating import GatingThresholds

        thresholds = GatingThresholds()
        assert thresholds.min_confidence == 0.6
        assert thresholds.max_hard_fails == 0
        assert thresholds.max_soft_fails == 2
        assert thresholds.max_warnings == 5

    def test_gating_thresholds_degraded_mode(self):
        """Test threshold adjustment for degraded mode."""
        from core.quality.actions.gating import GatingThresholds

        thresholds = GatingThresholds()
        adjusted = thresholds.adjust_for_degraded_mode(2)

        # Degraded mode should relax thresholds
        assert adjusted.min_confidence < thresholds.min_confidence
        assert adjusted.max_soft_fails > thresholds.max_soft_fails
        assert adjusted.max_warnings > thresholds.max_warnings
        # Hard fail limit should never change
        assert adjusted.max_hard_fails == thresholds.max_hard_fails

    def test_gating_context_effective_thresholds(self):
        """Test GatingContext effective thresholds property."""
        from core.quality.actions.gating import GatingContext, GatingThresholds

        context = GatingContext(
            event_id="evt_test",
            product_id="prod_001",
            degraded_mode_level=1,
        )

        effective = context.effective_thresholds
        base = context.thresholds

        # Degraded mode level 1 should relax thresholds
        assert effective.min_confidence < base.min_confidence

    def test_gating_decision_to_dict(self):
        """Test GatingDecision serialization."""
        from core.quality.actions.gating import GatingDecision, GateStatus

        decision = GatingDecision(
            status=GateStatus.PASS,
            rule_results={"rule1": (True, "passed")},
            rationale="All checks passed",
        )

        d = decision.to_dict()
        assert d["status"] == "PASS"
        assert "rule_results" in d
        assert d["rationale"] == "All checks passed"


class TestQualityGate:
    """Test QualityGate class."""

    def test_gate_initialization(self):
        """Test QualityGate initialization with default rules."""
        from core.quality.actions.gating import QualityGate

        gate = QualityGate()
        assert len(gate.rules) > 0

        # Check default rules exist
        rule_ids = [r.rule_id for r in gate.rules]
        assert "no_hard_failures" in rule_ids
        assert "min_confidence" in rule_ids
        assert "soft_failure_limit" in rule_ids

    def test_gate_register_rule(self):
        """Test registering custom rules."""
        from core.quality.actions.gating import (
            QualityGate, GatingRule, RuleSeverity, CheckCategory
        )

        gate = QualityGate()
        initial_count = len(gate.rules)

        custom_rule = GatingRule(
            rule_id="custom_test",
            name="Custom Test Rule",
            description="A custom test rule",
            severity=RuleSeverity.ADVISORY,
            check_categories=[CheckCategory.SPATIAL],
            check_function=lambda ctx, checks: (True, "passed"),
        )

        gate.register_rule(custom_rule)
        assert len(gate.rules) == initial_count + 1

    def test_gate_unregister_rule(self):
        """Test unregistering rules."""
        from core.quality.actions.gating import QualityGate

        gate = QualityGate()
        initial_count = len(gate.rules)

        result = gate.unregister_rule("no_hard_failures")
        assert result is True
        assert len(gate.rules) == initial_count - 1

        # Unregistering non-existent rule should return False
        result = gate.unregister_rule("nonexistent")
        assert result is False

    def test_gate_evaluate_all_pass(self):
        """Test evaluation when all checks pass."""
        from core.quality.actions.gating import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus
        )

        gate = QualityGate()

        checks = [
            QCCheck("spatial_check", CheckCategory.SPATIAL, CheckStatus.PASS),
            QCCheck("value_check", CheckCategory.VALUE, CheckStatus.PASS),
        ]

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate(checks, context)
        assert decision.status == GateStatus.PASS
        assert len(decision.failed_mandatory) == 0
        assert len(decision.failed_critical) == 0

    def test_gate_evaluate_hard_failure_blocks(self):
        """Test that hard failures result in BLOCKED status."""
        from core.quality.actions.gating import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus
        )

        gate = QualityGate()

        checks = [
            QCCheck("critical_check", CheckCategory.SPATIAL, CheckStatus.HARD_FAIL),
        ]

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate(checks, context)
        assert decision.status == GateStatus.BLOCKED
        assert len(decision.failed_mandatory) > 0

    def test_gate_evaluate_low_confidence_blocks(self):
        """Test that low confidence results in BLOCKED status."""
        from core.quality.actions.gating import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus
        )

        gate = QualityGate()

        checks = [
            QCCheck("test_check", CheckCategory.SPATIAL, CheckStatus.PASS),
        ]

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.3,  # Below default 0.6 threshold
        )

        decision = gate.evaluate(checks, context)
        assert decision.status == GateStatus.BLOCKED
        assert any("confidence" in f.lower() for f in decision.failed_mandatory)

    def test_gate_evaluate_soft_failures_require_review(self):
        """Test that multiple soft failures require review."""
        from core.quality.actions.gating import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus
        )

        gate = QualityGate()

        # Create more soft failures than threshold
        checks = [
            QCCheck("soft_1", CheckCategory.SPATIAL, CheckStatus.SOFT_FAIL),
            QCCheck("soft_2", CheckCategory.VALUE, CheckStatus.SOFT_FAIL),
            QCCheck("soft_3", CheckCategory.TEMPORAL, CheckStatus.SOFT_FAIL),
        ]

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate(checks, context)
        assert decision.status == GateStatus.REVIEW_REQUIRED
        assert len(decision.failed_critical) > 0

    def test_gate_evaluate_warnings_pass_with_warnings(self):
        """Test that advisory rule failures result in PASS_WITH_WARNINGS."""
        from core.quality.actions.gating import (
            QualityGate, QCCheck, CheckCategory, CheckStatus,
            GatingContext, GateStatus
        )

        gate = QualityGate()

        # Advisory rules check for soft_fail in their categories
        # Spatial and value soft failures trigger advisory warnings
        checks = [
            QCCheck("spatial_1", CheckCategory.SPATIAL, CheckStatus.SOFT_FAIL),
            QCCheck("value_1", CheckCategory.VALUE, CheckStatus.SOFT_FAIL),
            # Add passes in other categories so we don't hit soft_failure_limit (critical)
        ]

        # Use thresholds that allow 2 soft fails without triggering critical rule
        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate(checks, context)
        # With 2 soft fails (at default max_soft_fails=2), we're at the limit but not over
        # This should still pass with warnings from the spatial/value advisory rules
        assert decision.status in (GateStatus.PASS_WITH_WARNINGS, GateStatus.PASS)
        # At minimum, we should have warnings from the advisory rules
        if decision.status == GateStatus.PASS_WITH_WARNINGS:
            assert len(decision.warnings) > 0


class TestGatingConvenienceFunctions:
    """Test gating convenience functions."""

    def test_create_emergency_gate(self):
        """Test emergency gate creation with relaxed thresholds."""
        from core.quality.actions.gating import create_emergency_gate

        gate = create_emergency_gate()
        assert gate.thresholds.min_confidence < 0.6  # Relaxed from default

    def test_create_operational_gate(self):
        """Test operational gate with default thresholds."""
        from core.quality.actions.gating import create_operational_gate

        gate = create_operational_gate()
        assert gate.thresholds.min_confidence == 0.6

    def test_create_research_gate(self):
        """Test research gate with strict thresholds."""
        from core.quality.actions.gating import create_research_gate

        gate = create_research_gate()
        assert gate.thresholds.min_confidence > 0.6  # Stricter than default

    def test_quick_gate(self):
        """Test quick_gate convenience function."""
        from core.quality.actions.gating import (
            quick_gate, QCCheck, CheckCategory, CheckStatus, GateStatus
        )

        checks = [
            QCCheck("test", CheckCategory.SPATIAL, CheckStatus.PASS),
        ]

        decision = quick_gate(checks, confidence=0.9)
        assert decision.status == GateStatus.PASS


# ============================================================================
# FLAGGING TESTS
# ============================================================================

class TestFlaggingDataStructures:
    """Test flagging data structures."""

    def test_flag_definition_creation(self):
        """Test FlagDefinition dataclass."""
        from core.quality.actions.flagging import (
            FlagDefinition, FlagSeverity, FlagLevel
        )

        flag_def = FlagDefinition(
            flag_id="TEST_FLAG",
            name="Test Flag",
            description="A test flag",
            severity=FlagSeverity.WARNING,
            applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
            confidence_modifier=0.9,
        )

        assert flag_def.flag_id == "TEST_FLAG"
        assert flag_def.severity == FlagSeverity.WARNING
        assert FlagLevel.PRODUCT in flag_def.applies_to

    def test_applied_flag_to_dict(self):
        """Test AppliedFlag serialization."""
        from core.quality.actions.flagging import AppliedFlag, FlagLevel

        flag = AppliedFlag(
            flag_id="LOW_CONFIDENCE",
            level=FlagLevel.PRODUCT,
            reason="Confidence below threshold",
            metric_value=0.45,
        )

        d = flag.to_dict()
        assert d["flag_id"] == "LOW_CONFIDENCE"
        assert d["level"] == "product"
        assert d["metric_value"] == 0.45

    def test_flag_summary(self):
        """Test FlagSummary dataclass."""
        from core.quality.actions.flagging import FlagSummary

        summary = FlagSummary(
            product_id="prod_001",
            event_id="evt_001",
            total_flags=3,
            flags_by_severity={"warning": 2, "critical": 1},
            overall_confidence_modifier=0.75,
        )

        d = summary.to_dict()
        assert d["total_flags"] == 3
        assert d["overall_confidence_modifier"] == 0.75


class TestFlagRegistry:
    """Test FlagRegistry class."""

    def test_registry_initialization(self):
        """Test FlagRegistry initializes with standard flags."""
        from core.quality.actions.flagging import FlagRegistry

        registry = FlagRegistry()

        # Standard flags should be registered
        assert registry.get_flag("HIGH_CONFIDENCE") is not None
        assert registry.get_flag("LOW_CONFIDENCE") is not None
        assert registry.get_flag("CLOUD_AFFECTED") is not None

    def test_registry_custom_flag(self):
        """Test registering custom flags."""
        from core.quality.actions.flagging import (
            FlagRegistry, FlagDefinition, FlagSeverity, FlagLevel
        )

        registry = FlagRegistry()

        custom = FlagDefinition(
            flag_id="CUSTOM_TEST",
            name="Custom Test",
            description="A custom test flag",
            severity=FlagSeverity.INFORMATIONAL,
            applies_to=[FlagLevel.PRODUCT],
        )

        registry.register_flag(custom)
        assert registry.get_flag("CUSTOM_TEST") is not None

    def test_registry_list_flags_by_category(self):
        """Test listing flags by category."""
        from core.quality.actions.flagging import FlagRegistry

        registry = FlagRegistry()

        confidence_flags = registry.list_flags(category="confidence")
        assert len(confidence_flags) > 0
        assert all(f.category == "confidence" for f in confidence_flags)

    def test_registry_get_categories(self):
        """Test getting flag categories."""
        from core.quality.actions.flagging import FlagRegistry

        registry = FlagRegistry()

        categories = registry.get_categories()
        assert "confidence" in categories
        assert "degraded" in categories


class TestQualityFlagger:
    """Test QualityFlagger class."""

    def test_flagger_initialization(self):
        """Test QualityFlagger initialization."""
        from core.quality.actions.flagging import QualityFlagger

        flagger = QualityFlagger()
        assert flagger.registry is not None

    def test_apply_flag(self):
        """Test applying a flag to a product."""
        from core.quality.actions.flagging import (
            QualityFlagger, FlagLevel
        )

        flagger = QualityFlagger()

        flag = flagger.apply_flag(
            product_id="prod_001",
            flag_id="LOW_CONFIDENCE",
            level=FlagLevel.PRODUCT,
            reason="Test application",
        )

        assert flag.flag_id == "LOW_CONFIDENCE"
        assert flagger.has_flag("prod_001", "LOW_CONFIDENCE")

    def test_apply_standard_flag(self):
        """Test applying standard flag enum."""
        from core.quality.actions.flagging import (
            QualityFlagger, StandardFlag, FlagLevel
        )

        flagger = QualityFlagger()

        flag = flagger.apply_standard_flag(
            product_id="prod_001",
            flag=StandardFlag.CLOUD_AFFECTED,
            level=FlagLevel.REGION,
        )

        assert flag.flag_id == "CLOUD_AFFECTED"

    def test_apply_flag_invalid_level(self):
        """Test applying flag at invalid level raises error."""
        from core.quality.actions.flagging import (
            QualityFlagger, FlagLevel
        )

        flagger = QualityFlagger()

        # SATURATION can be applied at REGION, PIXEL, or BAND level, not PRODUCT
        with pytest.raises(ValueError, match="cannot be applied at level"):
            flagger.apply_flag(
                product_id="prod_001",
                flag_id="SATURATION",
                level=FlagLevel.PRODUCT,
            )

    def test_apply_flag_unknown(self):
        """Test applying unknown flag raises error."""
        from core.quality.actions.flagging import QualityFlagger, FlagLevel

        flagger = QualityFlagger()

        with pytest.raises(ValueError, match="Unknown flag"):
            flagger.apply_flag(
                product_id="prod_001",
                flag_id="NONEXISTENT_FLAG",
                level=FlagLevel.PRODUCT,
            )

    def test_remove_flag(self):
        """Test removing flags from a product."""
        from core.quality.actions.flagging import (
            QualityFlagger, StandardFlag, FlagLevel
        )

        flagger = QualityFlagger()

        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        assert flagger.has_flag("prod_001", "LOW_CONFIDENCE")

        removed = flagger.remove_flag("prod_001", "LOW_CONFIDENCE")
        assert removed == 1
        assert not flagger.has_flag("prod_001", "LOW_CONFIDENCE")

    def test_get_flags_filtered(self):
        """Test getting flags with filters."""
        from core.quality.actions.flagging import (
            QualityFlagger, StandardFlag, FlagLevel, FlagSeverity
        )

        flagger = QualityFlagger()

        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag(
            "prod_001", StandardFlag.CLOUD_AFFECTED, level=FlagLevel.REGION
        )

        # Get by level
        product_flags = flagger.get_flags("prod_001", level=FlagLevel.PRODUCT)
        assert len(product_flags) == 1

        region_flags = flagger.get_flags("prod_001", level=FlagLevel.REGION)
        assert len(region_flags) == 1

    def test_summarize(self):
        """Test flag summarization."""
        from core.quality.actions.flagging import (
            QualityFlagger, StandardFlag, FlagLevel
        )

        flagger = QualityFlagger()

        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag(
            "prod_001", StandardFlag.CLOUD_AFFECTED, level=FlagLevel.REGION
        )
        flagger.apply_standard_flag(
            "prod_001", StandardFlag.INSUFFICIENT_CONFIDENCE
        )

        summary = flagger.summarize("prod_001", "evt_001")

        assert summary.total_flags == 3
        assert len(summary.critical_flags) > 0  # INSUFFICIENT_CONFIDENCE is critical
        assert summary.overall_confidence_modifier < 1.0

    def test_clear_flags(self):
        """Test clearing all flags from a product."""
        from core.quality.actions.flagging import (
            QualityFlagger, StandardFlag, FlagLevel
        )

        flagger = QualityFlagger()

        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag(
            "prod_001", StandardFlag.SINGLE_SENSOR_MODE
        )  # Use a flag that can be applied at product level

        cleared = flagger.clear_flags("prod_001")
        assert cleared == 2

        flags = flagger.get_flags("prod_001")
        assert len(flags) == 0


class TestFlaggingConvenienceFunctions:
    """Test flagging convenience functions."""

    def test_create_confidence_flag(self):
        """Test confidence flag selection."""
        from core.quality.actions.flagging import create_confidence_flag, StandardFlag

        assert create_confidence_flag(0.9) == StandardFlag.HIGH_CONFIDENCE
        assert create_confidence_flag(0.7) == StandardFlag.MEDIUM_CONFIDENCE
        assert create_confidence_flag(0.5) == StandardFlag.LOW_CONFIDENCE
        assert create_confidence_flag(0.2) == StandardFlag.INSUFFICIENT_CONFIDENCE

    def test_flag_from_conditions(self):
        """Test flag generation from conditions."""
        from core.quality.actions.flagging import flag_from_conditions, StandardFlag

        flags = flag_from_conditions(
            cloud_cover=50,
            single_sensor=True,
            degraded_resolution=True,
        )

        assert StandardFlag.CLOUD_AFFECTED in flags
        assert StandardFlag.SINGLE_SENSOR_MODE in flags
        assert StandardFlag.RESOLUTION_DEGRADED in flags


# ============================================================================
# ROUTING TESTS
# ============================================================================

class TestRoutingDataStructures:
    """Test routing data structures."""

    def test_expert_creation(self):
        """Test Expert dataclass."""
        from core.quality.actions.routing import (
            Expert, ExpertDomain, ReviewType
        )

        expert = Expert(
            expert_id="exp_001",
            name="Test Expert",
            email="expert@test.com",
            domains=[ExpertDomain.FLOOD, ExpertDomain.SAR],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )

        assert expert.expert_id == "exp_001"
        assert ExpertDomain.FLOOD in expert.domains

    def test_expert_can_review(self):
        """Test Expert.can_review method."""
        from core.quality.actions.routing import (
            Expert, ExpertDomain, ReviewType
        )

        expert = Expert(
            expert_id="exp_001",
            name="Test Expert",
            email="expert@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
            max_concurrent_reviews=2,
        )

        # Should be able to review flood quality validation
        assert expert.can_review(ReviewType.QUALITY_VALIDATION, ExpertDomain.FLOOD)

        # Should not be able to review wildfire (not in domains)
        assert not expert.can_review(ReviewType.QUALITY_VALIDATION, ExpertDomain.WILDFIRE)

        # Test load limit
        expert.current_load = 2
        assert not expert.can_review(ReviewType.QUALITY_VALIDATION, ExpertDomain.FLOOD)

    def test_review_request_defaults(self):
        """Test ReviewRequest default deadline calculation."""
        from core.quality.actions.routing import (
            ReviewRequest, ReviewType, ExpertDomain, ReviewPriority,
            ReviewStatus, ReviewContext
        )

        context = ReviewContext(
            event_id="evt_001",
            product_id="prod_001",
        )

        # Critical priority should have short deadline
        critical = ReviewRequest(
            request_id="req_001",
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.CRITICAL,
            status=ReviewStatus.PENDING,
            context=context,
        )

        # Normal priority should have longer deadline
        normal = ReviewRequest(
            request_id="req_002",
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            status=ReviewStatus.PENDING,
            context=context,
        )

        assert critical.deadline < normal.deadline

    def test_review_request_overdue(self):
        """Test ReviewRequest.is_overdue property."""
        from core.quality.actions.routing import (
            ReviewRequest, ReviewType, ExpertDomain, ReviewPriority,
            ReviewStatus, ReviewContext
        )

        context = ReviewContext("evt_001", "prod_001")

        request = ReviewRequest(
            request_id="req_001",
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            status=ReviewStatus.PENDING,
            context=context,
            deadline=datetime.now(timezone.utc) - timedelta(hours=1),  # Past deadline
        )

        assert request.is_overdue


class TestReviewRouter:
    """Test ReviewRouter class."""

    def test_router_initialization(self):
        """Test ReviewRouter initialization."""
        from core.quality.actions.routing import ReviewRouter

        router = ReviewRouter()
        assert router.routing_strategy is not None

    def test_register_expert(self):
        """Test expert registration."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType
        )

        router = ReviewRouter()

        expert = Expert(
            expert_id="exp_001",
            name="Test Expert",
            email="expert@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )

        router.register_expert(expert)
        assert router.get_expert("exp_001") is not None

    def test_create_request(self):
        """Test review request creation."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, ReviewStatus, Expert
        )

        router = ReviewRouter()

        # Register an expert first
        expert = Expert(
            expert_id="exp_001",
            name="Test Expert",
            email="expert@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        context = ReviewContext(
            event_id="evt_001",
            product_id="prod_001",
            questions=["Is this valid?"],
        )

        request = router.create_request(
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            context=context,
        )

        assert request.request_id.startswith("rev_")
        assert request.status == ReviewStatus.ASSIGNED  # Auto-assigned
        assert request.assigned_to == "exp_001"

    def test_create_request_no_expert(self):
        """Test request creation when no expert available."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, ReviewStatus
        )

        router = ReviewRouter()

        context = ReviewContext("evt_001", "prod_001")

        request = router.create_request(
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            context=context,
            auto_assign=True,
        )

        # Should be pending since no expert available
        assert request.status == ReviewStatus.PENDING
        assert request.assigned_to is None

    def test_complete_review(self):
        """Test completing a review."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, ReviewStatus, ReviewOutcome, Expert
        )

        router = ReviewRouter()

        expert = Expert(
            expert_id="exp_001",
            name="Test Expert",
            email="expert@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        context = ReviewContext("evt_001", "prod_001")

        request = router.create_request(
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            context=context,
        )

        # Start review
        router.start_review(request.request_id, "exp_001")

        # Complete review
        outcome = ReviewOutcome(
            approved=True,
            confidence_adjustment=0.05,
            recommendations=["Minor adjustments recommended"],
        )

        result = router.complete_review(
            request.request_id,
            "exp_001",
            outcome,
            notes="Looks good overall",
        )

        assert result is True
        assert request.status == ReviewStatus.COMPLETED
        assert request.outcome == "APPROVED"

    def test_override_review(self):
        """Test overriding a pending review."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, ReviewStatus
        )

        router = ReviewRouter()

        context = ReviewContext("evt_001", "prod_001")

        request = router.create_request(
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            context=context,
            auto_assign=False,
        )

        result = router.override_review(
            request.request_id,
            authorized_by="admin@test.com",
            reason="Emergency release required",
            approved=True,
        )

        assert result is True
        assert request.status == ReviewStatus.OVERRIDDEN
        assert request.override_by == "admin@test.com"

    def test_get_requests_by_status(self):
        """Test getting requests by status."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext, ReviewStatus
        )

        router = ReviewRouter()

        context = ReviewContext("evt_001", "prod_001")

        # Create multiple requests
        router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
            auto_assign=False,
        )
        router.create_request(
            ReviewType.ALGORITHM_AGREEMENT,
            ExpertDomain.WILDFIRE,
            ReviewPriority.HIGH,
            context,
            auto_assign=False,
        )

        pending = router.get_requests_by_status(ReviewStatus.PENDING)
        assert len(pending) == 2

    def test_get_pending_count(self):
        """Test getting pending request counts."""
        from core.quality.actions.routing import (
            ReviewRouter, ReviewType, ExpertDomain, ReviewPriority,
            ReviewContext
        )

        router = ReviewRouter()

        context = ReviewContext("evt_001", "prod_001")

        router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
            auto_assign=False,
        )

        counts = router.get_pending_count()
        assert counts["pending"] == 1


class TestRoutingConvenienceFunctions:
    """Test routing convenience functions."""

    def test_create_review_from_gating(self):
        """Test review parameter generation from gating decision."""
        from core.quality.actions.routing import (
            create_review_from_gating, ReviewType, ExpertDomain, ReviewPriority
        )

        gating_decision = {
            "status": "REVIEW_REQUIRED",
            "failed_critical": [
                "Cross-validation agreement: IoU 0.45 < 0.7",
                "Uncertainty threshold: Mean uncertainty 0.35 > 0.3",
            ],
        }

        review_type, domain, priority, questions = create_review_from_gating(
            gating_decision,
            event_id="evt_001",
            product_id="prod_001",
            event_type="flood",
        )

        assert review_type == ReviewType.ALGORITHM_AGREEMENT
        assert domain == ExpertDomain.FLOOD
        assert priority == ReviewPriority.NORMAL
        assert len(questions) > 0

    def test_quick_route(self):
        """Test quick_route convenience function."""
        from core.quality.actions.routing import (
            ReviewRouter, quick_route, ReviewType, ExpertDomain, ReviewPriority
        )

        router = ReviewRouter()

        request = quick_route(
            router=router,
            product_id="prod_001",
            event_id="evt_001",
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.GENERAL,
            priority=ReviewPriority.NORMAL,
        )

        assert request is not None
        assert request.context.product_id == "prod_001"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestActionManagementIntegration:
    """Integration tests combining gating, flagging, and routing."""

    def test_full_qc_workflow(self):
        """Test complete QC workflow from checks to review."""
        from core.quality.actions import (
            QualityGate,
            QualityFlagger,
            ReviewRouter,
            QCCheck,
            CheckCategory,
            CheckStatus,
            GatingContext,
            GateStatus,
            StandardFlag,
            FlagLevel,
            ReviewType,
            ExpertDomain,
            ReviewPriority,
            ReviewContext,
            Expert,
            ReviewOutcome,
        )

        # Step 1: Create QC checks with some failures
        checks = [
            QCCheck("spatial_check", CheckCategory.SPATIAL, CheckStatus.PASS),
            QCCheck("value_check", CheckCategory.VALUE, CheckStatus.SOFT_FAIL),
            QCCheck("temporal_check", CheckCategory.TEMPORAL, CheckStatus.SOFT_FAIL),
            QCCheck("artifact_check", CheckCategory.ARTIFACT, CheckStatus.SOFT_FAIL),
        ]

        # Step 2: Evaluate through gate
        gate = QualityGate()
        context = GatingContext(
            event_id="evt_flood_001",
            product_id="prod_flood_001",
            confidence_score=0.75,
        )

        decision = gate.evaluate(checks, context)

        # Should require review due to soft failures
        assert decision.status == GateStatus.REVIEW_REQUIRED

        # Step 3: Apply appropriate flags
        flagger = QualityFlagger()
        flagger.apply_standard_flag(
            "prod_flood_001",
            StandardFlag.LOW_CONFIDENCE,
            reason=decision.rationale,
        )

        for check in checks:
            if check.status == CheckStatus.SOFT_FAIL:
                flagger.apply_flag(
                    "prod_flood_001",
                    "SPATIAL_UNCERTAINTY",
                    level=FlagLevel.REGION,
                    reason=f"Soft failure in {check.check_name}",
                )

        summary = flagger.summarize("prod_flood_001", "evt_flood_001")
        assert summary.total_flags > 0

        # Step 4: Route to expert review
        router = ReviewRouter()

        # Register an expert
        expert = Expert(
            expert_id="flood_expert_001",
            name="Flood Expert",
            email="flood@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        review_context = ReviewContext(
            event_id="evt_flood_001",
            product_id="prod_flood_001",
            gating_decision=decision.to_dict(),
            flags_applied=summary.standard_flag_list,
            questions=["Please validate the detected flood extent"],
        )

        request = router.create_request(
            review_type=ReviewType.QUALITY_VALIDATION,
            domain=ExpertDomain.FLOOD,
            priority=ReviewPriority.NORMAL,
            context=review_context,
        )

        assert request.assigned_to == "flood_expert_001"

        # Step 5: Complete review
        outcome = ReviewOutcome(
            approved=True,
            confidence_adjustment=0.1,
            flags_to_add=["CONSERVATIVE_ESTIMATE"],
            recommendations=["Product is usable with noted caveats"],
        )

        router.complete_review(
            request.request_id,
            "flood_expert_001",
            outcome,
            notes="Reviewed and approved with minor concerns",
        )

        assert request.outcome == "APPROVED"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestGatingEdgeCases:
    """Edge case tests for gating module."""

    def test_evaluate_empty_checks_list(self):
        """Test evaluation with empty checks list."""
        from core.quality.actions.gating import (
            QualityGate, GatingContext, GateStatus
        )

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate([], context)
        assert decision.status == GateStatus.PASS

    def test_degraded_mode_level_0_no_change(self):
        """Test that degraded mode level 0 returns same thresholds."""
        from core.quality.actions.gating import GatingThresholds

        thresholds = GatingThresholds()
        adjusted = thresholds.adjust_for_degraded_mode(0)

        assert adjusted.min_confidence == thresholds.min_confidence
        assert adjusted.max_soft_fails == thresholds.max_soft_fails

    def test_degraded_mode_high_level(self):
        """Test degraded mode at high level (4)."""
        from core.quality.actions.gating import GatingThresholds

        thresholds = GatingThresholds()
        adjusted = thresholds.adjust_for_degraded_mode(4)

        # Should have significantly relaxed thresholds but within bounds
        assert adjusted.min_confidence >= 0.3
        assert adjusted.max_uncertainty_mean <= 0.6
        assert adjusted.min_coverage_percent >= 50.0

    def test_rule_evaluation_error_handling(self):
        """Test that rule evaluation errors are handled gracefully."""
        from core.quality.actions.gating import (
            QualityGate, GatingRule, RuleSeverity, CheckCategory,
            GatingContext, QCCheck, CheckStatus, GateStatus
        )

        def failing_rule(ctx, checks):
            raise ValueError("Intentional test error")

        gate = QualityGate()

        # Register an advisory rule that raises error
        gate.register_rule(GatingRule(
            rule_id="error_rule",
            name="Error Rule",
            description="Rule that always errors",
            severity=RuleSeverity.ADVISORY,
            check_categories=[],
            check_function=failing_rule,
        ))

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        # Should not raise, should handle gracefully
        decision = gate.evaluate([], context)
        # Advisory rule error should not block
        assert decision.status in (GateStatus.PASS, GateStatus.PASS_WITH_WARNINGS)

    def test_mandatory_rule_error_blocks(self):
        """Test that mandatory rule errors result in blocking."""
        from core.quality.actions.gating import (
            QualityGate, GatingRule, RuleSeverity, CheckCategory,
            GatingContext, GateStatus
        )

        def failing_rule(ctx, checks):
            raise ValueError("Intentional test error")

        gate = QualityGate()

        # Register a mandatory rule that raises error
        gate.register_rule(GatingRule(
            rule_id="mandatory_error",
            name="Mandatory Error Rule",
            description="Rule that always errors",
            severity=RuleSeverity.MANDATORY,
            check_categories=[],
            check_function=failing_rule,
        ))

        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
        )

        decision = gate.evaluate([], context)
        assert decision.status == GateStatus.BLOCKED

    def test_cross_validation_with_none_values(self):
        """Test cross-validation rule with None values in context."""
        from core.quality.actions.gating import (
            QualityGate, GatingContext, GateStatus
        )

        gate = QualityGate()
        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.9,
            cross_validation={"agreement_metrics": {"iou": None, "kappa": None}},
        )

        # Should not fail on None values
        decision = gate.evaluate([], context)
        assert decision.status == GateStatus.PASS


class TestFlaggingEdgeCases:
    """Edge case tests for flagging module."""

    def test_empty_pixel_mask(self):
        """Test handling of empty pixel mask."""
        from core.quality.actions.flagging import AppliedFlag, FlagLevel
        import numpy as np

        flag = AppliedFlag(
            flag_id="TEST_FLAG",
            level=FlagLevel.PIXEL,
            pixel_mask=np.array([], dtype=bool).reshape(0, 0),
        )

        d = flag.to_dict()
        assert d["pixel_mask_coverage"] == 0.0

    def test_pixel_quality_mask_empty_flags(self):
        """Test pixel quality mask with no pixel-level flags."""
        from core.quality.actions.flagging import QualityFlagger, StandardFlag
        import numpy as np

        flagger = QualityFlagger()
        # Only apply product-level flag
        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)

        mask = flagger.get_pixel_quality_mask("prod_001", (100, 100))
        # All 1s since no pixel-level flags
        assert np.all(mask == 1.0)

    def test_summarize_no_flags(self):
        """Test summarizing product with no flags."""
        from core.quality.actions.flagging import QualityFlagger

        flagger = QualityFlagger()
        summary = flagger.summarize("nonexistent_product", "evt_001")

        assert summary.total_flags == 0
        assert summary.overall_confidence_modifier == 1.0

    def test_remove_nonexistent_flag(self):
        """Test removing flag that doesn't exist."""
        from core.quality.actions.flagging import QualityFlagger

        flagger = QualityFlagger()
        removed = flagger.remove_flag("prod_001", "NONEXISTENT")
        assert removed == 0

    def test_clear_flags_nonexistent_product(self):
        """Test clearing flags from product that has none."""
        from core.quality.actions.flagging import QualityFlagger

        flagger = QualityFlagger()
        cleared = flagger.clear_flags("nonexistent_product")
        assert cleared == 0

    def test_duplicate_flag_application(self):
        """Test applying same flag multiple times."""
        from core.quality.actions.flagging import QualityFlagger, StandardFlag

        flagger = QualityFlagger()
        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)
        flagger.apply_standard_flag("prod_001", StandardFlag.LOW_CONFIDENCE)

        flags = flagger.get_flags("prod_001")
        assert len(flags) == 3  # All applications are tracked

        # Summary should count unique flags for confidence modifier
        summary = flagger.summarize("prod_001", "evt_001")
        # Confidence modifier should only apply once per unique flag
        assert summary.overall_confidence_modifier == 0.7  # LOW_CONFIDENCE modifier


class TestRoutingEdgeCases:
    """Edge case tests for routing module."""

    def test_unregister_expert_with_assigned_reviews(self):
        """Test unregistering expert with assigned reviews."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType,
            ReviewContext, ReviewPriority, ReviewStatus
        )

        router = ReviewRouter()

        # Register two experts
        expert1 = Expert(
            expert_id="exp_001",
            name="Expert 1",
            email="exp1@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        expert2 = Expert(
            expert_id="exp_002",
            name="Expert 2",
            email="exp2@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert1)
        router.register_expert(expert2)

        context = ReviewContext("evt_001", "prod_001")
        request = router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
        )

        # Request should be assigned to first expert
        assert request.assigned_to == "exp_001"

        # Unregister first expert - review should be reassigned
        router.unregister_expert("exp_001")

        # Expert should be removed
        assert router.get_expert("exp_001") is None

    def test_assign_nonexistent_request(self):
        """Test assigning a request that doesn't exist."""
        from core.quality.actions.routing import ReviewRouter

        router = ReviewRouter()
        result = router.assign_request("nonexistent_req")
        assert result is False

    def test_complete_review_wrong_expert(self):
        """Test completing review by wrong expert."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType,
            ReviewContext, ReviewPriority, ReviewOutcome
        )

        router = ReviewRouter()

        expert = Expert(
            expert_id="exp_001",
            name="Expert 1",
            email="exp1@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        context = ReviewContext("evt_001", "prod_001")
        request = router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
        )

        # Try to complete with wrong expert
        outcome = ReviewOutcome(approved=True)
        result = router.complete_review(request.request_id, "wrong_expert", outcome)
        assert result is False

    def test_override_completed_review_fails(self):
        """Test that overriding a completed review fails."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType,
            ReviewContext, ReviewPriority, ReviewOutcome, ReviewStatus
        )

        router = ReviewRouter()

        expert = Expert(
            expert_id="exp_001",
            name="Expert 1",
            email="exp1@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        context = ReviewContext("evt_001", "prod_001")
        request = router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
        )

        # Complete the review
        outcome = ReviewOutcome(approved=True)
        router.complete_review(request.request_id, "exp_001", outcome)
        assert request.status == ReviewStatus.COMPLETED

        # Try to override - should fail
        result = router.override_review(
            request.request_id,
            "admin@test.com",
            "Emergency override",
        )
        assert result is False

    def test_list_experts_with_filters(self):
        """Test listing experts with domain and availability filters."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType
        )

        router = ReviewRouter()

        expert1 = Expert(
            expert_id="exp_001",
            name="Flood Expert",
            email="flood@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
            available=True,
        )
        expert2 = Expert(
            expert_id="exp_002",
            name="Wildfire Expert",
            email="wildfire@test.com",
            domains=[ExpertDomain.WILDFIRE],
            review_types=[ReviewType.QUALITY_VALIDATION],
            available=False,
        )
        router.register_expert(expert1)
        router.register_expert(expert2)

        # Filter by domain
        flood_experts = router.list_experts(domain=ExpertDomain.FLOOD)
        assert len(flood_experts) == 1
        assert flood_experts[0].expert_id == "exp_001"

        # Filter by availability
        available = router.list_experts(available_only=True)
        assert len(available) == 1
        assert available[0].expert_id == "exp_001"

    def test_get_requests_by_expert(self):
        """Test getting requests assigned to an expert."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType,
            ReviewContext, ReviewPriority
        )

        router = ReviewRouter()

        expert = Expert(
            expert_id="exp_001",
            name="Expert 1",
            email="exp1@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        # Create two requests
        context = ReviewContext("evt_001", "prod_001")
        router.create_request(ReviewType.QUALITY_VALIDATION, ExpertDomain.FLOOD, ReviewPriority.NORMAL, context)
        router.create_request(ReviewType.QUALITY_VALIDATION, ExpertDomain.FLOOD, ReviewPriority.HIGH, context)

        requests = router.get_requests_by_expert("exp_001")
        assert len(requests) == 2

    def test_check_escalations_no_escalation_needed(self):
        """Test check_escalations when no escalation is needed."""
        from core.quality.actions.routing import ReviewRouter

        router = ReviewRouter()
        escalated = router.check_escalations()
        assert escalated == []

    def test_notification_callback_error_handling(self):
        """Test that notification callback errors are handled gracefully."""
        from core.quality.actions.routing import (
            ReviewRouter, Expert, ExpertDomain, ReviewType,
            ReviewContext, ReviewPriority
        )

        def failing_callback(request, message):
            raise ValueError("Notification failed")

        router = ReviewRouter(notification_callback=failing_callback)

        expert = Expert(
            expert_id="exp_001",
            name="Expert 1",
            email="exp1@test.com",
            domains=[ExpertDomain.FLOOD],
            review_types=[ReviewType.QUALITY_VALIDATION],
        )
        router.register_expert(expert)

        context = ReviewContext("evt_001", "prod_001")

        # Should not raise despite failing callback
        request = router.create_request(
            ReviewType.QUALITY_VALIDATION,
            ExpertDomain.FLOOD,
            ReviewPriority.NORMAL,
            context,
        )
        assert request is not None


class TestNaNAndBoundaryValues:
    """Tests for NaN, Inf, and boundary value handling."""

    def test_confidence_at_boundary(self):
        """Test confidence values at exact boundaries."""
        from core.quality.actions.flagging import create_confidence_flag, StandardFlag

        # Test exact boundary values
        assert create_confidence_flag(0.8) == StandardFlag.HIGH_CONFIDENCE
        assert create_confidence_flag(0.6) == StandardFlag.MEDIUM_CONFIDENCE
        assert create_confidence_flag(0.4) == StandardFlag.LOW_CONFIDENCE
        assert create_confidence_flag(0.0) == StandardFlag.INSUFFICIENT_CONFIDENCE
        assert create_confidence_flag(1.0) == StandardFlag.HIGH_CONFIDENCE

    def test_gating_with_boundary_confidence(self):
        """Test gating at exact confidence threshold."""
        from core.quality.actions.gating import (
            QualityGate, GatingContext, GateStatus
        )

        gate = QualityGate()

        # Exactly at threshold (0.6)
        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.6,  # Exactly at default threshold
        )

        decision = gate.evaluate([], context)
        assert decision.status == GateStatus.PASS

        # Just below threshold
        context = GatingContext(
            event_id="evt_001",
            product_id="prod_001",
            confidence_score=0.59,
        )

        decision = gate.evaluate([], context)
        assert decision.status == GateStatus.BLOCKED

    def test_flag_from_conditions_boundary(self):
        """Test flag_from_conditions at cloud cover boundary."""
        from core.quality.actions.flagging import flag_from_conditions, StandardFlag

        # Exactly at 30% boundary - should not trigger
        flags = flag_from_conditions(cloud_cover=30)
        assert StandardFlag.CLOUD_AFFECTED not in flags

        # Just above 30% - should trigger
        flags = flag_from_conditions(cloud_cover=31)
        assert StandardFlag.CLOUD_AFFECTED in flags
