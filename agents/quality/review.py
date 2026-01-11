"""
Review Management for Quality Agent.

Provides the ReviewManager class for managing human review workflows:
- Queue results for human review
- Track review status
- Apply reviewer feedback
- Handle escalation logic for edge cases

This module integrates with core/quality/actions/routing.py to provide
review management capabilities for the Quality Agent.

Example:
    manager = ReviewManager()

    # Create review request
    request = await manager.create_review_request(
        event_id="evt_001",
        product_id="prod_001",
        review_type=ReviewType.QUALITY_VALIDATION,
        priority=ReviewPriority.NORMAL,
        context={"issues": issues},
    )

    # Track review
    status = manager.get_review_status(request.request_id)

    # Apply feedback
    await manager.apply_review_feedback(request.request_id, outcome)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# Import core routing module
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


logger = logging.getLogger(__name__)


class EscalationTrigger(Enum):
    """Triggers for escalating review requests."""
    DEADLINE_APPROACHING = "deadline_approaching"
    DEADLINE_PASSED = "deadline_passed"
    EXPERT_UNAVAILABLE = "expert_unavailable"
    HIGH_PRIORITY_UNASSIGNED = "high_priority_unassigned"
    REVIEWER_CONFLICT = "reviewer_conflict"
    MULTIPLE_FAILURES = "multiple_failures"


@dataclass
class EscalationRule:
    """
    Rule for escalating review requests.

    Attributes:
        rule_id: Unique rule identifier
        trigger: What triggers escalation
        condition: Condition function (request, context) -> bool
        action: Action to take on escalation
        priority_boost: How much to increase priority
        notify_targets: Who to notify
    """
    rule_id: str
    trigger: EscalationTrigger
    condition: Callable[["ActiveReview", Dict[str, Any]], bool]
    action: str = "escalate"  # escalate, reassign, notify, override
    priority_boost: int = 1
    notify_targets: List[str] = field(default_factory=list)


@dataclass
class ActiveReview:
    """
    An active review request being tracked.

    Attributes:
        request: The review request
        created_at: When request was created
        assigned_at: When request was assigned
        deadline: Review deadline
        status: Current status
        assigned_expert: Expert assigned to review
        escalation_count: Number of times escalated
        notes: Review notes
        metadata: Additional metadata
    """
    request: ReviewRequest
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_expert: Optional[str] = None
    escalation_count: int = 0
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def request_id(self) -> str:
        """Get the request ID."""
        return self.request.request_id

    @property
    def is_overdue(self) -> bool:
        """Check if review is overdue."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline

    @property
    def time_until_deadline(self) -> Optional[timedelta]:
        """Get time remaining until deadline."""
        if self.deadline is None:
            return None
        return self.deadline - datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "review_type": self.request.review_type.value if hasattr(self.request, "review_type") else "unknown",
            "priority": self.request.priority.value if hasattr(self.request, "priority") else "normal",
            "created_at": self.created_at.isoformat(),
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "status": self.status.value,
            "assigned_expert": self.assigned_expert,
            "escalation_count": self.escalation_count,
            "is_overdue": self.is_overdue,
            "notes": self.notes,
        }


@dataclass
class ReviewOutcomeResult:
    """
    Result of applying review outcome.

    Attributes:
        success: Whether outcome was applied successfully
        request_id: Review request ID
        outcome: The review outcome
        action_taken: Action taken based on outcome
        flags_updated: Quality flags updated
        product_status: New product status
        notes: Additional notes
    """
    success: bool
    request_id: str
    outcome: str
    action_taken: str
    flags_updated: List[str] = field(default_factory=list)
    product_status: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "request_id": self.request_id,
            "outcome": self.outcome,
            "action_taken": self.action_taken,
            "flags_updated": self.flags_updated,
            "product_status": self.product_status,
            "notes": self.notes,
        }


@dataclass
class ReviewManagerConfig:
    """
    Configuration for ReviewManager.

    Attributes:
        default_deadline_hours: Default deadline in hours
        escalation_enabled: Enable automatic escalation
        escalation_check_interval: Seconds between escalation checks
        max_escalations: Maximum escalations before auto-override
        routing_strategy: Strategy for assigning reviewers
        notification_enabled: Enable notifications
    """
    default_deadline_hours: float = 24.0
    escalation_enabled: bool = True
    escalation_check_interval: float = 300.0  # 5 minutes
    max_escalations: int = 3
    routing_strategy: str = "round_robin"
    notification_enabled: bool = True


class ReviewManager:
    """
    Manages human review workflows for the Quality Agent.

    Handles creation, tracking, escalation, and completion of
    review requests for quality control decisions.
    """

    def __init__(self, config: Optional[ReviewManagerConfig] = None):
        """
        Initialize ReviewManager.

        Args:
            config: Manager configuration
        """
        self.config = config or ReviewManagerConfig()
        self._logger = logging.getLogger(f"{__name__}.ReviewManager")

        # Core router
        self._router = ReviewRouter()

        # Active reviews
        self._active_reviews: Dict[str, ActiveReview] = {}
        self._completed_reviews: Dict[str, ActiveReview] = {}

        # Escalation rules
        self._escalation_rules: List[EscalationRule] = []
        self._setup_default_escalation_rules()

        # Callbacks
        self._on_review_created: List[Callable[[ActiveReview], None]] = []
        self._on_review_assigned: List[Callable[[ActiveReview], None]] = []
        self._on_review_completed: List[Callable[[ActiveReview, ReviewOutcomeResult], None]] = []
        self._on_escalation: List[Callable[[ActiveReview, EscalationTrigger], None]] = []

        # Background task handle
        self._escalation_task: Optional[asyncio.Task] = None

        self._logger.info("ReviewManager initialized")

    def _setup_default_escalation_rules(self) -> None:
        """Set up default escalation rules."""
        # Rule: Deadline approaching (< 2 hours remaining)
        self._escalation_rules.append(EscalationRule(
            rule_id="deadline_approaching",
            trigger=EscalationTrigger.DEADLINE_APPROACHING,
            condition=lambda r, ctx: (
                r.deadline is not None and
                r.status == ReviewStatus.PENDING and
                r.time_until_deadline is not None and
                r.time_until_deadline < timedelta(hours=2)
            ),
            action="notify",
            notify_targets=["supervisor"],
        ))

        # Rule: Deadline passed
        self._escalation_rules.append(EscalationRule(
            rule_id="deadline_passed",
            trigger=EscalationTrigger.DEADLINE_PASSED,
            condition=lambda r, ctx: r.is_overdue and r.status not in (
                ReviewStatus.COMPLETED, ReviewStatus.OVERRIDDEN
            ),
            action="escalate",
            priority_boost=1,
        ))

        # Rule: High priority unassigned for > 30 min
        self._escalation_rules.append(EscalationRule(
            rule_id="high_priority_unassigned",
            trigger=EscalationTrigger.HIGH_PRIORITY_UNASSIGNED,
            condition=lambda r, ctx: (
                hasattr(r.request, "priority") and
                r.request.priority in (ReviewPriority.CRITICAL, ReviewPriority.HIGH) and
                r.status == ReviewStatus.PENDING and
                r.assigned_expert is None and
                (datetime.now(timezone.utc) - r.created_at) > timedelta(minutes=30)
            ),
            action="escalate",
            priority_boost=1,
            notify_targets=["on_call"],
        ))

    async def start(self) -> None:
        """Start the review manager background tasks."""
        if self.config.escalation_enabled:
            self._escalation_task = asyncio.create_task(self._escalation_loop())
            self._logger.info("Escalation monitoring started")

    async def stop(self) -> None:
        """Stop the review manager background tasks."""
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass
            self._logger.info("Escalation monitoring stopped")

    async def create_review_request(
        self,
        event_id: str,
        product_id: str,
        review_type: ReviewType,
        priority: ReviewPriority = ReviewPriority.NORMAL,
        domain: ExpertDomain = ExpertDomain.GENERAL,
        context: Optional[Dict[str, Any]] = None,
        questions: Optional[List[str]] = None,
        deadline_hours: Optional[float] = None,
    ) -> ActiveReview:
        """
        Create a new review request.

        Args:
            event_id: Event identifier
            product_id: Product identifier
            review_type: Type of review needed
            priority: Review priority
            domain: Expert domain required
            context: Review context
            questions: Specific questions for reviewer
            deadline_hours: Custom deadline in hours

        Returns:
            ActiveReview object
        """
        context = context or {}
        questions = questions or []

        # Create review context
        review_context = ReviewContext(
            event_id=event_id,
            product_id=product_id,
            gating_decision=context.get("gating_decision"),
            quality_checks=context.get("quality_checks", []),
            flags_applied=context.get("flags_applied", []),
            questions=questions,
            visualizations=context.get("visualizations", []),
            related_products=context.get("related_products", []),
            metadata=context.get("metadata", {}),
        )

        # Create request via router
        request = self._router.create_request(
            review_type=review_type,
            domain=domain,
            priority=priority,
            context=review_context,
        )

        # Calculate deadline
        deadline_hrs = deadline_hours or self._get_default_deadline(priority)
        deadline = datetime.now(timezone.utc) + timedelta(hours=deadline_hrs)

        # Create active review
        active_review = ActiveReview(
            request=request,
            deadline=deadline,
            metadata={
                "event_id": event_id,
                "product_id": product_id,
                "review_type": review_type.value,
                "domain": domain.value,
            },
        )

        # Store
        self._active_reviews[request.request_id] = active_review

        # Try to assign
        await self._try_assign_review(active_review)

        # Notify callbacks
        for callback in self._on_review_created:
            try:
                callback(active_review)
            except Exception as e:
                self._logger.warning(f"Review created callback error: {e}")

        self._logger.info(
            f"Created review request {request.request_id}: "
            f"type={review_type.value}, priority={priority.value}"
        )

        return active_review

    async def get_review_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a review request.

        Args:
            request_id: Review request ID

        Returns:
            Status dictionary or None if not found
        """
        if request_id in self._active_reviews:
            return self._active_reviews[request_id].to_dict()
        if request_id in self._completed_reviews:
            return self._completed_reviews[request_id].to_dict()
        return None

    async def apply_review_feedback(
        self,
        request_id: str,
        outcome: str,
        reviewer_id: str,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> ReviewOutcomeResult:
        """
        Apply feedback from a completed review.

        Args:
            request_id: Review request ID
            outcome: Review outcome (approve, reject, needs_revision, override)
            reviewer_id: ID of reviewer
            feedback: Optional feedback data

        Returns:
            ReviewOutcomeResult with action taken
        """
        feedback = feedback or {}

        if request_id not in self._active_reviews:
            return ReviewOutcomeResult(
                success=False,
                request_id=request_id,
                outcome=outcome,
                action_taken="none",
                notes="Review request not found",
            )

        active_review = self._active_reviews[request_id]

        # Map outcome string to action
        action_taken = "unknown"
        flags_updated = []
        product_status = None

        if outcome == "approve":
            action_taken = "release_product"
            product_status = "approved"
            flags_updated = ["HUMAN_REVIEWED", "APPROVED"]

        elif outcome == "reject":
            action_taken = "block_product"
            product_status = "rejected"
            flags_updated = ["HUMAN_REVIEWED", "REJECTED"]

        elif outcome == "needs_revision":
            action_taken = "request_revision"
            product_status = "needs_revision"
            flags_updated = ["NEEDS_REVISION"]

        elif outcome == "override":
            action_taken = "override_gating"
            product_status = "override_approved"
            flags_updated = ["HUMAN_OVERRIDE", "APPROVED_WITH_OVERRIDE"]

        else:
            action_taken = "unknown_outcome"

        # Update review status
        active_review.status = ReviewStatus.COMPLETED
        active_review.notes.append(
            f"Completed by {reviewer_id} with outcome: {outcome}"
        )
        if feedback.get("notes"):
            active_review.notes.append(f"Reviewer notes: {feedback['notes']}")

        # Move to completed
        self._completed_reviews[request_id] = active_review
        del self._active_reviews[request_id]

        result = ReviewOutcomeResult(
            success=True,
            request_id=request_id,
            outcome=outcome,
            action_taken=action_taken,
            flags_updated=flags_updated,
            product_status=product_status,
            notes=feedback.get("notes", ""),
        )

        # Notify callbacks
        for callback in self._on_review_completed:
            try:
                callback(active_review, result)
            except Exception as e:
                self._logger.warning(f"Review completed callback error: {e}")

        self._logger.info(
            f"Review {request_id} completed: outcome={outcome}, action={action_taken}"
        )

        return result

    def register_expert(self, expert: Expert) -> None:
        """
        Register an expert reviewer.

        Args:
            expert: Expert to register
        """
        self._router.register_expert(expert)
        self._logger.info(f"Registered expert: {expert.name} ({expert.expert_id})")

    def unregister_expert(self, expert_id: str) -> None:
        """
        Unregister an expert reviewer.

        Args:
            expert_id: Expert ID to unregister
        """
        self._router.unregister_expert(expert_id)
        self._logger.info(f"Unregistered expert: {expert_id}")

    def get_active_reviews(
        self,
        status: Optional[ReviewStatus] = None,
        priority: Optional[ReviewPriority] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of active reviews with optional filtering.

        Args:
            status: Filter by status
            priority: Filter by priority

        Returns:
            List of review dictionaries
        """
        reviews = []
        for review in self._active_reviews.values():
            if status and review.status != status:
                continue
            if priority and hasattr(review.request, "priority"):
                if review.request.priority != priority:
                    continue
            reviews.append(review.to_dict())

        return reviews

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get review statistics."""
        active = list(self._active_reviews.values())
        completed = list(self._completed_reviews.values())

        pending_count = sum(1 for r in active if r.status == ReviewStatus.PENDING)
        assigned_count = sum(1 for r in active if r.status == ReviewStatus.ASSIGNED)
        in_progress_count = sum(1 for r in active if r.status == ReviewStatus.IN_PROGRESS)
        overdue_count = sum(1 for r in active if r.is_overdue)

        return {
            "total_active": len(active),
            "total_completed": len(completed),
            "pending": pending_count,
            "assigned": assigned_count,
            "in_progress": in_progress_count,
            "overdue": overdue_count,
            "escalation_count": sum(r.escalation_count for r in active),
        }

    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """Add a custom escalation rule."""
        self._escalation_rules.append(rule)
        self._logger.debug(f"Added escalation rule: {rule.rule_id}")

    # Callback registration

    def on_review_created(self, callback: Callable[[ActiveReview], None]) -> None:
        """Register callback for review creation."""
        self._on_review_created.append(callback)

    def on_review_assigned(self, callback: Callable[[ActiveReview], None]) -> None:
        """Register callback for review assignment."""
        self._on_review_assigned.append(callback)

    def on_review_completed(
        self,
        callback: Callable[[ActiveReview, ReviewOutcomeResult], None],
    ) -> None:
        """Register callback for review completion."""
        self._on_review_completed.append(callback)

    def on_escalation(
        self,
        callback: Callable[[ActiveReview, EscalationTrigger], None],
    ) -> None:
        """Register callback for escalations."""
        self._on_escalation.append(callback)

    # Internal methods

    async def _try_assign_review(self, review: ActiveReview) -> bool:
        """Try to assign a review to an expert."""
        try:
            # Use router to find best expert
            assignment = self._router.route(review.request)

            if assignment and assignment.expert_id:
                review.assigned_expert = assignment.expert_id
                review.assigned_at = datetime.now(timezone.utc)
                review.status = ReviewStatus.ASSIGNED

                # Notify callbacks
                for callback in self._on_review_assigned:
                    try:
                        callback(review)
                    except Exception as e:
                        self._logger.warning(f"Review assigned callback error: {e}")

                self._logger.info(
                    f"Assigned review {review.request_id} to {assignment.expert_id}"
                )
                return True

        except Exception as e:
            self._logger.warning(f"Failed to assign review: {e}")

        return False

    def _get_default_deadline(self, priority: ReviewPriority) -> float:
        """Get default deadline in hours based on priority."""
        deadlines = {
            ReviewPriority.CRITICAL: 1.0,
            ReviewPriority.HIGH: 4.0,
            ReviewPriority.NORMAL: self.config.default_deadline_hours,
            ReviewPriority.LOW: 72.0,
        }
        return deadlines.get(priority, self.config.default_deadline_hours)

    async def _escalation_loop(self) -> None:
        """Background loop for checking escalations."""
        while True:
            try:
                await asyncio.sleep(self.config.escalation_check_interval)
                await self._check_escalations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Escalation check error: {e}")

    async def _check_escalations(self) -> None:
        """Check all active reviews for escalation conditions."""
        context = {}  # Could include system state, expert availability, etc.

        for review in list(self._active_reviews.values()):
            for rule in self._escalation_rules:
                try:
                    if rule.condition(review, context):
                        await self._handle_escalation(review, rule)
                except Exception as e:
                    self._logger.warning(
                        f"Escalation rule {rule.rule_id} check error: {e}"
                    )

    async def _handle_escalation(
        self,
        review: ActiveReview,
        rule: EscalationRule,
    ) -> None:
        """Handle an escalation trigger."""
        review.escalation_count += 1
        review.notes.append(
            f"Escalated due to {rule.trigger.value} (count: {review.escalation_count})"
        )

        # Check max escalations
        if review.escalation_count >= self.config.max_escalations:
            self._logger.warning(
                f"Review {review.request_id} reached max escalations, "
                "auto-override may be needed"
            )
            review.notes.append("Maximum escalations reached")

        # Apply action
        if rule.action == "escalate":
            # Boost priority
            if hasattr(review.request, "priority"):
                priorities = list(ReviewPriority)
                current_idx = priorities.index(review.request.priority)
                new_idx = max(0, current_idx - rule.priority_boost)
                review.request.priority = priorities[new_idx]

        elif rule.action == "reassign":
            # Try to reassign to different expert
            review.assigned_expert = None
            review.status = ReviewStatus.PENDING
            await self._try_assign_review(review)

        elif rule.action == "notify":
            # Send notifications
            self._logger.info(
                f"Notification for review {review.request_id}: "
                f"trigger={rule.trigger.value}, targets={rule.notify_targets}"
            )

        # Notify callbacks
        for callback in self._on_escalation:
            try:
                callback(review, rule.trigger)
            except Exception as e:
                self._logger.warning(f"Escalation callback error: {e}")

        self._logger.info(
            f"Escalated review {review.request_id}: "
            f"trigger={rule.trigger.value}, action={rule.action}"
        )
