"""
Expert Review Routing System.

Routes quality issues requiring human review to appropriate experts,
managing review workflows with deadlines and priority tracking.

Key Concepts:
- Reviews are triggered by gating decisions (REVIEW_REQUIRED)
- Routing considers issue type, urgency, and expert availability
- Review requests include context and specific questions
- Reviews have deadlines with escalation paths
- Override capability for time-critical situations
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

logger = logging.getLogger(__name__)


class ReviewPriority(Enum):
    """Priority levels for review requests."""
    CRITICAL = "critical"       # Requires immediate attention (< 1 hour)
    HIGH = "high"               # Same-day review needed
    NORMAL = "normal"           # Within 24-48 hours
    LOW = "low"                 # Non-urgent, informational


class ReviewStatus(Enum):
    """Status of a review request."""
    PENDING = "pending"         # Awaiting assignment
    ASSIGNED = "assigned"       # Assigned to reviewer
    IN_PROGRESS = "in_progress" # Reviewer actively working
    COMPLETED = "completed"     # Review finished
    OVERRIDDEN = "overridden"   # Bypassed by authorized user
    ESCALATED = "escalated"     # Escalated due to deadline
    EXPIRED = "expired"         # Deadline passed without action


class ReviewType(Enum):
    """Types of review requests."""
    QUALITY_VALIDATION = "quality_validation"      # Validate QC results
    ALGORITHM_AGREEMENT = "algorithm_agreement"    # Resolve algorithm disagreements
    UNCERTAINTY_ASSESSMENT = "uncertainty_assessment"  # Assess uncertainty levels
    ARTIFACT_VERIFICATION = "artifact_verification"    # Verify detected artifacts
    BOUNDARY_VALIDATION = "boundary_validation"    # Validate geographic boundaries
    TEMPORAL_CONSISTENCY = "temporal_consistency"  # Check temporal evolution
    SENSOR_ANOMALY = "sensor_anomaly"              # Evaluate sensor anomalies
    GENERAL = "general"                            # General quality review


class ExpertDomain(Enum):
    """Domains of expertise for routing."""
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    STORM = "storm"
    SAR = "sar"
    OPTICAL = "optical"
    FUSION = "fusion"
    GIS = "gis"
    METEOROLOGY = "meteorology"
    GENERAL = "general"


@dataclass
class Expert:
    """
    Expert reviewer profile.

    Attributes:
        expert_id: Unique identifier
        name: Display name
        email: Contact email
        domains: Areas of expertise
        review_types: Types of reviews they can handle
        max_concurrent_reviews: Maximum simultaneous reviews
        current_load: Current number of assigned reviews
        available: Whether currently accepting reviews
        priority_access: Can handle critical priority
        timezone_offset: UTC offset in hours
    """
    expert_id: str
    name: str
    email: str
    domains: List[ExpertDomain]
    review_types: List[ReviewType]
    max_concurrent_reviews: int = 5
    current_load: int = 0
    available: bool = True
    priority_access: bool = False
    timezone_offset: int = 0

    def can_review(self, review_type: ReviewType, domain: ExpertDomain) -> bool:
        """Check if expert can handle this type of review."""
        if not self.available:
            return False
        if self.current_load >= self.max_concurrent_reviews:
            return False
        if review_type not in self.review_types and ReviewType.GENERAL not in self.review_types:
            return False
        if domain not in self.domains and ExpertDomain.GENERAL not in self.domains:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "expert_id": self.expert_id,
            "name": self.name,
            "email": self.email,
            "domains": [d.value for d in self.domains],
            "review_types": [r.value for r in self.review_types],
            "max_concurrent_reviews": self.max_concurrent_reviews,
            "current_load": self.current_load,
            "available": self.available,
            "priority_access": self.priority_access,
        }


@dataclass
class ReviewContext:
    """
    Context information for a review request.

    Attributes:
        event_id: Associated event
        product_id: Product being reviewed
        gating_decision: Result from quality gate
        quality_checks: Relevant QC check results
        flags_applied: Quality flags applied
        questions: Specific questions for reviewer
        visualizations: URLs to relevant visualizations
        related_products: Related products for comparison
        metadata: Additional context
    """
    event_id: str
    product_id: str
    gating_decision: Optional[Dict[str, Any]] = None
    quality_checks: List[Dict[str, Any]] = field(default_factory=list)
    flags_applied: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)
    related_products: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_id": self.event_id,
            "product_id": self.product_id,
            "gating_decision": self.gating_decision,
            "quality_checks": self.quality_checks,
            "flags_applied": self.flags_applied,
            "questions": self.questions,
            "visualizations": self.visualizations,
            "related_products": self.related_products,
            "metadata": self.metadata,
        }


@dataclass
class ReviewRequest:
    """
    A request for expert review.

    Attributes:
        request_id: Unique identifier
        review_type: Type of review needed
        domain: Relevant expertise domain
        priority: Review priority
        status: Current status
        context: Review context
        assigned_to: Expert ID if assigned
        created_at: When request was created
        deadline: When review must be completed
        escalation_time: When to escalate if incomplete
        completed_at: When review was completed
        outcome: Review outcome/decision
        notes: Reviewer notes
        override_by: If overridden, who authorized it
        override_reason: Why it was overridden
    """
    request_id: str
    review_type: ReviewType
    domain: ExpertDomain
    priority: ReviewPriority
    status: ReviewStatus
    context: ReviewContext
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    escalation_time: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    outcome: Optional[str] = None
    notes: str = ""
    override_by: Optional[str] = None
    override_reason: Optional[str] = None

    def __post_init__(self):
        """Set default deadline based on priority if not provided."""
        if self.deadline is None:
            self.deadline = self._default_deadline()
        if self.escalation_time is None:
            self.escalation_time = self._default_escalation()

    def _default_deadline(self) -> datetime:
        """Get default deadline based on priority."""
        now = datetime.now(timezone.utc)
        if self.priority == ReviewPriority.CRITICAL:
            return now + timedelta(hours=1)
        elif self.priority == ReviewPriority.HIGH:
            return now + timedelta(hours=8)
        elif self.priority == ReviewPriority.NORMAL:
            return now + timedelta(hours=48)
        else:
            return now + timedelta(hours=168)  # 1 week

    def _default_escalation(self) -> datetime:
        """Get default escalation time (halfway to deadline)."""
        if self.deadline is None:
            self.deadline = self._default_deadline()
        elapsed = (self.deadline - self.created_at) / 2
        return self.created_at + elapsed

    @property
    def is_overdue(self) -> bool:
        """Check if review is past deadline."""
        if self.deadline is None:
            return False
        return datetime.now(timezone.utc) > self.deadline

    @property
    def should_escalate(self) -> bool:
        """Check if review should be escalated."""
        if self.escalation_time is None:
            return False
        if self.status in (ReviewStatus.COMPLETED, ReviewStatus.OVERRIDDEN):
            return False
        return datetime.now(timezone.utc) > self.escalation_time

    @property
    def time_remaining(self) -> timedelta:
        """Get time remaining until deadline."""
        if self.deadline is None:
            return timedelta(0)
        remaining = self.deadline - datetime.now(timezone.utc)
        return max(remaining, timedelta(0))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "review_type": self.review_type.value,
            "domain": self.domain.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "context": self.context.to_dict(),
            "assigned_to": self.assigned_to,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "escalation_time": self.escalation_time.isoformat() if self.escalation_time else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "outcome": self.outcome,
            "notes": self.notes,
            "is_overdue": self.is_overdue,
            "time_remaining_seconds": self.time_remaining.total_seconds(),
        }


@dataclass
class ReviewOutcome:
    """
    Outcome of a completed review.

    Attributes:
        approved: Whether the product is approved
        confidence_adjustment: Adjustment to confidence score
        flags_to_add: Quality flags to add
        flags_to_remove: Quality flags to remove
        recommendations: Recommendations for product users
        follow_up_required: If additional action needed
        follow_up_notes: Notes for follow-up
    """
    approved: bool
    confidence_adjustment: float = 0.0
    flags_to_add: List[str] = field(default_factory=list)
    flags_to_remove: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    follow_up_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "approved": self.approved,
            "confidence_adjustment": self.confidence_adjustment,
            "flags_to_add": self.flags_to_add,
            "flags_to_remove": self.flags_to_remove,
            "recommendations": self.recommendations,
            "follow_up_required": self.follow_up_required,
            "follow_up_notes": self.follow_up_notes,
        }


class RoutingStrategy:
    """
    Strategy for selecting reviewers.

    Can be subclassed for custom routing logic.
    """

    def select_reviewer(
        self,
        request: ReviewRequest,
        experts: List[Expert],
    ) -> Optional[Expert]:
        """
        Select the best reviewer for a request.

        Default implementation uses load-balancing with capability matching.

        Args:
            request: Review request
            experts: Available experts

        Returns:
            Selected expert or None if no match
        """
        # Filter to capable experts
        candidates = [
            e for e in experts
            if e.can_review(request.review_type, request.domain)
        ]

        if not candidates:
            return None

        # For critical priority, require priority access
        if request.priority == ReviewPriority.CRITICAL:
            candidates = [e for e in candidates if e.priority_access]
            if not candidates:
                # Fall back to any capable expert
                candidates = [
                    e for e in experts
                    if e.can_review(request.review_type, request.domain)
                ]

        if not candidates:
            return None

        # Sort by current load (ascending) then by domain specificity
        def score(expert: Expert) -> Tuple[int, int]:
            # Lower is better: (load, -domain_match)
            domain_match = 1 if request.domain in expert.domains else 0
            return (expert.current_load, -domain_match)

        candidates.sort(key=score)
        return candidates[0]


class ReviewRouter:
    """
    Main expert review routing system.

    Manages review requests, assignments, and workflow.
    """

    def __init__(
        self,
        routing_strategy: Optional[RoutingStrategy] = None,
        notification_callback: Optional[Callable[[ReviewRequest, str], None]] = None,
    ):
        """
        Initialize review router.

        Args:
            routing_strategy: Custom routing strategy
            notification_callback: Function to call for notifications
        """
        self.routing_strategy = routing_strategy or RoutingStrategy()
        self.notification_callback = notification_callback

        self._experts: Dict[str, Expert] = {}
        self._requests: Dict[str, ReviewRequest] = {}
        self._by_status: Dict[ReviewStatus, Set[str]] = {s: set() for s in ReviewStatus}
        self._by_assignee: Dict[str, Set[str]] = {}

    def register_expert(self, expert: Expert) -> None:
        """Register an expert reviewer."""
        self._experts[expert.expert_id] = expert
        self._by_assignee[expert.expert_id] = set()
        logger.info(f"Registered expert: {expert.name} ({expert.expert_id})")

    def unregister_expert(self, expert_id: str) -> bool:
        """Unregister an expert reviewer."""
        if expert_id not in self._experts:
            return False

        # Reassign their pending reviews
        for request_id in list(self._by_assignee.get(expert_id, [])):
            request = self._requests.get(request_id)
            if request and request.status in (ReviewStatus.ASSIGNED, ReviewStatus.IN_PROGRESS):
                self._reassign(request)

        del self._experts[expert_id]
        if expert_id in self._by_assignee:
            del self._by_assignee[expert_id]
        return True

    def create_request(
        self,
        review_type: ReviewType,
        domain: ExpertDomain,
        priority: ReviewPriority,
        context: ReviewContext,
        auto_assign: bool = True,
    ) -> ReviewRequest:
        """
        Create a new review request.

        Args:
            review_type: Type of review
            domain: Expertise domain
            priority: Review priority
            context: Review context
            auto_assign: Whether to automatically assign

        Returns:
            Created review request
        """
        request_id = f"rev_{uuid.uuid4().hex[:12]}"

        request = ReviewRequest(
            request_id=request_id,
            review_type=review_type,
            domain=domain,
            priority=priority,
            status=ReviewStatus.PENDING,
            context=context,
        )

        self._requests[request_id] = request
        self._by_status[ReviewStatus.PENDING].add(request_id)

        logger.info(
            f"Created review request {request_id}: {review_type.value} "
            f"for {context.product_id} (priority: {priority.value})"
        )

        if auto_assign:
            self.assign_request(request_id)

        return request

    def assign_request(
        self,
        request_id: str,
        expert_id: Optional[str] = None,
    ) -> bool:
        """
        Assign a request to a reviewer.

        Args:
            request_id: Request to assign
            expert_id: Specific expert (uses routing if None)

        Returns:
            True if assignment successful
        """
        request = self._requests.get(request_id)
        if not request:
            logger.error(f"Request not found: {request_id}")
            return False

        if request.status not in (ReviewStatus.PENDING, ReviewStatus.ESCALATED):
            logger.warning(f"Cannot assign request in status {request.status.value}")
            return False

        # Find reviewer
        if expert_id:
            expert = self._experts.get(expert_id)
            if not expert or not expert.can_review(request.review_type, request.domain):
                logger.warning(f"Expert {expert_id} cannot handle this request")
                return False
        else:
            expert = self.routing_strategy.select_reviewer(
                request,
                list(self._experts.values()),
            )
            if not expert:
                logger.warning(f"No suitable expert found for request {request_id}")
                return False

        # Update assignment
        old_status = request.status
        request.assigned_to = expert.expert_id
        request.status = ReviewStatus.ASSIGNED

        self._by_status[old_status].discard(request_id)
        self._by_status[ReviewStatus.ASSIGNED].add(request_id)
        self._by_assignee[expert.expert_id].add(request_id)
        expert.current_load += 1

        logger.info(f"Assigned request {request_id} to {expert.name}")

        # Send notification
        self._notify(request, f"New review assigned: {request.review_type.value}")

        return True

    def start_review(self, request_id: str, expert_id: str) -> bool:
        """
        Mark a review as in progress.

        Args:
            request_id: Request to start
            expert_id: Expert starting the review

        Returns:
            True if successful
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.assigned_to != expert_id:
            logger.warning(f"Request {request_id} not assigned to {expert_id}")
            return False

        if request.status != ReviewStatus.ASSIGNED:
            return False

        self._by_status[ReviewStatus.ASSIGNED].discard(request_id)
        request.status = ReviewStatus.IN_PROGRESS
        self._by_status[ReviewStatus.IN_PROGRESS].add(request_id)

        logger.info(f"Review {request_id} started by {expert_id}")
        return True

    def complete_review(
        self,
        request_id: str,
        expert_id: str,
        outcome: ReviewOutcome,
        notes: str = "",
    ) -> bool:
        """
        Complete a review with outcome.

        Args:
            request_id: Request to complete
            expert_id: Expert completing the review
            outcome: Review outcome
            notes: Reviewer notes

        Returns:
            True if successful
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.assigned_to != expert_id:
            logger.warning(f"Request {request_id} not assigned to {expert_id}")
            return False

        if request.status not in (ReviewStatus.ASSIGNED, ReviewStatus.IN_PROGRESS):
            return False

        old_status = request.status
        self._by_status[old_status].discard(request_id)

        request.status = ReviewStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        request.outcome = "APPROVED" if outcome.approved else "REJECTED"
        request.notes = notes

        self._by_status[ReviewStatus.COMPLETED].add(request_id)

        # Update expert load
        expert = self._experts.get(expert_id)
        if expert:
            expert.current_load = max(0, expert.current_load - 1)

        logger.info(
            f"Review {request_id} completed: {request.outcome} "
            f"(confidence adjustment: {outcome.confidence_adjustment:+.2f})"
        )

        return True

    def override_review(
        self,
        request_id: str,
        authorized_by: str,
        reason: str,
        approved: bool = True,
    ) -> bool:
        """
        Override a pending review (emergency bypass).

        Args:
            request_id: Request to override
            authorized_by: Who authorized the override
            reason: Justification for override
            approved: Whether to approve the product

        Returns:
            True if successful
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status == ReviewStatus.COMPLETED:
            return False

        old_status = request.status
        self._by_status[old_status].discard(request_id)

        request.status = ReviewStatus.OVERRIDDEN
        request.completed_at = datetime.now(timezone.utc)
        request.outcome = "OVERRIDE_APPROVED" if approved else "OVERRIDE_REJECTED"
        request.override_by = authorized_by
        request.override_reason = reason

        self._by_status[ReviewStatus.OVERRIDDEN].add(request_id)

        # Free up assigned expert
        if request.assigned_to:
            expert = self._experts.get(request.assigned_to)
            if expert:
                expert.current_load = max(0, expert.current_load - 1)
            self._by_assignee[request.assigned_to].discard(request_id)

        logger.warning(
            f"Review {request_id} overridden by {authorized_by}: {reason}"
        )

        return True

    def check_escalations(self) -> List[str]:
        """
        Check for reviews needing escalation.

        Returns:
            List of escalated request IDs
        """
        escalated = []

        # Union of pending and assigned request IDs (sets)
        to_check = self._by_status[ReviewStatus.PENDING] | \
                   self._by_status[ReviewStatus.ASSIGNED]
        for request_id in list(to_check):
            request = self._requests.get(request_id)
            if request and request.should_escalate:
                self._escalate(request)
                escalated.append(request_id)

        return escalated

    def _escalate(self, request: ReviewRequest) -> None:
        """Escalate a review request."""
        old_status = request.status
        self._by_status[old_status].discard(request.request_id)

        request.status = ReviewStatus.ESCALATED
        # Increase priority
        if request.priority == ReviewPriority.LOW:
            request.priority = ReviewPriority.NORMAL
        elif request.priority == ReviewPriority.NORMAL:
            request.priority = ReviewPriority.HIGH
        elif request.priority == ReviewPriority.HIGH:
            request.priority = ReviewPriority.CRITICAL

        self._by_status[ReviewStatus.ESCALATED].add(request.request_id)

        logger.warning(
            f"Review {request.request_id} escalated to {request.priority.value}"
        )

        self._notify(request, f"Review ESCALATED to {request.priority.value}")

        # Try to reassign with higher priority
        self.assign_request(request.request_id)

    def _reassign(self, request: ReviewRequest) -> None:
        """Reassign a review to a different expert."""
        if request.assigned_to:
            old_expert = self._experts.get(request.assigned_to)
            if old_expert:
                old_expert.current_load = max(0, old_expert.current_load - 1)
            self._by_assignee[request.assigned_to].discard(request.request_id)
            request.assigned_to = None

        old_status = request.status
        self._by_status[old_status].discard(request.request_id)
        request.status = ReviewStatus.PENDING
        self._by_status[ReviewStatus.PENDING].add(request.request_id)

        self.assign_request(request.request_id)

    def _notify(self, request: ReviewRequest, message: str) -> None:
        """Send notification for a review request."""
        if self.notification_callback:
            try:
                self.notification_callback(request, message)
            except Exception as e:
                logger.error(f"Notification failed: {e}")

    def get_request(self, request_id: str) -> Optional[ReviewRequest]:
        """Get a review request by ID."""
        return self._requests.get(request_id)

    def get_requests_by_status(self, status: ReviewStatus) -> List[ReviewRequest]:
        """Get all requests with a given status."""
        return [
            self._requests[rid]
            for rid in self._by_status.get(status, set())
            if rid in self._requests
        ]

    def get_requests_by_expert(self, expert_id: str) -> List[ReviewRequest]:
        """Get all requests assigned to an expert."""
        return [
            self._requests[rid]
            for rid in self._by_assignee.get(expert_id, set())
            if rid in self._requests
        ]

    def get_pending_count(self) -> Dict[str, int]:
        """Get counts of pending/active reviews."""
        return {
            "pending": len(self._by_status[ReviewStatus.PENDING]),
            "assigned": len(self._by_status[ReviewStatus.ASSIGNED]),
            "in_progress": len(self._by_status[ReviewStatus.IN_PROGRESS]),
            "escalated": len(self._by_status[ReviewStatus.ESCALATED]),
            "completed": len(self._by_status[ReviewStatus.COMPLETED]),
            "overridden": len(self._by_status[ReviewStatus.OVERRIDDEN]),
        }

    def get_expert(self, expert_id: str) -> Optional[Expert]:
        """Get expert by ID."""
        return self._experts.get(expert_id)

    def list_experts(
        self,
        domain: Optional[ExpertDomain] = None,
        available_only: bool = False,
    ) -> List[Expert]:
        """List registered experts."""
        experts = list(self._experts.values())

        if domain:
            experts = [e for e in experts if domain in e.domains]
        if available_only:
            experts = [e for e in experts if e.available]

        return experts


# Convenience functions

def create_review_from_gating(
    gating_decision: Dict[str, Any],
    event_id: str,
    product_id: str,
    event_type: str = "general",
) -> Tuple[ReviewType, ExpertDomain, ReviewPriority, List[str]]:
    """
    Determine review parameters from a gating decision.

    Args:
        gating_decision: Gating decision dictionary
        event_id: Event ID
        product_id: Product ID
        event_type: Type of event (flood, wildfire, storm)

    Returns:
        (review_type, domain, priority, questions) tuple
    """
    # Map event type to domain
    domain_map = {
        "flood": ExpertDomain.FLOOD,
        "wildfire": ExpertDomain.WILDFIRE,
        "storm": ExpertDomain.STORM,
    }
    domain = domain_map.get(event_type.lower(), ExpertDomain.GENERAL)

    # Determine review type from failed rules
    failed_critical = gating_decision.get("failed_critical", [])
    review_type = ReviewType.QUALITY_VALIDATION

    for failure in failed_critical:
        failure_lower = failure.lower()
        if "cross" in failure_lower or "agreement" in failure_lower:
            review_type = ReviewType.ALGORITHM_AGREEMENT
            break
        elif "uncertainty" in failure_lower:
            review_type = ReviewType.UNCERTAINTY_ASSESSMENT
            break
        elif "artifact" in failure_lower:
            review_type = ReviewType.ARTIFACT_VERIFICATION
            break
        elif "spatial" in failure_lower or "boundary" in failure_lower:
            review_type = ReviewType.BOUNDARY_VALIDATION
            break
        elif "temporal" in failure_lower:
            review_type = ReviewType.TEMPORAL_CONSISTENCY
            break

    # Determine priority from gating status
    status = gating_decision.get("status", "REVIEW_REQUIRED")
    if status == "BLOCKED":
        priority = ReviewPriority.HIGH
    elif len(failed_critical) > 2:
        priority = ReviewPriority.HIGH
    else:
        priority = ReviewPriority.NORMAL

    # Generate questions
    questions = []
    for failure in failed_critical[:3]:
        questions.append(f"Please evaluate: {failure}")
    if not questions:
        questions.append("Please validate the overall quality of this product")

    return review_type, domain, priority, questions


def quick_route(
    router: ReviewRouter,
    product_id: str,
    event_id: str,
    review_type: ReviewType,
    domain: ExpertDomain = ExpertDomain.GENERAL,
    priority: ReviewPriority = ReviewPriority.NORMAL,
    questions: Optional[List[str]] = None,
) -> ReviewRequest:
    """
    Quick route a review request.

    Args:
        router: Review router instance
        product_id: Product to review
        event_id: Associated event
        review_type: Type of review
        domain: Expertise domain
        priority: Review priority
        questions: Specific questions

    Returns:
        Created review request
    """
    context = ReviewContext(
        event_id=event_id,
        product_id=product_id,
        questions=questions or ["Please review product quality"],
    )

    return router.create_request(
        review_type=review_type,
        domain=domain,
        priority=priority,
        context=context,
        auto_assign=True,
    )
