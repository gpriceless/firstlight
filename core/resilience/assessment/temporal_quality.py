"""
Temporal Quality Assessment.

Evaluates the temporal quality of data acquisitions including:
- Pre-event baseline availability
- Temporal gap assessment
- Acquisition timing relevance to event
- Overall temporal quality score (0-1)

Temporal quality is critical for change detection, event monitoring,
and establishing baseline conditions for comparison.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TemporalIssueSeverity(Enum):
    """Severity levels for temporal quality issues."""
    CRITICAL = "critical"   # Cannot perform temporal analysis
    HIGH = "high"           # Significant limitation
    MEDIUM = "medium"       # May impact analysis quality
    LOW = "low"             # Minor issue
    INFO = "info"           # Informational


class AcquisitionTiming(Enum):
    """Timing relevance of acquisition relative to event."""
    OPTIMAL = "optimal"         # Ideal timing
    GOOD = "good"               # Good timing
    ACCEPTABLE = "acceptable"   # Usable but not ideal
    POOR = "poor"               # Suboptimal timing
    UNUSABLE = "unusable"       # Too far from event


@dataclass
class TemporalQualityIssue:
    """
    A temporal quality issue found during assessment.

    Attributes:
        issue_type: Type of quality issue
        severity: Issue severity level
        description: Human-readable description
        gap_hours: Gap duration in hours (if applicable)
        recommendation: Suggested action
    """
    issue_type: str
    severity: TemporalIssueSeverity
    description: str
    gap_hours: Optional[float] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "description": self.description,
            "gap_hours": self.gap_hours,
            "recommendation": self.recommendation,
        }


@dataclass
class TemporalQualityConfig:
    """
    Configuration for temporal quality assessment.

    Attributes:
        max_baseline_gap_days: Maximum acceptable gap for baseline imagery
        max_post_event_delay_hours: Maximum delay after event for timely response
        min_baseline_images: Minimum number of baseline images
        ideal_baseline_period_days: Ideal baseline period length
        max_acquisition_gap_hours: Maximum gap between acquisitions

        event_types: Dictionary of event type-specific timing requirements
    """
    max_baseline_gap_days: int = 30
    max_post_event_delay_hours: int = 72
    min_baseline_images: int = 2
    ideal_baseline_period_days: int = 365
    max_acquisition_gap_hours: int = 168  # 1 week

    # Event-specific timing requirements (hours)
    event_timing_requirements: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "flood": {
            "optimal_post_event_hours": 24,
            "max_post_event_hours": 72,
            "baseline_importance": 0.7,
        },
        "wildfire": {
            "optimal_post_event_hours": 6,
            "max_post_event_hours": 48,
            "baseline_importance": 0.8,
        },
        "storm": {
            "optimal_post_event_hours": 12,
            "max_post_event_hours": 96,
            "baseline_importance": 0.5,
        },
    })


@dataclass
class TemporalQualityResult:
    """
    Result of temporal quality assessment.

    Attributes:
        overall_score: Overall temporal quality score (0-1)
        is_adequate: Whether temporal coverage is adequate
        has_baseline: Whether baseline imagery is available
        baseline_count: Number of baseline images
        baseline_span_days: Time span covered by baseline
        pre_event_gap_hours: Gap between last baseline and event
        post_event_delay_hours: Delay after event to first post-event image
        acquisition_timing: Timing relevance classification
        max_gap_hours: Maximum gap between consecutive acquisitions
        issues: List of quality issues
        metrics: Detailed metrics
        duration_seconds: Assessment duration
    """
    overall_score: float
    is_adequate: bool
    has_baseline: bool
    baseline_count: int
    baseline_span_days: float
    pre_event_gap_hours: float
    post_event_delay_hours: float
    acquisition_timing: AcquisitionTiming
    max_gap_hours: float
    issues: List[TemporalQualityIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def critical_count(self) -> int:
        """Count of critical issues."""
        return sum(1 for i in self.issues if i.severity == TemporalIssueSeverity.CRITICAL)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": round(self.overall_score, 3),
            "is_adequate": self.is_adequate,
            "has_baseline": self.has_baseline,
            "baseline_count": self.baseline_count,
            "baseline_span_days": round(self.baseline_span_days, 1),
            "pre_event_gap_hours": round(self.pre_event_gap_hours, 1),
            "post_event_delay_hours": round(self.post_event_delay_hours, 1),
            "acquisition_timing": self.acquisition_timing.value,
            "max_gap_hours": round(self.max_gap_hours, 1),
            "issue_count": len(self.issues),
            "critical_count": self.critical_count,
            "issues": [i.to_dict() for i in self.issues],
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


class TemporalQualityAssessor:
    """
    Assesses temporal quality of data acquisitions.

    Evaluates baseline availability, acquisition timing, and temporal
    gaps to produce an overall quality score and recommendations.

    Example:
        assessor = TemporalQualityAssessor()
        result = assessor.assess(
            acquisition_times=[datetime(2024, 1, 1), datetime(2024, 1, 5)],
            event_time=datetime(2024, 1, 3),
            event_type="flood"
        )
        if not result.has_baseline:
            print("No baseline imagery available")
    """

    def __init__(self, config: Optional[TemporalQualityConfig] = None):
        """
        Initialize the temporal quality assessor.

        Args:
            config: Configuration options
        """
        self.config = config or TemporalQualityConfig()

    def assess(
        self,
        acquisition_times: List[datetime],
        event_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
        event_duration_hours: Optional[float] = None,
        sensor_revisit_days: Optional[float] = None,
    ) -> TemporalQualityResult:
        """
        Assess temporal quality of acquisitions.

        Args:
            acquisition_times: List of acquisition timestamps
            event_time: Optional event start time
            event_type: Optional event type (flood, wildfire, storm)
            event_duration_hours: Optional event duration
            sensor_revisit_days: Optional expected sensor revisit time

        Returns:
            TemporalQualityResult with detailed assessment
        """
        import time
        start_time = time.time()

        issues = []
        metrics = {}

        # Validate input
        if not acquisition_times:
            return TemporalQualityResult(
                overall_score=0.0,
                is_adequate=False,
                has_baseline=False,
                baseline_count=0,
                baseline_span_days=0.0,
                pre_event_gap_hours=float('inf'),
                post_event_delay_hours=float('inf'),
                acquisition_timing=AcquisitionTiming.UNUSABLE,
                max_gap_hours=float('inf'),
                issues=[TemporalQualityIssue(
                    issue_type="no_acquisitions",
                    severity=TemporalIssueSeverity.CRITICAL,
                    description="No acquisition times provided",
                    recommendation="Acquire imagery for the area of interest"
                )],
                duration_seconds=time.time() - start_time
            )

        # Sort acquisition times
        sorted_times = sorted(acquisition_times)
        metrics["num_acquisitions"] = len(sorted_times)
        metrics["first_acquisition"] = sorted_times[0].isoformat()
        metrics["last_acquisition"] = sorted_times[-1].isoformat()

        # Calculate acquisition gaps
        gaps = []
        for i in range(1, len(sorted_times)):
            gap = (sorted_times[i] - sorted_times[i-1]).total_seconds() / 3600
            gaps.append(gap)

        max_gap_hours = max(gaps) if gaps else 0.0
        mean_gap_hours = np.mean(gaps) if gaps else 0.0

        metrics["max_gap_hours"] = float(max_gap_hours)
        metrics["mean_gap_hours"] = float(mean_gap_hours)

        # Check acquisition gaps
        if max_gap_hours > self.config.max_acquisition_gap_hours:
            issues.append(TemporalQualityIssue(
                issue_type="large_gap",
                severity=TemporalIssueSeverity.MEDIUM,
                description=f"Maximum acquisition gap of {max_gap_hours:.0f} hours",
                gap_hours=max_gap_hours,
                recommendation="Consider additional data sources for continuous coverage"
            ))

        # Initialize results
        has_baseline = False
        baseline_count = 0
        baseline_span_days = 0.0
        pre_event_gap_hours = 0.0
        post_event_delay_hours = 0.0
        acquisition_timing = AcquisitionTiming.GOOD

        # Event-specific analysis
        if event_time is not None:
            # Separate pre-event and post-event acquisitions
            pre_event = [t for t in sorted_times if t < event_time]
            post_event = [t for t in sorted_times if t >= event_time]

            metrics["pre_event_count"] = len(pre_event)
            metrics["post_event_count"] = len(post_event)

            # Baseline analysis
            if pre_event:
                has_baseline = True
                baseline_count = len(pre_event)

                # Calculate baseline span
                if len(pre_event) >= 2:
                    baseline_span_days = (pre_event[-1] - pre_event[0]).total_seconds() / 86400
                else:
                    baseline_span_days = 0.0

                # Pre-event gap
                pre_event_gap_hours = (event_time - pre_event[-1]).total_seconds() / 3600

                metrics["baseline_span_days"] = float(baseline_span_days)
                metrics["pre_event_gap_hours"] = float(pre_event_gap_hours)

                # Check baseline adequacy
                if baseline_count < self.config.min_baseline_images:
                    issues.append(TemporalQualityIssue(
                        issue_type="insufficient_baseline",
                        severity=TemporalIssueSeverity.MEDIUM,
                        description=f"Only {baseline_count} baseline images (need {self.config.min_baseline_images})",
                        recommendation="Acquire additional pre-event imagery"
                    ))

                if pre_event_gap_hours > self.config.max_baseline_gap_days * 24:
                    issues.append(TemporalQualityIssue(
                        issue_type="stale_baseline",
                        severity=TemporalIssueSeverity.HIGH,
                        description=f"Baseline gap of {pre_event_gap_hours/24:.0f} days before event",
                        gap_hours=pre_event_gap_hours,
                        recommendation="Use most recent available baseline or static reference"
                    ))
            else:
                issues.append(TemporalQualityIssue(
                    issue_type="no_baseline",
                    severity=TemporalIssueSeverity.HIGH,
                    description="No pre-event baseline imagery available",
                    recommendation="Use historical imagery or static baselines"
                ))

            # Post-event analysis
            if post_event:
                post_event_delay_hours = (post_event[0] - event_time).total_seconds() / 3600
                metrics["post_event_delay_hours"] = float(post_event_delay_hours)

                # Determine acquisition timing
                acquisition_timing = self._evaluate_timing(
                    post_event_delay_hours, event_type
                )

                if post_event_delay_hours > self.config.max_post_event_delay_hours:
                    issues.append(TemporalQualityIssue(
                        issue_type="late_acquisition",
                        severity=TemporalIssueSeverity.MEDIUM,
                        description=f"First post-event image {post_event_delay_hours:.0f} hours after event",
                        gap_hours=post_event_delay_hours,
                        recommendation="Prioritize rapid response for future events"
                    ))
            else:
                issues.append(TemporalQualityIssue(
                    issue_type="no_post_event",
                    severity=TemporalIssueSeverity.CRITICAL,
                    description="No post-event imagery available",
                    recommendation="Task sensor for immediate acquisition"
                ))
                acquisition_timing = AcquisitionTiming.UNUSABLE

        else:
            # No event time - assess general temporal coverage
            has_baseline = True
            baseline_count = len(sorted_times)
            baseline_span_days = (sorted_times[-1] - sorted_times[0]).total_seconds() / 86400

            metrics["baseline_span_days"] = float(baseline_span_days)

            if baseline_span_days < 30:
                issues.append(TemporalQualityIssue(
                    issue_type="short_timespan",
                    severity=TemporalIssueSeverity.LOW,
                    description=f"Acquisitions span only {baseline_span_days:.0f} days",
                    recommendation="Consider longer time period for robust analysis"
                ))

        # Check expected revisit rate
        if sensor_revisit_days is not None and gaps:
            expected_gap = sensor_revisit_days * 24
            actual_mean = mean_gap_hours

            if actual_mean > expected_gap * 2:
                issues.append(TemporalQualityIssue(
                    issue_type="below_expected_revisit",
                    severity=TemporalIssueSeverity.INFO,
                    description=f"Mean gap {actual_mean:.0f}h vs expected {expected_gap:.0f}h",
                    gap_hours=actual_mean,
                    recommendation="Check for cloud-affected or failed acquisitions"
                ))

        # Calculate overall score
        overall_score = self._calculate_overall_score(
            has_baseline,
            baseline_count,
            baseline_span_days,
            pre_event_gap_hours,
            post_event_delay_hours,
            acquisition_timing,
            max_gap_hours,
            event_type
        )

        # Determine adequacy
        is_adequate = (
            overall_score >= 0.4 and
            acquisition_timing not in [AcquisitionTiming.UNUSABLE]
        )

        duration = time.time() - start_time
        logger.info(
            f"Temporal quality assessment: score={overall_score:.2f}, "
            f"baseline={has_baseline}, timing={acquisition_timing.value}"
        )

        return TemporalQualityResult(
            overall_score=overall_score,
            is_adequate=is_adequate,
            has_baseline=has_baseline,
            baseline_count=baseline_count,
            baseline_span_days=baseline_span_days,
            pre_event_gap_hours=pre_event_gap_hours,
            post_event_delay_hours=post_event_delay_hours,
            acquisition_timing=acquisition_timing,
            max_gap_hours=max_gap_hours,
            issues=issues,
            metrics=metrics,
            duration_seconds=duration,
        )

    def _evaluate_timing(
        self,
        post_event_delay_hours: float,
        event_type: Optional[str]
    ) -> AcquisitionTiming:
        """Evaluate acquisition timing relevance."""
        # Get event-specific requirements
        if event_type and event_type in self.config.event_timing_requirements:
            requirements = self.config.event_timing_requirements[event_type]
            optimal = requirements.get("optimal_post_event_hours", 24)
            max_hours = requirements.get("max_post_event_hours", 72)
        else:
            optimal = 24
            max_hours = 72

        # Classify timing
        if post_event_delay_hours <= optimal:
            return AcquisitionTiming.OPTIMAL
        elif post_event_delay_hours <= optimal * 2:
            return AcquisitionTiming.GOOD
        elif post_event_delay_hours <= max_hours:
            return AcquisitionTiming.ACCEPTABLE
        elif post_event_delay_hours <= max_hours * 2:
            return AcquisitionTiming.POOR
        else:
            return AcquisitionTiming.UNUSABLE

    def _calculate_overall_score(
        self,
        has_baseline: bool,
        baseline_count: int,
        baseline_span_days: float,
        pre_event_gap_hours: float,
        post_event_delay_hours: float,
        acquisition_timing: AcquisitionTiming,
        max_gap_hours: float,
        event_type: Optional[str]
    ) -> float:
        """Calculate overall temporal quality score (0-1)."""
        scores = []

        # Baseline score
        if has_baseline:
            baseline_score = min(1.0, baseline_count / 5)
            if baseline_span_days > 0:
                span_score = min(1.0, baseline_span_days / self.config.ideal_baseline_period_days)
                baseline_score = (baseline_score + span_score) / 2
        else:
            baseline_score = 0.0

        scores.append(("baseline", baseline_score))

        # Pre-event gap score
        if pre_event_gap_hours > 0:
            max_gap_days = self.config.max_baseline_gap_days
            pre_event_score = max(0.0, 1.0 - pre_event_gap_hours / (max_gap_days * 24))
        else:
            pre_event_score = 1.0 if has_baseline else 0.0

        scores.append(("pre_event", pre_event_score))

        # Timing score
        timing_scores = {
            AcquisitionTiming.OPTIMAL: 1.0,
            AcquisitionTiming.GOOD: 0.85,
            AcquisitionTiming.ACCEPTABLE: 0.6,
            AcquisitionTiming.POOR: 0.3,
            AcquisitionTiming.UNUSABLE: 0.0,
        }
        timing_score = timing_scores.get(acquisition_timing, 0.5)
        scores.append(("timing", timing_score))

        # Gap regularity score
        if max_gap_hours > 0:
            gap_score = max(0.0, 1.0 - max_gap_hours / (self.config.max_acquisition_gap_hours * 2))
        else:
            gap_score = 1.0

        scores.append(("gaps", gap_score))

        # Weighted combination based on event type
        if event_type and event_type in self.config.event_timing_requirements:
            baseline_importance = self.config.event_timing_requirements[event_type].get(
                "baseline_importance", 0.5
            )
        else:
            baseline_importance = 0.5

        weights = {
            "baseline": baseline_importance * 0.4,
            "pre_event": (1 - baseline_importance) * 0.3,
            "timing": 0.4,
            "gaps": 0.2,
        }

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        overall = sum(weights[name] * score for name, score in scores)

        return float(np.clip(overall, 0.0, 1.0))

    def assess_for_change_detection(
        self,
        pre_event_times: List[datetime],
        post_event_times: List[datetime],
        minimum_pair_gap_hours: float = 24.0,
    ) -> Dict[str, Any]:
        """
        Assess temporal adequacy specifically for change detection.

        Args:
            pre_event_times: Pre-event acquisition times
            post_event_times: Post-event acquisition times
            minimum_pair_gap_hours: Minimum gap between pre/post pair

        Returns:
            Dictionary with change detection specific metrics
        """
        result = {
            "pre_event_count": len(pre_event_times),
            "post_event_count": len(post_event_times),
            "can_perform_change_detection": False,
            "best_pair": None,
            "pair_gap_hours": None,
            "issues": [],
        }

        if not pre_event_times or not post_event_times:
            result["issues"].append("Missing pre-event or post-event imagery")
            return result

        # Find best pair (closest pre/post that meet minimum gap)
        best_pair = None
        best_gap = float('inf')

        for pre in pre_event_times:
            for post in post_event_times:
                gap_hours = (post - pre).total_seconds() / 3600

                if gap_hours >= minimum_pair_gap_hours and gap_hours < best_gap:
                    best_gap = gap_hours
                    best_pair = (pre, post)

        if best_pair:
            result["can_perform_change_detection"] = True
            result["best_pair"] = {
                "pre_event": best_pair[0].isoformat(),
                "post_event": best_pair[1].isoformat(),
            }
            result["pair_gap_hours"] = best_gap

            # Check seasonal compatibility (same season is better)
            pre_month = best_pair[0].month
            post_month = best_pair[1].month
            if abs(pre_month - post_month) > 2 and abs(pre_month - post_month) < 10:
                result["issues"].append("Pre/post images from different seasons may show phenological changes")
        else:
            result["issues"].append("No valid pre/post pair found meeting gap requirements")

        return result


def assess_temporal_quality(
    acquisition_times: List[datetime],
    event_time: Optional[datetime] = None,
    event_type: Optional[str] = None,
    config: Optional[TemporalQualityConfig] = None,
) -> TemporalQualityResult:
    """
    Convenience function to assess temporal quality.

    Args:
        acquisition_times: List of acquisition timestamps
        event_time: Optional event time
        event_type: Optional event type
        config: Optional configuration

    Returns:
        TemporalQualityResult with assessment
    """
    assessor = TemporalQualityAssessor(config)
    return assessor.assess(acquisition_times, event_time, event_type)
