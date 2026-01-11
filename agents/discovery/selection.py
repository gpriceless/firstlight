"""
Dataset Selection Logic for Discovery Agent.

Provides:
- SelectionCriteria dataclass for defining selection requirements
- Constraint application (spatial, temporal, cloud cover)
- Scoring and ranking of candidates
- Fallback handling when primary sources unavailable
- Integration with core/data/selection/
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from core.data.discovery.base import DiscoveryResult
from core.data.providers.registry import Provider, ProviderRegistry


logger = logging.getLogger(__name__)


class SelectionOutcome(Enum):
    """Outcome of a selection attempt."""

    SELECTED = "selected"
    REJECTED_CONSTRAINT = "rejected_constraint"
    REJECTED_AVAILABILITY = "rejected_availability"
    REJECTED_QUALITY = "rejected_quality"
    FALLBACK_USED = "fallback_used"
    NO_CANDIDATES = "no_candidates"


@dataclass
class SelectionCriteria:
    """
    Criteria for dataset selection.

    Attributes:
        event_class: Event classification (e.g., "flood.coastal")
        data_types: Required data types
        constraints: Per-data-type constraints
        ranking_weights: Custom weights for ranking criteria
        min_spatial_coverage: Minimum required spatial coverage (0-100)
        max_cloud_cover: Maximum allowed cloud cover (0-100)
        max_resolution_m: Maximum allowed resolution in meters
        temporal_priority: Prefer "recent" or "closest_to_event"
        prefer_open_data: Whether to prefer open data sources
        allow_degraded: Whether to allow degraded quality data
    """

    event_class: str = ""
    data_types: Set[str] = field(default_factory=set)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ranking_weights: Dict[str, float] = field(default_factory=dict)
    min_spatial_coverage: float = 50.0
    max_cloud_cover: float = 80.0
    max_resolution_m: float = 100.0
    temporal_priority: str = "recent"
    prefer_open_data: bool = True
    allow_degraded: bool = True

    def get_default_weights(self) -> Dict[str, float]:
        """Get default ranking weights."""
        return {
            "spatial_coverage": 0.25,
            "temporal_proximity": 0.20,
            "resolution": 0.15,
            "cloud_cover": 0.15,
            "data_quality": 0.10,
            "access_cost": 0.10,
            "provider_preference": 0.05
        }

    def get_weights(self) -> Dict[str, float]:
        """Get effective ranking weights."""
        weights = self.get_default_weights()
        weights.update(self.ranking_weights)
        return weights


@dataclass
class CandidateEvaluation:
    """
    Evaluation result for a single candidate.

    Attributes:
        candidate: The discovery result being evaluated
        scores: Individual criterion scores (0-1)
        total_score: Weighted total score
        meets_hard_constraints: Whether hard constraints are met
        rejection_reasons: Reasons for rejection if any
        metadata: Additional evaluation metadata
    """

    candidate: DiscoveryResult
    scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    meets_hard_constraints: bool = True
    rejection_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "candidate_id": self.candidate.dataset_id,
            "provider": self.candidate.provider,
            "data_type": self.candidate.data_type,
            "scores": self.scores,
            "total_score": self.total_score,
            "meets_hard_constraints": self.meets_hard_constraints,
            "rejection_reasons": self.rejection_reasons,
            "metadata": self.metadata,
        }


@dataclass
class SelectionResult:
    """
    Result of dataset selection process.

    Attributes:
        selected: List of selected candidates (best first)
        rejected: List of rejected candidates with reasons
        by_data_type: Selected datasets grouped by data type
        selection_summary: Summary statistics
        fallbacks_used: Whether fallback sources were used
        degraded_mode: Whether operating in degraded mode
        trade_offs: Documented trade-off decisions
    """

    selected: List[CandidateEvaluation] = field(default_factory=list)
    rejected: List[CandidateEvaluation] = field(default_factory=list)
    by_data_type: Dict[str, List[CandidateEvaluation]] = field(default_factory=dict)
    selection_summary: Dict[str, Any] = field(default_factory=dict)
    fallbacks_used: bool = False
    degraded_mode: bool = False
    trade_offs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "selected": [e.to_dict() for e in self.selected],
            "rejected": [e.to_dict() for e in self.rejected],
            "by_data_type": {
                k: [e.to_dict() for e in v]
                for k, v in self.by_data_type.items()
            },
            "selection_summary": self.selection_summary,
            "fallbacks_used": self.fallbacks_used,
            "degraded_mode": self.degraded_mode,
            "trade_offs": self.trade_offs,
        }


class DatasetSelector:
    """
    Selects optimal datasets from discovery candidates.

    Features:
    - Hard constraint filtering (coverage, resolution, cloud)
    - Soft constraint scoring
    - Multi-criteria ranking
    - Fallback source handling
    - Trade-off documentation
    """

    def __init__(
        self,
        provider_registry: Optional[ProviderRegistry] = None
    ):
        """
        Initialize DatasetSelector.

        Args:
            provider_registry: Registry for provider metadata
        """
        self._provider_registry = provider_registry or ProviderRegistry()

    async def select(
        self,
        candidates: List[Any],  # Can be DiscoveryResult or dict
        criteria: SelectionCriteria
    ) -> SelectionResult:
        """
        Select optimal datasets from candidates.

        Args:
            candidates: List of candidate datasets
            criteria: Selection criteria

        Returns:
            SelectionResult with selected and rejected candidates
        """
        logger.info(
            f"Selecting from {len(candidates)} candidates for "
            f"data types: {criteria.data_types}"
        )

        # Convert dicts to DiscoveryResult if needed
        discovery_results = self._normalize_candidates(candidates)

        # Evaluate all candidates
        evaluations = []
        for candidate in discovery_results:
            evaluation = self._evaluate_candidate(candidate, criteria)
            evaluations.append(evaluation)

        # Separate passing and failing candidates
        passing = [e for e in evaluations if e.meets_hard_constraints]
        rejected = [e for e in evaluations if not e.meets_hard_constraints]

        # Score passing candidates
        weights = criteria.get_weights()
        for evaluation in passing:
            evaluation.total_score = self._calculate_total_score(
                evaluation.scores, weights
            )

        # Rank by total score
        passing.sort(key=lambda e: e.total_score, reverse=True)

        # Group by data type
        by_data_type: Dict[str, List[CandidateEvaluation]] = {}
        for evaluation in passing:
            data_type = evaluation.candidate.data_type
            if data_type not in by_data_type:
                by_data_type[data_type] = []
            by_data_type[data_type].append(evaluation)

        # Select best for each required data type
        selected: List[CandidateEvaluation] = []
        trade_offs: List[Dict[str, Any]] = []

        for data_type in criteria.data_types:
            type_candidates = by_data_type.get(data_type, [])

            if not type_candidates:
                logger.warning(f"No candidates available for data type: {data_type}")
                continue

            # Select best candidate
            best = type_candidates[0]
            selected.append(best)

            # Document trade-offs if alternatives exist
            if len(type_candidates) > 1:
                trade_offs.append({
                    "decision": f"Select {data_type} dataset",
                    "selected": best.candidate.dataset_id,
                    "selected_score": best.total_score,
                    "alternatives": [
                        {
                            "id": e.candidate.dataset_id,
                            "score": e.total_score
                        }
                        for e in type_candidates[1:4]  # Top 3 alternatives
                    ],
                    "rationale": self._generate_rationale(best, type_candidates[1:2])
                })

        # Build summary
        selection_summary = self._build_summary(
            candidates=discovery_results,
            selected=selected,
            rejected=rejected,
            criteria=criteria
        )

        # Check for degraded mode
        degraded_mode = self._check_degraded_mode(selected, criteria)

        return SelectionResult(
            selected=selected,
            rejected=rejected,
            by_data_type=by_data_type,
            selection_summary=selection_summary,
            fallbacks_used=False,
            degraded_mode=degraded_mode,
            trade_offs=trade_offs
        )

    async def rank_sources(
        self,
        sources: List[DiscoveryResult],
        criteria: Optional[SelectionCriteria] = None
    ) -> List[DiscoveryResult]:
        """
        Rank sources without full selection.

        Args:
            sources: Sources to rank
            criteria: Optional criteria for ranking

        Returns:
            Sources sorted by score (best first)
        """
        criteria = criteria or SelectionCriteria()
        weights = criteria.get_weights()

        # Score each source
        scored: List[Tuple[DiscoveryResult, float]] = []
        for source in sources:
            evaluation = self._evaluate_candidate(source, criteria)
            total_score = self._calculate_total_score(evaluation.scores, weights)
            scored.append((source, total_score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [source for source, _ in scored]

    def _normalize_candidates(
        self,
        candidates: List[Any]
    ) -> List[DiscoveryResult]:
        """Convert candidates to DiscoveryResult objects."""
        results = []

        for candidate in candidates:
            if isinstance(candidate, DiscoveryResult):
                results.append(candidate)
            elif isinstance(candidate, dict):
                # Convert dict to DiscoveryResult
                results.append(self._dict_to_discovery_result(candidate))
            else:
                logger.warning(f"Unknown candidate type: {type(candidate)}")

        return results

    def _dict_to_discovery_result(self, data: Dict[str, Any]) -> DiscoveryResult:
        """Convert dict to DiscoveryResult."""
        # Handle acquisition_time as string
        acq_time = data.get("acquisition_time")
        if isinstance(acq_time, str):
            acq_time = datetime.fromisoformat(acq_time.replace("Z", "+00:00"))
        elif acq_time is None:
            acq_time = datetime.now(timezone.utc)

        return DiscoveryResult(
            dataset_id=data.get("dataset_id", data.get("id", "")),
            provider=data.get("provider", ""),
            data_type=data.get("data_type", ""),
            source_uri=data.get("source_uri", data.get("uri", "")),
            format=data.get("source_format", data.get("format", "geotiff")),
            acquisition_time=acq_time,
            spatial_coverage_percent=data.get("spatial_coverage_percent", 0.0),
            resolution_m=data.get("resolution_m", 10.0),
            cloud_cover_percent=data.get("cloud_cover_percent"),
            quality_flag=data.get("quality_flag"),
            cost_tier=data.get("cost_tier"),
            metadata=data.get("metadata", {})
        )

    def _evaluate_candidate(
        self,
        candidate: DiscoveryResult,
        criteria: SelectionCriteria
    ) -> CandidateEvaluation:
        """
        Evaluate a single candidate against criteria.

        Args:
            candidate: Candidate to evaluate
            criteria: Selection criteria

        Returns:
            CandidateEvaluation with scores and constraint check
        """
        evaluation = CandidateEvaluation(candidate=candidate)

        # Check hard constraints
        self._check_hard_constraints(evaluation, criteria)

        # Calculate soft scores (even if hard constraints fail, for comparison)
        evaluation.scores = {
            "spatial_coverage": self._score_spatial_coverage(candidate, criteria),
            "temporal_proximity": self._score_temporal_proximity(candidate, criteria),
            "resolution": self._score_resolution(candidate, criteria),
            "cloud_cover": self._score_cloud_cover(candidate, criteria),
            "data_quality": self._score_data_quality(candidate),
            "access_cost": self._score_access_cost(candidate),
            "provider_preference": self._score_provider_preference(candidate)
        }

        return evaluation

    def _check_hard_constraints(
        self,
        evaluation: CandidateEvaluation,
        criteria: SelectionCriteria
    ) -> None:
        """Check if candidate meets hard constraints."""
        candidate = evaluation.candidate

        # Spatial coverage minimum
        if candidate.spatial_coverage_percent < criteria.min_spatial_coverage:
            evaluation.meets_hard_constraints = False
            evaluation.rejection_reasons.append(
                f"Insufficient spatial coverage: {candidate.spatial_coverage_percent:.1f}% "
                f"< {criteria.min_spatial_coverage:.1f}%"
            )

        # Cloud cover maximum (for optical)
        if candidate.data_type == "optical" and candidate.cloud_cover_percent is not None:
            if candidate.cloud_cover_percent > criteria.max_cloud_cover:
                evaluation.meets_hard_constraints = False
                evaluation.rejection_reasons.append(
                    f"Excessive cloud cover: {candidate.cloud_cover_percent:.1f}% "
                    f"> {criteria.max_cloud_cover:.1f}%"
                )

        # Resolution maximum
        if candidate.resolution_m > criteria.max_resolution_m:
            evaluation.meets_hard_constraints = False
            evaluation.rejection_reasons.append(
                f"Insufficient resolution: {candidate.resolution_m:.1f}m "
                f"> {criteria.max_resolution_m:.1f}m"
            )

        # Check data-type specific constraints
        type_constraints = criteria.constraints.get(candidate.data_type, {})

        # Additional cloud cover constraint from type-specific settings
        max_cloud = type_constraints.get("max_cloud_cover")
        if max_cloud is not None and candidate.cloud_cover_percent is not None:
            if candidate.cloud_cover_percent > max_cloud * 100:
                evaluation.meets_hard_constraints = False
                evaluation.rejection_reasons.append(
                    f"Cloud cover exceeds type constraint: "
                    f"{candidate.cloud_cover_percent:.1f}% > {max_cloud * 100:.1f}%"
                )

    def _score_spatial_coverage(
        self,
        candidate: DiscoveryResult,
        criteria: SelectionCriteria
    ) -> float:
        """Score based on spatial coverage (0-1)."""
        return min(candidate.spatial_coverage_percent / 100.0, 1.0)

    def _score_temporal_proximity(
        self,
        candidate: DiscoveryResult,
        criteria: SelectionCriteria
    ) -> float:
        """Score based on temporal proximity (0-1)."""
        import math

        # If no temporal constraints, assume recent is better
        reference_time = datetime.now(timezone.utc)

        # Calculate time difference in days
        time_diff_days = abs(
            (candidate.acquisition_time - reference_time).total_seconds() / 86400
        )

        # Exponential decay with 7-day half-life
        score = math.exp(-time_diff_days / 7.0)
        return score

    def _score_resolution(
        self,
        candidate: DiscoveryResult,
        criteria: SelectionCriteria
    ) -> float:
        """Score based on spatial resolution (0-1)."""
        import math

        # Better resolution = higher score
        # 10m = ~1.0, 100m = ~0.37, 1000m = ~0.0001
        score = math.exp(-candidate.resolution_m / 100.0)
        return min(score, 1.0)

    def _score_cloud_cover(
        self,
        candidate: DiscoveryResult,
        criteria: SelectionCriteria
    ) -> float:
        """Score based on cloud cover (0-1)."""
        if candidate.cloud_cover_percent is None:
            return 1.0  # N/A for non-optical

        # Lower cloud cover = higher score
        return 1.0 - (candidate.cloud_cover_percent / 100.0)

    def _score_data_quality(self, candidate: DiscoveryResult) -> float:
        """Score based on data quality (0-1)."""
        quality_scores = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.6,
            "poor": 0.3
        }
        return quality_scores.get(candidate.quality_flag or "good", 0.5)

    def _score_access_cost(self, candidate: DiscoveryResult) -> float:
        """Score based on access cost (0-1)."""
        cost_scores = {
            "open": 1.0,
            "open_restricted": 0.7,
            "commercial": 0.3
        }
        return cost_scores.get(candidate.cost_tier or "open", 0.5)

    def _score_provider_preference(self, candidate: DiscoveryResult) -> float:
        """Score based on provider preference (0-1)."""
        provider = self._provider_registry.get_provider(candidate.provider)
        if provider:
            return provider.metadata.get("preference_score", 0.5)
        return 0.5

    def _calculate_total_score(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted total score."""
        total = 0.0
        weight_sum = 0.0

        for criterion, weight in weights.items():
            if criterion in scores:
                total += scores[criterion] * weight
                weight_sum += weight

        # Normalize if weights don't sum to 1
        if weight_sum > 0 and weight_sum != 1.0:
            total /= weight_sum

        return total

    def _generate_rationale(
        self,
        selected: CandidateEvaluation,
        alternatives: List[CandidateEvaluation]
    ) -> str:
        """Generate human-readable selection rationale."""
        parts = []

        # Top scoring factors
        top_factors = sorted(
            selected.scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for factor, score in top_factors:
            if score > 0.7:
                parts.append(f"{factor.replace('_', ' ')}: {score:.2f}")

        if parts:
            return f"Top factors: {'; '.join(parts)}"

        return f"Best overall score: {selected.total_score:.3f}"

    def _build_summary(
        self,
        candidates: List[DiscoveryResult],
        selected: List[CandidateEvaluation],
        rejected: List[CandidateEvaluation],
        criteria: SelectionCriteria
    ) -> Dict[str, Any]:
        """Build selection summary statistics."""
        data_types_covered = list(set(
            e.candidate.data_type for e in selected
        ))
        data_types_missing = [
            dt for dt in criteria.data_types
            if dt not in data_types_covered
        ]

        # Rejection reasons summary
        rejection_counts: Dict[str, int] = {}
        for evaluation in rejected:
            for reason in evaluation.rejection_reasons:
                # Extract reason type
                reason_type = reason.split(":")[0]
                rejection_counts[reason_type] = rejection_counts.get(reason_type, 0) + 1

        return {
            "total_candidates": len(candidates),
            "selected_count": len(selected),
            "rejected_count": len(rejected),
            "data_types_covered": data_types_covered,
            "data_types_missing": data_types_missing,
            "rejection_breakdown": rejection_counts,
            "average_score": (
                sum(e.total_score for e in selected) / len(selected)
                if selected else 0.0
            )
        }

    def _check_degraded_mode(
        self,
        selected: List[CandidateEvaluation],
        criteria: SelectionCriteria
    ) -> bool:
        """Check if operating in degraded mode."""
        if not selected:
            return True

        # Missing required data types
        covered_types = set(e.candidate.data_type for e in selected)
        if not criteria.data_types.issubset(covered_types):
            return True

        # Low quality selections
        low_quality_count = sum(
            1 for e in selected
            if e.candidate.quality_flag in ("poor", "fair")
        )
        if low_quality_count > len(selected) / 2:
            return True

        # Low coverage
        low_coverage = any(
            e.candidate.spatial_coverage_percent < 70
            for e in selected
        )
        if low_coverage:
            return True

        return False


async def apply_fallback_sources(
    primary_result: SelectionResult,
    fallback_candidates: List[DiscoveryResult],
    criteria: SelectionCriteria
) -> SelectionResult:
    """
    Apply fallback sources when primary selection is incomplete.

    Args:
        primary_result: Result from primary selection
        fallback_candidates: Fallback candidates to consider
        criteria: Selection criteria

    Returns:
        Updated SelectionResult with fallbacks applied
    """
    selector = DatasetSelector()

    # Find missing data types
    covered_types = set(
        e.candidate.data_type for e in primary_result.selected
    )
    missing_types = criteria.data_types - covered_types

    if not missing_types:
        return primary_result

    logger.info(f"Applying fallback sources for missing types: {missing_types}")

    # Filter fallback candidates to missing types
    relevant_fallbacks = [
        c for c in fallback_candidates
        if c.data_type in missing_types
    ]

    if not relevant_fallbacks:
        return primary_result

    # Select from fallbacks with relaxed criteria
    relaxed_criteria = SelectionCriteria(
        event_class=criteria.event_class,
        data_types=missing_types,
        constraints=criteria.constraints,
        ranking_weights=criteria.ranking_weights,
        min_spatial_coverage=criteria.min_spatial_coverage * 0.5,  # Relax coverage
        max_cloud_cover=min(criteria.max_cloud_cover * 1.5, 100.0),  # Relax cloud
        max_resolution_m=criteria.max_resolution_m * 2.0,  # Relax resolution
        allow_degraded=True
    )

    fallback_result = await selector.select(relevant_fallbacks, relaxed_criteria)

    # Merge results
    combined_selected = primary_result.selected + fallback_result.selected
    combined_rejected = primary_result.rejected + fallback_result.rejected

    # Merge by_data_type
    merged_by_type = dict(primary_result.by_data_type)
    for data_type, evaluations in fallback_result.by_data_type.items():
        if data_type not in merged_by_type:
            merged_by_type[data_type] = []
        merged_by_type[data_type].extend(evaluations)

    return SelectionResult(
        selected=combined_selected,
        rejected=combined_rejected,
        by_data_type=merged_by_type,
        selection_summary={
            **primary_result.selection_summary,
            "fallback_sources_added": len(fallback_result.selected)
        },
        fallbacks_used=True,
        degraded_mode=primary_result.degraded_mode or fallback_result.degraded_mode,
        trade_offs=primary_result.trade_offs + fallback_result.trade_offs
    )
