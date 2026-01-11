"""
Deterministic algorithm selection.

Provides reproducible algorithm selection with version pinning,
selection hash generation, and comprehensive trade-off documentation.

This module ensures that given the same inputs, the same algorithms
will always be selected, enabling reproducibility of analysis results.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import logging

from core.analysis.library.registry import (
    AlgorithmMetadata,
    AlgorithmRegistry,
    AlgorithmCategory,
    DataType,
    get_global_registry,
    load_default_algorithms,
)
from core.data.discovery.base import DiscoveryResult
from core.data.evaluation.ranking import RankedCandidate, TradeOffRecord

logger = logging.getLogger(__name__)


# Version of the deterministic selection algorithm
SELECTOR_VERSION = "1.0.0"


class SelectionQuality(Enum):
    """Quality level of algorithm selection."""
    OPTIMAL = "optimal"          # All required data available, best algorithm selected
    ACCEPTABLE = "acceptable"    # Good selection with some trade-offs
    DEGRADED = "degraded"        # Fallback algorithm due to data limitations
    UNAVAILABLE = "unavailable"  # Cannot select any suitable algorithm


class SelectionReason(Enum):
    """Reasons for algorithm selection or rejection."""
    BEST_MATCH = "best_match"
    DATA_AVAILABLE = "data_available"
    EVENT_TYPE_MATCH = "event_type_match"
    RESOURCE_CONSTRAINED = "resource_constrained"
    MISSING_REQUIRED_DATA = "missing_required_data"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    EVENT_TYPE_MISMATCH = "event_type_mismatch"
    DEPRECATED = "deprecated"
    GPU_REQUIRED = "gpu_required"
    MEMORY_EXCEEDED = "memory_exceeded"


@dataclass
class DataAvailability:
    """
    Represents available data for algorithm selection.

    Tracks what data types, sensors, and datasets are available
    for the current analysis request.
    """
    data_types: Set[DataType] = field(default_factory=set)
    sensor_types: Set[str] = field(default_factory=set)
    datasets_by_type: Dict[str, List[DiscoveryResult]] = field(default_factory=dict)
    ranked_by_type: Dict[str, List[RankedCandidate]] = field(default_factory=dict)

    @classmethod
    def from_discovery_results(
        cls,
        results: List[DiscoveryResult],
        ranked: Optional[Dict[str, List[RankedCandidate]]] = None
    ) -> 'DataAvailability':
        """
        Create DataAvailability from discovery results.

        Args:
            results: List of discovered datasets
            ranked: Optional ranked candidates by type

        Returns:
            DataAvailability instance
        """
        data_types: Set[DataType] = set()
        sensor_types: Set[str] = set()
        datasets_by_type: Dict[str, List[DiscoveryResult]] = {}

        for result in results:
            # Convert string to DataType enum
            try:
                dt = DataType(result.data_type)
                data_types.add(dt)
            except ValueError:
                pass

            sensor_types.add(result.data_type)

            if result.data_type not in datasets_by_type:
                datasets_by_type[result.data_type] = []
            datasets_by_type[result.data_type].append(result)

        return cls(
            data_types=data_types,
            sensor_types=sensor_types,
            datasets_by_type=datasets_by_type,
            ranked_by_type=ranked or {}
        )

    def has_data_type(self, data_type: DataType) -> bool:
        """Check if a data type is available."""
        return data_type in self.data_types

    def has_all_required(self, required: List[DataType]) -> bool:
        """Check if all required data types are available."""
        return all(self.has_data_type(dt) for dt in required)

    def get_missing(self, required: List[DataType]) -> List[DataType]:
        """Get list of missing required data types."""
        return [dt for dt in required if not self.has_data_type(dt)]


@dataclass
class AlgorithmSelection:
    """
    Selected algorithm with full provenance information.

    Provides reproducible selection with:
    - Deterministic hash for verification
    - Version pinning for consistency
    - Full rationale and trade-off documentation
    - Rejection tracking for alternatives
    """

    algorithm_id: str
    algorithm_name: str
    version: str
    quality: SelectionQuality
    required_data_types: List[DataType]
    available_data_types: List[DataType]
    degraded_mode: bool = False
    rationale: str = ""
    selection_reason: SelectionReason = SelectionReason.BEST_MATCH
    alternatives_rejected: Dict[str, str] = field(default_factory=dict)
    execution_priority: int = 0
    deterministic_hash: str = ""
    version_lock: Dict[str, str] = field(default_factory=dict)
    selector_version: str = SELECTOR_VERSION
    selected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Generate deterministic hash if not provided."""
        if not self.deterministic_hash:
            self.deterministic_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Compute deterministic hash of selection.

        The hash ensures that identical selection criteria
        produce identical selection results.
        """
        # Build hashable content from selection parameters
        content = {
            "algorithm_id": self.algorithm_id,
            "version": self.version,
            "required_data": sorted([dt.value for dt in self.required_data_types]),
            "available_data": sorted([dt.value for dt in self.available_data_types]),
            "degraded_mode": self.degraded_mode,
            "selector_version": self.selector_version,
        }

        # Create deterministic JSON string
        json_str = json.dumps(content, sort_keys=True, separators=(',', ':'))

        # Return truncated SHA-256 hash
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def verify_hash(self) -> bool:
        """
        Verify that the stored hash matches computed hash.

        Returns:
            True if hash is valid, False if tampered
        """
        return self.deterministic_hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "algorithm_id": self.algorithm_id,
            "algorithm_name": self.algorithm_name,
            "version": self.version,
            "quality": self.quality.value,
            "required_data_types": [dt.value for dt in self.required_data_types],
            "available_data_types": [dt.value for dt in self.available_data_types],
            "degraded_mode": self.degraded_mode,
            "rationale": self.rationale,
            "selection_reason": self.selection_reason.value,
            "alternatives_rejected": self.alternatives_rejected,
            "execution_priority": self.execution_priority,
            "deterministic_hash": self.deterministic_hash,
            "version_lock": self.version_lock,
            "selector_version": self.selector_version,
            "selected_at": self.selected_at.isoformat(),
        }


@dataclass
class SelectionPlan:
    """
    Complete plan of algorithm selections for an event.

    Contains all selected algorithms, their execution order,
    and comprehensive provenance for the entire selection process.
    """

    event_class: str
    selections: Dict[str, AlgorithmSelection] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    trade_offs: List[TradeOffRecord] = field(default_factory=list)
    overall_quality: SelectionQuality = SelectionQuality.OPTIMAL
    degraded_mode: bool = False
    plan_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Compute plan hash after initialization."""
        if not self.plan_hash and self.selections:
            self.plan_hash = self._compute_plan_hash()

    def _compute_plan_hash(self) -> str:
        """Compute hash of entire selection plan."""
        content = {
            "event_class": self.event_class,
            "selections": {
                algo_id: sel.deterministic_hash
                for algo_id, sel in sorted(self.selections.items())
            },
            "execution_order": self.execution_order,
        }

        json_str = json.dumps(content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode()).hexdigest()[:24]

    def add_selection(self, selection: AlgorithmSelection) -> None:
        """Add an algorithm selection to the plan."""
        self.selections[selection.algorithm_id] = selection

        if selection.algorithm_id not in self.execution_order:
            self.execution_order.append(selection.algorithm_id)

        # Update overall quality
        if selection.quality == SelectionQuality.UNAVAILABLE:
            self.overall_quality = SelectionQuality.UNAVAILABLE
        elif selection.quality == SelectionQuality.DEGRADED:
            if self.overall_quality != SelectionQuality.UNAVAILABLE:
                self.overall_quality = SelectionQuality.DEGRADED
                self.degraded_mode = True
        elif selection.quality == SelectionQuality.ACCEPTABLE:
            if self.overall_quality == SelectionQuality.OPTIMAL:
                self.overall_quality = SelectionQuality.ACCEPTABLE

        # Recompute plan hash
        self.plan_hash = self._compute_plan_hash()

    def get_execution_sequence(self) -> List[AlgorithmSelection]:
        """Get selections in execution order."""
        return [
            self.selections[algo_id]
            for algo_id in self.execution_order
            if algo_id in self.selections
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "event_class": self.event_class,
            "selections": {
                algo_id: sel.to_dict()
                for algo_id, sel in self.selections.items()
            },
            "execution_order": self.execution_order,
            "trade_offs": [t.to_dict() for t in self.trade_offs],
            "overall_quality": self.overall_quality.value,
            "degraded_mode": self.degraded_mode,
            "plan_hash": self.plan_hash,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SelectionConstraints:
    """
    Constraints for algorithm selection.

    Defines resource limits, preferences, and requirements
    that filter which algorithms can be selected.
    """

    max_memory_mb: Optional[int] = None
    gpu_available: bool = False
    require_deterministic: bool = True
    allowed_categories: Optional[Set[AlgorithmCategory]] = None
    excluded_algorithms: Set[str] = field(default_factory=set)
    prefer_baseline: bool = True
    min_validation_score: Optional[float] = None

    def allows_algorithm(self, algorithm: AlgorithmMetadata) -> Tuple[bool, Optional[str]]:
        """
        Check if constraints allow an algorithm.

        Args:
            algorithm: Algorithm to check

        Returns:
            Tuple of (allowed, rejection_reason)
        """
        # Check deprecated
        if algorithm.deprecated:
            return False, SelectionReason.DEPRECATED.value

        # Check excluded list
        if algorithm.id in self.excluded_algorithms:
            return False, "explicitly_excluded"

        # Check category
        if self.allowed_categories and algorithm.category not in self.allowed_categories:
            return False, "category_not_allowed"

        # Check memory
        if self.max_memory_mb and algorithm.resources.memory_mb:
            if algorithm.resources.memory_mb > self.max_memory_mb:
                return False, SelectionReason.MEMORY_EXCEEDED.value

        # Check GPU requirement
        if algorithm.resources.gpu_required and not self.gpu_available:
            return False, SelectionReason.GPU_REQUIRED.value

        # Check determinism
        if self.require_deterministic and not algorithm.deterministic:
            return False, "non_deterministic"

        # Check validation score
        if self.min_validation_score and algorithm.validation:
            if algorithm.validation.accuracy_median:
                if algorithm.validation.accuracy_median < self.min_validation_score:
                    return False, "validation_score_too_low"

        return True, None


class DeterministicSelector:
    """
    Reproducible algorithm selection engine.

    Provides deterministic algorithm selection with:
    - Event type matching
    - Data availability checking
    - Resource constraint evaluation
    - Version pinning
    - Selection hash generation
    - Trade-off documentation
    """

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        constraints: Optional[SelectionConstraints] = None
    ):
        """
        Initialize deterministic selector.

        Args:
            registry: Algorithm registry (uses global if not provided)
            constraints: Selection constraints (uses defaults if not provided)
        """
        self.registry = registry or get_global_registry()
        self.constraints = constraints or SelectionConstraints()

        # Ensure registry is populated
        if not self.registry.algorithms:
            load_default_algorithms()

        # Algorithm priority order (lower is higher priority)
        self._category_priority = {
            AlgorithmCategory.BASELINE: 1,
            AlgorithmCategory.ADVANCED: 2,
            AlgorithmCategory.EXPERIMENTAL: 3,
        }

    def select_algorithms(
        self,
        event_class: str,
        data_availability: DataAvailability,
        constraints: Optional[SelectionConstraints] = None
    ) -> SelectionPlan:
        """
        Select algorithms for an event class.

        This is the main entry point for deterministic algorithm selection.
        Given an event class and available data, it selects the best
        algorithms in a reproducible manner.

        Args:
            event_class: Event classification (e.g., "flood.coastal")
            data_availability: Available data for analysis
            constraints: Optional override constraints

        Returns:
            SelectionPlan with all selected algorithms
        """
        constraints = constraints or self.constraints

        logger.info(
            f"Selecting algorithms for {event_class} with data types: "
            f"{[dt.value for dt in data_availability.data_types]}"
        )

        plan = SelectionPlan(event_class=event_class)

        # Find all candidate algorithms for this event type
        candidates = self.registry.search_by_event_type(event_class)

        if not candidates:
            logger.warning(f"No algorithms found for event type: {event_class}")
            return plan

        # Sort candidates by priority (category, then name for determinism)
        candidates = self._sort_candidates(candidates)

        # Track selected algorithms and their outputs
        selected_outputs: Set[str] = set()

        # Select algorithms
        for candidate in candidates:
            selection = self._evaluate_candidate(
                candidate,
                data_availability,
                constraints,
                selected_outputs
            )

            if selection and selection.quality != SelectionQuality.UNAVAILABLE:
                plan.add_selection(selection)

                # Track outputs for dependency resolution
                selected_outputs.update(candidate.outputs)

                logger.info(
                    f"Selected {selection.algorithm_id} v{selection.version} "
                    f"(quality: {selection.quality.value})"
                )

        # Determine execution order based on dependencies and sorting
        plan.execution_order = self._determine_execution_order(plan.selections, candidates)

        # Finalize plan hash after execution order is set
        plan.plan_hash = plan._compute_plan_hash()

        # Document trade-offs
        plan.trade_offs = self._document_trade_offs(
            candidates, plan.selections, event_class
        )

        logger.info(
            f"Selection plan complete: {len(plan.selections)} algorithms, "
            f"quality={plan.overall_quality.value}, hash={plan.plan_hash}"
        )

        return plan

    def select_single_algorithm(
        self,
        algorithm_id: str,
        data_availability: DataAvailability,
        constraints: Optional[SelectionConstraints] = None
    ) -> Optional[AlgorithmSelection]:
        """
        Select a specific algorithm by ID.

        Args:
            algorithm_id: Algorithm identifier
            data_availability: Available data
            constraints: Optional override constraints

        Returns:
            AlgorithmSelection if algorithm can be used, None otherwise
        """
        constraints = constraints or self.constraints

        algorithm = self.registry.get(algorithm_id)
        if not algorithm:
            logger.warning(f"Algorithm not found: {algorithm_id}")
            return None

        return self._evaluate_candidate(
            algorithm,
            data_availability,
            constraints,
            set()
        )

    def _sort_candidates(
        self,
        candidates: List[AlgorithmMetadata]
    ) -> List[AlgorithmMetadata]:
        """
        Sort candidates for deterministic selection.

        Sorts by:
        1. Category priority (baseline first)
        2. Validation score (higher is better)
        3. Algorithm ID (alphabetical for determinism)
        """
        def sort_key(algo: AlgorithmMetadata) -> Tuple:
            category_order = self._category_priority.get(algo.category, 99)

            # Validation score (negative for descending sort)
            validation_score = 0.0
            if algo.validation and algo.validation.accuracy_median:
                validation_score = -algo.validation.accuracy_median

            return (category_order, validation_score, algo.id)

        return sorted(candidates, key=sort_key)

    def _evaluate_candidate(
        self,
        algorithm: AlgorithmMetadata,
        data_availability: DataAvailability,
        constraints: SelectionConstraints,
        selected_outputs: Set[str]
    ) -> Optional[AlgorithmSelection]:
        """
        Evaluate if an algorithm can be selected.

        Args:
            algorithm: Candidate algorithm
            data_availability: Available data
            constraints: Selection constraints
            selected_outputs: Outputs from already-selected algorithms

        Returns:
            AlgorithmSelection if viable, None if not
        """
        alternatives_rejected: Dict[str, str] = {}

        # Check constraints
        allowed, reason = constraints.allows_algorithm(algorithm)
        if not allowed:
            return AlgorithmSelection(
                algorithm_id=algorithm.id,
                algorithm_name=algorithm.name,
                version=algorithm.version,
                quality=SelectionQuality.UNAVAILABLE,
                required_data_types=algorithm.required_data_types,
                available_data_types=list(data_availability.data_types),
                rationale=f"Rejected: {reason}",
                selection_reason=SelectionReason.RESOURCE_CONSTRAINED,
            )

        # Check data availability
        missing_data = data_availability.get_missing(algorithm.required_data_types)

        if missing_data:
            missing_str = ", ".join(dt.value for dt in missing_data)

            return AlgorithmSelection(
                algorithm_id=algorithm.id,
                algorithm_name=algorithm.name,
                version=algorithm.version,
                quality=SelectionQuality.UNAVAILABLE,
                required_data_types=algorithm.required_data_types,
                available_data_types=list(data_availability.data_types),
                rationale=f"Missing required data: {missing_str}",
                selection_reason=SelectionReason.MISSING_REQUIRED_DATA,
            )

        # Determine quality based on optional data
        quality = SelectionQuality.OPTIMAL
        degraded_mode = False
        rationale_parts = []

        # Check optional data availability
        optional_available = [
            dt for dt in algorithm.optional_data_types
            if data_availability.has_data_type(dt)
        ]
        optional_missing = [
            dt for dt in algorithm.optional_data_types
            if not data_availability.has_data_type(dt)
        ]

        if optional_missing:
            quality = SelectionQuality.ACCEPTABLE
            rationale_parts.append(
                f"Optional data unavailable: {[dt.value for dt in optional_missing]}"
            )

        # Check validation status
        if algorithm.validation:
            if not algorithm.validation.validated_regions:
                quality = SelectionQuality.ACCEPTABLE
                rationale_parts.append("Algorithm not validated in any region")
            elif algorithm.validation.accuracy_median and algorithm.validation.accuracy_median < 0.8:
                quality = SelectionQuality.ACCEPTABLE
                rationale_parts.append(
                    f"Validation accuracy: {algorithm.validation.accuracy_median:.1%}"
                )

        # Check category for experimental algorithms
        if algorithm.category == AlgorithmCategory.EXPERIMENTAL:
            quality = SelectionQuality.DEGRADED
            degraded_mode = True
            rationale_parts.append("Experimental algorithm")

        # Build version lock
        version_lock = {
            algorithm.id: algorithm.version,
        }

        # Build rationale
        if not rationale_parts:
            event_type_desc = algorithm.event_types[0] if algorithm.event_types else algorithm.id
            rationale = f"Best match for {event_type_desc} with all required data available"
        else:
            rationale = "; ".join(rationale_parts)

        return AlgorithmSelection(
            algorithm_id=algorithm.id,
            algorithm_name=algorithm.name,
            version=algorithm.version,
            quality=quality,
            required_data_types=algorithm.required_data_types,
            available_data_types=list(data_availability.data_types),
            degraded_mode=degraded_mode,
            rationale=rationale,
            selection_reason=SelectionReason.BEST_MATCH,
            alternatives_rejected=alternatives_rejected,
            version_lock=version_lock,
        )

    def _determine_execution_order(
        self,
        selections: Dict[str, AlgorithmSelection],
        sorted_candidates: Optional[List[AlgorithmMetadata]] = None
    ) -> List[str]:
        """
        Determine execution order based on dependencies and sorting.

        Sorts algorithms so that dependencies are executed first.
        Uses category-based ordering from sorted candidates when available.

        Args:
            selections: Map of algorithm ID to selection
            sorted_candidates: Pre-sorted candidate list for ordering

        Returns:
            Ordered list of algorithm IDs
        """
        if not selections:
            return []

        # Get algorithm metadata for dependency info
        algo_metadata = {
            algo_id: self.registry.get(algo_id)
            for algo_id in selections.keys()
        }

        # Build dependency graph
        dependencies: Dict[str, Set[str]] = {}
        for algo_id in selections.keys():
            metadata = algo_metadata.get(algo_id)
            if metadata:
                # Check if any required data types come from other selected algorithms
                deps = set()
                for other_id, other_meta in algo_metadata.items():
                    if other_meta and other_id != algo_id:
                        # Check if other algorithm's outputs are needed
                        for output in other_meta.outputs:
                            for required in metadata.required_data_types:
                                if output.lower() in required.value.lower():
                                    deps.add(other_id)
                dependencies[algo_id] = deps
            else:
                dependencies[algo_id] = set()

        # Use sorted_candidates order for tie-breaking if available
        candidate_order = {}
        if sorted_candidates:
            for i, candidate in enumerate(sorted_candidates):
                candidate_order[candidate.id] = i

        # Topological sort with category-aware ordering
        ordered = []
        remaining = set(selections.keys())

        while remaining:
            # Find algorithms with no unmet dependencies
            ready = [
                algo_id for algo_id in remaining
                if dependencies.get(algo_id, set()).issubset(set(ordered))
            ]

            if not ready:
                # Circular dependency or all have unmet deps - break tie by candidate order
                if candidate_order:
                    ready = [min(remaining, key=lambda x: candidate_order.get(x, float('inf')))]
                else:
                    ready = [min(remaining)]
            else:
                # Sort ready algorithms by candidate order (category priority)
                if candidate_order:
                    ready.sort(key=lambda x: candidate_order.get(x, float('inf')))
                else:
                    ready.sort()

            ordered.extend(ready)
            remaining -= set(ready)

        # Assign execution priority
        for priority, algo_id in enumerate(ordered):
            if algo_id in selections:
                selections[algo_id].execution_priority = priority

        return ordered

    def _document_trade_offs(
        self,
        candidates: List[AlgorithmMetadata],
        selections: Dict[str, AlgorithmSelection],
        context: str
    ) -> List[TradeOffRecord]:
        """
        Document trade-offs between selected algorithms.

        When multiple algorithms are selected for the same event type,
        documents why the ordering was chosen.

        Args:
            candidates: All candidate algorithms (sorted by priority)
            selections: Selected algorithms
            context: Decision context description

        Returns:
            List of TradeOffRecord objects
        """
        trade_offs = []

        selected_ids = list(selections.keys())
        if len(selected_ids) < 2:
            return trade_offs

        # Document trade-offs between selected algorithms with same event types
        for i, selected_id in enumerate(selected_ids):
            selection = selections[selected_id]
            selected_meta = self.registry.get(selected_id)

            if not selected_meta:
                continue

            # Find other selected algorithms with overlapping event types
            alternatives = []
            for other_id in selected_ids:
                if other_id == selected_id:
                    continue

                other_meta = self.registry.get(other_id)
                other_selection = selections.get(other_id)
                if not other_meta or not other_selection:
                    continue

                # Check for overlapping event types
                common_events = set(selected_meta.event_types) & set(other_meta.event_types)
                if common_events:
                    reason = f"Category: {other_meta.category.value}"
                    if other_selection.quality != selection.quality:
                        reason += f"; Quality: {other_selection.quality.value}"

                    alternatives.append({
                        "id": other_id,
                        "score": 0.5,  # Alternative score (lower than selected)
                        "reason": reason
                    })

            if alternatives:
                trade_off = TradeOffRecord(
                    decision_context=f"Algorithm selection for {context}",
                    selected_id=selected_id,
                    selected_score=1.0,
                    alternatives=alternatives[:3],  # Top 3 alternatives
                    rationale=selection.rationale,
                )
                trade_offs.append(trade_off)

        return trade_offs

    def verify_selection_plan(self, plan: SelectionPlan) -> bool:
        """
        Verify integrity of a selection plan.

        Checks that all hashes are valid and selection is reproducible.

        Args:
            plan: Selection plan to verify

        Returns:
            True if plan is valid, False otherwise
        """
        # Verify plan hash
        expected_hash = plan._compute_plan_hash()
        if plan.plan_hash != expected_hash:
            logger.error(f"Plan hash mismatch: {plan.plan_hash} != {expected_hash}")
            return False

        # Verify individual selection hashes
        for algo_id, selection in plan.selections.items():
            if not selection.verify_hash():
                logger.error(f"Selection hash invalid for {algo_id}")
                return False

        return True

    def update_constraints(self, constraints: SelectionConstraints) -> None:
        """Update selection constraints."""
        self.constraints = constraints

    def get_selector_info(self) -> Dict[str, Any]:
        """Get information about the selector."""
        return {
            "version": SELECTOR_VERSION,
            "registry_size": len(self.registry.algorithms),
            "constraints": {
                "max_memory_mb": self.constraints.max_memory_mb,
                "gpu_available": self.constraints.gpu_available,
                "require_deterministic": self.constraints.require_deterministic,
                "prefer_baseline": self.constraints.prefer_baseline,
            },
        }


def create_deterministic_selector(
    max_memory_mb: Optional[int] = None,
    gpu_available: bool = False,
    require_deterministic: bool = True,
    prefer_baseline: bool = True
) -> DeterministicSelector:
    """
    Factory function to create a configured deterministic selector.

    Args:
        max_memory_mb: Maximum memory constraint
        gpu_available: Whether GPU is available
        require_deterministic: Only select deterministic algorithms
        prefer_baseline: Prefer baseline over advanced algorithms

    Returns:
        Configured DeterministicSelector instance
    """
    constraints = SelectionConstraints(
        max_memory_mb=max_memory_mb,
        gpu_available=gpu_available,
        require_deterministic=require_deterministic,
        prefer_baseline=prefer_baseline,
    )

    return DeterministicSelector(constraints=constraints)
