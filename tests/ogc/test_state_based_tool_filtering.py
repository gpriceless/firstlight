"""
Tests for state-based algorithm tool filtering.

Covers:
(a) Job in ANALYZING phase returns only analysis-category algorithm tool
    schemas, excluding discovery and reporting algorithms.
(b) Job transitioning from DISCOVERING to ANALYZING causes the tool list
    to update accordingly.
(c) Job in a phase with no mapped algorithms returns an empty tool list
    rather than all algorithms as fallback.

Task 4.11
"""

import pytest
from typing import List

from core.analysis.library.registry import (
    AlgorithmCategory,
    AlgorithmMetadata,
    AlgorithmRegistry,
    DataType,
)
from core.ogc.processors.factory import (
    get_algorithms_for_phase,
    PHASE_ALGORITHM_CATEGORIES,
)
from agents.orchestrator.state_model import JobPhase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rich_registry() -> AlgorithmRegistry:
    """
    Registry with algorithms across categories for filtering tests.

    Includes:
    - 2 baseline algorithms (for ANALYZING phase)
    - 1 advanced algorithm (for ANALYZING phase)
    - 1 experimental algorithm (NOT in ANALYZING mapping)
    - 1 deprecated baseline algorithm (should always be excluded)
    """
    reg = AlgorithmRegistry()

    # Baseline algorithms
    reg.register(AlgorithmMetadata(
        id="flood.baseline.threshold_sar",
        name="SAR Backscatter Threshold",
        category=AlgorithmCategory.BASELINE,
        version="1.0.0",
        event_types=["flood.*"],
        required_data_types=[DataType.SAR],
        description="Flood detection via SAR backscatter thresholding.",
    ))

    reg.register(AlgorithmMetadata(
        id="wildfire.baseline.nbr",
        name="Normalized Burn Ratio",
        category=AlgorithmCategory.BASELINE,
        version="1.0.0",
        event_types=["wildfire.*"],
        required_data_types=[DataType.OPTICAL],
        description="Burn severity via dNBR.",
    ))

    # Advanced algorithm
    reg.register(AlgorithmMetadata(
        id="flood.advanced.ml_ensemble",
        name="ML Ensemble Flood Detection",
        category=AlgorithmCategory.ADVANCED,
        version="2.0.0",
        event_types=["flood.*"],
        required_data_types=[DataType.SAR, DataType.OPTICAL],
        description="ML-based flood detection ensemble.",
    ))

    # Experimental algorithm (not in ANALYZING mapping)
    reg.register(AlgorithmMetadata(
        id="flood.experimental.diffusion",
        name="Diffusion Model Flood",
        category=AlgorithmCategory.EXPERIMENTAL,
        version="0.1.0",
        event_types=["flood.*"],
        required_data_types=[DataType.SAR],
        description="Experimental diffusion-based approach.",
    ))

    # Deprecated baseline (should be excluded)
    reg.register(AlgorithmMetadata(
        id="flood.baseline.old_threshold",
        name="Old Threshold Method",
        category=AlgorithmCategory.BASELINE,
        version="0.5.0",
        event_types=["flood.*"],
        required_data_types=[DataType.SAR],
        deprecated=True,
        replacement_algorithm="flood.baseline.threshold_sar",
    ))

    return reg


# ---------------------------------------------------------------------------
# Test: ANALYZING phase returns only analysis-category algorithms
# ---------------------------------------------------------------------------

class TestAnalyzingPhaseFiltering:
    """
    (a) Job in ANALYZING phase returns only baseline + advanced algorithms,
    excluding experimental and deprecated.
    """

    def test_analyzing_returns_baseline_and_advanced(self, rich_registry):
        algos = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        ids = {a.id for a in algos}

        # Baseline and advanced should be present
        assert "flood.baseline.threshold_sar" in ids
        assert "wildfire.baseline.nbr" in ids
        assert "flood.advanced.ml_ensemble" in ids

    def test_analyzing_excludes_experimental(self, rich_registry):
        algos = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        ids = {a.id for a in algos}

        # Experimental should NOT be present
        assert "flood.experimental.diffusion" not in ids

    def test_analyzing_excludes_deprecated(self, rich_registry):
        algos = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        ids = {a.id for a in algos}

        # Deprecated should NOT be present
        assert "flood.baseline.old_threshold" not in ids

    def test_analyzing_returns_correct_count(self, rich_registry):
        algos = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        # 2 baseline + 1 advanced = 3 (excluding 1 experimental + 1 deprecated)
        assert len(algos) == 3


# ---------------------------------------------------------------------------
# Test: Phase transition updates tool list
# ---------------------------------------------------------------------------

class TestPhaseTransitionToolUpdate:
    """
    (b) Job transitioning from DISCOVERING to ANALYZING causes the tool
    list to update accordingly.
    """

    def test_discovering_to_analyzing_transition(self, rich_registry):
        """DISCOVERING returns empty, ANALYZING returns algorithms."""
        # Before transition: DISCOVERING phase
        before = get_algorithms_for_phase("DISCOVERING", registry=rich_registry)
        assert len(before) == 0

        # After transition: ANALYZING phase
        after = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        assert len(after) > 0
        assert any(a.id == "flood.baseline.threshold_sar" for a in after)

    def test_queued_to_analyzing_transition(self, rich_registry):
        """QUEUED returns empty, ANALYZING returns algorithms."""
        before = get_algorithms_for_phase("QUEUED", registry=rich_registry)
        assert len(before) == 0

        after = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        assert len(after) == 3

    def test_analyzing_to_reporting_transition(self, rich_registry):
        """ANALYZING returns algorithms, REPORTING returns empty."""
        before = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        assert len(before) == 3

        after = get_algorithms_for_phase("REPORTING", registry=rich_registry)
        assert len(after) == 0

    def test_analyzing_to_complete_transition(self, rich_registry):
        """ANALYZING returns algorithms, COMPLETE returns empty."""
        before = get_algorithms_for_phase("ANALYZING", registry=rich_registry)
        assert len(before) > 0

        after = get_algorithms_for_phase("COMPLETE", registry=rich_registry)
        assert len(after) == 0


# ---------------------------------------------------------------------------
# Test: Empty tool list (not fallback to all)
# ---------------------------------------------------------------------------

class TestEmptyToolListNoFallback:
    """
    (c) Job in a phase with no mapped algorithms returns an empty tool
    list rather than all algorithms as fallback.
    """

    def test_queued_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("QUEUED", registry=rich_registry)
        assert len(algos) == 0

    def test_discovering_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("DISCOVERING", registry=rich_registry)
        assert len(algos) == 0

    def test_ingesting_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("INGESTING", registry=rich_registry)
        assert len(algos) == 0

    def test_normalizing_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("NORMALIZING", registry=rich_registry)
        assert len(algos) == 0

    def test_reporting_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("REPORTING", registry=rich_registry)
        assert len(algos) == 0

    def test_complete_returns_empty(self, rich_registry):
        algos = get_algorithms_for_phase("COMPLETE", registry=rich_registry)
        assert len(algos) == 0

    def test_unknown_phase_returns_empty_not_all(self, rich_registry):
        """Unknown phase returns empty, not all algorithms as fallback."""
        algos = get_algorithms_for_phase("NONEXISTENT_PHASE", registry=rich_registry)
        assert len(algos) == 0


# ---------------------------------------------------------------------------
# Test: PHASE_ALGORITHM_CATEGORIES mapping completeness
# ---------------------------------------------------------------------------

class TestPhaseAlgorithmMapping:
    """Tests for the PHASE_ALGORITHM_CATEGORIES mapping itself."""

    def test_all_job_phases_covered(self):
        """Every JobPhase has a mapping entry."""
        for phase in JobPhase:
            assert phase.value in PHASE_ALGORITHM_CATEGORIES, (
                f"JobPhase.{phase.name} missing from PHASE_ALGORITHM_CATEGORIES"
            )

    def test_analyzing_allows_baseline_and_advanced(self):
        """ANALYZING allows BASELINE and ADVANCED categories."""
        categories = PHASE_ALGORITHM_CATEGORIES["ANALYZING"]
        assert AlgorithmCategory.BASELINE in categories
        assert AlgorithmCategory.ADVANCED in categories

    def test_analyzing_excludes_experimental(self):
        """ANALYZING does NOT include EXPERIMENTAL by default."""
        categories = PHASE_ALGORITHM_CATEGORIES["ANALYZING"]
        assert AlgorithmCategory.EXPERIMENTAL not in categories

    def test_non_analyzing_phases_are_empty(self):
        """All phases except ANALYZING have empty category sets."""
        for phase_value, categories in PHASE_ALGORITHM_CATEGORIES.items():
            if phase_value != "ANALYZING":
                assert len(categories) == 0, (
                    f"Phase {phase_value} should have no mapped categories, "
                    f"got {categories}"
                )
