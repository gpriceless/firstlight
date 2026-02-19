"""
Tests for the state model (JobPhase, JobStatus, transition validation,
and ExecutionStage mapping).

These are pure unit tests with no external dependencies.
"""

import pytest

from agents.orchestrator.state import ExecutionStage
from agents.orchestrator.state_model import (
    JobPhase,
    JobStatus,
    VALID_PHASE_STATUS_PAIRS,
    _TERMINAL_STATUSES,
    _PHASE_STATUSES,
    is_valid_phase_status,
    is_terminal_status,
    validate_transition,
    execution_stage_to_phase_status,
    phase_status_to_execution_stage,
)


# ---------------------------------------------------------------------------
# JobPhase and JobStatus enums
# ---------------------------------------------------------------------------


class TestJobPhaseEnum:
    """Tests for the JobPhase enum."""

    def test_all_phases_present(self):
        """All expected phases should be defined."""
        expected = {
            "QUEUED", "DISCOVERING", "INGESTING", "NORMALIZING",
            "ANALYZING", "REPORTING", "COMPLETE",
        }
        actual = {p.value for p in JobPhase}
        assert actual == expected

    def test_phase_is_str_enum(self):
        """JobPhase should be a str enum for easy serialization."""
        assert isinstance(JobPhase.QUEUED, str)
        assert JobPhase.QUEUED == "QUEUED"

    def test_phase_count(self):
        """There should be exactly 7 phases."""
        assert len(JobPhase) == 7


class TestJobStatusEnum:
    """Tests for the JobStatus enum."""

    def test_terminal_statuses_present(self):
        """FAILED and CANCELLED should be defined as universal statuses."""
        assert JobStatus.FAILED.value == "FAILED"
        assert JobStatus.CANCELLED.value == "CANCELLED"

    def test_complete_status_present(self):
        """COMPLETE status should exist for the COMPLETE phase."""
        assert JobStatus.COMPLETE.value == "COMPLETE"

    def test_per_phase_statuses_present(self):
        """Each phase should have its expected substatuses."""
        # Spot-check key statuses
        assert JobStatus.PENDING.value == "PENDING"
        assert JobStatus.VALIDATING.value == "VALIDATING"
        assert JobStatus.DISCOVERING.value == "DISCOVERING"
        assert JobStatus.INGESTING.value == "INGESTING"
        assert JobStatus.NORMALIZING.value == "NORMALIZING"
        assert JobStatus.ANALYZING.value == "ANALYZING"
        assert JobStatus.QUALITY_CHECK.value == "QUALITY_CHECK"
        assert JobStatus.REPORTING.value == "REPORTING"
        assert JobStatus.ASSEMBLING.value == "ASSEMBLING"


# ---------------------------------------------------------------------------
# Valid (phase, status) pairs
# ---------------------------------------------------------------------------


class TestValidPhaseStatusPairs:
    """Tests for VALID_PHASE_STATUS_PAIRS and is_valid_phase_status."""

    def test_all_phases_have_entries(self):
        """Every phase should have at least one valid status."""
        for phase in JobPhase:
            assert phase in VALID_PHASE_STATUS_PAIRS
            assert len(VALID_PHASE_STATUS_PAIRS[phase]) > 0

    def test_terminal_statuses_valid_in_every_phase(self):
        """FAILED and CANCELLED should be valid in every phase."""
        for phase in JobPhase:
            assert JobStatus.FAILED in VALID_PHASE_STATUS_PAIRS[phase]
            assert JobStatus.CANCELLED in VALID_PHASE_STATUS_PAIRS[phase]

    @pytest.mark.parametrize(
        "phase,status",
        [
            ("QUEUED", "PENDING"),
            ("QUEUED", "VALIDATING"),
            ("QUEUED", "VALIDATED"),
            ("QUEUED", "VALIDATION_FAILED"),
            ("DISCOVERING", "DISCOVERING"),
            ("DISCOVERING", "DISCOVERED"),
            ("DISCOVERING", "DISCOVERY_FAILED"),
            ("INGESTING", "INGESTING"),
            ("INGESTING", "INGESTED"),
            ("INGESTING", "INGESTION_FAILED"),
            ("NORMALIZING", "NORMALIZING"),
            ("NORMALIZING", "NORMALIZED"),
            ("NORMALIZING", "NORMALIZATION_FAILED"),
            ("ANALYZING", "ANALYZING"),
            ("ANALYZING", "QUALITY_CHECK"),
            ("ANALYZING", "ANALYZED"),
            ("ANALYZING", "ANALYSIS_FAILED"),
            ("REPORTING", "REPORTING"),
            ("REPORTING", "ASSEMBLING"),
            ("REPORTING", "REPORTED"),
            ("REPORTING", "REPORTING_FAILED"),
            ("COMPLETE", "COMPLETE"),
            # Terminal statuses in various phases
            ("QUEUED", "FAILED"),
            ("DISCOVERING", "CANCELLED"),
            ("ANALYZING", "FAILED"),
            ("COMPLETE", "CANCELLED"),
        ],
    )
    def test_valid_pairs_accepted(self, phase, status):
        """All defined valid (phase, status) pairs should be accepted."""
        assert is_valid_phase_status(phase, status), f"({phase}, {status}) should be valid"

    @pytest.mark.parametrize(
        "phase,status",
        [
            ("QUEUED", "DISCOVERING"),  # DISCOVERING is not valid in QUEUED
            ("QUEUED", "INGESTING"),
            ("DISCOVERING", "PENDING"),  # PENDING is not valid in DISCOVERING
            ("INGESTING", "QUALITY_CHECK"),
            ("COMPLETE", "PENDING"),  # PENDING is not valid in COMPLETE
            ("ANALYZING", "ASSEMBLING"),  # ASSEMBLING is for REPORTING
        ],
    )
    def test_invalid_pairs_rejected(self, phase, status):
        """Invalid (phase, status) combinations should be rejected."""
        assert not is_valid_phase_status(phase, status), f"({phase}, {status}) should be invalid"

    def test_invalid_phase_string_rejected(self):
        """A bogus phase string should be rejected."""
        assert not is_valid_phase_status("BOGUS_PHASE", "PENDING")

    def test_invalid_status_string_rejected(self):
        """A bogus status string should be rejected."""
        assert not is_valid_phase_status("QUEUED", "BOGUS_STATUS")


class TestTerminalStatus:
    """Tests for is_terminal_status."""

    def test_failed_is_terminal(self):
        assert is_terminal_status("FAILED")

    def test_cancelled_is_terminal(self):
        assert is_terminal_status("CANCELLED")

    def test_pending_is_not_terminal(self):
        assert not is_terminal_status("PENDING")

    def test_complete_is_not_terminal(self):
        """COMPLETE is a normal status, not a terminal one."""
        assert not is_terminal_status("COMPLETE")

    def test_bogus_is_not_terminal(self):
        assert not is_terminal_status("NONEXISTENT")


# ---------------------------------------------------------------------------
# Transition validation
# ---------------------------------------------------------------------------


class TestValidateTransition:
    """Tests for the validate_transition function."""

    def test_same_phase_status_change(self):
        """Changing status within the same phase should be valid."""
        assert validate_transition("QUEUED", "PENDING", "QUEUED", "VALIDATING")

    def test_forward_phase_transition(self):
        """Moving to the next phase should be valid."""
        assert validate_transition("QUEUED", "VALIDATED", "DISCOVERING", "DISCOVERING")
        assert validate_transition("DISCOVERING", "DISCOVERED", "INGESTING", "INGESTING")
        assert validate_transition("INGESTING", "INGESTED", "NORMALIZING", "NORMALIZING")
        assert validate_transition("NORMALIZING", "NORMALIZED", "ANALYZING", "ANALYZING")
        assert validate_transition("ANALYZING", "ANALYZED", "REPORTING", "REPORTING")
        assert validate_transition("REPORTING", "REPORTED", "COMPLETE", "COMPLETE")

    def test_skip_phase_transition(self):
        """Skipping phases forward should be valid (e.g., QUEUED -> ANALYZING)."""
        assert validate_transition("QUEUED", "VALIDATED", "ANALYZING", "ANALYZING")

    def test_backward_phase_transition_rejected(self):
        """Going backward in phase order should be rejected."""
        assert not validate_transition("DISCOVERING", "DISCOVERING", "QUEUED", "PENDING")
        assert not validate_transition("ANALYZING", "ANALYZING", "INGESTING", "INGESTING")
        assert not validate_transition("COMPLETE", "COMPLETE", "QUEUED", "PENDING")

    def test_transition_from_terminal_rejected(self):
        """Cannot transition from a terminal status."""
        assert not validate_transition("QUEUED", "FAILED", "QUEUED", "PENDING")
        assert not validate_transition("ANALYZING", "CANCELLED", "ANALYZING", "ANALYZING")

    def test_transition_from_complete_rejected(self):
        """Cannot transition from the COMPLETE phase."""
        assert not validate_transition("COMPLETE", "COMPLETE", "QUEUED", "PENDING")

    def test_transition_to_terminal_always_valid(self):
        """Any non-terminal state can transition to FAILED or CANCELLED."""
        assert validate_transition("QUEUED", "PENDING", "QUEUED", "FAILED")
        assert validate_transition("DISCOVERING", "DISCOVERING", "DISCOVERING", "CANCELLED")
        assert validate_transition("ANALYZING", "QUALITY_CHECK", "ANALYZING", "FAILED")
        assert validate_transition("REPORTING", "REPORTING", "REPORTING", "CANCELLED")

    def test_invalid_source_pair_rejected(self):
        """If the source (phase, status) is invalid, the transition is rejected."""
        assert not validate_transition("QUEUED", "DISCOVERING", "DISCOVERING", "DISCOVERING")

    def test_invalid_target_pair_rejected(self):
        """If the target (phase, status) is invalid, the transition is rejected."""
        assert not validate_transition("QUEUED", "PENDING", "QUEUED", "DISCOVERING")


# ---------------------------------------------------------------------------
# ExecutionStage <-> JobPhase/JobStatus mapping
# ---------------------------------------------------------------------------


class TestExecutionStageToPhaseStatus:
    """Tests for the forward mapping (ExecutionStage -> JobPhase/JobStatus)."""

    @pytest.mark.parametrize(
        "stage_value,expected_phase,expected_status",
        [
            ("pending", "QUEUED", "PENDING"),
            ("validating", "QUEUED", "VALIDATING"),
            ("discovery", "DISCOVERING", "DISCOVERING"),
            ("pipeline", "INGESTING", "INGESTING"),
            ("quality", "ANALYZING", "QUALITY_CHECK"),
            ("reporting", "REPORTING", "REPORTING"),
            ("assembly", "REPORTING", "ASSEMBLING"),
            ("completed", "COMPLETE", "COMPLETE"),
        ],
    )
    def test_forward_mapping(self, stage_value, expected_phase, expected_status):
        """Each ExecutionStage should map to the correct (phase, status)."""
        phase, status = execution_stage_to_phase_status(stage_value)
        assert phase.value == expected_phase
        assert status.value == expected_status

    def test_failed_maps_to_queued_failed(self):
        """FAILED maps to (QUEUED, FAILED) as placeholder -- actual phase is contextual."""
        phase, status = execution_stage_to_phase_status("failed")
        assert status == JobStatus.FAILED

    def test_cancelled_maps_to_queued_cancelled(self):
        """CANCELLED maps to (QUEUED, CANCELLED) as placeholder."""
        phase, status = execution_stage_to_phase_status("cancelled")
        assert status == JobStatus.CANCELLED

    def test_all_execution_stages_mapped(self):
        """Every ExecutionStage should have a mapping."""
        for stage in ExecutionStage:
            phase, status = execution_stage_to_phase_status(stage.value)
            assert isinstance(phase, JobPhase)
            assert isinstance(status, JobStatus)

    def test_invalid_stage_raises(self):
        """An invalid stage value should raise ValueError."""
        with pytest.raises(ValueError):
            execution_stage_to_phase_status("nonexistent_stage")


class TestPhaseStatusToExecutionStage:
    """Tests for the reverse mapping (JobPhase/JobStatus -> ExecutionStage)."""

    @pytest.mark.parametrize(
        "phase,status,expected_stage",
        [
            ("QUEUED", "PENDING", "pending"),
            ("QUEUED", "VALIDATING", "validating"),
            ("DISCOVERING", "DISCOVERING", "discovery"),
            ("INGESTING", "INGESTING", "pipeline"),
            ("NORMALIZING", "NORMALIZING", "pipeline"),
            ("ANALYZING", "ANALYZING", "pipeline"),
            ("ANALYZING", "QUALITY_CHECK", "quality"),
            ("REPORTING", "REPORTING", "reporting"),
            ("REPORTING", "ASSEMBLING", "assembly"),
            ("COMPLETE", "COMPLETE", "completed"),
        ],
    )
    def test_reverse_mapping(self, phase, status, expected_stage):
        """Key (phase, status) pairs should reverse-map correctly."""
        stage = phase_status_to_execution_stage(phase, status)
        assert stage.value == expected_stage

    def test_failed_reverse_mapping(self):
        """FAILED status should map to ExecutionStage.FAILED regardless of phase."""
        stage = phase_status_to_execution_stage("QUEUED", "FAILED")
        assert stage == ExecutionStage.FAILED
        stage = phase_status_to_execution_stage("ANALYZING", "FAILED")
        assert stage == ExecutionStage.FAILED

    def test_cancelled_reverse_mapping(self):
        """CANCELLED status should map to ExecutionStage.CANCELLED regardless of phase."""
        stage = phase_status_to_execution_stage("DISCOVERING", "CANCELLED")
        assert stage == ExecutionStage.CANCELLED

    def test_fallback_mapping_for_unmapped_status(self):
        """Statuses without exact mappings should fall back to the phase default."""
        # VALIDATED is in QUEUED phase but has no exact reverse mapping
        stage = phase_status_to_execution_stage("QUEUED", "VALIDATED")
        assert stage == ExecutionStage.PENDING  # Falls back to QUEUED default

    def test_ingesting_substatuses_map_to_pipeline(self):
        """Ingesting substatuses should fall back to PIPELINE."""
        stage = phase_status_to_execution_stage("INGESTING", "INGESTED")
        assert stage == ExecutionStage.PIPELINE

    def test_roundtrip_consistency(self):
        """Forward mapping followed by reverse should be idempotent for key stages."""
        for stage in [
            ExecutionStage.PENDING,
            ExecutionStage.VALIDATING,
            ExecutionStage.DISCOVERY,
            ExecutionStage.PIPELINE,
            ExecutionStage.QUALITY,
            ExecutionStage.REPORTING,
            ExecutionStage.ASSEMBLY,
            ExecutionStage.COMPLETED,
        ]:
            phase, status = execution_stage_to_phase_status(stage.value)
            roundtrip_stage = phase_status_to_execution_stage(phase.value, status.value)
            assert roundtrip_stage == stage, (
                f"Roundtrip failed for {stage}: "
                f"-> ({phase}, {status}) -> {roundtrip_stage}"
            )
