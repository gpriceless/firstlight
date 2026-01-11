"""
Tests for the Resilience Module.

Comprehensive test coverage for:
- Quality Assessment (optical, SAR, DEM, temporal)
- Fallback Chains (optical -> landsat, optical -> SAR, etc.)
- Degraded Mode Operations (mode detection, switching, partial coverage)
- Failure Handling (logging, recovery, provenance)
- Integration tests for full degraded pipelines
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch, MagicMock
import time

# Import degraded mode components
from core.resilience.degraded_mode.mode_manager import (
    DegradedModeLevel,
    DegradedModeTrigger,
    DegradedModeState,
    ModeTransition,
    DegradedModeConfig,
    DegradedModeManager,
    assess_degraded_mode,
)

from core.resilience.degraded_mode.partial_coverage import (
    GapFillMethod,
    CoverageQuality,
    CoverageGap,
    CoverageRegion,
    PartialAcquisition,
    CoverageMosaicResult,
    PartialCoverageConfig,
    PartialCoverageHandler,
    mosaic_partial_coverage,
)

from core.resilience.degraded_mode.low_confidence import (
    VotingMethod,
    CombinationMethod,
    DisagreementLevel,
    AlgorithmResult,
    DisagreementRegion,
    EnsembleResult,
    LowConfidenceConfig,
    LowConfidenceHandler,
    ensemble_voting,
)

# Import failure handling components
from core.resilience.failure.failure_log import (
    FailureComponent,
    FailureSeverity,
    FailureLog,
    FailureQuery,
    FailureStats,
    FailureLogger,
    log_failure,
    get_failure_logger,
)

from core.resilience.failure.recovery_strategies import (
    RecoveryStrategy,
    RecoveryOutcome,
    RecoveryAttempt,
    RecoveryResult,
    RecoveryConfig,
    RecoveryOrchestrator,
    retry_with_backoff,
)

from core.resilience.failure.provenance_tracking import (
    FallbackReason,
    FallbackType,
    FallbackDecision,
    FallbackChain,
    ProvenanceRecord,
    FallbackProvenanceConfig,
    FallbackProvenanceTracker,
    create_provenance_record,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_raster_100x100():
    """Sample 100x100 raster with some structure."""
    np.random.seed(42)
    data = np.random.rand(100, 100).astype(np.float32)
    # Add some structure
    data[20:40, 30:70] = 0.8  # High value region
    data[60:80, 20:50] = 0.2  # Low value region
    return data


@pytest.fixture
def sample_valid_mask_100x100():
    """Valid mask with some gaps."""
    mask = np.ones((100, 100), dtype=bool)
    # Create some gaps
    mask[10:20, 10:20] = False  # Gap 1
    mask[70:85, 60:90] = False  # Gap 2
    return mask


@pytest.fixture
def sample_partial_acquisitions():
    """Sample partial acquisitions for mosaicking."""
    np.random.seed(42)

    # Acquisition 1: covers left half
    data1 = np.random.rand(100, 100).astype(np.float32)
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[:, :60] = True

    # Acquisition 2: covers right half
    data2 = np.random.rand(100, 100).astype(np.float32)
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[:, 40:] = True

    return [
        PartialAcquisition(
            acquisition_id="acq_001",
            data=data1,
            bounds=(0, 0, 100, 100),
            valid_mask=mask1,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            quality_score=0.85,
            sensor="sentinel2",
        ),
        PartialAcquisition(
            acquisition_id="acq_002",
            data=data2,
            bounds=(0, 0, 100, 100),
            valid_mask=mask2,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            quality_score=0.90,
            sensor="sentinel2",
        ),
    ]


@pytest.fixture
def sample_algorithm_results():
    """Sample algorithm results for ensemble testing."""
    np.random.seed(42)
    shape = (50, 50)

    # Algorithm 1: Clear detection
    data1 = np.zeros(shape, dtype=np.float32)
    data1[10:30, 15:35] = 1.0  # Detected region
    conf1 = np.full(shape, 0.85, dtype=np.float32)

    # Algorithm 2: Similar but slightly different
    data2 = np.zeros(shape, dtype=np.float32)
    data2[12:32, 13:37] = 1.0  # Slightly shifted detection
    conf2 = np.full(shape, 0.80, dtype=np.float32)

    # Algorithm 3: Different result
    data3 = np.zeros(shape, dtype=np.float32)
    data3[8:28, 20:40] = 1.0  # More different
    conf3 = np.full(shape, 0.70, dtype=np.float32)

    return [
        AlgorithmResult(
            algorithm_id="algo_sar_threshold",
            algorithm_name="SAR Threshold",
            data=data1,
            confidence=conf1,
            overall_confidence=0.85,
        ),
        AlgorithmResult(
            algorithm_id="algo_ndwi",
            algorithm_name="NDWI",
            data=data2,
            confidence=conf2,
            overall_confidence=0.80,
        ),
        AlgorithmResult(
            algorithm_id="algo_change_detection",
            algorithm_name="Change Detection",
            data=data3,
            confidence=conf3,
            overall_confidence=0.70,
        ),
    ]


# ============================================================================
# TestQualityAssessment
# ============================================================================

class TestQualityAssessment:
    """Tests for quality assessment functionality."""

    def test_optical_quality_cloudy(self, sample_raster_100x100):
        """Test assessment with high cloud cover triggers degradation."""
        manager = DegradedModeManager()
        state = manager.assess_situation(cloud_cover=85.0)

        assert state.level in (DegradedModeLevel.MINIMAL, DegradedModeLevel.EMERGENCY)
        assert DegradedModeTrigger.HIGH_CLOUD_COVER in state.triggers
        assert state.confidence < 0.5

    def test_optical_quality_clear(self, sample_raster_100x100):
        """Test assessment with clear conditions stays in FULL mode."""
        manager = DegradedModeManager()
        state = manager.assess_situation(
            cloud_cover=10.0,
            spatial_coverage=95.0,
            available_sensors=["sentinel2", "landsat8", "modis"],
        )

        assert state.level == DegradedModeLevel.FULL
        assert state.confidence >= 0.8

    def test_sar_quality_noisy(self, sample_raster_100x100):
        """Test that low quality scores trigger degradation."""
        manager = DegradedModeManager()
        state = manager.assess_situation(
            data_quality_scores={"sar_backscatter": 0.3, "sar_coherence": 0.25},
        )

        assert state.level != DegradedModeLevel.FULL
        assert DegradedModeTrigger.DATA_QUALITY_LOW in state.triggers

    def test_dem_quality_voids(self, sample_raster_100x100):
        """Test spatial coverage gaps affect mode."""
        manager = DegradedModeManager()
        state = manager.assess_situation(spatial_coverage=60.0)

        assert state.level != DegradedModeLevel.FULL
        assert DegradedModeTrigger.SPATIAL_GAP in state.triggers

    def test_temporal_no_baseline(self, sample_raster_100x100):
        """Test missing baseline triggers appropriate mode."""
        manager = DegradedModeManager()
        state = manager.assess_situation(baseline_available=False)

        assert DegradedModeTrigger.MISSING_BASELINE in state.triggers
        assert state.confidence < 1.0


# ============================================================================
# TestFallbackChains
# ============================================================================

class TestFallbackChains:
    """Tests for fallback chain functionality."""

    def test_optical_fallback_to_landsat(self):
        """Test fallback from Sentinel-2 to Landsat-8."""
        tracker = FallbackProvenanceTracker(event_id="test_001")
        chain_id = tracker.start_chain("data_acquisition")

        decision = tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sentinel2_optical",
            fallback_strategy="landsat8_optical",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
            context={"cloud_cover": 85},
        )

        assert decision.original_strategy == "sentinel2_optical"
        assert decision.fallback_strategy == "landsat8_optical"
        assert decision.confidence_impact < 0  # Confidence decreased

    def test_optical_fallback_to_sar(self):
        """Test fallback from optical to SAR."""
        tracker = FallbackProvenanceTracker(event_id="test_002")
        chain_id = tracker.start_chain("data_acquisition")

        # First fallback: try Landsat
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sentinel2_optical",
            fallback_strategy="landsat8_optical",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
        )

        # Second fallback: go to SAR
        decision = tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="landsat8_optical",
            fallback_strategy="sentinel1_sar",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
        )

        chain = tracker._chains[chain_id]
        assert chain.fallback_depth == 2
        assert chain.final_confidence < chain.initial_confidence

    def test_sar_fallback_filtering(self):
        """Test SAR fallback with enhanced filtering."""
        tracker = FallbackProvenanceTracker(event_id="test_003")
        chain_id = tracker.start_chain("sar_processing")

        decision = tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sar_standard_filter",
            fallback_strategy="sar_enhanced_lee_filter",
            fallback_type=FallbackType.ALGORITHM_ALTERNATIVE,
            reason=FallbackReason.DATA_LOW_QUALITY,
            context={"speckle_level": "high"},
        )

        assert decision.fallback_type == FallbackType.ALGORITHM_ALTERNATIVE
        assert decision.reason == FallbackReason.DATA_LOW_QUALITY

    def test_dem_fallback_srtm(self):
        """Test DEM fallback from Copernicus to SRTM."""
        tracker = FallbackProvenanceTracker(event_id="test_004")
        chain_id = tracker.start_chain("dem_acquisition")

        decision = tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="copernicus_dem",
            fallback_strategy="srtm_dem",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.DATA_UNAVAILABLE,
        )

        assert decision.fallback_type == FallbackType.SENSOR_ALTERNATIVE

    def test_algorithm_fallback(self):
        """Test algorithm fallback when primary fails."""
        tracker = FallbackProvenanceTracker(event_id="test_005")
        chain_id = tracker.start_chain("algorithm_execution")

        # Record algorithm fallback
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="unet_segmentation",
            fallback_strategy="ndwi_threshold",
            fallback_type=FallbackType.ALGORITHM_ALTERNATIVE,
            reason=FallbackReason.ALGORITHM_FAILURE,
            context={"error": "GPU memory exceeded"},
        )

        record = tracker.generate_provenance_record()
        assert record.total_fallbacks == 1


# ============================================================================
# TestDegradedMode
# ============================================================================

class TestDegradedMode:
    """Tests for degraded mode operations."""

    def test_mode_detection(self):
        """Test automatic mode detection based on conditions."""
        # Test FULL mode
        state = assess_degraded_mode(
            cloud_cover=10.0,
            spatial_coverage=95.0,
            available_sensors=["sentinel1", "sentinel2"],
        )
        assert state.level == DegradedModeLevel.FULL

        # Test PARTIAL mode
        state = assess_degraded_mode(
            cloud_cover=50.0,
            spatial_coverage=70.0,
        )
        assert state.level == DegradedModeLevel.PARTIAL

        # Test MINIMAL mode
        state = assess_degraded_mode(
            cloud_cover=75.0,
            spatial_coverage=50.0,
        )
        assert state.level == DegradedModeLevel.MINIMAL

        # Test EMERGENCY mode
        state = assess_degraded_mode(
            cloud_cover=95.0,
            spatial_coverage=30.0,
            baseline_available=False,
        )
        assert state.level == DegradedModeLevel.EMERGENCY

    def test_mode_switching(self):
        """Test mode switching and transition tracking."""
        manager = DegradedModeManager()

        # Start in FULL mode
        state1 = manager.assess_situation(cloud_cover=10.0)
        assert state1.level == DegradedModeLevel.FULL

        # Conditions worsen -> PARTIAL
        state2 = manager.assess_situation(cloud_cover=55.0)
        assert state2.level == DegradedModeLevel.PARTIAL

        # Check transition was recorded
        history = manager.get_transition_history()
        assert len(history) >= 1

        # Verify transition details
        transition = history[-1]
        assert transition.from_level == DegradedModeLevel.FULL
        assert transition.to_level == DegradedModeLevel.PARTIAL
        assert transition.is_degradation

    def test_partial_coverage_handling(self, sample_partial_acquisitions):
        """Test handling of partial spatial coverage."""
        handler = PartialCoverageHandler()

        result = handler.mosaic_partial_acquisitions(
            sample_partial_acquisitions,
            fill_gaps=True,
        )

        assert result.coverage_percent > 90  # Should have most coverage
        assert len(result.sources_used) == 2
        assert result.fill_method != GapFillMethod.NONE

    def test_low_confidence_ensemble(self, sample_algorithm_results):
        """Test ensemble methods for low confidence situations."""
        handler = LowConfidenceHandler()

        result = handler.ensemble_voting(
            sample_algorithm_results,
            method=VotingMethod.WEIGHTED_MAJORITY,
        )

        assert result.overall_confidence > 0
        assert len(result.algorithms_used) == 3
        assert result.data.shape == (50, 50)


# ============================================================================
# TestFailureHandling
# ============================================================================

class TestFailureHandling:
    """Tests for failure logging and handling."""

    def test_failure_logging(self):
        """Test structured failure logging."""
        logger = FailureLogger()

        failure = logger.log_failure(
            component=FailureComponent.DISCOVERY,
            error_type="timeout",
            error_message="STAC query timed out after 30s",
            severity=FailureSeverity.ERROR,
            context={"catalog": "earth-search", "timeout": 30},
        )

        assert failure.failure_id is not None
        assert failure.component == FailureComponent.DISCOVERY
        assert failure.error_type == "timeout"
        assert "earth-search" in str(failure.context)

    def test_recovery_retry(self):
        """Test recovery with retry mechanism."""
        attempt_count = 0

        def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Network error")
            return "success"

        config = RecoveryConfig(
            max_retries=5,
            initial_delay_ms=10,
            max_delay_ms=100,
        )
        orchestrator = RecoveryOrchestrator(config)

        result = orchestrator.retry_with_backoff(flaky_operation)

        assert result.success
        assert result.result == "success"
        assert attempt_count == 3

    def test_recovery_alternative(self):
        """Test recovery using alternative strategies."""
        def primary_fails():
            raise ValueError("Primary failed")

        def alternative_works():
            return "alternative_result"

        orchestrator = RecoveryOrchestrator()
        result = orchestrator.attempt_recovery(
            operation=primary_fails,
            fallbacks=[
                (RecoveryStrategy.ALTERNATIVE, alternative_works),
            ],
        )

        assert result.success
        assert result.result == "alternative_result"
        assert result.strategy_used == RecoveryStrategy.ALTERNATIVE

    def test_provenance_tracking(self):
        """Test fallback provenance tracking."""
        tracker = FallbackProvenanceTracker(event_id="flood_001")

        chain_id = tracker.start_chain("analysis")
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sentinel2",
            fallback_strategy="landsat8",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
        )
        tracker.complete_chain(chain_id)

        tracker.record_data_source("landsat8", success=True)
        tracker.record_algorithm("ndwi", success=True)

        record = tracker.generate_provenance_record()

        assert record.event_id == "flood_001"
        assert record.total_fallbacks == 1
        assert "landsat8" in record.data_sources_used
        assert record.reproducibility_hash is not None


# ============================================================================
# TestIntegration
# ============================================================================

class TestIntegration:
    """Integration tests for full degraded pipelines."""

    def test_full_degraded_pipeline(self, sample_partial_acquisitions, sample_algorithm_results):
        """Test full pipeline in degraded mode."""
        # Simulate degraded conditions
        manager = DegradedModeManager()
        state = manager.assess_situation(
            cloud_cover=70.0,
            spatial_coverage=75.0,
            baseline_available=False,
        )

        assert state.is_degraded

        # Handle partial coverage
        coverage_handler = PartialCoverageHandler()
        mosaic_result = coverage_handler.mosaic_partial_acquisitions(
            sample_partial_acquisitions,
            fill_gaps=True,
        )

        assert mosaic_result.coverage_percent > 80

        # Run ensemble for low confidence
        confidence_handler = LowConfidenceHandler()
        ensemble_result = confidence_handler.ensemble_voting(sample_algorithm_results)

        assert ensemble_result.overall_confidence > 0.3

    def test_multiple_fallbacks(self):
        """Test pipeline with multiple sequential fallbacks."""
        tracker = FallbackProvenanceTracker(event_id="complex_001")

        # Chain of fallbacks
        chain_id = tracker.start_chain("data_acquisition")

        # Fallback 1: Sentinel-2 -> Landsat-8
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="sentinel2",
            fallback_strategy="landsat8",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
        )

        # Fallback 2: Landsat-8 -> Sentinel-1 SAR
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="landsat8",
            fallback_strategy="sentinel1_sar",
            fallback_type=FallbackType.SENSOR_ALTERNATIVE,
            reason=FallbackReason.CLOUD_COVER,
        )

        # Fallback 3: Resolution degradation
        tracker.record_fallback(
            chain_id=chain_id,
            original_strategy="10m_resolution",
            fallback_strategy="30m_resolution",
            fallback_type=FallbackType.RESOLUTION_DEGRADATION,
            reason=FallbackReason.RESOURCE_LIMIT,
        )

        tracker.complete_chain(chain_id)
        record = tracker.generate_provenance_record()

        assert record.total_fallbacks == 3
        assert record.overall_confidence < 1.0
        # Confidence should decrease with each fallback
        breakdown = tracker.get_confidence_breakdown()
        assert breakdown["final_confidence"] < breakdown["initial_confidence"]

    def test_recovery_from_failure(self):
        """Test complete recovery from a failure scenario."""
        # Set up failure logger
        failure_logger = FailureLogger()

        # Simulate operation that fails then succeeds
        attempts = []

        def operation_with_recovery():
            if len(attempts) < 2:
                attempts.append("fail")
                raise TimeoutError("Operation timed out")
            attempts.append("success")
            return {"data": [1, 2, 3]}

        # Set up recovery
        config = RecoveryConfig(
            max_retries=3,
            initial_delay_ms=5,
            enable_degradation=True,
        )
        orchestrator = RecoveryOrchestrator(config)

        # Track attempts
        logged_attempts = []

        def on_attempt(attempt):
            logged_attempts.append(attempt)
            if not attempt.success:
                failure_logger.log_failure(
                    component=FailureComponent.PIPELINE,
                    error_type="timeout",
                    error_message=attempt.error or "Unknown error",
                    recovery_attempted=True,
                )

        result = orchestrator.attempt_recovery(
            operation=operation_with_recovery,
            on_attempt=on_attempt,
        )

        assert result.success
        assert result.outcome == RecoveryOutcome.SUCCESS
        assert len(logged_attempts) == 3

        # Check failure logs
        failures = failure_logger.get_recent(10)
        assert len(failures) == 2  # Two failures before success


# ============================================================================
# Additional Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_acquisitions_raises(self):
        """Test that empty acquisition list raises error."""
        handler = PartialCoverageHandler()

        with pytest.raises(ValueError, match="No acquisitions"):
            handler.mosaic_partial_acquisitions([])

    def test_single_algorithm_insufficient(self):
        """Test that single algorithm is insufficient for ensemble."""
        handler = LowConfidenceHandler()

        single_result = AlgorithmResult(
            algorithm_id="single",
            algorithm_name="Single Algorithm",
            data=np.zeros((50, 50)),
            confidence=np.ones((50, 50)) * 0.8,
            overall_confidence=0.8,
        )

        with pytest.raises(ValueError, match="at least"):
            handler.ensemble_voting([single_result])

    def test_mismatched_shapes_raises(self):
        """Test that mismatched algorithm output shapes raise error."""
        handler = LowConfidenceHandler()

        results = [
            AlgorithmResult(
                algorithm_id="algo1",
                algorithm_name="Algo 1",
                data=np.zeros((50, 50)),
                confidence=np.ones((50, 50)),
                overall_confidence=0.8,
            ),
            AlgorithmResult(
                algorithm_id="algo2",
                algorithm_name="Algo 2",
                data=np.zeros((60, 60)),  # Different shape!
                confidence=np.ones((60, 60)),
                overall_confidence=0.8,
            ),
        ]

        with pytest.raises(ValueError, match="shapes don't match"):
            handler.ensemble_voting(results)

    def test_confidence_bounds(self):
        """Test confidence scores stay within 0-1."""
        manager = DegradedModeManager()

        # Extreme conditions
        state = manager.assess_situation(
            cloud_cover=100.0,
            spatial_coverage=0.0,
            baseline_available=False,
            data_quality_scores={"all": 0.0},
        )

        assert 0.0 <= state.confidence <= 1.0

    def test_recovery_timeout(self):
        """Test recovery respects timeout."""
        def slow_operation():
            time.sleep(0.1)
            raise ValueError("Still failing")

        config = RecoveryConfig(
            max_retries=10,
            timeout_ms=50,  # Very short timeout
            initial_delay_ms=1,
        )
        orchestrator = RecoveryOrchestrator(config)

        result = orchestrator.retry_with_backoff(slow_operation)

        assert not result.success
        assert result.outcome == RecoveryOutcome.TIMEOUT

    def test_mode_level_ordering(self):
        """Test mode level severity ordering."""
        assert DegradedModeLevel.FULL.severity < DegradedModeLevel.PARTIAL.severity
        assert DegradedModeLevel.PARTIAL.severity < DegradedModeLevel.MINIMAL.severity
        assert DegradedModeLevel.MINIMAL.severity < DegradedModeLevel.EMERGENCY.severity

    def test_failure_query_filters(self):
        """Test failure query filtering."""
        logger = FailureLogger()

        # Log various failures
        logger.log_failure(
            FailureComponent.DISCOVERY, "timeout", "Timeout 1",
            severity=FailureSeverity.ERROR,
        )
        logger.log_failure(
            FailureComponent.PIPELINE, "error", "Error 1",
            severity=FailureSeverity.WARNING,
        )
        logger.log_failure(
            FailureComponent.DISCOVERY, "network", "Network 1",
            severity=FailureSeverity.CRITICAL,
        )

        # Query by component
        results = logger.query(FailureQuery(component=FailureComponent.DISCOVERY))
        assert len(results) == 2

        # Query by severity
        results = logger.query(FailureQuery(severity_min=FailureSeverity.ERROR))
        assert len(results) == 2  # ERROR and CRITICAL

    def test_provenance_hash_consistency(self):
        """Test that provenance hash is consistent for same inputs."""
        record1 = create_provenance_record(
            event_id="test",
            fallback_decisions=[
                {"original": "s2", "fallback": "l8", "reason": "cloud_cover"},
            ],
            data_sources=["l8"],
            algorithms=["ndwi"],
        )

        record2 = create_provenance_record(
            event_id="test",
            fallback_decisions=[
                {"original": "s2", "fallback": "l8", "reason": "cloud_cover"},
            ],
            data_sources=["l8"],
            algorithms=["ndwi"],
        )

        assert record1.reproducibility_hash == record2.reproducibility_hash


class TestVotingMethods:
    """Tests for different voting methods."""

    @pytest.fixture
    def binary_results(self):
        """Create binary classification results for voting tests."""
        np.random.seed(42)
        shape = (30, 30)

        # All agree on most pixels
        base = np.zeros(shape, dtype=np.float32)
        base[10:20, 10:20] = 1.0

        results = []
        for i in range(3):
            data = base.copy()
            # Add some noise
            noise = np.random.random(shape) > 0.9
            data[noise] = 1 - data[noise]

            confidence = np.full(shape, 0.7 + i * 0.05, dtype=np.float32)

            results.append(AlgorithmResult(
                algorithm_id=f"algo_{i}",
                algorithm_name=f"Algorithm {i}",
                data=data,
                confidence=confidence,
                overall_confidence=0.7 + i * 0.05,
            ))

        return results

    def test_majority_vote(self, binary_results):
        """Test simple majority voting."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_voting(binary_results, method=VotingMethod.MAJORITY)

        assert result.data.shape == (30, 30)
        # Center region should be detected
        assert np.mean(result.data[10:20, 10:20]) > 0.5

    def test_weighted_majority_vote(self, binary_results):
        """Test confidence-weighted majority voting."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_voting(binary_results, method=VotingMethod.WEIGHTED_MAJORITY)

        assert result.data.shape == (30, 30)
        assert result.overall_confidence > 0

    def test_unanimous_vote(self, binary_results):
        """Test unanimous voting (strict)."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_voting(binary_results, method=VotingMethod.UNANIMOUS)

        # Unanimous is more conservative
        assert result.data.shape == (30, 30)

    def test_consensus_vote(self, binary_results):
        """Test consensus voting with threshold."""
        config = LowConfidenceConfig(agreement_threshold=0.7)
        handler = LowConfidenceHandler(config)
        result = handler.ensemble_voting(binary_results, method=VotingMethod.CONSENSUS)

        assert result.data.shape == (30, 30)


class TestCombinationMethods:
    """Tests for continuous output combination methods."""

    @pytest.fixture
    def continuous_results(self):
        """Create continuous results for combination tests."""
        np.random.seed(42)
        shape = (30, 30)

        results = []
        for i in range(3):
            data = np.random.rand(*shape).astype(np.float32) * 0.5 + 0.25
            confidence = np.full(shape, 0.6 + i * 0.1, dtype=np.float32)

            results.append(AlgorithmResult(
                algorithm_id=f"algo_{i}",
                algorithm_name=f"Algorithm {i}",
                data=data,
                confidence=confidence,
                overall_confidence=0.6 + i * 0.1,
            ))

        return results

    def test_mean_combination(self, continuous_results):
        """Test simple mean combination."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_combination(
            continuous_results, method=CombinationMethod.MEAN
        )

        assert result.data.shape == (30, 30)
        # Mean should be roughly middle of range
        assert 0.2 < np.mean(result.data) < 0.8

    def test_weighted_mean_combination(self, continuous_results):
        """Test confidence-weighted mean combination."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_combination(
            continuous_results, method=CombinationMethod.WEIGHTED_MEAN
        )

        assert result.data.shape == (30, 30)

    def test_median_combination(self, continuous_results):
        """Test median combination."""
        handler = LowConfidenceHandler()
        result = handler.ensemble_combination(
            continuous_results, method=CombinationMethod.MEDIAN
        )

        assert result.data.shape == (30, 30)


class TestPartialCoverage:
    """Tests for partial coverage handling."""

    def test_coverage_analysis(self, sample_raster_100x100, sample_valid_mask_100x100):
        """Test coverage analysis."""
        handler = PartialCoverageHandler()

        analysis = handler.analyze_coverage(
            sample_raster_100x100,
            sample_valid_mask_100x100,
        )

        assert "coverage_percent" in analysis
        assert analysis["coverage_percent"] < 100  # Some gaps
        assert "gap_count" in analysis
        assert analysis["gap_count"] > 0

    def test_gap_interpolation_nearest(self, sample_raster_100x100, sample_valid_mask_100x100):
        """Test nearest neighbor gap interpolation."""
        handler = PartialCoverageHandler()

        filled, confidence = handler.interpolate_gaps(
            sample_raster_100x100,
            sample_valid_mask_100x100,
            method=GapFillMethod.NEAREST,
        )

        # Gaps should be filled
        assert not np.any(np.isnan(filled))
        # Confidence should be lower in gap regions
        gap_confidence = confidence[~sample_valid_mask_100x100]
        valid_confidence = confidence[sample_valid_mask_100x100]
        assert np.mean(gap_confidence) < np.mean(valid_confidence)

    def test_gap_interpolation_linear(self, sample_raster_100x100, sample_valid_mask_100x100):
        """Test linear gap interpolation."""
        handler = PartialCoverageHandler()

        filled, confidence = handler.interpolate_gaps(
            sample_raster_100x100,
            sample_valid_mask_100x100,
            method=GapFillMethod.LINEAR,
        )

        # Most gaps should be filled
        fill_rate = np.mean(~np.isnan(filled))
        assert fill_rate > 0.9


class TestFailureStatistics:
    """Tests for failure statistics and analytics."""

    def test_failure_statistics(self):
        """Test failure statistics calculation."""
        logger = FailureLogger()

        # Log various failures
        for i in range(5):
            logger.log_failure(
                FailureComponent.DISCOVERY, "timeout", f"Timeout {i}",
                severity=FailureSeverity.ERROR,
                outcome="failed",
            )

        for i in range(3):
            logger.log_failure(
                FailureComponent.PIPELINE, "error", f"Error {i}",
                severity=FailureSeverity.WARNING,
                outcome="recovered",
            )

        stats = logger.get_statistics()

        assert stats.total_count == 8
        assert stats.by_component["discovery"] == 5
        assert stats.by_component["pipeline"] == 3
        assert stats.recovery_rate > 0

    def test_failure_patterns(self):
        """Test failure pattern detection."""
        logger = FailureLogger()

        # Create pattern: same error type from same component
        for i in range(5):
            logger.log_failure(
                FailureComponent.NETWORK, "connection_refused",
                f"Connection refused to service {i % 2}",
            )

        patterns = logger.find_patterns(min_occurrences=3)

        assert len(patterns) >= 1
        assert patterns[0]["occurrence_count"] >= 5


class TestModeTransitions:
    """Tests for mode transition tracking."""

    def test_transition_history(self):
        """Test transition history tracking."""
        manager = DegradedModeManager()

        # Create multiple transitions
        manager.assess_situation(cloud_cover=10)  # FULL
        manager.assess_situation(cloud_cover=50)  # PARTIAL
        manager.assess_situation(cloud_cover=80)  # MINIMAL

        history = manager.get_transition_history()

        assert len(history) >= 2

    def test_mode_statistics(self):
        """Test mode usage statistics."""
        manager = DegradedModeManager()

        # Create transitions
        manager.assess_situation(cloud_cover=10)
        manager.assess_situation(cloud_cover=50)
        manager.assess_situation(cloud_cover=80)
        manager.assess_situation(cloud_cover=20)  # Recovery

        stats = manager.get_mode_statistics()

        assert stats["total_transitions"] >= 3
        assert "degradations" in stats
        assert "recoveries" in stats

    def test_notification_callback(self):
        """Test mode change notification callbacks."""
        notifications = []

        def on_mode_change(state):
            notifications.append(state)

        config = DegradedModeConfig(
            enable_notifications=True,
            notification_callbacks=[on_mode_change],
        )
        manager = DegradedModeManager(config)

        # Trigger mode changes
        manager.assess_situation(cloud_cover=10)  # FULL
        manager.assess_situation(cloud_cover=60)  # PARTIAL

        assert len(notifications) >= 1


class TestRecoveryStrategies:
    """Tests for recovery strategy configurations."""

    def test_exponential_backoff(self):
        """Test exponential backoff delay calculation."""
        config = RecoveryConfig(
            initial_delay_ms=100,
            backoff_multiplier=2.0,
            max_delay_ms=1000,
            jitter=False,
        )
        orchestrator = RecoveryOrchestrator(config)

        # Calculate expected delays
        delays = [orchestrator._calculate_delay(i) for i in range(5)]

        assert delays[0] == 100
        assert delays[1] == 200
        assert delays[2] == 400
        assert delays[3] == 800
        assert delays[4] == 1000  # Capped at max

    def test_graceful_degradation(self):
        """Test graceful degradation fallback."""
        def always_fails():
            raise ValueError("Always fails")

        degraded_result = {"status": "degraded", "data": None}

        orchestrator = RecoveryOrchestrator()
        result = orchestrator.with_graceful_degradation(
            operation=always_fails,
            degraded_result=degraded_result,
            degraded_confidence=0.3,
        )

        assert result.success
        assert result.outcome == RecoveryOutcome.DEGRADED
        assert result.result == degraded_result
        assert result.confidence == 0.3

    def test_alternative_strategies(self):
        """Test trying multiple alternative strategies."""
        def primary():
            raise ValueError("Primary failed")

        def alt1():
            raise ValueError("Alt 1 failed")

        def alt2():
            return "Alt 2 succeeded"

        orchestrator = RecoveryOrchestrator()
        result = orchestrator.with_alternatives(
            primary=primary,
            alternatives=[alt1, alt2],
        )

        assert result.success
        assert result.result == "Alt 2 succeeded"
        assert result.attempt_count >= 3


# ============================================================================
# Test module runs correctly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
