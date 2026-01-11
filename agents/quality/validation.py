"""
Validation Execution for Quality Agent.

Provides the ValidationRunner class that executes various validation
strategies on analysis outputs:
- Sanity checks (spatial, temporal, value plausibility)
- Cross-model validation
- Cross-sensor validation
- Historical baseline comparison

This module integrates with core/quality/sanity/ and core/quality/validation/
to provide a unified interface for the Quality Agent.

Example:
    runner = ValidationRunner()

    # Run sanity checks
    sanity_result = await runner.run_sanity_checks(data, context)

    # Run cross-model validation
    model_result = await runner.run_model_validation(outputs)

    # Run cross-sensor validation
    sensor_result = await runner.run_sensor_validation(observations)

    # Run historical comparison
    historical_result = await runner.run_historical_validation(data, baseline)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import core sanity modules
from core.quality.sanity import (
    SanitySuite,
    SanitySuiteConfig,
    SanitySuiteResult,
    SpatialCoherenceChecker,
    SpatialCoherenceConfig,
    SpatialCoherenceResult,
    TemporalConsistencyChecker,
    TemporalConsistencyConfig,
    TemporalConsistencyResult,
    ValuePlausibilityChecker,
    ValuePlausibilityConfig,
    ValuePlausibilityResult,
    ValueType,
    ArtifactDetector,
    ArtifactDetectionConfig,
    ArtifactDetectionResult,
)

# Import core validation modules
from core.quality.validation import (
    CrossModelValidator,
    CrossModelConfig,
    CrossModelResult,
    ModelOutput,
    AgreementLevel,
    CrossSensorValidator,
    CrossSensorConfig,
    CrossSensorResult,
    SensorObservation,
    SensorType,
    ValidationOutcome,
    HistoricalValidator,
    HistoricalConfig,
    HistoricalResult,
    HistoricalBaseline,
    ConsensusGenerator,
    ConsensusConfig,
    ConsensusResult,
    ConsensusSource,
    ConsensusStrategy,
)


logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Mode for validation operations."""
    STRICT = "strict"          # All checks must pass
    STANDARD = "standard"      # Critical checks must pass
    LENIENT = "lenient"        # Warnings only for most issues
    QUICK = "quick"            # Minimal checks for speed


@dataclass
class ValidationRunnerConfig:
    """
    Configuration for ValidationRunner.

    Attributes:
        mode: Validation mode (strict, standard, lenient, quick)
        parallel_execution: Run independent checks in parallel
        cache_results: Cache validation results
        timeout_seconds: Timeout for individual checks
        sanity_config: Configuration for sanity suite
        model_config: Configuration for cross-model validation
        sensor_config: Configuration for cross-sensor validation
        historical_config: Configuration for historical validation
    """
    mode: ValidationMode = ValidationMode.STANDARD
    parallel_execution: bool = True
    cache_results: bool = True
    timeout_seconds: float = 60.0

    # Sub-configs
    sanity_config: Optional[SanitySuiteConfig] = None
    model_config: Optional[CrossModelConfig] = None
    sensor_config: Optional[CrossSensorConfig] = None
    historical_config: Optional[HistoricalConfig] = None


@dataclass
class SanityCheckResult:
    """
    Result from running sanity checks.

    Attributes:
        passed: Whether all critical checks passed
        overall_score: Combined quality score (0-1)
        spatial_result: Results from spatial coherence checks
        value_result: Results from value plausibility checks
        temporal_result: Results from temporal consistency checks
        artifact_result: Results from artifact detection
        issues: List of detected issues
        duration_seconds: Time taken for checks
    """
    passed: bool
    overall_score: float
    spatial_result: Optional[Dict[str, Any]] = None
    value_result: Optional[Dict[str, Any]] = None
    temporal_result: Optional[Dict[str, Any]] = None
    artifact_result: Optional[Dict[str, Any]] = None
    issues: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "spatial_result": self.spatial_result,
            "value_result": self.value_result,
            "temporal_result": self.temporal_result,
            "artifact_result": self.artifact_result,
            "issues": self.issues,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class ModelValidationResult:
    """
    Result from cross-model validation.

    Attributes:
        passed: Whether models agree sufficiently
        agreement_level: Level of agreement between models
        overall_agreement: Agreement score (0-1)
        comparisons: Pairwise comparison results
        ensemble_consensus: Consensus from ensemble if computed
        issues: List of disagreement issues
        duration_seconds: Time taken for validation
    """
    passed: bool
    agreement_level: str
    overall_agreement: float
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    ensemble_consensus: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "agreement_level": self.agreement_level,
            "overall_agreement": self.overall_agreement,
            "comparisons": self.comparisons,
            "ensemble_consensus": self.ensemble_consensus,
            "issues": self.issues,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SensorValidationResult:
    """
    Result from cross-sensor validation.

    Attributes:
        passed: Whether sensors agree sufficiently
        outcome: Overall validation outcome
        agreement_score: Agreement score (0-1)
        comparisons: Sensor pair comparison results
        physics_violations: Physics rule violations detected
        issues: List of disagreement issues
        duration_seconds: Time taken for validation
    """
    passed: bool
    outcome: str
    agreement_score: float
    comparisons: List[Dict[str, Any]] = field(default_factory=list)
    physics_violations: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "outcome": self.outcome,
            "agreement_score": self.agreement_score,
            "comparisons": self.comparisons,
            "physics_violations": self.physics_violations,
            "issues": self.issues,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class HistoricalValidationResult:
    """
    Result from historical baseline validation.

    Attributes:
        passed: Whether data is within historical norms
        is_anomalous: Whether anomaly was detected
        anomaly_score: Anomaly severity score
        anomalies: Detected anomalies
        baseline_comparison: Comparison with baseline statistics
        return_period: Estimated return period if applicable
        issues: List of historical issues
        duration_seconds: Time taken for validation
    """
    passed: bool
    is_anomalous: bool
    anomaly_score: float
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    baseline_comparison: Dict[str, Any] = field(default_factory=dict)
    return_period: Optional[float] = None
    issues: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": self.anomaly_score,
            "anomalies": self.anomalies,
            "baseline_comparison": self.baseline_comparison,
            "return_period": self.return_period,
            "issues": self.issues,
            "duration_seconds": self.duration_seconds,
        }


class ValidationRunner:
    """
    Executes validation operations for the Quality Agent.

    Provides methods for running various types of validation checks
    on analysis outputs, including sanity checks, cross-validation,
    and historical comparisons.
    """

    def __init__(self, config: Optional[ValidationRunnerConfig] = None):
        """
        Initialize ValidationRunner.

        Args:
            config: Runner configuration
        """
        self.config = config or ValidationRunnerConfig()
        self._logger = logging.getLogger(f"{__name__}.ValidationRunner")

        # Initialize validators
        self._sanity_suite = SanitySuite(self.config.sanity_config)
        self._model_validator = CrossModelValidator(self.config.model_config)
        self._sensor_validator = CrossSensorValidator(self.config.sensor_config)
        self._historical_validator = HistoricalValidator(self.config.historical_config)
        self._consensus_generator = ConsensusGenerator()

        # Individual checkers for targeted validation
        self._spatial_checker = SpatialCoherenceChecker()
        self._value_checker = ValuePlausibilityChecker()
        self._temporal_checker = TemporalConsistencyChecker()
        self._artifact_detector = ArtifactDetector()

        # Result cache
        self._cache: Dict[str, Any] = {}

        self._logger.info(f"ValidationRunner initialized with mode: {self.config.mode.value}")

    async def run_sanity_checks(
        self,
        data: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> SanityCheckResult:
        """
        Run sanity checks on data array.

        Args:
            data: Data array to validate
            context: Validation context with optional keys:
                - transform: Affine geotransform
                - mask: Boolean mask of valid pixels
                - timestamps: List of datetime objects
                - time_series_values: Time series values
                - tile_boundaries: Tile boundary indices
                - value_type: Type of values (confidence, extent, etc.)

        Returns:
            SanityCheckResult with check results
        """
        import time
        start_time = time.time()

        context = context or {}

        # Check cache
        cache_key = self._compute_cache_key("sanity", data, context)
        if self.config.cache_results and cache_key in self._cache:
            self._logger.debug(f"Returning cached sanity result for {cache_key[:8]}")
            return self._cache[cache_key]

        self._logger.info("Running sanity checks...")

        # Run sanity suite
        suite_result = self._sanity_suite.check(
            data=data,
            transform=context.get("transform"),
            mask=context.get("mask"),
            timestamps=context.get("timestamps"),
            time_series_values=context.get("time_series_values"),
            tile_boundaries=context.get("tile_boundaries"),
        )

        # Extract and format results
        issues = []

        spatial_result = None
        if suite_result.spatial:
            spatial_result = suite_result.spatial.to_dict()
            for issue in suite_result.spatial.issues:
                issues.append({
                    "type": "spatial",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                    "location": issue.location,
                })

        value_result = None
        if suite_result.values:
            value_result = suite_result.values.to_dict()
            for issue in suite_result.values.issues:
                issues.append({
                    "type": "value",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                })

        temporal_result = None
        if suite_result.temporal:
            temporal_result = suite_result.temporal.to_dict()
            for issue in suite_result.temporal.issues:
                issues.append({
                    "type": "temporal",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                })

        artifact_result = None
        if suite_result.artifacts:
            artifact_result = suite_result.artifacts.to_dict()
            for artifact in suite_result.artifacts.artifacts:
                issues.append({
                    "type": "artifact",
                    "artifact_type": artifact.artifact_type.value,
                    "severity": artifact.severity.value,
                    "message": artifact.description,
                    "location": artifact.location.to_dict() if artifact.location else None,
                })

        result = SanityCheckResult(
            passed=suite_result.passes_sanity,
            overall_score=suite_result.overall_score,
            spatial_result=spatial_result,
            value_result=value_result,
            temporal_result=temporal_result,
            artifact_result=artifact_result,
            issues=issues,
            duration_seconds=time.time() - start_time,
        )

        # Cache result
        if self.config.cache_results:
            self._cache[cache_key] = result

        self._logger.info(
            f"Sanity checks complete: passed={result.passed}, "
            f"score={result.overall_score:.3f}, issues={len(issues)}"
        )

        return result

    async def run_model_validation(
        self,
        model_outputs: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> ModelValidationResult:
        """
        Run cross-model validation on multiple model outputs.

        Args:
            model_outputs: List of model output dictionaries with keys:
                - model_id: Model identifier
                - data: Output array
                - confidence: Optional confidence array
                - metadata: Optional metadata
            context: Validation context

        Returns:
            ModelValidationResult with validation results
        """
        import time
        start_time = time.time()

        context = context or {}

        if len(model_outputs) < 2:
            return ModelValidationResult(
                passed=True,
                agreement_level="single_model",
                overall_agreement=1.0,
                issues=["Only one model output provided, skipping cross-model validation"],
                duration_seconds=time.time() - start_time,
            )

        self._logger.info(f"Running cross-model validation on {len(model_outputs)} models...")

        # Convert to ModelOutput objects
        outputs = []
        for i, mo in enumerate(model_outputs):
            data = mo.get("data")
            if isinstance(data, list):
                data = np.array(data)

            outputs.append(ModelOutput(
                model_id=mo.get("model_id", f"model_{i}"),
                data=data,
                confidence=mo.get("confidence"),
                metadata=mo.get("metadata", {}),
            ))

        # Run validation
        result = self._model_validator.validate(outputs)

        # Extract comparison results
        comparisons = []
        for comp in result.pairwise_comparisons:
            comparisons.append({
                "model_a": comp.model_a_id,
                "model_b": comp.model_b_id,
                "iou": comp.metrics.get("iou"),
                "correlation": comp.metrics.get("correlation"),
                "rmse": comp.metrics.get("rmse"),
                "agreement_score": comp.agreement_score,
            })

        # Determine if passed based on mode
        issues = []
        passed = True

        if result.agreement_level in (AgreementLevel.POOR, AgreementLevel.DISAGREEMENT):
            issues.append(f"Model disagreement detected: {result.agreement_level.value}")
            if self.config.mode == ValidationMode.STRICT:
                passed = False
            elif self.config.mode == ValidationMode.STANDARD:
                passed = result.agreement_level != AgreementLevel.DISAGREEMENT

        # Compute ensemble consensus if available
        ensemble_consensus = None
        if result.agreement_map is not None:
            try:
                from core.quality.validation import get_ensemble_consensus
                consensus_data = get_ensemble_consensus(outputs)
                ensemble_consensus = {
                    "mean": float(np.nanmean(consensus_data)) if consensus_data is not None else None,
                    "std": float(np.nanstd(consensus_data)) if consensus_data is not None else None,
                }
            except Exception as e:
                self._logger.warning(f"Failed to compute ensemble consensus: {e}")

        return ModelValidationResult(
            passed=passed,
            agreement_level=result.agreement_level.value,
            overall_agreement=result.overall_agreement,
            comparisons=comparisons,
            ensemble_consensus=ensemble_consensus,
            issues=issues,
            duration_seconds=time.time() - start_time,
        )

    async def run_sensor_validation(
        self,
        sensor_observations: List[Dict[str, Any]],
        observable: str = "water_extent",
        context: Optional[Dict[str, Any]] = None,
    ) -> SensorValidationResult:
        """
        Run cross-sensor validation on multiple sensor observations.

        Args:
            sensor_observations: List of sensor observation dictionaries with keys:
                - sensor_id: Sensor identifier
                - sensor_type: Type of sensor (optical, sar, etc.)
                - data: Observation array
                - acquisition_time: Acquisition timestamp
                - metadata: Optional metadata
            observable: What is being observed (water_extent, burn_area, etc.)
            context: Validation context

        Returns:
            SensorValidationResult with validation results
        """
        import time
        start_time = time.time()

        context = context or {}

        if len(sensor_observations) < 2:
            return SensorValidationResult(
                passed=True,
                outcome="single_sensor",
                agreement_score=1.0,
                issues=["Only one sensor observation provided, skipping cross-sensor validation"],
                duration_seconds=time.time() - start_time,
            )

        self._logger.info(f"Running cross-sensor validation on {len(sensor_observations)} sensors...")

        # Convert to SensorObservation objects
        observations = []
        for obs in sensor_observations:
            data = obs.get("data")
            if isinstance(data, list):
                data = np.array(data)

            # Map sensor type
            sensor_type_str = obs.get("sensor_type", "optical").lower()
            try:
                sensor_type = SensorType(sensor_type_str)
            except ValueError:
                sensor_type = SensorType.OPTICAL

            observations.append(SensorObservation(
                sensor_id=obs.get("sensor_id", "unknown"),
                sensor_type=sensor_type,
                data=data,
                acquisition_time=obs.get("acquisition_time"),
                metadata=obs.get("metadata", {}),
            ))

        # Run validation
        result = self._sensor_validator.validate(observations, observable=observable)

        # Extract comparison results
        comparisons = []
        for comp in result.comparisons:
            comparisons.append({
                "sensor_a": comp.sensor_a_id,
                "sensor_b": comp.sensor_b_id,
                "outcome": comp.outcome.value,
                "agreement_score": comp.agreement_score,
                "metrics": comp.metrics,
            })

        # Extract physics violations
        physics_violations = []
        for violation in result.physics_violations:
            physics_violations.append({
                "rule": violation.rule_name,
                "severity": violation.severity,
                "message": violation.message,
            })

        # Determine if passed
        issues = []
        passed = True

        if result.overall_outcome == ValidationOutcome.SIGNIFICANT_DISAGREEMENT:
            issues.append("Significant sensor disagreement detected")
            if self.config.mode in (ValidationMode.STRICT, ValidationMode.STANDARD):
                passed = False

        if physics_violations:
            issues.append(f"Physics violations detected: {len(physics_violations)}")
            if self.config.mode == ValidationMode.STRICT:
                passed = False

        return SensorValidationResult(
            passed=passed,
            outcome=result.overall_outcome.value,
            agreement_score=result.agreement_score,
            comparisons=comparisons,
            physics_violations=physics_violations,
            issues=issues,
            duration_seconds=time.time() - start_time,
        )

    async def run_historical_validation(
        self,
        current_data: np.ndarray,
        baseline: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> HistoricalValidationResult:
        """
        Validate current data against historical baseline.

        Args:
            current_data: Current data array to validate
            baseline: Historical baseline dictionary with keys:
                - mean: Historical mean
                - std: Historical standard deviation
                - min_value: Historical minimum
                - max_value: Historical maximum
                - percentiles: Dictionary of percentile values
            context: Validation context

        Returns:
            HistoricalValidationResult with validation results
        """
        import time
        start_time = time.time()

        context = context or {}

        self._logger.info("Running historical baseline validation...")

        # Create HistoricalBaseline object
        hist_baseline = HistoricalBaseline(
            mean=baseline.get("mean", 0.0),
            std=baseline.get("std", 1.0),
            min_value=baseline.get("min_value"),
            max_value=baseline.get("max_value"),
            percentiles=baseline.get("percentiles", {}),
        )

        # Run validation
        result = self._historical_validator.validate(current_data, hist_baseline)

        # Extract anomalies
        anomalies = []
        for anomaly in result.anomalies:
            anomalies.append({
                "type": anomaly.anomaly_type.value,
                "severity": anomaly.severity.value,
                "z_score": anomaly.z_score,
                "percentile": anomaly.percentile,
                "message": anomaly.message,
            })

        # Baseline comparison
        current_mean = float(np.nanmean(current_data))
        current_std = float(np.nanstd(current_data))
        current_min = float(np.nanmin(current_data))
        current_max = float(np.nanmax(current_data))

        baseline_comparison = {
            "current_mean": current_mean,
            "baseline_mean": hist_baseline.mean,
            "mean_difference": current_mean - hist_baseline.mean,
            "mean_z_score": (current_mean - hist_baseline.mean) / (hist_baseline.std + 1e-10),
            "current_std": current_std,
            "baseline_std": hist_baseline.std,
            "current_range": (current_min, current_max),
            "baseline_range": (hist_baseline.min_value, hist_baseline.max_value),
        }

        # Estimate return period if applicable
        return_period = None
        if result.return_period_years is not None:
            return_period = result.return_period_years

        # Determine if passed
        issues = []
        passed = True

        if result.is_anomalous:
            issues.append(f"Historical anomaly detected with score {result.anomaly_score:.2f}")
            if self.config.mode == ValidationMode.STRICT:
                passed = False
            elif self.config.mode == ValidationMode.STANDARD:
                # Only fail for severe anomalies
                passed = result.anomaly_score < 3.0

        return HistoricalValidationResult(
            passed=passed,
            is_anomalous=result.is_anomalous,
            anomaly_score=result.anomaly_score,
            anomalies=anomalies,
            baseline_comparison=baseline_comparison,
            return_period=return_period,
            issues=issues,
            duration_seconds=time.time() - start_time,
        )

    async def run_all_validations(
        self,
        data: np.ndarray,
        model_outputs: Optional[List[Dict[str, Any]]] = None,
        sensor_observations: Optional[List[Dict[str, Any]]] = None,
        historical_baseline: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run all applicable validations in parallel.

        Args:
            data: Primary data array
            model_outputs: Optional model outputs for cross-model validation
            sensor_observations: Optional sensor observations for cross-sensor validation
            historical_baseline: Optional historical baseline for comparison
            context: Validation context

        Returns:
            Dictionary with all validation results
        """
        import time
        start_time = time.time()

        context = context or {}
        results = {}

        if self.config.parallel_execution:
            # Run validations in parallel
            tasks = []

            # Always run sanity checks
            tasks.append(("sanity", self.run_sanity_checks(data, context)))

            if model_outputs and len(model_outputs) >= 2:
                tasks.append(("model", self.run_model_validation(model_outputs, context)))

            if sensor_observations and len(sensor_observations) >= 2:
                observable = context.get("observable", "water_extent")
                tasks.append(("sensor", self.run_sensor_validation(sensor_observations, observable, context)))

            if historical_baseline:
                tasks.append(("historical", self.run_historical_validation(data, historical_baseline, context)))

            # Execute in parallel
            task_results = await asyncio.gather(*[t[1] for t in tasks], return_exceptions=True)

            for (name, _), result in zip(tasks, task_results):
                if isinstance(result, Exception):
                    self._logger.error(f"{name} validation failed: {result}")
                    results[name] = {"error": str(result)}
                else:
                    results[name] = result.to_dict()

        else:
            # Run sequentially
            results["sanity"] = (await self.run_sanity_checks(data, context)).to_dict()

            if model_outputs and len(model_outputs) >= 2:
                results["model"] = (await self.run_model_validation(model_outputs, context)).to_dict()

            if sensor_observations and len(sensor_observations) >= 2:
                observable = context.get("observable", "water_extent")
                results["sensor"] = (await self.run_sensor_validation(sensor_observations, observable, context)).to_dict()

            if historical_baseline:
                results["historical"] = (await self.run_historical_validation(data, historical_baseline, context)).to_dict()

        results["total_duration_seconds"] = time.time() - start_time

        return results

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self._cache.clear()
        self._logger.debug("Validation cache cleared")

    def _compute_cache_key(
        self,
        validation_type: str,
        data: np.ndarray,
        context: Dict[str, Any],
    ) -> str:
        """Compute cache key for validation result."""
        import hashlib

        # Create hash from data and context
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()[:16]
        context_str = str(sorted(context.items()))
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:8]

        return f"{validation_type}_{data_hash}_{context_hash}"
