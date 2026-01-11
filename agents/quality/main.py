"""
Quality Agent - Orchestrates all QC operations.

The Quality Agent is responsible for:
- Validating analysis step outputs
- Running sanity checks on data
- Coordinating cross-validation between models and sensors
- Quantifying uncertainty
- Making pass/fail/review gating decisions
- Coordinating with human reviewers when needed
- Reporting quality status to other agents

This agent integrates the core quality control modules and provides
a high-level interface for the pipeline and orchestrator agents.

Example:
    config = AgentConfig(
        agent_id="quality_001",
        agent_type="quality",
        name="QualityAgent",
    )
    agent = QualityAgent(config)
    await agent.initialize()
    await agent.start()

    # Validate a step result
    decision = await agent.validate_step("step_001", result_data)
    if decision.status == GateDecision.REVIEW_REQUIRED:
        # Handle review
        ...
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.base import AgentMessage, AgentState, AgentType, BaseAgent, MessageType, RetryPolicy

# Import core quality modules
from core.quality.sanity import (
    SanitySuite,
    SanitySuiteConfig,
    SanitySuiteResult,
    SpatialCoherenceChecker,
    TemporalConsistencyChecker,
    ValuePlausibilityChecker,
    ArtifactDetector,
)
from core.quality.validation import (
    CrossModelValidator,
    CrossModelConfig,
    CrossModelResult,
    CrossSensorValidator,
    CrossSensorConfig,
    CrossSensorResult,
    HistoricalValidator,
    HistoricalConfig,
    HistoricalResult,
    ConsensusGenerator,
    ConsensusConfig,
    ConsensusResult,
)
from core.quality.uncertainty import (
    UncertaintyQuantifier,
    QuantificationConfig,
    UncertaintyMetrics,
    SpatialUncertaintyMapper,
    SpatialUncertaintyConfig,
    UncertaintySurface,
    QualityErrorPropagator,
    PropagationConfig,
    PropagationResult,
)
from core.quality.actions import (
    QualityGate,
    GatingThresholds,
    GatingContext,
    GatingDecision,
    GateStatus,
    QCCheck,
    CheckCategory,
    CheckStatus,
    QualityFlagger,
    FlagSummary,
    StandardFlag,
    ReviewRouter,
    ReviewRequest,
    ReviewPriority,
    ReviewType,
    ExpertDomain,
    ReviewContext,
)
from core.quality.reporting import (
    QAReportGenerator,
    QAReport,
    ReportFormat,
    ReportLevel,
)


logger = logging.getLogger(__name__)


class QAGateDecision(Enum):
    """High-level QA gate decisions."""
    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    REVIEW_REQUIRED = "review_required"
    FAIL = "fail"


@dataclass
class SanityResult:
    """Result from sanity checks."""
    passed: bool
    overall_score: float
    issues: List[Dict[str, Any]] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "duration_seconds": self.duration_seconds,
            "details": self.details,
        }


@dataclass
class ValidationResult:
    """Result from cross-validation."""
    passed: bool
    agreement_score: float
    model_validation: Optional[Dict[str, Any]] = None
    sensor_validation: Optional[Dict[str, Any]] = None
    historical_validation: Optional[Dict[str, Any]] = None
    consensus: Optional[Dict[str, Any]] = None
    issues: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "agreement_score": self.agreement_score,
            "model_validation": self.model_validation,
            "sensor_validation": self.sensor_validation,
            "historical_validation": self.historical_validation,
            "consensus": self.consensus,
            "issues": self.issues,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class UncertaintyMap:
    """Result from uncertainty quantification."""
    mean_uncertainty: float
    max_uncertainty: float
    spatial_uncertainty: Optional[np.ndarray] = None
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    hotspot_count: int = 0
    hotspot_locations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_uncertainty": self.mean_uncertainty,
            "max_uncertainty": self.max_uncertainty,
            "has_spatial_map": self.spatial_uncertainty is not None,
            "confidence_interval": self.confidence_interval,
            "hotspot_count": self.hotspot_count,
            "hotspot_locations": self.hotspot_locations,
            "metrics": self.metrics,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class GateDecision:
    """Decision from quality gating."""
    status: QAGateDecision
    confidence: float
    can_release: bool
    requires_review: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    flags_applied: List[str] = field(default_factory=list)
    review_request: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "can_release": self.can_release,
            "requires_review": self.requires_review,
            "reasons": self.reasons,
            "warnings": self.warnings,
            "flags_applied": self.flags_applied,
            "review_request": self.review_request,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class QAReport:
    """Quality assurance report."""
    event_id: str
    product_id: str
    overall_status: QAGateDecision
    confidence_score: float
    sanity_result: Optional[SanityResult] = None
    validation_result: Optional[ValidationResult] = None
    uncertainty_result: Optional[UncertaintyMap] = None
    gate_decision: Optional[GateDecision] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    format: ReportFormat = ReportFormat.JSON
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "product_id": self.product_id,
            "overall_status": self.overall_status.value,
            "confidence_score": self.confidence_score,
            "sanity_result": self.sanity_result.to_dict() if self.sanity_result else None,
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "uncertainty_result": self.uncertainty_result.to_dict() if self.uncertainty_result else None,
            "gate_decision": self.gate_decision.to_dict() if self.gate_decision else None,
            "generated_at": self.generated_at.isoformat(),
            "format": self.format.value,
            "report_path": self.report_path,
        }


@dataclass
class QualityAgentConfig:
    """Configuration specific to Quality Agent."""
    # Sanity check config
    run_spatial_checks: bool = True
    run_value_checks: bool = True
    run_temporal_checks: bool = True
    run_artifact_detection: bool = True

    # Validation config
    run_model_validation: bool = True
    run_sensor_validation: bool = True
    run_historical_validation: bool = True
    generate_consensus: bool = True

    # Uncertainty config
    run_uncertainty_quantification: bool = True
    run_spatial_uncertainty: bool = True

    # Gating config
    gate_mode: str = "operational"  # emergency, operational, research
    min_confidence_threshold: float = 0.6
    auto_review_threshold: float = 0.75

    # Reporting config
    default_report_format: ReportFormat = ReportFormat.JSON
    default_report_level: ReportLevel = ReportLevel.STANDARD
    report_output_path: Optional[str] = None


class QualityAgent(BaseAgent):
    """
    Quality Agent for orchestrating QC operations.

    Coordinates all quality control checks and makes pass/fail/review
    decisions for analysis outputs.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        quality_config: Optional[QualityAgentConfig] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        """
        Initialize Quality Agent.

        Args:
            agent_id: Unique agent identifier
            quality_config: Quality-specific configuration
            retry_policy: Retry configuration
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.QUALITY,
            retry_policy=retry_policy,
        )

        self.quality_config = quality_config or QualityAgentConfig()

        # Core components (initialized in initialize())
        self._sanity_suite: Optional[SanitySuite] = None
        self._model_validator: Optional[CrossModelValidator] = None
        self._sensor_validator: Optional[CrossSensorValidator] = None
        self._historical_validator: Optional[HistoricalValidator] = None
        self._consensus_generator: Optional[ConsensusGenerator] = None
        self._uncertainty_quantifier: Optional[UncertaintyQuantifier] = None
        self._spatial_mapper: Optional[SpatialUncertaintyMapper] = None
        self._error_propagator: Optional[QualityErrorPropagator] = None
        self._quality_gate: Optional[QualityGate] = None
        self._flagger: Optional[QualityFlagger] = None
        self._review_router: Optional[ReviewRouter] = None
        self._report_generator: Optional[QAReportGenerator] = None

        # State tracking
        self._validation_cache: Dict[str, Any] = {}
        self._active_reviews: Dict[str, ReviewRequest] = {}

        # Logger
        self._logger = logging.getLogger(f"{__name__}.{self.agent_id}")
        self._logger.info(f"QualityAgent created with config: {self.quality_config}")

    async def initialize(self) -> None:
        """Initialize quality control components."""
        self._logger.info("Initializing quality control components...")

        # Initialize sanity suite
        sanity_config = SanitySuiteConfig(
            run_spatial=self.quality_config.run_spatial_checks,
            run_values=self.quality_config.run_value_checks,
            run_temporal=self.quality_config.run_temporal_checks,
            run_artifacts=self.quality_config.run_artifact_detection,
        )
        self._sanity_suite = SanitySuite(sanity_config)

        # Initialize validators
        self._model_validator = CrossModelValidator()
        self._sensor_validator = CrossSensorValidator()
        self._historical_validator = HistoricalValidator()
        self._consensus_generator = ConsensusGenerator()

        # Initialize uncertainty components
        self._uncertainty_quantifier = UncertaintyQuantifier()
        self._spatial_mapper = SpatialUncertaintyMapper()
        self._error_propagator = QualityErrorPropagator()

        # Initialize gating and flagging
        if self.quality_config.gate_mode == "emergency":
            from core.quality.actions.gating import create_emergency_gate
            self._quality_gate = create_emergency_gate()
        elif self.quality_config.gate_mode == "research":
            from core.quality.actions.gating import create_research_gate
            self._quality_gate = create_research_gate()
        else:
            from core.quality.actions.gating import create_operational_gate
            self._quality_gate = create_operational_gate()

        self._flagger = QualityFlagger()
        self._review_router = ReviewRouter()

        # Initialize report generator
        self._report_generator = QAReportGenerator()

        # Message handlers mapping
        self._message_handlers = {
            "validate_step": self._handle_validate_step,
            "validate_final": self._handle_validate_final,
            "run_sanity": self._handle_run_sanity,
            "run_validation": self._handle_run_validation,
            "quantify_uncertainty": self._handle_quantify_uncertainty,
            "gate_decision": self._handle_gate_decision,
            "review_completed": self._handle_review_completed,
            "get_status": self._handle_get_status,
        }

        self._logger.info("Quality control components initialized")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        Args:
            message: Incoming message to process

        Returns:
            Optional response message
        """
        # Get command from payload
        command = message.payload.get("command", "")
        handler = self._message_handlers.get(command)

        if handler:
            try:
                response_payload = await handler(message)
                if response_payload:
                    return AgentMessage(
                        correlation_id=message.message_id,
                        from_agent=AgentType.QUALITY,
                        to_agent=message.from_agent,
                        message_type=MessageType.RESPONSE,
                        payload=response_payload,
                    )
            except Exception as e:
                self._logger.error(f"Error processing message {message.message_id}: {e}")
                return AgentMessage(
                    correlation_id=message.message_id,
                    from_agent=AgentType.QUALITY,
                    to_agent=message.from_agent,
                    message_type=MessageType.ERROR,
                    payload={"error": str(e)},
                )
        else:
            self._logger.warning(f"No handler for command: {command}")

        return None

    async def run(self) -> None:
        """
        Main agent execution loop.

        Processes messages from inbox until shutdown is requested.
        """
        self._logger.info("Quality Agent run loop started")

        while not self._shutdown_event.is_set():
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._inbox.get(),
                    timeout=1.0
                )
                response = await self.process_message(message)
                if response:
                    await self.send_message(response)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self._logger.error(f"Error in run loop: {e}")
                await self.on_error(e)

        self._logger.info("Quality Agent run loop ended")

    async def on_start(self) -> None:
        """Called when agent starts."""
        self._logger.info("Quality Agent started")

    async def on_error(self, error: Exception) -> None:
        """Called on error."""
        self._logger.error(f"Quality Agent error: {error}")

    # Public API methods

    async def validate_step(
        self,
        step_id: str,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """
        Validate a single pipeline step result.

        Args:
            step_id: Identifier of the pipeline step
            result: Step result data (must include 'data' array)
            context: Optional context (event_id, product_id, etc.)

        Returns:
            GateDecision with pass/fail/review status
        """
        import time
        start_time = time.time()

        context = context or {}
        self._logger.info(f"Validating step {step_id}")

        # Extract data array from result
        data = result.get("data")
        if data is None:
            return GateDecision(
                status=QAGateDecision.FAIL,
                confidence=0.0,
                can_release=False,
                requires_review=False,
                reasons=["No data provided for validation"],
                duration_seconds=time.time() - start_time,
            )

        if isinstance(data, list):
            data = np.array(data)

        # Run sanity checks
        sanity_result = await self.run_sanity_checks(data, context)

        # Build QC checks for gating
        qc_checks = self._build_qc_checks_from_sanity(sanity_result)

        # Run gating
        gating_context = GatingContext(
            event_id=context.get("event_id", "unknown"),
            product_id=context.get("product_id", "unknown"),
            confidence_threshold=self.quality_config.min_confidence_threshold,
        )

        gating_decision = self._quality_gate.evaluate(qc_checks, gating_context)

        # Convert to GateDecision
        decision = self._convert_gating_decision(gating_decision, sanity_result)
        decision.duration_seconds = time.time() - start_time

        self._logger.info(
            f"Step {step_id} validation complete: {decision.status.value}, "
            f"confidence={decision.confidence:.3f}"
        )

        return decision

    async def validate_final(
        self,
        results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> QAReport:
        """
        Perform final validation on complete analysis results.

        Args:
            results: Complete analysis results
            context: Validation context (event_id, product_id, etc.)

        Returns:
            Complete QA report
        """
        import time
        start_time = time.time()

        context = context or {}
        event_id = context.get("event_id", "unknown")
        product_id = context.get("product_id", "unknown")

        self._logger.info(f"Running final validation for {event_id}/{product_id}")

        # Extract primary result data
        data = results.get("data")
        if isinstance(data, list):
            data = np.array(data)

        # Run all validation steps
        sanity_result = None
        validation_result = None
        uncertainty_result = None

        # 1. Sanity checks
        if data is not None:
            sanity_result = await self.run_sanity_checks(data, context)

        # 2. Cross-validation
        model_outputs = results.get("model_outputs", [])
        sensor_observations = results.get("sensor_observations", [])
        historical_data = results.get("historical_baseline")

        if model_outputs or sensor_observations or historical_data:
            validation_result = await self.run_cross_validation(
                results=model_outputs,
                sensor_observations=sensor_observations,
                historical_baseline=historical_data,
                context=context,
            )

        # 3. Uncertainty quantification
        if data is not None:
            uncertainty_result = await self.quantify_uncertainty(
                {"data": data, "confidence": results.get("confidence")},
                context,
            )

        # 4. Gating decision
        qc_checks = []
        if sanity_result:
            qc_checks.extend(self._build_qc_checks_from_sanity(sanity_result))
        if validation_result:
            qc_checks.extend(self._build_qc_checks_from_validation(validation_result))
        if uncertainty_result:
            qc_checks.extend(self._build_qc_checks_from_uncertainty(uncertainty_result))

        gating_context = GatingContext(
            event_id=event_id,
            product_id=product_id,
            confidence_threshold=self.quality_config.min_confidence_threshold,
        )

        gating_decision = self._quality_gate.evaluate(qc_checks, gating_context)
        gate_decision = self._convert_gating_decision(
            gating_decision,
            sanity_result,
            validation_result,
            uncertainty_result,
        )

        # Determine overall status
        overall_status = QAGateDecision(gate_decision.status.value)
        confidence_score = gate_decision.confidence

        # Create QA report
        report = QAReport(
            event_id=event_id,
            product_id=product_id,
            overall_status=overall_status,
            confidence_score=confidence_score,
            sanity_result=sanity_result,
            validation_result=validation_result,
            uncertainty_result=uncertainty_result,
            gate_decision=gate_decision,
            format=self.quality_config.default_report_format,
        )

        self._logger.info(
            f"Final validation complete: {overall_status.value}, "
            f"confidence={confidence_score:.3f}, "
            f"duration={time.time() - start_time:.2f}s"
        )

        return report

    async def run_sanity_checks(
        self,
        data: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> SanityResult:
        """
        Run sanity checks on data array.

        Args:
            data: Data array to check
            context: Check context (timestamps, transform, etc.)

        Returns:
            SanityResult with check results
        """
        import time
        start_time = time.time()

        context = context or {}

        # Extract context parameters
        transform = context.get("transform")
        mask = context.get("mask")
        timestamps = context.get("timestamps")
        time_series_values = context.get("time_series_values")
        tile_boundaries = context.get("tile_boundaries")

        # Run sanity suite
        suite_result = self._sanity_suite.check(
            data=data,
            transform=transform,
            mask=mask,
            timestamps=timestamps,
            time_series_values=time_series_values,
            tile_boundaries=tile_boundaries,
        )

        # Convert to SanityResult
        issues = []
        critical_count = 0
        warning_count = 0

        if suite_result.spatial:
            for issue in suite_result.spatial.issues:
                issues.append({
                    "type": "spatial",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                })
                if issue.severity.value == "critical":
                    critical_count += 1
                elif issue.severity.value == "warning":
                    warning_count += 1

        if suite_result.values:
            for issue in suite_result.values.issues:
                issues.append({
                    "type": "value",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                })
                if issue.severity.value == "critical":
                    critical_count += 1
                elif issue.severity.value == "warning":
                    warning_count += 1

        if suite_result.temporal:
            for issue in suite_result.temporal.issues:
                issues.append({
                    "type": "temporal",
                    "check": issue.check_type.value,
                    "severity": issue.severity.value,
                    "message": issue.message,
                })
                if issue.severity.value == "critical":
                    critical_count += 1
                elif issue.severity.value == "warning":
                    warning_count += 1

        if suite_result.artifacts:
            for artifact in suite_result.artifacts.artifacts:
                issues.append({
                    "type": "artifact",
                    "artifact_type": artifact.artifact_type.value,
                    "severity": artifact.severity.value,
                    "message": artifact.description,
                })
                if artifact.severity.value == "critical":
                    critical_count += 1
                elif artifact.severity.value in ("high", "warning"):
                    warning_count += 1

        return SanityResult(
            passed=suite_result.passes_sanity,
            overall_score=suite_result.overall_score,
            issues=issues,
            critical_count=critical_count,
            warning_count=warning_count,
            duration_seconds=time.time() - start_time,
            details={
                "spatial": suite_result.spatial.to_dict() if suite_result.spatial else None,
                "values": suite_result.values.to_dict() if suite_result.values else None,
                "temporal": suite_result.temporal.to_dict() if suite_result.temporal else None,
                "artifacts": suite_result.artifacts.to_dict() if suite_result.artifacts else None,
            },
        )

    async def run_cross_validation(
        self,
        results: List[Dict[str, Any]],
        sensor_observations: Optional[List[Dict[str, Any]]] = None,
        historical_baseline: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Run cross-validation on multiple results.

        Args:
            results: List of model outputs to compare
            sensor_observations: Optional sensor observations for cross-sensor validation
            historical_baseline: Optional historical baseline for comparison
            context: Validation context

        Returns:
            ValidationResult with validation results
        """
        import time
        start_time = time.time()

        context = context or {}
        issues = []

        model_validation = None
        sensor_validation = None
        historical_validation = None
        consensus = None

        # Cross-model validation
        if self.quality_config.run_model_validation and len(results) >= 2:
            try:
                from core.quality.validation import ModelOutput
                model_outputs = []
                for i, r in enumerate(results):
                    model_outputs.append(ModelOutput(
                        model_id=r.get("model_id", f"model_{i}"),
                        data=np.array(r["data"]) if isinstance(r.get("data"), list) else r.get("data"),
                        confidence=r.get("confidence"),
                        metadata=r.get("metadata", {}),
                    ))

                result = self._model_validator.validate(model_outputs)
                model_validation = {
                    "agreement_level": result.agreement_level.value,
                    "overall_agreement": result.overall_agreement,
                    "pairwise_comparisons": len(result.pairwise_comparisons),
                }
                if result.agreement_level.value in ("poor", "disagreement"):
                    issues.append(f"Model disagreement detected: {result.agreement_level.value}")
            except Exception as e:
                self._logger.warning(f"Model validation failed: {e}")
                issues.append(f"Model validation error: {str(e)}")

        # Cross-sensor validation
        if self.quality_config.run_sensor_validation and sensor_observations:
            try:
                from core.quality.validation import SensorObservation
                observations = []
                for obs in sensor_observations:
                    observations.append(SensorObservation(
                        sensor_id=obs.get("sensor_id", "unknown"),
                        sensor_type=obs.get("sensor_type", "optical"),
                        data=np.array(obs["data"]) if isinstance(obs.get("data"), list) else obs.get("data"),
                        acquisition_time=obs.get("acquisition_time"),
                        metadata=obs.get("metadata", {}),
                    ))

                result = self._sensor_validator.validate(
                    observations,
                    observable=context.get("observable", "water_extent"),
                )
                sensor_validation = {
                    "overall_outcome": result.overall_outcome.value,
                    "agreement_score": result.agreement_score,
                    "comparisons": len(result.comparisons),
                }
                if result.overall_outcome.value == "significant_disagreement":
                    issues.append("Significant sensor disagreement detected")
            except Exception as e:
                self._logger.warning(f"Sensor validation failed: {e}")
                issues.append(f"Sensor validation error: {str(e)}")

        # Historical validation
        if self.quality_config.run_historical_validation and historical_baseline:
            try:
                from core.quality.validation import HistoricalBaseline
                baseline = HistoricalBaseline(
                    mean=historical_baseline.get("mean", 0),
                    std=historical_baseline.get("std", 1),
                    min_value=historical_baseline.get("min"),
                    max_value=historical_baseline.get("max"),
                    percentiles=historical_baseline.get("percentiles", {}),
                )
                current_data = results[0].get("data") if results else None
                if current_data is not None:
                    if isinstance(current_data, list):
                        current_data = np.array(current_data)
                    result = self._historical_validator.validate(current_data, baseline)
                    historical_validation = {
                        "is_anomalous": result.is_anomalous,
                        "anomaly_score": result.anomaly_score,
                        "anomalies_found": len(result.anomalies),
                    }
                    if result.is_anomalous:
                        issues.append(f"Historical anomaly detected, score={result.anomaly_score:.2f}")
            except Exception as e:
                self._logger.warning(f"Historical validation failed: {e}")
                issues.append(f"Historical validation error: {str(e)}")

        # Generate consensus
        if self.quality_config.generate_consensus and len(results) >= 2:
            try:
                from core.quality.validation import ConsensusSource
                sources = []
                for i, r in enumerate(results):
                    sources.append(ConsensusSource(
                        source_id=r.get("source_id", f"source_{i}"),
                        data=np.array(r["data"]) if isinstance(r.get("data"), list) else r.get("data"),
                        weight=r.get("weight", 1.0),
                        confidence=r.get("confidence"),
                    ))

                result = self._consensus_generator.generate(sources)
                consensus = {
                    "quality": result.quality.value,
                    "confidence_score": result.confidence_score,
                    "disagreement_regions": len(result.disagreement_regions),
                }
            except Exception as e:
                self._logger.warning(f"Consensus generation failed: {e}")
                issues.append(f"Consensus generation error: {str(e)}")

        # Calculate overall agreement score
        scores = []
        if model_validation:
            scores.append(model_validation.get("overall_agreement", 0.5))
        if sensor_validation:
            scores.append(sensor_validation.get("agreement_score", 0.5))
        if consensus:
            scores.append(consensus.get("confidence_score", 0.5))

        agreement_score = sum(scores) / len(scores) if scores else 1.0
        passed = len(issues) == 0 and agreement_score >= 0.6

        return ValidationResult(
            passed=passed,
            agreement_score=agreement_score,
            model_validation=model_validation,
            sensor_validation=sensor_validation,
            historical_validation=historical_validation,
            consensus=consensus,
            issues=issues,
            duration_seconds=time.time() - start_time,
        )

    async def quantify_uncertainty(
        self,
        result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> UncertaintyMap:
        """
        Quantify uncertainty in result.

        Args:
            result: Analysis result with data and optional confidence
            context: Uncertainty context

        Returns:
            UncertaintyMap with uncertainty metrics
        """
        import time
        start_time = time.time()

        context = context or {}

        data = result.get("data")
        if isinstance(data, list):
            data = np.array(data)

        confidence = result.get("confidence")
        if isinstance(confidence, list):
            confidence = np.array(confidence)

        metrics = {}
        spatial_uncertainty = None
        hotspots = []

        # Basic uncertainty metrics
        if confidence is not None:
            metrics["mean_confidence"] = float(np.nanmean(confidence))
            metrics["min_confidence"] = float(np.nanmin(confidence))
            metrics["max_confidence"] = float(np.nanmax(confidence))
            metrics["std_confidence"] = float(np.nanstd(confidence))

            # Uncertainty is 1 - confidence
            uncertainty = 1.0 - confidence
            mean_uncertainty = float(np.nanmean(uncertainty))
            max_uncertainty = float(np.nanmax(uncertainty))
        else:
            # Estimate uncertainty from data variance
            if data is not None and data.size > 0:
                data_std = float(np.nanstd(data))
                data_range = float(np.nanmax(data) - np.nanmin(data))
                # Coefficient of variation as proxy for uncertainty
                cv = data_std / (abs(float(np.nanmean(data))) + 1e-10)
                mean_uncertainty = min(1.0, cv)
                max_uncertainty = min(1.0, cv * 2)
                metrics["coefficient_of_variation"] = cv
            else:
                mean_uncertainty = 0.5
                max_uncertainty = 1.0

        # Spatial uncertainty mapping
        if self.quality_config.run_spatial_uncertainty and data is not None and data.ndim >= 2:
            try:
                surface = self._spatial_mapper.compute_uncertainty_surface(
                    data,
                    method="local_variance",
                )
                spatial_uncertainty = surface.uncertainty

                # Detect hotspots
                from core.quality.uncertainty import detect_uncertainty_hotspots
                hotspot_result = detect_uncertainty_hotspots(surface.uncertainty)
                hotspots = [
                    {
                        "location": h.centroid,
                        "mean_uncertainty": h.mean_uncertainty,
                        "area_pixels": h.area_pixels,
                    }
                    for h in hotspot_result.hotspots
                ]
                metrics["hotspot_count"] = len(hotspots)
            except Exception as e:
                self._logger.warning(f"Spatial uncertainty mapping failed: {e}")

        # Calculate confidence interval
        confidence_interval = (
            max(0.0, mean_uncertainty - 0.1),
            min(1.0, mean_uncertainty + 0.1),
        )

        return UncertaintyMap(
            mean_uncertainty=mean_uncertainty,
            max_uncertainty=max_uncertainty,
            spatial_uncertainty=spatial_uncertainty,
            confidence_interval=confidence_interval,
            hotspot_count=len(hotspots),
            hotspot_locations=hotspots,
            metrics=metrics,
            duration_seconds=time.time() - start_time,
        )

    async def decide_gate(
        self,
        qa_results: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> GateDecision:
        """
        Make gating decision based on QA results.

        Args:
            qa_results: Dictionary with sanity, validation, uncertainty results
            context: Gating context

        Returns:
            GateDecision with pass/fail/review status
        """
        import time
        start_time = time.time()

        context = context or {}
        qc_checks = []

        # Convert results to QC checks
        if "sanity" in qa_results:
            qc_checks.extend(self._build_qc_checks_from_sanity(qa_results["sanity"]))
        if "validation" in qa_results:
            qc_checks.extend(self._build_qc_checks_from_validation(qa_results["validation"]))
        if "uncertainty" in qa_results:
            qc_checks.extend(self._build_qc_checks_from_uncertainty(qa_results["uncertainty"]))

        # Run gating
        gating_context = GatingContext(
            event_id=context.get("event_id", "unknown"),
            product_id=context.get("product_id", "unknown"),
            confidence_threshold=self.quality_config.min_confidence_threshold,
        )

        gating_decision = self._quality_gate.evaluate(qc_checks, gating_context)
        decision = self._convert_gating_decision(gating_decision)
        decision.duration_seconds = time.time() - start_time

        return decision

    # Message handlers

    async def _handle_validate_step(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle validate_step request."""
        payload = message.payload
        step_id = payload.get("step_id", "unknown")
        result = payload.get("result", {})
        context = payload.get("context", {})

        decision = await self.validate_step(step_id, result, context)
        return {"decision": decision.to_dict()}

    async def _handle_validate_final(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle validate_final request."""
        payload = message.payload
        results = payload.get("results", {})
        context = payload.get("context", {})

        report = await self.validate_final(results, context)
        return {"report": report.to_dict()}

    async def _handle_run_sanity(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle run_sanity request."""
        payload = message.payload
        data = payload.get("data")
        if isinstance(data, list):
            data = np.array(data)
        context = payload.get("context", {})

        result = await self.run_sanity_checks(data, context)
        return {"result": result.to_dict()}

    async def _handle_run_validation(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle run_validation request."""
        payload = message.payload
        results = payload.get("results", [])
        sensor_observations = payload.get("sensor_observations")
        historical_baseline = payload.get("historical_baseline")
        context = payload.get("context", {})

        result = await self.run_cross_validation(
            results, sensor_observations, historical_baseline, context
        )
        return {"result": result.to_dict()}

    async def _handle_quantify_uncertainty(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle quantify_uncertainty request."""
        payload = message.payload
        result = payload.get("result", {})
        context = payload.get("context", {})

        uncertainty = await self.quantify_uncertainty(result, context)
        return {"result": uncertainty.to_dict()}

    async def _handle_gate_decision(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle gate_decision request."""
        payload = message.payload
        qa_results = payload.get("qa_results", {})
        context = payload.get("context", {})

        decision = await self.decide_gate(qa_results, context)
        return {"decision": decision.to_dict()}

    async def _handle_review_completed(self, message: AgentMessage) -> Optional[Dict[str, Any]]:
        """Handle review_completed notification."""
        payload = message.payload
        review_id = payload.get("review_id")
        outcome = payload.get("outcome")

        if review_id in self._active_reviews:
            del self._active_reviews[review_id]
            self._logger.info(f"Review {review_id} completed with outcome: {outcome}")
        return None

    async def _handle_get_status(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle get_status request."""
        status = self.get_status()
        status["active_reviews"] = len(self._active_reviews)
        status["cached_validations"] = len(self._validation_cache)
        return {"status": status}

    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "state": self.state.value,
            "is_running": self.is_running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }

    # Helper methods

    def _build_qc_checks_from_sanity(self, sanity_result: SanityResult) -> List[QCCheck]:
        """Build QC checks from sanity result."""
        checks = []

        # Overall sanity check
        checks.append(QCCheck(
            check_name="sanity_overall",
            category=CheckCategory.SPATIAL,
            status=CheckStatus.PASS if sanity_result.passed else CheckStatus.HARD_FAIL,
            metric_value=sanity_result.overall_score,
            threshold=0.6,
            details=f"Sanity score: {sanity_result.overall_score:.3f}",
        ))

        # Add individual issue checks
        for issue in sanity_result.issues:
            severity = issue.get("severity", "warning")
            if severity == "critical":
                status = CheckStatus.HARD_FAIL
            elif severity in ("high", "warning"):
                status = CheckStatus.WARNING
            else:
                status = CheckStatus.SOFT_FAIL

            category = CheckCategory.SPATIAL
            if issue.get("type") == "value":
                category = CheckCategory.VALUE
            elif issue.get("type") == "temporal":
                category = CheckCategory.TEMPORAL
            elif issue.get("type") == "artifact":
                category = CheckCategory.ARTIFACT

            checks.append(QCCheck(
                check_name=f"sanity_{issue.get('type', 'unknown')}_{issue.get('check', 'check')}",
                category=category,
                status=status,
                details=issue.get("message", ""),
            ))

        return checks

    def _build_qc_checks_from_validation(self, validation_result: ValidationResult) -> List[QCCheck]:
        """Build QC checks from validation result."""
        checks = []

        # Overall validation check
        checks.append(QCCheck(
            check_name="validation_overall",
            category=CheckCategory.CROSS_VALIDATION,
            status=CheckStatus.PASS if validation_result.passed else CheckStatus.SOFT_FAIL,
            metric_value=validation_result.agreement_score,
            threshold=0.6,
            details=f"Agreement score: {validation_result.agreement_score:.3f}",
        ))

        # Model validation check
        if validation_result.model_validation:
            agreement = validation_result.model_validation.get("overall_agreement", 0.5)
            checks.append(QCCheck(
                check_name="validation_cross_model",
                category=CheckCategory.CROSS_VALIDATION,
                status=CheckStatus.PASS if agreement >= 0.6 else CheckStatus.WARNING,
                metric_value=agreement,
                threshold=0.6,
                details=f"Model agreement: {validation_result.model_validation.get('agreement_level', 'unknown')}",
            ))

        # Historical validation check
        if validation_result.historical_validation:
            is_anomalous = validation_result.historical_validation.get("is_anomalous", False)
            checks.append(QCCheck(
                check_name="validation_historical",
                category=CheckCategory.HISTORICAL,
                status=CheckStatus.WARNING if is_anomalous else CheckStatus.PASS,
                metric_value=validation_result.historical_validation.get("anomaly_score", 0),
                threshold=2.0,  # Z-score threshold
                details="Historical anomaly detected" if is_anomalous else "Within historical range",
            ))

        return checks

    def _build_qc_checks_from_uncertainty(self, uncertainty_result: UncertaintyMap) -> List[QCCheck]:
        """Build QC checks from uncertainty result."""
        checks = []

        # Uncertainty level check
        status = CheckStatus.PASS
        if uncertainty_result.mean_uncertainty > 0.4:
            status = CheckStatus.WARNING
        if uncertainty_result.mean_uncertainty > 0.6:
            status = CheckStatus.SOFT_FAIL

        checks.append(QCCheck(
            check_name="uncertainty_level",
            category=CheckCategory.UNCERTAINTY,
            status=status,
            metric_value=uncertainty_result.mean_uncertainty,
            threshold=0.4,
            details=f"Mean uncertainty: {uncertainty_result.mean_uncertainty:.3f}",
        ))

        # Hotspot check
        if uncertainty_result.hotspot_count > 5:
            checks.append(QCCheck(
                check_name="uncertainty_hotspots",
                category=CheckCategory.UNCERTAINTY,
                status=CheckStatus.WARNING,
                metric_value=uncertainty_result.hotspot_count,
                threshold=5,
                details=f"High uncertainty hotspots: {uncertainty_result.hotspot_count}",
            ))

        return checks

    def _convert_gating_decision(
        self,
        gating_decision: GatingDecision,
        sanity_result: Optional[SanityResult] = None,
        validation_result: Optional[ValidationResult] = None,
        uncertainty_result: Optional[UncertaintyMap] = None,
    ) -> GateDecision:
        """Convert core GatingDecision to agent GateDecision."""
        # Map gate status
        status_map = {
            GateStatus.PASS: QAGateDecision.PASS,
            GateStatus.PASS_WITH_WARNINGS: QAGateDecision.PASS_WITH_WARNINGS,
            GateStatus.REVIEW_REQUIRED: QAGateDecision.REVIEW_REQUIRED,
            GateStatus.BLOCKED: QAGateDecision.FAIL,
        }
        status = status_map.get(gating_decision.status, QAGateDecision.FAIL)

        # Calculate confidence from various sources
        confidences = [gating_decision.confidence_score]
        if sanity_result:
            confidences.append(sanity_result.overall_score)
        if validation_result:
            confidences.append(validation_result.agreement_score)
        if uncertainty_result:
            confidences.append(1.0 - uncertainty_result.mean_uncertainty)

        confidence = sum(confidences) / len(confidences)

        # Build reasons and warnings
        reasons = [r.reason for r in gating_decision.rule_results if not r.passed]
        warnings = gating_decision.warnings

        # Determine if review is required
        requires_review = status == QAGateDecision.REVIEW_REQUIRED
        can_release = status in (QAGateDecision.PASS, QAGateDecision.PASS_WITH_WARNINGS)

        # Create review request if needed
        review_request = None
        if requires_review:
            review_request = {
                "priority": "normal",
                "type": "quality_validation",
                "reasons": reasons,
            }

        return GateDecision(
            status=status,
            confidence=confidence,
            can_release=can_release,
            requires_review=requires_review,
            reasons=reasons,
            warnings=warnings,
            flags_applied=gating_decision.flags_applied,
            review_request=review_request,
        )
