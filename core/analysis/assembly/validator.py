"""
Pipeline Validator for Event Intelligence Platform

Provides comprehensive pre-execution validation of pipeline graphs,
including structural checks, data type compatibility, resource estimation,
and algorithm requirement verification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import logging

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    Port,
    NodeType,
    DataType,
    EdgeType,
    CycleDetectedError
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    ResourceRequirements,
    get_global_registry,
    load_default_algorithms
)

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    ERROR = "error"      # Prevents execution
    WARNING = "warning"  # May cause issues
    INFO = "info"        # Informational


class ValidationCategory(Enum):
    """Categories of validation checks."""
    STRUCTURE = "structure"        # Graph structure issues
    CONNECTIVITY = "connectivity"  # Node connection issues
    DATA_TYPE = "data_type"        # Type compatibility issues
    ALGORITHM = "algorithm"        # Algorithm-related issues
    RESOURCE = "resource"          # Resource requirement issues
    PARAMETER = "parameter"        # Parameter validation issues
    QC = "qc"                      # Quality control issues


@dataclass
class ValidationIssue:
    """
    A single validation issue.

    Attributes:
        severity: Issue severity (error, warning, info)
        category: Issue category
        message: Human-readable description
        node_id: Related node ID (if applicable)
        edge_id: Related edge ID (if applicable)
        details: Additional details
    """
    severity: ValidationSeverity
    category: ValidationCategory
    message: str
    node_id: Optional[str] = None
    edge_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "node_id": self.node_id,
            "edge_id": self.edge_id,
            "details": self.details
        }

    def __str__(self) -> str:
        location = ""
        if self.node_id:
            location = f"[node:{self.node_id}] "
        elif self.edge_id:
            location = f"[edge:{self.edge_id}] "
        return f"[{self.severity.value.upper()}] {location}{self.message}"


@dataclass
class ResourceEstimate:
    """
    Estimated resource requirements for pipeline execution.

    Attributes:
        total_memory_mb: Estimated total memory (peak)
        peak_memory_mb: Estimated peak memory during execution
        gpu_required: Whether GPU is needed
        gpu_memory_mb: GPU memory required
        estimated_runtime_minutes: Estimated runtime
        parallelizable_steps: Number of steps that can run in parallel
        distributed_steps: Number of steps requiring distributed execution
    """
    total_memory_mb: int = 0
    peak_memory_mb: int = 0
    gpu_required: bool = False
    gpu_memory_mb: int = 0
    estimated_runtime_minutes: float = 0.0
    parallelizable_steps: int = 0
    distributed_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_memory_mb": self.total_memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_required": self.gpu_required,
            "gpu_memory_mb": self.gpu_memory_mb,
            "estimated_runtime_minutes": self.estimated_runtime_minutes,
            "parallelizable_steps": self.parallelizable_steps,
            "distributed_steps": self.distributed_steps
        }


@dataclass
class ValidationResult:
    """
    Result of pipeline validation.

    Attributes:
        is_valid: Whether pipeline is valid for execution
        issues: List of validation issues
        resource_estimate: Estimated resource requirements
        validation_time: Time taken to validate
        graph_hash: Hash of validated graph
    """
    is_valid: bool
    issues: List[ValidationIssue]
    resource_estimate: ResourceEstimate
    validation_time: float = 0.0
    graph_hash: Optional[str] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def infos(self) -> List[ValidationIssue]:
        """Get info-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.INFO]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "issues": [i.to_dict() for i in self.issues],
            "resource_estimate": self.resource_estimate.to_dict(),
            "validation_time": self.validation_time,
            "graph_hash": self.graph_hash,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos)
            }
        }

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        summary = f"[{status}] {len(self.errors)} errors, {len(self.warnings)} warnings"
        lines = [summary]
        for issue in self.issues:
            if issue.severity != ValidationSeverity.INFO:
                lines.append(f"  {issue}")
        return "\n".join(lines)


class PipelineValidator:
    """
    Comprehensive pipeline validator.

    Performs multiple validation passes:
    1. Structural validation (DAG properties)
    2. Connectivity validation (all nodes reachable)
    3. Type compatibility (data types match)
    4. Algorithm validation (requirements met)
    5. Resource estimation

    Usage:
        validator = PipelineValidator(registry)
        result = validator.validate(graph)
        if result.is_valid:
            # Safe to execute
    """

    # Default resource estimates when not specified
    DEFAULT_MEMORY_MB = 1024
    DEFAULT_RUNTIME_MINUTES = 5.0

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        strict_types: bool = True
    ):
        """
        Initialize pipeline validator.

        Args:
            registry: Algorithm registry (uses global if None)
            strict_types: If True, require exact type matches
        """
        if registry is None:
            registry = get_global_registry()
            if not registry.algorithms:
                load_default_algorithms()

        self.registry = registry
        self.strict_types = strict_types

        logger.info("Initialized PipelineValidator")

    def validate(
        self,
        graph: PipelineGraph,
        available_resources: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a pipeline graph.

        Args:
            graph: Pipeline graph to validate
            available_resources: Optional resource constraints to check against

        Returns:
            ValidationResult with issues and resource estimates
        """
        start_time = datetime.now(timezone.utc)
        issues: List[ValidationIssue] = []

        logger.info(f"Validating pipeline: {graph.id}")

        # Run validation passes
        self._validate_structure(graph, issues)
        self._validate_connectivity(graph, issues)
        self._validate_data_types(graph, issues)
        self._validate_algorithms(graph, issues)
        self._validate_parameters(graph, issues)
        self._validate_qc_gates(graph, issues)

        # Estimate resources
        resource_estimate = self._estimate_resources(graph)

        # Check against available resources
        if available_resources:
            self._check_resource_constraints(
                resource_estimate, available_resources, issues
            )

        # Determine validity
        has_errors = any(
            issue.severity == ValidationSeverity.ERROR
            for issue in issues
        )

        validation_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        result = ValidationResult(
            is_valid=not has_errors,
            issues=issues,
            resource_estimate=resource_estimate,
            validation_time=validation_time,
            graph_hash=graph.compute_hash()
        )

        logger.info(
            f"Validation complete: {len(issues)} issues found, "
            f"{'VALID' if result.is_valid else 'INVALID'}"
        )

        return result

    # --- Structural Validation ---

    def _validate_structure(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate graph structure."""

        # Check for empty graph
        if graph.is_empty():
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURE,
                message="Pipeline graph is empty"
            ))
            return

        # Check for DAG property (no cycles)
        try:
            graph.get_execution_order()
        except CycleDetectedError as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURE,
                message=f"Pipeline contains a cycle: {e}"
            ))

        # Check for input nodes
        input_nodes = graph.get_input_nodes()
        if not input_nodes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURE,
                message="Pipeline has no input nodes"
            ))

        # Check for output nodes
        output_nodes = graph.get_output_nodes()
        if not output_nodes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STRUCTURE,
                message="Pipeline has no output nodes"
            ))

        # Check for processor nodes
        processor_nodes = graph.get_processor_nodes()
        if not processor_nodes:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.STRUCTURE,
                message="Pipeline has no processing steps"
            ))

        # Validate node IDs are valid identifiers
        for node in graph.nodes:
            if not self._is_valid_identifier(node.id):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.STRUCTURE,
                    message=f"Node ID '{node.id}' contains invalid characters",
                    node_id=node.id
                ))

    def _is_valid_identifier(self, identifier: str) -> bool:
        """Check if identifier is valid (alphanumeric + underscore)."""
        if not identifier:
            return False
        return all(c.isalnum() or c in ('_', '-') for c in identifier)

    # --- Connectivity Validation ---

    def _validate_connectivity(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate node connectivity."""

        # Check for disconnected nodes
        for node in graph.nodes:
            incoming = graph.get_incoming_edges(node.id)
            outgoing = graph.get_outgoing_edges(node.id)

            if node.node_type == NodeType.INPUT:
                if not outgoing:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.CONNECTIVITY,
                        message=f"Input node '{node.id}' has no outgoing connections",
                        node_id=node.id
                    ))

            elif node.node_type == NodeType.OUTPUT:
                if not incoming:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONNECTIVITY,
                        message=f"Output node '{node.id}' has no incoming connections",
                        node_id=node.id
                    ))

            elif node.node_type == NodeType.PROCESSOR:
                if not incoming:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONNECTIVITY,
                        message=f"Processor '{node.id}' has no input connections",
                        node_id=node.id
                    ))
                if not outgoing:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.CONNECTIVITY,
                        message=f"Processor '{node.id}' has no output connections",
                        node_id=node.id
                    ))

        # Check required input ports are connected
        for node in graph.nodes:
            connected_ports = {
                edge.target_port
                for edge in graph.get_incoming_edges(node.id)
            }

            for port in node.input_ports:
                if port.required and port.name not in connected_ports:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.CONNECTIVITY,
                        message=f"Required input port '{port.name}' on '{node.id}' is not connected",
                        node_id=node.id,
                        details={"port": port.name}
                    ))

        # Check outputs are reachable from inputs
        input_ids = {n.id for n in graph.get_input_nodes()}
        for output_node in graph.get_output_nodes():
            ancestors = graph.get_ancestors(output_node.id)
            if not ancestors & input_ids:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.CONNECTIVITY,
                    message=f"Output '{output_node.id}' is not reachable from any input",
                    node_id=output_node.id
                ))

    # --- Data Type Validation ---

    def _validate_data_types(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate data type compatibility across edges."""

        for edge in graph.edges:
            source_node = graph.get_node(edge.source_node)
            target_node = graph.get_node(edge.target_node)

            if not source_node or not target_node:
                continue

            # Get source port type
            source_port = source_node.get_output_port(edge.source_port)
            target_port = target_node.get_input_port(edge.target_port)

            if not source_port:
                # Input nodes have implicit output port
                if source_node.node_type == NodeType.INPUT:
                    source_type = source_node.output_ports[0].data_type if source_node.output_ports else DataType.RASTER
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.DATA_TYPE,
                        message=f"Source port '{edge.source_port}' not found on '{edge.source_node}'",
                        edge_id=edge.id
                    ))
                    continue
            else:
                source_type = source_port.data_type

            if not target_port:
                # Output nodes have implicit input port
                if target_node.node_type == NodeType.OUTPUT:
                    target_type = target_node.input_ports[0].data_type if target_node.input_ports else DataType.RASTER
                else:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category=ValidationCategory.DATA_TYPE,
                        message=f"Target port '{edge.target_port}' not found on '{edge.target_node}'",
                        edge_id=edge.id
                    ))
                    continue
            else:
                target_type = target_port.data_type

            # Check type compatibility
            if not self._types_compatible(source_type, target_type):
                severity = ValidationSeverity.ERROR if self.strict_types else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    severity=severity,
                    category=ValidationCategory.DATA_TYPE,
                    message=f"Type mismatch: {source_type.value} -> {target_type.value}",
                    edge_id=edge.id,
                    details={
                        "source_type": source_type.value,
                        "target_type": target_type.value
                    }
                ))

    def _types_compatible(self, source: DataType, target: DataType) -> bool:
        """Check if data types are compatible."""
        if source == target:
            return True

        # Define compatible type pairs
        compatible_pairs = {
            # Raster types are generally compatible
            (DataType.RASTER, DataType.RASTER),
            # Scalar can go anywhere
            (DataType.SCALAR, DataType.RASTER),
            (DataType.SCALAR, DataType.TABLE),
        }

        return (source, target) in compatible_pairs

    # --- Algorithm Validation ---

    def _validate_algorithms(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate algorithm references and requirements."""

        for node in graph.get_processor_nodes():
            if not node.processor:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.ALGORITHM,
                    message=f"Processor node '{node.id}' has no algorithm specified",
                    node_id=node.id
                ))
                continue

            algorithm = self.registry.get(node.processor)

            if not algorithm:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.ALGORITHM,
                    message=f"Algorithm '{node.processor}' not found in registry",
                    node_id=node.id,
                    details={"processor": node.processor}
                ))
                continue

            # Check if algorithm is deprecated
            if algorithm.deprecated:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.ALGORITHM,
                    message=f"Algorithm '{algorithm.id}' is deprecated",
                    node_id=node.id,
                    details={
                        "replacement": algorithm.replacement_algorithm
                    }
                ))

            # Check input count matches requirements
            connected_inputs = len(graph.get_incoming_edges(node.id))
            required_inputs = len(algorithm.required_data_types)

            if connected_inputs < required_inputs:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.ALGORITHM,
                    message=f"Algorithm '{algorithm.id}' requires {required_inputs} inputs, "
                           f"but only {connected_inputs} connected",
                    node_id=node.id,
                    details={
                        "required": required_inputs,
                        "connected": connected_inputs
                    }
                ))

    # --- Parameter Validation ---

    def _validate_parameters(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate algorithm parameters."""

        for node in graph.get_processor_nodes():
            if not node.processor:
                continue

            algorithm = self.registry.get(node.processor)
            if not algorithm:
                continue

            # Check for unknown parameters
            if algorithm.parameter_schema:
                known_params = set(algorithm.parameter_schema.keys())
                provided_params = set(node.parameters.keys())
                unknown_params = provided_params - known_params

                for param in unknown_params:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PARAMETER,
                        message=f"Unknown parameter '{param}' for algorithm '{algorithm.id}'",
                        node_id=node.id,
                        details={"parameter": param}
                    ))

            # Basic type checking for parameters
            for param_name, param_value in node.parameters.items():
                if param_value is None:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        category=ValidationCategory.PARAMETER,
                        message=f"Parameter '{param_name}' has null value",
                        node_id=node.id,
                        details={"parameter": param_name}
                    ))

    # --- QC Gate Validation ---

    def _validate_qc_gates(
        self,
        graph: PipelineGraph,
        issues: List[ValidationIssue]
    ) -> None:
        """Validate QC gate configurations."""

        for node in graph.nodes:
            if not node.qc_gate or not node.qc_gate.enabled:
                continue

            # Check QC checks are specified
            if not node.qc_gate.checks:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.QC,
                    message=f"QC gate on '{node.id}' is enabled but has no checks defined",
                    node_id=node.id
                ))

            # Validate on_fail action
            valid_actions = {"continue", "warn", "abort"}
            if node.qc_gate.on_fail not in valid_actions:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.QC,
                    message=f"Invalid QC gate action '{node.qc_gate.on_fail}'",
                    node_id=node.id,
                    details={"valid_actions": list(valid_actions)}
                ))

    # --- Resource Estimation ---

    def _estimate_resources(self, graph: PipelineGraph) -> ResourceEstimate:
        """
        Estimate resource requirements for pipeline execution.

        Args:
            graph: Pipeline graph

        Returns:
            ResourceEstimate
        """
        total_memory = 0
        max_parallel_memory = 0
        gpu_required = False
        max_gpu_memory = 0
        total_runtime = 0.0
        distributed_count = 0

        # Get execution levels for parallel analysis
        try:
            levels = graph.get_execution_levels()
        except CycleDetectedError:
            levels = []

        parallelizable = sum(1 for level in levels if len(level) > 1)

        for node in graph.get_processor_nodes():
            algorithm = self.registry.get(node.processor) if node.processor else None

            if algorithm and algorithm.resources:
                resources = algorithm.resources

                # Memory
                mem = resources.memory_mb or self.DEFAULT_MEMORY_MB
                total_memory += mem

                # GPU
                if resources.gpu_required:
                    gpu_required = True
                if resources.gpu_memory_mb:
                    max_gpu_memory = max(max_gpu_memory, resources.gpu_memory_mb)

                # Runtime
                runtime = resources.max_runtime_minutes or self.DEFAULT_RUNTIME_MINUTES
                total_runtime += runtime

                # Distributed
                if resources.distributed:
                    distributed_count += 1
            else:
                # Default estimates
                total_memory += self.DEFAULT_MEMORY_MB
                total_runtime += self.DEFAULT_RUNTIME_MINUTES

        # Calculate peak memory (based on parallel execution)
        for level in levels:
            level_memory = 0
            for node_id in level:
                node = graph.get_node(node_id)
                if node and node.node_type == NodeType.PROCESSOR:
                    algorithm = self.registry.get(node.processor) if node.processor else None
                    if algorithm and algorithm.resources and algorithm.resources.memory_mb:
                        level_memory += algorithm.resources.memory_mb
                    else:
                        level_memory += self.DEFAULT_MEMORY_MB
            max_parallel_memory = max(max_parallel_memory, level_memory)

        return ResourceEstimate(
            total_memory_mb=total_memory,
            peak_memory_mb=max_parallel_memory or total_memory,
            gpu_required=gpu_required,
            gpu_memory_mb=max_gpu_memory,
            estimated_runtime_minutes=total_runtime,
            parallelizable_steps=parallelizable,
            distributed_steps=distributed_count
        )

    def _check_resource_constraints(
        self,
        estimate: ResourceEstimate,
        available: Dict[str, Any],
        issues: List[ValidationIssue]
    ) -> None:
        """Check estimated resources against available constraints."""

        # Memory check
        available_memory = available.get("memory_mb", float("inf"))
        if estimate.peak_memory_mb > available_memory:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.RESOURCE,
                message=f"Peak memory ({estimate.peak_memory_mb}MB) exceeds available ({available_memory}MB)",
                details={
                    "required": estimate.peak_memory_mb,
                    "available": available_memory
                }
            ))

        # GPU check
        gpu_available = available.get("gpu_available", False)
        if estimate.gpu_required and not gpu_available:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.RESOURCE,
                message="Pipeline requires GPU but none available"
            ))

        # GPU memory check
        available_gpu_memory = available.get("gpu_memory_mb", 0)
        if estimate.gpu_memory_mb > available_gpu_memory:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.RESOURCE,
                message=f"GPU memory ({estimate.gpu_memory_mb}MB) exceeds available ({available_gpu_memory}MB)",
                details={
                    "required": estimate.gpu_memory_mb,
                    "available": available_gpu_memory
                }
            ))

        # Runtime check
        max_runtime = available.get("max_runtime_minutes")
        if max_runtime and estimate.estimated_runtime_minutes > max_runtime:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.RESOURCE,
                message=f"Estimated runtime ({estimate.estimated_runtime_minutes:.1f}min) "
                       f"may exceed limit ({max_runtime}min)",
                details={
                    "estimated": estimate.estimated_runtime_minutes,
                    "limit": max_runtime
                }
            ))


def validate_pipeline(
    graph: PipelineGraph,
    registry: Optional[AlgorithmRegistry] = None,
    available_resources: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """
    Convenience function to validate a pipeline graph.

    Args:
        graph: Pipeline graph to validate
        registry: Algorithm registry (uses global if None)
        available_resources: Optional resource constraints

    Returns:
        ValidationResult
    """
    validator = PipelineValidator(registry=registry)
    return validator.validate(graph, available_resources)
