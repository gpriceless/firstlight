"""
Pipeline Assembly Module for Event Intelligence Platform

This module provides components for building, validating, and optimizing
analysis pipeline DAGs (Directed Acyclic Graphs).

Components:
- graph: Pipeline graph representation with nodes and edges
- assembler: DAG construction from specifications
- validator: Pre-execution validation
- optimizer: Execution optimization

Usage:
    from core.analysis.assembly import (
        PipelineGraph, PipelineNode, PipelineEdge,
        PipelineAssembler, PipelineSpec,
        PipelineValidator, validate_pipeline,
        PipelineOptimizer, optimize_pipeline
    )

    # Create from specification
    spec = PipelineSpec.from_yaml("my_pipeline.yaml")
    assembler = PipelineAssembler()
    result = assembler.assemble(spec)

    # Validate
    validation = validate_pipeline(result.graph)
    if validation.is_valid:
        # Optimize
        optimized = optimize_pipeline(result.graph)
        plan = optimized.plan
"""

from core.analysis.assembly.graph import (
    # Core graph structures
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    Port,
    QCGate,
    # Enums
    NodeType,
    DataType,
    EdgeType,
    NodeStatus,
    # Errors
    CycleDetectedError,
)

from core.analysis.assembly.assembler import (
    # Assembler classes
    PipelineAssembler,
    DynamicAssembler,
    # Specification classes
    PipelineSpec,
    InputSpec,
    OutputSpec,
    StepSpec,
    TemporalRole,
    # Results
    AssemblyResult,
    # Errors
    AssemblyError,
    InputNotFoundError,
    AlgorithmNotFoundError,
    ConnectionError,
)

from core.analysis.assembly.validator import (
    # Validator
    PipelineValidator,
    validate_pipeline,
    # Results
    ValidationResult,
    ValidationIssue,
    ResourceEstimate,
    # Enums
    ValidationSeverity,
    ValidationCategory,
)

from core.analysis.assembly.optimizer import (
    # Optimizer
    PipelineOptimizer,
    optimize_pipeline,
    # Results
    OptimizationResult,
    ExecutionPlan,
    ExecutionGroup,
    OptimizationConfig,
    # Enums
    OptimizationStrategy,
    SchedulingPolicy,
)

__all__ = [
    # Graph module
    'PipelineGraph',
    'PipelineNode',
    'PipelineEdge',
    'Port',
    'QCGate',
    'NodeType',
    'DataType',
    'EdgeType',
    'NodeStatus',
    'CycleDetectedError',

    # Assembler module
    'PipelineAssembler',
    'DynamicAssembler',
    'PipelineSpec',
    'InputSpec',
    'OutputSpec',
    'StepSpec',
    'TemporalRole',
    'AssemblyResult',
    'AssemblyError',
    'InputNotFoundError',
    'AlgorithmNotFoundError',
    'ConnectionError',

    # Validator module
    'PipelineValidator',
    'validate_pipeline',
    'ValidationResult',
    'ValidationIssue',
    'ResourceEstimate',
    'ValidationSeverity',
    'ValidationCategory',

    # Optimizer module
    'PipelineOptimizer',
    'optimize_pipeline',
    'OptimizationResult',
    'ExecutionPlan',
    'ExecutionGroup',
    'OptimizationConfig',
    'OptimizationStrategy',
    'SchedulingPolicy',
]
