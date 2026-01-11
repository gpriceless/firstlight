"""
Pipeline Optimizer for Execution Optimization.

This module provides optimization strategies for pipeline execution,
including:
- Execution order optimization for minimal memory footprint
- Parallelization analysis and grouping
- Common subexpression elimination
- Memory-constrained scheduling
- Checkpoint placement for fault tolerance
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
import logging
import copy

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    Port,
    NodeType,
    DataType,
    EdgeType,
    NodeStatus,
    CycleDetectedError
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    get_global_registry
)

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    MINIMIZE_MEMORY = "minimize_memory"      # Prioritize low memory usage
    MAXIMIZE_PARALLELISM = "maximize_parallelism"  # Prioritize parallel execution
    MINIMIZE_RUNTIME = "minimize_runtime"    # Prioritize fast completion
    BALANCED = "balanced"                    # Balance all factors
    FAULT_TOLERANT = "fault_tolerant"        # Prioritize recovery capability


class SchedulingPolicy(Enum):
    """Scheduling policies for execution."""
    FIFO = "fifo"                # First in, first out
    LIFO = "lifo"                # Last in, first out (depth-first)
    PRIORITY = "priority"        # Based on node priority
    MEMORY_AWARE = "memory_aware"  # Based on memory constraints
    CRITICAL_PATH = "critical_path"  # Prioritize critical path


@dataclass
class OptimizationConfig:
    """
    Configuration for pipeline optimization.

    Attributes:
        strategy: Main optimization strategy
        scheduling_policy: Node scheduling policy
        max_memory_mb: Maximum memory constraint
        max_parallel_nodes: Maximum concurrent nodes
        enable_cse: Enable common subexpression elimination
        enable_checkpoints: Enable checkpoint insertion
        checkpoint_interval_nodes: Insert checkpoint every N nodes
        min_parallelism: Minimum parallelization factor
    """
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.CRITICAL_PATH
    max_memory_mb: Optional[int] = None
    max_parallel_nodes: int = 8
    enable_cse: bool = True
    enable_checkpoints: bool = False
    checkpoint_interval_nodes: int = 5
    min_parallelism: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "strategy": self.strategy.value,
            "scheduling_policy": self.scheduling_policy.value,
            "max_memory_mb": self.max_memory_mb,
            "max_parallel_nodes": self.max_parallel_nodes,
            "enable_cse": self.enable_cse,
            "enable_checkpoints": self.enable_checkpoints,
            "checkpoint_interval_nodes": self.checkpoint_interval_nodes,
            "min_parallelism": self.min_parallelism
        }


@dataclass
class ExecutionGroup:
    """
    A group of nodes that can execute together.

    Attributes:
        id: Group identifier
        node_ids: Node IDs in this group
        level: Execution level (for ordering groups)
        estimated_memory_mb: Total memory for this group
        estimated_runtime_seconds: Estimated runtime
        can_parallelize: Whether nodes can run in parallel
        checkpoint_before: Insert checkpoint before this group
    """
    id: str
    node_ids: List[str]
    level: int
    estimated_memory_mb: int = 0
    estimated_runtime_seconds: float = 0.0
    can_parallelize: bool = True
    checkpoint_before: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "node_ids": self.node_ids,
            "level": self.level,
            "estimated_memory_mb": self.estimated_memory_mb,
            "estimated_runtime_seconds": self.estimated_runtime_seconds,
            "can_parallelize": self.can_parallelize,
            "checkpoint_before": self.checkpoint_before
        }


@dataclass
class ExecutionPlan:
    """
    Optimized execution plan for a pipeline.

    Attributes:
        groups: Ordered list of execution groups
        execution_order: Flat list of node IDs in execution order
        estimated_total_memory_mb: Peak memory requirement
        estimated_total_runtime_seconds: Total runtime estimate
        parallelism_factor: Average parallelism achieved
        checkpoints: List of checkpoint locations
        optimizations_applied: List of optimizations applied
    """
    groups: List[ExecutionGroup]
    execution_order: List[str]
    estimated_total_memory_mb: int = 0
    estimated_total_runtime_seconds: float = 0.0
    parallelism_factor: float = 1.0
    checkpoints: List[str] = field(default_factory=list)
    optimizations_applied: List[str] = field(default_factory=list)

    @property
    def num_groups(self) -> int:
        """Number of execution groups."""
        return len(self.groups)

    @property
    def num_checkpoints(self) -> int:
        """Number of checkpoints."""
        return len(self.checkpoints)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "groups": [g.to_dict() for g in self.groups],
            "execution_order": self.execution_order,
            "estimated_total_memory_mb": self.estimated_total_memory_mb,
            "estimated_total_runtime_seconds": self.estimated_total_runtime_seconds,
            "parallelism_factor": self.parallelism_factor,
            "checkpoints": self.checkpoints,
            "optimizations_applied": self.optimizations_applied,
            "summary": {
                "num_groups": self.num_groups,
                "num_nodes": len(self.execution_order),
                "num_checkpoints": self.num_checkpoints
            }
        }


@dataclass
class OptimizationResult:
    """
    Result of pipeline optimization.

    Attributes:
        original_graph: Reference to original graph
        optimized_graph: Optimized graph (may be modified)
        execution_plan: Execution plan
        config: Configuration used
        optimization_time: Time taken to optimize
        improvements: Dict of improvement metrics
    """
    original_graph: PipelineGraph
    optimized_graph: PipelineGraph
    execution_plan: ExecutionPlan
    config: OptimizationConfig
    optimization_time: float = 0.0
    improvements: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "original_graph_id": self.original_graph.id,
            "optimized_graph_id": self.optimized_graph.id,
            "execution_plan": self.execution_plan.to_dict(),
            "config": self.config.to_dict(),
            "optimization_time": self.optimization_time,
            "improvements": self.improvements
        }


class PipelineOptimizer:
    """
    Optimizes pipeline execution for performance and resource utilization.

    Features:
    - Multiple optimization strategies
    - Memory-constrained scheduling
    - Parallel execution grouping
    - Common subexpression elimination
    - Checkpoint placement for fault tolerance

    Usage:
        optimizer = PipelineOptimizer(registry)
        result = optimizer.optimize(graph, config)
        plan = result.execution_plan
    """

    # Default memory estimate when not specified by algorithm
    DEFAULT_MEMORY_MB = 512
    DEFAULT_RUNTIME_SECONDS = 30.0

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None
    ):
        """
        Initialize the optimizer.

        Args:
            registry: Algorithm registry for resource estimation
        """
        self.registry = registry or get_global_registry()
        logger.info("Initialized PipelineOptimizer")

    def optimize(
        self,
        graph: PipelineGraph,
        config: Optional[OptimizationConfig] = None
    ) -> OptimizationResult:
        """
        Optimize a pipeline graph.

        Args:
            graph: Pipeline graph to optimize
            config: Optimization configuration

        Returns:
            OptimizationResult with optimized graph and execution plan
        """
        start_time = datetime.now(timezone.utc)
        config = config or OptimizationConfig()

        logger.info(
            f"Optimizing pipeline {graph.id} with strategy {config.strategy.value}"
        )

        # Create a working copy of the graph
        optimized_graph = self._copy_graph(graph)
        optimizations_applied = []

        # Apply common subexpression elimination if enabled
        if config.enable_cse:
            cse_count = self._apply_cse(optimized_graph)
            if cse_count > 0:
                optimizations_applied.append(f"cse:{cse_count}")
                logger.info(f"CSE eliminated {cse_count} duplicate computations")

        # Create execution plan based on strategy
        execution_plan = self._create_execution_plan(optimized_graph, config)

        # Apply strategy-specific optimizations
        if config.strategy == OptimizationStrategy.MINIMIZE_MEMORY:
            self._optimize_for_memory(execution_plan, optimized_graph, config)
            optimizations_applied.append("memory_minimization")
        elif config.strategy == OptimizationStrategy.MAXIMIZE_PARALLELISM:
            self._optimize_for_parallelism(execution_plan, optimized_graph, config)
            optimizations_applied.append("parallelism_maximization")
        elif config.strategy == OptimizationStrategy.MINIMIZE_RUNTIME:
            self._optimize_for_runtime(execution_plan, optimized_graph, config)
            optimizations_applied.append("runtime_minimization")
        elif config.strategy == OptimizationStrategy.FAULT_TOLERANT:
            self._optimize_for_fault_tolerance(execution_plan, optimized_graph, config)
            optimizations_applied.append("fault_tolerance")

        # Insert checkpoints if enabled
        if config.enable_checkpoints:
            checkpoint_count = self._insert_checkpoints(
                execution_plan, config.checkpoint_interval_nodes
            )
            if checkpoint_count > 0:
                optimizations_applied.append(f"checkpoints:{checkpoint_count}")

        execution_plan.optimizations_applied = optimizations_applied

        # Calculate improvements
        improvements = self._calculate_improvements(graph, optimized_graph, execution_plan)

        optimization_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        logger.info(
            f"Optimization complete in {optimization_time:.3f}s: "
            f"{len(optimizations_applied)} optimizations applied"
        )

        return OptimizationResult(
            original_graph=graph,
            optimized_graph=optimized_graph,
            execution_plan=execution_plan,
            config=config,
            optimization_time=optimization_time,
            improvements=improvements
        )

    def create_execution_plan(
        self,
        graph: PipelineGraph,
        config: Optional[OptimizationConfig] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan without modifying the graph.

        Args:
            graph: Pipeline graph
            config: Optimization configuration

        Returns:
            ExecutionPlan
        """
        config = config or OptimizationConfig()
        return self._create_execution_plan(graph, config)

    def _copy_graph(self, graph: PipelineGraph) -> PipelineGraph:
        """Create a deep copy of the graph."""
        return PipelineGraph.from_dict(graph.to_dict())

    def _create_execution_plan(
        self,
        graph: PipelineGraph,
        config: OptimizationConfig
    ) -> ExecutionPlan:
        """Create initial execution plan from graph."""
        # Get execution levels
        try:
            levels = graph.get_execution_levels()
        except CycleDetectedError:
            # Fallback to basic topological order
            order = graph.get_execution_order()
            levels = [[node_id] for node_id in order]

        # Create execution groups
        groups = []
        execution_order = []
        total_memory = 0
        total_runtime = 0.0

        for level_idx, level in enumerate(levels):
            # Estimate resources for this level
            level_memory = 0
            level_runtime = 0.0

            for node_id in level:
                node = graph.get_node(node_id)
                if node:
                    mem, runtime = self._estimate_node_resources(node)
                    level_memory += mem
                    level_runtime = max(level_runtime, runtime)  # Parallel, so max not sum

            # Create group
            group = ExecutionGroup(
                id=f"group_{level_idx}",
                node_ids=sorted(level),  # Sort for determinism
                level=level_idx,
                estimated_memory_mb=level_memory,
                estimated_runtime_seconds=level_runtime,
                can_parallelize=len(level) <= config.max_parallel_nodes
            )
            groups.append(group)

            # Add to execution order
            execution_order.extend(sorted(level))

            # Track totals
            total_memory = max(total_memory, level_memory)  # Peak memory
            total_runtime += level_runtime

        # Calculate parallelism factor
        total_nodes = len(execution_order)
        parallelism_factor = total_nodes / len(groups) if groups else 1.0

        return ExecutionPlan(
            groups=groups,
            execution_order=execution_order,
            estimated_total_memory_mb=total_memory,
            estimated_total_runtime_seconds=total_runtime,
            parallelism_factor=parallelism_factor
        )

    def _estimate_node_resources(self, node: PipelineNode) -> Tuple[int, float]:
        """Estimate memory and runtime for a node."""
        if node.processor:
            algorithm = self.registry.get(node.processor)
            if algorithm and algorithm.resources:
                memory = algorithm.resources.memory_mb or self.DEFAULT_MEMORY_MB
                runtime = (algorithm.resources.max_runtime_minutes or 0.5) * 60
                return memory, runtime

        return self.DEFAULT_MEMORY_MB, self.DEFAULT_RUNTIME_SECONDS

    def _apply_cse(self, graph: PipelineGraph) -> int:
        """
        Apply common subexpression elimination.

        Finds nodes with identical configurations and merges them.

        Returns:
            Number of nodes eliminated
        """
        eliminated = 0

        # Group nodes by signature
        signatures: Dict[str, List[str]] = {}

        for node in graph.get_processor_nodes():
            sig = self._compute_node_signature(node)
            if sig not in signatures:
                signatures[sig] = []
            signatures[sig].append(node.id)

        # Merge duplicate nodes
        for sig, node_ids in signatures.items():
            if len(node_ids) > 1:
                # Keep first node, redirect others
                keep_id = node_ids[0]
                for remove_id in node_ids[1:]:
                    self._merge_nodes(graph, keep_id, remove_id)
                    eliminated += 1

        return eliminated

    def _compute_node_signature(self, node: PipelineNode) -> str:
        """Compute a signature for node equivalence checking."""
        import hashlib
        import json

        data = {
            "processor": node.processor,
            "parameters": node.parameters,
            "input_count": len(node.input_ports),
            "output_count": len(node.output_ports)
        }
        sig_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(sig_str.encode()).hexdigest()[:8]

    def _merge_nodes(
        self,
        graph: PipelineGraph,
        keep_id: str,
        remove_id: str
    ) -> None:
        """
        Merge two equivalent nodes, redirecting edges.

        Args:
            graph: Pipeline graph
            keep_id: Node to keep
            remove_id: Node to remove
        """
        # Get outgoing edges from remove_id
        outgoing = graph.get_outgoing_edges(remove_id)

        # Redirect edges to keep_id
        for edge in outgoing:
            # Remove old edge
            graph.remove_edge(edge.id)

            # Create new edge from keep_id
            new_edge = PipelineEdge(
                source_node=keep_id,
                source_port=edge.source_port,
                target_node=edge.target_node,
                target_port=edge.target_port,
                edge_type=edge.edge_type
            )
            try:
                graph.add_edge(new_edge)
            except ValueError:
                # Edge might already exist
                pass

        # Remove the duplicate node
        try:
            graph.remove_node(remove_id)
        except KeyError:
            pass

    def _optimize_for_memory(
        self,
        plan: ExecutionPlan,
        graph: PipelineGraph,
        config: OptimizationConfig
    ) -> None:
        """
        Optimize execution plan for minimum memory usage.

        Reorders nodes to minimize peak memory.
        """
        if not config.max_memory_mb:
            return

        # Split groups that exceed memory limit
        new_groups = []

        for group in plan.groups:
            if group.estimated_memory_mb > config.max_memory_mb:
                # Split into smaller groups
                split_groups = self._split_group_by_memory(
                    group, graph, config.max_memory_mb
                )
                new_groups.extend(split_groups)
            else:
                new_groups.append(group)

        # Update plan
        plan.groups = new_groups

        # Rebuild execution order
        plan.execution_order = []
        for group in plan.groups:
            plan.execution_order.extend(group.node_ids)

        # Recalculate peak memory
        plan.estimated_total_memory_mb = max(
            g.estimated_memory_mb for g in plan.groups
        ) if plan.groups else 0

    def _split_group_by_memory(
        self,
        group: ExecutionGroup,
        graph: PipelineGraph,
        max_memory: int
    ) -> List[ExecutionGroup]:
        """Split a group to fit within memory constraints."""
        result = []
        current_nodes = []
        current_memory = 0
        group_idx = 0

        for node_id in group.node_ids:
            node = graph.get_node(node_id)
            if node:
                mem, runtime = self._estimate_node_resources(node)

                if current_memory + mem > max_memory and current_nodes:
                    # Create group from current nodes
                    result.append(ExecutionGroup(
                        id=f"{group.id}_split_{group_idx}",
                        node_ids=current_nodes,
                        level=group.level,
                        estimated_memory_mb=current_memory,
                        can_parallelize=False
                    ))
                    group_idx += 1
                    current_nodes = []
                    current_memory = 0

                current_nodes.append(node_id)
                current_memory += mem

        # Add remaining nodes
        if current_nodes:
            result.append(ExecutionGroup(
                id=f"{group.id}_split_{group_idx}",
                node_ids=current_nodes,
                level=group.level,
                estimated_memory_mb=current_memory,
                can_parallelize=len(current_nodes) > 1
            ))

        return result

    def _optimize_for_parallelism(
        self,
        plan: ExecutionPlan,
        graph: PipelineGraph,
        config: OptimizationConfig
    ) -> None:
        """
        Optimize execution plan for maximum parallelism.

        Tries to maximize the number of nodes that can run concurrently.
        """
        # Merge consecutive groups if they can run in parallel
        merged_groups = []
        i = 0

        while i < len(plan.groups):
            current = plan.groups[i]

            # Try to merge with next groups
            merged_nodes = list(current.node_ids)
            merged_memory = current.estimated_memory_mb

            j = i + 1
            while j < len(plan.groups):
                next_group = plan.groups[j]

                # Check if we can merge (no dependencies between groups)
                can_merge = self._groups_independent(
                    merged_nodes, next_group.node_ids, graph
                )

                if can_merge:
                    merged_nodes.extend(next_group.node_ids)
                    merged_memory += next_group.estimated_memory_mb
                    j += 1
                else:
                    break

            # Create merged group
            if len(merged_nodes) <= config.max_parallel_nodes:
                merged_groups.append(ExecutionGroup(
                    id=f"parallel_group_{len(merged_groups)}",
                    node_ids=merged_nodes,
                    level=current.level,
                    estimated_memory_mb=merged_memory,
                    can_parallelize=True
                ))
            else:
                # Too many nodes for parallel execution
                merged_groups.append(current)

            i = j

        plan.groups = merged_groups

        # Recalculate parallelism factor
        total_nodes = sum(len(g.node_ids) for g in plan.groups)
        plan.parallelism_factor = total_nodes / len(plan.groups) if plan.groups else 1.0

    def _groups_independent(
        self,
        nodes_a: List[str],
        nodes_b: List[str],
        graph: PipelineGraph
    ) -> bool:
        """Check if two sets of nodes have no dependencies."""
        set_a = set(nodes_a)
        set_b = set(nodes_b)

        # Check for edges between the groups
        for node_id in nodes_a:
            successors = graph.get_successors(node_id)
            if successors & set_b:
                return False

        for node_id in nodes_b:
            predecessors = graph.get_predecessors(node_id)
            if predecessors & set_a:
                return False

        return True

    def _optimize_for_runtime(
        self,
        plan: ExecutionPlan,
        graph: PipelineGraph,
        config: OptimizationConfig
    ) -> None:
        """
        Optimize execution plan for minimum runtime.

        Prioritizes critical path and maximizes parallelism.
        """
        # Calculate critical path
        critical_path = self._find_critical_path(graph)

        # Prioritize critical path nodes
        for group in plan.groups:
            critical_nodes = [n for n in group.node_ids if n in critical_path]
            other_nodes = [n for n in group.node_ids if n not in critical_path]

            # Put critical nodes first
            group.node_ids = critical_nodes + other_nodes

    def _find_critical_path(self, graph: PipelineGraph) -> Set[str]:
        """Find the critical path (longest path) through the graph."""
        # Calculate longest path from inputs
        distances: Dict[str, float] = {}
        predecessors: Dict[str, Optional[str]] = {}

        for node in graph.iter_nodes(order="topological"):
            incoming = graph.get_incoming_edges(node.id)

            if not incoming:
                distances[node.id] = self._estimate_node_resources(node)[1]
                predecessors[node.id] = None
            else:
                max_dist = 0.0
                max_pred = None

                for edge in incoming:
                    source_dist = distances.get(edge.source_node, 0)
                    if source_dist > max_dist:
                        max_dist = source_dist
                        max_pred = edge.source_node

                node_runtime = self._estimate_node_resources(node)[1]
                distances[node.id] = max_dist + node_runtime
                predecessors[node.id] = max_pred

        # Find the endpoint with maximum distance
        if not distances:
            return set()

        end_node = max(distances, key=lambda n: distances[n])

        # Trace back the critical path
        critical_path = set()
        current = end_node

        while current:
            critical_path.add(current)
            current = predecessors.get(current)

        return critical_path

    def _optimize_for_fault_tolerance(
        self,
        plan: ExecutionPlan,
        graph: PipelineGraph,
        config: OptimizationConfig
    ) -> None:
        """
        Optimize execution plan for fault tolerance.

        Inserts checkpoints and limits parallel execution.
        """
        # Reduce parallelism to make restarts cheaper
        for group in plan.groups:
            if len(group.node_ids) > 2:
                group.can_parallelize = False

        # Mark checkpoint locations
        self._insert_checkpoints(plan, max(2, config.checkpoint_interval_nodes))

    def _insert_checkpoints(
        self,
        plan: ExecutionPlan,
        interval: int
    ) -> int:
        """Insert checkpoints at regular intervals."""
        checkpoint_count = 0
        nodes_since_checkpoint = 0

        for group in plan.groups:
            nodes_since_checkpoint += len(group.node_ids)

            if nodes_since_checkpoint >= interval:
                group.checkpoint_before = True
                plan.checkpoints.append(group.id)
                checkpoint_count += 1
                nodes_since_checkpoint = 0

        return checkpoint_count

    def _calculate_improvements(
        self,
        original: PipelineGraph,
        optimized: PipelineGraph,
        plan: ExecutionPlan
    ) -> Dict[str, Any]:
        """Calculate improvement metrics from optimization."""
        original_nodes = original.num_nodes
        optimized_nodes = optimized.num_nodes

        return {
            "nodes_eliminated": original_nodes - optimized_nodes,
            "parallelism_factor": plan.parallelism_factor,
            "num_groups": plan.num_groups,
            "num_checkpoints": plan.num_checkpoints,
            "peak_memory_mb": plan.estimated_total_memory_mb,
            "estimated_runtime_seconds": plan.estimated_total_runtime_seconds
        }


def optimize_pipeline(
    graph: PipelineGraph,
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
    max_memory_mb: Optional[int] = None,
    max_parallel_nodes: int = 8,
    enable_checkpoints: bool = False,
    registry: Optional[AlgorithmRegistry] = None
) -> OptimizationResult:
    """
    Convenience function to optimize a pipeline.

    Args:
        graph: Pipeline graph to optimize
        strategy: Optimization strategy
        max_memory_mb: Maximum memory constraint
        max_parallel_nodes: Maximum concurrent nodes
        enable_checkpoints: Enable checkpoint insertion
        registry: Algorithm registry

    Returns:
        OptimizationResult
    """
    config = OptimizationConfig(
        strategy=strategy,
        max_memory_mb=max_memory_mb,
        max_parallel_nodes=max_parallel_nodes,
        enable_checkpoints=enable_checkpoints
    )

    optimizer = PipelineOptimizer(registry=registry)
    return optimizer.optimize(graph, config)
