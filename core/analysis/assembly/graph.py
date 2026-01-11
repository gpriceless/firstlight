"""
Pipeline Graph Representation for Event Intelligence Platform

Provides directed acyclic graph (DAG) structures for representing
analysis pipelines with dependencies, inputs, outputs, and metadata.

Based on pipeline.schema.json specification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Iterator, Tuple
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the pipeline graph."""
    INPUT = "input"        # External data input
    PROCESSOR = "processor"  # Algorithm/processing step
    OUTPUT = "output"      # Final pipeline output
    CHECKPOINT = "checkpoint"  # State checkpoint
    QC_GATE = "qc_gate"    # Quality control check


class DataType(Enum):
    """Data types that flow through the pipeline."""
    RASTER = "raster"
    VECTOR = "vector"
    TABLE = "table"
    SCALAR = "scalar"
    REPORT = "report"


class EdgeType(Enum):
    """Types of edges in the pipeline graph."""
    DATA = "data"          # Data dependency
    CONTROL = "control"    # Control flow dependency
    OPTIONAL = "optional"  # Optional dependency


class NodeStatus(Enum):
    """Execution status of a node."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Port:
    """
    A port represents an input or output connection point on a node.

    Attributes:
        name: Port identifier (unique within node)
        data_type: Type of data expected/produced
        required: Whether this port must be connected
        description: Human-readable description
    """
    name: str
    data_type: DataType
    required: bool = True
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "required": self.required,
            "description": self.description
        }


@dataclass
class QCGate:
    """
    Quality control gate configuration.

    Attributes:
        enabled: Whether QC checks are active
        checks: List of check identifiers
        on_fail: Action when checks fail
    """
    enabled: bool = False
    checks: List[str] = field(default_factory=list)
    on_fail: str = "warn"  # continue, warn, abort

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "enabled": self.enabled,
            "checks": self.checks,
            "on_fail": self.on_fail
        }


@dataclass
class PipelineNode:
    """
    A node in the pipeline graph.

    Represents a processing step, input, or output in the pipeline.

    Attributes:
        id: Unique node identifier
        name: Human-readable name
        node_type: Type of node (input, processor, output, etc.)
        processor: Algorithm ID for processor nodes
        parameters: Algorithm parameters
        input_ports: List of input connection points
        output_ports: List of output connection points
        qc_gate: Quality control configuration
        metadata: Additional node metadata
    """
    id: str
    name: str
    node_type: NodeType
    processor: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    input_ports: List[Port] = field(default_factory=list)
    output_ports: List[Port] = field(default_factory=list)
    qc_gate: Optional[QCGate] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Execution state
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        """Validate node configuration."""
        if self.node_type == NodeType.PROCESSOR and not self.processor:
            raise ValueError(f"Processor node '{self.id}' must have a processor ID")

        # Ensure unique port names
        input_names = [p.name for p in self.input_ports]
        output_names = [p.name for p in self.output_ports]

        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Node '{self.id}' has duplicate input port names")
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Node '{self.id}' has duplicate output port names")

    def get_input_port(self, name: str) -> Optional[Port]:
        """Get input port by name."""
        for port in self.input_ports:
            if port.name == name:
                return port
        return None

    def get_output_port(self, name: str) -> Optional[Port]:
        """Get output port by name."""
        for port in self.output_ports:
            if port.name == name:
                return port
        return None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "processor": self.processor,
            "parameters": self.parameters,
            "input_ports": [p.to_dict() for p in self.input_ports],
            "output_ports": [p.to_dict() for p in self.output_ports],
            "qc_gate": self.qc_gate.to_dict() if self.qc_gate else None,
            "metadata": self.metadata,
            "status": self.status.value
        }
        if self.start_time:
            result["start_time"] = self.start_time.isoformat()
        if self.end_time:
            result["end_time"] = self.end_time.isoformat()
        if self.error_message:
            result["error_message"] = self.error_message
        return result


@dataclass
class PipelineEdge:
    """
    An edge in the pipeline graph.

    Represents a data or control flow connection between nodes.

    Attributes:
        source_node: ID of source node
        source_port: Name of source output port
        target_node: ID of target node
        target_port: Name of target input port
        edge_type: Type of edge (data, control, optional)
        metadata: Additional edge metadata
    """
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    edge_type: EdgeType = EdgeType.DATA
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique edge identifier."""
        return f"{self.source_node}.{self.source_port}->{self.target_node}.{self.target_port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
            "edge_type": self.edge_type.value,
            "metadata": self.metadata
        }


class CycleDetectedError(Exception):
    """Raised when a cycle is detected in the pipeline graph."""
    pass


class PipelineGraph:
    """
    Directed Acyclic Graph (DAG) representation of an analysis pipeline.

    Features:
    - Node and edge management
    - Topological sorting
    - Cycle detection
    - Execution order calculation
    - Graph traversal utilities

    Based on pipeline.schema.json structure.
    """

    def __init__(
        self,
        pipeline_id: str,
        name: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        applicable_classes: Optional[List[str]] = None
    ):
        """
        Initialize pipeline graph.

        Args:
            pipeline_id: Unique pipeline identifier
            name: Human-readable name
            version: Pipeline version (semver)
            description: Pipeline description
            applicable_classes: Event classes this pipeline supports
        """
        self.id = pipeline_id
        self.name = name
        self.version = version
        self.description = description
        self.applicable_classes = applicable_classes or []

        # Graph structure
        self._nodes: Dict[str, PipelineNode] = {}
        self._edges: Dict[str, PipelineEdge] = {}

        # Index structures for efficient lookup
        self._outgoing: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids
        self._incoming: Dict[str, Set[str]] = {}  # node_id -> set of edge_ids

        # Cached results
        self._topological_order: Optional[List[str]] = None
        self._is_valid: Optional[bool] = None

        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = datetime.now(timezone.utc)

        logger.info(f"Created pipeline graph: {pipeline_id} v{version}")

    def _invalidate_cache(self):
        """Invalidate cached computations after graph modification."""
        self._topological_order = None
        self._is_valid = None
        self.updated_at = datetime.now(timezone.utc)

    # --- Node Management ---

    def add_node(self, node: PipelineNode) -> None:
        """
        Add a node to the graph.

        Args:
            node: Node to add

        Raises:
            ValueError: If node ID already exists
        """
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists in graph")

        self._nodes[node.id] = node
        self._outgoing[node.id] = set()
        self._incoming[node.id] = set()

        self._invalidate_cache()
        logger.debug(f"Added node: {node.id} ({node.node_type.value})")

    def remove_node(self, node_id: str) -> PipelineNode:
        """
        Remove a node and all its edges from the graph.

        Args:
            node_id: ID of node to remove

        Returns:
            The removed node

        Raises:
            KeyError: If node not found
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node '{node_id}' not found in graph")

        # Remove all edges connected to this node
        edges_to_remove = list(self._incoming[node_id]) + list(self._outgoing[node_id])
        for edge_id in edges_to_remove:
            if edge_id in self._edges:
                self._remove_edge_internal(edge_id)

        # Remove node
        node = self._nodes.pop(node_id)
        del self._outgoing[node_id]
        del self._incoming[node_id]

        self._invalidate_cache()
        logger.debug(f"Removed node: {node_id}")
        return node

    def get_node(self, node_id: str) -> Optional[PipelineNode]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> List[PipelineNode]:
        """Get all nodes in the graph."""
        return list(self._nodes.values())

    @property
    def node_ids(self) -> Set[str]:
        """Get all node IDs."""
        return set(self._nodes.keys())

    # --- Edge Management ---

    def add_edge(self, edge: PipelineEdge) -> None:
        """
        Add an edge to the graph.

        Args:
            edge: Edge to add

        Raises:
            ValueError: If edge endpoints not found or edge creates a cycle
        """
        # Validate endpoints exist
        if edge.source_node not in self._nodes:
            raise ValueError(f"Source node '{edge.source_node}' not found")
        if edge.target_node not in self._nodes:
            raise ValueError(f"Target node '{edge.target_node}' not found")

        # Validate ports exist
        source = self._nodes[edge.source_node]
        target = self._nodes[edge.target_node]

        if source.node_type != NodeType.INPUT:
            if not source.get_output_port(edge.source_port):
                raise ValueError(
                    f"Output port '{edge.source_port}' not found on node '{edge.source_node}'"
                )

        if target.node_type != NodeType.OUTPUT:
            if not target.get_input_port(edge.target_port):
                raise ValueError(
                    f"Input port '{edge.target_port}' not found on node '{edge.target_node}'"
                )

        # Check for duplicate edge
        if edge.id in self._edges:
            raise ValueError(f"Edge '{edge.id}' already exists")

        # Add edge
        self._edges[edge.id] = edge
        self._outgoing[edge.source_node].add(edge.id)
        self._incoming[edge.target_node].add(edge.id)

        # Verify no cycle created
        try:
            self._topological_sort()
        except CycleDetectedError:
            # Rollback edge addition
            self._remove_edge_internal(edge.id)
            raise ValueError(
                f"Edge '{edge.id}' would create a cycle in the graph"
            )

        self._invalidate_cache()
        logger.debug(f"Added edge: {edge.id}")

    def _remove_edge_internal(self, edge_id: str) -> None:
        """Internal edge removal without cache invalidation."""
        if edge_id not in self._edges:
            return

        edge = self._edges[edge_id]
        self._outgoing[edge.source_node].discard(edge_id)
        self._incoming[edge.target_node].discard(edge_id)
        del self._edges[edge_id]

    def remove_edge(self, edge_id: str) -> PipelineEdge:
        """
        Remove an edge from the graph.

        Args:
            edge_id: ID of edge to remove

        Returns:
            The removed edge

        Raises:
            KeyError: If edge not found
        """
        if edge_id not in self._edges:
            raise KeyError(f"Edge '{edge_id}' not found in graph")

        edge = self._edges[edge_id]
        self._remove_edge_internal(edge_id)

        self._invalidate_cache()
        logger.debug(f"Removed edge: {edge_id}")
        return edge

    def get_edge(self, edge_id: str) -> Optional[PipelineEdge]:
        """Get edge by ID."""
        return self._edges.get(edge_id)

    @property
    def edges(self) -> List[PipelineEdge]:
        """Get all edges in the graph."""
        return list(self._edges.values())

    # --- Graph Properties ---

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self._edges)

    def is_empty(self) -> bool:
        """Check if graph has no nodes."""
        return len(self._nodes) == 0

    # --- Topological Operations ---

    def _topological_sort(self) -> List[str]:
        """
        Compute topological order using Kahn's algorithm.

        Returns:
            List of node IDs in topological order

        Raises:
            CycleDetectedError: If graph contains a cycle
        """
        if self._topological_order is not None:
            return self._topological_order

        # Compute in-degrees
        in_degree = {node_id: 0 for node_id in self._nodes}
        for edge in self._edges.values():
            in_degree[edge.target_node] += 1

        # Initialize queue with nodes having no incoming edges
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Get next node (sort for deterministic order)
            queue.sort()
            node_id = queue.pop(0)
            result.append(node_id)

            # Reduce in-degree for successors
            for edge_id in self._outgoing[node_id]:
                edge = self._edges[edge_id]
                in_degree[edge.target_node] -= 1
                if in_degree[edge.target_node] == 0:
                    queue.append(edge.target_node)

        # Check for cycle
        if len(result) != len(self._nodes):
            raise CycleDetectedError(
                f"Graph contains a cycle. Processed {len(result)} of {len(self._nodes)} nodes."
            )

        self._topological_order = result
        return result

    def get_execution_order(self) -> List[str]:
        """
        Get nodes in execution order (topological order).

        Returns:
            List of node IDs in the order they should be executed
        """
        return self._topological_sort()

    def get_execution_levels(self) -> List[List[str]]:
        """
        Get nodes grouped by execution level for parallel execution.

        Returns:
            List of lists, where each inner list contains nodes
            that can be executed in parallel.
        """
        # Compute in-degrees
        in_degree = {node_id: 0 for node_id in self._nodes}
        for edge in self._edges.values():
            in_degree[edge.target_node] += 1

        levels = []
        remaining = set(self._nodes.keys())

        while remaining:
            # Find nodes with no dependencies in remaining set
            level = [
                node_id for node_id in remaining
                if in_degree[node_id] == 0
            ]

            if not level:
                raise CycleDetectedError("Graph contains a cycle")

            level.sort()  # Deterministic ordering
            levels.append(level)

            # Remove nodes from remaining and update in-degrees
            for node_id in level:
                remaining.remove(node_id)
                for edge_id in self._outgoing[node_id]:
                    edge = self._edges[edge_id]
                    if edge.target_node in remaining:
                        in_degree[edge.target_node] -= 1

        return levels

    # --- Dependency Queries ---

    def get_predecessors(self, node_id: str) -> Set[str]:
        """
        Get immediate predecessor node IDs.

        Args:
            node_id: Target node ID

        Returns:
            Set of node IDs that have edges pointing to this node
        """
        predecessors = set()
        for edge_id in self._incoming.get(node_id, set()):
            edge = self._edges[edge_id]
            predecessors.add(edge.source_node)
        return predecessors

    def get_successors(self, node_id: str) -> Set[str]:
        """
        Get immediate successor node IDs.

        Args:
            node_id: Source node ID

        Returns:
            Set of node IDs that this node has edges pointing to
        """
        successors = set()
        for edge_id in self._outgoing.get(node_id, set()):
            edge = self._edges[edge_id]
            successors.add(edge.target_node)
        return successors

    def get_ancestors(self, node_id: str) -> Set[str]:
        """
        Get all ancestor node IDs (transitive predecessors).

        Args:
            node_id: Target node ID

        Returns:
            Set of all nodes that can reach this node
        """
        ancestors = set()
        queue = list(self.get_predecessors(node_id))

        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self.get_predecessors(current))

        return ancestors

    def get_descendants(self, node_id: str) -> Set[str]:
        """
        Get all descendant node IDs (transitive successors).

        Args:
            node_id: Source node ID

        Returns:
            Set of all nodes reachable from this node
        """
        descendants = set()
        queue = list(self.get_successors(node_id))

        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(self.get_successors(current))

        return descendants

    def get_incoming_edges(self, node_id: str) -> List[PipelineEdge]:
        """Get all edges pointing to a node."""
        return [
            self._edges[edge_id]
            for edge_id in self._incoming.get(node_id, set())
        ]

    def get_outgoing_edges(self, node_id: str) -> List[PipelineEdge]:
        """Get all edges from a node."""
        return [
            self._edges[edge_id]
            for edge_id in self._outgoing.get(node_id, set())
        ]

    # --- Node Type Queries ---

    def get_input_nodes(self) -> List[PipelineNode]:
        """Get all input nodes."""
        return [n for n in self._nodes.values() if n.node_type == NodeType.INPUT]

    def get_output_nodes(self) -> List[PipelineNode]:
        """Get all output nodes."""
        return [n for n in self._nodes.values() if n.node_type == NodeType.OUTPUT]

    def get_processor_nodes(self) -> List[PipelineNode]:
        """Get all processor nodes."""
        return [n for n in self._nodes.values() if n.node_type == NodeType.PROCESSOR]

    def get_root_nodes(self) -> List[PipelineNode]:
        """Get nodes with no incoming edges."""
        return [
            self._nodes[node_id]
            for node_id in self._nodes
            if len(self._incoming[node_id]) == 0
        ]

    def get_leaf_nodes(self) -> List[PipelineNode]:
        """Get nodes with no outgoing edges."""
        return [
            self._nodes[node_id]
            for node_id in self._nodes
            if len(self._outgoing[node_id]) == 0
        ]

    # --- Iteration ---

    def iter_nodes(self, order: str = "topological") -> Iterator[PipelineNode]:
        """
        Iterate over nodes.

        Args:
            order: Iteration order ("topological", "reverse", "insertion")

        Yields:
            PipelineNode objects in specified order
        """
        if order == "topological":
            for node_id in self.get_execution_order():
                yield self._nodes[node_id]
        elif order == "reverse":
            for node_id in reversed(self.get_execution_order()):
                yield self._nodes[node_id]
        else:
            for node in self._nodes.values():
                yield node

    def iter_edges(self) -> Iterator[PipelineEdge]:
        """Iterate over all edges."""
        for edge in self._edges.values():
            yield edge

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary representation.

        Compatible with pipeline.schema.json.
        """
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "applicable_classes": self.applicable_classes,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges.values()],
            "metadata": {
                **self.metadata,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "num_nodes": self.num_nodes,
                "num_edges": self.num_edges
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineGraph':
        """
        Create graph from dictionary representation.

        Args:
            data: Dictionary matching pipeline.schema.json

        Returns:
            Reconstructed PipelineGraph
        """
        graph = cls(
            pipeline_id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
            applicable_classes=data.get("applicable_classes", [])
        )

        # Add nodes
        for node_data in data.get("nodes", []):
            node = PipelineNode(
                id=node_data["id"],
                name=node_data.get("name", node_data["id"]),
                node_type=NodeType(node_data["node_type"]),
                processor=node_data.get("processor"),
                parameters=node_data.get("parameters", {}),
                input_ports=[
                    Port(
                        name=p["name"],
                        data_type=DataType(p["data_type"]),
                        required=p.get("required", True),
                        description=p.get("description")
                    )
                    for p in node_data.get("input_ports", [])
                ],
                output_ports=[
                    Port(
                        name=p["name"],
                        data_type=DataType(p["data_type"]),
                        required=p.get("required", True),
                        description=p.get("description")
                    )
                    for p in node_data.get("output_ports", [])
                ],
                qc_gate=QCGate(**node_data["qc_gate"]) if node_data.get("qc_gate") else None,
                metadata=node_data.get("metadata", {})
            )
            graph.add_node(node)

        # Add edges
        for edge_data in data.get("edges", []):
            edge = PipelineEdge(
                source_node=edge_data["source_node"],
                source_port=edge_data["source_port"],
                target_node=edge_data["target_node"],
                target_port=edge_data["target_port"],
                edge_type=EdgeType(edge_data.get("edge_type", "data")),
                metadata=edge_data.get("metadata", {})
            )
            graph.add_edge(edge)

        # Restore metadata
        if "metadata" in data:
            graph.metadata = {
                k: v for k, v in data["metadata"].items()
                if k not in ("created_at", "updated_at", "num_nodes", "num_edges")
            }

        return graph

    def to_json(self, indent: int = 2) -> str:
        """Convert graph to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'PipelineGraph':
        """Create graph from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # --- Hash and Comparison ---

    def compute_hash(self) -> str:
        """
        Compute deterministic hash of graph structure.

        Useful for caching and reproducibility checks.
        """
        # Create canonical representation
        canonical = {
            "id": self.id,
            "version": self.version,
            "nodes": sorted([
                (n.id, n.node_type.value, n.processor,
                 json.dumps(n.parameters, sort_keys=True))
                for n in self._nodes.values()
            ]),
            "edges": sorted([
                (e.source_node, e.source_port, e.target_node, e.target_port)
                for e in self._edges.values()
            ])
        }

        hash_input = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PipelineGraph(id='{self.id}', name='{self.name}', "
            f"nodes={self.num_nodes}, edges={self.num_edges})"
        )

    # --- Visualization Support ---

    def to_mermaid(self) -> str:
        """
        Generate Mermaid diagram representation.

        Returns:
            Mermaid flowchart syntax string
        """
        lines = ["flowchart TD"]

        # Add nodes with styling based on type
        for node in self._nodes.values():
            if node.node_type == NodeType.INPUT:
                lines.append(f"    {node.id}[/{node.name}/]")
            elif node.node_type == NodeType.OUTPUT:
                lines.append(f"    {node.id}[\\{node.name}\\]")
            elif node.node_type == NodeType.QC_GATE:
                lines.append(f"    {node.id}{{{node.name}}}")
            else:
                lines.append(f"    {node.id}[{node.name}]")

        # Add edges
        for edge in self._edges.values():
            label = edge.source_port if edge.source_port != "output" else ""
            if label:
                lines.append(f"    {edge.source_node} -->|{label}| {edge.target_node}")
            else:
                lines.append(f"    {edge.source_node} --> {edge.target_node}")

        return "\n".join(lines)
