"""
Pipeline Assembler for Event Intelligence Platform

Constructs pipeline DAGs from specifications, managing algorithm selection,
data flow connections, and dynamic pipeline composition.

Based on pipeline.schema.json and algorithm registry patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union
import logging
import yaml
import json

from core.analysis.assembly.graph import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    Port,
    QCGate,
    NodeType,
    DataType,
    EdgeType,
    NodeStatus,
    CycleDetectedError
)
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType as AlgorithmDataType,
    get_global_registry,
    load_default_algorithms
)

logger = logging.getLogger(__name__)


class AssemblyError(Exception):
    """Error during pipeline assembly."""
    pass


class InputNotFoundError(AssemblyError):
    """Referenced input not found."""
    pass


class AlgorithmNotFoundError(AssemblyError):
    """Algorithm not found in registry."""
    pass


class ConnectionError(AssemblyError):
    """Error connecting pipeline nodes."""
    pass


class TemporalRole(Enum):
    """Temporal role of pipeline inputs relative to event."""
    PRE_EVENT = "pre_event"
    POST_EVENT = "post_event"
    REFERENCE = "reference"
    ANY = "any"


@dataclass
class InputSpec:
    """
    Specification for a pipeline input.

    Attributes:
        name: Logical input name for reference in steps
        data_type: Type of input data
        source: Data source ID (e.g., sentinel1_grd)
        temporal_role: Temporal relationship to event
        required: Whether input is required
        description: Human-readable description
    """
    name: str
    data_type: DataType
    source: Optional[str] = None
    temporal_role: TemporalRole = TemporalRole.ANY
    required: bool = True
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.data_type.value,
            "source": self.source,
            "temporal_role": self.temporal_role.value,
            "required": self.required,
            "description": self.description
        }


@dataclass
class OutputSpec:
    """
    Specification for a pipeline output.

    Attributes:
        name: Output name
        data_type: Type of output data
        format: Output file format
        source_step: Step ID that produces this output
        description: Human-readable description
    """
    name: str
    data_type: DataType
    format: str = "geotiff"
    source_step: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": self.data_type.value,
            "format": self.format,
            "source_step": self.source_step,
            "description": self.description
        }


@dataclass
class StepSpec:
    """
    Specification for a pipeline processing step.

    Attributes:
        id: Unique step identifier
        processor: Algorithm ID from registry
        inputs: Input references (input_name or step_id.output_name)
        parameters: Algorithm parameter overrides
        outputs: Named outputs from this step
        qc_gate: Quality control configuration
    """
    id: str
    processor: str
    inputs: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    qc_gate: Optional[QCGate] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "processor": self.processor,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "qc_gate": self.qc_gate.to_dict() if self.qc_gate else None
        }


@dataclass
class PipelineSpec:
    """
    Complete pipeline specification.

    Compatible with pipeline.schema.json.
    """
    id: str
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    applicable_classes: List[str] = field(default_factory=list)
    inputs: List[InputSpec] = field(default_factory=list)
    steps: List[StepSpec] = field(default_factory=list)
    outputs: List[OutputSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "applicable_classes": self.applicable_classes,
            "inputs": [i.to_dict() for i in self.inputs],
            "steps": [s.to_dict() for s in self.steps],
            "outputs": [o.to_dict() for o in self.outputs],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineSpec':
        """Create from dictionary representation."""
        inputs = []
        for inp in data.get("inputs", []):
            inputs.append(InputSpec(
                name=inp["name"],
                data_type=DataType(inp["type"]),
                source=inp.get("source"),
                temporal_role=TemporalRole(inp.get("temporal_role", "any")),
                required=inp.get("required", True),
                description=inp.get("description")
            ))

        steps = []
        for step in data.get("steps", []):
            qc_gate = None
            if step.get("qc_gate"):
                qc_gate = QCGate(**step["qc_gate"])
            steps.append(StepSpec(
                id=step["id"],
                processor=step["processor"],
                inputs=step.get("inputs", []),
                parameters=step.get("parameters", {}),
                outputs=step.get("outputs", {}),
                qc_gate=qc_gate
            ))

        outputs = []
        for out in data.get("outputs", []):
            outputs.append(OutputSpec(
                name=out["name"],
                data_type=DataType(out["type"]),
                format=out.get("format", "geotiff"),
                source_step=out.get("source_step"),
                description=out.get("description")
            ))

        return cls(
            id=data["id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
            applicable_classes=data.get("applicable_classes", []),
            inputs=inputs,
            steps=steps,
            outputs=outputs,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'PipelineSpec':
        """Load from YAML file."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


@dataclass
class AssemblyResult:
    """
    Result of pipeline assembly.

    Attributes:
        graph: Assembled pipeline graph
        spec: Original pipeline specification
        algorithms_used: Map of step_id -> AlgorithmMetadata
        warnings: Assembly warnings
        assembly_time: Time taken to assemble
    """
    graph: PipelineGraph
    spec: PipelineSpec
    algorithms_used: Dict[str, AlgorithmMetadata]
    warnings: List[str] = field(default_factory=list)
    assembly_time: float = 0.0

    @property
    def success(self) -> bool:
        """Whether assembly was successful."""
        return self.graph is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "graph": self.graph.to_dict() if self.graph else None,
            "spec": self.spec.to_dict(),
            "algorithms_used": {
                step_id: algo.to_dict()
                for step_id, algo in self.algorithms_used.items()
            },
            "warnings": self.warnings,
            "assembly_time": self.assembly_time
        }


class PipelineAssembler:
    """
    Assembles pipeline graphs from specifications.

    Features:
    - Parse pipeline specifications from YAML/dict
    - Resolve algorithm references from registry
    - Build DAG with proper data flow connections
    - Validate step dependencies
    - Add QC gates where specified

    Usage:
        assembler = PipelineAssembler(registry)
        result = assembler.assemble(spec)
        if result.success:
            graph = result.graph
    """

    def __init__(
        self,
        registry: Optional[AlgorithmRegistry] = None,
        strict_mode: bool = True
    ):
        """
        Initialize pipeline assembler.

        Args:
            registry: Algorithm registry (uses global if None)
            strict_mode: If True, fail on missing algorithms; if False, warn
        """
        if registry is None:
            registry = get_global_registry()
            if not registry.algorithms:
                load_default_algorithms()

        self.registry = registry
        self.strict_mode = strict_mode

        logger.info(
            f"Initialized PipelineAssembler with {len(self.registry.algorithms)} algorithms"
        )

    def assemble(self, spec: PipelineSpec) -> AssemblyResult:
        """
        Assemble a pipeline graph from specification.

        Args:
            spec: Pipeline specification

        Returns:
            AssemblyResult containing graph, metadata, and any warnings
        """
        start_time = datetime.now(timezone.utc)
        warnings: List[str] = []
        algorithms_used: Dict[str, AlgorithmMetadata] = {}

        logger.info(f"Assembling pipeline: {spec.id} v{spec.version}")

        # Create graph
        graph = PipelineGraph(
            pipeline_id=spec.id,
            name=spec.name,
            version=spec.version,
            description=spec.description,
            applicable_classes=spec.applicable_classes
        )

        try:
            # Phase 1: Add input nodes
            input_names = self._add_input_nodes(graph, spec.inputs)

            # Phase 2: Add processing steps
            step_outputs = self._add_processing_steps(
                graph, spec.steps, input_names, algorithms_used, warnings
            )

            # Phase 3: Add output nodes
            self._add_output_nodes(graph, spec.outputs, step_outputs)

            # Phase 4: Validate graph
            self._validate_graph(graph)

            assembly_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.info(
                f"Assembled pipeline with {graph.num_nodes} nodes and {graph.num_edges} edges "
                f"in {assembly_time:.3f}s"
            )

            return AssemblyResult(
                graph=graph,
                spec=spec,
                algorithms_used=algorithms_used,
                warnings=warnings,
                assembly_time=assembly_time
            )

        except AssemblyError as e:
            logger.error(f"Pipeline assembly failed: {e}")
            raise

    def assemble_from_yaml(self, yaml_path: Path) -> AssemblyResult:
        """
        Assemble pipeline from YAML file.

        Args:
            yaml_path: Path to pipeline YAML file

        Returns:
            AssemblyResult
        """
        spec = PipelineSpec.from_yaml(yaml_path)
        return self.assemble(spec)

    def assemble_from_dict(self, data: Dict[str, Any]) -> AssemblyResult:
        """
        Assemble pipeline from dictionary.

        Args:
            data: Pipeline specification dictionary

        Returns:
            AssemblyResult
        """
        spec = PipelineSpec.from_dict(data)
        return self.assemble(spec)

    def _add_input_nodes(
        self,
        graph: PipelineGraph,
        inputs: List[InputSpec]
    ) -> Dict[str, str]:
        """
        Add input nodes to graph.

        Args:
            graph: Pipeline graph
            inputs: Input specifications

        Returns:
            Map of input name -> node ID
        """
        input_names: Dict[str, str] = {}

        for inp in inputs:
            node_id = f"input_{inp.name}"

            node = PipelineNode(
                id=node_id,
                name=inp.name,
                node_type=NodeType.INPUT,
                output_ports=[
                    Port(
                        name="output",
                        data_type=inp.data_type,
                        required=inp.required,
                        description=inp.description
                    )
                ],
                metadata={
                    "source": inp.source,
                    "temporal_role": inp.temporal_role.value,
                    "required": inp.required
                }
            )

            graph.add_node(node)
            input_names[inp.name] = node_id

            logger.debug(f"Added input node: {node_id}")

        return input_names

    def _add_processing_steps(
        self,
        graph: PipelineGraph,
        steps: List[StepSpec],
        input_names: Dict[str, str],
        algorithms_used: Dict[str, AlgorithmMetadata],
        warnings: List[str]
    ) -> Dict[str, Dict[str, str]]:
        """
        Add processing step nodes and edges to graph.

        Args:
            graph: Pipeline graph
            steps: Step specifications
            input_names: Map of input name -> node ID
            algorithms_used: Map to populate with algorithm metadata
            warnings: List to append warnings to

        Returns:
            Map of step_id -> {output_name -> full output reference}
        """
        step_outputs: Dict[str, Dict[str, str]] = {}

        for step in steps:
            # Resolve algorithm
            algorithm = self._resolve_algorithm(step.processor, warnings)
            if algorithm:
                algorithms_used[step.id] = algorithm

            # Create processor node
            node = self._create_processor_node(step, algorithm)
            graph.add_node(node)

            # Track step outputs
            step_outputs[step.id] = {
                port.name: f"{step.id}.{port.name}"
                for port in node.output_ports
            }

            # Connect inputs
            self._connect_step_inputs(
                graph, step, input_names, step_outputs
            )

            logger.debug(f"Added processing step: {step.id}")

        return step_outputs

    def _resolve_algorithm(
        self,
        processor_id: str,
        warnings: List[str]
    ) -> Optional[AlgorithmMetadata]:
        """
        Resolve algorithm from registry.

        Args:
            processor_id: Algorithm ID
            warnings: List to append warnings to

        Returns:
            AlgorithmMetadata or None if not found
        """
        algorithm = self.registry.get(processor_id)

        if algorithm is None:
            msg = f"Algorithm '{processor_id}' not found in registry"
            if self.strict_mode:
                raise AlgorithmNotFoundError(msg)
            else:
                warnings.append(msg)
                logger.warning(msg)

        return algorithm

    def _create_processor_node(
        self,
        step: StepSpec,
        algorithm: Optional[AlgorithmMetadata]
    ) -> PipelineNode:
        """
        Create processor node from step spec and algorithm metadata.

        Args:
            step: Step specification
            algorithm: Algorithm metadata (optional)

        Returns:
            PipelineNode for the processing step
        """
        # Determine input ports from algorithm or step inputs
        input_ports = []
        if algorithm:
            # Create ports for required data types
            for i, dt in enumerate(algorithm.required_data_types):
                input_ports.append(Port(
                    name=f"input_{i}",
                    data_type=DataType(dt.value) if hasattr(dt, 'value') else DataType.RASTER,
                    required=True
                ))
            for i, dt in enumerate(algorithm.optional_data_types):
                input_ports.append(Port(
                    name=f"optional_{i}",
                    data_type=DataType(dt.value) if hasattr(dt, 'value') else DataType.RASTER,
                    required=False
                ))
        else:
            # Create generic inputs based on step spec
            for i, input_ref in enumerate(step.inputs):
                input_ports.append(Port(
                    name=f"input_{i}",
                    data_type=DataType.RASTER,
                    required=True
                ))

        # Determine output ports
        output_ports = []
        if step.outputs:
            for name, output_type in step.outputs.items():
                output_ports.append(Port(
                    name=name,
                    data_type=DataType(output_type) if output_type in [dt.value for dt in DataType] else DataType.RASTER,
                    required=True
                ))
        elif algorithm and algorithm.outputs:
            for output_name in algorithm.outputs:
                output_ports.append(Port(
                    name=output_name,
                    data_type=DataType.RASTER,  # Default type
                    required=True
                ))
        else:
            # Default output
            output_ports.append(Port(
                name="output",
                data_type=DataType.RASTER,
                required=True
            ))

        # Merge parameters: algorithm defaults + step overrides
        parameters = {}
        if algorithm and algorithm.default_parameters:
            parameters.update(algorithm.default_parameters)
        parameters.update(step.parameters)

        # Build metadata
        metadata = {
            "processor": step.processor,
            "algorithm_version": algorithm.version if algorithm else None,
            "algorithm_category": algorithm.category.value if algorithm else None
        }

        return PipelineNode(
            id=step.id,
            name=algorithm.name if algorithm else step.processor,
            node_type=NodeType.PROCESSOR,
            processor=step.processor,
            parameters=parameters,
            input_ports=input_ports,
            output_ports=output_ports,
            qc_gate=step.qc_gate,
            metadata=metadata
        )

    def _connect_step_inputs(
        self,
        graph: PipelineGraph,
        step: StepSpec,
        input_names: Dict[str, str],
        step_outputs: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Connect step inputs to their sources.

        Args:
            graph: Pipeline graph
            step: Step specification
            input_names: Map of input name -> node ID
            step_outputs: Map of step_id -> {output_name -> reference}
        """
        for i, input_ref in enumerate(step.inputs):
            source_node, source_port = self._resolve_input_reference(
                input_ref, input_names, step_outputs
            )

            target_port = f"input_{i}"

            edge = PipelineEdge(
                source_node=source_node,
                source_port=source_port,
                target_node=step.id,
                target_port=target_port,
                edge_type=EdgeType.DATA
            )

            try:
                graph.add_edge(edge)
            except ValueError as e:
                raise ConnectionError(
                    f"Failed to connect {input_ref} to {step.id}.{target_port}: {e}"
                )

    def _resolve_input_reference(
        self,
        ref: str,
        input_names: Dict[str, str],
        step_outputs: Dict[str, Dict[str, str]]
    ) -> Tuple[str, str]:
        """
        Resolve an input reference to (node_id, port_name).

        Input references can be:
        - "input_name" -> refers to pipeline input
        - "step_id.output_name" -> refers to step output
        - "step_id" -> refers to default output of step

        Args:
            ref: Input reference string
            input_names: Map of input name -> node ID
            step_outputs: Map of step_id -> {output_name -> reference}

        Returns:
            Tuple of (node_id, port_name)
        """
        if "." in ref:
            # Step output reference
            parts = ref.split(".", 1)
            step_id = parts[0]
            output_name = parts[1]

            if step_id not in step_outputs:
                raise InputNotFoundError(
                    f"Referenced step '{step_id}' not found"
                )

            if output_name not in step_outputs[step_id]:
                # Try default output
                if "output" in step_outputs[step_id]:
                    output_name = "output"
                else:
                    raise InputNotFoundError(
                        f"Output '{output_name}' not found in step '{step_id}'"
                    )

            return step_id, output_name

        elif ref in input_names:
            # Pipeline input reference
            return input_names[ref], "output"

        elif ref in step_outputs:
            # Step reference without explicit output (use default)
            if "output" in step_outputs[ref]:
                return ref, "output"
            elif step_outputs[ref]:
                # Use first output
                first_output = list(step_outputs[ref].keys())[0]
                return ref, first_output
            else:
                raise InputNotFoundError(
                    f"Step '{ref}' has no outputs defined"
                )

        else:
            raise InputNotFoundError(
                f"Input reference '{ref}' not found in inputs or previous steps"
            )

    def _add_output_nodes(
        self,
        graph: PipelineGraph,
        outputs: List[OutputSpec],
        step_outputs: Dict[str, Dict[str, str]]
    ) -> None:
        """
        Add output nodes and connect them to sources.

        Args:
            graph: Pipeline graph
            outputs: Output specifications
            step_outputs: Map of step_id -> {output_name -> reference}
        """
        for out in outputs:
            node_id = f"output_{out.name}"

            node = PipelineNode(
                id=node_id,
                name=out.name,
                node_type=NodeType.OUTPUT,
                input_ports=[
                    Port(
                        name="input",
                        data_type=out.data_type,
                        required=True,
                        description=out.description
                    )
                ],
                metadata={
                    "format": out.format,
                    "source_step": out.source_step
                }
            )

            graph.add_node(node)

            # Connect to source step
            if out.source_step:
                source_node, source_port = self._resolve_output_source(
                    out.source_step, step_outputs
                )

                edge = PipelineEdge(
                    source_node=source_node,
                    source_port=source_port,
                    target_node=node_id,
                    target_port="input",
                    edge_type=EdgeType.DATA
                )

                try:
                    graph.add_edge(edge)
                except ValueError as e:
                    raise ConnectionError(
                        f"Failed to connect {out.source_step} to output {out.name}: {e}"
                    )

            logger.debug(f"Added output node: {node_id}")

    def _resolve_output_source(
        self,
        source_ref: str,
        step_outputs: Dict[str, Dict[str, str]]
    ) -> Tuple[str, str]:
        """
        Resolve output source reference.

        Args:
            source_ref: Source reference (step_id or step_id.output_name)
            step_outputs: Map of step_id -> {output_name -> reference}

        Returns:
            Tuple of (node_id, port_name)
        """
        if "." in source_ref:
            parts = source_ref.split(".", 1)
            step_id = parts[0]
            output_name = parts[1]

            if step_id not in step_outputs:
                raise InputNotFoundError(f"Source step '{step_id}' not found")

            return step_id, output_name

        else:
            if source_ref not in step_outputs:
                raise InputNotFoundError(f"Source step '{source_ref}' not found")

            # Use default output
            if "output" in step_outputs[source_ref]:
                return source_ref, "output"
            elif step_outputs[source_ref]:
                first_output = list(step_outputs[source_ref].keys())[0]
                return source_ref, first_output
            else:
                raise InputNotFoundError(
                    f"Source step '{source_ref}' has no outputs defined"
                )

    def _validate_graph(self, graph: PipelineGraph) -> None:
        """
        Validate assembled graph.

        Args:
            graph: Pipeline graph to validate

        Raises:
            AssemblyError: If validation fails
        """
        # Check for at least one input and output
        if not graph.get_input_nodes():
            raise AssemblyError("Pipeline has no input nodes")

        if not graph.get_output_nodes():
            raise AssemblyError("Pipeline has no output nodes")

        # Check connectivity - all outputs should be reachable from inputs
        input_ids = {n.id for n in graph.get_input_nodes()}
        output_ids = {n.id for n in graph.get_output_nodes()}

        for output_id in output_ids:
            ancestors = graph.get_ancestors(output_id)
            if not ancestors & input_ids:
                logger.warning(
                    f"Output '{output_id}' is not connected to any input"
                )

        # Validate topological order (will raise if cycle exists)
        try:
            graph.get_execution_order()
        except CycleDetectedError as e:
            raise AssemblyError(f"Pipeline contains a cycle: {e}")

        logger.info("Pipeline graph validation passed")


class DynamicAssembler(PipelineAssembler):
    """
    Dynamic pipeline assembler that can create pipelines on-the-fly.

    Extends PipelineAssembler with capabilities for:
    - Building pipelines from event class and available data
    - Automatic algorithm selection
    - Multi-algorithm ensemble creation
    """

    def create_for_event(
        self,
        event_class: str,
        available_data: Dict[str, DataType],
        max_algorithms: int = 1,
        require_deterministic: bool = False
    ) -> AssemblyResult:
        """
        Create a pipeline for an event class based on available data.

        Args:
            event_class: Event class (e.g., "flood.coastal")
            available_data: Map of data name -> data type
            max_algorithms: Maximum number of algorithms to use
            require_deterministic: Only use deterministic algorithms

        Returns:
            AssemblyResult with assembled pipeline
        """
        logger.info(f"Creating dynamic pipeline for event class: {event_class}")

        # Find suitable algorithms
        available_data_types = list(set(available_data.values()))
        algorithms = self.registry.search_by_requirements(
            event_type=event_class,
            available_data_types=[
                AlgorithmDataType(dt.value) if hasattr(dt, 'value') else dt
                for dt in available_data_types
            ],
            require_deterministic=require_deterministic
        )

        if not algorithms:
            raise AssemblyError(
                f"No algorithms found for event class '{event_class}' "
                f"with available data types: {[str(dt) for dt in available_data_types]}"
            )

        # Select top algorithms
        selected = algorithms[:max_algorithms]

        # Build pipeline spec
        spec = self._build_spec_from_algorithms(
            event_class, available_data, selected
        )

        return self.assemble(spec)

    def _build_spec_from_algorithms(
        self,
        event_class: str,
        available_data: Dict[str, DataType],
        algorithms: List[AlgorithmMetadata]
    ) -> PipelineSpec:
        """
        Build pipeline specification from selected algorithms.

        Args:
            event_class: Event class
            available_data: Available data map
            algorithms: Selected algorithms

        Returns:
            PipelineSpec
        """
        # Create inputs for all available data
        inputs = [
            InputSpec(
                name=name,
                data_type=data_type,
                required=True
            )
            for name, data_type in available_data.items()
        ]

        # Create steps for each algorithm
        steps = []
        for i, algo in enumerate(algorithms):
            # Match available data to algorithm requirements
            step_inputs = self._match_inputs_to_algorithm(
                available_data, algo
            )

            steps.append(StepSpec(
                id=f"step_{i}_{algo.id.replace('.', '_')}",
                processor=algo.id,
                inputs=step_inputs,
                outputs={"output": "raster"}
            ))

        # Create outputs for each step
        outputs = [
            OutputSpec(
                name=f"result_{i}",
                data_type=DataType.RASTER,
                format="cog",
                source_step=step.id
            )
            for i, step in enumerate(steps)
        ]

        return PipelineSpec(
            id=f"dynamic_{event_class.replace('.', '_')}",
            name=f"Dynamic Pipeline for {event_class}",
            version="1.0.0",
            description=f"Automatically assembled pipeline for {event_class}",
            applicable_classes=[event_class],
            inputs=inputs,
            steps=steps,
            outputs=outputs,
            metadata={
                "assembly_type": "dynamic",
                "event_class": event_class,
                "algorithms": [a.id for a in algorithms]
            }
        )

    def _match_inputs_to_algorithm(
        self,
        available_data: Dict[str, DataType],
        algorithm: AlgorithmMetadata
    ) -> List[str]:
        """
        Match available data to algorithm input requirements.

        Args:
            available_data: Available data map
            algorithm: Algorithm metadata

        Returns:
            List of input names that match algorithm requirements
        """
        matched = []

        for req_type in algorithm.required_data_types:
            req_type_value = req_type.value if hasattr(req_type, 'value') else req_type
            for name, data_type in available_data.items():
                data_type_value = data_type.value if hasattr(data_type, 'value') else data_type
                if req_type_value == data_type_value or self._types_compatible(req_type_value, data_type_value):
                    if name not in matched:
                        matched.append(name)
                        break

        return matched

    def _types_compatible(self, req_type: str, data_type: str) -> bool:
        """Check if data types are compatible."""
        # SAR and optical are both raster types
        if req_type in ("sar", "optical") and data_type == "raster":
            return True
        return req_type == data_type
