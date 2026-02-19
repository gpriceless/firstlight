"""
Dynamic OGC API Processes BaseProcessor factory.

For each algorithm in FirstLight's AlgorithmRegistry, generates a
pygeoapi BaseProcessor subclass at startup. The generated processors
expose vendor extension fields (x-firstlight-*) in their process
descriptions.

Also defines the JobPhase -> AlgorithmCategory mapping used by
state-based tool filtering (Task 4.11).
"""

import logging
import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Type

from core.analysis.library.registry import (
    AlgorithmCategory,
    AlgorithmMetadata,
    AlgorithmRegistry,
    get_global_registry,
)
from agents.orchestrator.state_model import JobPhase

logger = logging.getLogger(__name__)


# ============================================================================
# JobPhase -> AlgorithmCategory mapping for state-based tool filtering
# ============================================================================

PHASE_ALGORITHM_CATEGORIES: Dict[str, Set[AlgorithmCategory]] = {
    JobPhase.QUEUED.value: set(),
    JobPhase.DISCOVERING.value: set(),
    JobPhase.INGESTING.value: set(),
    JobPhase.NORMALIZING.value: set(),
    JobPhase.ANALYZING.value: {
        AlgorithmCategory.BASELINE,
        AlgorithmCategory.ADVANCED,
    },
    JobPhase.REPORTING.value: set(),
    JobPhase.COMPLETE.value: set(),
}


def get_algorithms_for_phase(
    phase: str,
    registry: Optional[AlgorithmRegistry] = None,
) -> List[AlgorithmMetadata]:
    """
    Return the algorithms available during a given job phase.

    Uses PHASE_ALGORITHM_CATEGORIES to filter by category. If the phase
    has no mapped categories, returns an empty list (NOT all algorithms).

    Args:
        phase: A JobPhase value string (e.g. "ANALYZING").
        registry: Optional registry instance; defaults to global.

    Returns:
        List of AlgorithmMetadata matching the phase's categories.
    """
    if registry is None:
        registry = get_global_registry()

    allowed_categories = PHASE_ALGORITHM_CATEGORIES.get(phase, set())
    if not allowed_categories:
        return []

    results: List[AlgorithmMetadata] = []
    for algo in registry.list_all():
        if algo.deprecated:
            continue
        if algo.category in allowed_categories:
            results.append(algo)
    return results


# ============================================================================
# OGC Processor factory
# ============================================================================

def _normalize_id(algorithm_id: str) -> str:
    """Normalize an algorithm ID to a valid OGC process identifier."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", algorithm_id)


def _build_inputs(algo: AlgorithmMetadata) -> Dict[str, Any]:
    """
    Build OGC process inputs from AlgorithmMetadata.parameter_schema.

    If the algorithm has a parameter_schema dict, convert each property
    to an OGC input descriptor. Otherwise return a generic AOI input.
    """
    inputs: Dict[str, Any] = {}

    # Always include AOI input
    inputs["aoi"] = {
        "title": "Area of Interest",
        "description": "GeoJSON geometry defining the analysis area",
        "schema": {"type": "object"},
        "minOccurs": 1,
        "maxOccurs": 1,
    }

    # Always include event_type
    inputs["event_type"] = {
        "title": "Event Type",
        "description": "Event type to analyze (e.g., flood, wildfire, storm)",
        "schema": {"type": "string"},
        "minOccurs": 1,
        "maxOccurs": 1,
    }

    # Add algorithm-specific parameters
    schema = algo.parameter_schema
    if isinstance(schema, dict) and "properties" in schema:
        for prop_name, prop_def in schema["properties"].items():
            inputs[prop_name] = {
                "title": prop_def.get("title", prop_name),
                "description": prop_def.get("description", ""),
                "schema": {
                    "type": prop_def.get("type", "string"),
                },
                "minOccurs": 1 if prop_name in schema.get("required", []) else 0,
                "maxOccurs": 1,
            }

    return inputs


def _build_outputs(algo: AlgorithmMetadata) -> Dict[str, Any]:
    """Build OGC process outputs from AlgorithmMetadata.outputs."""
    outputs: Dict[str, Any] = {}

    if algo.outputs:
        for i, output_name in enumerate(algo.outputs):
            outputs[_normalize_id(output_name)] = {
                "title": output_name,
                "description": f"Output: {output_name}",
                "schema": {
                    "type": "object",
                    "contentMediaType": "application/geo+json",
                },
            }
    else:
        # Default output
        outputs["result"] = {
            "title": "Analysis Result",
            "description": "Analysis output product",
            "schema": {
                "type": "object",
                "contentMediaType": "application/geo+json",
            },
        }

    return outputs


def _build_vendor_extensions(algo: AlgorithmMetadata) -> Dict[str, Any]:
    """Build x-firstlight-* vendor extension fields."""
    extensions: Dict[str, Any] = {
        "x-firstlight-category": algo.category.value,
    }

    # Resource requirements
    if algo.resources:
        res = {
            "memory_mb": algo.resources.memory_mb,
            "gpu_required": algo.resources.gpu_required,
            "gpu_memory_mb": algo.resources.gpu_memory_mb,
            "max_runtime_minutes": algo.resources.max_runtime_minutes,
            "distributed": algo.resources.distributed,
        }
        extensions["x-firstlight-resource-requirements"] = {
            k: v for k, v in res.items() if v is not None and v is not False
        }
        if not extensions["x-firstlight-resource-requirements"]:
            extensions["x-firstlight-resource-requirements"] = {}

    # Reasoning / confidence / escalation placeholders for LLM control
    extensions["x-firstlight-reasoning"] = {
        "supported": True,
        "description": (
            "LLM agents can attach reasoning and confidence scores "
            "to job transitions and parameter adjustments."
        ),
    }
    extensions["x-firstlight-confidence"] = {
        "supported": True,
        "field": "confidence",
        "range": [0.0, 1.0],
    }
    extensions["x-firstlight-escalation"] = {
        "supported": True,
        "severity_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
    }

    return extensions


def build_processor_config(algo: AlgorithmMetadata) -> Dict[str, Any]:
    """
    Build a pygeoapi-compatible processor configuration dict.

    This can be used to populate a pygeoapi config YAML or to
    dynamically register a processor.

    Args:
        algo: Algorithm metadata from the registry.

    Returns:
        Dict suitable for pygeoapi processor config.
    """
    proc_id = _normalize_id(algo.id)
    vendor = _build_vendor_extensions(algo)

    config = {
        "type": "process",
        "processor": {
            "name": f"core.ogc.processors.factory.FirstLightProcessor",
        },
        "id": proc_id,
        "title": algo.name,
        "description": algo.description or algo.name,
        "version": algo.version,
        "keywords": list(algo.event_types),
        "inputs": _build_inputs(algo),
        "outputs": _build_outputs(algo),
        "metadata": [
            {"role": "algorithm-id", "value": algo.id},
        ],
        "links": [],
        "example": {},
        **vendor,
    }

    return config


def build_processor_class(
    algo: AlgorithmMetadata,
) -> Optional[Type]:
    """
    Dynamically create a pygeoapi BaseProcessor subclass for an algorithm.

    The class is created via type() so that pygeoapi's processor loading
    can instantiate it. If pygeoapi is not installed, returns None.

    Args:
        algo: Algorithm metadata from the registry.

    Returns:
        A BaseProcessor subclass, or None if pygeoapi is unavailable.
    """
    try:
        from pygeoapi.process.base import BaseProcessor
    except ImportError:
        logger.debug("pygeoapi not installed, cannot build processor class")
        return None

    proc_id = _normalize_id(algo.id)
    vendor = _build_vendor_extensions(algo)

    # Build the process metadata dict that pygeoapi expects
    process_metadata = {
        "id": proc_id,
        "title": algo.name,
        "description": algo.description or algo.name,
        "version": algo.version,
        "keywords": list(algo.event_types),
        "inputs": _build_inputs(algo),
        "outputs": _build_outputs(algo),
        "links": [],
        "example": {},
        **vendor,
    }

    # Capture algo in closure
    _algo = algo
    _metadata = process_metadata

    class _DynamicProcessor(BaseProcessor):
        """Dynamically generated processor for {algo_name}."""

        def __init__(self, processor_def: Dict[str, Any]):
            # Merge our metadata into whatever pygeoapi passes
            merged = {**_metadata, **processor_def}
            super().__init__(merged, _metadata)
            self.algorithm_id = _algo.id
            self.algorithm_metadata = _algo

        def execute(self, data: Dict[str, Any]) -> Any:
            """
            Execute the algorithm.

            For async execution (Prefer: respond-async), this is called
            by a Taskiq worker. For sync, this runs inline.
            """
            # Import here to avoid circular deps
            from core.ogc.processors.factory import _execute_algorithm
            return _execute_algorithm(self.algorithm_id, data)

        def __repr__(self) -> str:
            return f"<FirstLightProcessor id={proc_id}>"

    # Give the class a meaningful name
    class_name = f"Processor_{proc_id}"
    _DynamicProcessor.__name__ = class_name
    _DynamicProcessor.__qualname__ = class_name
    _DynamicProcessor.__doc__ = (
        f"OGC Processor for {algo.name} ({algo.id})"
    )

    return _DynamicProcessor


def _execute_algorithm(algorithm_id: str, data: Dict[str, Any]) -> Any:
    """
    Stub execution entry point for an algorithm.

    In production this would delegate to the orchestrator or run
    the algorithm directly. For now, returns a status dict.
    """
    logger.info("OGC execute request for algorithm %s", algorithm_id)
    return (
        "application/json",
        {
            "status": "accepted",
            "algorithm_id": algorithm_id,
            "message": (
                "Algorithm execution enqueued. Use the job status "
                "endpoint to track progress."
            ),
        },
    )


def build_all_processors(
    registry: Optional[AlgorithmRegistry] = None,
) -> Dict[str, Type]:
    """
    Build processor classes for all non-deprecated algorithms.

    Args:
        registry: Optional registry instance; defaults to global.

    Returns:
        Dict mapping process id -> processor class.
    """
    if registry is None:
        registry = get_global_registry()

    processors: Dict[str, Type] = {}

    for algo in registry.list_all():
        if algo.deprecated:
            continue

        cls = build_processor_class(algo)
        if cls is not None:
            proc_id = _normalize_id(algo.id)
            processors[proc_id] = cls
            logger.info("Built OGC processor: %s -> %s", algo.id, proc_id)

    logger.info("Built %d OGC processors", len(processors))
    return processors


def get_processor_config(
    registry: Optional[AlgorithmRegistry] = None,
) -> Dict[str, Any]:
    """
    Build a complete pygeoapi resources config for all algorithms.

    Returns a dict suitable for the 'resources' section of a pygeoapi
    configuration YAML. Each algorithm becomes a process resource.

    Args:
        registry: Optional registry instance; defaults to global.

    Returns:
        Dict mapping process id -> pygeoapi resource config.
    """
    if registry is None:
        registry = get_global_registry()

    resources: Dict[str, Any] = {}

    for algo in registry.list_all():
        if algo.deprecated:
            continue

        proc_id = _normalize_id(algo.id)
        resources[proc_id] = build_processor_config(algo)

    return resources
