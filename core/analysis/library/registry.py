"""
Algorithm Registry for Event Intelligence Platform

Manages algorithm metadata, version tracking, requirement specifications,
and algorithm discovery for dynamic pipeline assembly.

Based on OPENSPEC.md algorithm registry specification.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlgorithmCategory(Enum):
    """Algorithm maturity categories"""
    BASELINE = "baseline"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


class DataType(Enum):
    """Data type categories"""
    OPTICAL = "optical"
    SAR = "sar"
    DEM = "dem"
    WEATHER = "weather"
    ANCILLARY = "ancillary"
    FORECAST = "forecast"


@dataclass
class ResourceRequirements:
    """Computational resource requirements"""
    memory_mb: Optional[int] = None
    gpu_required: bool = False
    gpu_memory_mb: Optional[int] = None
    max_runtime_minutes: Optional[int] = None
    distributed: bool = False


@dataclass
class ValidationMetrics:
    """Algorithm validation statistics"""
    accuracy_min: Optional[float] = None
    accuracy_max: Optional[float] = None
    accuracy_median: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    validated_regions: List[str] = field(default_factory=list)
    validation_dataset_count: int = 0
    last_validated: Optional[str] = None


@dataclass
class AlgorithmMetadata:
    """
    Complete metadata for an algorithm in the registry.

    Based on OPENSPEC.md algorithm_registry specification:
    - Unique ID with hierarchical naming
    - Category (baseline/advanced/experimental)
    - Event type patterns (with wildcard support)
    - Data requirements
    - Computational requirements
    - Validation metrics
    - Version tracking
    """

    # Identification
    id: str  # e.g., "flood.baseline.threshold_sar"
    name: str  # Human-readable name
    category: AlgorithmCategory
    version: str

    # Event type matching
    event_types: List[str]  # Patterns like "flood.*", "flood.coastal"

    # Data requirements
    required_data_types: List[DataType]
    optional_data_types: List[DataType] = field(default_factory=list)
    required_bands: List[str] = field(default_factory=list)
    required_polarizations: List[str] = field(default_factory=list)

    # Computational requirements
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)

    # Validation and accuracy
    validation: Optional[ValidationMetrics] = None

    # Determinism
    deterministic: bool = True
    seed_required: bool = False

    # Description and references
    description: Optional[str] = None
    references: List[str] = field(default_factory=list)

    # Execution
    module_path: Optional[str] = None  # Python module path
    class_name: Optional[str] = None  # Class name for instantiation

    # Parameters
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    parameter_schema: Dict[str, Any] = field(default_factory=dict)

    # Outputs
    outputs: List[str] = field(default_factory=list)

    # Metadata
    added_date: Optional[str] = None
    last_updated: Optional[str] = None
    deprecated: bool = False
    replacement_algorithm: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        result = asdict(self)
        # Convert enums to strings
        result['category'] = self.category.value
        result['required_data_types'] = [dt.value for dt in self.required_data_types]
        result['optional_data_types'] = [dt.value for dt in self.optional_data_types]
        return result

    def matches_event_type(self, event_type: str) -> bool:
        """
        Check if algorithm supports the given event type.
        Supports wildcard matching (e.g., "flood.*" matches "flood.coastal")
        """
        for pattern in self.event_types:
            if self._matches_pattern(event_type, pattern):
                return True
        return False

    @staticmethod
    def _matches_pattern(value: str, pattern: str) -> bool:
        """
        Match value against pattern with wildcard support.

        Examples:
            "flood.coastal" matches "flood.*"
            "flood.coastal" matches "flood.coastal"
            "flood.coastal.storm_surge" matches "flood.*"
            "flood.coastal.storm_surge" matches "flood.coastal.*"
        """
        if pattern == value:
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return value.startswith(prefix + ".")

        return False


class AlgorithmRegistry:
    """
    Central registry for all analysis algorithms.

    Features:
    - Load algorithms from YAML definitions
    - Search by event type, data availability, requirements
    - Version tracking and compatibility checking
    - Algorithm discovery and filtering
    """

    def __init__(self):
        self.algorithms: Dict[str, AlgorithmMetadata] = {}
        self._index_by_event_type: Dict[str, Set[str]] = {}
        self._index_by_category: Dict[AlgorithmCategory, Set[str]] = {
            cat: set() for cat in AlgorithmCategory
        }

    def register(self, algorithm: AlgorithmMetadata) -> None:
        """
        Register an algorithm in the registry.

        Args:
            algorithm: Algorithm metadata to register
        """
        if algorithm.id in self.algorithms:
            logger.warning(f"Algorithm {algorithm.id} already registered, overwriting")

        self.algorithms[algorithm.id] = algorithm

        # Update indexes
        for event_type in algorithm.event_types:
            if event_type not in self._index_by_event_type:
                self._index_by_event_type[event_type] = set()
            self._index_by_event_type[event_type].add(algorithm.id)

        self._index_by_category[algorithm.category].add(algorithm.id)

        logger.info(f"Registered algorithm: {algorithm.id} v{algorithm.version}")

    def get(self, algorithm_id: str) -> Optional[AlgorithmMetadata]:
        """Get algorithm by ID"""
        return self.algorithms.get(algorithm_id)

    def list_all(self) -> List[AlgorithmMetadata]:
        """Get all registered algorithms"""
        return list(self.algorithms.values())

    def list_by_category(self, category: AlgorithmCategory) -> List[AlgorithmMetadata]:
        """Get all algorithms in a category"""
        return [
            self.algorithms[algo_id]
            for algo_id in self._index_by_category[category]
        ]

    def search_by_event_type(self, event_type: str) -> List[AlgorithmMetadata]:
        """
        Find algorithms supporting the given event type.
        Handles wildcard matching.

        Args:
            event_type: Event type string (e.g., "flood.coastal")

        Returns:
            List of matching algorithms
        """
        matches = []
        for algorithm in self.algorithms.values():
            if algorithm.matches_event_type(event_type):
                matches.append(algorithm)
        return matches

    def search_by_data_availability(
        self,
        available_data_types: List[DataType],
        event_type: Optional[str] = None
    ) -> List[AlgorithmMetadata]:
        """
        Find algorithms that can run with available data.

        Args:
            available_data_types: List of available data types
            event_type: Optional event type filter

        Returns:
            List of algorithms with all required data available
        """
        available_set = set(available_data_types)
        matches = []

        for algorithm in self.algorithms.values():
            # Check event type if specified
            if event_type and not algorithm.matches_event_type(event_type):
                continue

            # Check if all required data types are available
            required_set = set(algorithm.required_data_types)
            if required_set.issubset(available_set):
                matches.append(algorithm)

        return matches

    def search_by_requirements(
        self,
        event_type: Optional[str] = None,
        available_data_types: Optional[List[DataType]] = None,
        max_memory_mb: Optional[int] = None,
        gpu_available: bool = False,
        require_deterministic: bool = False
    ) -> List[AlgorithmMetadata]:
        """
        Advanced search with multiple filters.

        Args:
            event_type: Event type pattern
            available_data_types: Available data types
            max_memory_mb: Maximum available memory
            gpu_available: Whether GPU is available
            require_deterministic: Only return deterministic algorithms

        Returns:
            List of matching algorithms
        """
        matches = []

        for algorithm in self.algorithms.values():
            # Skip deprecated
            if algorithm.deprecated:
                continue

            # Event type filter
            if event_type and not algorithm.matches_event_type(event_type):
                continue

            # Data availability filter
            if available_data_types:
                available_set = set(available_data_types)
                required_set = set(algorithm.required_data_types)
                if not required_set.issubset(available_set):
                    continue

            # Memory constraint
            if max_memory_mb and algorithm.resources.memory_mb:
                if algorithm.resources.memory_mb > max_memory_mb:
                    continue

            # GPU requirement
            if algorithm.resources.gpu_required and not gpu_available:
                continue

            # Determinism requirement
            if require_deterministic and not algorithm.deterministic:
                continue

            matches.append(algorithm)

        return matches

    def load_from_yaml(self, yaml_path: Path) -> None:
        """
        Load algorithm definitions from YAML file.

        Expected format:
        algorithms:
          flood.baseline.threshold_sar:
            name: "SAR Backscatter Threshold"
            category: "baseline"
            version: "1.0.0"
            ...
        """
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            if 'algorithms' not in data:
                logger.error(f"No 'algorithms' key in {yaml_path}")
                return

            for algo_id, algo_def in data['algorithms'].items():
                # Parse data types
                required_data_types = [
                    DataType(dt) for dt in algo_def.get('required_data_types', [])
                ]
                optional_data_types = [
                    DataType(dt) for dt in algo_def.get('optional_data_types', [])
                ]

                # Parse resources
                resources_dict = algo_def.get('resources', {})
                resources = ResourceRequirements(
                    memory_mb=resources_dict.get('memory_mb'),
                    gpu_required=resources_dict.get('gpu_required', False),
                    gpu_memory_mb=resources_dict.get('gpu_memory_mb'),
                    max_runtime_minutes=resources_dict.get('max_runtime_minutes'),
                    distributed=resources_dict.get('distributed', False)
                )

                # Parse validation
                validation_dict = algo_def.get('validation', {})
                validation = None
                if validation_dict:
                    validation = ValidationMetrics(
                        accuracy_min=validation_dict.get('accuracy_min'),
                        accuracy_max=validation_dict.get('accuracy_max'),
                        accuracy_median=validation_dict.get('accuracy_median'),
                        precision=validation_dict.get('precision'),
                        recall=validation_dict.get('recall'),
                        f1_score=validation_dict.get('f1_score'),
                        validated_regions=validation_dict.get('validated_regions', []),
                        validation_dataset_count=validation_dict.get('validation_dataset_count', 0),
                        last_validated=validation_dict.get('last_validated')
                    )

                # Create metadata
                metadata = AlgorithmMetadata(
                    id=algo_id,
                    name=algo_def['name'],
                    category=AlgorithmCategory(algo_def['category']),
                    version=algo_def['version'],
                    event_types=algo_def['event_types'],
                    required_data_types=required_data_types,
                    optional_data_types=optional_data_types,
                    required_bands=algo_def.get('required_bands', []),
                    required_polarizations=algo_def.get('required_polarizations', []),
                    resources=resources,
                    validation=validation,
                    deterministic=algo_def.get('deterministic', True),
                    seed_required=algo_def.get('seed_required', False),
                    description=algo_def.get('description'),
                    references=algo_def.get('references', []),
                    module_path=algo_def.get('module_path'),
                    class_name=algo_def.get('class_name'),
                    default_parameters=algo_def.get('default_parameters', {}),
                    parameter_schema=algo_def.get('parameter_schema', {}),
                    outputs=algo_def.get('outputs', []),
                    added_date=algo_def.get('added_date'),
                    last_updated=algo_def.get('last_updated'),
                    deprecated=algo_def.get('deprecated', False),
                    replacement_algorithm=algo_def.get('replacement_algorithm')
                )

                self.register(metadata)

            logger.info(f"Loaded {len(data['algorithms'])} algorithms from {yaml_path}")

        except Exception as e:
            logger.error(f"Error loading algorithms from {yaml_path}: {e}")
            raise

    def load_from_directory(self, directory: Path, recursive: bool = True) -> None:
        """
        Load all YAML algorithm definitions from a directory.

        Args:
            directory: Directory to search
            recursive: Whether to search subdirectories
        """
        pattern = "**/*.yaml" if recursive else "*.yaml"

        for yaml_file in directory.glob(pattern):
            if yaml_file.name.startswith('.'):
                continue

            logger.info(f"Loading algorithms from {yaml_file}")
            try:
                self.load_from_yaml(yaml_file)
            except Exception as e:
                logger.error(f"Failed to load {yaml_file}: {e}")

    def export_to_yaml(self, output_path: Path) -> None:
        """Export all algorithms to YAML file"""
        export_data = {
            'algorithms': {
                algo_id: algo.to_dict()
                for algo_id, algo in self.algorithms.items()
            }
        }

        with open(output_path, 'w') as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported {len(self.algorithms)} algorithms to {output_path}")

    def list_tool_schemas(self, exclude_deprecated: bool = True) -> List[Dict[str, Any]]:
        """
        List OpenAI-compatible function-calling schemas for all algorithms.

        Args:
            exclude_deprecated: If True, exclude deprecated algorithms.

        Returns:
            List of tool schema dicts with name, description, parameters.
        """
        import re

        schemas = []
        for algo in self.algorithms.values():
            if exclude_deprecated and algo.deprecated:
                continue

            # Normalize algorithm ID to valid function name
            func_name = re.sub(r"[^a-zA-Z0-9_]", "_", algo.id)

            # Build description
            desc_parts = []
            if algo.description:
                desc_parts.append(algo.description)
            if algo.event_types:
                desc_parts.append(f"Supports event types: {', '.join(algo.event_types)}")
            if algo.category:
                desc_parts.append(f"Category: {algo.category.value}")
            description = ". ".join(desc_parts) if desc_parts else algo.name

            # Build parameters
            parameters = algo.parameter_schema if algo.parameter_schema else {
                "type": "object",
                "properties": {},
            }
            if "type" not in parameters:
                parameters["type"] = "object"

            schemas.append({
                "name": func_name,
                "description": description,
                "parameters": parameters,
            })

        return schemas

    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'total_algorithms': len(self.algorithms),
            'by_category': {
                cat.value: len(self._index_by_category[cat])
                for cat in AlgorithmCategory
            },
            'deprecated': sum(1 for algo in self.algorithms.values() if algo.deprecated),
            'deterministic': sum(1 for algo in self.algorithms.values() if algo.deterministic),
            'gpu_required': sum(
                1 for algo in self.algorithms.values()
                if algo.resources.gpu_required
            )
        }


# Global registry instance
_global_registry: Optional[AlgorithmRegistry] = None


def get_global_registry() -> AlgorithmRegistry:
    """Get or create the global algorithm registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = AlgorithmRegistry()
    return _global_registry


def load_default_algorithms() -> AlgorithmRegistry:
    """
    Load algorithms from default locations.

    Searches:
    1. openspec/definitions/algorithms/
    2. core/analysis/library/definitions/
    """
    registry = get_global_registry()

    # Try standard locations
    base_paths = [
        Path(__file__).parent.parent.parent.parent / "openspec" / "definitions" / "algorithms",
        Path(__file__).parent / "definitions"
    ]

    for path in base_paths:
        if path.exists():
            logger.info(f"Loading algorithms from {path}")
            registry.load_from_directory(path)

    return registry


if __name__ == "__main__":
    # CLI interface for testing
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Algorithm Registry CLI")
    parser.add_argument("--list-algorithms", action="store_true", help="List all algorithms")
    parser.add_argument("--event-type", type=str, help="Search by event type")
    parser.add_argument("--stats", action="store_true", help="Show registry statistics")
    parser.add_argument("--load", type=str, help="Load algorithms from YAML file")

    args = parser.parse_args()

    registry = get_global_registry()

    if args.load:
        registry.load_from_yaml(Path(args.load))
    else:
        load_default_algorithms()

    if args.list_algorithms:
        print("\nRegistered Algorithms:")
        print("=" * 80)
        for algo in registry.list_all():
            print(f"\n{algo.id} (v{algo.version})")
            print(f"  Name: {algo.name}")
            print(f"  Category: {algo.category.value}")
            print(f"  Event Types: {', '.join(algo.event_types)}")
            print(f"  Required Data: {[dt.value for dt in algo.required_data_types]}")
            if algo.deprecated:
                print(f"  [DEPRECATED]")

    if args.event_type:
        matches = registry.search_by_event_type(args.event_type)
        print(f"\nAlgorithms for event type '{args.event_type}':")
        print("=" * 80)
        for algo in matches:
            print(f"  - {algo.id} (v{algo.version}): {algo.name}")

    if args.stats:
        stats = registry.get_statistics()
        print("\nRegistry Statistics:")
        print("=" * 80)
        for key, value in stats.items():
            print(f"  {key}: {value}")
