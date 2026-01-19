"""Event class registry for hierarchical taxonomy management.

Handles loading event class definitions from YAML files, wildcard matching,
and metadata lookup for event types.
"""

import fnmatch
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


class EventClass:
    """Represents a single event class in the taxonomy."""

    def __init__(
        self,
        class_path: str,
        description: str,
        keywords: Optional[List[str]] = None,
        required_data_types: Optional[List[str]] = None,
        optional_data_types: Optional[List[str]] = None,
        required_observables: Optional[List[str]] = None,
        optional_observables: Optional[List[str]] = None,
        pipelines: Optional[List[str]] = None,
        children: Optional[Dict[str, "EventClass"]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize an event class.

        Args:
            class_path: Full class path (e.g., 'flood.coastal.storm_surge')
            description: Human-readable description
            keywords: List of keywords for NLP matching
            required_data_types: Required data type categories
            optional_data_types: Optional data type categories
            required_observables: Required output observables
            optional_observables: Optional output observables
            pipelines: Applicable pipeline IDs
            children: Child event classes
            metadata: Additional metadata
        """
        self.class_path = class_path
        self.description = description
        self.keywords = keywords or []
        self.required_data_types = required_data_types or []
        self.optional_data_types = optional_data_types or []
        self.required_observables = required_observables or []
        self.optional_observables = optional_observables or []
        self.pipelines = pipelines or []
        self.children = children or {}
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"EventClass('{self.class_path}')"

    def matches_wildcard(self, pattern: str) -> bool:
        """Check if this class matches a wildcard pattern.

        Args:
            pattern: Wildcard pattern (e.g., 'flood.*', 'flood.coastal.*')

        Returns:
            True if class matches pattern
        """
        return fnmatch.fnmatch(self.class_path, pattern)

    def get_all_keywords(self, include_children: bool = True) -> Set[str]:
        """Get all keywords for this class and optionally its children.

        Args:
            include_children: Whether to include keywords from child classes

        Returns:
            Set of all keywords
        """
        keywords = set(self.keywords)
        if include_children:
            for child in self.children.values():
                keywords.update(child.get_all_keywords(include_children=True))
        return keywords

    def get_descendant_paths(self) -> List[str]:
        """Get all descendant class paths.

        Returns:
            List of all descendant class paths
        """
        paths = [self.class_path]
        for child in self.children.values():
            paths.extend(child.get_descendant_paths())
        return paths


class EventClassRegistry:
    """Registry for managing event class taxonomy."""

    def __init__(self, definitions_dir: Optional[Path] = None):
        """Initialize the registry.

        Args:
            definitions_dir: Directory containing event class YAML files.
                           Defaults to openspec/definitions/event_classes/
        """
        if definitions_dir is None:
            # Default to openspec/definitions/event_classes relative to project root
            project_root = Path(os.environ.get(
                "FIRSTLIGHT_ROOT",
                Path(__file__).parent.parent.parent
            ))
            definitions_dir = project_root / "openspec" / "definitions" / "event_classes"

        self.definitions_dir = Path(definitions_dir)
        self._classes: Dict[str, EventClass] = {}
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> list of class paths

        # Load all event class definitions
        self._load_definitions()

    def _load_definitions(self):
        """Load all event class definitions from YAML files."""
        if not self.definitions_dir.exists():
            raise FileNotFoundError(
                f"Event class definitions directory not found: {self.definitions_dir}"
            )

        for yaml_file in self.definitions_dir.glob("*.yaml"):
            self._load_definition_file(yaml_file)

        # Build keyword index after all classes are loaded
        self._build_keyword_index()

    def _load_definition_file(self, file_path: Path):
        """Load a single event class definition file.

        Args:
            file_path: Path to YAML definition file
        """
        with open(file_path) as f:
            definition = yaml.safe_load(f)

        root_class = definition.get("class")
        if not root_class:
            raise ValueError(f"Missing 'class' field in {file_path}")

        # Parse taxonomy tree
        taxonomy = definition.get("taxonomy", {})
        if root_class in taxonomy:
            self._parse_taxonomy_node(root_class, taxonomy[root_class], root_class)

    def _parse_taxonomy_node(
        self, name: str, node: Dict[str, Any], parent_path: Optional[str] = None
    ) -> EventClass:
        """Recursively parse a taxonomy node.

        Args:
            name: Node name
            node: Node data from YAML
            parent_path: Parent class path

        Returns:
            Parsed EventClass
        """
        # Build class path
        class_path = f"{parent_path}.{name}" if parent_path and parent_path != name else name

        # Parse children first
        children_data = node.get("children", {})
        children = {}
        for child_name, child_node in children_data.items():
            child_class = self._parse_taxonomy_node(child_name, child_node, class_path)
            children[child_name] = child_class

        # Create event class
        event_class = EventClass(
            class_path=class_path,
            description=node.get("description", ""),
            keywords=node.get("keywords", []),
            required_data_types=node.get("required_data_types", []),
            optional_data_types=node.get("optional_data_types", []),
            required_observables=node.get("required_observables", []),
            optional_observables=node.get("optional_observables", []),
            pipelines=node.get("pipelines", []),
            children=children,
        )

        # Register this class
        self._classes[class_path] = event_class

        return event_class

    def _build_keyword_index(self):
        """Build keyword -> class_path index for fast lookup."""
        self._keyword_index = {}
        for class_path, event_class in self._classes.items():
            for keyword in event_class.keywords:
                keyword_lower = keyword.lower()
                if keyword_lower not in self._keyword_index:
                    self._keyword_index[keyword_lower] = []
                self._keyword_index[keyword_lower].append(class_path)

    def get_class(self, class_path: str) -> Optional[EventClass]:
        """Get an event class by its path.

        Args:
            class_path: Class path (e.g., 'flood.coastal.storm_surge')

        Returns:
            EventClass if found, None otherwise
        """
        return self._classes.get(class_path)

    def get_classes_by_wildcard(self, pattern: str) -> List[EventClass]:
        """Get all classes matching a wildcard pattern.

        Args:
            pattern: Wildcard pattern (e.g., 'flood.*', '*.storm_surge')

        Returns:
            List of matching EventClass instances
        """
        matching_classes = []
        for class_path, event_class in self._classes.items():
            if event_class.matches_wildcard(pattern):
                matching_classes.append(event_class)
        return matching_classes

    def get_classes_by_keyword(self, keyword: str) -> List[EventClass]:
        """Get event classes that match a keyword.

        Args:
            keyword: Keyword to search for (case-insensitive)

        Returns:
            List of matching EventClass instances
        """
        keyword_lower = keyword.lower()
        class_paths = self._keyword_index.get(keyword_lower, [])
        return [self._classes[path] for path in class_paths]

    def search_by_keywords(self, keywords: List[str]) -> Dict[str, int]:
        """Search for classes by multiple keywords and rank by matches.

        Args:
            keywords: List of keywords to search for

        Returns:
            Dictionary mapping class_path -> match_count, sorted by match count
        """
        match_counts: Dict[str, int] = {}

        for keyword in keywords:
            matching_classes = self.get_classes_by_keyword(keyword)
            for event_class in matching_classes:
                if event_class.class_path not in match_counts:
                    match_counts[event_class.class_path] = 0
                match_counts[event_class.class_path] += 1

        return dict(sorted(match_counts.items(), key=lambda x: x[1], reverse=True))

    def list_all_classes(self) -> List[str]:
        """Get list of all registered class paths.

        Returns:
            List of class paths
        """
        return sorted(self._classes.keys())

    def get_root_classes(self) -> List[EventClass]:
        """Get top-level event classes.

        Returns:
            List of root EventClass instances
        """
        root_classes = []
        for class_path, event_class in self._classes.items():
            if "." not in class_path:
                root_classes.append(event_class)
        return root_classes


# Global registry instance
_registry: Optional[EventClassRegistry] = None


def get_registry() -> EventClassRegistry:
    """Get the global event class registry instance.

    Returns:
        EventClassRegistry singleton
    """
    global _registry
    if _registry is None:
        _registry = EventClassRegistry()
    return _registry


if __name__ == "__main__":
    # Demo usage
    registry = get_registry()

    print("Available event classes:")
    for class_path in registry.list_all_classes():
        event_class = registry.get_class(class_path)
        print(f"  {class_path}: {event_class.description}")

    print("\nFlood classes (using wildcard 'flood.*'):")
    for event_class in registry.get_classes_by_wildcard("flood.*"):
        print(f"  {event_class.class_path}")

    print("\nClasses matching keyword 'storm surge':")
    for event_class in registry.get_classes_by_keyword("storm surge"):
        print(f"  {event_class.class_path}")
