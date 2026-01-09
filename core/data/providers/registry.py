"""
Provider registry.

Central registry for all data providers with preference ordering,
capability metadata, and applicability rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import fnmatch


@dataclass
class Provider:
    """Data provider configuration."""

    # Identity
    id: str
    provider: str
    type: str  # optical, sar, dem, weather, ancillary

    # Capabilities
    capabilities: Dict[str, Any] = field(default_factory=dict)

    # Access configuration
    access: Dict[str, Any] = field(default_factory=dict)

    # Applicability rules
    applicability: Dict[str, Any] = field(default_factory=dict)

    # Cost information
    cost: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches_event_class(self, event_class: str) -> bool:
        """
        Check if provider is applicable to event class.

        Supports wildcard matching (e.g., "flood.*" matches "flood.coastal").
        """
        applicable_classes = self.applicability.get("event_classes", [])

        for pattern in applicable_classes:
            if fnmatch.fnmatch(event_class, pattern):
                return True

        return False


class ProviderRegistry:
    """
    Central registry for data providers.

    Manages provider configurations, preference ordering (open → restricted → commercial),
    and applicability rules.
    """

    def __init__(self):
        self.providers: Dict[str, Provider] = {}
        self._load_default_providers()

    def _load_default_providers(self):
        """Load default provider configurations."""
        # This will be populated with actual provider instances
        # For now, we initialize an empty registry
        pass

    def register(self, provider: Provider):
        """Register a provider."""
        self.providers[provider.id] = provider

    def get_provider(self, provider_id: str) -> Optional[Provider]:
        """Get provider by ID."""
        return self.providers.get(provider_id)

    def get_all_providers(self) -> List[Provider]:
        """Get all registered providers."""
        return list(self.providers.values())

    def get_applicable_providers(
        self,
        event_class: str,
        data_types: Optional[List[str]] = None
    ) -> List[Provider]:
        """
        Get providers applicable to event class and data types.

        Args:
            event_class: Event class (e.g., "flood.coastal")
            data_types: Optional list of data types to filter by

        Returns:
            List of applicable providers, ordered by preference
        """
        applicable = []

        for provider in self.providers.values():
            # Filter by data type if specified
            if data_types and provider.type not in data_types:
                continue

            # Check event class match
            if provider.matches_event_class(event_class):
                applicable.append(provider)

        # Sort by preference: open → open_restricted → commercial
        return self._sort_by_preference(applicable)

    def get_providers_by_type(self, data_type: str) -> List[Provider]:
        """Get all providers of a specific data type."""
        return [p for p in self.providers.values() if p.type == data_type]

    def _sort_by_preference(self, providers: List[Provider]) -> List[Provider]:
        """
        Sort providers by preference.

        Preference order:
        1. Open data sources (tier: "open")
        2. Open but restricted (tier: "open_restricted")
        3. Commercial (tier: "commercial")

        Within each tier, sort by provider-specific preference score.
        """
        tier_priority = {
            "open": 0,
            "open_restricted": 1,
            "commercial": 2
        }

        def sort_key(provider: Provider):
            tier = provider.cost.get("tier", "open")
            tier_rank = tier_priority.get(tier, 3)
            preference_score = provider.metadata.get("preference_score", 0.5)

            # Lower tier_rank is better, higher preference_score is better
            return (tier_rank, -preference_score)

        return sorted(providers, key=sort_key)

    def load_from_definitions(self, definitions_path: str):
        """
        Load provider definitions from YAML files.

        Args:
            definitions_path: Path to directory containing datasource YAML files
        """
        import os
        import yaml

        if not os.path.exists(definitions_path):
            return

        for filename in os.listdir(definitions_path):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(definitions_path, filename)

                try:
                    with open(filepath, 'r') as f:
                        data = yaml.safe_load(f)

                    # Extract datasource definition
                    datasource = data.get("datasource", {})

                    provider = Provider(
                        id=datasource.get("id", filename.replace(".yaml", "")),
                        provider=datasource.get("provider", "unknown"),
                        type=datasource.get("type", "unknown"),
                        capabilities=datasource.get("capabilities", {}),
                        access=datasource.get("access", {}),
                        applicability=datasource.get("applicability", {}),
                        cost=datasource.get("cost", {}),
                        metadata=datasource.get("metadata", {})
                    )

                    self.register(provider)

                except Exception as e:
                    print(f"Error loading provider definition {filename}: {e}")
