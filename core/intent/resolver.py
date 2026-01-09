"""Intent resolution orchestration.

Combines registry lookup, NLP classification, and user overrides to resolve
event intent from natural language or explicit specifications.
"""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, Optional

from .classifier import EventClassifier, get_classifier
from .registry import EventClassRegistry, get_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentResolution:
    """Represents a resolved intent."""

    def __init__(
        self,
        resolved_class: str,
        source: str,  # 'explicit' or 'inferred'
        confidence: float,
        original_input: Optional[str] = None,
        inferred_class: Optional[str] = None,
        inferred_confidence: Optional[float] = None,
        alternatives: Optional[list] = None,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize intent resolution.

        Args:
            resolved_class: Final resolved event class
            source: Resolution source ('explicit' or 'inferred')
            confidence: Final confidence score
            original_input: Original natural language input (if any)
            inferred_class: NLP-inferred class (if applicable)
            inferred_confidence: NLP inference confidence
            alternatives: Alternative class suggestions
            parameters: Extracted parameters
            metadata: Additional metadata
        """
        self.resolved_class = resolved_class
        self.source = source
        self.confidence = confidence
        self.original_input = original_input
        self.inferred_class = inferred_class
        self.inferred_confidence = inferred_confidence
        self.alternatives = alternatives or []
        self.parameters = parameters or {}
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format matching intent schema.

        Returns:
            Dictionary representation
        """
        result = {
            "input": {},
            "output": {
                "resolved_class": self.resolved_class,
                "source": self.source,
                "confidence": self.confidence,
            },
        }

        # Add input section
        if self.original_input:
            result["input"]["natural_language"] = self.original_input
        if self.source == "explicit":
            result["input"]["explicit_class"] = self.resolved_class

        # Add resolution section if inferred
        if self.source == "inferred" and self.inferred_class:
            result["resolution"] = {
                "inferred_class": self.inferred_class,
                "confidence": self.inferred_confidence,
            }
            if self.alternatives:
                result["resolution"]["alternatives"] = self.alternatives

        # Add parameters if any
        if self.parameters:
            result["output"]["parameters"] = self.parameters

        return result

    def __repr__(self) -> str:
        return (
            f"IntentResolution(class='{self.resolved_class}', "
            f"source='{self.source}', confidence={self.confidence:.2f})"
        )


class IntentResolver:
    """Resolves event intent from various input formats."""

    def __init__(
        self,
        registry: Optional[EventClassRegistry] = None,
        classifier: Optional[EventClassifier] = None,
    ):
        """Initialize the resolver.

        Args:
            registry: Event class registry
            classifier: Event classifier
        """
        self.registry = registry or get_registry()
        self.classifier = classifier or get_classifier()

    def resolve(
        self,
        natural_language: Optional[str] = None,
        explicit_class: Optional[str] = None,
        allow_override: bool = True,
    ) -> Optional[IntentResolution]:
        """Resolve intent from natural language or explicit class.

        Args:
            natural_language: Natural language event description
            explicit_class: Explicitly specified event class
            allow_override: If True, explicit_class overrides NLP inference

        Returns:
            IntentResolution if successful, None otherwise
        """
        if not natural_language and not explicit_class:
            logger.error("Either natural_language or explicit_class must be provided")
            return None

        # Handle explicit class specification
        if explicit_class and allow_override:
            return self._resolve_explicit(explicit_class, natural_language)

        # Handle natural language inference
        if natural_language:
            return self._resolve_natural_language(natural_language, explicit_class)

        # Fallback to explicit if NLP failed
        if explicit_class:
            return self._resolve_explicit(explicit_class, None)

        return None

    def _resolve_explicit(
        self,
        explicit_class: str,
        original_input: Optional[str] = None
    ) -> Optional[IntentResolution]:
        """Resolve from explicit class specification.

        Args:
            explicit_class: Explicit event class path
            original_input: Original natural language input (for logging)

        Returns:
            IntentResolution if class is valid, None otherwise
        """
        # Validate that class exists in registry
        event_class = self.registry.get_class(explicit_class)

        if not event_class:
            logger.warning(
                f"Explicit class '{explicit_class}' not found in registry. "
                f"Available classes: {self.registry.list_all_classes()[:5]}..."
            )
            return None

        logger.info(f"Resolved to explicit class: {explicit_class}")

        # Extract parameters from class metadata if available
        parameters = self._extract_parameters(event_class)

        return IntentResolution(
            resolved_class=explicit_class,
            source="explicit",
            confidence=1.0,
            original_input=original_input,
            parameters=parameters,
            metadata={
                "resolved_at": datetime.now(UTC).isoformat(),
                "resolution_method": "explicit",
            },
        )

    def _resolve_natural_language(
        self,
        natural_language: str,
        hint_class: Optional[str] = None
    ) -> Optional[IntentResolution]:
        """Resolve from natural language description.

        Args:
            natural_language: Natural language event description
            hint_class: Optional hint for class (not enforced)

        Returns:
            IntentResolution if classification successful, None otherwise
        """
        logger.info(f"Classifying natural language: '{natural_language}'")

        # Use NLP classifier
        classification = self.classifier.classify(natural_language)

        if not classification:
            logger.warning(f"Failed to classify: '{natural_language}'")
            return None

        # Validate inferred class exists
        event_class = self.registry.get_class(classification.inferred_class)
        if not event_class:
            logger.error(
                f"Classifier returned invalid class: {classification.inferred_class}"
            )
            return None

        logger.info(
            f"Inferred class: {classification.inferred_class} "
            f"(confidence: {classification.confidence:.2f})"
        )

        # Extract parameters
        parameters = self._extract_parameters(event_class)
        parameters.update(self._extract_parameters_from_text(natural_language))

        return IntentResolution(
            resolved_class=classification.inferred_class,
            source="inferred",
            confidence=classification.confidence,
            original_input=natural_language,
            inferred_class=classification.inferred_class,
            inferred_confidence=classification.confidence,
            alternatives=classification.alternatives,
            parameters=parameters,
            metadata={
                "resolved_at": datetime.now(UTC).isoformat(),
                "resolution_method": "nlp_classification",
                "matched_keywords": classification.matched_keywords,
            },
        )

    def _extract_parameters(self, event_class) -> Dict[str, Any]:
        """Extract event-specific parameters from class definition.

        Args:
            event_class: EventClass instance

        Returns:
            Dictionary of parameters
        """
        # For now, extract basic info from class path
        parameters = {}

        # Parse class path to extract hierarchy
        parts = event_class.class_path.split(".")
        if len(parts) >= 2:
            parameters["event_type"] = parts[0]  # e.g., 'flood'
            parameters["event_subtype"] = parts[1]  # e.g., 'coastal'

        if len(parts) >= 3:
            parameters["event_variant"] = parts[2]  # e.g., 'storm_surge'

        return parameters

    def _extract_parameters_from_text(self, text: str) -> Dict[str, Any]:
        """Extract additional parameters from natural language text.

        Args:
            text: Natural language text

        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        text_lower = text.lower()

        # Extract temporal indicators
        if "after" in text_lower or "following" in text_lower:
            parameters["temporal_context"] = "post_event"
        elif "during" in text_lower:
            parameters["temporal_context"] = "active"
        elif "before" in text_lower or "forecasted" in text_lower:
            parameters["temporal_context"] = "pre_event"

        # Extract causation hints
        if "hurricane" in text_lower or "tropical storm" in text_lower:
            parameters["causation"] = "tropical_cyclone"
        elif "tornado" in text_lower:
            parameters["causation"] = "tornado"

        return parameters


# Global resolver instance
_resolver: Optional[IntentResolver] = None


def get_resolver() -> IntentResolver:
    """Get the global intent resolver instance.

    Returns:
        IntentResolver singleton
    """
    global _resolver
    if _resolver is None:
        _resolver = IntentResolver()
    return _resolver


def resolve_intent(
    natural_language: Optional[str] = None,
    explicit_class: Optional[str] = None,
) -> Optional[IntentResolution]:
    """Resolve event intent.

    Args:
        natural_language: Natural language event description
        explicit_class: Explicitly specified event class

    Returns:
        IntentResolution if successful, None otherwise
    """
    return get_resolver().resolve(natural_language, explicit_class)


if __name__ == "__main__":
    import sys
    import json

    # Demo usage
    test_cases = [
        {"natural_language": "flooding after hurricane in Miami"},
        {"natural_language": "forest fire with severe burn scars"},
        {"explicit_class": "storm.severe_convective"},
        {
            "natural_language": "coastal flooding",
            "explicit_class": "flood.coastal.storm_surge"
        },
    ]

    if len(sys.argv) > 1:
        # Use command line argument
        test_cases = [{"natural_language": " ".join(sys.argv[1:])}]

    resolver = get_resolver()

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}:")
        print(f"{'='*60}")
        print(f"Input: {test_case}")

        resolution = resolver.resolve(**test_case)

        if resolution:
            print(f"\nResolution: {resolution}")
            print(f"\nFull output (Intent Schema format):")
            print(json.dumps(resolution.to_dict(), indent=2))
        else:
            print("\nNo resolution found")
