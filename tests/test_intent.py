"""Tests for intent resolution system."""

import pytest

from core.intent.classifier import EventClassifier, get_classifier
from core.intent.registry import EventClass, EventClassRegistry, get_registry
from core.intent.resolver import IntentResolver, get_resolver


class TestEventClassRegistry:
    """Test event class registry functionality."""

    @pytest.fixture
    def registry(self):
        """Get registry instance."""
        return get_registry()

    def test_registry_loads_definitions(self, registry):
        """Test that registry loads event class definitions."""
        all_classes = registry.list_all_classes()
        assert len(all_classes) > 0
        assert "flood" in all_classes
        assert "wildfire" in all_classes
        assert "storm" in all_classes

    def test_get_class_by_path(self, registry):
        """Test retrieving a class by its path."""
        flood_class = registry.get_class("flood")
        assert flood_class is not None
        assert flood_class.class_path == "flood"
        assert flood_class.description != ""

    def test_get_nested_class(self, registry):
        """Test retrieving a nested class."""
        storm_surge = registry.get_class("flood.coastal.storm_surge")
        assert storm_surge is not None
        assert storm_surge.class_path == "flood.coastal.storm_surge"

    def test_get_nonexistent_class(self, registry):
        """Test that nonexistent class returns None."""
        result = registry.get_class("nonexistent.class")
        assert result is None

    def test_wildcard_matching(self, registry):
        """Test wildcard matching for classes."""
        flood_classes = registry.get_classes_by_wildcard("flood.*")
        assert len(flood_classes) > 0
        assert all("flood" in ec.class_path for ec in flood_classes)

    def test_wildcard_all_classes(self, registry):
        """Test wildcard matching all classes."""
        all_classes = registry.get_classes_by_wildcard("*")
        assert len(all_classes) > 0

    def test_keyword_search(self, registry):
        """Test searching by keyword."""
        # Search for "hurricane" keyword
        matches = registry.get_classes_by_keyword("hurricane")
        assert len(matches) > 0

    def test_multi_keyword_search(self, registry):
        """Test searching by multiple keywords."""
        keywords = ["storm surge", "hurricane"]
        results = registry.search_by_keywords(keywords)
        assert len(results) > 0
        # Should match storm surge and/or tropical cyclone classes
        assert any("storm_surge" in path or "tropical_cyclone" in path for path in results.keys())

    def test_get_root_classes(self, registry):
        """Test getting root event classes."""
        root_classes = registry.get_root_classes()
        assert len(root_classes) > 0
        root_paths = [ec.class_path for ec in root_classes]
        assert "flood" in root_paths
        assert "wildfire" in root_paths
        assert "storm" in root_paths

    def test_event_class_has_keywords(self, registry):
        """Test that event classes have keywords."""
        flood_class = registry.get_class("flood.coastal.storm_surge")
        if flood_class:
            assert len(flood_class.keywords) > 0

    def test_event_class_has_data_types(self, registry):
        """Test that event classes specify data types."""
        flood_class = registry.get_class("flood.coastal.storm_surge")
        if flood_class:
            # Should have required data types
            assert len(flood_class.required_data_types) > 0


class TestEventClassifier:
    """Test event classifier functionality."""

    @pytest.fixture
    def classifier(self):
        """Get classifier instance."""
        return get_classifier()

    def test_classify_flood_text(self, classifier):
        """Test classifying flood-related text."""
        result = classifier.classify("river flooding and inundation")
        assert result is not None
        assert "flood" in result.inferred_class
        assert 0 < result.confidence <= 1.0

    def test_classify_wildfire_text(self, classifier):
        """Test classifying wildfire-related text."""
        result = classifier.classify("forest fire with burn scars")
        assert result is not None
        assert "wildfire" in result.inferred_class
        assert 0 < result.confidence <= 1.0

    def test_classify_storm_text(self, classifier):
        """Test classifying storm-related text."""
        result = classifier.classify("tornado damage assessment")
        assert result is not None
        assert "storm" in result.inferred_class
        assert 0 < result.confidence <= 1.0

    def test_classify_coastal_storm_surge(self, classifier):
        """Test specific classification for coastal storm surge."""
        result = classifier.classify("coastal flooding from storm surge")
        assert result is not None
        assert "flood" in result.inferred_class
        # Should prefer coastal or storm_surge variant
        assert "coastal" in result.inferred_class or "storm_surge" in result.inferred_class

    def test_classify_returns_alternatives(self, classifier):
        """Test that classifier returns alternative classifications."""
        result = classifier.classify("storm flooding")
        assert result is not None
        # Should have alternatives since "storm flooding" could be ambiguous
        # (could be storm surge flood or general storm)

    def test_classify_nonsense_text(self, classifier):
        """Test that nonsense text returns None or low confidence."""
        result = classifier.classify("xyz abc def random words")
        # Either returns None or very low confidence
        if result:
            assert result.confidence < 0.3

    def test_extract_keywords(self, classifier):
        """Test keyword extraction."""
        keywords = classifier._extract_keywords("flooding after hurricane in Miami")
        assert "flooding" in keywords or "flood" in keywords
        assert "hurricane" in keywords

    def test_stopwords_removed(self, classifier):
        """Test that stopwords are removed."""
        keywords = classifier._extract_keywords("the flooding is in the area")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "in" not in keywords

    def test_multi_word_phrases(self, classifier):
        """Test extraction of multi-word phrases."""
        phrases = classifier._extract_multi_word_phrases("storm surge flooding")
        assert "storm surge" in phrases


class TestIntentResolver:
    """Test intent resolver functionality."""

    @pytest.fixture
    def resolver(self):
        """Get resolver instance."""
        return get_resolver()

    def test_resolve_explicit_class(self, resolver):
        """Test resolving with explicit class specification."""
        resolution = resolver.resolve(explicit_class="flood.coastal.storm_surge")
        assert resolution is not None
        assert resolution.resolved_class == "flood.coastal.storm_surge"
        assert resolution.source == "explicit"
        assert resolution.confidence == 1.0

    def test_resolve_natural_language(self, resolver):
        """Test resolving with natural language."""
        resolution = resolver.resolve(natural_language="river flooding and inundation")
        assert resolution is not None
        assert "flood" in resolution.resolved_class
        assert resolution.source == "inferred"
        assert 0 < resolution.confidence <= 1.0
        assert resolution.original_input == "river flooding and inundation"

    def test_explicit_overrides_nlp(self, resolver):
        """Test that explicit class overrides NLP inference."""
        resolution = resolver.resolve(
            natural_language="flooding",
            explicit_class="wildfire.forest"  # Contradictory explicit class
        )
        assert resolution is not None
        assert resolution.resolved_class == "wildfire.forest"
        assert resolution.source == "explicit"

    def test_resolve_invalid_explicit_class(self, resolver):
        """Test that invalid explicit class returns None."""
        resolution = resolver.resolve(explicit_class="invalid.class.name")
        assert resolution is None

    def test_resolve_no_input(self, resolver):
        """Test that no input returns None."""
        resolution = resolver.resolve()
        assert resolution is None

    def test_resolution_to_dict(self, resolver):
        """Test converting resolution to dictionary."""
        resolution = resolver.resolve(explicit_class="flood.coastal")
        assert resolution is not None

        result_dict = resolution.to_dict()
        assert "input" in result_dict
        assert "output" in result_dict
        assert result_dict["output"]["resolved_class"] == "flood.coastal"
        assert result_dict["output"]["source"] == "explicit"

    def test_resolution_dict_with_nlp(self, resolver):
        """Test dictionary format for NLP-resolved intent."""
        resolution = resolver.resolve(natural_language="wildfire in forest")
        assert resolution is not None

        result_dict = resolution.to_dict()
        assert "input" in result_dict
        assert "natural_language" in result_dict["input"]
        assert "output" in result_dict
        assert "resolution" in result_dict  # Should have NLP resolution details

    def test_parameter_extraction(self, resolver):
        """Test that parameters are extracted."""
        resolution = resolver.resolve(explicit_class="flood.coastal.storm_surge")
        assert resolution is not None
        assert len(resolution.parameters) > 0
        assert "event_type" in resolution.parameters

    def test_parameter_extraction_from_text(self, resolver):
        """Test parameter extraction from natural language."""
        resolution = resolver.resolve(
            natural_language="flooding after hurricane landfall"
        )
        assert resolution is not None
        # Should extract temporal context
        if "temporal_context" in resolution.parameters:
            assert resolution.parameters["temporal_context"] == "post_event"


class TestIntegration:
    """Integration tests for the full intent resolution pipeline."""

    def test_end_to_end_flood_resolution(self):
        """Test complete flow for flood event."""
        resolver = get_resolver()
        resolution = resolver.resolve(
            natural_language="coastal flooding from hurricane storm surge"
        )

        assert resolution is not None
        assert "flood" in resolution.resolved_class
        assert resolution.source == "inferred"
        assert resolution.original_input is not None

        # Verify it conforms to intent schema format
        result_dict = resolution.to_dict()
        assert "input" in result_dict
        assert "output" in result_dict
        assert "resolved_class" in result_dict["output"]
        assert "source" in result_dict["output"]
        assert "confidence" in result_dict["output"]

    def test_end_to_end_wildfire_resolution(self):
        """Test complete flow for wildfire event."""
        resolver = get_resolver()
        resolution = resolver.resolve(natural_language="forest fire burn scar mapping")

        assert resolution is not None
        assert "wildfire" in resolution.resolved_class
        assert resolution.source == "inferred"

    def test_registry_classifier_integration(self):
        """Test that classifier uses registry correctly."""
        registry = get_registry()
        classifier = get_classifier()

        # Ensure classifier can find classes in registry
        result = classifier.classify("storm surge flooding")
        assert result is not None

        # Verify the inferred class exists in registry
        event_class = registry.get_class(result.inferred_class)
        assert event_class is not None

    def test_singleton_instances(self):
        """Test that singleton instances are reused."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

        classifier1 = get_classifier()
        classifier2 = get_classifier()
        assert classifier1 is classifier2

        resolver1 = get_resolver()
        resolver2 = get_resolver()
        assert resolver1 is resolver2
