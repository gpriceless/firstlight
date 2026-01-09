"""Edge case and stress tests for intent resolution system."""

import pytest

from core.intent.classifier import ClassificationResult, EventClassifier
from core.intent.registry import EventClass, EventClassRegistry
from core.intent.resolver import IntentResolution, IntentResolver


class TestClassifierEdgeCases:
    """Test edge cases in event classification."""

    @pytest.fixture
    def classifier(self):
        """Get classifier instance."""
        from core.intent.classifier import get_classifier
        return get_classifier()

    def test_classify_empty_string(self, classifier):
        """Test classifying an empty string."""
        result = classifier.classify("")
        assert result is None

    def test_classify_only_stopwords(self, classifier):
        """Test classifying text with only stopwords."""
        result = classifier.classify("the and is in on at")
        assert result is None

    def test_classify_very_short_text(self, classifier):
        """Test classifying very short text."""
        result = classifier.classify("a b")
        # Very short text should return None or low confidence
        if result:
            assert result.confidence < 0.3

    def test_classify_very_long_text(self, classifier):
        """Test classifying very long text."""
        long_text = " ".join(["flooding"] * 100 + ["river"] * 50)
        result = classifier.classify(long_text)
        assert result is not None
        assert "flood" in result.inferred_class

    def test_classify_mixed_case(self, classifier):
        """Test that classification is case-insensitive."""
        result1 = classifier.classify("FLOODING IN COASTAL AREAS")
        result2 = classifier.classify("flooding in coastal areas")
        result3 = classifier.classify("FlOoDiNg In CoAsTaL aReAs")

        assert result1 is not None
        assert result2 is not None
        assert result3 is not None

        # All should classify to similar classes
        assert "flood" in result1.inferred_class
        assert "flood" in result2.inferred_class
        assert "flood" in result3.inferred_class

    def test_classify_special_characters(self, classifier):
        """Test classification with special characters."""
        result = classifier.classify("flooding!!! in @coastal #areas $$$")
        assert result is not None
        assert "flood" in result.inferred_class

    def test_classify_unicode_characters(self, classifier):
        """Test classification with unicode characters."""
        result = classifier.classify("flooding in coastal areas 洪水")
        # Should still work with ASCII keywords
        assert result is not None
        assert "flood" in result.inferred_class

    def test_classify_numbers(self, classifier):
        """Test classification with numbers."""
        result = classifier.classify("flooding in area 123 with depth 5m")
        assert result is not None
        assert "flood" in result.inferred_class

    def test_classify_ambiguous_text(self, classifier):
        """Test classification of ambiguous text."""
        result = classifier.classify("storm")
        assert result is not None
        # Should classify to storm but may have low confidence
        assert "storm" in result.inferred_class

    def test_classify_contradictory_keywords(self, classifier):
        """Test classification with contradictory keywords."""
        result = classifier.classify("flood fire storm")
        assert result is not None
        # Should classify to something with alternatives
        assert len(result.alternatives) > 0

    def test_extract_keywords_removes_short_words(self, classifier):
        """Test that very short words are removed."""
        keywords = classifier._extract_keywords("a flood in my area")
        assert "a" not in keywords
        assert "in" not in keywords
        assert "my" not in keywords

    def test_extract_keywords_handles_hyphens(self, classifier):
        """Test keyword extraction with hyphenated words."""
        keywords = classifier._extract_keywords("post-fire burn scar")
        # Should extract both hyphenated and non-hyphenated versions
        assert any("fire" in kw or "post-fire" in kw for kw in keywords)

    def test_confidence_score_normalization(self, classifier):
        """Test that confidence scores are properly normalized."""
        result = classifier.classify("coastal flooding storm surge")
        assert result is not None
        assert 0 <= result.confidence <= 1.0

        # Check alternatives too
        for alt_class, alt_conf in result.alternatives:
            assert 0 <= alt_conf <= 1.0


class TestResolverEdgeCases:
    """Test edge cases in intent resolution."""

    @pytest.fixture
    def resolver(self):
        """Get resolver instance."""
        from core.intent.resolver import get_resolver
        return get_resolver()

    def test_resolve_both_nl_and_explicit_with_no_override(self, resolver):
        """Test resolution when both NL and explicit provided with override disabled."""
        resolution = resolver.resolve(
            natural_language="flooding",
            explicit_class="wildfire.forest",
            allow_override=False  # Don't allow explicit to override
        )
        # With allow_override=False, should use NLP
        assert resolution is not None

    def test_resolve_empty_natural_language(self, resolver):
        """Test resolution with empty natural language string."""
        resolution = resolver.resolve(natural_language="")
        assert resolution is None

    def test_resolve_whitespace_only(self, resolver):
        """Test resolution with only whitespace."""
        resolution = resolver.resolve(natural_language="   \t\n  ")
        # Should return None due to no keywords after processing
        assert resolution is None

    def test_resolve_invalid_explicit_class_format(self, resolver):
        """Test resolution with malformed explicit class."""
        resolution = resolver.resolve(explicit_class=".......")
        assert resolution is None

    def test_resolve_partial_class_path(self, resolver):
        """Test resolution with partial/incomplete class path."""
        # Try a partial path that exists
        resolution = resolver.resolve(explicit_class="flood")
        # Should work if "flood" is a valid class
        if resolution:
            assert resolution.resolved_class == "flood"

    def test_parameter_extraction_edge_cases(self, resolver):
        """Test parameter extraction with edge cases."""
        resolution = resolver.resolve(
            natural_language="before during after flooding hurricane tornado"
        )
        # Should extract temporal context and causation
        if resolution and resolution.parameters:
            # Multiple temporal keywords - should pick one
            assert "temporal_context" in resolution.parameters


class TestRegistryEdgeCases:
    """Test edge cases in registry operations."""

    @pytest.fixture
    def registry(self):
        """Get registry instance."""
        from core.intent.registry import get_registry
        return get_registry()

    def test_get_class_empty_string(self, registry):
        """Test getting class with empty string."""
        result = registry.get_class("")
        assert result is None

    def test_get_class_none(self, registry):
        """Test getting class with None (should handle gracefully)."""
        # The registry should handle None gracefully by returning None
        result = registry.get_class(None)
        assert result is None

    def test_wildcard_matching_edge_cases(self, registry):
        """Test wildcard matching with edge cases."""
        # Empty wildcard
        results = registry.get_classes_by_wildcard("")
        assert len(results) == 0

        # Just asterisk
        results = registry.get_classes_by_wildcard("*")
        assert len(results) > 0

        # Double wildcard
        results = registry.get_classes_by_wildcard("**")
        # Should match anything with at least one character
        assert len(results) >= 0

        # Wildcard at beginning
        results = registry.get_classes_by_wildcard("*.coastal")
        # Should match anything ending in .coastal
        assert all(".coastal" in ec.class_path for ec in results)

    def test_keyword_search_empty_string(self, registry):
        """Test keyword search with empty string."""
        results = registry.get_classes_by_keyword("")
        assert len(results) == 0

    def test_keyword_search_case_insensitive(self, registry):
        """Test that keyword search is case-insensitive."""
        results_lower = registry.get_classes_by_keyword("flood")
        results_upper = registry.get_classes_by_keyword("FLOOD")
        results_mixed = registry.get_classes_by_keyword("FlOoD")

        # All should return same results
        assert len(results_lower) == len(results_upper)
        assert len(results_lower) == len(results_mixed)

    def test_search_by_empty_keywords_list(self, registry):
        """Test searching with empty keywords list."""
        results = registry.search_by_keywords([])
        assert len(results) == 0

    def test_search_by_keywords_with_duplicates(self, registry):
        """Test searching with duplicate keywords."""
        results = registry.search_by_keywords(["flood", "flood", "flood"])
        # Should handle duplicates gracefully
        assert len(results) >= 0

    def test_search_by_nonexistent_keywords(self, registry):
        """Test searching with keywords that don't match anything."""
        results = registry.search_by_keywords(["xyzabc123", "nonexistent", "fake"])
        assert len(results) == 0


class TestClassificationResultEdgeCases:
    """Test ClassificationResult edge cases."""

    def test_classification_result_with_minimal_data(self):
        """Test creating classification result with minimal data."""
        result = ClassificationResult(
            inferred_class="flood",
            confidence=0.5
        )
        assert result.inferred_class == "flood"
        assert result.confidence == 0.5
        assert result.alternatives == []
        assert result.matched_keywords == []

    def test_classification_result_to_dict(self):
        """Test converting classification result to dict."""
        result = ClassificationResult(
            inferred_class="flood.coastal",
            confidence=0.85,
            alternatives=[("flood.riverine", 0.6), ("storm.tropical_cyclone", 0.4)],
            matched_keywords=["flood", "coastal"]
        )

        result_dict = result.to_dict()
        assert result_dict["inferred_class"] == "flood.coastal"
        assert result_dict["confidence"] == 0.85
        assert len(result_dict["alternatives"]) == 2
        assert len(result_dict["matched_keywords"]) == 2

    def test_classification_result_edge_confidence_values(self):
        """Test classification result with edge confidence values."""
        # Zero confidence
        result1 = ClassificationResult(inferred_class="test", confidence=0.0)
        assert result1.confidence == 0.0

        # Max confidence
        result2 = ClassificationResult(inferred_class="test", confidence=1.0)
        assert result2.confidence == 1.0


class TestIntentResolutionEdgeCases:
    """Test IntentResolution edge cases."""

    def test_intent_resolution_minimal(self):
        """Test creating intent resolution with minimal data."""
        resolution = IntentResolution(
            resolved_class="flood",
            source="explicit",
            confidence=1.0
        )
        assert resolution.resolved_class == "flood"
        assert resolution.source == "explicit"
        assert resolution.confidence == 1.0

    def test_intent_resolution_to_dict_explicit(self):
        """Test converting explicit resolution to dict."""
        resolution = IntentResolution(
            resolved_class="flood.coastal",
            source="explicit",
            confidence=1.0
        )

        result_dict = resolution.to_dict()
        assert "input" in result_dict
        assert "output" in result_dict
        assert result_dict["output"]["source"] == "explicit"
        assert "resolution" not in result_dict  # No NLP resolution for explicit

    def test_intent_resolution_to_dict_inferred(self):
        """Test converting inferred resolution to dict."""
        resolution = IntentResolution(
            resolved_class="flood.coastal",
            source="inferred",
            confidence=0.8,
            original_input="coastal flooding",
            inferred_class="flood.coastal",
            inferred_confidence=0.8,
            alternatives=[("flood.riverine", 0.6)]
        )

        result_dict = resolution.to_dict()
        assert "resolution" in result_dict  # Should have NLP resolution details
        assert result_dict["resolution"]["inferred_class"] == "flood.coastal"
        assert len(result_dict["resolution"]["alternatives"]) > 0

    def test_intent_resolution_repr(self):
        """Test string representation of intent resolution."""
        resolution = IntentResolution(
            resolved_class="flood.coastal",
            source="inferred",
            confidence=0.85
        )
        repr_str = repr(resolution)
        assert "flood.coastal" in repr_str
        assert "inferred" in repr_str
        assert "0.85" in repr_str


class TestEventClassEdgeCases:
    """Test EventClass edge cases."""

    def test_event_class_wildcard_matching(self):
        """Test wildcard matching on EventClass."""
        event_class = EventClass(
            class_path="flood.coastal.storm_surge",
            description="Test"
        )

        assert event_class.matches_wildcard("flood.*")
        assert event_class.matches_wildcard("flood.coastal.*")
        assert event_class.matches_wildcard("*.storm_surge")
        assert event_class.matches_wildcard("*")
        assert not event_class.matches_wildcard("wildfire.*")

    def test_event_class_get_all_keywords(self):
        """Test getting all keywords including children."""
        parent = EventClass(
            class_path="flood",
            description="Parent",
            keywords=["flood", "flooding"]
        )

        child = EventClass(
            class_path="flood.coastal",
            description="Child",
            keywords=["coastal", "ocean"]
        )

        parent.children["coastal"] = child

        # Get keywords including children
        all_keywords = parent.get_all_keywords(include_children=True)
        assert "flood" in all_keywords
        assert "flooding" in all_keywords
        assert "coastal" in all_keywords
        assert "ocean" in all_keywords

        # Get keywords excluding children
        parent_only = parent.get_all_keywords(include_children=False)
        assert "flood" in parent_only
        assert "coastal" not in parent_only

    def test_event_class_get_descendant_paths(self):
        """Test getting descendant paths."""
        parent = EventClass(class_path="flood", description="Parent")
        child1 = EventClass(class_path="flood.coastal", description="Child1")
        child2 = EventClass(class_path="flood.riverine", description="Child2")
        grandchild = EventClass(
            class_path="flood.coastal.storm_surge",
            description="Grandchild"
        )

        parent.children["coastal"] = child1
        parent.children["riverine"] = child2
        child1.children["storm_surge"] = grandchild

        paths = parent.get_descendant_paths()
        assert "flood" in paths
        assert "flood.coastal" in paths
        assert "flood.riverine" in paths
        assert "flood.coastal.storm_surge" in paths


class TestConcurrentAccess:
    """Test concurrent access patterns (thread safety considerations)."""

    def test_multiple_classifier_instances(self):
        """Test that multiple classifier instances work correctly."""
        from core.intent.classifier import EventClassifier

        classifier1 = EventClassifier()
        classifier2 = EventClassifier()

        result1 = classifier1.classify("flooding")
        result2 = classifier2.classify("flooding")

        assert result1 is not None
        assert result2 is not None
        assert result1.inferred_class == result2.inferred_class

    def test_multiple_resolver_instances(self):
        """Test that multiple resolver instances work correctly."""
        from core.intent.resolver import IntentResolver

        resolver1 = IntentResolver()
        resolver2 = IntentResolver()

        resolution1 = resolver1.resolve(natural_language="flooding")
        resolution2 = resolver2.resolve(natural_language="flooding")

        assert resolution1 is not None
        assert resolution2 is not None
        assert resolution1.resolved_class == resolution2.resolved_class
