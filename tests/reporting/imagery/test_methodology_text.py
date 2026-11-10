"""
Tests for Methodology Text Module (VIS-1.4 Task 9)

Verifies:
- Methodology explanations at different detail levels
- Algorithm-specific method details
- Limitations text generation
- Confidence explanations
- Accessible, non-technical language (NO jargon)
- Proper handling of edge cases
"""

import pytest

from core.reporting.imagery.methodology_text import (
    EventType,
    METHODOLOGY_TEMPLATES,
    LIMITATIONS_TEMPLATES,
    ALGORITHM_METHOD_DETAILS,
    get_methodology_explanation,
    get_limitations_text,
    get_confidence_explanation,
)


# ============================================================================
# Coverage Tests - Ensure All Event Types Have Content
# ============================================================================


class TestMethodologyTemplatesCoverage:
    """Verify all event types have methodology templates."""

    def test_all_event_types_have_methodology(self):
        """All EventType enum members have methodology templates."""
        for event_type in EventType:
            assert event_type in METHODOLOGY_TEMPLATES, (
                f"Missing methodology template for {event_type.value}"
            )

    def test_all_event_types_have_limitations(self):
        """All EventType enum members have limitations templates."""
        for event_type in EventType:
            assert event_type in LIMITATIONS_TEMPLATES, (
                f"Missing limitations template for {event_type.value}"
            )

    def test_all_methodologies_are_nonempty(self):
        """All methodology templates have content."""
        for event_type, text in METHODOLOGY_TEMPLATES.items():
            assert len(text.strip()) > 0, (
                f"Empty methodology template for {event_type.value}"
            )
            assert len(text.strip()) > 50, (
                f"Methodology template too short for {event_type.value}"
            )

    def test_all_limitations_are_nonempty(self):
        """All limitations templates have content."""
        for event_type, text in LIMITATIONS_TEMPLATES.items():
            assert len(text.strip()) > 0, (
                f"Empty limitations template for {event_type.value}"
            )


class TestAlgorithmMethodDetailsCoverage:
    """Verify algorithm method details are comprehensive."""

    def test_flood_algorithms_covered(self):
        """Common flood algorithms have method details."""
        flood_algorithms = ["threshold_sar", "ndwi", "change_detection"]
        for algo in flood_algorithms:
            assert algo in ALGORITHM_METHOD_DETAILS, (
                f"Missing method detail for flood algorithm: {algo}"
            )

    def test_wildfire_algorithms_covered(self):
        """Common wildfire algorithms have method details."""
        wildfire_algorithms = ["thermal_anomaly", "nbr_differenced", "ba_classifier"]
        for algo in wildfire_algorithms:
            assert algo in ALGORITHM_METHOD_DETAILS, (
                f"Missing method detail for wildfire algorithm: {algo}"
            )

    def test_storm_algorithms_covered(self):
        """Common storm algorithms have method details."""
        storm_algorithms = ["wind_damage", "structural_damage"]
        for algo in storm_algorithms:
            assert algo in ALGORITHM_METHOD_DETAILS, (
                f"Missing method detail for storm algorithm: {algo}"
            )


# ============================================================================
# Language Quality Tests - NO JARGON
# ============================================================================


class TestLanguageAccessibility:
    """Verify all text uses accessible, non-technical language."""

    # Technical jargon that should NOT appear in explanations
    FORBIDDEN_JARGON = [
        "backscatter",
        "radiometric",
        "spectral index",
        "spectral indices",
        "ndwi",
        "ndvi",
        "nbr",
        "dnbr",
        "sar",  # Should say "radar" instead
        "synthetic aperture",
        "orthorectified",
        "geotiff",
        "raster",
        "pixel",  # Okay in technical context, but avoid if possible
        "band math",
        "reflectance coefficient",
        "emissivity",
        "radiance",
    ]

    def test_methodology_templates_no_jargon(self):
        """Methodology templates avoid technical jargon."""
        for event_type, text in METHODOLOGY_TEMPLATES.items():
            text_lower = text.lower()
            for jargon in self.FORBIDDEN_JARGON:
                assert jargon.lower() not in text_lower, (
                    f"Jargon '{jargon}' found in {event_type.value} methodology: {text}"
                )

    def test_limitations_templates_no_jargon(self):
        """Limitations templates avoid technical jargon."""
        for event_type, text in LIMITATIONS_TEMPLATES.items():
            text_lower = text.lower()
            for jargon in self.FORBIDDEN_JARGON:
                assert jargon.lower() not in text_lower, (
                    f"Jargon '{jargon}' found in {event_type.value} limitations: {text}"
                )

    def test_algorithm_details_no_jargon(self):
        """Algorithm method details avoid technical jargon."""
        for algo_id, text in ALGORITHM_METHOD_DETAILS.items():
            text_lower = text.lower()
            for jargon in self.FORBIDDEN_JARGON:
                assert jargon.lower() not in text_lower, (
                    f"Jargon '{jargon}' found in {algo_id} detail: {text}"
                )

    def test_uses_accessible_terms(self):
        """Explanations use accessible alternatives to jargon."""
        # Check that accessible terms are used
        flood_text = METHODOLOGY_TEMPLATES[EventType.FLOOD].lower()
        assert "radar" in flood_text or "satellite" in flood_text

        wildfire_text = METHODOLOGY_TEMPLATES[EventType.WILDFIRE].lower()
        assert "vegetation" in wildfire_text or "burned" in wildfire_text

    def test_no_unexplained_acronyms(self):
        """No unexplained acronyms (3+ uppercase letters)."""
        import re

        all_texts = (
            list(METHODOLOGY_TEMPLATES.values()) +
            list(LIMITATIONS_TEMPLATES.values()) +
            list(ALGORITHM_METHOD_DETAILS.values())
        )

        for text in all_texts:
            # Find acronyms (3+ consecutive uppercase letters)
            acronyms = re.findall(r'\b[A-Z]{3,}\b', text)

            # No unexplained acronyms should appear
            assert len(acronyms) == 0, (
                f"Unexplained acronyms found: {acronyms} in text: {text[:100]}..."
            )


class TestLanguageTone:
    """Verify language is factual and professional (no hyperbole)."""

    FORBIDDEN_HYPERBOLE = [
        "devastating",
        "catastrophic",
        "massive",
        "enormous",
        "incredible",
        "amazing",
        "unprecedented",
        "historic",
        "tragic",
        "!",  # No exclamation marks
    ]

    def test_no_hyperbolic_language(self):
        """No hyperbolic or emotional language."""
        all_texts = (
            list(METHODOLOGY_TEMPLATES.values()) +
            list(LIMITATIONS_TEMPLATES.values()) +
            list(ALGORITHM_METHOD_DETAILS.values())
        )

        for text in all_texts:
            text_lower = text.lower()
            for phrase in self.FORBIDDEN_HYPERBOLE:
                assert phrase.lower() not in text_lower, (
                    f"Hyperbolic phrase '{phrase}' found in: {text[:100]}..."
                )

    def test_uses_factual_descriptors(self):
        """Uses factual, measurable language."""
        # Should use words like "detects", "identifies", "measures", "compares"
        factual_terms = ["detect", "identify", "measure", "compare", "analyze", "distinguish"]

        flood_text = METHODOLOGY_TEMPLATES[EventType.FLOOD].lower()
        assert any(term in flood_text for term in factual_terms)


# ============================================================================
# Function Tests - get_methodology_explanation()
# ============================================================================


class TestGetMethodologyExplanation:
    """Test get_methodology_explanation function."""

    def test_flood_basic(self):
        """Returns flood methodology without algorithm details."""
        text = get_methodology_explanation("flood")

        assert "flood" in text.lower()
        assert "water" in text.lower()
        assert len(text) > 100
        assert "Note:" not in text  # No limitations by default

    def test_flood_with_algorithm(self):
        """Returns flood methodology with algorithm-specific details."""
        text = get_methodology_explanation("flood", algorithm_id="flood.baseline.threshold_sar")

        assert "flood" in text.lower()
        assert "radar" in text.lower()
        assert len(text) > len(METHODOLOGY_TEMPLATES[EventType.FLOOD])

    def test_flood_with_limitations(self):
        """Returns flood methodology with limitations included."""
        text = get_methodology_explanation("flood", include_limitations=True)

        assert "flood" in text.lower()
        assert "Note:" in text
        assert "limitations" in text.lower()

    def test_flood_complete(self):
        """Returns complete flood methodology with algorithm and limitations."""
        text = get_methodology_explanation(
            "flood",
            algorithm_id="flood.baseline.ndwi",
            include_limitations=True
        )

        assert "flood" in text.lower()
        assert "optical" in text.lower() or "light" in text.lower()
        assert "Note:" in text
        assert "limitations" in text.lower()

        # Should have multiple paragraphs (base methodology may have multiple paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        assert len(paragraphs) >= 3, f"Expected at least 3 paragraphs, got {len(paragraphs)}"

    def test_wildfire_basic(self):
        """Returns wildfire methodology."""
        text = get_methodology_explanation("wildfire")

        assert "wildfire" in text.lower() or "fire" in text.lower()
        assert "vegetation" in text.lower() or "burned" in text.lower()

    def test_wildfire_with_algorithm(self):
        """Returns wildfire methodology with burn severity details."""
        text = get_methodology_explanation("wildfire", algorithm_id="wildfire.baseline.nbr_differenced")

        assert "vegetation" in text.lower()
        assert "reflects" in text.lower() or "reflectance" in text.lower()
        assert "severity" in text.lower()

    def test_storm_basic(self):
        """Returns storm methodology."""
        text = get_methodology_explanation("storm")

        assert "storm" in text.lower()
        assert "damage" in text.lower()

    def test_event_type_with_subtype(self):
        """Handles event types with subtypes (e.g., 'flood.coastal')."""
        text = get_methodology_explanation("flood.coastal")

        # Should treat as "flood"
        assert "flood" in text.lower()
        assert len(text) > 100

    def test_unknown_event_type(self):
        """Returns generic methodology for unknown event types."""
        text = get_methodology_explanation("unknown_event")

        assert "satellite" in text.lower()
        assert "imagery" in text.lower()
        assert len(text) > 100

    def test_algorithm_id_extraction(self):
        """Correctly extracts algorithm method from full ID."""
        # Full algorithm IDs
        text1 = get_methodology_explanation("flood", "flood.baseline.threshold_sar")
        assert "radar" in text1.lower()

        text2 = get_methodology_explanation("wildfire", "wildfire.baseline.thermal_anomaly")
        assert "thermal" in text2.lower()

        # Partial algorithm IDs
        text3 = get_methodology_explanation("flood", "threshold_sar")
        assert "radar" in text3.lower()

    def test_unknown_algorithm_id(self):
        """Handles unknown algorithm IDs gracefully."""
        text = get_methodology_explanation("flood", algorithm_id="flood.unknown.method")

        # Should still return base methodology
        assert "flood" in text.lower()
        assert len(text) > 0


# ============================================================================
# Function Tests - get_limitations_text()
# ============================================================================


class TestGetLimitationsText:
    """Test get_limitations_text function."""

    def test_flood_limitations(self):
        """Returns flood-specific limitations."""
        text = get_limitations_text("flood")

        assert "Note:" in text
        assert "limitations" in text.lower()
        assert "tree cover" in text.lower() or "vegetation" in text.lower()

    def test_wildfire_limitations(self):
        """Returns wildfire-specific limitations."""
        text = get_limitations_text("wildfire")

        assert "Note:" in text
        assert "limitations" in text.lower()
        assert "smoke" in text.lower() or "cloud" in text.lower()

    def test_storm_limitations(self):
        """Returns storm-specific limitations."""
        text = get_limitations_text("storm")

        assert "Note:" in text
        assert "limitations" in text.lower()
        assert "interior" in text.lower() or "ground level" in text.lower()

    def test_event_type_with_subtype(self):
        """Handles event subtypes (e.g., 'flood.coastal')."""
        text = get_limitations_text("flood.coastal")

        # Should treat as "flood"
        assert "Note:" in text
        assert "flood" in text.lower()

    def test_unknown_event_type(self):
        """Returns generic limitations for unknown event types."""
        text = get_limitations_text("unknown_event")

        assert "Note:" in text
        assert "satellite" in text.lower()
        assert "limitations" in text.lower()


# ============================================================================
# Function Tests - get_confidence_explanation()
# ============================================================================


class TestGetConfidenceExplanation:
    """Test get_confidence_explanation function."""

    def test_high_confidence(self):
        """Explains high confidence (>= 80%)."""
        text = get_confidence_explanation(85.0)

        assert "85%" in text
        assert "high certainty" in text.lower()
        assert "Detection confidence" in text

    def test_moderate_confidence(self):
        """Explains moderate confidence (60-79%)."""
        text = get_confidence_explanation(70.0)

        assert "70%" in text
        assert "moderate certainty" in text.lower()

    def test_low_confidence(self):
        """Explains low confidence (< 60%)."""
        text = get_confidence_explanation(45.0)

        assert "45%" in text
        assert "lower certainty" in text.lower()

    def test_boundary_values(self):
        """Tests boundary values between confidence levels."""
        # Exactly 80% should be high
        assert "high certainty" in get_confidence_explanation(80.0).lower()

        # Exactly 60% should be moderate
        assert "moderate certainty" in get_confidence_explanation(60.0).lower()

        # Just below 60% should be lower
        assert "lower certainty" in get_confidence_explanation(59.9).lower()

    def test_formatting(self):
        """Confidence formatted as integer percentage."""
        text = get_confidence_explanation(85.7)

        # Should round to 86%
        assert "86%" in text
        assert "85.7%" not in text

    def test_out_of_range_values(self):
        """Handles values outside 0-100 range."""
        # Above 100
        text_high = get_confidence_explanation(150.0)
        assert "100%" in text_high

        # Below 0
        text_low = get_confidence_explanation(-10.0)
        assert "0%" in text_low

    def test_explanation_includes_guidance(self):
        """Explanation includes guidance on interpreting confidence."""
        text = get_confidence_explanation(85.0)

        assert "higher values indicate" in text.lower() or "higher" in text.lower()
        assert "reliable" in text.lower() or "certainty" in text.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_complete_flood_report_explanation(self):
        """Complete flood report with all components."""
        methodology = get_methodology_explanation(
            event_type="flood",
            algorithm_id="flood.baseline.threshold_sar",
            include_limitations=True
        )

        limitations = get_limitations_text("flood")
        confidence = get_confidence_explanation(82.0)

        # Verify all components are distinct and valid
        assert methodology != limitations
        assert methodology != confidence
        assert limitations != confidence

        # All should have meaningful content
        assert len(methodology) > 200
        assert len(limitations) > 50
        assert len(confidence) > 50

        # Methodology should include all parts
        assert "flood" in methodology.lower()
        assert "radar" in methodology.lower()
        assert "Note:" in methodology

    def test_wildfire_report_with_burn_severity(self):
        """Wildfire report with burn severity algorithm."""
        methodology = get_methodology_explanation(
            event_type="wildfire",
            algorithm_id="wildfire.baseline.nbr_differenced",
            include_limitations=True
        )

        # Should explain vegetation reflectance and burn detection
        assert "vegetation" in methodology.lower()
        assert "burned" in methodology.lower() or "burn" in methodology.lower()
        assert "reflects" in methodology.lower()

        # Should include limitations
        assert "Note:" in methodology
        assert "smoke" in methodology.lower() or "cloud" in methodology.lower()

    def test_minimal_information_scenario(self):
        """Scenario with minimal information (just event type)."""
        methodology = get_methodology_explanation("storm")
        limitations = get_limitations_text("storm")
        confidence = get_confidence_explanation(75.0)

        # Should all generate valid content
        assert len(methodology) > 100
        assert len(limitations) > 50
        assert len(confidence) > 50

        # Should be factual and professional
        assert "storm" in methodology.lower()
        assert "Note:" in limitations


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_event_type(self):
        """Handles empty event type."""
        text = get_methodology_explanation("")

        # Should return generic methodology
        assert len(text) > 0
        assert "satellite" in text.lower()

    def test_none_algorithm_id(self):
        """Handles None algorithm ID."""
        text = get_methodology_explanation("flood", algorithm_id=None)

        # Should return base methodology
        assert "flood" in text.lower()
        assert len(text) > 0

    def test_whitespace_in_inputs(self):
        """Handles whitespace in inputs."""
        text = get_methodology_explanation(" flood ", algorithm_id="  threshold_sar  ")

        # Should still work
        assert "flood" in text.lower()

    def test_case_insensitive_event_type(self):
        """Handles different case in event type."""
        text1 = get_methodology_explanation("flood")
        text2 = get_methodology_explanation("FLOOD")
        text3 = get_methodology_explanation("Flood")

        # All should return the same methodology
        assert text1 == text2 == text3

    def test_confidence_decimal_handling(self):
        """Handles various decimal confidence values."""
        # Test rounding
        assert "85%" in get_confidence_explanation(85.4)
        assert "86%" in get_confidence_explanation(85.5)

        # Test very small values
        text_small = get_confidence_explanation(0.01)
        assert "0%" in text_small

        # Test values near 100
        text_high = get_confidence_explanation(99.9)
        assert "100%" in text_high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
