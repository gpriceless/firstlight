"""
VIS-1.4 Integration Tests

Tests complete annotation pipeline with all VIS-1.4 components working together:
- Area metrics calculation and formatting
- Population impact assessment
- Caption generation
- Algorithm descriptions
- Methodology explanations
- OSM landmark labeling
- Locator maps
- Attribution blocks

This validates that all components integrate correctly and produce
professional, factual output.
"""

import pytest
import numpy as np
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon, box

from core.reporting.imagery import (
    # Area metrics
    calculate_area_metrics,
    format_area,
    # Population impact
    calculate_population_impact,
    format_impact_summary,
    FacilityCount,
    # Captions
    generate_detection_caption,
    generate_map_title,
    Verbosity,
    CaptionContext,
    # Algorithm descriptions
    get_algorithm_description,
    get_sensor_description,
    # Methodology text
    EventType,
    get_methodology_explanation,
    get_limitations_text,
    get_combined_methodology,
    # Maps and attribution
    create_locator_map,
    create_attribution_block,
    sentinel2_source,
    osm_source,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def detection_geometry():
    """Sample flood detection polygon."""
    return Polygon([
        (-95.3698, 29.7604),  # Houston area
        (-95.3698, 29.7804),
        (-95.3498, 29.7804),
        (-95.3498, 29.7604),
        (-95.3698, 29.7604),
    ])


@pytest.fixture
def detection_context():
    """Sample detection context for captions."""
    return CaptionContext(
        event_type="flood",
        algorithm_id="flood.baseline.threshold_sar",
        sensor="sentinel-1",
        acquisition_date=datetime(2024, 9, 15, 10, 30),
        area_km2=12.5,
        location_name="Houston, Texas",
    )


# ============================================================================
# Integration Tests
# ============================================================================

class TestFullAnnotationPipeline:
    """Test complete annotation generation workflow."""

    def test_flood_detection_complete_annotation(self, detection_geometry, detection_context):
        """Generate complete annotation suite for a flood detection."""
        # 1. Calculate area metrics
        area_metrics = calculate_area_metrics(detection_geometry)
        assert area_metrics.total_area_km2 > 0
        area_text = format_area(area_metrics.total_area_km2)
        assert "km²" in area_text or "km2" in area_text
        assert "football" not in area_text.lower()  # No analogies

        # 2. Calculate population impact
        pop_impact = calculate_population_impact(
            detection_geometry,
            population=25000,
            facilities=FacilityCount(hospitals=2, schools=5, emergency=1),
        )
        assert pop_impact.population_in_area >= 0
        impact_summary = format_impact_summary(pop_impact, verbosity="brief")
        assert "in detection area" in impact_summary.lower()
        assert "at risk" not in impact_summary.lower()  # No sensational language

        # 3. Generate caption
        caption = generate_detection_caption(
            context=detection_context,
            verbosity=Verbosity.STANDARD,
        )
        assert len(caption) > 0
        # Check for forbidden phrases
        forbidden = ["football", "devastating", "catastrophic", "massive",
                    "horrific", "tragic", "unprecedented", "historic"]
        for phrase in forbidden:
            assert phrase.lower() not in caption.lower()

        # 4. Get algorithm description
        algo_desc = get_algorithm_description("flood.baseline.threshold_sar")
        assert len(algo_desc) > 0
        assert "radar" in algo_desc.lower()  # Factual description

        # 5. Get sensor description
        sensor_desc = get_sensor_description("sentinel-1")
        assert len(sensor_desc) > 0

        # 6. Get methodology explanation
        methodology = get_methodology_explanation(EventType.FLOOD)
        assert len(methodology) > 0
        assert "water" in methodology.lower()
        assert "imagery" in methodology.lower()

        # 7. Get limitations text
        limitations = get_limitations_text(
            EventType.FLOOD,
            sensor="sentinel-1",
            cloud_cover=0.15,
        )
        assert len(limitations) > 0
        assert "limitation" in limitations.lower() or "accuracy" in limitations.lower()

        # 8. Create locator map
        locator = create_locator_map(
            detection_bounds=detection_geometry.bounds,
            background_bounds=(-96, 29, -94, 30.5),
            size_px=(200, 200),
        )
        assert locator.shape == (200, 200, 4)  # RGBA

        # 9. Create attribution block
        attribution = create_attribution_block([sentinel2_source(), osm_source()])
        assert len(attribution) > 0
        assert "sentinel" in attribution.lower() or "copernicus" in attribution.lower()

        # All components executed successfully
        print("\n=== COMPLETE ANNOTATION SUITE ===")
        print(f"Area: {area_text}")
        print(f"Impact: {impact_summary}")
        print(f"Caption: {caption}")
        print(f"Algorithm: {algo_desc}")
        print(f"Methodology: {methodology[:100]}...")
        print(f"Limitations: {limitations[:100]}...")
        print(f"Attribution: {attribution}")


    def test_wildfire_detection_annotation(self):
        """Generate annotations for wildfire detection."""
        # Create wildfire detection geometry
        wildfire_geometry = Polygon([
            (-122.4, 37.8),  # California area
            (-122.4, 37.85),
            (-122.35, 37.85),
            (-122.35, 37.8),
            (-122.4, 37.8),
        ])

        # Area metrics
        area_metrics = calculate_area_metrics(wildfire_geometry)
        assert area_metrics.total_area_km2 > 0

        # Caption
        context = CaptionContext(
            event_type="wildfire",
            algorithm_id="wildfire.baseline.nbr_differenced",
            sensor="sentinel-2",
            acquisition_date=datetime(2024, 8, 20, 14, 0),
            area_km2=area_metrics.total_area_km2,
            location_name="Northern California",
        )
        caption = generate_detection_caption(context, Verbosity.DETAILED)
        assert len(caption) > 0

        # Methodology
        methodology = get_methodology_explanation(EventType.WILDFIRE)
        assert "vegetation" in methodology.lower()

        # Limitations
        limitations = get_limitations_text(EventType.WILDFIRE, cloud_cover=0.05)
        assert len(limitations) > 0


    def test_map_title_generation(self):
        """Test map title generation for various scenarios."""
        titles = [
            generate_map_title("flood", "Houston, Texas"),
            generate_map_title("wildfire", "Northern California"),
            generate_map_title("storm", "Florida Coast"),
        ]

        for title in titles:
            assert len(title) > 0
            # Check for forbidden phrases
            forbidden = ["devastating", "catastrophic", "massive"]
            for phrase in forbidden:
                assert phrase.lower() not in title.lower()


class TestLanguageStandards:
    """Verify all output adheres to professional language standards."""

    def test_no_sensational_language_in_captions(self, detection_context):
        """Verify captions contain no sensational language."""
        forbidden_phrases = [
            "football field", "football fields",
            "devastating", "catastrophic", "massive",
            "horrific", "tragic",
            "unprecedented", "historic",
            "at risk", "in danger",
        ]

        verbosities = [Verbosity.BRIEF, Verbosity.STANDARD, Verbosity.DETAILED]

        for verbosity in verbosities:
            caption = generate_detection_caption(detection_context, verbosity)

            for phrase in forbidden_phrases:
                assert phrase.lower() not in caption.lower(), \
                    f"Caption contains forbidden phrase '{phrase}': {caption}"


    def test_no_exclamations_in_generated_text(self, detection_context):
        """Verify generated text contains no exclamation marks."""
        caption = generate_detection_caption(detection_context, Verbosity.DETAILED)
        assert "!" not in caption, "Caption contains exclamation mark"

        methodology = get_methodology_explanation(EventType.FLOOD)
        assert "!" not in methodology, "Methodology contains exclamation mark"

        limitations = get_limitations_text(EventType.FLOOD)
        assert "!" not in limitations, "Limitations text contains exclamation mark"


    def test_area_formatting_no_analogies(self):
        """Verify area formatting uses metric units without analogies."""
        test_areas = [0.5, 1.2, 10.5, 100.0, 1000.0]

        for area_km2 in test_areas:
            formatted = format_area(area_km2)
            assert "km" in formatted  # Contains metric unit
            assert "football" not in formatted.lower()  # No analogies
            assert "soccer" not in formatted.lower()
            assert "basketball" not in formatted.lower()


    def test_population_impact_neutral_language(self, detection_geometry):
        """Verify population impact uses neutral, factual language."""
        impact = calculate_population_impact(
            detection_geometry,
            population=50000,
            facilities=FacilityCount(hospitals=3, schools=8, emergency=2),
        )

        summary = format_impact_summary(impact, verbosity="detailed")

        # Should use neutral terms
        assert "in detection area" in summary.lower() or "within detection area" in summary.lower()

        # Should NOT use sensational terms
        forbidden = ["at risk", "in danger", "threatened", "imperiled"]
        for phrase in forbidden:
            assert phrase.lower() not in summary.lower()


class TestCrossComponentIntegration:
    """Test interactions between different annotation components."""

    def test_caption_references_methodology(self, detection_context):
        """Verify caption and methodology use consistent terminology."""
        caption = generate_detection_caption(detection_context, Verbosity.DETAILED)
        methodology = get_methodology_explanation(EventType.FLOOD)

        # Both should mention water/flood
        assert "water" in caption.lower() or "flood" in caption.lower()
        assert "water" in methodology.lower() or "flood" in methodology.lower()


    def test_algorithm_description_matches_sensor(self):
        """Verify algorithm descriptions match their sensor types."""
        # SAR algorithms should reference radar
        sar_algo = get_algorithm_description("flood.baseline.threshold_sar")
        assert "radar" in sar_algo.lower()

        # Optical algorithms should reference optical/infrared
        optical_algo = get_algorithm_description("flood.baseline.ndwi")
        assert "infrared" in optical_algo.lower() or "light" in optical_algo.lower()


    def test_methodology_limitations_consistency(self):
        """Verify methodology and limitations are consistent."""
        methodology = get_methodology_explanation(EventType.FLOOD)
        limitations = get_limitations_text(EventType.FLOOD)

        # Limitations should reference concepts from methodology
        if "cloud" in methodology.lower():
            # If methodology mentions clouds, limitations should too
            assert "cloud" in limitations.lower()


class TestOutputQuality:
    """Test overall quality and completeness of generated output."""

    def test_all_event_types_have_complete_content(self):
        """Verify all event types have complete methodology and limitations."""
        event_types = [EventType.FLOOD, EventType.WILDFIRE, EventType.STORM]

        for event_type in event_types:
            # Methodology exists and is substantial
            methodology = get_methodology_explanation(event_type)
            assert len(methodology) > 100, f"Methodology too short for {event_type}"
            assert "\n" in methodology, f"Methodology not multi-paragraph for {event_type}"

            # Limitations exist and are substantial
            limitations = get_limitations_text(event_type)
            assert len(limitations) > 100, f"Limitations too short for {event_type}"

            # Combined output is coherent
            combined = get_combined_methodology(event_type)
            assert methodology in combined
            assert limitations in combined


    def test_caption_length_appropriate(self, detection_context):
        """Verify captions are appropriate length for their verbosity."""
        brief = generate_detection_caption(detection_context, Verbosity.BRIEF)
        standard = generate_detection_caption(detection_context, Verbosity.STANDARD)
        detailed = generate_detection_caption(detection_context, Verbosity.DETAILED)

        # Brief should be shortest
        assert len(brief) < len(standard)
        assert len(standard) <= len(detailed)

        # None should be excessively long
        assert len(detailed) < 500, "Detailed caption is too long"


    def test_professional_tone_throughout(self, detection_context, detection_geometry):
        """Verify all output maintains professional, factual tone."""
        # Generate all types of output
        caption = generate_detection_caption(detection_context, Verbosity.STANDARD)
        methodology = get_methodology_explanation(EventType.FLOOD)
        limitations = get_limitations_text(EventType.FLOOD)
        impact = calculate_population_impact(detection_geometry, population=10000)
        impact_text = format_impact_summary(impact)
        area = format_area(12.5)

        all_text = f"{caption} {methodology} {limitations} {impact_text} {area}"

        # Professional indicators
        assert "detection" in all_text.lower() or "analysis" in all_text.lower()
        assert "imagery" in all_text.lower() or "satellite" in all_text.lower()

        # No informal language
        informal_terms = ["huge", "tiny", "tons of", "lots of", "gonna", "wanna"]
        for term in informal_terms:
            assert term.lower() not in all_text.lower()

        # No ALL CAPS (except acronyms)
        words = all_text.split()
        all_caps_words = [w for w in words if w.isupper() and len(w) > 3]
        # A few acronyms are OK, but not full sentences in caps
        assert len(all_caps_words) < 5, f"Too many all-caps words: {all_caps_words}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
