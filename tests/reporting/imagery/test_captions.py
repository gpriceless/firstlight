"""
Tests for Caption Generation Module (VIS-1.4 Task 4).

Verifies:
- Caption generation at all verbosity levels
- Map title generation
- Factual, professional language (NO hyperbole)
- Proper integration with area_metrics and impact_metrics
"""

import pytest
from datetime import datetime

from core.reporting.imagery.captions import (
    Verbosity,
    CaptionContext,
    generate_detection_caption,
    generate_map_title,
)
from core.reporting.imagery.area_metrics import AreaMetrics
from core.reporting.imagery.impact_metrics import PopulationImpact, FacilityCount


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def small_area_metrics():
    """Small detection area (< 100 hectares)."""
    return AreaMetrics(
        hectares=24.5,
        acres=60.5,
        square_kilometers=0.245,
        square_miles=0.095,
        percent_of_aoi=18.0
    )


@pytest.fixture
def large_area_metrics():
    """Large detection area (>= 100 hectares)."""
    return AreaMetrics(
        hectares=2450.0,
        acres=6053.0,
        square_kilometers=24.5,
        square_miles=9.5,
        percent_of_aoi=18.0
    )


@pytest.fixture
def population_impact_full():
    """Full population impact with all metrics."""
    return PopulationImpact(
        estimated_population=4200,
        residential_structures=1800,
        facilities=FacilityCount(
            hospitals=1,
            schools=3,
            fire_stations=0,
            police_stations=0,
            shelters=0
        ),
        road_km_affected=12.5
    )


@pytest.fixture
def population_impact_minimal():
    """Minimal population impact."""
    return PopulationImpact(
        estimated_population=0,
        residential_structures=0,
        facilities=FacilityCount(),
        road_km_affected=0.0
    )


# ============================================================================
# Caption Generation Tests
# ============================================================================


def test_brief_caption_minimal(small_area_metrics):
    """Test brief caption with minimal context."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=small_area_metrics,
        detection_date=datetime(2026, 1, 25)
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.BRIEF)

    assert "Flood extent:" in caption
    assert "24.5 hectares" in caption
    assert "60.5 acres" in caption
    assert "2026-01-25" in caption
    assert caption.endswith(".")


def test_brief_caption_no_date(small_area_metrics):
    """Test brief caption without date."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=small_area_metrics
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.BRIEF)

    assert "Flood extent:" in caption
    assert "hectares" in caption
    assert caption.endswith(".")


def test_standard_caption_full(small_area_metrics, population_impact_full):
    """Test standard caption with full context."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=small_area_metrics,
        population_impact=population_impact_full
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.STANDARD)

    # Should include area with percentage
    assert "Flood extent:" in caption
    assert "24.5 hectares" in caption
    assert "18% of analysis area" in caption

    # Should include population
    assert "Estimated affected population: 4,200" in caption


def test_standard_caption_area_only(large_area_metrics):
    """Test standard caption with only area metrics."""
    context = CaptionContext(
        event_type="wildfire",
        area_metrics=large_area_metrics
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.STANDARD)

    # Should use km² for large areas
    assert "Wildfire extent:" in caption
    assert "24.5 km²" in caption
    assert "9.5 mi²" in caption
    assert "18% of analysis area" in caption


def test_detailed_caption_full(large_area_metrics, population_impact_full):
    """Test detailed caption with all available information."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=large_area_metrics,
        population_impact=population_impact_full,
        algorithm_id="sar_water_detection",
        detection_date=datetime(2026, 1, 25),
        confidence_mean=0.85
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Area
    assert "Flood extent:" in caption
    assert "24.5 km²" in caption
    assert "18% of analysis area" in caption

    # Population
    assert "Estimated affected population: 4,200" in caption

    # Structures
    assert "1,800 residential structures" in caption
    assert "in detection area" in caption

    # Facilities
    assert "Critical facilities in detection area:" in caption
    assert "3 schools" in caption
    assert "1 hospital" in caption

    # Metadata
    assert "2026-01-25" in caption
    assert "85% average confidence" in caption


def test_detailed_caption_minimal(small_area_metrics, population_impact_minimal):
    """Test detailed caption with minimal data."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=small_area_metrics,
        population_impact=population_impact_minimal
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Should still generate valid caption
    assert "Flood extent:" in caption
    assert "hectares" in caption

    # Should NOT include zero population (check implementation handles gracefully)
    # Population impact with 0 should not add sentence


def test_verbosity_length_progression(large_area_metrics, population_impact_full):
    """Test that caption length increases with verbosity."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=large_area_metrics,
        population_impact=population_impact_full,
        detection_date=datetime(2026, 1, 25),
        confidence_mean=0.85
    )

    brief = generate_detection_caption(context, verbosity=Verbosity.BRIEF)
    standard = generate_detection_caption(context, verbosity=Verbosity.STANDARD)
    detailed = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Length should increase
    assert len(brief) < len(standard) < len(detailed)


# ============================================================================
# Map Title Tests
# ============================================================================


def test_map_title_event_only():
    """Test map title with only event type."""
    title = generate_map_title("flood")
    assert title == "Flood Detection"


def test_map_title_with_location():
    """Test map title with location."""
    title = generate_map_title("flood", location_name="Harris County, TX")
    assert title == "Flood Detection - Harris County, TX"


def test_map_title_with_date():
    """Test map title with date."""
    title = generate_map_title("wildfire", detection_date=datetime(2026, 1, 25))
    assert title == "Wildfire Detection - 2026-01-25"


def test_map_title_full():
    """Test map title with all components."""
    title = generate_map_title(
        "flood",
        location_name="Harris County, TX",
        detection_date=datetime(2026, 1, 25, 14, 30)
    )
    assert title == "Flood Detection - Harris County, TX - 2026-01-25"


def test_map_title_event_type_formatting():
    """Test that event types are formatted correctly."""
    assert generate_map_title("flood") == "Flood Detection"
    assert generate_map_title("wildfire") == "Wildfire Detection"
    assert generate_map_title("storm") == "Storm Detection"
    assert generate_map_title("burn_severity") == "Burn Severity Detection"
    assert generate_map_title("wind_damage") == "Wind Damage Detection"


# ============================================================================
# Language Validation Tests - NO HYPERBOLE
# ============================================================================


FORBIDDEN_PHRASES = [
    "devastating",
    "catastrophic",
    "massive",
    "horrific",
    "tragic",
    "unprecedented",
    "historic",
    "!",  # Exclamation marks
    "at risk",
    "in danger",
]


@pytest.mark.parametrize("verbosity", [Verbosity.BRIEF, Verbosity.STANDARD, Verbosity.DETAILED])
def test_no_hyperbolic_language(verbosity, large_area_metrics, population_impact_full):
    """Test that generated captions contain NO hyperbolic language."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=large_area_metrics,
        population_impact=population_impact_full,
        detection_date=datetime(2026, 1, 25),
        confidence_mean=0.85
    )

    caption = generate_detection_caption(context, verbosity=verbosity)

    # Check for forbidden phrases
    caption_lower = caption.lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase.lower() not in caption_lower, \
            f"Forbidden phrase '{phrase}' found in caption: {caption}"


def test_uses_in_detection_area_not_at_risk(population_impact_full):
    """Test that captions use 'in detection area' not 'at risk' or 'in danger'."""
    context = CaptionContext(
        event_type="flood",
        population_impact=population_impact_full
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Should say "in detection area"
    assert "in detection area" in caption.lower()

    # Should NOT say "at risk" or "in danger"
    assert "at risk" not in caption.lower()
    assert "in danger" not in caption.lower()


def test_factual_tone_numbers_only(large_area_metrics, population_impact_full):
    """Test that captions use factual numbers and measurements."""
    context = CaptionContext(
        event_type="flood",
        area_metrics=large_area_metrics,
        population_impact=population_impact_full
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Should contain actual numbers
    assert "4,200" in caption  # Population
    assert "1,800" in caption  # Structures
    assert "24.5 km²" in caption  # Area

    # Should NOT contain vague qualifiers
    vague_terms = ["many", "numerous", "countless", "extensive", "widespread"]
    caption_lower = caption.lower()
    for term in vague_terms:
        assert term not in caption_lower


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_context():
    """Test caption generation with minimal context."""
    context = CaptionContext(event_type="flood")

    caption = generate_detection_caption(context, verbosity=Verbosity.BRIEF)

    # Should still generate valid caption
    assert "Flood" in caption
    assert len(caption) > 0


def test_zero_population_not_displayed():
    """Test that zero population is not displayed as 'Estimated: 0'."""
    impact = PopulationImpact(
        estimated_population=0,
        residential_structures=0,
        facilities=FacilityCount(),
        road_km_affected=0.0
    )

    context = CaptionContext(
        event_type="flood",
        population_impact=impact
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.STANDARD)

    # Should not include population line if zero
    # (current implementation checks for > 0)
    assert "Estimated affected population: 0" not in caption


def test_facility_pluralization():
    """Test that facility counts are properly pluralized."""
    # Single facility
    impact_single = PopulationImpact(
        estimated_population=100,
        residential_structures=0,
        facilities=FacilityCount(hospitals=1),
        road_km_affected=0.0
    )

    context = CaptionContext(
        event_type="flood",
        population_impact=impact_single
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)
    assert "1 hospital" in caption  # Singular
    assert "hospitals" not in caption  # Not plural

    # Multiple facilities
    impact_multiple = PopulationImpact(
        estimated_population=100,
        residential_structures=0,
        facilities=FacilityCount(schools=3),
        road_km_affected=0.0
    )

    context_multiple = CaptionContext(
        event_type="flood",
        population_impact=impact_multiple
    )

    caption_multiple = generate_detection_caption(context_multiple, verbosity=Verbosity.DETAILED)
    assert "3 schools" in caption_multiple  # Plural


def test_confidence_formatting():
    """Test confidence percentage formatting."""
    context = CaptionContext(
        event_type="flood",
        confidence_mean=0.856  # Should round to 85% or 86%
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Should show as percentage without decimal
    assert "85% average confidence" in caption or "86% average confidence" in caption
    assert ".8" not in caption  # No decimal values


# ============================================================================
# Integration Tests
# ============================================================================


def test_real_world_flood_scenario():
    """Test realistic flood detection scenario."""
    area = AreaMetrics(
        hectares=1250.0,
        acres=3088.0,
        square_kilometers=12.5,
        square_miles=4.8,
        percent_of_aoi=23.0
    )

    impact = PopulationImpact(
        estimated_population=8500,
        residential_structures=3200,
        facilities=FacilityCount(
            hospitals=2,
            schools=5,
            fire_stations=1,
            police_stations=1,
            shelters=2
        ),
        road_km_affected=45.0
    )

    context = CaptionContext(
        event_type="flood",
        area_metrics=area,
        population_impact=impact,
        detection_date=datetime(2026, 1, 25),
        confidence_mean=0.82
    )

    caption = generate_detection_caption(context, verbosity=Verbosity.DETAILED)

    # Verify all components present
    assert "Flood extent: 12.5 km²" in caption
    assert "23% of analysis area" in caption
    assert "8,500" in caption
    assert "3,200 residential structures" in caption
    assert "5 schools" in caption
    assert "2 hospitals" in caption
    assert "2026-01-25" in caption
    assert "82% average confidence" in caption

    # Verify professional tone
    assert caption.endswith(".")
    caption_lower = caption.lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase.lower() not in caption_lower


def test_real_world_wildfire_scenario():
    """Test realistic wildfire/burn severity scenario."""
    area = AreaMetrics(
        hectares=35000.0,
        acres=86487.0,
        square_kilometers=350.0,
        square_miles=135.1,
        percent_of_aoi=None  # May not have AOI for wildfires
    )

    context = CaptionContext(
        event_type="burn_severity",
        area_metrics=area,
        detection_date=datetime(2026, 1, 20)
    )

    # Brief for situation reports
    brief = generate_detection_caption(context, verbosity=Verbosity.BRIEF)
    assert "Burn Severity extent:" in brief
    assert "2026-01-20" in brief

    # Standard for daily updates
    standard = generate_detection_caption(context, verbosity=Verbosity.STANDARD)
    assert "350.0 km²" in standard
    assert "135.1 mi²" in standard


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
