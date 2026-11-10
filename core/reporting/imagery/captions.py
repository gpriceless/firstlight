"""
Caption Generation Module (VIS-1.4 Task 4).

Auto-generates factual captions for detection maps and imagery products.
Uses PROFESSIONAL, OBJECTIVE language only - NO hyperbolic phrases.

This is emergency response software. All language is factual and appropriate
for official reports and decision-making.

Provides:
- Verbosity: Control caption detail level (BRIEF, STANDARD, DETAILED)
- CaptionContext: Context for generating detection captions
- generate_detection_caption: Create factual caption from metrics
- generate_map_title: Create clean map titles
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from core.reporting.imagery.area_metrics import AreaMetrics, format_area
from core.reporting.imagery.impact_metrics import PopulationImpact


class Verbosity(Enum):
    """Control level of detail in generated captions."""

    BRIEF = "brief"  # 1 sentence
    STANDARD = "standard"  # 2-3 sentences
    DETAILED = "detailed"  # Full paragraph


@dataclass
class CaptionContext:
    """Context for generating detection captions.

    Attributes:
        event_type: Type of event ("flood", "wildfire", "storm")
        area_metrics: Area measurements from detection
        population_impact: Population and infrastructure impacts
        algorithm_id: Detection algorithm identifier
        detection_date: Date of detection/imagery
        confidence_mean: Average confidence score (0-1)
    """

    event_type: str
    area_metrics: Optional[AreaMetrics] = None
    population_impact: Optional[PopulationImpact] = None
    algorithm_id: Optional[str] = None
    detection_date: Optional[datetime] = None
    confidence_mean: Optional[float] = None


def generate_detection_caption(
    context: CaptionContext,
    verbosity: Verbosity = Verbosity.STANDARD
) -> str:
    """Generate factual caption for detection results.

    Creates professional, objective descriptions of detection results.
    Language is appropriate for emergency response reports.

    FORBIDDEN PHRASES (never appear in output):
        - "devastating", "catastrophic", "massive"
        - "horrific", "tragic", "disaster" (as adjective)
        - "unprecedented", "historic"
        - Exclamation marks
        - "at risk", "in danger" (use "in detection area")

    Args:
        context: Detection context with metrics
        verbosity: Level of detail (BRIEF, STANDARD, DETAILED)

    Returns:
        Factual caption text

    Examples:
        BRIEF:
            "Flood extent: 2,450 hectares detected on 2026-01-25."

        STANDARD:
            "Flood extent: 2,450 hectares (6,050 acres), representing 18% of the
             analysis area. Estimated affected population: 4,200."

        DETAILED:
            "Flood extent: 2,450 hectares (6,050 acres), representing 18% of the
             analysis area. Estimated affected population: 4,200. Approximately
             1,800 residential structures in detection area. Critical facilities:
             3 schools, 1 hospital. Detection performed using radar imagery on
             2026-01-25 with 85% average confidence."
    """
    if verbosity == Verbosity.BRIEF:
        return _generate_brief_caption(context)
    elif verbosity == Verbosity.STANDARD:
        return _generate_standard_caption(context)
    else:  # DETAILED
        return _generate_detailed_caption(context)


def generate_map_title(
    event_type: str,
    location_name: Optional[str] = None,
    detection_date: Optional[datetime] = None
) -> str:
    """Generate factual map title.

    Creates clean, professional titles for map products. Includes event type,
    optional location, and optional date.

    Args:
        event_type: Type of event ("flood", "wildfire", "storm")
        location_name: Optional location descriptor
        detection_date: Optional detection/imagery date

    Returns:
        Formatted map title

    Examples:
        >>> generate_map_title("flood")
        "Flood Detection"

        >>> generate_map_title("flood", location_name="Harris County, TX")
        "Flood Detection - Harris County, TX"

        >>> generate_map_title("wildfire", detection_date=datetime(2026, 1, 25))
        "Wildfire Detection - 2026-01-25"

        >>> generate_map_title(
        ...     "flood",
        ...     location_name="Harris County, TX",
        ...     detection_date=datetime(2026, 1, 25)
        ... )
        "Flood Detection - Harris County, TX - 2026-01-25"
    """
    # Capitalize event type appropriately
    event_display = _format_event_type(event_type)

    # Build title parts
    parts = [f"{event_display} Detection"]

    if location_name:
        parts.append(location_name)

    if detection_date:
        date_str = detection_date.strftime("%Y-%m-%d")
        parts.append(date_str)

    return " - ".join(parts)


# ============================================================================
# Private Implementation
# ============================================================================


def _generate_brief_caption(context: CaptionContext) -> str:
    """Generate brief (1 sentence) caption."""
    event_display = _format_event_type(context.event_type)

    # Basic format: "[Event] extent: [area] detected on [date]."
    parts = [f"{event_display} extent:"]

    if context.area_metrics:
        area_text = format_area(context.area_metrics, include_percentage=False)
        parts.append(area_text)
    else:
        parts.append("area detected")

    if context.detection_date:
        date_str = context.detection_date.strftime("%Y-%m-%d")
        parts.append(f"on {date_str}")

    caption = " ".join(parts)
    if not caption.endswith("."):
        caption += "."

    return caption


def _generate_standard_caption(context: CaptionContext) -> str:
    """Generate standard (2-3 sentence) caption."""
    sentences = []

    # Sentence 1: Area with percentage
    if context.area_metrics:
        event_display = _format_event_type(context.event_type)
        area_text = format_area(context.area_metrics, include_percentage=True)
        sentences.append(f"{event_display} extent: {area_text}.")

    # Sentence 2: Population impact if available
    if context.population_impact:
        if context.population_impact.estimated_population > 0:
            pop = context.population_impact.estimated_population
            sentences.append(f"Estimated affected population: {pop:,}.")

    return " ".join(sentences)


def _generate_detailed_caption(context: CaptionContext) -> str:
    """Generate detailed (full paragraph) caption."""
    sentences = []

    # Sentence 1: Area with percentage
    if context.area_metrics:
        event_display = _format_event_type(context.event_type)
        area_text = format_area(context.area_metrics, include_percentage=True)
        sentences.append(f"{event_display} extent: {area_text}.")

    # Sentence 2: Population
    if context.population_impact:
        if context.population_impact.estimated_population > 0:
            pop = context.population_impact.estimated_population
            sentences.append(f"Estimated affected population: {pop:,}.")

        # Sentence 3: Residential structures
        if context.population_impact.residential_structures > 0:
            structures = context.population_impact.residential_structures
            sentences.append(
                f"Approximately {structures:,} residential structures "
                "in detection area."
            )

        # Sentence 4: Critical facilities
        facilities = context.population_impact.facilities
        facility_parts = []

        if facilities.schools > 0:
            plural = "s" if facilities.schools > 1 else ""
            facility_parts.append(f"{facilities.schools} school{plural}")
        if facilities.hospitals > 0:
            plural = "s" if facilities.hospitals > 1 else ""
            facility_parts.append(f"{facilities.hospitals} hospital{plural}")
        if facilities.fire_stations > 0:
            plural = "s" if facilities.fire_stations > 1 else ""
            facility_parts.append(f"{facilities.fire_stations} fire station{plural}")
        if facilities.police_stations > 0:
            plural = "s" if facilities.police_stations > 1 else ""
            facility_parts.append(f"{facilities.police_stations} police station{plural}")
        if facilities.shelters > 0:
            plural = "s" if facilities.shelters > 1 else ""
            facility_parts.append(f"{facilities.shelters} shelter{plural}")

        if facility_parts:
            facilities_text = ", ".join(facility_parts)
            sentences.append(f"Critical facilities in detection area: {facilities_text}.")

    # Sentence 5: Detection metadata
    meta_parts = []
    if context.detection_date:
        date_str = context.detection_date.strftime("%Y-%m-%d")
        meta_parts.append(f"on {date_str}")
    if context.confidence_mean is not None:
        confidence_pct = int(context.confidence_mean * 100)
        meta_parts.append(f"with {confidence_pct}% average confidence")

    if meta_parts:
        meta_text = " ".join(meta_parts)
        sentences.append(f"Detection performed {meta_text}.")

    return " ".join(sentences)


def _format_event_type(event_type: str) -> str:
    """Format event type for display.

    Args:
        event_type: Event type string ("flood", "wildfire", "storm")

    Returns:
        Properly capitalized display name

    Examples:
        >>> _format_event_type("flood")
        "Flood"
        >>> _format_event_type("wildfire")
        "Wildfire"
        >>> _format_event_type("burn_severity")
        "Burn Severity"
    """
    # Handle common event types
    event_lower = event_type.lower()

    if event_lower == "flood":
        return "Flood"
    elif event_lower in ("wildfire", "fire"):
        return "Wildfire"
    elif event_lower == "storm":
        return "Storm"
    elif event_lower in ("burn_severity", "burn severity"):
        return "Burn Severity"
    elif event_lower == "wind_damage":
        return "Wind Damage"
    else:
        # Default: capitalize each word, replace underscores with spaces
        return event_type.replace("_", " ").title()
