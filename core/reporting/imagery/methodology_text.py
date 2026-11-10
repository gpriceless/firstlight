"""
Methodology Text Module for VIS-1.4 Task 9

Provides plain-language explanations of detection methodology and data limitations
for use in automated reports. Designed for non-technical audiences (emergency managers,
local officials) who need to understand WHAT was detected and WHY certain limitations exist.

Language Guidelines:
- Write for a city council member, not a scientist
- Avoid technical jargon (e.g., "backscatter", "spectral indices", "NDWI")
- Use accessible terms (e.g., "radar signals", "water signatures", "vegetation health")
- Be honest about limitations
- Keep explanations to 2-3 short paragraphs maximum
"""

from typing import Optional
from enum import Enum


class EventType(Enum):
    """Supported event types for methodology explanations."""
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    STORM = "storm"


# ============================================================================
# Methodology Templates by Event Type
# ============================================================================

METHODOLOGY_TEMPLATES = {
    EventType.FLOOD: """
Flood detection analysis identifies areas where water is present that was not
present before the event. This is done by comparing satellite imagery from
before and after the reported flooding. Areas showing water signatures in the
post-event imagery that were dry in pre-event imagery are classified as flooded.

Water is detected based on how the surface reflects radar signals or light.
Water appears distinctly different from land in both radar and optical satellite
imagery, allowing automated systems to map its extent.
""".strip(),

    EventType.WILDFIRE: """
Wildfire analysis uses satellite imagery to map burned areas and assess burn
severity. The analysis compares images from before and after the fire to identify
areas where vegetation has been damaged or destroyed.

Burned vegetation reflects light differently than healthy vegetation, particularly
in wavelengths beyond what the human eye can see. These changes allow satellites
to detect and map fire-affected areas, even distinguishing between lightly scorched
and completely burned zones.
""".strip(),

    EventType.STORM: """
Storm damage assessment uses satellite imagery to identify areas where vegetation
or structures have been damaged by high winds, flooding, or debris. The analysis
compares before and after imagery to detect changes consistent with storm damage.

Damaged vegetation, debris fields, and structural changes are visible in high-resolution
satellite imagery. Computer analysis identifies patterns of damage across large areas
that would be difficult to assess from ground level alone.
""".strip(),
}


# ============================================================================
# Data Limitations by Event Type
# ============================================================================

LIMITATIONS_TEMPLATES = {
    EventType.FLOOD: """
Note: Satellite-based flood mapping has limitations. Water under dense tree cover
may not be detected, as satellites cannot see through vegetation canopies. Urban
flooding can be difficult to distinguish from shadows cast by tall buildings.
Small-scale flooding (such as individual flooded basements) may not be visible
at satellite resolution. This analysis should be used alongside ground-based
observations and local knowledge.
""".strip(),

    EventType.WILDFIRE: """
Note: Satellite burn mapping has limitations. Small fires or spot fires between
satellite passes may be missed. Burn severity assessments indicate vegetation damage
but do not measure soil damage or subsurface impacts. Cloud cover or heavy smoke
during imagery collection can obscure burned areas. This analysis should be verified
with ground surveys for critical decisions such as declaring areas safe for reentry.
""".strip(),

    EventType.STORM: """
Note: Satellite storm damage assessment has limitations. Damage to building interiors
or infrastructure below ground level is not visible from satellites. Small-scale damage
(such as individual broken windows or roof shingles) may not be detectable at satellite
resolution. Cloud cover during or immediately after the storm can delay imagery collection.
This analysis should be supplemented with ground-based damage assessments.
""".strip(),
}


# ============================================================================
# Algorithm-Specific Method Explanations
# ============================================================================

ALGORITHM_METHOD_DETAILS = {
    # Flood algorithms
    "threshold_sar": """
This analysis used radar imagery that can see through clouds and darkness. Radar signals
bounce off surfaces differently depending on whether they hit water or land - water appears
very dark in radar images while land appears brighter. By analyzing these signal differences,
the system identifies areas covered by water.
""".strip(),

    "ndwi": """
This analysis used optical satellite imagery that measures how surfaces reflect different
colors of light, including some wavelengths invisible to the human eye. Water absorbs certain
wavelengths that vegetation and soil reflect strongly. By comparing these reflectance patterns,
the system distinguishes water from dry land.
""".strip(),

    "change_detection": """
This analysis compared satellite images captured before and after the event. Areas that changed
from dry land to water between the two image dates are classified as newly flooded. This approach
helps distinguish flood water from permanent water bodies like lakes and rivers.
""".strip(),

    # Wildfire algorithms
    "thermal_anomaly": """
This analysis used thermal infrared sensors that detect heat. Active fires emit intense heat
signatures visible from space, even through light smoke. The system identifies spots significantly
hotter than their surroundings as active fire locations.
""".strip(),

    "nbr_differenced": """
This analysis compared how vegetation reflects infrared light before and after the fire. Healthy
vegetation has a distinct reflectance pattern that changes dramatically when burned. By measuring
these changes, the system maps burned areas and estimates severity levels from light scorching
to complete combustion.
""".strip(),

    "ba_classifier": """
This analysis used multiple satellite measurements (surface reflectance, temperature, vegetation
patterns) to identify burned areas. A statistical model trained on known burned areas classifies
each location as burned or unburned based on its characteristics in post-fire imagery.
""".strip(),

    # Storm algorithms
    "wind_damage": """
This analysis compared vegetation patterns before and after the storm. High winds cause changes
visible from space: flattened crops, fallen trees, and disrupted forest canopies. The system
detects these changes by comparing pre- and post-storm imagery.
""".strip(),

    "structural_damage": """
This analysis used high-resolution imagery to identify damaged structures. Changes such as
missing roofs, collapsed walls, or debris fields are detected by comparing before and after
images. The system highlights areas with significant structural changes for further investigation.
""".strip(),
}


# ============================================================================
# Main Functions
# ============================================================================


def get_methodology_explanation(
    event_type: str,
    algorithm_id: Optional[str] = None,
    include_limitations: bool = False
) -> str:
    """
    Generate plain-language methodology explanation.

    Args:
        event_type: Type of event ("flood", "wildfire", "storm")
        algorithm_id: Optional specific algorithm used (e.g., "threshold_sar", "ndwi")
        include_limitations: Whether to include data limitations section

    Returns:
        2-3 paragraph explanation suitable for non-technical readers

    Example:
        >>> get_methodology_explanation("flood", "threshold_sar", include_limitations=True)
        "Flood detection analysis identifies areas where water is present...
         This analysis used radar imagery that can see through clouds...
         Note: Satellite-based flood mapping has limitations..."
    """
    # Normalize event type
    event_type_normalized = event_type.strip().lower().split(".")[0]  # Handle "flood.coastal" -> "flood"

    try:
        event_enum = EventType(event_type_normalized)
    except ValueError:
        # Unknown event type - return generic explanation
        return _get_generic_methodology()

    # Start with base methodology
    paragraphs = [METHODOLOGY_TEMPLATES[event_enum]]

    # Add algorithm-specific details if provided
    if algorithm_id:
        method_detail = _get_algorithm_method_detail(algorithm_id.strip())
        if method_detail:
            paragraphs.append(method_detail)

    # Add limitations if requested
    if include_limitations:
        paragraphs.append(LIMITATIONS_TEMPLATES[event_enum])

    return "\n\n".join(paragraphs)


def get_limitations_text(
    event_type: str,
    sensor: Optional[str] = None,
    cloud_cover: Optional[float] = None,
) -> str:
    """
    Get data limitations disclaimer for reports.

    Args:
        event_type: Type of event ("flood", "wildfire", "storm")
        sensor: Optional sensor name (for future enhancements)
        cloud_cover: Optional cloud cover fraction (for future enhancements)

    Returns:
        Plain-language limitations disclaimer

    Example:
        >>> get_limitations_text("flood")
        "Note: Satellite-based flood mapping has limitations. Water under dense..."
    """
    # Normalize event type
    event_type_normalized = event_type.lower().split(".")[0]

    try:
        event_enum = EventType(event_type_normalized)
        return LIMITATIONS_TEMPLATES[event_enum]
    except ValueError:
        return _get_generic_limitations()


def get_combined_methodology(
    event_type: str,
    algorithm_id: Optional[str] = None,
    sensor: Optional[str] = None,
    cloud_cover: Optional[float] = None,
) -> str:
    """
    Get combined methodology and limitations text for reports.

    Convenience function that combines methodology explanation and
    limitations into a single text block suitable for report sections.

    Args:
        event_type: Type of event ("flood", "wildfire", "storm")
        algorithm_id: Optional specific algorithm used
        sensor: Optional sensor name
        cloud_cover: Optional cloud cover fraction

    Returns:
        Combined methodology and limitations text with "Methodology" and
        "Limitations" headers

    Example:
        >>> text = get_combined_methodology("flood", algorithm_id="threshold_sar")
        >>> "Methodology" in text
        True
        >>> "Limitations" in text
        True
    """
    methodology = get_methodology_explanation(
        event_type=event_type,
        algorithm_id=algorithm_id,
        include_limitations=False,
    )
    limitations = get_limitations_text(
        event_type=event_type,
        sensor=sensor,
        cloud_cover=cloud_cover,
    )

    return f"Methodology\n\n{methodology}\n\nLimitations\n\n{limitations}"


def get_confidence_explanation(confidence_percent: float) -> str:
    """
    Explain what confidence score means in plain language.

    Args:
        confidence_percent: Confidence value as percentage (0-100)

    Returns:
        Plain-language explanation of confidence level

    Example:
        >>> get_confidence_explanation(85.0)
        "Detection confidence: 85%. This indicates high certainty in the mapped extent.
         Higher values indicate more reliable detection."
    """
    # Ensure value is in 0-100 range
    confidence = max(0, min(100, confidence_percent))

    # Determine confidence level description
    if confidence >= 80:
        level_desc = "high certainty"
    elif confidence >= 60:
        level_desc = "moderate certainty"
    else:
        level_desc = "lower certainty"

    return (
        f"Detection confidence: {confidence:.0f}%. This indicates {level_desc} in the mapped extent. "
        f"Higher values indicate more reliable detection."
    )


# ============================================================================
# Helper Functions
# ============================================================================


def _get_algorithm_method_detail(algorithm_id: str) -> Optional[str]:
    """
    Extract algorithm-specific method details.

    Args:
        algorithm_id: Full algorithm ID (e.g., "flood.baseline.threshold_sar")

    Returns:
        Method detail text, or None if no specific detail available
    """
    # Extract method name from algorithm ID
    # "flood.baseline.threshold_sar" -> "threshold_sar"
    # "wildfire.baseline.nbr_differenced" -> "nbr_differenced"
    parts = algorithm_id.split(".")
    if len(parts) >= 3:
        method_name = parts[2]
    elif len(parts) == 2:
        method_name = parts[1]
    else:
        method_name = algorithm_id

    return ALGORITHM_METHOD_DETAILS.get(method_name)


def _get_generic_methodology() -> str:
    """
    Return generic methodology explanation for unknown event types.

    Returns:
        Generic explanation text
    """
    return """
This analysis used satellite imagery to detect and map areas affected by the event.
Satellite sensors measure how the Earth's surface reflects or emits different types
of energy (such as visible light, infrared, or radar signals). By analyzing these
measurements, automated systems can identify patterns associated with the event being
studied.

The analysis compared imagery from before and after the event to identify changes
consistent with the event type. This approach helps distinguish new impacts from
pre-existing conditions.
""".strip()


def _get_generic_limitations() -> str:
    """
    Return generic limitations disclaimer for unknown event types.

    Returns:
        Generic limitations text
    """
    return """
Note: Satellite-based analysis has limitations. Satellite sensors have finite resolution
and may not detect small-scale features. Cloud cover or atmospheric conditions can obscure
the surface. Timing of satellite passes may miss short-duration events. This analysis should
be verified with ground-based observations for critical decisions.
""".strip()
