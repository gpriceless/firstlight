"""
Algorithm Descriptions for Detection Products

Maps algorithm IDs to plain-language descriptions suitable for non-technical audiences.
These descriptions explain WHAT is detected, not HOW the detection works.

Designed for use in automated reports, visualizations, and decision products.
"""

from typing import Optional


# Map algorithm IDs to plain-language descriptions
ALGORITHM_DESCRIPTIONS = {
    # Flood detection algorithms
    "flood.baseline.threshold_sar": "Water detection using radar imagery that can see through clouds",
    "flood.baseline.ndwi": "Water detection using visible and near-infrared light reflectance",
    "flood.baseline.change_detection": "Comparison of before and after imagery to identify new water",
    "flood.baseline.hand_model": "Flood mapping using terrain elevation data",

    # Wildfire detection algorithms
    "wildfire.baseline.thermal_anomaly": "Active fire detection using thermal infrared sensors",
    "wildfire.baseline.nbr_differenced": "Burn severity mapping using vegetation health indicators",
    "wildfire.baseline.ba_classifier": "Burned area mapping using spectral analysis",

    # Storm damage algorithms
    "storm.baseline.wind_damage": "Vegetation damage assessment using imagery comparison",
    "storm.baseline.structural_damage": "Building damage detection using visual analysis",

    # Advanced algorithms
    "flood.advanced.ensemble_fusion": "Combined analysis using multiple detection approaches",
    "flood.advanced.unet_segmentation": "AI-powered detection using trained neural network",
}


# Sensor descriptions for common data sources
SENSOR_DESCRIPTIONS = {
    "sentinel-1": "Sentinel-1 radar",
    "sentinel-2": "Sentinel-2 optical",
    "landsat-8": "Landsat-8 optical",
    "landsat-9": "Landsat-9 optical",
    "modis": "MODIS thermal",
    "viirs": "VIIRS thermal",
    "planet": "Planet optical",
    "sar": "radar",
    "optical": "optical",
    "thermal": "thermal infrared",
}


def get_algorithm_description(algorithm_id: str) -> str:
    """Get plain-language description for an algorithm.

    Args:
        algorithm_id: The algorithm identifier (e.g., "flood.baseline.threshold_sar")

    Returns:
        Human-readable description, or generic fallback if not found

    Examples:
        >>> get_algorithm_description("flood.baseline.threshold_sar")
        "Water detection using radar imagery that can see through clouds"

        >>> get_algorithm_description("unknown.algorithm")
        "Analysis using specialized detection algorithm"
    """
    # Try exact match
    if algorithm_id in ALGORITHM_DESCRIPTIONS:
        return ALGORITHM_DESCRIPTIONS[algorithm_id]

    # Try partial match (e.g., "threshold_sar" from "flood.baseline.threshold_sar")
    for key, description in ALGORITHM_DESCRIPTIONS.items():
        if algorithm_id in key or key in algorithm_id:
            return description

    # Generic fallback
    return "Analysis using specialized detection algorithm"


def get_sensor_description(sensor: str) -> str:
    """Get plain-language description for a sensor.

    Args:
        sensor: The sensor name or type

    Returns:
        Human-readable sensor description

    Examples:
        >>> get_sensor_description("sentinel-1")
        "Sentinel-1 radar"

        >>> get_sensor_description("sar")
        "radar"
    """
    sensor_lower = sensor.lower()

    # Try exact match
    if sensor_lower in SENSOR_DESCRIPTIONS:
        return SENSOR_DESCRIPTIONS[sensor_lower]

    # Try partial match
    for key, description in SENSOR_DESCRIPTIONS.items():
        if key in sensor_lower or sensor_lower in key:
            return description

    # Return as-is if no match
    return sensor


def get_detection_method_text(
    algorithm_id: str,
    event_type: str,
    sensor: Optional[str] = None
) -> str:
    """Generate detection method explanation for reports.

    Creates a 1-2 sentence explanation of how the detection was performed,
    suitable for inclusion in automated reports and visualizations.

    Args:
        algorithm_id: The algorithm identifier
        event_type: The event type being detected (e.g., "flood", "wildfire")
        sensor: Optional sensor name or type

    Returns:
        Plain-language explanation of detection method

    Examples:
        >>> get_detection_method_text("flood.baseline.threshold_sar", "flood", "sentinel-1")
        "Flooding detected using radar imagery that can see through clouds. Analysis performed on Sentinel-1 radar data."

        >>> get_detection_method_text("wildfire.baseline.nbr_differenced", "wildfire", "sentinel-2")
        "Burn severity mapped using vegetation health indicators from Sentinel-2 optical imagery."
    """
    # Get algorithm description
    algo_desc = get_algorithm_description(algorithm_id)

    # Determine what was detected based on event type
    detection_subject = _get_detection_subject(event_type, algo_desc)

    # Build first sentence based on algorithm
    if "burn severity" in algo_desc.lower() or "severity mapping" in algo_desc.lower():
        # For burn severity, use "mapped" verb
        first_sentence = f"{detection_subject} mapped {algo_desc.lower()}"
    elif "detection" in algo_desc.lower():
        # Already says "detection", use as-is but add event context
        first_sentence = algo_desc.replace("detection", f"detected for {event_type.split('.')[0]}")
    else:
        # Standard pattern: "{Subject} detected using..."
        first_sentence = f"{detection_subject} detected {algo_desc.lower()}"

    # Capitalize first letter
    first_sentence = first_sentence[0].upper() + first_sentence[1:]

    # Add sensor information if provided
    if sensor:
        sensor_desc = get_sensor_description(sensor)
        second_sentence = f"Analysis performed on {sensor_desc} data."
        return f"{first_sentence}. {second_sentence}"

    return f"{first_sentence}."


def _get_detection_subject(event_type: str, algorithm_desc: str) -> str:
    """Get the appropriate subject for detection text based on event type.

    Args:
        event_type: The event type (e.g., "flood", "wildfire.urban")
        algorithm_desc: The algorithm description to check for keywords

    Returns:
        Subject text (e.g., "Flooding", "Burn severity")
    """
    # Extract base event type
    base_type = event_type.split(".")[0].lower()

    # Check for specific keywords in algorithm description
    desc_lower = algorithm_desc.lower()

    # If it's burn severity mapping, use that
    if "burn severity" in desc_lower or "severity mapping" in desc_lower:
        return "Burn severity"

    # Standard subject map
    subject_map = {
        "flood": "Flooding",
        "wildfire": "Fire extent",
        "fire": "Fire extent",
        "storm": "Storm damage",
        "hurricane": "Hurricane damage",
        "tornado": "Tornado damage",
    }

    return subject_map.get(base_type, "Event")


def format_algorithm_summary(
    algorithm_id: str,
    event_type: str,
    sensor: Optional[str] = None,
    area_detected_ha: Optional[float] = None,
) -> str:
    """Format a complete summary of detection method and results.

    Combines algorithm description with optional detection statistics
    for a complete summary suitable for reports.

    Args:
        algorithm_id: The algorithm identifier
        event_type: The event type being detected
        sensor: Optional sensor name
        area_detected_ha: Optional detected area in hectares

    Returns:
        Formatted summary text

    Examples:
        >>> format_algorithm_summary("flood.baseline.threshold_sar", "flood", "sentinel-1", 250.5)
        "Flooding detected using radar imagery that can see through clouds. Analysis performed on Sentinel-1 radar data. Total area detected: 250.5 hectares."
    """
    # Get base detection text
    method_text = get_detection_method_text(algorithm_id, event_type, sensor)

    # Add area if provided
    if area_detected_ha is not None:
        area_text = f" Total area detected: {area_detected_ha:.1f} hectares."
        return method_text + area_text

    return method_text
