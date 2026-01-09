"""
Baseline Flood Detection Algorithms

This module contains well-validated, interpretable baseline algorithms
for flood extent detection and susceptibility mapping.

Algorithms:
    - threshold_sar: SAR backscatter threshold detection
    - ndwi_optical: NDWI-based optical flood detection
    - change_detection: Pre/post temporal change detection
    - hand_model: Height Above Nearest Drainage susceptibility
"""

from .threshold_sar import (
    ThresholdSARAlgorithm,
    ThresholdSARConfig,
    ThresholdSARResult
)

from .ndwi_optical import (
    NDWIOpticalAlgorithm,
    NDWIOpticalConfig,
    NDWIOpticalResult
)

from .change_detection import (
    ChangeDetectionAlgorithm,
    ChangeDetectionConfig,
    ChangeDetectionResult
)

from .hand_model import (
    HANDModelAlgorithm,
    HANDModelConfig,
    HANDModelResult
)

__all__ = [
    # Threshold SAR
    "ThresholdSARAlgorithm",
    "ThresholdSARConfig",
    "ThresholdSARResult",
    # NDWI Optical
    "NDWIOpticalAlgorithm",
    "NDWIOpticalConfig",
    "NDWIOpticalResult",
    # Change Detection
    "ChangeDetectionAlgorithm",
    "ChangeDetectionConfig",
    "ChangeDetectionResult",
    # HAND Model
    "HANDModelAlgorithm",
    "HANDModelConfig",
    "HANDModelResult",
]

# Algorithm registry for easy lookup
FLOOD_ALGORITHMS = {
    "flood.baseline.threshold_sar": ThresholdSARAlgorithm,
    "flood.baseline.ndwi_optical": NDWIOpticalAlgorithm,
    "flood.baseline.change_detection": ChangeDetectionAlgorithm,
    "flood.baseline.hand_model": HANDModelAlgorithm,
}


def get_algorithm(algorithm_id: str):
    """
    Get algorithm class by ID.

    Args:
        algorithm_id: Algorithm identifier (e.g., "flood.baseline.threshold_sar")

    Returns:
        Algorithm class

    Raises:
        KeyError: If algorithm_id is not found
    """
    if algorithm_id not in FLOOD_ALGORITHMS:
        available = ", ".join(FLOOD_ALGORITHMS.keys())
        raise KeyError(f"Unknown algorithm: {algorithm_id}. Available: {available}")

    return FLOOD_ALGORITHMS[algorithm_id]


def list_algorithms():
    """
    List all available flood algorithms.

    Returns:
        List of (algorithm_id, algorithm_class) tuples
    """
    return list(FLOOD_ALGORITHMS.items())
