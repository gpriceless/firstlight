"""
Baseline Storm Damage Detection Algorithms

This module contains well-validated, interpretable baseline algorithms
for storm damage assessment, including vegetation and structural damage.

Algorithms:
    - wind_damage: Wind damage vegetation detection using NDVI change
    - structural_damage: Building/infrastructure damage assessment
"""

from .wind_damage import (
    WindDamageDetection,
    WindDamageConfig,
    WindDamageResult,
)

from .structural_damage import (
    StructuralDamageAssessment,
    StructuralDamageConfig,
    StructuralDamageResult,
)

__all__ = [
    # Wind Damage Detection
    "WindDamageDetection",
    "WindDamageConfig",
    "WindDamageResult",
    # Structural Damage Assessment
    "StructuralDamageAssessment",
    "StructuralDamageConfig",
    "StructuralDamageResult",
]

# Algorithm registry for easy lookup
STORM_ALGORITHMS = {
    "storm.baseline.wind_damage": WindDamageDetection,
    "storm.baseline.structural_damage": StructuralDamageAssessment,
}


def get_algorithm(algorithm_id: str):
    """
    Get algorithm class by ID.

    Args:
        algorithm_id: Algorithm identifier (e.g., "storm.baseline.wind_damage")

    Returns:
        Algorithm class

    Raises:
        KeyError: If algorithm_id is not found
    """
    if algorithm_id not in STORM_ALGORITHMS:
        available = ", ".join(STORM_ALGORITHMS.keys())
        raise KeyError(f"Unknown algorithm: {algorithm_id}. Available: {available}")

    return STORM_ALGORITHMS[algorithm_id]


def list_algorithms():
    """
    List all available storm algorithms.

    Returns:
        List of (algorithm_id, algorithm_class) tuples
    """
    return list(STORM_ALGORITHMS.items())
