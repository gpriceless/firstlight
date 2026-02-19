"""
OGC API Processes processor implementations.

This package dynamically generates pygeoapi BaseProcessor subclasses
from FirstLight's AlgorithmRegistry at startup.
"""

from core.ogc.processors.factory import (
    build_processor_class,
    build_all_processors,
    get_processor_config,
    PHASE_ALGORITHM_CATEGORIES,
)

__all__ = [
    "build_processor_class",
    "build_all_processors",
    "get_processor_config",
    "PHASE_ALGORITHM_CATEGORIES",
]
