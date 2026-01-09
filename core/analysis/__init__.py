"""
Analysis and modeling layer for event intelligence.

This module contains:
- Algorithm library and registry
- Pipeline assembly and execution
- Multi-sensor fusion
- Algorithm selection logic
"""

from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType,
    get_global_registry,
    load_default_algorithms
)

__all__ = [
    'AlgorithmRegistry',
    'AlgorithmMetadata',
    'AlgorithmCategory',
    'DataType',
    'get_global_registry',
    'load_default_algorithms'
]
