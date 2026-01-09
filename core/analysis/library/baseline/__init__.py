"""
Baseline Algorithm Library

Collection of well-validated, interpretable baseline algorithms for
hazard detection and mapping across different event types.

Modules:
    - flood: Flood detection and susceptibility algorithms
    - wildfire: Wildfire detection and burn severity algorithms
    - storm: Storm damage assessment algorithms (planned)
"""

from . import flood
from . import wildfire

__all__ = [
    "flood",
    "wildfire",
]
