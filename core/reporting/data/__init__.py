"""
Data integration clients for reporting.

Provides access to external data sources for enriching disaster reports:
- Census Bureau: Population and demographic data
- Infrastructure databases: Critical facilities and services
- Emergency resources: Contact information and resources
"""

from .census_client import CensusClient, PopulationData
from .infrastructure_client import (
    InfrastructureClient,
    InfrastructureFeature,
    InfrastructureType,
)
from .emergency_resources import (
    EmergencyResources,
    EmergencyContact,
    StateEmergencyInfo,
    DisasterType,
)

__all__ = [
    "CensusClient",
    "PopulationData",
    "InfrastructureClient",
    "InfrastructureFeature",
    "InfrastructureType",
    "EmergencyResources",
    "EmergencyContact",
    "StateEmergencyInfo",
    "DisasterType",
]
