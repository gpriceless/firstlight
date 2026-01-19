"""
Discovery Agent Package - Data discovery and acquisition coordination.

This package provides the Discovery Agent which wraps the core data broker
and handles all data discovery and acquisition tasks for the FirstLight
orchestration system.

Components:
- DiscoveryAgent: Main agent for handling discovery requests
- CatalogQueryManager: Parallel STAC catalog query management
- DatasetSelector: Dataset selection and ranking logic
- AcquisitionManager: Data download coordination

Example usage:
    from agents.discovery import DiscoveryAgent

    # Create and start the agent
    agent = DiscoveryAgent()
    await agent.start()
    asyncio.create_task(agent.run())

    # Send discovery request via message
    from agents.base import AgentMessage, AgentType, MessageType

    message = AgentMessage(
        from_agent=AgentType.ORCHESTRATOR,
        to_agent=AgentType.DISCOVERY,
        message_type=MessageType.REQUEST,
        payload={
            "request_type": "discovery_request",
            "event_id": "flood_2024_001",
            "spatial": {
                "type": "Polygon",
                "coordinates": [[[-122.5, 37.5], [-122.5, 38.5],
                                [-121.5, 38.5], [-121.5, 37.5], [-122.5, 37.5]]]
            },
            "temporal": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-10T00:00:00Z"
            },
            "intent_class": "flood.coastal",
            "data_types": ["optical", "sar", "dem"]
        }
    )

    await agent.receive_message(message)
"""

# Main agent
from agents.discovery.main import (
    DiscoveryAgent,
    DiscoveryRequest,
    DiscoveryResponse,
    DiscoveryStatus,
    AvailabilityStatus,
)

# Catalog query management
from agents.discovery.catalog import (
    CatalogQueryManager,
    CatalogQueryResult,
    CatalogQueryStatus,
)

# Dataset selection
from agents.discovery.selection import (
    DatasetSelector,
    SelectionCriteria,
    SelectionResult,
    SelectionOutcome,
    CandidateEvaluation,
    apply_fallback_sources,
)

# Data acquisition
from agents.discovery.acquisition import (
    AcquisitionManager,
    AcquisitionRequest,
    AcquisitionResult,
    AcquisitionStatus,
    DatasetInfo,
    DownloadProgress,
    DownloadStatus,
    DownloadedFile,
    batch_acquire,
)


__all__ = [
    # Main agent
    "DiscoveryAgent",
    "DiscoveryRequest",
    "DiscoveryResponse",
    "DiscoveryStatus",
    "AvailabilityStatus",
    # Catalog
    "CatalogQueryManager",
    "CatalogQueryResult",
    "CatalogQueryStatus",
    # Selection
    "DatasetSelector",
    "SelectionCriteria",
    "SelectionResult",
    "SelectionOutcome",
    "CandidateEvaluation",
    "apply_fallback_sources",
    # Acquisition
    "AcquisitionManager",
    "AcquisitionRequest",
    "AcquisitionResult",
    "AcquisitionStatus",
    "DatasetInfo",
    "DownloadProgress",
    "DownloadStatus",
    "DownloadedFile",
    "batch_acquire",
]
