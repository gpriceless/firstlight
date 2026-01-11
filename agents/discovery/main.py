"""
Discovery Agent - Main agent for data discovery and acquisition.

Wraps the core data broker functionality and provides an agent interface
for the Orchestrator to coordinate data discovery tasks.

Key responsibilities:
- Handle discovery requests from Orchestrator
- Query multiple STAC catalogs in parallel
- Rank and select optimal data sources
- Coordinate data acquisition
- Manage retries and fallbacks for catalog queries
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from agents.base import (
    AgentMessage,
    AgentState,
    AgentType,
    BaseAgent,
    MessageContext,
    MessageType,
    RetryPolicy,
)
from agents.discovery.catalog import CatalogQueryManager, CatalogQueryResult
from agents.discovery.selection import DatasetSelector, SelectionCriteria, SelectionResult
from agents.discovery.acquisition import AcquisitionManager, AcquisitionRequest, AcquisitionResult

from core.data.broker import BrokerQuery, BrokerResponse, DataBroker
from core.data.discovery.base import DiscoveryResult, DiscoveryError
from core.data.providers.registry import Provider, ProviderRegistry


logger = logging.getLogger(__name__)


class DiscoveryStatus(Enum):
    """Status of a discovery operation."""

    PENDING = "pending"
    QUERYING = "querying"
    RANKING = "ranking"
    SELECTING = "selecting"
    ACQUIRING = "acquiring"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some sources succeeded, others failed


@dataclass
class DiscoveryRequest:
    """
    Request for data discovery from the Orchestrator.

    Attributes:
        request_id: Unique request identifier
        event_id: Event being processed
        spatial: GeoJSON geometry or bbox
        temporal: Temporal extent with start/end
        intent_class: Event classification (e.g., "flood.coastal")
        data_types: Required data types
        constraints: Per-data-type constraints
        ranking_weights: Custom ranking weights
        cache_policy: Cache usage policy
        priority: Request priority
        metadata: Additional request metadata
    """

    request_id: str
    event_id: str
    spatial: Dict[str, Any]
    temporal: Dict[str, str]
    intent_class: str
    data_types: List[str] = field(default_factory=list)
    constraints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ranking_weights: Optional[Dict[str, float]] = None
    cache_policy: str = "prefer_cache"
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_broker_query(self) -> BrokerQuery:
        """Convert to BrokerQuery for core data broker."""
        return BrokerQuery(
            event_id=self.event_id,
            spatial=self.spatial,
            temporal=self.temporal,
            intent_class=self.intent_class,
            data_types=self.data_types,
            constraints=self.constraints,
            ranking_weights=self.ranking_weights,
            cache_policy=self.cache_policy,
        )


@dataclass
class DiscoveryResponse:
    """
    Response from discovery operation.

    Attributes:
        request_id: Original request ID
        event_id: Event being processed
        status: Discovery operation status
        broker_response: Response from data broker
        catalog_results: Raw catalog query results
        selection_result: Final dataset selection
        acquisition_result: Data acquisition result (if requested)
        timing: Timing statistics
        errors: Any errors encountered
    """

    request_id: str
    event_id: str
    status: DiscoveryStatus
    broker_response: Optional[BrokerResponse] = None
    catalog_results: List[CatalogQueryResult] = field(default_factory=list)
    selection_result: Optional[SelectionResult] = None
    acquisition_result: Optional[AcquisitionResult] = None
    timing: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "request_id": self.request_id,
            "event_id": self.event_id,
            "status": self.status.value,
            "broker_response": self.broker_response.to_dict() if self.broker_response else None,
            "catalog_results": [r.to_dict() for r in self.catalog_results],
            "selection_result": self.selection_result.to_dict() if self.selection_result else None,
            "acquisition_result": self.acquisition_result.to_dict() if self.acquisition_result else None,
            "timing": self.timing,
            "errors": self.errors,
        }


@dataclass
class AvailabilityStatus:
    """
    Status of data source availability.

    Attributes:
        source_id: Data source identifier
        available: Whether source is currently available
        last_checked: When availability was last checked
        estimated_latency_ms: Estimated query latency
        error_message: Error if unavailable
        metadata: Additional availability metadata
    """

    source_id: str
    available: bool
    last_checked: datetime
    estimated_latency_ms: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "available": self.available,
            "last_checked": self.last_checked.isoformat(),
            "estimated_latency_ms": self.estimated_latency_ms,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class DiscoveryAgent(BaseAgent):
    """
    Discovery Agent for data discovery and acquisition tasks.

    Wraps the core data broker and provides:
    - Multi-catalog parallel queries
    - Dataset ranking and selection
    - Data acquisition coordination
    - Retry and fallback handling
    - Cache management for catalog results
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        data_broker: Optional[DataBroker] = None,
        provider_registry: Optional[ProviderRegistry] = None,
        retry_policy: Optional[RetryPolicy] = None,
        max_concurrent_catalog_queries: int = 10,
        catalog_query_timeout_seconds: float = 30.0,
        cache_catalog_results: bool = True,
        cache_ttl_seconds: int = 300,
        enable_fallback_sources: bool = True,
        acquisition_enabled: bool = True,
    ):
        """
        Initialize DiscoveryAgent.

        Args:
            agent_id: Optional agent identifier
            data_broker: Optional data broker instance
            provider_registry: Optional provider registry
            retry_policy: Retry policy configuration
            max_concurrent_catalog_queries: Max parallel queries
            catalog_query_timeout_seconds: Timeout per query
            cache_catalog_results: Whether to cache results
            cache_ttl_seconds: Cache TTL
            enable_fallback_sources: Enable fallback sources
            acquisition_enabled: Enable data acquisition
        """
        super().__init__(
            agent_id=agent_id or "discovery",
            agent_type=AgentType.DISCOVERY,
            retry_policy=retry_policy,
        )

        # Configuration
        self._max_concurrent_catalog_queries = max_concurrent_catalog_queries
        self._catalog_query_timeout_seconds = catalog_query_timeout_seconds
        self._cache_catalog_results = cache_catalog_results
        self._cache_ttl_seconds = cache_ttl_seconds
        self._enable_fallback_sources = enable_fallback_sources
        self._acquisition_enabled = acquisition_enabled

        # Core components
        self._data_broker = data_broker
        self._provider_registry = provider_registry

        # Agent-specific components (initialized on start)
        self._catalog_manager: Optional[CatalogQueryManager] = None
        self._dataset_selector: Optional[DatasetSelector] = None
        self._acquisition_manager: Optional[AcquisitionManager] = None

        # State tracking
        self._active_discoveries: Dict[str, DiscoveryRequest] = {}
        self._availability_cache: Dict[str, AvailabilityStatus] = {}
        self._discovery_history: List[DiscoveryResponse] = []

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming messages.

        Handles:
        - discovery_request: Start data discovery
        - availability_check: Check source availability
        - acquisition_request: Download data

        Args:
            message: Incoming message

        Returns:
            Response message if applicable
        """
        payload = message.payload
        request_type = payload.get("request_type", "discovery_request")

        try:
            if request_type == "discovery_request":
                response = await self._handle_discovery_request(message)
            elif request_type == "availability_check":
                response = await self._handle_availability_check(message)
            elif request_type == "acquisition_request":
                response = await self._handle_acquisition_request(message)
            else:
                response = {
                    "success": False,
                    "error": f"Unknown request type: {request_type}"
                }

            return message.create_response(response)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return message.create_response({
                "success": False,
                "error": str(e)
            })

    async def run(self) -> None:
        """
        Main agent execution loop.

        Processes messages from the inbox until shutdown.
        """
        logger.info(f"Discovery Agent {self.agent_id} starting run loop")

        # Initialize components
        await self._initialize_components()

        while not self._shutdown_event.is_set():
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(
                    self._inbox.get(),
                    timeout=1.0
                )

                # Process message
                response = await self.process_message(message)
                if response:
                    await self.send_message(response)

            except asyncio.TimeoutError:
                # No message received, continue loop
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in run loop: {e}")
                await self.on_error(e)

        # Cleanup
        await self._cleanup_components()
        logger.info(f"Discovery Agent {self.agent_id} run loop stopped")

    async def _initialize_components(self) -> None:
        """Initialize agent components."""
        logger.info("Initializing Discovery Agent components...")

        # Initialize provider registry if not provided
        if self._provider_registry is None:
            self._provider_registry = ProviderRegistry()
            try:
                import os
                definitions_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../openspec/definitions/datasources"
                )
                if os.path.exists(definitions_path):
                    self._provider_registry.load_from_definitions(definitions_path)
            except Exception as e:
                logger.warning(f"Failed to load provider definitions: {e}")

        # Initialize data broker if not provided
        if self._data_broker is None:
            self._data_broker = DataBroker(provider_registry=self._provider_registry)
            await self._register_discovery_adapters()

        # Initialize catalog query manager
        self._catalog_manager = CatalogQueryManager(
            max_concurrent_queries=self._max_concurrent_catalog_queries,
            query_timeout_seconds=self._catalog_query_timeout_seconds,
            cache_enabled=self._cache_catalog_results,
            cache_ttl_seconds=self._cache_ttl_seconds,
        )

        # Initialize dataset selector
        self._dataset_selector = DatasetSelector(
            provider_registry=self._provider_registry
        )

        # Initialize acquisition manager
        if self._acquisition_enabled:
            self._acquisition_manager = AcquisitionManager()

        logger.info("Discovery Agent components initialized")

    async def _cleanup_components(self) -> None:
        """Clean up agent resources."""
        if self._catalog_manager:
            await self._catalog_manager.close()

        if self._acquisition_manager:
            await self._acquisition_manager.close()

        self._availability_cache.clear()

    async def _register_discovery_adapters(self) -> None:
        """Register discovery adapters with the data broker."""
        from core.data.discovery.stac import STACAdapter

        stac_adapter = STACAdapter()
        self._data_broker.register_adapter(stac_adapter)
        logger.debug("Registered STAC discovery adapter")

    # Public API methods

    async def discover_data(self, query: BrokerQuery) -> BrokerResponse:
        """
        Main entry point for data discovery.

        Wraps the core data broker's discover method with additional
        agent-level functionality including retries and fallbacks.

        Args:
            query: Broker query with spatial/temporal/intent parameters

        Returns:
            BrokerResponse with selected datasets and metadata
        """
        logger.info(
            f"Starting discovery for event {query.event_id}, "
            f"intent: {query.intent_class}, data_types: {query.data_types}"
        )

        start_time = datetime.now(timezone.utc)

        try:
            response = await self._data_broker.discover(query)

            elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Discovery completed for {query.event_id} in {elapsed_ms:.1f}ms. "
                f"Selected {len(response.selected_datasets)} datasets"
            )

            return response

        except Exception as e:
            logger.error(f"Discovery failed for {query.event_id}: {e}")
            raise

    async def check_availability(self, source_id: str) -> AvailabilityStatus:
        """
        Check availability of a data source.

        Args:
            source_id: Data source identifier

        Returns:
            AvailabilityStatus with current availability info
        """
        # Check cache first
        cached = self._availability_cache.get(source_id)
        if cached:
            cache_age = (datetime.now(timezone.utc) - cached.last_checked).total_seconds()
            if cache_age < self._cache_ttl_seconds:
                return cached

        # Query provider registry
        provider = self._provider_registry.get_provider(source_id)
        if not provider:
            return AvailabilityStatus(
                source_id=source_id,
                available=False,
                last_checked=datetime.now(timezone.utc),
                error_message=f"Unknown provider: {source_id}"
            )

        # Check endpoint availability
        start_time = datetime.now(timezone.utc)
        try:
            available = await self._check_endpoint_availability(provider)
            latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            status = AvailabilityStatus(
                source_id=source_id,
                available=available,
                last_checked=datetime.now(timezone.utc),
                estimated_latency_ms=latency_ms,
                metadata={"provider_type": provider.type}
            )

        except Exception as e:
            status = AvailabilityStatus(
                source_id=source_id,
                available=False,
                last_checked=datetime.now(timezone.utc),
                error_message=str(e)
            )

        # Update cache
        self._availability_cache[source_id] = status
        return status

    async def _check_endpoint_availability(self, provider: Provider) -> bool:
        """Check if provider endpoint is accessible."""
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available, skipping endpoint check")
            return True  # Assume available if we can't check

        endpoint = provider.access.get("endpoint")
        if not endpoint:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    endpoint,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 500
        except Exception:
            return False

    async def rank_sources(
        self,
        sources: List[DiscoveryResult],
        criteria: Optional[SelectionCriteria] = None
    ) -> List[DiscoveryResult]:
        """
        Rank discovered data sources.

        Args:
            sources: List of discovered sources to rank
            criteria: Optional selection criteria

        Returns:
            Sources sorted by ranking (best first)
        """
        if not sources:
            return []

        if self._dataset_selector is None:
            # Fall back to simple scoring
            return sorted(
                sources,
                key=lambda s: (
                    s.spatial_coverage_percent,
                    -getattr(s, 'cloud_cover_percent', 0) if s.cloud_cover_percent else 0
                ),
                reverse=True
            )

        return await self._dataset_selector.rank_sources(sources, criteria)

    async def handle_discovery_failure(
        self,
        error: Exception,
        request: DiscoveryRequest
    ) -> Optional[DiscoveryResponse]:
        """
        Handle discovery failure with fallback strategies.

        Args:
            error: The exception that occurred
            request: Original discovery request

        Returns:
            Partial response if fallback succeeded, None otherwise
        """
        logger.warning(
            f"Handling discovery failure for {request.event_id}: {error}"
        )

        errors = [{
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }]

        if not self._enable_fallback_sources:
            return DiscoveryResponse(
                request_id=request.request_id,
                event_id=request.event_id,
                status=DiscoveryStatus.FAILED,
                errors=errors
            )

        # Try fallback sources
        logger.info("Attempting fallback sources...")

        try:
            fallback_results = []

            for data_type in request.data_types:
                providers = self._provider_registry.get_providers_by_type(data_type)

                # Filter to open data sources as fallback
                fallback_providers = [
                    p for p in providers
                    if p.cost.get("tier") == "open"
                ]

                if fallback_providers and self._catalog_manager:
                    results = await self._catalog_manager.query_catalogs(
                        providers=fallback_providers[:3],
                        spatial=request.spatial,
                        temporal=request.temporal,
                        constraints=request.constraints.get(data_type, {})
                    )
                    fallback_results.extend(results)

            if fallback_results:
                return DiscoveryResponse(
                    request_id=request.request_id,
                    event_id=request.event_id,
                    status=DiscoveryStatus.PARTIAL,
                    catalog_results=fallback_results,
                    errors=errors
                )

        except Exception as fallback_error:
            errors.append({
                "type": type(fallback_error).__name__,
                "message": str(fallback_error),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": "fallback"
            })

        return DiscoveryResponse(
            request_id=request.request_id,
            event_id=request.event_id,
            status=DiscoveryStatus.FAILED,
            errors=errors
        )

    # Message handlers

    async def _handle_discovery_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle discovery request from Orchestrator."""
        payload = message.payload

        # Parse request
        request = DiscoveryRequest(
            request_id=message.message_id,
            event_id=payload.get("event_id", ""),
            spatial=payload.get("spatial", {}),
            temporal=payload.get("temporal", {}),
            intent_class=payload.get("intent_class", ""),
            data_types=payload.get("data_types", []),
            constraints=payload.get("constraints", {}),
            ranking_weights=payload.get("ranking_weights"),
            cache_policy=payload.get("cache_policy", "prefer_cache"),
            priority=payload.get("priority", 5),
            metadata=payload.get("metadata", {}),
        )

        # Track active discovery
        self._active_discoveries[request.request_id] = request

        try:
            response = await self._execute_discovery(request)
            return response.to_dict()

        except Exception as e:
            error_response = await self.handle_discovery_failure(e, request)
            if error_response:
                return error_response.to_dict()
            return {"success": False, "error": str(e)}

        finally:
            self._active_discoveries.pop(request.request_id, None)

    async def _handle_availability_check(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle availability check request."""
        source_id = message.payload.get("source_id", "")

        status = await self.check_availability(source_id)
        return status.to_dict()

    async def _handle_acquisition_request(self, message: AgentMessage) -> Dict[str, Any]:
        """Handle data acquisition request."""
        if not self._acquisition_manager:
            return {
                "success": False,
                "error": "Acquisition not enabled"
            }

        payload = message.payload

        request = AcquisitionRequest(
            request_id=message.message_id,
            datasets=payload.get("datasets", []),
            output_path=payload.get("output_path", ""),
            format_options=payload.get("format_options", {}),
        )

        try:
            result = await self._acquisition_manager.acquire(request)
            return result.to_dict()
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _execute_discovery(self, request: DiscoveryRequest) -> DiscoveryResponse:
        """
        Execute full discovery workflow.

        Args:
            request: Discovery request

        Returns:
            Discovery response with results
        """
        timing: Dict[str, float] = {}
        start_time = datetime.now(timezone.utc)

        # Phase 1: Query catalogs
        phase_start = datetime.now(timezone.utc)
        catalog_results: List[CatalogQueryResult] = []

        if self._catalog_manager:
            providers = self._provider_registry.get_applicable_providers(
                event_class=request.intent_class,
                data_types=request.data_types
            )

            catalog_results = await self._catalog_manager.query_catalogs(
                providers=providers,
                spatial=request.spatial,
                temporal=request.temporal,
                constraints=request.constraints
            )

        timing["catalog_query_ms"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds() * 1000

        # Phase 2: Run broker discovery
        phase_start = datetime.now(timezone.utc)
        broker_response = await self.discover_data(request.to_broker_query())
        timing["broker_discovery_ms"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds() * 1000

        # Phase 3: Dataset selection
        phase_start = datetime.now(timezone.utc)
        selection_result: Optional[SelectionResult] = None

        if self._dataset_selector and broker_response.selected_datasets:
            criteria = SelectionCriteria(
                event_class=request.intent_class,
                data_types=set(request.data_types),
                constraints=request.constraints,
                ranking_weights=request.ranking_weights or {},
            )
            selection_result = await self._dataset_selector.select(
                candidates=broker_response.selected_datasets,
                criteria=criteria
            )

        timing["selection_ms"] = (
            datetime.now(timezone.utc) - phase_start
        ).total_seconds() * 1000

        # Calculate total time
        timing["total_ms"] = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds() * 1000

        # Determine status
        if broker_response.selected_datasets:
            status = DiscoveryStatus.COMPLETED
        elif catalog_results:
            status = DiscoveryStatus.PARTIAL
        else:
            status = DiscoveryStatus.FAILED

        response = DiscoveryResponse(
            request_id=request.request_id,
            event_id=request.event_id,
            status=status,
            broker_response=broker_response,
            catalog_results=catalog_results,
            selection_result=selection_result,
            timing=timing,
        )

        # Store in history
        self._discovery_history.append(response)

        return response
