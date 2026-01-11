"""
Webhook Management for Event Notifications.

Provides webhook registration, event delivery with retries,
signature verification, and delivery logging.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class WebhookConfig(BaseModel):
    """Webhook system configuration."""

    # Delivery settings
    max_retries: int = Field(default=5, description="Maximum delivery retry attempts")
    retry_delay_seconds: int = Field(default=30, description="Initial retry delay")
    retry_backoff_multiplier: float = Field(default=2.0, description="Exponential backoff multiplier")
    max_retry_delay_seconds: int = Field(default=3600, description="Maximum retry delay (1 hour)")

    # Timeout settings
    delivery_timeout_seconds: int = Field(default=30, description="HTTP request timeout")
    connect_timeout_seconds: int = Field(default=10, description="Connection timeout")

    # Security settings
    signature_header: str = Field(default="X-Webhook-Signature", description="Signature header name")
    timestamp_header: str = Field(default="X-Webhook-Timestamp", description="Timestamp header name")
    id_header: str = Field(default="X-Webhook-ID", description="Webhook ID header name")
    signature_tolerance_seconds: int = Field(default=300, description="Max age for signature validation")

    # Limits
    max_webhooks_per_user: int = Field(default=10, description="Max webhooks per user")
    max_payload_size_bytes: int = Field(default=1_000_000, description="Max payload size (1MB)")

    # Queue settings
    queue_max_size: int = Field(default=10000, description="Max items in delivery queue")
    batch_size: int = Field(default=100, description="Batch size for processing")

    class Config:
        env_prefix = "WEBHOOK_"


def get_webhook_config() -> WebhookConfig:
    """Get webhook configuration from environment."""
    return WebhookConfig(
        max_retries=int(os.getenv("WEBHOOK_MAX_RETRIES", "5")),
        retry_delay_seconds=int(os.getenv("WEBHOOK_RETRY_DELAY", "30")),
        retry_backoff_multiplier=float(os.getenv("WEBHOOK_BACKOFF_MULTIPLIER", "2.0")),
        max_retry_delay_seconds=int(os.getenv("WEBHOOK_MAX_RETRY_DELAY", "3600")),
        delivery_timeout_seconds=int(os.getenv("WEBHOOK_TIMEOUT", "30")),
        connect_timeout_seconds=int(os.getenv("WEBHOOK_CONNECT_TIMEOUT", "10")),
        signature_header=os.getenv("WEBHOOK_SIGNATURE_HEADER", "X-Webhook-Signature"),
        timestamp_header=os.getenv("WEBHOOK_TIMESTAMP_HEADER", "X-Webhook-Timestamp"),
        id_header=os.getenv("WEBHOOK_ID_HEADER", "X-Webhook-ID"),
        signature_tolerance_seconds=int(os.getenv("WEBHOOK_SIGNATURE_TOLERANCE", "300")),
        max_webhooks_per_user=int(os.getenv("WEBHOOK_MAX_PER_USER", "10")),
        max_payload_size_bytes=int(os.getenv("WEBHOOK_MAX_PAYLOAD_SIZE", "1000000")),
        queue_max_size=int(os.getenv("WEBHOOK_QUEUE_SIZE", "10000")),
        batch_size=int(os.getenv("WEBHOOK_BATCH_SIZE", "100")),
    )


# =============================================================================
# Event Types
# =============================================================================


class WebhookEventType(str, Enum):
    """Types of webhook events."""

    # Event lifecycle
    EVENT_CREATED = "event.created"
    EVENT_STARTED = "event.started"
    EVENT_PROGRESS = "event.progress"
    EVENT_COMPLETED = "event.completed"
    EVENT_FAILED = "event.failed"
    EVENT_CANCELLED = "event.cancelled"

    # Product lifecycle
    PRODUCT_READY = "product.ready"
    PRODUCT_EXPIRED = "product.expired"

    # Quality checks
    QUALITY_PASSED = "quality.passed"
    QUALITY_FAILED = "quality.failed"
    QUALITY_REVIEW_REQUIRED = "quality.review_required"

    # System events
    SYSTEM_MAINTENANCE = "system.maintenance"
    SYSTEM_ALERT = "system.alert"

    # Test event
    TEST = "test"


# Event type groups for subscription filtering
EVENT_TYPE_GROUPS = {
    "all": set(WebhookEventType),
    "events": {
        WebhookEventType.EVENT_CREATED,
        WebhookEventType.EVENT_STARTED,
        WebhookEventType.EVENT_PROGRESS,
        WebhookEventType.EVENT_COMPLETED,
        WebhookEventType.EVENT_FAILED,
        WebhookEventType.EVENT_CANCELLED,
    },
    "products": {
        WebhookEventType.PRODUCT_READY,
        WebhookEventType.PRODUCT_EXPIRED,
    },
    "quality": {
        WebhookEventType.QUALITY_PASSED,
        WebhookEventType.QUALITY_FAILED,
        WebhookEventType.QUALITY_REVIEW_REQUIRED,
    },
    "system": {
        WebhookEventType.SYSTEM_MAINTENANCE,
        WebhookEventType.SYSTEM_ALERT,
    },
}


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class Webhook:
    """Represents a registered webhook endpoint."""

    webhook_id: str
    user_id: str
    url: str
    secret: str  # Used for signature generation
    event_types: Set[WebhookEventType]
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_secret: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "webhook_id": self.webhook_id,
            "user_id": self.user_id,
            "url": self.url,
            "event_types": [e.value for e in self.event_types],
            "name": self.name,
            "description": self.description,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "metadata": self.metadata,
        }
        if include_secret:
            result["secret"] = self.secret
        return result


@dataclass
class WebhookEvent:
    """Represents a webhook event to be delivered."""

    event_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: Optional[str] = None  # Source event/job ID

    def to_dict(self) -> Dict[str, Any]:
        """Convert to delivery payload."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "data": self.payload,
        }


@dataclass
class DeliveryAttempt:
    """Record of a webhook delivery attempt."""

    attempt_id: str
    webhook_id: str
    event_id: str
    attempt_number: int
    timestamp: datetime
    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "attempt_id": self.attempt_id,
            "webhook_id": self.webhook_id,
            "event_id": self.event_id,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "status_code": self.status_code,
            "response_body": self.response_body[:500] if self.response_body else None,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


@dataclass
class DeliveryTask:
    """A pending delivery task in the queue."""

    task_id: str
    webhook: Webhook
    event: WebhookEvent
    attempt_number: int = 0
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Signature Generation and Verification
# =============================================================================


def generate_signature(payload: bytes, secret: str, timestamp: int) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Args:
        payload: The JSON payload bytes
        secret: The webhook secret key
        timestamp: Unix timestamp of the request

    Returns:
        The signature as a hex string prefixed with algorithm
    """
    # Combine timestamp and payload for signature
    signed_content = f"{timestamp}.{payload.decode()}"
    signature = hmac.new(
        secret.encode(),
        signed_content.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


def verify_signature(
    payload: bytes,
    signature: str,
    secret: str,
    timestamp: int,
    tolerance_seconds: int = 300,
) -> bool:
    """
    Verify a webhook signature.

    Args:
        payload: The raw request body bytes
        signature: The signature from the request header
        secret: The webhook secret key
        timestamp: The timestamp from the request header
        tolerance_seconds: Maximum age of the request

    Returns:
        True if signature is valid and timestamp is within tolerance
    """
    # Check timestamp tolerance
    now = int(time.time())
    if abs(now - timestamp) > tolerance_seconds:
        logger.warning(f"Webhook signature timestamp too old: {now - timestamp}s")
        return False

    # Generate expected signature
    expected = generate_signature(payload, secret, timestamp)

    # Constant-time comparison
    return hmac.compare_digest(expected, signature)


# =============================================================================
# Webhook Store
# =============================================================================


class WebhookStore:
    """
    Storage interface for webhooks.

    In production, this should be backed by a database.
    This in-memory implementation is for development and testing.
    """

    def __init__(self):
        self._webhooks: Dict[str, Webhook] = {}
        self._user_webhooks: Dict[str, Set[str]] = {}  # user_id -> webhook_ids
        self._delivery_log: List[DeliveryAttempt] = []
        self._max_log_entries = 10000

    def create(
        self,
        user_id: str,
        url: str,
        event_types: Set[WebhookEventType],
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Webhook:
        """Create a new webhook."""
        webhook_id = str(uuid4())
        secret = secrets.token_urlsafe(32)

        webhook = Webhook(
            webhook_id=webhook_id,
            user_id=user_id,
            url=url,
            secret=secret,
            event_types=event_types,
            name=name,
            description=description,
            metadata=metadata or {},
        )

        self._webhooks[webhook_id] = webhook

        if user_id not in self._user_webhooks:
            self._user_webhooks[user_id] = set()
        self._user_webhooks[user_id].add(webhook_id)

        logger.info(f"Created webhook {webhook_id} for user {user_id}")
        return webhook

    def get(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self._webhooks.get(webhook_id)

    def get_by_user(self, user_id: str) -> List[Webhook]:
        """Get all webhooks for a user."""
        webhook_ids = self._user_webhooks.get(user_id, set())
        return [self._webhooks[wid] for wid in webhook_ids if wid in self._webhooks]

    def get_for_event_type(self, event_type: WebhookEventType) -> List[Webhook]:
        """Get all active webhooks subscribed to an event type."""
        return [
            w for w in self._webhooks.values()
            if w.is_active and event_type in w.event_types
        ]

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        event_types: Optional[Set[WebhookEventType]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Webhook]:
        """Update a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if webhook is None:
            return None

        if url is not None:
            webhook.url = url
        if event_types is not None:
            webhook.event_types = event_types
        if name is not None:
            webhook.name = name
        if description is not None:
            webhook.description = description
        if is_active is not None:
            webhook.is_active = is_active
        if metadata is not None:
            webhook.metadata.update(metadata)

        webhook.updated_at = datetime.now(timezone.utc)

        logger.info(f"Updated webhook {webhook_id}")
        return webhook

    def delete(self, webhook_id: str) -> bool:
        """Delete a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if webhook is None:
            return False

        del self._webhooks[webhook_id]

        if webhook.user_id in self._user_webhooks:
            self._user_webhooks[webhook.user_id].discard(webhook_id)

        logger.info(f"Deleted webhook {webhook_id}")
        return True

    def regenerate_secret(self, webhook_id: str) -> Optional[str]:
        """Regenerate the secret for a webhook."""
        webhook = self._webhooks.get(webhook_id)
        if webhook is None:
            return None

        new_secret = secrets.token_urlsafe(32)
        webhook.secret = new_secret
        webhook.updated_at = datetime.now(timezone.utc)

        logger.info(f"Regenerated secret for webhook {webhook_id}")
        return new_secret

    def log_delivery_attempt(self, attempt: DeliveryAttempt) -> None:
        """Log a delivery attempt."""
        self._delivery_log.append(attempt)

        # Trim log if too large
        if len(self._delivery_log) > self._max_log_entries:
            self._delivery_log = self._delivery_log[-self._max_log_entries:]

        # Update webhook stats
        webhook = self._webhooks.get(attempt.webhook_id)
        if webhook:
            if attempt.success:
                webhook.success_count += 1
            else:
                webhook.failure_count += 1
            webhook.last_triggered_at = attempt.timestamp

    def get_delivery_log(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DeliveryAttempt]:
        """Get delivery log, optionally filtered by webhook."""
        logs = self._delivery_log
        if webhook_id:
            logs = [l for l in logs if l.webhook_id == webhook_id]
        return logs[-limit:]


# =============================================================================
# Webhook Manager
# =============================================================================


class WebhookManager:
    """
    Manages webhook registrations and event delivery.

    Handles webhook lifecycle, event queuing, delivery with retries,
    and delivery logging.
    """

    def __init__(
        self,
        config: Optional[WebhookConfig] = None,
        store: Optional[WebhookStore] = None,
    ):
        self.config = config or get_webhook_config()
        self.store = store or WebhookStore()

        # Delivery queue
        self._queue: asyncio.Queue[DeliveryTask] = asyncio.Queue(maxsize=self.config.queue_max_size)
        self._retry_queue: asyncio.Queue[DeliveryTask] = asyncio.Queue()

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Background workers
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self, num_workers: int = 3) -> None:
        """Start the webhook delivery workers."""
        if self._running:
            return

        self._running = True

        # Create HTTP client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_seconds,
                read=self.config.delivery_timeout_seconds,
                write=self.config.delivery_timeout_seconds,
                pool=5.0,
            ),
            follow_redirects=False,
        )

        # Start delivery workers
        for i in range(num_workers):
            worker = asyncio.create_task(self._delivery_worker(f"worker-{i}"))
            self._workers.append(worker)

        # Start retry scheduler
        retry_worker = asyncio.create_task(self._retry_scheduler())
        self._workers.append(retry_worker)

        logger.info(f"Started webhook manager with {num_workers} workers")

    async def stop(self) -> None:
        """Stop the webhook delivery workers."""
        self._running = False

        # Cancel workers
        for worker in self._workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        self._workers.clear()

        # Close HTTP client
        if self._client:
            await self._client.aclose()
            self._client = None

        logger.info("Stopped webhook manager")

    # -------------------------------------------------------------------------
    # Webhook Management
    # -------------------------------------------------------------------------

    def register(
        self,
        user_id: str,
        url: str,
        event_types: Union[Set[WebhookEventType], List[str], str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Webhook:
        """
        Register a new webhook endpoint.

        Args:
            user_id: The user ID registering the webhook
            url: The webhook endpoint URL
            event_types: Event types to subscribe to (set, list, or group name)
            name: Optional webhook name
            description: Optional description
            metadata: Optional custom metadata

        Returns:
            The created Webhook

        Raises:
            ValueError: If user has too many webhooks or invalid event types
        """
        # Check user limit
        existing = self.store.get_by_user(user_id)
        if len(existing) >= self.config.max_webhooks_per_user:
            raise ValueError(
                f"Maximum webhooks ({self.config.max_webhooks_per_user}) reached for user"
            )

        # Parse event types
        if isinstance(event_types, str):
            # Check if it's a group name
            if event_types in EVENT_TYPE_GROUPS:
                event_types = EVENT_TYPE_GROUPS[event_types]
            else:
                event_types = {WebhookEventType(event_types)}
        elif isinstance(event_types, list):
            event_types = {WebhookEventType(e) for e in event_types}

        return self.store.create(
            user_id=user_id,
            url=url,
            event_types=event_types,
            name=name,
            description=description,
            metadata=metadata,
        )

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        return self.store.delete(webhook_id)

    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self.store.get(webhook_id)

    def list_webhooks(self, user_id: str) -> List[Webhook]:
        """List all webhooks for a user."""
        return self.store.get_by_user(user_id)

    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        event_types: Optional[Union[Set[WebhookEventType], List[str]]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Webhook]:
        """Update a webhook configuration."""
        # Parse event types if provided
        if event_types is not None:
            if isinstance(event_types, list):
                event_types = {WebhookEventType(e) for e in event_types}

        return self.store.update(
            webhook_id=webhook_id,
            url=url,
            event_types=event_types,
            name=name,
            description=description,
            is_active=is_active,
        )

    def regenerate_secret(self, webhook_id: str) -> Optional[str]:
        """Regenerate the secret for a webhook."""
        return self.store.regenerate_secret(webhook_id)

    # -------------------------------------------------------------------------
    # Event Dispatch
    # -------------------------------------------------------------------------

    async def dispatch(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        source: Optional[str] = None,
    ) -> str:
        """
        Dispatch a webhook event to all subscribers.

        Args:
            event_type: The type of event
            payload: The event payload data
            source: Optional source identifier (e.g., event ID)

        Returns:
            The generated event ID
        """
        event = WebhookEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            payload=payload,
            source=source,
        )

        # Find subscribed webhooks
        webhooks = self.store.get_for_event_type(event_type)

        if not webhooks:
            logger.debug(f"No webhooks subscribed to {event_type.value}")
            return event.event_id

        # Queue delivery tasks
        for webhook in webhooks:
            task = DeliveryTask(
                task_id=str(uuid4()),
                webhook=webhook,
                event=event,
            )
            try:
                self._queue.put_nowait(task)
            except asyncio.QueueFull:
                logger.error(f"Webhook queue full, dropping event {event.event_id}")

        logger.info(
            f"Dispatched event {event_type.value} to {len(webhooks)} webhooks"
        )
        return event.event_id

    async def dispatch_test(self, webhook_id: str) -> Optional[DeliveryAttempt]:
        """
        Send a test event to a specific webhook.

        Args:
            webhook_id: The webhook to test

        Returns:
            The delivery attempt result, or None if webhook not found
        """
        webhook = self.store.get(webhook_id)
        if webhook is None:
            return None

        event = WebhookEvent(
            event_id=str(uuid4()),
            event_type=WebhookEventType.TEST,
            payload={"message": "This is a test webhook delivery"},
        )

        # Deliver synchronously for testing
        return await self._deliver(webhook, event, attempt_number=1)

    # -------------------------------------------------------------------------
    # Delivery Workers
    # -------------------------------------------------------------------------

    async def _delivery_worker(self, name: str) -> None:
        """Background worker that processes the delivery queue."""
        logger.info(f"Webhook delivery worker {name} started")

        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # Deliver
                attempt = await self._deliver(
                    task.webhook,
                    task.event,
                    task.attempt_number + 1,
                )

                # Handle failure - schedule retry
                if not attempt.success and task.attempt_number < self.config.max_retries:
                    # Calculate next retry time
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff_multiplier ** task.attempt_number
                    )
                    delay = min(delay, self.config.max_retry_delay_seconds)

                    task.attempt_number += 1
                    task.scheduled_at = datetime.now(timezone.utc) + timedelta(seconds=delay)

                    await self._retry_queue.put(task)
                    logger.info(
                        f"Scheduled retry {task.attempt_number} for webhook "
                        f"{task.webhook.webhook_id} in {delay}s"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in delivery worker {name}: {e}")

        logger.info(f"Webhook delivery worker {name} stopped")

    async def _retry_scheduler(self) -> None:
        """Background worker that schedules retries."""
        pending_retries: List[DeliveryTask] = []

        while self._running:
            try:
                # Check for new retries
                try:
                    while True:
                        task = self._retry_queue.get_nowait()
                        pending_retries.append(task)
                except asyncio.QueueEmpty:
                    pass

                # Process due retries
                now = datetime.now(timezone.utc)
                still_pending = []

                for task in pending_retries:
                    if task.scheduled_at <= now:
                        try:
                            self._queue.put_nowait(task)
                        except asyncio.QueueFull:
                            still_pending.append(task)
                    else:
                        still_pending.append(task)

                pending_retries = still_pending

                # Sleep a bit
                await asyncio.sleep(1.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry scheduler: {e}")

    async def _deliver(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        attempt_number: int,
    ) -> DeliveryAttempt:
        """
        Deliver an event to a webhook endpoint.

        Args:
            webhook: The webhook to deliver to
            event: The event to deliver
            attempt_number: The attempt number (1-based)

        Returns:
            The delivery attempt result
        """
        start_time = time.time()

        # Prepare payload
        payload_dict = event.to_dict()
        payload_bytes = json.dumps(payload_dict, default=str).encode()

        # Check payload size
        if len(payload_bytes) > self.config.max_payload_size_bytes:
            logger.error(f"Payload too large for webhook {webhook.webhook_id}")
            attempt = DeliveryAttempt(
                attempt_id=str(uuid4()),
                webhook_id=webhook.webhook_id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_message="Payload too large",
            )
            self.store.log_delivery_attempt(attempt)
            return attempt

        # Generate signature
        timestamp = int(time.time())
        signature = generate_signature(payload_bytes, webhook.secret, timestamp)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            self.config.signature_header: signature,
            self.config.timestamp_header: str(timestamp),
            self.config.id_header: event.event_id,
            "User-Agent": "MultiverseDive-Webhook/1.0",
        }

        try:
            response = await self._client.post(
                webhook.url,
                content=payload_bytes,
                headers=headers,
            )

            duration_ms = int((time.time() - start_time) * 1000)

            # Consider 2xx as success
            success = 200 <= response.status_code < 300

            attempt = DeliveryAttempt(
                attempt_id=str(uuid4()),
                webhook_id=webhook.webhook_id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                timestamp=datetime.now(timezone.utc),
                success=success,
                status_code=response.status_code,
                response_body=response.text[:1000] if response.text else None,
                duration_ms=duration_ms,
            )

            if success:
                logger.debug(
                    f"Delivered event {event.event_id} to webhook {webhook.webhook_id}"
                )
            else:
                logger.warning(
                    f"Webhook {webhook.webhook_id} returned {response.status_code}"
                )

        except httpx.TimeoutException as e:
            duration_ms = int((time.time() - start_time) * 1000)
            attempt = DeliveryAttempt(
                attempt_id=str(uuid4()),
                webhook_id=webhook.webhook_id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_message=f"Timeout: {str(e)}",
                duration_ms=duration_ms,
            )
            logger.warning(f"Webhook {webhook.webhook_id} timed out")

        except httpx.RequestError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            attempt = DeliveryAttempt(
                attempt_id=str(uuid4()),
                webhook_id=webhook.webhook_id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_message=f"Request error: {str(e)}",
                duration_ms=duration_ms,
            )
            logger.warning(f"Webhook {webhook.webhook_id} request failed: {e}")

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            attempt = DeliveryAttempt(
                attempt_id=str(uuid4()),
                webhook_id=webhook.webhook_id,
                event_id=event.event_id,
                attempt_number=attempt_number,
                timestamp=datetime.now(timezone.utc),
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                duration_ms=duration_ms,
            )
            logger.error(f"Unexpected error delivering to webhook {webhook.webhook_id}: {e}")

        # Log the attempt
        self.store.log_delivery_attempt(attempt)

        return attempt

    # -------------------------------------------------------------------------
    # Delivery Log
    # -------------------------------------------------------------------------

    def get_delivery_log(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[DeliveryAttempt]:
        """Get delivery log entries."""
        return self.store.get_delivery_log(webhook_id=webhook_id, limit=limit)


# =============================================================================
# Global Instance
# =============================================================================


_webhook_manager: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager instance."""
    global _webhook_manager
    if _webhook_manager is None:
        _webhook_manager = WebhookManager()
    return _webhook_manager


async def initialize_webhooks() -> WebhookManager:
    """Initialize and start the global webhook manager."""
    manager = get_webhook_manager()
    await manager.start()
    return manager


async def shutdown_webhooks() -> None:
    """Shutdown the global webhook manager."""
    global _webhook_manager
    if _webhook_manager is not None:
        await _webhook_manager.stop()
        _webhook_manager = None
