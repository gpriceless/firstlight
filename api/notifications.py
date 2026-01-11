"""
Notification Dispatcher for Multi-Channel Notifications.

Provides unified notification dispatch across webhooks, WebSockets,
with queuing and batch support.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from api.webhooks import (
    WebhookEventType,
    WebhookManager,
    get_webhook_manager,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class NotificationConfig(BaseModel):
    """Notification system configuration."""

    # Queue settings
    queue_max_size: int = Field(default=10000, description="Max items in notification queue")
    batch_size: int = Field(default=50, description="Batch size for processing")
    batch_delay_ms: int = Field(default=100, description="Delay between batches")

    # Channel settings
    enable_webhooks: bool = Field(default=True, description="Enable webhook notifications")
    enable_websockets: bool = Field(default=True, description="Enable WebSocket notifications")

    # Filtering
    dedupe_window_seconds: int = Field(default=5, description="Window for deduplicating notifications")

    # Priority settings
    high_priority_immediate: bool = Field(default=True, description="Send high priority immediately")

    class Config:
        env_prefix = "NOTIFICATION_"


def get_notification_config() -> NotificationConfig:
    """Get notification configuration from environment."""
    return NotificationConfig(
        queue_max_size=int(os.getenv("NOTIFICATION_QUEUE_SIZE", "10000")),
        batch_size=int(os.getenv("NOTIFICATION_BATCH_SIZE", "50")),
        batch_delay_ms=int(os.getenv("NOTIFICATION_BATCH_DELAY_MS", "100")),
        enable_webhooks=os.getenv("NOTIFICATION_WEBHOOKS", "true").lower() == "true",
        enable_websockets=os.getenv("NOTIFICATION_WEBSOCKETS", "true").lower() == "true",
        dedupe_window_seconds=int(os.getenv("NOTIFICATION_DEDUPE_WINDOW", "5")),
        high_priority_immediate=os.getenv("NOTIFICATION_HIGH_PRIORITY_IMMEDIATE", "true").lower() == "true",
    )


# =============================================================================
# Notification Types
# =============================================================================


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""

    WEBHOOK = "webhook"
    WEBSOCKET = "websocket"
    ALL = "all"


@dataclass
class Notification:
    """Represents a notification to be dispatched."""

    notification_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: Set[NotificationChannel] = field(default_factory=lambda: {NotificationChannel.ALL})
    target_users: Optional[Set[str]] = None  # None = broadcast to all
    source: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dedupe_key: Optional[str] = None  # For deduplication
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "notification_id": self.notification_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "priority": self.priority.value,
            "channels": [c.value for c in self.channels],
            "target_users": list(self.target_users) if self.target_users else None,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class NotificationResult:
    """Result of a notification dispatch."""

    notification_id: str
    success: bool
    channels_sent: Dict[str, bool]  # channel -> success
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# WebSocket Connection Manager
# =============================================================================


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time notifications.

    This is a simple in-memory implementation. For production scale,
    consider using Redis pub/sub or a dedicated WebSocket service.
    """

    def __init__(self):
        # user_id -> list of websocket connections
        self._connections: Dict[str, List[Any]] = {}
        # subscription_id -> (user_id, event_types)
        self._subscriptions: Dict[str, tuple] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self,
        websocket: Any,
        user_id: str,
        event_types: Optional[Set[WebhookEventType]] = None,
    ) -> str:
        """
        Register a WebSocket connection.

        Args:
            websocket: The WebSocket connection object
            user_id: The user ID for this connection
            event_types: Optional set of event types to subscribe to

        Returns:
            Subscription ID for managing this connection
        """
        async with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = []
            self._connections[user_id].append(websocket)

            subscription_id = str(uuid4())
            self._subscriptions[subscription_id] = (user_id, event_types or set(WebhookEventType))

            logger.info(f"WebSocket connected: user={user_id}, subscription={subscription_id}")
            return subscription_id

    async def disconnect(self, subscription_id: str) -> None:
        """Disconnect a WebSocket by subscription ID."""
        async with self._lock:
            if subscription_id not in self._subscriptions:
                return

            user_id, _ = self._subscriptions[subscription_id]
            del self._subscriptions[subscription_id]

            # Note: We don't remove from _connections here as we don't
            # have the websocket reference. The send_to_user method
            # will handle dead connections.

            logger.info(f"WebSocket disconnected: subscription={subscription_id}")

    async def disconnect_user(self, user_id: str) -> None:
        """Disconnect all WebSockets for a user."""
        async with self._lock:
            if user_id in self._connections:
                self._connections[user_id].clear()

            # Remove subscriptions for this user
            to_remove = [
                sid for sid, (uid, _) in self._subscriptions.items()
                if uid == user_id
            ]
            for sid in to_remove:
                del self._subscriptions[sid]

    async def send_to_user(
        self,
        user_id: str,
        message: Dict[str, Any],
        event_type: Optional[WebhookEventType] = None,
    ) -> int:
        """
        Send a message to all WebSocket connections for a user.

        Args:
            user_id: The target user ID
            message: The message to send (will be JSON encoded)
            event_type: Optional event type for filtering subscriptions

        Returns:
            Number of messages sent
        """
        import json

        async with self._lock:
            connections = self._connections.get(user_id, [])
            if not connections:
                return 0

            # Filter by event type subscriptions
            if event_type:
                valid_connections = []
                for ws in connections:
                    # Find subscription for this connection
                    for sid, (uid, types) in self._subscriptions.items():
                        if uid == user_id and (not types or event_type in types):
                            valid_connections.append(ws)
                            break
                connections = valid_connections

            message_json = json.dumps(message, default=str)
            sent_count = 0
            dead_connections = []

            for ws in connections:
                try:
                    await ws.send_text(message_json)
                    sent_count += 1
                except Exception as e:
                    logger.warning(f"Failed to send WebSocket message: {e}")
                    dead_connections.append(ws)

            # Remove dead connections
            for dead_ws in dead_connections:
                if dead_ws in self._connections.get(user_id, []):
                    self._connections[user_id].remove(dead_ws)

            return sent_count

    async def broadcast(
        self,
        message: Dict[str, Any],
        event_type: Optional[WebhookEventType] = None,
    ) -> int:
        """
        Broadcast a message to all connected WebSockets.

        Args:
            message: The message to send
            event_type: Optional event type for filtering

        Returns:
            Total number of messages sent
        """
        total_sent = 0
        user_ids = list(self._connections.keys())

        for user_id in user_ids:
            sent = await self.send_to_user(user_id, message, event_type)
            total_sent += sent

        return total_sent

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return sum(len(conns) for conns in self._connections.values())

    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user."""
        return len(self._connections.get(user_id, []))


# =============================================================================
# Notification Dispatcher
# =============================================================================


class NotificationDispatcher:
    """
    Unified notification dispatcher that routes notifications
    to appropriate channels (webhooks, WebSockets, etc.).

    Supports queuing, batching, deduplication, and priority handling.
    """

    def __init__(
        self,
        config: Optional[NotificationConfig] = None,
        webhook_manager: Optional[WebhookManager] = None,
        websocket_manager: Optional[WebSocketConnectionManager] = None,
    ):
        self.config = config or get_notification_config()
        self._webhook_manager = webhook_manager
        self._websocket_manager = websocket_manager or WebSocketConnectionManager()

        # Notification queue
        self._queue: asyncio.Queue[Notification] = asyncio.Queue(
            maxsize=self.config.queue_max_size
        )

        # Deduplication cache
        self._recent_dedupe_keys: Dict[str, datetime] = {}
        self._dedupe_lock = asyncio.Lock()

        # Background workers
        self._worker_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "notifications_sent": 0,
            "notifications_dropped": 0,
            "webhooks_sent": 0,
            "websockets_sent": 0,
            "deduplicated": 0,
        }

    @property
    def webhook_manager(self) -> WebhookManager:
        """Get the webhook manager, creating if needed."""
        if self._webhook_manager is None:
            self._webhook_manager = get_webhook_manager()
        return self._webhook_manager

    @property
    def websocket_manager(self) -> WebSocketConnectionManager:
        """Get the WebSocket connection manager."""
        return self._websocket_manager

    async def start(self) -> None:
        """Start the notification dispatcher."""
        if self._running:
            return

        self._running = True

        # Start webhook manager if needed
        if self.config.enable_webhooks:
            await self.webhook_manager.start()

        # Start processing worker
        self._worker_task = asyncio.create_task(self._process_queue())

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_dedupe_cache())

        logger.info("Notification dispatcher started")

    async def stop(self) -> None:
        """Stop the notification dispatcher."""
        self._running = False

        # Cancel workers
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Stop webhook manager
        if self.config.enable_webhooks and self._webhook_manager:
            await self._webhook_manager.stop()

        logger.info("Notification dispatcher stopped")

    # -------------------------------------------------------------------------
    # Notification Dispatch
    # -------------------------------------------------------------------------

    async def notify(
        self,
        event_type: WebhookEventType,
        payload: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channels: Optional[Set[NotificationChannel]] = None,
        target_users: Optional[Set[str]] = None,
        source: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a notification through configured channels.

        Args:
            event_type: Type of event being notified
            payload: Notification payload data
            priority: Notification priority level
            channels: Specific channels to use (default: all)
            target_users: Specific users to notify (default: broadcast)
            source: Source identifier (e.g., event ID)
            dedupe_key: Key for deduplication
            metadata: Additional metadata

        Returns:
            Notification ID
        """
        notification = Notification(
            notification_id=str(uuid4()),
            event_type=event_type,
            payload=payload,
            priority=priority,
            channels=channels or {NotificationChannel.ALL},
            target_users=target_users,
            source=source,
            dedupe_key=dedupe_key,
            metadata=metadata or {},
        )

        # Check deduplication
        if dedupe_key:
            is_duplicate = await self._check_dedupe(dedupe_key)
            if is_duplicate:
                self._stats["deduplicated"] += 1
                logger.debug(f"Notification deduplicated: {dedupe_key}")
                return notification.notification_id

        # High priority and critical notifications bypass queue
        if (
            self.config.high_priority_immediate
            and priority in (NotificationPriority.HIGH, NotificationPriority.CRITICAL)
        ):
            await self._dispatch_notification(notification)
        else:
            # Queue for batch processing
            try:
                self._queue.put_nowait(notification)
            except asyncio.QueueFull:
                self._stats["notifications_dropped"] += 1
                logger.warning(f"Notification queue full, dropping: {notification.notification_id}")

        return notification.notification_id

    async def notify_event_started(
        self,
        event_id: str,
        event_type: str,
        area: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method for event started notifications."""
        return await self.notify(
            event_type=WebhookEventType.EVENT_STARTED,
            payload={
                "event_id": event_id,
                "event_type": event_type,
                "area": area,
                "status": "started",
            },
            source=event_id,
            target_users={user_id} if user_id else None,
            dedupe_key=f"event_started:{event_id}",
        )

    async def notify_event_progress(
        self,
        event_id: str,
        progress: float,
        stage: str,
        message: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method for event progress notifications."""
        return await self.notify(
            event_type=WebhookEventType.EVENT_PROGRESS,
            payload={
                "event_id": event_id,
                "progress": progress,
                "stage": stage,
                "message": message,
            },
            priority=NotificationPriority.LOW,
            source=event_id,
            target_users={user_id} if user_id else None,
            # Dedupe progress updates for same stage
            dedupe_key=f"event_progress:{event_id}:{stage}:{int(progress * 10)}",
        )

    async def notify_event_completed(
        self,
        event_id: str,
        products: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method for event completed notifications."""
        return await self.notify(
            event_type=WebhookEventType.EVENT_COMPLETED,
            payload={
                "event_id": event_id,
                "status": "completed",
                "products": products,
            },
            priority=NotificationPriority.HIGH,
            source=event_id,
            target_users={user_id} if user_id else None,
            dedupe_key=f"event_completed:{event_id}",
        )

    async def notify_event_failed(
        self,
        event_id: str,
        error: str,
        details: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method for event failed notifications."""
        return await self.notify(
            event_type=WebhookEventType.EVENT_FAILED,
            payload={
                "event_id": event_id,
                "status": "failed",
                "error": error,
                "details": details,
            },
            priority=NotificationPriority.HIGH,
            source=event_id,
            target_users={user_id} if user_id else None,
            dedupe_key=f"event_failed:{event_id}",
        )

    async def notify_product_ready(
        self,
        event_id: str,
        product_id: str,
        product_type: str,
        download_url: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> str:
        """Convenience method for product ready notifications."""
        return await self.notify(
            event_type=WebhookEventType.PRODUCT_READY,
            payload={
                "event_id": event_id,
                "product_id": product_id,
                "product_type": product_type,
                "download_url": download_url,
            },
            source=event_id,
            target_users={user_id} if user_id else None,
            dedupe_key=f"product_ready:{product_id}",
        )

    # -------------------------------------------------------------------------
    # Batch Notifications
    # -------------------------------------------------------------------------

    async def notify_batch(
        self,
        notifications: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Send multiple notifications as a batch.

        Each notification dict should contain:
        - event_type: str
        - payload: dict
        - priority: str (optional)
        - channels: list (optional)
        - target_users: list (optional)

        Returns:
            List of notification IDs
        """
        notification_ids = []

        for notif_dict in notifications:
            event_type = WebhookEventType(notif_dict["event_type"])
            priority = NotificationPriority(notif_dict.get("priority", "normal"))

            channels = None
            if "channels" in notif_dict:
                channels = {NotificationChannel(c) for c in notif_dict["channels"]}

            target_users = None
            if "target_users" in notif_dict:
                target_users = set(notif_dict["target_users"])

            notif_id = await self.notify(
                event_type=event_type,
                payload=notif_dict["payload"],
                priority=priority,
                channels=channels,
                target_users=target_users,
                source=notif_dict.get("source"),
                dedupe_key=notif_dict.get("dedupe_key"),
            )
            notification_ids.append(notif_id)

        return notification_ids

    # -------------------------------------------------------------------------
    # Internal Processing
    # -------------------------------------------------------------------------

    async def _dispatch_notification(self, notification: Notification) -> NotificationResult:
        """Dispatch a notification to all applicable channels."""
        channels_sent = {}

        # Determine which channels to use
        use_webhooks = (
            self.config.enable_webhooks
            and (
                NotificationChannel.ALL in notification.channels
                or NotificationChannel.WEBHOOK in notification.channels
            )
        )
        use_websockets = (
            self.config.enable_websockets
            and (
                NotificationChannel.ALL in notification.channels
                or NotificationChannel.WEBSOCKET in notification.channels
            )
        )

        # Send via webhooks
        if use_webhooks:
            try:
                await self.webhook_manager.dispatch(
                    event_type=notification.event_type,
                    payload=notification.payload,
                    source=notification.source,
                )
                channels_sent["webhook"] = True
                self._stats["webhooks_sent"] += 1
            except Exception as e:
                logger.error(f"Error dispatching webhook: {e}")
                channels_sent["webhook"] = False

        # Send via WebSockets
        if use_websockets:
            try:
                message = notification.to_dict()

                if notification.target_users:
                    # Send to specific users
                    for user_id in notification.target_users:
                        await self._websocket_manager.send_to_user(
                            user_id,
                            message,
                            notification.event_type,
                        )
                else:
                    # Broadcast to all
                    await self._websocket_manager.broadcast(
                        message,
                        notification.event_type,
                    )

                channels_sent["websocket"] = True
                self._stats["websockets_sent"] += 1
            except Exception as e:
                logger.error(f"Error dispatching WebSocket: {e}")
                channels_sent["websocket"] = False

        self._stats["notifications_sent"] += 1

        return NotificationResult(
            notification_id=notification.notification_id,
            success=any(channels_sent.values()),
            channels_sent=channels_sent,
        )

    async def _process_queue(self) -> None:
        """Background worker that processes the notification queue."""
        logger.info("Notification queue processor started")

        while self._running:
            try:
                # Collect batch of notifications
                batch: List[Notification] = []

                for _ in range(self.config.batch_size):
                    try:
                        notification = await asyncio.wait_for(
                            self._queue.get(),
                            timeout=self.config.batch_delay_ms / 1000,
                        )
                        batch.append(notification)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                if batch:
                    for notification in batch:
                        try:
                            await self._dispatch_notification(notification)
                        except Exception as e:
                            logger.error(
                                f"Error processing notification {notification.notification_id}: {e}"
                            )

                # Small delay between batches
                if len(batch) < self.config.batch_size:
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in notification processor: {e}")
                await asyncio.sleep(1)

        logger.info("Notification queue processor stopped")

    async def _check_dedupe(self, dedupe_key: str) -> bool:
        """Check if a notification should be deduplicated."""
        async with self._dedupe_lock:
            now = datetime.now(timezone.utc)

            if dedupe_key in self._recent_dedupe_keys:
                last_sent = self._recent_dedupe_keys[dedupe_key]
                if (now - last_sent).total_seconds() < self.config.dedupe_window_seconds:
                    return True  # Duplicate

            self._recent_dedupe_keys[dedupe_key] = now
            return False

    async def _cleanup_dedupe_cache(self) -> None:
        """Periodically clean up the deduplication cache."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                async with self._dedupe_lock:
                    now = datetime.now(timezone.utc)
                    cutoff = self.config.dedupe_window_seconds * 2

                    keys_to_remove = [
                        key
                        for key, timestamp in self._recent_dedupe_keys.items()
                        if (now - timestamp).total_seconds() > cutoff
                    ]

                    for key in keys_to_remove:
                        del self._recent_dedupe_keys[key]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dedupe cleanup: {e}")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "websocket_connections": self._websocket_manager.get_connection_count(),
            "dedupe_cache_size": len(self._recent_dedupe_keys),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "notifications_sent": 0,
            "notifications_dropped": 0,
            "webhooks_sent": 0,
            "websockets_sent": 0,
            "deduplicated": 0,
        }


# =============================================================================
# Global Instance
# =============================================================================


_notification_dispatcher: Optional[NotificationDispatcher] = None


def get_notification_dispatcher() -> NotificationDispatcher:
    """Get the global notification dispatcher instance."""
    global _notification_dispatcher
    if _notification_dispatcher is None:
        _notification_dispatcher = NotificationDispatcher()
    return _notification_dispatcher


async def initialize_notifications() -> NotificationDispatcher:
    """Initialize and start the global notification dispatcher."""
    dispatcher = get_notification_dispatcher()
    await dispatcher.start()
    return dispatcher


async def shutdown_notifications() -> None:
    """Shutdown the global notification dispatcher."""
    global _notification_dispatcher
    if _notification_dispatcher is not None:
        await _notification_dispatcher.stop()
        _notification_dispatcher = None
