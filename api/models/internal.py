"""
Pydantic models for the internal/partner API endpoints.

Request and response models for webhook subscription management,
metrics responses, and queue summaries.
"""

import ipaddress
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Webhook Subscription Models (Task 3.5)
# =============================================================================

# Private/loopback IP ranges that must be rejected for SSRF protection
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
]


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address is in a private/loopback range."""
    try:
        addr = ipaddress.ip_address(ip_str)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        return False


def validate_webhook_url(url: str) -> str:
    """
    Validate a webhook URL for SSRF protection.

    Requirements:
    - Must be HTTPS (reject http://)
    - Must not resolve to private/loopback IPs
    - Must have a valid hostname
    """
    parsed = urlparse(url)

    # Require HTTPS
    if parsed.scheme != "https":
        raise ValueError("Webhook URL must use HTTPS")

    # Must have a hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Webhook URL must have a valid hostname")

    # Resolve hostname and check for private IPs
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
        for addr_info in addr_infos:
            ip = addr_info[4][0]
            if _is_private_ip(ip):
                raise ValueError(
                    f"Webhook URL resolves to private/loopback address: {ip}"
                )
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    return url


class CreateWebhookRequest(BaseModel):
    """Request body for registering a webhook subscription."""

    target_url: str = Field(
        ...,
        description="HTTPS URL to deliver webhook events to",
        max_length=2048,
    )
    event_filter: List[str] = Field(
        default_factory=list,
        description="Event types to subscribe to (empty = all events)",
    )

    @field_validator("target_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate HTTPS URL and reject private/loopback IPs."""
        return validate_webhook_url(v)


class WebhookResponse(BaseModel):
    """Response for a webhook subscription."""

    subscription_id: str = Field(..., description="Subscription UUID")
    customer_id: str = Field(..., description="Owner customer ID")
    target_url: str = Field(..., description="Delivery target URL")
    event_filter: List[str] = Field(
        default_factory=list, description="Subscribed event types"
    )
    active: bool = Field(default=True, description="Whether the subscription is active")
    created_at: datetime = Field(..., description="Creation timestamp")


class CursorPaginatedWebhooksResponse(BaseModel):
    """Cursor-based paginated response for webhook listing."""

    items: List[WebhookResponse] = Field(..., description="Webhook subscriptions")
    next_cursor: Optional[str] = Field(
        default=None, description="Cursor for next page (null if no more items)"
    )
    has_more: bool = Field(
        default=False, description="Whether more items are available"
    )


# =============================================================================
# DLQ Models
# =============================================================================


class DLQEntry(BaseModel):
    """Dead letter queue entry."""

    dlq_id: str = Field(..., description="DLQ entry UUID")
    subscription_id: str = Field(..., description="Associated subscription UUID")
    event_seq: int = Field(..., description="Original event sequence number")
    payload: Dict[str, Any] = Field(..., description="Event payload that failed")
    last_error: Optional[str] = Field(
        default=None, description="Last error message"
    )
    attempt_count: int = Field(..., description="Number of delivery attempts")
    failed_at: datetime = Field(..., description="When the final failure occurred")
