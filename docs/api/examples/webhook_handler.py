#!/usr/bin/env python3
"""
Example: Webhook handler for Multiverse Dive event notifications.

This script demonstrates how to:
1. Set up a FastAPI webhook receiver
2. Verify webhook signatures
3. Process different event types
4. Handle delivery retries

Requirements:
    pip install fastapi uvicorn

Usage:
    # Start the webhook server
    uvicorn webhook_handler:app --host 0.0.0.0 --port 8080

    # Register webhook with Multiverse Dive API
    curl -X POST https://api.multiverse-dive.io/v1/webhooks \
        -H "X-API-Key: your_key" \
        -H "Content-Type: application/json" \
        -d '{"url": "https://your-server.com/webhook", "events": ["*"], "secret": "your_secret"}'
"""

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Header, Request, BackgroundTasks
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "your_webhook_secret_here")

# FastAPI application
app = FastAPI(
    title="Multiverse Dive Webhook Handler",
    description="Example webhook receiver for event notifications",
    version="1.0.0",
)


# ============================================================================
# Models
# ============================================================================

class WebhookPayload(BaseModel):
    """Webhook delivery payload."""
    event_type: str
    timestamp: str
    event_id: Optional[str] = None
    payload: Dict[str, Any] = {}


class WebhookResponse(BaseModel):
    """Webhook response."""
    status: str
    message: str


# ============================================================================
# Signature Verification
# ============================================================================

def verify_signature(
    payload: bytes,
    signature: Optional[str],
    secret: str
) -> bool:
    """
    Verify webhook signature using HMAC-SHA256.

    Args:
        payload: Raw request body
        signature: Signature from X-Webhook-Signature header
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    if not signature:
        logger.warning("No signature provided")
        return False

    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    expected_signature = f"sha256={expected}"

    is_valid = hmac.compare_digest(expected_signature, signature)

    if not is_valid:
        logger.warning(f"Invalid signature: expected {expected_signature}, got {signature}")

    return is_valid


# ============================================================================
# Event Handlers
# ============================================================================

async def handle_event_submitted(event_id: str, payload: Dict[str, Any]):
    """Handle event.submitted notification."""
    logger.info(f"Event submitted: {event_id}")
    logger.info(f"  Priority: {payload.get('priority', 'unknown')}")
    logger.info(f"  Intent: {payload.get('intent', {}).get('class', 'unknown')}")

    # Add your custom logic here
    # Example: Send notification, update database, trigger alerts


async def handle_event_started(event_id: str, payload: Dict[str, Any]):
    """Handle event.started notification."""
    logger.info(f"Processing started: {event_id}")

    # Add your custom logic here


async def handle_event_progress(event_id: str, payload: Dict[str, Any]):
    """Handle event.progress notification."""
    progress = payload.get("progress", 0)
    phase = payload.get("current_phase", "unknown")

    logger.info(f"Progress update: {event_id} - {progress:.0%} ({phase})")

    # Add your custom logic here
    # Example: Update progress bar, send status to frontend


async def handle_event_completed(event_id: str, payload: Dict[str, Any]):
    """Handle event.completed notification."""
    logger.info(f"Event completed: {event_id}")
    logger.info(f"  Products: {payload.get('products', [])}")
    logger.info(f"  Quality score: {payload.get('quality_score', 'N/A')}")

    # Add your custom logic here
    # Example: Download products, send completion email, update dashboard


async def handle_event_failed(event_id: str, payload: Dict[str, Any]):
    """Handle event.failed notification."""
    error = payload.get("error", "Unknown error")
    logger.error(f"Event failed: {event_id}")
    logger.error(f"  Error: {error}")

    # Add your custom logic here
    # Example: Send alert, log error, trigger retry


# Event handler mapping
EVENT_HANDLERS = {
    "event.submitted": handle_event_submitted,
    "event.started": handle_event_started,
    "event.progress": handle_event_progress,
    "event.completed": handle_event_completed,
    "event.failed": handle_event_failed,
}


# ============================================================================
# Webhook Endpoint
# ============================================================================

@app.post("/webhook", response_model=WebhookResponse)
async def receive_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_webhook_signature: Optional[str] = Header(None),
    x_webhook_delivery_id: Optional[str] = Header(None),
):
    """
    Receive and process webhook deliveries from Multiverse Dive.

    The endpoint verifies the signature (if configured), parses the payload,
    and dispatches to the appropriate event handler.
    """
    # Get raw body for signature verification
    body = await request.body()

    # Verify signature if secret is configured
    if WEBHOOK_SECRET and WEBHOOK_SECRET != "your_webhook_secret_here":
        if not verify_signature(body, x_webhook_signature, WEBHOOK_SECRET):
            logger.warning(f"Signature verification failed for delivery {x_webhook_delivery_id}")
            raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    try:
        data = json.loads(body)
        webhook = WebhookPayload(**data)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {e}")

    logger.info(f"Received webhook: {webhook.event_type} (delivery: {x_webhook_delivery_id})")

    # Get handler for event type
    handler = EVENT_HANDLERS.get(webhook.event_type)

    if handler:
        # Run handler in background to respond quickly
        background_tasks.add_task(
            handler,
            webhook.event_id or "unknown",
            webhook.payload
        )
    else:
        logger.warning(f"No handler for event type: {webhook.event_type}")

    return WebhookResponse(
        status="received",
        message=f"Webhook {webhook.event_type} received successfully"
    )


@app.get("/webhook/health")
async def webhook_health():
    """Health check endpoint for the webhook handler."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ============================================================================
# Application Startup
# ============================================================================

@app.on_event("startup")
async def startup():
    """Log configuration on startup."""
    logger.info("Webhook handler starting...")
    if WEBHOOK_SECRET == "your_webhook_secret_here":
        logger.warning("WEBHOOK_SECRET not configured - signature verification disabled!")
    else:
        logger.info("Signature verification enabled")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.environ.get("WEBHOOK_HOST", "0.0.0.0")
    port = int(os.environ.get("WEBHOOK_PORT", "8080"))

    print(f"Starting webhook handler on {host}:{port}")
    print(f"Webhook endpoint: http://{host}:{port}/webhook")

    uvicorn.run(
        "webhook_handler:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
