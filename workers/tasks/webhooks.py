"""
Webhook delivery Taskiq task.

Delivers CloudEvents payloads to registered webhook endpoints with:
- HMAC-SHA256 signing via X-FirstLight-Signature-256 header
- Exponential backoff retry: 5 attempts, ~5s/~10s/~20s/~40s/~80s + jitter
- DLQ insertion after all retries exhausted
- Redis idempotency keys to prevent duplicate delivery (Task 3.7)
- No HTTP redirect following (redirects treated as failures)

Task 3.6 + 3.7
"""

import hashlib
import hmac
import json
import logging
import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# Retry configuration per spec
MAX_ATTEMPTS = 5
BASE_DELAY_SECONDS = 5.0
BACKOFF_MULTIPLIER = 2.0
MAX_DELAY_SECONDS = 300.0  # 5 minutes cap
DELIVERY_TIMEOUT_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 10

# Idempotency key TTL (24 hours)
IDEMPOTENCY_TTL_SECONDS = 86400


def _compute_hmac_signature(body: bytes, secret_key: str) -> str:
    """
    Compute HMAC-SHA256 signature for webhook payload.

    Returns the signature in the format: sha256=<hex-digest>
    """
    signature = hmac.new(
        secret_key.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={signature}"


def _calculate_delay(attempt: int) -> float:
    """
    Calculate the delay for a retry attempt with exponential backoff + jitter.

    Schedule: ~5s, ~10s, ~20s, ~40s, ~80s
    """
    delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** attempt)
    delay = min(delay, MAX_DELAY_SECONDS)
    # Add jitter: random value in [0, 1)
    delay += random.uniform(0, 1)
    return delay


async def _check_idempotency_key(
    event_seq: int,
    subscription_id: str,
) -> bool:
    """
    Check if this delivery has already been completed (idempotency check).

    Uses Redis key: webhook:delivered:{event_seq}:{subscription_id}

    Returns True if already delivered (should skip), False if not yet delivered.
    """
    try:
        import redis.asyncio as aioredis
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url)
        try:
            key = f"webhook:delivered:{event_seq}:{subscription_id}"
            exists = await r.exists(key)
            return bool(exists)
        finally:
            await r.close()
    except ImportError:
        logger.warning("redis not available for idempotency check, proceeding")
        return False
    except Exception as e:
        logger.warning("Idempotency check failed: %s, proceeding with delivery", e)
        return False


async def _set_idempotency_key(
    event_seq: int,
    subscription_id: str,
) -> None:
    """
    Set the idempotency key after successful delivery.

    Key: webhook:delivered:{event_seq}:{subscription_id}
    TTL: 24 hours
    """
    try:
        import redis.asyncio as aioredis
        import os

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = aioredis.from_url(redis_url)
        try:
            key = f"webhook:delivered:{event_seq}:{subscription_id}"
            await r.setex(key, IDEMPOTENCY_TTL_SECONDS, "1")
        finally:
            await r.close()
    except ImportError:
        logger.warning("redis not available for idempotency key, skipping")
    except Exception as e:
        logger.warning("Failed to set idempotency key: %s", e)


async def _insert_to_dlq(
    subscription_id: str,
    event_seq: int,
    payload: Dict[str, Any],
    last_error: str,
    attempt_count: int,
) -> None:
    """Insert a failed delivery into the dead letter queue."""
    try:
        import asyncpg
        from api.config import get_settings

        settings = get_settings()
        db = settings.database
        dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"

        conn = await asyncpg.connect(dsn)
        try:
            await conn.execute(
                """
                INSERT INTO webhook_dlq
                    (subscription_id, event_seq, payload, last_error, attempt_count)
                VALUES ($1, $2, $3::jsonb, $4, $5)
                """,
                uuid.UUID(subscription_id),
                event_seq,
                json.dumps(payload, default=str),
                last_error,
                attempt_count,
            )
            logger.info(
                "Inserted failed delivery to DLQ: subscription=%s event_seq=%d",
                subscription_id,
                event_seq,
            )
        finally:
            await conn.close()
    except Exception as e:
        logger.error("Failed to insert to DLQ: %s", e)


async def deliver_webhook(
    subscription_id: str,
    target_url: str,
    secret_key: str,
    event_seq: int,
    payload: Dict[str, Any],
    attempt: int = 0,
) -> bool:
    """
    Deliver a webhook payload to the registered endpoint.

    This is the core delivery function called by the Taskiq task.

    Args:
        subscription_id: The webhook subscription UUID.
        target_url: The HTTPS delivery URL.
        secret_key: The HMAC secret for signing.
        event_seq: The event sequence number.
        payload: The CloudEvents envelope payload.
        attempt: Current attempt number (0-based).

    Returns:
        True if delivery succeeded, False otherwise.
    """
    # Check idempotency key (Task 3.7)
    if await _check_idempotency_key(event_seq, subscription_id):
        logger.info(
            "Skipping duplicate delivery: subscription=%s event_seq=%d",
            subscription_id,
            event_seq,
        )
        return True

    body = json.dumps(payload, default=str).encode("utf-8")
    signature = _compute_hmac_signature(body, secret_key)

    headers = {
        "Content-Type": "application/json",
        "X-FirstLight-Signature-256": signature,
        "User-Agent": "FirstLight-Webhook/1.0",
    }

    last_error = ""

    for current_attempt in range(attempt, MAX_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=CONNECT_TIMEOUT_SECONDS,
                    read=DELIVERY_TIMEOUT_SECONDS,
                    write=DELIVERY_TIMEOUT_SECONDS,
                    pool=5.0,
                ),
                follow_redirects=False,  # Do NOT follow redirects (treat as failure)
            ) as client:
                response = await client.post(
                    target_url,
                    content=body,
                    headers=headers,
                )

            # Check for redirect (treat as failure per spec)
            if 300 <= response.status_code < 400:
                last_error = f"Redirect response ({response.status_code}) treated as failure"
                logger.warning(
                    "Webhook delivery redirect: subscription=%s attempt=%d status=%d",
                    subscription_id,
                    current_attempt + 1,
                    response.status_code,
                )
            elif 200 <= response.status_code < 300:
                # Success
                logger.info(
                    "Webhook delivered: subscription=%s event_seq=%d attempt=%d",
                    subscription_id,
                    event_seq,
                    current_attempt + 1,
                )
                # Set idempotency key (Task 3.7)
                await _set_idempotency_key(event_seq, subscription_id)
                return True
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(
                    "Webhook delivery failed: subscription=%s attempt=%d status=%d",
                    subscription_id,
                    current_attempt + 1,
                    response.status_code,
                )

        except httpx.TimeoutException as e:
            last_error = f"Timeout: {str(e)}"
            logger.warning(
                "Webhook timeout: subscription=%s attempt=%d error=%s",
                subscription_id,
                current_attempt + 1,
                e,
            )
        except httpx.RequestError as e:
            last_error = f"Request error: {str(e)}"
            logger.warning(
                "Webhook request error: subscription=%s attempt=%d error=%s",
                subscription_id,
                current_attempt + 1,
                e,
            )
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            logger.error(
                "Unexpected webhook error: subscription=%s attempt=%d error=%s",
                subscription_id,
                current_attempt + 1,
                e,
            )

        # Wait before retrying (if not the last attempt)
        if current_attempt < MAX_ATTEMPTS - 1:
            import asyncio

            delay = _calculate_delay(current_attempt)
            logger.info(
                "Webhook retry scheduled: subscription=%s attempt=%d delay=%.1fs",
                subscription_id,
                current_attempt + 2,
                delay,
            )
            await asyncio.sleep(delay)

    # All attempts exhausted — insert to DLQ
    await _insert_to_dlq(
        subscription_id=subscription_id,
        event_seq=event_seq,
        payload=payload,
        last_error=last_error,
        attempt_count=MAX_ATTEMPTS,
    )

    return False


# Register as Taskiq task
try:
    from workers.taskiq_app import broker

    @broker.task(task_name="deliver_webhook")
    async def deliver_webhook_task(
        subscription_id: str,
        target_url: str,
        secret_key: str,
        event_seq: int,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Taskiq task wrapper for webhook delivery.

        This is the entry point called by the Taskiq worker.
        The actual delivery logic is in deliver_webhook().
        """
        return await deliver_webhook(
            subscription_id=subscription_id,
            target_url=target_url,
            secret_key=secret_key,
            event_seq=event_seq,
            payload=payload,
            attempt=0,
        )

except ImportError:
    logger.warning("Taskiq not available — webhook task not registered")
