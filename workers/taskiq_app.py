"""
Taskiq broker configuration.

Defines the central TaskiqBroker backed by Redis Streams. All background
tasks (webhook delivery, maintenance, OGC execution) are registered
against this broker instance.

The broker URL is configured via the REDIS_URL environment variable.
Redis authentication uses the REDIS_PASSWORD injected during Phase 0
credential hardening.

Usage:
    # Import the broker in task modules:
    from workers.taskiq_app import broker

    # Register a task:
    @broker.task
    async def my_task(arg: str) -> None:
        ...

    # Run the worker (CLI):
    taskiq worker workers.taskiq_app:broker
"""

import logging
import os

logger = logging.getLogger(__name__)


def _build_redis_url() -> str:
    """
    Build the Redis URL from environment variables.

    Supports REDIS_URL directly, or constructs from REDIS_HOST, REDIS_PORT,
    REDIS_PASSWORD, and REDIS_DB components.
    """
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        return redis_url

    host = os.getenv("REDIS_HOST", "localhost")
    port = os.getenv("REDIS_PORT", "6379")
    password = os.getenv("REDIS_PASSWORD", "")
    db = os.getenv("REDIS_DB", "0")

    if password:
        return f"redis://:{password}@{host}:{port}/{db}"
    return f"redis://{host}:{port}/{db}"


def _create_broker():
    """
    Create the Taskiq broker instance.

    Uses taskiq-redis RedisAsyncResultBackend for result storage and
    ListQueueBroker for task distribution via Redis lists.

    Falls back to InMemoryBroker if taskiq-redis is not installed
    (e.g., in test environments without the control-plane extra).
    """
    redis_url = _build_redis_url()

    try:
        from taskiq_redis import ListQueueBroker, RedisAsyncResultBackend

        result_backend = RedisAsyncResultBackend(
            redis_url=redis_url,
        )

        _broker = ListQueueBroker(
            url=redis_url,
        ).with_result_backend(result_backend)

        logger.info("Taskiq broker configured with Redis at %s", redis_url.split("@")[-1])
        return _broker

    except ImportError:
        logger.warning(
            "taskiq-redis not installed; falling back to InMemoryBroker. "
            "Install the control-plane extra: pip install firstlight[control-plane]"
        )
        from taskiq import InMemoryBroker

        return InMemoryBroker()


# The global broker instance used by all task modules.
broker = _create_broker()
