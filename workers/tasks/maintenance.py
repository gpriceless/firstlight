"""
Database maintenance tasks for job_events partitioning.

Provides a daily maintenance task that:
1. Creates next month's partition if it doesn't exist
2. Detaches partitions older than FIRSTLIGHT_EVENT_RETENTION_DAYS (default 90)
3. Drops archived partitions after verification

This task runs as a scheduled asyncio task in the application lifespan,
not as a Taskiq task (because it needs direct database DDL access).

Task 3.10
"""

import logging
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# Default retention period (days)
DEFAULT_EVENT_RETENTION_DAYS = 90


def get_retention_days() -> int:
    """Get the event retention period from environment."""
    return int(os.getenv("FIRSTLIGHT_EVENT_RETENTION_DAYS", str(DEFAULT_EVENT_RETENTION_DAYS)))


async def run_partition_maintenance() -> dict:
    """
    Run partition maintenance for the job_events table.

    This function:
    1. Creates next month's partition if it doesn't exist
    2. Detaches partitions older than the retention period
    3. Drops detached partitions

    Returns a summary dict of actions taken.
    """
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg not available, skipping partition maintenance")
        return {"error": "asyncpg not available"}

    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    dsn = f"postgresql://{db.user}:{db.password}@{db.host}:{db.port}/{db.name}"

    conn = await asyncpg.connect(dsn)
    summary = {
        "partitions_created": [],
        "partitions_detached": [],
        "partitions_dropped": [],
        "errors": [],
    }

    try:
        # 1. Create next month's partition (and month after for safety)
        today = date.today()
        for month_offset in range(1, 4):
            target_month = today.replace(day=1) + timedelta(days=32 * month_offset)
            target_month = target_month.replace(day=1)
            next_month = (target_month + timedelta(days=32)).replace(day=1)

            partition_name = f"job_events_{target_month.strftime('%Y_%m')}"

            # Check if partition already exists
            exists = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM pg_class
                    WHERE relname = $1 AND relkind = 'r'
                )
                """,
                partition_name,
            )

            if not exists:
                try:
                    await conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {partition_name}
                        PARTITION OF job_events
                        FOR VALUES FROM ('{target_month}') TO ('{next_month}')
                        """
                    )
                    summary["partitions_created"].append(partition_name)
                    logger.info("Created partition: %s", partition_name)
                except Exception as e:
                    error_msg = f"Failed to create partition {partition_name}: {e}"
                    summary["errors"].append(error_msg)
                    logger.error(error_msg)

        # 2. Find and detach old partitions
        retention_days = get_retention_days()
        cutoff_date = today - timedelta(days=retention_days)
        cutoff_month = cutoff_date.replace(day=1)

        # List all partitions of job_events
        partitions = await conn.fetch(
            """
            SELECT c.relname AS partition_name
            FROM pg_inherits i
            JOIN pg_class c ON i.inhrelid = c.oid
            JOIN pg_class p ON i.inhparent = p.oid
            WHERE p.relname = 'job_events'
              AND c.relname != 'job_events_default'
            ORDER BY c.relname
            """
        )

        for row in partitions:
            partition_name = row["partition_name"]

            # Parse the year_month from the partition name (format: job_events_YYYY_MM)
            try:
                parts = partition_name.replace("job_events_", "").split("_")
                if len(parts) == 2:
                    year = int(parts[0])
                    month = int(parts[1])
                    partition_date = date(year, month, 1)

                    # Check if this partition is older than retention
                    if partition_date < cutoff_month:
                        # Check if partition has any data (verification)
                        count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {partition_name}"
                        )

                        # Detach the partition
                        try:
                            await conn.execute(
                                f"ALTER TABLE job_events DETACH PARTITION {partition_name}"
                            )
                            summary["partitions_detached"].append(partition_name)
                            logger.info(
                                "Detached partition: %s (%d rows)",
                                partition_name,
                                count,
                            )

                            # 3. Drop the detached partition
                            await conn.execute(f"DROP TABLE IF EXISTS {partition_name}")
                            summary["partitions_dropped"].append(partition_name)
                            logger.info("Dropped partition: %s", partition_name)

                        except Exception as e:
                            error_msg = f"Failed to detach/drop {partition_name}: {e}"
                            summary["errors"].append(error_msg)
                            logger.error(error_msg)
            except (ValueError, IndexError):
                # Skip partitions that don't match the expected naming pattern
                continue

    except Exception as e:
        summary["errors"].append(f"Maintenance failed: {e}")
        logger.error("Partition maintenance failed: %s", e)
    finally:
        await conn.close()

    return summary


async def partition_maintenance_loop() -> None:
    """
    Background loop that runs partition maintenance daily.

    Registered as an asyncio task in the application lifespan.
    """
    import asyncio

    # Wait for initial startup
    await asyncio.sleep(60)

    while True:
        try:
            logger.info("Running daily partition maintenance")
            summary = await run_partition_maintenance()
            logger.info(
                "Partition maintenance complete: created=%d detached=%d dropped=%d errors=%d",
                len(summary["partitions_created"]),
                len(summary["partitions_detached"]),
                len(summary["partitions_dropped"]),
                len(summary["errors"]),
            )
        except Exception as e:
            logger.error("Partition maintenance loop error: %s", e)

        # Wait 24 hours before next run
        await asyncio.sleep(86400)
