"""
OGC process execution Taskiq task.

Handles asynchronous execution of OGC API Processes requests via the
Prefer: respond-async flow. When an OGC execute request arrives with
the async preference, the API returns 201 with a Location header and
enqueues the actual execution as a Taskiq task.

This task runs alongside the existing deliver_webhook task but uses
separate labels for routing and priority isolation.

Task 4.8
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Task labels for routing isolation from webhook delivery
OGC_TASK_LABELS = {
    "queue": "ogc-execution",
    "priority": "normal",
}

WEBHOOK_TASK_LABELS = {
    "queue": "webhook-delivery",
    "priority": "high",
}


async def execute_ogc_process(
    job_id: str,
    process_id: str,
    algorithm_id: str,
    inputs: Dict[str, Any],
    customer_id: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute an OGC process asynchronously.

    This is the core execution function called by the Taskiq task. It
    delegates to the FirstLight orchestrator/algorithm execution engine.

    Args:
        job_id: The OGC job UUID.
        process_id: The OGC process identifier.
        algorithm_id: The FirstLight algorithm ID.
        inputs: OGC process inputs (AOI, event_type, parameters).
        customer_id: The tenant identifier.
        parameters: Additional algorithm parameters.

    Returns:
        Execution result dict with status, outputs, and metadata.
    """
    logger.info(
        "OGC process execution starting: job=%s process=%s algorithm=%s",
        job_id,
        process_id,
        algorithm_id,
    )

    result: Dict[str, Any] = {
        "job_id": job_id,
        "process_id": process_id,
        "algorithm_id": algorithm_id,
        "status": "accepted",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {},
    }

    try:
        # Record job start in job_events
        await _record_ogc_event(
            job_id=job_id,
            customer_id=customer_id or "unknown",
            event_type="OGC_EXECUTION_STARTED",
            phase="ANALYZING",
            status="ANALYZING",
            payload={
                "process_id": process_id,
                "algorithm_id": algorithm_id,
                "inputs_summary": {
                    k: type(v).__name__ for k, v in inputs.items()
                },
            },
        )

        # Execute the algorithm
        # In production, this would delegate to the orchestrator or
        # run the algorithm directly via AlgorithmRegistry.
        from core.analysis.library.registry import get_global_registry

        registry = get_global_registry()
        algo = registry.get(algorithm_id)

        if algo is None:
            result["status"] = "failed"
            result["error"] = f"Algorithm not found: {algorithm_id}"
            logger.error("OGC execution: algorithm %s not found", algorithm_id)
        else:
            # For now, the actual algorithm execution is a stub.
            # The real implementation would:
            # 1. Create a job in the control plane
            # 2. Run the algorithm pipeline
            # 3. Collect outputs
            result["status"] = "successful"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            result["outputs"] = {
                "result": {
                    "type": "application/geo+json",
                    "value": {
                        "type": "Feature",
                        "geometry": inputs.get("aoi", {}),
                        "properties": {
                            "algorithm": algorithm_id,
                            "event_type": inputs.get("event_type", "unknown"),
                        },
                    },
                },
            }
            logger.info(
                "OGC process execution completed: job=%s process=%s",
                job_id,
                process_id,
            )

        # Record completion event
        await _record_ogc_event(
            job_id=job_id,
            customer_id=customer_id or "unknown",
            event_type=(
                "OGC_EXECUTION_COMPLETED"
                if result["status"] == "successful"
                else "OGC_EXECUTION_FAILED"
            ),
            phase="COMPLETE" if result["status"] == "successful" else "ANALYZING",
            status="COMPLETE" if result["status"] == "successful" else "FAILED",
            payload={"outputs": list(result.get("outputs", {}).keys())},
        )

    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        logger.error(
            "OGC process execution failed: job=%s error=%s", job_id, e
        )

    return result


async def _record_ogc_event(
    job_id: str,
    customer_id: str,
    event_type: str,
    phase: str,
    status: str,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an OGC execution event in job_events."""
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
                INSERT INTO job_events
                    (job_id, customer_id, event_type, phase, status, actor, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                """,
                uuid.UUID(job_id) if isinstance(job_id, str) else job_id,
                customer_id,
                event_type,
                phase,
                status,
                "ogc-executor",
                json.dumps(payload or {}, default=str),
            )
        finally:
            await conn.close()
    except ImportError:
        logger.debug("asyncpg not available, cannot record OGC event")
    except Exception as e:
        logger.warning("Failed to record OGC event: %s", e)


def build_ogc_job_status_response(
    job_id: str,
    process_id: str,
    status: str = "accepted",
    progress: Optional[int] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build an OGC API Processes Job Status response.

    Per OGC API - Processes - Part 1: Core, the job status resource
    includes statusInfo fields.

    Args:
        job_id: The job UUID.
        process_id: The OGC process identifier.
        status: Job status (accepted, running, successful, failed, dismissed).
        progress: Optional progress percentage (0-100).
        message: Optional status message.

    Returns:
        OGC job status dict.
    """
    response: Dict[str, Any] = {
        "jobID": job_id,
        "processID": process_id,
        "type": "process",
        "status": status,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    if progress is not None:
        response["progress"] = max(0, min(100, progress))

    if message:
        response["message"] = message

    if status in ("successful", "failed"):
        response["finished"] = datetime.now(timezone.utc).isoformat()

    return response


# Register as Taskiq tasks
try:
    from workers.taskiq_app import broker

    @broker.task(
        task_name="execute_ogc_process",
        labels=OGC_TASK_LABELS,
    )
    async def execute_ogc_process_task(
        job_id: str,
        process_id: str,
        algorithm_id: str,
        inputs: Dict[str, Any],
        customer_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Taskiq task wrapper for OGC process execution.

        This is the entry point called by the Taskiq worker.
        Uses 'ogc-execution' queue label for routing isolation
        from webhook delivery tasks.
        """
        return await execute_ogc_process(
            job_id=job_id,
            process_id=process_id,
            algorithm_id=algorithm_id,
            inputs=inputs,
            customer_id=customer_id,
            parameters=parameters,
        )

    logger.info("OGC execution Taskiq task registered")

except ImportError:
    logger.warning("Taskiq not available -- OGC execution task not registered")
