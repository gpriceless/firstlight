"""
CloudEvents v1.0 envelope builder.

Wraps job_events rows into CloudEvents v1.0 compliant envelopes for
SSE streaming and webhook delivery. Each event uses the type namespace
`io.firstlight.job.<event_type>` and includes FirstLight-specific
extension attributes.

Spec: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def build_cloudevent(
    event_seq: int,
    job_id: str,
    customer_id: str,
    event_type: str,
    phase: str,
    status: str,
    occurred_at: Optional[datetime] = None,
    payload: Optional[Dict[str, Any]] = None,
    reasoning: Optional[str] = None,
    actor: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a CloudEvents v1.0 envelope from a job_events row.

    Args:
        event_seq: The event sequence number (used as CloudEvent id).
        job_id: The job UUID.
        customer_id: The tenant/customer identifier.
        event_type: The event type string (e.g., "STATE_TRANSITION").
        phase: Current job phase at time of event.
        status: Current job status at time of event.
        occurred_at: When the event occurred. Defaults to now (UTC).
        payload: Optional structured payload data.
        reasoning: Optional reasoning text from LLM agent.
        actor: Who triggered the event.

    Returns:
        A dictionary conforming to CloudEvents v1.0 structured content mode.
    """
    if occurred_at is None:
        occurred_at = datetime.now(timezone.utc)

    # Normalize event_type to CloudEvents type namespace
    ce_type = f"io.firstlight.job.{event_type.lower().replace('_', '.')}"

    envelope: Dict[str, Any] = {
        # Required CloudEvents attributes
        "specversion": "1.0",
        "type": ce_type,
        "source": f"/jobs/{job_id}",
        "id": str(event_seq),
        "time": occurred_at.isoformat() if isinstance(occurred_at, datetime) else str(occurred_at),
        "datacontenttype": "application/json",
        # FirstLight extension attributes
        "firstlight_job_id": str(job_id),
        "firstlight_customer_id": customer_id,
        "firstlight_phase": phase,
        "firstlight_status": status,
        # Data payload
        "data": payload or {},
    }

    # Add optional fields to data if present
    if reasoning is not None:
        envelope["data"]["reasoning"] = reasoning
    if actor is not None:
        envelope["data"]["actor"] = actor

    return envelope


def cloudevent_to_sse_frame(
    envelope: Dict[str, Any],
) -> str:
    """
    Format a CloudEvents envelope as an SSE frame.

    Returns a string in SSE wire format:
        id: <event_seq>
        event: <type>
        data: <json>

    Args:
        envelope: A CloudEvents v1.0 envelope dict.

    Returns:
        SSE-formatted string ready to be written to the streaming response.
    """
    event_id = envelope.get("id", "")
    event_type = envelope.get("type", "message")
    data_json = json.dumps(envelope, default=str)

    lines = [
        f"id: {event_id}",
        f"event: {event_type}",
        f"data: {data_json}",
        "",  # Empty line terminates the frame
        "",
    ]
    return "\n".join(lines)


def parse_job_event_notification(notification_payload: str) -> Dict[str, Any]:
    """
    Parse a pg_notify payload from the job_events trigger.

    The notification payload is a JSON string produced by notify_job_event().

    Args:
        notification_payload: Raw string from pg_notify.

    Returns:
        Parsed dict with event fields.
    """
    return json.loads(notification_payload)
