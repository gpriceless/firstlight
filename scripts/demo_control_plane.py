#!/usr/bin/env python3
"""
FirstLight LLM Control Plane -- Live Demo for MAIA Analytics
=============================================================

This script walks through the entire FirstLight Control Plane lifecycle,
showing how an LLM agent (like MAIA) would interact with the platform
to submit, monitor, steer, and consume geospatial analysis jobs.

Usage:
    python scripts/demo_control_plane.py
    python scripts/demo_control_plane.py --base-url http://api.firstlight.dev:8000
    python scripts/demo_control_plane.py --dry-run
    python scripts/demo_control_plane.py --speed fast

Requirements:
    pip install requests
    pip install sseclient-py   (optional, for nicer SSE handling)
"""

import argparse
import json
import sys
import textwrap
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    sys.exit(1)


# =============================================================================
# ANSI Color Helpers
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    WHITE = "\033[97m"
    UNDERLINE = "\033[4m"


def banner(text: str) -> None:
    """Print a large section banner."""
    width = 72
    border = Colors.BLUE + "=" * width + Colors.RESET
    print()
    print(border)
    print(Colors.BOLD + Colors.WHITE + text.center(width) + Colors.RESET)
    print(border)
    print()


def section(number: int, title: str) -> None:
    """Print a numbered section header."""
    print()
    print(
        Colors.BOLD + Colors.MAGENTA
        + f"  [{number}] {title}"
        + Colors.RESET
    )
    print(Colors.DIM + "  " + "-" * 68 + Colors.RESET)


def explain(text: str) -> None:
    """Print an explanation line in cyan."""
    for line in textwrap.wrap(text, width=68):
        print(Colors.CYAN + "  > " + line + Colors.RESET)


def success(text: str) -> None:
    """Print a success line in green."""
    print(Colors.GREEN + "  [OK] " + text + Colors.RESET)


def warn(text: str) -> None:
    """Print a warning line in yellow."""
    print(Colors.YELLOW + "  [!] " + text + Colors.RESET)


def error(text: str) -> None:
    """Print an error line in red."""
    print(Colors.RED + "  [ERROR] " + text + Colors.RESET)


def info(text: str) -> None:
    """Print an info line."""
    print(Colors.DIM + "  " + text + Colors.RESET)


def show_request(method: str, path: str, body: Optional[dict] = None) -> None:
    """Print a formatted HTTP request."""
    print()
    print(
        Colors.BOLD + Colors.WHITE
        + f"    {method} {path}"
        + Colors.RESET
    )
    if body is not None:
        formatted = json.dumps(body, indent=4)
        for line in formatted.split("\n"):
            print(Colors.DIM + "    " + line + Colors.RESET)


def show_response(status_code: int, body: Any) -> None:
    """Print a formatted HTTP response."""
    color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
    print(
        color + f"    HTTP {status_code}"
        + Colors.RESET
    )
    if body is not None:
        if isinstance(body, dict):
            formatted = json.dumps(body, indent=4, default=str)
        else:
            formatted = str(body)
        for line in formatted.split("\n"):
            print(Colors.DIM + "    " + line + Colors.RESET)
    print()


def show_event(event_data: dict) -> None:
    """Print a formatted SSE event."""
    etype = event_data.get("type", event_data.get("event_type", "unknown"))
    phase = event_data.get("phase", "")
    status = event_data.get("status", "")
    print(
        Colors.YELLOW + f"  << EVENT: {etype}"
        + Colors.DIM + f"  phase={phase} status={status}"
        + Colors.RESET
    )


def pause(seconds: float, label: str = "") -> None:
    """Pause with a visual indicator."""
    if seconds <= 0:
        return
    if label:
        print(Colors.DIM + f"  ... {label} ({seconds:.1f}s)" + Colors.RESET, end="", flush=True)
    else:
        print(Colors.DIM + f"  ... ({seconds:.1f}s)" + Colors.RESET, end="", flush=True)
    time.sleep(seconds)
    print()


# =============================================================================
# Speed Profiles
# =============================================================================

SPEED_PROFILES = {
    "fast": {
        "between_steps": 0.5,
        "processing_sim": 0.3,
        "transition_pause": 0.2,
    },
    "normal": {
        "between_steps": 1.5,
        "processing_sim": 1.0,
        "transition_pause": 0.5,
    },
    "slow": {
        "between_steps": 3.0,
        "processing_sim": 2.0,
        "transition_pause": 1.0,
    },
}


# =============================================================================
# Demo AOI Data -- Houston, TX (flood scenario)
# =============================================================================

# Approximate polygon around Houston Ship Channel / Buffalo Bayou area
# WGS84 (EPSG:4326), ~25 km2 coverage of historically flood-prone area
HOUSTON_AOI = {
    "type": "Polygon",
    "coordinates": [[
        [-95.3698, 29.7604],   # NW corner -- near downtown Houston
        [-95.2900, 29.7604],   # NE corner -- east of Ship Channel
        [-95.2900, 29.7100],   # SE corner -- south of Turning Basin
        [-95.3698, 29.7100],   # SW corner
        [-95.3698, 29.7604],   # close the ring
    ]],
}


# =============================================================================
# SSE Listener Thread
# =============================================================================

class SSEListener:
    """Background SSE event listener."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.events: List[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start listening for SSE events in a background thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the SSE listener."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_new_events(self) -> List[dict]:
        """Drain and return all collected events."""
        events = list(self.events)
        self.events.clear()
        return events

    def _listen(self) -> None:
        """Internal listener loop."""
        url = f"{self.base_url}/internal/v1/events/stream"
        headers = {"Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            # Try sseclient-py first for proper SSE parsing
            try:
                import sseclient
                resp = requests.get(url, headers=headers, stream=True, timeout=120)
                client = sseclient.SSEClient(resp)
                for event in client.events():
                    if self._stop.is_set():
                        break
                    try:
                        data = json.loads(event.data)
                        self.events.append(data)
                    except (json.JSONDecodeError, AttributeError):
                        pass
            except ImportError:
                # Fall back to raw streaming
                resp = requests.get(url, headers=headers, stream=True, timeout=120)
                for line in resp.iter_lines(decode_unicode=True):
                    if self._stop.is_set():
                        break
                    if line and line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            self.events.append(data)
                        except json.JSONDecodeError:
                            pass
        except requests.exceptions.ConnectionError:
            pass  # Server not available -- handled gracefully
        except Exception:
            pass


# =============================================================================
# Demo Runner
# =============================================================================

class ControlPlaneDemo:
    """Orchestrates the full control plane demo."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        speed: str = "normal",
        dry_run: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.speed = SPEED_PROFILES.get(speed, SPEED_PROFILES["normal"])
        self.dry_run = dry_run
        self.job_id: Optional[str] = None
        self.escalation_id: Optional[str] = None
        self.sse_listener: Optional[SSEListener] = None

        # State tracking for transitions
        self.current_phase = "QUEUED"
        self.current_status = "PENDING"

    @property
    def _headers(self) -> dict:
        """Standard request headers."""
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        label: str = "",
    ) -> Tuple[int, Any]:
        """Make an HTTP request, or simulate in dry-run mode."""
        url = f"{self.base_url}{path}"
        show_request(method, path, body)

        if self.dry_run:
            info("[DRY RUN] Would send the above request")
            return 200, {"dry_run": True, "note": "No actual request sent"}

        try:
            resp = requests.request(
                method=method,
                url=url,
                json=body,
                headers=self._headers,
                timeout=30,
            )
            try:
                resp_body = resp.json()
            except ValueError:
                resp_body = resp.text or None
            show_response(resp.status_code, resp_body)
            return resp.status_code, resp_body
        except requests.exceptions.ConnectionError:
            error(f"Cannot connect to {self.base_url}")
            error("Is the FirstLight API server running?")
            error(f"  Start with: flight serve   (or uvicorn api.app:app)")
            return 0, None

    def _drain_events(self) -> None:
        """Show any SSE events that have arrived."""
        if self.sse_listener is None:
            return
        events = self.sse_listener.get_new_events()
        if events:
            print()
            info("SSE events received:")
            for ev in events:
                show_event(ev)

    # -------------------------------------------------------------------------
    # Demo Steps
    # -------------------------------------------------------------------------

    def step_banner(self) -> None:
        """Print the opening banner."""
        banner("FirstLight LLM Control Plane Demo")
        explain(
            "Welcome to the FirstLight Control Plane walkthrough. "
            "This demo shows how an LLM agent -- such as MAIA -- "
            "interacts with FirstLight's geospatial intelligence platform."
        )
        print()
        explain(
            "FirstLight exposes four API surfaces:"
        )
        print()
        info("  1. /control/v1/*   -- LLM Control API (agent-facing)")
        info("     Jobs, transitions, reasoning, parameters, escalations, tools")
        print()
        info("  2. /internal/v1/*  -- Partner Integration API (MAIA backend)")
        info("     SSE event stream, webhooks, metrics, queue summary")
        print()
        info("  3. /oapi/*         -- OGC API Processes (standards-compliant)")
        info("     Process listing and execution per OGC spec")
        print()
        info("  4. /stac/*         -- STAC Catalog (results discovery)")
        info("     Browse and search published analysis results")
        print()

        if self.dry_run:
            warn("DRY RUN MODE -- No actual HTTP requests will be sent")
        print()
        info(f"Target: {self.base_url}")
        info(f"Speed:  {[k for k, v in SPEED_PROFILES.items() if v == self.speed][0]}")
        print()
        pause(self.speed["between_steps"], "Starting demo")

    def step_1_connectivity(self) -> bool:
        """Step 1: Check API connectivity."""
        section(1, "Setup & Discovery")
        explain(
            "First, we verify the API is reachable and discover what "
            "algorithms are available as LLM-callable tools."
        )

        # Health check
        explain("Checking API health...")
        status_code, body = self._request("GET", "/health")
        if status_code == 0 and not self.dry_run:
            error("API is not reachable. Aborting demo.")
            error(f"Expected server at {self.base_url}")
            return False
        if status_code >= 200 and status_code < 300:
            success("API is reachable")
        elif self.dry_run:
            success("[DRY RUN] Would verify connectivity")

        pause(self.speed["between_steps"])

        # Tool schemas
        explain(
            "Fetching available tool schemas. These are the algorithms "
            "FirstLight exposes in OpenAI function-calling format, so any "
            "LLM can call them natively."
        )
        status_code, body = self._request("GET", "/control/v1/tools")
        if body and isinstance(body, dict) and "tools" in body:
            tool_count = len(body["tools"])
            success(f"Discovered {tool_count} algorithm tool(s)")
            for tool in body["tools"][:5]:
                info(f"    - {tool.get('name', '?')}: {tool.get('description', '')[:60]}...")
        elif self.dry_run:
            success("[DRY RUN] Would list tool schemas")

        pause(self.speed["between_steps"])

        # OGC processes
        explain(
            "Checking OGC API Processes endpoint. This provides a "
            "standards-compliant view of the same algorithms for "
            "interoperability with GIS tools and OGC clients."
        )
        status_code, body = self._request("GET", "/oapi/processes")
        if status_code == 200:
            processes = body.get("processes", []) if isinstance(body, dict) else []
            success(f"OGC Processes endpoint available ({len(processes)} processes)")
        elif status_code == 404:
            info("OGC Processes endpoint not yet deployed (404)")
        elif self.dry_run:
            success("[DRY RUN] Would check OGC processes")

        explain(
            "TAKEAWAY: MAIA can discover all available algorithms at "
            "runtime -- no hardcoded knowledge needed."
        )

        pause(self.speed["between_steps"])
        return True

    def step_2_create_job(self) -> bool:
        """Step 2: Create a new analysis job."""
        section(2, "Create an Analysis Job")
        explain(
            "An LLM agent submits a job by providing an event type, "
            "an area of interest (AOI) as GeoJSON, and optional parameters."
        )
        explain(
            "Scenario: MAIA's NLP pipeline detected flood-related NOAA "
            "alerts in the Houston, TX area. It creates a flood detection "
            "job covering the Ship Channel / Buffalo Bayou region."
        )

        request_body = {
            "event_type": "flood",
            "aoi": HOUSTON_AOI,
            "parameters": {
                "sensitivity": "medium",
                "min_area_km2": 0.1,
                "include_sar": True,
            },
            "reasoning": (
                "Detected potential flood event in Houston area based on "
                "NOAA Weather Alert WEA-2026-0451. Multiple river gauges "
                "reporting above-flood-stage levels along Buffalo Bayou. "
                "Initiating SAR and optical flood detection analysis."
            ),
        }

        status_code, body = self._request(
            "POST", "/control/v1/jobs", request_body
        )

        if body and isinstance(body, dict) and "job_id" in body:
            self.job_id = body["job_id"]
            self.current_phase = body.get("phase", "QUEUED")
            self.current_status = body.get("status", "PENDING")
            success(f"Job created: {self.job_id}")
            info(f"  Phase: {self.current_phase}  Status: {self.current_status}")
        elif self.dry_run:
            self.job_id = "dry-run-00000000-0000-0000-0000-000000000000"
            success("[DRY RUN] Would create job (using placeholder ID)")

        explain(
            "TAKEAWAY: The job is queued with MAIA's reasoning preserved "
            "as an audit trail. The AOI is stored as PostGIS geometry in "
            "EPSG:4326 -- enabling spatial queries across all jobs."
        )

        pause(self.speed["between_steps"])
        return self.job_id is not None

    def step_3_start_sse(self) -> None:
        """Step 3: Start SSE event listener."""
        section(3, "Start Real-Time Event Stream")
        explain(
            "MAIA's backend would open a persistent SSE connection to "
            "receive real-time events for all jobs. Each event is wrapped "
            "in a CloudEvents v1.0 envelope for interoperability."
        )
        explain(
            "The stream supports Last-Event-ID for reconnection without "
            "missing events, customer_id scoping for multi-tenant isolation, "
            "and backpressure protection (max 500 buffered events)."
        )

        show_request("GET", "/internal/v1/events/stream")
        info("    Accept: text/event-stream")
        info("    Connection: keep-alive")
        print()

        if not self.dry_run:
            self.sse_listener = SSEListener(self.base_url, self.api_key)
            self.sse_listener.start()
            pause(1.0, "Connecting SSE stream")
            success("SSE listener started in background thread")
        else:
            success("[DRY RUN] Would open SSE stream")

        explain(
            "TAKEAWAY: Events stream in real time -- no polling needed. "
            "MAIA sees every state change, reasoning entry, and escalation "
            "the instant it happens."
        )

        pause(self.speed["between_steps"])

    def step_4_phase_transitions(self) -> None:
        """Step 4: Walk through the pipeline phase transitions."""
        section(4, "Pipeline Phase Transitions")
        explain(
            "A job moves through a defined lifecycle: QUEUED -> DISCOVERING "
            "-> INGESTING -> NORMALIZING -> ANALYZING -> REPORTING -> COMPLETE. "
            "Each transition is atomic with optimistic concurrency (TOCTOU guard)."
        )
        explain(
            "The expected_phase/expected_status fields prevent race conditions: "
            "if another actor already moved the job, the transition fails with "
            "HTTP 409 instead of silently corrupting state."
        )

        # Define the transition path
        transitions = [
            {
                "from_phase": "QUEUED", "from_status": "PENDING",
                "to_phase": "QUEUED", "to_status": "VALIDATING",
                "label": "Validating input parameters and AOI geometry",
                "detail": "Checks coordinate bounds, vertex count, CRS validity",
            },
            {
                "from_phase": "QUEUED", "from_status": "VALIDATING",
                "to_phase": "QUEUED", "to_status": "VALIDATED",
                "label": "Validation passed",
                "detail": "AOI is a valid WGS84 polygon with reasonable bounds",
            },
            {
                "from_phase": "QUEUED", "from_status": "VALIDATED",
                "to_phase": "DISCOVERING", "to_status": "DISCOVERING",
                "label": "Searching satellite data catalogs (STAC, Copernicus, USGS)",
                "detail": "Spatial+temporal query against STAC APIs for Sentinel-1/2 scenes",
            },
            {
                "from_phase": "DISCOVERING", "from_status": "DISCOVERING",
                "to_phase": "DISCOVERING", "to_status": "DISCOVERED",
                "label": "Found matching scenes",
                "detail": "Located 3 Sentinel-1 GRD scenes and 2 Sentinel-2 L2A scenes",
            },
            {
                "from_phase": "DISCOVERING", "from_status": "DISCOVERED",
                "to_phase": "INGESTING", "to_status": "INGESTING",
                "label": "Downloading and staging satellite imagery",
                "detail": "Fetching COG tiles from AWS S3 / Copernicus Data Space",
            },
            {
                "from_phase": "INGESTING", "from_status": "INGESTING",
                "to_phase": "INGESTING", "to_status": "INGESTED",
                "label": "Imagery staged locally",
                "detail": "5 scenes downloaded, checksums verified, 2.3 GB total",
            },
            {
                "from_phase": "INGESTING", "from_status": "INGESTED",
                "to_phase": "NORMALIZING", "to_status": "NORMALIZING",
                "label": "Band alignment, CRS reprojection, radiometric calibration",
                "detail": "Reprojecting to UTM 15N (EPSG:32615) for Houston area",
            },
            {
                "from_phase": "NORMALIZING", "from_status": "NORMALIZING",
                "to_phase": "NORMALIZING", "to_status": "NORMALIZED",
                "label": "Normalization complete",
                "detail": "All scenes co-registered, resampled to 10m, calibrated",
            },
            {
                "from_phase": "NORMALIZING", "from_status": "NORMALIZED",
                "to_phase": "ANALYZING", "to_status": "ANALYZING",
                "label": "Running flood detection algorithms",
                "detail": "SAR coherence change + NDWI thresholding + ML classifier",
            },
        ]

        for i, t in enumerate(transitions):
            pause(self.speed["transition_pause"])

            explain(f"{t['label']}")
            info(f"  {t['detail']}")

            request_body = {
                "expected_phase": t["from_phase"],
                "expected_status": t["from_status"],
                "target_phase": t["to_phase"],
                "target_status": t["to_status"],
                "reason": t["label"],
            }

            status_code, body = self._request(
                "POST",
                f"/control/v1/jobs/{self.job_id}/transition",
                request_body,
            )

            if body and isinstance(body, dict):
                self.current_phase = body.get("phase", t["to_phase"])
                self.current_status = body.get("status", t["to_status"])
                success(
                    f"Transitioned to {self.current_phase}/{self.current_status}"
                )
            elif self.dry_run:
                self.current_phase = t["to_phase"]
                self.current_status = t["to_status"]
                success(
                    f"[DRY RUN] {t['from_phase']}/{t['from_status']} "
                    f"-> {t['to_phase']}/{t['to_status']}"
                )

            # Show SSE events
            self._drain_events()

            # Simulate processing time
            if i in (2, 4, 6, 8):
                pause(
                    self.speed["processing_sim"],
                    "simulating processing",
                )

        explain(
            "TAKEAWAY: Every transition is recorded as an immutable event. "
            "The full audit trail shows exactly what happened, when, and why."
        )

        pause(self.speed["between_steps"])

    def step_5_inject_reasoning(self) -> None:
        """Step 5: Inject reasoning during analysis."""
        section(5, "Inject LLM Reasoning")
        explain(
            "LLM agents can inject their analytical reasoning at any "
            "decision point. This creates a traceable chain of thought "
            "that auditors can review alongside the algorithmic results."
        )

        # First reasoning entry -- during ANALYZING
        explain(
            "MAIA reviews intermediate SAR coherence results and adds "
            "its interpretation:"
        )

        reasoning_1 = {
            "reasoning": (
                "SAR coherence analysis indicates significant ground "
                "displacement in grid cells 4-7, consistent with surface "
                "water accumulation. Coherence drop from 0.85 to 0.23 in "
                "the Buffalo Bayou corridor. Cross-referencing with USGS "
                "gauge station 08074000 confirms water levels 4.2ft above "
                "flood stage. Confidence: HIGH. Recommending focused "
                "analysis on eastern quadrant where displacement is greatest."
            ),
            "confidence": 0.85,
            "payload": {
                "affected_grid_cells": [4, 5, 6, 7],
                "coherence_drop": {"from": 0.85, "to": 0.23},
                "gauge_station": "USGS-08074000",
                "water_level_above_flood_stage_ft": 4.2,
            },
        }

        status_code, body = self._request(
            "POST",
            f"/control/v1/jobs/{self.job_id}/reasoning",
            reasoning_1,
        )

        if body and isinstance(body, dict) and "event_seq" in body:
            success(f"Reasoning recorded as event #{body['event_seq']}")
        elif self.dry_run:
            success("[DRY RUN] Would record reasoning entry")

        self._drain_events()
        pause(self.speed["between_steps"])

        # Transition through QUALITY_CHECK
        explain("Analysis enters quality check phase:")
        status_code, body = self._request(
            "POST",
            f"/control/v1/jobs/{self.job_id}/transition",
            {
                "expected_phase": "ANALYZING",
                "expected_status": "ANALYZING",
                "target_phase": "ANALYZING",
                "target_status": "QUALITY_CHECK",
                "reason": "Running QA checks on detection output",
            },
        )
        if body and isinstance(body, dict):
            self.current_phase = body.get("phase", "ANALYZING")
            self.current_status = body.get("status", "QUALITY_CHECK")
        elif self.dry_run:
            self.current_phase = "ANALYZING"
            self.current_status = "QUALITY_CHECK"

        pause(self.speed["processing_sim"], "running quality checks")

        explain(
            "TAKEAWAY: Reasoning entries are first-class citizens in the "
            "event log -- not afterthoughts. Every LLM decision is recorded "
            "with confidence scores and structured payloads."
        )

        pause(self.speed["between_steps"])

    def step_6_adjust_parameters(self) -> None:
        """Step 6: Adjust parameters mid-flight."""
        section(6, "Adjust Parameters Mid-Flight")
        explain(
            "Based on intermediate results, the LLM agent can tune algorithm "
            "parameters without restarting the job. This uses JSON merge-patch "
            "semantics (RFC 7386) -- only the changed keys are sent."
        )
        explain(
            "MAIA decides to increase sensitivity and lower the change "
            "threshold after seeing strong coherence signals:"
        )

        patch_body = {
            "sensitivity": "high",
            "min_change_threshold": 0.15,
            "focus_quadrant": "NE",
        }

        status_code, body = self._request(
            "PATCH",
            f"/control/v1/jobs/{self.job_id}/parameters",
            patch_body,
        )

        if status_code == 200 and body:
            success("Parameters updated successfully")
            info(f"  New parameters: {json.dumps(body, indent=2)[:200]}")
        elif self.dry_run:
            success("[DRY RUN] Would patch parameters")

        self._drain_events()

        explain(
            "TAKEAWAY: Parameter tuning is a merge-patch operation. The LLM "
            "sends only what changed. Setting a key to null removes it. "
            "Terminal jobs reject parameter changes (HTTP 409)."
        )

        pause(self.speed["between_steps"])

    def step_7_escalation(self) -> None:
        """Step 7: Demonstrate the escalation workflow."""
        section(7, "Escalation Workflow")
        explain(
            "When the LLM encounters something outside its confidence "
            "threshold, it escalates to a human operator. Escalations have "
            "severity levels (LOW, MEDIUM, HIGH, CRITICAL) and structured "
            "context for the reviewer."
        )

        # Create escalation
        explain(
            "MAIA detects an anomalous pattern -- possible sensor artifact "
            "vs. real flood extent -- and escalates for human review:"
        )

        escalation_body = {
            "severity": "HIGH",
            "reason": (
                "Detected anomalous reflectance pattern in grid cell 6 that "
                "could indicate either (a) genuine flash flood extent beyond "
                "historical norms or (b) SAR sensor artifact from wind-roughened "
                "water surface. Confidence in flood classification: 0.62 -- below "
                "the 0.75 threshold for autonomous reporting. Requesting human "
                "review of the SAR backscatter imagery before finalizing results."
            ),
            "context": {
                "grid_cell": 6,
                "confidence": 0.62,
                "threshold": 0.75,
                "possible_causes": [
                    "genuine_flood_extent",
                    "wind_roughened_surface_artifact",
                ],
                "recommended_action": "Review SAR backscatter in QuickLook viewer",
            },
        }

        status_code, body = self._request(
            "POST",
            f"/control/v1/jobs/{self.job_id}/escalations",
            escalation_body,
        )

        if body and isinstance(body, dict) and "escalation_id" in body:
            self.escalation_id = body["escalation_id"]
            success(f"Escalation created: {self.escalation_id}")
            info(f"  Severity: {body.get('severity', 'HIGH')}")
        elif self.dry_run:
            self.escalation_id = "dry-run-esc-00000000"
            success("[DRY RUN] Would create escalation")

        self._drain_events()
        pause(self.speed["processing_sim"], "human reviews the imagery")

        # Resolve escalation
        explain(
            "The human operator reviews the SAR imagery and confirms "
            "the flood extent is real:"
        )

        resolve_body = {
            "resolution": (
                "Reviewed SAR backscatter in QuickLook viewer. Pattern in "
                "grid cell 6 is consistent with genuine flood extent -- the "
                "irregular boundary matches the Greens Bayou overflow pattern "
                "from historical events. Wind artifact ruled out based on "
                "VV/VH polarization ratio analysis. Classification confidence "
                "upgraded to 0.91. Approved for reporting."
            ),
        }

        status_code, body = self._request(
            "PATCH",
            f"/control/v1/jobs/{self.job_id}/escalations/{self.escalation_id}",
            resolve_body,
        )

        if status_code == 200 and body:
            success("Escalation resolved")
            info(f"  Resolved by: {body.get('resolved_by', 'operator')}")
        elif self.dry_run:
            success("[DRY RUN] Would resolve escalation")

        self._drain_events()

        explain(
            "TAKEAWAY: Escalation is a first-class workflow, not an error "
            "state. The job continues processing while awaiting human input. "
            "The full escalation context and resolution are preserved in the "
            "audit trail."
        )

        pause(self.speed["between_steps"])

    def step_8_complete_pipeline(self) -> None:
        """Step 8: Complete the remaining pipeline transitions."""
        section(8, "Complete the Pipeline")
        explain(
            "After the escalation is resolved, we move through the final "
            "pipeline stages to completion."
        )

        remaining_transitions = [
            {
                "from_phase": "ANALYZING", "from_status": "QUALITY_CHECK",
                "to_phase": "ANALYZING", "to_status": "ANALYZED",
                "label": "Analysis complete -- flood extent map generated",
            },
            {
                "from_phase": "ANALYZING", "from_status": "ANALYZED",
                "to_phase": "REPORTING", "to_status": "REPORTING",
                "label": "Generating analysis products and report",
            },
            {
                "from_phase": "REPORTING", "from_status": "REPORTING",
                "to_phase": "REPORTING", "to_status": "ASSEMBLING",
                "label": "Assembling final deliverables (GeoTIFF, GeoJSON, PDF)",
            },
        ]

        for t in remaining_transitions:
            pause(self.speed["transition_pause"])
            explain(t["label"])

            status_code, body = self._request(
                "POST",
                f"/control/v1/jobs/{self.job_id}/transition",
                {
                    "expected_phase": t["from_phase"],
                    "expected_status": t["from_status"],
                    "target_phase": t["to_phase"],
                    "target_status": t["to_status"],
                    "reason": t["label"],
                },
            )

            if body and isinstance(body, dict):
                self.current_phase = body.get("phase", t["to_phase"])
                self.current_status = body.get("status", t["to_status"])
                success(f"{self.current_phase}/{self.current_status}")
            elif self.dry_run:
                self.current_phase = t["to_phase"]
                self.current_status = t["to_status"]
                success(f"[DRY RUN] {t['to_phase']}/{t['to_status']}")

            self._drain_events()

        # Inject final reasoning during ASSEMBLING
        explain(
            "MAIA adds a final reasoning entry summarizing its conclusions:"
        )

        reasoning_final = {
            "reasoning": (
                "Flood detection analysis complete for Houston Ship Channel "
                "area. Key findings: (1) Buffalo Bayou overflow detected "
                "along 4.2km stretch between Waugh Dr and US-59. (2) Estimated "
                "flood extent: 3.8 km2. (3) 94 structures identified within "
                "flood boundary using building footprint overlay. (4) SAR "
                "coherence method corroborated by NDWI optical analysis with "
                "87% spatial agreement. Confidence in overall assessment: 0.91."
            ),
            "confidence": 0.91,
            "payload": {
                "flood_extent_km2": 3.8,
                "affected_structures": 94,
                "sar_optical_agreement": 0.87,
                "primary_method": "SAR coherence change detection",
                "corroboration": "NDWI optical thresholding",
            },
        }

        status_code, body = self._request(
            "POST",
            f"/control/v1/jobs/{self.job_id}/reasoning",
            reasoning_final,
        )

        if body and isinstance(body, dict) and "event_seq" in body:
            success(f"Final reasoning recorded as event #{body['event_seq']}")
        elif self.dry_run:
            success("[DRY RUN] Would record final reasoning")

        pause(self.speed["processing_sim"], "assembling final products")

        # Complete the pipeline
        final_transitions = [
            {
                "from_phase": "REPORTING", "from_status": "ASSEMBLING",
                "to_phase": "REPORTING", "to_status": "REPORTED",
                "label": "Report assembled and validated",
            },
            {
                "from_phase": "REPORTING", "from_status": "REPORTED",
                "to_phase": "COMPLETE", "to_status": "COMPLETE",
                "label": "Job complete -- results published to STAC catalog",
            },
        ]

        for t in final_transitions:
            pause(self.speed["transition_pause"])
            explain(t["label"])

            status_code, body = self._request(
                "POST",
                f"/control/v1/jobs/{self.job_id}/transition",
                {
                    "expected_phase": t["from_phase"],
                    "expected_status": t["from_status"],
                    "target_phase": t["to_phase"],
                    "target_status": t["to_status"],
                    "reason": t["label"],
                },
            )

            if body and isinstance(body, dict):
                self.current_phase = body.get("phase", t["to_phase"])
                self.current_status = body.get("status", t["to_status"])
                success(f"{self.current_phase}/{self.current_status}")
            elif self.dry_run:
                self.current_phase = t["to_phase"]
                self.current_status = t["to_status"]
                success(f"[DRY RUN] {t['to_phase']}/{t['to_status']}")

            self._drain_events()

        success("Pipeline complete")

        explain(
            "TAKEAWAY: The job has traversed 7 phases and 15+ state "
            "transitions. Every step is recorded, auditable, and reproducible."
        )

        pause(self.speed["between_steps"])

    def step_9_metrics(self) -> None:
        """Step 9: Check pipeline health metrics."""
        section(9, "Pipeline Health & Queue Status")
        explain(
            "MAIA's operations dashboard polls these endpoints to monitor "
            "platform health. Metrics are served from materialized views "
            "that refresh every 30 seconds -- responses are under 100ms."
        )

        # Pipeline metrics
        explain("Pipeline health metrics:")
        status_code, body = self._request("GET", "/internal/v1/metrics")
        if status_code == 200 and body:
            success("Metrics retrieved")
            if isinstance(body, dict):
                info(f"  Jobs completed (1h):  {body.get('jobs_completed_1h', 0)}")
                info(f"  Jobs failed (1h):     {body.get('jobs_failed_1h', 0)}")
                info(f"  Jobs completed (24h): {body.get('jobs_completed_24h', 0)}")
                info(f"  P50 duration:         {body.get('p50_duration_s', 0):.1f}s")
                info(f"  P95 duration:         {body.get('p95_duration_s', 0):.1f}s")
                info(f"  Active SSE conns:     {body.get('active_sse_connections', 0)}")
        elif self.dry_run:
            success("[DRY RUN] Would fetch pipeline metrics")

        pause(self.speed["between_steps"])

        # Queue summary
        explain("Queue summary (per-phase breakdown):")
        status_code, body = self._request("GET", "/internal/v1/queue/summary")
        if status_code == 200 and body:
            success("Queue summary retrieved")
            if isinstance(body, dict):
                counts = body.get("per_phase_counts", {})
                if counts:
                    for phase, count in counts.items():
                        info(f"  {phase}: {count} job(s)")
                info(f"  Stuck jobs:         {body.get('stuck_count', 0)}")
                info(f"  Awaiting review:    {body.get('awaiting_review_count', 0)}")
        elif self.dry_run:
            success("[DRY RUN] Would fetch queue summary")

        explain(
            "TAKEAWAY: Metrics are cheap to query (materialized views, "
            "not live aggregation). MAIA's dashboard can poll every few "
            "seconds without impacting API performance."
        )

        pause(self.speed["between_steps"])

    def step_10_stac(self) -> None:
        """Step 10: Verify STAC publishing."""
        section(10, "STAC Catalog -- Results Discovery")
        explain(
            "Every completed analysis is automatically published as a "
            "STAC Item. This makes results discoverable by any "
            "STAC-compatible client -- QGIS, ArcGIS, custom dashboards, "
            "or other LLM agents."
        )

        # List collections
        explain("Available STAC collections:")
        status_code, body = self._request("GET", "/stac/collections")
        if status_code == 200 and body:
            collections = body.get("collections", []) if isinstance(body, dict) else []
            success(f"Found {len(collections)} collection(s)")
            for coll in collections[:5]:
                info(f"  - {coll.get('id', '?')}: {coll.get('title', coll.get('description', ''))[:50]}")
        elif status_code == 404:
            info("STAC endpoint not yet deployed (results would appear here after first job)")
        elif self.dry_run:
            success("[DRY RUN] Would list STAC collections")

        pause(self.speed["between_steps"])

        # Try to get the published item
        if self.job_id and not self.dry_run:
            explain(
                f"Looking for our completed job's STAC item:"
            )
            status_code, body = self._request(
                "GET", f"/stac/collections/flood/items/{self.job_id}"
            )
            if status_code == 200 and body:
                success("STAC Item found")
                if isinstance(body, dict):
                    info(f"  ID:       {body.get('id', '?')}")
                    info(f"  Type:     {body.get('type', '?')}")
                    bbox = body.get("bbox", [])
                    if bbox:
                        info(f"  BBox:     [{', '.join(f'{x:.4f}' for x in bbox)}]")
                    assets = body.get("assets", {})
                    if assets:
                        info(f"  Assets:   {', '.join(assets.keys())}")
            elif status_code == 404:
                info("STAC Item not yet published (async publishing may still be in progress)")
        elif self.dry_run:
            explain("Looking for our completed job in the STAC catalog:")
            show_request("GET", f"/stac/collections/flood/items/{self.job_id}")
            info("[DRY RUN] Would look up the STAC item for this job")

        explain(
            "TAKEAWAY: Results are immediately discoverable through standard "
            "STAC APIs. No proprietary format -- just open geospatial standards."
        )

        pause(self.speed["between_steps"])

    def step_summary(self) -> None:
        """Print the final summary."""
        banner("Demo Complete")

        explain("Here is what we just demonstrated:")
        print()
        info("  1. TOOL DISCOVERY       -- LLM discovers algorithms at runtime")
        info("  2. JOB CREATION         -- Submit analysis with AOI + reasoning")
        info("  3. REAL-TIME EVENTS     -- SSE stream with CloudEvents envelopes")
        info("  4. PHASE TRANSITIONS    -- Atomic state machine with TOCTOU guards")
        info("  5. REASONING INJECTION  -- LLM records its chain of thought")
        info("  6. PARAMETER TUNING     -- Adjust algorithms mid-flight")
        info("  7. ESCALATION WORKFLOW  -- Human-in-the-loop when LLM is uncertain")
        info("  8. PIPELINE COMPLETION  -- Full lifecycle from QUEUED to COMPLETE")
        info("  9. HEALTH METRICS       -- Operations dashboard data")
        info("  10. STAC PUBLISHING     -- Results discoverable via open standards")
        print()

        explain("The four API surfaces and their purposes:")
        print()
        print(
            Colors.BOLD + Colors.WHITE
            + "    Surface             Audience         Purpose"
            + Colors.RESET
        )
        print(Colors.DIM + "    " + "-" * 60 + Colors.RESET)
        info("    /control/v1/*       LLM Agents       Read/write job state")
        info("    /internal/v1/*      Partner Backend   Events, metrics, webhooks")
        info("    /oapi/*             GIS Tools         OGC-compliant processes")
        info("    /stac/*             Any Client        Discover analysis results")
        print()

        explain("Endpoint map:")
        print()
        endpoints = [
            ("POST", "/control/v1/jobs", "Create a new analysis job"),
            ("GET", "/control/v1/jobs", "List jobs (filterable by phase, bbox)"),
            ("GET", "/control/v1/jobs/{id}", "Get full job detail"),
            ("POST", "/control/v1/jobs/{id}/transition", "Atomic state transition"),
            ("PATCH", "/control/v1/jobs/{id}/parameters", "Tune parameters (merge-patch)"),
            ("POST", "/control/v1/jobs/{id}/reasoning", "Inject LLM reasoning"),
            ("POST", "/control/v1/jobs/{id}/escalations", "Create escalation"),
            ("PATCH", "/control/v1/jobs/{id}/escalations/{eid}", "Resolve escalation"),
            ("GET", "/control/v1/tools", "Discover algorithm tool schemas"),
            ("GET", "/internal/v1/events/stream", "SSE event stream"),
            ("GET", "/internal/v1/metrics", "Pipeline health metrics"),
            ("GET", "/internal/v1/queue/summary", "Queue status summary"),
        ]

        for method, path, desc in endpoints:
            method_color = {
                "GET": Colors.GREEN,
                "POST": Colors.YELLOW,
                "PATCH": Colors.CYAN,
            }.get(method, Colors.WHITE)
            print(
                f"    {method_color}{method:6s}{Colors.RESET} "
                f"{Colors.WHITE}{path:50s}{Colors.RESET} "
                f"{Colors.DIM}{desc}{Colors.RESET}"
            )

        print()
        if self.job_id:
            info(f"  Demo job ID: {self.job_id}")
        print()
        success("End of demo. Questions?")
        print()

    # -------------------------------------------------------------------------
    # Main Runner
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full demo sequence."""
        try:
            self.step_banner()

            if not self.step_1_connectivity():
                return

            if not self.step_2_create_job():
                error("Job creation failed. Cannot continue demo.")
                return

            self.step_3_start_sse()
            self.step_4_phase_transitions()
            self.step_5_inject_reasoning()
            self.step_6_adjust_parameters()
            self.step_7_escalation()
            self.step_8_complete_pipeline()
            self.step_9_metrics()
            self.step_10_stac()
            self.step_summary()

        except KeyboardInterrupt:
            print()
            warn("Demo interrupted by user")
        finally:
            if self.sse_listener:
                self.sse_listener.stop()


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FirstLight LLM Control Plane Demo for MAIA Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/demo_control_plane.py
              python scripts/demo_control_plane.py --dry-run
              python scripts/demo_control_plane.py --base-url http://api.firstlight.dev:8000
              python scripts/demo_control_plane.py --speed fast --api-key sk-demo-123
        """),
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="FirstLight API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for authentication (Bearer token)",
    )
    parser.add_argument(
        "--speed",
        choices=["fast", "normal", "slow"],
        default="normal",
        help="Demo pacing: fast (0.2-0.5s), normal (0.5-1.5s), slow (1-3s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any API calls",
    )

    args = parser.parse_args()

    demo = ControlPlaneDemo(
        base_url=args.base_url,
        api_key=args.api_key,
        speed=args.speed,
        dry_run=args.dry_run,
    )
    demo.run()


if __name__ == "__main__":
    main()
