#!/usr/bin/env python3
"""
FirstLight Control Plane — Claude Agent Demo.

Connects Claude to the live FirstLight API and lets it drive a flood
analysis job end-to-end. Claude discovers algorithms, creates a job,
walks it through each pipeline phase, injects reasoning, tunes
parameters, escalates when uncertain, and completes the analysis.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python scripts/claude_agent_demo.py

Optional env vars:
    FIRSTLIGHT_API_URL  — API base URL  (default: http://localhost:8000)
    FIRSTLIGHT_API_KEY  — API key       (default: demo-18254ee7d18f5926)
    CLAUDE_MODEL        — Model to use  (default: claude-sonnet-4-20250514)
"""

import json
import os
import sys
import time

import anthropic
import requests

# ── Config ──────────────────────────────────────────────────────────────

API_BASE = os.environ.get(
    "FIRSTLIGHT_API_URL", "http://localhost:8000"
).rstrip("/")

API_KEY = os.environ.get("FIRSTLIGHT_API_KEY", "demo-18254ee7d18f5926")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")

MAX_TURNS = 35

# ── ANSI Colors ─────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"

# ── Tool Definitions ────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "discover_algorithms",
        "description": (
            "Discover available analysis algorithms and their parameter "
            "schemas. Returns tools in OpenAI function-calling format. "
            "Call this first to understand what the platform can do."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "create_job",
        "description": (
            "Create a new analysis job. Returns the job_id you will use "
            "for all subsequent operations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "description": "Event to analyze: flood, wildfire, storm, etc.",
                },
                "aoi": {
                    "type": "object",
                    "description": "GeoJSON Polygon geometry for the area of interest.",
                },
                "parameters": {
                    "type": "object",
                    "description": "Initial algorithm parameters.",
                    "default": {},
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why you are creating this job.",
                },
            },
            "required": ["event_type", "aoi", "reasoning"],
        },
    },
    {
        "name": "get_job_status",
        "description": "Get the current phase, status, parameters, and AOI of a job.",
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
            },
            "required": ["job_id"],
        },
    },
    {
        "name": "transition_job",
        "description": (
            "Atomically move a job from one (phase, status) to another. "
            "You MUST specify the expected current state as a TOCTOU guard. "
            "If the job is not in the expected state, the call fails with 409."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "expected_phase": {"type": "string"},
                "expected_status": {"type": "string"},
                "target_phase": {"type": "string"},
                "target_status": {"type": "string"},
                "reason": {
                    "type": "string",
                    "description": "Why you are making this transition.",
                },
            },
            "required": [
                "job_id",
                "expected_phase",
                "expected_status",
                "target_phase",
                "target_status",
                "reason",
            ],
        },
    },
    {
        "name": "inject_reasoning",
        "description": (
            "Record your analysis reasoning at the current phase. "
            "This is a first-class audit record — not a comment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "reasoning": {
                    "type": "string",
                    "description": "Your analysis, observations, and logic.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0.",
                },
                "payload": {
                    "type": "object",
                    "description": "Structured supporting data.",
                    "default": {},
                },
            },
            "required": ["job_id", "reasoning", "confidence"],
        },
    },
    {
        "name": "adjust_parameters",
        "description": (
            "Adjust algorithm parameters mid-analysis. Uses JSON merge-patch: "
            "only send the keys you want to change."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "description": "Parameters to update.",
                },
            },
            "required": ["job_id", "parameters"],
        },
    },
    {
        "name": "create_escalation",
        "description": (
            "Escalate to a human analyst when you are uncertain. "
            "The job continues processing — escalation is not a failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "severity": {
                    "type": "string",
                    "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                },
                "reason": {
                    "type": "string",
                    "description": "What you need help with.",
                },
                "context": {
                    "type": "object",
                    "description": "Structured context for the reviewer.",
                    "default": {},
                },
            },
            "required": ["job_id", "severity", "reason"],
        },
    },
    {
        "name": "resolve_escalation",
        "description": (
            "Resolve an open escalation (simulating a human analyst response)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {"type": "string"},
                "escalation_id": {"type": "string"},
                "resolution": {
                    "type": "string",
                    "description": "The human analyst's resolution.",
                },
            },
            "required": ["job_id", "escalation_id", "resolution"],
        },
    },
    {
        "name": "query_context",
        "description": (
            "Query the geospatial context lakehouse. Returns datasets, "
            "buildings, infrastructure, or weather within a bounding box."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "layer": {
                    "type": "string",
                    "enum": ["datasets", "buildings", "infrastructure", "weather"],
                    "description": "Which context layer to query.",
                },
                "bbox": {
                    "type": "string",
                    "description": "west,south,east,north in decimal degrees.",
                },
            },
            "required": ["layer", "bbox"],
        },
    },
    {
        "name": "get_pipeline_metrics",
        "description": (
            "Get pipeline health metrics: jobs completed/failed, "
            "durations, queue status, escalation backlog."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

# ── System Prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an LLM agent connected to the FirstLight geospatial event \
intelligence platform via its Control Plane API.

You will analyze a flood event in the Houston Ship Channel area by \
driving a job through the full pipeline, making decisions at every phase.

## Available Tools
You have tools to discover algorithms, create jobs, transition state, \
inject reasoning, adjust parameters, escalate to humans, query the \
geospatial context lakehouse, and check pipeline metrics.

## Pipeline Phases
You drive each transition manually using transition_job. The valid \
progression (each arrow is a transition you must make):

  QUEUED/PENDING → QUEUED/VALIDATING → QUEUED/VALIDATED
  → DISCOVERING/DISCOVERING → DISCOVERING/DISCOVERED
  → INGESTING/INGESTING → INGESTING/INGESTED
  → NORMALIZING/NORMALIZING → NORMALIZING/NORMALIZED
  → ANALYZING/ANALYZING → ANALYZING/QUALITY_CHECK → ANALYZING/ANALYZED
  → REPORTING/REPORTING → REPORTING/ASSEMBLING → REPORTING/REPORTED
  → COMPLETE/COMPLETE

## Scenario
NOAA has issued a flood warning for the Houston Ship Channel area. \
USGS gauge 08074000 (Buffalo Bayou at Shepherd Dr) reads 4.2 feet \
above flood stage. Sentinel-1 SAR imagery from this morning shows \
significant backscatter changes in the Ship Channel corridor.

## Your Mission
1. Discover available algorithms
2. Create a flood analysis job for the Houston Ship Channel \
   (use a realistic GeoJSON polygon)
3. Walk the job through EVERY phase — you make each transition
4. At key phases, inject reasoning explaining what you observe \
   and why you are making each decision. Be specific and technical \
   (SAR coherence, backscatter coefficients, gauge readings, etc.)
5. During ANALYZING, adjust parameters based on your observations
6. During QUALITY_CHECK, create an escalation for something you \
   are genuinely uncertain about, then resolve it (simulating the \
   human analyst response)
7. Query the context lakehouse at least once to cross-reference
8. Complete the job and check pipeline metrics

## Style
- Be a real analyst. Specific, technical, decisive.
- Your reasoning entries should read like an expert's field notes.
- Make every decision count — don't just go through the motions.
- Drive the job all the way to COMPLETE/COMPLETE.
"""

# ── Tool Execution ──────────────────────────────────────────────────────


def execute_tool(name: str, inputs: dict) -> dict:
    """Execute a tool call against the live FirstLight API."""
    try:
        if name == "discover_algorithms":
            r = requests.get(f"{API_BASE}/control/v1/tools", headers=HEADERS, timeout=15)
            return r.json()

        elif name == "create_job":
            body = {
                "event_type": inputs["event_type"],
                "aoi": inputs["aoi"],
                "parameters": inputs.get("parameters", {}),
                "reasoning": inputs["reasoning"],
            }
            r = requests.post(
                f"{API_BASE}/control/v1/jobs", headers=HEADERS, json=body, timeout=15
            )
            return r.json()

        elif name == "get_job_status":
            r = requests.get(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}",
                headers=HEADERS,
                timeout=15,
            )
            return r.json()

        elif name == "transition_job":
            body = {
                "expected_phase": inputs["expected_phase"],
                "expected_status": inputs["expected_status"],
                "target_phase": inputs["target_phase"],
                "target_status": inputs["target_status"],
                "reason": inputs["reason"],
            }
            r = requests.post(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}/transition",
                headers=HEADERS,
                json=body,
                timeout=15,
            )
            return r.json()

        elif name == "inject_reasoning":
            body = {
                "reasoning": inputs["reasoning"],
                "confidence": inputs["confidence"],
                "payload": inputs.get("payload", {}),
            }
            r = requests.post(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}/reasoning",
                headers=HEADERS,
                json=body,
                timeout=15,
            )
            return r.json()

        elif name == "adjust_parameters":
            r = requests.patch(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}/parameters",
                headers=HEADERS,
                json=inputs["parameters"],
                timeout=15,
            )
            return r.json()

        elif name == "create_escalation":
            body = {
                "severity": inputs["severity"],
                "reason": inputs["reason"],
                "context": inputs.get("context", {}),
            }
            r = requests.post(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}/escalations",
                headers=HEADERS,
                json=body,
                timeout=15,
            )
            return r.json()

        elif name == "resolve_escalation":
            body = {"resolution": inputs["resolution"]}
            r = requests.patch(
                f"{API_BASE}/control/v1/jobs/{inputs['job_id']}/escalations/{inputs['escalation_id']}",
                headers=HEADERS,
                json=body,
                timeout=15,
            )
            return r.json()

        elif name == "query_context":
            layer = inputs["layer"]
            bbox = inputs["bbox"]
            r = requests.get(
                f"{API_BASE}/control/v1/context/{layer}?bbox={bbox}",
                headers=HEADERS,
                timeout=15,
            )
            return r.json()

        elif name == "get_pipeline_metrics":
            r = requests.get(
                f"{API_BASE}/internal/v1/metrics", headers=HEADERS, timeout=15
            )
            return r.json()

        else:
            return {"error": f"Unknown tool: {name}"}

    except requests.exceptions.ConnectionError:
        return {"error": f"Connection failed — is the API running at {API_BASE}?"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except Exception as e:
        return {"error": str(e)}


# ── Display Helpers ─────────────────────────────────────────────────────


def print_banner():
    """Print the demo banner."""
    print(f"\n{BOLD}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  FirstLight LLM Control Plane — Claude Agent Demo{RESET}")
    print(f"{BOLD}{'═' * 70}{RESET}")
    print(f"  {DIM}API:{RESET}      {API_BASE}")
    print(f"  {DIM}Model:{RESET}    {MODEL}")
    print(f"  {DIM}Scenario:{RESET} Houston Ship Channel Flood Analysis")
    print(f"{BOLD}{'═' * 70}{RESET}\n")


def print_thinking(turn: int):
    """Print the thinking indicator."""
    print(f"\n{DIM}{'─' * 60}{RESET}")
    print(f"  {MAGENTA}{BOLD}CLAUDE{RESET} {DIM}(turn {turn}){RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")


def print_text(text: str):
    """Print Claude's text output."""
    print(f"\n{text}")


def print_tool_call(name: str, inputs: dict):
    """Print a tool call."""
    # Choose color based on tool type
    colors = {
        "discover_algorithms": BLUE,
        "create_job": GREEN,
        "get_job_status": CYAN,
        "transition_job": YELLOW,
        "inject_reasoning": MAGENTA,
        "adjust_parameters": YELLOW,
        "create_escalation": RED,
        "resolve_escalation": GREEN,
        "query_context": CYAN,
        "get_pipeline_metrics": BLUE,
    }
    color = colors.get(name, RESET)

    print(f"\n  {color}{BOLD}▶ {name}{RESET}")

    # Show key inputs concisely
    for key, val in inputs.items():
        val_str = json.dumps(val) if isinstance(val, (dict, list)) else str(val)
        if len(val_str) > 120:
            val_str = val_str[:117] + "..."
        print(f"    {DIM}{key}:{RESET} {val_str}")


def print_tool_result(name: str, result: dict):
    """Print a tool result concisely."""
    result_str = json.dumps(result, indent=2)

    # Truncate long results
    if len(result_str) > 400:
        result_str = result_str[:397] + "..."

    print(f"    {DIM}→ {result_str}{RESET}")


def print_transition(inputs: dict):
    """Print a state transition prominently."""
    frm = f"{inputs['expected_phase']}/{inputs['expected_status']}"
    to = f"{inputs['target_phase']}/{inputs['target_status']}"
    print(f"\n  {YELLOW}{BOLD}  ┌─ STATE TRANSITION ─────────────────────┐{RESET}")
    print(f"  {YELLOW}{BOLD}  │  {frm:>25s}  →  {to:<20s}│{RESET}")
    print(f"  {YELLOW}{BOLD}  └─────────────────────────────────────────┘{RESET}")


def print_completion():
    """Print the demo completion banner."""
    print(f"\n{BOLD}{GREEN}{'═' * 70}{RESET}")
    print(f"{BOLD}{GREEN}  Demo Complete — Full Pipeline Executed{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 70}{RESET}\n")


# ── Main Agent Loop ─────────────────────────────────────────────────────


def main():
    # Verify API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}Error: ANTHROPIC_API_KEY environment variable not set.{RESET}")
        sys.exit(1)

    # Verify API connectivity
    try:
        r = requests.get(f"{API_BASE}/api/v1/health", headers=HEADERS, timeout=5)
        if r.status_code != 200:
            print(f"{RED}Warning: API health check returned {r.status_code}{RESET}")
    except requests.exceptions.ConnectionError:
        print(f"{RED}Error: Cannot connect to {API_BASE}{RESET}")
        print(f"{DIM}Is the API server running?{RESET}")
        sys.exit(1)

    client = anthropic.Anthropic()
    print_banner()

    messages = [
        {
            "role": "user",
            "content": (
                "Begin the flood analysis for the Houston Ship Channel. "
                "Drive the job through every phase of the pipeline."
            ),
        }
    ]

    turn = 0
    start_time = time.time()

    while turn < MAX_TURNS:
        turn += 1
        print_thinking(turn)

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        assistant_content = response.content
        has_tool_use = any(
            block.type == "tool_use" for block in assistant_content
        )

        # Print text blocks
        for block in assistant_content:
            if hasattr(block, "text") and block.text and block.text.strip():
                print_text(block.text)

        # If no tool use, check if done
        if not has_tool_use:
            if response.stop_reason == "end_turn":
                print_completion()
                elapsed = time.time() - start_time
                print(f"  {DIM}Turns: {turn} | Time: {elapsed:.1f}s{RESET}\n")
                break
            continue

        # Process tool calls
        tool_results = []
        for block in assistant_content:
            if block.type == "tool_use":
                # Special display for transitions
                if block.name == "transition_job":
                    print_transition(block.input)

                print_tool_call(block.name, block.input)
                result = execute_tool(block.name, block.input)
                print_tool_result(block.name, result)

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        # Append to conversation
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append({"role": "user", "content": tool_results})

    else:
        print(f"\n  {YELLOW}[Reached {MAX_TURNS} turn limit]{RESET}\n")

    # Print event count from the DB via the API
    print(f"  {DIM}All events were recorded to PostGIS and streamed via SSE.{RESET}")
    print(
        f"  {DIM}View at: {API_BASE}/internal/v1/events/stream{RESET}\n"
    )


if __name__ == "__main__":
    main()
