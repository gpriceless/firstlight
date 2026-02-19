#!/usr/bin/env python3
"""
FirstLight Context Data Lakehouse -- Live Demo
================================================

This script demonstrates the "lakehouse effect": context data accumulated
during one job is automatically reused by subsequent jobs that overlap
spatially. Buildings, infrastructure, weather observations, and satellite
scene metadata are stored once and shared across analyses.

Usage:
    python scripts/demo_lakehouse.py
    python scripts/demo_lakehouse.py --base-url http://localhost:8000
    python scripts/demo_lakehouse.py --dry-run
    python scripts/demo_lakehouse.py --speed fast

Requirements:
    pip install requests
"""

import argparse
import json
import sys
import textwrap
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
        formatted = json.dumps(body, indent=4, default=str)
        for line in formatted.split("\n")[:20]:
            print(Colors.DIM + "    " + line + Colors.RESET)
        if len(formatted.split("\n")) > 20:
            print(Colors.DIM + "    ... (truncated)" + Colors.RESET)


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
        for line in formatted.split("\n")[:15]:
            print(Colors.DIM + "    " + line + Colors.RESET)
        if len(formatted.split("\n")) > 15:
            print(Colors.DIM + "    ... (truncated)" + Colors.RESET)
    print()


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
        "between_steps": 0.3,
        "processing_sim": 0.2,
    },
    "normal": {
        "between_steps": 1.0,
        "processing_sim": 0.5,
    },
    "slow": {
        "between_steps": 2.5,
        "processing_sim": 1.5,
    },
}


# =============================================================================
# Demo AOI Data -- Houston, TX (flood scenario)
# =============================================================================

# Job A: core Houston area (Ship Channel / Buffalo Bayou)
HOUSTON_AOI_A = {
    "type": "Polygon",
    "coordinates": [[
        [-95.45, 29.71],
        [-95.35, 29.71],
        [-95.35, 29.78],
        [-95.45, 29.78],
        [-95.45, 29.71],
    ]],
}

# Job B: overlapping area (shifted east, ~60% overlap with Job A)
HOUSTON_AOI_B = {
    "type": "Polygon",
    "coordinates": [[
        [-95.40, 29.72],
        [-95.30, 29.72],
        [-95.30, 29.79],
        [-95.40, 29.79],
        [-95.40, 29.72],
    ]],
}

# Houston bbox for queries
HOUSTON_BBOX = "-95.50,29.70,-95.25,29.80"


# =============================================================================
# Demo Runner
# =============================================================================

class LakehouseDemo:
    """Orchestrates the context data lakehouse demo."""

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
        self.job_a_id: Optional[str] = None
        self.job_b_id: Optional[str] = None
        self._api_key_raw: Optional[str] = None

    @property
    def _headers(self) -> dict:
        """Standard request headers."""
        h = {"Content-Type": "application/json"}
        if self._api_key_raw:
            h["X-API-Key"] = self._api_key_raw
        elif self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
    ) -> Tuple[int, Any]:
        """Make an HTTP request, or simulate in dry-run mode."""
        url = f"{self.base_url}{path}"
        show_request(method, path, body)

        if self.dry_run:
            info("[DRY RUN] Would send the above request")
            return 200, {"dry_run": True}

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
            return 0, None

    # -------------------------------------------------------------------------
    # Demo Steps
    # -------------------------------------------------------------------------

    def step_banner(self) -> None:
        """Print the opening banner."""
        banner("FirstLight Context Data Lakehouse Demo")
        explain(
            "This demo shows how the lakehouse accumulates context data "
            "(satellite scenes, buildings, infrastructure, weather) across "
            "multiple analysis jobs, enabling automatic reuse when areas "
            "overlap spatially."
        )
        print()
        explain("The demo walks through these stages:")
        info("  1. Setup & API connectivity check")
        info("  2. Submit Job A for a Houston flood zone")
        info("  3. Simulate pipeline context data ingest (118 items)")
        info("  4. Query lakehouse summary")
        info("  5. Submit Job B for an overlapping area")
        info("  6. Simulate Job B -- show data reuse")
        info("  7. Query per-job context usage")
        info("  8. Summary: the lakehouse effect in numbers")
        print()

        if self.dry_run:
            warn("DRY RUN MODE -- No actual HTTP requests will be sent")
        info(f"Target: {self.base_url}")
        print()
        pause(self.speed["between_steps"], "Starting demo")

    def step_1_connectivity(self) -> bool:
        """Step 1: Check connectivity and set up auth."""
        section(1, "Setup & Connectivity")
        explain("Checking that the API is reachable and context endpoints exist.")

        status_code, body = self._request("GET", "/health")
        if status_code == 0 and not self.dry_run:
            error("API is not reachable. Aborting demo.")
            return False
        success("API is reachable")

        pause(self.speed["between_steps"])

        # Create a dev API key for authentication
        explain("Creating a development API key for demo requests.")
        if not self.dry_run:
            try:
                from api.auth import create_development_key
                raw_key, api_key_obj = create_development_key(user_id="demo-lakehouse")
                self._api_key_raw = raw_key
                success(f"Dev API key created (role: {api_key_obj.role})")
            except Exception as e:
                warn(f"Could not create dev key ({e}). Proceeding without auth.")
        else:
            success("[DRY RUN] Would create dev API key")

        pause(self.speed["between_steps"])

        # Verify context endpoints exist
        explain("Verifying context query endpoints...")
        status_code, body = self._request("GET", "/control/v1/context/summary")
        if status_code in (200, 401, 403):
            success("Context endpoints are registered")
        elif status_code == 404:
            error("Context endpoints not found (404). Is the lakehouse deployed?")
            return False
        elif self.dry_run:
            success("[DRY RUN] Would verify context endpoints")

        pause(self.speed["between_steps"])
        return True

    def step_2_create_job_a(self) -> bool:
        """Step 2: Create Job A."""
        section(2, "Create Job A -- Houston Flood Zone")
        explain(
            "Submit a flood detection job covering the Houston Ship Channel "
            "area. This is the first job -- no context data exists yet."
        )

        request_body = {
            "event_type": "flood",
            "aoi": HOUSTON_AOI_A,
            "parameters": {
                "sensitivity": "high",
                "include_sar": True,
            },
            "reasoning": (
                "NOAA flood alert for Houston area. Initiating SAR-based "
                "flood detection for Buffalo Bayou corridor."
            ),
        }

        status_code, body = self._request(
            "POST", "/control/v1/jobs", request_body
        )

        if body and isinstance(body, dict) and "job_id" in body:
            self.job_a_id = body["job_id"]
            success(f"Job A created: {self.job_a_id}")
        elif self.dry_run:
            self.job_a_id = "dry-run-job-a-00000000"
            success("[DRY RUN] Would create Job A")
        else:
            error("Failed to create Job A")
            return False

        pause(self.speed["between_steps"])
        return True

    def step_3_ingest_context(self) -> None:
        """Step 3: Simulate pipeline context data ingest for Job A."""
        section(3, "Simulate Pipeline -- Ingest Context Data for Job A")
        explain(
            "During pipeline execution, the discovery and analysis agents "
            "store context data into the lakehouse. We simulate this by "
            "inserting synthetic data directly via the ContextRepository."
        )

        if self.dry_run:
            success("[DRY RUN] Would insert 100 buildings, 5 infrastructure, 10 weather, 3 datasets")
            pause(self.speed["between_steps"])
            return

        explain("Inserting context data via ContextRepository...")
        try:
            import asyncio
            asyncio.run(self._ingest_context_async(self.job_a_id, is_first_job=True))
        except Exception as e:
            error(f"Context ingest failed: {e}")
            warn("Demo will continue but reuse stats may be empty.")

        pause(self.speed["between_steps"])

    async def _ingest_context_async(
        self, job_id: str, is_first_job: bool = True
    ) -> None:
        """Async helper to ingest synthetic context data."""
        from uuid import UUID

        from core.context.repository import ContextRepository
        from core.context.models import DatasetRecord
        from core.context.stubs import (
            generate_buildings,
            generate_infrastructure,
            generate_weather,
        )
        from api.config import get_settings

        settings = get_settings()
        db = settings.database
        repo = ContextRepository(
            host=db.host,
            port=db.port,
            database=db.name,
            user=db.user,
            password=db.password,
        )
        await repo.connect()

        try:
            job_uuid = UUID(job_id)
            bbox = (-95.45, 29.71, -95.35, 29.78)

            # Buildings (100)
            buildings = generate_buildings(bbox, count=100, seed=42)
            results = await repo.store_batch(
                job_uuid, "context_buildings", buildings
            )
            ingested = sum(1 for r in results if r.usage_type == "ingested")
            reused = sum(1 for r in results if r.usage_type == "reused")
            success(f"Buildings: {ingested} ingested, {reused} reused")

            # Infrastructure (5)
            infra = generate_infrastructure(bbox, count=5, seed=42)
            results = await repo.store_batch(
                job_uuid, "context_infrastructure", infra
            )
            ingested = sum(1 for r in results if r.usage_type == "ingested")
            reused = sum(1 for r in results if r.usage_type == "reused")
            success(f"Infrastructure: {ingested} ingested, {reused} reused")

            # Weather (10)
            weather = generate_weather(bbox, count=10, seed=42)
            results = await repo.store_batch(
                job_uuid, "context_weather", weather
            )
            ingested = sum(1 for r in results if r.usage_type == "ingested")
            reused = sum(1 for r in results if r.usage_type == "reused")
            success(f"Weather: {ingested} ingested, {reused} reused")

            # Datasets / satellite scenes (3)
            from datetime import timedelta
            import random
            rng = random.Random(42)
            for i in range(3):
                record = DatasetRecord(
                    source="earth_search",
                    source_id=f"S2A_MSIL2A_2026021{i}_T15RUN",
                    geometry={
                        "type": "Polygon",
                        "coordinates": [[
                            [-95.50, 29.65],
                            [-95.25, 29.65],
                            [-95.25, 29.85],
                            [-95.50, 29.85],
                            [-95.50, 29.65],
                        ]],
                    },
                    properties={
                        "platform": "sentinel-2a",
                        "constellation": "sentinel-2",
                    },
                    acquisition_date=datetime(2026, 2, 10 + i, tzinfo=timezone.utc),
                    cloud_cover=rng.uniform(5, 30),
                    resolution_m=10.0,
                    bands=["B02", "B03", "B04", "B08"],
                )
                result = await repo.store_dataset(job_uuid, record)
                info(f"  Scene {record.source_id}: {result.usage_type}")

            success("All context data for Job A ingested (118 items total)")

        finally:
            await repo.close()

    def step_4_query_summary(self) -> None:
        """Step 4: Query lakehouse summary."""
        section(4, "Query Lakehouse Summary")
        explain(
            "After Job A's pipeline, the lakehouse has accumulated context "
            "data. Let's see the overall statistics."
        )

        status_code, body = self._request(
            "GET", "/control/v1/context/summary"
        )

        if status_code == 200 and body:
            total = body.get("total_rows", "?")
            success(f"Lakehouse contains {total} context rows")

            tables = body.get("tables", {})
            for label, info_data in tables.items():
                if isinstance(info_data, dict):
                    count = info_data.get("row_count", 0)
                    sources = info_data.get("sources", [])
                    info(f"  {label}: {count} rows (sources: {', '.join(sources) or 'none'})")

            usage = body.get("usage_stats", {})
            if usage:
                info(f"  Usage: ingested={usage.get('ingested', 0)}, reused={usage.get('reused', 0)}")
        elif self.dry_run:
            success("[DRY RUN] Would show lakehouse statistics")

        pause(self.speed["between_steps"])

        # Also try querying buildings with bbox
        explain("Querying buildings within the Houston bbox:")
        status_code, body = self._request(
            "GET", f"/control/v1/context/buildings?bbox={HOUSTON_BBOX}&page_size=5"
        )
        if status_code == 200 and body:
            total = body.get("total", 0)
            items = body.get("items", [])
            success(f"Found {total} buildings in Houston bbox (showing first {len(items)})")
            for item in items[:3]:
                info(f"  {item.get('source_id', '?')}: {item.get('properties', {}).get('type', 'unknown')}")
        elif self.dry_run:
            success("[DRY RUN] Would query buildings")

        pause(self.speed["between_steps"])

    def step_5_create_job_b(self) -> bool:
        """Step 5: Create Job B with overlapping AOI."""
        section(5, "Create Job B -- Overlapping Area")
        explain(
            "Submit a second flood detection job that overlaps ~60% with "
            "Job A's coverage area. The lakehouse should recognize that "
            "most context data already exists."
        )

        request_body = {
            "event_type": "flood",
            "aoi": HOUSTON_AOI_B,
            "parameters": {
                "sensitivity": "high",
                "include_sar": True,
            },
            "reasoning": (
                "Follow-up flood analysis for eastern Houston. Overlaps with "
                "previous analysis area to provide continuity coverage."
            ),
        }

        status_code, body = self._request(
            "POST", "/control/v1/jobs", request_body
        )

        if body and isinstance(body, dict) and "job_id" in body:
            self.job_b_id = body["job_id"]
            success(f"Job B created: {self.job_b_id}")
        elif self.dry_run:
            self.job_b_id = "dry-run-job-b-00000000"
            success("[DRY RUN] Would create Job B")
        else:
            error("Failed to create Job B")
            return False

        pause(self.speed["between_steps"])
        return True

    def step_6_ingest_with_reuse(self) -> None:
        """Step 6: Simulate Job B pipeline -- show reuse."""
        section(6, "Simulate Pipeline -- Job B Reuses Existing Context")
        explain(
            "When Job B's pipeline runs, it attempts to store the same "
            "context data. Because these buildings, infrastructure, and "
            "weather stations already exist (same source + source_id), "
            "the lakehouse links them with usage_type='reused' instead of "
            "inserting duplicates."
        )

        if self.dry_run:
            success("[DRY RUN] Would demonstrate data reuse")
            info("  Expected: ~60% of buildings reused, all infra reused, etc.")
            pause(self.speed["between_steps"])
            return

        explain("Inserting the same context data for Job B...")
        try:
            import asyncio
            asyncio.run(self._ingest_context_async(self.job_b_id, is_first_job=False))
        except Exception as e:
            error(f"Context ingest for Job B failed: {e}")

        pause(self.speed["between_steps"])

    def step_7_query_job_context(self) -> None:
        """Step 7: Query per-job context usage."""
        section(7, "Per-Job Context Usage")
        explain(
            "Query the context usage summary for each job to see the "
            "lakehouse effect in action."
        )

        # Job A context
        if self.job_a_id:
            explain(f"Job A context usage:")
            status_code, body = self._request(
                "GET", f"/control/v1/jobs/{self.job_a_id}/context"
            )
            if status_code == 200 and body:
                total_i = body.get("total_ingested", 0)
                total_r = body.get("total_reused", 0)
                success(
                    f"Job A: {total_i} ingested, {total_r} reused "
                    f"(total: {body.get('total', 0)})"
                )
            elif self.dry_run:
                success("[DRY RUN] Would show Job A context (all ingested)")

        pause(self.speed["between_steps"])

        # Job B context
        if self.job_b_id:
            explain(f"Job B context usage:")
            status_code, body = self._request(
                "GET", f"/control/v1/jobs/{self.job_b_id}/context"
            )
            if status_code == 200 and body:
                total_i = body.get("total_ingested", 0)
                total_r = body.get("total_reused", 0)
                success(
                    f"Job B: {total_i} ingested, {total_r} reused "
                    f"(total: {body.get('total', 0)})"
                )
            elif self.dry_run:
                success("[DRY RUN] Would show Job B context (mostly reused)")

        pause(self.speed["between_steps"])

    def step_summary(self) -> None:
        """Step 8: Print final summary."""
        banner("Demo Complete -- The Lakehouse Effect")

        explain("Here is what we demonstrated:")
        print()
        info("  1. Job A analyzed a Houston flood zone")
        info("  2. The pipeline accumulated 118 context data items:")
        info("     - 100 building footprints")
        info("     - 5 infrastructure facilities")
        info("     - 10 weather observations")
        info("     - 3 satellite scenes")
        info("  3. Job B overlapped ~60% with Job A's area")
        info("  4. When Job B ran, it REUSED existing context data")
        info("     instead of re-fetching it from external sources")
        print()

        explain("The lakehouse effect means:")
        print()
        info("  - No duplicate data: each building/scene stored ONCE")
        info("  - Faster processing: skip re-fetching known data")
        info("  - Full provenance: junction table tracks which job used what")
        info("  - Spatial intelligence: data accumulates over time and space")
        print()

        explain("Context API Endpoints:")
        print()
        endpoints = [
            ("GET", "/control/v1/context/datasets", "Query satellite scenes"),
            ("GET", "/control/v1/context/buildings", "Query building footprints"),
            ("GET", "/control/v1/context/infrastructure", "Query infrastructure"),
            ("GET", "/control/v1/context/weather", "Query weather observations"),
            ("GET", "/control/v1/context/summary", "Lakehouse statistics"),
            ("GET", "/control/v1/jobs/{id}/context", "Per-job context usage"),
        ]
        for method, path, desc in endpoints:
            print(
                f"    {Colors.GREEN}{method:6s}{Colors.RESET} "
                f"{Colors.WHITE}{path:45s}{Colors.RESET} "
                f"{Colors.DIM}{desc}{Colors.RESET}"
            )

        print()
        if self.job_a_id:
            info(f"  Job A ID: {self.job_a_id}")
        if self.job_b_id:
            info(f"  Job B ID: {self.job_b_id}")
        print()
        success("End of lakehouse demo.")
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

            if not self.step_2_create_job_a():
                error("Job A creation failed. Cannot continue.")
                return

            self.step_3_ingest_context()
            self.step_4_query_summary()

            if not self.step_5_create_job_b():
                error("Job B creation failed. Cannot continue.")
                return

            self.step_6_ingest_with_reuse()
            self.step_7_query_job_context()
            self.step_summary()

        except KeyboardInterrupt:
            print()
            warn("Demo interrupted by user")


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="FirstLight Context Data Lakehouse Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/demo_lakehouse.py
              python scripts/demo_lakehouse.py --dry-run
              python scripts/demo_lakehouse.py --base-url http://api.firstlight.dev:8000
              python scripts/demo_lakehouse.py --speed fast
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
        help="API key for authentication",
    )
    parser.add_argument(
        "--speed",
        choices=["fast", "normal", "slow"],
        default="normal",
        help="Demo pacing: fast, normal, slow",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without making any API calls",
    )

    args = parser.parse_args()

    demo = LakehouseDemo(
        base_url=args.base_url,
        api_key=args.api_key,
        speed=args.speed,
        dry_run=args.dry_run,
    )
    demo.run()


if __name__ == "__main__":
    main()
