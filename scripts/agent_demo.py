#!/usr/bin/env python3
"""
FirstLight Control Plane — Agent Demo.

Simulates an LLM agent driving a flood analysis job through the full
pipeline, making every API call against the live control plane. Each
step shows what the agent "thinks", then executes the real API call.

This demonstrates the full control plane surface:
  - Algorithm discovery (LLM tool router)
  - Job creation with reasoning
  - Manual state transitions through all 7 phases
  - Reasoning injection with confidence scores
  - Mid-flight parameter tuning
  - Human-in-the-loop escalation + resolution
  - Context lakehouse spatial queries
  - Pipeline health metrics

Usage:
    python scripts/agent_demo.py

Optional env vars:
    FIRSTLIGHT_API_URL  — default: http://localhost:8000
    FIRSTLIGHT_API_KEY  — default: demo-18254ee7d18f5926
"""

import json
import os
import sys
import time

import requests

# ── Config ──────────────────────────────────────────────────────────────

API_BASE = os.environ.get("FIRSTLIGHT_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.environ.get("FIRSTLIGHT_API_KEY", "demo-18254ee7d18f5926")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ── ANSI ────────────────────────────────────────────────────────────────

B = "\033[1m"
D = "\033[2m"
R = "\033[0m"
CY = "\033[36m"
GR = "\033[32m"
YL = "\033[33m"
RD = "\033[31m"
MG = "\033[35m"
BL = "\033[34m"

# ── Helpers ─────────────────────────────────────────────────────────────

JOB_ID = None  # set after create


def banner():
    print(f"\n{B}{'═' * 70}{R}")
    print(f"{B}{CY}  FirstLight LLM Control Plane — Agent Demo{R}")
    print(f"{B}{'═' * 70}{R}")
    print(f"  {D}API:{R}      {API_BASE}")
    print(f"  {D}Scenario:{R} Houston Ship Channel Flood Analysis")
    print(f"  {D}Agent:{R}    Simulated LLM analyst (live API calls)")
    print(f"{B}{'═' * 70}{R}\n")


def think(text):
    """Print agent reasoning."""
    print(f"\n  {MG}{B}AGENT REASONING:{R}")
    for line in text.strip().split("\n"):
        print(f"  {MG}{line.strip()}{R}")


def action(label):
    """Print action header."""
    print(f"\n  {BL}{B}▶ {label}{R}")


def result(data, max_len=400):
    """Print API result."""
    s = json.dumps(data, indent=2)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    print(f"  {D}→ {s}{R}")


def transition_box(frm, to):
    """Print a state transition box."""
    print(f"\n  {YL}{B}  ┌─ STATE TRANSITION ─────────────────────────┐{R}")
    print(f"  {YL}{B}  │  {frm:>28s}  →  {to:<16s}│{R}")
    print(f"  {YL}{B}  └─────────────────────────────────────────────┘{R}")


def phase_header(num, name):
    """Print a phase separator."""
    print(f"\n{B}{'─' * 70}{R}")
    print(f"  {B}{GR}Phase {num}: {name}{R}")
    print(f"{B}{'─' * 70}{R}")


def api_get(path):
    r = requests.get(f"{API_BASE}{path}", headers=HEADERS, timeout=15)
    return r.json()


def api_post(path, body=None):
    r = requests.post(f"{API_BASE}{path}", headers=HEADERS, json=body or {}, timeout=15)
    return r.json()


def api_patch(path, body):
    r = requests.patch(f"{API_BASE}{path}", headers=HEADERS, json=body, timeout=15)
    return r.json()


def do_transition(expected_phase, expected_status, target_phase, target_status, reason):
    """Execute a state transition against the live API."""
    transition_box(
        f"{expected_phase}/{expected_status}",
        f"{target_phase}/{target_status}",
    )
    action(f"POST /control/v1/jobs/{{id}}/transition")
    resp = api_post(f"/control/v1/jobs/{JOB_ID}/transition", {
        "expected_phase": expected_phase,
        "expected_status": expected_status,
        "target_phase": target_phase,
        "target_status": target_status,
        "reason": reason,
    })
    result(resp)
    if "error" in str(resp).lower() and "conflict" in str(resp).lower():
        print(f"  {RD}Transition conflict — checking current state...{R}")
        result(api_get(f"/control/v1/jobs/{JOB_ID}"))
    return resp


def do_reasoning(reasoning, confidence, payload=None):
    """Inject a reasoning entry."""
    action(f"POST /control/v1/jobs/{{id}}/reasoning")
    resp = api_post(f"/control/v1/jobs/{JOB_ID}/reasoning", {
        "reasoning": reasoning,
        "confidence": confidence,
        "payload": payload or {},
    })
    result(resp)
    return resp


# ── Demo Execution ──────────────────────────────────────────────────────


def main():
    global JOB_ID

    # Connectivity check
    try:
        h = api_get("/api/v1/health")
        if h.get("status") != "healthy":
            print(f"{RD}API not healthy: {h}{R}")
            sys.exit(1)
    except Exception as e:
        print(f"{RD}Cannot connect to {API_BASE}: {e}{R}")
        sys.exit(1)

    banner()
    start = time.time()

    # ================================================================
    # Step 1: Algorithm Discovery
    # ================================================================

    phase_header(0, "ALGORITHM DISCOVERY")

    think("""
        Starting flood analysis workflow. First, I need to discover what
        algorithms this platform supports and their parameter schemas.
        This tells me what tools I can use and how to configure them.
    """)

    action("GET /control/v1/tools")
    tools = api_get("/control/v1/tools")
    result(tools)

    tool_count = len(tools) if isinstance(tools, list) else tools.get("total", "?")
    think(f"""
        Platform reports {tool_count} available algorithms. I can see flood
        detection, SAR analysis, and spectral classifiers in the registry.
        Each tool exposes an OpenAI function-calling schema — I can invoke
        them natively through my tool_use interface.
    """)

    time.sleep(0.5)

    # ================================================================
    # Step 2: Job Creation
    # ================================================================

    phase_header(1, "JOB CREATION")

    think("""
        NOAA has issued a flood warning for the Houston Ship Channel.
        USGS gauge 08074000 (Buffalo Bayou at Shepherd Dr) reads 4.2ft
        above flood stage. I'm creating a flood analysis job covering
        the Ship Channel corridor from Galveston Bay to downtown Houston.

        Initial parameters: medium sensitivity, SAR+optical fusion,
        minimum 10m resolution. I'll adjust these once I see the data.
    """)

    action("POST /control/v1/jobs")
    aoi = {
        "type": "Polygon",
        "coordinates": [[
            [-95.3632, 29.7012],
            [-95.3632, 29.7802],
            [-95.0514, 29.7802],
            [-95.0514, 29.7012],
            [-94.8946, 29.6142],
            [-94.7853, 29.5638],
            [-94.7853, 29.5218],
            [-94.9452, 29.5218],
            [-95.0514, 29.5692],
            [-95.2108, 29.6422],
            [-95.3632, 29.7012],
        ]]
    }
    job = api_post("/control/v1/jobs", {
        "event_type": "flood",
        "aoi": aoi,
        "parameters": {
            "sensitivity": "medium",
            "include_sar": True,
            "include_optical": True,
            "min_resolution_m": 10,
        },
        "reasoning": (
            "NOAA Flood Warning WEA-2026-0451 issued for Houston Ship Channel. "
            "USGS gauge 08074000 at Buffalo Bayou reads 31.2ft (4.2ft above "
            "flood stage of 27.0ft). Sentinel-1 SAR pass at 06:14 UTC shows "
            "significant backscatter anomaly in the Ship Channel corridor. "
            "Initiating rapid flood extent mapping for emergency response."
        ),
    })
    result(job)

    JOB_ID = job.get("job_id")
    if not JOB_ID:
        print(f"{RD}Job creation failed: {job}{R}")
        sys.exit(1)

    print(f"\n  {GR}{B}Job created: {JOB_ID}{R}")

    time.sleep(0.5)

    # ================================================================
    # Step 3: QUEUED — Validation
    # ================================================================

    phase_header(2, "QUEUED — Validation")

    think("""
        Job is in QUEUED/PENDING. I need to validate the inputs before
        proceeding. Checking AOI geometry validity, coordinate bounds,
        and parameter compatibility with the flood detection algorithms.
    """)

    do_transition("QUEUED", "PENDING", "QUEUED", "VALIDATING",
                  "Validating AOI geometry and flood detection parameters")

    do_reasoning(
        "AOI covers approximately 250 km² of the Houston Ship Channel "
        "corridor from Galveston Bay (29.52°N) to downtown Houston "
        "(29.78°N). Polygon has 11 vertices, all within valid WGS84 bounds. "
        "Parameters are compatible with the SAR flood detection algorithm. "
        "Sentinel-1 IW mode provides 10m resolution — meets min_resolution_m "
        "constraint. Validation passed.",
        confidence=0.95,
        payload={
            "aoi_area_km2": 248.7,
            "aoi_vertex_count": 11,
            "crs": "EPSG:4326",
            "resolution_check": "PASS",
        }
    )

    do_transition("QUEUED", "VALIDATING", "QUEUED", "VALIDATED",
                  "AOI and parameters validated successfully")

    time.sleep(0.3)

    # ================================================================
    # Step 4: DISCOVERING — Catalog Search
    # ================================================================

    phase_header(3, "DISCOVERING — Satellite Data Search")

    think("""
        Validation passed. Moving to discovery phase. I need to search
        STAC catalogs for Sentinel-1 SAR imagery over the AOI from the
        last 48 hours. Also looking for Sentinel-2 optical if cloud
        cover permits — SAR/optical fusion improves flood classification.
    """)

    do_transition("QUEUED", "VALIDATED", "DISCOVERING", "DISCOVERING",
                  "Searching STAC catalogs for Sentinel-1/2 imagery over Houston Ship Channel")

    do_reasoning(
        "STAC catalog search initiated against Copernicus Data Space. "
        "Query: Sentinel-1 IW GRD, last 48h, intersecting AOI. "
        "Found 3 SAR scenes: S1A descending pass (06:14 UTC today), "
        "S1B ascending pass (18:02 UTC yesterday), S1A descending "
        "(06:14 UTC yesterday — pre-event baseline). "
        "Sentinel-2 search: 1 scene from yesterday but 78% cloud cover "
        "over AOI — will use SAR-only workflow. "
        "Pre-event baseline available — enables coherence change detection.",
        confidence=0.92,
        payload={
            "scenes_found": {
                "sentinel_1_sar": 3,
                "sentinel_2_optical": 1,
            },
            "sar_scenes": [
                {"id": "S1A_IW_GRDH_20260219T0614", "orbit": "descending", "age_hours": 13},
                {"id": "S1B_IW_GRDH_20260218T1802", "orbit": "ascending", "age_hours": 25},
                {"id": "S1A_IW_GRDH_20260218T0614", "orbit": "descending", "age_hours": 37},
            ],
            "optical_rejected": "78% cloud cover exceeds 30% threshold",
            "workflow": "SAR-only with coherence change detection",
        }
    )

    do_transition("DISCOVERING", "DISCOVERING", "DISCOVERING", "DISCOVERED",
                  "Found 3 Sentinel-1 SAR scenes, 1 pre-event baseline. Optical rejected (78% cloud).")

    time.sleep(0.3)

    # ================================================================
    # Step 5: INGESTING — Download and Stage
    # ================================================================

    phase_header(4, "INGESTING — Download and Stage Imagery")

    think("""
        Discovery complete. 3 SAR scenes identified. Downloading GRD
        products from Copernicus Data Space. The pre-event scene from
        yesterday gives me a baseline for coherence change detection —
        that's the strongest signal for flood mapping with SAR.
    """)

    do_transition("DISCOVERING", "DISCOVERED", "INGESTING", "INGESTING",
                  "Downloading 3 Sentinel-1 GRD scenes from Copernicus Data Space")

    do_reasoning(
        "Ingestion progress: 3 scenes queued for download. "
        "S1A_20260219 (today's pass): 1.2 GB, downloading... "
        "S1B_20260218 (yesterday ascending): 1.1 GB, downloading... "
        "S1A_20260218 (pre-event baseline): 1.2 GB, downloading... "
        "Total: 3.5 GB. Checksums will be verified against ESA manifests. "
        "Estimated COG conversion: 45 seconds per scene.",
        confidence=0.88,
        payload={
            "total_download_gb": 3.5,
            "scenes_downloading": 3,
            "format": "GeoTIFF → COG",
        }
    )

    do_transition("INGESTING", "INGESTING", "INGESTING", "INGESTED",
                  "All 3 scenes downloaded, checksums verified, COG conversion complete")

    time.sleep(0.3)

    # ================================================================
    # Step 6: NORMALIZING — CRS Reprojection, Calibration
    # ================================================================

    phase_header(5, "NORMALIZING — CRS Reprojection and Calibration")

    think("""
        Imagery staged. Now I need to co-register the scenes, reproject
        to UTM Zone 15N (EPSG:32615 — covers Houston), and apply
        radiometric terrain correction. The ascending and descending
        orbits have different look angles — I need to account for that
        in the calibration to make the backscatter values comparable.
    """)

    do_transition("INGESTING", "INGESTED", "NORMALIZING", "NORMALIZING",
                  "Reprojecting to EPSG:32615, radiometric terrain correction, co-registration")

    do_reasoning(
        "Normalization pipeline: "
        "1) CRS reprojection: WGS84 Geographic → UTM 15N (EPSG:32615). "
        "2) Radiometric calibration: DN → sigma-nought (σ⁰) in dB. "
        "3) Terrain correction: SRTM 30m DEM applied (flat terrain in "
        "Ship Channel — minimal geometric distortion expected). "
        "4) Co-registration: ascending/descending orbit alignment via "
        "cross-correlation on stable targets (bridges, port infrastructure). "
        "RMS co-registration error: 0.3 pixels — within tolerance.",
        confidence=0.90,
        payload={
            "target_crs": "EPSG:32615",
            "calibration": "sigma-nought (dB)",
            "dem": "SRTM 30m",
            "coregistration_rmse_px": 0.3,
        }
    )

    do_transition("NORMALIZING", "NORMALIZING", "NORMALIZING", "NORMALIZED",
                  "Scenes co-registered, calibrated to σ⁰, reprojected to UTM 15N")

    time.sleep(0.3)

    # ================================================================
    # Step 7: ANALYZING — Flood Detection
    # ================================================================

    phase_header(6, "ANALYZING — Flood Detection")

    think("""
        Data is normalized. Running SAR flood detection now. I'll use
        coherence change detection between today's scene and yesterday's
        baseline, plus backscatter thresholding on the VV polarization.
        The Ship Channel has strong scatterers (cranes, tanks) that
        give me good reference points.
    """)

    do_transition("NORMALIZING", "NORMALIZED", "ANALYZING", "ANALYZING",
                  "Running SAR coherence change detection and backscatter thresholding")

    do_reasoning(
        "Flood detection results (preliminary): "
        "Coherence change detection (today vs. pre-event baseline): "
        "Mean coherence dropped from 0.82 to 0.31 in the Buffalo Bayou "
        "corridor (29.72-29.76°N). This is a strong flood signal — "
        "water surfaces decorrelate SAR coherence. "
        "VV backscatter thresholding (Otsu method): σ⁰ threshold = -14.2 dB. "
        "Pixels below threshold classified as open water. "
        "Detected flood extent: ~47 km² of inundation, concentrated "
        "along Buffalo Bayou from Shepherd Dr to the San Jacinto confluence. "
        "Double-bounce signature detected in residential areas east of "
        "I-610 — indicates urban flooding (water between buildings).",
        confidence=0.85,
        payload={
            "coherence_pre": 0.82,
            "coherence_post": 0.31,
            "coherence_drop": 0.51,
            "backscatter_threshold_db": -14.2,
            "flood_extent_km2": 47.3,
            "double_bounce_detected": True,
            "affected_corridor": "Buffalo Bayou: Shepherd Dr → San Jacinto River",
        }
    )

    # Parameter adjustment based on findings
    think("""
        The coherence drop is very strong (0.51) and the flood extent
        is larger than expected. I'm increasing sensitivity to HIGH to
        capture the full extent including shallow inundation at the margins.
        Also lowering the backscatter threshold to catch partially
        submerged vegetation along the bayou banks.
    """)

    action("PATCH /control/v1/jobs/{id}/parameters")
    params = api_patch(f"/control/v1/jobs/{JOB_ID}/parameters", {
        "sensitivity": "high",
        "backscatter_threshold_db": -15.5,
        "include_partial_inundation": True,
        "double_bounce_detection": True,
    })
    result(params)

    do_reasoning(
        "Parameters adjusted: sensitivity HIGH, backscatter threshold "
        "lowered to -15.5 dB (from -14.2), enabled partial inundation "
        "and double-bounce detection. Re-running classification with "
        "updated parameters. "
        "Updated flood extent: ~58 km² (22% increase from parameter tuning). "
        "The additional 11 km² is mostly shallow inundation along "
        "tributary channels and low-lying residential areas.",
        confidence=0.87,
        payload={
            "previous_extent_km2": 47.3,
            "updated_extent_km2": 58.1,
            "increase_pct": 22.8,
            "new_threshold_db": -15.5,
        }
    )

    time.sleep(0.3)

    # ================================================================
    # Step 8: QUALITY CHECK — with Escalation
    # ================================================================

    phase_header(7, "ANALYZING — Quality Check + Escalation")

    do_transition("ANALYZING", "ANALYZING", "ANALYZING", "QUALITY_CHECK",
                  "Running quality checks on flood detection output")

    think("""
        Quality check in progress. Most of the flood extent looks solid —
        strong coherence drop, clean backscatter signature. But there's
        an anomalous region in grid cell H-7 near the Deer Park
        industrial complex. The backscatter is low but the coherence
        is still moderate (0.58). Could be:
          1) Real flood extent (water + oil film altering backscatter)
          2) Industrial surface change (new pavement, equipment)
          3) SAR processing artifact from the tank farm sidelobes

        I'm not confident enough to classify this. Escalating.
    """)

    # Create escalation
    action("POST /control/v1/jobs/{id}/escalations")
    esc = api_post(f"/control/v1/jobs/{JOB_ID}/escalations", {
        "severity": "HIGH",
        "reason": (
            "Anomalous SAR signature in grid cell H-7 near Deer Park "
            "industrial complex (29.715°N, -95.118°W). Backscatter "
            "is -16.3 dB (below flood threshold) but coherence remains "
            "moderate at 0.58 (expected <0.35 for water). Possible "
            "explanations: (1) flood with oil film, (2) industrial "
            "surface change, (3) tank farm sidelobe artifact. "
            "Confidence 0.52 — below 0.75 auto-classification threshold. "
            "Human review requested."
        ),
        "context": {
            "grid_cell": "H-7",
            "location": {"lat": 29.715, "lon": -95.118},
            "backscatter_db": -16.3,
            "coherence": 0.58,
            "expected_coherence": "<0.35",
            "confidence": 0.52,
            "possible_causes": [
                "flood_with_oil_film",
                "industrial_surface_change",
                "tank_farm_sidelobe",
            ],
        },
    })
    result(esc)

    esc_id = esc.get("escalation_id")
    print(f"\n  {RD}{B}Escalation created: {esc_id}{R}")

    time.sleep(1)

    # Simulate human resolution
    think("""
        [HUMAN ANALYST RESPONSE]
        Reviewed Sentinel-1 imagery for grid cell H-7. Cross-referenced
        with Planet SkySat tasking from 14:30 UTC — optical confirmation
        shows standing water in the Deer Park industrial staging area.
        The moderate coherence is explained by partially submerged
        shipping containers acting as corner reflectors. This is genuine
        flood extent, not an artifact. Confidence upgraded.
    """)

    if esc_id:
        action(f"PATCH /control/v1/jobs/{{id}}/escalations/{esc_id}")
        resolution = api_patch(f"/control/v1/jobs/{JOB_ID}/escalations/{esc_id}", {
            "resolution": (
                "Confirmed genuine flood extent in grid cell H-7. "
                "Planet SkySat optical imagery (14:30 UTC) shows standing "
                "water in Deer Park industrial staging area. Moderate "
                "coherence (0.58) explained by partially submerged shipping "
                "containers acting as corner reflectors (double-bounce). "
                "Reclassify as FLOOD with confidence 0.88. "
                "— Reviewed by Senior Analyst, Houston Emergency Ops"
            ),
        })
        result(resolution)
        print(f"\n  {GR}{B}Escalation resolved{R}")

    do_reasoning(
        "Quality check complete. Escalation for grid cell H-7 resolved: "
        "confirmed flood extent (optical cross-reference). Updated "
        "classification incorporates the Deer Park area. "
        "Final flood extent: 61.4 km². "
        "Quality metrics: 94.2% classification confidence (weighted avg), "
        "3 false-positive candidates reviewed (2 confirmed, 1 reclassified). "
        "No false negatives detected in spot-check sample.",
        confidence=0.91,
        payload={
            "final_flood_extent_km2": 61.4,
            "classification_confidence_pct": 94.2,
            "escalations_created": 1,
            "escalations_resolved": 1,
            "false_positives_reviewed": 3,
        }
    )

    do_transition("ANALYZING", "QUALITY_CHECK", "ANALYZING", "ANALYZED",
                  "Quality checks passed — flood extent finalized at 61.4 km²")

    time.sleep(0.3)

    # ================================================================
    # Step 9: Context Lakehouse Query
    # ================================================================

    phase_header(8, "CONTEXT QUERY — Impact Assessment")

    think("""
        Analysis is complete. Before generating the report, I need
        to cross-reference the flood extent with the context lakehouse
        to assess impact: buildings affected, critical infrastructure
        at risk, population in the inundation zone.
    """)

    bbox = "-95.37,29.52,-94.78,29.79"

    action(f"GET /control/v1/context/buildings?bbox={bbox}")
    buildings = api_get(f"/control/v1/context/buildings?bbox={bbox}")
    result(buildings)

    action(f"GET /control/v1/context/infrastructure?bbox={bbox}")
    infra = api_get(f"/control/v1/context/infrastructure?bbox={bbox}")
    result(infra)

    action(f"GET /control/v1/context/weather?bbox={bbox}")
    weather = api_get(f"/control/v1/context/weather?bbox={bbox}")
    result(weather)

    do_reasoning(
        "Context lakehouse query results for flood impact assessment: "
        "Queried buildings, infrastructure, and weather layers within "
        "the AOI bounding box. The lakehouse accumulates context data "
        "across jobs — previous analyses in this area would have "
        "pre-populated building footprints and infrastructure locations, "
        "enabling faster impact assessment (the lakehouse effect). "
        "This cross-referencing capability is what transforms raw flood "
        "extent into actionable intelligence: not just 'where is the "
        "water' but 'what is the water affecting.'",
        confidence=0.88,
        payload={
            "layers_queried": ["buildings", "infrastructure", "weather"],
            "bbox": bbox,
        }
    )

    time.sleep(0.3)

    # ================================================================
    # Step 10: REPORTING — Generate Products
    # ================================================================

    phase_header(9, "REPORTING — Product Generation")

    think("""
        Moving to reporting phase. Assembling the final deliverables:
        - GeoTIFF flood extent raster (binary mask + confidence layer)
        - GeoJSON flood boundary polygons
        - Impact assessment summary (buildings, infrastructure, population)
        - STAC Item for catalog publication
    """)

    do_transition("ANALYZING", "ANALYZED", "REPORTING", "REPORTING",
                  "Generating flood analysis products and impact report")

    do_transition("REPORTING", "REPORTING", "REPORTING", "ASSEMBLING",
                  "Assembling final deliverables: GeoTIFF, GeoJSON, PDF report, STAC Item")

    do_reasoning(
        "Final report assembly: "
        "1) Flood extent raster: 10m resolution binary mask + confidence "
        "layer (GeoTIFF, COG format, EPSG:32615). "
        "2) Flood boundary vectors: simplified polygons at 20m tolerance "
        "(GeoJSON, WGS84). 3 connected components: Buffalo Bayou main "
        "channel (42.1 km²), Brays Bayou tributary (11.8 km²), "
        "Deer Park industrial (7.5 km²). "
        "3) Impact summary: flood extent 61.4 km², estimated buildings "
        "in inundation zone based on context lakehouse data. "
        "4) STAC Item: publishing to /stac/collections/flood for "
        "downstream discovery by GIS clients and other AI agents.",
        confidence=0.93,
        payload={
            "products": [
                "flood_extent_10m.tif (COG, EPSG:32615)",
                "flood_boundary.geojson (WGS84)",
                "impact_report.pdf",
                "stac_item.json",
            ],
            "flood_components": [
                {"name": "Buffalo Bayou main", "area_km2": 42.1},
                {"name": "Brays Bayou tributary", "area_km2": 11.8},
                {"name": "Deer Park industrial", "area_km2": 7.5},
            ],
            "total_flood_km2": 61.4,
        }
    )

    do_transition("REPORTING", "ASSEMBLING", "REPORTING", "REPORTED",
                  "All products assembled and validated")

    time.sleep(0.3)

    # ================================================================
    # Step 11: COMPLETE
    # ================================================================

    phase_header(10, "COMPLETE")

    do_transition("REPORTING", "REPORTED", "COMPLETE", "COMPLETE",
                  "Job complete — Houston Ship Channel flood analysis delivered")

    # Final status check
    action("GET /control/v1/jobs/{id}")
    final = api_get(f"/control/v1/jobs/{JOB_ID}")
    result(final, max_len=600)

    time.sleep(0.3)

    # ================================================================
    # Step 12: Pipeline Metrics
    # ================================================================

    phase_header(11, "PIPELINE METRICS")

    think("""
        Job complete. Checking pipeline health metrics to confirm
        the system is operating normally and this job's stats are
        reflected in the dashboard.
    """)

    action("GET /internal/v1/metrics")
    metrics = api_get("/internal/v1/metrics")
    result(metrics)

    # ================================================================
    # Done
    # ================================================================

    elapsed = time.time() - start
    event_count = "35+"

    print(f"\n{B}{GR}{'═' * 70}{R}")
    print(f"{B}{GR}  Demo Complete — Full Pipeline Executed{R}")
    print(f"{B}{GR}{'═' * 70}{R}")
    print(f"""
  {B}Job ID:{R}       {JOB_ID}
  {B}Final State:{R}  COMPLETE/COMPLETE
  {B}Flood Extent:{R} 61.4 km²
  {B}Events:{R}       {event_count} recorded to PostGIS
  {B}Escalations:{R}  1 created, 1 resolved
  {B}Time:{R}         {elapsed:.1f}s

  {D}What you just saw:{R}
  {D}An LLM agent drove a geospatial analysis job through the full{R}
  {D}FirstLight pipeline — discovering data, classifying floods,{R}
  {D}adjusting parameters mid-flight, escalating to a human when{R}
  {D}uncertain, and producing decision products. Every API call hit{R}
  {D}the live control plane. Every event was recorded to PostGIS{R}
  {D}and broadcast via SSE.{R}

  {D}Stream events:{R}  {API_BASE}/internal/v1/events/stream
  {D}Swagger UI:{R}     {API_BASE}/api/docs
  {D}STAC catalog:{R}   {API_BASE}/stac/
""")


if __name__ == "__main__":
    main()
