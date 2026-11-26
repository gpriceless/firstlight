"""
Run Command - Execute full pipeline from specification to products.

Usage:
    flight run --area miami.geojson --event flood --profile laptop --output ./products/
    flight run --bbox -82.2,26.3,-81.7,26.8 --event flood --output ./out/ --synthetic
"""

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import click
import numpy as np

logger = logging.getLogger("flight.run")


# Data mode constants
DATA_MODE_REAL = "real"
DATA_MODE_SYNTHETIC = "synthetic"


# Execution profiles
PROFILES = {
    "laptop": {
        "description": "Low-power laptop (4GB RAM, 2 cores)",
        "memory_mb": 2048,
        "max_workers": 2,
        "tile_size": 256,
        "parallel_downloads": 1,
    },
    "workstation": {
        "description": "Desktop workstation (16GB RAM, 4-8 cores)",
        "memory_mb": 8192,
        "max_workers": 4,
        "tile_size": 512,
        "parallel_downloads": 4,
    },
    "cloud": {
        "description": "Cloud instance (32GB+ RAM, 16+ cores)",
        "memory_mb": 32768,
        "max_workers": 16,
        "tile_size": 1024,
        "parallel_downloads": 8,
    },
    "edge": {
        "description": "Edge device (1GB RAM, 1 core)",
        "memory_mb": 1024,
        "max_workers": 1,
        "tile_size": 128,
        "parallel_downloads": 1,
    },
}


# The run options that decide which scenes are found and how they are read.
# A completed stage is reused only when all of them match.
RUN_INPUT_KEYS = (
    "area_path", "bbox", "analyze_bbox", "max_cloud", "start_date", "end_date",
    "event_type", "algorithm", "data_mode",
)


class WorkflowState:
    """Manage workflow state for resume capability."""

    STAGES = ["discover", "ingest", "analyze", "validate", "export"]

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.state_file = workdir / ".workflow_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return {
            "started_at": datetime.now().isoformat(),
            "current_stage": None,
            "completed_stages": [],
            "stage_results": {},
            "config": {},
        }

    def save(self):
        """Save state to file."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            # Stage results can hold numpy scalars (np.bool_, np.float32).
            json.dump(
                self.state, f, indent=2,
                default=lambda o: o.item() if hasattr(o, "item") else str(o),
            )

    def start_stage(self, stage: str):
        """Mark a stage as started."""
        self.state["current_stage"] = stage
        self.state["stage_results"][stage] = {
            "started_at": datetime.now().isoformat(),
            "status": "in_progress",
        }
        self.save()

    def complete_stage(self, stage: str, result: Dict[str, Any] = None):
        """Mark a stage as completed."""
        self.state["completed_stages"].append(stage)
        self.state["current_stage"] = None
        self.state["stage_results"][stage].update({
            "completed_at": datetime.now().isoformat(),
            "status": "completed",
            "result": result or {},
        })
        self.save()

    def fail_stage(self, stage: str, error: str):
        """Mark a stage as failed."""
        self.state["stage_results"][stage].update({
            "failed_at": datetime.now().isoformat(),
            "status": "failed",
            "error": error,
        })
        self.save()

    def is_completed(self, stage: str) -> bool:
        """Check if a stage is completed."""
        return stage in self.state["completed_stages"]

    def get_resume_stage(self) -> Optional[str]:
        """Get the stage to resume from."""
        for stage in self.STAGES:
            if not self.is_completed(stage):
                return stage
        return None


@click.command("run")
@click.option(
    "--area",
    "-a",
    "area_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GeoJSON file defining the area of interest.",
)
@click.option(
    "--bbox",
    "-b",
    type=str,
    help="Bounding box as min_lon,min_lat,max_lon,max_lat.",
)
@click.option(
    "--start",
    "-s",
    "start_date",
    type=str,
    default=None,
    help="Start date (YYYY-MM-DD). Default: 7 days ago.",
)
@click.option(
    "--end",
    "-e",
    "end_date",
    type=str,
    default=None,
    help="End date (YYYY-MM-DD). Default: today; a resumed run keeps its stored date.",
)
@click.option(
    "--event",
    "-t",
    "event_type",
    type=click.Choice(["flood", "wildfire", "storm"], case_sensitive=False),
    required=True,
    help="Event type to analyze.",
)
@click.option(
    "--profile",
    "-p",
    type=click.Choice(list(PROFILES.keys()), case_sensitive=False),
    default="workstation",
    help="Execution profile (default: workstation).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for all products.",
)
@click.option(
    "--algorithm",
    type=str,
    default=None,
    help="Specific algorithm to use (default: auto-select).",
)
@click.option(
    "--formats",
    "-f",
    type=str,
    default="geotiff,geojson",
    help="Output formats (default: geotiff,geojson).",
)
@click.option(
    "--skip-validate",
    is_flag=True,
    default=False,
    help="Skip validation step.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show execution plan without running.",
)
@click.option(
    "--synthetic",
    is_flag=True,
    default=False,
    help="Use synthetic data instead of downloading real satellite imagery. Useful for testing.",
)
@click.option(
    "--max-cloud",
    "max_cloud",
    type=float,
    default=None,
    help=(
        "Max cloud cover percentage for discover. Defaults to 90 for wildfire "
        "(post-event observations are often the only options and frequently "
        "cloudy) and 30 for flood/storm."
    ),
)
@click.pass_obj
def run(
    ctx,
    area_path: Optional[Path],
    bbox: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    event_type: str,
    profile: str,
    output_path: Path,
    algorithm: Optional[str],
    formats: str,
    skip_validate: bool,
    dry_run: bool,
    synthetic: bool,
    max_cloud: Optional[float],
):
    """
    Execute full analysis pipeline from specification to products.

    Runs the complete workflow: discover data, ingest, analyze, validate,
    and export results. Supports profile-based execution for different
    hardware configurations and can be interrupted and resumed.

    By default, downloads and processes REAL satellite data from STAC catalogs.
    Use --synthetic flag for testing with generated data.

    \b
    Examples:
        # Run flood analysis with real satellite data (default)
        flight run --area miami.geojson --event flood --profile laptop --output ./products/

        # Run with specific dates and algorithm
        flight run --area miami.geojson --start 2024-09-15 --end 2024-09-20 \\
            --event flood --algorithm sar_threshold --output ./products/

        # Run for wildfire with cloud profile
        flight run --bbox -120.5,34.0,-120.0,34.5 --event wildfire \\
            --profile cloud --output ./products/

        # Run with synthetic data for testing
        flight run --bbox -82.2,26.3,-81.7,26.8 --event flood --output ./test/ --synthetic
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    if not area_path and not bbox:
        raise click.BadParameter("Either --area or --bbox must be provided")

    # A date left to its default is resolved once, when the run starts, and
    # stored with it. Resuming reuses the stored date: today's default would
    # be a different input every day, and every second for the end date.
    state = WorkflowState(output_path)
    resumed = (state.state.get("config") or {}) if state.state.get("completed_stages") else {}
    if end_date:
        end_dt = parse_date(end_date)
    elif resumed.get("end_date"):
        end_dt = datetime.fromisoformat(resumed["end_date"])
        click.echo(f"  Resuming with the stored end date {end_dt.date()}")
    else:
        end_dt = datetime.now()
    if start_date:
        start_dt = parse_date(start_date)
    elif resumed.get("start_date"):
        start_dt = datetime.fromisoformat(resumed["start_date"])
        click.echo(f"  Resuming with the stored start date {start_dt.date()}")
    else:
        start_dt = end_dt - timedelta(days=7)

    # Get profile configuration
    profile_config = PROFILES[profile]

    # Determine data mode
    data_mode = DATA_MODE_SYNTHETIC if synthetic else DATA_MODE_REAL

    # Pick a max-cloud default that actually reflects the event. Wildfire
    # post-event windows are often the only available acquisition and are
    # frequently above 30% cloud (the Max Road Miramar scene was 78.9%).
    if max_cloud is None:
        max_cloud = 90.0 if event_type.lower() == "wildfire" else 30.0

    # If only --area was provided, derive a bbox string for the analyze loader
    # so it can window-read to the AOI instead of the full Sentinel-2 tile
    # (full-tile reads dominate stats with surrounding vegetation).
    analyze_bbox: Optional[str] = bbox
    if analyze_bbox is None and area_path is not None:
        try:
            from cli.commands.discover import (
                extract_bbox_from_geometry,
                load_geometry,
            )

            geometry = load_geometry(area_path, None)
            derived = extract_bbox_from_geometry(geometry)
            if derived and len(derived) == 4:
                analyze_bbox = ",".join(f"{v:.6f}" for v in derived)
                logger.info(
                    "Derived AOI bbox from %s: %s", area_path, analyze_bbox,
                )
        except Exception as e:
            logger.warning(f"Could not derive bbox from area {area_path}: {e}")

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  FirstLight - Full Pipeline Execution")
    click.echo(f"{'=' * 60}")
    click.echo(f"\n  Event type: {event_type}")
    click.echo(f"  Time window: {start_dt.date()} to {end_dt.date()}")
    click.echo(f"  Profile: {profile} - {profile_config['description']}")
    click.echo(f"  Data mode: {data_mode.upper()}")
    click.echo(f"  Output: {output_path}")
    if algorithm:
        click.echo(f"  Algorithm: {algorithm}")
    click.echo(f"  Export formats: {formats}")

    if dry_run:
        click.echo(f"\n[DRY RUN] Execution plan:")
        click.echo("  1. Discover available data")
        click.echo("  2. Ingest and normalize data")
        click.echo("  3. Run analysis algorithm")
        if not skip_validate:
            click.echo("  4. Validate results")
        click.echo(f"  {'5' if not skip_validate else '4'}. Export products")
        return

    new_config = {
        "area_path": str(area_path) if area_path else None,
        "bbox": bbox,
        "analyze_bbox": analyze_bbox,
        "max_cloud": max_cloud,
        "start_date": start_dt.isoformat(),
        "end_date": end_dt.isoformat(),
        "event_type": event_type,
        "profile": profile,
        "algorithm": algorithm,
        "formats": formats,
        "data_mode": data_mode,
        "skip_validate": skip_validate,
    }
    # Completed stages are skipped, so a directory holding a run with other
    # inputs would hand that run's scenes to this one. Refuse instead.
    old_config = state.state.get("config") or {}
    if state.state.get("completed_stages"):
        changed = sorted(
            key for key in RUN_INPUT_KEYS if old_config.get(key) != new_config.get(key)
        )
        if changed:
            raise click.UsageError(
                f"{output_path} holds a run with different inputs ({', '.join(changed)}). "
                "Resuming would reuse its scenes for this run. Use a new --output, "
                f"or remove {state.state_file} to start over."
            )
    state.state["config"] = new_config
    state.save()

    # Execute pipeline stages
    start_time = time.time()

    try:
        # Stage 1: Discover
        if not state.is_completed("discover"):
            click.echo(f"\n[1/{'5' if not skip_validate else '4'}] Discovering data...")
            state.start_stage("discover")
            discover_result = run_discover(
                area_path=area_path,
                bbox=bbox,
                start_date=start_dt,
                end_date=end_dt,
                event_type=event_type,
                output_path=output_path,
                max_cloud=max_cloud,
            )
            state.complete_stage("discover", discover_result)
            click.echo(f"    Found {discover_result.get('count', 0)} datasets")
            from core.data.discovery.stac_client import describe_sources

            for line in describe_sources(discover_result.get("sources") or {}):
                click.echo(f"    {line}")
        else:
            click.echo(f"\n[1/5] Discover: skipped (already completed)")

        # Stage 2: Ingest
        if not state.is_completed("ingest"):
            click.echo(f"\n[2/{'5' if not skip_validate else '4'}] Ingesting data...")
            state.start_stage("ingest")
            ingest_result = run_ingest(
                discovery_file=output_path / "discovery.json",
                output_path=output_path / "data",
                profile_config=profile_config,
                data_mode=data_mode,
                bbox=analyze_bbox,
                event_type=event_type,
            )
            ingested_count = ingest_result.get("count", 0)
            total = ingest_result.get("total", ingested_count)
            if data_mode == DATA_MODE_REAL and ingested_count == 0:
                raise RuntimeError(
                    f"Ingest produced 0 items in real mode — aborting before "
                    f"analyze to avoid fabricated results. Detail: 0 of {total} "
                    f"items ingested successfully"
                )
            state.complete_stage("ingest", ingest_result)
            click.echo(f"    Ingested {ingested_count} items")
            failed_items = ingest_result.get("failed") or []
            if failed_items:
                click.echo(
                    f"    Warning: {len(failed_items)} of {total} items failed: "
                    f"{failed_items}"
                )
        else:
            click.echo(f"\n[2/5] Ingest: skipped (already completed)")

        # Stage 3: Analyze
        if not state.is_completed("analyze"):
            click.echo(f"\n[3/{'5' if not skip_validate else '4'}] Running analysis...")
            state.start_stage("analyze")
            # Use the midpoint of the search window as the fire boundary date.
            # Scenes before this date are treated as pre-fire, on/after as post-fire.
            event_dt = start_dt + (end_dt - start_dt) / 2
            analyze_result = run_analyze(
                input_path=output_path / "data",
                output_path=output_path / "results",
                event_type=event_type,
                algorithm=algorithm,
                profile_config=profile_config,
                data_mode=data_mode,
                bbox=analyze_bbox,
                event_date=event_dt,
            )
            state.complete_stage("analyze", analyze_result)
            click.echo(f"    Algorithm: {analyze_result.get('algorithm', 'auto')}")
            if analyze_result.get("statistics"):
                stats = analyze_result["statistics"]
                if "flood_area_ha" in stats:
                    click.echo(f"    Flood area: {stats['flood_area_ha']:.1f} hectares")
                if "burned_area_ha" in stats:
                    click.echo(f"    Burned area: {stats['burned_area_ha']:.1f} hectares")
                if "damage_area_ha" in stats:
                    click.echo(f"    Damage area: {stats['damage_area_ha']:.1f} hectares")
                if "cloud_masked_percent" in stats:
                    click.echo(
                        f"    Cloud and shadow masked: {stats['cloud_masked_percent']:.1f}% of the box"
                    )
                for line in _no_data_lines(stats, event_type):
                    click.echo(f"    {line}")
        else:
            click.echo(f"\n[3/5] Analyze: skipped (already completed)")

        # Stage 4: Validate (optional)
        if not skip_validate:
            if not state.is_completed("validate"):
                click.echo(f"\n[4/5] Validating results...")
                state.start_stage("validate")
                validate_result = run_validate(
                    input_path=output_path / "results",
                    output_path=output_path,
                    data_mode=data_mode,
                )
                state.complete_stage("validate", validate_result)
                score = validate_result.get("score", 0)
                status = "PASSED" if validate_result.get("passed") else "FAILED"
                click.echo(f"    Quality score: {score:.2f} ({status})")
            else:
                click.echo(f"\n[4/5] Validate: skipped (already completed)")

        # Stage 5: Export
        export_stage = "5" if not skip_validate else "4"
        total_stages = "5" if not skip_validate else "4"
        if not state.is_completed("export"):
            click.echo(f"\n[{export_stage}/{total_stages}] Exporting products...")
            state.start_stage("export")
            export_result = run_export(
                input_path=output_path / "results",
                output_path=output_path / "products",
                formats=formats,
                data_mode=data_mode,
                bbox=bbox,
            )
            state.complete_stage("export", export_result)
            click.echo(f"    Exported: {', '.join(export_result.get('formats', []))}")
        else:
            click.echo(f"\n[{export_stage}/{total_stages}] Export: skipped (already completed)")

        # Success
        elapsed = time.time() - start_time
        click.echo(f"\n{'=' * 60}")
        click.echo(f"  Pipeline Complete!")
        click.echo(f"{'=' * 60}")
        click.echo(f"\n  Elapsed time: {elapsed:.1f}s")
        click.echo(f"  Output directory: {output_path}")
        _echo_failed_sources(output_path / "discovery.json")
        click.echo(f"\n  Products:")
        products_dir = output_path / "products"
        if products_dir.exists():
            for p in products_dir.iterdir():
                if p.is_file():
                    click.echo(f"    - {p.name}")

    except KeyboardInterrupt:
        click.echo(f"\n\nInterrupted. Use 'flight resume --workdir {output_path}' to continue.")
        state.save()
        sys.exit(130)

    except Exception as e:
        current = state.state.get("current_stage")
        if current:
            state.fail_stage(current, str(e))
        logger.error(f"Pipeline failed: {e}")
        click.echo(f"\nError: {e}", err=True)
        click.echo(f"Use 'flight resume --workdir {output_path}' to retry.")
        sys.exit(1)


def parse_date(date_str: str) -> datetime:
    """Parse date string."""
    formats = ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date: {date_str}")


def _echo_failed_sources(discovery_file: Path) -> None:
    """Repeat, at the end of a run, any source whose search failed."""
    from core.data.discovery.stac_client import describe_sources

    try:
        sources = json.loads(discovery_file.read_text()).get("sources") or {}
    except (OSError, ValueError):
        return
    failed = {k: v for k, v in sources.items() if v.get("status") == "failed"}
    for line in describe_sources(failed):
        click.echo(f"  Warning: {line}")


def run_discover(
    area_path: Optional[Path],
    bbox: Optional[str],
    start_date: datetime,
    end_date: datetime,
    event_type: str,
    output_path: Path,
    max_cloud: float = 30.0,
) -> Dict[str, Any]:
    """Run discovery stage. A failed search stops the run; it never writes stand-in ids."""
    from cli.commands.discover import perform_discovery, load_geometry

    geometry = load_geometry(area_path, bbox)
    results = perform_discovery(
        geometry=geometry,
        start=start_date,
        end=end_date,
        event_type=event_type,
        sources=None,
        max_cloud=max_cloud,
        config={},
    )
    sources = dict(getattr(results, "sources", {}) or {})

    discovery_file = output_path / "discovery.json"
    with open(discovery_file, "w") as f:
        json.dump(
            {"count": len(results), "sources": sources, "results": results}, f, indent=2
        )

    return {"count": len(results), "file": str(discovery_file), "sources": sources}


def run_ingest(
    discovery_file: Path,
    output_path: Path,
    profile_config: Dict[str, Any],
    data_mode: str = DATA_MODE_REAL,
    bbox: Optional[str] = None,
    event_type: str = "flood",
) -> Dict[str, Any]:
    """Run ingestion stage - download real data or generate synthetic."""
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        with open(discovery_file) as f:
            discovery = json.load(f)

        items = discovery.get("results", [])
        count = len(items)

        if data_mode == DATA_MODE_REAL:
            # Real mode: Use the actual ingest logic to download satellite data
            # Fail hard if ingest fails - never silently fall back to synthetic
            from cli.commands.ingest import process_item
            from core.data.ingestion.validation.config import (
                ValidationConfigError,
                load_config,
            )

            # Read the validation config before any download, so a broken
            # file stops the run at no cost.
            load_config()

            box = None
            if bbox:
                parts = [float(x) for x in bbox.split(",")]
                if len(parts) == 4:
                    box = parts
            ingested = 0
            failed = []
            for item in items:
                try:
                    success = process_item(
                        item=item,
                        output_path=output_path,
                        output_format="cog",
                        normalize=True,
                        target_crs=None,
                        target_resolution=None,
                        bbox=box,
                    )
                    if success:
                        ingested += 1
                    else:
                        failed.append(item.get('id', 'unknown'))
                except ValidationConfigError:
                    raise
                except Exception as e:
                    logger.error(f"Failed to ingest {item.get('id')}: {e}")
                    failed.append(item.get('id', 'unknown'))

            if ingested == 0 and items:
                raise RuntimeError(
                    f"Real data ingestion failed: 0 of {len(items)} items ingested successfully. "
                    f"Failed items: {failed[:5]}{'...' if len(failed) > 5 else ''}"
                )

            if failed:
                logger.warning(
                    f"Partial ingest: {ingested} of {len(items)} items succeeded. "
                    f"Failed: {failed}"
                )

            return {
                "count": ingested,
                "total": len(items),
                "path": str(output_path),
                "mode": "real",
                "failed": failed,
            }

        # Synthetic mode: Generate realistic synthetic data
        elif data_mode == DATA_MODE_SYNTHETIC:
            return _generate_synthetic_ingest_data(
                items=items,
                output_path=output_path,
                bbox=bbox,
                event_type=event_type,
            )

        return {"count": count, "path": str(output_path)}

    except Exception as e:
        # A failed ingest means there is no data to analyze, in either mode.
        # Propagate so the run stops and exits nonzero.
        logger.error(f"Ingest failed: {e}")
        raise


def _generate_synthetic_ingest_data(
    items: List[Dict[str, Any]],
    output_path: Path,
    bbox: Optional[str],
    event_type: str,
) -> Dict[str, Any]:
    """Generate realistic synthetic satellite data for testing."""
    count = len(items)
    size = 512  # 512x512 pixels at 10m = 5.12km x 5.12km

    # Parse bbox if provided
    bbox_coords = None
    if bbox:
        try:
            bbox_coords = [float(x) for x in bbox.split(",")]
        except ValueError:
            pass

    # Create metadata for each item
    for item in items:
        item_dir = output_path / item.get("source", "unknown") / item.get("id", "unknown")
        item_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = item_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(item, f, indent=2)

    # Generate synthetic raster data based on event type
    synthetic_dir = output_path / "synthetic"
    synthetic_dir.mkdir(exist_ok=True)

    np.random.seed(42)

    if event_type.lower() == "flood":
        # Generate SAR backscatter data (dB scale: land ~-10dB, water ~-20dB)
        sar_data = np.random.normal(-10, 2.5, (size, size)).astype(np.float32)
        # Add flood patterns
        river = (slice(None), slice(int(size*0.7), None))
        sar_data[river] = np.random.normal(-20, 2, sar_data[river].shape)
        lake = (slice(int(size*0.4), int(size*0.6)), slice(int(size*0.2), int(size*0.8)))
        sar_data[lake] = np.random.normal(-18, 1.5, sar_data[lake].shape)
        np.save(synthetic_dir / "sar_vv.npy", sar_data)

        # Generate optical bands for NDWI
        green = np.random.normal(0.12, 0.03, (size, size)).astype(np.float32)
        nir = np.random.normal(0.35, 0.06, (size, size)).astype(np.float32)
        np.save(synthetic_dir / "green.npy", np.clip(green, 0, 1))
        np.save(synthetic_dir / "nir.npy", np.clip(nir, 0, 1))

    elif event_type.lower() == "wildfire":
        # Generate pre/post fire imagery
        pre_nir = np.clip(np.random.normal(0.4, 0.08, (size, size)), 0.1, 0.6).astype(np.float32)
        pre_swir = np.clip(np.random.normal(0.15, 0.04, (size, size)), 0.05, 0.3).astype(np.float32)

        post_nir = pre_nir.copy()
        post_swir = pre_swir.copy()

        # Create burn scar
        burn_mask = np.zeros((size, size), dtype=bool)
        burn_mask[int(size*0.2):int(size*0.7), int(size*0.3):int(size*0.8)] = True

        post_nir[burn_mask] = np.random.normal(0.1, 0.03, np.sum(burn_mask))
        post_swir[burn_mask] = np.random.normal(0.35, 0.05, np.sum(burn_mask))

        np.save(synthetic_dir / "pre_nir.npy", pre_nir)
        np.save(synthetic_dir / "pre_swir.npy", pre_swir)
        np.save(synthetic_dir / "post_nir.npy", post_nir)
        np.save(synthetic_dir / "post_swir.npy", post_swir)

    else:  # storm
        # Generate pre/post storm optical imagery
        pre = np.random.normal(0.3, 0.05, (size, size)).astype(np.float32)
        post = pre.copy()
        damage_mask = np.random.random((size, size)) > 0.7
        post[damage_mask] = np.random.normal(0.15, 0.05, np.sum(damage_mask))
        np.save(synthetic_dir / "pre_optical.npy", np.clip(pre, 0, 1))
        np.save(synthetic_dir / "post_optical.npy", np.clip(post, 0, 1))

    # Generate DEM
    dem = np.zeros((size, size), dtype=np.float32)
    x = np.linspace(0, 1, size)
    dem += (8 * (1 - x))[np.newaxis, :]
    dem += np.random.normal(0, 0.5, (size, size))
    np.save(synthetic_dir / "dem.npy", np.clip(dem, -2, 50))

    return {"count": count, "path": str(output_path), "mode": "synthetic"}


def run_analyze(
    input_path: Path,
    output_path: Path,
    event_type: str,
    algorithm: Optional[str],
    profile_config: Dict[str, Any],
    data_mode: str = DATA_MODE_REAL,
    bbox: Optional[str] = None,
    event_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Run analysis stage - execute real algorithms or generate synthetic results."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Select algorithm based on event type
    if not algorithm:
        algorithm_map = {
            "flood": "sar_threshold",
            "wildfire": "dnbr",
            "storm": "wind_damage",
        }
        algorithm = algorithm_map.get(event_type.lower(), "sar_threshold")

    statistics = {}

    # Run analysis based on data mode
    if data_mode == DATA_MODE_REAL:
        # Real mode: fail hard if analysis fails - never mix synthetic data
        result = _run_real_analysis(
            input_path=input_path,
            output_path=output_path,
            event_type=event_type,
            algorithm=algorithm,
            profile_config=profile_config,
            bbox=bbox,
            event_date=event_date,
        )
        if not result.get("success"):
            error_msg = result.get("error", "Unknown analysis error")
            raise RuntimeError(f"Real analysis failed: {error_msg}")
        statistics = result.get("statistics", {})

    elif data_mode == DATA_MODE_SYNTHETIC:
        result = _run_synthetic_analysis(
            input_path=input_path,
            output_path=output_path,
            event_type=event_type,
            algorithm=algorithm,
        )
        statistics = result.get("statistics", {})

    # Save metadata
    metadata = {
        "algorithm": algorithm,
        "event_type": event_type,
        "profile": profile_config,
        "data_mode": data_mode,
        "statistics": statistics,
        "completed_at": datetime.now().isoformat(),
    }
    with open(output_path / "analysis_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {"algorithm": algorithm, "path": str(output_path), "statistics": statistics}


def _run_real_analysis(
    input_path: Path,
    output_path: Path,
    event_type: str,
    algorithm: str,
    profile_config: Dict[str, Any],
    bbox: Optional[str] = None,
    event_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Execute real analysis algorithms on ingested data."""
    from cli.commands.analyze import ALGORITHMS, load_algorithm

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Load and initialize the algorithm (params=None for defaults)
    algo_instance = load_algorithm(algorithm, params=None)

    # Find input data files
    raster_files = list(input_path.rglob("*.tif"))
    npy_files = list(input_path.rglob("*.npy"))

    if not raster_files and not npy_files:
        raise FileNotFoundError(
            f"No real input rasters found in {input_path}. "
            "Refusing to run analysis without real data — this would produce "
            "fabricated results."
        )

    # Execute based on event type
    if event_type.lower() == "flood":
        return _analyze_flood(algo_instance, input_path, output_path, raster_files, npy_files)
    elif event_type.lower() == "wildfire":
        return _analyze_wildfire(
            algo_instance,
            input_path,
            output_path,
            raster_files,
            npy_files,
            event_date=event_date,
            bbox=bbox,
        )
    else:
        return _analyze_storm(
            algo_instance,
            input_path,
            output_path,
            raster_files,
            npy_files,
            event_date=event_date,
            bbox=bbox,
        )


def _analyze_flood(
    algorithm,
    input_path: Path,
    output_path: Path,
    raster_files: List[Path],
    npy_files: List[Path],
) -> Dict[str, Any]:
    """Run flood detection analysis."""
    # Try to load SAR data
    sar_data = None
    synthetic_dir = input_path / "synthetic"
    grid: Optional[Dict[str, Any]] = None

    if (synthetic_dir / "sar_vv.npy").exists():
        sar_data = np.load(synthetic_dir / "sar_vv.npy")
    elif npy_files:
        for f in npy_files:
            if "sar" in f.name.lower():
                sar_data = np.load(f)
                break

    # Only a raster named as SAR backscatter counts. Ingest also writes
    # Sentinel-2 bands under the same work dir, and an optical band read as
    # SAR would produce a flood map from the wrong physics.
    sar_rasters = [
        f for f in raster_files if any(k in f.stem.lower() for k in ("sar", "vv"))
    ]
    if sar_data is None and sar_rasters:
        try:
            import rasterio
            with rasterio.open(sar_rasters[0]) as src:
                sar_data = src.read(1)
                pixel_size_m = float(abs(src.transform.a)) or 10.0
                grid = _grid_info_from_rasterio(
                    crs=src.crs,
                    transform=src.transform,
                    shape=(src.height, src.width),
                    pixel_size_m=pixel_size_m,
                    source="raster",
                    extra={"raster": sar_rasters[0].name},
                )
        except Exception as e:
            logger.warning(f"Could not load raster: {e}")

    if sar_data is None:
        raise FileNotFoundError(
            "Real flood analysis requires SAR data, but none was found in "
            f"{input_path}. Refusing to fabricate synthetic data in real mode."
        )

    # Execute algorithm
    result = algorithm.execute(sar_data, pixel_size_m=10.0)

    # Save outputs
    np.save(output_path / "flood_extent.npy", result.flood_extent)
    np.save(output_path / "confidence.npy", result.confidence_raster)

    if grid is not None:
        with open(output_path / "grid.json", "w") as f:
            json.dump(grid, f, indent=2)

    return {
        "success": True,
        "statistics": result.statistics,
    }


def _parse_sentinel2_item_date(item_id: str) -> Optional[datetime]:
    """Parse acquisition date from a Sentinel-2 item id.

    Expected format: ``S2{A|B}_<tile>_<YYYYMMDD>_<rev>_<level>`` (the date is the
    third underscore-delimited segment). Returns ``None`` if the id does not
    match this layout — the caller is responsible for skipping such scenes.
    """
    parts = item_id.split("_")
    if len(parts) < 3:
        return None
    date_segment = parts[2]
    if len(date_segment) != 8 or not date_segment.isdigit():
        return None
    try:
        return datetime.strptime(date_segment, "%Y%m%d")
    except ValueError:
        return None


def _scl_clear_share(scl, bbox: Optional[str] = None) -> float:
    """Share of the area of interest that a scene's SCL layers say was seen.

    ``scl`` is one SCL path or the paths of every tile of a same-day scene.
    They are mosaicked over ``bbox`` (geographic min_lon,min_lat,max_lon,max_lat),
    so any part of the box no tile covers counts as no data, not as clear. With
    no bbox the first raster's extent is the area.
    """
    scl_arr = _scl_over_box(scl, bbox)
    if scl_arr.size == 0:
        return 0.0
    return float((~_scl_cloud_mask(scl_arr)).mean())


def _scl_no_data_share(scl, bbox: Optional[str] = None) -> float:
    """Share of the area of interest where a scene has no pixels at all.

    SCL class 0 is no data, and the mosaic writes 0 where no tile of the day
    covers the box, so this is the part of the box the scene never saw.
    """
    scl_arr = _scl_over_box(scl, bbox)
    if scl_arr.size == 0:
        return 1.0
    return float((scl_arr == 0).mean())


def _scl_over_box(scl, bbox: Optional[str] = None) -> np.ndarray:
    paths = [scl] if isinstance(scl, (str, Path)) else list(scl)
    return _read_mosaic(paths, _aoi_grid(paths[0], bbox), categorical=True)


def _find_sentinel2_pre_post_scenes(
    input_path: Path,
    event_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
    bands: Tuple[str, ...] = ("nir", "swir22"),
    optional: Tuple[str, ...] = (),
    hazard: str = "wildfire",
) -> Dict[str, Dict[str, Any]]:
    """Discover Sentinel-2 scenes and pick one before and one after the event.

    The ingest stage writes scenes to ``<input_path>/sentinel2/<item_id>/`` with
    one ``<band>.tif`` per band and ``scl.tif``. A scene is a candidate when it
    has every band in ``bands`` and an SCL layer. Returns a dict with ``pre``
    and ``post`` keys, each mapping to ``{item_id, date, tiles, scl,
    clear_share}`` plus one path list per band (``None`` for an ``optional``
    band some tile lacks). Tiles from the same day are one scene, with every
    path a list over its tiles, so a box across a tile edge is mosaicked. On each
    side the scene with the largest clear share of the area of interest wins
    (``bbox``, or the whole raster); a tie goes to the scene nearest the event.
    Raises ``FileNotFoundError`` if there isn't at least one scene on each
    side, or if no scene has an SCL layer to mask cloud with.
    """
    s2_root = input_path / "sentinel2"
    band_list = " + ".join(b.upper() for b in bands)
    candidates: List[Dict[str, Any]] = []
    no_scl: List[str] = []
    if s2_root.is_dir():
        for item_dir in sorted(s2_root.iterdir()):
            if not item_dir.is_dir():
                continue
            paths = {b: item_dir / f"{b}.tif" for b in bands}
            if not all(path.exists() for path in paths.values()):
                continue
            scene_date = _parse_sentinel2_item_date(item_dir.name)
            if scene_date is None:
                continue
            scl = item_dir / "scl.tif"
            if not scl.exists():
                no_scl.append(item_dir.name)
                continue
            for b in optional:
                path = item_dir / f"{b}.tif"
                paths[b] = path if path.exists() else None
            candidates.append({
                "item_id": item_dir.name,
                "date": scene_date,
                "scl": scl,
                **paths,
            })

    # Tiles of one day are one scene: a box across a tile edge is mosaicked.
    by_date: Dict[datetime, List[Dict[str, Any]]] = {}
    for tile in candidates:
        by_date.setdefault(tile["date"], []).append(tile)
    candidates = [
        {
            "item_id": "+".join(t["item_id"] for t in tiles),
            "date": day,
            "tiles": [t["item_id"] for t in tiles],
            "scl": [t["scl"] for t in tiles],
            **{b: [t[b] for t in tiles] for b in bands},
            # An optional band is used only if every tile has it.
            **{
                b: [t[b] for t in tiles] if all(t[b] for t in tiles) else None
                for b in optional
            },
        }
        for day, tiles in by_date.items()
    ]

    if no_scl and not candidates:
        raise FileNotFoundError(
            f"Real {hazard} analysis needs the Sentinel-2 SCL layer to mask "
            f"cloud and shadow, but these scenes have none: {no_scl}. "
            "Re-run ingest so scl.tif is downloaded; refusing to count cloud "
            "as ground."
        )
    if no_scl:
        logger.warning(
            "Skipping Sentinel-2 scenes with no SCL layer (cannot mask cloud): %s",
            no_scl,
        )

    if not candidates:
        raise FileNotFoundError(
            f"Real {hazard} analysis requires pre/post {band_list} rasters, "
            f"but none were found in {input_path}. Refusing to fabricate "
            "synthetic data in real mode."
        )

    candidates.sort(key=lambda s: s["date"])
    if event_date is None:
        # Split at the midpoint of available scene dates so we always end up
        # with at least one pre and one post scene when 2+ are present.
        midpoint_idx = len(candidates) // 2
        split_date = candidates[midpoint_idx]["date"]
        pre = [s for s in candidates if s["date"] < split_date]
        post = [s for s in candidates if s["date"] >= split_date]
        if not pre and len(candidates) >= 2:
            pre = candidates[:1]
            post = candidates[1:]
    else:
        pre = [s for s in candidates if s["date"] < event_date]
        post = [s for s in candidates if s["date"] >= event_date]

    if not pre or not post:
        raise FileNotFoundError(
            f"Real {hazard} analysis requires both pre-event and post-event "
            f"Sentinel-2 scenes ({band_list}) under {input_path}/sentinel2/, "
            f"but only {len(candidates)} scene(s) were found"
            + (f" relative to event date {event_date.date()}" if event_date else "")
            + ". Refusing to fabricate synthetic data in real mode."
        )

    # A scene that is all cloud over the box measures nothing, however close
    # to the event it is. Rank by clear share over the area of interest, then
    # by nearness to the event (latest pre, earliest post).
    for scene in pre + post:
        scene["clear_share"] = _scl_clear_share(scene["scl"], bbox)
        logger.info(
            "Sentinel-2 %s (%s): %.1f%% of the area clear",
            scene["item_id"], scene["date"].date(), 100 * scene["clear_share"],
        )
    pre_choice = max(enumerate(pre), key=lambda e: (e[1]["clear_share"], e[0]))[1]
    post_choice = max(enumerate(post), key=lambda e: (e[1]["clear_share"], -e[0]))[1]
    for side, scene in (("pre", pre_choice), ("post", post_choice)):
        scene["no_data_share"] = _scl_no_data_share(scene["scl"], bbox)
        if round(100 * scene["no_data_share"], 1) > 0:
            logger.warning(
                "The %s-event scene %s (%s) has no data over %.1f%% of the area",
                side, scene["item_id"], scene["date"].date(),
                100 * scene["no_data_share"],
            )
    return {"pre": pre_choice, "post": post_choice}


def _aoi_grid(reference: Path, bbox: Optional[str] = None) -> Dict[str, Any]:
    """The analysis grid: ``reference``'s CRS and pixel alignment over ``bbox``.

    The grid covers the whole box even where ``reference`` does not, so other
    tiles can fill it. With no usable bbox it is ``reference``'s own extent.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    with rasterio.open(reference) as src:
        grid = {
            "crs": src.crs,
            "transform": src.transform,
            "shape": (src.height, src.width),
            "pixel_size": float(abs(src.transform.a)),
        }
        parts = [float(x) for x in bbox.split(",")] if bbox else []
        if len(parts) == 4:
            bounds = transform_bounds("EPSG:4326", src.crs, *parts, densify_pts=21)
            window = from_bounds(*bounds, transform=src.transform)
            window = window.round_offsets().round_lengths()
            grid["transform"] = src.window_transform(window)
            grid["shape"] = (int(window.height), int(window.width))
        elif bbox:
            logger.warning(f"Ignoring bbox {bbox!r}: need min_lon,min_lat,max_lon,max_lat")
    return grid


def _analysis_grid(*scenes: Dict[str, Any], bbox: Optional[str] = None) -> Dict[str, Any]:
    """Grid on the NIR tile holding most of the box, so the fewest pixels are warped."""
    import rasterio

    def pixels(path: Path) -> int:
        with rasterio.open(path) as src:
            return src.width * src.height

    nir = [path for scene in scenes for path in scene["nir"]]
    return _aoi_grid(max(nir, key=pixels), bbox)


def _read_mosaic(
    paths: List[Path], grid: Dict[str, Any], categorical: bool = False,
) -> np.ndarray:
    """Read one band from each tile onto ``grid`` and merge them.

    Reflectance is resampled bilinearly into float32 with NaN where no tile has
    data (source value 0 is Sentinel-2 no data). Categorical layers such as SCL
    are resampled by nearest neighbour in their own dtype, 0 where uncovered,
    which SCL reads as no data. A pixel takes the first tile that has it away
    from that tile's edge, where bilinear has neighbours on every side; a tile's
    edge pixels are used only where no other tile covers.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    fill = 0 if categorical else np.nan
    warped = []
    for path in paths:
        with rasterio.open(path) as src:
            tile = np.full(
                grid["shape"], fill, dtype=src.dtypes[0] if categorical else np.float32
            )
            reproject(
                source=rasterio.band(src, 1),
                destination=tile,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                dst_nodata=fill,
                resampling=Resampling.nearest if categorical else Resampling.bilinear,
            )
            inner = np.zeros((src.height, src.width), dtype=np.uint8)
            inner[1:-1, 1:-1] = 1
            interior = np.zeros(grid["shape"], dtype=np.uint8)
            reproject(
                source=inner,
                destination=interior,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=grid["transform"],
                dst_crs=grid["crs"],
                resampling=Resampling.nearest,
            )
        warped.append((tile, interior.astype(bool)))

    out = np.full(grid["shape"], fill, dtype=warped[0][0].dtype)
    for edges_too in (False, True):
        for tile, interior in warped:
            empty = (out == 0) if categorical else np.isnan(out)
            take = empty if edges_too else empty & interior
            out[take] = tile[take]
    return out


def _to_reflectance(arr: np.ndarray) -> np.ndarray:
    """Sentinel-2 L2A DN to 0..1 reflectance; arrays already in 0..1 pass through."""
    seen = arr[np.isfinite(arr)]
    if seen.size and seen.max() > 1.5:
        return (arr / 10000.0).astype(np.float32)
    return arr.astype(np.float32)


def _load_sentinel2_pre_post(
    input_path: Path,
    event_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
) -> tuple:
    """Load pre/post NIR and SWIR22 arrays from an ingested Sentinel-2 tree.

    Returns ``(pre_nir, pre_swir, post_nir, post_swir, pixel_size_m, grid,
    cloud_mask)``. Every band is mosaicked over the box from all tiles of its
    day onto one NIR grid (see ``_analysis_grid``), so SWIR22 and SCL share the
    NIR shape. ``cloud_mask`` is True where either scene's SCL says no data,
    saturated, cloud, cirrus or cloud shadow, which includes any part of the
    box no tile covers. ``grid`` is a serializable dict of the output CRS and
    affine transform so the export stage can produce georeferenced products.
    """
    scenes = _find_sentinel2_pre_post_scenes(
        input_path, event_date=event_date, bbox=bbox
    )
    pre = scenes["pre"]
    post = scenes["post"]
    aoi = _analysis_grid(pre, post, bbox=bbox)

    def _read(paths: List[Path]) -> np.ndarray:
        return _to_reflectance(_read_mosaic(paths, aoi))

    pre_nir, pre_swir = _read(pre["nir"]), _read(pre["swir22"])
    post_nir, post_swir = _read(post["nir"]), _read(post["swir22"])
    cloud_mask = _pre_post_cloud_mask(pre, post, aoi)

    logger.info(
        "Wildfire real-mode loader: pre=%s (%s) post=%s (%s) shape=%s pixel=%sm",
        pre["item_id"], pre["date"].date(),
        post["item_id"], post["date"].date(),
        aoi["shape"], aoi["pixel_size"],
    )
    return (
        pre_nir, pre_swir, post_nir, post_swir,
        aoi["pixel_size"], _scene_grid_info(aoi, pre, post), cloud_mask,
    )


def _scene_grid_info(aoi: Dict[str, Any], pre: Dict[str, Any], post: Dict[str, Any]):
    return _grid_info_from_rasterio(
        crs=aoi["crs"],
        transform=aoi["transform"],
        shape=aoi["shape"],
        pixel_size_m=aoi["pixel_size"],
        source="sentinel2",
        extra={
            "pre_item_id": pre["item_id"],
            "post_item_id": post["item_id"],
            "pre_clear_share": round(pre["clear_share"], 4),
            "post_clear_share": round(post["clear_share"], 4),
            "pre_date": pre["date"].date().isoformat(),
            "post_date": post["date"].date().isoformat(),
            "pre_no_data_share": round(pre["no_data_share"], 4),
            "post_no_data_share": round(post["no_data_share"], 4),
        },
    )


def _pre_post_cloud_mask(
    pre: Dict[str, Any], post: Dict[str, Any], aoi: Dict[str, Any],
) -> np.ndarray:
    """True where either scene's SCL says the surface was not seen."""
    masks = [
        _scl_cloud_mask(_read_mosaic(scene["scl"], aoi, categorical=True))
        for scene in (pre, post)
    ]
    return masks[0] | masks[1]


def _no_data_statistics(grid: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Per side, the scene date and the share of the box it has no pixels for."""
    stats: Dict[str, Any] = {}
    for side in ("pre", "post"):
        share = (grid or {}).get(f"{side}_no_data_share")
        if share is not None:
            stats[f"{side}_date"] = grid[f"{side}_date"]
            stats[f"{side}_no_data_percent"] = round(100.0 * share, 1)
    return stats


def _no_data_lines(stats: Dict[str, Any], event_type: str) -> List[str]:
    """Plain sentences naming the part of the box a side has no scene for."""
    noun = {"wildfire": "fire"}.get(event_type.lower(), event_type.lower())
    lines = []
    for side in ("pre", "post"):
        percent = stats.get(f"{side}_no_data_percent")
        if percent:
            lines.append(
                f"No {side}-{noun} data over {percent:.1f}% of the box: the "
                f"{stats[f'{side}_date']} scene has no pixels there, so that "
                "part is masked, not measured."
            )
    return lines


def _scl_cloud_mask(scl: np.ndarray) -> np.ndarray:
    """True where an SCL class map says the surface was not seen.

    Masks no data, saturated, cloud (medium and high), thin cirrus and cloud
    shadow. Shadow is masked because it darkens NIR and reads as burn in dNBR.
    """
    from core.reporting.imagery.cloud_mask import CloudMask, CloudMaskConfig

    masker = CloudMask(CloudMaskConfig(mask_clouds=True, mask_cirrus=True, mask_shadows=True))
    return masker.from_scl_band(scl).mask


def _load_sentinel2_storm_pre_post(
    input_path: Path,
    event_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
) -> tuple:
    """Load pre/post red + NIR (+ optional blue) arrays for storm analysis.

    Returns ``(bands, pixel_size_m, grid, cloud_mask)`` where ``bands`` is a
    dict with keys ``red_pre, red_post, nir_pre, nir_post`` (all float32
    reflectance) plus ``blue_pre, blue_post`` (each either an array or
    ``None``). ``cloud_mask`` is True where either scene's SCL says no data,
    saturated, cloud, cirrus or cloud shadow: cloud brightens red and drops
    NDVI, which reads as vegetation loss. Every band is mosaicked over the box
    from all tiles of its day onto one NIR grid, as for wildfire.
    """
    scenes = _find_sentinel2_pre_post_scenes(
        input_path, event_date=event_date, bbox=bbox,
        bands=("red", "nir"), optional=("blue",), hazard="storm",
    )
    pre = scenes["pre"]
    post = scenes["post"]
    aoi = _analysis_grid(pre, post, bbox=bbox)

    def _read(paths: Optional[List[Path]]) -> Optional[np.ndarray]:
        return _to_reflectance(_read_mosaic(paths, aoi)) if paths else None

    bands: Dict[str, Optional[np.ndarray]] = {
        "red_pre": _read(pre["red"]),
        "red_post": _read(post["red"]),
        "nir_pre": _read(pre["nir"]),
        "nir_post": _read(post["nir"]),
        "blue_pre": _read(pre["blue"]),
        "blue_post": _read(post["blue"]),
    }

    logger.info(
        "Storm real-mode loader: pre=%s (%s) post=%s (%s) shape=%s pixel=%sm blue=%s",
        pre["item_id"], pre["date"].date(),
        post["item_id"], post["date"].date(),
        aoi["shape"], aoi["pixel_size"],
        bool(bands["blue_pre"] is not None and bands["blue_post"] is not None),
    )

    cloud_mask = _pre_post_cloud_mask(pre, post, aoi)
    return bands, aoi["pixel_size"], _scene_grid_info(aoi, pre, post), cloud_mask


def _grid_info_from_rasterio(
    crs,
    transform,
    shape: tuple,
    pixel_size_m: float,
    source: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize a rasterio CRS + Affine pair into a sidecar dict.

    The export stage round-trips this via :func:`_load_grid_info` to reconstruct
    real georeferencing for the .npy analyze outputs. Stores both ``crs`` (a
    PROJ-style string when available) and ``crs_wkt`` (full WKT) so consumers
    have a robust path even for non-EPSG CRSes.
    """
    crs_str: Optional[str]
    crs_wkt: Optional[str]
    try:
        crs_str = crs.to_string() if crs is not None else None
    except Exception:
        crs_str = None
    try:
        crs_wkt = crs.to_wkt() if crs is not None else None
    except Exception:
        crs_wkt = None

    grid: Dict[str, Any] = {
        "crs": crs_str,
        "crs_wkt": crs_wkt,
        "transform": [
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        ],
        "shape": [int(shape[0]), int(shape[1])],
        "pixel_size_m": float(pixel_size_m),
        "source": source,
    }
    if extra:
        grid.update(extra)
    return grid


def _load_grid_info(input_path: Path) -> Optional[Dict[str, Any]]:
    """Return the analyze stage's grid.json sidecar, if present."""
    grid_path = input_path / "grid.json"
    if not grid_path.exists():
        return None
    try:
        with open(grid_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load grid.json sidecar: {e}")
        return None


def _analyze_wildfire(
    algorithm,
    input_path: Path,
    output_path: Path,
    raster_files: List[Path],
    npy_files: List[Path],
    event_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
) -> Dict[str, Any]:
    """Run wildfire/burn severity analysis."""
    synthetic_dir = input_path / "synthetic"
    grid: Optional[Dict[str, Any]] = None
    cloud_mask: Optional[np.ndarray] = None

    # Load pre/post imagery
    if (synthetic_dir / "pre_nir.npy").exists():
        pre_nir = np.load(synthetic_dir / "pre_nir.npy")
        pre_swir = np.load(synthetic_dir / "pre_swir.npy")
        post_nir = np.load(synthetic_dir / "post_nir.npy")
        post_swir = np.load(synthetic_dir / "post_swir.npy")
        pixel_size_m = 10.0
    else:
        # Real mode: discover Sentinel-2 scenes and group into pre/post.
        # _load_sentinel2_pre_post raises FileNotFoundError if pre+post are not
        # both available — that matches the existing "refuse to fabricate"
        # contract and is required by the wildfire loader regression test.
        pre_nir, pre_swir, post_nir, post_swir, pixel_size_m, grid, cloud_mask = (
            _load_sentinel2_pre_post(
                input_path,
                event_date=event_date,
                bbox=bbox,
            )
        )

    # Execute algorithm
    result = algorithm.execute(
        nir_pre=pre_nir,
        swir_pre=pre_swir,
        nir_post=post_nir,
        swir_post=post_swir,
        pixel_size_m=pixel_size_m,
        cloud_mask=cloud_mask,
    )
    statistics = dict(result.statistics)
    if cloud_mask is not None:
        statistics["cloud_masked_percent"] = float(100.0 * cloud_mask.mean())
        logger.info(
            "Cloud and shadow masked from SCL: %.1f%% of the AOI",
            statistics["cloud_masked_percent"],
        )
    statistics.update(_no_data_statistics(grid))

    # Save outputs
    np.save(output_path / "dnbr.npy", result.dnbr_map)
    np.save(output_path / "burn_severity.npy", result.burn_severity)
    np.save(output_path / "burn_extent.npy", result.burn_extent)
    np.save(output_path / "confidence.npy", result.confidence_raster)

    # Persist the analyze-stage grid so the export stage can produce properly
    # georeferenced GeoTIFFs / GeoJSONs. Synthetic runs leave grid
    # unset — the export stage falls back to its bbox-based placeholder.
    if grid is not None:
        with open(output_path / "grid.json", "w") as f:
            json.dump(grid, f, indent=2)

    return {
        "success": True,
        "statistics": statistics,
    }


def _analyze_storm(
    algorithm,
    input_path: Path,
    output_path: Path,
    raster_files: List[Path],
    npy_files: List[Path],
    event_date: Optional[datetime] = None,
    bbox: Optional[str] = None,
) -> Dict[str, Any]:
    """Run storm damage analysis on real or synthetic data."""
    synthetic_dir = input_path / "synthetic"
    grid: Optional[Dict[str, Any]] = None
    cloud_mask: Optional[np.ndarray] = None
    algo_class = type(algorithm).__name__

    if (synthetic_dir / "pre_optical.npy").exists():
        # Synthetic path: a single-band optical image stands in for everything.
        pre_optical = np.load(synthetic_dir / "pre_optical.npy").astype(np.float32)
        post_optical = np.load(synthetic_dir / "post_optical.npy").astype(np.float32)
        pixel_size_m = 10.0

        if algo_class == "StructuralDamageAssessment":
            result = algorithm.execute(
                optical_pre=pre_optical,
                optical_post=post_optical,
                pixel_size_m=pixel_size_m,
            )
        else:
            # WindDamageDetection (default): synthetic optical doubles as both
            # red and NIR so the NDVI-based detector still produces a result.
            result = algorithm.execute(
                red_pre=pre_optical,
                nir_pre=pre_optical,
                red_post=post_optical,
                nir_post=post_optical,
                pixel_size_m=pixel_size_m,
            )
    else:
        # Real mode: load Sentinel-2 red + NIR (+ optional blue) per scene.
        bands, pixel_size_m, grid, cloud_mask = _load_sentinel2_storm_pre_post(
            input_path, event_date=event_date, bbox=bbox,
        )

        if algo_class == "StructuralDamageAssessment":
            # Structural damage assesses a single optical band; red is the most
            # informative for built-up areas under Sentinel-2 10 m bands. It
            # takes no cloud mask but treats NaN as no data.
            result = algorithm.execute(
                optical_pre=np.where(cloud_mask, np.nan, bands["red_pre"]),
                optical_post=np.where(cloud_mask, np.nan, bands["red_post"]),
                pixel_size_m=pixel_size_m,
            )
        else:
            kwargs: Dict[str, Any] = {
                "red_pre": bands["red_pre"],
                "nir_pre": bands["nir_pre"],
                "red_post": bands["red_post"],
                "nir_post": bands["nir_post"],
                "cloud_mask": cloud_mask,
                "pixel_size_m": pixel_size_m,
            }
            if bands["blue_pre"] is not None and bands["blue_post"] is not None:
                kwargs["blue_pre"] = bands["blue_pre"]
                kwargs["blue_post"] = bands["blue_post"]
            result = algorithm.execute(**kwargs)

    # Extract the damage mask under whatever name the result uses.
    damage = getattr(result, "damage_extent", None)
    if damage is None:
        damage = getattr(result, "damage_map", None)
    if damage is None:
        damage = getattr(result, "damage_mask", None)
    if damage is None:
        raise AttributeError(
            f"Storm algorithm {algo_class} returned a result without a known "
            "damage attribute (damage_extent/damage_map/damage_mask)."
        )

    confidence = getattr(result, "confidence_raster", None)
    if confidence is None:
        confidence = getattr(result, "confidence_map", None)

    if confidence is not None and cloud_mask is not None:
        # Unseen is no data, not zero confidence: validate's coverage check
        # counts NaN, so a box under cloud cannot pass as fully observed.
        confidence = np.where(cloud_mask, np.nan, np.asarray(confidence, dtype=np.float32))

    np.save(output_path / "damage_extent.npy", np.asarray(damage))
    if confidence is not None:
        np.save(output_path / "confidence.npy", np.asarray(confidence))

    statistics = dict(result.statistics)
    if cloud_mask is not None:
        statistics["cloud_masked_percent"] = float(100.0 * cloud_mask.mean())
        logger.info(
            "Cloud and shadow masked from SCL: %.1f%% of the AOI",
            statistics["cloud_masked_percent"],
        )
    statistics.update(_no_data_statistics(grid))

    if grid is not None:
        with open(output_path / "grid.json", "w") as f:
            json.dump(grid, f, indent=2)

    return {
        "success": True,
        "statistics": statistics,
    }


def _run_synthetic_analysis(
    input_path: Path,
    output_path: Path,
    event_type: str,
    algorithm: str,
) -> Dict[str, Any]:
    """Run analysis using synthetic data generation."""
    size = 512
    np.random.seed(42)

    if event_type.lower() == "flood":
        # Generate flood extent
        flood_extent = np.zeros((size, size), dtype=np.uint8)
        flood_extent[:, int(size*0.7):] = 1
        flood_extent[int(size*0.4):int(size*0.6), int(size*0.2):int(size*0.8)] = 1

        confidence = np.where(flood_extent == 1, 0.85, 0.95).astype(np.float32)
        confidence += np.random.normal(0, 0.05, (size, size))
        confidence = np.clip(confidence, 0, 1)

        np.save(output_path / "flood_extent.npy", flood_extent)
        np.save(output_path / "confidence.npy", confidence)

        flood_pixels = int(np.sum(flood_extent))
        flood_area_ha = flood_pixels * (10.0 ** 2) / 10000

        return {
            "success": True,
            "statistics": {
                "flood_pixels": flood_pixels,
                "flood_area_ha": flood_area_ha,
                "flood_percent": 100 * flood_pixels / (size * size),
                "mean_confidence": float(np.mean(confidence[flood_extent == 1])),
            }
        }

    elif event_type.lower() == "wildfire":
        # Generate burn severity
        burn_mask = np.zeros((size, size), dtype=bool)
        burn_mask[int(size*0.2):int(size*0.7), int(size*0.3):int(size*0.8)] = True

        dnbr = np.random.normal(0.1, 0.05, (size, size)).astype(np.float32)
        dnbr[burn_mask] = np.random.normal(0.5, 0.15, np.sum(burn_mask))

        severity = np.zeros((size, size), dtype=np.uint8)
        severity[dnbr > 0.1] = 1
        severity[dnbr > 0.27] = 2
        severity[dnbr > 0.44] = 3
        severity[dnbr > 0.66] = 4

        np.save(output_path / "dnbr.npy", dnbr)
        np.save(output_path / "burn_severity.npy", severity)
        np.save(output_path / "burn_extent.npy", burn_mask.astype(np.uint8))

        burned_pixels = int(np.sum(burn_mask))
        burned_area_ha = burned_pixels * (10.0 ** 2) / 10000

        return {
            "success": True,
            "statistics": {
                "burned_pixels": burned_pixels,
                "burned_area_ha": burned_area_ha,
                "mean_dnbr": float(np.mean(dnbr[burn_mask])),
                "high_severity_percent": 100 * np.sum(severity == 4) / burned_pixels if burned_pixels > 0 else 0,
            }
        }

    else:  # storm
        damage = np.random.random((size, size)) > 0.7
        np.save(output_path / "damage_extent.npy", damage.astype(np.uint8))

        damaged_pixels = int(np.sum(damage))
        return {
            "success": True,
            "statistics": {
                "damaged_pixels": damaged_pixels,
                "damaged_area_ha": damaged_pixels * (10.0 ** 2) / 10000,
                "damage_percent": 100 * damaged_pixels / (size * size),
            }
        }


def run_validate(
    input_path: Path,
    output_path: Path,
    data_mode: str = DATA_MODE_REAL,
) -> Dict[str, Any]:
    """Run validation stage - real QC checks or synthetic scoring."""
    import random

    if data_mode == DATA_MODE_REAL:
        # Real mode: fail hard if validation fails - never use synthetic scores
        from cli.commands.validate import run_quality_checks

        # Run actual quality checks
        check_results = run_quality_checks(
            input_path=input_path,
            checks=["spatial_coherence", "value_range", "coverage"],
        )

        # Calculate overall score
        scores = [r.get("score", 0) for r in check_results]
        score = sum(scores) / len(scores) if scores else 0.0
        passed = all(r.get("passed", False) for r in check_results)

        report = {
            "score": score,
            "passed": passed,
            "checks": check_results,
            "mode": "real",
        }

        with open(output_path / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    elif data_mode == DATA_MODE_SYNTHETIC:
        # Synthetic validation - only used when --synthetic flag is set
        score = random.uniform(0.75, 0.98)
        passed = score >= 0.7

        report = {
            "score": score,
            "passed": passed,
            "checks": [
                {"check_id": "spatial_coherence", "passed": True, "score": random.uniform(0.8, 1.0)},
                {"check_id": "value_range", "passed": True, "score": random.uniform(0.85, 1.0)},
                {"check_id": "coverage", "passed": True, "score": random.uniform(0.7, 0.95)},
            ],
            "mode": "synthetic",
        }

        with open(output_path / "validation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        return report

    else:
        raise ValueError(f"Unknown data mode: {data_mode}")


def run_export(
    input_path: Path,
    output_path: Path,
    formats: str,
    data_mode: str = DATA_MODE_REAL,
    bbox: Optional[str] = None,
) -> Dict[str, Any]:
    """Run export stage - create real GeoTIFFs/GeoJSON or placeholders."""
    output_path.mkdir(parents=True, exist_ok=True)

    format_list = [f.strip().lower() for f in formats.split(",")]
    exported = []

    # Parse bbox for georeferencing
    bbox_coords = [-180, -90, 180, 90]  # Default global
    if bbox:
        try:
            bbox_coords = [float(x) for x in bbox.split(",")]
        except ValueError:
            pass

    # Find result arrays
    result_arrays = {}
    for npy_file in input_path.glob("*.npy"):
        name = npy_file.stem
        result_arrays[name] = np.load(npy_file)

    if not result_arrays:
        logger.warning("No result arrays found for export")

    # Prefer the analyze stage's grid.json sidecar over the bbox
    # placeholder so exported products land at real-world coordinates.
    grid = _load_grid_info(input_path)

    for fmt in format_list:
        try:
            if fmt == "geotiff":
                _export_geotiff(result_arrays, output_path, bbox_coords, data_mode, grid=grid)
                exported.append(fmt)

            elif fmt == "geojson":
                _export_geojson(result_arrays, output_path, bbox_coords, data_mode, grid=grid)
                exported.append(fmt)

            elif fmt == "png":
                _export_png(result_arrays, output_path)
                exported.append(fmt)

            elif fmt == "pdf":
                _export_pdf(result_arrays, output_path, bbox_coords)
                exported.append(fmt)

        except Exception as e:
            logger.warning(f"Export to {fmt} failed: {e}")

    return {"formats": exported, "path": str(output_path)}


def _severity_nodata() -> int:
    """The burn severity value that means no valid observation."""
    from core.analysis.library.baseline.wildfire.nbr_differenced import (
        DifferencedNBRAlgorithm,
    )

    return DifferencedNBRAlgorithm.SEVERITY_NODATA


def _export_geotiff(
    result_arrays: Dict[str, np.ndarray],
    output_path: Path,
    bbox: List[float],
    data_mode: str,
    grid: Optional[Dict[str, Any]] = None,
) -> None:
    """Export results as GeoTIFFs."""
    try:
        import rasterio
        from rasterio.transform import from_bounds, Affine
        from rasterio.crs import CRS
    except ImportError:
        logger.warning("rasterio not available, creating placeholder GeoTIFFs")
        for name in result_arrays:
            (output_path / f"{name}.tif").write_bytes(b"")
        return

    # Prefer the analyze-stage grid (real CRS + transform of the source scene)
    # over the bbox-based placeholder. Falls back to the placeholder for
    # synthetic runs or when the sidecar is missing.
    grid_crs = None
    grid_transform = None
    grid_shape: Optional[tuple] = None
    if grid:
        try:
            crs_str = grid.get("crs") or grid.get("crs_wkt")
            if crs_str:
                grid_crs = CRS.from_string(crs_str)
            t = grid.get("transform")
            if t and len(t) == 6:
                grid_transform = Affine(*[float(x) for x in t])
            s = grid.get("shape")
            if s and len(s) == 2:
                grid_shape = (int(s[0]), int(s[1]))
        except Exception as e:
            logger.warning(f"Could not deserialize grid.json sidecar: {e}")
            grid_crs = None
            grid_transform = None

    placeholder_crs = CRS.from_epsg(4326)

    for name, data in result_arrays.items():
        if data.ndim != 2:
            continue

        output_file = output_path / f"{name}.tif"

        if grid_crs is not None and grid_transform is not None and (
            grid_shape is None or grid_shape == data.shape
        ):
            crs = grid_crs
            transform = grid_transform
        else:
            if grid_crs is not None and grid_shape and grid_shape != data.shape:
                logger.warning(
                    f"grid.json shape {grid_shape} does not match {name}.npy "
                    f"shape {data.shape}; falling back to bbox placeholder."
                )
            crs = placeholder_crs
            transform = from_bounds(
                bbox[0], bbox[1], bbox[2], bbox[3], data.shape[1], data.shape[0]
            )

        # Determine dtype and the value that marks no observation (cloud,
        # shadow, outside the scene) so readers do not take it as a value.
        nodata = None
        if data.dtype == np.bool_:
            dtype = 'uint8'
            data = data.astype(np.uint8)
        elif np.issubdtype(data.dtype, np.floating):
            dtype = 'float32'
            data = data.astype(np.float32)
            nodata = float("nan")
        else:
            dtype = str(data.dtype)
            if name == "burn_severity":
                nodata = _severity_nodata()

        with rasterio.open(
            output_file, 'w', driver='GTiff',
            height=data.shape[0], width=data.shape[1],
            count=1, dtype=dtype, crs=crs, transform=transform,
            compress='lzw', nodata=nodata,
        ) as dst:
            dst.write(data, 1)

        logger.info(f"Exported {output_file}")


def _export_geojson(
    result_arrays: Dict[str, np.ndarray],
    output_path: Path,
    bbox: List[float],
    data_mode: str,
    grid: Optional[Dict[str, Any]] = None,
) -> None:
    """Export results as GeoJSON vectors."""
    from scipy import ndimage

    # Find extent/mask arrays
    extent_array = None
    for name in ["flood_extent", "burn_extent", "damage_extent"]:
        if name in result_arrays:
            extent_array = result_arrays[name]
            break

    if extent_array is None:
        # Create empty GeoJSON
        geojson = {"type": "FeatureCollection", "features": []}
        with open(output_path / "result.geojson", "w") as f:
            json.dump(geojson, f)
        return

    # Prefer the analyze-stage grid: when present, polygonize the
    # mask with rasterio.features.shapes against the real affine transform and
    # reproject each polygon to EPSG:4326. Falls back to the legacy bbox-based
    # bounding-box projection otherwise.
    if grid:
        try:
            features = _polygonize_with_grid(extent_array, grid)
            geojson = {
                "type": "FeatureCollection",
                "name": "Analysis Results",
                "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                "features": features,
            }
            with open(output_path / "result.geojson", "w") as f:
                json.dump(geojson, f, indent=2)
            logger.info(
                f"Exported GeoJSON with {len(features)} features (grid sidecar)"
            )
            return
        except Exception as e:
            logger.warning(
                f"grid-based GeoJSON export failed ({e}); falling back to bbox."
            )

    # Label connected regions
    labeled, num_features = ndimage.label(extent_array > 0)
    size = extent_array.shape

    geojson = {
        "type": "FeatureCollection",
        "name": "Analysis Results",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": []
    }

    for i in range(1, min(num_features + 1, 100)):
        region_mask = labeled == i
        area_ha = np.sum(region_mask) * (10.0 ** 2) / 10000

        if area_ha < 0.5:
            continue

        rows, cols = np.where(region_mask)
        if len(rows) == 0:
            continue

        # Convert to geographic coordinates
        min_lon = bbox[0] + (cols.min() / size[1]) * (bbox[2] - bbox[0])
        max_lon = bbox[0] + (cols.max() / size[1]) * (bbox[2] - bbox[0])
        min_lat = bbox[1] + (1 - rows.max() / size[0]) * (bbox[3] - bbox[1])
        max_lat = bbox[1] + (1 - rows.min() / size[0]) * (bbox[3] - bbox[1])

        feature = {
            "type": "Feature",
            "properties": {
                "id": i,
                "area_ha": round(area_ha, 2),
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [min_lon, min_lat], [max_lon, min_lat],
                    [max_lon, max_lat], [min_lon, max_lat],
                    [min_lon, min_lat]
                ]]
            }
        }
        geojson["features"].append(feature)

    with open(output_path / "result.geojson", "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info(f"Exported GeoJSON with {len(geojson['features'])} features")


def _polygonize_with_grid(
    extent_array: np.ndarray, grid: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Polygonize a 0/1 mask using the analyze-stage grid + reproject to WGS84."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.features import shapes
    from rasterio.transform import Affine
    from rasterio.warp import transform_geom

    crs_str = grid.get("crs") or grid.get("crs_wkt")
    src_crs = CRS.from_string(crs_str) if crs_str else None
    t = grid["transform"]
    affine = Affine(*[float(x) for x in t])
    pixel_area_m2 = float(abs(affine.a) * abs(affine.e))

    mask = (extent_array > 0).astype(np.uint8)
    features: List[Dict[str, Any]] = []
    feature_id = 0
    for geom, val in shapes(mask, mask=(mask == 1), transform=affine):
        if int(val) != 1:
            continue
        feature_id += 1
        # Approximate area from the source mask using the polygon's coverage
        # is non-trivial; instead use the simpler per-feature pixel count below.
        if src_crs is not None and src_crs.to_string() != "EPSG:4326":
            geom_out = transform_geom(src_crs, "EPSG:4326", geom)
        else:
            geom_out = geom
        features.append({
            "type": "Feature",
            "properties": {
                "id": feature_id,
                "area_ha": None,
            },
            "geometry": geom_out,
        })

    # Backfill per-feature area_ha by counting source-CRS pixels per connected
    # component. Skip if it would dominate runtime — small masks only.
    if pixel_area_m2 > 0 and mask.size <= 2048 * 2048:
        from scipy import ndimage as _nd

        labeled, _ = _nd.label(mask)
        if labeled.max() == len(features):
            for i, feat in enumerate(features, start=1):
                pixel_count = int(np.sum(labeled == i))
                feat["properties"]["area_ha"] = round(
                    pixel_count * pixel_area_m2 / 10000.0, 2
                )

    return features


def _export_png(result_arrays: Dict[str, np.ndarray], output_path: Path) -> None:
    """Export results as PNG visualizations."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping PNG export")
        return

    for name, data in result_arrays.items():
        if data.ndim != 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        if "extent" in name or "mask" in name:
            im = ax.imshow(data, cmap='Blues')
        elif "severity" in name:
            im = ax.imshow(
                np.ma.masked_equal(data, _severity_nodata()),
                cmap='RdYlGn_r', vmin=0, vmax=4,
            )
        elif "confidence" in name:
            im = ax.imshow(data, cmap='viridis', vmin=0, vmax=1)
        elif "dnbr" in name:
            im = ax.imshow(data, cmap='RdYlGn_r', vmin=-0.5, vmax=1.0)
        else:
            im = ax.imshow(data, cmap='viridis')

        ax.set_title(name.replace("_", " ").title())
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(output_path / f"{name}.png", dpi=150)
        plt.close()

        logger.info(f"Exported {name}.png")


def _export_pdf(
    result_arrays: Dict[str, np.ndarray],
    output_path: Path,
    bbox: List[float],
) -> None:
    """Export results as PDF report."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        logger.warning("matplotlib not available, skipping PDF export")
        return

    pdf_path = output_path / "analysis_report.pdf"

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, "FirstLight Analysis Report", ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ha='center', fontsize=12)
        fig.text(0.5, 0.4, f"Bounding Box: {bbox}", ha='center', fontsize=10)
        pdf.savefig(fig)
        plt.close()

        # Result pages
        for name, data in result_arrays.items():
            if data.ndim != 2:
                continue

            fig, ax = plt.subplots(figsize=(11, 8.5))

            if "extent" in name:
                im = ax.imshow(data, cmap='Blues', extent=bbox)
            elif "severity" in name:
                im = ax.imshow(data, cmap='RdYlGn_r', vmin=0, vmax=4, extent=bbox)
            else:
                im = ax.imshow(data, cmap='viridis', extent=bbox)

            ax.set_title(name.replace("_", " ").title(), fontsize=14)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    logger.info(f"Exported PDF report to {pdf_path}")
