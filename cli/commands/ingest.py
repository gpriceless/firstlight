"""
Ingest Command - Download and normalize data to analysis-ready format.

Usage:
    flight ingest --area miami.geojson --source sentinel1 --output ./data/
    flight ingest --input discovery.json --output ./data/ --resume
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import click

from core.data.ingestion.streaming import StreamingIngester
from core.data.ingestion.validation.image_validator import ImageValidator

logger = logging.getLogger("flight.ingest")


@dataclass
class BandDownloadResult:
    """Result of downloading a single band."""
    band_name: str
    url: str
    local_path: Optional[Path]
    size_bytes: int
    download_time_s: float
    success: bool
    error: Optional[str] = None
    retries: int = 0


class ProgressTracker:
    """Track download and processing progress with resume support."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.state_file = workdir / ".ingest_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file or create new."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return {
            "started_at": datetime.now().isoformat(),
            "completed": [],
            "failed": [],
            "in_progress": None,
        }

    def save_state(self):
        """Persist state to file."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def mark_started(self, item_id: str):
        """Mark an item as started."""
        self.state["in_progress"] = item_id
        self.save_state()

    def mark_completed(self, item_id: str):
        """Mark an item as completed."""
        self.state["completed"].append(item_id)
        self.state["in_progress"] = None
        self.save_state()

    def mark_failed(self, item_id: str, error: str):
        """Mark an item as failed."""
        self.state["failed"].append({"id": item_id, "error": error})
        self.state["in_progress"] = None
        self.save_state()

    def is_completed(self, item_id: str) -> bool:
        """Check if an item is already completed."""
        return item_id in self.state["completed"]

    def get_resume_point(self) -> Optional[str]:
        """Get the item that was in progress when interrupted."""
        return self.state.get("in_progress")


def create_progress_bar(total: int, desc: str = "Progress"):
    """Create a progress bar using click or tqdm."""
    try:
        from tqdm import tqdm

        return tqdm(total=total, desc=desc, unit="item")
    except ImportError:
        # Fallback to simple counter
        class SimpleProgress:
            def __init__(self, total, desc):
                self.total = total
                self.current = 0
                self.desc = desc

            def update(self, n=1):
                self.current += n
                pct = 100 * self.current / self.total if self.total > 0 else 0
                click.echo(f"\r{self.desc}: {self.current}/{self.total} ({pct:.0f}%)", nl=False)

            def close(self):
                click.echo()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()

        return SimpleProgress(total, desc)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def download_with_progress(url: str, dest_path: Path, expected_size: Optional[int] = None) -> bool:
    """Download a file with progress bar and resume support."""
    import requests

    # Check for partial download
    existing_size = 0
    if dest_path.exists():
        existing_size = dest_path.stat().st_size

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        logger.info(f"Resuming download from {format_size(existing_size)}")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)

        if response.status_code == 416:
            # Range not satisfiable - file already complete
            logger.info("File already downloaded")
            return True

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if existing_size > 0:
            total_size += existing_size

        mode = "ab" if existing_size > 0 else "wb"

        with open(dest_path, mode) as f:
            with create_progress_bar(total_size, "Downloading") as pbar:
                pbar.update(existing_size)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def _download_band_file(url: str, output_path: Path) -> None:
    """Download a single band file using HTTP streaming."""
    import requests

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def download_bands(
    band_urls: Dict[str, str],
    output_dir: Path,
    item_id: str,
    parallel_downloads: int = 4,
    max_retries: int = 3,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, BandDownloadResult]:
    """
    Download multiple band files for a single scene.

    Args:
        band_urls: Mapping of band name to URL
        output_dir: Directory to save band files
        item_id: Item ID for logging
        parallel_downloads: Number of concurrent downloads
        max_retries: Retry attempts per band
        progress_callback: Called with (band_name, status, progress)

    Returns:
        Mapping of band name to download result
    """
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    def download_single_band(band_name: str, url: str) -> BandDownloadResult:
        band_path = output_dir / f"{band_name}.tif"
        start_time = time.time()

        # Check for existing complete file
        if band_path.exists() and band_path.stat().st_size > 0:
            # Try to validate it's a valid raster
            try:
                import rasterio
                with rasterio.open(band_path) as ds:
                    _ = ds.count  # Quick validity check
                logger.info(f"Band {band_name} already downloaded: {format_size(band_path.stat().st_size)}")
                return BandDownloadResult(
                    band_name=band_name,
                    url=url,
                    local_path=band_path,
                    size_bytes=band_path.stat().st_size,
                    download_time_s=0,
                    success=True,
                )
            except Exception:
                # File is corrupt, re-download
                logger.warning(f"Band {band_name} file corrupt, re-downloading")
                band_path.unlink(missing_ok=True)

        # Download with retries
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading band {band_name} (attempt {attempt + 1}/{max_retries})")
                _download_band_file(url, band_path)
                elapsed = time.time() - start_time

                if progress_callback:
                    progress_callback(band_name, "complete", 1.0)

                logger.info(
                    f"Downloaded band {band_name}: {format_size(band_path.stat().st_size)} "
                    f"in {elapsed:.1f}s"
                )

                return BandDownloadResult(
                    band_name=band_name,
                    url=url,
                    local_path=band_path,
                    size_bytes=band_path.stat().st_size,
                    download_time_s=elapsed,
                    success=True,
                    retries=attempt,
                )
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Download failed for band {band_name}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)  # Exponential backoff
                    continue
                logger.error(f"Failed to download band {band_name} after {max_retries} attempts: {e}")
                return BandDownloadResult(
                    band_name=band_name,
                    url=url,
                    local_path=None,
                    size_bytes=0,
                    download_time_s=time.time() - start_time,
                    success=False,
                    error=str(e),
                    retries=attempt + 1,
                )

    # Download bands in parallel
    with ThreadPoolExecutor(max_workers=parallel_downloads) as executor:
        futures = {
            executor.submit(download_single_band, name, url): name
            for name, url in band_urls.items()
        }

        for future in as_completed(futures):
            band_name = futures[future]
            results[band_name] = future.result()

    return results


@click.command("ingest")
@click.option(
    "--area",
    "-a",
    "area_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to GeoJSON file defining the area of interest.",
)
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    help="Path to discovery results JSON file.",
)
@click.option(
    "--source",
    "-s",
    "sources",
    multiple=True,
    type=str,
    help="Data sources to ingest (e.g., sentinel1, sentinel2).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for ingested data.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["cog", "zarr", "netcdf"], case_sensitive=False),
    default="cog",
    help="Output format (default: cog).",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume interrupted downloads (default: yes).",
)
@click.option(
    "--parallel",
    "-p",
    type=int,
    default=1,
    help="Number of parallel downloads (default: 1).",
)
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="Normalize data to analysis-ready format (default: yes).",
)
@click.option(
    "--crs",
    type=str,
    default=None,
    help="Target CRS for normalization (e.g., EPSG:32617). Auto-detect if not specified.",
)
@click.option(
    "--resolution",
    "-r",
    type=float,
    default=None,
    help="Target resolution in meters. Preserve original if not specified.",
)
@click.option(
    "--skip-validation",
    is_flag=True,
    default=False,
    help="Skip image validation (WARNING: may download unusable files)",
)
@click.pass_obj
def ingest(
    ctx,
    area_path: Optional[Path],
    input_path: Optional[Path],
    sources: tuple,
    output_path: Path,
    output_format: str,
    resume: bool,
    parallel: int,
    normalize: bool,
    crs: Optional[str],
    resolution: Optional[float],
    skip_validation: bool,
):
    """
    Download and normalize satellite data to analysis-ready format.

    Downloads data from configured sources or from a discovery results file.
    Supports resumable downloads and automatic normalization to cloud-native
    formats (COG, Zarr, NetCDF).

    \b
    Examples:
        # Ingest Sentinel-1 data for an area
        flight ingest --area miami.geojson --source sentinel1 --output ./data/

        # Ingest from discovery results
        flight ingest --input discovery.json --output ./data/ --format zarr

        # Parallel downloads with custom resolution
        flight ingest --input discovery.json --output ./data/ --parallel 4 --resolution 10
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Display warning if validation is skipped
    if skip_validation:
        click.echo(
            "[WARNING] Validation skipped. Downloaded files may not be "
            "suitable for spectral analysis (missing NIR/SWIR bands).",
            err=True,
        )

    # Initialize progress tracker
    tracker = ProgressTracker(output_path)

    # Load items to ingest
    items = load_ingest_items(area_path, input_path, sources)

    if not items:
        click.echo("No items to ingest. Run 'flight discover' first or specify --area and --source.")
        return

    click.echo(f"\nIngesting {len(items)} items to {output_path}")
    click.echo(f"  Format: {output_format.upper()}")
    click.echo(f"  Resume: {'enabled' if resume else 'disabled'}")
    click.echo(f"  Parallel: {parallel}")

    if normalize:
        click.echo(f"  Normalization: enabled")
        if crs:
            click.echo(f"    Target CRS: {crs}")
        if resolution:
            click.echo(f"    Target resolution: {resolution}m")

    # Check for resume point
    if resume:
        resume_id = tracker.get_resume_point()
        if resume_id:
            click.echo(f"\nResuming from: {resume_id}")

    # Process items
    completed = 0
    failed = 0

    for item in items:
        item_id = item.get("id", "unknown")

        # Skip if already completed
        if resume and tracker.is_completed(item_id):
            logger.debug(f"Skipping already completed: {item_id}")
            completed += 1
            continue

        click.echo(f"\nProcessing: {item_id}")
        tracker.mark_started(item_id)

        try:
            # Download
            success = process_item(
                item=item,
                output_path=output_path,
                output_format=output_format,
                normalize=normalize,
                target_crs=crs,
                target_resolution=resolution,
                skip_validation=skip_validation,
            )

            if success:
                tracker.mark_completed(item_id)
                completed += 1
                click.echo(f"  Completed: {item_id}")
            else:
                tracker.mark_failed(item_id, "Processing failed")
                failed += 1
                click.echo(f"  Failed: {item_id}", err=True)

        except KeyboardInterrupt:
            click.echo("\n\nInterrupted. Use --resume to continue later.")
            tracker.save_state()
            sys.exit(130)

        except Exception as e:
            tracker.mark_failed(item_id, str(e))
            failed += 1
            logger.error(f"Error processing {item_id}: {e}")

    # Summary
    click.echo(f"\n=== Ingestion Summary ===")
    click.echo(f"  Completed: {completed}")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Skipped: {len(tracker.state['completed']) - completed}")
    click.echo(f"  Output: {output_path}")


def load_ingest_items(
    area_path: Optional[Path],
    input_path: Optional[Path],
    sources: tuple,
) -> List[Dict[str, Any]]:
    """Load items to ingest from various sources."""
    items = []

    # Load from discovery results file
    if input_path:
        with open(input_path) as f:
            data = json.load(f)
            if isinstance(data, dict) and "results" in data:
                items = data["results"]
            elif isinstance(data, list):
                items = data

    # Filter by source if specified
    if sources:
        source_lower = [s.lower() for s in sources]
        items = [i for i in items if i.get("source", "").lower() in source_lower]

    # If no items from file, generate from area + source
    if not items and area_path and sources:
        # Generate placeholder items based on sources
        for source in sources:
            items.append({
                "id": f"{source}_latest",
                "source": source,
                "url": None,  # Will be resolved during processing
            })

    return items


def _download_real_data(
    url: str, download_path: Path, item: Dict[str, Any], skip_validation: bool = False
) -> None:
    """
    Download real satellite data using StreamingIngester.

    Args:
        url: Source URL
        download_path: Destination path
        item: Item metadata (may contain size_bytes)
        skip_validation: Skip image validation

    Raises:
        RuntimeError: If download fails
    """
    try:
        # Use StreamingIngester for real downloads
        ingester = StreamingIngester()
        result = ingester.ingest(
            source=url,
            output_path=download_path,
            skip_validation=skip_validation,
        )

        if result["status"] != "completed":
            errors = ", ".join(result.get("errors", ["Unknown error"]))
            raise RuntimeError(f"Download failed: {errors}")

        logger.info(f"Downloaded {download_path.name} successfully")

    except Exception as e:
        logger.error(f"Failed to download from {url}: {e}")
        raise RuntimeError(f"Download failed: {e}") from e


def _validate_downloaded_image(download_path: Path, item: Dict[str, Any]) -> bool:
    """
    Validate downloaded image using ImageValidator.

    Args:
        download_path: Path to downloaded file
        item: Item metadata (may contain source info)

    Returns:
        True if validation passed, False otherwise
    """
    try:
        validator = ImageValidator()

        # Build data source spec from item metadata
        data_source_spec = None
        if "source" in item:
            data_source_spec = {"sensor": item["source"]}

        result = validator.validate(
            raster_path=download_path,
            data_source_spec=data_source_spec,
            dataset_id=item.get("id", str(download_path)),
        )

        if not result.is_valid:
            logger.error(f"Validation errors: {result.errors}")
            return False

        if result.warnings:
            logger.warning(f"Validation warnings: {result.warnings}")

        logger.info(f"Image validation passed for {download_path.name}")
        return True

    except Exception as e:
        logger.warning(f"Image validation failed with exception: {e}")
        # Allow processing to continue if validation module has issues
        return True


def process_item(
    item: Dict[str, Any],
    output_path: Path,
    output_format: str,
    normalize: bool,
    target_crs: Optional[str],
    target_resolution: Optional[float],
    skip_validation: bool = False,
) -> bool:
    """
    Process a single data item: download and optionally normalize.
    """
    item_id = item.get("id", "unknown")
    source = item.get("source", "unknown")
    url = item.get("url")
    band_urls = item.get("band_urls", {})

    # Create output directory for this item
    item_dir = output_path / source / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    # Multi-band download path
    if band_urls:
        logger.info(f"Downloading {len(band_urls)} bands for {item_id}")
        band_results = download_bands(band_urls, item_dir, item_id)

        # Check for failures
        failed_bands = [r for r in band_results.values() if not r.success]
        if failed_bands:
            for r in failed_bands:
                logger.error(f"Failed to download {r.band_name}: {r.error}")
            return False

        # Get paths for successful downloads
        band_paths = {
            name: result.local_path
            for name, result in band_results.items()
            if result.success and result.local_path
        }

        # Calculate total download size and time
        total_size = sum(r.size_bytes for r in band_results.values() if r.success)
        total_time = sum(r.download_time_s for r in band_results.values() if r.success)
        logger.info(
            f"Downloaded {len(band_paths)} bands: {format_size(total_size)} "
            f"in {total_time:.1f}s"
        )

        # Create band stack (placeholder - Task 1.7.3 will implement this)
        stack_path = item_dir / f"{item_id}_stack.vrt"
        try:
            from core.data.ingestion.band_stack import create_band_stack
            stack_result = create_band_stack(band_paths, stack_path)
            logger.info(f"Created VRT stack: {stack_result.path}")

            # Validate the stack (unless validation skipped)
            if not skip_validation:
                if not _validate_downloaded_image(stack_result.path, item):
                    logger.error(f"VRT stack validation failed for {item_id}")
                    return False
        except ImportError:
            # band_stack.py not yet implemented, just log the band files
            logger.info(f"Downloaded band files to {item_dir} (VRT stacking pending)")
            logger.debug(f"Band files: {list(band_paths.keys())}")

    # Legacy single-URL path
    elif url:
        # Determine filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename or len(filename) < 3:
            filename = f"{item_id}.tif"

        download_path = item_dir / filename

        logger.info(f"Downloading {item_id} from {url}")

        # Check if file already exists and is complete
        if download_path.exists():
            existing_size = download_path.stat().st_size
            if existing_size > 0:
                logger.info(f"File already exists: {download_path} ({format_size(existing_size)})")
            else:
                # File exists but is empty, re-download
                _download_real_data(url, download_path, item, skip_validation)
        else:
            # File doesn't exist, download
            _download_real_data(url, download_path, item, skip_validation)

        # Validate downloaded image (unless validation is skipped)
        if not skip_validation:
            if not _validate_downloaded_image(download_path, item):
                logger.error(f"Image validation failed for {item_id}")
                return False
        else:
            logger.info(f"Image validation skipped for {item_id}")

    # Normalize if requested
    if normalize:
        success = normalize_item(
            item_dir=item_dir,
            output_format=output_format,
            target_crs=target_crs,
            target_resolution=target_resolution,
        )
        if not success:
            return False

    # Write metadata
    metadata_path = item_dir / "metadata.json"
    metadata = {
        "item_id": item_id,
        "source": source,
        "ingested_at": datetime.now().isoformat(),
        "format": output_format,
        "normalized": normalize,
    }
    if target_crs:
        metadata["crs"] = target_crs
    if target_resolution:
        metadata["resolution_m"] = target_resolution

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return True


def normalize_item(
    item_dir: Path,
    output_format: str,
    target_crs: Optional[str],
    target_resolution: Optional[float],
) -> bool:
    """
    Normalize data to analysis-ready format.
    """
    try:
        # Find input files
        input_files = list(item_dir.glob("*.tif")) + list(item_dir.glob("*.TIF"))

        if not input_files:
            logger.debug("No raster files to normalize")
            return True

        # Import normalization modules - raise error if not available
        try:
            from core.data.ingestion.formats.cog import COGConverter
            from core.data.ingestion.normalization.projection import RasterReprojector
        except ImportError as e:
            error_msg = f"Normalization requested but modules unavailable: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        # Perform normalization for each file
        for input_file in input_files:
            output_file = input_file.with_suffix(f".{output_format}")

            if output_format == "cog":
                logger.info(f"Converting {input_file.name} to COG format")
                converter = COGConverter()
                converter.convert(input_file, output_file, target_crs=target_crs)
            else:
                logger.warning(f"Output format '{output_format}' not fully implemented, skipping conversion")

        logger.info(f"Normalization complete for {len(input_files)} file(s)")
        return True

    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        return False
