"""
Discover Command - Find available data for an area and time window.

Usage:
    flight discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood
    flight discover --bbox -80.5,25.5,-80.0,26.0 --start 2024-09-15 --end 2024-09-20 --event flood
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import click

logger = logging.getLogger("flight.discover")


class STACError(Exception):
    """Base exception for STAC-related errors."""
    pass


class NetworkError(STACError):
    """Network-related errors that can be retried."""
    pass


class QueryError(STACError):
    """Query parameter errors that should not be retried."""
    pass


def format_size(size_bytes: Optional[int]) -> str:
    """Format bytes to human-readable string."""
    if size_bytes is None:
        return "Unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def parse_date(date_str: str) -> datetime:
    """Parse date string in various formats."""
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise click.BadParameter(f"Cannot parse date: {date_str}")


def load_geometry(area_path: Optional[Path], bbox: Optional[str]) -> Dict[str, Any]:
    """Load geometry from file or bounding box string."""
    if area_path:
        if not area_path.exists():
            raise click.BadParameter(f"Area file not found: {area_path}")

        suffix = area_path.suffix.lower()
        if suffix == ".geojson" or suffix == ".json":
            with open(area_path) as f:
                geojson = json.load(f)
            # Extract geometry from GeoJSON
            if geojson.get("type") == "FeatureCollection":
                if geojson.get("features"):
                    return geojson["features"][0]["geometry"]
                raise click.BadParameter("GeoJSON FeatureCollection has no features")
            elif geojson.get("type") == "Feature":
                return geojson["geometry"]
            else:
                return geojson
        else:
            raise click.BadParameter(f"Unsupported area file format: {suffix}")

    elif bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise click.BadParameter(
                "Bounding box must have 4 values: min_lon,min_lat,max_lon,max_lat"
            )
        min_lon, min_lat, max_lon, max_lat = parts
        return {
            "type": "Polygon",
            "coordinates": [
                [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat],
                ]
            ],
        }
    else:
        raise click.BadParameter("Either --area or --bbox must be provided")


def format_table_row(columns: List[str], widths: List[int]) -> str:
    """Format a table row with proper spacing."""
    parts = []
    for col, width in zip(columns, widths):
        parts.append(str(col).ljust(width)[:width])
    return "  ".join(parts)


def extract_bbox_from_geometry(geometry: Dict[str, Any]) -> List[float]:
    """
    Extract bounding box from GeoJSON geometry.

    Args:
        geometry: GeoJSON geometry dict

    Returns:
        Bounding box as [west, south, east, north]
    """
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        # Flatten polygon coordinates
        all_coords = coords[0]  # Outer ring
    elif geom_type == "MultiPolygon":
        # Flatten all polygon coordinates
        all_coords = []
        for polygon in coords:
            all_coords.extend(polygon[0])
    elif geom_type == "Point":
        # Single point - create small bbox around it
        lon, lat = coords
        return [lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01]
    elif geom_type == "LineString":
        all_coords = coords
    else:
        # Fallback for unknown types
        logger.warning(f"Unknown geometry type: {geom_type}, using default bbox")
        return [-180, -90, 180, 90]

    # Calculate bbox
    lons = [c[0] for c in all_coords]
    lats = [c[1] for c in all_coords]

    return [min(lons), min(lats), max(lons), max(lats)]


@click.command("discover")
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
    required=True,
    type=str,
    help="Start date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    "-e",
    "end_date",
    required=True,
    type=str,
    help="End date (YYYY-MM-DD).",
)
@click.option(
    "--event",
    "-t",
    "event_type",
    type=click.Choice(["flood", "wildfire", "storm"], case_sensitive=False),
    required=True,
    help="Event type to optimize data discovery for.",
)
@click.option(
    "--source",
    "-S",
    "sources",
    multiple=True,
    type=str,
    help="Filter by specific data sources (e.g., sentinel1, sentinel2, landsat).",
)
@click.option(
    "--max-cloud",
    type=float,
    default=30.0,
    help="Maximum cloud cover percentage for optical data (default: 30).",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output file path for results (JSON format).",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["table", "json", "csv"], case_sensitive=False),
    default="table",
    help="Output format (default: table).",
)
@click.pass_obj
def discover(
    ctx,
    area_path: Optional[Path],
    bbox: Optional[str],
    start_date: str,
    end_date: str,
    event_type: str,
    sources: tuple,
    max_cloud: float,
    output_path: Optional[Path],
    output_format: str,
):
    """
    Discover available satellite data for an area and time window.

    Queries multiple data catalogs (STAC, WMS/WCS) to find available
    satellite imagery matching the specified criteria. Results include
    cloud cover, resolution, and availability information.

    \b
    Examples:
        # Discover flood-relevant data for Miami
        flight discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 --event flood

        # Use bounding box instead of file
        flight discover --bbox -80.5,25.5,-80.0,26.0 --start 2024-09-15 --end 2024-09-20 --event flood

        # Filter by source and output as JSON
        flight discover --area miami.geojson --start 2024-09-15 --end 2024-09-20 \\
            --event flood --source sentinel1 --format json --output results.json
    """
    # Parse inputs
    start = parse_date(start_date)
    end = parse_date(end_date)
    geometry = load_geometry(area_path, bbox)

    if start > end:
        raise click.BadParameter("Start date must be before end date")

    click.echo(f"\nDiscovering data for {event_type} event...")
    click.echo(f"  Time window: {start.date()} to {end.date()}")
    click.echo(f"  Max cloud cover: {max_cloud}%")
    if sources:
        click.echo(f"  Sources: {', '.join(sources)}")

    # Perform discovery
    results = perform_discovery(
        geometry=geometry,
        start=start,
        end=end,
        event_type=event_type,
        sources=list(sources) if sources else None,
        max_cloud=max_cloud,
        config=ctx.config if ctx else {},
    )

    # Output results
    if output_format == "json":
        output_json(results, output_path)
    elif output_format == "csv":
        output_csv(results, output_path)
    else:
        output_table(results)

    # Summary
    click.echo(f"\nFound {len(results)} datasets")
    if output_path:
        click.echo(f"Results saved to: {output_path}")


def perform_discovery(
    geometry: Dict[str, Any],
    start: datetime,
    end: datetime,
    event_type: str,
    sources: Optional[List[str]],
    max_cloud: float,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Perform data discovery across configured catalogs.

    This function queries STAC catalogs and other data sources
    to find available satellite imagery.

    Raises:
        NetworkError: For transient network failures (will be retried)
        QueryError: For invalid query parameters (will not be retried)
        STACError: For other STAC-related errors
    """
    # Extract bounding box from geometry
    bbox = extract_bbox_from_geometry(geometry)

    # Import STAC client (raise clear error if not available)
    try:
        from core.data.discovery.stac_client import discover_data
    except ImportError as e:
        raise STACError(
            "STAC client not available. Install pystac-client: pip install pystac-client"
        ) from e

    # Perform discovery with retry logic for transient failures
    results = _discover_with_retry(
        bbox=bbox,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        event_type=event_type,
        max_cloud_cover=max_cloud,
        discover_func=discover_data,
    )

    logger.info(f"Found {len(results)} datasets via STAC")

    # Filter by source if specified
    if sources:
        source_lower = [s.lower() for s in sources]
        results = [r for r in results if r["source"].lower() in source_lower]

    # Sort by priority and datetime
    def sort_key(r):
        priority_order = {"primary": 0, "secondary": 1, "ancillary": 2}
        return (priority_order.get(r.get("priority", "secondary"), 1), r.get("datetime", ""))

    results.sort(key=sort_key)

    return results


def _discover_with_retry(
    bbox: List[float],
    start_date: str,
    end_date: str,
    event_type: str,
    max_cloud_cover: float,
    discover_func,
    max_attempts: int = 3,
) -> List[Dict[str, Any]]:
    """
    Execute STAC discovery with exponential backoff retry for transient failures.

    Args:
        bbox: Bounding box [west, south, east, north]
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        event_type: Event type (flood, wildfire, storm)
        max_cloud_cover: Maximum cloud cover percentage
        discover_func: The discovery function to call
        max_attempts: Maximum retry attempts (default: 3)

    Returns:
        List of discovery results

    Raises:
        NetworkError: After exhausting retries for network failures
        QueryError: For invalid query parameters (no retry)
        STACError: For other errors (no retry)
    """
    for attempt in range(max_attempts):
        try:
            results = discover_func(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                event_type=event_type,
                max_cloud_cover=max_cloud_cover,
            )

            # Empty results are valid - return as-is
            return results

        except Exception as e:
            error_msg = str(e).lower()

            # Categorize the error
            is_network_error = any(
                keyword in error_msg
                for keyword in [
                    "timeout",
                    "connection",
                    "dns",
                    "network",
                    "unreachable",
                    "refused",
                    "reset",
                ]
            )

            is_query_error = any(
                keyword in error_msg
                for keyword in [
                    "invalid",
                    "parameter",
                    "validation",
                    "bbox",
                    "collection",
                ]
            )

            # Query errors should not be retried
            if is_query_error:
                raise QueryError(
                    f"Invalid query parameters: {e}\n"
                    f"Check that bbox={bbox}, dates={start_date}/{end_date}, and event_type={event_type} are valid."
                ) from e

            # Network errors - retry with exponential backoff
            if is_network_error:
                if attempt < max_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Network error on attempt {attempt + 1}/{max_attempts}: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise NetworkError(
                        f"Network error after {max_attempts} attempts: {e}\n"
                        f"Check your internet connection and try again."
                    ) from e

            # Other errors - don't retry, raise immediately
            raise STACError(f"STAC discovery failed: {e}") from e

    # Should never reach here, but just in case
    return []


def output_table(results: List[Dict[str, Any]]):
    """Output results as a formatted table."""
    if not results:
        click.echo("\nNo datasets found matching criteria.")
        return

    click.echo("\n")
    widths = [40, 12, 20, 8, 8, 10]
    headers = ["ID", "Source", "Date/Time", "Cloud%", "Res(m)", "Size"]
    click.echo(format_table_row(headers, widths))
    click.echo("-" * (sum(widths) + len(widths) * 2))

    for r in results:
        dt_str = r.get("datetime", "")
        if dt_str and "T" in dt_str:
            dt_str = dt_str.split("T")[0]

        cloud = r.get("cloud_cover")
        cloud_str = f"{cloud:.0f}" if cloud is not None else "N/A"

        res = r.get("resolution_m")
        res_str = f"{res:.0f}" if res is not None else "N/A"

        size_str = format_size(r.get("size_bytes"))

        row = [
            r.get("id", "unknown"),
            r.get("source", "unknown"),
            dt_str,
            cloud_str,
            res_str,
            size_str,
        ]
        click.echo(format_table_row(row, widths))


def output_json(results: List[Dict[str, Any]], output_path: Optional[Path]):
    """Output results as JSON."""
    output = {
        "count": len(results),
        "results": results,
    }

    json_str = json.dumps(output, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)
    else:
        click.echo(json_str)


def output_csv(results: List[Dict[str, Any]], output_path: Optional[Path]):
    """Output results as CSV."""
    import csv
    import io

    if not results:
        click.echo("No results to export.")
        return

    fields = ["id", "source", "datetime", "cloud_cover", "resolution_m", "size_bytes", "url"]

    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    else:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
        click.echo(output.getvalue())
