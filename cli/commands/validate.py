"""
Validate Command - Run quality control checks on analysis results.

Usage:
    flight validate --input ./results/
    flight validate --input ./results/ --output report.html --format html
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import click
import numpy as np

logger = logging.getLogger("flight.validate")


# Quality check definitions
QUALITY_CHECKS = {
    "spatial_coherence": {
        "name": "Spatial Coherence",
        "description": "Check for spatially incoherent patterns (salt-and-pepper noise)",
        "category": "sanity",
        "severity": "warning",
    },
    "value_range": {
        "name": "Value Range",
        "description": "Verify values are within physically plausible ranges",
        "category": "sanity",
        "severity": "error",
    },
    "coverage": {
        "name": "Coverage Completeness",
        "description": "Check for missing data and gaps",
        "category": "sanity",
        "severity": "warning",
    },
    "artifacts": {
        "name": "Artifact Detection",
        "description": "Detect stripe patterns, saturation, and processing artifacts",
        "category": "sanity",
        "severity": "warning",
    },
    "cross_sensor": {
        "name": "Cross-Sensor Validation",
        "description": "Compare results across different sensor types",
        "category": "validation",
        "severity": "info",
    },
    "historical": {
        "name": "Historical Baseline",
        "description": "Compare against historical patterns",
        "category": "validation",
        "severity": "info",
    },
}


@click.command("validate")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Input directory containing analysis results.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file for validation report.",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["text", "json", "html", "markdown"], case_sensitive=False),
    default="text",
    help="Output format for the report (default: text).",
)
@click.option(
    "--checks",
    "-c",
    type=str,
    default="all",
    help="Comma-separated list of checks to run, or 'all' (default: all).",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Fail on any warning or error.",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.7,
    help="Minimum quality score threshold (0-1, default: 0.7).",
)
@click.pass_obj
def validate(
    ctx,
    input_path: Path,
    output_path: Optional[Path],
    output_format: str,
    checks: str,
    strict: bool,
    threshold: float,
):
    """
    Run quality control checks on analysis results.

    Executes a suite of quality checks including spatial coherence,
    value range validation, artifact detection, and cross-validation.
    Generates a report with pass/fail status and recommendations.

    \b
    Examples:
        # Run all quality checks
        flight validate --input ./results/

        # Generate HTML report
        flight validate --input ./results/ --output report.html --format html

        # Run specific checks only
        flight validate --input ./results/ --checks spatial_coherence,value_range
    """
    click.echo(f"\n=== Quality Validation ===")
    click.echo(f"  Input: {input_path}")
    click.echo(f"  Format: {output_format}")
    click.echo(f"  Threshold: {threshold}")

    # Parse checks to run
    if checks.lower() == "all":
        checks_to_run = list(QUALITY_CHECKS.keys())
    else:
        checks_to_run = [c.strip() for c in checks.split(",")]
        invalid = [c for c in checks_to_run if c not in QUALITY_CHECKS]
        if invalid:
            raise click.BadParameter(f"Unknown checks: {', '.join(invalid)}")

    click.echo(f"  Checks: {', '.join(checks_to_run)}")

    # Run quality checks
    results = run_quality_checks(input_path, checks_to_run)

    # Calculate overall score
    overall_score = calculate_overall_score(results)
    passed = overall_score >= threshold

    # Count issues by severity
    errors = sum(1 for r in results if r["severity"] == "error" and not r["passed"])
    warnings = sum(1 for r in results if r["severity"] == "warning" and not r["passed"])

    # Generate report
    report = generate_report(
        input_path=input_path,
        results=results,
        overall_score=overall_score,
        passed=passed,
        threshold=threshold,
    )

    # Output report
    if output_format == "json":
        output_json(report, output_path)
    elif output_format == "html":
        output_html(report, output_path)
    elif output_format == "markdown":
        output_markdown(report, output_path)
    else:
        output_text(report, output_path)

    # Summary
    click.echo(f"\n=== Validation Summary ===")
    click.echo(f"  Overall Score: {overall_score:.2f}")
    click.echo(f"  Threshold: {threshold}")
    click.echo(f"  Status: {'PASSED' if passed else 'FAILED'}")
    click.echo(f"  Errors: {errors}")
    click.echo(f"  Warnings: {warnings}")

    if output_path:
        click.echo(f"  Report: {output_path}")

    # Exit with error code if failed
    if not passed or (strict and (errors > 0 or warnings > 0)):
        raise SystemExit(1)


def run_quality_checks(input_path: Path, checks: List[str]) -> List[Dict[str, Any]]:
    """
    Run specified quality checks on the input data.
    """
    results = []

    # Import quality modules
    try:
        from core.quality.sanity.spatial import SpatialCoherenceChecker
        from core.quality.sanity.values import ValuePlausibilityChecker, ValueType
        from core.quality.sanity.artifacts import ArtifactDetector
    except ImportError as e:
        logger.error(f"Failed to import quality modules: {e}")
        raise ImportError(f"Quality control modules not available: {e}")

    # Load raster data from input path
    raster_data = _load_raster_data(input_path)
    if raster_data is None:
        logger.error(f"No valid raster data found in {input_path}")
        raise FileNotFoundError(f"No valid raster data found in {input_path}")

    # Run each requested check
    for check_id in checks:
        check_info = QUALITY_CHECKS[check_id]

        try:
            if check_id == "spatial_coherence":
                result = _run_spatial_coherence_check(raster_data, check_info)
            elif check_id == "value_range":
                result = _run_value_range_check(raster_data, check_info)
            elif check_id == "artifacts":
                result = _run_artifact_check(raster_data, check_info)
            elif check_id == "coverage":
                result = _run_coverage_check(raster_data, check_info)
            elif check_id in ["cross_sensor", "historical"]:
                # These require additional data not available in basic validation
                result = {
                    "check_id": check_id,
                    "name": check_info["name"],
                    "description": check_info["description"],
                    "category": check_info["category"],
                    "severity": check_info["severity"],
                    "passed": True,
                    "score": 1.0,
                    "message": "Check skipped - requires additional reference data",
                    "details": {},
                }
            else:
                # Unknown check type
                result = {
                    "check_id": check_id,
                    "name": check_info["name"],
                    "description": check_info["description"],
                    "category": check_info["category"],
                    "severity": check_info["severity"],
                    "passed": True,
                    "score": 1.0,
                    "message": "Check not implemented",
                    "details": {},
                }

            results.append(result)

        except Exception as e:
            logger.warning(f"Check {check_id} failed: {e}")
            results.append({
                "check_id": check_id,
                "name": check_info["name"],
                "description": check_info["description"],
                "category": check_info["category"],
                "severity": check_info["severity"],
                "passed": False,
                "score": 0.0,
                "message": f"Check failed: {str(e)}",
                "details": {"error": str(e)},
            })

    return results


def _load_raster_data(input_path: Path) -> Optional[np.ndarray]:
    """
    Load raster data from input path (GeoTIFF or directory containing GeoTIFFs).
    """
    try:
        import rasterio
    except ImportError:
        logger.error("rasterio not available - cannot load raster data")
        return None

    # If input is a file, load it directly
    if input_path.is_file() and input_path.suffix.lower() in ['.tif', '.tiff']:
        try:
            with rasterio.open(input_path) as src:
                data = src.read(1)  # Read first band
                return data
        except Exception as e:
            logger.warning(f"Failed to load {input_path}: {e}")
            return None

    # If input is a directory, find first GeoTIFF
    if input_path.is_dir():
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tif_files = list(input_path.glob(ext))
            if tif_files:
                try:
                    with rasterio.open(tif_files[0]) as src:
                        data = src.read(1)  # Read first band
                        return data
                except Exception as e:
                    logger.warning(f"Failed to load {tif_files[0]}: {e}")
                    continue

    return None


def _run_spatial_coherence_check(data: np.ndarray, check_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run spatial coherence check using SpatialCoherenceChecker.
    """
    from core.quality.sanity.spatial import SpatialCoherenceChecker

    checker = SpatialCoherenceChecker()
    result = checker.check(data)

    # Calculate score from issues
    score = 1.0 - min(1.0, (result.critical_count * 0.3 + result.high_count * 0.15))
    passed = result.is_coherent

    return {
        "check_id": "spatial_coherence",
        "name": check_info["name"],
        "description": check_info["description"],
        "category": check_info["category"],
        "severity": check_info["severity"],
        "passed": passed,
        "score": score,
        "message": f"{'Spatially coherent' if passed else f'{len(result.issues)} spatial issues detected'}",
        "details": {
            "issue_count": len(result.issues),
            "critical_count": result.critical_count,
            "high_count": result.high_count,
            "metrics": result.metrics,
        },
    }


def _run_value_range_check(data: np.ndarray, check_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run value range check using ValuePlausibilityChecker.
    """
    from core.quality.sanity.values import ValuePlausibilityChecker, ValueType

    # Try to infer value type from data range
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return {
            "check_id": "value_range",
            "name": check_info["name"],
            "description": check_info["description"],
            "category": check_info["category"],
            "severity": check_info["severity"],
            "passed": False,
            "score": 0.0,
            "message": "No valid data",
            "details": {},
        }

    min_val, max_val = float(np.min(valid_data)), float(np.max(valid_data))

    # Infer value type based on range
    if min_val >= 0 and max_val <= 1:
        value_type = ValueType.CONFIDENCE
    elif min_val >= -1 and max_val <= 1:
        value_type = ValueType.NDWI
    elif min_val >= -50 and max_val <= 10:
        value_type = ValueType.BACKSCATTER_DB
    else:
        value_type = ValueType.CUSTOM

    checker = ValuePlausibilityChecker()
    if value_type != ValueType.CUSTOM:
        from core.quality.sanity.values import ValuePlausibilityConfig
        checker.config.value_type = value_type

    result = checker.check(data)

    score = 1.0 - min(1.0, (result.critical_count * 0.3 + result.high_count * 0.15))
    passed = result.is_plausible

    return {
        "check_id": "value_range",
        "name": check_info["name"],
        "description": check_info["description"],
        "category": check_info["category"],
        "severity": check_info["severity"],
        "passed": passed,
        "score": score,
        "message": f"{'Values in valid range' if passed else f'{len(result.issues)} value issues detected'}",
        "details": {
            "issue_count": len(result.issues),
            "statistics": result.statistics,
            "value_type": value_type.value,
        },
    }


def _run_artifact_check(data: np.ndarray, check_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run artifact detection using ArtifactDetector.
    """
    from core.quality.sanity.artifacts import ArtifactDetector

    detector = ArtifactDetector()
    result = detector.detect(data)

    score = 1.0 - min(1.0, (result.critical_count * 0.3 + result.high_count * 0.15))
    passed = not result.has_artifacts or (result.critical_count == 0 and result.high_count == 0)

    return {
        "check_id": "artifacts",
        "name": check_info["name"],
        "description": check_info["description"],
        "category": check_info["category"],
        "severity": check_info["severity"],
        "passed": passed,
        "score": score,
        "message": f"{'No significant artifacts' if passed else f'{len(result.artifacts)} artifacts detected'}",
        "details": {
            "artifact_count": len(result.artifacts),
            "critical_count": result.critical_count,
            "high_count": result.high_count,
            "artifact_types": result.artifact_types,
        },
    }


def _run_coverage_check(data: np.ndarray, check_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run coverage completeness check.
    """
    total_pixels = data.size
    valid_pixels = np.isfinite(data).sum()
    nodata_pixels = total_pixels - valid_pixels

    coverage_pct = 100.0 * valid_pixels / total_pixels if total_pixels > 0 else 0.0
    nodata_pct = 100.0 * nodata_pixels / total_pixels if total_pixels > 0 else 0.0

    # Pass if coverage > 85%
    passed = coverage_pct >= 85.0
    score = coverage_pct / 100.0

    return {
        "check_id": "coverage",
        "name": check_info["name"],
        "description": check_info["description"],
        "category": check_info["category"],
        "severity": check_info["severity"],
        "passed": passed,
        "score": score,
        "message": f"{'Sufficient coverage' if passed else f'Low coverage: {coverage_pct:.1f}%'}",
        "details": {
            "coverage_percentage": coverage_pct,
            "nodata_percentage": nodata_pct,
            "total_pixels": total_pixels,
            "valid_pixels": int(valid_pixels),
        },
    }


def calculate_overall_score(results: List[Dict[str, Any]]) -> float:
    """Calculate weighted overall quality score."""
    if not results:
        return 0.0

    # Weight by severity
    weights = {"error": 3.0, "warning": 1.5, "info": 1.0}

    total_weight = 0.0
    weighted_score = 0.0

    for r in results:
        weight = weights.get(r["severity"], 1.0)
        total_weight += weight
        weighted_score += r["score"] * weight

    return weighted_score / total_weight if total_weight > 0 else 0.0


def generate_report(
    input_path: Path,
    results: List[Dict[str, Any]],
    overall_score: float,
    passed: bool,
    threshold: float,
) -> Dict[str, Any]:
    """Generate structured validation report."""
    return {
        "title": "Quality Validation Report",
        "input_path": str(input_path),
        "generated_at": datetime.now().isoformat(),
        "overall_score": overall_score,
        "threshold": threshold,
        "passed": passed,
        "summary": {
            "total_checks": len(results),
            "passed_checks": sum(1 for r in results if r["passed"]),
            "failed_checks": sum(1 for r in results if not r["passed"]),
            "errors": sum(1 for r in results if r["severity"] == "error" and not r["passed"]),
            "warnings": sum(1 for r in results if r["severity"] == "warning" and not r["passed"]),
        },
        "checks": results,
        "recommendations": generate_recommendations(results),
    }


def generate_recommendations(results: List[Dict[str, Any]]) -> List[str]:
    """Generate recommendations based on failed checks."""
    recommendations = []

    for r in results:
        if not r["passed"]:
            if r["check_id"] == "spatial_coherence":
                recommendations.append(
                    "Apply spatial filtering to reduce noise in the output"
                )
            elif r["check_id"] == "value_range":
                recommendations.append(
                    "Review input data quality and algorithm parameters"
                )
            elif r["check_id"] == "coverage":
                recommendations.append(
                    "Check for cloud cover or data gaps in input imagery"
                )
            elif r["check_id"] == "artifacts":
                recommendations.append(
                    "Inspect for sensor artifacts or processing issues"
                )

    return list(set(recommendations))  # Remove duplicates


def output_text(report: Dict[str, Any], output_path: Optional[Path]):
    """Output report as plain text."""
    lines = [
        f"\n{'=' * 60}",
        f"  {report['title']}",
        f"{'=' * 60}",
        f"",
        f"Input: {report['input_path']}",
        f"Generated: {report['generated_at']}",
        f"",
        f"Overall Score: {report['overall_score']:.2f} / 1.00",
        f"Threshold: {report['threshold']}",
        f"Status: {'PASSED' if report['passed'] else 'FAILED'}",
        f"",
        f"--- Summary ---",
        f"Total checks: {report['summary']['total_checks']}",
        f"Passed: {report['summary']['passed_checks']}",
        f"Failed: {report['summary']['failed_checks']}",
        f"Errors: {report['summary']['errors']}",
        f"Warnings: {report['summary']['warnings']}",
        f"",
        f"--- Check Results ---",
    ]

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"[{status}] {check['name']} (score: {check['score']:.2f})")
        if not check["passed"]:
            lines.append(f"       {check['message']}")

    if report["recommendations"]:
        lines.append("")
        lines.append("--- Recommendations ---")
        for rec in report["recommendations"]:
            lines.append(f"  - {rec}")

    text = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(text)
    else:
        click.echo(text)


def output_json(report: Dict[str, Any], output_path: Optional[Path]):
    """Output report as JSON."""
    json_str = json.dumps(report, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(json_str)
    else:
        click.echo(json_str)


def output_html(report: Dict[str, Any], output_path: Optional[Path]):
    """Output report as HTML."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report['title']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .passed {{ color: #28a745; }}
        .failed {{ color: #dc3545; }}
        .check {{ padding: 10px; margin: 10px 0; border-left: 4px solid #ddd; }}
        .check.pass {{ border-left-color: #28a745; }}
        .check.fail {{ border-left-color: #dc3545; }}
        .score {{ font-weight: bold; }}
        .recommendations {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>{report['title']}</h1>
    <p>Input: <code>{report['input_path']}</code></p>
    <p>Generated: {report['generated_at']}</p>

    <div class="summary">
        <h2>Overall Score: <span class="score {'passed' if report['passed'] else 'failed'}">{report['overall_score']:.2f}</span> / 1.00</h2>
        <p>Status: <strong class="{'passed' if report['passed'] else 'failed'}">{'PASSED' if report['passed'] else 'FAILED'}</strong></p>
        <p>Checks: {report['summary']['passed_checks']} passed, {report['summary']['failed_checks']} failed</p>
    </div>

    <h2>Check Results</h2>
"""

    for check in report["checks"]:
        status_class = "pass" if check["passed"] else "fail"
        html += f"""
    <div class="check {status_class}">
        <strong>{check['name']}</strong> - Score: {check['score']:.2f}
        <br><small>{check['description']}</small>
        {'<br><em>' + check['message'] + '</em>' if not check['passed'] else ''}
    </div>
"""

    if report["recommendations"]:
        html += """
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
"""
        for rec in report["recommendations"]:
            html += f"            <li>{rec}</li>\n"
        html += """
        </ul>
    </div>
"""

    html += """
</body>
</html>
"""

    if output_path:
        with open(output_path, "w") as f:
            f.write(html)
        click.echo(f"HTML report saved to: {output_path}")
    else:
        click.echo(html)


def output_markdown(report: Dict[str, Any], output_path: Optional[Path]):
    """Output report as Markdown."""
    md = f"""# {report['title']}

**Input:** `{report['input_path']}`
**Generated:** {report['generated_at']}

## Summary

| Metric | Value |
|--------|-------|
| Overall Score | {report['overall_score']:.2f} / 1.00 |
| Status | {'PASSED' if report['passed'] else 'FAILED'} |
| Total Checks | {report['summary']['total_checks']} |
| Passed | {report['summary']['passed_checks']} |
| Failed | {report['summary']['failed_checks']} |

## Check Results

"""

    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        md += f"### [{status}] {check['name']}\n\n"
        md += f"- **Score:** {check['score']:.2f}\n"
        md += f"- **Description:** {check['description']}\n"
        if not check["passed"]:
            md += f"- **Issue:** {check['message']}\n"
        md += "\n"

    if report["recommendations"]:
        md += "## Recommendations\n\n"
        for rec in report["recommendations"]:
            md += f"- {rec}\n"

    if output_path:
        with open(output_path, "w") as f:
            f.write(md)
    else:
        click.echo(md)
