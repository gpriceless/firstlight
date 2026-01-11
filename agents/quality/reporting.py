"""
QA Reporting for Quality Agent.

Provides the QAReportGenerator class for generating quality assurance
reports in multiple formats:
- JSON: Machine-readable format
- HTML: Human-readable with visualizations
- Markdown: Lightweight documentation format
- Text: Simple plain text summary

This module integrates with core/quality/reporting/ to provide
comprehensive reporting capabilities for the Quality Agent.

Example:
    generator = QAReportGenerator()

    # Generate report from QA results
    report = await generator.generate_report(
        event_id="evt_001",
        product_id="prod_001",
        qa_results=qa_results,
        format=ReportFormat.HTML,
    )

    # Save report
    await generator.save_report(report, "/path/to/reports")

    # Generate summary
    summary = generator.generate_summary(qa_results)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import core reporting modules
from core.quality.reporting import (
    QAReportGenerator as CoreReportGenerator,
    QAReport as CoreQAReport,
    ReportFormat,
    ReportLevel,
    ReportSection,
    ReportConfig,
    ReportMetadata,
    QualitySummary,
    CheckReport,
    CrossValidationReport,
    UncertaintySummaryReport,
    GatingReport,
    FlagReport,
    ActionReport,
    Recommendation,
)

# Import diagnostics if available
try:
    from core.quality.reporting import (
        DiagnosticGenerator,
        Diagnostics,
        DiagnosticLevel,
    )
    _DIAGNOSTICS_AVAILABLE = True
except ImportError:
    _DIAGNOSTICS_AVAILABLE = False


logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of reports that can be generated."""
    FULL = "full"               # Complete QA report
    SUMMARY = "summary"         # Brief summary only
    DIAGNOSTIC = "diagnostic"   # Detailed diagnostic report
    COMPARISON = "comparison"   # Comparison between runs
    EXECUTIVE = "executive"     # High-level executive summary


@dataclass
class ReportGeneratorConfig:
    """
    Configuration for QAReportGenerator.

    Attributes:
        default_format: Default output format
        default_level: Default detail level
        include_visualizations: Include visual elements in HTML
        include_recommendations: Include action recommendations
        include_provenance: Include provenance information
        output_directory: Default output directory
        template_directory: Custom template directory
        max_issues_in_summary: Maximum issues to show in summary
    """
    default_format: ReportFormat = ReportFormat.JSON
    default_level: ReportLevel = ReportLevel.STANDARD
    include_visualizations: bool = True
    include_recommendations: bool = True
    include_provenance: bool = True
    output_directory: Optional[str] = None
    template_directory: Optional[str] = None
    max_issues_in_summary: int = 10


@dataclass
class GeneratedReport:
    """
    A generated QA report.

    Attributes:
        report_id: Unique report identifier
        event_id: Associated event
        product_id: Associated product
        report_type: Type of report
        format: Output format
        content: Report content (string or dict)
        metadata: Report metadata
        generated_at: Generation timestamp
        file_path: Path where saved (if applicable)
    """
    report_id: str
    event_id: str
    product_id: str
    report_type: ReportType
    format: ReportFormat
    content: Union[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "event_id": self.event_id,
            "product_id": self.product_id,
            "report_type": self.report_type.value,
            "format": self.format.value,
            "generated_at": self.generated_at.isoformat(),
            "file_path": self.file_path,
            "metadata": self.metadata,
        }


@dataclass
class ReportSummary:
    """
    Quick summary of QA results.

    Attributes:
        status: Overall status (pass/fail/review)
        confidence: Confidence score
        total_checks: Total checks performed
        passed_checks: Passed checks
        failed_checks: Failed checks
        warning_checks: Checks with warnings
        key_issues: Top issues
        recommendations: Action recommendations
    """
    status: str
    confidence: float
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    key_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status,
            "confidence": self.confidence,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "key_issues": self.key_issues,
            "recommendations": self.recommendations,
        }


class QAReportGenerator:
    """
    Generates QA reports for the Quality Agent.

    Provides methods for generating various types of quality assurance
    reports in multiple formats, with support for customization and
    templating.
    """

    def __init__(self, config: Optional[ReportGeneratorConfig] = None):
        """
        Initialize QAReportGenerator.

        Args:
            config: Generator configuration
        """
        self.config = config or ReportGeneratorConfig()
        self._logger = logging.getLogger(f"{__name__}.QAReportGenerator")

        # Core report generator
        self._core_generator = CoreReportGenerator()

        # Diagnostic generator if available
        self._diagnostic_generator = None
        if _DIAGNOSTICS_AVAILABLE:
            self._diagnostic_generator = DiagnosticGenerator()

        # Report counter for IDs
        self._report_counter = 0

        self._logger.info("QAReportGenerator initialized")

    async def generate_report(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        report_type: ReportType = ReportType.FULL,
        format: Optional[ReportFormat] = None,
        level: Optional[ReportLevel] = None,
    ) -> GeneratedReport:
        """
        Generate a QA report from quality results.

        Args:
            event_id: Event identifier
            product_id: Product identifier
            qa_results: Dictionary containing QA results with keys:
                - sanity: SanityResult or dict
                - validation: ValidationResult or dict
                - uncertainty: UncertaintyMap or dict
                - gate_decision: GateDecision or dict
            report_type: Type of report to generate
            format: Output format (defaults to config)
            level: Detail level (defaults to config)

        Returns:
            GeneratedReport object
        """
        import time
        start_time = time.time()

        format = format or self.config.default_format
        level = level or self.config.default_level

        self._report_counter += 1
        report_id = f"qa_report_{event_id}_{product_id}_{self._report_counter}"

        self._logger.info(
            f"Generating {report_type.value} report for {event_id}/{product_id} "
            f"in {format.value} format"
        )

        # Generate content based on report type
        if report_type == ReportType.SUMMARY:
            content = self._generate_summary_content(qa_results, format)
        elif report_type == ReportType.DIAGNOSTIC:
            content = self._generate_diagnostic_content(qa_results, format)
        elif report_type == ReportType.EXECUTIVE:
            content = self._generate_executive_content(qa_results, format)
        else:
            content = self._generate_full_content(
                event_id, product_id, qa_results, format, level
            )

        # Create report object
        report = GeneratedReport(
            report_id=report_id,
            event_id=event_id,
            product_id=product_id,
            report_type=report_type,
            format=format,
            content=content,
            metadata={
                "level": level.value,
                "generation_time_seconds": time.time() - start_time,
                "sections_included": self._get_included_sections(qa_results),
            },
        )

        self._logger.info(
            f"Report {report_id} generated in {time.time() - start_time:.2f}s"
        )

        return report

    async def save_report(
        self,
        report: GeneratedReport,
        output_dir: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Save a report to file.

        Args:
            report: Report to save
            output_dir: Output directory (defaults to config)
            filename: Custom filename

        Returns:
            Path to saved file
        """
        output_dir = output_dir or self.config.output_directory or "."
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        if filename is None:
            ext = self._get_file_extension(report.format)
            filename = f"{report.report_id}.{ext}"

        file_path = output_path / filename

        # Write content
        if report.format == ReportFormat.JSON:
            if isinstance(report.content, dict):
                with open(file_path, "w") as f:
                    json.dump(report.content, f, indent=2, default=str)
            else:
                with open(file_path, "w") as f:
                    f.write(report.content)
        else:
            with open(file_path, "w") as f:
                f.write(report.content if isinstance(report.content, str) else json.dumps(report.content))

        report.file_path = str(file_path)

        self._logger.info(f"Report saved to {file_path}")

        return str(file_path)

    def generate_summary(
        self,
        qa_results: Dict[str, Any],
    ) -> ReportSummary:
        """
        Generate a quick summary of QA results.

        Args:
            qa_results: QA results dictionary

        Returns:
            ReportSummary object
        """
        # Determine overall status
        status = "pass"
        if qa_results.get("gate_decision"):
            gd = qa_results["gate_decision"]
            status = gd.get("status", "pass") if isinstance(gd, dict) else getattr(gd, "status", "pass")
            if hasattr(status, "value"):
                status = status.value

        # Calculate confidence
        confidence = 0.8
        if qa_results.get("gate_decision"):
            gd = qa_results["gate_decision"]
            confidence = gd.get("confidence", 0.8) if isinstance(gd, dict) else getattr(gd, "confidence", 0.8)

        # Count checks
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0

        if qa_results.get("sanity"):
            sanity = qa_results["sanity"]
            if isinstance(sanity, dict):
                issues = sanity.get("issues", [])
                total_checks += len(issues) + 1  # Plus overall check
                if sanity.get("passed"):
                    passed_checks += 1
                else:
                    failed_checks += 1
                warning_checks += sanity.get("warning_count", 0)
            else:
                total_checks += 1
                if getattr(sanity, "passed", True):
                    passed_checks += 1
                else:
                    failed_checks += 1

        if qa_results.get("validation"):
            val = qa_results["validation"]
            if isinstance(val, dict):
                total_checks += 1
                if val.get("passed"):
                    passed_checks += 1
                else:
                    failed_checks += 1
            else:
                total_checks += 1
                if getattr(val, "passed", True):
                    passed_checks += 1
                else:
                    failed_checks += 1

        # Extract key issues
        key_issues = []
        if qa_results.get("gate_decision"):
            gd = qa_results["gate_decision"]
            reasons = gd.get("reasons", []) if isinstance(gd, dict) else getattr(gd, "reasons", [])
            key_issues.extend(reasons[:self.config.max_issues_in_summary])

        if qa_results.get("sanity"):
            sanity = qa_results["sanity"]
            issues = sanity.get("issues", []) if isinstance(sanity, dict) else getattr(sanity, "issues", [])
            for issue in issues[:5]:
                if isinstance(issue, dict):
                    key_issues.append(issue.get("message", str(issue)))
                else:
                    key_issues.append(str(issue))

        # Generate recommendations
        recommendations = self._generate_recommendations(qa_results)

        return ReportSummary(
            status=status,
            confidence=confidence,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warning_checks=warning_checks,
            key_issues=key_issues[:self.config.max_issues_in_summary],
            recommendations=recommendations[:5],
        )

    def summarize_quality_metrics(
        self,
        qa_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a dictionary of key quality metrics.

        Args:
            qa_results: QA results dictionary

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Sanity metrics
        if qa_results.get("sanity"):
            sanity = qa_results["sanity"]
            if isinstance(sanity, dict):
                metrics["sanity_score"] = sanity.get("overall_score", 1.0)
                metrics["sanity_passed"] = sanity.get("passed", True)
                metrics["sanity_issues"] = len(sanity.get("issues", []))
            else:
                metrics["sanity_score"] = getattr(sanity, "overall_score", 1.0)
                metrics["sanity_passed"] = getattr(sanity, "passed", True)

        # Validation metrics
        if qa_results.get("validation"):
            val = qa_results["validation"]
            if isinstance(val, dict):
                metrics["agreement_score"] = val.get("agreement_score", 1.0)
                metrics["validation_passed"] = val.get("passed", True)
            else:
                metrics["agreement_score"] = getattr(val, "agreement_score", 1.0)
                metrics["validation_passed"] = getattr(val, "passed", True)

        # Uncertainty metrics
        if qa_results.get("uncertainty"):
            unc = qa_results["uncertainty"]
            if isinstance(unc, dict):
                metrics["mean_uncertainty"] = unc.get("mean_uncertainty", 0.0)
                metrics["max_uncertainty"] = unc.get("max_uncertainty", 0.0)
                metrics["hotspot_count"] = unc.get("hotspot_count", 0)
            else:
                metrics["mean_uncertainty"] = getattr(unc, "mean_uncertainty", 0.0)
                metrics["max_uncertainty"] = getattr(unc, "max_uncertainty", 0.0)

        # Gate decision metrics
        if qa_results.get("gate_decision"):
            gd = qa_results["gate_decision"]
            if isinstance(gd, dict):
                metrics["gate_confidence"] = gd.get("confidence", 0.0)
                metrics["can_release"] = gd.get("can_release", False)
            else:
                metrics["gate_confidence"] = getattr(gd, "confidence", 0.0)
                metrics["can_release"] = getattr(gd, "can_release", False)

        return metrics

    # Private methods

    def _generate_full_content(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        format: ReportFormat,
        level: ReportLevel,
    ) -> Union[str, Dict[str, Any]]:
        """Generate full report content."""
        if format == ReportFormat.JSON:
            return self._generate_json_report(event_id, product_id, qa_results, level)
        elif format == ReportFormat.HTML:
            return self._generate_html_report(event_id, product_id, qa_results, level)
        elif format == ReportFormat.MARKDOWN:
            return self._generate_markdown_report(event_id, product_id, qa_results, level)
        else:
            return self._generate_text_report(event_id, product_id, qa_results, level)

    def _generate_json_report(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        level: ReportLevel,
    ) -> Dict[str, Any]:
        """Generate JSON report content."""
        report = {
            "report_type": "quality_assessment",
            "version": "1.0",
            "event_id": event_id,
            "product_id": product_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
        }

        # Summary
        summary = self.generate_summary(qa_results)
        report["summary"] = summary.to_dict()

        # Metrics
        report["metrics"] = self.summarize_quality_metrics(qa_results)

        # Detailed results
        if level in (ReportLevel.STANDARD, ReportLevel.DETAILED, ReportLevel.DEBUG):
            if qa_results.get("sanity"):
                sanity = qa_results["sanity"]
                report["sanity_checks"] = sanity if isinstance(sanity, dict) else sanity.to_dict()

            if qa_results.get("validation"):
                val = qa_results["validation"]
                report["validation"] = val if isinstance(val, dict) else val.to_dict()

            if qa_results.get("uncertainty"):
                unc = qa_results["uncertainty"]
                report["uncertainty"] = unc if isinstance(unc, dict) else unc.to_dict()

            if qa_results.get("gate_decision"):
                gd = qa_results["gate_decision"]
                report["gate_decision"] = gd if isinstance(gd, dict) else gd.to_dict()

        # Recommendations
        if self.config.include_recommendations:
            report["recommendations"] = self._generate_recommendations(qa_results)

        return report

    def _generate_html_report(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        level: ReportLevel,
    ) -> str:
        """Generate HTML report content."""
        summary = self.generate_summary(qa_results)
        metrics = self.summarize_quality_metrics(qa_results)

        # Determine status color
        status_colors = {
            "pass": "#28a745",
            "pass_with_warnings": "#ffc107",
            "review_required": "#fd7e14",
            "fail": "#dc3545",
        }
        status_color = status_colors.get(summary.status, "#6c757d")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QA Report - {event_id}/{product_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; border-bottom: 1px solid #eee; }}
        .header h1 {{ margin: 0 0 10px 0; color: #333; }}
        .header .meta {{ color: #666; font-size: 14px; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 4px; color: white; font-weight: bold; background: {status_color}; }}
        .section {{ padding: 20px; border-bottom: 1px solid #eee; }}
        .section h2 {{ margin: 0 0 15px 0; color: #333; font-size: 18px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; }}
        .metric {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric .value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric .label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
        .issues {{ list-style: none; padding: 0; margin: 0; }}
        .issues li {{ padding: 10px; margin: 5px 0; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; }}
        .recommendations {{ list-style: none; padding: 0; margin: 0; }}
        .recommendations li {{ padding: 10px; margin: 5px 0; background: #d4edda; border-left: 4px solid #28a745; border-radius: 4px; }}
        .footer {{ padding: 15px 20px; text-align: center; color: #666; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Quality Assessment Report</h1>
            <div class="meta">
                <span>Event: <strong>{event_id}</strong></span> |
                <span>Product: <strong>{product_id}</strong></span> |
                <span>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
            </div>
        </div>

        <div class="section">
            <h2>Overall Status</h2>
            <span class="status-badge">{summary.status.upper().replace('_', ' ')}</span>
            <p style="margin-top: 15px;">Confidence: <strong>{summary.confidence:.1%}</strong></p>
        </div>

        <div class="section">
            <h2>Quality Metrics</h2>
            <div class="metrics">
                <div class="metric">
                    <div class="value">{summary.total_checks}</div>
                    <div class="label">Total Checks</div>
                </div>
                <div class="metric">
                    <div class="value" style="color: #28a745;">{summary.passed_checks}</div>
                    <div class="label">Passed</div>
                </div>
                <div class="metric">
                    <div class="value" style="color: #dc3545;">{summary.failed_checks}</div>
                    <div class="label">Failed</div>
                </div>
                <div class="metric">
                    <div class="value" style="color: #ffc107;">{summary.warning_checks}</div>
                    <div class="label">Warnings</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics.get('sanity_score', 1.0):.2f}</div>
                    <div class="label">Sanity Score</div>
                </div>
                <div class="metric">
                    <div class="value">{metrics.get('mean_uncertainty', 0.0):.2f}</div>
                    <div class="label">Uncertainty</div>
                </div>
            </div>
        </div>
"""

        if summary.key_issues:
            html += """
        <div class="section">
            <h2>Key Issues</h2>
            <ul class="issues">
"""
            for issue in summary.key_issues:
                html += f"                <li>{issue}</li>\n"
            html += """            </ul>
        </div>
"""

        if summary.recommendations:
            html += """
        <div class="section">
            <h2>Recommendations</h2>
            <ul class="recommendations">
"""
            for rec in summary.recommendations:
                html += f"                <li>{rec}</li>\n"
            html += """            </ul>
        </div>
"""

        html += """
        <div class="footer">
            Generated by Multiverse Dive Quality Agent
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_markdown_report(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        level: ReportLevel,
    ) -> str:
        """Generate Markdown report content."""
        summary = self.generate_summary(qa_results)
        metrics = self.summarize_quality_metrics(qa_results)

        md = f"""# Quality Assessment Report

**Event:** {event_id}
**Product:** {product_id}
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

## Overall Status

**Status:** {summary.status.upper().replace('_', ' ')}
**Confidence:** {summary.confidence:.1%}

## Quality Metrics

| Metric | Value |
|--------|-------|
| Total Checks | {summary.total_checks} |
| Passed | {summary.passed_checks} |
| Failed | {summary.failed_checks} |
| Warnings | {summary.warning_checks} |
| Sanity Score | {metrics.get('sanity_score', 1.0):.2f} |
| Mean Uncertainty | {metrics.get('mean_uncertainty', 0.0):.2f} |

"""

        if summary.key_issues:
            md += "## Key Issues\n\n"
            for issue in summary.key_issues:
                md += f"- {issue}\n"
            md += "\n"

        if summary.recommendations:
            md += "## Recommendations\n\n"
            for rec in summary.recommendations:
                md += f"- {rec}\n"
            md += "\n"

        md += "---\n*Generated by Multiverse Dive Quality Agent*\n"

        return md

    def _generate_text_report(
        self,
        event_id: str,
        product_id: str,
        qa_results: Dict[str, Any],
        level: ReportLevel,
    ) -> str:
        """Generate plain text report content."""
        summary = self.generate_summary(qa_results)

        lines = [
            "=" * 60,
            "QUALITY ASSESSMENT REPORT",
            "=" * 60,
            f"Event: {event_id}",
            f"Product: {product_id}",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "-" * 60,
            "OVERALL STATUS",
            "-" * 60,
            f"Status: {summary.status.upper().replace('_', ' ')}",
            f"Confidence: {summary.confidence:.1%}",
            "",
            "-" * 60,
            "CHECKS SUMMARY",
            "-" * 60,
            f"Total: {summary.total_checks}",
            f"Passed: {summary.passed_checks}",
            f"Failed: {summary.failed_checks}",
            f"Warnings: {summary.warning_checks}",
            "",
        ]

        if summary.key_issues:
            lines.extend([
                "-" * 60,
                "KEY ISSUES",
                "-" * 60,
            ])
            for i, issue in enumerate(summary.key_issues, 1):
                lines.append(f"{i}. {issue}")
            lines.append("")

        if summary.recommendations:
            lines.extend([
                "-" * 60,
                "RECOMMENDATIONS",
                "-" * 60,
            ])
            for i, rec in enumerate(summary.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _generate_summary_content(
        self,
        qa_results: Dict[str, Any],
        format: ReportFormat,
    ) -> Union[str, Dict[str, Any]]:
        """Generate summary-only content."""
        summary = self.generate_summary(qa_results)

        if format == ReportFormat.JSON:
            return summary.to_dict()
        else:
            return f"Status: {summary.status}, Confidence: {summary.confidence:.1%}"

    def _generate_diagnostic_content(
        self,
        qa_results: Dict[str, Any],
        format: ReportFormat,
    ) -> Union[str, Dict[str, Any]]:
        """Generate diagnostic content."""
        if not _DIAGNOSTICS_AVAILABLE:
            return {"error": "Diagnostics module not available"}

        # Use diagnostic generator if available
        try:
            diagnostics = self._diagnostic_generator.generate(qa_results)
            if format == ReportFormat.JSON:
                return diagnostics.to_dict() if hasattr(diagnostics, "to_dict") else {"diagnostics": str(diagnostics)}
            else:
                return str(diagnostics)
        except Exception as e:
            return {"error": f"Failed to generate diagnostics: {e}"}

    def _generate_executive_content(
        self,
        qa_results: Dict[str, Any],
        format: ReportFormat,
    ) -> Union[str, Dict[str, Any]]:
        """Generate executive summary content."""
        summary = self.generate_summary(qa_results)

        if format == ReportFormat.JSON:
            return {
                "status": summary.status,
                "confidence": summary.confidence,
                "recommendation": summary.recommendations[0] if summary.recommendations else "No action required",
            }
        else:
            status_emoji = {"pass": "PASS", "fail": "FAIL", "review_required": "REVIEW NEEDED"}.get(summary.status, summary.status)
            return f"Quality Assessment: {status_emoji} (Confidence: {summary.confidence:.0%})"

    def _generate_recommendations(
        self,
        qa_results: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on QA results."""
        recommendations = []

        # Based on gate decision
        if qa_results.get("gate_decision"):
            gd = qa_results["gate_decision"]
            status = gd.get("status") if isinstance(gd, dict) else getattr(gd, "status", None)
            if status:
                status_val = status.value if hasattr(status, "value") else str(status)
                if status_val == "review_required":
                    recommendations.append("Product requires expert review before release")
                elif status_val == "fail":
                    recommendations.append("Product should not be released in current state")

        # Based on sanity
        if qa_results.get("sanity"):
            sanity = qa_results["sanity"]
            if isinstance(sanity, dict):
                if not sanity.get("passed", True):
                    recommendations.append("Address sanity check failures before proceeding")
                if sanity.get("critical_count", 0) > 0:
                    recommendations.append("Critical issues detected - immediate attention required")
            else:
                if not getattr(sanity, "passed", True):
                    recommendations.append("Address sanity check failures before proceeding")

        # Based on uncertainty
        if qa_results.get("uncertainty"):
            unc = qa_results["uncertainty"]
            mean_unc = unc.get("mean_uncertainty", 0) if isinstance(unc, dict) else getattr(unc, "mean_uncertainty", 0)
            if mean_unc > 0.4:
                recommendations.append("High uncertainty detected - consider additional data sources")

        if not recommendations:
            recommendations.append("Product meets quality standards for release")

        return recommendations

    def _get_included_sections(self, qa_results: Dict[str, Any]) -> List[str]:
        """Get list of sections included in report."""
        sections = ["summary"]
        if qa_results.get("sanity"):
            sections.append("sanity_checks")
        if qa_results.get("validation"):
            sections.append("validation")
        if qa_results.get("uncertainty"):
            sections.append("uncertainty")
        if qa_results.get("gate_decision"):
            sections.append("gate_decision")
        return sections

    def _get_file_extension(self, format: ReportFormat) -> str:
        """Get file extension for format."""
        extensions = {
            ReportFormat.JSON: "json",
            ReportFormat.HTML: "html",
            ReportFormat.MARKDOWN: "md",
            ReportFormat.TEXT: "txt",
        }
        return extensions.get(format, "txt")
