"""
FirstLight Reporting System

Human-readable report generation for geospatial event analysis.

Modules:
- templates: Jinja2-based HTML report generation
- components: Reusable UI components (metric cards, alerts, etc.)
- maps: Static and interactive map generation
- data: External data integrations (Census, OSM, emergency resources)
- web: Interactive web reports
- pdf: PDF report generation
- utils: Utility functions (color handling, etc.)
"""

from core.reporting.templates import (
    ReportTemplateEngine,
    ExecutiveSummaryGenerator,
    ExecutiveSummaryData,
)

__all__ = [
    "ReportTemplateEngine",
    "ExecutiveSummaryGenerator",
    "ExecutiveSummaryData",
]
