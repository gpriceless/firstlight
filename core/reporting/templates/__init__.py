"""
FirstLight Reporting - Template System

Provides Jinja2-based template rendering for generating HTML reports.
"""

from .base import ReportTemplateEngine
from .executive_summary import (
    ExecutiveSummaryGenerator,
    ExecutiveSummaryData,
)

__all__ = [
    "ReportTemplateEngine",
    "ExecutiveSummaryGenerator",
    "ExecutiveSummaryData",
]
