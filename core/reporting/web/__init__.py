"""
Interactive web report generation.

This module provides HTML report generation with embedded interactive maps,
before/after sliders, and mobile-responsive layouts.
"""

from core.reporting.web.interactive_report import (
    InteractiveReportGenerator,
    WebReportConfig,
)

__all__ = [
    'InteractiveReportGenerator',
    'WebReportConfig',
]
