"""
FirstLight Reporting Components

Base component classes for building HTML reports using the FirstLight design system.
All components render to HTML with proper CSS classes and XSS protection.
"""

from .base import (
    BaseComponent,
    ComponentStatus,
    MetricCard,
    AlertBox,
    SeverityBadge,
    DataTable,
    Legend,
    SummaryGrid,
)

__all__ = [
    "BaseComponent",
    "ComponentStatus",
    "MetricCard",
    "AlertBox",
    "SeverityBadge",
    "DataTable",
    "Legend",
    "SummaryGrid",
]
