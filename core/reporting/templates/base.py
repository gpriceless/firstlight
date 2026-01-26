"""
FirstLight Reporting - Base Template Engine

Provides Jinja2 template rendering with custom filters for human-readable output.
"""

from pathlib import Path
from typing import Any, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape


class ReportTemplateEngine:
    """
    Template engine for generating HTML reports.

    Uses Jinja2 for templating with custom filters for formatting
    numbers, areas, and populations in human-readable formats.

    Security: Autoescape is enabled for HTML and XML to prevent XSS.
    """

    def __init__(self, template_dir: Optional[Path] = None):
        """
        Initialize template engine.

        Args:
            template_dir: Directory containing HTML templates.
                         Defaults to templates/html subdirectory.
        """
        self.template_dir = template_dir or Path(__file__).parent / "html"
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
        )
        self._register_filters()

    def _register_filters(self):
        """Register custom Jinja2 filters for data formatting."""
        self.env.filters['format_number'] = lambda x: f"{x:,}" if x is not None else "N/A"
        self.env.filters['format_hectares'] = self._format_hectares
        self.env.filters['format_population'] = self._format_population
        self.env.filters['format_percent'] = self._format_percent

    @staticmethod
    def _format_hectares(value: Optional[float]) -> str:
        """
        Convert hectares to human-readable format with multiple units.

        Args:
            value: Area in hectares

        Returns:
            Formatted string with hectares, acres, and football field equivalents

        Example:
            3026.5 hectares -> "3,027 hectares (7,479 acres, ~9,900 football fields)"
        """
        if value is None:
            return "N/A"

        acres = value * 2.471
        # 1 American football field (with end zones) = ~0.535 hectares
        football_fields = int(value / 0.535)

        return (
            f"{value:,.0f} hectares "
            f"({acres:,.0f} acres, "
            f"~{football_fields:,} football fields)"
        )

    @staticmethod
    def _format_population(value: Optional[int]) -> str:
        """
        Format population with appropriate scale.

        Args:
            value: Population count

        Returns:
            Formatted string with appropriate scale (million, thousand, etc.)

        Example:
            12000 -> "12 thousand people"
            1500000 -> "1.5 million people"
        """
        if value is None:
            return "N/A"

        if value >= 1_000_000:
            return f"{value/1_000_000:.1f} million people"
        elif value >= 1_000:
            return f"{value/1_000:.0f} thousand people"
        return f"{value:,} people"

    @staticmethod
    def _format_percent(value: Optional[float], decimals: int = 1) -> str:
        """
        Format percentage value.

        Args:
            value: Percentage (0-100)
            decimals: Number of decimal places

        Returns:
            Formatted percentage string

        Example:
            28.87 -> "28.9%"
        """
        if value is None:
            return "N/A"

        return f"{value:.{decimals}f}%"

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render a template with context data.

        Args:
            template_name: Name of the template file (e.g., "executive_summary.html")
            context: Dictionary of template variables

        Returns:
            Rendered HTML string

        Raises:
            jinja2.TemplateNotFound: If template file doesn't exist
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
