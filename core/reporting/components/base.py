"""
FirstLight Reporting - Base Component Classes

Provides dataclasses that represent report components and render to HTML.
All components generate HTML using CSS classes from the design system.

Security: All user-provided content is escaped using html.escape() to prevent XSS.
"""

import html
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
from abc import ABC, abstractmethod


class ComponentStatus(Enum):
    """Status levels for components that indicate severity or importance."""
    SUCCESS = "success"
    WARNING = "warning"
    DANGER = "danger"
    INFO = "info"


class BaseComponent(ABC):
    """Base class for all report components."""

    @abstractmethod
    def to_html(self) -> str:
        """
        Render component to HTML string.

        Returns:
            HTML string representation of the component
        """
        pass

    def get_css_classes(self) -> list[str]:
        """
        Get CSS classes for this component.

        Returns:
            List of CSS class names
        """
        return []


@dataclass
class MetricCard(BaseComponent):
    """
    A metric display card for showing key statistics.

    Used for KPIs, summary stats, and quick facts in reports.
    Maps to .fl-metric-card CSS classes.
    """
    value: str
    label: str
    status: Optional[ComponentStatus] = None
    icon: Optional[str] = None
    subtitle: Optional[str] = None

    def to_html(self) -> str:
        """Render metric card to HTML."""
        status_class = f"fl-metric-card--{self.status.value}" if self.status else ""

        # Escape all user content
        safe_value = html.escape(str(self.value))
        safe_label = html.escape(self.label)
        safe_subtitle = html.escape(self.subtitle) if self.subtitle else None

        subtitle_html = f'<div class="fl-metric-card__subtext">{safe_subtitle}</div>' if safe_subtitle else ''

        return f'''<div class="fl-metric-card {status_class}">
    <div class="fl-metric-card__value">{safe_value}</div>
    <div class="fl-metric-card__label">{safe_label}</div>
    {subtitle_html}
</div>'''

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this metric card."""
        classes = ["fl-metric-card"]
        if self.status:
            classes.append(f"fl-metric-card--{self.status.value}")
        return classes


@dataclass
class AlertBox(BaseComponent):
    """
    An alert/notification box for contextual feedback messages.

    Used for warnings, errors, success messages, and informational content.
    Maps to .fl-alert CSS classes.
    """
    message: str
    level: ComponentStatus = ComponentStatus.INFO
    title: Optional[str] = None

    def to_html(self) -> str:
        """Render alert box to HTML."""
        # Escape all user content
        safe_message = html.escape(self.message)
        safe_title = html.escape(self.title) if self.title else None

        # Map status to alert level class
        level_map = {
            ComponentStatus.SUCCESS: "success",
            ComponentStatus.WARNING: "warning",
            ComponentStatus.DANGER: "critical",
            ComponentStatus.INFO: "info",
        }
        level_class = level_map.get(self.level, "info")

        title_html = f'<div class="fl-alert__title">{safe_title}</div>' if safe_title else ''

        return f'''<div class="fl-alert fl-alert--{level_class}">
    <div class="fl-alert__content">
        {title_html}
        <div class="fl-alert__body">{safe_message}</div>
    </div>
</div>'''

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this alert box."""
        level_map = {
            ComponentStatus.SUCCESS: "success",
            ComponentStatus.WARNING: "warning",
            ComponentStatus.DANGER: "critical",
            ComponentStatus.INFO: "info",
        }
        level_class = level_map.get(self.level, "info")
        return ["fl-alert", f"fl-alert--{level_class}"]


@dataclass
class SeverityBadge(BaseComponent):
    """
    A severity indicator badge.

    Used for flood/fire severity levels and other categorical data.
    Maps to .fl-badge CSS classes.

    Example levels:
    - Flood: "flood-minimal", "flood-minor", "flood-moderate", "flood-significant",
             "flood-severe", "flood-extreme"
    - Fire: "fire-unburned", "fire-low", "fire-mod-low", "fire-moderate",
            "fire-mod-high", "fire-high"
    - Generic: "success", "warning", "danger", "info", "uncertain"
    - Confidence: "conf-high", "conf-medium", "conf-low", "conf-very-low"
    """
    label: str
    level: str

    def to_html(self) -> str:
        """Render severity badge to HTML."""
        # Escape user content
        safe_label = html.escape(self.label)

        # Sanitize level to prevent arbitrary class injection
        # Level should match known badge patterns
        safe_level = "".join(c for c in self.level if c.isalnum() or c == "-").lower()

        return f'<span class="fl-badge fl-badge--{safe_level}">{safe_label}</span>'

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this badge."""
        safe_level = "".join(c for c in self.level if c.isalnum() or c == "-").lower()
        return ["fl-badge", f"fl-badge--{safe_level}"]


@dataclass
class DataTable(BaseComponent):
    """
    A data table for structured data presentation.

    Maps to .fl-table CSS classes.
    """
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    striped: bool = True
    hoverable: bool = False
    compact: bool = False

    def to_html(self) -> str:
        """Render data table to HTML."""
        # Build CSS classes
        classes = ["fl-table"]
        if self.striped:
            classes.append("fl-table--striped")
        if self.hoverable:
            classes.append("fl-table--hoverable")
        if self.compact:
            classes.append("fl-table--compact")

        table_classes = " ".join(classes)

        # Escape caption
        caption_html = ""
        if self.caption:
            safe_caption = html.escape(self.caption)
            caption_html = f"\n    <caption>{safe_caption}</caption>"

        # Build header row
        header_cells = []
        for header in self.headers:
            safe_header = html.escape(header)
            header_cells.append(f'        <th scope="col">{safe_header}</th>')
        header_row = "\n".join(header_cells)

        # Build data rows
        data_rows = []
        for row in self.rows:
            cells = []
            for cell in row:
                safe_cell = html.escape(str(cell))
                cells.append(f"            <td>{safe_cell}</td>")
            row_html = "\n".join(cells)
            data_rows.append(f"        <tr>\n{row_html}\n        </tr>")

        rows_html = "\n".join(data_rows)

        return f'''<table class="{table_classes}">{caption_html}
    <thead>
        <tr>
{header_row}
        </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
</table>'''

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this table."""
        classes = ["fl-table"]
        if self.striped:
            classes.append("fl-table--striped")
        if self.hoverable:
            classes.append("fl-table--hoverable")
        if self.compact:
            classes.append("fl-table--compact")
        return classes


@dataclass
class Legend(BaseComponent):
    """
    A map legend for data visualization.

    Supports both categorical (discrete items) and continuous (gradient) legends.
    Maps to .fl-legend CSS classes.
    """
    title: str
    items: List[tuple[str, str]]  # (color, label) pairs
    continuous: bool = False
    gradient_class: Optional[str] = None  # e.g., "flood", "fire", "confidence"

    def to_html(self) -> str:
        """Render legend to HTML."""
        safe_title = html.escape(self.title)

        if self.continuous:
            return self._render_continuous_legend(safe_title)
        else:
            return self._render_categorical_legend(safe_title)

    def _render_categorical_legend(self, safe_title: str) -> str:
        """Render categorical legend with discrete items."""
        legend_items = []
        for color, label in self.items:
            # Escape label
            safe_label = html.escape(label)
            # Sanitize color (should be hex or CSS color name)
            safe_color = html.escape(color)

            legend_items.append(
                f'        <div class="fl-legend__item">\n'
                f'            <div class="fl-legend__color" style="background-color: {safe_color};"></div>\n'
                f'            <div class="fl-legend__label">{safe_label}</div>\n'
                f'        </div>'
            )

        items_html = "\n".join(legend_items)

        return f'''<div class="fl-legend">
    <div class="fl-legend__title">{safe_title}</div>
    <div class="fl-legend__items">
{items_html}
    </div>
</div>'''

    def _render_continuous_legend(self, safe_title: str) -> str:
        """Render continuous gradient legend."""
        # Sanitize gradient class
        safe_gradient = ""
        if self.gradient_class:
            safe_gradient = "".join(c for c in self.gradient_class if c.isalnum() or c == "-").lower()

        gradient_html = f'<div class="fl-legend__gradient fl-legend__gradient--{safe_gradient}"></div>' if safe_gradient else ''

        # Build labels (min and max from items)
        labels_html = ""
        if self.items and len(self.items) >= 2:
            min_label = html.escape(self.items[0][1])
            max_label = html.escape(self.items[-1][1])
            labels_html = f'''        <div class="fl-legend__labels">
            <span>{min_label}</span>
            <span>{max_label}</span>
        </div>'''

        return f'''<div class="fl-legend fl-legend--continuous">
    <div class="fl-legend__title">{safe_title}</div>
    {gradient_html}
    {labels_html}
</div>'''

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this legend."""
        classes = ["fl-legend"]
        if self.continuous:
            classes.append("fl-legend--continuous")
        return classes


@dataclass
class SummaryGrid(BaseComponent):
    """
    A grid of metric cards for dashboard layouts.

    Maps to .fl-summary-grid CSS classes.
    """
    cards: List[MetricCard]
    columns: int = 4

    def to_html(self) -> str:
        """Render summary grid to HTML."""
        # Validate columns
        if self.columns not in [2, 3, 4]:
            self.columns = 4

        grid_class = f"fl-summary-grid--{self.columns}col"

        # Render all cards
        cards_html = []
        for card in self.cards:
            cards_html.append(f"    {card.to_html()}")

        cards_joined = "\n".join(cards_html)

        return f'''<div class="fl-summary-grid {grid_class}">
{cards_joined}
</div>'''

    def get_css_classes(self) -> list[str]:
        """Get CSS classes for this grid."""
        columns = self.columns if self.columns in [2, 3, 4] else 4
        return ["fl-summary-grid", f"fl-summary-grid--{columns}col"]
