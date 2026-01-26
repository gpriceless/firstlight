"""
FirstLight Reporting - Executive Summary Generator

Generates executive summary reports in plain language for decision-makers.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class ExecutiveSummaryData:
    """
    Data for executive summary report.

    Contains all information needed to generate a one-page
    executive summary suitable for emergency managers and officials.
    """

    # Event information
    event_name: str
    event_type: str  # "flood", "wildfire", "hurricane"
    location: str
    event_date: datetime

    # Key metrics
    affected_area_hectares: float
    affected_area_percent: float
    confidence_score: float

    # Impact estimates
    estimated_population: Optional[int] = None
    estimated_housing_units: Optional[int] = None
    infrastructure_affected: Optional[dict] = None

    # Analysis metadata
    analysis_date: Optional[datetime] = None
    data_sources: Optional[List[str]] = None

    # Status indicators
    severity: str = "moderate"  # minimal, moderate, significant, severe, extreme
    qc_status: str = "PASS"

    def __post_init__(self):
        """Set defaults and validate data."""
        if self.analysis_date is None:
            self.analysis_date = datetime.now()

        if self.data_sources is None:
            self.data_sources = ["Satellite imagery"]

        # Validate severity level
        valid_severities = ["minimal", "moderate", "significant", "severe", "extreme"]
        if self.severity not in valid_severities:
            raise ValueError(
                f"Invalid severity '{self.severity}'. "
                f"Must be one of: {', '.join(valid_severities)}"
            )


class ExecutiveSummaryGenerator:
    """
    Generate executive summary reports.

    Produces plain-language HTML reports suitable for non-technical
    audiences including emergency managers, local officials, and media.
    """

    def __init__(self, template_engine):
        """
        Initialize generator.

        Args:
            template_engine: ReportTemplateEngine instance
        """
        self.engine = template_engine

    def generate(self, data: ExecutiveSummaryData) -> str:
        """
        Generate HTML executive summary.

        Args:
            data: Executive summary data

        Returns:
            Rendered HTML report

        Example:
            >>> engine = ReportTemplateEngine()
            >>> generator = ExecutiveSummaryGenerator(engine)
            >>> data = ExecutiveSummaryData(
            ...     event_name="Hurricane Ian",
            ...     event_type="flood",
            ...     location="Fort Myers, FL",
            ...     event_date=datetime(2022, 9, 28),
            ...     affected_area_hectares=3026.5,
            ...     affected_area_percent=28.9,
            ...     confidence_score=0.90,
            ...     severity="severe"
            ... )
            >>> html = generator.generate(data)
        """
        context = self._build_context(data)
        context['plain_language_summary'] = self.generate_plain_language_summary(data)
        return self.engine.render("executive_summary.html", context)

    def _build_context(self, data: ExecutiveSummaryData) -> dict:
        """
        Build template context from data.

        Args:
            data: Executive summary data

        Returns:
            Dictionary suitable for template rendering
        """
        return {
            "event": {
                "name": data.event_name,
                "type": data.event_type,
                "location": data.location,
                "date": data.event_date.strftime("%B %d, %Y"),
            },
            "metrics": {
                "area_hectares": data.affected_area_hectares,
                "area_percent": data.affected_area_percent,
                "confidence": data.confidence_score * 100,  # Convert to percentage
                "severity": data.severity,
            },
            "impact": {
                "population": data.estimated_population,
                "housing": data.estimated_housing_units,
                "infrastructure": data.infrastructure_affected or {},
            },
            "qc_status": data.qc_status,
            "data_sources": ", ".join(data.data_sources) if data.data_sources else "N/A",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        }

    def generate_plain_language_summary(self, data: ExecutiveSummaryData) -> str:
        """
        Generate plain-language summary paragraph.

        Creates a narrative summary suitable for non-technical readers,
        using relatable comparisons and avoiding jargon.

        Args:
            data: Executive summary data

        Returns:
            Plain English summary paragraph

        Example:
            "Hurricane Ian caused severe flooding across the Fort Myers area.
            Our satellite analysis detected approximately 3,027 hectares of
            standing water, covering about 28.9% of the study area. An estimated
            12,000 people may be affected."
        """
        severity_words = {
            "minimal": "minor",
            "moderate": "moderate",
            "significant": "significant",
            "severe": "severe",
            "extreme": "catastrophic",
        }

        event_type_text = {
            "flood": "flooding",
            "wildfire": "fire damage",
            "hurricane": "hurricane impacts",
        }

        severity_desc = severity_words.get(data.severity, "significant")
        event_desc = event_type_text.get(data.event_type, "damage")

        # Build summary
        summary = (
            f"{data.event_name} caused {severity_desc} {event_desc} "
            f"across the {data.location} area. "
        )

        # Add area information
        if data.event_type == "flood":
            summary += (
                f"Our satellite analysis detected approximately "
                f"{data.affected_area_hectares:,.0f} hectares of standing water, "
                f"covering about {data.affected_area_percent:.1f}% of the study area."
            )
        elif data.event_type == "wildfire":
            summary += (
                f"Our satellite analysis detected approximately "
                f"{data.affected_area_hectares:,.0f} hectares of burned area, "
                f"covering about {data.affected_area_percent:.1f}% of the study area."
            )
        else:
            summary += (
                f"Our analysis identified approximately "
                f"{data.affected_area_hectares:,.0f} hectares of affected area, "
                f"covering about {data.affected_area_percent:.1f}% of the study area."
            )

        # Add population impact if available
        if data.estimated_population:
            summary += f" An estimated {data.estimated_population:,} people may be affected."

        return summary.strip()
