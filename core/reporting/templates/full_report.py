"""
FirstLight Reporting - Full Report Generator

Generates comprehensive multi-section analysis reports for detailed documentation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class WhatHappenedSection:
    """Data for 'What Happened' section."""

    # Plain language description
    event_description: str

    # Timeline entries: [{date, event}, ...]
    timeline: List[Dict[str, str]] = field(default_factory=list)

    # Affected area names (neighborhoods, districts)
    affected_areas: List[str] = field(default_factory=list)

    # Plain language severity description
    severity_description: str = "moderate impact"

    # Optional: Area-by-area breakdown
    area_breakdown: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self):
        """Validate data."""
        if not self.event_description:
            raise ValueError("event_description is required")


@dataclass
class WhoIsAffectedSection:
    """Data for 'Who Is Affected' section."""

    # Population estimates
    estimated_population: int
    estimated_housing_units: int

    # Vulnerable populations description (optional)
    vulnerable_populations: Optional[str] = None

    # Infrastructure counts: {type: count}
    # Example: {"Hospital": 3, "School": 15, "Fire Station": 2}
    infrastructure: Dict[str, int] = field(default_factory=dict)

    # Detailed facility list (optional)
    # [{name, type, status, address}, ...]
    facilities: Optional[List[Dict[str, str]]] = None

    # Transportation impact (optional)
    transportation: Optional[Dict[str, Any]] = None


@dataclass
class EmergencyResources:
    """Emergency contact and action information."""

    # Emergency contacts: [{name, number, type}, ...]
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list)

    # What to do actions
    what_to_do: List[str] = field(default_factory=list)

    # Additional resources (optional)
    resources: Optional[List[Dict[str, str]]] = None


@dataclass
class TechnicalDetails:
    """Technical appendix data."""

    # Methodology description
    methodology: str

    # Data sources list
    data_sources: List[str]

    # QC results: {check_name: result}
    qc_results: Dict[str, Any]

    # Overall confidence score (0-1)
    confidence_score: float

    # Processing details (optional)
    processing_details: Optional[Dict[str, Any]] = None

    # Validation metrics (optional)
    validation_metrics: Optional[Dict[str, Any]] = None


@dataclass
class FullReportData:
    """Complete data for full analysis report."""

    # Header information (required)
    event_name: str
    event_type: str  # "flood", "wildfire", etc.
    location: str
    event_date: datetime

    # Core sections (required)
    executive_summary: str
    what_happened: WhatHappenedSection
    who_is_affected: WhoIsAffectedSection
    technical: TechnicalDetails

    # Report identification (optional)
    report_date: datetime = field(default_factory=datetime.now)
    report_id: Optional[str] = None
    classification: str = "UNCLASSIFIED"

    # Emergency resources (optional)
    emergency: EmergencyResources = field(default_factory=EmergencyResources)

    # Key metrics for cover page (optional)
    affected_area_hectares: Optional[float] = None
    severity: str = "moderate"
    confidence_score: Optional[float] = None

    # Map placeholders (optional)
    maps: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Generate report ID if not provided."""
        if self.report_id is None:
            # Generate ID from location and date
            location_code = self.location[:2].upper()
            event_code = self.event_type[:4].upper()
            date_str = self.event_date.strftime("%Y%m%d")
            self.report_id = f"{location_code}-{event_code}-{date_str}"


class FullReportGenerator:
    """Generate complete multi-section analysis reports."""

    def __init__(self, template_engine):
        """
        Initialize generator.

        Args:
            template_engine: ReportTemplateEngine instance
        """
        self.engine = template_engine

    def generate(self, data: FullReportData) -> str:
        """
        Generate complete HTML report.

        Args:
            data: Full report data

        Returns:
            Rendered HTML report

        Example:
            >>> engine = ReportTemplateEngine()
            >>> generator = FullReportGenerator(engine)
            >>> data = FullReportData(
            ...     event_name="Hurricane Ian",
            ...     event_type="flood",
            ...     location="Fort Myers, FL",
            ...     event_date=datetime(2022, 9, 28),
            ...     executive_summary="Hurricane Ian caused severe flooding...",
            ...     what_happened=WhatHappenedSection(...),
            ...     who_is_affected=WhoIsAffectedSection(...),
            ...     technical=TechnicalDetails(...)
            ... )
            >>> html = generator.generate(data)
        """
        context = self._build_context(data)
        return self.engine.render("full_report.html", context)

    def _build_context(self, data: FullReportData) -> dict:
        """
        Build template context from data.

        Args:
            data: Full report data

        Returns:
            Dictionary suitable for template rendering
        """
        return {
            # Header info
            "event": {
                "name": data.event_name,
                "type": data.event_type,
                "location": data.location,
                "date": data.event_date.strftime("%B %d, %Y"),
                "date_iso": data.event_date.isoformat(),
            },
            "report": {
                "id": data.report_id,
                "date": data.report_date.strftime("%B %d, %Y"),
                "date_iso": data.report_date.isoformat(),
                "classification": data.classification,
            },

            # Key metrics
            "metrics": {
                "area_hectares": data.affected_area_hectares,
                "severity": data.severity,
                "confidence": (data.confidence_score * 100
                             if data.confidence_score else
                             data.technical.confidence_score * 100),
            },

            # Content sections
            "executive_summary": data.executive_summary,
            "what_happened": self._format_what_happened(data.what_happened),
            "who_is_affected": self._format_who_is_affected(data.who_is_affected),

            # Technical appendix
            "technical": {
                "methodology": data.technical.methodology,
                "data_sources": data.technical.data_sources,
                "qc_results": data.technical.qc_results,
                "confidence": data.technical.confidence_score * 100,
                "processing": data.technical.processing_details,
                "validation": data.technical.validation_metrics,
            },

            # Emergency info
            "emergency": {
                "contacts": data.emergency.emergency_contacts,
                "actions": data.emergency.what_to_do,
                "resources": data.emergency.resources,
            },

            # Maps
            "maps": data.maps,

            # Metadata
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        }

    def _format_what_happened(self, section: WhatHappenedSection) -> dict:
        """Format 'What Happened' section data."""
        return {
            "description": section.event_description,
            "timeline": section.timeline,
            "affected_areas": section.affected_areas,
            "severity_description": section.severity_description,
            "area_breakdown": section.area_breakdown or [],
        }

    def _format_who_is_affected(self, section: WhoIsAffectedSection) -> dict:
        """Format 'Who Is Affected' section data."""
        return {
            "population": section.estimated_population,
            "housing_units": section.estimated_housing_units,
            "vulnerable_populations": section.vulnerable_populations,
            "infrastructure": section.infrastructure,
            "facilities": section.facilities or [],
            "transportation": section.transportation,
        }
