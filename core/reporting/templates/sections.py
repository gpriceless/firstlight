"""
FirstLight Reporting - Reusable Section Generators

Generates individual report sections as standalone HTML fragments.
"""

from typing import Dict, List, Any, Optional
from .full_report import WhatHappenedSection, WhoIsAffectedSection


class SectionGenerator:
    """Generate individual report sections as HTML fragments."""

    def __init__(self, template_engine):
        """
        Initialize section generator.

        Args:
            template_engine: ReportTemplateEngine instance
        """
        self.engine = template_engine

    def what_happened_section(self, data: WhatHappenedSection) -> str:
        """
        Generate 'What Happened' HTML section.

        Args:
            data: What happened section data

        Returns:
            HTML section fragment

        Example:
            >>> section = generator.what_happened_section(
            ...     WhatHappenedSection(
            ...         event_description="Hurricane Ian caused severe flooding...",
            ...         timeline=[
            ...             {"date": "Sep 23", "event": "Storm forms"},
            ...             {"date": "Sep 28", "event": "Landfall"}
            ...         ],
            ...         affected_areas=["Downtown", "Riverside"],
            ...         severity_description="severe flooding"
            ...     )
            ... )
        """
        context = {
            "description": data.event_description,
            "timeline": data.timeline,
            "affected_areas": data.affected_areas,
            "severity": data.severity_description,
            "breakdown": data.area_breakdown or [],
        }
        return self.engine.render("sections/what_happened.html", context)

    def who_is_affected_section(self, data: WhoIsAffectedSection) -> str:
        """
        Generate 'Who Is Affected' HTML section.

        Args:
            data: Who is affected section data

        Returns:
            HTML section fragment

        Example:
            >>> section = generator.who_is_affected_section(
            ...     WhoIsAffectedSection(
            ...         estimated_population=12000,
            ...         estimated_housing_units=4800,
            ...         infrastructure={"Hospital": 3, "School": 15}
            ...     )
            ... )
        """
        context = {
            "population": data.estimated_population,
            "housing": data.estimated_housing_units,
            "vulnerable": data.vulnerable_populations,
            "infrastructure": data.infrastructure,
            "facilities": data.facilities or [],
            "transportation": data.transportation,
        }
        return self.engine.render("sections/who_is_affected.html", context)

    def emergency_resources_section(
        self,
        contacts: List[Dict[str, str]],
        actions: List[str],
        resources: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate emergency resources HTML section.

        Args:
            contacts: Emergency contact list
                [{name, number, type}, ...]
            actions: What to do action list
            resources: Additional resource links (optional)
                [{name, url, description}, ...]

        Returns:
            HTML section fragment

        Example:
            >>> section = generator.emergency_resources_section(
            ...     contacts=[
            ...         {"name": "FEMA", "number": "1-800-621-3362", "type": "Federal"},
            ...         {"name": "Local EM", "number": "239-555-0100", "type": "Local"}
            ...     ],
            ...     actions=[
            ...         "Do not enter flood waters",
            ...         "Check on neighbors",
            ...         "Document damage for insurance"
            ...     ]
            ... )
        """
        context = {
            "contacts": contacts,
            "actions": actions,
            "resources": resources or [],
        }
        return self.engine.render("sections/emergency_resources.html", context)

    def technical_appendix(
        self,
        methodology: str,
        qc_results: Dict[str, Any],
        data_sources: Optional[List[str]] = None,
        processing_details: Optional[Dict[str, Any]] = None,
        validation_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate technical appendix HTML section.

        Args:
            methodology: Methodology description
            qc_results: Quality control results
            data_sources: Data source list (optional)
            processing_details: Processing parameters (optional)
            validation_metrics: Validation metrics (optional)

        Returns:
            HTML section fragment

        Example:
            >>> section = generator.technical_appendix(
            ...     methodology="Flood detection using SAR imagery...",
            ...     qc_results={
            ...         "geometric_check": "PASS",
            ...         "radiometric_check": "PASS",
            ...         "cloud_cover": "2.3%"
            ...     },
            ...     data_sources=["Sentinel-1", "Sentinel-2"]
            ... )
        """
        context = {
            "methodology": methodology,
            "qc_results": qc_results,
            "data_sources": data_sources or [],
            "processing": processing_details,
            "validation": validation_metrics,
        }
        return self.engine.render("sections/technical_appendix.html", context)

    def table_of_contents(
        self,
        sections: List[Dict[str, Any]]
    ) -> str:
        """
        Generate table of contents HTML.

        Args:
            sections: Section list with page numbers
                [
                    {
                        "number": "1",
                        "title": "Executive Summary",
                        "page": 3,
                        "subsections": [
                            {"number": "1.1", "title": "Overview", "page": 3}
                        ]
                    },
                    ...
                ]

        Returns:
            HTML table of contents

        Example:
            >>> toc = generator.table_of_contents([
            ...     {"number": "1", "title": "Executive Summary", "page": 3},
            ...     {"number": "2", "title": "What Happened", "page": 4}
            ... ])
        """
        context = {"sections": sections}
        return self.engine.render("sections/table_of_contents.html", context)

    def cover_page(
        self,
        event_name: str,
        event_type: str,
        location: str,
        event_date: str,
        report_id: str,
        generated_date: str,
        classification: str = "UNCLASSIFIED",
        key_stats: Optional[Dict[str, Any]] = None,
        hero_image: Optional[str] = None
    ) -> str:
        """
        Generate cover page HTML.

        Args:
            event_name: Event name
            event_type: Event type
            location: Location
            event_date: Event date (formatted)
            report_id: Report identifier
            generated_date: Report generation date
            classification: Security classification
            key_stats: Key statistics for cover
            hero_image: Hero image path/URL

        Returns:
            HTML cover page

        Example:
            >>> cover = generator.cover_page(
            ...     event_name="Hurricane Ian",
            ...     event_type="Flood Analysis",
            ...     location="Fort Myers, Florida",
            ...     event_date="September 28, 2022",
            ...     report_id="FL-IAN-2022-001",
            ...     generated_date="January 26, 2026"
            ... )
        """
        context = {
            "event": {
                "name": event_name,
                "type": event_type,
                "location": location,
                "date": event_date,
            },
            "report": {
                "id": report_id,
                "generated": generated_date,
                "classification": classification,
            },
            "stats": key_stats or {},
            "hero_image": hero_image,
        }
        return self.engine.render("sections/cover_page.html", context)

    def map_page(
        self,
        title: str,
        subtitle: str,
        map_image: str,
        legend_items: List[Dict[str, str]],
        scale: str,
        data_sources: str,
        analysis_date: str,
        coordinate_system: str,
        caption: Optional[str] = None
    ) -> str:
        """
        Generate map page HTML.

        Args:
            title: Map title
            subtitle: Map subtitle
            map_image: Map image path/URL
            legend_items: Legend items [{color, label}, ...]
            scale: Scale text (e.g., "0 --- 2 km")
            data_sources: Data sources text
            analysis_date: Analysis date
            coordinate_system: CRS description
            caption: Optional caption

        Returns:
            HTML map page

        Example:
            >>> map_page = generator.map_page(
            ...     title="Flood Extent Map",
            ...     subtitle="Hurricane Ian | Fort Myers, FL | Sep 28, 2022",
            ...     map_image="maps/flood_extent.png",
            ...     legend_items=[
            ...         {"color": "#90CDF4", "label": "Minor (<0.3m)"},
            ...         {"color": "#4299E1", "label": "Moderate (0.3-1m)"}
            ...     ],
            ...     scale="0 --- 2 km",
            ...     data_sources="Sentinel-1, Sentinel-2",
            ...     analysis_date="September 29, 2022",
            ...     coordinate_system="WGS 84 / UTM Zone 17N"
            ... )
        """
        context = {
            "title": title,
            "subtitle": subtitle,
            "map_image": map_image,
            "legend": legend_items,
            "scale": scale,
            "sources": data_sources,
            "analysis_date": analysis_date,
            "crs": coordinate_system,
            "caption": caption,
        }
        return self.engine.render("sections/map_page.html", context)
