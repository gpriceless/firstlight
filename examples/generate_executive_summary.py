#!/usr/bin/env python3
"""
Example: Generate an Executive Summary Report

Demonstrates how to use the template system to generate
a plain-language executive summary HTML report.
"""

from datetime import datetime
from pathlib import Path

from core.reporting.templates import (
    ReportTemplateEngine,
    ExecutiveSummaryGenerator,
    ExecutiveSummaryData,
)


def main():
    """Generate example executive summary report."""

    # Create sample data for Hurricane Ian
    data = ExecutiveSummaryData(
        event_name="Hurricane Ian",
        event_type="flood",
        location="Fort Myers, FL",
        event_date=datetime(2022, 9, 28),
        affected_area_hectares=3026.5,
        affected_area_percent=28.9,
        confidence_score=0.90,
        estimated_population=12000,
        estimated_housing_units=4800,
        infrastructure_affected={
            "hospital": 3,
            "school": 15,
            "fire station": 2,
        },
        severity="severe",
        data_sources=["Sentinel-1 SAR", "Sentinel-2 Optical"],
    )

    # Initialize template engine and generator
    engine = ReportTemplateEngine()
    generator = ExecutiveSummaryGenerator(engine)

    # Generate HTML report
    html = generator.generate(data)

    # Save to file
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "hurricane_ian_executive_summary.html"
    output_file.write_text(html)

    print(f"✅ Executive summary generated: {output_file}")
    print(f"📄 Open in browser to view the report")

    # Also show the plain language summary
    summary = generator.generate_plain_language_summary(data)
    print(f"\n📝 Plain Language Summary:")
    print(f"{summary}")


if __name__ == "__main__":
    main()
