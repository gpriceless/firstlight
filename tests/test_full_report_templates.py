"""
Test full report template generation.
"""

from datetime import datetime
from core.reporting.templates.full_report import (
    FullReportData,
    WhatHappenedSection,
    WhoIsAffectedSection,
    TechnicalDetails,
    EmergencyResources,
)


def test_what_happened_section_creation():
    """Test WhatHappenedSection dataclass."""
    section = WhatHappenedSection(
        event_description="Hurricane Ian caused severe flooding",
        timeline=[
            {"date": "Sep 23", "event": "Storm forms"},
            {"date": "Sep 28", "event": "Landfall"},
        ],
        affected_areas=["Downtown", "Riverside"],
        severity_description="severe flooding",
    )

    assert section.event_description == "Hurricane Ian caused severe flooding"
    assert len(section.timeline) == 2
    assert len(section.affected_areas) == 2
    assert section.severity_description == "severe flooding"


def test_who_is_affected_section_creation():
    """Test WhoIsAffectedSection dataclass."""
    section = WhoIsAffectedSection(
        estimated_population=12000,
        estimated_housing_units=4800,
        infrastructure={"Hospital": 3, "School": 15},
    )

    assert section.estimated_population == 12000
    assert section.estimated_housing_units == 4800
    assert section.infrastructure["Hospital"] == 3
    assert section.infrastructure["School"] == 15


def test_technical_details_creation():
    """Test TechnicalDetails dataclass."""
    technical = TechnicalDetails(
        methodology="SAR-based flood detection",
        data_sources=["Sentinel-1", "Sentinel-2"],
        qc_results={"geometric_check": "PASS", "radiometric_check": "PASS"},
        confidence_score=0.90,
    )

    assert technical.methodology == "SAR-based flood detection"
    assert len(technical.data_sources) == 2
    assert technical.qc_results["geometric_check"] == "PASS"
    assert technical.confidence_score == 0.90


def test_emergency_resources_creation():
    """Test EmergencyResources dataclass."""
    emergency = EmergencyResources(
        emergency_contacts=[
            {"name": "FEMA", "number": "1-800-621-3362", "type": "Federal"}
        ],
        what_to_do=["Do not enter flood waters", "Check on neighbors"],
    )

    assert len(emergency.emergency_contacts) == 1
    assert len(emergency.what_to_do) == 2
    assert emergency.emergency_contacts[0]["name"] == "FEMA"


def test_full_report_data_creation():
    """Test FullReportData dataclass."""
    data = FullReportData(
        event_name="Hurricane Ian",
        event_type="flood",
        location="Fort Myers, FL",
        event_date=datetime(2022, 9, 28),
        executive_summary="Hurricane Ian caused severe flooding across Fort Myers.",
        what_happened=WhatHappenedSection(
            event_description="Storm surge and rainfall caused widespread flooding"
        ),
        who_is_affected=WhoIsAffectedSection(
            estimated_population=12000, estimated_housing_units=4800
        ),
        technical=TechnicalDetails(
            methodology="SAR analysis",
            data_sources=["Sentinel-1"],
            qc_results={"check": "PASS"},
            confidence_score=0.90,
        ),
    )

    assert data.event_name == "Hurricane Ian"
    assert data.event_type == "flood"
    assert data.location == "Fort Myers, FL"
    assert data.report_id is not None  # Should be auto-generated
    assert "FO" in data.report_id  # Location code
    assert "FLOO" in data.report_id  # Event code


def test_full_report_data_with_custom_report_id():
    """Test FullReportData with custom report ID."""
    data = FullReportData(
        event_name="Hurricane Ian",
        event_type="flood",
        location="Fort Myers, FL",
        event_date=datetime(2022, 9, 28),
        report_id="FL-IAN-2022-001",
        executive_summary="Test summary",
        what_happened=WhatHappenedSection(event_description="Test"),
        who_is_affected=WhoIsAffectedSection(
            estimated_population=1000, estimated_housing_units=500
        ),
        technical=TechnicalDetails(
            methodology="Test",
            data_sources=["Test"],
            qc_results={},
            confidence_score=0.5,
        ),
    )

    assert data.report_id == "FL-IAN-2022-001"


def test_what_happened_section_validation():
    """Test WhatHappenedSection validation."""
    try:
        WhatHappenedSection(event_description="")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "event_description is required" in str(e)


if __name__ == "__main__":
    # Run all tests
    tests = [
        test_what_happened_section_creation,
        test_who_is_affected_section_creation,
        test_technical_details_creation,
        test_emergency_resources_creation,
        test_full_report_data_creation,
        test_full_report_data_with_custom_report_id,
        test_what_happened_section_validation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
