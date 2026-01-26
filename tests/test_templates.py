"""
Tests for FirstLight reporting templates.
"""

import pytest
from datetime import datetime
from pathlib import Path

from core.reporting.templates import (
    ReportTemplateEngine,
    ExecutiveSummaryGenerator,
    ExecutiveSummaryData,
)


class TestReportTemplateEngine:
    """Test the base template engine."""

    def test_init_default_template_dir(self):
        """Test initialization with default template directory."""
        engine = ReportTemplateEngine()
        assert engine.template_dir.exists()
        assert engine.template_dir.name == "html"

    def test_init_custom_template_dir(self, tmp_path):
        """Test initialization with custom template directory."""
        custom_dir = tmp_path / "templates"
        custom_dir.mkdir()
        engine = ReportTemplateEngine(template_dir=custom_dir)
        assert engine.template_dir == custom_dir

    def test_format_number_filter(self):
        """Test number formatting filter."""
        engine = ReportTemplateEngine()
        filter_func = engine.env.filters['format_number']

        assert filter_func(1000) == "1,000"
        assert filter_func(1000000) == "1,000,000"
        assert filter_func(None) == "N/A"

    def test_format_hectares_filter(self):
        """Test hectares formatting with multiple units."""
        engine = ReportTemplateEngine()
        result = engine._format_hectares(3026.5)

        assert "3,026 hectares" in result  # Note: 3026.5 rounds to 3,026
        assert "7,478 acres" in result
        assert "football fields" in result

    def test_format_hectares_none(self):
        """Test hectares formatting with None value."""
        engine = ReportTemplateEngine()
        result = engine._format_hectares(None)
        assert result == "N/A"

    def test_format_population_millions(self):
        """Test population formatting for millions."""
        engine = ReportTemplateEngine()
        result = engine._format_population(1_500_000)
        assert result == "1.5 million people"

    def test_format_population_thousands(self):
        """Test population formatting for thousands."""
        engine = ReportTemplateEngine()
        result = engine._format_population(12_000)
        assert result == "12 thousand people"

    def test_format_population_hundreds(self):
        """Test population formatting for hundreds."""
        engine = ReportTemplateEngine()
        result = engine._format_population(500)
        assert result == "500 people"

    def test_format_population_none(self):
        """Test population formatting with None value."""
        engine = ReportTemplateEngine()
        result = engine._format_population(None)
        assert result == "N/A"

    def test_format_percent(self):
        """Test percentage formatting."""
        engine = ReportTemplateEngine()

        assert engine._format_percent(28.87, 1) == "28.9%"
        assert engine._format_percent(100.0, 0) == "100%"
        assert engine._format_percent(None) == "N/A"


class TestExecutiveSummaryData:
    """Test the executive summary data model."""

    def test_valid_initialization(self):
        """Test creating valid summary data."""
        data = ExecutiveSummaryData(
            event_name="Hurricane Ian",
            event_type="flood",
            location="Fort Myers, FL",
            event_date=datetime(2022, 9, 28),
            affected_area_hectares=3026.5,
            affected_area_percent=28.9,
            confidence_score=0.90,
        )

        assert data.event_name == "Hurricane Ian"
        assert data.severity == "moderate"  # default
        assert data.qc_status == "PASS"  # default
        assert data.analysis_date is not None  # auto-set
        assert data.data_sources is not None  # auto-set

    def test_invalid_severity(self):
        """Test that invalid severity raises error."""
        with pytest.raises(ValueError, match="Invalid severity"):
            ExecutiveSummaryData(
                event_name="Test",
                event_type="flood",
                location="Test Location",
                event_date=datetime.now(),
                affected_area_hectares=100,
                affected_area_percent=10,
                confidence_score=0.8,
                severity="invalid",
            )

    def test_all_severity_levels(self):
        """Test all valid severity levels."""
        valid_severities = ["minimal", "moderate", "significant", "severe", "extreme"]

        for severity in valid_severities:
            data = ExecutiveSummaryData(
                event_name="Test",
                event_type="flood",
                location="Test",
                event_date=datetime.now(),
                affected_area_hectares=100,
                affected_area_percent=10,
                confidence_score=0.8,
                severity=severity,
            )
            assert data.severity == severity


class TestExecutiveSummaryGenerator:
    """Test the executive summary generator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample executive summary data."""
        return ExecutiveSummaryData(
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
        )

    def test_plain_language_summary_flood(self, sample_data):
        """Test plain language summary for flood events."""
        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)

        summary = generator.generate_plain_language_summary(sample_data)

        assert "Hurricane Ian" in summary
        assert "severe flooding" in summary
        assert "Fort Myers, FL" in summary
        assert "3,026 hectares" in summary  # Note: 3026.5 rounds to 3,026
        assert "28.9%" in summary
        assert "12,000 people" in summary

    def test_plain_language_summary_wildfire(self):
        """Test plain language summary for wildfire events."""
        data = ExecutiveSummaryData(
            event_name="Camp Fire",
            event_type="wildfire",
            location="Paradise, CA",
            event_date=datetime(2018, 11, 8),
            affected_area_hectares=5000,
            affected_area_percent=75.0,
            confidence_score=0.95,
            severity="extreme",
        )

        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)
        summary = generator.generate_plain_language_summary(data)

        assert "catastrophic fire damage" in summary
        assert "burned area" in summary

    def test_build_context(self, sample_data):
        """Test building template context from data."""
        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)

        context = generator._build_context(sample_data)

        assert context["event"]["name"] == "Hurricane Ian"
        assert context["event"]["type"] == "flood"
        assert context["event"]["location"] == "Fort Myers, FL"
        assert context["metrics"]["area_hectares"] == 3026.5
        assert context["metrics"]["confidence"] == 90.0  # Converted to percentage
        assert context["impact"]["population"] == 12000
        assert context["impact"]["housing"] == 4800
        assert "hospital" in context["impact"]["infrastructure"]

    def test_generate_html(self, sample_data):
        """Test generating complete HTML report."""
        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)

        html = generator.generate(sample_data)

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<title>Hurricane Ian - Executive Summary</title>" in html
        assert "HURRICANE IAN" in html
        assert "Fort Myers, FL" in html

        # Check metrics are present
        assert "3,026" in html  # Formatted hectares
        assert "28.9%" in html
        assert "12,000" in html  # Population
        assert "90" in html  # Confidence percentage

        # Check severity badge
        assert "SEVERE" in html

        # Check infrastructure
        assert "hospital" in html.lower()
        assert "school" in html.lower()

    def test_generate_html_minimal_data(self):
        """Test generating HTML with minimal data (no population, infrastructure)."""
        data = ExecutiveSummaryData(
            event_name="Test Event",
            event_type="flood",
            location="Test Location",
            event_date=datetime(2024, 1, 1),
            affected_area_hectares=100,
            affected_area_percent=10,
            confidence_score=0.8,
        )

        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)
        html = generator.generate(data)

        assert "<!DOCTYPE html>" in html
        assert "Test Event" in html
        assert "100" in html  # Area
        assert "10" in html or "10.0" in html  # Percentage

    def test_xss_protection(self):
        """Test that XSS attacks are prevented by autoescape."""
        data = ExecutiveSummaryData(
            event_name="<script>alert('xss')</script>",
            event_type="flood",
            location="<img src=x onerror=alert(1)>",
            event_date=datetime.now(),
            affected_area_hectares=100,
            affected_area_percent=10,
            confidence_score=0.8,
        )

        engine = ReportTemplateEngine()
        generator = ExecutiveSummaryGenerator(engine)
        html = generator.generate(data)

        # Check that scripts are escaped
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        # Check that dangerous HTML is escaped (even if attribute name remains)
        assert "alert" not in html or "&lt;" in html  # Script content should be escaped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
