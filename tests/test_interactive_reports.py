"""
Tests for interactive web report generation.

Tests the InteractiveReportGenerator including:
- HTML generation with embedded maps
- Before/after slider generation
- CSS and JS embedding
- Mobile responsive output
- Print styles
"""

import pytest
from pathlib import Path
from datetime import datetime

from core.reporting.web.interactive_report import (
    InteractiveReportGenerator,
    WebReportConfig,
)
from core.reporting.templates.full_report import (
    FullReportData,
    WhatHappenedSection,
    WhoIsAffectedSection,
    TechnicalDetails,
    EmergencyResources,
)
from core.reporting.maps.base import MapBounds


@pytest.fixture
def sample_report_data():
    """Create sample report data for testing."""
    return FullReportData(
        event_name="Hurricane Ian",
        event_type="flood",
        location="Fort Myers, FL",
        event_date=datetime(2022, 9, 28),
        report_date=datetime(2022, 9, 29),
        report_id="FL-FLOOD-20220928",
        executive_summary=(
            "Hurricane Ian caused severe flooding across Fort Myers. "
            "Approximately 3,026 hectares of standing water detected."
        ),
        what_happened=WhatHappenedSection(
            event_description="Hurricane Ian made landfall as Category 4 storm.",
            timeline=[
                {"date": "Sep 28", "event": "Landfall"},
                {"date": "Sep 29", "event": "Peak flooding"},
            ],
            affected_areas=["Downtown", "Riverside"],
            severity_description="Severe flooding in low-lying areas",
        ),
        who_is_affected=WhoIsAffectedSection(
            estimated_population=12000,
            estimated_housing_units=4800,
            infrastructure={
                "Hospital": 3,
                "School": 15,
                "Fire Station": 2,
            },
        ),
        technical=TechnicalDetails(
            methodology="Satellite-based flood detection using SAR and optical imagery.",
            data_sources=["Sentinel-1", "Sentinel-2"],
            qc_results={"geometry_valid": "PASS", "coverage": "PASS"},
            confidence_score=0.90,
        ),
        emergency=EmergencyResources(
            emergency_contacts=[
                {"name": "FEMA", "number": "1-800-621-3362", "type": "Federal Emergency"},
                {"name": "Local EM", "number": "239-555-0100", "type": "Local Emergency"},
            ],
            what_to_do=[
                "Do not enter flood waters",
                "Check on neighbors",
                "Document damage for insurance",
            ],
        ),
        affected_area_hectares=3026.5,
        severity="severe",
        confidence_score=0.90,
    )


@pytest.fixture
def sample_flood_geojson():
    """Create sample GeoJSON for testing."""
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-81.9, 26.4],
                        [-81.8, 26.4],
                        [-81.8, 26.5],
                        [-81.9, 26.5],
                        [-81.9, 26.4],
                    ]],
                },
                "properties": {
                    "severity": "moderate",
                },
            },
        ],
    }


@pytest.fixture
def sample_bounds():
    """Create sample map bounds."""
    return MapBounds(
        min_lon=-82.0,
        min_lat=26.3,
        max_lon=-81.7,
        max_lat=26.6,
    )


class TestWebReportConfig:
    """Test WebReportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WebReportConfig()

        assert config.include_interactive_map is True
        assert config.include_before_after_slider is True
        assert config.mobile_responsive is True
        assert config.embed_css is True
        assert config.embed_js is True
        assert config.enable_print_styles is True
        assert config.collapsible_sections is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebReportConfig(
            include_interactive_map=False,
            embed_css=False,
            collapsible_sections=False,
        )

        assert config.include_interactive_map is False
        assert config.embed_css is False
        assert config.collapsible_sections is False


class TestInteractiveReportGenerator:
    """Test InteractiveReportGenerator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = InteractiveReportGenerator()

        assert generator.config is not None
        assert generator.template_engine is not None
        assert generator.map_generator is not None

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = WebReportConfig(embed_css=False)
        generator = InteractiveReportGenerator(config)

        assert generator.config.embed_css is False

    def test_generate_basic_report(self, sample_report_data):
        """Test basic report generation without map or slider."""
        config = WebReportConfig(
            include_interactive_map=False,
            include_before_after_slider=False,
        )
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check HTML structure
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "Hurricane Ian" in html
        assert "Fort Myers, FL" in html
        assert "3,026" in html  # Area in hectares
        assert "12,000" in html  # Population

    def test_generate_with_embedded_map(
        self, sample_report_data, sample_flood_geojson, sample_bounds
    ):
        """Test report generation with embedded interactive map."""
        generator = InteractiveReportGenerator()

        html = generator.generate(
            report_data=sample_report_data,
            flood_geojson=sample_flood_geojson,
            bounds=sample_bounds,
        )

        # Check for map container
        assert "fl-map-container" in html
        # Map HTML should be embedded
        assert "folium" in html.lower() or "leaflet" in html.lower()

    def test_generate_with_infrastructure(
        self, sample_report_data, sample_flood_geojson, sample_bounds
    ):
        """Test report generation with infrastructure markers."""
        infrastructure = [
            {
                "lon": -81.85,
                "lat": 26.45,
                "type": "hospital",
                "name": "Memorial Hospital",
                "address": "123 Main St",
            },
            {
                "lon": -81.87,
                "lat": 26.47,
                "type": "school",
                "name": "Fort Myers High",
                "address": "456 Oak Ave",
            },
        ]

        generator = InteractiveReportGenerator()

        html = generator.generate(
            report_data=sample_report_data,
            flood_geojson=sample_flood_geojson,
            infrastructure=infrastructure,
            bounds=sample_bounds,
        )

        # Check HTML generated
        assert "Hurricane Ian" in html
        assert "fl-map-container" in html

    def test_generate_before_after_slider(self, sample_report_data):
        """Test before/after slider generation."""
        generator = InteractiveReportGenerator()

        html = generator.generate(
            report_data=sample_report_data,
            before_image_url="https://example.com/before.jpg",
            after_image_url="https://example.com/after.jpg",
        )

        # Check for slider elements
        assert "fl-before-after" in html
        assert "before.jpg" in html
        assert "after.jpg" in html
        assert "fl-before-after__slider" in html

    def test_embedded_css(self, sample_report_data):
        """Test CSS embedding."""
        config = WebReportConfig(embed_css=True)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for embedded CSS
        assert "<style>" in html
        assert ".fl-before-after" in html
        assert ".fl-sticky-header" in html
        assert ".fl-collapsible" in html

    def test_embedded_js(self, sample_report_data):
        """Test JavaScript embedding."""
        config = WebReportConfig(embed_js=True)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for embedded JavaScript
        assert "<script>" in html
        assert "initBeforeAfterSlider" in html
        assert "initCollapsibleSections" in html
        assert "initPrintButton" in html

    def test_external_css_js(self, sample_report_data):
        """Test external CSS and JS references."""
        config = WebReportConfig(embed_css=False, embed_js=False)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for external references
        assert "interactive.css" in html
        assert "interactive.js" in html
        # Should not have embedded styles
        assert "initBeforeAfterSlider" not in html

    def test_collapsible_sections(self, sample_report_data):
        """Test collapsible sections."""
        config = WebReportConfig(collapsible_sections=True)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for collapsible elements
        assert "fl-collapsible" in html
        assert "fl-collapsible__header" in html
        assert "fl-collapsible__content" in html
        assert "fl-collapsible__icon" in html

    def test_non_collapsible_sections(self, sample_report_data):
        """Test non-collapsible sections."""
        config = WebReportConfig(
            collapsible_sections=False,
            embed_css=False,
            embed_js=False
        )
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Should not have collapsible HTML structure when not embedded
        # (CSS/JS files may still contain these classes/functions)
        assert "fl-collapsible__header" not in html
        assert "fl-collapsible--expanded" not in html

    def test_sticky_header(self, sample_report_data):
        """Test sticky header presence."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for sticky header
        assert "fl-sticky-header" in html
        assert "Hurricane Ian" in html
        assert "Print" in html

    def test_emergency_resources(self, sample_report_data):
        """Test emergency resources section."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for emergency content
        assert "FEMA" in html
        assert "1-800-621-3362" in html
        assert "Do not enter flood waters" in html

    def test_technical_details(self, sample_report_data):
        """Test technical details section."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for technical content
        assert "Sentinel-1" in html
        assert "Sentinel-2" in html
        assert "90%" in html  # Confidence
        assert "SAR" in html

    def test_print_styles(self, sample_report_data):
        """Test print styles inclusion."""
        config = WebReportConfig(enable_print_styles=True)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for print media query
        assert "@media print" in html
        assert "no-print" in html

    def test_mobile_responsive(self, sample_report_data):
        """Test mobile responsive styles."""
        config = WebReportConfig(mobile_responsive=True)
        generator = InteractiveReportGenerator(config)

        html = generator.generate(sample_report_data)

        # Check for responsive styles
        assert "@media (max-width" in html or "viewport" in html

    def test_save_report(self, sample_report_data, tmp_path):
        """Test saving report to file."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)
        output_path = tmp_path / "report.html"

        saved_path = generator.save(html, output_path)

        # Verify file was created
        assert saved_path.exists()
        assert saved_path.read_text(encoding='utf-8') == html

    def test_save_creates_directories(self, sample_report_data, tmp_path):
        """Test save creates parent directories."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)
        output_path = tmp_path / "reports" / "2022" / "report.html"

        saved_path = generator.save(html, output_path)

        # Verify directory structure was created
        assert saved_path.exists()
        assert saved_path.parent.exists()

    def test_metadata_in_report(self, sample_report_data):
        """Test metadata presence in report."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for metadata
        assert "FL-FLOOD-20220928" in html  # Report ID
        assert "Generated" in html
        assert "FirstLight" in html

    def test_footer_disclaimer(self, sample_report_data):
        """Test footer disclaimer presence."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for disclaimer
        assert "automated analysis" in html.lower()
        assert "satellite imagery" in html.lower()
        assert "verify" in html.lower()


class TestBeforeAfterSlider:
    """Test before/after slider component generation."""

    def test_slider_html_structure(self):
        """Test slider HTML structure."""
        generator = InteractiveReportGenerator()

        slider_html = generator._generate_before_after_slider(
            before_url="https://example.com/before.jpg",
            after_url="https://example.com/after.jpg",
            before_date="September 25, 2022",
            after_date="September 29, 2022",
        )

        # Check structure
        assert "fl-before-after" in slider_html
        assert "fl-before-after__container" in slider_html
        assert "fl-before-after__wrapper" in slider_html
        assert "fl-before-after__before" in slider_html
        assert "fl-before-after__after" in slider_html
        assert "fl-before-after__slider" in slider_html
        assert "fl-before-after__handle" in slider_html
        assert "fl-before-after__labels" in slider_html

    def test_slider_urls(self):
        """Test URLs are properly embedded."""
        generator = InteractiveReportGenerator()

        slider_html = generator._generate_before_after_slider(
            before_url="https://example.com/before.jpg",
            after_url="https://example.com/after.jpg",
            before_date="Sep 25",
            after_date="Sep 29",
        )

        assert "https://example.com/before.jpg" in slider_html
        assert "https://example.com/after.jpg" in slider_html

    def test_slider_dates(self):
        """Test dates are properly displayed."""
        generator = InteractiveReportGenerator()

        slider_html = generator._generate_before_after_slider(
            before_url="before.jpg",
            after_url="after.jpg",
            before_date="September 25, 2022",
            after_date="September 29, 2022",
        )

        assert "September 25, 2022" in slider_html
        assert "September 29, 2022" in slider_html


class TestAccessibility:
    """Test accessibility features."""

    def test_lang_attribute(self, sample_report_data):
        """Test HTML lang attribute."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        assert 'lang="en"' in html

    def test_meta_viewport(self, sample_report_data):
        """Test viewport meta tag for mobile."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        assert 'name="viewport"' in html

    def test_semantic_html(self, sample_report_data):
        """Test use of semantic HTML elements."""
        generator = InteractiveReportGenerator()

        html = generator.generate(sample_report_data)

        # Check for semantic elements
        assert "<header" in html or "<div class=\"fl-sticky-header\"" in html
        assert "<footer" in html
        assert "<button" in html

    def test_alt_text_support(self, sample_report_data):
        """Test images have alt text support."""
        generator = InteractiveReportGenerator()

        html = generator.generate(
            report_data=sample_report_data,
            before_image_url="before.jpg",
            after_image_url="after.jpg",
        )

        # Check for alt attributes
        assert 'alt=' in html


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
