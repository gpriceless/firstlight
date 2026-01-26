"""
Tests for PDF report generator.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from core.reporting.pdf import PDFReportGenerator, PDFConfig, PageSize

# Check if optional dependencies are available
try:
    import weasyprint
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


class TestPDFConfig:
    """Test PDFConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PDFConfig()
        assert config.page_size == PageSize.LETTER
        assert config.orientation == "portrait"
        assert config.dpi == 300
        assert config.include_bleed is False
        assert config.bleed_mm == 3.0
        assert config.embed_fonts is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PDFConfig(
            page_size=PageSize.A4,
            orientation="landscape",
            dpi=150,
            include_bleed=True,
        )
        assert config.page_size == PageSize.A4
        assert config.orientation == "landscape"
        assert config.dpi == 150
        assert config.include_bleed is True


class TestPDFReportGenerator:
    """Test PDFReportGenerator class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        generator = PDFReportGenerator()
        assert generator.config is not None
        assert isinstance(generator.config, PDFConfig)
        assert generator.config.page_size == PageSize.LETTER

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = PDFConfig(page_size=PageSize.A4, dpi=150)
        generator = PDFReportGenerator(config)
        assert generator.config.page_size == PageSize.A4
        assert generator.config.dpi == 150

    def test_get_print_css_default(self):
        """Test print CSS generation with default config."""
        generator = PDFReportGenerator()
        css = generator._get_print_css()

        # Check for key CSS features
        assert "@page" in css
        assert "letter portrait" in css
        assert "margin: 1in" in css
        assert "counter(page)" in css
        assert "counter(pages)" in css
        assert ".fl-map-container" in css
        assert "display: none" in css  # Hide interactive elements

    def test_get_print_css_with_bleed(self):
        """Test print CSS generation with bleed margins."""
        config = PDFConfig(include_bleed=True, bleed_mm=3.0)
        generator = PDFReportGenerator(config)
        css = generator._get_print_css()

        # Should have larger margin to accommodate bleed
        # 1 inch + 3mm = 1 + 3/25.4 ≈ 1.118 inches
        assert "@page" in css
        assert "margin:" in css

    def test_get_print_css_landscape(self):
        """Test print CSS generation with landscape orientation."""
        config = PDFConfig(orientation="landscape")
        generator = PDFReportGenerator(config)
        css = generator._get_print_css()

        assert "letter landscape" in css

    def test_get_print_css_a4(self):
        """Test print CSS generation with A4 page size."""
        config = PDFConfig(page_size=PageSize.A4)
        generator = PDFReportGenerator(config)
        css = generator._get_print_css()

        assert "A4 portrait" in css

    @pytest.mark.skipif(not WEASYPRINT_AVAILABLE, reason="weasyprint not installed")
    @patch('weasyprint.HTML')
    @patch('weasyprint.CSS')
    def test_generate_success(self, mock_css, mock_html):
        """Test successful PDF generation."""
        # Setup mocks
        mock_html_instance = Mock()
        mock_html.return_value = mock_html_instance

        generator = PDFReportGenerator()
        html_content = "<html><body><h1>Test Report</h1></body></html>"
        output_path = Path("/tmp/test_report.pdf")

        # Generate PDF
        result = generator.generate(html_content, output_path)

        # Verify HTML was created
        mock_html.assert_called_once_with(string=html_content)

        # Verify write_pdf was called
        mock_html_instance.write_pdf.assert_called_once()
        call_args = mock_html_instance.write_pdf.call_args
        assert call_args[0][0] == output_path

        # Verify result
        assert result == output_path

    def test_generate_missing_weasyprint(self):
        """Test error when WeasyPrint is not installed."""
        with patch.dict('sys.modules', {'weasyprint': None}):
            generator = PDFReportGenerator()
            html_content = "<html><body>Test</body></html>"
            output_path = Path("/tmp/test.pdf")

            with pytest.raises(RuntimeError, match="WeasyPrint is required"):
                generator.generate(html_content, output_path)

    def test_generate_map_page(self):
        """Test map page HTML generation."""
        generator = PDFReportGenerator()

        html = generator.generate_map_page(
            map_image_path=Path("/tmp/map.png"),
            title="Flood Extent Map",
            caption="Hurricane Ian flood extent, September 28, 2022"
        )

        # Verify HTML structure
        assert "fl-map-page" in html
        assert "fl-report-section" in html
        assert "Flood Extent Map" in html
        assert "/tmp/map.png" in html
        assert "Hurricane Ian" in html
        assert "<img" in html
        assert "fl-map-container" in html

    @pytest.mark.skipif(not PYPDF_AVAILABLE, reason="pypdf not installed")
    @patch('pypdf.PdfWriter')
    def test_combine_reports_success(self, mock_pdf_writer):
        """Test successful PDF combination."""
        # Setup mocks
        mock_writer_instance = Mock()
        mock_pdf_writer.return_value = mock_writer_instance

        generator = PDFReportGenerator()

        # Create temporary files
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f2, \
             tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as out:

            pdf1 = Path(f1.name)
            pdf2 = Path(f2.name)
            output = Path(out.name)

            try:
                # Write some dummy content
                f1.write(b"PDF1")
                f2.write(b"PDF2")
                f1.flush()
                f2.flush()

                # Combine PDFs
                result = generator.combine_reports(
                    pdf_paths=[pdf1, pdf2],
                    output_path=output
                )

                # Verify append was called for each PDF
                assert mock_writer_instance.append.call_count == 2

                # Verify write was called
                assert mock_writer_instance.write.called

                # Verify result
                assert result == output

            finally:
                # Cleanup
                pdf1.unlink(missing_ok=True)
                pdf2.unlink(missing_ok=True)
                output.unlink(missing_ok=True)

    def test_combine_reports_missing_pypdf(self):
        """Test error when pypdf is not installed."""
        with patch.dict('sys.modules', {'pypdf': None}):
            generator = PDFReportGenerator()

            with pytest.raises(RuntimeError, match="pypdf is required"):
                generator.combine_reports(
                    pdf_paths=[Path("/tmp/1.pdf")],
                    output_path=Path("/tmp/combined.pdf")
                )

    @pytest.mark.skipif(not PYPDF_AVAILABLE, reason="pypdf not installed")
    @patch('pypdf.PdfWriter')
    def test_combine_reports_missing_input(self, mock_pdf_writer):
        """Test error when input PDF doesn't exist."""
        generator = PDFReportGenerator()

        with pytest.raises(FileNotFoundError):
            generator.combine_reports(
                pdf_paths=[Path("/nonexistent/file.pdf")],
                output_path=Path("/tmp/output.pdf")
            )


class TestPageSize:
    """Test PageSize enum."""

    def test_page_size_values(self):
        """Test PageSize enum values."""
        assert PageSize.LETTER.value == "letter"
        assert PageSize.A4.value == "A4"
        assert PageSize.TABLOID.value == "tabloid"

    def test_page_size_members(self):
        """Test PageSize enum members."""
        assert len(PageSize) == 3
        assert PageSize.LETTER in PageSize
        assert PageSize.A4 in PageSize
        assert PageSize.TABLOID in PageSize
