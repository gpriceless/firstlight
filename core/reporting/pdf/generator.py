"""
FirstLight Reporting - PDF Report Generator

Converts HTML reports to print-ready PDFs using WeasyPrint.
"""

from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
from enum import Enum


class PageSize(Enum):
    """Standard paper sizes for PDF output."""
    LETTER = "letter"  # 8.5x11" (US standard)
    A4 = "A4"          # 210x297mm (International standard)
    TABLOID = "tabloid"  # 11x17" (Large format)


@dataclass
class PDFConfig:
    """
    Configuration for PDF generation.

    Attributes:
        page_size: Paper size (letter, A4, or tabloid)
        orientation: Page orientation ("portrait" or "landscape")
        dpi: Output resolution (300 for print, 150 for draft)
        include_bleed: Whether to add bleed margins for professional printing
        bleed_mm: Bleed size in millimeters (standard is 3mm)
        embed_fonts: Whether to embed fonts in the PDF
    """
    page_size: PageSize = PageSize.LETTER
    orientation: str = "portrait"  # or "landscape"
    dpi: int = 300
    include_bleed: bool = False
    bleed_mm: float = 3.0
    embed_fonts: bool = True


class PDFReportGenerator:
    """
    Generate print-ready PDF reports from HTML content.

    Uses WeasyPrint for HTML-to-PDF conversion with professional
    print specifications including proper margins, page numbers,
    and font embedding.

    Example:
        >>> config = PDFConfig(page_size=PageSize.LETTER, dpi=300)
        >>> generator = PDFReportGenerator(config)
        >>> pdf_path = generator.generate(
        ...     html_content="<html>...</html>",
        ...     output_path=Path("report.pdf")
        ... )
    """

    def __init__(self, config: Optional[PDFConfig] = None):
        """
        Initialize PDF generator.

        Args:
            config: PDF configuration. Uses defaults if not provided.
        """
        self.config = config or PDFConfig()

    def generate(
        self,
        html_content: str,
        output_path: Path,
        static_maps: Optional[List[Path]] = None,
    ) -> Path:
        """
        Convert HTML report to PDF.

        Args:
            html_content: HTML string to convert
            output_path: Where to save the PDF
            static_maps: Optional list of paths to map images to embed

        Returns:
            Path to the generated PDF

        Raises:
            RuntimeError: If WeasyPrint is not installed
            ValueError: If HTML content is invalid
        """
        try:
            from weasyprint import HTML, CSS
        except ImportError:
            raise RuntimeError(
                "WeasyPrint is required for PDF generation. "
                "Install with: pip install weasyprint"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load print-specific CSS
        print_css = self._get_print_css()

        # Generate PDF from HTML
        html = HTML(string=html_content)
        html.write_pdf(
            output_path,
            stylesheets=[CSS(string=print_css)],
        )

        return output_path

    def _get_print_css(self) -> str:
        """
        Get CSS optimized for print output.

        Returns:
            CSS string with @page rules, print-specific styling,
            and overrides for interactive elements.
        """
        # Calculate margins
        margin = "1in" if not self.config.include_bleed else f"{1 + self.config.bleed_mm/25.4}in"

        # Page size string
        page_size = self.config.page_size.value
        orientation = self.config.orientation

        return f'''
        @page {{
            size: {page_size} {orientation};
            margin: {margin};

            @top-center {{
                content: "FirstLight Analysis Report";
                font-family: sans-serif;
                font-size: 10pt;
                color: #718096;
            }}

            @bottom-center {{
                content: "Page " counter(page) " of " counter(pages);
                font-family: sans-serif;
                font-size: 10pt;
                color: #718096;
            }}
        }}

        /* Base typography for print */
        body {{
            font-size: 11pt;
            line-height: 1.4;
            color: #000;
        }}

        /* Prevent orphans and widows */
        p {{
            orphans: 3;
            widows: 3;
        }}

        /* Keep important elements together */
        .fl-map-container,
        .fl-metric-card,
        .fl-alert-box {{
            break-inside: avoid;
        }}

        /* Force page breaks for major sections */
        .fl-report-section {{
            break-before: page;
        }}

        /* Hide interactive elements in print */
        .fl-before-after__slider,
        .fl-collapsible__toggle,
        .fl-interactive-controls,
        button,
        .fl-web-only {{
            display: none !important;
        }}

        /* Show both images in before/after comparisons */
        .fl-before-after__before,
        .fl-before-after__after {{
            position: static !important;
            width: 48% !important;
            display: inline-block;
            vertical-align: top;
        }}

        /* Ensure high-resolution images */
        img {{
            max-width: 100%;
            height: auto;
            image-rendering: -webkit-optimize-contrast;
            image-rendering: crisp-edges;
        }}

        /* Map-specific styling */
        .fl-map-container {{
            margin: 1rem 0;
            page-break-inside: avoid;
        }}

        .fl-map-container img {{
            width: 100%;
            height: auto;
        }}

        /* Table of contents links */
        a {{
            color: #2B6CB0;
            text-decoration: none;
        }}

        a[href^="#"] {{
            border-bottom: 1px dotted #2B6CB0;
        }}

        /* Headings */
        h1, h2, h3 {{
            break-after: avoid;
            color: #1A365D;
        }}

        h1 {{
            font-size: 24pt;
            margin-top: 0;
        }}

        h2 {{
            font-size: 18pt;
            margin-top: 1.5em;
        }}

        h3 {{
            font-size: 14pt;
            margin-top: 1em;
        }}

        /* Metric cards in print */
        .fl-metric-card {{
            border: 1px solid #CBD5E0;
            padding: 0.75rem;
            margin: 0.5rem;
            border-radius: 4pt;
            break-inside: avoid;
        }}

        /* Alert boxes */
        .fl-alert-box {{
            border-left: 4pt solid #E53E3E;
            padding: 0.75rem;
            margin: 0.75rem 0;
            background: #FFF5F5;
            break-inside: avoid;
        }}

        /* Cover page - no headers/footers */
        .fl-cover {{
            page: cover;
        }}

        @page cover {{
            margin: 0;
            @top-center {{ content: none; }}
            @bottom-center {{ content: none; }}
        }}

        /* Table styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 10pt;
        }}

        th {{
            background: #EDF2F7;
            padding: 0.5rem;
            text-align: left;
            border-bottom: 2pt solid #CBD5E0;
        }}

        td {{
            padding: 0.5rem;
            border-bottom: 1pt solid #E2E8F0;
        }}

        /* Footer meta information */
        .fl-footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1pt solid #E2E8F0;
            font-size: 9pt;
            color: #718096;
        }}
        '''

    def generate_map_page(
        self,
        map_image_path: Path,
        title: str,
        caption: str,
    ) -> str:
        """
        Generate HTML for a full-page map.

        Creates a standalone page with a map image, title, and caption
        suitable for insertion into a PDF report.

        Args:
            map_image_path: Path to the map PNG image
            title: Map title
            caption: Description or caption text

        Returns:
            HTML string for the map page
        """
        return f'''
        <div class="fl-map-page fl-report-section">
            <h2>{title}</h2>
            <div class="fl-map-container">
                <img src="{map_image_path}" alt="{title}">
            </div>
            <p class="fl-map-caption">{caption}</p>
        </div>
        '''

    def combine_reports(
        self,
        pdf_paths: List[Path],
        output_path: Path,
    ) -> Path:
        """
        Combine multiple PDFs into one.

        Useful for merging main report with appendices or
        combining reports from multiple events.

        Args:
            pdf_paths: List of PDF file paths to combine
            output_path: Where to save the combined PDF

        Returns:
            Path to the combined PDF

        Raises:
            RuntimeError: If pypdf is not installed
        """
        try:
            from pypdf import PdfWriter
        except ImportError:
            raise RuntimeError(
                "pypdf is required for PDF merging. "
                "Install with: pip install pypdf"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Combine PDFs
        writer = PdfWriter()
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            writer.append(str(pdf_path))

        # Write combined PDF
        with open(output_path, 'wb') as f:
            writer.write(f)

        return output_path
