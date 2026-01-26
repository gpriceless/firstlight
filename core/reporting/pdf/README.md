# FirstLight PDF Report Generator

Print-ready PDF generation from HTML reports using WeasyPrint.

## Features

- 300 DPI high-resolution output
- Multiple page sizes (US Letter, A4, Tabloid)
- Professional print margins with optional bleed
- Automatic page numbers and headers
- Font embedding
- Proper page breaks and orphan/widow control
- Before/after comparison support
- Interactive elements automatically hidden in print

## Installation

The PDF generator requires WeasyPrint:

```bash
pip install weasyprint
```

For PDF merging functionality, also install pypdf:

```bash
pip install pypdf
```

## Usage

### Basic PDF Generation

```python
from pathlib import Path
from core.reporting.pdf import PDFReportGenerator, PDFConfig, PageSize

# Create generator with default config (US Letter, 300 DPI)
generator = PDFReportGenerator()

# Generate PDF from HTML
html_content = "<html><body><h1>Flood Report</h1></body></html>"
pdf_path = generator.generate(
    html_content=html_content,
    output_path=Path("report.pdf")
)
```

### Custom Configuration

```python
from core.reporting.pdf import PDFConfig, PageSize

# A4 paper, landscape orientation, with bleed
config = PDFConfig(
    page_size=PageSize.A4,
    orientation="landscape",
    dpi=300,
    include_bleed=True,
    bleed_mm=3.0,
    embed_fonts=True
)

generator = PDFReportGenerator(config)
```

### Generating Full Report PDFs

```python
from core.reporting.templates import ReportTemplateEngine
from core.reporting.pdf import PDFReportGenerator

# Render HTML report
engine = ReportTemplateEngine()
html = engine.render("full_report.html", context={
    "event": event_data,
    "analysis": analysis_results,
    "maps": map_paths
})

# Convert to PDF
generator = PDFReportGenerator()
pdf = generator.generate(html, output_path=Path("full_report.pdf"))
```

### Map Page Generation

```python
# Generate standalone map page
map_html = generator.generate_map_page(
    map_image_path=Path("flood_extent.png"),
    title="Hurricane Ian Flood Extent",
    caption="Fort Myers, Florida - September 28, 2022"
)

# This HTML can be embedded in larger reports
```

### Combining PDFs

```python
# Merge multiple PDFs
combined = generator.combine_reports(
    pdf_paths=[
        Path("executive_summary.pdf"),
        Path("full_analysis.pdf"),
        Path("appendix.pdf")
    ],
    output_path=Path("complete_report.pdf")
)
```

## Configuration Options

### Page Sizes

- `PageSize.LETTER` - 8.5" x 11" (US standard)
- `PageSize.A4` - 210mm x 297mm (International)
- `PageSize.TABLOID` - 11" x 17" (Large format)

### Orientation

- `"portrait"` - Vertical orientation (default)
- `"landscape"` - Horizontal orientation

### Resolution

- `dpi=300` - Professional print quality (default)
- `dpi=150` - Draft quality (smaller file size)

### Print Options

- `include_bleed=True` - Add 3mm bleed for professional printing
- `embed_fonts=True` - Embed fonts for consistent rendering (default)

## CSS Features

The generator applies print-optimized CSS automatically:

### Page Setup
- Automatic page breaks for major sections
- Headers with report title
- Footers with page numbers

### Typography
- Orphan/widow control (minimum 3 lines)
- Proper heading hierarchy
- Print-friendly font sizes (11pt body)

### Interactive Elements
- Sliders, buttons, and controls hidden
- Before/after images shown side-by-side
- Map controls removed

### Images
- High-resolution rendering
- Crisp edge rendering for maps
- Proper aspect ratio preservation

## Technical Details

### Print CSS Location

The print-specific CSS is embedded in the generator but also available as a standalone file at:

```
core/reporting/pdf/print_styles.css
```

This can be used for previewing print output in browsers.

### File Structure

```
core/reporting/pdf/
├── __init__.py           # Module exports
├── generator.py          # PDFReportGenerator class
├── print_styles.css      # Standalone print CSS
└── README.md            # This file
```

### Dependencies

**Required:**
- None (imports are conditional)

**Optional:**
- `weasyprint` - For PDF generation
- `pypdf` - For PDF merging

The code will raise helpful errors if optional dependencies are missing.

## Testing

Run tests with:

```bash
./run_tests.py --file test_pdf_generator
```

Note: Tests requiring weasyprint/pypdf will be skipped if not installed.

## Integration with Reporting Pipeline

### From Template Engine

```python
from core.reporting.templates import ReportTemplateEngine
from core.reporting.pdf import PDFReportGenerator

# 1. Render HTML
engine = ReportTemplateEngine()
html = engine.render("full_report.html", context)

# 2. Convert to PDF
generator = PDFReportGenerator()
pdf = generator.generate(html, output_path)
```

### From CLI

```bash
flight export <analysis-id> --format pdf --output report.pdf
```

### From API

```python
POST /events/{event_id}/export
{
  "format": "pdf",
  "page_size": "letter",
  "dpi": 300
}
```

## Cartographic Specifications

For map-specific requirements, see:
- `docs/design/CARTOGRAPHIC_SPEC.md` - Map standards
- `docs/design/DESIGN_SYSTEM.md` - Visual design system

## Examples

Example reports are generated as part of the test suite. Run:

```bash
./run_tests.py --file test_pdf_generator --verbose
```

## Troubleshooting

### WeasyPrint Installation Issues

WeasyPrint requires system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev python3-pip python3-cffi libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
pip install weasyprint
```

**macOS:**
```bash
brew install cairo pango gdk-pixbuf libffi
pip install weasyprint
```

### Font Issues

If fonts aren't rendering correctly:
1. Ensure `embed_fonts=True` in config
2. Specify fonts explicitly in HTML/CSS
3. Install required system fonts

### Large File Sizes

To reduce PDF file size:
- Use `dpi=150` for drafts
- Compress map images before embedding
- Use JPEG for photos, PNG for diagrams

## Future Enhancements

Deferred features (see ROADMAP.md):
- [ ] Black & white pattern overlays (R2.6.7)
- [ ] Custom header/footer templates
- [ ] Watermark support
- [ ] Digital signatures
- [ ] PDF/A compliance for archival

---

**Version:** 1.0.0
**Last Updated:** 2026-01-26
**Epic:** R2.6 Print-Ready PDF Outputs
