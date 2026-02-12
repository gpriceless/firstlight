"""Interactive web report generator."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

from core.reporting.templates.base import ReportTemplateEngine
from core.reporting.templates.full_report import FullReportData
from core.reporting.maps.folium_map import InteractiveMapGenerator
from core.reporting.maps.base import MapBounds
from core.reporting.imagery import ImageManifest


@dataclass
class WebReportConfig:
    """Configuration for web reports."""
    include_interactive_map: bool = True
    include_before_after_slider: bool = True
    mobile_responsive: bool = True
    embed_css: bool = True  # Inline CSS vs external file
    embed_js: bool = True   # Inline JS vs external file
    enable_print_styles: bool = True
    collapsible_sections: bool = True


class InteractiveReportGenerator:
    """Generate interactive HTML reports with embedded maps."""

    def __init__(self, config: Optional[WebReportConfig] = None):
        """
        Initialize interactive report generator.

        Args:
            config: Web report configuration
        """
        self.config = config or WebReportConfig()
        self.template_engine = ReportTemplateEngine(
            template_dir=Path(__file__).parent / "html"
        )
        self.map_generator = InteractiveMapGenerator()

    def generate(
        self,
        report_data: FullReportData,
        flood_geojson: Optional[dict] = None,
        infrastructure: Optional[List[dict]] = None,
        before_image_url: Optional[str] = None,
        after_image_url: Optional[str] = None,
        bounds: Optional[MapBounds] = None,
        image_manifest: Optional[ImageManifest] = None,
    ) -> str:
        """
        Generate complete interactive HTML report.

        Args:
            report_data: Full report data structure
            flood_geojson: Optional GeoJSON for flood extent overlay
            infrastructure: Optional list of infrastructure features
            before_image_url: Optional URL for before image (deprecated - use image_manifest)
            after_image_url: Optional URL for after image (deprecated - use image_manifest)
            bounds: Optional map bounds (required if including map)
            image_manifest: Optional ImageManifest with locally generated images

        Returns:
            Complete HTML document as string

        Example:
            >>> config = WebReportConfig()
            >>> generator = InteractiveReportGenerator(config)
            >>> html = generator.generate(
            ...     report_data=report_data,
            ...     flood_geojson=geojson_dict,
            ...     bounds=MapBounds(-82.0, 26.3, -81.5, 26.7),
            ...     image_manifest=manifest
            ... )
        """
        # Generate embedded map HTML if requested
        map_html = None
        if self.config.include_interactive_map and bounds and flood_geojson:
            map_html = self.map_generator.generate_flood_map(
                flood_geojson=flood_geojson,
                bounds=bounds,
                infrastructure=infrastructure,
                title=f"{report_data.event_name} - Flood Extent"
            )

        # Generate before/after slider HTML if requested
        # Prefer image_manifest over deprecated URL parameters
        slider_html = None
        if self.config.include_before_after_slider:
            if image_manifest and image_manifest.before_image and image_manifest.after_image:
                # Use local generated images from manifest
                slider_html = self._generate_before_after_slider(
                    before_url=str(image_manifest.before_image),
                    after_url=str(image_manifest.after_image),
                    before_date=report_data.event_date.strftime("%B %d, %Y"),
                    after_date=report_data.report_date.strftime("%B %d, %Y"),
                )
            elif before_image_url and after_image_url:
                # Fallback to deprecated URL parameters
                slider_html = self._generate_before_after_slider(
                    before_url=before_image_url,
                    after_url=after_image_url,
                    before_date=report_data.event_date.strftime("%B %d, %Y"),
                    after_date=report_data.report_date.strftime("%B %d, %Y"),
                )

        # Build template context
        context = {
            'config': self.config,
            'report_data': report_data,
            'map_html': map_html,
            'slider_html': slider_html,
            'image_manifest': image_manifest,
            'css': self._generate_css() if self.config.embed_css else None,
            'js': self._generate_js() if self.config.embed_js else None,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        }

        return self.template_engine.render("interactive_report.html", context)

    def _generate_before_after_slider(
        self,
        before_url: str,
        after_url: str,
        before_date: str,
        after_date: str,
    ) -> str:
        """
        Generate before/after image comparison slider.

        Args:
            before_url: URL to before image
            after_url: URL to after image
            before_date: Date label for before image
            after_date: Date label for after image

        Returns:
            HTML string for slider component
        """
        return f'''
<div class="fl-before-after">
    <div class="fl-before-after__header">
        <h3 class="fl-before-after__title">Before/After Comparison</h3>
        <p class="fl-before-after__subtitle">Drag the slider to compare images</p>
    </div>
    <div class="fl-before-after__container">
        <div class="fl-before-after__wrapper">
            <img class="fl-before-after__before" src="{before_url}" alt="Before - {before_date}">
            <div class="fl-before-after__overlay">
                <img class="fl-before-after__after" src="{after_url}" alt="After - {after_date}">
                <div class="fl-before-after__slider">
                    <div class="fl-before-after__handle">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M5 12h14M12 5l7 7-7 7"/>
                        </svg>
                    </div>
                </div>
            </div>
        </div>
        <div class="fl-before-after__labels">
            <span class="fl-before-after__label fl-before-after__label--before">{before_date}</span>
            <span class="fl-before-after__label fl-before-after__label--after">{after_date}</span>
        </div>
    </div>
</div>
'''

    def _generate_css(self) -> str:
        """
        Generate embedded CSS from design system and interactive components.

        Returns:
            CSS string to be embedded in <style> tag
        """
        return '''
/* Interactive Report Styles */

/* Before/After Slider */
.fl-before-after {
    margin: 2rem 0;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
}

.fl-before-after__header {
    padding: 1.5rem;
    border-bottom: 1px solid #E2E8F0;
}

.fl-before-after__title {
    margin: 0 0 0.5rem 0;
    font-size: 1.25rem;
    font-weight: 600;
    color: #2D3748;
}

.fl-before-after__subtitle {
    margin: 0;
    font-size: 0.875rem;
    color: #718096;
}

.fl-before-after__container {
    padding: 1.5rem;
}

.fl-before-after__wrapper {
    position: relative;
    width: 100%;
    overflow: hidden;
    border-radius: 4px;
    background: #F7FAFC;
}

.fl-before-after__before {
    display: block;
    width: 100%;
    height: auto;
}

.fl-before-after__overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 50%;
    height: 100%;
    overflow: hidden;
}

.fl-before-after__after {
    display: block;
    width: 200%;
    height: 100%;
    object-fit: cover;
}

.fl-before-after__slider {
    position: absolute;
    top: 0;
    right: 0;
    width: 4px;
    height: 100%;
    background: #3182CE;
    cursor: ew-resize;
    z-index: 10;
}

.fl-before-after__handle {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 48px;
    height: 48px;
    background: white;
    border: 2px solid #3182CE;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #3182CE;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    pointer-events: none;
}

.fl-before-after__labels {
    display: flex;
    justify-content: space-between;
    margin-top: 1rem;
    padding: 0 0.5rem;
}

.fl-before-after__label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #4A5568;
}

/* Sticky Header */
.fl-sticky-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: white;
    border-bottom: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.fl-sticky-header__content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.fl-sticky-header__title {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
    color: #2D3748;
}

.fl-sticky-header__actions {
    display: flex;
    gap: 0.75rem;
}

.fl-sticky-header__button {
    padding: 0.5rem 1rem;
    background: white;
    border: 1px solid #CBD5E0;
    border-radius: 4px;
    color: #2D3748;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
}

.fl-sticky-header__button:hover {
    background: #F7FAFC;
    border-color: #A0AEC0;
}

/* Collapsible Sections */
.fl-collapsible {
    margin: 1.5rem 0;
}

.fl-collapsible__header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    background: #F7FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s ease;
}

.fl-collapsible__header:hover {
    background: #EDF2F7;
}

.fl-collapsible__title {
    margin: 0;
    font-size: 1.125rem;
    font-weight: 600;
    color: #2D3748;
}

.fl-collapsible__icon {
    width: 20px;
    height: 20px;
    color: #718096;
    transition: transform 0.2s ease;
}

.fl-collapsible--expanded .fl-collapsible__icon {
    transform: rotate(180deg);
}

.fl-collapsible__content {
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease;
}

.fl-collapsible--expanded .fl-collapsible__content {
    max-height: 5000px;
}

.fl-collapsible__body {
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    border-top: none;
    border-radius: 0 0 8px 8px;
}

/* Map Container */
.fl-map-container {
    margin: 2rem 0;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    overflow: hidden;
    background: white;
}

.fl-map-container iframe {
    width: 100%;
    height: 600px;
    border: none;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .fl-sticky-header__content {
        flex-direction: column;
        gap: 0.75rem;
        align-items: flex-start;
    }

    .fl-sticky-header__actions {
        width: 100%;
        justify-content: stretch;
    }

    .fl-sticky-header__button {
        flex: 1;
    }

    .fl-before-after__handle {
        width: 36px;
        height: 36px;
    }

    .fl-map-container iframe {
        height: 400px;
    }
}

/* Print Styles */
@media print {
    .fl-sticky-header,
    .fl-sticky-header__actions,
    .fl-collapsible__icon,
    .no-print {
        display: none !important;
    }

    .fl-collapsible__content {
        max-height: none !important;
    }

    .fl-collapsible__header {
        cursor: default;
        background: white;
    }

    .fl-before-after__slider {
        display: none;
    }

    .fl-before-after__overlay {
        width: 50%;
    }
}
'''

    def _generate_js(self) -> str:
        """
        Generate embedded JavaScript for interactivity.

        Returns:
            JavaScript string to be embedded in <script> tag
        """
        return '''
// Interactive Report JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initBeforeAfterSlider();
    initCollapsibleSections();
    initPrintButton();
});

// Before/After Slider
function initBeforeAfterSlider() {
    const sliders = document.querySelectorAll('.fl-before-after__slider');

    sliders.forEach(slider => {
        const container = slider.closest('.fl-before-after__wrapper');
        const overlay = container.querySelector('.fl-before-after__overlay');

        let isDragging = false;

        function updateSlider(x) {
            const rect = container.getBoundingClientRect();
            const offsetX = x - rect.left;
            const percentage = Math.max(0, Math.min(100, (offsetX / rect.width) * 100));
            overlay.style.width = percentage + '%';
        }

        slider.addEventListener('mousedown', () => isDragging = true);
        slider.addEventListener('touchstart', () => isDragging = true);

        document.addEventListener('mousemove', (e) => {
            if (isDragging) {
                e.preventDefault();
                updateSlider(e.clientX);
            }
        });

        document.addEventListener('touchmove', (e) => {
            if (isDragging) {
                e.preventDefault();
                updateSlider(e.touches[0].clientX);
            }
        });

        document.addEventListener('mouseup', () => isDragging = false);
        document.addEventListener('touchend', () => isDragging = false);

        // Initial position
        updateSlider(rect.left + rect.width / 2);
    });
}

// Collapsible Sections
function initCollapsibleSections() {
    const headers = document.querySelectorAll('.fl-collapsible__header');

    headers.forEach(header => {
        header.addEventListener('click', function() {
            const collapsible = this.closest('.fl-collapsible');
            collapsible.classList.toggle('fl-collapsible--expanded');
        });
    });

    // Expand first section by default
    const firstCollapsible = document.querySelector('.fl-collapsible');
    if (firstCollapsible) {
        firstCollapsible.classList.add('fl-collapsible--expanded');
    }
}

// Print Button
function initPrintButton() {
    const printButtons = document.querySelectorAll('[data-action="print"]');

    printButtons.forEach(button => {
        button.addEventListener('click', () => {
            window.print();
        });
    });
}
'''

    def save(self, html: str, output_path: Path) -> Path:
        """
        Save report to file.

        Args:
            html: Rendered HTML string
            output_path: Path to save file

        Returns:
            Path to saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding='utf-8')
        return output_path
