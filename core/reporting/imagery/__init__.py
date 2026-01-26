# Visual Product Generation Pipeline (VIS-1.0)
#
# This module handles satellite imagery rendering and visual product generation:
# - VIS-1.1: ImageryRenderer - Base satellite imagery rendering
# - VIS-1.2: BeforeAfterGenerator - Temporal comparison images
# - VIS-1.3: OverlayRenderer - Detection result overlays
# - VIS-1.4: AnnotationLayer - Contextual annotations and labels
# - VIS-1.5: ReportVisualPipeline - Integration with reporting system
#
# See ROADMAP.md for implementation status.

from .band_combinations import (
    BandComposite,
    SENTINEL2_COMPOSITES,
    LANDSAT_COMPOSITES,
    SAR_VISUALIZATIONS,
    get_bands_for_composite,
    get_available_composites,
)
from .cloud_mask import (
    CloudMask,
    CloudMaskConfig,
    CloudMaskResult,
    SCLClass,
)
from .histogram import (
    HistogramStretch,
    apply_stretch,
    calculate_stddev_bounds,
    calculate_stretch_bounds,
    stretch_band,
)
from .renderer import (
    ImageryRenderer,
    RendererConfig,
    RenderedImage,
)
from .export import (
    ExportConfig,
    ImageExporter,
    add_alpha_channel,
    resize_for_web,
)
from .utils import (
    detect_partial_coverage,
    get_valid_data_bounds,
    handle_nodata,
    normalize_to_uint8,
    stack_bands_to_rgb,
)

__all__ = [
    # Band combinations (VIS-1.1.3, VIS-1.1.4)
    "BandComposite",
    "SENTINEL2_COMPOSITES",
    "LANDSAT_COMPOSITES",
    "SAR_VISUALIZATIONS",
    "get_bands_for_composite",
    "get_available_composites",
    # Cloud masking (VIS-1.1.6)
    "CloudMask",
    "CloudMaskConfig",
    "CloudMaskResult",
    "SCLClass",
    # Histogram stretching (VIS-1.1.5)
    "HistogramStretch",
    "apply_stretch",
    "calculate_stddev_bounds",
    "calculate_stretch_bounds",
    "stretch_band",
    # Main renderer (VIS-1.1.2)
    "ImageryRenderer",
    "RendererConfig",
    "RenderedImage",
    # Export (VIS-1.1 Task 4.1)
    "ExportConfig",
    "ImageExporter",
    "add_alpha_channel",
    "resize_for_web",
    # Utilities
    "normalize_to_uint8",
    "stack_bands_to_rgb",
    "handle_nodata",
    "detect_partial_coverage",
    "get_valid_data_bounds",
]
