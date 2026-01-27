"""
Detection overlay rendering for FirstLight visualizations (VIS-1.3 Complete).

This module blends detection masks (flood/fire) onto satellite imagery backgrounds
with configurable transparency, severity colormaps, and optional map furniture.

Core functionality:
- Blend detection results onto RGB satellite imagery
- Apply severity-based coloring (flood: blue, fire: burn severity colors)
- Confidence-based alpha transparency (low confidence = more transparent)
- Optional polygon outline rendering
- Auto-generated legend integration
- Scale bar and north arrow components
- Pattern fills for black-and-white printing accessibility

Example:
    from core.reporting.imagery import ImageryRenderer, DetectionOverlay
    from core.reporting.imagery.color_scales import get_flood_palette
    import numpy as np

    # Render background satellite imagery
    renderer = ImageryRenderer()
    result = renderer.render("scene.tif", sensor="sentinel2")
    background = result.rgb_array  # (H, W, 3)

    # Detection results from flood algorithm
    detection_mask = np.array(...)  # Binary mask (H, W)
    confidence = np.array(...)      # Confidence scores (H, W)

    # Create overlay
    overlay = DetectionOverlay(overlay_type="flood")
    result = overlay.render(background, detection_mask, confidence=confidence)

    # Result contains composite image + legend
    composite = result.composite_image  # (H, W, 4) RGBA
    legend = result.legend_image        # (H, W, 4) RGBA
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .color_scales import apply_colormap, get_palette_by_name
from .legend import LegendRenderer, LegendConfig


@dataclass
class OverlayConfig:
    """
    Configuration for detection overlay rendering.

    Attributes:
        overlay_type: Type of detection ("flood", "fire")
        confidence_threshold: Ignore detections below this confidence (0-1)
        alpha_base: Base transparency level (0-1, where 1 = opaque)
        use_confidence_alpha: Modulate alpha by confidence (low conf = more transparent)
        show_outline: Draw polygon outlines around detections
        outline_color: RGB color for outlines
        outline_width: Line width for outlines in pixels
        show_scale_bar: Draw scale bar on the image
        show_north_arrow: Draw north arrow on the image
        use_bw_patterns: Use hatching patterns for B&W accessibility
        pixel_size_meters: Ground sampling distance in meters (for scale bar calculation)
    """

    overlay_type: str = "flood"
    confidence_threshold: float = 0.3
    alpha_base: float = 0.6
    use_confidence_alpha: bool = True
    show_outline: bool = True
    outline_color: Tuple[int, int, int] = (255, 255, 255)
    outline_width: int = 2
    show_scale_bar: bool = True
    show_north_arrow: bool = True
    use_bw_patterns: bool = False
    pixel_size_meters: float = 10.0  # Default: Sentinel-2 resolution

    def __post_init__(self):
        """Validate configuration."""
        valid_types = ["flood", "fire"]
        if self.overlay_type not in valid_types:
            raise ValueError(
                f"Invalid overlay_type: {self.overlay_type}. "
                f"Choose from: {valid_types}"
            )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                f"confidence_threshold must be 0-1, got {self.confidence_threshold}"
            )

        if not 0 <= self.alpha_base <= 1:
            raise ValueError(f"alpha_base must be 0-1, got {self.alpha_base}")

        if self.outline_width < 0:
            raise ValueError(
                f"outline_width must be non-negative, got {self.outline_width}"
            )


@dataclass
class OverlayResult:
    """
    Result from detection overlay rendering.

    Attributes:
        composite_image: RGBA image with detection overlay blended on background
        legend_image: RGBA legend image (can be composited separately)
        metadata: Dictionary containing overlay statistics
    """

    composite_image: np.ndarray
    legend_image: np.ndarray
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate result arrays."""
        if self.composite_image.ndim != 3:
            raise ValueError(
                f"composite_image must be 3D array, got shape {self.composite_image.shape}"
            )

        if self.legend_image.ndim != 3:
            raise ValueError(
                f"legend_image must be 3D array, got shape {self.legend_image.shape}"
            )


class DetectionOverlay:
    """
    Renders detection overlays on satellite imagery backgrounds.

    Blends detection masks with severity/confidence-based coloring onto
    RGB satellite imagery. Handles alpha transparency, outline rendering,
    and automatic legend generation.
    """

    def __init__(self, config: Optional[OverlayConfig] = None):
        """
        Initialize detection overlay renderer.

        Args:
            config: Overlay configuration (uses defaults if None)
        """
        self.config = config or OverlayConfig()
        self._palette = get_palette_by_name(self.config.overlay_type)

    def render(
        self,
        background: np.ndarray,
        detection_mask: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        severity: Optional[np.ndarray] = None,
    ) -> OverlayResult:
        """
        Render detection overlay on background imagery.

        Args:
            background: RGB background image (H, W, 3) with dtype uint8 or float
            detection_mask: Binary detection mask (H, W) - True where detections occur
            confidence: Optional confidence scores (H, W) in range 0-1
            severity: Optional severity values (H, W) for color mapping
                     If None, uses confidence as severity proxy

        Returns:
            OverlayResult with composite image, legend, and metadata

        Raises:
            ValueError: If array shapes don't match or dtypes are invalid
        """
        # Validate inputs
        background = self._validate_background(background)
        detection_mask = self._validate_mask(detection_mask, background.shape[:2])

        # Handle optional arrays
        if confidence is not None:
            confidence = self._validate_confidence(confidence, background.shape[:2])
        else:
            confidence = np.ones_like(detection_mask, dtype=np.float32)

        if severity is not None:
            severity = self._validate_severity(severity, background.shape[:2])
        else:
            # Use confidence as proxy for severity if not provided
            severity = confidence.copy()

        # Apply confidence threshold
        thresholded_mask = self._apply_threshold(detection_mask, confidence)

        # Calculate detection area statistics
        total_pixels = detection_mask.size
        detected_pixels = np.sum(thresholded_mask)
        detection_area_pct = (detected_pixels / total_pixels) * 100

        if detected_pixels == 0:
            # No detections after thresholding - return background unchanged
            legend = self._generate_legend()
            return OverlayResult(
                composite_image=self._ensure_rgba(background),
                legend_image=legend,
                metadata={
                    "overlay_type": self.config.overlay_type,
                    "detection_area_pct": 0.0,
                    "confidence_range": (0.0, 0.0),
                    "severity_range": (0.0, 0.0),
                    "confidence_threshold": self.config.confidence_threshold,
                },
            )

        # Create colored overlay from severity values
        colored_overlay = self._create_colored_overlay(
            thresholded_mask, severity, confidence
        )

        # Apply B&W patterns if enabled
        if self.config.use_bw_patterns:
            colored_overlay = self._apply_bw_patterns(
                colored_overlay, severity, thresholded_mask
            )

        # Add outlines if enabled
        if self.config.show_outline:
            colored_overlay = self._add_outline(colored_overlay, thresholded_mask)

        # Blend overlay onto background
        composite = self._blend_overlay(background, colored_overlay)

        # Add scale bar if enabled
        if self.config.show_scale_bar:
            composite = self._add_scale_bar(composite, position="bottom-left")

        # Add north arrow if enabled
        if self.config.show_north_arrow:
            composite = self._add_north_arrow(composite, position="top-right")

        # Generate legend
        legend = self._generate_legend()

        # Calculate statistics
        detected_confidence = confidence[thresholded_mask]
        detected_severity = severity[thresholded_mask]

        metadata = {
            "overlay_type": self.config.overlay_type,
            "detection_area_pct": float(detection_area_pct),
            "confidence_range": (
                float(np.min(detected_confidence)),
                float(np.max(detected_confidence)),
            ),
            "severity_range": (
                float(np.min(detected_severity)),
                float(np.max(detected_severity)),
            ),
            "confidence_threshold": self.config.confidence_threshold,
        }

        return OverlayResult(
            composite_image=composite, legend_image=legend, metadata=metadata
        )

    def render_with_legend(
        self,
        background: np.ndarray,
        detection_mask: np.ndarray,
        confidence: Optional[np.ndarray] = None,
        severity: Optional[np.ndarray] = None,
        legend_position: str = "top-right",
    ) -> np.ndarray:
        """
        Convenience method to render overlay with legend composited onto image.

        Args:
            background: RGB background image (H, W, 3)
            detection_mask: Binary detection mask (H, W)
            confidence: Optional confidence scores (H, W)
            severity: Optional severity values (H, W)
            legend_position: Corner position for legend
                           ("top-left", "top-right", "bottom-left", "bottom-right")

        Returns:
            RGBA image with overlay and legend composited
        """
        # Render overlay
        result = self.render(background, detection_mask, confidence, severity)

        # Composite legend onto image
        composite_with_legend = self._composite_legend(
            result.composite_image, result.legend_image, legend_position
        )

        return composite_with_legend

    # === Private Methods ===

    def _add_scale_bar(
        self, image: np.ndarray, position: str = "bottom-left"
    ) -> np.ndarray:
        """
        Add scale bar to the image.

        Args:
            image: RGBA image array
            position: Corner position ("bottom-left", "bottom-right", "top-left", "top-right")

        Returns:
            Image with scale bar drawn
        """
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        h, w = image.shape[:2]
        margin = 20

        # Calculate appropriate scale bar length
        # Aim for 100-200 pixels, round to nice numbers in meters
        target_px = 150
        target_m = target_px * self.config.pixel_size_meters

        # Round to nice number (100, 200, 500, 1000, 2000, 5000, etc.)
        nice_numbers = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        scale_m = min(nice_numbers, key=lambda x: abs(x - target_m))
        scale_px = int(scale_m / self.config.pixel_size_meters)

        # Format label
        if scale_m >= 1000:
            label = f"{scale_m / 1000:.0f} km"
        else:
            label = f"{scale_m:.0f} m"

        # Position calculations
        bar_height = 5
        text_margin = 5

        if position == "bottom-left":
            x1 = margin
            y1 = h - margin - bar_height
        elif position == "bottom-right":
            x1 = w - margin - scale_px
            y1 = h - margin - bar_height
        elif position == "top-left":
            x1 = margin
            y1 = margin + 20  # Space for text
        else:  # top-right
            x1 = w - margin - scale_px
            y1 = margin + 20

        x2 = x1 + scale_px
        y2 = y1 + bar_height

        # Draw white background rectangle
        draw.rectangle([x1 - 2, y1 - 15, x2 + 2, y2 + 2], fill=(255, 255, 255, 200))

        # Draw black scale bar
        draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0, 255), outline=(255, 255, 255))

        # Draw label
        draw.text((x1, y1 - 15), label, fill=(0, 0, 0, 255))

        return np.array(img_pil)

    def _add_north_arrow(
        self, image: np.ndarray, position: str = "top-right"
    ) -> np.ndarray:
        """
        Add north arrow to the image.

        Args:
            image: RGBA image array
            position: Corner position ("top-left", "top-right", "bottom-left", "bottom-right")

        Returns:
            Image with north arrow drawn
        """
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)

        h, w = image.shape[:2]
        margin = 20
        arrow_size = 40

        # Position calculations
        if position == "top-left":
            cx = margin + arrow_size // 2
            cy = margin + arrow_size // 2
        elif position == "top-right":
            cx = w - margin - arrow_size // 2
            cy = margin + arrow_size // 2
        elif position == "bottom-left":
            cx = margin + arrow_size // 2
            cy = h - margin - arrow_size // 2
        else:  # bottom-right
            cx = w - margin - arrow_size // 2
            cy = h - margin - arrow_size // 2

        # Draw white circle background
        draw.ellipse(
            [cx - arrow_size // 2, cy - arrow_size // 2, cx + arrow_size // 2, cy + arrow_size // 2],
            fill=(255, 255, 255, 200),
            outline=(0, 0, 0)
        )

        # Draw north arrow (triangle pointing up)
        arrow_height = arrow_size * 0.6
        arrow_width = arrow_size * 0.3

        points = [
            (cx, cy - arrow_height / 2),  # Top point
            (cx - arrow_width / 2, cy + arrow_height / 2),  # Bottom left
            (cx + arrow_width / 2, cy + arrow_height / 2),  # Bottom right
        ]

        draw.polygon(points, fill=(0, 0, 0, 255), outline=(0, 0, 0))

        # Draw "N" label
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = None

        text_y = cy + arrow_height / 2 + 5
        draw.text((cx, text_y), "N", fill=(0, 0, 0, 255), anchor="mt", font=font)

        return np.array(img_pil)

    def _apply_bw_patterns(
        self, overlay: np.ndarray, severity: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """
        Apply hatching patterns for B&W printing accessibility.

        Different severity levels get different hatching patterns:
        - Low severity: Light dots
        - Medium severity: Diagonal lines
        - High severity: Crosshatch

        Args:
            overlay: RGBA overlay array
            severity: Severity values (H, W)
            mask: Detection mask (H, W)

        Returns:
            Overlay with patterns applied
        """
        h, w = overlay.shape[:2]
        pattern_overlay = overlay.copy()

        # Define severity ranges
        if self.config.overlay_type == "flood":
            # 5 levels: 0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0
            ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        else:  # fire
            # 6 levels from unburned to high severity
            ranges = [(-0.5, -0.1), (-0.1, 0.1), (0.1, 0.27), (0.27, 0.44), (0.44, 0.66), (0.66, 1.3)]

        # Create pattern for each severity level
        for level, (vmin, vmax) in enumerate(ranges):
            in_range = mask & (severity >= vmin) & (severity < vmax)
            if level == len(ranges) - 1:
                in_range = mask & (severity >= vmin) & (severity <= vmax)

            if not np.any(in_range):
                continue

            # Apply pattern based on level
            if level == 0:
                # Light dots (every 6th pixel)
                pattern = np.zeros((h, w), dtype=bool)
                pattern[::6, ::6] = True
            elif level == 1:
                # Sparse dots (every 4th pixel)
                pattern = np.zeros((h, w), dtype=bool)
                pattern[::4, ::4] = True
            elif level == 2:
                # Diagonal lines (/)
                pattern = np.zeros((h, w), dtype=bool)
                for i in range(h):
                    pattern[i, i % w::8] = True
            elif level == 3:
                # Dense diagonal lines (/)
                pattern = np.zeros((h, w), dtype=bool)
                for i in range(h):
                    pattern[i, i % w::4] = True
            elif level == 4:
                # Crosshatch (+ and /)
                pattern = np.zeros((h, w), dtype=bool)
                pattern[::3, :] = True
                pattern[:, ::3] = True
            else:
                # Dense crosshatch
                pattern = np.zeros((h, w), dtype=bool)
                pattern[::2, :] = True
                pattern[:, ::2] = True

            # Apply pattern: darken pixels where pattern is True
            pattern_mask = in_range & pattern
            pattern_overlay[pattern_mask, :3] = (pattern_overlay[pattern_mask, :3] * 0.5).astype(np.uint8)

        return pattern_overlay

    def _validate_background(self, background: np.ndarray) -> np.ndarray:
        """Validate and convert background to uint8 RGB."""
        if background.ndim != 3:
            raise ValueError(
                f"background must be 3D array (H, W, C), got shape {background.shape}"
            )

        if background.shape[2] not in (3, 4):
            raise ValueError(
                f"background must have 3 or 4 channels, got {background.shape[2]}"
            )

        # Convert to uint8 if float
        if background.dtype in (np.float32, np.float64):
            if background.max() <= 1.0:
                background = (background * 255).astype(np.uint8)
            else:
                background = background.astype(np.uint8)

        # Convert RGBA to RGB if needed
        if background.shape[2] == 4:
            background = background[:, :, :3]

        return background

    def _validate_mask(
        self, mask: np.ndarray, expected_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Validate detection mask."""
        if mask.ndim != 2:
            raise ValueError(f"detection_mask must be 2D array, got shape {mask.shape}")

        if mask.shape != expected_shape:
            raise ValueError(
                f"detection_mask shape {mask.shape} doesn't match "
                f"background shape {expected_shape}"
            )

        # Convert to boolean
        return mask.astype(bool)

    def _validate_confidence(
        self, confidence: np.ndarray, expected_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Validate confidence array."""
        if confidence.ndim != 2:
            raise ValueError(f"confidence must be 2D array, got shape {confidence.shape}")

        if confidence.shape != expected_shape:
            raise ValueError(
                f"confidence shape {confidence.shape} doesn't match "
                f"expected shape {expected_shape}"
            )

        # Ensure float and clip to [0, 1]
        confidence = confidence.astype(np.float32)
        return np.clip(confidence, 0.0, 1.0)

    def _validate_severity(
        self, severity: np.ndarray, expected_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Validate severity array."""
        if severity.ndim != 2:
            raise ValueError(f"severity must be 2D array, got shape {severity.shape}")

        if severity.shape != expected_shape:
            raise ValueError(
                f"severity shape {severity.shape} doesn't match "
                f"expected shape {expected_shape}"
            )

        return severity.astype(np.float32)

    def _apply_threshold(
        self, mask: np.ndarray, confidence: np.ndarray
    ) -> np.ndarray:
        """Apply confidence threshold to detection mask."""
        return mask & (confidence >= self.config.confidence_threshold)

    def _create_colored_overlay(
        self,
        mask: np.ndarray,
        severity: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """Create RGBA colored overlay from severity values."""
        # Apply colormap to severity values
        # Set non-detection pixels to NaN so they become transparent
        severity_masked = np.where(mask, severity, np.nan)
        colored = apply_colormap(severity_masked, self._palette, nodata_value=np.nan)

        # Modulate alpha by confidence if enabled
        if self.config.use_confidence_alpha:
            # Confidence modulates the alpha channel
            # Low confidence -> more transparent
            alpha_modulation = confidence * self.config.alpha_base
            colored[:, :, 3] = np.where(
                mask,
                (colored[:, :, 3] / 255.0 * alpha_modulation * 255).astype(np.uint8),
                0,
            )
        else:
            # Use constant alpha_base
            colored[:, :, 3] = np.where(
                mask, int(self.config.alpha_base * 255), 0
            )

        return colored

    def _add_outline(self, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Add polygon outlines to overlay using edge detection."""
        # Find edges using erosion
        # An edge pixel is True in original mask but False after erosion
        struct = ndimage.generate_binary_structure(2, 1)  # 4-connectivity
        eroded = ndimage.binary_erosion(mask, structure=struct)
        edges = mask & ~eroded

        # Dilate edges by outline_width
        if self.config.outline_width > 1:
            dilate_struct = ndimage.generate_binary_structure(2, 1)
            for _ in range(self.config.outline_width - 1):
                edges = ndimage.binary_dilation(edges, structure=dilate_struct)

        # Apply outline color to edge pixels
        overlay_with_outline = overlay.copy()
        overlay_with_outline[edges, :3] = self.config.outline_color
        overlay_with_outline[edges, 3] = 255  # Full opacity for outline

        return overlay_with_outline

    def _blend_overlay(
        self, background: np.ndarray, overlay: np.ndarray
    ) -> np.ndarray:
        """Blend RGBA overlay onto RGB background using alpha compositing."""
        # Ensure background is RGBA
        bg_rgba = self._ensure_rgba(background)

        # Extract alpha channel from overlay (0-1 range)
        alpha_overlay = overlay[:, :, 3:4] / 255.0
        alpha_bg = bg_rgba[:, :, 3:4] / 255.0

        # Alpha blending formula: C = C_fg * alpha_fg + C_bg * alpha_bg * (1 - alpha_fg)
        # Result alpha: alpha_out = alpha_fg + alpha_bg * (1 - alpha_fg)

        # Composite RGB channels
        rgb_overlay = overlay[:, :, :3]
        rgb_bg = bg_rgba[:, :, :3]

        rgb_out = (
            rgb_overlay * alpha_overlay + rgb_bg * alpha_bg * (1 - alpha_overlay)
        ).astype(np.uint8)

        # Composite alpha channel
        alpha_out = (alpha_overlay + alpha_bg * (1 - alpha_overlay)) * 255

        # Combine into RGBA
        result = np.dstack([rgb_out, alpha_out.astype(np.uint8)])

        return result

    def _ensure_rgba(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image to RGBA with full opacity."""
        if image.shape[2] == 4:
            return image

        # Add alpha channel (fully opaque)
        height, width = image.shape[:2]
        alpha = np.full((height, width, 1), 255, dtype=np.uint8)
        return np.dstack([image, alpha])

    def _generate_legend(self) -> np.ndarray:
        """Generate legend for the overlay."""
        legend_config = LegendConfig(
            title=self._palette.name, position="top-right", width=200, font_size=12
        )
        renderer = LegendRenderer(legend_config)
        legend = renderer.generate_discrete(self._palette)
        return legend

    def _composite_legend(
        self, image: np.ndarray, legend: np.ndarray, position: str
    ) -> np.ndarray:
        """Composite legend onto image at specified position."""
        img_h, img_w = image.shape[:2]
        leg_h, leg_w = legend.shape[:2]

        # Calculate legend position
        margin = 10
        if position == "top-left":
            x, y = margin, margin
        elif position == "top-right":
            x, y = img_w - leg_w - margin, margin
        elif position == "bottom-left":
            x, y = margin, img_h - leg_h - margin
        elif position == "bottom-right":
            x, y = img_w - leg_w - margin, img_h - leg_h - margin
        else:
            raise ValueError(f"Invalid legend position: {position}")

        # Ensure legend fits in image
        if x < 0 or y < 0 or x + leg_w > img_w or y + leg_h > img_h:
            raise ValueError(
                f"Legend ({leg_w}x{leg_h}) doesn't fit in image ({img_w}x{img_h}) at position {position}"
            )

        # Composite legend using alpha blending
        result = image.copy()
        legend_region = result[y : y + leg_h, x : x + leg_w]

        # Alpha blend legend onto image region
        alpha_legend = legend[:, :, 3:4] / 255.0
        alpha_image = legend_region[:, :, 3:4] / 255.0

        blended_rgb = (
            legend[:, :, :3] * alpha_legend
            + legend_region[:, :, :3] * alpha_image * (1 - alpha_legend)
        ).astype(np.uint8)

        blended_alpha = (alpha_legend + alpha_image * (1 - alpha_legend)) * 255

        result[y : y + leg_h, x : x + leg_w] = np.dstack(
            [blended_rgb, blended_alpha.astype(np.uint8)]
        )

        return result
