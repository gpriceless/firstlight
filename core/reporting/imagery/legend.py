"""
Automatic legend generation for detection overlays (VIS-1.3 Task 1.2).

This module generates legend images for flood/wildfire detection visualizations.
Legends are rendered as transparent PNG images that can be overlaid on maps.

Uses PIL/Pillow for lightweight rendering without matplotlib dependencies.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color_scales import ColorPalette


@dataclass
class LegendConfig:
    """
    Configuration for legend rendering.

    Attributes:
        title: Legend title text
        position: Corner position ("top-left", "top-right", "bottom-left", "bottom-right")
        width: Legend width in pixels
        height: Legend height in pixels (None = auto-calculate)
        font_size: Font size for text
        background_alpha: Background opacity (0-1)
        show_title: Whether to show the title
    """

    title: str = "Legend"
    position: str = "top-right"
    width: int = 200
    height: Optional[int] = None
    font_size: int = 12
    background_alpha: float = 0.8
    show_title: bool = True

    def __post_init__(self):
        """Validate configuration."""
        valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        if self.position not in valid_positions:
            raise ValueError(
                f"Invalid position: {self.position}. Choose from: {valid_positions}"
            )

        if self.width <= 0:
            raise ValueError(f"Width must be positive, got {self.width}")

        if self.height is not None and self.height <= 0:
            raise ValueError(f"Height must be positive, got {self.height}")

        if not 0 <= self.background_alpha <= 1:
            raise ValueError(
                f"Background alpha must be between 0 and 1, got {self.background_alpha}"
            )

        if self.font_size <= 0:
            raise ValueError(f"Font size must be positive, got {self.font_size}")


class LegendRenderer:
    """
    Renders legends for detection overlays.

    Creates legend images with color scales and labels for discrete severity
    levels or continuous value ranges. Outputs transparent PNG images suitable
    for overlay on satellite imagery or maps.
    """

    # Layout constants
    PADDING = 10
    COLOR_BOX_HEIGHT = 20
    COLOR_BOX_MARGIN = 5
    GRADIENT_HEIGHT = 20
    TEXT_MARGIN = 5
    TITLE_BOTTOM_MARGIN = 10

    def __init__(self, config: Optional[LegendConfig] = None):
        """
        Initialize legend renderer.

        Args:
            config: Legend configuration (uses defaults if None)
        """
        self.config = config or LegendConfig()

        # Try to load a default font, fall back to PIL's default
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", self.config.font_size)
            self.title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", self.config.font_size + 2)
        except (OSError, IOError):
            # Fall back to default font if truetype not available
            self.font = ImageFont.load_default()
            self.title_font = ImageFont.load_default()

    def generate_discrete(
        self, palette: ColorPalette, output_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Generate legend for discrete severity levels.

        Creates a legend with color boxes and labels for each severity level.
        Suitable for flood/wildfire severity classifications.

        Args:
            palette: ColorPalette defining colors and labels
            output_path: Optional path to save PNG file

        Returns:
            RGBA numpy array with shape (H, W, 4) and dtype uint8

        Examples:
            >>> from core.reporting.imagery.color_scales import get_flood_palette
            >>> renderer = LegendRenderer()
            >>> palette = get_flood_palette()
            >>> legend = renderer.generate_discrete(palette)
            >>> legend.shape
            (height, 200, 4)
        """
        # Calculate required height
        height = self._calculate_discrete_height(palette)

        # Create transparent image
        img = Image.new("RGBA", (self.config.width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        y_offset = self.PADDING

        # Draw title if enabled
        if self.config.show_title:
            title_text = self.config.title or palette.name
            y_offset = self._draw_text(
                draw, title_text, y_offset, self.title_font, bold=True
            )
            y_offset += self.TITLE_BOTTOM_MARGIN

        # Draw each color box with label
        for color, label in zip(palette.colors, palette.labels):
            y_offset = self._draw_color_box(draw, color, label, y_offset)

        # Add semi-transparent background
        img_with_bg = self._add_background(img)

        # Convert to numpy array
        result = np.array(img_with_bg)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_with_bg.save(output_path, "PNG")

        return result

    def generate_continuous(
        self,
        palette: ColorPalette,
        min_val: float,
        max_val: float,
        output_path: Optional[Path] = None,
    ) -> np.ndarray:
        """
        Generate legend for continuous value range.

        Creates a legend with a gradient color bar and min/max value labels.
        Suitable for continuous data like water depth or burn severity indices.

        Args:
            palette: ColorPalette defining colors
            min_val: Minimum value for the scale
            max_val: Maximum value for the scale
            output_path: Optional path to save PNG file

        Returns:
            RGBA numpy array with shape (H, W, 4) and dtype uint8

        Examples:
            >>> from core.reporting.imagery.color_scales import get_flood_palette
            >>> renderer = LegendRenderer()
            >>> palette = get_flood_palette()
            >>> legend = renderer.generate_continuous(palette, 0.0, 1.0)
            >>> legend.shape
            (height, 200, 4)
        """
        # Calculate required height
        height = self._calculate_continuous_height()

        # Create transparent image
        img = Image.new("RGBA", (self.config.width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        y_offset = self.PADDING

        # Draw title if enabled
        if self.config.show_title:
            title_text = self.config.title or palette.name
            y_offset = self._draw_text(
                draw, title_text, y_offset, self.title_font, bold=True
            )
            y_offset += self.TITLE_BOTTOM_MARGIN

        # Draw gradient bar
        gradient_y = y_offset
        gradient_x1 = self.PADDING
        gradient_x2 = self.config.width - self.PADDING
        gradient_width = gradient_x2 - gradient_x1

        # Create gradient by drawing vertical segments
        n_colors = len(palette.colors)
        segment_width = gradient_width / n_colors

        for i, color in enumerate(palette.colors):
            x1 = gradient_x1 + int(i * segment_width)
            x2 = gradient_x1 + int((i + 1) * segment_width)
            y1 = gradient_y
            y2 = gradient_y + self.GRADIENT_HEIGHT

            # Convert RGBA tuple to format PIL expects
            rgb_color = color[:3]  # Take only RGB, handle alpha separately
            draw.rectangle([x1, y1, x2, y2], fill=rgb_color)

        y_offset = gradient_y + self.GRADIENT_HEIGHT + self.TEXT_MARGIN

        # Draw min value (left aligned)
        min_text = f"{min_val:.2f}"
        self._draw_text_at_position(
            draw, min_text, gradient_x1, y_offset, self.font, align="left"
        )

        # Draw max value (right aligned)
        max_text = f"{max_val:.2f}"
        self._draw_text_at_position(
            draw, max_text, gradient_x2, y_offset, self.font, align="right"
        )

        y_offset += self._get_text_height(min_text) + self.PADDING

        # Add semi-transparent background
        img_with_bg = self._add_background(img)

        # Convert to numpy array
        result = np.array(img_with_bg)

        # Save to file if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img_with_bg.save(output_path, "PNG")

        return result

    def _draw_color_box(
        self, draw: ImageDraw.ImageDraw, color: Tuple[int, int, int, int], label: str, y_offset: int
    ) -> int:
        """
        Draw a color box with label.

        Args:
            draw: PIL ImageDraw object
            color: RGBA color tuple
            label: Text label for the color
            y_offset: Current vertical offset

        Returns:
            Updated y_offset after drawing
        """
        # Draw color box (left side)
        box_x1 = self.PADDING
        box_y1 = y_offset
        box_x2 = box_x1 + self.COLOR_BOX_HEIGHT
        box_y2 = box_y1 + self.COLOR_BOX_HEIGHT

        # Convert RGBA to RGB for PIL (alpha handled by image mode)
        rgb_color = color[:3]
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=rgb_color, outline="black")

        # Draw label (right of box)
        text_x = box_x2 + self.TEXT_MARGIN
        text_y = y_offset + (self.COLOR_BOX_HEIGHT - self._get_text_height(label)) // 2
        draw.text((text_x, text_y), label, fill="black", font=self.font)

        return y_offset + self.COLOR_BOX_HEIGHT + self.COLOR_BOX_MARGIN

    def _draw_text(
        self, draw: ImageDraw.ImageDraw, text: str, y_offset: int, font: ImageFont.ImageFont, bold: bool = False
    ) -> int:
        """
        Draw centered text.

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            y_offset: Current vertical offset
            font: Font to use
            bold: Whether text is bold (for spacing)

        Returns:
            Updated y_offset after drawing
        """
        # Get text dimensions using textbbox
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center horizontally
        text_x = (self.config.width - text_width) // 2
        text_y = y_offset

        draw.text((text_x, text_y), text, fill="black", font=font)

        return y_offset + text_height + self.TEXT_MARGIN

    def _draw_text_at_position(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        font: ImageFont.ImageFont,
        align: str = "left",
    ):
        """
        Draw text at specific position with alignment.

        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            x: X coordinate
            y: Y coordinate
            font: Font to use
            align: Text alignment ("left", "right", "center")
        """
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        if align == "right":
            x = x - text_width
        elif align == "center":
            x = x - text_width // 2

        draw.text((x, y), text, fill="black", font=font)

    def _get_text_height(self, text: str) -> int:
        """Get text height in pixels."""
        # Create temporary image to measure text
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=self.font)
        return bbox[3] - bbox[1]

    def _calculate_discrete_height(self, palette: ColorPalette) -> int:
        """Calculate required height for discrete legend."""
        height = self.PADDING  # Top padding

        # Title height
        if self.config.show_title:
            title_text = self.config.title or palette.name
            height += self._get_text_height(title_text) + self.TEXT_MARGIN
            height += self.TITLE_BOTTOM_MARGIN

        # Color boxes
        n_colors = len(palette.colors)
        height += n_colors * (self.COLOR_BOX_HEIGHT + self.COLOR_BOX_MARGIN)
        height -= self.COLOR_BOX_MARGIN  # Remove last margin

        # Bottom padding
        height += self.PADDING

        return height if self.config.height is None else self.config.height

    def _calculate_continuous_height(self) -> int:
        """Calculate required height for continuous legend."""
        height = self.PADDING  # Top padding

        # Title height
        if self.config.show_title:
            height += self._get_text_height("Sample") + self.TEXT_MARGIN
            height += self.TITLE_BOTTOM_MARGIN

        # Gradient bar
        height += self.GRADIENT_HEIGHT + self.TEXT_MARGIN

        # Value labels
        height += self._get_text_height("0.00") + self.PADDING

        return height if self.config.height is None else self.config.height

    def _add_background(self, img: Image.Image) -> Image.Image:
        """
        Add semi-transparent background for readability.

        Args:
            img: Input image with transparent background

        Returns:
            Image with semi-transparent white background
        """
        # Create white background with alpha
        bg_alpha = int(255 * self.config.background_alpha)
        background = Image.new("RGBA", img.size, (255, 255, 255, bg_alpha))

        # Composite legend over background
        result = Image.alpha_composite(background, img)

        return result
