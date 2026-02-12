"""
Before/After Temporal Image Generator for VIS-1.2.

Generates temporal comparison products from satellite imagery pairs:
- Side-by-side comparisons: Before and after images placed horizontally
- Labeled comparisons: Date labels added to each panel
- Animated GIFs: Alternating before/after for change visualization
- Custom styling: Configurable gaps, labels, and timing

Used for visualizing change detection results, disaster impact assessment,
and temporal analysis of events like floods, wildfires, and infrastructure changes.

Part of VIS-1.2: Before/After Image Generation

Example:
    # Generate side-by-side comparison
    config = BeforeAfterConfig(
        time_window_days=30,
        max_cloud_cover=20.0
    )
    generator = BeforeAfterGenerator(config)

    result = BeforeAfterResult(
        before_image=before_array,
        after_image=after_array,
        before_date=datetime(2024, 1, 1),
        after_date=datetime(2024, 1, 15)
    )

    # Create comparison products
    side_by_side = generator.generate_side_by_side(result)
    labeled = generator.generate_labeled_comparison(result)
    gif = generator.generate_animated_gif(result, Path("comparison.gif"))
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class BeforeAfterConfig:
    """Configuration for temporal image selection.

    Attributes:
        time_window_days: Maximum time difference between before/after images
        max_cloud_cover: Maximum acceptable cloud cover percentage (0-100)
        min_coverage: Minimum data coverage percentage required (0-100)
    """

    time_window_days: int = 30
    max_cloud_cover: float = 20.0
    min_coverage: float = 80.0

    def __post_init__(self):
        """Validate configuration."""
        if self.time_window_days <= 0:
            raise ValueError(
                f"time_window_days must be positive, got {self.time_window_days}"
            )

        if not 0 <= self.max_cloud_cover <= 100:
            raise ValueError(
                f"max_cloud_cover must be in [0, 100], got {self.max_cloud_cover}"
            )

        if not 0 <= self.min_coverage <= 100:
            raise ValueError(
                f"min_coverage must be in [0, 100], got {self.min_coverage}"
            )


@dataclass
class BeforeAfterResult:
    """Result of temporal image pair selection.

    Attributes:
        before_image: Before image as RGB numpy array (H, W, 3)
        after_image: After image as RGB numpy array (H, W, 3)
        before_date: Acquisition date of before image
        after_date: Acquisition date of after image
        before_cloud_cover: Cloud cover percentage of before image (0-100)
        after_cloud_cover: Cloud cover percentage of after image (0-100)
        metadata: Additional metadata about the image pair
    """

    before_image: np.ndarray
    after_image: np.ndarray
    before_date: datetime
    after_date: datetime
    before_cloud_cover: float = 0.0
    after_cloud_cover: float = 0.0
    metadata: dict = None

    def __post_init__(self):
        """Validate result."""
        if self.before_image.shape != self.after_image.shape:
            raise ValueError(
                f"Before and after images must have same shape. "
                f"Got before={self.before_image.shape}, after={self.after_image.shape}"
            )

        if self.before_image.ndim != 3 or self.before_image.shape[2] != 3:
            raise ValueError(
                f"Images must be RGB (H, W, 3), got shape {self.before_image.shape}"
            )

        if self.before_date >= self.after_date:
            raise ValueError(
                f"before_date must be earlier than after_date. "
                f"Got before={self.before_date}, after={self.after_date}"
            )

        if self.metadata is None:
            self.metadata = {}


@dataclass
class OutputConfig:
    """Configuration for output styling.

    Attributes:
        gap_width: Width of white gap between side-by-side images in pixels
        label_font_size: Font size for date labels in points
        label_background_alpha: Transparency of label background (0.0-1.0)
        gif_frame_duration_ms: Duration of each frame in animated GIF (milliseconds)
    """

    gap_width: int = 20
    label_font_size: int = 24
    label_background_alpha: float = 0.7
    gif_frame_duration_ms: int = 1000

    def __post_init__(self):
        """Validate configuration."""
        if self.gap_width < 0:
            raise ValueError(f"gap_width must be non-negative, got {self.gap_width}")

        if self.label_font_size <= 0:
            raise ValueError(
                f"label_font_size must be positive, got {self.label_font_size}"
            )

        if not 0.0 <= self.label_background_alpha <= 1.0:
            raise ValueError(
                f"label_background_alpha must be in [0.0, 1.0], "
                f"got {self.label_background_alpha}"
            )

        if self.gif_frame_duration_ms <= 0:
            raise ValueError(
                f"gif_frame_duration_ms must be positive, "
                f"got {self.gif_frame_duration_ms}"
            )


class BeforeAfterGenerator:
    """Generator for before/after temporal comparison products.

    Creates visual comparison products from temporal image pairs:
    - Side-by-side comparisons with customizable gap
    - Labeled comparisons with acquisition dates
    - Animated GIFs for change visualization

    Examples:
        # Basic side-by-side
        generator = BeforeAfterGenerator()
        composite = generator.generate_side_by_side(result)

        # Custom styling
        output_config = OutputConfig(gap_width=50, label_font_size=32)
        composite = generator.generate_side_by_side(result, config=output_config)

        # Labeled comparison
        labeled = generator.generate_labeled_comparison(result)

        # Animated GIF
        gif_path = generator.generate_animated_gif(
            result,
            Path("comparison.gif"),
            frame_duration_ms=500
        )
    """

    def __init__(self, config: Optional[BeforeAfterConfig] = None):
        """Initialize generator with optional configuration.

        Args:
            config: Configuration for temporal image selection.
                   If None, uses default config.
        """
        self.config = config if config is not None else BeforeAfterConfig()

    def generate_side_by_side(
        self,
        result: BeforeAfterResult,
        output_path: Optional[Path] = None,
        config: Optional[OutputConfig] = None
    ) -> np.ndarray:
        """Generate side-by-side comparison image.

        Creates a composite image with before and after images placed side by side
        with a white gap between them.

        Args:
            result: BeforeAfterResult containing the image pair
            output_path: Optional path to save the output image
            config: Optional OutputConfig for customization

        Returns:
            Side-by-side composite as numpy array with shape (H, W, 3)

        Example:
            >>> composite = generator.generate_side_by_side(result)
            >>> # Or save to file
            >>> composite = generator.generate_side_by_side(
            ...     result, output_path=Path("comparison.png")
            ... )
        """
        if config is None:
            config = OutputConfig()

        height, width = result.before_image.shape[:2]
        gap = config.gap_width

        # Create composite array: before + gap + after
        composite_width = width * 2 + gap
        composite = np.ones((height, composite_width, 3), dtype=np.uint8) * 255

        # Place before image on left
        composite[:, :width, :] = result.before_image

        # Gap is already white (255)

        # Place after image on right
        composite[:, width + gap:, :] = result.after_image

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(composite).save(output_path)

        return composite

    def add_date_labels(
        self,
        image: np.ndarray,
        date: datetime,
        position: str = "bottom",
        config: Optional[OutputConfig] = None
    ) -> np.ndarray:
        """Add date label to an image.

        Args:
            image: Input image as numpy array (H, W, 3)
            date: Date to display
            position: Label position: 'top', 'bottom', 'top-left', 'top-right',
                     'bottom-left', 'bottom-right'
            config: Optional OutputConfig for label styling

        Returns:
            Image with label added (copy of input)

        Raises:
            ValueError: If position is not valid
        """
        if config is None:
            config = OutputConfig()

        valid_positions = {
            "top", "bottom", "top-left", "top-right",
            "bottom-left", "bottom-right"
        }
        if position not in valid_positions:
            raise ValueError(
                f"Invalid position '{position}'. Must be one of {valid_positions}"
            )

        # Convert to PIL Image for text rendering
        pil_img = Image.fromarray(image.copy())
        draw = ImageDraw.Draw(pil_img, mode="RGBA")

        # Format date string
        date_str = date.strftime("%Y-%m-%d")

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                     config.label_font_size)
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc",
                                         config.label_font_size)
            except:
                font = ImageFont.load_default()

        # Get text bounding box
        bbox = draw.textbbox((0, 0), date_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate position
        img_height, img_width = image.shape[:2]
        padding = 10

        if position == "top":
            x = (img_width - text_width) // 2
            y = padding
        elif position == "bottom":
            x = (img_width - text_width) // 2
            y = img_height - text_height - padding * 2
        elif position == "top-left":
            x = padding
            y = padding
        elif position == "top-right":
            x = img_width - text_width - padding * 2
            y = padding
        elif position == "bottom-left":
            x = padding
            y = img_height - text_height - padding * 2
        elif position == "bottom-right":
            x = img_width - text_width - padding * 2
            y = img_height - text_height - padding * 2

        # Draw semi-transparent background
        bg_alpha = int(config.label_background_alpha * 255)
        bg_box = [
            x - padding,
            y - padding,
            x + text_width + padding,
            y + text_height + padding
        ]
        draw.rectangle(bg_box, fill=(0, 0, 0, bg_alpha))

        # Draw text
        draw.text((x, y), date_str, fill=(255, 255, 255, 255), font=font)

        # Convert back to numpy array
        return np.array(pil_img.convert("RGB"))

    def generate_labeled_comparison(
        self,
        result: BeforeAfterResult,
        output_path: Optional[Path] = None,
        config: Optional[OutputConfig] = None
    ) -> np.ndarray:
        """Generate side-by-side comparison with date labels.

        Creates a composite with before and after images side by side, each
        labeled with its acquisition date.

        Args:
            result: BeforeAfterResult containing the image pair
            output_path: Optional path to save the output image
            config: Optional OutputConfig for customization

        Returns:
            Labeled composite as numpy array with shape (H, W, 3)

        Example:
            >>> labeled = generator.generate_labeled_comparison(result)
            >>> # Labels show dates at bottom of each panel
        """
        if config is None:
            config = OutputConfig()

        # First create side-by-side
        composite = self.generate_side_by_side(result, config=config)

        height, width = result.before_image.shape[:2]
        gap = config.gap_width

        # Label before image (left panel)
        before_panel = composite[:, :width, :].copy()
        before_labeled = self.add_date_labels(
            before_panel,
            result.before_date,
            position="bottom",
            config=config
        )
        composite[:, :width, :] = before_labeled

        # Label after image (right panel)
        after_panel = composite[:, width + gap:, :].copy()
        after_labeled = self.add_date_labels(
            after_panel,
            result.after_date,
            position="bottom",
            config=config
        )
        composite[:, width + gap:, :] = after_labeled

        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(composite).save(output_path)

        return composite

    def generate_animated_gif(
        self,
        result: BeforeAfterResult,
        output_path: Path,
        frame_duration_ms: Optional[int] = None,
        config: Optional[OutputConfig] = None
    ) -> Path:
        """Generate animated GIF alternating between before and after images.

        Creates a GIF that flips between the before and after images, useful for
        visualizing changes over time.

        Args:
            result: BeforeAfterResult containing the image pair
            output_path: Path where GIF will be saved
            frame_duration_ms: Duration of each frame in milliseconds
                              (overrides config if provided)
            config: Optional OutputConfig for defaults

        Returns:
            Path to the created GIF file

        Example:
            >>> gif_path = generator.generate_animated_gif(
            ...     result,
            ...     Path("comparison.gif"),
            ...     frame_duration_ms=500
            ... )
        """
        if config is None:
            config = OutputConfig()

        if frame_duration_ms is None:
            frame_duration_ms = config.gif_frame_duration_ms

        # Convert images to PIL
        before_pil = Image.fromarray(result.before_image)
        after_pil = Image.fromarray(result.after_image)

        # Create output directory if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as animated GIF
        before_pil.save(
            output_path,
            save_all=True,
            append_images=[after_pil],
            duration=frame_duration_ms,
            loop=0  # Loop forever
        )

        return output_path
