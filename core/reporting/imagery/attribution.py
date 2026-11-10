"""
Attribution block generation for FirstLight visualizations (VIS-1.4 Task 3).

Provides proper attribution for all data sources used in satellite imagery
products, ensuring compliance with data provider requirements and giving
credit to OpenStreetMap contributors, satellite operators, etc.

Features:
- Structured data source tracking
- Auto-formatted attribution text
- Positioning in image corners
- Compliance with OSM, ESA, NASA attribution requirements

Example:
    from core.reporting.imagery.attribution import (
        DataSource,
        create_attribution_block,
        format_attribution_text,
    )
    from datetime import datetime

    # Define data sources
    sources = [
        DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25)),
        DataSource("OpenStreetMap", "OSM Contributors"),
    ]

    # Create attribution block
    attribution = create_attribution_block(sources, position="bottom-right")

    # Or just get formatted text
    text = format_attribution_text(sources)
    # "Data: Sentinel-2 (ESA, 2026-01-25), OpenStreetMap (OSM Contributors)"
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class DataSource:
    """
    Represents a single data source for attribution.

    Attributes:
        name: Name of the data source (e.g., "Sentinel-2", "Landsat-8")
        provider: Provider organization (e.g., "ESA", "NASA", "USGS")
        acquisition_date: Optional date when data was acquired
        license: Optional license information (e.g., "CC-BY-SA", "Public Domain")

    Example:
        # Satellite imagery source
        sentinel = DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25))

        # OpenStreetMap source (no acquisition date)
        osm = DataSource(
            "OpenStreetMap",
            "OSM Contributors",
            license="ODbL"
        )
    """

    name: str
    provider: str
    acquisition_date: Optional[datetime] = None
    license: Optional[str] = None

    def __post_init__(self):
        """Validate data source."""
        if not self.name or not self.provider:
            raise ValueError("name and provider are required")


def format_attribution_text(sources: List[DataSource]) -> str:
    """
    Format data sources as citation text.

    Creates a human-readable attribution string listing all data sources
    with their providers and acquisition dates.

    Args:
        sources: List of DataSource objects to format

    Returns:
        Formatted attribution text string

    Example:
        sources = [
            DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25)),
            DataSource("Landsat-8", "NASA/USGS", datetime(2026, 1, 20)),
            DataSource("OpenStreetMap", "OSM Contributors"),
        ]

        text = format_attribution_text(sources)
        # Returns: "Data: Sentinel-2 (ESA, 2026-01-25), Landsat-8 (NASA/USGS, 2026-01-20), OpenStreetMap (OSM Contributors)"
    """
    if not sources:
        return "Data: No sources specified"

    # Format each source
    formatted_sources = []
    for source in sources:
        parts = [source.name]

        # Add provider and date in parentheses
        info_parts = [source.provider]
        if source.acquisition_date:
            date_str = source.acquisition_date.strftime("%Y-%m-%d")
            info_parts.append(date_str)

        parts.append(f"({', '.join(info_parts)})")

        formatted_sources.append(" ".join(parts))

    # Combine all sources
    return "Data: " + ", ".join(formatted_sources)


def create_attribution_block(
    sources: List[DataSource],
    position: str = "bottom-right",
    width: Optional[int] = None,
    background_color: Tuple[int, int, int, int] = (255, 255, 255, 230),
    text_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
    font_size: int = 10,
    padding: int = 8,
) -> np.ndarray:
    """
    Create attribution block listing all data sources.

    Generates a semi-transparent attribution block that can be composited
    onto satellite imagery. The block includes proper citations for all
    data sources used.

    Args:
        sources: List of DataSource objects to attribute
        position: Corner position for block placement
                 ("top-left", "top-right", "bottom-left", "bottom-right")
                 This affects text alignment, not final positioning
        width: Optional fixed width in pixels (auto-calculated if None)
        background_color: RGBA background color
        text_color: RGBA text color
        font_size: Font size in points
        padding: Padding around text in pixels

    Returns:
        RGBA image array ready for compositing

    Example:
        sources = [
            DataSource("Sentinel-2", "ESA", datetime(2026, 1, 25)),
            DataSource("OpenStreetMap", "OSM Contributors"),
        ]

        attribution = create_attribution_block(sources)

        # Composite onto image (bottom-right corner)
        h, w = attribution.shape[:2]
        img_h, img_w = main_image.shape[:2]
        margin = 10
        y1 = img_h - h - margin
        x1 = img_w - w - margin
        main_image[y1:y1+h, x1:x1+w] = attribution
    """
    # Validate inputs
    valid_positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
    if position not in valid_positions:
        raise ValueError(
            f"Invalid position: {position}. Choose from: {valid_positions}"
        )

    # Format attribution text
    text = format_attribution_text(sources)

    # Load font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size
        )
    except OSError:
        # Fallback to default font
        font = ImageFont.load_default()

    # Calculate text size
    # Create temporary image to measure text
    temp_img = Image.new("RGBA", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate block size
    if width is None:
        block_width = text_width + 2 * padding
    else:
        block_width = width

    block_height = text_height + 2 * padding

    # Create attribution block image
    img = Image.new("RGBA", (block_width, block_height), background_color)
    draw = ImageDraw.Draw(img)

    # Draw text
    # Center text horizontally if width is specified
    if width is None:
        text_x = padding
    else:
        text_x = (block_width - text_width) // 2

    text_y = padding

    draw.text((text_x, text_y), text, fill=text_color, font=font)

    # Add subtle border
    draw.rectangle(
        [0, 0, block_width - 1, block_height - 1],
        outline=(0, 0, 0, 100),
        width=1,
    )

    return np.array(img)


def add_license_info(sources: List[DataSource]) -> str:
    """
    Generate license information text for data sources.

    Creates a detailed license block for use in reports or metadata.
    Useful for compliance with data provider requirements.

    Args:
        sources: List of DataSource objects

    Returns:
        Multi-line license information text

    Example:
        sources = [
            DataSource("Sentinel-2", "ESA", license="CC-BY-SA 3.0 IGO"),
            DataSource("OpenStreetMap", "OSM Contributors", license="ODbL"),
        ]

        license_text = add_license_info(sources)
        # Returns:
        # Sentinel-2: Copyright ESA, licensed under CC-BY-SA 3.0 IGO
        # OpenStreetMap: Copyright OSM Contributors, licensed under ODbL
    """
    if not sources:
        return "No license information available"

    lines = []
    for source in sources:
        if source.license:
            line = f"{source.name}: Copyright {source.provider}, licensed under {source.license}"
        else:
            line = f"{source.name}: Copyright {source.provider}"
        lines.append(line)

    return "\n".join(lines)


# Common data sources for convenience

def sentinel2_source(acquisition_date: datetime) -> DataSource:
    """Create DataSource for Sentinel-2 imagery."""
    return DataSource(
        "Sentinel-2",
        "ESA",
        acquisition_date=acquisition_date,
        license="CC-BY-SA 3.0 IGO"
    )


def landsat8_source(acquisition_date: datetime) -> DataSource:
    """Create DataSource for Landsat-8 imagery."""
    return DataSource(
        "Landsat-8",
        "NASA/USGS",
        acquisition_date=acquisition_date,
        license="Public Domain"
    )


def landsat9_source(acquisition_date: datetime) -> DataSource:
    """Create DataSource for Landsat-9 imagery."""
    return DataSource(
        "Landsat-9",
        "NASA/USGS",
        acquisition_date=acquisition_date,
        license="Public Domain"
    )


def osm_source() -> DataSource:
    """Create DataSource for OpenStreetMap."""
    return DataSource(
        "OpenStreetMap",
        "OSM Contributors",
        license="ODbL"
    )


def opentopodata_source() -> DataSource:
    """Create DataSource for OpenTopoData elevation data."""
    return DataSource(
        "OpenTopoData",
        "Various Contributors",
        license="Various"
    )
