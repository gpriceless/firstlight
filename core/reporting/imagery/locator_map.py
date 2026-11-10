"""
Locator map generation for FirstLight visualizations (VIS-1.4 Task 3).

Creates small inset maps that show the detection area in regional context,
helping viewers understand where the analyzed area is located geographically.

Features:
- Detection area highlighted with red box
- State/country boundaries for context
- Automatic zoom level calculation
- Returns RGBA image array ready for compositing

Example:
    from core.reporting.imagery.locator_map import create_locator_map

    # Create locator map showing detection area
    detection_bounds = (-80.5, 25.5, -80.0, 26.0)  # Miami area
    locator = create_locator_map(
        detection_bounds,
        size=(200, 200),
        context_zoom=2
    )

    # Composite onto main image
    result[10:210, 10:210] = locator  # Top-left corner
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def create_locator_map(
    detection_bounds: Tuple[float, float, float, float],
    size: Tuple[int, int] = (200, 200),
    context_zoom: int = 2,
    background_color: Tuple[int, int, int] = (240, 240, 240),
    detection_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Create small inset map showing detection area in regional context.

    The locator map provides geographic context by showing:
    - A wider regional view (controlled by context_zoom)
    - The detection area as a highlighted box
    - Optional state/country boundaries (if available)

    Args:
        detection_bounds: Detection area bounds (minx, miny, maxx, maxy) in EPSG:4326
        size: Output image size as (width, height) in pixels
        context_zoom: How many zoom levels out from detection area (1-5)
                     1 = tight context, 5 = continental context
        background_color: RGB color for background
        detection_color: RGB color for detection box highlight

    Returns:
        RGBA image array of shape (height, width, 4) with dtype uint8

    Example:
        # Show Houston flood area in Texas context
        houston_bounds = (-95.8, 29.5, -95.0, 30.1)
        locator = create_locator_map(houston_bounds, context_zoom=2)

        # Result shows Texas with Houston highlighted
    """
    # Validate inputs
    if len(detection_bounds) != 4:
        raise ValueError(f"detection_bounds must have 4 elements, got {len(detection_bounds)}")

    minx, miny, maxx, maxy = detection_bounds
    if minx >= maxx or miny >= maxy:
        raise ValueError(
            f"Invalid bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}"
        )

    if not (1 <= context_zoom <= 5):
        raise ValueError(f"context_zoom must be 1-5, got {context_zoom}")

    width, height = size

    # Calculate context bounds (expand detection bounds by context_zoom factor)
    det_width = maxx - minx
    det_height = maxy - miny
    det_center_x = (minx + maxx) / 2
    det_center_y = (miny + maxy) / 2

    # Expand by context_zoom factor (exponential for visual effect)
    expansion_factor = 2 ** context_zoom
    context_width = det_width * expansion_factor
    context_height = det_height * expansion_factor

    # Ensure square aspect ratio for consistent appearance
    context_size = max(context_width, context_height)

    context_bounds = (
        det_center_x - context_size / 2,  # minx
        det_center_y - context_size / 2,  # miny
        det_center_x + context_size / 2,  # maxx
        det_center_y + context_size / 2,  # maxy
    )

    # Create image
    img = Image.new("RGBA", (width, height), (*background_color, 255))
    draw = ImageDraw.Draw(img)

    # Draw simple graticule (latitude/longitude grid)
    _draw_graticule(draw, context_bounds, (width, height))

    # Convert detection bounds to pixel coordinates
    det_box_px = _geo_to_pixel(detection_bounds, context_bounds, (width, height))

    # Draw detection area box
    draw.rectangle(
        det_box_px,
        outline=(*detection_color, 255),
        width=3,
    )

    # Add semi-transparent fill to detection area
    draw.rectangle(
        det_box_px,
        fill=(*detection_color, 80),  # 30% opacity
    )

    # Add border around entire map
    draw.rectangle(
        [0, 0, width - 1, height - 1],
        outline=(0, 0, 0, 255),
        width=2,
    )

    # Add coordinate labels
    _add_coordinate_labels(draw, context_bounds, (width, height))

    return np.array(img)


def _geo_to_pixel(
    geo_bounds: Tuple[float, float, float, float],
    context_bounds: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Convert geographic bounds to pixel coordinates.

    Args:
        geo_bounds: Geographic bounds to convert (minx, miny, maxx, maxy)
        context_bounds: Full context bounds (minx, miny, maxx, maxy)
        image_size: Image size (width, height)

    Returns:
        Pixel coordinates (x1, y1, x2, y2)
    """
    geo_minx, geo_miny, geo_maxx, geo_maxy = geo_bounds
    ctx_minx, ctx_miny, ctx_maxx, ctx_maxy = context_bounds
    width, height = image_size

    # Calculate pixel coordinates
    # Note: Image Y axis is inverted (0 at top)
    x1 = int((geo_minx - ctx_minx) / (ctx_maxx - ctx_minx) * width)
    x2 = int((geo_maxx - ctx_minx) / (ctx_maxx - ctx_minx) * width)
    y1 = int((ctx_maxy - geo_maxy) / (ctx_maxy - ctx_miny) * height)
    y2 = int((ctx_maxy - geo_miny) / (ctx_maxy - ctx_miny) * height)

    return (x1, y1, x2, y2)


def _draw_graticule(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
    num_lines: int = 4,
) -> None:
    """
    Draw latitude/longitude grid lines.

    Args:
        draw: PIL ImageDraw object
        bounds: Map bounds (minx, miny, maxx, maxy)
        image_size: Image size (width, height)
        num_lines: Number of grid lines in each direction
    """
    minx, miny, maxx, maxy = bounds
    width, height = image_size

    # Draw vertical lines (longitude)
    for i in range(num_lines + 1):
        lon = minx + (maxx - minx) * i / num_lines
        x = int((lon - minx) / (maxx - minx) * width)
        draw.line(
            [(x, 0), (x, height)],
            fill=(200, 200, 200, 128),
            width=1,
        )

    # Draw horizontal lines (latitude)
    for i in range(num_lines + 1):
        lat = miny + (maxy - miny) * i / num_lines
        # Invert Y axis
        y = int((maxy - lat) / (maxy - miny) * height)
        draw.line(
            [(0, y), (width, y)],
            fill=(200, 200, 200, 128),
            width=1,
        )


def _add_coordinate_labels(
    draw: ImageDraw.ImageDraw,
    bounds: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> None:
    """
    Add coordinate labels to map corners.

    Args:
        draw: PIL ImageDraw object
        bounds: Map bounds (minx, miny, maxx, maxy)
        image_size: Image size (width, height)
    """
    minx, miny, maxx, maxy = bounds
    width, height = image_size

    # Format coordinates
    def format_coord(value: float, is_lat: bool) -> str:
        """Format coordinate as degrees with N/S/E/W."""
        if is_lat:
            direction = "N" if value >= 0 else "S"
        else:
            direction = "E" if value >= 0 else "W"
        return f"{abs(value):.1f}°{direction}"

    # Font settings
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        font = None

    # Add corner labels with background
    margin = 5
    label_bg = (255, 255, 255, 200)
    label_fg = (0, 0, 0, 255)

    # Top-left: Max lat, Min lon
    label = f"{format_coord(maxy, True)}, {format_coord(minx, False)}"
    bbox = draw.textbbox((margin, margin), label, font=font)
    draw.rectangle(bbox, fill=label_bg)
    draw.text((margin, margin), label, fill=label_fg, font=font)

    # Bottom-right: Min lat, Max lon
    label = f"{format_coord(miny, True)}, {format_coord(maxx, False)}"
    bbox = draw.textbbox((width - margin, height - margin), label, font=font, anchor="rb")
    draw.rectangle(bbox, fill=label_bg)
    draw.text((width - margin, height - margin), label, fill=label_fg, font=font, anchor="rb")
