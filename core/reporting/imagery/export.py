"""
Image export functionality with georeferencing support (VIS-1.1 Task 4.1).

Provides export capabilities for rendered imagery:
- Web-optimized PNG for reports and web display
- Georeferenced GeoTIFF preserving spatial metadata
- High-resolution PNG (300 DPI) for PDF embedding
- JPEG export for web thumbnails

Supports transparency via alpha channels and automatic web resizing.

Example:
    # Export to PNG
    exporter = ImageExporter()
    exporter.export_png(rgb_array, Path("output.png"))

    # Export to georeferenced GeoTIFF
    exporter.export_geotiff(
        rgb_array,
        Path("output.tif"),
        crs="EPSG:4326",
        transform=affine_transform
    )

    # High-res PNG for print
    exporter.export_high_res(rgb_array, Path("print.png"), dpi=300)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


@dataclass
class ExportConfig:
    """Configuration for image export operations.

    Attributes:
        format: Output format ('png', 'geotiff', 'jpeg')
        dpi: Dots per inch (72 for web, 300 for print)
        compression: Compression method ('lzw' for tiff, 0-9 for PNG)
        quality: JPEG quality (1-100, higher is better)
    """

    format: str = "png"
    dpi: int = 72
    compression: str = "lzw"
    quality: int = 85

    def __post_init__(self):
        """Validate configuration."""
        valid_formats = ("png", "geotiff", "jpeg")
        if self.format not in valid_formats:
            raise ValueError(
                f"format must be one of {valid_formats}, got '{self.format}'"
            )

        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")

        if not 1 <= self.quality <= 100:
            raise ValueError(f"quality must be 1-100, got {self.quality}")


class ImageExporter:
    """Export rendered imagery to various formats with georeferencing support.

    Handles export to:
    - PNG (with optional compression and DPI settings)
    - GeoTIFF (preserves CRS and geotransform)
    - JPEG (compressed for web thumbnails)

    Supports alpha channel for transparency and automatic web resizing.

    Examples:
        # Basic PNG export
        exporter = ImageExporter()
        exporter.export_png(rgb_array, Path("output.png"))

        # Configure export settings
        config = ExportConfig(format="png", dpi=300, quality=95)
        exporter = ImageExporter(config)

        # Export with transparency
        rgb_with_alpha = add_alpha_channel(rgb_array, valid_mask)
        exporter.export_png(rgb_with_alpha, Path("transparent.png"))
    """

    def __init__(self, config: Optional[ExportConfig] = None):
        """Initialize exporter with optional configuration.

        Args:
            config: Export configuration. If None, uses defaults.
        """
        self.config = config if config is not None else ExportConfig()

    def export_png(
        self,
        rgb: np.ndarray,
        output_path: Path,
        dpi: Optional[int] = None,
        compression: Optional[int] = None,
    ) -> Path:
        """Export RGB array as PNG with optional compression.

        Supports both RGB (3-channel) and RGBA (4-channel) arrays.
        Automatically sets DPI metadata for proper sizing.

        Args:
            rgb: RGB or RGBA array with shape (H, W, 3) or (H, W, 4), dtype uint8
            output_path: Path to save PNG file
            dpi: Override DPI setting (defaults to config.dpi or 72)
            compression: PNG compression level 0-9 (0=none, 9=max)

        Returns:
            Path to saved PNG file

        Raises:
            ValueError: If array shape or dtype is invalid

        Example:
            # Web-optimized PNG
            exporter.export_png(rgb, Path("web.png"), dpi=72)

            # High-compression PNG
            exporter.export_png(rgb, Path("compressed.png"), compression=9)
        """
        # Validate input
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(
                f"rgb must have shape (H, W, 3) or (H, W, 4), got {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")

        # Determine DPI
        dpi_to_use = dpi if dpi is not None else self.config.dpi

        # Determine compression
        compress_level = compression if compression is not None else 6

        # Create PIL Image
        if rgb.shape[2] == 3:
            img = Image.fromarray(rgb, mode="RGB")
        else:
            img = Image.fromarray(rgb, mode="RGBA")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        dpi_tuple = (dpi_to_use, dpi_to_use)
        img.save(
            output_path,
            format="PNG",
            dpi=dpi_tuple,
            compress_level=compress_level,
        )

        return output_path

    def export_geotiff(
        self,
        rgb: np.ndarray,
        output_path: Path,
        crs: str,
        transform,  # Affine transform
        nodata: Optional[int] = None,
        compression: Optional[str] = None,
    ) -> Path:
        """Export RGB array as georeferenced GeoTIFF.

        Preserves spatial metadata (CRS and geotransform) for GIS compatibility.
        Requires rasterio to be installed.

        Args:
            rgb: RGB array with shape (H, W, 3), dtype uint8
            output_path: Path to save GeoTIFF file
            crs: Coordinate reference system (e.g., 'EPSG:4326')
            transform: Affine transform for georeferencing
            nodata: Optional nodata value for each band
            compression: Compression method ('lzw', 'deflate', 'jpeg')

        Returns:
            Path to saved GeoTIFF file

        Raises:
            ImportError: If rasterio is not installed
            ValueError: If array shape or dtype is invalid

        Example:
            from affine import Affine

            # Create geotransform
            transform = Affine.translation(lon_min, lat_max) * Affine.scale(
                pixel_width, -pixel_height
            )

            # Export georeferenced TIFF
            exporter.export_geotiff(
                rgb, Path("geo.tif"),
                crs="EPSG:4326",
                transform=transform
            )
        """
        try:
            import rasterio
            from rasterio.transform import Affine
        except ImportError:
            raise ImportError(
                "rasterio is required for GeoTIFF export. "
                "Install with: pip install rasterio"
            )

        # Validate input
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must have shape (H, W, 3) for GeoTIFF, got {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")

        # Ensure transform is Affine type
        if not isinstance(transform, Affine):
            raise TypeError(
                f"transform must be rasterio.transform.Affine, got {type(transform)}"
            )

        # Determine compression
        compress_method = compression if compression is not None else self.config.compression

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Rasterio expects (bands, height, width) so transpose
        rgb_bands = np.transpose(rgb, (2, 0, 1))  # (3, H, W)

        height, width = rgb.shape[:2]

        # Write GeoTIFF
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,  # RGB = 3 bands
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
            compress=compress_method,
            nodata=nodata,
        ) as dst:
            # Write each band
            dst.write(rgb_bands[0], 1)  # Red
            dst.write(rgb_bands[1], 2)  # Green
            dst.write(rgb_bands[2], 3)  # Blue

            # Set band descriptions
            dst.set_band_description(1, "Red")
            dst.set_band_description(2, "Green")
            dst.set_band_description(3, "Blue")

        return output_path

    def export_high_res(
        self,
        rgb: np.ndarray,
        output_path: Path,
        dpi: int = 300,
    ) -> Path:
        """Export high-resolution PNG for PDF embedding or print.

        Creates a 300 DPI PNG suitable for high-quality print output.
        This is a convenience wrapper around export_png with print settings.

        Args:
            rgb: RGB or RGBA array with shape (H, W, 3) or (H, W, 4), dtype uint8
            output_path: Path to save PNG file
            dpi: DPI for print quality (default 300)

        Returns:
            Path to saved PNG file

        Example:
            # 300 DPI for standard print
            exporter.export_high_res(rgb, Path("print.png"))

            # 600 DPI for high-end printing
            exporter.export_high_res(rgb, Path("hires.png"), dpi=600)
        """
        return self.export_png(rgb, output_path, dpi=dpi, compression=9)

    def export_jpeg(
        self,
        rgb: np.ndarray,
        output_path: Path,
        quality: Optional[int] = None,
    ) -> Path:
        """Export RGB array as JPEG for web thumbnails.

        JPEG provides better compression than PNG for photographic imagery
        but does not support transparency. Use for web thumbnails and previews.

        Args:
            rgb: RGB array with shape (H, W, 3), dtype uint8
            output_path: Path to save JPEG file
            quality: JPEG quality 1-100 (defaults to config.quality or 85)

        Returns:
            Path to saved JPEG file

        Raises:
            ValueError: If array shape is invalid or includes alpha channel

        Example:
            # Standard quality thumbnail
            exporter.export_jpeg(rgb, Path("thumb.jpg"), quality=85)

            # High-quality preview
            exporter.export_jpeg(rgb, Path("preview.jpg"), quality=95)
        """
        # Validate input
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"rgb must have shape (H, W, 3) for JPEG (no alpha), got {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")

        # Determine quality
        quality_to_use = quality if quality is not None else self.config.quality

        # Create PIL Image
        img = Image.fromarray(rgb, mode="RGB")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save JPEG
        img.save(output_path, format="JPEG", quality=quality_to_use, optimize=True)

        return output_path


def add_alpha_channel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Add alpha channel to RGB array based on validity mask.

    Converts RGB image to RGBA by adding transparency channel.
    Pixels marked as invalid (False) in the mask become fully transparent.

    Args:
        rgb: RGB array with shape (H, W, 3), dtype uint8
        mask: Boolean validity mask with shape (H, W) - True = valid, False = nodata

    Returns:
        RGBA array with shape (H, W, 4), dtype uint8

    Raises:
        ValueError: If shapes don't match or dtypes are invalid

    Example:
        # Create transparent PNG from rendered image with nodata
        rgb_with_alpha = add_alpha_channel(result.rgb_array, result.valid_mask)
        exporter.export_png(rgb_with_alpha, Path("transparent.png"))
    """
    # Validate inputs
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")

    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    if rgb.shape[:2] != mask.shape:
        raise ValueError(
            f"rgb spatial dimensions {rgb.shape[:2]} must match mask shape {mask.shape}"
        )

    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")

    # Create alpha channel: 255 for valid, 0 for invalid
    alpha = np.where(mask, 255, 0).astype(np.uint8)

    # Stack RGB with alpha
    rgba = np.dstack([rgb, alpha])

    return rgba


def resize_for_web(rgb: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
    """Resize large images for web display while preserving aspect ratio.

    Downsamples images that exceed max_dimension on either axis.
    Small images are not upsampled.

    Args:
        rgb: RGB or RGBA array with shape (H, W, 3) or (H, W, 4), dtype uint8
        max_dimension: Maximum width or height in pixels (default 2048)

    Returns:
        Resized array with same number of channels, dtype uint8

    Example:
        # Resize large satellite image for web
        web_size = resize_for_web(rgb, max_dimension=1024)

        # No change if already small
        small = resize_for_web(rgb_500x500, max_dimension=2048)
        assert small.shape == (500, 500, 3)
    """
    # Validate input
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError(
            f"rgb must have shape (H, W, 3) or (H, W, 4), got {rgb.shape}"
        )

    if rgb.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb.dtype}")

    height, width = rgb.shape[:2]

    # Check if resize needed
    if height <= max_dimension and width <= max_dimension:
        return rgb

    # Calculate new dimensions preserving aspect ratio
    scale = max_dimension / max(height, width)
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Use PIL for high-quality resize
    mode = "RGB" if rgb.shape[2] == 3 else "RGBA"
    img = Image.fromarray(rgb, mode=mode)
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return np.array(img_resized, dtype=np.uint8)
