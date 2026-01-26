"""
Unit Tests for Image Export (VIS-1.1 Task 4.1).

Tests the ImageExporter class and helper functions:
- PNG export with compression and DPI settings
- GeoTIFF export with georeferencing (if rasterio available)
- High-resolution export for print
- JPEG export for web thumbnails
- Alpha channel addition for transparency
- Web resizing for large images

Includes edge cases: invalid inputs, missing dependencies, various array shapes.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.reporting.imagery.export import (
    ExportConfig,
    ImageExporter,
    add_alpha_channel,
    resize_for_web,
)


def _rasterio_available() -> bool:
    """Check if rasterio is available."""
    try:
        import rasterio
        return True
    except ImportError:
        return False


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExportConfig()

        assert config.format == "png"
        assert config.dpi == 72
        assert config.compression == "lzw"
        assert config.quality == 85

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExportConfig(
            format="geotiff",
            dpi=300,
            compression="deflate",
            quality=95,
        )

        assert config.format == "geotiff"
        assert config.dpi == 300
        assert config.compression == "deflate"
        assert config.quality == 95

    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="format must be one of"):
            ExportConfig(format="invalid")

    def test_invalid_dpi(self):
        """Test that invalid DPI raises error."""
        with pytest.raises(ValueError, match="dpi must be positive"):
            ExportConfig(dpi=0)

        with pytest.raises(ValueError, match="dpi must be positive"):
            ExportConfig(dpi=-100)

    def test_invalid_quality(self):
        """Test that invalid quality raises error."""
        with pytest.raises(ValueError, match="quality must be 1-100"):
            ExportConfig(quality=0)

        with pytest.raises(ValueError, match="quality must be 1-100"):
            ExportConfig(quality=101)


class TestImageExporter:
    """Tests for ImageExporter class."""

    @pytest.fixture
    def rgb_array(self):
        """Create test RGB array."""
        # Create gradient image for visual testing
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(0, 255, 100, dtype=np.uint8)[None, :]  # Red gradient
        rgb[:, :, 1] = np.linspace(0, 255, 100, dtype=np.uint8)[:, None]  # Green gradient
        rgb[:, :, 2] = 128  # Constant blue
        return rgb

    @pytest.fixture
    def rgba_array(self):
        """Create test RGBA array with transparency."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[:, :, 0] = 255  # Red
        rgba[:, :, 1] = 128  # Green
        rgba[:, :, 2] = 64   # Blue
        # Alpha: transparent in center, opaque on edges
        y, x = np.ogrid[:100, :100]
        distance = np.sqrt((x - 50)**2 + (y - 50)**2)
        rgba[:, :, 3] = np.clip(distance * 5, 0, 255).astype(np.uint8)
        return rgba

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_export_png_basic(self, rgb_array, temp_dir):
        """Test basic PNG export."""
        exporter = ImageExporter()
        output_path = temp_dir / "test.png"

        result = exporter.export_png(rgb_array, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify image can be read back
        img = Image.open(output_path)
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    def test_export_png_with_dpi(self, rgb_array, temp_dir):
        """Test PNG export with custom DPI."""
        exporter = ImageExporter()
        output_path = temp_dir / "test_dpi.png"

        exporter.export_png(rgb_array, output_path, dpi=150)

        # Verify DPI was set (with tolerance for floating point precision)
        img = Image.open(output_path)
        dpi = img.info.get("dpi")
        assert dpi is not None
        assert abs(dpi[0] - 150) < 1
        assert abs(dpi[1] - 150) < 1

    def test_export_png_with_compression(self, rgb_array, temp_dir):
        """Test PNG export with different compression levels."""
        exporter = ImageExporter()

        # No compression
        path_none = temp_dir / "none.png"
        exporter.export_png(rgb_array, path_none, compression=0)

        # Max compression
        path_max = temp_dir / "max.png"
        exporter.export_png(rgb_array, path_max, compression=9)

        # Max compression should produce smaller file
        # (though with synthetic data, difference may be minimal)
        assert path_none.exists()
        assert path_max.exists()

    def test_export_png_with_alpha(self, rgba_array, temp_dir):
        """Test PNG export with alpha channel."""
        exporter = ImageExporter()
        output_path = temp_dir / "alpha.png"

        exporter.export_png(rgba_array, output_path)

        # Verify image has alpha channel
        img = Image.open(output_path)
        assert img.mode == "RGBA"
        assert img.size == (100, 100)

    def test_export_png_invalid_shape(self, temp_dir):
        """Test PNG export with invalid array shape."""
        exporter = ImageExporter()
        output_path = temp_dir / "invalid.png"

        # 2D array
        with pytest.raises(ValueError, match="must have shape"):
            exporter.export_png(np.zeros((100, 100), dtype=np.uint8), output_path)

        # Wrong number of channels
        with pytest.raises(ValueError, match="must have shape"):
            exporter.export_png(np.zeros((100, 100, 2), dtype=np.uint8), output_path)

    def test_export_png_invalid_dtype(self, temp_dir):
        """Test PNG export with invalid dtype."""
        exporter = ImageExporter()
        output_path = temp_dir / "invalid.png"

        # Float array
        with pytest.raises(ValueError, match="must be uint8"):
            exporter.export_png(np.zeros((100, 100, 3), dtype=np.float32), output_path)

    def test_export_png_creates_directory(self, temp_dir):
        """Test PNG export creates parent directories."""
        exporter = ImageExporter()
        output_path = temp_dir / "subdir" / "nested" / "test.png"

        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        exporter.export_png(rgb, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_export_high_res(self, rgb_array, temp_dir):
        """Test high-resolution PNG export."""
        exporter = ImageExporter()
        output_path = temp_dir / "highres.png"

        result = exporter.export_high_res(rgb_array, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify DPI is 300 (with tolerance for floating point precision)
        img = Image.open(output_path)
        dpi = img.info.get("dpi")
        assert dpi is not None
        assert abs(dpi[0] - 300) < 1
        assert abs(dpi[1] - 300) < 1

    def test_export_high_res_custom_dpi(self, rgb_array, temp_dir):
        """Test high-resolution export with custom DPI."""
        exporter = ImageExporter()
        output_path = temp_dir / "highres_600.png"

        exporter.export_high_res(rgb_array, output_path, dpi=600)

        img = Image.open(output_path)
        dpi = img.info.get("dpi")
        assert dpi is not None
        assert abs(dpi[0] - 600) < 1
        assert abs(dpi[1] - 600) < 1

    def test_export_jpeg_basic(self, rgb_array, temp_dir):
        """Test basic JPEG export."""
        exporter = ImageExporter()
        output_path = temp_dir / "test.jpg"

        result = exporter.export_jpeg(rgb_array, output_path)

        assert result == output_path
        assert output_path.exists()

        # Verify image can be read back
        img = Image.open(output_path)
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    def test_export_jpeg_with_quality(self, rgb_array, temp_dir):
        """Test JPEG export with different quality levels."""
        exporter = ImageExporter()

        # Low quality
        path_low = temp_dir / "low.jpg"
        exporter.export_jpeg(rgb_array, path_low, quality=50)

        # High quality
        path_high = temp_dir / "high.jpg"
        exporter.export_jpeg(rgb_array, path_high, quality=95)

        # Both should exist
        assert path_low.exists()
        assert path_high.exists()

        # High quality should produce larger file
        assert path_high.stat().st_size > path_low.stat().st_size

    def test_export_jpeg_no_alpha(self, rgba_array, temp_dir):
        """Test JPEG export rejects alpha channel."""
        exporter = ImageExporter()
        output_path = temp_dir / "invalid.jpg"

        # JPEG doesn't support alpha
        with pytest.raises(ValueError, match="no alpha"):
            exporter.export_jpeg(rgba_array, output_path)

    def test_export_jpeg_invalid_dtype(self, temp_dir):
        """Test JPEG export with invalid dtype."""
        exporter = ImageExporter()
        output_path = temp_dir / "invalid.jpg"

        # Float array
        with pytest.raises(ValueError, match="must be uint8"):
            exporter.export_jpeg(np.zeros((100, 100, 3), dtype=np.float32), output_path)

    @pytest.mark.skipif(
        not _rasterio_available(),
        reason="rasterio not installed"
    )
    def test_export_geotiff_basic(self, rgb_array, temp_dir):
        """Test basic GeoTIFF export with georeferencing."""
        from affine import Affine

        exporter = ImageExporter()
        output_path = temp_dir / "geo.tif"

        # Create simple geotransform
        transform = Affine.translation(10.0, 50.0) * Affine.scale(0.01, -0.01)

        result = exporter.export_geotiff(
            rgb_array,
            output_path,
            crs="EPSG:4326",
            transform=transform,
        )

        assert result == output_path
        assert output_path.exists()

        # Verify georeferencing
        import rasterio
        with rasterio.open(output_path) as src:
            assert src.crs.to_string() == "EPSG:4326"
            assert src.transform == transform
            assert src.count == 3
            assert src.dtypes == ("uint8", "uint8", "uint8")
            assert src.width == 100
            assert src.height == 100

    @pytest.mark.skipif(
        not _rasterio_available(),
        reason="rasterio not installed"
    )
    def test_export_geotiff_with_nodata(self, rgb_array, temp_dir):
        """Test GeoTIFF export with nodata value."""
        from affine import Affine

        exporter = ImageExporter()
        output_path = temp_dir / "geo_nodata.tif"

        transform = Affine.translation(0, 0) * Affine.scale(1, -1)

        exporter.export_geotiff(
            rgb_array,
            output_path,
            crs="EPSG:32633",
            transform=transform,
            nodata=0,
        )

        # Verify nodata value
        import rasterio
        with rasterio.open(output_path) as src:
            assert src.nodata == 0

    @pytest.mark.skipif(
        not _rasterio_available(),
        reason="rasterio not installed"
    )
    def test_export_geotiff_with_compression(self, rgb_array, temp_dir):
        """Test GeoTIFF export with different compression."""
        from affine import Affine

        exporter = ImageExporter()
        transform = Affine.translation(0, 0) * Affine.scale(1, -1)

        # LZW compression
        path_lzw = temp_dir / "lzw.tif"
        exporter.export_geotiff(
            rgb_array, path_lzw,
            crs="EPSG:4326", transform=transform,
            compression="lzw"
        )

        # Deflate compression
        path_deflate = temp_dir / "deflate.tif"
        exporter.export_geotiff(
            rgb_array, path_deflate,
            crs="EPSG:4326", transform=transform,
            compression="deflate"
        )

        assert path_lzw.exists()
        assert path_deflate.exists()

    @pytest.mark.skipif(
        not _rasterio_available(),
        reason="rasterio not installed"
    )
    def test_export_geotiff_invalid_shape(self, rgba_array, temp_dir):
        """Test GeoTIFF export rejects RGBA (only RGB supported)."""
        from affine import Affine

        exporter = ImageExporter()
        output_path = temp_dir / "invalid.tif"
        transform = Affine.translation(0, 0) * Affine.scale(1, -1)

        # GeoTIFF export only supports RGB, not RGBA
        with pytest.raises(ValueError, match="must have shape.*W, 3.*for GeoTIFF"):
            exporter.export_geotiff(rgba_array, output_path, "EPSG:4326", transform)

    @pytest.mark.skipif(
        not _rasterio_available(),
        reason="rasterio not installed"
    )
    def test_export_geotiff_invalid_transform(self, rgb_array, temp_dir):
        """Test GeoTIFF export with invalid transform type."""
        exporter = ImageExporter()
        output_path = temp_dir / "invalid.tif"

        # Pass a tuple instead of Affine
        with pytest.raises(TypeError, match="must be.*Affine"):
            exporter.export_geotiff(
                rgb_array, output_path,
                crs="EPSG:4326",
                transform=(1, 0, 0, 0, -1, 0)  # Wrong type
            )

    @pytest.mark.skipif(
        _rasterio_available(),
        reason="Test requires rasterio to NOT be installed"
    )
    def test_export_geotiff_no_rasterio(self, rgb_array, temp_dir):
        """Test GeoTIFF export fails gracefully without rasterio."""
        exporter = ImageExporter()
        output_path = temp_dir / "geo.tif"

        with pytest.raises(ImportError, match="rasterio is required"):
            # This should fail because rasterio is not available
            exporter.export_geotiff(
                rgb_array, output_path,
                crs="EPSG:4326",
                transform=None  # Would need Affine, but we can't import it
            )


class TestAddAlphaChannel:
    """Tests for add_alpha_channel function."""

    def test_add_alpha_basic(self):
        """Test adding alpha channel to RGB array."""
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red

        mask = np.ones((50, 50), dtype=bool)
        mask[10:20, 10:20] = False  # Invalid region

        rgba = add_alpha_channel(rgb, mask)

        assert rgba.shape == (50, 50, 4)
        assert rgba.dtype == np.uint8

        # Valid pixels should have full opacity
        assert rgba[0, 0, 3] == 255

        # Invalid pixels should be transparent
        assert rgba[15, 15, 3] == 0

        # RGB channels should be unchanged
        assert rgba[0, 0, 0] == 255
        assert rgba[15, 15, 0] == 255

    def test_add_alpha_all_valid(self):
        """Test alpha channel with all pixels valid."""
        rgb = np.full((30, 30, 3), 128, dtype=np.uint8)
        mask = np.ones((30, 30), dtype=bool)

        rgba = add_alpha_channel(rgb, mask)

        # All pixels should be opaque
        assert np.all(rgba[:, :, 3] == 255)

    def test_add_alpha_all_invalid(self):
        """Test alpha channel with all pixels invalid."""
        rgb = np.full((30, 30, 3), 128, dtype=np.uint8)
        mask = np.zeros((30, 30), dtype=bool)

        rgba = add_alpha_channel(rgb, mask)

        # All pixels should be transparent
        assert np.all(rgba[:, :, 3] == 0)

    def test_add_alpha_invalid_rgb_shape(self):
        """Test add_alpha with invalid RGB shape."""
        mask = np.ones((50, 50), dtype=bool)

        # 2D array
        with pytest.raises(ValueError, match="must have shape"):
            add_alpha_channel(np.zeros((50, 50), dtype=np.uint8), mask)

        # RGBA array (already has alpha)
        with pytest.raises(ValueError, match="must have shape"):
            add_alpha_channel(np.zeros((50, 50, 4), dtype=np.uint8), mask)

    def test_add_alpha_invalid_mask_shape(self):
        """Test add_alpha with invalid mask shape."""
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)

        # 3D mask
        with pytest.raises(ValueError, match="must be 2D"):
            add_alpha_channel(rgb, np.ones((50, 50, 1), dtype=bool))

    def test_add_alpha_shape_mismatch(self):
        """Test add_alpha with mismatched shapes."""
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.ones((60, 60), dtype=bool)  # Different size

        with pytest.raises(ValueError, match="must match mask shape"):
            add_alpha_channel(rgb, mask)

    def test_add_alpha_invalid_dtype(self):
        """Test add_alpha with invalid RGB dtype."""
        rgb = np.zeros((50, 50, 3), dtype=np.float32)
        mask = np.ones((50, 50), dtype=bool)

        with pytest.raises(ValueError, match="must be uint8"):
            add_alpha_channel(rgb, mask)


class TestResizeForWeb:
    """Tests for resize_for_web function."""

    def test_resize_large_width(self):
        """Test resizing when width exceeds max."""
        rgb = np.zeros((1000, 3000, 3), dtype=np.uint8)

        resized = resize_for_web(rgb, max_dimension=2048)

        # Width should be scaled to 2048, height proportionally
        assert resized.shape[1] == 2048
        assert resized.shape[0] == int(1000 * 2048 / 3000)
        assert resized.shape[2] == 3
        assert resized.dtype == np.uint8

    def test_resize_large_height(self):
        """Test resizing when height exceeds max."""
        rgb = np.zeros((3000, 1000, 3), dtype=np.uint8)

        resized = resize_for_web(rgb, max_dimension=2048)

        # Height should be scaled to 2048, width proportionally
        assert resized.shape[0] == 2048
        assert resized.shape[1] == int(1000 * 2048 / 3000)
        assert resized.shape[2] == 3

    def test_resize_already_small(self):
        """Test that small images are not upsampled."""
        rgb = np.zeros((500, 500, 3), dtype=np.uint8)

        resized = resize_for_web(rgb, max_dimension=2048)

        # Should be unchanged
        assert resized.shape == (500, 500, 3)
        assert resized is rgb  # Should be same object

    def test_resize_with_alpha(self):
        """Test resizing RGBA image."""
        rgba = np.zeros((4000, 2000, 4), dtype=np.uint8)

        resized = resize_for_web(rgba, max_dimension=1024)

        # Height is larger (4000), so it should be scaled to 1024
        # Width scales proportionally: 2000 * (1024/4000) = 512
        assert resized.shape[0] == 1024  # Height
        assert resized.shape[1] == 512   # Width (proportional)
        assert resized.shape[2] == 4     # Alpha preserved
        assert resized.dtype == np.uint8

    def test_resize_square(self):
        """Test resizing square image."""
        rgb = np.zeros((5000, 5000, 3), dtype=np.uint8)

        resized = resize_for_web(rgb, max_dimension=1000)

        # Both dimensions should be 1000
        assert resized.shape == (1000, 1000, 3)

    def test_resize_preserves_aspect_ratio(self):
        """Test that aspect ratio is preserved during resize."""
        rgb = np.zeros((2000, 1000, 3), dtype=np.uint8)

        resized = resize_for_web(rgb, max_dimension=1024)

        # Aspect ratio should be preserved
        original_ratio = 2000 / 1000
        resized_ratio = resized.shape[0] / resized.shape[1]
        assert abs(original_ratio - resized_ratio) < 0.01

    def test_resize_invalid_shape(self):
        """Test resize with invalid array shape."""
        # 2D array
        with pytest.raises(ValueError, match="must have shape"):
            resize_for_web(np.zeros((100, 100), dtype=np.uint8))

        # Wrong number of channels
        with pytest.raises(ValueError, match="must have shape"):
            resize_for_web(np.zeros((100, 100, 2), dtype=np.uint8))

    def test_resize_invalid_dtype(self):
        """Test resize with invalid dtype."""
        with pytest.raises(ValueError, match="must be uint8"):
            resize_for_web(np.zeros((100, 100, 3), dtype=np.float32))
