"""
Integration tests for VIS-1.5 Report Visual Pipeline.

Tests the complete visual product generation pipeline including:
- Satellite imagery rendering
- Before/after comparison generation
- Detection overlay rendering
- Web and print optimization
- Caching functionality
- Manifest generation
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

import numpy as np
import pytest
from PIL import Image

from core.reporting.imagery import (
    ReportVisualPipeline,
    PipelineConfig,
    ImageManifest,
)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_bands():
    """Create sample band data."""
    # Create synthetic RGB bands (512x512)
    height, width = 512, 512

    # Red band - gradient
    red = np.linspace(0, 4000, height * width).reshape(height, width).astype(np.float32)

    # Green band - different gradient
    green = np.linspace(1000, 3000, height * width).reshape(height, width).astype(np.float32)

    # Blue band - uniform
    blue = np.full((height, width), 2000, dtype=np.float32)

    return {
        "B04": red,    # Sentinel-2 Red
        "B03": green,  # Sentinel-2 Green
        "B02": blue,   # Sentinel-2 Blue
    }


@pytest.fixture
def sample_detection_mask():
    """Create sample detection mask."""
    height, width = 512, 512
    mask = np.zeros((height, width), dtype=bool)

    # Add some detected regions
    mask[100:200, 100:300] = True  # Rectangle
    mask[300:400, 200:350] = True  # Another rectangle

    return mask


@pytest.fixture
def sample_confidence():
    """Create sample confidence scores."""
    height, width = 512, 512
    confidence = np.random.uniform(0.5, 1.0, (height, width)).astype(np.float32)
    return confidence


class TestPipelineConfig:
    """Test PipelineConfig validation."""

    def test_default_config(self):
        """Test default configuration."""
        config = PipelineConfig()

        assert config.cache_ttl_hours == 24
        assert config.web_max_width == 1200
        assert config.web_jpeg_quality == 85
        assert config.print_dpi == 300
        assert config.overlay_alpha == 0.6

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            cache_ttl_hours=12,
            web_max_width=800,
            print_dpi=600,
            overlay_alpha=0.8
        )

        assert config.cache_ttl_hours == 12
        assert config.web_max_width == 800
        assert config.print_dpi == 600
        assert config.overlay_alpha == 0.8

    def test_invalid_cache_ttl(self):
        """Test validation of cache TTL."""
        with pytest.raises(ValueError, match="cache_ttl_hours must be non-negative"):
            PipelineConfig(cache_ttl_hours=-1)

    def test_invalid_web_quality(self):
        """Test validation of web JPEG quality."""
        with pytest.raises(ValueError, match="web_jpeg_quality must be 1-100"):
            PipelineConfig(web_jpeg_quality=101)

    def test_invalid_overlay_alpha(self):
        """Test validation of overlay alpha."""
        with pytest.raises(ValueError, match="overlay_alpha must be 0-1"):
            PipelineConfig(overlay_alpha=1.5)


class TestImageManifest:
    """Test ImageManifest serialization."""

    def test_manifest_creation(self, temp_output_dir):
        """Test manifest creation."""
        manifest = ImageManifest(
            satellite_image=temp_output_dir / "satellite.png",
            detection_overlay=temp_output_dir / "overlay.png",
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
        )

        assert manifest.satellite_image == temp_output_dir / "satellite.png"
        assert manifest.detection_overlay == temp_output_dir / "overlay.png"
        assert manifest.generated_at == datetime(2024, 1, 15, 12, 0, 0)

    def test_manifest_to_dict(self, temp_output_dir):
        """Test manifest serialization to dict."""
        manifest = ImageManifest(
            satellite_image=temp_output_dir / "satellite.png",
            generated_at=datetime(2024, 1, 15, 12, 0, 0),
            cache_valid_until=datetime(2024, 1, 16, 12, 0, 0),
            metadata={"sensor": "sentinel2"}
        )

        data = manifest.to_dict()

        assert data["satellite_image"] == str(temp_output_dir / "satellite.png")
        assert data["generated_at"] == "2024-01-15T12:00:00"
        assert data["cache_valid_until"] == "2024-01-16T12:00:00"
        assert data["metadata"]["sensor"] == "sentinel2"

    def test_manifest_from_dict(self, temp_output_dir):
        """Test manifest deserialization from dict."""
        data = {
            "satellite_image": str(temp_output_dir / "satellite.png"),
            "generated_at": "2024-01-15T12:00:00",
            "cache_valid_until": "2024-01-16T12:00:00",
            "metadata": {"sensor": "sentinel2"},
            "before_image": None,
            "after_image": None,
            "detection_overlay": None,
            "side_by_side": None,
            "labeled_comparison": None,
            "animated_gif": None,
            "web_satellite_image": None,
            "web_detection_overlay": None,
            "print_satellite_image": None,
            "print_detection_overlay": None,
        }

        manifest = ImageManifest.from_dict(data)

        assert manifest.satellite_image == temp_output_dir / "satellite.png"
        assert manifest.generated_at == datetime(2024, 1, 15, 12, 0, 0)
        assert manifest.metadata["sensor"] == "sentinel2"


class TestReportVisualPipeline:
    """Test ReportVisualPipeline orchestration."""

    def test_pipeline_initialization(self, temp_output_dir):
        """Test pipeline initialization creates directories."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        assert pipeline.output_dir == temp_output_dir
        assert (temp_output_dir / ".cache").exists()
        assert (temp_output_dir / "web").exists()
        assert (temp_output_dir / "print").exists()

    def test_generate_satellite_imagery(self, temp_output_dir, sample_bands):
        """Test satellite imagery generation."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        output_path = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2",
            composite_name="true_color"
        )

        # Check file exists
        assert output_path.exists()
        assert output_path.suffix == ".png"

        # Check image can be loaded
        img = Image.open(output_path)
        assert img.mode == "RGB"
        assert img.size == (512, 512)

    def test_generate_before_after(self, temp_output_dir, sample_bands):
        """Test before/after comparison generation."""
        # Create slightly different "before" bands
        before_bands = {
            key: arr * 0.9 for key, arr in sample_bands.items()
        }

        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        before_path, after_path, side_by_side, labeled, gif = pipeline.generate_before_after(
            before_bands=before_bands,
            after_bands=sample_bands,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
            sensor="sentinel2"
        )

        # Check all files exist
        assert before_path.exists()
        assert after_path.exists()
        assert side_by_side.exists()
        assert labeled.exists()
        assert gif is not None and gif.exists()

        # Check side-by-side dimensions (should be wider)
        img = Image.open(side_by_side)
        assert img.width > 512  # Wider than single image
        assert img.height == 512

    def test_generate_detection_overlay(self, temp_output_dir, sample_bands, sample_detection_mask):
        """Test detection overlay generation."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        # First generate satellite image to use as background
        sat_path = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Load as array
        background = np.array(Image.open(sat_path))

        # Generate overlay
        overlay_path = pipeline.generate_detection_overlay(
            background=background,
            detection_mask=sample_detection_mask,
            overlay_type="flood"
        )

        # Check file exists
        assert overlay_path.exists()

        # Check image has alpha channel (RGBA)
        img = Image.open(overlay_path)
        assert img.mode == "RGBA"
        assert img.size == (512, 512)

    def test_generate_all_visuals_minimal(self, temp_output_dir, sample_bands):
        """Test complete visual generation with minimal inputs."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        manifest = pipeline.generate_all_visuals(
            bands=sample_bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15)
        )

        # Check satellite image generated
        assert manifest.satellite_image is not None
        assert manifest.satellite_image.exists()

        # Check web and print versions generated
        assert manifest.web_satellite_image is not None
        assert manifest.web_satellite_image.exists()
        assert manifest.print_satellite_image is not None
        assert manifest.print_satellite_image.exists()

        # Check metadata
        assert manifest.metadata["sensor"] == "sentinel2"
        assert manifest.metadata["event_date"] == "2024-01-15T00:00:00"
        assert manifest.metadata["has_before_after"] is False
        assert manifest.metadata["has_detection_overlay"] is False

    def test_generate_all_visuals_complete(
        self, temp_output_dir, sample_bands, sample_detection_mask, sample_confidence
    ):
        """Test complete visual generation with all inputs."""
        # Create before bands
        before_bands = {
            key: arr * 0.9 for key, arr in sample_bands.items()
        }

        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        manifest = pipeline.generate_all_visuals(
            bands=sample_bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15),
            detection_mask=sample_detection_mask,
            confidence=sample_confidence,
            overlay_type="flood",
            before_bands=before_bands,
            before_date=datetime(2024, 1, 1)
        )

        # Check all products generated
        assert manifest.satellite_image is not None and manifest.satellite_image.exists()
        assert manifest.before_image is not None and manifest.before_image.exists()
        assert manifest.after_image is not None and manifest.after_image.exists()
        assert manifest.side_by_side is not None and manifest.side_by_side.exists()
        assert manifest.labeled_comparison is not None and manifest.labeled_comparison.exists()
        assert manifest.detection_overlay is not None and manifest.detection_overlay.exists()

        # Check web/print versions
        assert manifest.web_satellite_image is not None and manifest.web_satellite_image.exists()
        assert manifest.web_detection_overlay is not None and manifest.web_detection_overlay.exists()
        assert manifest.print_satellite_image is not None and manifest.print_satellite_image.exists()
        assert manifest.print_detection_overlay is not None and manifest.print_detection_overlay.exists()

        # Check metadata
        assert manifest.metadata["has_before_after"] is True
        assert manifest.metadata["has_detection_overlay"] is True
        assert manifest.metadata["overlay_type"] == "flood"

    def test_web_optimization(self, temp_output_dir, sample_bands):
        """Test web image optimization."""
        pipeline = ReportVisualPipeline(
            output_dir=temp_output_dir,
            config=PipelineConfig(web_max_width=600, web_jpeg_quality=75)
        )

        # Generate satellite image
        sat_path = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Get web-optimized version
        web_path = pipeline.get_web_optimized(sat_path)

        assert web_path.exists()
        assert web_path.suffix == ".jpg"

        # Check image is resized
        img = Image.open(web_path)
        assert img.width == 600  # Resized to max width
        assert img.height == 600  # Maintains aspect ratio

        # Check file is compressed
        web_size = web_path.stat().st_size
        original_size = sat_path.stat().st_size
        assert web_size < original_size  # Web version should be smaller

    def test_print_optimization(self, temp_output_dir, sample_bands):
        """Test print image optimization."""
        pipeline = ReportVisualPipeline(
            output_dir=temp_output_dir,
            config=PipelineConfig(print_dpi=300)
        )

        # Generate satellite image
        sat_path = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Get print-optimized version
        print_path = pipeline.get_print_optimized(sat_path)

        assert print_path.exists()
        assert print_path.suffix == ".png"

        # Check DPI metadata
        img = Image.open(print_path)
        assert img.info.get("dpi") == (300, 300)

    def test_caching(self, temp_output_dir, sample_bands):
        """Test caching prevents regeneration."""
        pipeline = ReportVisualPipeline(
            output_dir=temp_output_dir,
            config=PipelineConfig(cache_ttl_hours=24)
        )

        # Generate image first time
        path1 = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Record modification time
        mtime1 = path1.stat().st_mtime

        # Generate again - should use cache
        path2 = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Should return same path
        assert path1 == path2

        # File should not be regenerated
        mtime2 = path2.stat().st_mtime
        assert mtime1 == mtime2

    def test_cache_disabled(self, temp_output_dir, sample_bands):
        """Test caching can be disabled."""
        pipeline = ReportVisualPipeline(
            output_dir=temp_output_dir,
            config=PipelineConfig(cache_ttl_hours=0)  # Disable caching
        )

        # Generate image twice - should regenerate each time
        path1 = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        path2 = pipeline.generate_satellite_imagery(
            bands=sample_bands,
            sensor="sentinel2"
        )

        # Paths should be different (new cache keys each time)
        # Both files should exist
        assert path1.exists()
        assert path2.exists()

    def test_manifest_persistence(self, temp_output_dir, sample_bands):
        """Test manifest is saved to and loaded from disk."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        # Generate visuals
        manifest = pipeline.generate_all_visuals(
            bands=sample_bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15)
        )

        # Check manifest.json exists
        manifest_file = temp_output_dir / "manifest.json"
        assert manifest_file.exists()

        # Load manifest
        loaded_manifest = pipeline.load_manifest()
        assert loaded_manifest is not None
        assert loaded_manifest.satellite_image == manifest.satellite_image
        assert loaded_manifest.metadata["sensor"] == "sentinel2"

    def test_manifest_cache_expiry(self, temp_output_dir, sample_bands):
        """Test manifest cache expiry."""
        pipeline = ReportVisualPipeline(
            output_dir=temp_output_dir,
            config=PipelineConfig(cache_ttl_hours=1)
        )

        # Generate visuals
        manifest = pipeline.generate_all_visuals(
            bands=sample_bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15)
        )

        # Manually expire the manifest
        manifest.cache_valid_until = datetime.now() - timedelta(hours=1)
        pipeline._save_manifest(manifest)

        # Try to load - should return None because expired
        loaded = pipeline.load_manifest()
        assert loaded is None

    def test_error_handling_missing_file(self, temp_output_dir):
        """Test error handling for missing source file."""
        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        with pytest.raises(FileNotFoundError):
            pipeline.get_web_optimized(temp_output_dir / "nonexistent.png")


class TestInteractiveReportIntegration:
    """Test integration with InteractiveReportGenerator."""

    def test_manifest_integration(self, temp_output_dir, sample_bands):
        """Test that manifest integrates with InteractiveReportGenerator."""
        # This is a smoke test - full integration would require
        # a complete report data structure

        pipeline = ReportVisualPipeline(output_dir=temp_output_dir)

        manifest = pipeline.generate_all_visuals(
            bands=sample_bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15)
        )

        # Verify manifest has all required attributes
        assert hasattr(manifest, "satellite_image")
        assert hasattr(manifest, "before_image")
        assert hasattr(manifest, "after_image")
        assert hasattr(manifest, "detection_overlay")
        assert hasattr(manifest, "web_satellite_image")

        # Verify paths can be converted to strings for templates
        assert str(manifest.satellite_image) is not None
        if manifest.web_satellite_image:
            assert str(manifest.web_satellite_image) is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
