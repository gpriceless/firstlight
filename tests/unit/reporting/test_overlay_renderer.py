"""
Unit tests for detection overlay renderer (VIS-1.3 Task 10).

Tests all overlay rendering features:
- Flood and fire detection overlays
- Confidence-based transparency
- Polygon outlines
- Scale bar and north arrow
- B&W pattern fills for accessibility
- Legend generation
"""

import numpy as np
import pytest

from core.reporting.imagery.overlay import (
    DetectionOverlay,
    OverlayConfig,
    OverlayResult,
)
from core.reporting.imagery.color_scales import get_flood_palette, get_fire_palette


class TestOverlayConfig:
    """Test OverlayConfig validation and defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OverlayConfig()
        assert config.overlay_type == "flood"
        assert config.confidence_threshold == 0.3
        assert config.alpha_base == 0.6
        assert config.use_confidence_alpha is True
        assert config.show_outline is True
        assert config.show_scale_bar is True
        assert config.show_north_arrow is True
        assert config.use_bw_patterns is False

    def test_invalid_overlay_type(self):
        """Test validation rejects invalid overlay types."""
        with pytest.raises(ValueError, match="Invalid overlay_type"):
            OverlayConfig(overlay_type="invalid")

    def test_invalid_confidence_threshold(self):
        """Test validation rejects invalid confidence threshold."""
        with pytest.raises(ValueError, match="confidence_threshold must be 0-1"):
            OverlayConfig(confidence_threshold=1.5)

    def test_invalid_alpha_base(self):
        """Test validation rejects invalid alpha base."""
        with pytest.raises(ValueError, match="alpha_base must be 0-1"):
            OverlayConfig(alpha_base=-0.1)

    def test_invalid_outline_width(self):
        """Test validation rejects negative outline width."""
        with pytest.raises(ValueError, match="outline_width must be non-negative"):
            OverlayConfig(outline_width=-1)


class TestOverlayResult:
    """Test OverlayResult dataclass validation."""

    def test_valid_result(self):
        """Test valid overlay result."""
        composite = np.zeros((100, 100, 4), dtype=np.uint8)
        legend = np.zeros((50, 50, 4), dtype=np.uint8)
        result = OverlayResult(
            composite_image=composite, legend_image=legend, metadata={}
        )
        assert result.composite_image.shape == (100, 100, 4)
        assert result.legend_image.shape == (50, 50, 4)

    def test_invalid_composite_shape(self):
        """Test validation rejects invalid composite shape."""
        composite = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
        legend = np.zeros((50, 50, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="composite_image must be 3D array"):
            OverlayResult(composite_image=composite, legend_image=legend)

    def test_invalid_legend_shape(self):
        """Test validation rejects invalid legend shape."""
        composite = np.zeros((100, 100, 4), dtype=np.uint8)
        legend = np.zeros((50, 50), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(ValueError, match="legend_image must be 3D array"):
            OverlayResult(composite_image=composite, legend_image=legend)


class TestDetectionOverlay:
    """Test DetectionOverlay rendering functionality."""

    @pytest.fixture
    def background(self):
        """Create synthetic RGB background."""
        bg = np.zeros((256, 256, 3), dtype=np.uint8)
        bg[:, :, 0] = 100  # Red
        bg[:, :, 1] = 120  # Green
        bg[:, :, 2] = 140  # Blue
        return bg

    @pytest.fixture
    def detection_mask(self):
        """Create synthetic detection mask."""
        mask = np.zeros((256, 256), dtype=bool)
        mask[50:200, 50:200] = True  # Square detection region
        return mask

    @pytest.fixture
    def confidence(self):
        """Create synthetic confidence array."""
        conf = np.zeros((256, 256), dtype=np.float32)
        conf[50:200, 50:200] = 0.8  # High confidence
        return conf

    @pytest.fixture
    def severity(self):
        """Create synthetic severity array."""
        sev = np.zeros((256, 256), dtype=np.float32)
        sev[50:100, 50:200] = 0.3  # Low severity
        sev[100:150, 50:200] = 0.6  # Medium severity
        sev[150:200, 50:200] = 0.9  # High severity
        return sev

    def test_flood_overlay_basic(self, background, detection_mask, confidence):
        """Test basic flood overlay rendering."""
        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence)

        assert result.composite_image.shape == (256, 256, 4)
        assert result.composite_image.dtype == np.uint8
        assert result.legend_image.shape[2] == 4
        assert "overlay_type" in result.metadata
        assert result.metadata["overlay_type"] == "flood"

    def test_fire_overlay_basic(self, background, detection_mask, confidence, severity):
        """Test basic fire overlay rendering."""
        config = OverlayConfig(overlay_type="fire", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence, severity=severity)

        assert result.composite_image.shape == (256, 256, 4)
        assert result.composite_image.dtype == np.uint8
        assert result.metadata["overlay_type"] == "fire"

    def test_confidence_based_transparency(self, background):
        """Test that confidence affects transparency."""
        # Create mask with two regions: low and high confidence
        mask = np.ones((256, 256), dtype=bool)
        confidence = np.zeros((256, 256), dtype=np.float32)
        confidence[:, :128] = 0.3  # Low confidence (left)
        confidence[:, 128:] = 0.9  # High confidence (right)

        # Use dark background to see transparency effect
        dark_bg = np.ones((256, 256, 3), dtype=np.uint8) * 50

        config = OverlayConfig(
            overlay_type="flood",
            use_confidence_alpha=True,
            show_scale_bar=False,
            show_north_arrow=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(dark_bg, mask, confidence=confidence)

        # Check that high confidence region has more overlay color (blue for flood)
        blue_low = result.composite_image[:, :128, 2].mean()
        blue_high = result.composite_image[:, 128:, 2].mean()
        assert blue_high > blue_low, "High confidence should show more overlay color"

    def test_outline_rendering(self, background, detection_mask, confidence):
        """Test polygon outline rendering."""
        config = OverlayConfig(
            overlay_type="flood",
            show_outline=True,
            outline_width=2,
            show_scale_bar=False,
            show_north_arrow=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence)

        # Outline should be visible at edges of detection region
        # Check a few edge pixels have high alpha (outline is fully opaque)
        assert result.composite_image[50, 50, 3] == 255  # Top-left corner
        assert result.composite_image.shape == (256, 256, 4)

    def test_no_detections(self, background):
        """Test handling of empty detection mask."""
        mask = np.zeros((256, 256), dtype=bool)
        confidence = np.zeros((256, 256), dtype=np.float32)

        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, mask, confidence=confidence)

        assert result.metadata["detection_area_pct"] == 0.0
        assert result.metadata["confidence_range"] == (0.0, 0.0)

    def test_confidence_threshold(self, background, detection_mask):
        """Test confidence threshold filtering."""
        # Create confidence below threshold
        low_confidence = np.ones((256, 256), dtype=np.float32) * 0.2

        config = OverlayConfig(
            overlay_type="flood",
            confidence_threshold=0.3,
            show_scale_bar=False,
            show_north_arrow=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=low_confidence)

        # All detections should be filtered out
        assert result.metadata["detection_area_pct"] == 0.0

    def test_scale_bar_rendering(self, background, detection_mask, confidence):
        """Test scale bar component."""
        config = OverlayConfig(
            overlay_type="flood",
            show_scale_bar=True,
            pixel_size_meters=10.0,
            show_north_arrow=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence)

        # Scale bar adds visual elements, can't easily verify without image inspection
        # Just verify rendering completes and shape is correct
        assert result.composite_image.shape == (256, 256, 4)

    def test_north_arrow_rendering(self, background, detection_mask, confidence):
        """Test north arrow component."""
        config = OverlayConfig(
            overlay_type="flood",
            show_north_arrow=True,
            show_scale_bar=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence)

        # North arrow adds visual elements
        assert result.composite_image.shape == (256, 256, 4)

    def test_bw_patterns(self, background, detection_mask, confidence, severity):
        """Test B&W pattern fills for accessibility."""
        config = OverlayConfig(
            overlay_type="flood",
            use_bw_patterns=True,
            show_scale_bar=False,
            show_north_arrow=False
        )
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence, severity=severity)

        # Patterns should be applied (darkens some pixels)
        assert result.composite_image.shape == (256, 256, 4)

    def test_render_with_legend(self, background, detection_mask, confidence):
        """Test render_with_legend convenience method."""
        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render_with_legend(
            background, detection_mask, confidence=confidence, legend_position="top-right"
        )

        # Result should have legend composited onto image
        assert result.shape == (256, 256, 4)
        assert result.dtype == np.uint8

    def test_invalid_background_shape(self, detection_mask, confidence):
        """Test validation of background shape."""
        invalid_bg = np.zeros((256, 256), dtype=np.uint8)  # 2D instead of 3D
        config = OverlayConfig(overlay_type="flood")
        overlay = DetectionOverlay(config=config)

        with pytest.raises(ValueError, match="background must be 3D array"):
            overlay.render(invalid_bg, detection_mask, confidence=confidence)

    def test_shape_mismatch(self, background):
        """Test validation of shape mismatches."""
        mask = np.zeros((128, 128), dtype=bool)  # Different size
        confidence = np.zeros((256, 256), dtype=np.float32)

        config = OverlayConfig(overlay_type="flood")
        overlay = DetectionOverlay(config=config)

        with pytest.raises(ValueError, match="doesn't match"):
            overlay.render(background, mask, confidence=confidence)

    def test_float_background_conversion(self, detection_mask, confidence):
        """Test float background conversion to uint8."""
        float_bg = np.random.rand(256, 256, 3).astype(np.float32)  # 0-1 range
        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(float_bg, detection_mask, confidence=confidence)

        assert result.composite_image.dtype == np.uint8

    def test_rgba_background_handling(self, detection_mask, confidence):
        """Test RGBA background is converted to RGB."""
        rgba_bg = np.zeros((256, 256, 4), dtype=np.uint8)
        rgba_bg[:, :, :3] = 128  # Gray RGB
        rgba_bg[:, :, 3] = 255  # Full alpha

        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(rgba_bg, detection_mask, confidence=confidence)

        assert result.composite_image.shape == (256, 256, 4)

    def test_metadata_completeness(self, background, detection_mask, confidence):
        """Test that metadata contains all expected fields."""
        config = OverlayConfig(overlay_type="flood", show_scale_bar=False, show_north_arrow=False)
        overlay = DetectionOverlay(config=config)
        result = overlay.render(background, detection_mask, confidence=confidence)

        expected_keys = [
            "overlay_type",
            "detection_area_pct",
            "confidence_range",
            "severity_range",
            "confidence_threshold",
        ]
        for key in expected_keys:
            assert key in result.metadata, f"Missing metadata key: {key}"

    def test_legend_generation(self):
        """Test that legend is properly generated."""
        config = OverlayConfig(overlay_type="flood")
        overlay = DetectionOverlay(config=config)

        # Create minimal data to trigger legend generation
        bg = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        conf = np.ones((100, 100), dtype=np.float32) * 0.8

        result = overlay.render(bg, mask, confidence=conf)

        # Legend should be non-empty RGBA image
        assert result.legend_image.size > 0
        assert result.legend_image.shape[2] == 4
        assert result.legend_image.dtype == np.uint8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
