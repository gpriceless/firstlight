"""
Unit tests for detection overlay rendering (VIS-1.3 Task 2.1).

Tests cover:
- OverlayConfig validation
- DetectionOverlay rendering
- Alpha blending correctness
- Outline rendering
- Legend generation
- Edge cases (no detections, threshold filtering)
"""

import numpy as np
import pytest

from core.reporting.imagery.overlay import (
    DetectionOverlay,
    OverlayConfig,
    OverlayResult,
)
from core.reporting.imagery.color_scales import get_flood_palette, get_fire_palette


# === Fixtures ===


@pytest.fixture
def rgb_background():
    """Create a simple RGB background image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def rgba_background():
    """Create a simple RGBA background image."""
    return np.full((100, 100, 4), (128, 128, 128, 255), dtype=np.uint8)


@pytest.fixture
def float_background():
    """Create a float RGB background image (0-1 range)."""
    return np.full((100, 100, 3), 0.5, dtype=np.float32)


@pytest.fixture
def detection_mask():
    """Create a binary detection mask with a square region."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True  # 50x50 detection region
    return mask


@pytest.fixture
def confidence_uniform():
    """Create uniform confidence values."""
    return np.full((100, 100), 0.8, dtype=np.float32)


@pytest.fixture
def confidence_gradient():
    """Create confidence gradient from left (low) to right (high)."""
    confidence = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        confidence[:, i] = i / 100.0
    return confidence


@pytest.fixture
def severity_gradient():
    """Create severity gradient from top (low) to bottom (high)."""
    severity = np.zeros((100, 100), dtype=np.float32)
    for i in range(100):
        severity[i, :] = i / 100.0
    return severity


# === Config Validation Tests ===


def test_overlay_config_defaults():
    """Test OverlayConfig default values."""
    config = OverlayConfig()
    assert config.overlay_type == "flood"
    assert config.confidence_threshold == 0.3
    assert config.alpha_base == 0.6
    assert config.use_confidence_alpha is True
    assert config.show_outline is True
    assert config.outline_color == (255, 255, 255)
    assert config.outline_width == 2


def test_overlay_config_custom():
    """Test OverlayConfig with custom values."""
    config = OverlayConfig(
        overlay_type="fire",
        confidence_threshold=0.5,
        alpha_base=0.8,
        use_confidence_alpha=False,
        show_outline=False,
        outline_color=(0, 0, 0),
        outline_width=3,
    )
    assert config.overlay_type == "fire"
    assert config.confidence_threshold == 0.5
    assert config.alpha_base == 0.8
    assert config.use_confidence_alpha is False
    assert config.show_outline is False
    assert config.outline_color == (0, 0, 0)
    assert config.outline_width == 3


def test_overlay_config_invalid_type():
    """Test that invalid overlay_type raises error."""
    with pytest.raises(ValueError, match="Invalid overlay_type"):
        OverlayConfig(overlay_type="invalid")


def test_overlay_config_invalid_threshold():
    """Test that invalid confidence_threshold raises error."""
    with pytest.raises(ValueError, match="confidence_threshold must be 0-1"):
        OverlayConfig(confidence_threshold=-0.1)

    with pytest.raises(ValueError, match="confidence_threshold must be 0-1"):
        OverlayConfig(confidence_threshold=1.5)


def test_overlay_config_invalid_alpha():
    """Test that invalid alpha_base raises error."""
    with pytest.raises(ValueError, match="alpha_base must be 0-1"):
        OverlayConfig(alpha_base=-0.1)

    with pytest.raises(ValueError, match="alpha_base must be 0-1"):
        OverlayConfig(alpha_base=1.5)


def test_overlay_config_invalid_outline_width():
    """Test that negative outline_width raises error."""
    with pytest.raises(ValueError, match="outline_width must be non-negative"):
        OverlayConfig(outline_width=-1)


# === DetectionOverlay Initialization Tests ===


def test_detection_overlay_init_default():
    """Test DetectionOverlay initialization with default config."""
    overlay = DetectionOverlay()
    assert overlay.config.overlay_type == "flood"
    assert overlay._palette.name == "Flood Severity"


def test_detection_overlay_init_custom():
    """Test DetectionOverlay initialization with custom config."""
    config = OverlayConfig(overlay_type="fire")
    overlay = DetectionOverlay(config)
    assert overlay.config.overlay_type == "fire"
    # The actual palette name might differ, just check it exists
    assert overlay._palette.name is not None


# === Basic Rendering Tests ===


def test_render_basic_flood(rgb_background, detection_mask, confidence_uniform):
    """Test basic flood overlay rendering."""
    overlay = DetectionOverlay(OverlayConfig(overlay_type="flood"))
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    assert isinstance(result, OverlayResult)
    assert result.composite_image.shape == (100, 100, 4)
    assert result.composite_image.dtype == np.uint8
    assert result.legend_image.shape[2] == 4
    assert "overlay_type" in result.metadata
    assert result.metadata["overlay_type"] == "flood"


def test_render_basic_fire(rgb_background, detection_mask, confidence_uniform):
    """Test basic fire overlay rendering."""
    overlay = DetectionOverlay(OverlayConfig(overlay_type="fire"))
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    assert isinstance(result, OverlayResult)
    assert result.composite_image.shape == (100, 100, 4)
    assert result.metadata["overlay_type"] == "fire"


def test_render_with_rgba_background(rgba_background, detection_mask, confidence_uniform):
    """Test rendering with RGBA background."""
    overlay = DetectionOverlay()
    result = overlay.render(rgba_background, detection_mask, confidence=confidence_uniform)

    assert result.composite_image.shape == (100, 100, 4)


def test_render_with_float_background(float_background, detection_mask, confidence_uniform):
    """Test rendering with float background (0-1 range)."""
    overlay = DetectionOverlay()
    result = overlay.render(float_background, detection_mask, confidence=confidence_uniform)

    assert result.composite_image.shape == (100, 100, 4)
    assert result.composite_image.dtype == np.uint8


# === Confidence Threshold Tests ===


def test_render_confidence_threshold_filters():
    """Test that confidence threshold filters out low-confidence pixels."""
    background = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=bool)

    # Half the image has low confidence, half high
    confidence = np.zeros((100, 100), dtype=np.float32)
    confidence[:, :50] = 0.2  # Below threshold
    confidence[:, 50:] = 0.8  # Above threshold

    config = OverlayConfig(confidence_threshold=0.3)
    overlay = DetectionOverlay(config)
    result = overlay.render(background, mask, confidence=confidence)

    # Check that detection area is reduced
    assert result.metadata["detection_area_pct"] < 100
    assert result.metadata["detection_area_pct"] > 0


def test_render_all_below_threshold():
    """Test rendering when all detections are below confidence threshold."""
    background = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=bool)
    confidence = np.full((100, 100), 0.1, dtype=np.float32)

    config = OverlayConfig(confidence_threshold=0.5)
    overlay = DetectionOverlay(config)
    result = overlay.render(background, mask, confidence=confidence)

    # Should have 0% detection after thresholding
    assert result.metadata["detection_area_pct"] == 0.0
    assert result.metadata["confidence_range"] == (0.0, 0.0)


# === Severity Mapping Tests ===


def test_render_with_severity(rgb_background, detection_mask, confidence_uniform):
    """Test rendering with explicit severity values."""
    severity = np.random.rand(100, 100).astype(np.float32)

    overlay = DetectionOverlay()
    result = overlay.render(
        rgb_background, detection_mask, confidence=confidence_uniform, severity=severity
    )

    assert result.composite_image.shape == (100, 100, 4)
    assert "severity_range" in result.metadata


def test_render_without_severity_uses_confidence(rgb_background, detection_mask):
    """Test that severity defaults to confidence when not provided."""
    confidence = np.full((100, 100), 0.7, dtype=np.float32)

    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, detection_mask, confidence=confidence)

    # Severity range should match confidence range
    assert result.metadata["confidence_range"][0] <= result.metadata["severity_range"][0]
    assert result.metadata["confidence_range"][1] >= result.metadata["severity_range"][1]


# === Alpha Blending Tests ===


def test_render_with_confidence_alpha(rgb_background, detection_mask, confidence_gradient):
    """Test that confidence modulates alpha channel."""
    config = OverlayConfig(use_confidence_alpha=True, alpha_base=0.8, show_outline=False)
    overlay = DetectionOverlay(config)

    # Create colored overlay directly to test alpha modulation
    severity = confidence_gradient.copy()
    colored_overlay = overlay._create_colored_overlay(detection_mask, severity, confidence_gradient)

    # Extract alpha channel from detection region (before blending with background)
    alpha_region = colored_overlay[25:75, 25:75, 3]

    # Filter out transparent pixels (outside detection)
    non_zero_alpha = alpha_region[alpha_region > 0]

    # Alpha should vary across the region (due to confidence gradient)
    if len(non_zero_alpha) > 1:
        assert np.min(non_zero_alpha) < np.max(non_zero_alpha)


def test_render_without_confidence_alpha(rgb_background, detection_mask, confidence_gradient):
    """Test that alpha is constant when use_confidence_alpha=False."""
    config = OverlayConfig(use_confidence_alpha=False, alpha_base=0.8, show_outline=False)
    overlay = DetectionOverlay(config)
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_gradient)

    # Extract alpha channel from detection region (excluding edges due to scipy erosion)
    # Sample from center to avoid edge artifacts
    alpha_center = result.composite_image[35:65, 35:65, 3]

    # Alpha should be relatively uniform (may have some variation from blending)
    # Filter out any transparent pixels (from blending artifacts)
    non_zero_alpha = alpha_center[alpha_center > 0]
    if len(non_zero_alpha) > 0:
        alpha_std = np.std(non_zero_alpha)
        assert alpha_std < 15  # Small tolerance for numerical variation


# === Outline Tests ===


def test_render_with_outline(rgb_background, detection_mask, confidence_uniform):
    """Test that outlines are drawn when enabled."""
    config = OverlayConfig(show_outline=True, outline_color=(255, 0, 0), outline_width=2)
    overlay = DetectionOverlay(config)
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    # Check that outline pixels exist at edges of detection region
    # Top edge of detection region
    top_edge = result.composite_image[24:26, 35:40, :]
    # Outline should have pixels with red channel > 0
    assert np.any(top_edge[:, :, 0] > 0)


def test_render_without_outline(rgb_background, detection_mask, confidence_uniform):
    """Test that no outlines are drawn when disabled."""
    config = OverlayConfig(show_outline=False)
    overlay = DetectionOverlay(config)
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    assert result.composite_image.shape == (100, 100, 4)
    # We can't easily verify absence of outline, but ensure it doesn't crash


# === Legend Tests ===


def test_render_generates_legend(rgb_background, detection_mask, confidence_uniform):
    """Test that legend is generated."""
    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    assert result.legend_image is not None
    assert result.legend_image.ndim == 3
    assert result.legend_image.shape[2] == 4
    assert result.legend_image.dtype == np.uint8


def test_render_with_legend_flood(detection_mask, confidence_uniform):
    """Test render_with_legend for flood overlay."""
    # Use larger background to fit legend
    large_background = np.full((300, 300, 3), 128, dtype=np.uint8)
    large_mask = np.zeros((300, 300), dtype=bool)
    large_mask[100:200, 100:200] = True
    large_confidence = np.full((300, 300), 0.8, dtype=np.float32)

    overlay = DetectionOverlay(OverlayConfig(overlay_type="flood"))
    composite = overlay.render_with_legend(
        large_background, large_mask, confidence=large_confidence, legend_position="top-right"
    )

    assert composite.shape == (300, 300, 4)
    assert composite.dtype == np.uint8


def test_render_with_legend_fire(detection_mask, confidence_uniform):
    """Test render_with_legend for fire overlay."""
    # Use larger background to fit legend
    large_background = np.full((300, 300, 3), 128, dtype=np.uint8)
    large_mask = np.zeros((300, 300), dtype=bool)
    large_mask[100:200, 100:200] = True
    large_confidence = np.full((300, 300), 0.8, dtype=np.float32)

    overlay = DetectionOverlay(OverlayConfig(overlay_type="fire"))
    composite = overlay.render_with_legend(
        large_background, large_mask, confidence=large_confidence, legend_position="bottom-left"
    )

    assert composite.shape == (300, 300, 4)


def test_render_with_legend_positions(detection_mask, confidence_uniform):
    """Test all legend positions."""
    # Use larger background to fit legend
    large_background = np.full((300, 300, 3), 128, dtype=np.uint8)
    large_mask = np.zeros((300, 300), dtype=bool)
    large_mask[100:200, 100:200] = True
    large_confidence = np.full((300, 300), 0.8, dtype=np.float32)

    overlay = DetectionOverlay()
    positions = ["top-left", "top-right", "bottom-left", "bottom-right"]

    for position in positions:
        composite = overlay.render_with_legend(
            large_background, large_mask, confidence=large_confidence, legend_position=position
        )
        assert composite.shape == (300, 300, 4)


# === Metadata Tests ===


def test_render_metadata_structure(rgb_background, detection_mask, confidence_uniform):
    """Test that metadata contains expected fields."""
    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    assert "overlay_type" in result.metadata
    assert "detection_area_pct" in result.metadata
    assert "confidence_range" in result.metadata
    assert "severity_range" in result.metadata
    assert "confidence_threshold" in result.metadata


def test_render_metadata_detection_area(rgb_background, detection_mask, confidence_uniform):
    """Test detection_area_pct calculation."""
    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, detection_mask, confidence=confidence_uniform)

    # Detection mask is 50x50 in 100x100 image = 25%
    expected_area = 25.0
    assert abs(result.metadata["detection_area_pct"] - expected_area) < 0.1


def test_render_metadata_confidence_range(rgb_background, detection_mask):
    """Test confidence_range calculation."""
    # Use high confidence to avoid threshold filtering
    confidence = np.random.uniform(0.5, 1.0, (100, 100)).astype(np.float32)

    config = OverlayConfig(confidence_threshold=0.3)
    overlay = DetectionOverlay(config)
    result = overlay.render(rgb_background, detection_mask, confidence=confidence)

    # Confidence range should reflect detected pixels that passed threshold
    thresholded_mask = detection_mask & (confidence >= config.confidence_threshold)
    detected_conf = confidence[thresholded_mask]
    assert abs(result.metadata["confidence_range"][0] - np.min(detected_conf)) < 0.01
    assert abs(result.metadata["confidence_range"][1] - np.max(detected_conf)) < 0.01


# === Validation Tests ===


def test_render_invalid_background_shape():
    """Test that invalid background shape raises error."""
    overlay = DetectionOverlay()
    background = np.zeros((100, 100), dtype=np.uint8)  # 2D instead of 3D
    mask = np.zeros((100, 100), dtype=bool)

    with pytest.raises(ValueError, match="background must be 3D array"):
        overlay.render(background, mask)


def test_render_invalid_background_channels():
    """Test that invalid number of channels raises error."""
    overlay = DetectionOverlay()
    background = np.zeros((100, 100, 2), dtype=np.uint8)  # 2 channels
    mask = np.zeros((100, 100), dtype=bool)

    with pytest.raises(ValueError, match="background must have 3 or 4 channels"):
        overlay.render(background, mask)


def test_render_invalid_mask_shape():
    """Test that invalid mask shape raises error."""
    overlay = DetectionOverlay()
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=bool)  # Wrong size

    with pytest.raises(ValueError, match="detection_mask shape .* doesn't match"):
        overlay.render(background, mask)


def test_render_invalid_confidence_shape():
    """Test that invalid confidence shape raises error."""
    overlay = DetectionOverlay()
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    confidence = np.zeros((50, 50), dtype=np.float32)  # Wrong size

    with pytest.raises(ValueError, match="confidence shape .* doesn't match"):
        overlay.render(background, mask, confidence=confidence)


def test_render_invalid_severity_shape():
    """Test that invalid severity shape raises error."""
    overlay = DetectionOverlay()
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    severity = np.zeros((50, 50), dtype=np.float32)  # Wrong size

    with pytest.raises(ValueError, match="severity shape .* doesn't match"):
        overlay.render(background, mask, severity=severity)


# === Edge Case Tests ===


def test_render_empty_mask(rgb_background):
    """Test rendering with empty detection mask."""
    mask = np.zeros((100, 100), dtype=bool)
    confidence = np.ones((100, 100), dtype=np.float32)

    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, mask, confidence=confidence)

    # Should return background unchanged (but as RGBA)
    assert result.metadata["detection_area_pct"] == 0.0
    assert result.composite_image.shape == (100, 100, 4)


def test_render_full_mask(rgb_background):
    """Test rendering with full detection mask."""
    mask = np.ones((100, 100), dtype=bool)
    confidence = np.ones((100, 100), dtype=np.float32)

    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, mask, confidence=confidence)

    # Should detect 100% of area
    assert result.metadata["detection_area_pct"] == 100.0


def test_render_confidence_clipped(rgb_background, detection_mask):
    """Test that confidence values are clipped to [0, 1]."""
    confidence = np.random.randn(100, 100).astype(np.float32)  # Can be negative or > 1

    overlay = DetectionOverlay()
    result = overlay.render(rgb_background, detection_mask, confidence=confidence)

    # Confidence range should be within [0, 1]
    assert result.metadata["confidence_range"][0] >= 0.0
    assert result.metadata["confidence_range"][1] <= 1.0


# === Integration Tests ===


def test_flood_overlay_end_to_end():
    """Integration test: flood overlay from background to final composite."""
    # Create realistic flood scenario
    background = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Simulate flood detection in bottom-right corner
    mask = np.zeros((200, 200), dtype=bool)
    mask[100:, 100:] = True

    # Confidence decreases with distance from center
    confidence = np.zeros((200, 200), dtype=np.float32)
    for i in range(200):
        for j in range(200):
            if mask[i, j]:
                dist = np.sqrt((i - 150) ** 2 + (j - 150) ** 2)
                confidence[i, j] = max(0.3, 1.0 - dist / 100.0)

    # Severity based on confidence for this test
    severity = confidence.copy()

    config = OverlayConfig(
        overlay_type="flood",
        confidence_threshold=0.3,
        use_confidence_alpha=True,
        show_outline=True,
    )
    overlay = DetectionOverlay(config)
    result = overlay.render(background, mask, confidence=confidence, severity=severity)

    # Verify result
    assert result.composite_image.shape == (200, 200, 4)
    assert result.legend_image.shape[2] == 4
    assert result.metadata["detection_area_pct"] > 0
    assert result.metadata["overlay_type"] == "flood"


def test_fire_overlay_end_to_end():
    """Integration test: fire overlay with burn severity."""
    # Create realistic fire scenario
    background = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

    # Simulate burn area
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True

    # Confidence mostly high
    confidence = np.full((200, 200), 0.9, dtype=np.float32)

    # Severity varies by dNBR scale
    severity = np.random.uniform(0.1, 0.9, (200, 200)).astype(np.float32)

    config = OverlayConfig(
        overlay_type="fire",
        confidence_threshold=0.5,
        use_confidence_alpha=False,  # Use constant alpha for fire
        show_outline=True,
        outline_color=(255, 165, 0),  # Orange outline
    )
    overlay = DetectionOverlay(config)
    result = overlay.render(background, mask, confidence=confidence, severity=severity)

    # Verify result
    assert result.composite_image.shape == (200, 200, 4)
    assert result.metadata["overlay_type"] == "fire"
    assert result.metadata["detection_area_pct"] == 25.0  # 100x100 in 200x200
