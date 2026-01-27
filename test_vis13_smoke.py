#!/usr/bin/env python3
"""
VIS-1.3 Smoke Test - Detection Overlay Rendering

Validates that overlay rendering works end-to-end with synthetic data.
"""

import numpy as np
from core.reporting.imagery.overlay import DetectionOverlay, OverlayConfig
from core.reporting.imagery.color_scales import get_flood_palette, get_fire_palette


def test_flood_overlay_rendering():
    """Test flood overlay with synthetic data."""
    print("Testing flood overlay rendering...")

    # Create synthetic RGB background (512x512)
    h, w = 512, 512
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = 100  # Red channel
    background[:, :, 1] = 120  # Green channel
    background[:, :, 2] = 140  # Blue channel

    # Create synthetic flood detection mask with confidence
    detection_mask = np.zeros((h, w), dtype=bool)
    detection_mask[100:400, 100:400] = True  # Large flooded area

    confidence = np.zeros((h, w), dtype=np.float32)
    confidence[100:400, 100:400] = 0.8  # High confidence

    # Render overlay
    config = OverlayConfig(overlay_type="flood")
    overlay = DetectionOverlay(config=config)
    result = overlay.render(background, detection_mask, confidence=confidence)

    # Verify output shape and type
    assert result.composite_image.shape == (h, w, 4), f"Expected (512, 512, 4), got {result.composite_image.shape}"
    assert result.composite_image.dtype == np.uint8, f"Expected uint8, got {result.composite_image.dtype}"

    # Verify legend was generated
    assert result.legend_image is not None, "Legend should be generated"
    assert result.legend_image.shape[2] == 4, "Legend should be RGBA"

    # Verify alpha channel is used (should have transparency)
    alpha = result.composite_image[:, :, 3]
    assert alpha.max() > 0, "Alpha channel should have non-zero values"

    # Verify blue tones in flooded area (flood uses blue palette)
    flooded_region = result.composite_image[200:300, 200:300, :]
    blue_channel = flooded_region[:, :, 2]
    assert blue_channel.mean() > 100, "Flooded area should have blue tones"

    print("✓ Flood overlay rendering passed")
    return result


def test_fire_overlay_rendering():
    """Test wildfire overlay with synthetic data."""
    print("Testing wildfire overlay rendering...")

    # Create synthetic RGB background (512x512)
    h, w = 512, 512
    background = np.zeros((h, w, 3), dtype=np.uint8)
    background[:, :, 0] = 80   # Red channel
    background[:, :, 1] = 100  # Green channel
    background[:, :, 2] = 60   # Blue channel

    # Create synthetic burn severity mask (0 = unburned, 1 = severe)
    severity_mask = np.zeros((h, w), dtype=np.float32)
    severity_mask[100:200, 100:400] = 0.3  # Low severity
    severity_mask[200:300, 100:400] = 0.6  # Moderate severity
    severity_mask[300:400, 100:400] = 0.9  # High severity

    confidence = np.ones((h, w), dtype=np.float32) * 0.85

    # Render overlay
    config = OverlayConfig(overlay_type="fire")
    overlay = DetectionOverlay(config=config)
    result = overlay.render(background, severity_mask > 0, confidence=confidence, severity=severity_mask)

    # Verify output shape and type
    assert result.composite_image.shape == (h, w, 4), f"Expected (512, 512, 4), got {result.composite_image.shape}"
    assert result.composite_image.dtype == np.uint8, f"Expected uint8, got {result.composite_image.dtype}"

    # Verify legend was generated
    assert result.legend_image is not None, "Legend should be generated"

    # Verify colors vary by severity (high severity should have more red)
    high_severity_region = result.composite_image[320:380, 200:300, :]
    red_channel = high_severity_region[:, :, 0]
    assert red_channel.mean() > 100, "High severity area should have red/orange tones"

    print("✓ Wildfire overlay rendering passed")
    return result


def test_confidence_affects_transparency():
    """Test that confidence affects alpha transparency."""
    print("Testing confidence-based transparency...")

    h, w = 256, 256
    background = np.ones((h, w, 3), dtype=np.uint8) * 50  # Dark background
    detection_mask = np.ones((h, w), dtype=bool)

    # Two regions with different confidence
    confidence = np.zeros((h, w), dtype=np.float32)
    confidence[:, :128] = 0.3   # Low confidence (left half)
    confidence[:, 128:] = 0.95  # High confidence (right half)

    config = OverlayConfig(overlay_type="flood")
    overlay = DetectionOverlay(config=config)
    result = overlay.render(background, detection_mask, confidence=confidence)

    # Check that RGB values differ between regions (higher confidence = more overlay color, less background)
    # Blue channel should be strongest in flood overlay
    blue_low = result.composite_image[:, :128, 2].mean()
    blue_high = result.composite_image[:, 128:, 2].mean()

    # High confidence should have more of the overlay's blue color
    assert blue_high > blue_low, f"High confidence blue ({blue_high:.1f}) should exceed low confidence blue ({blue_low:.1f})"

    print(f"  Low confidence region blue: {blue_low:.1f}")
    print(f"  High confidence region blue: {blue_high:.1f}")
    print("✓ Confidence-based transparency passed")


def test_palette_colors():
    """Test that flood and fire palettes have correct color tones."""
    print("Testing color palette accuracy...")

    # Test flood palette (should be blue gradient)
    flood_palette = get_flood_palette()
    flood_colors = np.array(flood_palette.colors)

    # Blue channel should be dominant in flood colors
    blue_channel = flood_colors[:, 2]
    assert blue_channel.mean() > flood_colors[:, 0].mean(), "Flood palette should be dominated by blue"
    assert blue_channel.mean() > flood_colors[:, 1].mean(), "Flood palette should be dominated by blue"

    # Test fire palette (should progress from green → yellow → orange → red)
    fire_palette = get_fire_palette()
    fire_colors = np.array(fire_palette.colors)

    # First color should be greenish (unburned)
    assert fire_colors[0, 1] > fire_colors[0, 0], "Unburned color should be greenish"

    # Last color should be reddish (severe)
    assert fire_colors[-1, 0] > fire_colors[-1, 1], "Severe burn color should be reddish"

    print("✓ Color palette accuracy passed")


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("VIS-1.3 Detection Overlay Rendering - Smoke Test")
    print("="*60 + "\n")

    try:
        test_flood_overlay_rendering()
        test_fire_overlay_rendering()
        test_confidence_affects_transparency()
        test_palette_colors()

        print("\n" + "="*60)
        print("✓ ALL SMOKE TESTS PASSED")
        print("="*60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
