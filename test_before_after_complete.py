#!/usr/bin/env python3
"""
Quick verification test for the fixed before_after.py module.

This test ensures all dataclasses and the BeforeAfterGenerator class
are properly defined and functional.
"""

from datetime import datetime
from pathlib import Path
import tempfile

import numpy as np

from core.reporting.imagery import (
    BeforeAfterConfig,
    BeforeAfterGenerator,
    BeforeAfterResult,
    OutputConfig,
)


def test_dataclasses():
    """Test that all dataclasses can be instantiated."""
    print("Testing dataclasses...")

    # Test BeforeAfterConfig
    config = BeforeAfterConfig(
        time_window_days=30,
        max_cloud_cover=20.0,
        min_coverage=80.0
    )
    assert config.time_window_days == 30
    print("  ✓ BeforeAfterConfig")

    # Test OutputConfig
    output_config = OutputConfig(
        gap_width=20,
        label_font_size=24,
        label_background_alpha=0.7,
        gif_frame_duration_ms=1000
    )
    assert output_config.gap_width == 20
    print("  ✓ OutputConfig")

    # Test BeforeAfterResult
    before_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    after_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = BeforeAfterResult(
        before_image=before_img,
        after_image=after_img,
        before_date=datetime(2024, 1, 1),
        after_date=datetime(2024, 1, 15)
    )
    assert result.before_image.shape == (100, 100, 3)
    print("  ✓ BeforeAfterResult")


def test_generator():
    """Test BeforeAfterGenerator methods."""
    print("\nTesting BeforeAfterGenerator...")

    # Create generator
    config = BeforeAfterConfig()
    generator = BeforeAfterGenerator(config)
    print("  ✓ BeforeAfterGenerator instantiation")

    # Create test images
    before_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    after_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = BeforeAfterResult(
        before_image=before_img,
        after_image=after_img,
        before_date=datetime(2024, 1, 1),
        after_date=datetime(2024, 1, 15)
    )

    # Test side-by-side
    composite = generator.generate_side_by_side(result)
    assert composite.shape == (100, 220, 3)  # 100x100 + 20px gap + 100x100
    print("  ✓ generate_side_by_side")

    # Test labeled comparison
    labeled = generator.generate_labeled_comparison(result)
    assert labeled.shape == (100, 220, 3)
    print("  ✓ generate_labeled_comparison")

    # Test animated GIF
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.gif"
        gif_path = generator.generate_animated_gif(result, output_path)
        assert gif_path.exists()
        print("  ✓ generate_animated_gif")


def test_file_saving():
    """Test file saving functionality."""
    print("\nTesting file saving...")

    generator = BeforeAfterGenerator()
    before_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    after_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = BeforeAfterResult(
        before_image=before_img,
        after_image=after_img,
        before_date=datetime(2024, 1, 1),
        after_date=datetime(2024, 1, 15)
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test saving side-by-side
        side_by_side_path = Path(tmpdir) / "side_by_side.png"
        generator.generate_side_by_side(result, output_path=side_by_side_path)
        assert side_by_side_path.exists()
        print("  ✓ Side-by-side saves to file")

        # Test saving labeled
        labeled_path = Path(tmpdir) / "labeled.png"
        generator.generate_labeled_comparison(result, output_path=labeled_path)
        assert labeled_path.exists()
        print("  ✓ Labeled comparison saves to file")


if __name__ == "__main__":
    print("=" * 60)
    print("Before/After Module Verification Test")
    print("=" * 60)

    test_dataclasses()
    test_generator()
    test_file_saving()

    print("\n" + "=" * 60)
    print("✅ All tests passed! before_after.py is complete and functional.")
    print("=" * 60)
