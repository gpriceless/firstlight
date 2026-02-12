#!/usr/bin/env python3
"""
Demonstration of VIS-1.2 Before/After Image Generation with Visual Outputs.

This script demonstrates the complete workflow from synthetic data to
side-by-side comparisons, labeled images, and animated GIFs.
"""

from datetime import datetime
from pathlib import Path
import numpy as np

from core.reporting.imagery.before_after import (
    BeforeAfterConfig,
    BeforeAfterGenerator,
    BeforeAfterResult,
    OutputConfig,
)


def create_synthetic_data():
    """Create synthetic before/after satellite imagery for demonstration."""

    # Create 512x512 synthetic images
    height, width = 512, 512

    # Before image: Blue water (simulating pre-flood normal conditions)
    before_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Add some variation for realism
    before_image[:, :, 2] = 100 + np.random.randint(-20, 20, (height, width))  # Blue
    before_image[:, :, 1] = 80 + np.random.randint(-10, 10, (height, width))   # Green
    before_image[:, :, 0] = 60 + np.random.randint(-10, 10, (height, width))   # Red

    # After image: Dark blue water (simulating flood - more water coverage)
    after_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Simulate flooding in lower half
    after_image[:256, :, 2] = 100 + np.random.randint(-20, 20, (256, width))   # Normal area
    after_image[:256, :, 1] = 80 + np.random.randint(-10, 10, (256, width))
    after_image[:256, :, 0] = 60 + np.random.randint(-10, 10, (256, width))

    # Flooded area - darker, more blue
    after_image[256:, :, 2] = 140 + np.random.randint(-10, 10, (256, width))   # More blue
    after_image[256:, :, 1] = 50 + np.random.randint(-10, 10, (256, width))    # Less green
    after_image[256:, :, 0] = 40 + np.random.randint(-10, 10, (256, width))    # Less red

    return before_image, after_image


def main():
    """Run the demonstration."""

    print("VIS-1.2 Before/After Visual Output Demonstration")
    print("=" * 60)

    # Create output directory
    output_dir = Path("outputs/vis_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Step 1: Generate synthetic imagery
    print("\n1. Generating synthetic satellite imagery...")
    before_img, after_img = create_synthetic_data()
    print(f"   Before image: {before_img.shape}")
    print(f"   After image: {after_img.shape}")

    # Step 2: Create BeforeAfterResult
    print("\n2. Creating BeforeAfterResult...")
    result = BeforeAfterResult(
        before_image=before_img,
        after_image=after_img,
        before_date=datetime(2024, 8, 5),
        after_date=datetime(2024, 8, 20),
        before_cloud_cover=5.0,
        after_cloud_cover=10.0,
        metadata={
            'event_date': '2024-08-15',
            'temporal_separation_days': 15,
            'composite_name': 'true_color',
            'sensor': 'sentinel2',
        }
    )
    print(f"   Temporal separation: {(result.after_date - result.before_date).days} days")

    # Step 3: Initialize generator
    print("\n3. Initializing BeforeAfterGenerator...")
    config = BeforeAfterConfig(
        time_window_days=30,
        max_cloud_cover=20.0,
        min_coverage=80.0,
    )
    generator = BeforeAfterGenerator(config)

    # Step 4: Generate side-by-side composite
    print("\n4. Generating side-by-side composite...")
    composite_path = output_dir / "01_side_by_side.png"
    composite = generator.generate_side_by_side(result, output_path=composite_path)
    print(f"   Saved: {composite_path}")
    print(f"   Shape: {composite.shape}")

    # Step 5: Generate labeled comparison (main report output)
    print("\n5. Generating labeled comparison...")
    comparison_path = output_dir / "02_labeled_comparison.png"
    comparison = generator.generate_labeled_comparison(
        result,
        output_path=comparison_path
    )
    print(f"   Saved: {comparison_path}")
    print(f"   This is the main output for reports!")

    # Step 6: Generate with custom styling
    print("\n6. Generating custom-styled comparison...")
    custom_config = OutputConfig(
        gap_width=20,
        label_font_size=32,
        label_background_alpha=0.5,
    )
    custom_path = output_dir / "03_custom_styled.png"
    custom = generator.generate_labeled_comparison(
        result,
        output_path=custom_path,
        config=custom_config
    )
    print(f"   Saved: {custom_path}")
    print(f"   Custom: 20px gap, 32pt font, 50% transparent labels")

    # Step 7: Generate animated GIF
    print("\n7. Generating animated GIF...")
    gif_path = output_dir / "04_animation.gif"
    generator.generate_animated_gif(
        result,
        output_path=gif_path,
        frame_duration_ms=1500  # 1.5 seconds per frame
    )
    print(f"   Saved: {gif_path}")
    print(f"   Animation: 1.5s per frame, loops forever")

    # Step 8: Generate fast animation
    print("\n8. Generating fast-paced GIF...")
    fast_gif_path = output_dir / "05_fast_animation.gif"
    fast_config = OutputConfig(gif_frame_duration_ms=500)
    generator.generate_animated_gif(
        result,
        output_path=fast_gif_path,
        config=fast_config
    )
    print(f"   Saved: {fast_gif_path}")
    print(f"   Animation: 0.5s per frame (faster pace)")

    # Summary
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated {len(list(output_dir.iterdir()))} output files:")
    for i, path in enumerate(sorted(output_dir.iterdir()), 1):
        print(f"  {i}. {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    print("\nNext Steps:")
    print("  - Open the PNG files to see the side-by-side comparisons")
    print("  - Open the GIF files to see the animations")
    print("  - These outputs are ready for integration into HTML/PDF reports")
    print("\nFor production use:")
    print("  - Replace synthetic data with real STAC-discovered imagery")
    print("  - Use generate_labeled_comparison() as the main report method")
    print("  - Customize OutputConfig for your visual style requirements")


if __name__ == "__main__":
    main()
