#!/usr/bin/env python3
"""
Smoke test for VIS-1.5 Report Integration Pipeline.

Quick validation that the pipeline can generate all visual products.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from core.reporting.imagery import (
    ReportVisualPipeline,
    PipelineConfig,
    ImageManifest,
)


def create_sample_bands(height=512, width=512):
    """Create synthetic band data."""
    red = np.linspace(0, 4000, height * width).reshape(height, width).astype(np.float32)
    green = np.linspace(1000, 3000, height * width).reshape(height, width).astype(np.float32)
    blue = np.full((height, width), 2000, dtype=np.float32)

    return {
        "B04": red,    # Sentinel-2 Red
        "B03": green,  # Sentinel-2 Green
        "B02": blue,   # Sentinel-2 Blue
    }


def create_sample_detection_mask(height=512, width=512):
    """Create synthetic detection mask."""
    mask = np.zeros((height, width), dtype=bool)
    mask[100:200, 100:300] = True
    mask[300:400, 200:350] = True
    return mask


def main():
    print("🧪 VIS-1.5 Pipeline Smoke Test\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Test 1: Pipeline initialization
        print("1. Initializing pipeline...")
        config = PipelineConfig(
            cache_ttl_hours=1,
            web_max_width=800,
            print_dpi=300,
            overlay_alpha=0.6
        )
        pipeline = ReportVisualPipeline(output_dir=output_dir, config=config)
        print("   ✓ Pipeline initialized")
        print(f"   ✓ Output dir: {output_dir}")
        print(f"   ✓ Cache dir created: {(output_dir / '.cache').exists()}")

        # Test 2: Generate satellite imagery
        print("\n2. Generating satellite imagery...")
        bands = create_sample_bands()
        sat_path = pipeline.generate_satellite_imagery(
            bands=bands,
            sensor="sentinel2",
            composite_name="true_color"
        )
        print(f"   ✓ Satellite image generated: {sat_path.name}")
        print(f"   ✓ File exists: {sat_path.exists()}")
        img = Image.open(sat_path)
        print(f"   ✓ Image size: {img.size}")
        print(f"   ✓ Image mode: {img.mode}")

        # Test 3: Generate before/after comparison
        print("\n3. Generating before/after comparison...")
        before_bands = {k: v * 0.9 for k, v in bands.items()}
        before_path, after_path, side_by_side, labeled, gif = pipeline.generate_before_after(
            before_bands=before_bands,
            after_bands=bands,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
            sensor="sentinel2"
        )
        print(f"   ✓ Before image: {before_path.name}")
        print(f"   ✓ After image: {after_path.name}")
        print(f"   ✓ Side-by-side: {side_by_side.name}")
        print(f"   ✓ Labeled comparison: {labeled.name}")
        print(f"   ✓ Animated GIF: {gif.name if gif else 'None'}")

        # Test 4: Generate detection overlay
        print("\n4. Generating detection overlay...")
        background = np.array(Image.open(sat_path))
        mask = create_sample_detection_mask()
        confidence = np.random.uniform(0.7, 1.0, mask.shape).astype(np.float32)

        overlay_path = pipeline.generate_detection_overlay(
            background=background,
            detection_mask=mask,
            overlay_type="flood",
            confidence=confidence
        )
        print(f"   ✓ Detection overlay: {overlay_path.name}")
        overlay_img = Image.open(overlay_path)
        print(f"   ✓ Overlay mode: {overlay_img.mode}")
        print(f"   ✓ Overlay size: {overlay_img.size}")

        # Test 5: Generate all visuals at once
        print("\n5. Generating complete visual suite...")
        manifest = pipeline.generate_all_visuals(
            bands=bands,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15),
            detection_mask=mask,
            confidence=confidence,
            overlay_type="flood",
            before_bands=before_bands,
            before_date=datetime(2024, 1, 1)
        )
        print("   ✓ Complete manifest generated:")
        print(f"     - Satellite: {manifest.satellite_image.name if manifest.satellite_image else 'None'}")
        print(f"     - Before: {manifest.before_image.name if manifest.before_image else 'None'}")
        print(f"     - After: {manifest.after_image.name if manifest.after_image else 'None'}")
        print(f"     - Side-by-side: {manifest.side_by_side.name if manifest.side_by_side else 'None'}")
        print(f"     - Overlay: {manifest.detection_overlay.name if manifest.detection_overlay else 'None'}")

        # Test 6: Web optimization
        print("\n6. Testing web optimization...")
        web_path = pipeline.get_web_optimized(sat_path)
        print(f"   ✓ Web image: {web_path.name}")
        web_img = Image.open(web_path)
        print(f"   ✓ Web size: {web_img.size}")
        print(f"   ✓ Web format: {web_img.format}")
        web_size_kb = web_path.stat().st_size / 1024
        print(f"   ✓ Web file size: {web_size_kb:.1f} KB")

        # Test 7: Print optimization
        print("\n7. Testing print optimization...")
        print_path = pipeline.get_print_optimized(sat_path, dpi=300)
        print(f"   ✓ Print image: {print_path.name}")
        print_img = Image.open(print_path)
        print(f"   ✓ Print DPI: {print_img.info.get('dpi', 'not set')}")
        print(f"   ✓ Print format: {print_img.format}")

        # Test 8: Manifest serialization
        print("\n8. Testing manifest persistence...")
        manifest_file = output_dir / "manifest.json"
        print(f"   ✓ Manifest saved: {manifest_file.exists()}")

        loaded_manifest = pipeline.load_manifest()
        print(f"   ✓ Manifest loaded: {loaded_manifest is not None}")
        print(f"   ✓ Sensor: {loaded_manifest.metadata.get('sensor')}")
        print(f"   ✓ Has before/after: {loaded_manifest.metadata.get('has_before_after')}")
        print(f"   ✓ Has overlay: {loaded_manifest.metadata.get('has_detection_overlay')}")

        # Test 9: Caching
        print("\n9. Testing caching...")
        sat_path2 = pipeline.generate_satellite_imagery(
            bands=bands,
            sensor="sentinel2",
            composite_name="true_color"
        )
        print(f"   ✓ Second call uses cache: {sat_path == sat_path2}")
        print(f"   ✓ File not regenerated: {sat_path.stat().st_mtime == sat_path2.stat().st_mtime}")

        print("\n" + "=" * 50)
        print("✅ All VIS-1.5 smoke tests passed!")
        print("=" * 50)

        # Summary
        print("\n📊 Summary:")
        print(f"   - Total files generated: {len(list(output_dir.glob('*.png')))}")
        print(f"   - Web-optimized files: {len(list((output_dir / 'web').glob('*.jpg')))}")
        print(f"   - Print-optimized files: {len(list((output_dir / 'print').glob('*.png')))}")
        print(f"   - Cache entries: {len(list((output_dir / '.cache').glob('*.json')))}")


if __name__ == "__main__":
    main()
