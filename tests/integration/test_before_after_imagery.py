"""
Integration Tests for Before/After Image Generation (VIS-1.2).

Tests the integration of before/after comparison products:
- BeforeAfterGenerator basic functionality
- Image alignment with comparison.py utilities
- Side-by-side composites
- Date-labeled comparisons
- Animated GIF generation
- Histogram normalization
- Difference calculation
- Co-registration with spatial alignment

These tests use synthetic data and can run offline without STAC access.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

# Skip tests if dependencies not available
rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from core.reporting.imagery import (
    BeforeAfterConfig,
    BeforeAfterGenerator,
    BeforeAfterResult,
    OutputConfig,
    AlignedPair,
    ComparisonConfig,
    coregister_images,
    normalize_histograms,
    calculate_difference,
)


class SyntheticImageFactory:
    """Factory for creating synthetic test imagery."""

    @staticmethod
    def create_rgb_array(
        width: int = 200,
        height: int = 200,
        mean: float = 128,
        std: float = 30,
        seed: int = 42,
    ) -> np.ndarray:
        """Create synthetic RGB image array.

        Args:
            width: Image width in pixels
            height: Image height in pixels
            mean: Mean pixel value (0-255)
            std: Standard deviation of pixel values
            seed: Random seed for reproducibility

        Returns:
            RGB array with shape (height, width, 3) and dtype uint8
        """
        np.random.seed(seed)

        # Create image with some structure (gradients + noise)
        r_channel = np.random.normal(mean, std, (height, width))
        g_channel = np.random.normal(mean - 10, std, (height, width))
        b_channel = np.random.normal(mean - 20, std, (height, width))

        # Add some gradients for visual interest
        y_gradient = np.linspace(0, 30, height)[:, np.newaxis]
        x_gradient = np.linspace(0, 20, width)[np.newaxis, :]

        r_channel += y_gradient
        g_channel += x_gradient
        b_channel += y_gradient * 0.5

        # Clip and convert to uint8
        rgb = np.stack([r_channel, g_channel, b_channel], axis=2)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        return rgb

    @staticmethod
    def create_changed_image(
        base_image: np.ndarray,
        change_type: str = "brightening",
        change_intensity: float = 30,
    ) -> np.ndarray:
        """Create a changed version of the base image.

        Args:
            base_image: Base RGB image
            change_type: Type of change ('brightening', 'darkening', 'localized')
            change_intensity: Intensity of change (0-255)

        Returns:
            Changed RGB array with same shape as base
        """
        changed = base_image.astype(np.float32).copy()

        if change_type == "brightening":
            # Overall brightening
            changed += change_intensity
        elif change_type == "darkening":
            # Overall darkening
            changed -= change_intensity
        elif change_type == "localized":
            # Localized change in center region
            h, w = changed.shape[:2]
            center_y, center_x = h // 2, w // 2
            y_start, y_end = center_y - 30, center_y + 30
            x_start, x_end = center_x - 30, center_x + 30
            changed[y_start:y_end, x_start:x_end] += change_intensity
        else:
            raise ValueError(f"Unknown change_type: {change_type}")

        # Clip to valid range
        changed = np.clip(changed, 0, 255).astype(np.uint8)

        return changed


@pytest.mark.integration
class TestBeforeAfterGeneratorBasic:
    """Tests for basic BeforeAfterGenerator functionality."""

    def test_generator_creation_default_config(self):
        """Test creating generator with default configuration."""
        generator = BeforeAfterGenerator()

        assert generator.config is not None
        assert isinstance(generator.config, BeforeAfterConfig)
        assert generator.config.time_window_days == 30
        assert generator.config.max_cloud_cover == 20.0

    def test_generator_creation_custom_config(self):
        """Test creating generator with custom configuration."""
        config = BeforeAfterConfig(
            time_window_days=60,
            max_cloud_cover=10.0,
            min_coverage=90.0,
        )
        generator = BeforeAfterGenerator(config)

        assert generator.config.time_window_days == 60
        assert generator.config.max_cloud_cover == 10.0
        assert generator.config.min_coverage == 90.0

    def test_generate_side_by_side_basic(self):
        """Test basic side-by-side comparison generation."""
        # Create synthetic images
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        generator = BeforeAfterGenerator()
        composite = generator.generate_side_by_side(result)

        # Check dimensions
        assert composite.shape[0] == before_img.shape[0]  # Same height
        expected_width = before_img.shape[1] * 2 + 20  # 2 images + default gap
        assert composite.shape[1] == expected_width
        assert composite.shape[2] == 3  # RGB

        # Check that images are present
        # Left panel should match before image
        assert np.array_equal(composite[:, :before_img.shape[1], :], before_img)

        # Right panel should match after image
        gap = 20
        assert np.array_equal(
            composite[:, before_img.shape[1] + gap :, :], after_img
        )

        # Gap should be white (255)
        gap_region = composite[:, before_img.shape[1] : before_img.shape[1] + gap, :]
        assert np.all(gap_region == 255)

    def test_generate_side_by_side_custom_gap(self):
        """Test side-by-side with custom gap width."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        generator = BeforeAfterGenerator()
        output_config = OutputConfig(gap_width=50)
        composite = generator.generate_side_by_side(result, config=output_config)

        # Check gap width
        expected_width = before_img.shape[1] * 2 + 50
        assert composite.shape[1] == expected_width

        # Gap should be white
        gap_region = composite[:, before_img.shape[1] : before_img.shape[1] + 50, :]
        assert np.all(gap_region == 255)

    def test_generate_side_by_side_with_output_path(self, tmp_path):
        """Test saving side-by-side composite to file."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        output_path = tmp_path / "side_by_side.png"
        generator = BeforeAfterGenerator()
        composite = generator.generate_side_by_side(result, output_path=output_path)

        # File should exist
        assert output_path.exists()
        assert output_path.suffix == ".png"

        # Load and verify
        from PIL import Image

        loaded = np.array(Image.open(output_path))
        assert np.array_equal(loaded, composite)


@pytest.mark.integration
class TestDateLabels:
    """Tests for date label functionality."""

    def test_add_date_labels_bottom(self):
        """Test adding date label at bottom center."""
        generator = BeforeAfterGenerator()
        image = SyntheticImageFactory.create_rgb_array()
        date = datetime(2024, 1, 15)

        labeled = generator.add_date_labels(image, date, position="bottom")

        # Image should be modified (copy returned)
        assert labeled.shape == image.shape
        assert not np.array_equal(labeled, image)  # Should be different

        # Labeled image should have text region that differs from original
        # Check bottom region where label should be
        bottom_region = labeled[-50:, :, :]
        original_bottom = image[-50:, :, :]
        assert not np.array_equal(bottom_region, original_bottom)

    def test_add_date_labels_all_positions(self):
        """Test all label positions."""
        positions = [
            "top",
            "bottom",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ]

        generator = BeforeAfterGenerator()
        image = SyntheticImageFactory.create_rgb_array()
        date = datetime(2024, 1, 15)

        for position in positions:
            labeled = generator.add_date_labels(image, date, position=position)
            assert labeled.shape == image.shape
            assert not np.array_equal(labeled, image)

    def test_add_date_labels_invalid_position(self):
        """Test that invalid position raises ValueError."""
        generator = BeforeAfterGenerator()
        image = SyntheticImageFactory.create_rgb_array()
        date = datetime(2024, 1, 15)

        with pytest.raises(ValueError, match="Invalid position"):
            generator.add_date_labels(image, date, position="invalid")

    def test_add_date_labels_custom_font_size(self):
        """Test custom font size for labels."""
        generator = BeforeAfterGenerator()
        image = SyntheticImageFactory.create_rgb_array()
        date = datetime(2024, 1, 15)

        config = OutputConfig(label_font_size=32)
        labeled = generator.add_date_labels(image, date, config=config)

        assert labeled.shape == image.shape
        assert not np.array_equal(labeled, image)


@pytest.mark.integration
class TestLabeledComparison:
    """Tests for labeled comparison generation."""

    def test_generate_labeled_comparison(self):
        """Test generating labeled side-by-side comparison."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        generator = BeforeAfterGenerator()
        labeled = generator.generate_labeled_comparison(result)

        # Check dimensions
        assert labeled.shape[0] == before_img.shape[0]
        expected_width = before_img.shape[1] * 2 + 20
        assert labeled.shape[1] == expected_width
        assert labeled.shape[2] == 3

        # Labels should be present (bottom regions should differ from plain side-by-side)
        plain = generator.generate_side_by_side(result)
        assert not np.array_equal(labeled, plain)

        # Bottom regions should have labels
        bottom_left = labeled[-50:, :before_img.shape[1], :]
        bottom_right = labeled[-50:, before_img.shape[1] + 20 :, :]

        plain_bottom_left = plain[-50:, :before_img.shape[1], :]
        plain_bottom_right = plain[-50:, before_img.shape[1] + 20 :, :]

        assert not np.array_equal(bottom_left, plain_bottom_left)
        assert not np.array_equal(bottom_right, plain_bottom_right)

    def test_generate_labeled_comparison_with_output(self, tmp_path):
        """Test saving labeled comparison to file."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        output_path = tmp_path / "labeled_comparison.png"
        generator = BeforeAfterGenerator()
        labeled = generator.generate_labeled_comparison(result, output_path=output_path)

        # File should exist
        assert output_path.exists()

        # Load and verify
        from PIL import Image

        loaded = np.array(Image.open(output_path))
        assert np.array_equal(loaded, labeled)


@pytest.mark.integration
class TestAnimatedGif:
    """Tests for animated GIF generation."""

    def test_generate_animated_gif(self, tmp_path):
        """Test generating animated GIF from image pair."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_changed_image(
            before_img, change_type="brightening", change_intensity=40
        )

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        output_path = tmp_path / "comparison.gif"
        generator = BeforeAfterGenerator()
        gif_path = generator.generate_animated_gif(result, output_path)

        # File should exist
        assert gif_path.exists()
        assert gif_path.suffix == ".gif"

        # Verify it's a valid GIF with multiple frames
        from PIL import Image

        with Image.open(gif_path) as gif:
            assert gif.format == "GIF"
            assert gif.is_animated
            assert gif.n_frames >= 2

    def test_generate_animated_gif_custom_duration(self, tmp_path):
        """Test animated GIF with custom frame duration."""
        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        output_path = tmp_path / "fast_comparison.gif"
        generator = BeforeAfterGenerator()
        gif_path = generator.generate_animated_gif(
            result, output_path, frame_duration_ms=500
        )

        assert gif_path.exists()

        # Verify it's animated
        from PIL import Image

        with Image.open(gif_path) as gif:
            assert gif.is_animated

    def test_generate_animated_gif_creates_parent_dir(self, tmp_path):
        """Test that GIF generation creates parent directories."""
        output_path = tmp_path / "subdir" / "nested" / "comparison.gif"

        before_img = SyntheticImageFactory.create_rgb_array(seed=1)
        after_img = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
        )

        generator = BeforeAfterGenerator()
        gif_path = generator.generate_animated_gif(result, output_path)

        assert gif_path.exists()
        assert gif_path.parent.exists()


@pytest.mark.integration
class TestImageAlignment:
    """Tests for image co-registration and alignment."""

    def test_coregister_images_same_extent(self):
        """Test co-registering images with same extent."""
        # Create two images with same spatial properties
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=1)
        after = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=2)

        # Same transform for both
        transform = from_bounds(-10, -10, 10, 10, 100, 100)

        config = ComparisonConfig(target_resolution=0.2)
        aligned = coregister_images(
            before, after, transform, transform, config=config
        )

        # Should have same shape
        assert aligned.before.shape == aligned.after.shape

        # Should have expected resolution
        assert aligned.resolution == 0.2

        # Bounds should be correct
        assert aligned.bounds == (-10, -10, 10, 10)

    def test_coregister_images_different_resolutions(self):
        """Test co-registering images with different resolutions."""
        # Create images with different sizes (simulating different resolutions)
        before = SyntheticImageFactory.create_rgb_array(width=50, height=50, seed=1)
        after = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=2)

        # Different transforms (same extent, different pixel counts)
        before_transform = from_bounds(-10, -10, 10, 10, 50, 50)
        after_transform = from_bounds(-10, -10, 10, 10, 100, 100)

        config = ComparisonConfig(target_resolution=0.2)
        aligned = coregister_images(
            before, after, before_transform, after_transform, config=config
        )

        # Both should be resampled to same dimensions
        assert aligned.before.shape == aligned.after.shape

        # Should have target resolution
        assert aligned.resolution == 0.2

    def test_coregister_images_intersection(self):
        """Test co-registration cropping to intersection."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=1)
        after = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=2)

        # Overlapping but not identical extents
        before_transform = from_bounds(-10, -10, 10, 10, 100, 100)
        after_transform = from_bounds(-5, -5, 15, 15, 100, 100)

        config = ComparisonConfig(
            target_resolution=0.2, crop_to_intersection=True
        )
        aligned = coregister_images(
            before, after, before_transform, after_transform, config=config
        )

        # Both images should be aligned
        assert aligned.before.shape == aligned.after.shape

        # Bounds should be intersection
        expected_bounds = (-5, -5, 10, 10)
        assert aligned.bounds == expected_bounds

    def test_coregister_images_union(self):
        """Test co-registration padding to union."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=1)
        after = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=2)

        before_transform = from_bounds(-10, -10, 10, 10, 100, 100)
        after_transform = from_bounds(-5, -5, 15, 15, 100, 100)

        config = ComparisonConfig(
            target_resolution=0.2, crop_to_intersection=False
        )
        aligned = coregister_images(
            before, after, before_transform, after_transform, config=config
        )

        # Both images should be aligned
        assert aligned.before.shape == aligned.after.shape

        # Bounds should be union
        expected_bounds = (-10, -10, 15, 15)
        assert aligned.bounds == expected_bounds

    def test_coregister_images_no_overlap_raises(self):
        """Test that non-overlapping images raise error with crop_to_intersection."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=1)
        after = SyntheticImageFactory.create_rgb_array(width=100, height=100, seed=2)

        # Non-overlapping extents
        before_transform = from_bounds(-20, -20, -10, -10, 100, 100)
        after_transform = from_bounds(10, 10, 20, 20, 100, 100)

        config = ComparisonConfig(crop_to_intersection=True)

        with pytest.raises(ValueError, match="do not overlap"):
            coregister_images(before, after, before_transform, after_transform, config=config)


@pytest.mark.integration
class TestHistogramNormalization:
    """Tests for histogram normalization."""

    def test_normalize_histograms_match_method(self):
        """Test histogram matching normalization."""
        # Create two images with different brightness
        before = SyntheticImageFactory.create_rgb_array(mean=100, std=20, seed=1)
        after = SyntheticImageFactory.create_rgb_array(mean=150, std=20, seed=2)

        # Normalize using histogram matching
        before_norm, after_norm = normalize_histograms(before, after, method="match")

        # Before should be unchanged
        assert np.array_equal(before_norm, before)

        # After should be modified to match before's histogram
        assert not np.array_equal(after_norm, after)

        # After normalized should have similar mean to before
        # (allowing for some variation due to histogram matching)
        before_mean = np.mean(before)
        after_norm_mean = np.mean(after_norm)
        assert abs(before_mean - after_norm_mean) < 20  # Within 20 units

    def test_normalize_histograms_standardize_method(self):
        """Test standardization normalization."""
        before = SyntheticImageFactory.create_rgb_array(mean=100, std=30, seed=1)
        after = SyntheticImageFactory.create_rgb_array(mean=150, std=40, seed=2)

        # Normalize using standardization
        before_norm, after_norm = normalize_histograms(
            before, after, method="standardize"
        )

        # Both should be modified
        assert not np.array_equal(before_norm, before)
        assert not np.array_equal(after_norm, after)

        # Both should have approximately zero mean and unit variance
        # (within valid pixels, excluding zeros)
        before_valid = before_norm[before_norm > 0]
        after_valid = after_norm[after_norm > 0]

        # Means should be close to zero
        assert abs(np.mean(before_valid)) < 10
        assert abs(np.mean(after_valid)) < 10

    def test_normalize_histograms_invalid_method(self):
        """Test that invalid method raises ValueError."""
        before = SyntheticImageFactory.create_rgb_array(seed=1)
        after = SyntheticImageFactory.create_rgb_array(seed=2)

        with pytest.raises(ValueError, match="Invalid normalization method"):
            normalize_histograms(before, after, method="invalid")

    def test_normalize_histograms_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100)
        after = SyntheticImageFactory.create_rgb_array(width=150, height=150)

        with pytest.raises(ValueError, match="same shape"):
            normalize_histograms(before, after)


@pytest.mark.integration
class TestDifferenceCalculation:
    """Tests for difference calculation."""

    def test_calculate_difference_basic(self):
        """Test basic difference calculation."""
        before = SyntheticImageFactory.create_rgb_array(mean=100, std=10, seed=1)
        after = SyntheticImageFactory.create_changed_image(
            before, change_type="brightening", change_intensity=50
        )

        diff = calculate_difference(before, after)

        # Difference should have same shape
        assert diff.shape == before.shape

        # Most differences should be positive (brightening)
        assert np.mean(diff) > 0

        # Check magnitude is reasonable
        mean_diff = np.mean(diff)
        assert 40 < mean_diff < 60  # Around the 50 intensity change

    def test_calculate_difference_darkening(self):
        """Test difference with darkening change."""
        before = SyntheticImageFactory.create_rgb_array(mean=150, std=10, seed=1)
        after = SyntheticImageFactory.create_changed_image(
            before, change_type="darkening", change_intensity=40
        )

        diff = calculate_difference(before, after)

        # Most differences should be negative (darkening)
        assert np.mean(diff) < 0

    def test_calculate_difference_no_change(self):
        """Test difference with identical images."""
        before = SyntheticImageFactory.create_rgb_array(seed=1)
        after = before.copy()

        diff = calculate_difference(before, after)

        # Difference should be all zeros
        assert np.allclose(diff, 0)

    def test_calculate_difference_localized_change(self):
        """Test difference with localized change."""
        before = SyntheticImageFactory.create_rgb_array(mean=100, std=10, seed=1)
        after = SyntheticImageFactory.create_changed_image(
            before, change_type="localized", change_intensity=80
        )

        diff = calculate_difference(before, after)

        # Center region should have large differences
        h, w = diff.shape[:2]
        center_y, center_x = h // 2, w // 2
        center_diff = diff[center_y - 20 : center_y + 20, center_x - 20 : center_x + 20]

        # Center should have significant change
        assert np.mean(center_diff) > 30

        # Corners should have minimal change
        corner_diff = diff[:20, :20]
        assert abs(np.mean(corner_diff)) < 10

    def test_calculate_difference_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100)
        after = SyntheticImageFactory.create_rgb_array(width=150, height=150)

        with pytest.raises(ValueError, match="same shape"):
            calculate_difference(before, after)


@pytest.mark.integration
class TestBeforeAfterResult:
    """Tests for BeforeAfterResult validation."""

    def test_before_after_result_valid(self):
        """Test creating valid BeforeAfterResult."""
        before = SyntheticImageFactory.create_rgb_array(seed=1)
        after = SyntheticImageFactory.create_rgb_array(seed=2)

        result = BeforeAfterResult(
            before_image=before,
            after_image=after,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 15),
            before_cloud_cover=5.0,
            after_cloud_cover=8.0,
            metadata={"sensor": "Sentinel-2"},
        )

        assert result.before_image.shape == before.shape
        assert result.after_image.shape == after.shape
        assert result.before_date < result.after_date
        assert result.metadata["sensor"] == "Sentinel-2"

    def test_before_after_result_shape_mismatch(self):
        """Test that shape mismatch raises ValueError."""
        before = SyntheticImageFactory.create_rgb_array(width=100, height=100)
        after = SyntheticImageFactory.create_rgb_array(width=150, height=150)

        with pytest.raises(ValueError, match="same shape"):
            BeforeAfterResult(
                before_image=before,
                after_image=after,
                before_date=datetime(2024, 1, 1),
                after_date=datetime(2024, 1, 15),
            )

    def test_before_after_result_wrong_dimensions(self):
        """Test that non-RGB images raise ValueError."""
        before = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # Grayscale
        after = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be RGB"):
            BeforeAfterResult(
                before_image=before,
                after_image=after,
                before_date=datetime(2024, 1, 1),
                after_date=datetime(2024, 1, 15),
            )

    def test_before_after_result_date_order_error(self):
        """Test that incorrect date order raises ValueError."""
        before = SyntheticImageFactory.create_rgb_array(seed=1)
        after = SyntheticImageFactory.create_rgb_array(seed=2)

        with pytest.raises(ValueError, match="before_date must be earlier"):
            BeforeAfterResult(
                before_image=before,
                after_image=after,
                before_date=datetime(2024, 1, 15),  # After date
                after_date=datetime(2024, 1, 1),  # Before date
            )


@pytest.mark.integration
class TestEndToEndWorkflow:
    """End-to-end integration tests combining multiple components."""

    def test_full_comparison_workflow(self, tmp_path):
        """Test complete before/after comparison workflow."""
        # 1. Create synthetic before/after images with changes
        before_img = SyntheticImageFactory.create_rgb_array(mean=120, std=25, seed=10)
        after_img = SyntheticImageFactory.create_changed_image(
            before_img, change_type="brightening", change_intensity=35
        )

        # 2. Create BeforeAfterResult
        result = BeforeAfterResult(
            before_image=before_img,
            after_image=after_img,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 2, 1),
            before_cloud_cover=3.5,
            after_cloud_cover=7.2,
            metadata={"event": "test_flood", "sensor": "Sentinel-2"},
        )

        # 3. Create generator
        config = BeforeAfterConfig(time_window_days=45, max_cloud_cover=15.0)
        generator = BeforeAfterGenerator(config)

        # 4. Generate all products
        output_config = OutputConfig(gap_width=30, label_font_size=28)

        # Side-by-side
        side_by_side_path = tmp_path / "side_by_side.png"
        side_by_side = generator.generate_side_by_side(
            result, output_path=side_by_side_path, config=output_config
        )

        # Labeled comparison
        labeled_path = tmp_path / "labeled.png"
        labeled = generator.generate_labeled_comparison(
            result, output_path=labeled_path, config=output_config
        )

        # Animated GIF
        gif_path = tmp_path / "animation.gif"
        gif = generator.generate_animated_gif(
            result, gif_path, frame_duration_ms=800, config=output_config
        )

        # 5. Verify all outputs
        assert side_by_side_path.exists()
        assert labeled_path.exists()
        assert gif_path.exists()

        # Check dimensions
        assert side_by_side.shape[1] == before_img.shape[1] * 2 + 30
        assert labeled.shape[1] == before_img.shape[1] * 2 + 30

        # Labeled should differ from plain side-by-side
        assert not np.array_equal(side_by_side, labeled)

    def test_workflow_with_alignment(self, tmp_path):
        """Test workflow with image alignment step."""
        # Create images with different resolutions
        before_small = SyntheticImageFactory.create_rgb_array(
            width=80, height=80, seed=1
        )
        after_large = SyntheticImageFactory.create_rgb_array(
            width=120, height=120, seed=2
        )

        # Create transforms
        before_transform = from_bounds(-10, -10, 10, 10, 80, 80)
        after_transform = from_bounds(-10, -10, 10, 10, 120, 120)

        # Align images first
        config = ComparisonConfig(target_resolution=0.25)
        aligned = coregister_images(
            before_small, after_large, before_transform, after_transform, config=config
        )

        # Now both images have same dimensions
        assert aligned.before.shape == aligned.after.shape

        # Create BeforeAfterResult with aligned images
        result = BeforeAfterResult(
            before_image=aligned.before,
            after_image=aligned.after,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 20),
        )

        # Generate comparison products
        generator = BeforeAfterGenerator()
        comparison = generator.generate_labeled_comparison(result)

        assert comparison.shape[0] == aligned.before.shape[0]

    def test_workflow_with_normalization(self, tmp_path):
        """Test workflow with histogram normalization."""
        # Create images with different brightness
        before_dark = SyntheticImageFactory.create_rgb_array(mean=80, std=20, seed=1)
        after_bright = SyntheticImageFactory.create_rgb_array(mean=160, std=20, seed=2)

        # Normalize histograms
        before_norm, after_norm = normalize_histograms(
            before_dark, after_bright, method="match"
        )

        # Create result with normalized images
        result = BeforeAfterResult(
            before_image=before_norm,
            after_image=after_norm,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 25),
        )

        # Generate comparison
        generator = BeforeAfterGenerator()
        comparison = generator.generate_side_by_side(result)

        # Images should now have more consistent appearance
        assert comparison.shape[2] == 3

    def test_workflow_with_difference_analysis(self, tmp_path):
        """Test workflow including difference calculation."""
        # Create images with known change
        before = SyntheticImageFactory.create_rgb_array(mean=100, std=15, seed=1)
        after = SyntheticImageFactory.create_changed_image(
            before, change_type="localized", change_intensity=60
        )

        # Calculate difference
        diff = calculate_difference(before, after)

        # Find changed regions
        change_mask = np.abs(diff) > 30
        change_pixels = np.sum(change_mask)

        # Verify change detection
        assert change_pixels > 0  # Some change detected

        # Generate comparison with original images
        result = BeforeAfterResult(
            before_image=before,
            after_image=after,
            before_date=datetime(2024, 1, 1),
            after_date=datetime(2024, 1, 18),
            metadata={"change_pixels": int(change_pixels)},
        )

        generator = BeforeAfterGenerator()
        output_path = tmp_path / "change_comparison.png"
        comparison = generator.generate_labeled_comparison(
            result, output_path=output_path
        )

        assert output_path.exists()
        assert result.metadata["change_pixels"] == change_pixels


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
