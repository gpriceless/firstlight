"""Unit tests for cloud masking functionality.

Tests cloud mask generation from Sentinel-2 Scene Classification Layer (SCL)
and application to RGB imagery.
"""

import numpy as np
import pytest

from core.reporting.imagery.cloud_mask import (
    CloudMask,
    CloudMaskConfig,
    CloudMaskResult,
    SCLClass,
)


class TestSCLClass:
    """Test SCL class enumeration."""

    def test_all_scl_values_defined(self):
        """All 12 SCL values are defined."""
        assert len(SCLClass) == 12

    def test_scl_value_ranges(self):
        """SCL values are in expected 0-11 range."""
        values = [scl.value for scl in SCLClass]
        assert min(values) == 0
        assert max(values) == 11
        assert len(set(values)) == 12  # All unique

    def test_specific_scl_values(self):
        """Specific SCL classes have correct values."""
        assert SCLClass.NO_DATA == 0
        assert SCLClass.SATURATED_DEFECTIVE == 1
        assert SCLClass.CLOUD_SHADOWS == 3
        assert SCLClass.VEGETATION == 4
        assert SCLClass.WATER == 6
        assert SCLClass.CLOUD_MEDIUM_PROBABILITY == 8
        assert SCLClass.CLOUD_HIGH_PROBABILITY == 9
        assert SCLClass.THIN_CIRRUS == 10
        assert SCLClass.SNOW_ICE == 11


class TestCloudMaskConfig:
    """Test cloud mask configuration."""

    def test_default_config(self):
        """Default configuration masks clouds and cirrus."""
        config = CloudMaskConfig()
        assert config.mask_clouds is True
        assert config.mask_cirrus is True
        assert config.mask_shadows is False
        assert config.mask_snow is False
        assert config.transparency == 0.0

    def test_custom_config(self):
        """Custom configuration is preserved."""
        config = CloudMaskConfig(
            mask_clouds=False,
            mask_cirrus=False,
            mask_shadows=True,
            mask_snow=True,
            transparency=0.5
        )
        assert config.mask_clouds is False
        assert config.mask_cirrus is False
        assert config.mask_shadows is True
        assert config.mask_snow is True
        assert config.transparency == 0.5

    def test_invalid_transparency_raises_error(self):
        """Transparency outside 0-1 range raises error."""
        with pytest.raises(ValueError, match="transparency must be between 0.0 and 1.0"):
            CloudMaskConfig(transparency=-0.1)

        with pytest.raises(ValueError, match="transparency must be between 0.0 and 1.0"):
            CloudMaskConfig(transparency=1.5)

    def test_boundary_transparency_values(self):
        """Boundary transparency values (0.0 and 1.0) are valid."""
        config1 = CloudMaskConfig(transparency=0.0)
        assert config1.transparency == 0.0

        config2 = CloudMaskConfig(transparency=1.0)
        assert config2.transparency == 1.0


class TestCloudMaskResult:
    """Test cloud mask result dataclass."""

    def test_valid_result_creation(self):
        """Valid cloud mask result can be created."""
        mask = np.zeros((10, 10), dtype=bool)
        result = CloudMaskResult(
            mask=mask,
            cloud_percentage=25.5,
            classes_masked=[SCLClass.CLOUD_HIGH_PROBABILITY]
        )
        assert result.mask.shape == (10, 10)
        assert result.cloud_percentage == 25.5
        assert result.classes_masked == [SCLClass.CLOUD_HIGH_PROBABILITY]

    def test_invalid_mask_dimension_raises_error(self):
        """Mask must be 2D array."""
        mask_3d = np.zeros((10, 10, 3), dtype=bool)
        with pytest.raises(ValueError, match="mask must be 2D array"):
            CloudMaskResult(
                mask=mask_3d,
                cloud_percentage=0.0,
                classes_masked=[]
            )

    def test_invalid_cloud_percentage_raises_error(self):
        """Cloud percentage outside 0-100 range raises error."""
        mask = np.zeros((10, 10), dtype=bool)

        with pytest.raises(ValueError, match="cloud_percentage must be between 0 and 100"):
            CloudMaskResult(
                mask=mask,
                cloud_percentage=-5.0,
                classes_masked=[]
            )

        with pytest.raises(ValueError, match="cloud_percentage must be between 0 and 100"):
            CloudMaskResult(
                mask=mask,
                cloud_percentage=105.0,
                classes_masked=[]
            )


class TestCloudMask:
    """Test CloudMask class functionality."""

    def test_default_initialization(self):
        """CloudMask initializes with default config."""
        masker = CloudMask()
        assert masker.config is not None
        assert masker.config.mask_clouds is True
        assert masker.config.mask_cirrus is True

    def test_custom_config_initialization(self):
        """CloudMask accepts custom config."""
        config = CloudMaskConfig(mask_shadows=True, transparency=0.3)
        masker = CloudMask(config=config)
        assert masker.config.mask_shadows is True
        assert masker.config.transparency == 0.3

    def test_from_scl_band_basic(self):
        """Basic cloud mask generation from SCL band."""
        scl = np.array([
            [4, 4, 8],  # vegetation, vegetation, cloud medium
            [9, 10, 4]  # cloud high, cirrus, vegetation
        ])

        masker = CloudMask()  # Default: mask clouds and cirrus
        result = masker.from_scl_band(scl)

        expected_mask = np.array([
            [False, False, True],   # vegetation valid, cloud masked
            [True, True, False]     # clouds masked, vegetation valid
        ])

        assert np.array_equal(result.mask, expected_mask)
        assert result.cloud_percentage == 50.0  # 3 out of 6 pixels are clouds
        assert SCLClass.CLOUD_MEDIUM_PROBABILITY in result.classes_masked
        assert SCLClass.CLOUD_HIGH_PROBABILITY in result.classes_masked
        assert SCLClass.THIN_CIRRUS in result.classes_masked

    def test_from_scl_band_no_clouds(self):
        """SCL band with no clouds."""
        scl = np.array([
            [4, 5, 6],  # vegetation, bare soil, water
            [4, 4, 7]   # vegetation, vegetation, unclassified
        ])

        masker = CloudMask()
        result = masker.from_scl_band(scl)

        # Only no_data and saturated pixels are always masked (none present here)
        assert np.sum(result.mask) == 0
        assert result.cloud_percentage == 0.0

    def test_from_scl_band_with_nodata(self):
        """SCL band with no data pixels."""
        scl = np.array([
            [0, 4, 4],  # no data, vegetation, vegetation
            [4, 1, 4]   # vegetation, saturated, vegetation
        ])

        masker = CloudMask()
        result = masker.from_scl_band(scl)

        # No data and saturated pixels are always masked
        assert result.mask[0, 0]  # no data
        assert result.mask[1, 1]  # saturated
        assert np.sum(result.mask) == 2

    def test_from_scl_band_mask_shadows_enabled(self):
        """Mask cloud shadows when enabled."""
        scl = np.array([
            [3, 4, 4],  # cloud shadow, vegetation, vegetation
            [4, 3, 4]   # vegetation, cloud shadow, vegetation
        ])

        # Default config: shadows NOT masked
        masker_no_shadows = CloudMask()
        result1 = masker_no_shadows.from_scl_band(scl)
        assert np.sum(result1.mask) == 0  # shadows not masked

        # Custom config: shadows masked
        config = CloudMaskConfig(mask_shadows=True)
        masker_with_shadows = CloudMask(config=config)
        result2 = masker_with_shadows.from_scl_band(scl)
        assert result2.mask[0, 0]  # shadow masked
        assert result2.mask[1, 1]  # shadow masked
        assert np.sum(result2.mask) == 2

    def test_from_scl_band_mask_snow_enabled(self):
        """Mask snow/ice when enabled."""
        scl = np.array([
            [11, 4, 4],  # snow/ice, vegetation, vegetation
            [4, 11, 4]   # vegetation, snow/ice, vegetation
        ])

        # Default config: snow NOT masked
        masker_no_snow = CloudMask()
        result1 = masker_no_snow.from_scl_band(scl)
        assert np.sum(result1.mask) == 0

        # Custom config: snow masked
        config = CloudMaskConfig(mask_snow=True)
        masker_with_snow = CloudMask(config=config)
        result2 = masker_with_snow.from_scl_band(scl)
        assert result2.mask[0, 0]
        assert result2.mask[1, 1]
        assert np.sum(result2.mask) == 2

    def test_from_scl_band_clouds_disabled(self):
        """Don't mask clouds when disabled."""
        scl = np.array([
            [8, 9, 10],  # cloud medium, cloud high, cirrus
            [4, 4, 4]    # vegetation, vegetation, vegetation
        ])

        config = CloudMaskConfig(mask_clouds=False, mask_cirrus=False)
        masker = CloudMask(config=config)
        result = masker.from_scl_band(scl)

        # Clouds not masked
        assert not result.mask[0, 0]  # cloud medium
        assert not result.mask[0, 1]  # cloud high
        assert not result.mask[0, 2]  # cirrus
        assert np.sum(result.mask) == 0

        # But cloud percentage is still calculated
        assert result.cloud_percentage == 50.0

    def test_from_scl_band_invalid_dimension(self):
        """SCL band must be 2D."""
        scl_1d = np.array([4, 8, 9])
        scl_3d = np.zeros((10, 10, 3))

        masker = CloudMask()

        with pytest.raises(ValueError, match="SCL band must be 2D array"):
            masker.from_scl_band(scl_1d)

        with pytest.raises(ValueError, match="SCL band must be 2D array"):
            masker.from_scl_band(scl_3d)

    def test_apply_to_image_basic(self):
        """Apply mask to RGB image."""
        rgb = np.full((3, 3, 3), 100, dtype=np.uint8)  # All pixels = 100
        mask = np.array([
            [False, True, False],
            [True, False, True],
            [False, False, False]
        ])

        masker = CloudMask()
        result = masker.apply_to_image(rgb, mask, fill_value=255)

        # Masked pixels should be white (255)
        assert np.all(result[0, 1] == 255)  # Masked pixel
        assert np.all(result[1, 0] == 255)  # Masked pixel
        assert np.all(result[1, 2] == 255)  # Masked pixel

        # Unmasked pixels should be unchanged (100)
        assert np.all(result[0, 0] == 100)
        assert np.all(result[1, 1] == 100)
        assert np.all(result[2, 2] == 100)

    def test_apply_to_image_fill_value_black(self):
        """Apply mask with black fill value."""
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([
            [True, False],
            [False, True]
        ])

        masker = CloudMask()
        result = masker.apply_to_image(rgb, mask, fill_value=0)

        assert np.all(result[0, 0] == 0)    # Black
        assert np.all(result[1, 1] == 0)    # Black
        assert np.all(result[0, 1] == 100)  # Unchanged
        assert np.all(result[1, 0] == 100)  # Unchanged

    def test_apply_to_image_with_transparency(self):
        """Apply mask with partial transparency."""
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.array([
            [True, False],
            [False, False]
        ])

        # 50% transparency: blend between original (100) and fill (200)
        config = CloudMaskConfig(transparency=0.5)
        masker = CloudMask(config=config)
        result = masker.apply_to_image(rgb, mask, fill_value=200)

        # Masked pixel should be blended: 0.5 * 100 + 0.5 * 200 = 150
        assert np.all(result[0, 0] == 150)

        # Unmasked pixels unchanged
        assert np.all(result[0, 1] == 100)
        assert np.all(result[1, 0] == 100)

    def test_apply_to_image_full_transparency(self):
        """Apply mask with full transparency (no masking)."""
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)  # All masked

        config = CloudMaskConfig(transparency=1.0)
        masker = CloudMask(config=config)
        result = masker.apply_to_image(rgb, mask, fill_value=255)

        # All pixels should be unchanged (transparency = 1.0)
        assert np.all(result == 100)

    def test_apply_to_image_invalid_rgb_shape(self):
        """RGB image must be 3D with 3 channels."""
        mask = np.zeros((10, 10), dtype=bool)
        masker = CloudMask()

        # 2D grayscale
        rgb_2d = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="rgb must be 3D array with 3 channels"):
            masker.apply_to_image(rgb_2d, mask)

        # 4 channels (RGBA)
        rgb_4ch = np.zeros((10, 10, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="rgb must be 3D array with 3 channels"):
            masker.apply_to_image(rgb_4ch, mask)

    def test_apply_to_image_shape_mismatch(self):
        """RGB and mask shapes must match."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        mask_wrong = np.zeros((5, 5), dtype=bool)

        masker = CloudMask()
        with pytest.raises(ValueError, match="mask shape .* doesn't match rgb spatial shape"):
            masker.apply_to_image(rgb, mask_wrong)

    def test_apply_to_image_invalid_fill_value(self):
        """Fill value must be 0-255."""
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=bool)
        masker = CloudMask()

        with pytest.raises(ValueError, match="fill_value must be 0-255"):
            masker.apply_to_image(rgb, mask, fill_value=-1)

        with pytest.raises(ValueError, match="fill_value must be 0-255"):
            masker.apply_to_image(rgb, mask, fill_value=256)

    def test_get_cloud_percentage_simple(self):
        """Calculate cloud percentage correctly."""
        scl = np.array([
            [4, 8, 9],   # vegetation, cloud medium, cloud high
            [10, 4, 4]   # cirrus, vegetation, vegetation
        ])

        masker = CloudMask()
        percentage = masker.get_cloud_percentage(scl)

        # 3 cloud pixels (8, 9, 10) out of 6 total = 50%
        assert percentage == 50.0

    def test_get_cloud_percentage_no_clouds(self):
        """Cloud percentage is 0 when no clouds present."""
        scl = np.full((10, 10), SCLClass.VEGETATION, dtype=np.uint8)

        masker = CloudMask()
        percentage = masker.get_cloud_percentage(scl)

        assert percentage == 0.0

    def test_get_cloud_percentage_all_clouds(self):
        """Cloud percentage is 100 when all clouds."""
        scl = np.full((10, 10), SCLClass.CLOUD_HIGH_PROBABILITY, dtype=np.uint8)

        masker = CloudMask()
        percentage = masker.get_cloud_percentage(scl)

        assert percentage == 100.0

    def test_get_cloud_percentage_invalid_dimension(self):
        """get_cloud_percentage requires 2D array."""
        scl_1d = np.array([4, 8, 9])

        masker = CloudMask()
        with pytest.raises(ValueError, match="SCL band must be 2D array"):
            masker.get_cloud_percentage(scl_1d)

    def test_get_classification_stats_simple(self):
        """Get classification statistics."""
        scl = np.array([
            [4, 4, 8],   # 2 vegetation, 1 cloud medium
            [9, 10, 6]   # 1 cloud high, 1 cirrus, 1 water
        ])

        masker = CloudMask()
        stats = masker.get_classification_stats(scl)

        # Check specific classes
        assert stats['VEGETATION'] == pytest.approx(33.33, abs=0.01)  # 2/6
        assert stats['WATER'] == pytest.approx(16.67, abs=0.01)       # 1/6
        assert stats['CLOUD_MEDIUM_PROBABILITY'] == pytest.approx(16.67, abs=0.01)  # 1/6
        assert stats['CLOUD_HIGH_PROBABILITY'] == pytest.approx(16.67, abs=0.01)    # 1/6
        assert stats['THIN_CIRRUS'] == pytest.approx(16.67, abs=0.01)              # 1/6

        # All classes should be present
        assert len(stats) == 12

        # All percentages should sum to 100
        assert sum(stats.values()) == pytest.approx(100.0, abs=0.01)

    def test_get_classification_stats_uniform(self):
        """Stats for uniform classification."""
        scl = np.full((10, 10), SCLClass.WATER, dtype=np.uint8)

        masker = CloudMask()
        stats = masker.get_classification_stats(scl)

        # 100% water, 0% everything else
        assert stats['WATER'] == 100.0
        for scl_class in SCLClass:
            if scl_class != SCLClass.WATER:
                assert stats[scl_class.name] == 0.0

    def test_get_classification_stats_invalid_dimension(self):
        """get_classification_stats requires 2D array."""
        scl_3d = np.zeros((10, 10, 3))

        masker = CloudMask()
        with pytest.raises(ValueError, match="SCL band must be 2D array"):
            masker.get_classification_stats(scl_3d)


class TestCloudMaskIntegration:
    """Integration tests combining mask generation and application."""

    def test_full_workflow(self):
        """Complete workflow: SCL -> mask -> apply to RGB."""
        # Create SCL band with clouds
        scl = np.array([
            [4, 4, 8, 9],    # vegetation, vegetation, cloud medium, cloud high
            [10, 4, 4, 4],   # cirrus, vegetation, vegetation, vegetation
            [4, 3, 4, 4],    # vegetation, shadow, vegetation, vegetation
            [4, 4, 11, 4]    # vegetation, vegetation, snow, vegetation
        ])

        # Create RGB image
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:, :, 0] = 100  # Red channel
        rgb[:, :, 1] = 150  # Green channel
        rgb[:, :, 2] = 200  # Blue channel

        # Create masker
        masker = CloudMask()

        # Generate mask
        result = masker.from_scl_band(scl)

        # Verify cloud percentage
        # Clouds: (0,2), (0,3), (1,0) = 3 out of 16 = 18.75%
        assert result.cloud_percentage == 18.75

        # Apply mask
        masked_rgb = masker.apply_to_image(rgb, result.mask, fill_value=255)

        # Check masked pixels are white
        assert np.all(masked_rgb[0, 2] == 255)  # Cloud medium
        assert np.all(masked_rgb[0, 3] == 255)  # Cloud high
        assert np.all(masked_rgb[1, 0] == 255)  # Cirrus

        # Check unmasked pixels are unchanged
        assert np.all(masked_rgb[0, 0] == [100, 150, 200])  # Vegetation
        assert np.all(masked_rgb[1, 1] == [100, 150, 200])  # Vegetation

        # Shadow not masked by default
        assert np.all(masked_rgb[2, 1] == [100, 150, 200])

        # Snow not masked by default
        assert np.all(masked_rgb[3, 2] == [100, 150, 200])

    def test_comprehensive_masking(self):
        """Test all masking options enabled."""
        scl = np.array([
            [8, 9, 10, 3],   # cloud medium, cloud high, cirrus, shadow
            [11, 0, 1, 4]    # snow, no data, saturated, vegetation
        ])

        # Enable all masking options
        config = CloudMaskConfig(
            mask_clouds=True,
            mask_cirrus=True,
            mask_shadows=True,
            mask_snow=True
        )
        masker = CloudMask(config=config)

        result = masker.from_scl_band(scl)

        # All pixels except vegetation should be masked
        expected_mask = np.array([
            [True, True, True, True],    # All masked
            [True, True, True, False]    # Only vegetation unmasked
        ])

        assert np.array_equal(result.mask, expected_mask)

    def test_minimal_masking(self):
        """Test minimal masking (only no data and saturated)."""
        scl = np.array([
            [8, 9, 10, 3],   # clouds, cirrus, shadow
            [11, 0, 1, 4]    # snow, no data, saturated, vegetation
        ])

        # Disable all optional masking
        config = CloudMaskConfig(
            mask_clouds=False,
            mask_cirrus=False,
            mask_shadows=False,
            mask_snow=False
        )
        masker = CloudMask(config=config)

        result = masker.from_scl_band(scl)

        # Only no data (1,1) and saturated (1,2) masked
        expected_mask = np.array([
            [False, False, False, False],
            [False, True, True, False]
        ])

        assert np.array_equal(result.mask, expected_mask)

    def test_large_scene_performance(self):
        """Test with realistic Sentinel-2 scene size."""
        # Sentinel-2 tile is typically 10980x10980 pixels
        # Use smaller size for test but representative
        height, width = 1000, 1000

        # Create SCL with 30% clouds
        scl = np.full((height, width), SCLClass.VEGETATION, dtype=np.uint8)
        num_cloud_pixels = int(height * width * 0.3)

        # Randomly place clouds
        rng = np.random.default_rng(42)
        cloud_indices = rng.choice(height * width, num_cloud_pixels, replace=False)
        scl.ravel()[cloud_indices] = SCLClass.CLOUD_HIGH_PROBABILITY

        # Create masker
        masker = CloudMask()

        # Generate mask (should be fast)
        result = masker.from_scl_band(scl)

        # Verify cloud percentage
        assert result.cloud_percentage == pytest.approx(30.0, abs=0.1)

        # Apply to RGB image
        rgb = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        masked_rgb = masker.apply_to_image(rgb, result.mask)

        # Verify masked pixels
        cloud_pixels = (scl == SCLClass.CLOUD_HIGH_PROBABILITY)
        assert np.all(masked_rgb[cloud_pixels] == 255)
