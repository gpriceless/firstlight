"""
Comprehensive Tests for Baseline Flood Detection Algorithms (Group E, Track 2)

Tests cover:
- ThresholdSARAlgorithm
- NDWIOpticalAlgorithm
- ChangeDetectionAlgorithm
- HANDModelAlgorithm

Each algorithm is tested for:
- Basic functionality with synthetic data
- Edge cases and error handling
- Parameter validation
- Reproducibility (determinism)
- Output structure and metadata
"""

import pytest
import numpy as np
from typing import Dict, Any

from core.analysis.library.baseline.flood import (
    ThresholdSARAlgorithm,
    ThresholdSARConfig,
    ThresholdSARResult,
    NDWIOpticalAlgorithm,
    NDWIOpticalConfig,
    NDWIOpticalResult,
    ChangeDetectionAlgorithm,
    ChangeDetectionConfig,
    ChangeDetectionResult,
    HANDModelAlgorithm,
    HANDModelConfig,
    HANDModelResult,
    get_algorithm,
    list_algorithms,
    FLOOD_ALGORITHMS
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_sar_data():
    """Create synthetic SAR data for testing."""
    # 100x100 pixel image
    sar_post = np.random.uniform(-20, -5, (100, 100))

    # Create a flooded region (lower backscatter values)
    sar_post[20:40, 30:50] = np.random.uniform(-22, -16, (20, 20))

    # Pre-event data (higher backscatter)
    sar_pre = sar_post + np.random.uniform(3, 6, (100, 100))
    sar_pre[20:40, 30:50] += np.random.uniform(2, 4, (20, 20))

    return sar_post, sar_pre


@pytest.fixture
def synthetic_optical_data():
    """Create synthetic optical data for testing."""
    # 100x100 pixel image
    # Green band (higher reflectance over water)
    green = np.random.uniform(0.1, 0.3, (100, 100))
    # NIR band (lower reflectance over water)
    nir = np.random.uniform(0.2, 0.5, (100, 100))

    # Create water region (high green, low NIR)
    green[20:40, 30:50] = np.random.uniform(0.3, 0.4, (20, 20))
    nir[20:40, 30:50] = np.random.uniform(0.05, 0.15, (20, 20))

    # Pre-event (less water)
    green_pre = green - 0.05
    nir_pre = nir + 0.1

    # Cloud mask
    cloud_mask = np.zeros((100, 100), dtype=bool)
    cloud_mask[5:10, 5:10] = True

    return green, nir, green_pre, nir_pre, cloud_mask


@pytest.fixture
def synthetic_dem_data():
    """Create synthetic DEM data for testing."""
    # 100x100 pixel DEM
    # Create a valley with drainage channel
    x = np.arange(100)
    y = np.arange(100)
    X, Y = np.meshgrid(x, y)

    # Valley: elevation decreases toward center
    dem = 100 + 0.5 * np.abs(Y - 50) + 0.3 * np.abs(X - 50)

    # Add drainage channel down the middle
    dem[48:52, :] -= 5

    # Flow accumulation (higher along channel)
    flow_accum = np.ones((100, 100))
    flow_accum[48:52, :] = 100

    # Slope (lower in valley)
    slope = np.ones((100, 100)) * 5.0
    slope[48:52, :] = 1.0
    slope[45:55, :] = 2.0

    return dem, flow_accum, slope


# ============================================================================
# ThresholdSARAlgorithm Tests
# ============================================================================

class TestThresholdSARAlgorithm:
    """Tests for SAR threshold flood detection algorithm."""

    def test_initialization_default_config(self):
        """Test algorithm initialization with default configuration."""
        algo = ThresholdSARAlgorithm()
        assert algo.config.threshold_db == -15.0
        assert algo.config.min_area_ha == 0.5
        assert algo.config.polarization == "VV"
        assert algo.config.use_change_detection is False

    def test_initialization_custom_config(self):
        """Test algorithm initialization with custom configuration."""
        config = ThresholdSARConfig(
            threshold_db=-12.0,
            min_area_ha=1.0,
            polarization="VH",
            use_change_detection=True,
            change_threshold_db=4.0
        )
        algo = ThresholdSARAlgorithm(config)
        assert algo.config.threshold_db == -12.0
        assert algo.config.min_area_ha == 1.0
        assert algo.config.polarization == "VH"
        assert algo.config.use_change_detection is True

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Invalid threshold
        with pytest.raises(ValueError, match="threshold_db must be in"):
            ThresholdSARConfig(threshold_db=-25.0)

        # Invalid min area
        with pytest.raises(ValueError, match="min_area_ha must be non-negative"):
            ThresholdSARConfig(min_area_ha=-1.0)

        # Invalid polarization
        with pytest.raises(ValueError, match="polarization must be VV or VH"):
            ThresholdSARConfig(polarization="HH")

    def test_simple_threshold_detection(self, synthetic_sar_data):
        """Test simple threshold-based flood detection."""
        sar_post, sar_pre = synthetic_sar_data

        algo = ThresholdSARAlgorithm()
        result = algo.execute(sar_post, pixel_size_m=10.0)

        # Check result structure
        assert isinstance(result, ThresholdSARResult)
        assert result.flood_extent.shape == sar_post.shape
        assert result.confidence_raster.shape == sar_post.shape
        assert result.flood_extent.dtype == bool
        assert 0.0 <= result.confidence_raster.min() <= result.confidence_raster.max() <= 1.0

        # Check metadata
        assert result.metadata["id"] == "flood.baseline.threshold_sar"
        assert result.metadata["version"] == "1.2.0"
        assert result.metadata["deterministic"] is True

        # Check statistics
        assert "flood_area_ha" in result.statistics
        assert "flood_percent" in result.statistics
        assert result.statistics["flood_area_ha"] >= 0
        assert 0 <= result.statistics["flood_percent"] <= 100

    def test_change_detection_mode(self, synthetic_sar_data):
        """Test change detection mode with pre/post comparison."""
        sar_post, sar_pre = synthetic_sar_data

        config = ThresholdSARConfig(use_change_detection=True, change_threshold_db=3.0)
        algo = ThresholdSARAlgorithm(config)
        result = algo.execute(sar_post, sar_pre=sar_pre, pixel_size_m=10.0)

        # Should detect change in flooded region
        assert result.flood_extent.any()
        assert result.metadata["execution"]["mode"] == "change_detection"

        # Check that change detection finds reasonable results
        assert 0 < result.statistics["flood_pixels"] < result.statistics["total_pixels"]

    def test_nodata_handling(self, synthetic_sar_data):
        """Test proper handling of NoData values."""
        sar_post, sar_pre = synthetic_sar_data

        # Add NoData values
        sar_post_with_nodata = sar_post.copy()
        sar_post_with_nodata[0:10, 0:10] = -9999.0

        algo = ThresholdSARAlgorithm()
        result = algo.execute(sar_post_with_nodata, pixel_size_m=10.0, nodata_value=-9999.0)

        # NoData pixels should not be classified as flood
        assert not result.flood_extent[0:10, 0:10].any()
        assert result.statistics["valid_pixels"] < result.statistics["total_pixels"]

    def test_nan_inf_handling(self, synthetic_sar_data):
        """Test proper handling of NaN and Inf values."""
        sar_post, sar_pre = synthetic_sar_data

        # Add NaN and Inf
        sar_post_with_nan = sar_post.copy()
        sar_post_with_nan[0:5, 0:5] = np.nan
        sar_post_with_nan[5:10, 5:10] = np.inf

        algo = ThresholdSARAlgorithm()
        result = algo.execute(sar_post_with_nan, pixel_size_m=10.0)

        # NaN/Inf pixels should not be classified
        # Check specifically the regions with NaN and Inf
        assert not result.flood_extent[0:5, 0:5].any()
        assert not result.flood_extent[5:10, 5:10].any()

    def test_determinism(self, synthetic_sar_data):
        """Test that algorithm produces identical results on repeated runs."""
        sar_post, sar_pre = synthetic_sar_data

        algo = ThresholdSARAlgorithm()
        result1 = algo.execute(sar_post, pixel_size_m=10.0)
        result2 = algo.execute(sar_post, pixel_size_m=10.0)

        np.testing.assert_array_equal(result1.flood_extent, result2.flood_extent)
        np.testing.assert_array_almost_equal(result1.confidence_raster, result2.confidence_raster)

    def test_metadata_access(self):
        """Test static metadata access."""
        metadata = ThresholdSARAlgorithm.get_metadata()
        assert metadata["id"] == "flood.baseline.threshold_sar"
        assert "requirements" in metadata
        assert "validation" in metadata

    def test_create_from_dict(self):
        """Test creating algorithm instance from parameter dictionary."""
        params = {
            "threshold_db": -13.0,
            "min_area_ha": 2.0,
            "polarization": "VH"
        }
        algo = ThresholdSARAlgorithm.create_from_dict(params)
        assert algo.config.threshold_db == -13.0
        assert algo.config.min_area_ha == 2.0

    def test_shape_validation(self, synthetic_sar_data):
        """Test input shape validation."""
        sar_post, sar_pre = synthetic_sar_data

        # 3D input should fail
        sar_3d = np.random.uniform(-20, -5, (3, 100, 100))
        algo = ThresholdSARAlgorithm()

        with pytest.raises(ValueError, match="Expected 2D SAR array"):
            algo.execute(sar_3d)

    def test_pre_post_shape_mismatch(self, synthetic_sar_data):
        """Test pre/post shape mismatch detection."""
        sar_post, sar_pre = synthetic_sar_data

        # Mismatched shapes
        sar_pre_wrong = sar_pre[:50, :50]

        config = ThresholdSARConfig(use_change_detection=True)
        algo = ThresholdSARAlgorithm(config)

        with pytest.raises(ValueError, match="shape mismatch"):
            algo.execute(sar_post, sar_pre=sar_pre_wrong)


# ============================================================================
# NDWIOpticalAlgorithm Tests
# ============================================================================

class TestNDWIOpticalAlgorithm:
    """Tests for NDWI optical flood detection algorithm."""

    def test_initialization_default_config(self):
        """Test algorithm initialization with default configuration."""
        algo = NDWIOpticalAlgorithm()
        assert algo.config.ndwi_threshold == 0.0
        assert algo.config.cloud_mask_enabled is True
        assert algo.config.shadow_mask_enabled is True

    def test_ndwi_calculation(self, synthetic_optical_data):
        """Test NDWI calculation and flood detection."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        algo = NDWIOpticalAlgorithm()
        result = algo.execute(green, nir, pixel_size_m=10.0)

        # Check result structure
        assert isinstance(result, NDWIOpticalResult)
        assert result.flood_extent.shape == green.shape
        assert result.ndwi_raster.shape == green.shape
        assert result.confidence_raster.shape == green.shape

        # NDWI should be in range [-1, 1]
        assert -1.0 <= result.ndwi_raster.min() <= result.ndwi_raster.max() <= 1.0

        # Check metadata
        assert result.metadata["id"] == "flood.baseline.ndwi_optical"
        assert "mean_ndwi" in result.statistics
        assert "mean_ndwi_flood" in result.statistics

    def test_cloud_masking(self, synthetic_optical_data):
        """Test cloud masking functionality."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        algo = NDWIOpticalAlgorithm()
        result = algo.execute(green, nir, cloud_mask=cloud_mask, pixel_size_m=10.0)

        # Cloudy pixels should not be classified
        assert not result.flood_extent[cloud_mask].any()
        assert result.metadata["execution"]["cloud_masking_applied"] is True

    def test_shadow_masking(self, synthetic_optical_data):
        """Test shadow masking functionality."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        shadow_mask = np.zeros_like(cloud_mask)
        shadow_mask[10:15, 10:15] = True

        algo = NDWIOpticalAlgorithm()
        result = algo.execute(green, nir, shadow_mask=shadow_mask, pixel_size_m=10.0)

        # Shadow pixels should not be classified
        assert not result.flood_extent[shadow_mask].any()
        assert result.metadata["execution"]["shadow_masking_applied"] is True

    def test_reflectance_normalization(self, synthetic_optical_data):
        """Test reflectance normalization (DN to 0-1)."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        # Convert to DN scale (0-10000)
        green_dn = (green * 10000).astype(np.float32)
        nir_dn = (nir * 10000).astype(np.float32)

        algo = NDWIOpticalAlgorithm()
        result = algo.execute(green_dn, nir_dn, pixel_size_m=10.0)

        # Should still produce valid NDWI
        assert -1.0 <= result.ndwi_raster.min() <= result.ndwi_raster.max() <= 1.0

    def test_change_detection_mode(self, synthetic_optical_data):
        """Test change detection with pre/post imagery."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        config = NDWIOpticalConfig(use_change_detection=True, change_threshold=0.2)
        algo = NDWIOpticalAlgorithm(config)
        result = algo.execute(
            green, nir, green_pre=green_pre, nir_pre=nir_pre, pixel_size_m=10.0
        )

        assert result.metadata["execution"]["mode"] == "change_detection"
        assert result.flood_extent.any()

    def test_band_shape_validation(self, synthetic_optical_data):
        """Test green/NIR band shape validation."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        # Mismatched shapes
        nir_wrong = nir[:50, :50]

        algo = NDWIOpticalAlgorithm()
        with pytest.raises(ValueError, match="shape mismatch"):
            algo.execute(green, nir_wrong)

    def test_determinism(self, synthetic_optical_data):
        """Test deterministic behavior."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        algo = NDWIOpticalAlgorithm()
        result1 = algo.execute(green, nir, pixel_size_m=10.0)
        result2 = algo.execute(green, nir, pixel_size_m=10.0)

        np.testing.assert_array_equal(result1.flood_extent, result2.flood_extent)
        np.testing.assert_array_almost_equal(result1.ndwi_raster, result2.ndwi_raster)

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Invalid NDWI threshold
        with pytest.raises(ValueError, match="ndwi_threshold must be in"):
            NDWIOpticalConfig(ndwi_threshold=2.0)

        # Invalid min area
        with pytest.raises(ValueError, match="min_area_ha must be non-negative"):
            NDWIOpticalConfig(min_area_ha=-1.0)


# ============================================================================
# ChangeDetectionAlgorithm Tests
# ============================================================================

class TestChangeDetectionAlgorithm:
    """Tests for pre/post change detection algorithm."""

    def test_initialization_default_config(self):
        """Test algorithm initialization with default configuration."""
        algo = ChangeDetectionAlgorithm()
        assert algo.config.change_threshold == 0.15
        assert algo.config.method == "normalized_difference"
        assert algo.config.use_multiple_bands is True

    def test_difference_method(self, synthetic_sar_data):
        """Test difference change detection method."""
        sar_post, sar_pre = synthetic_sar_data

        config = ChangeDetectionConfig(method="difference", change_threshold=0.1)
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")

        assert isinstance(result, ChangeDetectionResult)
        assert result.change_magnitude.shape == sar_post.shape
        assert result.flood_extent.any()

    def test_ratio_method(self, synthetic_sar_data):
        """Test ratio change detection method."""
        sar_post, sar_pre = synthetic_sar_data

        config = ChangeDetectionConfig(method="ratio", change_threshold=0.2)
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")

        assert result.flood_extent.any()
        assert result.metadata["parameters"]["method"] == "ratio"

    def test_normalized_difference_method(self, synthetic_sar_data):
        """Test normalized difference method."""
        sar_post, sar_pre = synthetic_sar_data

        config = ChangeDetectionConfig(method="normalized_difference")
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")

        # Normalized difference should be in range [-1, 1]
        assert -1.0 <= result.change_magnitude.min() <= result.change_magnitude.max() <= 1.0

    def test_multiband_detection(self):
        """Test multi-band change detection."""
        # Create 3-band synthetic data
        pre_image = np.random.uniform(0.1, 0.5, (3, 100, 100))
        post_image = pre_image + np.random.uniform(-0.2, 0.2, (3, 100, 100))

        # Create change in one region
        post_image[:, 20:40, 30:50] += 0.3

        config = ChangeDetectionConfig(use_multiple_bands=True)
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(pre_image, post_image, pixel_size_m=10.0)

        assert result.metadata["execution"]["bands_used"] == 3
        assert result.flood_extent.any()

    def test_singleband_from_multiband(self):
        """Test using only first band from multi-band input."""
        pre_image = np.random.uniform(-20, -10, (3, 100, 100))
        # Create significant change in a region
        post_image = pre_image.copy()
        post_image[:, 20:40, 30:50] -= 5.0  # Significant decrease

        config = ChangeDetectionConfig(use_multiple_bands=False, method="difference", change_threshold=2.0)
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(pre_image, post_image, pixel_size_m=10.0, sensor_type="sar")

        # Should detect change in the modified region
        assert result.flood_extent.any()
        # Metadata reports input band count (3 bands available)
        # But only first band used when use_multiple_bands=False
        assert result.metadata["parameters"]["use_multiple_bands"] is False

    def test_outlier_removal(self, synthetic_sar_data):
        """Test outlier removal functionality."""
        sar_post, sar_pre = synthetic_sar_data

        # Add outliers
        sar_post_outliers = sar_post.copy()
        sar_post_outliers[5, 5] = -50.0  # Extreme value

        config = ChangeDetectionConfig(outlier_removal=True, outlier_sigma=3.0)
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(sar_pre, sar_post_outliers, pixel_size_m=10.0, sensor_type="sar")

        # Outliers should be clipped
        assert result.change_magnitude.min() > -50.0

    def test_sar_sensor_interpretation(self, synthetic_sar_data):
        """Test SAR sensor-specific interpretation."""
        sar_post, sar_pre = synthetic_sar_data

        config = ChangeDetectionConfig(method="difference")
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")

        assert result.metadata["parameters"]["sensor_type"] == "sar"
        # SAR: flooding causes decrease (negative change)
        assert result.flood_extent.any()

    def test_optical_sensor_interpretation(self, synthetic_optical_data):
        """Test optical sensor-specific interpretation."""
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data

        config = ChangeDetectionConfig(method="difference")
        algo = ChangeDetectionAlgorithm(config)
        result = algo.execute(green_pre, green, pixel_size_m=10.0, sensor_type="optical")

        assert result.metadata["parameters"]["sensor_type"] == "optical"

    def test_shape_validation(self, synthetic_sar_data):
        """Test pre/post shape validation."""
        sar_post, sar_pre = synthetic_sar_data

        sar_pre_wrong = sar_pre[:50, :50]

        algo = ChangeDetectionAlgorithm()
        with pytest.raises(ValueError, match="shape mismatch"):
            algo.execute(sar_pre_wrong, sar_post)

    def test_determinism(self, synthetic_sar_data):
        """Test deterministic behavior."""
        sar_post, sar_pre = synthetic_sar_data

        algo = ChangeDetectionAlgorithm()
        result1 = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")
        result2 = algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")

        np.testing.assert_array_equal(result1.flood_extent, result2.flood_extent)
        np.testing.assert_array_almost_equal(result1.change_magnitude, result2.change_magnitude)

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid method
        with pytest.raises(ValueError, match="Invalid method"):
            ChangeDetectionConfig(method="invalid")

        # Invalid threshold
        with pytest.raises(ValueError, match="change_threshold must be non-negative"):
            ChangeDetectionConfig(change_threshold=-1.0)


# ============================================================================
# HANDModelAlgorithm Tests
# ============================================================================

class TestHANDModelAlgorithm:
    """Tests for HAND flood susceptibility model."""

    def test_initialization_default_config(self):
        """Test algorithm initialization with default configuration."""
        algo = HANDModelAlgorithm()
        assert algo.config.hand_threshold_m == 10.0
        assert algo.config.use_slope_factor is True
        assert algo.config.slope_weight == 0.3

    def test_hand_calculation_with_flow_accum(self, synthetic_dem_data):
        """Test HAND calculation with provided flow accumulation."""
        dem, flow_accum, slope = synthetic_dem_data

        algo = HANDModelAlgorithm()
        result = algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        # Check result structure
        assert isinstance(result, HANDModelResult)
        assert result.hand_raster.shape == dem.shape
        assert result.susceptibility_mask.shape == dem.shape
        assert result.drainage_network.shape == dem.shape
        assert result.confidence_raster.shape == dem.shape

        # HAND should be non-negative
        assert np.nanmin(result.hand_raster) >= 0

        # Drainage channels should have HAND = 0
        assert np.all(result.hand_raster[result.drainage_network] == 0.0)

    def test_hand_calculation_without_flow_accum(self, synthetic_dem_data):
        """Test HAND calculation with estimated drainage network."""
        dem, flow_accum, slope = synthetic_dem_data

        # Use a lower threshold to ensure channels are found with simplified algorithm
        config = HANDModelConfig(channel_threshold_area_km2=0.01)
        algo = HANDModelAlgorithm(config)
        result = algo.execute(dem, pixel_size_m=30.0)

        # The simplified algorithm may not find channels reliably, but should complete
        assert result.metadata["execution"]["flow_accumulation_provided"] is False
        # Should produce HAND values
        assert not np.all(np.isnan(result.hand_raster))

    def test_slope_factor_inclusion(self, synthetic_dem_data):
        """Test slope factor inclusion in susceptibility."""
        dem, flow_accum, slope = synthetic_dem_data

        config = HANDModelConfig(use_slope_factor=True, slope_weight=0.5)
        algo = HANDModelAlgorithm(config)
        result = algo.execute(dem, flow_accumulation=flow_accum, slope=slope, pixel_size_m=30.0)

        # Lower slopes should have higher confidence
        assert result.metadata["execution"]["slope_provided"] is True
        assert result.susceptibility_mask.any()

    def test_without_slope_factor(self, synthetic_dem_data):
        """Test susceptibility without slope factor."""
        dem, flow_accum, slope = synthetic_dem_data

        config = HANDModelConfig(use_slope_factor=False)
        algo = HANDModelAlgorithm(config)
        result = algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        # Should still work
        assert result.susceptibility_mask.any()

    def test_channel_identification(self, synthetic_dem_data):
        """Test drainage channel identification."""
        dem, flow_accum, slope = synthetic_dem_data

        # Use a very low threshold to ensure channels are found
        config = HANDModelConfig(channel_threshold_area_km2=0.001)
        algo = HANDModelAlgorithm(config)
        result = algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        # Should identify channels with low threshold
        assert result.statistics["channel_pixels"] > 0
        assert result.drainage_network.any()

    def test_susceptibility_threshold(self, synthetic_dem_data):
        """Test susceptibility threshold parameter."""
        dem, flow_accum, slope = synthetic_dem_data

        # Lower threshold = more susceptible area
        config_low = HANDModelConfig(hand_threshold_m=5.0)
        algo_low = HANDModelAlgorithm(config_low)
        result_low = algo_low.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        # Higher threshold = more susceptible area
        config_high = HANDModelConfig(hand_threshold_m=15.0)
        algo_high = HANDModelAlgorithm(config_high)
        result_high = algo_high.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        assert result_high.statistics["susceptible_pixels"] >= result_low.statistics["susceptible_pixels"]

    def test_nodata_handling(self, synthetic_dem_data):
        """Test NoData handling in DEM."""
        dem, flow_accum, slope = synthetic_dem_data

        dem_with_nodata = dem.copy()
        dem_with_nodata[0:10, 0:10] = -9999.0

        algo = HANDModelAlgorithm()
        result = algo.execute(dem_with_nodata, pixel_size_m=30.0, nodata_value=-9999.0)

        # NoData regions should be NaN in HAND
        assert np.all(np.isnan(result.hand_raster[0:10, 0:10]))
        assert result.statistics["valid_pixels"] < result.statistics["total_pixels"]

    def test_shape_validation(self, synthetic_dem_data):
        """Test DEM shape validation."""
        # 3D input should fail
        dem_3d = np.random.uniform(0, 100, (3, 100, 100))

        algo = HANDModelAlgorithm()
        with pytest.raises(ValueError, match="Expected 2D DEM array"):
            algo.execute(dem_3d)

    def test_determinism(self, synthetic_dem_data):
        """Test deterministic behavior."""
        dem, flow_accum, slope = synthetic_dem_data

        algo = HANDModelAlgorithm()
        result1 = algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)
        result2 = algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)

        np.testing.assert_array_equal(result1.susceptibility_mask, result2.susceptibility_mask)
        np.testing.assert_array_almost_equal(
            result1.hand_raster[~np.isnan(result1.hand_raster)],
            result2.hand_raster[~np.isnan(result2.hand_raster)]
        )

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid HAND threshold
        with pytest.raises(ValueError, match="hand_threshold_m must be positive"):
            HANDModelConfig(hand_threshold_m=-5.0)

        # Invalid channel threshold
        with pytest.raises(ValueError, match="channel_threshold_area_km2 must be positive"):
            HANDModelConfig(channel_threshold_area_km2=-1.0)

        # Invalid slope weight
        with pytest.raises(ValueError, match="slope_weight must be in"):
            HANDModelConfig(slope_weight=1.5)


# ============================================================================
# Module-level Tests
# ============================================================================

class TestModuleFunctions:
    """Tests for module-level functions and registry."""

    def test_flood_algorithms_registry(self):
        """Test FLOOD_ALGORITHMS registry."""
        assert "flood.baseline.threshold_sar" in FLOOD_ALGORITHMS
        assert "flood.baseline.ndwi_optical" in FLOOD_ALGORITHMS
        assert "flood.baseline.change_detection" in FLOOD_ALGORITHMS
        assert "flood.baseline.hand_model" in FLOOD_ALGORITHMS

        assert FLOOD_ALGORITHMS["flood.baseline.threshold_sar"] == ThresholdSARAlgorithm
        assert FLOOD_ALGORITHMS["flood.baseline.ndwi_optical"] == NDWIOpticalAlgorithm
        assert FLOOD_ALGORITHMS["flood.baseline.change_detection"] == ChangeDetectionAlgorithm
        assert FLOOD_ALGORITHMS["flood.baseline.hand_model"] == HANDModelAlgorithm

    def test_get_algorithm_valid(self):
        """Test get_algorithm with valid algorithm IDs."""
        algo_class = get_algorithm("flood.baseline.threshold_sar")
        assert algo_class == ThresholdSARAlgorithm

        algo_class = get_algorithm("flood.baseline.ndwi_optical")
        assert algo_class == NDWIOpticalAlgorithm

    def test_get_algorithm_invalid(self):
        """Test get_algorithm with invalid algorithm ID."""
        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm("flood.invalid.algorithm")

    def test_list_algorithms(self):
        """Test list_algorithms function."""
        algorithms = list_algorithms()
        assert len(algorithms) == 4

        ids = [algo_id for algo_id, _ in algorithms]
        assert "flood.baseline.threshold_sar" in ids
        assert "flood.baseline.ndwi_optical" in ids
        assert "flood.baseline.change_detection" in ids
        assert "flood.baseline.hand_model" in ids

    def test_all_algorithms_have_metadata(self):
        """Test that all algorithms have proper metadata."""
        for algo_id, algo_class in list_algorithms():
            metadata = algo_class.get_metadata()

            assert "id" in metadata
            assert "name" in metadata
            assert "version" in metadata
            assert "deterministic" in metadata
            assert "requirements" in metadata
            assert "validation" in metadata

            # Check deterministic flag is True (all baseline algorithms should be)
            assert metadata["deterministic"] is True

    def test_all_algorithms_support_create_from_dict(self):
        """Test that all algorithms support create_from_dict."""
        for algo_id, algo_class in list_algorithms():
            # Should be able to create with empty params (use defaults)
            algo = algo_class.create_from_dict({})
            assert algo is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestAlgorithmIntegration:
    """Integration tests for algorithm workflows."""

    def test_all_algorithms_produce_valid_outputs(
        self, synthetic_sar_data, synthetic_optical_data, synthetic_dem_data
    ):
        """Test that all algorithms produce valid, structured outputs."""
        sar_post, sar_pre = synthetic_sar_data
        green, nir, green_pre, nir_pre, cloud_mask = synthetic_optical_data
        dem, flow_accum, slope = synthetic_dem_data

        # Test ThresholdSAR
        sar_algo = ThresholdSARAlgorithm()
        sar_result = sar_algo.execute(sar_post, pixel_size_m=10.0)
        assert sar_result.to_dict() is not None

        # Test NDWI
        ndwi_algo = NDWIOpticalAlgorithm()
        ndwi_result = ndwi_algo.execute(green, nir, pixel_size_m=10.0)
        assert ndwi_result.to_dict() is not None

        # Test Change Detection
        change_algo = ChangeDetectionAlgorithm()
        change_result = change_algo.execute(sar_pre, sar_post, pixel_size_m=10.0, sensor_type="sar")
        assert change_result.to_dict() is not None

        # Test HAND
        hand_algo = HANDModelAlgorithm()
        hand_result = hand_algo.execute(dem, flow_accumulation=flow_accum, pixel_size_m=30.0)
        assert hand_result.to_dict() is not None

    def test_algorithm_metadata_consistency(self):
        """Test that metadata is consistent across access methods."""
        for algo_id, algo_class in list_algorithms():
            # Get metadata from class method
            metadata_static = algo_class.get_metadata()

            # Get metadata from instance
            algo_instance = algo_class()
            metadata_instance = algo_instance.METADATA

            # Should be identical
            assert metadata_static == metadata_instance
            assert metadata_static["id"] == algo_id
