"""
Integration Tests for VIS-1.1 Imagery Rendering Pipeline.

Tests the complete end-to-end workflow of the imagery rendering system,
verifying that all modules work together correctly:
- Band combinations
- Histogram stretching
- Image rendering
- SAR processing
- Cloud masking

Tests real-world scenarios, graceful degradation, performance, and interoperability.

Part of VIS-1.1 Task 4.2: Integration Testing
"""

import time
from pathlib import Path

import numpy as np
import pytest

# Import from specific modules to avoid export dependencies (PIL)
from core.reporting.imagery.band_combinations import (
    BandComposite,
    get_bands_for_composite,
)
from core.reporting.imagery.cloud_mask import CloudMask, CloudMaskConfig, SCLClass
from core.reporting.imagery.histogram import HistogramStretch
from core.reporting.imagery.renderer import ImageryRenderer, RendererConfig
from core.reporting.imagery.sar_processor import SARConfig, SARProcessor
from core.reporting.imagery.utils import (
    detect_partial_coverage,
    normalize_to_uint8,
    stack_bands_to_rgb,
)


class TestEndToEndOpticalWorkflow:
    """End-to-end tests for Sentinel-2 optical imagery workflow."""

    def create_sentinel2_scene(self, shape=(1000, 1000), include_scl=False):
        """Create synthetic Sentinel-2 scene with realistic reflectance values."""
        rng = np.random.default_rng(42)

        bands = {
            "B02": rng.normal(0.08, 0.02, shape).clip(0, 1),  # Blue
            "B03": rng.normal(0.12, 0.03, shape).clip(0, 1),  # Green
            "B04": rng.normal(0.15, 0.04, shape).clip(0, 1),  # Red
            "B08": rng.normal(0.35, 0.08, shape).clip(0, 1),  # NIR
        }

        if include_scl:
            # Create SCL band with mostly vegetation (4), some clouds (8, 9)
            scl = np.full(shape, SCLClass.VEGETATION, dtype=np.uint8)
            # Add some clouds in upper-left corner
            scl[0:100, 0:100] = SCLClass.CLOUD_HIGH_PROBABILITY
            # Add some cirrus
            scl[100:150, 0:50] = SCLClass.THIN_CIRRUS
            bands["SCL"] = scl

        return bands

    def test_sentinel2_true_color_complete_workflow(self):
        """Test complete Sentinel-2 true color rendering workflow."""
        # Create synthetic scene
        bands = self.create_sentinel2_scene(shape=(500, 500))

        # Configure renderer
        config = RendererConfig(
            composite_name="true_color",
            stretch_method=HistogramStretch.LINEAR_2PCT,
            output_format="uint8"
        )
        renderer = ImageryRenderer(config)

        # Render
        result = renderer.render(bands, sensor="sentinel2")

        # Verify output structure
        assert result.rgb_array.shape == (500, 500, 3)
        assert result.rgb_array.dtype == np.uint8
        assert result.valid_mask.shape == (500, 500)
        assert result.valid_mask.dtype == bool

        # Verify metadata
        assert result.metadata["sensor"] == "sentinel2"
        assert result.metadata["composite_name"] == "true_color"
        assert result.metadata["bands_used"] == ("B04", "B03", "B02")
        assert result.metadata["stretch_method"] == "linear_2pct"
        assert result.metadata["output_format"] == "uint8"

        # Verify coverage near 100%
        assert result.metadata["coverage_percent"] > 99.0

        # Verify RGB values are distributed across range
        assert result.rgb_array.min() < 50  # Has dark values
        assert result.rgb_array.max() > 200  # Has bright values

    def test_sentinel2_false_color_ir_workflow(self):
        """Test Sentinel-2 false color infrared workflow."""
        bands = self.create_sentinel2_scene(shape=(400, 400))

        config = RendererConfig(
            composite_name="false_color_ir",
            stretch_method=HistogramStretch.LINEAR_2PCT,
            output_format="uint8"
        )
        renderer = ImageryRenderer(config)

        result = renderer.render(bands, sensor="sentinel2")

        # Verify composite used
        assert result.metadata["composite_name"] == "false_color_ir"
        assert result.metadata["bands_used"] == ("B08", "B04", "B03")

        # Verify output shape and type
        assert result.rgb_array.shape == (400, 400, 3)
        assert result.rgb_array.dtype == np.uint8

        # In false color IR, vegetation should appear red (high NIR)
        # Since B08 (NIR) is mapped to red channel and has higher values
        # Red channel should have higher average than other channels
        red_mean = result.rgb_array[:, :, 0].mean()
        green_mean = result.rgb_array[:, :, 1].mean()
        blue_mean = result.rgb_array[:, :, 2].mean()

        # Red channel (NIR) should be brighter due to higher NIR reflectance
        assert red_mean > green_mean
        assert red_mean > blue_mean

    def test_sentinel2_with_nodata_regions(self):
        """Test Sentinel-2 rendering with nodata regions."""
        bands = self.create_sentinel2_scene(shape=(300, 300))

        # Add nodata regions (simulating partial coverage)
        nodata_value = -9999.0
        bands["B02"][0:50, 0:50] = nodata_value
        bands["B03"][0:50, 0:50] = nodata_value
        bands["B04"][0:50, 0:50] = nodata_value
        bands["B08"][0:50, 0:50] = nodata_value

        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2", nodata_value=nodata_value)

        # Verify coverage is reduced
        expected_coverage = (300*300 - 50*50) / (300*300) * 100
        assert result.metadata["coverage_percent"] == pytest.approx(expected_coverage, rel=0.01)

        # Verify valid mask excludes nodata region
        assert not result.valid_mask[25, 25]  # Inside nodata region
        assert result.valid_mask[100, 100]  # Outside nodata region


class TestEndToEndSARWorkflow:
    """End-to-end tests for Sentinel-1 SAR workflow."""

    def create_sentinel1_scene(self, shape=(1000, 1000)):
        """Create synthetic Sentinel-1 SAR scene with realistic backscatter values."""
        rng = np.random.default_rng(42)

        # Sentinel-1 linear backscatter values
        # Typical range: 0.0001 to 1.0 (corresponds to -40 to 0 dB)
        vv = rng.lognormal(mean=-3.5, sigma=0.5, size=shape).clip(0.0001, 1.0)
        vh = rng.lognormal(mean=-4.5, sigma=0.5, size=shape).clip(0.0001, 0.5)

        return {"VV": vv, "VH": vh}

    def test_sar_vv_single_pol_workflow(self):
        """Test SAR VV single polarization workflow."""
        bands = self.create_sentinel1_scene(shape=(500, 500))

        config = SARConfig(
            visualization="vv",
            db_min=-20.0,
            db_max=5.0,
            apply_speckle_filter=False
        )
        processor = SARProcessor(config)

        result = processor.process(bands["VV"])

        # Verify output
        assert result.rgb_array.shape == (500, 500)
        assert result.rgb_array.dtype == np.float32
        assert np.all((result.rgb_array >= 0) & (result.rgb_array <= 1))

        # Verify metadata
        assert result.metadata["visualization"] == "vv"
        assert result.metadata["db_range"] == (-20.0, 5.0)
        assert result.metadata["speckle_filtered"] is False

        # Verify valid mask
        assert result.valid_mask.shape == (500, 500)
        assert result.valid_mask.dtype == bool
        assert np.sum(result.valid_mask) > 0.99 * 500 * 500  # Most pixels valid

    def test_sar_dual_pol_workflow(self):
        """Test SAR dual polarization RGB composite workflow."""
        bands = self.create_sentinel1_scene(shape=(400, 400))

        config = SARConfig(
            visualization="dual_pol",
            db_min=-20.0,
            db_max=5.0
        )
        processor = SARProcessor(config)

        result = processor.process(bands["VV"], bands["VH"])

        # Verify RGB output
        assert result.rgb_array.shape == (400, 400, 3)
        assert result.rgb_array.dtype == np.float32
        assert np.all((result.rgb_array >= 0) & (result.rgb_array <= 1))

        # Verify metadata
        assert result.metadata["visualization"] == "dual_pol"

        # All channels should have some data (not all zeros)
        for channel in range(3):
            channel_mean = result.rgb_array[:, :, channel].mean()
            assert channel_mean > 0.0  # Has content

        # VV and VH channels should have reasonable values
        # (ratio channel might be lower, VH can be quite dark)
        assert result.rgb_array[:, :, 0].mean() > 0.03  # VV channel
        assert result.rgb_array[:, :, 1].mean() > 0.03  # VH channel

    def test_sar_with_speckle_filtering(self):
        """Test SAR processing with speckle filtering."""
        bands = self.create_sentinel1_scene(shape=(200, 200))

        # Process without filtering
        config_no_filter = SARConfig(visualization="vv", apply_speckle_filter=False)
        processor_no_filter = SARProcessor(config_no_filter)
        result_no_filter = processor_no_filter.process(bands["VV"])

        # Process with filtering
        config_filter = SARConfig(visualization="vv", apply_speckle_filter=True)
        processor_filter = SARProcessor(config_filter)
        result_filter = processor_filter.process(bands["VV"])

        # Filtered result should have lower variance
        var_no_filter = np.var(result_no_filter.rgb_array[result_no_filter.valid_mask])
        var_filter = np.var(result_filter.rgb_array[result_filter.valid_mask])

        assert var_filter < var_no_filter

        # Metadata should reflect filtering
        assert result_filter.metadata["speckle_filtered"] is True


class TestGracefulDegradation:
    """Tests for graceful degradation when data is incomplete."""

    def test_missing_swir_bands_fallback_to_true_color(self):
        """Test fallback from SWIR to true color when SWIR bands missing."""
        # Create bands without SWIR (B11, B12, B8A)
        bands = {
            "B02": np.random.rand(200, 200) * 0.2,
            "B03": np.random.rand(200, 200) * 0.25,
            "B04": np.random.rand(200, 200) * 0.3,
            "B08": np.random.rand(200, 200) * 0.4,
        }

        # Request SWIR composite
        config = RendererConfig(composite_name="swir")
        renderer = ImageryRenderer(config)

        # Should fall back to true_color
        result = renderer.render(bands, sensor="sentinel2")

        assert result.metadata["composite_name"] == "true_color"
        assert result.metadata["bands_used"] == ("B04", "B03", "B02")
        assert result.rgb_array.shape == (200, 200, 3)

    def test_partial_coverage_handling(self):
        """Test handling of partial coverage with nodata."""
        bands = {
            "B02": np.random.rand(300, 300) * 0.2,
            "B03": np.random.rand(300, 300) * 0.25,
            "B04": np.random.rand(300, 300) * 0.3,
        }

        # Set half the scene to nodata
        nodata = -9999.0
        bands["B02"][150:, :] = nodata
        bands["B03"][150:, :] = nodata
        bands["B04"][150:, :] = nodata

        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2", nodata_value=nodata)

        # Verify coverage percentage
        coverage = detect_partial_coverage(bands["B04"], nodata_value=nodata)
        assert coverage == pytest.approx(50.0, rel=0.01)
        assert result.metadata["coverage_percent"] == pytest.approx(50.0, rel=0.01)

        # Verify valid mask
        assert not result.valid_mask[200, 100]  # In nodata region
        assert result.valid_mask[100, 100]  # In valid region

    def test_no_valid_composite_available_error(self):
        """Test error when no valid composite can be created."""
        # Only provide NIR band, missing all visible bands
        bands = {"B08": np.random.rand(100, 100)}

        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Cannot render composite.*missing bands"):
            renderer.render(bands, sensor="sentinel2")


class TestCloudMaskingIntegration:
    """Tests for cloud masking integration with optical imagery."""

    def test_cloud_mask_generation_and_application(self):
        """Test generating cloud mask from SCL and applying to RGB."""
        # Create Sentinel-2 scene with SCL
        rng = np.random.default_rng(42)
        shape = (300, 300)

        bands = {
            "B02": rng.normal(0.08, 0.02, shape).clip(0, 1),
            "B03": rng.normal(0.12, 0.03, shape).clip(0, 1),
            "B04": rng.normal(0.15, 0.04, shape).clip(0, 1),
        }

        # Create SCL with clouds
        scl = np.full(shape, SCLClass.VEGETATION, dtype=np.uint8)
        # Add 100x100 cloud region (11.1% of 300x300 scene)
        scl[0:100, 0:100] = SCLClass.CLOUD_HIGH_PROBABILITY
        scl[100:130, 0:30] = SCLClass.THIN_CIRRUS  # Additional 1%

        # Render to RGB
        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2")

        # Apply cloud mask
        masker = CloudMask()
        mask_result = masker.from_scl_band(scl)
        masked_rgb = masker.apply_to_image(result.rgb_array, mask_result.mask, fill_value=255)

        # Verify cloud percentage (100*100 + 30*30) / (300*300) * 100 = 12.1%
        assert mask_result.cloud_percentage > 10.0
        assert mask_result.cloud_percentage < 15.0

        # Verify masked pixels are white (255)
        assert np.all(masked_rgb[50, 50] == 255)  # In cloud region
        assert not np.all(masked_rgb[200, 200] == 255)  # In clear region

        # Verify mask shape matches RGB
        assert masked_rgb.shape == result.rgb_array.shape

    def test_cloud_mask_with_partial_transparency(self):
        """Test cloud masking with partial transparency."""
        shape = (200, 200)
        rgb = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        scl = np.full(shape, SCLClass.VEGETATION, dtype=np.uint8)
        scl[50:100, 50:100] = SCLClass.CLOUD_HIGH_PROBABILITY

        # Create mask with 50% transparency
        config = CloudMaskConfig(mask_clouds=True, transparency=0.5)
        masker = CloudMask(config)

        mask_result = masker.from_scl_band(scl)
        masked_rgb = masker.apply_to_image(rgb, mask_result.mask, fill_value=255)

        # Masked pixels should be blend of original and white
        # 50% transparency means 0.5 * original + 0.5 * 255
        original_pixel = rgb[75, 75]
        masked_pixel = masked_rgb[75, 75]

        expected = (0.5 * original_pixel + 0.5 * 255).astype(np.uint8)
        np.testing.assert_array_equal(masked_pixel, expected)


class TestPerformanceBenchmarks:
    """Performance benchmark tests to ensure < 5 second target."""

    def test_optical_rendering_performance(self):
        """Test optical rendering meets performance target."""
        # Create 1000x1000 scene (approximately 100 km² at 10m resolution)
        bands = {
            "B02": np.random.rand(1000, 1000) * 0.2,
            "B03": np.random.rand(1000, 1000) * 0.25,
            "B04": np.random.rand(1000, 1000) * 0.3,
        }

        renderer = ImageryRenderer()

        # Time the rendering
        start = time.time()
        result = renderer.render(bands, sensor="sentinel2")
        elapsed = time.time() - start

        # Verify performance target (< 5 seconds)
        assert elapsed < 5.0, f"Rendering took {elapsed:.2f}s, exceeds 5s target"

        # Verify result is valid
        assert result.rgb_array.shape == (1000, 1000, 3)

    def test_sar_processing_performance(self):
        """Test SAR processing meets performance target."""
        # Create 1000x1000 SAR scene
        vv = np.random.lognormal(mean=-3.5, sigma=0.5, size=(1000, 1000)).clip(0.0001, 1.0)
        vh = np.random.lognormal(mean=-4.5, sigma=0.5, size=(1000, 1000)).clip(0.0001, 0.5)

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)

        # Time the processing
        start = time.time()
        result = processor.process(vv, vh)
        elapsed = time.time() - start

        # Verify performance target
        assert elapsed < 5.0, f"SAR processing took {elapsed:.2f}s, exceeds 5s target"

        # Verify result is valid
        assert result.rgb_array.shape == (1000, 1000, 3)

    def test_cloud_masking_performance(self):
        """Test cloud masking meets performance target."""
        scl = np.random.randint(0, 12, size=(1000, 1000), dtype=np.uint8)
        rgb = np.random.randint(0, 256, size=(1000, 1000, 3), dtype=np.uint8)

        masker = CloudMask()

        # Time the masking
        start = time.time()
        mask_result = masker.from_scl_band(scl)
        masked_rgb = masker.apply_to_image(rgb, mask_result.mask)
        elapsed = time.time() - start

        # Should be very fast (< 1 second)
        assert elapsed < 1.0, f"Cloud masking took {elapsed:.2f}s, exceeds 1s target"


class TestModuleInteroperability:
    """Tests verifying all modules work together correctly."""

    def test_all_modules_import_correctly(self):
        """Test that all modules import without errors."""
        # If we got here, all imports at the top succeeded
        assert True

    def test_dataclass_compatibility(self):
        """Test dataclass compatibility across modules."""
        # Create instances of all dataclasses
        from core.reporting.imagery.band_combinations import BandComposite
        from core.reporting.imagery.renderer import RenderedImage, RendererConfig
        from core.reporting.imagery.sar_processor import SARConfig, SARResult
        from core.reporting.imagery.cloud_mask import CloudMaskConfig, CloudMaskResult

        # BandComposite
        composite = BandComposite(
            name="test",
            bands=("B04", "B03", "B02"),
            description="Test composite",
            sensor="sentinel2"
        )
        assert composite.name == "test"

        # RendererConfig
        config = RendererConfig(composite_name="true_color")
        assert config.composite_name == "true_color"

        # RenderedImage
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rendered = RenderedImage(rgb_array=rgb)
        assert rendered.rgb_array.shape == (10, 10, 3)

        # SARConfig
        sar_config = SARConfig(visualization="vv")
        assert sar_config.visualization == "vv"

        # SARResult
        sar_result = SARResult(
            rgb_array=np.zeros((10, 10)),
            metadata={},
            valid_mask=np.ones((10, 10), dtype=bool)
        )
        assert sar_result.rgb_array.shape == (10, 10)

        # CloudMaskConfig
        cloud_config = CloudMaskConfig(mask_clouds=True)
        assert cloud_config.mask_clouds is True

        # CloudMaskResult
        cloud_result = CloudMaskResult(
            mask=np.zeros((10, 10), dtype=bool),
            cloud_percentage=10.0,
            classes_masked=[SCLClass.CLOUD_HIGH_PROBABILITY]
        )
        assert cloud_result.cloud_percentage == 10.0

    def test_rendered_image_passes_through_pipeline(self):
        """Test RenderedImage can be used in subsequent processing."""
        # Render optical imagery
        bands = {
            "B02": np.random.rand(100, 100) * 0.2,
            "B03": np.random.rand(100, 100) * 0.25,
            "B04": np.random.rand(100, 100) * 0.3,
        }
        renderer = ImageryRenderer()
        result = renderer.render(bands, sensor="sentinel2")

        # Use the rendered image for further processing
        # Convert to normalized format
        normalized = result.rgb_array.astype(np.float32) / 255.0
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        # Stack with valid mask
        stacked = np.dstack([normalized, result.valid_mask.astype(np.float32)])
        assert stacked.shape == (100, 100, 4)

    def test_utility_functions_interoperate(self):
        """Test utility functions work together."""
        from core.reporting.imagery.utils import (
            detect_partial_coverage,
            get_valid_data_bounds,
            handle_nodata,
            normalize_to_uint8,
            stack_bands_to_rgb,
        )

        # Create test data
        band = np.random.rand(100, 100)
        band[0:20, 0:20] = -9999  # Nodata region (400 pixels out of 10000 = 4%)

        # Detect coverage
        coverage = detect_partial_coverage(band, nodata_value=-9999)
        # Expected: (10000 - 400) / 10000 * 100 = 96%
        assert 95 < coverage < 97

        # Handle nodata
        filled, mask = handle_nodata(band, nodata_value=-9999)
        assert not mask[10, 10]  # Nodata region
        assert mask[50, 50]  # Valid region

        # Get valid bounds (nodata only covers top-left corner, not whole rows)
        bounds = get_valid_data_bounds(band, nodata_value=-9999)
        # Should span entire array since valid data exists in all rows/cols
        # (nodata is only in 0:20, 0:20 corner)
        assert bounds == (0, 100, 0, 100)  # Full bounds

        # Stack bands
        r = g = b = band.clip(0, 1)
        rgb_float = stack_bands_to_rgb(r, g, b)
        assert rgb_float.shape == (100, 100, 3)

        # Normalize to uint8
        rgb_uint8 = normalize_to_uint8(rgb_float, 0.0, 1.0)
        assert rgb_uint8.dtype == np.uint8
        assert rgb_uint8.shape == (100, 100, 3)


class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns."""

    def test_complete_sentinel2_to_png_workflow(self):
        """Test complete workflow from Sentinel-2 bands to PNG-ready RGB."""
        # Create realistic Sentinel-2 scene
        rng = np.random.default_rng(42)
        shape = (512, 512)

        bands = {
            "B02": rng.normal(0.08, 0.02, shape).clip(0, 1),
            "B03": rng.normal(0.12, 0.03, shape).clip(0, 1),
            "B04": rng.normal(0.15, 0.04, shape).clip(0, 1),
            "B08": rng.normal(0.35, 0.08, shape).clip(0, 1),
        }

        # Add partial nodata
        bands["B02"][0:50, 0:50] = np.nan
        bands["B03"][0:50, 0:50] = np.nan
        bands["B04"][0:50, 0:50] = np.nan

        # Create SCL for cloud masking
        scl = np.full(shape, SCLClass.VEGETATION, dtype=np.uint8)
        scl[100:200, 100:200] = SCLClass.CLOUD_HIGH_PROBABILITY

        # Step 1: Render to RGB
        config = RendererConfig(
            composite_name="true_color",
            stretch_method=HistogramStretch.LINEAR_2PCT,
            output_format="uint8"
        )
        renderer = ImageryRenderer(config)
        result = renderer.render(bands, sensor="sentinel2")

        # Step 2: Apply cloud mask
        masker = CloudMask(CloudMaskConfig(mask_clouds=True))
        mask_result = masker.from_scl_band(scl)
        final_rgb = masker.apply_to_image(result.rgb_array, mask_result.mask, fill_value=255)

        # Verify final output is PNG-ready
        assert final_rgb.dtype == np.uint8
        assert final_rgb.shape == (512, 512, 3)
        assert final_rgb.min() >= 0
        assert final_rgb.max() <= 255

        # Verify nodata region handled
        assert not result.valid_mask[25, 25]

        # Verify cloud region masked
        assert np.all(final_rgb[150, 150] == 255)

    def test_sentinel1_flood_detection_workflow(self):
        """Test realistic flood detection workflow with Sentinel-1."""
        rng = np.random.default_rng(42)
        shape = (400, 400)

        # Create scene with normal land and flooded areas
        vv = rng.lognormal(mean=-3.0, sigma=0.3, size=shape).clip(0.0001, 1.0)
        vh = rng.lognormal(mean=-4.0, sigma=0.3, size=shape).clip(0.0001, 0.5)

        # Add flooded region (very low backscatter)
        vv[150:250, 150:250] = rng.uniform(0.0001, 0.001, size=(100, 100))
        vh[150:250, 150:250] = rng.uniform(0.00005, 0.0005, size=(100, 100))

        # Process for flood detection
        config = SARConfig(
            visualization="flood_detection",
            db_min=-20.0,
            db_max=5.0,
            apply_speckle_filter=True
        )
        processor = SARProcessor(config)
        result = processor.process(vv, vh)

        # Verify flood region is darker than land
        flood_brightness = result.rgb_array[200, 200].mean()
        land_brightness = result.rgb_array[50, 50].mean()

        assert flood_brightness < land_brightness * 0.7

        # Convert to uint8 for visualization
        rgb_uint8 = (result.rgb_array * 255).astype(np.uint8)
        assert rgb_uint8.dtype == np.uint8
        assert rgb_uint8.shape == (400, 400, 3)

    def test_multi_composite_comparison(self):
        """Test rendering same scene with multiple composites."""
        bands = {
            "B02": np.random.rand(200, 200) * 0.2,
            "B03": np.random.rand(200, 200) * 0.25,
            "B04": np.random.rand(200, 200) * 0.3,
            "B08": np.random.rand(200, 200) * 0.4,
        }

        # Render with different composites
        composites = ["true_color", "false_color_ir"]
        results = {}

        for composite_name in composites:
            config = RendererConfig(composite_name=composite_name)
            renderer = ImageryRenderer(config)
            result = renderer.render(bands, sensor="sentinel2")
            results[composite_name] = result

        # Verify all results are valid but different
        assert results["true_color"].rgb_array.shape == (200, 200, 3)
        assert results["false_color_ir"].rgb_array.shape == (200, 200, 3)

        # Results should be different (different band combinations)
        assert not np.array_equal(
            results["true_color"].rgb_array,
            results["false_color_ir"].rgb_array
        )

        # Metadata should differ
        assert results["true_color"].metadata["bands_used"] == ("B04", "B03", "B02")
        assert results["false_color_ir"].metadata["bands_used"] == ("B08", "B04", "B03")


class TestErrorHandling:
    """Tests for comprehensive error handling across pipeline."""

    def test_invalid_sensor_raises_clear_error(self):
        """Test clear error message for invalid sensor."""
        bands = {"B02": np.random.rand(10, 10)}
        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="Unsupported sensor"):
            renderer.render(bands, sensor="invalid_sensor_name")

    def test_missing_required_bands_raises_clear_error(self):
        """Test clear error message for missing bands."""
        bands = {"B02": np.random.rand(10, 10)}  # Only blue, missing red and green
        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="missing bands.*B04.*B03"):
            renderer.render(bands, sensor="sentinel2")

    def test_mismatched_shapes_raises_clear_error(self):
        """Test clear error message for mismatched band shapes."""
        bands = {
            "B02": np.random.rand(100, 100),
            "B03": np.random.rand(100, 100),
            "B04": np.random.rand(50, 50),  # Different shape
        }
        renderer = ImageryRenderer()

        with pytest.raises(ValueError, match="shapes must match"):
            renderer.render(bands, sensor="sentinel2")

    def test_sar_missing_vh_raises_clear_error(self):
        """Test clear error message when VH required but missing."""
        vv = np.random.rand(100, 100)

        config = SARConfig(visualization="dual_pol")
        processor = SARProcessor(config)

        with pytest.raises(ValueError, match="requires VH polarization"):
            processor.process(vv)  # Missing vh parameter
