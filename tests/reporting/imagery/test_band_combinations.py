"""Unit tests for satellite imagery band combination definitions.

Tests all band composites for Sentinel-2, Landsat 8/9, and Sentinel-1 SAR.
"""

import pytest
from core.reporting.imagery.band_combinations import (
    BandComposite,
    SENTINEL2_COMPOSITES,
    LANDSAT_COMPOSITES,
    SAR_VISUALIZATIONS,
    get_bands_for_composite,
    get_available_composites,
)


class TestBandComposite:
    """Test BandComposite dataclass validation and behavior."""

    def test_valid_composite_creation(self):
        """Valid band composite can be created."""
        composite = BandComposite(
            name="test",
            bands=("B04", "B03", "B02"),
            description="Test composite",
            sensor="sentinel2"
        )
        assert composite.name == "test"
        assert composite.bands == ("B04", "B03", "B02")
        assert composite.description == "Test composite"
        assert composite.sensor == "sentinel2"
        assert composite.stretch_range == (2.0, 98.0)  # Default

    def test_custom_stretch_range(self):
        """Custom stretch range is preserved."""
        composite = BandComposite(
            name="test",
            bands=("VV", "VH", "VV"),
            description="SAR test",
            sensor="sentinel1",
            stretch_range=(-20.0, 5.0)
        )
        assert composite.stretch_range == (-20.0, 5.0)

    def test_invalid_band_count_raises_error(self):
        """BandComposite requires exactly 3 bands."""
        with pytest.raises(ValueError, match="exactly 3 bands"):
            BandComposite(
                name="invalid",
                bands=("B04", "B03"),  # Only 2 bands
                description="Invalid",
                sensor="sentinel2"
            )

        with pytest.raises(ValueError, match="exactly 3 bands"):
            BandComposite(
                name="invalid",
                bands=("B04", "B03", "B02", "B08"),  # 4 bands
                description="Invalid",
                sensor="sentinel2"
            )


class TestSentinel2Composites:
    """Test Sentinel-2 band composite definitions."""

    def test_true_color_composite(self):
        """Sentinel-2 true color uses B04/B03/B02."""
        composite = SENTINEL2_COMPOSITES["true_color"]
        assert composite.name == "true_color"
        assert composite.bands == ("B04", "B03", "B02")
        assert composite.sensor == "sentinel2"
        assert "visible spectrum" in composite.description.lower()

    def test_false_color_ir_composite(self):
        """Sentinel-2 false color IR uses B08/B04/B03."""
        composite = SENTINEL2_COMPOSITES["false_color_ir"]
        assert composite.name == "false_color_ir"
        assert composite.bands == ("B08", "B04", "B03")
        assert composite.sensor == "sentinel2"
        assert "vegetation" in composite.description.lower()

    def test_swir_composite(self):
        """Sentinel-2 SWIR uses B12/B8A/B04."""
        composite = SENTINEL2_COMPOSITES["swir"]
        assert composite.name == "swir"
        assert composite.bands == ("B12", "B8A", "B04")
        assert composite.sensor == "sentinel2"
        assert "infrared" in composite.description.lower()

    def test_agriculture_composite(self):
        """Sentinel-2 agriculture composite exists."""
        composite = SENTINEL2_COMPOSITES["agriculture"]
        assert composite.name == "agriculture"
        assert composite.bands == ("B11", "B08", "B02")
        assert composite.sensor == "sentinel2"

    def test_geology_composite(self):
        """Sentinel-2 geology composite exists."""
        composite = SENTINEL2_COMPOSITES["geology"]
        assert composite.name == "geology"
        assert composite.bands == ("B12", "B11", "B02")
        assert composite.sensor == "sentinel2"

    def test_all_composites_have_3_bands(self):
        """All Sentinel-2 composites have exactly 3 bands."""
        for name, composite in SENTINEL2_COMPOSITES.items():
            assert len(composite.bands) == 3, f"{name} should have 3 bands"

    def test_all_composites_are_sentinel2(self):
        """All Sentinel-2 composites have correct sensor."""
        for name, composite in SENTINEL2_COMPOSITES.items():
            assert composite.sensor == "sentinel2", f"{name} should be sentinel2"


class TestLandsatComposites:
    """Test Landsat 8/9 band composite definitions."""

    def test_true_color_composite(self):
        """Landsat true color uses SR_B4/SR_B3/SR_B2."""
        composite = LANDSAT_COMPOSITES["true_color"]
        assert composite.name == "true_color"
        assert composite.bands == ("SR_B4", "SR_B3", "SR_B2")
        assert composite.sensor == "landsat8"
        assert "visible spectrum" in composite.description.lower()

    def test_false_color_ir_composite(self):
        """Landsat false color IR uses SR_B5/SR_B4/SR_B3."""
        composite = LANDSAT_COMPOSITES["false_color_ir"]
        assert composite.name == "false_color_ir"
        assert composite.bands == ("SR_B5", "SR_B4", "SR_B3")
        assert composite.sensor == "landsat8"
        assert "vegetation" in composite.description.lower()

    def test_swir_composite(self):
        """Landsat SWIR uses SR_B7/SR_B5/SR_B4."""
        composite = LANDSAT_COMPOSITES["swir"]
        assert composite.name == "swir"
        assert composite.bands == ("SR_B7", "SR_B5", "SR_B4")
        assert composite.sensor == "landsat8"
        assert "infrared" in composite.description.lower()

    def test_agriculture_composite(self):
        """Landsat agriculture composite exists."""
        composite = LANDSAT_COMPOSITES["agriculture"]
        assert composite.name == "agriculture"
        assert composite.bands == ("SR_B6", "SR_B5", "SR_B2")
        assert composite.sensor == "landsat8"

    def test_geology_composite(self):
        """Landsat geology composite exists."""
        composite = LANDSAT_COMPOSITES["geology"]
        assert composite.name == "geology"
        assert composite.bands == ("SR_B7", "SR_B6", "SR_B2")
        assert composite.sensor == "landsat8"

    def test_all_composites_have_3_bands(self):
        """All Landsat composites have exactly 3 bands."""
        for name, composite in LANDSAT_COMPOSITES.items():
            assert len(composite.bands) == 3, f"{name} should have 3 bands"

    def test_all_composites_are_landsat(self):
        """All Landsat composites have correct sensor."""
        for name, composite in LANDSAT_COMPOSITES.items():
            assert composite.sensor == "landsat8", f"{name} should be landsat8"


class TestSARVisualizations:
    """Test Sentinel-1 SAR visualization definitions."""

    def test_vv_single_polarization(self):
        """VV single polarization uses VV for all channels."""
        viz = SAR_VISUALIZATIONS["vv_single"]
        assert viz.name == "vv_single"
        assert viz.bands == ("VV", "VV", "VV")
        assert viz.sensor == "sentinel1"
        assert "water" in viz.description.lower()
        assert viz.stretch_range == (-20.0, 5.0)

    def test_vh_single_polarization(self):
        """VH single polarization uses VH for all channels."""
        viz = SAR_VISUALIZATIONS["vh_single"]
        assert viz.name == "vh_single"
        assert viz.bands == ("VH", "VH", "VH")
        assert viz.sensor == "sentinel1"
        assert viz.stretch_range == (-25.0, 0.0)

    def test_dual_pol_composite(self):
        """Dual polarization composite uses both VV and VH."""
        viz = SAR_VISUALIZATIONS["dual_pol"]
        assert viz.name == "dual_pol"
        assert viz.bands == ("VV", "VH", "VV/VH")
        assert viz.sensor == "sentinel1"
        assert "dual" in viz.description.lower()

    def test_flood_detection_composite(self):
        """Flood detection composite optimized for water."""
        viz = SAR_VISUALIZATIONS["flood_detection"]
        assert viz.name == "flood_detection"
        assert viz.bands == ("VV", "VV", "VH")
        assert viz.sensor == "sentinel1"
        assert "flood" in viz.description.lower()

    def test_all_visualizations_have_3_bands(self):
        """All SAR visualizations have exactly 3 bands."""
        for name, viz in SAR_VISUALIZATIONS.items():
            assert len(viz.bands) == 3, f"{name} should have 3 bands"

    def test_all_visualizations_are_sentinel1(self):
        """All SAR visualizations have correct sensor."""
        for name, viz in SAR_VISUALIZATIONS.items():
            assert viz.sensor == "sentinel1", f"{name} should be sentinel1"

    def test_sar_has_db_scale_stretch(self):
        """SAR visualizations use dB scale stretch ranges."""
        for name, viz in SAR_VISUALIZATIONS.items():
            assert viz.stretch_range is not None, f"{name} should have stretch_range"
            min_val, max_val = viz.stretch_range
            # dB values are typically negative to small positive
            assert min_val < 0, f"{name} should have negative min dB"
            assert max_val > min_val, f"{name} should have valid range"


class TestGetBandsForComposite:
    """Test get_bands_for_composite function."""

    def test_sentinel2_true_color(self):
        """Get Sentinel-2 true color composite."""
        composite = get_bands_for_composite('sentinel2', 'true_color')
        assert composite.bands == ("B04", "B03", "B02")
        assert composite.sensor == "sentinel2"

    def test_sentinel2_case_insensitive(self):
        """Sensor name is case-insensitive."""
        composite = get_bands_for_composite('SENTINEL2', 'true_color')
        assert composite.bands == ("B04", "B03", "B02")

    def test_sentinel2_alias(self):
        """Sentinel-2 aliases work."""
        composite1 = get_bands_for_composite('sentinel-2', 'true_color')
        composite2 = get_bands_for_composite('s2', 'true_color')
        assert composite1.bands == composite2.bands

    def test_landsat8_true_color(self):
        """Get Landsat-8 true color composite."""
        composite = get_bands_for_composite('landsat8', 'true_color')
        assert composite.bands == ("SR_B4", "SR_B3", "SR_B2")
        assert composite.sensor == "landsat8"

    def test_landsat9_uses_same_composites(self):
        """Landsat-9 uses same composite definitions as Landsat-8."""
        composite = get_bands_for_composite('landsat9', 'true_color')
        assert composite.bands == ("SR_B4", "SR_B3", "SR_B2")

    def test_landsat_aliases(self):
        """Landsat aliases work."""
        composite1 = get_bands_for_composite('landsat-8', 'true_color')
        composite2 = get_bands_for_composite('l8', 'true_color')
        assert composite1.bands == composite2.bands

    def test_sentinel1_vv_single(self):
        """Get Sentinel-1 VV visualization."""
        viz = get_bands_for_composite('sentinel1', 'vv_single')
        assert viz.bands == ("VV", "VV", "VV")
        assert viz.sensor == "sentinel1"

    def test_sentinel1_aliases(self):
        """Sentinel-1 aliases work."""
        viz1 = get_bands_for_composite('sentinel-1', 'vv_single')
        viz2 = get_bands_for_composite('s1', 'vv_single')
        assert viz1.bands == viz2.bands

    def test_invalid_sensor_raises_error(self):
        """Invalid sensor raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported sensor"):
            get_bands_for_composite('sentinel3', 'true_color')

    def test_invalid_composite_raises_error(self):
        """Invalid composite name raises ValueError."""
        with pytest.raises(ValueError, match="not available"):
            get_bands_for_composite('sentinel2', 'nonexistent')

    def test_error_message_includes_available_composites(self):
        """Error message lists available composites."""
        with pytest.raises(ValueError, match="Available composites"):
            get_bands_for_composite('sentinel2', 'invalid_name')


class TestGetAvailableComposites:
    """Test get_available_composites function."""

    def test_sentinel2_composites(self):
        """Get all available Sentinel-2 composites."""
        composites = get_available_composites('sentinel2')
        assert 'true_color' in composites
        assert 'false_color_ir' in composites
        assert 'swir' in composites
        assert 'agriculture' in composites
        assert 'geology' in composites
        assert len(composites) == 5

    def test_landsat_composites(self):
        """Get all available Landsat composites."""
        composites = get_available_composites('landsat8')
        assert 'true_color' in composites
        assert 'false_color_ir' in composites
        assert 'swir' in composites
        assert 'agriculture' in composites
        assert 'geology' in composites
        assert len(composites) == 5

    def test_sentinel1_visualizations(self):
        """Get all available Sentinel-1 visualizations."""
        visualizations = get_available_composites('sentinel1')
        assert 'vv_single' in visualizations
        assert 'vh_single' in visualizations
        assert 'dual_pol' in visualizations
        assert 'flood_detection' in visualizations
        assert len(visualizations) == 4

    def test_case_insensitive(self):
        """Sensor name is case-insensitive."""
        composites = get_available_composites('SENTINEL2')
        assert 'true_color' in composites

    def test_sensor_aliases(self):
        """Sensor aliases return same composites."""
        s2_composites = get_available_composites('sentinel2')
        s2_alt_composites = get_available_composites('s2')
        assert s2_composites == s2_alt_composites

    def test_invalid_sensor_raises_error(self):
        """Invalid sensor raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported sensor"):
            get_available_composites('sentinel3')

    def test_error_message_includes_supported_sensors(self):
        """Error message lists supported sensors."""
        with pytest.raises(ValueError, match="Supported sensors"):
            get_available_composites('invalid_sensor')


class TestCompositeParity:
    """Test that optical sensors have equivalent composites."""

    def test_sentinel2_and_landsat_have_same_composite_names(self):
        """Sentinel-2 and Landsat have matching composite names."""
        s2_names = set(SENTINEL2_COMPOSITES.keys())
        landsat_names = set(LANDSAT_COMPOSITES.keys())
        assert s2_names == landsat_names, "Sentinel-2 and Landsat should have same composite names"

    def test_composite_descriptions_match_purpose(self):
        """True color and false color composites have matching descriptions."""
        s2_true = SENTINEL2_COMPOSITES["true_color"]
        l8_true = LANDSAT_COMPOSITES["true_color"]
        # Descriptions should be similar (both mention visible/natural)
        assert "color" in s2_true.description.lower()
        assert "color" in l8_true.description.lower()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_band_tuple_immutability(self):
        """Band tuples are immutable."""
        composite = SENTINEL2_COMPOSITES["true_color"]
        # Tuples are immutable, so this should fail with TypeError
        with pytest.raises(TypeError, match="does not support item assignment"):
            composite.bands[0] = "B08"

    def test_composite_dict_contains_all_required(self):
        """All composite dictionaries are non-empty."""
        assert len(SENTINEL2_COMPOSITES) > 0
        assert len(LANDSAT_COMPOSITES) > 0
        assert len(SAR_VISUALIZATIONS) > 0

    def test_no_duplicate_composite_names_within_sensor(self):
        """No duplicate composite names within same sensor."""
        s2_names = list(SENTINEL2_COMPOSITES.keys())
        assert len(s2_names) == len(set(s2_names)), "Sentinel-2 has duplicate composite names"

        landsat_names = list(LANDSAT_COMPOSITES.keys())
        assert len(landsat_names) == len(set(landsat_names)), "Landsat has duplicate composite names"

        sar_names = list(SAR_VISUALIZATIONS.keys())
        assert len(sar_names) == len(set(sar_names)), "Sentinel-1 has duplicate visualization names"
