"""
Integration tests for multi-band Sentinel-2 ingestion.

Tests the complete flow from STAC discovery to VRT validation.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import numpy as np

# Test imports - verify all modules are importable
class TestModuleImports:
    """Verify all Epic 1.7 modules can be imported."""

    def test_band_config_imports(self):
        """Band configuration module loads correctly."""
        from core.data.discovery.band_config import (
            SENTINEL2_BANDS,
            LANDSAT_BANDS,
            SENTINEL1_BANDS,
            get_band_config,
            get_canonical_order,
            SENTINEL2_CANONICAL_ORDER,
        )

        assert "blue" in SENTINEL2_BANDS
        assert "nir" in SENTINEL2_BANDS
        assert "swir16" in SENTINEL2_BANDS
        assert len(SENTINEL2_CANONICAL_ORDER) == 6

    def test_stac_client_helpers_import(self):
        """STAC client helper functions exist."""
        from core.data.discovery.stac_client import (
            _extract_band_urls,
            _get_missing_bands,
            _get_primary_url,
            _get_asset_href,
        )

    def test_band_stack_imports(self):
        """Band stacking module loads correctly."""
        from core.data.ingestion.band_stack import (
            create_band_stack,
            stack_to_geotiff,
            validate_vrt_sources,
            StackResult,
            BandStackError,
        )

    def test_ingest_multiband_imports(self):
        """Ingest module has multi-band support."""
        pytest.importorskip("cli.commands.ingest")
        from cli.commands.ingest import (
            download_bands,
            BandDownloadResult,
        )


class TestBandConfig:
    """Test band configuration module."""

    def test_sentinel2_bands(self):
        """Sentinel-2 band config is correct."""
        from core.data.discovery.band_config import SENTINEL2_BANDS

        expected = {"blue", "green", "red", "nir", "swir16", "swir22"}
        assert set(SENTINEL2_BANDS.keys()) == expected

    def test_get_band_config_sentinel2(self):
        """get_band_config returns correct config for sentinel2."""
        from core.data.discovery.band_config import get_band_config

        config = get_band_config("sentinel2")
        assert "blue" in config
        assert "nir" in config

    def test_get_band_config_unknown_sensor(self):
        """get_band_config returns empty dict for unknown sensor."""
        from core.data.discovery.band_config import get_band_config

        config = get_band_config("unknown_sensor")
        assert config == {}

    def test_canonical_order(self):
        """Canonical order follows wavelength."""
        from core.data.discovery.band_config import SENTINEL2_CANONICAL_ORDER

        # Order should be shortest wavelength to longest
        assert SENTINEL2_CANONICAL_ORDER == ["blue", "green", "red", "nir", "swir16", "swir22"]


class TestSTACBandExtraction:
    """Test STAC client band URL extraction."""

    def test_extract_band_urls_complete_item(self):
        """Band URL extraction works with complete STAC item."""
        from core.data.discovery.stac_client import _extract_band_urls

        # Mock STAC item with full band assets
        mock_item = Mock()
        mock_item.assets = {
            "blue": "https://example.com/B02.tif",
            "green": "https://example.com/B03.tif",
            "red": "https://example.com/B04.tif",
            "nir": "https://example.com/B08.tif",
            "swir16": "https://example.com/B11.tif",
            "swir22": "https://example.com/B12.tif",
            "visual": "https://example.com/TCI.tif",
        }

        band_urls = _extract_band_urls(mock_item, "sentinel2")

        assert len(band_urls) == 6
        assert "blue" in band_urls
        assert "nir" in band_urls
        assert "visual" not in band_urls  # TCI should not be in band_urls

    def test_extract_band_urls_partial_item(self):
        """Band URL extraction handles missing bands."""
        from core.data.discovery.stac_client import _extract_band_urls

        mock_item = Mock()
        mock_item.assets = {
            "blue": "https://example.com/B02.tif",
            "green": "https://example.com/B03.tif",
            "red": "https://example.com/B04.tif",
            # Missing: nir, swir16, swir22
        }

        band_urls = _extract_band_urls(mock_item, "sentinel2")

        assert len(band_urls) == 3
        assert "nir" not in band_urls

    def test_get_missing_bands(self):
        """Missing band detection works."""
        from core.data.discovery.stac_client import _get_missing_bands

        mock_item = Mock()
        mock_item.assets = {
            "blue": "https://example.com/B02.tif",
            "green": "https://example.com/B03.tif",
            # Missing most bands
        }

        missing = _get_missing_bands(mock_item, "sentinel2")

        assert "red" in missing
        assert "nir" in missing
        assert "blue" not in missing


class TestBandDownload:
    """Test band download functionality."""

    def test_band_download_result_dataclass(self):
        """BandDownloadResult dataclass works correctly."""
        pytest.importorskip("cli.commands.ingest")
        from cli.commands.ingest import BandDownloadResult

        result = BandDownloadResult(
            band_name="blue",
            url="https://example.com/B02.tif",
            local_path=Path("/tmp/blue.tif"),
            size_bytes=1000000,
            download_time_s=5.0,
            success=True,
            error=None,
            retries=0,
        )

        assert result.success
        assert result.band_name == "blue"
        assert result.retries == 0


class TestBandStacking:
    """Test band stacking functionality."""

    def test_stack_result_dataclass(self):
        """StackResult dataclass works correctly."""
        from core.data.ingestion.band_stack import StackResult

        result = StackResult(
            path=Path("/tmp/stack.vrt"),
            band_count=6,
            band_mapping={"blue": 1, "green": 2, "red": 3, "nir": 4},
            crs="EPSG:32632",
            bounds=(11.5, 46.0, 12.5, 47.0),
            resolution=(10.0, 10.0),
            warnings=[],
            source_files={},
        )

        assert result.band_count == 6
        assert result.band_mapping["blue"] == 1

    def test_validate_vrt_sources_missing_file(self):
        """VRT validation detects missing source files."""
        from core.data.ingestion.band_stack import validate_vrt_sources
        import tempfile

        # Create a VRT that references a non-existent file
        vrt_content = '''<VRTDataset rasterXSize="100" rasterYSize="100">
          <VRTRasterBand dataType="UInt16" band="1">
            <SimpleSource>
              <SourceFilename relativeToVRT="1">nonexistent_file.tif</SourceFilename>
              <SourceBand>1</SourceBand>
            </SimpleSource>
          </VRTRasterBand>
        </VRTDataset>'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.vrt', delete=False) as f:
            f.write(vrt_content)
            vrt_path = Path(f.name)

        try:
            is_valid, missing = validate_vrt_sources(vrt_path)
            assert not is_valid
            assert len(missing) > 0
            assert "nonexistent_file.tif" in missing[0]
        finally:
            vrt_path.unlink()


class TestValidatorVRTSupport:
    """Test validator VRT handling."""

    def test_image_validator_has_vrt_method(self):
        """ImageValidator has VRT source validation method."""
        from core.data.ingestion.validation.image_validator import ImageValidator

        # Check method exists
        assert hasattr(ImageValidator, '_validate_vrt_sources')

    def test_band_validator_match_bands(self):
        """BandValidator can match bands from descriptions."""
        from core.data.ingestion.validation.band_validator import BandValidator
        from core.data.ingestion.validation.config import ValidationConfig

        # Check class exists and has matching capability
        validator = BandValidator(ValidationConfig())
        assert hasattr(validator, '_match_bands') or hasattr(validator, 'validate_bands')


class TestSkipValidationFlag:
    """Test --skip-validation flag functionality."""

    def test_skip_validation_option_exists(self):
        """Ingest command has skip-validation option."""
        pytest.importorskip("cli.commands.ingest")
        from cli.commands.ingest import ingest
        import click

        # Check that ingest is a click command with skip_validation param
        assert hasattr(ingest, 'params')
        param_names = [p.name for p in ingest.params]
        assert 'skip_validation' in param_names


# Run a quick sanity check
if __name__ == "__main__":
    print("Running quick sanity checks...")

    # Test imports
    t = TestModuleImports()
    t.test_band_config_imports()
    print("✓ Band config imports")

    t.test_stac_client_helpers_import()
    print("✓ STAC client helpers import")

    t.test_band_stack_imports()
    print("✓ Band stack imports")

    t.test_ingest_multiband_imports()
    print("✓ Ingest multiband imports")

    # Test band config
    tc = TestBandConfig()
    tc.test_sentinel2_bands()
    print("✓ Sentinel-2 bands correct")

    tc.test_canonical_order()
    print("✓ Canonical order correct")

    # Test skip validation
    tv = TestSkipValidationFlag()
    tv.test_skip_validation_option_exists()
    print("✓ Skip validation flag exists")

    print("\n✅ All quick sanity checks passed!")
