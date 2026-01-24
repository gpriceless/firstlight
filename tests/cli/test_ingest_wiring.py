"""
Tests for CLI Ingest Command Wiring to StreamingIngester.

Verifies Epic 1.2 implementation:
- Real data download (not text placeholders)
- Image validation integration
- Normalization pipeline
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add project root to path BEFORE any imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import rasterio
    from rasterio.transform import from_bounds

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# Test helpers


def create_mock_geotiff(path: Path, width: int = 100, height: int = 100):
    """Create a mock GeoTIFF file for testing."""
    if not HAS_RASTERIO:
        pytest.skip("rasterio not available")

    data = np.random.randint(0, 255, (height, width), dtype=np.uint8)

    transform = from_bounds(
        west=-180, south=-90, east=-170, north=-80, width=width, height=height
    )

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=rasterio.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


@pytest.fixture
def ingest_module():
    """Import the ingest module inside a fixture to ensure path is set up."""
    import importlib.util
    import sys

    # Use importlib to load the module
    spec = importlib.util.spec_from_file_location(
        "ingest",
        str(project_root / "cli" / "commands" / "ingest.py")
    )
    ingest = importlib.util.module_from_spec(spec)
    sys.modules["ingest"] = ingest
    spec.loader.exec_module(ingest)
    return ingest


# Tests


def test_imports_succeed():
    """Test that required imports are available."""
    from core.data.ingestion.streaming import StreamingIngester
    from core.data.ingestion.validation.image_validator import ImageValidator

    assert StreamingIngester is not None
    assert ImageValidator is not None


def test_download_real_data_uses_streaming_ingester(tmp_path, ingest_module):
    """Test that _download_real_data uses StreamingIngester, not text placeholder."""
    # Create a mock GeoTIFF for the ingester to "download"
    mock_source = tmp_path / "source.tif"
    create_mock_geotiff(mock_source)

    download_path = tmp_path / "downloaded.tif"

    # Mock the StreamingIngester
    with patch.object(ingest_module, "StreamingIngester") as MockIngester:
        mock_instance = MockIngester.return_value
        mock_instance.ingest.return_value = {
            "status": "completed",
            "output_path": str(download_path),
        }

        # Call the function
        item = {"id": "test_item", "source": "test_source"}
        ingest_module._download_real_data(str(mock_source), download_path, item)

        # Verify StreamingIngester was called
        MockIngester.assert_called_once()
        mock_instance.ingest.assert_called_once_with(
            source=str(mock_source),
            output_path=download_path,
        )


def test_download_real_data_raises_on_failure(tmp_path, ingest_module):
    """Test that _download_real_data raises RuntimeError on download failure."""
    download_path = tmp_path / "downloaded.tif"

    # Mock the StreamingIngester to return failure
    with patch.object(ingest_module, "StreamingIngester") as MockIngester:
        mock_instance = MockIngester.return_value
        mock_instance.ingest.return_value = {
            "status": "failed",
            "errors": ["Network error", "Timeout"],
        }

        # Call should raise RuntimeError
        item = {"id": "test_item"}
        with pytest.raises(RuntimeError, match="Download failed"):
            ingest_module._download_real_data("https://example.com/data.tif", download_path, item)


def test_validate_downloaded_image_uses_validator(tmp_path, ingest_module):
    """Test that _validate_downloaded_image uses ImageValidator."""
    # Create a mock GeoTIFF
    image_path = tmp_path / "test.tif"
    create_mock_geotiff(image_path)

    # Mock the ImageValidator
    with patch.object(ingest_module, "ImageValidator") as MockValidator:
        mock_instance = MockValidator.return_value
        mock_result = MagicMock()
        mock_result.is_valid = True
        mock_result.warnings = []
        mock_result.errors = []
        mock_instance.validate.return_value = mock_result

        # Call the function
        item = {"id": "test_item", "source": "sentinel2"}
        result = ingest_module._validate_downloaded_image(image_path, item)

        # Verify validator was called
        MockValidator.assert_called_once()
        mock_instance.validate.assert_called_once()
        assert result is True


def test_validate_downloaded_image_returns_false_on_invalid(tmp_path, ingest_module):
    """Test that _validate_downloaded_image returns False when validation fails."""
    # Create a mock GeoTIFF
    image_path = tmp_path / "test.tif"
    create_mock_geotiff(image_path)

    # Mock the ImageValidator to return invalid
    with patch.object(ingest_module, "ImageValidator") as MockValidator:
        mock_instance = MockValidator.return_value
        mock_result = MagicMock()
        mock_result.is_valid = False
        mock_result.warnings = ["Low quality"]
        mock_result.errors = ["Blank band detected"]
        mock_instance.validate.return_value = mock_result

        # Call the function
        item = {"id": "test_item"}
        result = ingest_module._validate_downloaded_image(image_path, item)

        # Should return False
        assert result is False


def test_process_item_downloads_and_validates(tmp_path, ingest_module):
    """Test that process_item integrates download and validation."""
    # Mock the download and validation functions
    with (
        patch.object(ingest_module, "_download_real_data") as mock_download,
        patch.object(ingest_module, "_validate_downloaded_image") as mock_validate,
    ):
        mock_validate.return_value = True

        # Call process_item
        item = {
            "id": "test_item",
            "source": "sentinel1",
            "url": "https://example.com/data.tif",
        }
        result = ingest_module.process_item(
            item=item,
            output_path=tmp_path,
            output_format="cog",
            normalize=False,
            target_crs=None,
            target_resolution=None,
        )

        # Verify download was called
        mock_download.assert_called_once()

        # Verify validation was called
        mock_validate.assert_called_once()

        # Should succeed
        assert result is True


def test_process_item_fails_on_validation_failure(tmp_path, ingest_module):
    """Test that process_item fails when validation fails."""
    # Mock download to succeed, validation to fail
    with (
        patch.object(ingest_module, "_download_real_data") as mock_download,
        patch.object(ingest_module, "_validate_downloaded_image") as mock_validate,
    ):
        mock_validate.return_value = False

        # Call process_item
        item = {
            "id": "test_item",
            "source": "sentinel1",
            "url": "https://example.com/data.tif",
        }
        result = ingest_module.process_item(
            item=item,
            output_path=tmp_path,
            output_format="cog",
            normalize=False,
            target_crs=None,
            target_resolution=None,
        )

        # Should fail
        assert result is False


def test_no_text_placeholder_in_production_path(ingest_module):
    """Test that no text placeholders are written in the production code path."""
    # Read the source code
    import inspect

    source = inspect.getsource(ingest_module.process_item)

    # Verify no write_text with mock data
    assert "write_text" not in source or "Mock data" not in source
    assert "# Mock download" not in source


def test_downloaded_file_is_real_raster(tmp_path, ingest_module):
    """Test that downloaded files are real raster data, not text files."""
    # Create a real GeoTIFF source
    source_file = tmp_path / "source.tif"
    create_mock_geotiff(source_file)

    download_path = tmp_path / "downloaded.tif"

    # Mock ingester to copy the file
    with patch.object(ingest_module, "StreamingIngester") as MockIngester:
        mock_instance = MockIngester.return_value

        def mock_ingest(source, output_path):
            # Simulate real download by copying
            import shutil

            shutil.copy(source_file, output_path)
            return {"status": "completed", "output_path": str(output_path)}

        mock_instance.ingest.side_effect = mock_ingest

        # Download the file
        item = {"id": "test_item"}
        ingest_module._download_real_data(str(source_file), download_path, item)

        # Verify the file is a real raster, not a text file
        assert download_path.exists()

        if HAS_RASTERIO:
            # Should be openable as raster
            with rasterio.open(download_path) as src:
                assert src.count > 0
                assert src.width > 0
                assert src.height > 0
        else:
            # At least verify it's not a text file
            with open(download_path, "rb") as f:
                header = f.read(100)
                # GeoTIFF magic number
                assert header.startswith(b"II\x2a\x00") or header.startswith(b"MM\x00\x2a")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
