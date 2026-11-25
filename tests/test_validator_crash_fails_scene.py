"""A validation check that crashes fails its scene and says which check."""

from pathlib import Path

import pytest

from cli.commands import ingest
from core.data.ingestion.band_stack import create_band_stack
from core.data.ingestion.validation import image_validator
from core.data.ingestion.validation.band_validator import BandValidator
from core.data.ingestion.validation.config import load_config
from core.data.ingestion.validation.image_validator import ImageValidator

SCENE_DIR = (
    Path(__file__).resolve().parent / "fixtures/s2_cloud/S2A_17RNJ_20260511_0_L2A"
)
ITEM = {"id": "S2A_17RNJ_20260511_0_L2A", "source": "sentinel2"}


@pytest.fixture
def stack(tmp_path, monkeypatch):
    """The fixture's three spectral bands stacked as ingest stacks them.

    The fixture has no blue or green band, so only red and nir are required.
    """
    bands = {b: SCENE_DIR / f"{b}.tif" for b in ("red", "nir", "swir22")}
    path = create_band_stack(bands, tmp_path / "stack.vrt").path
    config = load_config(use_environment=False)
    config.required_optical_bands = ["red", "nir"]
    monkeypatch.setattr(image_validator, "load_config", lambda: config)
    return path


def _boom(*args, **kwargs):
    raise RuntimeError("check blew up")


def test_real_scene_passes_without_a_crash(stack):
    assert ingest._validate_downloaded_image(stack, ITEM) is True


def test_band_check_crash_fails_scene_and_names_the_check(stack, monkeypatch):
    monkeypatch.setattr(BandValidator, "validate_bands", _boom)

    result = ImageValidator().validate(stack, {"sensor": "sentinel2"}, ITEM["id"])

    assert result.is_valid is False
    assert result.errors == ["optical band check crashed: RuntimeError: check blew up"]
    assert ingest._validate_downloaded_image(stack, ITEM) is False


def test_validator_crash_fails_scene_and_names_validator(stack, monkeypatch, caplog):
    monkeypatch.setattr(ImageValidator, "validate", _boom)

    assert ingest._validate_downloaded_image(stack, ITEM) is False
    assert "ImageValidator crashed on stack.vrt: RuntimeError: check blew up" in caplog.text
