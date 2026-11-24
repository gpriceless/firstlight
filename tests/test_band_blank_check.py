"""The blank-band check judges a band's spread in the unit its metadata declares.

It used to hold every band to a floor of 1 digital number, so a real float
reflectance band, whose spread is a few hundredths, was called blank.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from core.data.ingestion.validation.band_validator import BandValidator
from core.data.ingestion.validation.config import ValidationConfig

RED = Path(__file__).parent / "fixtures" / "s2_cloud" / "S2A_17RNJ_20260511_0_L2A" / "red.tif"
EARTH_SEARCH = {"scale": 0.0001, "offset": 0.0}


def _check(path: Path, scaling=None):
    validator = BandValidator(ValidationConfig())
    with rasterio.open(path) as ds:
        result = validator.validate_band_content(ds, 1, reflectance_scaling=scaling)
    return [e for e in result.errors if "appears blank" in e], result.statistics


def _as_reflectance(tmp_path: Path) -> Path:
    # The real band, written as float32 reflectance (DN / 10000).
    with rasterio.open(RED) as src:
        profile, dn = src.profile, src.read(1)
    profile.update(dtype="float32", nodata=0.0)
    out = tmp_path / "red_reflectance.tif"
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(dn.astype("float32") / 10000, 1)
    return out


def _constant(tmp_path: Path, value, dtype) -> Path:
    out = tmp_path / f"blank_{dtype}.tif"
    with rasterio.open(
        out, "w", driver="GTiff", height=64, width=64, count=1, dtype=dtype,
        crs="EPSG:32617", transform=from_origin(500000, 2900000, 10, 10),
    ) as ds:
        ds.write(np.full((64, 64), value, dtype=dtype), 1)
    return out


def test_real_float_reflectance_band_is_not_blank(tmp_path):
    blank, stats = _check(_as_reflectance(tmp_path))

    assert 0.01 < stats.std_dev < 0.2  # real spread, below the old floor of 1
    assert blank == []


def test_real_dn_band_with_declared_scale_is_not_blank():
    blank, stats = _check(RED, EARTH_SEARCH)

    assert stats.std_dev > 100
    assert blank == []


def test_constant_float_band_is_still_blank(tmp_path):
    blank, _ = _check(_constant(tmp_path, 0.25, "float32"))

    assert len(blank) == 1
    assert "reflectance" in blank[0]


def test_constant_dn_band_is_still_blank(tmp_path):
    blank, _ = _check(_constant(tmp_path, 2500, "uint16"), EARTH_SEARCH)

    assert len(blank) == 1
    assert "reflectance" in blank[0]


def test_integer_band_with_no_scale_is_judged_in_dn(tmp_path):
    blank, _ = _check(_constant(tmp_path, 2500, "uint16"))

    assert len(blank) == 1
    assert "DN" in blank[0]


def test_shipped_config_loads_both_blank_floors():
    config = ValidationConfig.from_yaml(
        str(Path(__file__).parent.parent / "config" / "ingestion.yaml")
    )

    assert config.optical.std_dev_min == 1.0
    assert config.optical.std_dev_min_reflectance == 0.0001
