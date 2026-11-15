"""A band download refuses a response that is not a TIFF.

The landsatlook mirror answers an anonymous request for a Landsat band with
200 text/html: a redirect to the USGS login page. Ingest used to save that
page as ``blue.tif`` and fail later with "not a supported format". The
response below is shaped like the real one for item
LC08_L2SP_015042_20260425_02_T1, band SR_B2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from cli.commands.ingest import NotATiffError, _download_band_file, download_bands

LANDSAT_URL = (
    "https://landsatlook.usgs.gov/data/collection02/level-2/standard/oli-tirs/"
    "2026/015/042/LC08_L2SP_015042_20260425_20260505_02_T1/"
    "LC08_L2SP_015042_20260425_20260505_02_T1_SR_B2.TIF"
)
LOGIN_URL = "https://ers.cr.usgs.gov/login"
REAL_TIFF = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "s2_cloud" / "S2A_17RNJ_20260531_0_L2A" / "nir.tif"
)


class _Response:
    def __init__(self, body: bytes, url: str, content_type: str):
        self._body = body
        self.url = url
        self.headers = {"Content-Type": content_type}
        self.status_code = 200

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def close(self):
        pass


def _login_page():
    return _Response(
        b"<!DOCTYPE html><html><head><title>Login</title></head></html>",
        LOGIN_URL,
        "text/html; charset=UTF-8",
    )


def test_login_page_is_refused_and_not_saved(tmp_path):
    dest = tmp_path / "blue.tif"
    with patch("requests.get", return_value=_login_page()):
        with pytest.raises(NotATiffError, match=r"text/html.* from https://ers\.cr\.usgs\.gov/login"):
            _download_band_file(LANDSAT_URL, dest)
    assert not dest.exists()


def test_a_real_tiff_is_saved_byte_for_byte(tmp_path):
    """Positive control: the same check lets a real GeoTIFF through."""
    body = REAL_TIFF.read_bytes()
    dest = tmp_path / "nir.tif"
    with patch("requests.get", return_value=_Response(body, "https://x/nir.tif", "image/tiff")):
        _download_band_file("https://x/nir.tif", dest)
    assert dest.read_bytes() == body


def test_band_failure_names_the_cause_without_retrying(tmp_path):
    with patch("requests.get", return_value=_login_page()) as get:
        results = download_bands({"blue": LANDSAT_URL}, tmp_path, "LC08_L2SP_015042_20260425_02_T1")
    assert get.call_count == 1
    blue = results["blue"]
    assert not blue.success
    assert "did not return a TIFF" in blue.error and LOGIN_URL in blue.error
    assert not (tmp_path / "blue.tif").exists()
