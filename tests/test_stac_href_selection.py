"""
Regression tests for LGT-412: STAC asset href selection.

The Element84 `landsat-c2-l2` collection returns `s3://usgs-landsat/...`
as the primary asset href; that scheme is unreachable from the HTTP-only
downloader. The discovery layer must surface an HTTPS-reachable URL
(via the STAC `alternate-assets` extension or a known bucket -> HTTPS
mapping) so the downstream ingest pipeline succeeds without an S3
client.
"""

from datetime import datetime
from typing import Any, Dict

from core.data.discovery.stac_client import (
    STACItem,
    _pick_downloadable_href,
    _translate_s3_uri,
)


class _FakeAsset:
    def __init__(self, href: str, extra_fields: Dict[str, Any] | None = None) -> None:
        self.href = href
        self.extra_fields = extra_fields or {}


class _FakePystacItem:
    def __init__(self, item_id: str, assets: Dict[str, Any]) -> None:
        self.id = item_id
        self.collection_id = "landsat-c2-l2"
        self.datetime = datetime(2026, 4, 25, 12, 0)
        self.bbox = [-80.5, 25.9, -80.4, 26.0]
        self.geometry: Dict[str, Any] = {}
        self.properties = {"eo:cloud_cover": 12.0}
        self.assets = assets


class TestPickDownloadableHref:
    """`_pick_downloadable_href` prefers HTTP(S) over s3:// at extract time."""

    def test_returns_primary_href_when_already_https(self) -> None:
        asset = _FakeAsset("https://example.com/blue.tif")
        assert _pick_downloadable_href(asset) == "https://example.com/blue.tif"

    def test_falls_back_to_alternate_https(self) -> None:
        asset = _FakeAsset(
            "s3://usgs-landsat/collection02/foo/B2.TIF",
            extra_fields={
                "alternate": {
                    "https": {"href": "https://landsatlook.usgs.gov/data/foo/B2.TIF"},
                    "s3": {"href": "s3://usgs-landsat/foo/B2.TIF"},
                }
            },
        )
        assert (
            _pick_downloadable_href(asset)
            == "https://landsatlook.usgs.gov/data/foo/B2.TIF"
        )

    def test_translates_known_s3_bucket_to_https(self) -> None:
        asset = _FakeAsset(
            "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2026/"
            "015/042/LC08_L2SP_015042_20260425_20260505_02_T1/"
            "LC08_L2SP_015042_20260425_20260505_02_T1_SR_B2.TIF"
        )
        href = _pick_downloadable_href(asset)
        assert href.startswith("https://landsatlook.usgs.gov/data/")
        assert href.endswith("_SR_B2.TIF")

    def test_unknown_s3_bucket_returns_primary(self) -> None:
        # Unknown bucket — caller must see the s3:// URL so the failure is
        # actionable downstream, not silently dropped.
        asset = _FakeAsset("s3://unknown-bucket/foo.tif")
        assert _pick_downloadable_href(asset) == "s3://unknown-bucket/foo.tif"

    def test_dict_asset_with_alternate(self) -> None:
        asset = {
            "href": "s3://usgs-landsat/foo/B5.TIF",
            "alternate": {"https": {"href": "https://example.com/foo/B5.TIF"}},
        }
        assert _pick_downloadable_href(asset) == "https://example.com/foo/B5.TIF"


class TestSTACItemFromPystac:
    """`STACItem.from_pystac` must downgrade s3:// hrefs to HTTPS during extract."""

    def test_landsat_assets_resolve_to_https(self) -> None:
        assets = {
            "blue": _FakeAsset(
                "s3://usgs-landsat/foo/B2.TIF",
                extra_fields={
                    "alternate": {
                        "https": {"href": "https://landsatlook.usgs.gov/data/foo/B2.TIF"}
                    }
                },
            ),
            "red": _FakeAsset(
                "s3://usgs-landsat/collection02/foo/B4.TIF",
            ),
        }
        pystac_item = _FakePystacItem("LC08_L2SP_015042_20260425", assets)
        item = STACItem.from_pystac(pystac_item)

        # Both bands now downloadable over HTTPS.
        for url in item.assets.values():
            assert url.startswith("https://"), f"Got non-HTTPS asset url: {url}"

    def test_does_not_break_sentinel2_https_hrefs(self) -> None:
        # Sentinel-2 from Element84 is already HTTPS — pass-through must be
        # bit-identical, no churn through the s3 mapping path.
        assets = {
            "B04": _FakeAsset(
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/foo/B04.tif"
            ),
            "B08": _FakeAsset(
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/foo/B08.tif"
            ),
        }
        pystac_item = _FakePystacItem("S2A_2026_05_01", assets)
        pystac_item.collection_id = "sentinel-2-l2a"
        item = STACItem.from_pystac(pystac_item)

        assert (
            item.assets["B04"]
            == "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/foo/B04.tif"
        )
        assert (
            item.assets["B08"]
            == "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/foo/B08.tif"
        )


class TestTranslateS3Uri:
    """The known-bucket s3:// translator is conservative — only known mappings."""

    def test_translates_usgs_landsat(self) -> None:
        translated = _translate_s3_uri(
            "s3://usgs-landsat/collection02/level-2/standard/oli-tirs/2026/015/042/scene/B10.TIF"
        )
        assert translated == (
            "https://landsatlook.usgs.gov/data/collection02/level-2/standard/"
            "oli-tirs/2026/015/042/scene/B10.TIF"
        )

    def test_returns_none_for_unknown_bucket(self) -> None:
        assert _translate_s3_uri("s3://random-bucket/foo.tif") is None

    def test_returns_none_for_https(self) -> None:
        assert _translate_s3_uri("https://example.com/foo.tif") is None

    def test_returns_none_for_empty_key(self) -> None:
        assert _translate_s3_uri("s3://usgs-landsat/") is None
