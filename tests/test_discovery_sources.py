"""Discovery says which sources were searched, which had nothing and which failed.

A failed source used to be a log line and a smaller result with a clean exit.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from core.data.discovery import stac_client
from core.data.discovery.stac_client import STACItem, describe_sources, discover_data

ITEM = STACItem(
    id="S2A_10SFH_20250115_0_L2A",
    collection="sentinel-2-l2a",
    datetime=datetime(2025, 1, 15, 18, 0),
    bbox=[-118.7, 33.9, -118.4, 34.2],
    geometry={},
    cloud_cover=4.0,
    assets={"nir": "https://example.invalid/B08.tif"},
    properties={},
)
BOX = [-118.60, 34.02, -118.50, 34.10]


def _discover(client_cls):
    client_cls.return_value.search_sentinel2.return_value = [ITEM]
    client_cls.return_value.search.return_value = []
    return discover_data(
        bbox=BOX, start_date="2025-01-01", end_date="2025-01-31", event_type="wildfire",
    )


def test_a_failed_source_is_reported_as_failed_not_empty():
    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_landsat.side_effect = RuntimeError("HTTP 502")
        results = _discover(client_cls)

    assert len(results) == 1
    assert results.sources["sentinel2"] == {"status": "found", "count": 1, "error": None}
    assert results.sources["landsat"]["status"] == "failed"
    assert "HTTP 502" in results.sources["landsat"]["error"]
    lines = describe_sources(results.sources)
    assert lines[0].startswith("landsat was not searched: the search failed (HTTP 502)")


def test_a_source_that_answered_with_nothing_is_empty_not_failed():
    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_landsat.return_value = []
        results = _discover(client_cls)

    assert results.sources["landsat"] == {"status": "empty", "count": 0, "error": None}
    assert describe_sources(results.sources) == [
        "sentinel2 was searched and returned 1 scene(s).",
        "landsat was searched and has no scenes for this box and window.",
    ]


def test_every_source_failing_raises_instead_of_an_empty_list():
    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_sentinel2.side_effect = RuntimeError("connection reset")
        client_cls.return_value.search_landsat.side_effect = RuntimeError("connection reset")
        with pytest.raises(RuntimeError, match="connection reset"):
            discover_data(
                bbox=BOX, start_date="2025-01-01", end_date="2025-01-31",
                event_type="wildfire",
            )


def test_run_discover_prints_and_records_a_failed_source(tmp_path):
    import json

    from cli.commands.run import run_discover

    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_landsat.side_effect = RuntimeError("HTTP 502")
        client_cls.return_value.search_sentinel2.return_value = [ITEM]
        client_cls.return_value.search.return_value = []
        out = run_discover(
            area_path=None, bbox=",".join(str(v) for v in BOX),
            start_date=datetime(2025, 1, 1), end_date=datetime(2025, 1, 31),
            event_type="wildfire", output_path=tmp_path,
        )

    saved = json.loads((tmp_path / "discovery.json").read_text())
    assert saved["sources"]["landsat"]["status"] == "failed"
    assert out["sources"]["landsat"]["status"] == "failed"
