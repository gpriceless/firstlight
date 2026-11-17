"""Discovery reports item size as unknown instead of a guessed constant.

Earth Search Sentinel-2 L2A assets carry no ``file:size`` (checked on
S2A_17RNJ_20260531_0_L2A), and ingest reads only the part of each band the
box needs, so no whole-item number is the download size.
"""

from datetime import datetime
from unittest.mock import patch

from core.data.discovery import stac_client
from core.data.discovery.stac_client import STACItem, discover_data

ITEM = STACItem(
    id="S2A_17RNJ_20260531_0_L2A",
    collection="sentinel-2-l2a",
    datetime=datetime(2026, 5, 31, 16, 0),
    bbox=[-80.9, 25.2, -79.8, 26.2],
    geometry={},
    cloud_cover=12.0,
    assets={"nir": "https://example.invalid/B08.tif"},
    properties={},
)


def test_sentinel2_item_size_is_unknown_not_guessed():
    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_sentinel2.return_value = [ITEM]
        results = discover_data(
            bbox=[-80.55, 25.97, -80.35, 26.10],
            start_date="2026-04-19", end_date="2026-05-31", event_type="storm",
        )

    assert len(results) == 1
    assert results[0]["size_bytes"] is None
