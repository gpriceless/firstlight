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


def _tile(tile: str, day: int, cloud: float) -> STACItem:
    return STACItem(
        id=f"S2B_{tile}_202209{day:02d}_0_L2A",
        collection="sentinel-2-l2a",
        datetime=datetime(2022, 9, day, 16, 15),
        bbox=[], geometry={}, cloud_cover=cloud,
        assets={"nir": "https://example.invalid/B08.tif"}, properties={},
    )


def test_a_kept_day_brings_its_tiles_over_the_cloud_limit():
    # Fort Myers, 25 Sep 2022: Earth Search holds 17RLK at 29.8% cloud and
    # 17RMK at 36.8%. A 30% limit kept one tile of the day and lost the box's
    # east part. 17RMK on 27 Sep (100%) is on a day with no kept tile.
    kept = [_tile("17RLK", 25, 29.8)]
    by_day = {"2022-09-25": [_tile("17RLK", 25, 29.8), _tile("17RMK", 25, 36.8)]}

    def search(bbox, start_date, end_date, max_cloud_cover, **_):
        if max_cloud_cover is None:
            return by_day.get(start_date, [])
        return kept

    with patch.object(stac_client, "STACClient") as client_cls:
        client_cls.return_value.search_sentinel2.side_effect = search
        results = discover_data(
            bbox=[-82.2, 26.3, -81.7, 26.8],
            start_date="2022-09-15", end_date="2022-10-12", event_type="storm",
            max_cloud_cover=30,
        )

    assert sorted(r["id"] for r in results) == [
        "S2B_17RLK_20220925_0_L2A", "S2B_17RMK_20220925_0_L2A",
    ]
