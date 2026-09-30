"""Render chrome-free Sentinel-2 composites of the Max Road Fire AOI for publication.

Reads B08/B04/B03 (false colour) and B04/B03/B02 (true colour) straight from the Earth
Search COGs for the April 4 and May 9 2026 scenes, over the same 10 m grid the analysis run
used, and writes clean rectangles of imagery with no title, legend or credit baked in:

    firstlight-2026-04-04.png   false colour, April 4
    firstlight-2026-05-09.png   false colour, May 9
    firstlight-hero.png         wide crop of the May 9 false colour
    firstlight-2026-05-09-true-colour.png   true colour alternative for the hero

Both dates share one stretch per channel, computed over the pixels of both scenes together,
so any difference between the two panes is a difference in the data and not in the display.
These are surface-reflectance composites; no index, severity or classification is computed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.windows import from_bounds

OUT_DIR = Path(__file__).resolve().parent / "site"

BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/17/R/NJ/2026"
SCENES = {
    "2026-04-04": f"{BASE}/4/S2B_17RNJ_20260404_0_L2A",
    "2026-05-09": f"{BASE}/5/S2C_17RNJ_20260509_0_L2A",
}

# The analysis grid: EPSG:32617, 10 m, origin as recorded by the run.
GRID = Affine(10.0, 0.0, 550005.29, 0.0, -10.0, 2884742.29)
WIDTH, HEIGHT = 2212, 2225

FALSE_COLOUR = ("B08", "B04", "B03")
TRUE_COLOUR = ("B04", "B03", "B02")
HERO_ASPECT = 2.6


def read_band(scene: str, band: str) -> np.ndarray:
    left, top = GRID.c, GRID.f
    bounds = (left, top - HEIGHT * 10, left + WIDTH * 10, top)
    with rasterio.open(f"{scene}/{band}.tif") as src:
        window = from_bounds(*bounds, src.transform)
        return src.read(1, window=window, out_shape=(HEIGHT, WIDTH), boundless=True).astype(
            np.float32
        )


def shared_stretch(stacks: list[np.ndarray]) -> list[np.ndarray]:
    """2-98 percentile stretch per channel, limits shared across every stack."""
    out = [np.empty_like(s) for s in stacks]
    for ch in range(3):
        values = np.concatenate([s[..., ch][s[..., ch] > 0].ravel() for s in stacks])
        lo, hi = np.percentile(values, [2, 98])
        for i, s in enumerate(stacks):
            out[i][..., ch] = np.clip((s[..., ch] - lo) / (hi - lo), 0, 1)
    return out


def to_png(rgb: np.ndarray, path: Path) -> None:
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(path, optimize=True)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    cache = {
        date: {b: read_band(url, b) for b in ("B08", "B04", "B03", "B02")}
        for date, url in SCENES.items()
    }

    false = shared_stretch([np.dstack([cache[d][b] for b in FALSE_COLOUR]) for d in SCENES])
    true = shared_stretch([np.dstack([cache[d][b] for b in TRUE_COLOUR]) for d in SCENES])

    for date, rgb in zip(SCENES, false):
        to_png(rgb, OUT_DIR / f"firstlight-{date}.png")
    to_png(true[1], OUT_DIR / "firstlight-2026-05-09-true-colour.png")

    band_h = int(round(WIDTH / HERO_ASPECT))
    y0 = (HEIGHT - band_h) // 2
    to_png(false[1][y0 : y0 + band_h], OUT_DIR / "firstlight-hero.png")

    provenance = {
        "sensor": "Sentinel-2 MSI, Level-2A surface reflectance",
        "tile": "17RNJ",
        "scenes": {d: u.rsplit("/", 1)[1] for d, u in SCENES.items()},
        "source": "Element84 Earth Search, sentinel-2-l2a COGs on AWS Open Data",
        "crs": "EPSG:32617",
        "gsd_m": 10,
        "grid_origin": [GRID.c, GRID.f],
        "size": [WIDTH, HEIGHT],
        "false_colour": "B08 NIR, B04 red, B03 green -> R, G, B",
        "true_colour": "B04, B03, B02 -> R, G, B",
        "stretch": "linear 2-98 percentile per channel, limits shared across both dates",
        "hero_crop": {"x": 0, "y": y0, "width": WIDTH, "height": band_h},
        "licence": "Copernicus Sentinel data, free and open; "
        "'Contains modified Copernicus Sentinel data 2026'",
        "script": "florida_wildfire_test/render_site_imagery.py",
    }
    (OUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
