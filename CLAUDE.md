# FirstLight

Read `ARCHITECTURE.md` before changing code, and `LESSONS.md` before fixing a bug.
`history/left-in-the-dust.md` lists everything deleted or abandoned and where it went.

## What it does

An area, a time window and an event type (flood, wildfire, storm) go in. An extent
raster, a severity map, GeoJSON and a report come out.

- Only the CLI runs end to end: `flight run --bbox ... --event ...` (`cli/commands/run.py`).
- The API (`api/`) stores events and never processes them; `GET /products` returns mock
  records. Do not quote anything from it as output.
- Reports, maps and before and after imagery come from `scripts/` and `examples/` through
  `core/reporting`, not from `flight run`.

## Data

- Discovery is Earth Search STAC (`core/data/discovery/stac_client.py`); imagery is
  Sentinel-2 L2A COGs on AWS.
- Landsat assets come back as `s3://usgs-landsat/...` on a requester-pays bucket.
  `stac_client` rewrites them to the landsatlook HTTPS mirror. Keep that rewrite.
- Sentinel-1 is retired (see "Retired" in `ARCHITECTURE.md`): real-mode flood has no input.
- Algorithms are fixed per hazard: `sar_threshold` flood, `dnbr` wildfire,
  `wind_damage` storm. Wildfire defaults to `max_cloud` 90 because post-fire scenes are
  cloudy; the others default to 30.

## Carry the CRS

Every product keeps the grid it was computed on.

- The analyze stage writes a `grid.json` sidecar (CRS, transform, shape). Export reads it
  and writes GeoTIFFs on the source UTM grid.
- GeoJSON is polygonised on that grid, then reprojected to EPSG:4326.
- If `grid.json` is missing or its shape disagrees, export falls back to a bbox
  placeholder in EPSG:4326 and logs a warning. Treat that warning as a bug, not a result.
- Never write a raster without its CRS and transform, and never assume degrees.
- An AOI is an area: Polygon or MultiPolygon. `api/models/requests.py` still accepts any
  GeoJSON type; the Polygon-only check lived in the retired control plane.

## Real or fake

- Discover failure writes mock item ids and carries on; real mode then fails at ingest.
- `--synthetic` generates arrays. Synthetic output is for exercising code paths only; it
  never goes in a README, figure, report or claim. Analyzers once fabricated severity
  statistics; check any number against the rasters that made it.

## Retired, do not revive without a spec

The LLM control plane and the MAIA showcase were deprecated on 2026-03-18. Neither is on
`main`. See "Retired" in `ARCHITECTURE.md`.

## Working here

- Tests: `python -m pytest <file> --import-mode=importlib`, one file at a time. Not
  `run_tests.py`.
- Boundaries: `lint-imports` must stay green (5 contracts, `.importlinter`).
- `zarr.DirectoryStore` is gone in zarr 3; 10 tests in `tests/test_ingestion.py` fail on it.
