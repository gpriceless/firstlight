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
- The LLM control plane and MAIA showcase are retired and off `main`. Do not revive them
  without a spec; see "Retired" in `ARCHITECTURE.md`.

## Data

- Discovery is Earth Search STAC (`core/data/discovery/stac_client.py`); imagery is
  Sentinel-2 L2A COGs on AWS.
- Landsat assets are `s3://usgs-landsat/...` (requester-pays). The landsatlook mirror
  `stac_client` rewrites them to now serves a login page: Landsat ingest is broken.
- Earth Search Sentinel-2 L2A pixels already have the 04.00 offset removed: DN/10000 is
  reflectance. Do not also apply the `raster:bands` offset of -0.1.
- Sentinel-1 is retired (see "Retired" in `ARCHITECTURE.md`): real-mode flood has no input.
- Algorithms are fixed per hazard: `sar_threshold` flood, `dnbr` wildfire,
  `wind_damage` storm. Wildfire defaults to `max_cloud` 90 because post-fire scenes are
  cloudy; the others default to 30. Wildfire and storm refuse a scene with no `scl.tif`
  and mask cloud and shadow with it (wildfire dNBR NaN, severity 255; storm confidence
  NaN). A fix to one optical hazard is not done until the other has it too.

## Carry the CRS

Every product keeps the grid it was computed on.

- The analyze stage writes a `grid.json` sidecar (CRS, transform, shape). Export reads it
  and writes GeoTIFFs on the source UTM grid.
- GeoJSON is polygonised on that grid, then reprojected to EPSG:4326.
- If `grid.json` is missing or its shape disagrees, export falls back to a bbox
  placeholder in EPSG:4326 and logs a warning. Treat that warning as a bug, not a result.
- Never write a raster without its CRS and transform, and never assume degrees.
- An AOI is an area: `SpatialSpec` refuses anything but Polygon and MultiPolygon.

## Real or fake

- Discover failure stops the run. Never write stand-in item ids for data that is not there.
- Pick an input by what its consumer needs (the declared band or layer, the clear share of
  the box), or refuse and name it. Sort order and nearest date picked wrong six times.
- `--synthetic` output exercises code paths only, never a README, figure, report or claim.
  Analyzers once fabricated severity statistics; check a number against its rasters.

## Working here

- Tests: `python -m pytest --import-mode=importlib`, the same command the Tests workflow
  runs. Docker and e2e tests are off by default; `-m docker` or `-m e2e` runs them.
- Test against a small real raster (`tests/fixtures/`) and assert the value that matters. A
  mocked rasterio, `isinstance(exit_code, int)`, or a no-data count nothing writes cannot fail.
- Boundaries: `lint-imports` must stay green (4 contracts, `.importlinter`).
