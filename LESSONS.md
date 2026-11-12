# FirstLight lessons

Read at session start. Add an entry after every resolved bug. At most 40 entries; when
two entries share a root cause, merge them into a pattern.

Seeded on 2026-11-10 from the February agent memory file, cut to what still holds on
`main`. The original is kept verbatim in `history/left-in-the-dust.md`.

## Patterns

### Removing an export breaks importers you did not open (2)
- Seen: `_events_store` left `api/routes/products.py` and `status.py` when events moved
  to the database; both still imported it, and the app would not start.
- Why it spreads: `api/routes/__init__.py` imports every route module, so one broken
  import takes the whole app down.
- Do: before removing or renaming a module-level name, `grep -rn` the whole repo for it,
  tests included, and fix every importer in the same change.

### A mock of the boundary hid a real bug (3)
- Seen: a test mocked rasterio, so nobody saw `flight analyze` index a result object as
  a dict and fail every run; flood analysis read the first GeoTIFF as SAR, which is a
  Sentinel-2 band when only optical scenes were ingested; three `flight run` tests
  asserted only that the exit code was an int.
- Do: test against a small real raster and assert the value that matters. A test that
  cannot fail is not coverage.

## Entries

### 2026-02 AOI geometry type checked too late
- Bug: Point and LineString AOIs passed request validation and failed at a
  MULTIPOLYGON 4326 column.
- Lesson: validate the geometry type at the boundary, not only the coordinates. An AOI
  is Polygon or MultiPolygon.
- Fixed 2026-11-12: `SpatialSpec` refuses anything but Polygon and MultiPolygon
  (`tests/api/test_aoi_geometry.py`).

### 2026-02 Use pytest directly
- `run_tests.py` collected differently and tripped `jsonschema` import errors. It is
  deleted; `python -m pytest --import-mode=importlib` is the command CI runs.

### 2026-11 zarr 3 removed DirectoryStore
- Use `zarr.storage.LocalStore`, `create_array`, and `zarr_format=2` so numcodecs
  compressors and the xarray `_ARRAY_DIMENSIONS` attribute still work. Through xarray,
  the encoding key is `compressors`, a list.
