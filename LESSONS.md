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

## Entries

### 2026-02 AOI geometry type checked too late
- Bug: Point and LineString AOIs passed request validation and failed at a
  MULTIPOLYGON 4326 column.
- Lesson: validate the geometry type at the boundary, not only the coordinates. An AOI
  is Polygon or MultiPolygon.
- Still true: the column was in the retired control plane, but `api/models/requests.py`
  on `main` still accepts every GeoJSON type.

### 2026-02 Use pytest directly
- `run_tests.py` collects differently and trips `jsonschema` import errors. Use
  `python -m pytest`, one file at a time with `--import-mode=importlib`.

### 2026-11 zarr 3 removed DirectoryStore
- `core/data/ingestion/formats/zarr.py` still calls `zarr.DirectoryStore`; 10 tests in
  `tests/test_ingestion.py` fail on it. Not fixed yet.
