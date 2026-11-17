# FirstLight lessons evicted

Text that left `LESSONS.md` at a weekly consolidation, kept verbatim. Each block says
where its rule went.

## 2026-11-16

Promoted: input chosen by sort order or nearest date became a CLAUDE.md line under
"Real or fake". Mocked boundary and checks that cannot fail became a CLAUDE.md line under
"Working here". Removing an export is caught by blast radius (refactor-prep skill) and the
API tests that import the app. Stand-in for a failed lookup was already a CLAUDE.md line.
The AOI and pytest entries were fixed and already in CLAUDE.md. The unique-value entry
was a single incident.

### Patterns

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

### Whatever file sorts first is not the input (5)
- Seen: flood analysis read the first GeoTIFF as SAR; `flight analyze` read whatever
  `rglob` returned first and could run the SAR threshold on a reflectance band (0%
  water where the VV band gave 100%); the validate stage read the first `.npy` in the
  results dir, the boolean burn mask, and the value check threw on it. That throw was
  scored 0 and gave Max Road its 0.67 FAILED; on dNBR it is 0.87, failed on coverage.
  `estimate_tiles` sized the run from the first GeoTIFF too.
- Do: pick an input by what the consumer declares it needs, or refuse and name it.
  `select_input_raster` (analyze) and `_resolve_measurement_layer` (validate) do.

### A failed lookup must not write a stand-in (2)
- Bug: discovery failure in `flight run` wrote three invented Sentinel-1 ids and went on;
  `flight ingest --area` with no discovery file invented `<source>_latest` items.
- Lesson: a missing input is an error at the stage that missed it. Retired sources
  fail with a usage error that says so (`RETIRED_SOURCES`).

### Entries

### 2026-11 The nearest scene was all cloud
- Bug: wildfire took the scenes nearest the window midpoint. For Max Road that was
  11 May, 0% of the box clear, so the burn map was all no data.
- Lesson: rank scenes by what they show of the box (SCL clear share), then by date.
  Scene-wide `eo:cloud_cover` ranks them wrong: 31 May is 58.9% for the tile, 15.1%
  over the box.

### 2026-11 Coverage passed on cloud
- Bug: dNBR filled every pixel, so a pre-fire scene 29% under cloud and shadow showed
  0% no data and the coverage check passed. The algorithm took a `cloud_mask` that no
  caller passed.
- Lesson: a no-data check is only as good as the writer that marks no data. Test it on
  a real raster with known cloud (`tests/fixtures/s2_cloud`) and show the unmasked run
  reporting none.

### 2026-11 Comparing against every unique value
- Bug: the dominant-value test looped `(data == v).sum()` over `np.unique(data)`. On 2.9
  million float pixels that ran past 290 s; `return_counts=True` takes under a second.

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

### A fixed chooser had a twin
- Wildfire scene choice was fixed to rank by clear share; storm kept its own copy of
  the old nearest-date pick for a day. Before closing a fix to a picker, grep for its
  siblings (`_find_*_scenes`) and route them through the fixed one, then delete them.
- Promoted to `CLAUDE.md` on its second occurrence, below.

### Storm counted cloud as damage
- Storm took the SCL clear-share chooser from wildfire but not its cloud mask, so cloud
  dropped NDVI and read as vegetation loss. Max Road: 864.6 ha of damage became
  295.5 ha once 39.1% of the box was masked, and validate went from a 1.00 pass to a
  0.87 fail on coverage, because masked confidence had been 0 rather than no data.
- Second fix made to one hazard's path but not its twin; promoted to `CLAUDE.md`.
