# FirstLight status

One line per capability, each with its evidence. A capability missing here is not
verified either way.

| Capability | State | Evidence |
|---|---|---|
| Wildfire cloud and shadow mask from Sentinel-2 SCL | works | `tests/cli/test_wildfire_cloud_mask.py` on a real 1 km block of the Max Road scenes; full Max Road run masked 39.1% of the box |
| Validate reads the measurement layer (dNBR, then confidence) | works, under 2 s on 2.9 Mpx | `tests/cli/test_validate_wiring.py::TestValueCheckOnDnbr`; Max Road scores 0.87, failed on coverage 60.9% |
| `flight ingest --area` | refused with a pointer to `flight discover` | `tests/cli/test_ingest_wiring.py::test_area_alone_is_refused_not_given_placeholder_ids` |
| `flight analyze` tile estimate | counts the declared input raster | `tests/cli/test_analyze_wiring.py::TestAnalyzeInputSelection::test_tile_estimate_counts_the_declared_raster` |
| Scene choice by AOI cloud | missing: the midpoint split can pick a scene that is 100% cloud over the box | Max Road run with 4/19 to 5/31 picked post 11 May, 100% masked |
