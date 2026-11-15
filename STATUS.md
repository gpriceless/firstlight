# FirstLight status

One line per capability, each with its evidence. A capability missing here is not
verified either way.

| Capability | State | Evidence |
|---|---|---|
| Wildfire cloud and shadow mask from Sentinel-2 SCL | works | `tests/cli/test_wildfire_cloud_mask.py` on a real 1 km block of the Max Road scenes; full Max Road run masked 39.1% of the box |
| Validate reads the measurement layer (dNBR, then confidence) | works, under 2 s on 2.9 Mpx | `tests/cli/test_validate_wiring.py::TestValueCheckOnDnbr`; Max Road scores 0.87, failed on coverage 60.9% |
| `flight ingest --area` | refused with a pointer to `flight discover` | `tests/cli/test_ingest_wiring.py::test_area_alone_is_refused_not_given_placeholder_ids` |
| `flight analyze` tile estimate | counts the declared input raster | `tests/cli/test_analyze_wiring.py::TestAnalyzeInputSelection::test_tile_estimate_counts_the_declared_raster` |
| Wildfire scene choice by clear share over the box | works: each side of the event takes the scene with most of the box clear in SCL | `tests/cli/test_wildfire_cloud_mask.py::test_a_fully_clouded_scene_loses_to_a_clearer_one`; Max Road 4/19 to 5/31 now takes 9 May (70.9% clear) and 31 May (84.9%), not 11 May (0%); score 0.87, failed on coverage 60.9% |
| Storm scene choice | still the nearest scene to the window midpoint, no cloud layer | `_find_sentinel2_storm_pre_post_scenes` |
| Landsat ingest | broken: landsatlook redirects to the USGS login page. Ingest now refuses the HTML and names the redirect instead of saving it as `blue.tif` | `tests/cli/test_band_download.py`; `LC08_L2SP_015042_20260425_02_T1` SR_B2 live: text/html from ers.cr.usgs.gov/login, nothing saved |
| Sentinel-2 reflectance scale | DN/10000 is right for Earth Search L2A | `S2A_17RNJ_20260531_0_L2A`: baseline 05.12, `boa_offset_applied` true; water NIR p5 64 DN, 13.8% of the scene below 1000 DN |
