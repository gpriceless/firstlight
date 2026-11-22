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
| Storm scene choice by clear share over the box | works: the same chooser as wildfire, asked for red and NIR | `tests/cli/test_storm_scene_choice.py::test_storm_run_records_the_chosen_scenes_and_clear_shares` fails on the old pick (11 May, all cloud) |
| Storm cloud and shadow mask from Sentinel-2 SCL | works: masked pixels are no damage and no-data confidence, and the run prints the masked share | `tests/cli/test_storm_scene_choice.py::test_storm_damage_map_does_not_count_cloud`; Max Road storm run 864.6 ha before, 295.5 ha after with 39.1% masked; validate 1.00 pass before, 0.87 fail on coverage 60.9% after |
| Same-day mosaic across tiles | works for wildfire and storm: discovery takes every tile of a day it keeps, even over the cloud limit; tiles of one day are one scene, read over the whole box; a part with no pixels is masked and the run says so in a sentence per side | `tests/cli/test_tile_mosaic.py`, `tests/test_discovery_size.py::test_a_kept_day_brings_its_tiles_over_the_cloud_limit`; Fort Myers box, storm, 15 Sep to 12 Oct 2022: 25 Sep pre-storm reads 17RLK and 17RMK (17RMK was dropped at 36.8% tile cloud), no part of the box without data |
| Synthetic run on a failed ingest | stops and exits 1; synthetic arrays are 512 by 512 for all three hazards | `tests/cli/test_run_failfast.py::TestRunCommandFailFast::test_synthetic_ingest_failure_exits_nonzero`, `test_synthetic_ingest_writes_full_size_arrays` |
| Full test suite in a shared editable venv | green locally: tests import this tree's `api`, time budgets skip unless CI or `-m benchmark` | `python -m pytest --import-mode=importlib`: 2776 passed, 30 skipped; `-m benchmark`: 9 passed |
| Normalise | rewrites each band as a COG in place, one file per band | `tests/cli/test_normalize_in_place.py` |
| Discovery item size | unknown: Earth Search assets carry no file size | `tests/test_discovery_size.py` |
| Discovery says which sources failed | works: each source is found, empty or failed in discovery.json, and a failed one is named in a sentence after discover and in the final summary; every source failing stops the run | `tests/test_discovery_sources.py`; Palisades box, wildfire, 20 Dec 2024 to 31 Jan 2025, Landsat search made to fail: 6 Sentinel-2 scenes, Landsat named as not searched in both places |
| Ingest range check | works in reflectance from each item's raster:bands scale, on the band mean | `tests/test_band_range_check.py`; same Palisades run: 12 image validations, no range warning (3 of 6 scenes warned before, on cloud tops near 1.7) |
| Ingest reads only the box | works on Earth Search COGs by HTTP range reads; a source that cannot be range-read is downloaded whole | `tests/cli/test_band_window.py`; Max Road 19 Apr to 31 May: 11,278,525,752 bytes of whole files before, 501,936,628 bytes fetched and 227,890,821 on disk after, same scenes and score 0.87 |
| Landsat ingest | broken: landsatlook redirects to the USGS login page. Ingest now refuses the HTML and names the redirect instead of saving it as `blue.tif` | `tests/cli/test_band_download.py`; `LC08_L2SP_015042_20260425_02_T1` SR_B2 live: text/html from ers.cr.usgs.gov/login, nothing saved |
| Sentinel-2 reflectance scale | DN/10000 is right for Earth Search L2A | `S2A_17RNJ_20260531_0_L2A`: baseline 05.12, `boa_offset_applied` true; water NIR p5 64 DN, 13.8% of the scene below 1000 DN |
