# Left in the dust

Every abandoned subsystem and large dead-code cluster in FirstLight, with one call each:
**revive with a spec** or **delete**. The owner approved the delete calls below, and they
were carried out on 2026-11-09. Everything deleted from git still lives at commit
`4f9d341` (`git show 4f9d341:<path>`); each dropped stash and the origin copy of the
lakehouse branch are kept as refs under `refs/attic/` in the local clone.

Checked 2026-11-05 at commit 109431f. Evidence for "reached by nothing" and "tests only"
is an AST import-reachability pass from the entry points in `ARCHITECTURE.md`, which
counts imports inside functions; vulture at 60% confidence (`vulture agents api cli core
openspec scripts examples run_*.py --min-confidence 60`, 1,342 lines of findings) agrees
on the clusters. Line counts are Python source lines.

## Code

| Item | Lines | Evidence | Call | Why |
|---|---|---|---|---|
| `agents/discovery`, `agents/pipeline`, `agents/quality`, `agents/reporting` | 14,837 | Imported by nothing, not even tests | deleted, last at `4f9d341` | The four worker agents were never registered with the orchestrator. The CLI does their job for real. |
| `agents/orchestrator`, `agents/base.py` | 5,506 | Only its own `main.py`; Dockerfile CMD names a module that does not exist | deleted, last at `4f9d341` | Orchestrates agents that do not exist. If the service comes back it will not be this design (see the redesign notes in the state report). |
| `tests/test_agents.py` | 2,544 | Defines its own copies of the agent classes and tests those | deleted, last at `4f9d341` | Tests nothing in `agents/`. Goes with the agents. |
| `core/resilience/` and `tests/test_resilience.py` | 11,641 | Tests only | deleted, last at `4f9d341` | Degraded-mode framework no pipeline calls. `flight run` fails hard instead, which is the right behaviour. |
| `core/analysis/forecast/` and `tests/test_forecast.py` | 5,195 | Tests only | deleted, last at `4f9d341` | No forecast product exists or is planned. |
| `core/analysis/fusion/` | 4,462 | Tests only | deleted, last at `4f9d341` | Multi-sensor fusion never reached the CLI; each hazard uses one algorithm. |
| `core/analysis/selection/` | 1,770 | Tests only | deleted, last at `4f9d341` | The CLI picks algorithms from a fixed map (`cli/commands/run.py:747`). |
| `core/data/cache/`, `core/data/storage/`, `core/data/selection/`, `core/data/evaluation/` | 7,883 | Tests only | deleted, last at `4f9d341` | Cache, object-storage and constraint engines with no caller. |
| `core/quality/validation/`, `core/quality/actions/` | 6,355 | Tests only | deleted, last at `4f9d341` | Cross-validation and action management never wired; `core.quality.sanity` is what runs. |
| `core/data/discovery/stac.py`, `provider_api.py`, `wms_wcs.py` | 1,041 | `stac.py` only from the dead agents; the other two from nothing | deleted, last at `4f9d341` | `stac_client.py` is the live discovery path. |
| `core/reporting/components/`, `core/reporting/templates/sections.py` | 694 | Imported by nothing | deleted, last at `4f9d341` | Superseded by the templates the scripts use. |
| `core/provenance/` | 0 | Empty package | deleted, last at `4f9d341` | Placeholder. |
| `core/analysis/execution/sedona_backend.py`, `sedona_adapters.py`, `tests/test_sedona_execution.py` | 3,689 | Re-exported from `core/analysis/execution/__init__.py`, called by nothing; pyspark is not a dependency | deleted, last at `4f9d341` | Spark scale-out has no cluster and goes against local-first. The re-export and the router's Spark paths went with it. |
| Tests for the deleted modules: `test_cache`, `test_constraints`, `test_fusion`, `test_fusion_core`, `test_quality`, `test_quality_validation`, `test_ranking`, `test_selection`, `test_fusion_core_comprehensive`, `test_k8s_manifests`; the cache classes in `test_ingestion`, the storage class in `test_tiling`, the actions and validation tests in `test_quality_integration` and `test_quality_reporting`, the worker and on-prem tests in `test_docker_compose` and `test_docker_builds` | | Exercised only deleted code | deleted, last at `4f9d341` | |
| `tests/api/test_intent_resolution.py` | 295 | Imported `EventIntent`, `SpatialExtent` and `TemporalWindow` from `api.models.requests`, which never had them | deleted, last at `4f9d341` | Written against an API that does not exist; it failed collection. |
| `core/reporting/imagery/{area_metrics,attribution,captions,impact_metrics,locator_map,algorithm_descriptions,methodology_text}.py` and their tests | 1,907, plus 3,181 of tests | Imported by `core/reporting/imagery/__init__.py` | revived | Committed to main on 2026-11-09 so a clean clone imports. |
| `api/` | 9,649 | Live code, but events are never processed and products are mock records | revive with a spec, or delete | The owner's call. See "the one decision" in the state report. |

## Infrastructure

| Item | Evidence | Call | Why |
|---|---|---|---|
| `deploy/aws`, `deploy/azure`, `deploy/kubernetes`, `deploy/edge`, `deploy/on-prem`, `deploy/gcp` (2,375 lines) | Never deployed; staging host is `staging.firstlight.example.com` | deleted, last at `4f9d341` | Template output, not infrastructure. |
| `.github/workflows/deploy-staging.yml`, `deploy-production.yml`, `docker-push.yml`, `docker-build.yml` | 43 of 43 staging runs and 17 of 17 Docker builds on main failed | deleted, last at `4f9d341` | Red on every push and they test nothing. Replaced by the boundary-contract workflow; a test workflow comes with the next fix. |
| `docker/worker/Dockerfile` | CMD prints "Worker ready" and exits | deleted, last at `4f9d341` | Placeholder. |
| Declared but unimported dependencies: spacy, nltk, s3fs, scikit-learn, loguru, planetary-computer | No import in `api cli core scripts examples openspec` or the tests | removed from `pyproject.toml` | Install weight for nothing. folium was added: `core/reporting/maps/folium_map.py` imports it. |

## Local only (not in git, main checkout)

These lived only on this machine, so they were moved, not deleted, into
`~/projects/firstlight-attic-2026-11-10/` with their relative paths kept. The owner
deletes the attic when sure. The real Max Road run output that sat in a tracker-named
folder is now `florida_wildfire_test/max-road-run/` in the checkout.

| Item | Call | Why |
|---|---|---|
| `google-cloud-sdk/` (692 MB, git-ignored) | moved to the attic | A vendored SDK; install it system-wide if GCP work returns. |
| Root completion summaries: `VIS-1.2_REQUEST_3_COMPLETION_SUMMARY.md`, `VIS-1.3_COMPLETION_SUMMARY.md`, `VIS14_TASK9_COMPLETE.md`, `VIS-1.5_COMPLETION_SUMMARY.md`, `GITHUB_PRESENTATION_ANALYSIS.md`, `SECURITY_AUDIT.md`, `AGENTS.md` (git-ignored) | moved to the attic | Agent run reports; the work they describe is in git. |
| Root smoke scripts `test_vis13_smoke.py`, `test_vis14_*.py`, `test_vis15_smoke.py`, `test_before_after_complete.py`, `generate_detection_overlay.py` (git-ignored) | moved to the attic | One-off checks outside the test suite. |
| Empty `__pycache__`-only dirs left from another branch: `core/context`, `core/events`, `core/ogc`, `core/stac`, `api/routes/control`, `agents/orchestrator/backends`, `workers`, `tests/{context,geospatial,ogc,stac,state}` | deleted | Bytecode only, no source. |

## Branches and history

`main` (138 commits) and the old local branches share no merge base: the old branches
predate the history rewrite of 2026-03-30. The old local `main` is kept as tag
`archive/local-main-2026-03-30`; the control plane branch is kept as tag
`archive/llm-control-plane`.

| Ref | Last commit | Evidence | Call |
|---|---|---|---|
| `refactor/rename-to-firstlight` | 2026-01-18 | Tree identical to `945f832` on main | deleted |
| `feature/multi-band-asset-download` | 2026-01-25 | Tree identical to `13c1481` on main | deleted |
| `feature/report-2.0-human-readable-reporting` | 2026-01-26 | Tree identical to `6980ab7` on main | deleted |
| `feature/vis-1.0` | 2026-01-26 | Tree identical to `a05d887` on main | deleted |
| `feature/context-data-lakehouse` (local) | 2026-02-19 | Ancestor of tag `archive/llm-control-plane` | deleted |
| `origin/feature/context-data-lakehouse` | 2026-02-19 | Same tree as the local branch, other commit ids, not on main; the control plane deprecated on 2026-03-18 | deleted; a rewritten copy of the local branch (same tree, other commit ids), so it is kept as `refs/attic/origin-context-data-lakehouse` |
| `feature/viz-storytelling-phase4-track-a`, `-track-b`, local and origin | 2026-05-16 | 0 commits ahead of main | deleted, local and origin |
| 7 stashes, 2026-01-26 to 2026-02-18 | | Superseded by main | dropped; each kept as `refs/attic/stash-N-<date>`. `stash@{0}` matches the imagery modules on main except one word in `methodology_text.py`, and also held design notes and a study note that never reached main |
| `generate_readme_images.py` (untracked) | | Draws a synthetic river valley; its images were shown in the README as product output | moved to the attic (the images left the README on 2026-11-09) |
| `openspec/changes/viz-storytelling-phase4/` (untracked) | | Proposal for map storytelling; its first part is on main, the rest never started | moved to the attic; revive with a spec if map storytelling is still wanted |
| `documentation/`, `mkdocs.yml`, `docs/study-guide/` (untracked) | | Generated docs from 2026-04-24, one study note | moved to the attic |
