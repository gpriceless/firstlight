# Left in the dust

Every abandoned subsystem and large dead-code cluster in FirstLight, with one call each:
**revive with a spec** or **delete**. Nothing here has been deleted yet. Deleting is
irreversible, so the paths below go to the owner for approval first.

Checked 2026-11-05 at commit 109431f. Evidence for "reached by nothing" and "tests only"
is an AST import-reachability pass from the entry points in `ARCHITECTURE.md`, which
counts imports inside functions; vulture at 60% confidence (`vulture agents api cli core
openspec scripts examples run_*.py --min-confidence 60`, 1,342 lines of findings) agrees
on the clusters. Line counts are Python source lines.

## Code

| Item | Lines | Evidence | Call | Why |
|---|---|---|---|---|
| `agents/discovery`, `agents/pipeline`, `agents/quality`, `agents/reporting` | 14,837 | Imported by nothing, not even tests | delete | The four worker agents were never registered with the orchestrator. The CLI does their job for real. |
| `agents/orchestrator`, `agents/base.py` | 5,506 | Only its own `main.py`; Dockerfile CMD names a module that does not exist | delete | Orchestrates agents that do not exist. If the service comes back it will not be this design (see the redesign notes in the state report). |
| `tests/test_agents.py` | 2,544 | Defines its own copies of the agent classes and tests those | delete | Tests nothing in `agents/`. Goes with the agents. |
| `core/resilience/` and `tests/test_resilience.py` | 11,641 | Tests only | delete | Degraded-mode framework no pipeline calls. `flight run` fails hard instead, which is the right behaviour. |
| `core/analysis/forecast/` and `tests/test_forecast.py` | 5,195 | Tests only | delete | No forecast product exists or is planned. |
| `core/analysis/fusion/` | 4,462 | Tests only | delete | Multi-sensor fusion never reached the CLI; each hazard uses one algorithm. |
| `core/analysis/selection/` | 1,770 | Tests only | delete | The CLI picks algorithms from a fixed map (`cli/commands/run.py:747`). |
| `core/data/cache/`, `core/data/storage/`, `core/data/selection/`, `core/data/evaluation/` | 7,883 | Tests only | delete | Cache, object-storage and constraint engines with no caller. |
| `core/quality/validation/`, `core/quality/actions/` | 6,355 | Tests only | delete | Cross-validation and action management never wired; `core.quality.sanity` is what runs. |
| `core/data/discovery/stac.py`, `provider_api.py`, `wms_wcs.py` | 1,041 | `stac.py` only from the dead agents; the other two from nothing | delete | `stac_client.py` is the live discovery path. |
| `core/reporting/components/`, `core/reporting/templates/sections.py` | 694 | Imported by nothing | delete | Superseded by the templates the scripts use. |
| `core/provenance/` | 0 | Empty package | delete | Placeholder. |
| `core/analysis/execution/sedona_backend.py`, `sedona_adapters.py`, `tests/test_sedona_execution.py` | 3,689 | Re-exported from `core/analysis/execution/__init__.py`, called by nothing; pyspark is not a dependency | delete | Spark scale-out has no cluster and goes against local-first. Remove the re-export with it. |
| Uncommitted `core/reporting/imagery/{area_metrics,attribution,captions,impact_metrics,locator_map,algorithm_descriptions,methodology_text}.py` and their tests (main checkout only) | 1,907, plus 3,181 of tests | Committed `core/reporting/imagery/__init__.py` imports them since 2026-02-12 | revive | Without them a clean clone cannot import `core.reporting.imagery`. Commit them; no spec needed, it is a fix. |
| `api/` | 9,649 | Live code, but events are never processed and products are mock records | revive with a spec, or delete | The owner's call. See "the one decision" in the state report. |

## Infrastructure

| Item | Evidence | Call | Why |
|---|---|---|---|
| `deploy/aws`, `deploy/azure`, `deploy/kubernetes`, `deploy/edge`, `deploy/on-prem`, `deploy/gcp` (2,375 lines) | Never deployed; staging host is `staging.firstlight.example.com` | delete | Template output, not infrastructure. |
| `.github/workflows/deploy-staging.yml`, `deploy-production.yml`, `docker-push.yml`, `docker-build.yml` | 43 of 43 staging runs and 17 of 17 Docker builds on main failed | delete | Red on every push and they test nothing. Replaced by the boundary-contract workflow; a test workflow comes with the next fix. |
| `docker/worker/Dockerfile` | CMD prints "Worker ready" and exits | delete | Placeholder. |
| Declared but unimported dependencies: spacy, nltk, s3fs, scikit-learn, loguru, planetary-computer | `grep` for an import in `agents api cli core` returns nothing | delete | Install weight for nothing; spacy alone is large. |

## Local only (not in git, main checkout)

These live only on this machine, so deleting them loses them. Owner approval per path.

| Item | Call | Why |
|---|---|---|
| `google-cloud-sdk/` (692 MB, git-ignored) | delete | A vendored SDK; install it system-wide if GCP work returns. |
| Root completion summaries: `VIS-1.2_REQUEST_3_COMPLETION_SUMMARY.md`, `VIS-1.3_COMPLETION_SUMMARY.md`, `VIS14_TASK9_COMPLETE.md`, `VIS-1.5_COMPLETION_SUMMARY.md`, `GITHUB_PRESENTATION_ANALYSIS.md`, `SECURITY_AUDIT.md`, `AGENTS.md` (git-ignored) | delete | Agent run reports; the work they describe is in git. |
| Root smoke scripts `test_vis13_smoke.py`, `test_vis14_*.py`, `test_vis15_smoke.py`, `test_before_after_complete.py`, `generate_detection_overlay.py` (git-ignored) | delete | One-off checks outside the test suite. |
| Empty `__pycache__`-only dirs left from another branch: `core/context`, `core/events`, `core/ogc`, `core/stac`, `api/routes/control`, `agents/orchestrator/backends`, `workers`, `tests/{context,geospatial,ogc,stac,state}` | delete | Bytecode only, no source. |

## Branches and history

`main` (138 commits) and the old local branches share no merge base: the old branches
predate the history rewrite of 2026-03-30. The old local `main` is kept as tag
`archive/local-main-2026-03-30`; the control plane branch is kept as tag
`archive/llm-control-plane`.

| Ref | Last commit | Evidence | Call |
|---|---|---|---|
| `refactor/rename-to-firstlight` | 2026-01-18 | Tree identical to `945f832` on main | delete |
| `feature/multi-band-asset-download` | 2026-01-25 | Tree identical to `13c1481` on main | delete |
| `feature/report-2.0-human-readable-reporting` | 2026-01-26 | Tree identical to `6980ab7` on main | delete |
| `feature/vis-1.0` | 2026-01-26 | Tree identical to `a05d887` on main | delete |
| `feature/context-data-lakehouse` (local) | 2026-02-19 | Ancestor of tag `archive/llm-control-plane` | delete |
| `origin/feature/context-data-lakehouse` | 2026-02-19 | Same tree as the tag, 70 commits not on main; the control plane deprecated on 2026-03-18 | delete (the tag keeps it locally) |
| `feature/viz-storytelling-phase4-track-a`, `-track-b`, local and origin | 2026-05-16 | 0 commits ahead of main | delete |
| 7 stashes, 2026-01-26 to 2026-02-18 | | `stash@{0}` is byte-identical to the uncommitted imagery modules | delete after the imagery modules are committed |
| `generate_readme_images.py` (untracked) and `docs/images/detection_overlay.png`, `before_after.png` | | The script draws a synthetic river valley; the README shows those images as product output | delete, and replace the images with output from the real Max Road run |
| `openspec/changes/viz-storytelling-phase4/` (untracked) | | Proposal for map storytelling; its first part is on main, the rest never started | revive with a spec if map storytelling is still wanted, else delete |
| `documentation/`, `mkdocs.yml`, `docs/study-guide/` (untracked) | | Generated docs from 2026-04-24, one study note | delete (regenerate from code if wanted) |
