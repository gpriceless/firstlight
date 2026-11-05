# Architecture: FirstLight

Last checked against the code: 2026-11-05, commit 109431f. Read by the owner: not yet.

Sources: `.harness/import-graph.json` (regex map), an AST reachability pass that also
counts imports inside functions, the import-linter contracts in `.importlinter`, and
reading the entry points listed below. Anything not opened is marked "inferred from
imports".

## What it is for

Turn an area, a time window and an event type (flood, wildfire, storm) into satellite
derived products: an extent raster, a severity map, GeoJSON, and an HTML or PDF report.
The CLI is the only path that does this end to end today.

## Entry points

| Entry point | How it starts | What it serves | State |
|---|---|---|---|
| `cli/main.py` | `flight` console script (`pyproject.toml`) | `discover`, `ingest`, `analyze`, `validate`, `export`, `run`, `status`, `resume` | Live. `flight run` is the real pipeline. |
| `api/main.py` | `uvicorn api.main:app` (Dockerfiles) | FastAPI: events, status, products, catalog, health | Shell. Events are stored and never processed; products are generated mock records (`api/routes/products.py:42`). |
| `agents/orchestrator/main.py` | `python -m agents.orchestrator.main` | Async orchestrator that delegates to discovery, pipeline, quality and reporting agents | Not wired. Nothing registers the four worker agents, and the Dockerfile CMD points at `agents.orchestrator.agent`, which does not exist. |
| `docker/worker/Dockerfile` | compose `worker` service | Prints "Worker ready" and exits | Placeholder. |
| `run_real_analysis.py`, `run_campfire_analysis.py` | by hand | One-off scripts against real scenes | Standalone, import little from the package. |
| `scripts/*.py`, `examples/*.py`, `florida_wildfire_test/render_site_imagery.py` | by hand | Report and imagery demos | Live but ad hoc. |

## Modules

Line counts are Python source lines. "Prod" means reachable by import from an entry point
above; "tests only" means only the test suite imports it.

| Module | Owns | Depends on | Must not depend on | Reach |
|---|---|---|---|---|
| `cli/` (6.3k) | Click commands; `run.py` (2,063 lines) holds the staged pipeline and the per-hazard Sentinel-2 loaders | `core.data`, `core.analysis`, `core.quality.sanity` (lazy imports inside functions) | `api`, `agents` | prod |
| `api/` (9.6k) | FastAPI app, JWT auth, rate limit, SQLite via aiosqlite, webhooks | `core.intent`, `core.analysis.library`, `core.data.providers`, `openspec` | `cli`, `agents` | prod, but a shell |
| `agents/` (20.3k) | Orchestrator plus discovery, pipeline, quality and reporting agents on a shared base | `core.*` | `api`, `cli` | orchestrator only; the four worker agents (14.8k) are reached by nothing, not even tests |
| `core/intent/` (1.0k) | Event taxonomy and resolver. Keyword matching, not a language model (`core/intent/classifier.py:3`) | nothing in `core` | the rest of `core` | prod |
| `core/data/` (32.8k) | Providers registry, STAC discovery, ingestion (COG, Zarr, streaming, tiling, validation), cache, storage | nothing above it | `core.analysis`, `core.quality`, `core.reporting` | ingestion, discovery, providers prod; `cache`, `storage`, `selection`, `evaluation` tests only |
| `core/analysis/` (31.3k) | Algorithm library (flood, wildfire, storm baselines; U-Net and ensemble flood), assembly graph, execution runners (serial, Dask, Sedona/Spark) | `core.data`, `core.execution` | `core.quality`, `core.reporting` | library, assembly, execution prod; `forecast`, `fusion`, `selection` tests only |
| `core/execution/` (3.3k) | Profiles, run state, tiling | nothing in `core` | analysis, quality, reporting | prod |
| `core/quality/` (15.3k) | Sanity checks, uncertainty, QA report; validation and actions | `core.analysis` at most | `core.reporting` | sanity, uncertainty, reporting prod; `validation`, `actions` tests only |
| `core/resilience/` (10.5k) | Degraded mode, failure handling, fallbacks, quality assessment | nothing in `core` | analysis, quality, reporting | tests only |
| `core/reporting/` (12.6k) | Templates, imagery renderers, maps (static, perimeters), PDF, interactive web report, census and infrastructure clients | `core.reporting.*` only | everything below it in reverse | prod via `scripts/` and `examples/`, not via the CLI |
| `openspec/` (0.3k) | JSON schema validator for event and intent specs | none | | prod (API) |

Totals from the reachability pass: about 87.8k lines reachable from an entry point,
39.0k reachable only from tests, 16.6k reachable from nothing.

## Boundaries

All five hold today (`lint-imports`, 5 kept, 0 broken). A planted violation turned
three of them red, so the checker can fail.

- core does not import the api, cli or agents
- api, cli and agents do not import each other
- core layers: reporting over quality over analysis over data
- core.execution and core.resilience do not import analysis, quality or reporting
- core.intent imports nothing else in core

## Data flow

The one path that runs on real data, `flight run --bbox ... --event ...`:

1. discover: `core.data.discovery.stac_client.discover_data` queries Earth Search STAC.
   On error it writes mock item ids and carries on (`cli/commands/run.py:548`); real
   mode then fails at ingest.
2. ingest: downloads COG bands into the work dir. Real mode fails hard on error;
   `--synthetic` generates arrays instead (`cli/commands/run.py:649`).
3. analyze: picks the per-hazard algorithm (`sar_threshold`, `dnbr`, `wind_damage`),
   reads pre and post Sentinel-2 scenes, runs it, writes rasters and statistics.
4. validate: `core.quality.sanity` checks on the outputs.
5. export: GeoTIFF with the source grid's CRS and transform, GeoJSON by polygonising.

Reports (HTML, PDF, maps, before and after imagery) are produced by `scripts/` and
`examples/` from `core.reporting`, not by `flight run`.

The API path (`POST /events` to resolve intent, queue, process, `GET /products`) stops
after "queue". Nothing consumes the queue.

## Where state lives

- `flight run` work dirs: `<output>/` with a `.workflow_state.json` per run (`cli/commands/run.py:69`).
- API: SQLite file via aiosqlite (`api/database.py`); Redis only as an optional
  rate-limit store (`api/rate_limit.py`). Compose also declares Postgres, which no code
  imports a driver for.
- External: Earth Search STAC and Sentinel-2 COGs on AWS; census and infrastructure
  clients in `core/reporting/data` call public APIs.

## Known debt

- Committed `core/reporting/imagery/__init__.py` imports seven modules that were never
  committed. On a clean clone `core.reporting.imagery`, `core.reporting.web` and
  everything that imports them fail to import, and 5 test modules fail to collect.
- The 3 modules in `tests/api/` fail to collect (`No module named 'api.dependencies'`)
  while `python -c "import api.database"` works from the repo root. Suspected cause:
  the `tests/api` package shadows the top-level `api` package. Not traced further.
- `tests/test_agents.py` defines its own copies of the agent classes and tests those,
  not the code in `agents/`.
- God files: `cli/commands/run.py` 2,063, `agents/base.py` 1,706,
  `core/quality/reporting/qa_report.py` 1,520, `core/analysis/execution/sedona_backend.py` 1,487.
- Declared dependencies with no importer in the package: spacy, nltk, s3fs,
  scikit-learn, loguru, planetary-computer.
- Every GitHub Actions run on main fails: staging deploy targets
  `staging.firstlight.example.com` (43 of 43 runs failed at SSH setup), the Docker build
  fails after an hour (17 of 17). No workflow runs the tests.
- Test suite on 2026-11-05, one file at a time with `--import-mode=importlib`: 3,223
  passed, 58 failed, 15 skipped; 13 files fail to collect (the missing imagery modules
  and the API tests); `tests/test_cli.py` and `tests/test_docker_builds.py` did not
  finish in 180 seconds. Most failures are tests that drifted from refactored code
  (`tests/cli/test_*_wiring.py`) or need pyspark (`tests/test_sedona_execution.py`, 23).
- Dead and test-only clusters with a revive or delete call each: `history/left-in-the-dust.md`.
