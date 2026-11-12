# Architecture: FirstLight

Last checked against the code: 2026-11-10, commit 1dc46c7. Read by the owner: 2026-11-11.

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
| `run_real_analysis.py`, `run_campfire_analysis.py` | by hand | One-off scripts against real scenes | Standalone, import little from the package. |
| `scripts/*.py`, `examples/*.py`, `florida_wildfire_test/render_site_imagery.py` | by hand | Report and imagery demos | Live but ad hoc. |

## Modules

Line counts are Python source lines. "Prod" means reachable by import from an entry point
above; "tests only" means only the test suite imports it.

| Module | Owns | Depends on | Must not depend on | Reach |
|---|---|---|---|---|
| `cli/` (6.3k) | Click commands; `run.py` (2,063 lines) holds the staged pipeline and the per-hazard Sentinel-2 loaders | `core.data`, `core.analysis`, `core.quality.sanity` (lazy imports inside functions) | `api` | prod |
| `api/` (9.6k) | FastAPI app, JWT auth, rate limit, SQLite via aiosqlite, webhooks | `core.intent`, `core.analysis.library`, `core.data.providers`, `openspec` | `cli` | prod, but a shell |
| `core/intent/` (1.0k) | Event taxonomy and resolver. Keyword matching, not a language model (`core/intent/classifier.py:3`) | nothing in `core` | the rest of `core` | prod |
| `core/data/` (23.9k) | Providers registry, STAC discovery (`stac_client.py`), ingestion (COG, Zarr, streaming, tiling, validation, enrichment, persistence) | nothing above it | `core.analysis`, `core.quality`, `core.reporting` | prod; `ingestion/enrichment` tests only; `broker.py` reached by nothing |
| `core/analysis/` (18.3k) | Algorithm library (flood, wildfire, storm baselines; U-Net and ensemble flood), assembly graph, execution runners (serial, tiled, Dask) and the router that picks one | `core.data`, `core.execution` | `core.quality`, `core.reporting` | library and execution prod; `assembly` tests only |
| `core/execution/` (3.3k) | Profiles, run state, tiling | nothing in `core` | analysis, quality, reporting | prod |
| `core/quality/` (9.0k) | Sanity checks, uncertainty, QA report | `core.analysis` at most | `core.reporting` | prod |
| `core/reporting/` (13.8k) | Templates, imagery renderers, maps (static, perimeters), PDF, interactive web report, census and infrastructure clients | `core.reporting.*` only | everything below it in reverse | prod via `scripts/` and `examples/`, not via the CLI |
| `openspec/` (0.3k) | JSON schema validator for event and intent specs | none | | prod (API) |

Totals from the reachability pass on 2026-11-09: 81.7k lines reachable from an entry
point, 7.7k reachable only from tests, 1.0k reachable from nothing. The same pass on the
tree before the delete gave 94.2k, 39.0k and 17.1k. Algorithms loaded through the
registry by name can show as tests only; the pass follows imports, not strings.

## Boundaries

All five hold today (`lint-imports`, 5 kept, 0 broken). A planted violation turned
three of them red, so the checker can fail.

- core does not import the api or cli
- api and cli do not import each other
- core layers: reporting over quality over analysis over data
- core.execution does not import analysis, quality or reporting
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

- God files: `cli/commands/run.py` 2,063, `core/quality/reporting/qa_report.py` 1,520.
- Two workflows: the boundary-contract check and the Tests workflow, which runs the
  whole suite except Docker-marked tests (`.github/workflows/tests.yml`).
- `core/analysis/execution/` (runner, checkpoints, Dask, tiled runner) is reached only by
  its tests since `flight analyze` stopped importing a runner class that never existed.
- Dead and test-only clusters with a revive or delete call each: `history/left-in-the-dust.md`.

## Retired

Deprecated on 2026-03-18 by the owner (`docs/branch-triage-llm-control-plane.md`). None
of it is on `main` and nothing here should be rebuilt without a new spec.

- The LLM control plane: control API, event stream, OGC API Processes and pygeoapi,
  pgSTAC, the PostGIS job schema, the Taskiq and Redis worker queue, tenant middleware.
  The code is kept only as tag `archive/llm-control-plane`. Its proposal still sits in
  `openspec/changes/add-llm-control-plane/` as a record, not as planned work.
- The MAIA showcase: demo scripts and notebooks built for MAIA, and the separate
  `maia_pipeline` repo. Not part of FirstLight.
- The triage listed a few things on that branch worth keeping; the triage file says
  which. The context lakehouse branch is kept as
  `refs/attic/origin-context-data-lakehouse`; reviving it needs a spec.

- Sentinel-1 ingestion, retired 2026-11-12. Discovery can search the `sentinel-1-grd`
  collection and `core/data/providers/sar/sentinel1.py` describes it, but no loader
  downloads or calibrates S1 scenes; the only S1 item ids the CLI writes are the mock
  ones in the discover fallback (`cli/commands/run.py:554`). So `flight run --event
  flood` has no real input and stops at analyze asking for SAR data. Earth Search
  GRD assets sit on a requester-pays bucket and are uncalibrated. A real flood path
  needs a spec: an analysis-ready S1 RTC source, or an optical water index on
  Sentinel-2.
- Landsat stays: it is read through the public landsatlook HTTPS mirror, not the
  requester-pays bucket (see `stac_client`).

An agent memory file written in February still described both as in progress. It is
kept verbatim in `history/left-in-the-dust.md` under "Old agent memory".
