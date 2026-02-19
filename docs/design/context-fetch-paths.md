# Context Data Fetch Paths Recon

**Date:** 2026-02-19
**Task:** 2.1b from add-context-data-lakehouse
**Purpose:** Document which context data types are actually fetched by the pipeline today.

## Methodology

Inspected:
- `agents/discovery/main.py` (DiscoveryAgent)
- `agents/pipeline/main.py` (PipelineAgent)
- `agents/discovery/catalog.py` (CatalogQueryManager)
- `core/data/discovery/base.py` (DiscoveryResult)
- `core/data/broker.py` (DataBroker)

Searched for: buildings, infrastructure, weather, context, footprint, overture, osm,
building, hospital, temperature, precipitation, wind, noaa, era5, openmeteo.

## Findings

### 1. Datasets (satellite scenes)

**Status: FETCHED via DiscoveryAgent**

The DiscoveryAgent queries STAC catalogs (`CatalogQueryManager.query_catalogs`)
and returns `CatalogQueryResult` objects containing `List[DiscoveryResult]`.
Each DiscoveryResult represents a satellite scene with provider, dataset_id,
acquisition_time, resolution, cloud cover, and source URI.

**Integration point:** After `catalog_results` is populated in
`DiscoveryAgent._execute_discovery()` (around line 698). Implemented in Task 2.1.

**Geometry note:** DiscoveryResult does NOT carry per-scene geometry. The STAC
item geometry is available during parsing in `CatalogQueryManager._stac_item_to_result`
but is not forwarded into the DiscoveryResult dataclass. Context storage uses the
request bbox as a geometry approximation.

### 2. Building Footprints

**Status: NOT FETCHED -- no existing fetch path**

Zero references to buildings, footprints, OSM, Overture, or Microsoft building
data in either `agents/pipeline/main.py` or `agents/discovery/main.py`.
The PipelineAgent assembles analysis pipelines (flood detection, wildfire
analysis) but does not fetch any ancillary building data.

**Decision:** Create synthetic stub generator in `core/context/stubs.py` (Task 2.2).

### 3. Infrastructure (hospitals, fire stations, etc.)

**Status: NOT FETCHED -- no existing fetch path**

No references to infrastructure, hospitals, fire stations, HIFLD, or critical
facilities anywhere in the pipeline or discovery agents.

**Decision:** Create synthetic stub generator in `core/context/stubs.py` (Task 2.3).

### 4. Weather Observations

**Status: NOT FETCHED -- no existing fetch path**

No references to weather, temperature, precipitation, wind, NOAA, ERA5, or
OpenMeteo in the pipeline agent. The DiscoveryAgent handles "weather" as a
data_type category for STAC catalogs, but this fetches satellite-derived
weather products (raster scenes), not point observation data.

**Decision:** Create synthetic stub generator in `core/context/stubs.py` (Task 2.4).

## Summary

| Context Type     | Existing Fetch Path? | Integration Strategy           |
|-----------------|---------------------|-------------------------------|
| Datasets        | Yes (STAC catalogs) | Wire DiscoveryAgent to repo   |
| Buildings       | No                  | Synthetic stub + repo store    |
| Infrastructure  | No                  | Synthetic stub + repo store    |
| Weather         | No                  | Synthetic stub + repo store    |

## Future Work

Real data integration for buildings, infrastructure, and weather will require:
- Overture Maps or OSM Overpass API for building footprints
- HIFLD or OpenStreetMap for critical infrastructure
- NOAA/ERA5/OpenMeteo APIs for weather observations

These are separate features beyond the scope of the lakehouse schema work.
