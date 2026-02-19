# Project Context

## Purpose
FirstLight is a geospatial event intelligence platform that converts (area, time window, event type) into decision products. It processes satellite imagery (Sentinel-1 SAR, Sentinel-2 optical, Landsat) to detect floods, wildfires, and storms, producing analysis results with confidence scores and provenance tracking.

## Tech Stack
- **Language:** Python 3.12, fully async/await
- **API:** FastAPI with Uvicorn (4 workers)
- **Database:** PostgreSQL 15 (planned PostGIS), SQLite (current state backend via aiosqlite)
- **Cache/Queue:** Redis 7
- **Geospatial:** GDAL 3.8+, rasterio, xarray, geopandas, shapely 2.0+, pyproj, pystac/pystac-client
- **Container:** Docker Compose (postgres:15-alpine, redis:7-alpine)
- **Testing:** pytest, 518+ tests across flood/wildfire/storm/schemas

## Project Conventions

### Code Style
- Type hints on all function signatures
- Pydantic models for data validation and serialization
- Dataclasses for internal state objects
- Async/await throughout (no sync blocking calls)

### Architecture Patterns
- **Agent System:** BaseAgent ABC with AgentRegistry and MessageBus (`agents/base.py`)
- **State Machine:** ExecutionStage enum with SQLite-backed StateManager (`agents/orchestrator/state.py`)
- **Algorithm Registry:** AlgorithmMetadata dataclass with category/event_type/resource metadata (`core/analysis/library/registry.py`)
- **Pipeline:** Discovery → Ingestion → Normalization → Analysis → Quality → Reporting
- **API layers:** Public routes at `/api/v1/`, internal status at `/api/v1/status`

### Testing Strategy
- `./run_tests.py` runner with category filtering (flood, wildfire, schemas, etc.)
- Unit tests per algorithm, integration tests per pipeline stage
- Synthetic raster fixtures for deterministic testing

### Git Workflow
- Feature branches from `main`
- Conventional commits (feat:, fix:, chore:, docs:)
- PR-based merges

## Domain Context
- **AOIs** (Areas of Interest) are GeoJSON polygons/multipolygons in WGS84 (EPSG:4326)
- **Events** are natural disasters (flood, wildfire, storm) with spatial and temporal bounds
- **Algorithms** are sensor-specific detectors (SAR threshold, NDWI, dNBR, etc.)
- **STAC** (SpatioTemporal Asset Catalog) is the standard for discovering and publishing EO data
- **OGC API Processes** is the standard for exposing geospatial processing as HTTP services
- **MAIA Analytics** is the target integration partner — conversational geospatial layer

## Important Constraints
- All spatial storage in EPSG:4326; metric calculations use geography casts
- Must maintain backward compatibility with existing CLI (`flight` command) and agent system
- Auth system exists but defaults to disabled — security hardening required
- Current state backend is SQLite — migration to PostGIS is part of the control plane work

## External Dependencies
- **Earth Search (AWS):** STAC catalog for Sentinel/Landsat data discovery
- **Microsoft Planetary Computer:** Alternative STAC catalog
- **Copernicus Data Space:** Sentinel data access
- **USGS:** Landsat data access
