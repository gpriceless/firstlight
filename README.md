# Multiverse Dive

A cloud-native, agent-orchestrated geospatial event intelligence platform.

## Overview

Multiverse Dive transforms an area, time window, and event type into complete, reproducible decision products within hours. The platform autonomously discovers multi-sensor earth observation data, assembles validated analytical pipelines, performs quality checks, and publishes both machine-readable datasets and human-readable reports with full provenance.

## Key Capabilities

- **Situation-agnostic**: Structured event specifications allow the same agents and pipelines to handle floods, wildfires, storms, and other hazards
- **Autonomous data discovery**: Multi-source broker for optical, SAR, DEM, weather, and ancillary data with preference for open sources
- **Intelligent sensor selection**: Atmospheric-aware switching between optical and SAR, multi-sensor fusion, degraded mode handling
- **Dynamic pipeline assembly**: Algorithm selection based on data availability and constraints with hybrid rule-based and ML approaches
- **Quality control**: Automated plausibility checks, cross-validation, uncertainty quantification, and consensus generation
- **Full provenance**: Complete lineage tracking from source data through final products

## Documentation

- [OpenSpec](OPENSPEC.md) - Full system specification and design

## Tech Stack

- Python + FastAPI
- JSON Schema + YAML specifications
- Serverless-first deployment

## License

TBD
