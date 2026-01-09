# Claude Context

Geospatial event intelligence platform that converts (area, time window, event type) into decision products.

## Core Concept

Situation-agnostic specifications enable the same agent-orchestrated pipelines to handle any hazard type (flood, wildfire, storm) without bespoke logic.

## Architecture

- **OpenSpec**: JSON Schema + YAML for event, intent, data source, pipeline, and provenance specifications
- **Data Broker**: Multi-source discovery with constraint evaluation, ranking, and open-source preference
- **Analysis Layer**: Algorithm library with dynamic pipeline assembly and hybrid rule/ML selection
- **Quality Control**: Automated sanity checks, cross-validation, consensus generation

## Key Files

- `OPENSPEC.md` - Complete system design specification
