## ADDED Requirements

### Requirement: pgSTAC Schema Installation
The system SHALL install the pgSTAC schema into the existing PostgreSQL instance as a dedicated schema namespace, separate from application tables, without requiring a separate database or service.

#### Scenario: First-time installation succeeds
- **WHEN** the application starts and pgSTAC schema objects are absent
- **THEN** the system runs pgSTAC migrations and the pgSTAC schema is present with all required tables and functions

#### Scenario: Re-installation is idempotent
- **WHEN** the application starts and pgSTAC schema objects already exist
- **THEN** the system completes startup without error and existing catalog data is preserved

#### Scenario: Schema namespace isolation
- **WHEN** pgSTAC schema objects are installed
- **THEN** they reside in a namespace that does not conflict with application tables in the default schema

---

### Requirement: STAC API Mount
The system SHALL mount stac-fastapi-pgstac as an ASGI sub-application at the `/stac` path prefix, exposing a STAC API v1.0-compliant endpoint accessible to any standard STAC client.

#### Scenario: STAC conformance endpoint reachable
- **WHEN** a client sends GET /stac/conformance
- **THEN** the response lists OGC API Features and STAC API conformance classes

#### Scenario: Collections and items browsable
- **WHEN** a client sends GET /stac/collections
- **THEN** the response lists all registered STAC Collections in JSON

#### Scenario: Mount does not interfere with other routes
- **WHEN** requests are sent to /control/v1/ or /oapi/ paths
- **THEN** those routes are handled by their respective handlers without interference from the /stac mount

---

### Requirement: Analysis Result Publishing
Every completed analysis result MUST be published as a STAC Item in the pgSTAC catalog. Publishing SHALL occur as part of the pipeline's reporting stage, after quality checks pass.

#### Scenario: Result published after successful analysis
- **WHEN** a pipeline job transitions to the COMPLETE phase
- **THEN** a STAC Item representing that result is upserted into the pgSTAC catalog within the same transaction or immediately after commit

#### Scenario: Result is queryable by area and time
- **WHEN** a STAC client sends a POST /stac/search with a bbox and datetime filter matching the result
- **THEN** the published Item appears in the response features

#### Scenario: Failed jobs do not produce Items
- **WHEN** a pipeline job fails before reaching quality checks
- **THEN** no STAC Item is published for that job

---

### Requirement: Processing Extension Fields
Every published STAC Item MUST include the `processing` extension with fields that describe the algorithm level, software provenance, and processing lineage.

#### Scenario: Processing level recorded
- **WHEN** a STAC Item is published
- **THEN** the Item's properties include `processing:level` set to `L3`

#### Scenario: Software version recorded
- **WHEN** a STAC Item is published
- **THEN** the Item's properties include `processing:software` as an object mapping the algorithm name to its version string

#### Scenario: Lineage description recorded
- **WHEN** a STAC Item is published
- **THEN** the Item's properties include `processing:lineage` as a human-readable string naming the algorithm and sensor used to produce the result

---

### Requirement: Provenance Links
Every published STAC Item MUST include `derived_from` links pointing to the source imagery STAC Items that were ingested to produce the result.

#### Scenario: Source items linked
- **WHEN** a STAC Item is published and the pipeline recorded one or more source STAC Item URIs during ingestion
- **THEN** the Item contains one `derived_from` link per source Item, each with `rel=derived_from` and the source Item's canonical URI as `href`

#### Scenario: Missing source URIs handled gracefully
- **WHEN** a STAC Item is published and no source STAC Item URIs were recorded during ingestion
- **THEN** publishing succeeds and no `derived_from` links are emitted; a warning is logged

---

### Requirement: COG Output Format
All raster assets attached to published STAC Items MUST be written as Cloud-Optimized GeoTIFFs (COG) with a standard tiling and compression profile to enable range-request access and dynamic tiling via titiler-pgstac.

**Implementation note:** The existing pipeline already produces COG output via `COGConfig` in `agents/reporting/products.py` with 512x512 tile blocks, DEFLATE compression, horizontal differencing predictor, and overview factors [2, 4, 8, 16, 32]. The STAC publisher MUST reuse the existing `ProductGenerator.generate_raster()` output path (with `as_cog=True`) rather than introducing a separate COG conversion step. No new COG generation capability needs to be built; this requirement validates that the existing output format meets the STAC publishing profile.

#### Scenario: Output raster meets COG profile
- **WHEN** a raster result is written to storage by the existing `ProductGenerator` with `as_cog=True`
- **THEN** the file has TILED layout, DEFLATE compression, and 512x512 internal tile blocks, matching the `COGConfig` defaults

#### Scenario: COG asset registered in Item
- **WHEN** a STAC Item is published
- **THEN** the Item's assets map includes the COG file with `type=image/tiff; application=geotiff; profile=cloud-optimized` and a reachable `href`

---

### Requirement: Idempotent Publishing via Upsert
The system SHALL use pypgstac's upsert operation to publish STAC Items, ensuring that re-publishing the same result (same Item ID) overwrites the previous record rather than raising an error or creating a duplicate.

#### Scenario: Re-publishing same result is safe
- **WHEN** a STAC Item with a given ID is published a second time (e.g., after a pipeline retry)
- **THEN** the catalog contains exactly one Item with that ID, reflecting the most recent publication

#### Scenario: Distinct results coexist
- **WHEN** two pipeline jobs with different job IDs produce results over the same AOI and time window
- **THEN** each result is stored as a separate STAC Item with a unique ID
