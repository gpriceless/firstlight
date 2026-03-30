## ADDED Requirements

### Requirement: AOI Column Type
The PostGIS jobs table SHALL store area-of-interest geometries as `GEOMETRY(MULTIPOLYGON, 4326)`. At insert time, any incoming `POLYGON` geometry MUST be promoted to `MULTIPOLYGON` via `ST_Multi()` so that all stored geometries share a single concrete type. Callers MAY submit either `POLYGON` or `MULTIPOLYGON` GeoJSON; the schema absorbs the difference transparently.

#### Scenario: Polygon input promoted to MultiPolygon
- **WHEN** a job is created with a single-polygon AOI
- **THEN** the stored geometry is a `MULTIPOLYGON` with one ring, indistinguishable in structure from a natively multi-part AOI

#### Scenario: MultiPolygon input stored unchanged
- **WHEN** a job is created with a multi-part AOI (e.g., two separated flood zones)
- **THEN** the stored geometry is the same `MULTIPOLYGON` with both parts intact and no data is lost

---

### Requirement: Geography Casts for Metric Calculations
All distance and area calculations against the `aoi` column SHALL cast to `::geography` at query time rather than storing a separate geography column. `ST_Area(aoi::geography)` MUST return square metres with sub-1% error at global latitudes. `ST_DWithin(aoi::geography, point::geography, metres)` MUST accept a distance threshold in metres without requiring a reprojection step from the caller.

#### Scenario: Area query returns metric result
- **WHEN** a query computes the area of a stored AOI
- **THEN** the result is in square metres and does not require the caller to apply a projection correction

#### Scenario: Proximity filter uses metre threshold
- **WHEN** a query filters jobs whose AOI is within a specified metre radius of a coordinate
- **THEN** the filter uses the geography cast and the threshold is expressed in metres, not degrees

---

### Requirement: Spatial Index on AOI
A GIST index SHALL exist on the `aoi` column of the jobs table. The index MUST be created before the table is populated and MUST be used automatically by the query planner for bounding-box overlap queries (`&&` operator) and `ST_Intersects` predicates.

#### Scenario: Bounding-box intersection query uses index
- **WHEN** a query selects jobs whose AOI intersects a given bounding envelope
- **THEN** the query planner uses the GIST index and does not perform a sequential scan on tables with more than one row

---

### Requirement: AOI Validity Constraints
The schema SHALL enforce three validity constraints on every inserted or updated `aoi` value. First, `ST_IsValid(aoi)` MUST be true — self-intersecting or otherwise malformed geometries are rejected. Second, `ST_IsEmpty(aoi)` MUST be false — empty geometry collections are rejected. Third, the area MUST fall within a reasonable operational bound (greater than 0.01 km² and no larger than 5,000,000 km², approximately half the continental United States) to prevent runaway processing jobs.

#### Scenario: Self-intersecting geometry rejected
- **WHEN** a job submission contains a bowtie or figure-eight polygon
- **THEN** the insert fails with a constraint violation before any pipeline work begins

#### Scenario: Empty geometry rejected
- **WHEN** a job submission contains a geometry with no coordinates
- **THEN** the insert fails with a constraint violation

#### Scenario: Oversized AOI rejected
- **WHEN** a job submission contains an AOI larger than the configured maximum area
- **THEN** the insert fails with a constraint violation and the caller receives a clear error

---

### Requirement: Computed Area and Bounding Box Columns
The jobs table SHALL expose two denormalized spatial columns populated automatically on insert and kept in sync on update. `aoi_area_km2` SHALL be a `NUMERIC(12, 4)` column storing `ST_Area(aoi::geography) / 1e6` (area in square kilometres). `bbox` SHALL store the axis-aligned bounding rectangle as `GEOMETRY(POLYGON, 4326)` produced by `ST_Envelope(aoi)`, enabling cheap bounding-box queries without recomputing from the full geometry.

**Implementation note — generated column feasibility:** PostgreSQL `GENERATED ALWAYS AS ... STORED` columns require all referenced functions to be marked `IMMUTABLE`. `ST_Envelope(aoi)` is IMMUTABLE in PostGIS 3.4 and MAY be implemented as a generated column. `ST_Area(aoi::geography)`, however, involves a geometry-to-geography cast that is not guaranteed IMMUTABLE across all PostGIS builds; if the migration fails with an error about non-immutable functions, implement `aoi_area_km2` as a trigger-maintained column instead: create an `AFTER INSERT OR UPDATE OF aoi` trigger that executes `NEW.aoi_area_km2 := ST_Area(NEW.aoi::geography) / 1e6`. Prefer the generated column approach if the PostGIS version supports it; fall back to the trigger approach otherwise. The engineering task MUST verify which approach works against the target PostGIS 3.4 image before committing the migration.

#### Scenario: Area column populated on insert
- **WHEN** a job is created with a valid AOI
- **THEN** `aoi_area_km2` contains the correct area in km² with four decimal places and required no explicit value from the caller

#### Scenario: Bounding box column populated on insert
- **WHEN** a job is created with a multi-part AOI
- **THEN** `bbox` contains the minimum enclosing rectangle covering all parts, usable in map viewport calculations without loading the full geometry

---

### Requirement: GeoJSON Round-Trip
The schema SHALL support lossless GeoJSON serialization and deserialization for every stored AOI. `ST_AsGeoJSON(aoi)` MUST return a valid RFC 7946 GeoJSON geometry string with coordinates in longitude/latitude order. `ST_GeomFromGeoJSON(input)` MUST be the canonical write path for converting incoming GeoJSON to a PostGIS geometry before the `ST_Multi()` promotion step.

#### Scenario: Stored AOI serializes to valid GeoJSON
- **WHEN** a job's AOI is read via the API
- **THEN** the coordinate array is in `[longitude, latitude]` order and validates against the GeoJSON specification

#### Scenario: GeoJSON input deserializes without precision loss
- **WHEN** a GeoJSON polygon with six-decimal-place coordinates is written and immediately read back
- **THEN** the returned coordinates match the input to at least five decimal places (~1 metre precision)

---

### Requirement: PostGIS Docker Image
The Docker Compose service for PostgreSQL SHALL use the image `postgis/postgis:15-3.4-alpine` in place of `postgres:15-alpine`. The image swap MUST be the only change required to the service definition — port mapping, volume mounts, and environment variables remain unchanged. The PostGIS extension MUST be available in the database without any additional `CREATE EXTENSION` step being required of application code at startup (the image initializes it automatically).

#### Scenario: PostGIS functions available after compose up
- **WHEN** the stack is started from a clean state with `docker compose up`
- **THEN** `ST_GeomFromGeoJSON`, `ST_Area`, and `ST_IsValid` are callable without any manual extension installation

#### Scenario: Existing volume data survives image swap
- **WHEN** an existing Postgres data volume is used with the PostGIS image
- **THEN** the database starts successfully and prior non-spatial tables are accessible
