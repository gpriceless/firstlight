-- Migration 007: Context Data Lakehouse Tables
--
-- Creates four context tables and a junction table for the Context Data
-- Lakehouse. Context data (satellite scenes, building footprints,
-- infrastructure facilities, weather observations) is stored as it flows
-- through the pipeline, creating a persistent geospatial knowledge base.
--
-- Deduplication uses natural keys (source, source_id) with
-- INSERT ... ON CONFLICT DO NOTHING. The junction table tracks which jobs
-- consumed which context rows and whether the data was freshly ingested
-- or reused from a previous job.
--
-- All statements are idempotent (IF NOT EXISTS).
--
-- Depends on: 001_control_plane_schema.sql (jobs table)


-- ============================================================================
-- Table: datasets
-- ============================================================================
-- Satellite scenes discovered and used by the pipeline. Each row represents
-- one scene from a STAC catalog. Geometry is MULTIPOLYGON because scene
-- footprints can be multi-part.

CREATE TABLE IF NOT EXISTS datasets (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(100) NOT NULL,
    source_id           VARCHAR(500) NOT NULL,
    geometry            GEOMETRY(MULTIPOLYGON, 4326) NOT NULL,
    properties          JSONB DEFAULT '{}',
    acquisition_date    TIMESTAMPTZ NOT NULL,
    cloud_cover         NUMERIC(5, 2),
    resolution_m        NUMERIC(8, 2),
    bands               TEXT[],
    file_path           TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT now(),
    ingested_by_job_id  UUID REFERENCES jobs(job_id) ON DELETE SET NULL,

    CONSTRAINT uq_datasets_source UNIQUE (source, source_id)
);

-- Spatial index for geospatial queries
CREATE INDEX IF NOT EXISTS idx_datasets_geometry_gist
    ON datasets USING GIST (geometry);

-- B-tree index for source filtering (composite UNIQUE covers prefix but explicit is clearer)
CREATE INDEX IF NOT EXISTS idx_datasets_source
    ON datasets (source);

-- B-tree index for temporal queries
CREATE INDEX IF NOT EXISTS idx_datasets_acquisition_date
    ON datasets (acquisition_date);


-- ============================================================================
-- Table: context_buildings
-- ============================================================================
-- Building footprints from OSM, Overture, or other sources. Each row is one
-- building polygon. Uses POLYGON (not MULTIPOLYGON) because individual
-- building footprints are single polygons.

CREATE TABLE IF NOT EXISTS context_buildings (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(100) NOT NULL,
    source_id           VARCHAR(500) NOT NULL,
    geometry            GEOMETRY(POLYGON, 4326) NOT NULL,
    properties          JSONB DEFAULT '{}',
    ingested_at         TIMESTAMPTZ DEFAULT now(),
    ingested_by_job_id  UUID REFERENCES jobs(job_id) ON DELETE SET NULL,

    CONSTRAINT uq_context_buildings_source UNIQUE (source, source_id)
);

-- Spatial index for geospatial queries
CREATE INDEX IF NOT EXISTS idx_context_buildings_geometry_gist
    ON context_buildings USING GIST (geometry);

-- B-tree index for source filtering
CREATE INDEX IF NOT EXISTS idx_context_buildings_source
    ON context_buildings (source);


-- ============================================================================
-- Table: context_infrastructure
-- ============================================================================
-- Critical infrastructure facilities (hospitals, fire stations, power plants,
-- bridges, etc.). Uses generic GEOMETRY because facilities can be points
-- (fire stations) or polygons (hospital campuses).

CREATE TABLE IF NOT EXISTS context_infrastructure (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(100) NOT NULL,
    source_id           VARCHAR(500) NOT NULL,
    geometry            GEOMETRY(GEOMETRY, 4326) NOT NULL,
    properties          JSONB DEFAULT '{}',
    ingested_at         TIMESTAMPTZ DEFAULT now(),
    ingested_by_job_id  UUID REFERENCES jobs(job_id) ON DELETE SET NULL,

    CONSTRAINT uq_context_infrastructure_source UNIQUE (source, source_id)
);

-- Spatial index for geospatial queries
CREATE INDEX IF NOT EXISTS idx_context_infrastructure_geometry_gist
    ON context_infrastructure USING GIST (geometry);

-- B-tree index for source filtering
CREATE INDEX IF NOT EXISTS idx_context_infrastructure_source
    ON context_infrastructure (source);


-- ============================================================================
-- Table: context_weather
-- ============================================================================
-- Weather observations from NOAA, ERA5, or other meteorological sources.
-- Uses POINT geometry for station locations or grid centroids.

CREATE TABLE IF NOT EXISTS context_weather (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source              VARCHAR(100) NOT NULL,
    source_id           VARCHAR(500) NOT NULL,
    geometry            GEOMETRY(POINT, 4326) NOT NULL,
    properties          JSONB DEFAULT '{}',
    observation_time    TIMESTAMPTZ NOT NULL,
    ingested_at         TIMESTAMPTZ DEFAULT now(),
    ingested_by_job_id  UUID REFERENCES jobs(job_id) ON DELETE SET NULL,

    CONSTRAINT uq_context_weather_source UNIQUE (source, source_id)
);

-- Spatial index for geospatial queries
CREATE INDEX IF NOT EXISTS idx_context_weather_geometry_gist
    ON context_weather USING GIST (geometry);

-- B-tree index for source filtering
CREATE INDEX IF NOT EXISTS idx_context_weather_source
    ON context_weather (source);

-- B-tree index for temporal queries
CREATE INDEX IF NOT EXISTS idx_context_weather_observation_time
    ON context_weather (observation_time);


-- ============================================================================
-- Table: job_context_usage (Junction)
-- ============================================================================
-- Links jobs to context rows they consumed. Tracks provenance and the
-- "lakehouse effect" -- whether context was freshly ingested or reused
-- from a previous job.

CREATE TABLE IF NOT EXISTS job_context_usage (
    job_id          UUID NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
    context_table   VARCHAR(50) NOT NULL
                    CHECK (context_table IN (
                        'datasets',
                        'context_buildings',
                        'context_infrastructure',
                        'context_weather'
                    )),
    context_id      UUID NOT NULL,
    usage_type      VARCHAR(20) NOT NULL
                    CHECK (usage_type IN ('ingested', 'reused')),
    linked_at       TIMESTAMPTZ DEFAULT now(),

    PRIMARY KEY (job_id, context_table, context_id)
);

-- B-tree index for reverse lookups ("which jobs used this context row?")
CREATE INDEX IF NOT EXISTS idx_job_context_usage_reverse
    ON job_context_usage (context_table, context_id);


-- Comments
COMMENT ON TABLE datasets IS
    'Satellite scenes discovered and used by the pipeline. '
    'Deduplicated by (source, source_id). Geometry is scene footprint as MULTIPOLYGON.';

COMMENT ON TABLE context_buildings IS
    'Building footprints from OSM, Overture, or other sources. '
    'Deduplicated by (source, source_id). Geometry is individual POLYGON.';

COMMENT ON TABLE context_infrastructure IS
    'Critical infrastructure facilities (hospitals, fire stations, etc.). '
    'Deduplicated by (source, source_id). Geometry is generic (point or polygon).';

COMMENT ON TABLE context_weather IS
    'Weather observations from NOAA, ERA5, or other sources. '
    'Deduplicated by (source, source_id). Geometry is POINT.';

COMMENT ON TABLE job_context_usage IS
    'Junction table linking jobs to context rows. Tracks whether each context '
    'item was freshly ingested or reused from a previous job.';
