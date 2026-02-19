-- FirstLight Database Initialization
-- Ensures PostGIS extension is available for geospatial operations.

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
