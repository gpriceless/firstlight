# FirstLight Production Readiness - Implementation Specifications

**Created:** 2026-01-23
**Engineering Manager:** Ratchet
**Purpose:** Detailed specifications for coder agents

---

## Stream A: CLI-to-Core Wiring (Parallel)

These four epics can be implemented in parallel by separate coder agents.

---

### Epic 1.1: Wire CLI Analyze Command

**Agent:** coder-analyze
**Priority:** P0 (Critical)
**Estimated Effort:** 2-3 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/cli/commands/analyze.py`

#### Current State (Problem)

The analyze command falls back to a `MockAlgorithm` class (lines 394-408) when imports fail:

```python
# Return mock algorithm for demonstration
class MockAlgorithm:
    def __init__(self, params=None):
        self.params = params or {}

    def execute(self, data, **kwargs):
        return {"flood_extent": None, "confidence": None}

    def process_tile(self, tile_data, context=None):
        import numpy as np
        return (tile_data < -15.0).astype(np.uint8)

return MockAlgorithm(params)
```

Also, when no raster files found (lines 523-528):
```python
# No raster files - create mock output
mock_result = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8)
```

#### Core Modules to Wire

1. `core.analysis.library.registry.AlgorithmRegistry` - Algorithm discovery and instantiation
2. `core.analysis.library.baseline.flood.threshold_sar.ThresholdSARAlgorithm` - SAR flood detection
3. `core.analysis.library.baseline.flood.ndwi_optical.NDWIFloodAlgorithm` - Optical flood detection
4. `core.analysis.execution.runner.PipelineRunner` - Already partially wired (line 432)
5. `core.analysis.execution.tiled_runner.TiledRunner` - Already partially wired (line 433)

#### Implementation Tasks

1. **Remove MockAlgorithm fallback**
   - Replace lines 394-408 with proper error propagation
   - Raise `ImportError` or custom `AlgorithmNotFoundError` with descriptive message
   - Do NOT silently return mock data

2. **Wire algorithm registry lookup**
   - Import: `from core.analysis.library.registry import AlgorithmRegistry`
   - Get algorithm by name: `registry.get_algorithm(algorithm_name)`
   - Map event types to algorithms:
     - `flood` -> `threshold_sar` or `ndwi_optical`
     - `wildfire` -> `dNBR` or `thermal_anomaly`
     - `storm` -> `wind_damage`

3. **Remove random output fallback**
   - Replace lines 523-528 with proper error:
   ```python
   raise FileNotFoundError(f"No input rasters found in {input_path}")
   ```

4. **Add unit tests**
   - Create `/home/gprice/projects/firstlight/tests/cli/test_analyze_wiring.py`
   - Test algorithm registry lookup
   - Test error propagation for missing algorithms
   - Test error propagation for missing input files

#### Success Criteria

- [x] `flight analyze --algorithm sar_threshold` invokes `ThresholdSARAlgorithm.run()` *(Completed 2026-01-23)*
- [x] Output contains actual flood extent data (not random) *(Completed 2026-01-23)*
- [x] Missing algorithm raises descriptive `AlgorithmNotFoundError` *(Completed 2026-01-23)*
- [x] Missing input files raises descriptive `FileNotFoundError` *(Completed 2026-01-23)*
- [x] No `MockAlgorithm` class in codebase *(Completed 2026-01-23)*
- [x] No `np.random.randint` in analyze code path *(Completed 2026-01-23)*

---

### Epic 1.2: Wire CLI Ingest Command

**Agent:** coder-ingest
**Priority:** P0 (Critical)
**Estimated Effort:** 3-4 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/cli/commands/ingest.py`

#### Current State (Problem)

Lines 428-432 write a text placeholder instead of downloading real data:
```python
except ImportError:
    # Mock download for demonstration
    logger.warning("requests not available, simulating download")
    time.sleep(0.5)  # Simulate download time
    download_path.write_text(f"# Mock data for {item_id}\n")
```

#### Core Modules to Wire

1. `core.data.ingestion.streaming.StreamingIngester` - Real data download
2. `core.data.ingestion.formats.cog.COGConverter` - Already partially wired (line 491)
3. `core.data.ingestion.normalization.projection.RasterReprojector` - Already partially wired (line 485)
4. `core.data.ingestion.validation.image_validator.ImageValidator` - Validate downloaded images
5. `core.data.ingestion.validation.band_validator.BandValidator` - Validate band data

#### Implementation Tasks

1. **Replace placeholder download with StreamingIngester**
   - Import: `from core.data.ingestion.streaming import StreamingIngester`
   - Use `ingester.download_stac_item(url, output_path)` for real downloads
   - Handle network errors with proper retry logic

2. **Wire image validation**
   - Import: `from core.data.ingestion.validation.image_validator import ImageValidator`
   - Call `validator.validate(downloaded_file)` after download
   - Report validation failures clearly

3. **Wire normalization properly**
   - The normalization code at lines 483-497 already attempts to wire correctly
   - Ensure fallback doesn't silently skip normalization
   - Raise error if normalization required but modules unavailable

4. **Add integration tests**
   - Create `/home/gprice/projects/firstlight/tests/cli/test_ingest_wiring.py`
   - Test download with mock STAC responses
   - Test image validation integration
   - Test normalization pipeline

#### Success Criteria

- [x] `flight ingest` downloads real satellite data as GeoTIFF/COG (Completed 2026-01-23)
- [x] Downloaded files contain valid raster data (not text) (Completed 2026-01-23)
- [x] Image validation catches blank/corrupt images (Completed 2026-01-23)
- [x] Normalization applies CRS and resolution transformations (Completed 2026-01-23)
- [x] No text placeholder writes in codebase (Completed 2026-01-23)

---

### Epic 1.3: Wire CLI Validate Command

**Agent:** coder-validate
**Priority:** P0 (Critical)
**Estimated Effort:** 2-3 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/cli/commands/validate.py`

#### Current State (Problem)

Lines 235-275 return random scores instead of real quality checks:
```python
# Mock quality checks for demonstration
import random

for check_id in checks:
    check_info = QUALITY_CHECKS[check_id]

    # Generate mock results
    score = random.uniform(0.6, 1.0)
    passed = score >= 0.7
```

#### Core Modules to Wire

1. `core.quality.sanity.spatial.SpatialCoherenceCheck` - Spatial quality
2. `core.quality.sanity.values.ValueRangeCheck` - Value range validation
3. `core.quality.sanity.artifacts.ArtifactDetector` - Stripe/saturation detection
4. `core.quality.sanity.temporal.TemporalConsistencyCheck` - Temporal validation
5. `core.quality.validation.cross_sensor.CrossSensorValidator` - Cross-sensor checks

#### Implementation Tasks

1. **Remove random score generation**
   - Delete lines 235-275 mock implementation
   - Raise error if quality modules unavailable

2. **Wire spatial coherence check**
   - Import: `from core.quality.sanity.spatial import SpatialCoherenceCheck`
   - Call: `check.run(raster_data)` for real coherence score
   - Map result to check format

3. **Wire value range check**
   - Import: `from core.quality.sanity.values import ValueRangeCheck`
   - Call: `check.run(raster_data, expected_range)`
   - Report outliers and range violations

4. **Wire artifact detection**
   - Import: `from core.quality.sanity.artifacts import ArtifactDetector`
   - Call: `detector.detect(raster_data)`
   - Report stripe patterns and hot pixels

5. **Wire temporal consistency**
   - Import: `from core.quality.sanity.temporal import TemporalConsistencyCheck`
   - Call: `check.run(time_series_data)`
   - Report temporal anomalies

6. **Add tests**
   - Create `/home/gprice/projects/firstlight/tests/cli/test_validate_wiring.py`
   - Test each quality check independently
   - Test aggregate scoring

#### Success Criteria

- [x] `flight validate` produces real QC scores from raster data *(Completed 2026-01-23)*
- [x] No `random.uniform()` in validation code *(Completed 2026-01-23)*
- [x] Spatial coherence check returns real scores *(Completed 2026-01-23)*
- [x] Artifact detection identifies stripes and hot pixels *(Completed 2026-01-23)*
- [x] Temporal consistency check works with time series *(Completed 2026-01-23)*

---

### Epic 1.6: Wire CLI Discover Command

**Agent:** coder-discover
**Priority:** P0 (Critical)
**Estimated Effort:** 1-2 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/cli/commands/discover.py`

#### Current State (Problem)

Lines 333-342 fall back to mock results on any error:
```python
except ImportError as e:
    logger.debug(f"STAC client not available ({e}), using mock data")
    results = generate_mock_results(...)
except Exception as e:
    logger.warning(f"STAC discovery failed ({e}), falling back to mock data")
    results = generate_mock_results(...)
```

#### Core Modules to Wire

1. `core.data.discovery.stac_client.STACClient` - STAC catalog access
2. Already attempts to use STAC client, but falls back too eagerly

#### Implementation Tasks

1. **Remove mock fallback for STAC failures**
   - Only fall back on specific network errors (timeout, DNS failure)
   - For query errors, raise descriptive exception
   - For auth errors, prompt for credentials

2. **Improve error handling**
   - Distinguish between:
     - Network unreachable -> Retry with backoff
     - Invalid query parameters -> Raise with suggestion
     - No results found -> Return empty list (not mock)
     - Auth failure -> Raise with credential instructions

3. **Add retry logic**
   - Import: `from tenacity import retry, stop_after_attempt, wait_exponential`
   - Wrap STAC calls with retry decorator
   - Max 3 attempts with exponential backoff

4. **Add tests**
   - Create `/home/gprice/projects/firstlight/tests/cli/test_discover_wiring.py`
   - Test STAC success path
   - Test network error retry
   - Test query error propagation

#### Success Criteria

- [x] `flight discover` never falls back to mocks *(Completed 2026-01-23)*
- [x] Clear error messages on STAC failure *(Completed 2026-01-23)*
- [x] Retry logic for transient network errors *(Completed 2026-01-23)*
- [x] Empty results returned as empty list (not mock data) *(Completed 2026-01-23)*
- [x] `generate_mock_results()` function can be removed *(Completed 2026-01-23)*

---

## Stream B: API Production Hardening (Parallel)

These three epics can be implemented in parallel by separate coder agents.

---

### Epic 3.1: Database Persistence

**Agent:** coder-database
**Priority:** P1 (High)
**Estimated Effort:** 4-5 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/api/dependencies.py`
2. `/home/gprice/projects/firstlight/api/routes/events.py`
3. Create: `/home/gprice/projects/firstlight/api/database.py`
4. Create: `/home/gprice/projects/firstlight/api/models/database.py`

#### Current State (Problem)

Lines 67-70 in dependencies.py:
```python
async def execute(self, query: str, params: Optional[Dict] = None) -> Any:
    """Execute a database query."""
    # Placeholder for query execution
    raise NotImplementedError("Database layer not yet implemented")
```

Events stored in memory dict in `api/routes/events.py`.

#### Implementation Tasks

1. **Choose SQLite for simplicity** (can upgrade to PostgreSQL later)
   - Add `aiosqlite` to requirements
   - Create database file at `data/firstlight.db`

2. **Define database schema**
   - Create `/home/gprice/projects/firstlight/api/models/database.py`:
   ```python
   # Tables: events, executions, products
   CREATE_EVENTS_TABLE = """
   CREATE TABLE IF NOT EXISTS events (
       id TEXT PRIMARY KEY,
       status TEXT NOT NULL,
       priority TEXT NOT NULL,
       intent_json TEXT,
       spatial_json TEXT,
       temporal_json TEXT,
       created_at TIMESTAMP,
       updated_at TIMESTAMP
   )
   """
   ```

3. **Implement DatabaseSession.execute()**
   - Use aiosqlite for async operations
   - Implement connection pooling
   - Add transaction support

4. **Replace in-memory _events_store**
   - In `api/routes/events.py`, replace dict with DB calls
   - Add CRUD operations for events table

5. **Add migration support**
   - Create `/home/gprice/projects/firstlight/api/migrations/` directory
   - Add initial schema migration

6. **Add tests**
   - Create `/home/gprice/projects/firstlight/tests/api/test_database.py`
   - Test CRUD operations
   - Test connection pooling
   - Test data persistence across restarts

#### Success Criteria

- [x] Events persist across API restarts *(Completed 2026-01-23)*
- [x] Database operations are async *(Completed 2026-01-23)*
- [x] Connection pooling prevents exhaustion *(Completed 2026-01-23)*
- [x] Migration system in place *(Completed 2026-01-23)*
- [x] No `NotImplementedError` in database code *(Completed 2026-01-23)*

---

### Epic 3.2: JWT Authentication

**Agent:** coder-auth
**Priority:** P1 (High)
**Estimated Effort:** 3-4 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/api/dependencies.py`
2. `/home/gprice/projects/firstlight/api/auth.py`
3. Create: `/home/gprice/projects/firstlight/api/jwt_handler.py`

#### Current State (Problem)

Line 356 in dependencies.py:
```python
# TODO: Implement actual JWT validation
# For now, just return the token
return credentials.credentials
```

#### Implementation Tasks

1. **Add PyJWT dependency**
   - Add `PyJWT>=2.8.0` to requirements.txt
   - Add `python-jose[cryptography]` for RS256 support

2. **Create JWT handler**
   - Create `/home/gprice/projects/firstlight/api/jwt_handler.py`:
   ```python
   import jwt
   from datetime import datetime, timedelta

   class JWTHandler:
       def __init__(self, secret: str, algorithm: str = "HS256"):
           self.secret = secret
           self.algorithm = algorithm

       def decode(self, token: str) -> dict:
           return jwt.decode(token, self.secret, algorithms=[self.algorithm])

       def validate(self, token: str) -> bool:
           # Check signature, expiration, issuer
           ...
   ```

3. **Implement JWT validation in get_bearer_token()**
   - Decode token with signature verification
   - Check expiration (`exp` claim)
   - Check issuer (`iss` claim)
   - Return 401 for invalid tokens

4. **Add role-based access control**
   - Define roles: `admin`, `analyst`, `viewer`
   - Add `roles` claim to JWT
   - Create `require_role(role)` dependency

5. **Add refresh token support**
   - Create `/api/auth/refresh` endpoint
   - Issue new access token from valid refresh token

6. **Add environment configuration**
   - `JWT_SECRET` environment variable
   - `JWT_ALGORITHM` (default: HS256)
   - `JWT_EXPIRATION_HOURS` (default: 24)

7. **Add tests**
   - Create `/home/gprice/projects/firstlight/tests/api/test_jwt_auth.py`
   - Test valid token acceptance
   - Test expired token rejection
   - Test invalid signature rejection
   - Test role-based access

#### Success Criteria

- [x] Invalid JWT tokens are rejected with 401 *(Completed 2026-01-23)*
- [x] Expired tokens return 401 *(Completed 2026-01-23)*
- [x] Wrong signature returns 401 *(Completed 2026-01-23)*
- [ ] Role-based access enforced *(Not implemented - deferred)*
- [x] JWT secret from environment variable *(Completed 2026-01-23)*
- [x] No `TODO` comments for JWT validation *(Completed 2026-01-23)*

---

### Epic 3.3: Wire Intent Resolution API

**Agent:** coder-intent
**Priority:** P1 (High)
**Estimated Effort:** 2-3 hours

#### Files to Modify

1. `/home/gprice/projects/firstlight/api/routes/events.py`

#### Current State (Problem)

Lines 199-208 default to hardcoded intent:
```python
if not request.intent.event_class and request.intent.natural_language:
    # TODO: Call intent resolution service
    # For now, default to generic flood
    resolved_intent.event_class = "flood.riverine"
    resolved_intent.confidence = 0.75
```

#### Core Modules to Wire

1. `core.intent.resolver.IntentResolver` - Main resolver class
2. `core.intent.classifier.EventClassifier` - NLP classification
3. `core.intent.registry.EventClassRegistry` - Valid event class lookup

#### Implementation Tasks

1. **Import intent resolver**
   - Add: `from core.intent.resolver import IntentResolver`
   - Create resolver instance at module level or as dependency

2. **Wire natural language parsing**
   - Replace TODO with:
   ```python
   from core.intent.resolver import IntentResolver

   resolver = IntentResolver()
   resolution = resolver.resolve(
       natural_language=request.intent.natural_language,
       parameters=request.intent.parameters
   )
   resolved_intent.event_class = resolution.resolved_class
   resolved_intent.confidence = resolution.confidence
   ```

3. **Add intent validation**
   - Validate resolved class against taxonomy
   - Reject unknown event classes with 400 error
   - Return alternatives if confidence low

4. **Add error handling**
   - Handle NLP failures gracefully
   - Return reasonable default with low confidence
   - Log resolution failures for analysis

5. **Add tests**
   - Create `/home/gprice/projects/firstlight/tests/api/test_intent_resolution.py`
   - Test "flooding after hurricane" -> flood.coastal
   - Test "forest fire in California" -> wildfire.forest
   - Test invalid intents rejection

#### Success Criteria

- [x] Natural language intent is resolved correctly *(Completed 2026-01-23)*
- [x] "flooding after hurricane" -> flood.coastal *(Completed 2026-01-23)*
- [x] "forest fire" -> wildfire.forest *(Completed 2026-01-23)*
- [x] Invalid intents rejected with 400 *(Completed 2026-01-23)*
- [x] No `TODO: Call intent resolution` in code *(Completed 2026-01-23)*

---

## Dependencies and Sequencing

### Independent Work (Can Run in Parallel)

| Stream | Epics | Agents |
|--------|-------|--------|
| A (CLI) | 1.1, 1.2, 1.3, 1.6 | 4 parallel agents |
| B (API) | 3.1, 3.2, 3.3 | 3 parallel agents |

**Total: 7 parallel work streams**

### Sequential Work (Blocked)

| Epic | Blocked By | Reason |
|------|------------|--------|
| 1.4 (Export) | 1.1 | Needs real analysis output to export |
| 1.5 (Run) | 1.1, 1.3, 1.4 | Orchestrates all sub-commands |
| 2.1 (Agent) | Phase 1 | Follows CLI wiring patterns |
| 4.x (Testing) | 1-3 | Needs implementations to test |

---

## Quality Gates

Before merging any epic:

1. **Build passes** - No compilation or type errors
2. **Tests pass** - All existing 518+ tests still pass
3. **New tests added** - Epic-specific tests included
4. **No mock/stub in production path** - Real implementations only
5. **Documentation updated** - Docstrings and comments current

---

## Communication Protocol

Each coder agent should:

1. Post to Matrix #product when starting epic
2. Post progress updates every 15-30 minutes
3. Post immediately if blocked
4. Post completion with summary

Example:
```
[EPIC 1.1] Starting CLI Analyze wiring.
• Removing MockAlgorithm class
• Wiring to AlgorithmRegistry
• ETA: 2 hours
```

---

*Implementation specs created by Ratchet (Engineering Manager)*
