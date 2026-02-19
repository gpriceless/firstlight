# API Surface Map and Auth Boundary

This document defines the authentication boundary for all API surfaces in FirstLight.

## Auth Boundary Summary

| Path Prefix | Auth Required | Permission | Notes |
|-------------|--------------|------------|-------|
| `GET /health`, `/ready` | No | -- | Health probes |
| `GET /api/v1/health/*` | No | -- | Health probes |
| `GET /oapi/` | No | -- | OGC landing page |
| `GET /oapi/conformance` | No | -- | OGC conformance |
| `GET /oapi/processes` | No | -- | OGC process listing |
| `GET /oapi/processes/{id}` | No | -- | OGC process description |
| `POST /oapi/processes/{id}/execution` | Yes | API key | OGC job execution |
| `GET /oapi/jobs/*` | Yes | API key | OGC job status/results |
| `GET /api/v1/*` | Yes | API key | Core API |
| `GET /control/v1/*` | Yes | API key + tenant | LLM Control Plane |
| `POST /control/v1/*` | Yes | API key + tenant | LLM Control Plane |
| `PATCH /control/v1/*` | Yes | API key + tenant | LLM Control Plane |
| `GET /internal/v1/*` | Yes | API key + `internal:read` | Partner API |
| `POST /internal/v1/*` | Yes | API key + `internal:read` | Partner API |
| `DELETE /internal/v1/*` | Yes | API key + `internal:read` | Partner API |
| `GET /stac/*` | Configurable | -- | STAC catalog (read) |

## OGC Auth Boundary

OGC API Processes follows a split auth model:

- **Discovery paths are public** (no authentication required):
  - `GET /oapi/` -- Landing page
  - `GET /oapi/conformance` -- Conformance classes
  - `GET /oapi/processes` -- Process listing
  - `GET /oapi/processes/{id}` -- Process description (including vendor extensions)

- **Execution paths require authentication**:
  - `POST /oapi/processes/{id}/execution` -- Submit a job for execution
  - `GET /oapi/jobs/{jobId}` -- Job status
  - `GET /oapi/jobs/{jobId}/results` -- Job results

This split allows public discoverability of available algorithms while
protecting compute resources and data behind authentication.

pygeoapi's internal auth is disabled. All authentication is handled by
FirstLight's TenantMiddleware, which intercepts requests before they
reach pygeoapi.

## Implementation

The auth boundary is implemented via:

1. **`TenantMiddleware`** (in `api/middleware.py`):
   - `DEFAULT_AUTH_EXEMPT_PATHS` lists exact paths that skip auth
   - `DEFAULT_AUTH_EXEMPT_PATTERNS` lists regex patterns for parameterized exempt paths
   - All other paths require a valid API key with `customer_id`

2. **Route-level permissions** (in individual route files):
   - `state:read`, `state:write`, `escalation:manage` for control plane
   - `internal:read` for partner API
   - Standard API key validation for core API

## STAC Auth Boundary

STAC catalog endpoints at `/stac/*` are mounted separately. Read access
(browsing collections, items) is configurable. Write access (if exposed)
requires authentication.
