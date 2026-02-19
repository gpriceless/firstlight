"""
Control Plane Job Endpoints.

Provides CRUD operations and state transitions for jobs
accessible to LLM agents via the Control API.

Tasks: 2.2 (list), 2.3 (create), 2.4 (get), 2.5 (transition),
       2.6 (patch params), 2.7 (reasoning)
"""

import json
import logging
import math
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Path, Query, Request, status

from api.models.control import (
    CreateJobRequest,
    JobDetailLinks,
    JobDetailResponse,
    JobResponse,
    JobSummary,
    PaginatedJobsResponse,
    ReasoningRequest,
    ReasoningResponse,
    TransitionRequest,
    TransitionResponse,
)
from api.models.errors import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from api.routes.control import get_current_customer
from agents.orchestrator.backends.base import StateConflictError
from agents.orchestrator.state_model import (
    is_valid_phase_status,
    is_terminal_status,
    validate_transition,
    JobPhase,
    JobStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["LLM Control - Jobs"])


# =============================================================================
# Backend Helpers
# =============================================================================


def _get_backend():
    """
    Get the active StateBackend instance.

    In a full deployment this would come from app state or a dependency.
    For now, we use the PostGIS backend factory from config.
    """
    from agents.orchestrator.backends.postgis_backend import PostGISStateBackend
    from api.config import get_settings

    settings = get_settings()
    db = settings.database
    backend = PostGISStateBackend(
        host=db.host,
        port=db.port,
        database=db.name,
        user=db.user,
        password=db.password,
    )
    return backend


async def _get_connected_backend():
    """Get a connected backend instance."""
    backend = _get_backend()
    await backend.connect()
    return backend


# =============================================================================
# Task 2.2: GET /control/v1/jobs — List jobs for authenticated tenant
# =============================================================================


@router.get(
    "",
    response_model=PaginatedJobsResponse,
    summary="List jobs for the authenticated tenant",
    description=(
        "List jobs with optional filtering by phase, status, event_type, "
        "bbox, and pagination support."
    ),
)
async def list_jobs(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
    phase: Annotated[
        Optional[str],
        Query(description="Filter by job phase (e.g., QUEUED, ANALYZING)"),
    ] = None,
    status_filter: Annotated[
        Optional[str],
        Query(alias="status", description="Filter by job status"),
    ] = None,
    event_type: Annotated[
        Optional[str],
        Query(description="Filter by event type"),
    ] = None,
    bbox: Annotated[
        Optional[str],
        Query(
            description=(
                "Bounding box filter as west,south,east,north in WGS84 decimal degrees "
                "(e.g., bbox=-122.5,37.5,-121.5,38.5)"
            ),
        ),
    ] = None,
    page: Annotated[
        int,
        Query(ge=1, description="Page number (1-based)"),
    ] = 1,
    page_size: Annotated[
        int,
        Query(ge=1, le=100, description="Items per page (max 100)"),
    ] = 20,
) -> PaginatedJobsResponse:
    """List jobs for the authenticated tenant with filtering and pagination."""
    # Parse and validate bbox
    bbox_values = None
    if bbox is not None:
        try:
            parts = [float(x.strip()) for x in bbox.split(",")]
        except (ValueError, AttributeError):
            raise ValidationError(
                message="bbox must be four comma-separated numbers: west,south,east,north"
            )
        if len(parts) != 4:
            raise ValidationError(
                message="bbox must be four comma-separated numbers: west,south,east,north"
            )
        west, south, east, north = parts
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValidationError(
                message="bbox longitude values must be between -180 and 180"
            )
        if not (-90 <= south <= 90 and -90 <= north <= 90):
            raise ValidationError(
                message="bbox latitude values must be between -90 and 90"
            )
        if south > north:
            raise ValidationError(
                message="bbox south must be less than or equal to north"
            )
        bbox_values = (west, south, east, north)

    backend = await _get_connected_backend()
    try:
        offset = (page - 1) * page_size

        # Get all matching jobs for total count
        all_jobs = await backend.list_jobs(
            customer_id=customer_id,
            phase=phase,
            status=status_filter,
            event_type=event_type,
            limit=10000,
            offset=0,
        )

        # Apply bbox filter at the application level
        if bbox_values is not None:
            west, south, east, north = bbox_values
            all_jobs = [
                job for job in all_jobs
                if _job_intersects_bbox(job.aoi, west, south, east, north)
            ]

        total = len(all_jobs)

        # Apply pagination
        page_jobs = all_jobs[offset:offset + page_size]

        items = [
            JobSummary(
                job_id=job.job_id,
                phase=job.phase,
                status=job.status,
                event_type=job.event_type,
                aoi_area_km2=_compute_aoi_area_km2(job.aoi),
                created_at=job.created_at,
                updated_at=job.updated_at,
            )
            for job in page_jobs
        ]

        return PaginatedJobsResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )
    finally:
        await backend.close()


# =============================================================================
# Task 2.3: POST /control/v1/jobs — Create a new job
# =============================================================================


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new job",
    description="Create a new job with event type, AOI geometry, and parameters.",
)
async def create_job(
    request: Request,
    body: CreateJobRequest,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> JobResponse:
    """
    Create a new job.

    The AOI geometry (any GeoJSON type) is promoted to MultiPolygon via ST_Multi.
    Validates GeoJSON coordinate bounds and vertex count at the API boundary.
    """
    backend = await _get_connected_backend()
    try:
        initial_phase = JobPhase.QUEUED.value
        initial_status = JobStatus.PENDING.value

        job_state = await backend.create_job(
            customer_id=customer_id,
            event_type=body.event_type,
            aoi_geojson=body.aoi,
            phase=initial_phase,
            status=initial_status,
            parameters=body.parameters,
        )

        # Record job.created event
        await backend.record_event(
            job_id=job_state.job_id,
            customer_id=customer_id,
            event_type="job.created",
            phase=initial_phase,
            status=initial_status,
            actor=customer_id,
            reasoning=body.reasoning,
            payload={
                "event_type": body.event_type,
                "parameters": body.parameters,
            },
        )

        return JobResponse(
            job_id=job_state.job_id,
            phase=job_state.phase,
            status=job_state.status,
            event_type=job_state.event_type,
            created_at=job_state.created_at,
        )
    finally:
        await backend.close()


# =============================================================================
# Task 2.4: GET /control/v1/jobs/{job_id} — Full job detail
# =============================================================================


@router.get(
    "/{job_id}",
    response_model=JobDetailResponse,
    summary="Get job details",
    description="Get full job detail including phase, status, AOI, parameters, and links.",
    responses={404: {"description": "Job not found or access denied"}},
)
async def get_job(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> JobDetailResponse:
    """
    Get full job detail.

    Returns 404 if the job does not exist or if the customer_id does not match
    (to avoid leaking existence of other tenants' jobs).
    """
    backend = await _get_connected_backend()
    try:
        job = await backend.get_state(job_id)

        if job is None or job.customer_id != customer_id:
            raise NotFoundError(message=f"Job '{job_id}' not found")

        return JobDetailResponse(
            job_id=job.job_id,
            phase=job.phase,
            status=job.status,
            event_type=job.event_type,
            aoi=job.aoi,
            aoi_area_km2=_compute_aoi_area_km2(job.aoi),
            parameters=job.parameters,
            orchestrator_id=job.orchestrator_id,
            created_at=job.created_at,
            updated_at=job.updated_at,
            links=JobDetailLinks(
                self=f"/control/v1/jobs/{job.job_id}",
                events=f"/control/v1/jobs/{job.job_id}/events",
                checkpoints=f"/control/v1/jobs/{job.job_id}/checkpoints",
            ),
        )
    finally:
        await backend.close()


# =============================================================================
# Task 2.5: POST /control/v1/jobs/{job_id}/transition — State transition
# =============================================================================


@router.post(
    "/{job_id}/transition",
    response_model=TransitionResponse,
    summary="Transition job state",
    description=(
        "Atomically transition a job from one (phase, status) to another. "
        "The expected_phase and expected_status are the TOCTOU guard."
    ),
    responses={
        409: {"description": "State conflict -- current state does not match expected"},
        422: {"description": "Invalid target (phase, status) pair"},
    },
)
async def transition_job(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    body: TransitionRequest,
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> TransitionResponse:
    """
    Transition a job from one state to another with TOCTOU protection.

    Requires state:write permission.
    """
    # Validate target (phase, status) pair
    if not is_valid_phase_status(body.target_phase, body.target_status):
        raise ValidationError(
            message=(
                f"Invalid target (phase, status) pair: "
                f"({body.target_phase}, {body.target_status})"
            )
        )

    # Validate the transition itself
    if not validate_transition(
        body.expected_phase, body.expected_status,
        body.target_phase, body.target_status,
    ):
        raise ValidationError(
            message=(
                f"Invalid transition from ({body.expected_phase}, {body.expected_status}) "
                f"to ({body.target_phase}, {body.target_status})"
            )
        )

    backend = await _get_connected_backend()
    try:
        # Verify tenant access first
        job = await backend.get_state(job_id)
        if job is None or job.customer_id != customer_id:
            raise NotFoundError(message=f"Job '{job_id}' not found")

        try:
            new_state = await backend.transition(
                job_id=job_id,
                expected_phase=body.expected_phase,
                expected_status=body.expected_status,
                new_phase=body.target_phase,
                new_status=body.target_status,
                reason=body.reason,
                actor=customer_id,
            )
        except StateConflictError as e:
            raise ConflictError(
                message=(
                    f"State conflict: expected ({e.expected_phase}, {e.expected_status}), "
                    f"but current state is ({e.actual_phase}, {e.actual_status})"
                )
            )

        return TransitionResponse(
            job_id=new_state.job_id,
            phase=new_state.phase,
            status=new_state.status,
            updated_at=new_state.updated_at,
        )
    finally:
        await backend.close()


# =============================================================================
# Task 2.6: PATCH /control/v1/jobs/{job_id}/parameters — Merge-patch params
# =============================================================================


@router.patch(
    "/{job_id}/parameters",
    summary="Patch job parameters",
    description=(
        "Apply a JSON merge-patch to the job's parameters. "
        "Validates keys against the algorithm's parameter schema. "
        "Rejects if job is in a terminal state."
    ),
    responses={
        409: {"description": "Job is in a terminal state"},
    },
)
async def patch_parameters(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> Dict[str, Any]:
    """
    Apply a JSON merge-patch to the job's parameters.

    Requires state:write permission.
    """
    # Parse body manually for merge-patch
    body_bytes = await request.body()
    try:
        patch = json.loads(body_bytes)
    except (json.JSONDecodeError, ValueError):
        raise ValidationError(message="Request body must be valid JSON")

    if not isinstance(patch, dict):
        raise ValidationError(message="Request body must be a JSON object")

    backend = await _get_connected_backend()
    try:
        job = await backend.get_state(job_id)
        if job is None or job.customer_id != customer_id:
            raise NotFoundError(message=f"Job '{job_id}' not found")

        # Reject if in terminal state
        if is_terminal_status(job.status) or job.phase == JobPhase.COMPLETE.value:
            raise ConflictError(
                message=(
                    f"Cannot update parameters: job is in terminal state "
                    f"({job.phase}, {job.status})"
                )
            )

        # Validate against algorithm parameter schema if available
        _validate_against_algorithm_schema(job.event_type, patch)

        # Apply merge-patch
        merged = dict(job.parameters)
        for key, value in patch.items():
            if value is None:
                merged.pop(key, None)  # JSON merge-patch: null removes key
            else:
                merged[key] = value

        # Update parameters in backend
        await backend.set_state(
            job_id=job_id,
            phase=job.phase,
            status=job.status,
            parameters=merged,
        )

        # Record event
        await backend.record_event(
            job_id=job_id,
            customer_id=customer_id,
            event_type="job.parameters_updated",
            phase=job.phase,
            status=job.status,
            actor=customer_id,
            payload={"patch": patch, "resulting_parameters": merged},
        )

        return merged
    finally:
        await backend.close()


# =============================================================================
# Task 2.7: POST /control/v1/jobs/{job_id}/reasoning — Append reasoning
# =============================================================================


@router.post(
    "/{job_id}/reasoning",
    response_model=ReasoningResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Append a reasoning entry",
    description=(
        "Append a reasoning entry to the job's event log. "
        "Used by LLM agents to record their decision-making process."
    ),
)
async def append_reasoning(
    job_id: Annotated[str, Path(description="Job identifier (UUID)")],
    body: ReasoningRequest,
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> ReasoningResponse:
    """
    Append a reasoning entry to job_events.

    The reasoning text is limited to 64KB and null bytes are rejected.
    """
    backend = await _get_connected_backend()
    try:
        job = await backend.get_state(job_id)
        if job is None or job.customer_id != customer_id:
            raise NotFoundError(message=f"Job '{job_id}' not found")

        event_seq = await backend.record_event(
            job_id=job_id,
            customer_id=customer_id,
            event_type="job.reasoning",
            phase=job.phase,
            status=job.status,
            actor=customer_id,
            reasoning=body.reasoning,
            payload={
                "confidence": body.confidence,
                **(body.payload or {}),
            },
        )

        return ReasoningResponse(event_seq=event_seq)
    finally:
        await backend.close()


# =============================================================================
# Utility Functions
# =============================================================================


def _extract_bbox_from_geojson(geojson: Dict[str, Any]) -> Optional[tuple]:
    """Extract bounding box (west, south, east, north) from GeoJSON geometry."""
    try:
        coords = geojson.get("coordinates", [])
        geom_type = geojson.get("type", "")

        all_lons: List[float] = []
        all_lats: List[float] = []

        if geom_type == "Point":
            all_lons.append(coords[0])
            all_lats.append(coords[1])
        elif geom_type in ("MultiPoint", "LineString"):
            for pt in coords:
                all_lons.append(pt[0])
                all_lats.append(pt[1])
        elif geom_type in ("MultiLineString", "Polygon"):
            for ring in coords:
                for pt in ring:
                    all_lons.append(pt[0])
                    all_lats.append(pt[1])
        elif geom_type == "MultiPolygon":
            for polygon in coords:
                for ring in polygon:
                    for pt in ring:
                        all_lons.append(pt[0])
                        all_lats.append(pt[1])

        if all_lons and all_lats:
            return (min(all_lons), min(all_lats), max(all_lons), max(all_lats))
    except (IndexError, TypeError, KeyError):
        pass
    return None


def _job_intersects_bbox(
    aoi: Optional[Dict[str, Any]],
    west: float,
    south: float,
    east: float,
    north: float,
) -> bool:
    """Check if a job's AOI intersects with the given bbox."""
    if aoi is None:
        return True  # Include jobs without AOI
    job_bbox = _extract_bbox_from_geojson(aoi)
    if job_bbox is None:
        return True  # Can't determine — include
    jw, js, je, jn = job_bbox
    # Check for non-intersection and negate
    return not (je < west or jw > east or jn < south or js > north)


def _compute_aoi_area_km2(aoi: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Compute approximate area in km2 from GeoJSON.

    Uses a simple spherical approximation.
    """
    if aoi is None:
        return None

    bbox = _extract_bbox_from_geojson(aoi)
    if bbox is None:
        return None

    west, south, east, north = bbox
    R = 6371.0  # Earth radius in km

    lat1 = math.radians(south)
    lat2 = math.radians(north)
    lon1 = math.radians(west)
    lon2 = math.radians(east)

    area = R * R * abs(math.sin(lat1) - math.sin(lat2)) * abs(lon1 - lon2)
    return round(area, 4)


def _validate_against_algorithm_schema(
    event_type: str, patch: Dict[str, Any]
) -> None:
    """
    Validate parameter patch against the algorithm's declared parameter_schema.

    If the algorithm registry is not available or no schema is defined,
    the validation is skipped (permissive mode).
    """
    try:
        from core.analysis.library.registry import get_global_registry

        registry = get_global_registry()
        algorithms = registry.search_by_event_type(event_type)

        if not algorithms:
            return  # No algorithm found — skip validation

        # Use the first matching algorithm's schema
        algo = algorithms[0]
        if not algo.parameter_schema:
            return  # No schema defined — skip validation

        # Check if submitted keys are in the schema
        schema_properties = algo.parameter_schema.get("properties", {})
        if schema_properties:
            unknown_keys = set(patch.keys()) - set(schema_properties.keys())
            if unknown_keys:
                raise ValidationError(
                    message=(
                        f"Unknown parameter keys for algorithm '{algo.id}': "
                        f"{', '.join(sorted(unknown_keys))}"
                    )
                )

        # Full JSON Schema validation if jsonschema is available
        try:
            import jsonschema
            jsonschema.validate(instance=patch, schema=algo.parameter_schema)
        except ImportError:
            pass  # jsonschema not installed — skip full validation
        except jsonschema.ValidationError as e:
            raise ValidationError(
                message=f"Parameter validation failed: {e.message}"
            )

    except ImportError:
        pass  # Registry not available — skip validation
