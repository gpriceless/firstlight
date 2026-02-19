"""
Control Plane Tool Schema Endpoints.

Provides endpoints for discovering available tool schemas
(OpenAI-compatible function-calling format) for LLM agents.

Task 2.9: GET /control/v1/tools + generate_tool_schema + list_tool_schemas
"""

import logging
import re
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Request

from api.models.control import ToolSchema, ToolsResponse
from api.routes.control import get_current_customer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["LLM Control - Tools"])


# =============================================================================
# Tool Schema Generation (Task 2.9)
# =============================================================================


def generate_tool_schema(algorithm_id: str) -> Optional[Dict[str, Any]]:
    """
    Generate an OpenAI-compatible function-calling schema from an algorithm.

    Reads the algorithm's AlgorithmMetadata fields and returns a schema
    with name, description, and parameters (JSON Schema).

    Args:
        algorithm_id: The algorithm identifier.

    Returns:
        An OpenAI-compatible tool schema dict, or None if the algorithm
        is not found or is deprecated.
    """
    try:
        from core.analysis.library.registry import get_global_registry

        registry = get_global_registry()
        algo = registry.get(algorithm_id)

        if algo is None:
            return None

        if algo.deprecated:
            return None

        # Normalize algorithm ID to a valid function name
        # Replace dots and other special chars with underscores
        func_name = re.sub(r"[^a-zA-Z0-9_]", "_", algo.id)

        # Build description
        description_parts = []
        if algo.description:
            description_parts.append(algo.description)
        if algo.event_types:
            description_parts.append(
                f"Supports event types: {', '.join(algo.event_types)}"
            )
        if algo.category:
            description_parts.append(f"Category: {algo.category.value}")
        description = ". ".join(description_parts) if description_parts else algo.name

        # Build parameters from parameter_schema
        parameters = algo.parameter_schema if algo.parameter_schema else {
            "type": "object",
            "properties": {},
        }

        # Ensure it's a valid JSON Schema
        if "type" not in parameters:
            parameters["type"] = "object"

        return {
            "name": func_name,
            "description": description,
            "parameters": parameters,
        }

    except ImportError:
        logger.warning("Algorithm registry not available for tool schema generation")
        return None


def list_tool_schemas(exclude_deprecated: bool = True) -> List[Dict[str, Any]]:
    """
    Convenience method to list all available tool schemas.

    Args:
        exclude_deprecated: If True, exclude deprecated algorithms.

    Returns:
        List of OpenAI-compatible tool schema dicts.
    """
    try:
        from core.analysis.library.registry import get_global_registry

        registry = get_global_registry()
        schemas = []

        for algo in registry.list_all():
            if exclude_deprecated and algo.deprecated:
                continue

            schema = generate_tool_schema(algo.id)
            if schema is not None:
                schemas.append(schema)

        return schemas

    except ImportError:
        logger.warning("Algorithm registry not available for tool schema listing")
        return []


# =============================================================================
# Endpoint
# =============================================================================


@router.get(
    "",
    response_model=ToolsResponse,
    summary="List available tool schemas",
    description=(
        "Returns all available tool schemas for the authenticated tenant "
        "in OpenAI-compatible function-calling format."
    ),
)
async def get_tools(
    request: Request,
    customer_id: Annotated[str, Depends(get_current_customer)],
) -> ToolsResponse:
    """
    Get all available tool schemas.

    Returns tool schemas derived from the algorithm registry,
    excluding deprecated algorithms.
    """
    schemas = list_tool_schemas(exclude_deprecated=True)

    tools = [
        ToolSchema(
            name=s["name"],
            description=s["description"],
            parameters=s["parameters"],
        )
        for s in schemas
    ]

    return ToolsResponse(tools=tools)
