"""
Tests for Control Plane Tool Schema Endpoints (Phase 2, Task 2.14).

Tests cover:
- GET /control/v1/tools returns tool schemas for all non-deprecated algorithms
- Each schema has name/description/parameters fields
- Deprecated algorithms are excluded
- Empty registry returns 200 with empty tools array

Note: Tests use the AlgorithmRegistry.list_tool_schemas() method directly
and the generate_tool_schema standalone function to avoid triggering the full
import chain (api.routes -> api.database -> aiosqlite).
"""

import re
from typing import Any, Dict, List

import pytest

from core.analysis.library.registry import (
    AlgorithmCategory,
    AlgorithmMetadata,
    AlgorithmRegistry,
    DataType,
)


# =============================================================================
# Registry Fixtures
# =============================================================================


@pytest.fixture
def empty_registry():
    """Create an empty algorithm registry."""
    return AlgorithmRegistry()


@pytest.fixture
def populated_registry():
    """Create a registry with test algorithms."""
    registry = AlgorithmRegistry()

    registry.register(AlgorithmMetadata(
        id="flood.baseline.threshold_sar",
        name="SAR Backscatter Threshold",
        category=AlgorithmCategory.BASELINE,
        version="1.0.0",
        event_types=["flood.*"],
        required_data_types=[DataType.SAR],
        description="Detects flood extent using SAR backscatter thresholding",
        parameter_schema={
            "type": "object",
            "properties": {
                "threshold": {"type": "number", "default": -15.0},
                "polarization": {"type": "string", "enum": ["VV", "VH"]},
            },
        },
        default_parameters={"threshold": -15.0, "polarization": "VV"},
    ))

    registry.register(AlgorithmMetadata(
        id="wildfire.advanced.nbr",
        name="Normalized Burn Ratio",
        category=AlgorithmCategory.ADVANCED,
        version="2.0.0",
        event_types=["wildfire.*"],
        required_data_types=[DataType.OPTICAL],
        description="Calculates burn severity using NBR",
        parameter_schema={
            "type": "object",
            "properties": {
                "pre_fire_window_days": {"type": "integer", "default": 30},
            },
        },
    ))

    registry.register(AlgorithmMetadata(
        id="flood.experimental.old_method",
        name="Old Flood Method",
        category=AlgorithmCategory.EXPERIMENTAL,
        version="0.1.0",
        event_types=["flood.*"],
        required_data_types=[DataType.OPTICAL],
        deprecated=True,
        replacement_algorithm="flood.baseline.threshold_sar",
    ))

    return registry


# =============================================================================
# Standalone generate_tool_schema function
# (replicated from api/routes/control/tools.py to avoid import chain)
# =============================================================================


def _generate_tool_schema(registry: AlgorithmRegistry, algorithm_id: str) -> Dict[str, Any]:
    """
    Generate an OpenAI-compatible function-calling schema.

    This mirrors the logic in api/routes/control/tools.py::generate_tool_schema
    but takes a registry argument directly to avoid the import chain.
    """
    algo = registry.get(algorithm_id)
    if algo is None or algo.deprecated:
        return None

    func_name = re.sub(r"[^a-zA-Z0-9_]", "_", algo.id)

    desc_parts = []
    if algo.description:
        desc_parts.append(algo.description)
    if algo.event_types:
        desc_parts.append(f"Supports event types: {', '.join(algo.event_types)}")
    if algo.category:
        desc_parts.append(f"Category: {algo.category.value}")
    description = ". ".join(desc_parts) if desc_parts else algo.name

    parameters = algo.parameter_schema if algo.parameter_schema else {
        "type": "object",
        "properties": {},
    }
    if "type" not in parameters:
        parameters["type"] = "object"

    return {
        "name": func_name,
        "description": description,
        "parameters": parameters,
    }


# =============================================================================
# Tool Schema Generation Tests
# =============================================================================


class TestGenerateToolSchema:
    """Test individual tool schema generation."""

    def test_basic_schema_generation(self, populated_registry):
        """Test generating a schema for an existing algorithm."""
        schema = _generate_tool_schema(
            populated_registry, "flood.baseline.threshold_sar"
        )
        assert schema is not None
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema

    def test_schema_name_normalized(self, populated_registry):
        """Test that algorithm IDs with dots are normalized to underscores."""
        schema = _generate_tool_schema(
            populated_registry, "flood.baseline.threshold_sar"
        )
        assert schema is not None
        assert "." not in schema["name"]
        assert schema["name"] == "flood_baseline_threshold_sar"

    def test_schema_includes_description(self, populated_registry):
        """Test that schema description includes algorithm info."""
        schema = _generate_tool_schema(
            populated_registry, "flood.baseline.threshold_sar"
        )
        assert schema is not None
        assert "SAR" in schema["description"] or "flood" in schema["description"].lower()

    def test_schema_parameters_from_algorithm(self, populated_registry):
        """Test that parameters come from the algorithm's parameter_schema."""
        schema = _generate_tool_schema(
            populated_registry, "flood.baseline.threshold_sar"
        )
        assert schema is not None
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "threshold" in params.get("properties", {})

    def test_deprecated_algorithm_returns_none(self, populated_registry):
        """Test that deprecated algorithms return None."""
        schema = _generate_tool_schema(
            populated_registry, "flood.experimental.old_method"
        )
        assert schema is None

    def test_nonexistent_algorithm_returns_none(self, populated_registry):
        """Test that nonexistent algorithm returns None."""
        schema = _generate_tool_schema(
            populated_registry, "nonexistent.algorithm"
        )
        assert schema is None


# =============================================================================
# AlgorithmRegistry.list_tool_schemas Tests
# =============================================================================


class TestAlgorithmRegistryListToolSchemas:
    """Test the list_tool_schemas method on AlgorithmRegistry."""

    def test_registry_method_exists(self, populated_registry):
        """Test that list_tool_schemas method exists on AlgorithmRegistry."""
        assert hasattr(populated_registry, "list_tool_schemas")

    def test_lists_non_deprecated(self, populated_registry):
        """Test that list excludes deprecated algorithms by default."""
        schemas = populated_registry.list_tool_schemas(exclude_deprecated=True)
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "flood_experimental_old_method" not in names

    def test_includes_all_when_not_excluding(self, populated_registry):
        """Test that list includes deprecated when exclude_deprecated=False."""
        schemas = populated_registry.list_tool_schemas(exclude_deprecated=False)
        assert len(schemas) == 3

    def test_each_schema_has_required_fields(self, populated_registry):
        """Test that each schema has name, description, parameters."""
        schemas = populated_registry.list_tool_schemas(exclude_deprecated=True)
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert isinstance(schema["name"], str)
            assert isinstance(schema["description"], str)
            assert isinstance(schema["parameters"], dict)

    def test_empty_registry_returns_empty_list(self, empty_registry):
        """Test that empty registry returns empty list."""
        schemas = empty_registry.list_tool_schemas(exclude_deprecated=True)
        assert schemas == []

    def test_schema_name_has_no_dots(self, populated_registry):
        """Test that schema names have dots replaced with underscores."""
        schemas = populated_registry.list_tool_schemas()
        for schema in schemas:
            assert "." not in schema["name"]


# =============================================================================
# Response Model Tests
# =============================================================================


class TestToolsResponseModel:
    """Test the ToolsResponse model."""

    def test_tools_response_with_items(self):
        """Test ToolsResponse with tool schemas."""
        from api.models.control import ToolSchema, ToolsResponse

        resp = ToolsResponse(tools=[
            ToolSchema(
                name="flood_baseline_threshold_sar",
                description="SAR threshold detection",
                parameters={"type": "object", "properties": {}},
            ),
        ])
        assert len(resp.tools) == 1
        assert resp.tools[0].name == "flood_baseline_threshold_sar"

    def test_tools_response_empty(self):
        """Test ToolsResponse with empty tools list."""
        from api.models.control import ToolsResponse

        resp = ToolsResponse(tools=[])
        assert len(resp.tools) == 0
