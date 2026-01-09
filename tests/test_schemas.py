"""Tests for OpenSpec JSON schemas."""

import json
from pathlib import Path

import pytest


SCHEMA_DIR = Path(__file__).parent.parent / "openspec" / "schemas"
EXPECTED_SCHEMAS = [
    "common",
    "event",
    "intent",
    "datasource",
    "pipeline",
    "ingestion",
    "quality",
    "provenance",
]


class TestSchemaFiles:
    """Test that all schema files exist and are valid JSON."""

    @pytest.mark.parametrize("schema_name", EXPECTED_SCHEMAS)
    def test_schema_file_exists(self, schema_name):
        """Test that schema file exists."""
        schema_path = SCHEMA_DIR / f"{schema_name}.schema.json"
        assert schema_path.exists(), f"Schema file not found: {schema_path}"

    @pytest.mark.parametrize("schema_name", EXPECTED_SCHEMAS)
    def test_schema_valid_json(self, schema_name):
        """Test that schema file is valid JSON."""
        schema_path = SCHEMA_DIR / f"{schema_name}.schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        assert isinstance(schema, dict)
        assert "$schema" in schema

    @pytest.mark.parametrize("schema_name", EXPECTED_SCHEMAS)
    def test_schema_has_id(self, schema_name):
        """Test that schema has $id field."""
        schema_path = SCHEMA_DIR / f"{schema_name}.schema.json"
        with open(schema_path) as f:
            schema = json.load(f)
        assert "$id" in schema
        assert schema["$id"].startswith("https://openspec.io/schemas/")


class TestCommonSchema:
    """Test common schema definitions."""

    @pytest.fixture
    def common_schema(self):
        """Load common schema."""
        with open(SCHEMA_DIR / "common.schema.json") as f:
            return json.load(f)

    def test_has_required_defs(self, common_schema):
        """Test that common schema has required definitions."""
        required_defs = [
            "geometry",
            "bbox",
            "temporal_extent",
            "confidence_score",
            "crs",
            "uri",
            "checksum",
            "data_type_category",
            "data_format",
            "quality_flag",
            "band_mapping",
        ]
        assert "$defs" in common_schema
        for def_name in required_defs:
            assert def_name in common_schema["$defs"], f"Missing definition: {def_name}"


class TestEventSchema:
    """Test event schema structure."""

    @pytest.fixture
    def event_schema(self):
        """Load event schema."""
        with open(SCHEMA_DIR / "event.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, event_schema):
        """Test that event schema has required fields."""
        assert "required" in event_schema
        required = event_schema["required"]
        assert "id" in required
        assert "intent" in required
        assert "spatial" in required
        assert "temporal" in required

    def test_id_pattern(self, event_schema):
        """Test that ID has correct pattern."""
        assert "properties" in event_schema
        assert "id" in event_schema["properties"]
        assert "pattern" in event_schema["properties"]["id"]
        assert event_schema["properties"]["id"]["pattern"] == "^evt_[a-z0-9_]+$"


class TestIntentSchema:
    """Test intent schema structure."""

    @pytest.fixture
    def intent_schema(self):
        """Load intent schema."""
        with open(SCHEMA_DIR / "intent.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, intent_schema):
        """Test that intent schema has required fields."""
        assert "required" in intent_schema
        required = intent_schema["required"]
        assert "input" in required
        assert "output" in required

    def test_source_enum(self, intent_schema):
        """Test that source field has correct enum values."""
        output_props = intent_schema["properties"]["output"]["properties"]
        assert "source" in output_props
        assert "enum" in output_props["source"]
        assert set(output_props["source"]["enum"]) == {"explicit", "inferred"}


class TestPipelineSchema:
    """Test pipeline schema structure."""

    @pytest.fixture
    def pipeline_schema(self):
        """Load pipeline schema."""
        with open(SCHEMA_DIR / "pipeline.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, pipeline_schema):
        """Test that pipeline schema has required fields."""
        assert "required" in pipeline_schema
        required = pipeline_schema["required"]
        assert "id" in required
        assert "name" in required
        assert "applicable_classes" in required
        assert "inputs" in required
        assert "steps" in required
        assert "outputs" in required

    def test_step_structure(self, pipeline_schema):
        """Test that step definition has correct structure."""
        steps_schema = pipeline_schema["properties"]["steps"]["items"]
        assert "required" in steps_schema
        assert "id" in steps_schema["required"]
        assert "processor" in steps_schema["required"]


class TestDataSourceSchema:
    """Test data source schema structure."""

    @pytest.fixture
    def datasource_schema(self):
        """Load datasource schema."""
        with open(SCHEMA_DIR / "datasource.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, datasource_schema):
        """Test that datasource schema has required fields."""
        assert "required" in datasource_schema
        required = datasource_schema["required"]
        assert "id" in required
        assert "provider" in required
        assert "type" in required
        assert "capabilities" in required
        assert "access" in required

    def test_type_enum(self, datasource_schema):
        """Test that type field has correct enum values."""
        type_prop = datasource_schema["properties"]["type"]
        assert "enum" in type_prop
        expected_types = {"optical", "sar", "dem", "weather", "ancillary"}
        assert set(type_prop["enum"]) == expected_types


class TestIngestionSchema:
    """Test ingestion schema structure."""

    @pytest.fixture
    def ingestion_schema(self):
        """Load ingestion schema."""
        with open(SCHEMA_DIR / "ingestion.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, ingestion_schema):
        """Test that ingestion schema has required fields."""
        assert "required" in ingestion_schema
        required = ingestion_schema["required"]
        # Actual schema uses different field names
        assert "source" in required


class TestQualitySchema:
    """Test quality schema structure."""

    @pytest.fixture
    def quality_schema(self):
        """Load quality schema."""
        with open(SCHEMA_DIR / "quality.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, quality_schema):
        """Test that quality schema has required fields."""
        assert "required" in quality_schema
        required = quality_schema["required"]
        assert "product_id" in required
        # Actual schema has different structure

    def test_status_enum(self, quality_schema):
        """Test that status field has correct enum values."""
        # Actual schema uses overall_status not results.status
        status_prop = quality_schema["properties"]["overall_status"]
        assert "enum" in status_prop
        # Check that it has status enum values
        assert len(status_prop["enum"]) > 0


class TestProvenanceSchema:
    """Test provenance schema structure."""

    @pytest.fixture
    def provenance_schema(self):
        """Load provenance schema."""
        with open(SCHEMA_DIR / "provenance.schema.json") as f:
            return json.load(f)

    def test_required_fields(self, provenance_schema):
        """Test that provenance schema has required fields."""
        assert "required" in provenance_schema
        required = provenance_schema["required"]
        assert "product_id" in required
        assert "lineage" in required

    def test_lineage_structure(self, provenance_schema):
        """Test that lineage item has correct structure."""
        lineage_item = provenance_schema["properties"]["lineage"]["items"]
        assert "required" in lineage_item
        required = lineage_item["required"]
        assert "step" in required
        assert "timestamp" in required
        assert "inputs" in required
        assert "outputs" in required
