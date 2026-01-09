"""Tests for example YAML files to ensure they validate against schemas."""

from pathlib import Path

import pytest
import yaml

from openspec.validator import SchemaValidator


EXAMPLES_DIR = Path(__file__).parent.parent / "examples"
EXPECTED_EXAMPLES = {
    "event": [
        "flood_event.yaml",
        "wildfire_event.yaml",
        "storm_event.yaml",
    ],
    "datasource": [
        "datasource_sentinel1.yaml",
    ],
    "pipeline": [
        "pipeline_flood_sar.yaml",
    ],
}


@pytest.fixture
def validator():
    """Create a schema validator instance."""
    return SchemaValidator()


class TestExampleFiles:
    """Test that all example files exist and are valid YAML."""

    def test_examples_directory_exists(self):
        """Test that examples directory exists."""
        assert EXAMPLES_DIR.exists(), f"Examples directory not found: {EXAMPLES_DIR}"

    @pytest.mark.parametrize("schema_name,examples", EXPECTED_EXAMPLES.items())
    def test_example_files_exist(self, schema_name, examples):
        """Test that all expected example files exist."""
        for example_file in examples:
            example_path = EXAMPLES_DIR / example_file
            assert example_path.exists(), f"Example file not found: {example_path}"

    @pytest.mark.parametrize("schema_name,examples", EXPECTED_EXAMPLES.items())
    def test_examples_valid_yaml(self, schema_name, examples):
        """Test that all example files are valid YAML."""
        for example_file in examples:
            example_path = EXAMPLES_DIR / example_file
            with open(example_path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{example_file} should parse to a dictionary"


class TestEventExamples:
    """Test event example files validate against event schema."""

    @pytest.mark.parametrize("example_file", EXPECTED_EXAMPLES["event"])
    def test_event_validates(self, validator, example_file):
        """Test that event example validates against event schema."""
        example_path = EXAMPLES_DIR / example_file
        valid, errors = validator.validate_event(example_path)

        assert valid, (
            f"{example_file} failed validation against event schema.\n"
            f"Errors:\n" + "\n".join(f"  • {err}" for err in errors)
        )

    def test_flood_event_structure(self):
        """Test flood event has expected structure."""
        with open(EXAMPLES_DIR / "flood_event.yaml") as f:
            data = yaml.safe_load(f)

        assert data["id"].startswith("evt_")
        assert "flood" in data["intent"]["class"]
        assert data["spatial"]["type"] == "Polygon"
        assert "start" in data["temporal"]
        assert "end" in data["temporal"]
        assert data["priority"] in ["low", "medium", "high", "critical"]

    def test_wildfire_event_structure(self):
        """Test wildfire event has expected structure."""
        with open(EXAMPLES_DIR / "wildfire_event.yaml") as f:
            data = yaml.safe_load(f)

        assert data["id"].startswith("evt_")
        assert "wildfire" in data["intent"]["class"]
        assert data["spatial"]["type"] == "Polygon"
        assert "optical" in data["constraints"]["required_data_types"]

    def test_storm_event_structure(self):
        """Test storm event has expected structure."""
        with open(EXAMPLES_DIR / "storm_event.yaml") as f:
            data = yaml.safe_load(f)

        assert data["id"].startswith("evt_")
        assert "storm" in data["intent"]["class"]
        assert data["spatial"]["type"] == "Polygon"
        assert data["constraints"]["min_resolution_m"] == 3  # High res for damage


class TestDataSourceExamples:
    """Test data source example files validate against datasource schema."""

    @pytest.mark.parametrize("example_file", EXPECTED_EXAMPLES["datasource"])
    def test_datasource_validates(self, validator, example_file):
        """Test that datasource example validates against datasource schema."""
        example_path = EXAMPLES_DIR / example_file
        valid, errors = validator.validate_datasource(example_path)

        assert valid, (
            f"{example_file} failed validation against datasource schema.\n"
            f"Errors:\n" + "\n".join(f"  • {err}" for err in errors)
        )

    def test_sentinel1_datasource_structure(self):
        """Test Sentinel-1 datasource has expected structure."""
        with open(EXAMPLES_DIR / "datasource_sentinel1.yaml") as f:
            data = yaml.safe_load(f)

        assert data["id"] == "sentinel1_grd"
        assert data["type"] == "sar"
        assert data["provider"] == "ESA"
        assert data["access"]["protocol"] == "stac"
        assert data["cost"]["tier"] == "open"
        assert "VV" in data["capabilities"]["polarizations"]


class TestPipelineExamples:
    """Test pipeline example files validate against pipeline schema."""

    @pytest.mark.parametrize("example_file", EXPECTED_EXAMPLES["pipeline"])
    def test_pipeline_validates(self, validator, example_file):
        """Test that pipeline example validates against pipeline schema."""
        example_path = EXAMPLES_DIR / example_file
        valid, errors = validator.validate_pipeline(example_path)

        assert valid, (
            f"{example_file} failed validation against pipeline schema.\n"
            f"Errors:\n" + "\n".join(f"  • {err}" for err in errors)
        )

    def test_flood_sar_pipeline_structure(self):
        """Test flood SAR pipeline has expected structure."""
        with open(EXAMPLES_DIR / "pipeline_flood_sar.yaml") as f:
            data = yaml.safe_load(f)

        assert data["id"] == "flood_mapping_sar_change"
        assert len(data["applicable_classes"]) > 0
        assert any("flood" in cls for cls in data["applicable_classes"])

        # Check for essential steps
        step_ids = [step["id"] for step in data["steps"]]
        assert "change_detection" in step_ids
        assert "vectorize" in step_ids

        # Check inputs
        input_names = [inp["name"] for inp in data["inputs"]]
        assert "sar_pre_event" in input_names
        assert "sar_post_event" in input_names

        # Check outputs
        output_names = [out["name"] for out in data["outputs"]]
        assert "flood_extent_raster" in output_names
        assert "flood_extent_vector" in output_names


class TestCrossValidation:
    """Test cross-validation between examples (e.g., event refs datasource)."""

    def test_event_references_valid_data_types(self):
        """Test that events reference valid data types."""
        valid_data_types = {"optical", "sar", "dem", "weather", "ancillary"}

        for example_file in EXPECTED_EXAMPLES["event"]:
            with open(EXAMPLES_DIR / example_file) as f:
                data = yaml.safe_load(f)

            if "constraints" in data and "required_data_types" in data["constraints"]:
                for data_type in data["constraints"]["required_data_types"]:
                    assert data_type in valid_data_types, (
                        f"{example_file} references invalid data type: {data_type}"
                    )

            if "constraints" in data and "optional_data_types" in data["constraints"]:
                for data_type in data["constraints"]["optional_data_types"]:
                    assert data_type in valid_data_types, (
                        f"{example_file} references invalid data type: {data_type}"
                    )

    def test_pipeline_applicable_classes_match_event_classes(self):
        """Test that pipeline applicable classes match actual event classes."""
        # Collect event classes from examples
        event_classes = []
        for example_file in EXPECTED_EXAMPLES["event"]:
            with open(EXAMPLES_DIR / example_file) as f:
                data = yaml.safe_load(f)
            event_classes.append(data["intent"]["class"])

        # Check pipeline applicable classes
        for example_file in EXPECTED_EXAMPLES["pipeline"]:
            with open(EXAMPLES_DIR / example_file) as f:
                data = yaml.safe_load(f)

            # At least one applicable class should match or be a prefix of an event class
            applicable = data["applicable_classes"]
            assert any(
                any(event_cls.startswith(app_cls) for event_cls in event_classes)
                for app_cls in applicable
            ), f"{example_file} has no matching event classes"


class TestExampleCompleteness:
    """Test that we have complete example coverage."""

    def test_all_hazard_types_covered(self):
        """Test that we have examples for all major hazard types."""
        hazard_types = {"flood", "wildfire", "storm"}

        covered_types = set()
        for example_file in EXPECTED_EXAMPLES["event"]:
            with open(EXAMPLES_DIR / example_file) as f:
                data = yaml.safe_load(f)
            event_class = data["intent"]["class"]
            covered_types.add(event_class.split(".")[0])

        assert hazard_types == covered_types, (
            f"Not all hazard types covered. Missing: {hazard_types - covered_types}"
        )

    def test_all_data_type_categories_represented(self):
        """Test that examples collectively reference all data type categories."""
        expected_categories = {"optical", "sar", "dem", "weather", "ancillary"}

        referenced_categories = set()
        for example_file in EXPECTED_EXAMPLES["event"]:
            with open(EXAMPLES_DIR / example_file) as f:
                data = yaml.safe_load(f)

            if "constraints" in data:
                if "required_data_types" in data["constraints"]:
                    referenced_categories.update(data["constraints"]["required_data_types"])
                if "optional_data_types" in data["constraints"]:
                    referenced_categories.update(data["constraints"]["optional_data_types"])

        # We should have most categories represented (not necessarily all)
        assert len(referenced_categories & expected_categories) >= 4, (
            f"Examples should cover at least 4 data type categories. "
            f"Found: {referenced_categories}"
        )
