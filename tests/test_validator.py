"""Tests for OpenSpec validator."""

from pathlib import Path

import pytest

from openspec.validator import SchemaValidator, get_validator, validate_file


class TestValidatorInitialization:
    """Test validator initialization."""

    def test_default_initialization(self):
        """Test validator with default schema directory."""
        validator = SchemaValidator()
        assert validator.schema_dir.exists()
        assert validator.schema_dir.name == "schemas"

    def test_custom_schema_dir(self, tmp_path):
        """Test validator with custom schema directory."""
        validator = SchemaValidator(schema_dir=tmp_path)
        assert validator.schema_dir == tmp_path


class TestSchemaLoading:
    """Test schema loading functionality."""

    def test_load_event_schema(self):
        """Test loading event schema."""
        validator = SchemaValidator()
        schema = validator._load_schema("event")
        assert schema is not None
        assert "$schema" in schema
        assert schema["$id"] == "https://openspec.io/schemas/event.schema.json"

    def test_load_intent_schema(self):
        """Test loading intent schema."""
        validator = SchemaValidator()
        schema = validator._load_schema("intent")
        assert schema is not None
        assert schema["$id"] == "https://openspec.io/schemas/intent.schema.json"

    def test_schema_caching(self):
        """Test that schemas are cached after first load."""
        validator = SchemaValidator()
        schema1 = validator._load_schema("event")
        schema2 = validator._load_schema("event")
        assert schema1 is schema2  # Should be same object due to caching

    def test_load_nonexistent_schema(self):
        """Test loading a schema that doesn't exist."""
        validator = SchemaValidator()
        with pytest.raises(FileNotFoundError) as exc_info:
            validator._load_schema("nonexistent")
        assert "Schema file not found" in str(exc_info.value)


class TestEventValidation:
    """Test event specification validation."""

    @pytest.fixture
    def validator(self):
        """Get validator instance."""
        return SchemaValidator()

    @pytest.fixture
    def valid_event(self):
        """Create a valid event specification."""
        return {
            "id": "evt_test_001",
            "intent": {
                "class": "flood.coastal.storm_surge",
                "source": "explicit",
                "confidence": 0.95
            },
            "spatial": {
                "type": "Polygon",
                "coordinates": [[
                    [-80.0, 25.0],
                    [-79.0, 25.0],
                    [-79.0, 26.0],
                    [-80.0, 26.0],
                    [-80.0, 25.0]
                ]],
                "crs": "EPSG:4326"
            },
            "temporal": {
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-10T23:59:59Z"
            }
        }

    def test_valid_event(self, validator, valid_event):
        """Test validation of a valid event."""
        is_valid, errors = validator.validate_event(valid_event)
        assert is_valid
        assert len(errors) == 0

    def test_missing_required_field(self, validator, valid_event):
        """Test validation fails when required field is missing."""
        del valid_event["id"]
        is_valid, errors = validator.validate_event(valid_event)
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_id_pattern(self, validator, valid_event):
        """Test validation fails for invalid ID pattern."""
        valid_event["id"] = "invalid-id-format"
        is_valid, errors = validator.validate_event(valid_event)
        assert not is_valid
        assert any("pattern" in err.lower() for err in errors)

    def test_invalid_event_class_pattern(self, validator, valid_event):
        """Test validation fails for invalid event class pattern."""
        valid_event["intent"]["class"] = "Invalid.Class.Name"
        is_valid, errors = validator.validate_event(valid_event)
        assert not is_valid

    def test_invalid_confidence_score(self, validator, valid_event):
        """Test validation fails for confidence score outside 0-1 range."""
        valid_event["intent"]["confidence"] = 1.5
        is_valid, errors = validator.validate_event(valid_event)
        assert not is_valid


class TestIntentValidation:
    """Test intent resolution validation."""

    @pytest.fixture
    def validator(self):
        """Get validator instance."""
        return SchemaValidator()

    @pytest.fixture
    def valid_intent(self):
        """Create a valid intent specification."""
        return {
            "input": {
                "natural_language": "flooding in coastal areas"
            },
            "output": {
                "resolved_class": "flood.coastal",
                "source": "inferred",
                "confidence": 0.87
            }
        }

    def test_valid_intent(self, validator, valid_intent):
        """Test validation of a valid intent."""
        is_valid, errors = validator.validate_intent(valid_intent)
        assert is_valid
        assert len(errors) == 0

    def test_invalid_source_enum(self, validator, valid_intent):
        """Test validation fails for invalid source value."""
        valid_intent["output"]["source"] = "invalid_source"
        is_valid, errors = validator.validate_intent(valid_intent)
        assert not is_valid

    def test_with_alternatives(self, validator, valid_intent):
        """Test intent with alternatives."""
        valid_intent["resolution"] = {
            "inferred_class": "flood.coastal",
            "confidence": 0.87,
            "alternatives": [
                {"class": "flood.riverine", "confidence": 0.12}
            ]
        }
        is_valid, errors = validator.validate_intent(valid_intent)
        assert is_valid


class TestExampleFiles:
    """Test that example files validate correctly."""

    @pytest.fixture
    def examples_dir(self):
        """Get examples directory."""
        return Path(__file__).parent.parent / "examples"

    def test_flood_example_exists(self, examples_dir):
        """Test that flood example file exists."""
        flood_example = examples_dir / "flood_event.yaml"
        assert flood_example.exists()

    def test_wildfire_example_exists(self, examples_dir):
        """Test that wildfire example file exists."""
        wildfire_example = examples_dir / "wildfire_event.yaml"
        assert wildfire_example.exists()

    def test_storm_example_exists(self, examples_dir):
        """Test that storm example file exists."""
        storm_example = examples_dir / "storm_event.yaml"
        assert storm_example.exists()

    @pytest.mark.parametrize("example_file", [
        "flood_event.yaml",
        "wildfire_event.yaml",
        "storm_event.yaml"
    ])
    def test_example_validates(self, examples_dir, example_file):
        """Test that example files validate against event schema."""
        example_path = examples_dir / example_file
        is_valid, errors = validate_file(example_path, "event")
        if not is_valid:
            print(f"\nValidation errors for {example_file}:")
            for error in errors:
                print(f"  • {error}")
        assert is_valid, f"{example_file} failed validation"


class TestValidatorConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_validator(self):
        """Test get_validator returns a SchemaValidator instance."""
        validator = get_validator()
        assert isinstance(validator, SchemaValidator)

    def test_get_validator_singleton(self):
        """Test get_validator returns same instance."""
        validator1 = get_validator()
        validator2 = get_validator()
        assert validator1 is validator2

    def test_validate_file_function(self, tmp_path):
        """Test validate_file convenience function."""
        # Create a temporary valid event file
        event_file = tmp_path / "test_event.yaml"
        event_file.write_text("""
id: evt_test_001
intent:
  class: flood.coastal
  source: explicit
spatial:
  type: Polygon
  coordinates:
    - [[-80.0, 25.0], [-79.0, 25.0], [-79.0, 26.0], [-80.0, 26.0], [-80.0, 25.0]]
temporal:
  start: "2024-01-01T00:00:00Z"
  end: "2024-01-10T00:00:00Z"
""")

        is_valid, errors = validate_file(event_file, "event")
        assert is_valid
        assert len(errors) == 0
