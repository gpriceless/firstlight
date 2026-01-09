"""OpenSpec schema validation utilities.

Provides validation for all OpenSpec schemas with helpful error messages,
schema loading and caching, and validation utilities for each schema type.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from jsonschema import Draft202012Validator, RefResolver, ValidationError


class SchemaValidator:
    """Schema validator with caching and helpful error messages."""

    def __init__(self, schema_dir: Optional[Path] = None):
        """Initialize the validator.

        Args:
            schema_dir: Directory containing JSON schema files.
                       Defaults to openspec/schemas/ relative to this file.
        """
        if schema_dir is None:
            schema_dir = Path(__file__).parent / "schemas"
        self.schema_dir = Path(schema_dir)
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validator_cache: Dict[str, Draft202012Validator] = {}

    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema from disk with caching.

        Args:
            schema_name: Name of the schema (e.g., 'event', 'intent')

        Returns:
            Loaded schema dictionary

        Raises:
            FileNotFoundError: If schema file doesn't exist
            json.JSONDecodeError: If schema file is invalid JSON
        """
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]

        schema_path = self.schema_dir / f"{schema_name}.schema.json"
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Schema file not found: {schema_path}\n"
                f"Available schemas in {self.schema_dir}: "
                f"{[f.stem.replace('.schema', '') for f in self.schema_dir.glob('*.schema.json')]}"
            )

        with open(schema_path) as f:
            schema = json.load(f)

        self._schema_cache[schema_name] = schema
        return schema

    def _get_validator(self, schema_name: str) -> Draft202012Validator:
        """Get a validator for the specified schema with reference resolution.

        Args:
            schema_name: Name of the schema

        Returns:
            Configured validator instance
        """
        if schema_name in self._validator_cache:
            return self._validator_cache[schema_name]

        schema = self._load_schema(schema_name)

        # Set up reference resolver to handle $ref references
        store = {}
        for schema_file in self.schema_dir.glob("*.schema.json"):
            with open(schema_file) as f:
                s = json.load(f)
                store[s.get("$id", "")] = s

        resolver = RefResolver.from_schema(schema, store=store)
        validator = Draft202012Validator(schema, resolver=resolver)

        self._validator_cache[schema_name] = validator
        return validator

    def validate(
        self, instance: Union[Dict[str, Any], str, Path], schema_name: str
    ) -> Tuple[bool, List[str]]:
        """Validate an instance against a schema.

        Args:
            instance: Instance to validate (dict, YAML string, or file path)
            schema_name: Name of the schema to validate against

        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Load instance if it's a file path or string
        if isinstance(instance, (str, Path)):
            instance_data = self._load_instance(instance)
        else:
            instance_data = instance

        validator = self._get_validator(schema_name)
        errors = []

        try:
            validator.validate(instance_data)
            return True, []
        except ValidationError as e:
            # Provide helpful error messages
            error_path = " → ".join(str(p) for p in e.absolute_path) if e.absolute_path else "root"
            error_msg = f"Validation error at '{error_path}': {e.message}"

            # Add context about the failing value
            if e.instance is not None:
                error_msg += f"\nFailing value: {e.instance}"

            # Add expected schema information
            if e.schema:
                if "enum" in e.schema:
                    error_msg += f"\nExpected one of: {e.schema['enum']}"
                if "pattern" in e.schema:
                    error_msg += f"\nExpected pattern: {e.schema['pattern']}"

            errors.append(error_msg)

            # Collect all validation errors, not just the first one
            for error in validator.iter_errors(instance_data):
                if error != e:  # Skip the one we already processed
                    error_path = " → ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
                    errors.append(f"At '{error_path}': {error.message}")

        return False, errors

    def _load_instance(self, source: Union[str, Path]) -> Dict[str, Any]:
        """Load an instance from a file or YAML string.

        Args:
            source: File path or YAML string

        Returns:
            Loaded instance dictionary
        """
        # Try as file path first
        source_path = Path(source)
        if source_path.exists():
            with open(source_path) as f:
                if source_path.suffix in [".yaml", ".yml"]:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)

        # Try as YAML string
        try:
            return yaml.safe_load(source)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML/JSON string: {e}")

    def validate_event(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate an event specification.

        Args:
            instance: Event instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "event")

    def validate_intent(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate an intent resolution specification.

        Args:
            instance: Intent instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "intent")

    def validate_datasource(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate a data source configuration.

        Args:
            instance: Data source instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "datasource")

    def validate_pipeline(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate a pipeline definition.

        Args:
            instance: Pipeline instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "pipeline")

    def validate_ingestion(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate an ingestion configuration.

        Args:
            instance: Ingestion instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "ingestion")

    def validate_quality(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate a quality control report.

        Args:
            instance: Quality report instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "quality")

    def validate_provenance(self, instance: Union[Dict, str, Path]) -> Tuple[bool, List[str]]:
        """Validate a provenance record.

        Args:
            instance: Provenance instance to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return self.validate(instance, "provenance")


# Convenience instance for module-level usage
_default_validator: Optional[SchemaValidator] = None


def get_validator() -> SchemaValidator:
    """Get the default schema validator instance.

    Returns:
        Default SchemaValidator instance
    """
    global _default_validator
    if _default_validator is None:
        _default_validator = SchemaValidator()
    return _default_validator


def validate_file(file_path: Union[str, Path], schema_name: str) -> Tuple[bool, List[str]]:
    """Validate a YAML/JSON file against a schema.

    Args:
        file_path: Path to the file to validate
        schema_name: Name of the schema to validate against

    Returns:
        Tuple of (is_valid, error_messages)
    """
    return get_validator().validate(file_path, schema_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m openspec.validator <file> [schema_name]")
        print("Example: python -m openspec.validator examples/flood_event.yaml event")
        sys.exit(1)

    file_path = sys.argv[1]
    schema_name = sys.argv[2] if len(sys.argv) > 2 else "event"

    valid, errors = validate_file(file_path, schema_name)

    if valid:
        print(f"✓ {file_path} is valid against {schema_name} schema")
        sys.exit(0)
    else:
        print(f"✗ {file_path} failed validation against {schema_name} schema")
        print("\nErrors:")
        for error in errors:
            print(f"  • {error}")
        sys.exit(1)
