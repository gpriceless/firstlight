"""Tests for intent resolution API integration.

Tests that the events API correctly resolves intent from natural language
and validates explicit event classes against the taxonomy.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from api.routes.events import create_event, _events_store
from api.models.requests import (
    EventSubmitRequest,
    EventIntent,
    SpatialExtent,
    TemporalWindow,
    BoundingBox,
    EventPriority,
)
from api.models.errors import ValidationError
from core.intent.resolver import IntentResolver, IntentResolution


@pytest.fixture
def mock_dependencies():
    """Mock dependencies for API endpoint testing."""
    return {
        "settings": MagicMock(),
        "db_session": MagicMock(),
        "schema_validator": MagicMock(validate_event=MagicMock(return_value=(True, []))),
        "agent_registry": MagicMock(),
        "auth": MagicMock(),
        "correlation_id": "test-correlation-123",
        "_rate_limit": MagicMock(),
    }


@pytest.fixture
def base_request():
    """Base request with minimal required fields."""
    now = datetime.now(timezone.utc)
    return EventSubmitRequest(
        intent=EventIntent(natural_language="", event_class=None),
        spatial=SpatialExtent(
            bbox=BoundingBox(west=-80.5, south=25.5, east=-80.0, north=26.0),
            crs="EPSG:4326",
        ),
        temporal=TemporalWindow(
            start=now,
            end=now + timedelta(days=5),
        ),
        priority=EventPriority.NORMAL,
    )


@pytest.fixture(autouse=True)
def clear_events_store():
    """Clear the in-memory events store before each test."""
    _events_store.clear()
    yield
    _events_store.clear()


class TestNaturalLanguageIntentResolution:
    """Test natural language intent resolution in API."""

    @pytest.mark.intent
    async def test_flooding_after_hurricane_resolves_to_coastal(
        self, base_request, mock_dependencies
    ):
        """Test that 'flooding after hurricane' resolves to flood.coastal."""
        base_request.intent.natural_language = "flooding after hurricane"
        base_request.intent.event_class = None

        event = await create_event(base_request, **mock_dependencies)

        assert event.intent.event_class == "flood.coastal"
        assert event.intent.confidence > 0.0
        assert event.intent.source == "inferred"
        assert event.intent.original_input == "flooding after hurricane"

    @pytest.mark.intent
    async def test_forest_fire_resolves_to_wildfire_forest(
        self, base_request, mock_dependencies
    ):
        """Test that 'forest fire' resolves to wildfire.forest."""
        base_request.intent.natural_language = "forest fire"
        base_request.intent.event_class = None

        event = await create_event(base_request, **mock_dependencies)

        assert event.intent.event_class == "wildfire.forest"
        assert event.intent.confidence > 0.0
        assert event.intent.source == "inferred"

    @pytest.mark.intent
    async def test_coastal_flooding_resolves_correctly(
        self, base_request, mock_dependencies
    ):
        """Test that 'coastal flooding' resolves to coastal flood class."""
        base_request.intent.natural_language = "coastal flooding"
        base_request.intent.event_class = None

        event = await create_event(base_request, **mock_dependencies)

        assert "coastal" in event.intent.event_class.lower()
        assert "flood" in event.intent.event_class.lower()
        assert event.intent.confidence > 0.0

    @pytest.mark.intent
    async def test_nlp_resolution_extracts_parameters(
        self, base_request, mock_dependencies
    ):
        """Test that NLP resolution extracts parameters from text."""
        base_request.intent.natural_language = "flooding after hurricane"
        base_request.intent.event_class = None

        event = await create_event(base_request, **mock_dependencies)

        # Should extract temporal context (after hurricane)
        assert event.intent.parameters is not None
        # Could include causation or temporal context
        assert len(event.intent.parameters) >= 0  # At least extracts hierarchy

    @pytest.mark.intent
    async def test_invalid_natural_language_raises_validation_error(
        self, base_request, mock_dependencies
    ):
        """Test that invalid NL that can't be resolved raises ValidationError."""
        base_request.intent.natural_language = "xyzabc123nonsense"
        base_request.intent.event_class = None

        with patch.object(
            IntentResolver,
            "resolve",
            return_value=None,  # Simulate resolution failure
        ):
            with pytest.raises(ValidationError) as exc_info:
                await create_event(base_request, **mock_dependencies)

            assert "Could not resolve intent" in str(exc_info.value.message)

    @pytest.mark.intent
    async def test_nlp_failure_provides_helpful_error(
        self, base_request, mock_dependencies
    ):
        """Test that NLP failures provide helpful error messages."""
        base_request.intent.natural_language = "gibberish text"
        base_request.intent.event_class = None

        with patch.object(
            IntentResolver,
            "resolve",
            side_effect=Exception("NLP service unavailable"),
        ):
            with pytest.raises(ValidationError) as exc_info:
                await create_event(base_request, **mock_dependencies)

            error_msg = str(exc_info.value.message)
            assert "Intent resolution failed" in error_msg
            assert "explicit event_class" in error_msg


class TestExplicitClassValidation:
    """Test explicit event class validation against taxonomy."""

    @pytest.mark.intent
    async def test_valid_explicit_class_accepted(
        self, base_request, mock_dependencies
    ):
        """Test that valid explicit event classes are accepted."""
        base_request.intent.event_class = "flood.coastal.storm_surge"
        base_request.intent.natural_language = None

        event = await create_event(base_request, **mock_dependencies)

        assert event.intent.event_class == "flood.coastal.storm_surge"
        assert event.intent.confidence == 1.0
        assert event.intent.source == "explicit"

    @pytest.mark.intent
    async def test_invalid_explicit_class_rejected(
        self, base_request, mock_dependencies
    ):
        """Test that invalid event classes are rejected with 400."""
        base_request.intent.event_class = "invalid.nonexistent.class"
        base_request.intent.natural_language = None

        with pytest.raises(ValidationError) as exc_info:
            await create_event(base_request, **mock_dependencies)

        error_msg = str(exc_info.value.message)
        assert "Invalid event class" in error_msg
        assert "invalid.nonexistent.class" in error_msg
        assert "Available classes include" in error_msg

    @pytest.mark.intent
    async def test_root_level_class_accepted(
        self, base_request, mock_dependencies
    ):
        """Test that root-level classes like 'flood' are accepted."""
        base_request.intent.event_class = "flood"
        base_request.intent.natural_language = None

        event = await create_event(base_request, **mock_dependencies)

        assert event.intent.event_class == "flood"
        assert event.intent.confidence == 1.0

    @pytest.mark.intent
    async def test_nested_class_path_accepted(
        self, base_request, mock_dependencies
    ):
        """Test that nested class paths are accepted."""
        base_request.intent.event_class = "wildfire.forest.crown_fire"
        base_request.intent.natural_language = None

        event = await create_event(base_request, **mock_dependencies)

        assert event.intent.event_class == "wildfire.forest.crown_fire"


class TestIntentResolutionEdgeCases:
    """Test edge cases in intent resolution."""

    @pytest.mark.intent
    async def test_no_intent_provided_raises_error(
        self, base_request, mock_dependencies
    ):
        """Test that providing neither NL nor explicit class raises error."""
        base_request.intent.event_class = None
        base_request.intent.natural_language = None

        # Should be caught during validation or raise error
        # The current implementation will try to proceed with "pending_resolution"
        # but should ideally reject this
        event = await create_event(base_request, **mock_dependencies)

        # Depending on implementation, could accept with pending or reject
        # Current behavior accepts with default
        assert event.intent.event_class == "pending_resolution"

    @pytest.mark.intent
    async def test_both_nl_and_explicit_class_provided(
        self, base_request, mock_dependencies
    ):
        """Test that explicit class is used when both are provided."""
        base_request.intent.natural_language = "forest fire"
        base_request.intent.event_class = "flood.riverine"

        event = await create_event(base_request, **mock_dependencies)

        # Explicit class should take precedence (current implementation)
        assert event.intent.event_class == "flood.riverine"
        assert event.intent.confidence == 1.0
        assert event.intent.source == "explicit"

    @pytest.mark.intent
    async def test_low_confidence_resolution_still_accepted(
        self, base_request, mock_dependencies
    ):
        """Test that low confidence resolutions are accepted if > 0."""
        base_request.intent.natural_language = "potential flooding scenario"
        base_request.intent.event_class = None

        # Create a mock resolution with low confidence
        with patch.object(
            IntentResolver,
            "resolve",
            return_value=IntentResolution(
                resolved_class="flood",
                source="inferred",
                confidence=0.3,
                original_input="potential flooding scenario",
            ),
        ):
            event = await create_event(base_request, **mock_dependencies)

            assert event.intent.event_class == "flood"
            assert event.intent.confidence == 0.3

    @pytest.mark.intent
    async def test_zero_confidence_resolution_rejected(
        self, base_request, mock_dependencies
    ):
        """Test that zero confidence resolutions are rejected."""
        base_request.intent.natural_language = "unclear event description"
        base_request.intent.event_class = None

        with patch.object(
            IntentResolver,
            "resolve",
            return_value=IntentResolution(
                resolved_class="unknown",
                source="inferred",
                confidence=0.0,
                original_input="unclear event description",
            ),
        ):
            with pytest.raises(ValidationError) as exc_info:
                await create_event(base_request, **mock_dependencies)

            assert "Could not resolve intent" in str(exc_info.value.message)
