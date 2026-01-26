"""Tests for emergency resources module."""

import pytest
from core.reporting.data.emergency_resources import (
    EmergencyResources,
    EmergencyContact,
    StateEmergencyInfo,
    DisasterType,
)


@pytest.fixture
def resources():
    """Create EmergencyResources instance."""
    return EmergencyResources()


class TestNationalResources:
    """Tests for national emergency resources."""

    def test_get_national_resources(self, resources):
        """Test retrieval of national resources."""
        national = resources.get_national_resources()

        assert len(national) > 0
        assert all(isinstance(c, EmergencyContact) for c in national)

        # Check for key national resources
        names = [c.name for c in national]
        assert "FEMA" in names
        assert "American Red Cross" in names
        assert "National Weather Service" in names

    def test_national_resources_have_required_fields(self, resources):
        """Test that national resources have proper data."""
        national = resources.get_national_resources()

        for contact in national:
            assert contact.name
            assert contact.description
            # At least one contact method must exist
            assert contact.phone or contact.url

    def test_fema_contact_info(self, resources):
        """Test FEMA contact information."""
        national = resources.get_national_resources()
        fema = next((c for c in national if c.name == "FEMA"), None)

        assert fema is not None
        assert fema.phone == "1-800-621-3362"
        assert "fema.gov" in fema.url.lower()


class TestStateResources:
    """Tests for state-specific resources."""

    def test_get_state_resources(self, resources):
        """Test retrieval of state resources."""
        state_info = resources.get_state_resources("FL")

        assert isinstance(state_info, StateEmergencyInfo)
        assert state_info.state_name == "Florida"
        assert state_info.state_abbrev == "FL"
        assert state_info.emergency_mgmt_url
        assert state_info.emergency_mgmt_phone

    def test_required_states_present(self, resources):
        """Test that required states are in the database."""
        required_states = ["FL", "TX", "CA", "LA", "NC"]

        for state in required_states:
            state_info = resources.get_state_resources(state)
            assert state_info is not None
            assert state_info.state_abbrev == state

    def test_state_case_insensitive(self, resources):
        """Test that state lookup is case-insensitive."""
        upper = resources.get_state_resources("FL")
        lower = resources.get_state_resources("fl")

        assert upper.state_abbrev == lower.state_abbrev

    def test_invalid_state_raises_error(self, resources):
        """Test that invalid state raises KeyError."""
        with pytest.raises(KeyError):
            resources.get_state_resources("XX")

    def test_state_has_governor_office_url(self, resources):
        """Test that states have governor office URLs."""
        state_info = resources.get_state_resources("FL")
        assert state_info.governor_office_url is not None
        assert "gov" in state_info.governor_office_url.lower()


class TestDisasterSpecificResources:
    """Tests for disaster-type-specific resources."""

    def test_flood_resources(self, resources):
        """Test flood-specific resources."""
        flood = resources.get_disaster_specific_resources(DisasterType.FLOOD)

        assert len(flood) > 0
        assert all(isinstance(c, EmergencyContact) for c in flood)

        # Check for flood-specific resources
        names = [c.name.lower() for c in flood]
        assert any("flood" in name for name in names)

    def test_wildfire_resources(self, resources):
        """Test wildfire-specific resources."""
        wildfire = resources.get_disaster_specific_resources(DisasterType.WILDFIRE)

        assert len(wildfire) > 0

        # Check for wildfire-specific resources
        names = [c.name.lower() for c in wildfire]
        descriptions = [c.description.lower() for c in wildfire]
        combined = names + descriptions

        assert any("fire" in text for text in combined)
        assert any("air" in text or "smoke" in text for text in combined)

    def test_hurricane_resources(self, resources):
        """Test hurricane-specific resources."""
        hurricane = resources.get_disaster_specific_resources(DisasterType.HURRICANE)

        assert len(hurricane) > 0

        # Should include National Hurricane Center
        names = [c.name for c in hurricane]
        assert any("hurricane" in name.lower() for name in names)

    def test_tornado_resources(self, resources):
        """Test tornado-specific resources."""
        tornado = resources.get_disaster_specific_resources(DisasterType.TORNADO)

        assert len(tornado) > 0
        names = [c.name.lower() for c in tornado]
        assert any("tornado" in name or "storm" in name for name in names)

    def test_earthquake_resources(self, resources):
        """Test earthquake-specific resources."""
        earthquake = resources.get_disaster_specific_resources(DisasterType.EARTHQUAKE)

        assert len(earthquake) > 0

        # Should include USGS
        names = [c.name.lower() for c in earthquake]
        assert any("earthquake" in name or "usgs" in name for name in names)


class TestRoadClosureInfo:
    """Tests for road closure information."""

    def test_get_road_closure_url(self, resources):
        """Test retrieval of state road closure URLs."""
        url = resources.get_road_closure_url("FL")

        assert url
        assert url.startswith("http")

    def test_road_closure_urls_for_required_states(self, resources):
        """Test that required states have road closure URLs."""
        required_states = ["FL", "TX", "CA", "LA", "NC"]

        for state in required_states:
            url = resources.get_road_closure_url(state)
            assert url
            assert url.startswith("http")

    def test_unknown_state_returns_fallback(self, resources):
        """Test that unknown states get federal fallback URL."""
        url = resources.get_road_closure_url("XX")

        assert url
        assert url.startswith("http")
        # Should be federal fallback
        assert "fhwa" in url or "dot.gov" in url


class TestWhatToDoActions:
    """Tests for disaster-specific action items."""

    def test_flood_actions_present(self, resources):
        """Test that flood actions are available."""
        section = resources.generate_resources_section("FL", DisasterType.FLOOD)

        actions = section['what_to_do']
        assert len(actions) > 0

        # Check for key flood safety messages
        combined = " ".join(actions).lower()
        assert "flood" in combined or "water" in combined

    def test_wildfire_actions_present(self, resources):
        """Test that wildfire actions are available."""
        section = resources.generate_resources_section("CA", DisasterType.WILDFIRE)

        actions = section['what_to_do']
        assert len(actions) > 0

        # Check for key wildfire safety messages
        combined = " ".join(actions).lower()
        assert any(word in combined for word in ["evacuate", "smoke", "fire"])

    def test_hurricane_actions_present(self, resources):
        """Test that hurricane actions are available."""
        section = resources.generate_resources_section("FL", DisasterType.HURRICANE)

        actions = section['what_to_do']
        assert len(actions) > 0
        assert len(actions) >= 5  # Should have multiple action items

    def test_tornado_actions_present(self, resources):
        """Test that tornado actions are available."""
        section = resources.generate_resources_section("TX", DisasterType.TORNADO)

        actions = section['what_to_do']
        assert len(actions) > 0

        # Check for tornado shelter guidance
        combined = " ".join(actions).lower()
        assert "shelter" in combined or "basement" in combined

    def test_earthquake_actions_present(self, resources):
        """Test that earthquake actions are available."""
        section = resources.generate_resources_section("CA", DisasterType.EARTHQUAKE)

        actions = section['what_to_do']
        assert len(actions) > 0

        # Check for Drop, Cover, Hold On
        combined = " ".join(actions).lower()
        assert any(word in combined for word in ["drop", "cover", "hold"])


class TestGenerateResourcesSection:
    """Tests for complete resource section generation."""

    def test_generate_resources_section(self, resources):
        """Test generation of complete resources section."""
        section = resources.generate_resources_section(
            state_abbrev="FL",
            disaster_type=DisasterType.HURRICANE
        )

        # Check all required keys present
        assert 'national' in section
        assert 'state' in section
        assert 'disaster_specific' in section
        assert 'road_info_url' in section
        assert 'what_to_do' in section

    def test_section_has_national_resources(self, resources):
        """Test that section includes national resources."""
        section = resources.generate_resources_section("FL", DisasterType.FLOOD)

        national = section['national']
        assert len(national) > 0
        assert all(isinstance(c, EmergencyContact) for c in national)

    def test_section_has_state_info(self, resources):
        """Test that section includes state information."""
        section = resources.generate_resources_section("TX", DisasterType.FLOOD)

        state = section['state']
        assert isinstance(state, StateEmergencyInfo)
        assert state.state_abbrev == "TX"

    def test_section_has_disaster_resources(self, resources):
        """Test that section includes disaster-specific resources."""
        section = resources.generate_resources_section("CA", DisasterType.WILDFIRE)

        disaster = section['disaster_specific']
        assert len(disaster) > 0

    def test_section_has_road_info(self, resources):
        """Test that section includes road closure URL."""
        section = resources.generate_resources_section("FL", DisasterType.HURRICANE)

        url = section['road_info_url']
        assert url
        assert url.startswith("http")

    def test_section_has_actions(self, resources):
        """Test that section includes what to do actions."""
        section = resources.generate_resources_section("FL", DisasterType.FLOOD)

        actions = section['what_to_do']
        assert len(actions) > 0
        assert all(isinstance(a, str) for a in actions)

    def test_section_with_county_name(self, resources):
        """Test that county name is accepted (even if not used yet)."""
        section = resources.generate_resources_section(
            state_abbrev="FL",
            disaster_type=DisasterType.HURRICANE,
            county_name="Miami-Dade"
        )

        # Should still generate complete section
        assert all(k in section for k in ['national', 'state', 'disaster_specific'])


class TestDataCompleteness:
    """Tests for data completeness and accuracy."""

    def test_all_disaster_types_have_resources(self, resources):
        """Test that all disaster types have specific resources."""
        for disaster_type in DisasterType:
            disaster_resources = resources.get_disaster_specific_resources(disaster_type)
            assert len(disaster_resources) > 0, f"No resources for {disaster_type.value}"

    def test_all_disaster_types_have_actions(self, resources):
        """Test that all disaster types have action items."""
        for disaster_type in DisasterType:
            section = resources.generate_resources_section("FL", disaster_type)
            actions = section['what_to_do']
            assert len(actions) > 0, f"No actions for {disaster_type.value}"

    def test_all_contacts_have_descriptions(self, resources):
        """Test that all contacts have descriptions."""
        national = resources.get_national_resources()

        for contact in national:
            assert contact.description
            assert len(contact.description) > 10  # Non-trivial description

    def test_phone_numbers_formatted_correctly(self, resources):
        """Test that phone numbers follow expected format."""
        national = resources.get_national_resources()

        for contact in national:
            if contact.phone:
                # Should be formatted like 1-800-xxx-xxxx or 988
                assert contact.phone.replace("-", "").replace(" ", "").isdigit() or \
                       contact.phone == "988"

    def test_urls_are_https(self, resources):
        """Test that URLs use HTTPS where possible."""
        national = resources.get_national_resources()

        for contact in national:
            if contact.url:
                # Most should be HTTPS
                assert contact.url.startswith("http")


class TestDisasterTypeEnum:
    """Tests for DisasterType enum."""

    def test_disaster_types_defined(self):
        """Test that expected disaster types are defined."""
        expected = ["FLOOD", "WILDFIRE", "HURRICANE", "TORNADO", "EARTHQUAKE"]

        for disaster in expected:
            assert hasattr(DisasterType, disaster)

    def test_disaster_type_values(self):
        """Test disaster type string values."""
        assert DisasterType.FLOOD.value == "flood"
        assert DisasterType.WILDFIRE.value == "wildfire"
        assert DisasterType.HURRICANE.value == "hurricane"
        assert DisasterType.TORNADO.value == "tornado"
        assert DisasterType.EARTHQUAKE.value == "earthquake"
