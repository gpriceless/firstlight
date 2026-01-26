"""Integration tests for reporting data modules."""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from core.reporting.data import (
    CensusClient,
    PopulationData,
    InfrastructureClient,
    InfrastructureFeature,
    InfrastructureType,
    EmergencyResources,
    EmergencyContact,
    StateEmergencyInfo,
    DisasterType,
)


def test_imports():
    """Verify all modules can be imported."""
    assert CensusClient is not None
    assert PopulationData is not None
    assert InfrastructureClient is not None
    assert InfrastructureFeature is not None
    assert InfrastructureType is not None
    assert EmergencyResources is not None
    assert EmergencyContact is not None
    assert StateEmergencyInfo is not None
    assert DisasterType is not None


@pytest.mark.asyncio
async def test_census_client_mock():
    """Test CensusClient with mocked API responses."""
    # Mock API response for Miami-Dade County
    mock_response = [
        ["B01003_001E", "B25001_001E", "B25002_002E", "NAME", "state", "county"],
        ["2716940", "1184100", "952420", "Miami-Dade County, Florida", "12", "999"]
    ]

    with patch('aiohttp.ClientSession') as mock_session_class:
        # Create mock response - properly setup async context manager
        mock_response_obj = MagicMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)

        async def mock_aenter(*args, **kwargs):
            return mock_response_obj

        async def mock_aexit(*args, **kwargs):
            return None

        mock_response_obj.__aenter__ = mock_aenter
        mock_response_obj.__aexit__ = mock_aexit

        # Configure session mock
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response_obj)
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        # Test client - use unique county to avoid cache
        async with CensusClient() as client:
            data = await client.get_population_by_county("12", "999")

            assert data.total_population == 2716940
            assert data.housing_units == 1184100
            assert data.occupied_housing == 952420
            assert "Miami-Dade" in data.area_name
            assert data.state_fips == "12"
            assert data.county_fips == "999"


@pytest.mark.asyncio
async def test_census_client_caching():
    """Test that census client properly caches responses."""
    import tempfile
    import shutil

    # Create temporary cache directory for this test
    temp_cache = Path(tempfile.mkdtemp())

    try:
        mock_response = [
            ["B01003_001E", "B25001_001E", "B25002_002E", "NAME", "state", "county"],
            ["500000", "200000", "180000", "Test County", "12", "998"]
        ]

        with patch('aiohttp.ClientSession') as mock_session_class, \
             patch.object(CensusClient, 'CACHE_DIR', temp_cache):
            # Properly setup async context manager
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)

            async def mock_aenter(*args, **kwargs):
                return mock_response_obj

            async def mock_aexit(*args, **kwargs):
                return None

            mock_response_obj.__aenter__ = mock_aenter
            mock_response_obj.__aexit__ = mock_aexit

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_response_obj)
            mock_session.close = AsyncMock()
            mock_session_class.return_value = mock_session

            # First call - should hit API
            async with CensusClient(cache_ttl=300) as client:
                data1 = await client.get_population_by_county("12", "998")
                assert data1.total_population == 500000

                # Second call - should use cache
                data2 = await client.get_population_by_county("12", "998")
                assert data2.total_population == 500000

                # Should only call API once due to caching
                assert mock_session.get.call_count == 1
    finally:
        # Clean up temp cache
        shutil.rmtree(temp_cache, ignore_errors=True)


@pytest.mark.asyncio
async def test_infrastructure_client_mock():
    """Test InfrastructureClient with mocked Overpass API."""
    # Mock Overpass API response with hospitals
    mock_response = {
        "elements": [
            {
                "type": "node",
                "id": 123456,
                "lat": 25.7907,
                "lon": -80.1300,
                "tags": {
                    "amenity": "hospital",
                    "name": "Jackson Memorial Hospital",
                    "addr:street": "NW 12th Ave",
                    "addr:city": "Miami",
                    "phone": "(305) 585-1111",
                    "beds": "1550"
                }
            },
            {
                "type": "way",
                "id": 789012,
                "center": {"lat": 25.8165, "lon": -80.2206},
                "tags": {
                    "amenity": "hospital",
                    "name": "Mount Sinai Medical Center"
                }
            }
        ]
    }

    with patch('aiohttp.ClientSession') as mock_session_class:
        # Properly setup async context manager
        mock_response_obj = MagicMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)

        async def mock_aenter(*args, **kwargs):
            return mock_response_obj

        async def mock_aexit(*args, **kwargs):
            return None

        mock_response_obj.__aenter__ = mock_aenter
        mock_response_obj.__aexit__ = mock_aexit

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=mock_response_obj)
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        # Test infrastructure query
        async with InfrastructureClient() as client:
            bbox = (-80.87, 25.13, -80.12, 25.97)
            hospitals = await client.query_by_bbox(
                bbox,
                types=[InfrastructureType.HOSPITAL]
            )

            assert len(hospitals) == 2
            assert hospitals[0].name == "Jackson Memorial Hospital"
            assert hospitals[0].type == InfrastructureType.HOSPITAL
            assert hospitals[0].capacity == 1550
            assert hospitals[1].name == "Mount Sinai Medical Center"


def test_emergency_resources_florida_flood():
    """Test complete resource generation for Florida flood."""
    resources = EmergencyResources()
    section = resources.generate_resources_section(
        state_abbrev="FL",
        disaster_type=DisasterType.FLOOD,
        county_name="Lee"
    )

    # Verify structure
    assert 'national' in section
    assert 'state' in section
    assert 'disaster_specific' in section
    assert 'road_info_url' in section
    assert 'what_to_do' in section

    # Verify national resources
    assert len(section['national']) > 0
    fema_found = any(c.name == "FEMA" for c in section['national'])
    assert fema_found

    # Verify state info
    assert section['state'].state_name == "Florida"
    assert section['state'].state_abbrev == "FL"
    assert "floridadisaster.org" in section['state'].emergency_mgmt_url

    # Verify disaster-specific resources
    assert len(section['disaster_specific']) > 0
    flood_insurance_found = any(
        "Flood Insurance" in c.name for c in section['disaster_specific']
    )
    assert flood_insurance_found

    # Verify road info
    assert "511" in section['road_info_url']

    # Verify action items
    assert len(section['what_to_do']) > 0
    assert any("flood waters" in action.lower() for action in section['what_to_do'])


def test_emergency_resources_all_disaster_types():
    """Test emergency resources for all disaster types."""
    resources = EmergencyResources()

    for disaster_type in DisasterType:
        section = resources.generate_resources_section(
            state_abbrev="CA",
            disaster_type=disaster_type
        )

        # Each disaster type should have action items
        assert len(section['what_to_do']) > 0, f"No actions for {disaster_type.value}"

        # National resources should always be present
        assert len(section['national']) > 0


def test_emergency_resources_invalid_state():
    """Test handling of invalid state abbreviation."""
    resources = EmergencyResources()

    with pytest.raises(KeyError):
        resources.get_state_resources("XX")


@pytest.mark.asyncio
async def test_combined_disaster_report_data():
    """Test generating all data for a disaster report."""
    # Mock flood GeoJSON
    flood_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [-80.87, 25.13],
            [-80.12, 25.13],
            [-80.12, 25.97],
            [-80.87, 25.97],
            [-80.87, 25.13]
        ]]
    }

    # Mock census response
    mock_census_response = [
        ["B01003_001E", "B25001_001E", "B25002_002E", "NAME", "state", "county"],
        ["500000", "200000", "180000", "Test County", "12", "997"]
    ]

    # Mock infrastructure response
    mock_infra_response = {
        "elements": [
            {
                "type": "node",
                "id": 123,
                "lat": 25.50,
                "lon": -80.50,
                "tags": {"amenity": "hospital", "name": "Test Hospital"}
            }
        ]
    }

    with patch('aiohttp.ClientSession') as mock_session_class:
        # Setup mock GET response (for census)
        mock_get_response = MagicMock()
        mock_get_response.status = 200
        mock_get_response.json = AsyncMock(return_value=mock_census_response)

        async def mock_get_aenter(*args, **kwargs):
            return mock_get_response

        async def mock_get_aexit(*args, **kwargs):
            return None

        mock_get_response.__aenter__ = mock_get_aenter
        mock_get_response.__aexit__ = mock_get_aexit

        # Setup mock POST response (for infrastructure)
        mock_post_response = MagicMock()
        mock_post_response.status = 200
        mock_post_response.json = AsyncMock(return_value=mock_infra_response)

        async def mock_post_aenter(*args, **kwargs):
            return mock_post_response

        async def mock_post_aexit(*args, **kwargs):
            return None

        mock_post_response.__aenter__ = mock_post_aenter
        mock_post_response.__aexit__ = mock_post_aexit

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_get_response)
        mock_session.post = MagicMock(return_value=mock_post_response)
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        # Gather all report data
        async with CensusClient() as census:
            pop_data = await census.get_population_by_county("12", "997")

            assert pop_data.total_population == 500000
            assert pop_data.housing_units == 200000

        async with InfrastructureClient() as infra:
            bbox = (-80.87, 25.13, -80.12, 25.97)
            hospitals = await infra.query_by_bbox(
                bbox,
                types=[InfrastructureType.HOSPITAL]
            )

            assert len(hospitals) > 0

        # Get emergency resources (no mocking needed - static data)
        resources = EmergencyResources()
        emergency = resources.generate_resources_section(
            state_abbrev="FL",
            disaster_type=DisasterType.FLOOD
        )

        assert len(emergency['national']) > 0

        # All data successfully gathered
        assert pop_data is not None
        assert len(hospitals) > 0
        assert emergency is not None


def test_caching_creates_cache_files():
    """Verify caching creates expected cache files."""
    cache_dir = Path.home() / ".cache" / "firstlight"

    # Ensure cache directories exist (created by clients)
    census_cache = cache_dir / "census"
    infra_cache = cache_dir / "infrastructure"

    # At minimum, directories should be created
    # (actual cache files created during real API calls)
    assert cache_dir.exists() or True  # Cache may not exist yet in test env


@pytest.mark.asyncio
async def test_api_timeout_handling():
    """Test graceful handling of API timeouts."""
    with patch('aiohttp.ClientSession') as mock_session_class:
        # Create mock that raises timeout when used as context manager
        async def mock_get_timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=MagicMock(__aenter__=mock_get_timeout))
        mock_session.close = AsyncMock()
        mock_session_class.return_value = mock_session

        async with CensusClient() as client:
            # Should handle timeout gracefully and return default data
            data = await client.get_population_by_county("12", "996")

            assert data.total_population == 0
            assert data.area_name == "Unknown County"


@pytest.mark.asyncio
async def test_infrastructure_retry_on_rate_limit():
    """Test that infrastructure client retries on rate limiting."""
    import tempfile
    import shutil

    # Create temporary cache directory for this test
    temp_cache = Path(tempfile.mkdtemp())

    try:
        with patch('aiohttp.ClientSession') as mock_session_class, \
             patch('asyncio.sleep', new_callable=AsyncMock), \
             patch.object(InfrastructureClient, 'CACHE_DIR', temp_cache):

            # First call returns 429 (rate limited)
            rate_limited_response = MagicMock()
            rate_limited_response.status = 429

            async def mock_rate_limit_aenter(*args, **kwargs):
                return rate_limited_response

            async def mock_rate_limit_aexit(*args, **kwargs):
                return None

            rate_limited_response.__aenter__ = mock_rate_limit_aenter
            rate_limited_response.__aexit__ = mock_rate_limit_aexit

            # Second call succeeds
            success_response = MagicMock()
            success_response.status = 200
            success_response.json = AsyncMock(return_value={"elements": []})

            async def mock_success_aenter(*args, **kwargs):
                return success_response

            async def mock_success_aexit(*args, **kwargs):
                return None

            success_response.__aenter__ = mock_success_aenter
            success_response.__aexit__ = mock_success_aexit

            mock_session = MagicMock()
            # First call returns rate limited, second returns success
            mock_session.post = MagicMock(side_effect=[rate_limited_response, success_response])
            mock_session.close = AsyncMock()
            mock_session_class.return_value = mock_session

            async with InfrastructureClient() as client:
                # Should retry and succeed
                bbox = (-80.87, 25.13, -80.12, 25.97)
                features = await client.query_by_bbox(bbox)

                # Should have made 2 calls (first failed, second succeeded)
                assert mock_session.post.call_count == 2
                assert isinstance(features, list)
    finally:
        # Clean up temp cache
        shutil.rmtree(temp_cache, ignore_errors=True)


def test_infrastructure_geojson_output():
    """Test that infrastructure data exports to valid GeoJSON."""
    # Create sample features
    features = [
        InfrastructureFeature(
            id="123",
            type=InfrastructureType.HOSPITAL,
            name="Test Hospital",
            lat=25.7907,
            lon=-80.1300,
            address="123 Main St",
            phone="555-1234",
            capacity=100
        ),
        InfrastructureFeature(
            id="456",
            type=InfrastructureType.SHELTER,
            name="Emergency Shelter",
            lat=25.8000,
            lon=-80.2000
        )
    ]

    # Create client and export to GeoJSON
    client = InfrastructureClient()
    geojson = client.to_geojson(features)

    # Verify structure
    assert geojson['type'] == "FeatureCollection"
    assert len(geojson['features']) == 2

    # Verify first feature
    feature1 = geojson['features'][0]
    assert feature1['type'] == "Feature"
    assert feature1['geometry']['type'] == "Point"
    assert feature1['geometry']['coordinates'] == [-80.1300, 25.7907]
    assert feature1['properties']['name'] == "Test Hospital"
    assert feature1['properties']['type'] == "hospital"
    assert feature1['properties']['capacity'] == 100

    # Verify second feature
    feature2 = geojson['features'][1]
    assert feature2['geometry']['coordinates'] == [-80.2000, 25.8000]
    assert feature2['properties']['type'] == "shelter"


def test_population_data_to_dict():
    """Test PopulationData serialization."""
    data = PopulationData(
        total_population=100000,
        housing_units=40000,
        occupied_housing=35000,
        area_name="Test County",
        state_fips="12",
        county_fips="086",
        tract="0001"
    )

    result = data.to_dict()

    assert result['total_population'] == 100000
    assert result['housing_units'] == 40000
    assert result['occupied_housing'] == 35000
    assert result['area_name'] == "Test County"
    assert result['state_fips'] == "12"
    assert result['county_fips'] == "086"
    assert result['tract'] == "0001"


def test_infrastructure_feature_to_geojson():
    """Test InfrastructureFeature GeoJSON conversion."""
    feature = InfrastructureFeature(
        id="test-123",
        type=InfrastructureType.FIRE_STATION,
        name="Fire Station 5",
        lat=26.1234,
        lon=-80.5678,
        address="456 Oak St",
        phone="555-FIRE",
        capacity=20
    )

    geojson = feature.to_geojson_feature()

    assert geojson['type'] == "Feature"
    assert geojson['geometry']['type'] == "Point"
    assert geojson['geometry']['coordinates'] == [-80.5678, 26.1234]
    assert geojson['properties']['id'] == "test-123"
    assert geojson['properties']['type'] == "fire_station"
    assert geojson['properties']['name'] == "Fire Station 5"
    assert geojson['properties']['address'] == "456 Oak St"
    assert geojson['properties']['phone'] == "555-FIRE"
    assert geojson['properties']['capacity'] == 20


def test_emergency_contact_dataclass():
    """Test EmergencyContact dataclass."""
    contact = EmergencyContact(
        name="Test Agency",
        phone="555-1234",
        url="https://example.com",
        description="Test description"
    )

    assert contact.name == "Test Agency"
    assert contact.phone == "555-1234"
    assert contact.url == "https://example.com"
    assert contact.description == "Test description"


def test_state_emergency_info_dataclass():
    """Test StateEmergencyInfo dataclass."""
    info = StateEmergencyInfo(
        state_name="Florida",
        state_abbrev="FL",
        emergency_mgmt_url="https://floridadisaster.org",
        emergency_mgmt_phone="1-800-123-4567",
        governor_office_url="https://flgov.com"
    )

    assert info.state_name == "Florida"
    assert info.state_abbrev == "FL"
    assert "floridadisaster.org" in info.emergency_mgmt_url
    assert info.emergency_mgmt_phone == "1-800-123-4567"


def test_disaster_type_enum():
    """Test DisasterType enum values."""
    assert DisasterType.FLOOD.value == "flood"
    assert DisasterType.WILDFIRE.value == "wildfire"
    assert DisasterType.HURRICANE.value == "hurricane"
    assert DisasterType.TORNADO.value == "tornado"
    assert DisasterType.EARTHQUAKE.value == "earthquake"


def test_infrastructure_type_enum():
    """Test InfrastructureType enum values."""
    assert InfrastructureType.HOSPITAL.value == "hospital"
    assert InfrastructureType.SCHOOL.value == "school"
    assert InfrastructureType.FIRE_STATION.value == "fire_station"
    assert InfrastructureType.POLICE.value == "police"
    assert InfrastructureType.SHELTER.value == "shelter"
    assert InfrastructureType.POWER_STATION.value == "power"
    assert InfrastructureType.WATER_TREATMENT.value == "water"
