"""
Unit tests for population impact metrics (VIS-1.4 Task 3).

Tests cover:
- FacilityCount dataclass and methods
- PopulationImpact dataclass and methods
- calculate_population_impact() with mocked clients
- format_impact_summary() language guidelines
- Edge cases (zero values, missing clients)
"""

import numpy as np
import pytest

from core.reporting.imagery.impact_metrics import (
    FacilityCount,
    PopulationImpact,
    calculate_population_impact,
    format_impact_summary,
    _estimate_residential_structures,
    _calculate_road_length_km,
)


# === Fixtures ===


@pytest.fixture
def sample_bounds():
    """Sample bounding box (Miami area)."""
    return (-80.87, 25.13, -80.12, 25.97)


@pytest.fixture
def detection_mask():
    """Sample detection mask."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[25:75, 25:75] = True
    return mask


@pytest.fixture
def mock_census_client():
    """Mock census client with test data."""
    class MockCensusClient:
        async def get_population_by_bbox(self, bbox):
            # Simulate returning population data
            from core.reporting.data.census_client import PopulationData
            return [
                PopulationData(
                    total_population=42000,
                    housing_units=18000,
                    occupied_housing=16500,
                    area_name="Test County",
                    state_fips="12",
                    county_fips="086"
                )
            ]
    return MockCensusClient()


@pytest.fixture
def mock_infrastructure_client():
    """Mock infrastructure client with test data."""
    class MockInfrastructureClient:
        async def find_in_flood_zone(self, flood_geojson, types=None):
            # Simulate returning facility data
            from core.reporting.data.infrastructure_client import (
                InfrastructureFeature,
                InfrastructureType
            )
            return {
                "in_flood_zone": [
                    InfrastructureFeature(
                        id="1",
                        type=InfrastructureType.HOSPITAL,
                        name="Test Hospital",
                        lat=25.5,
                        lon=-80.5
                    ),
                    InfrastructureFeature(
                        id="2",
                        type=InfrastructureType.SCHOOL,
                        name="Test School 1",
                        lat=25.6,
                        lon=-80.6
                    ),
                    InfrastructureFeature(
                        id="3",
                        type=InfrastructureType.SCHOOL,
                        name="Test School 2",
                        lat=25.7,
                        lon=-80.7
                    ),
                ],
                "nearby": [],
                "by_type": {
                    "hospital": 1,
                    "school": 2
                }
            }
    return MockInfrastructureClient()


# === FacilityCount Tests ===


def test_facility_count_defaults():
    """Test FacilityCount default values."""
    facilities = FacilityCount()
    assert facilities.hospitals == 0
    assert facilities.schools == 0
    assert facilities.fire_stations == 0
    assert facilities.police_stations == 0
    assert facilities.shelters == 0
    assert facilities.total() == 0


def test_facility_count_custom():
    """Test FacilityCount with custom values."""
    facilities = FacilityCount(
        hospitals=2,
        schools=5,
        fire_stations=1,
        police_stations=1,
        shelters=3
    )
    assert facilities.hospitals == 2
    assert facilities.schools == 5
    assert facilities.fire_stations == 1
    assert facilities.police_stations == 1
    assert facilities.shelters == 3
    assert facilities.total() == 12


def test_facility_count_total():
    """Test total() calculation."""
    facilities = FacilityCount(hospitals=1, schools=3)
    assert facilities.total() == 4

    facilities = FacilityCount()
    assert facilities.total() == 0


def test_facility_count_to_dict():
    """Test to_dict() conversion."""
    facilities = FacilityCount(hospitals=1, schools=2)
    data = facilities.to_dict()

    assert data["hospitals"] == 1
    assert data["schools"] == 2
    assert data["fire_stations"] == 0
    assert data["police_stations"] == 0
    assert data["shelters"] == 0
    assert data["total"] == 3


# === PopulationImpact Tests ===


def test_population_impact_creation():
    """Test PopulationImpact creation."""
    facilities = FacilityCount(hospitals=1, schools=2)
    impact = PopulationImpact(
        estimated_population=5000,
        residential_structures=2000,
        facilities=facilities,
        road_km_affected=15.5
    )

    assert impact.estimated_population == 5000
    assert impact.residential_structures == 2000
    assert impact.facilities.hospitals == 1
    assert impact.facilities.schools == 2
    assert impact.road_km_affected == 15.5


def test_population_impact_to_dict():
    """Test to_dict() conversion."""
    facilities = FacilityCount(hospitals=1)
    impact = PopulationImpact(
        estimated_population=1000,
        residential_structures=400,
        facilities=facilities,
        road_km_affected=5.2
    )

    data = impact.to_dict()
    assert data["estimated_population"] == 1000
    assert data["residential_structures"] == 400
    assert data["facilities"]["hospitals"] == 1
    assert data["facilities"]["total"] == 1
    assert data["road_km_affected"] == 5.2


# === calculate_population_impact Tests ===


def test_calculate_population_impact_no_clients(sample_bounds):
    """Test with no clients provided - should return zero impact."""
    impact = calculate_population_impact(sample_bounds)

    assert impact.estimated_population == 0
    assert impact.residential_structures == 0
    assert impact.facilities.total() == 0
    assert impact.road_km_affected == 0.0


def test_calculate_population_impact_with_mask(sample_bounds, detection_mask):
    """Test with detection mask (currently not used)."""
    impact = calculate_population_impact(
        sample_bounds,
        detection_mask=detection_mask
    )

    # Should still return zero without clients
    assert impact.estimated_population == 0


# === format_impact_summary Tests ===


def test_format_impact_summary_zero_impact():
    """Test formatting with zero values."""
    facilities = FacilityCount()
    impact = PopulationImpact(
        estimated_population=0,
        residential_structures=0,
        facilities=facilities,
        road_km_affected=0.0
    )

    summary = format_impact_summary(impact)
    assert "Not available" in summary


def test_format_impact_summary_full_impact():
    """Test formatting with all values present."""
    facilities = FacilityCount(
        hospitals=1,
        schools=3,
        fire_stations=1,
        police_stations=1,
        shelters=2
    )
    impact = PopulationImpact(
        estimated_population=4200,
        residential_structures=1800,
        facilities=facilities,
        road_km_affected=12.5
    )

    summary = format_impact_summary(impact)

    # Check for factual language (no hyperbole)
    assert "Estimated affected population: 4,200" in summary
    assert "1,800" in summary
    assert "residential structures" in summary
    assert "1 hospital" in summary
    assert "3 schools" in summary
    assert "1 fire station" in summary
    assert "1 police station" in summary
    assert "2 shelters" in summary
    assert "12.5 km" in summary

    # Ensure NO hyperbolic language
    assert "devastating" not in summary.lower()
    assert "massive" not in summary.lower()
    assert "catastrophic" not in summary.lower()
    assert "severe" not in summary.lower()
    assert "extreme" not in summary.lower()


def test_format_impact_summary_pluralization():
    """Test proper plural handling in facility counts."""
    # Single hospital
    facilities = FacilityCount(hospitals=1)
    impact = PopulationImpact(
        estimated_population=1000,
        residential_structures=400,
        facilities=facilities,
        road_km_affected=5.0
    )
    summary = format_impact_summary(impact)
    assert "1 hospital" in summary  # Singular
    assert "hospitals" not in summary.split("1 hospital")[0]  # No plural before

    # Multiple hospitals
    facilities = FacilityCount(hospitals=2)
    impact = PopulationImpact(
        estimated_population=1000,
        residential_structures=400,
        facilities=facilities,
        road_km_affected=5.0
    )
    summary = format_impact_summary(impact)
    assert "2 hospitals" in summary  # Plural


def test_format_impact_summary_partial_facilities():
    """Test formatting with only some facility types."""
    facilities = FacilityCount(schools=2, shelters=1)
    impact = PopulationImpact(
        estimated_population=1500,
        residential_structures=600,
        facilities=facilities,
        road_km_affected=8.0
    )

    summary = format_impact_summary(impact)
    assert "2 schools" in summary
    assert "1 shelter" in summary
    assert "hospital" not in summary  # Not present


def test_format_impact_summary_professional_tone():
    """Test that language is professional and factual."""
    facilities = FacilityCount(hospitals=1, schools=2)
    impact = PopulationImpact(
        estimated_population=5000,
        residential_structures=2000,
        facilities=facilities,
        road_km_affected=10.0
    )

    summary = format_impact_summary(impact)

    # Professional phrases present
    assert "Estimated" in summary or "Approximately" in summary
    assert "detection area" in summary or "affected area" in summary

    # Unprofessional phrases absent
    forbidden_words = [
        "devastating", "massive", "catastrophic", "severe",
        "extreme", "terrible", "awful", "critical impact",
        "emergency", "crisis", "disaster"  # These describe events, not impacts
    ]
    summary_lower = summary.lower()
    for word in forbidden_words:
        assert word not in summary_lower, f"Unprofessional word '{word}' found in summary"


# === Helper Function Tests ===


def test_estimate_residential_structures():
    """Test residential structure estimation."""
    # Zero housing units
    assert _estimate_residential_structures(0) == 0

    # 1500 housing units -> ~1000 structures (1.5 units per structure)
    assert _estimate_residential_structures(1500) == 1000

    # 3000 housing units -> ~2000 structures
    assert _estimate_residential_structures(3000) == 2000


def test_calculate_road_length_km(sample_bounds):
    """Test road length estimation."""
    road_length = _calculate_road_length_km(sample_bounds)

    # Should return a positive value
    assert road_length > 0

    # Sanity check: shouldn't be absurdly large or small
    # Sample bounds (-80.87, 25.13, -80.12, 25.97) is roughly 83km x 93km = 7719 km²
    # At 2 km/km² density = ~15,438 km roads
    # Allow reasonable range
    assert road_length < 20000  # Upper bound check
    assert road_length > 5000  # Lower bound check


def test_calculate_road_length_km_small_area():
    """Test road length for very small area."""
    # 0.01 x 0.01 degree box (~1.1 km x 1.1 km)
    small_bounds = (-80.0, 25.0, -79.99, 25.01)
    road_length = _calculate_road_length_km(small_bounds)

    # Should be small but positive
    assert road_length > 0
    assert road_length < 10  # Very small area
