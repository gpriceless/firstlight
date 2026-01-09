"""Tests for data provider registry and loader."""

import pytest

from core.data.providers.loader import create_default_registry, load_all_providers
from core.data.providers.registry import Provider, ProviderRegistry


class TestProviderRegistry:
    """Test provider registry functionality."""

    @pytest.fixture
    def empty_registry(self):
        """Create an empty registry for testing."""
        return ProviderRegistry()

    @pytest.fixture
    def populated_registry(self):
        """Create a registry populated with all providers."""
        return create_default_registry()

    def test_registry_initialization(self, empty_registry):
        """Test that registry initializes properly."""
        assert isinstance(empty_registry, ProviderRegistry)
        assert isinstance(empty_registry.providers, dict)

    def test_register_provider(self, empty_registry):
        """Test registering a provider."""
        provider = Provider(
            id="test_provider",
            provider="test_org",
            type="optical",
            capabilities={"resolution_m": 10},
            applicability={"event_classes": ["flood.*"]},
        )
        empty_registry.register(provider)

        assert "test_provider" in empty_registry.providers
        retrieved = empty_registry.get_provider("test_provider")
        assert retrieved is not None
        assert retrieved.id == "test_provider"

    def test_get_nonexistent_provider(self, empty_registry):
        """Test getting a provider that doesn't exist."""
        result = empty_registry.get_provider("nonexistent")
        assert result is None

    def test_get_all_providers(self, populated_registry):
        """Test getting all providers."""
        all_providers = populated_registry.get_all_providers()
        assert len(all_providers) > 0
        assert all(isinstance(p, Provider) for p in all_providers)

    def test_provider_has_required_fields(self, populated_registry):
        """Test that all providers have required fields."""
        all_providers = populated_registry.get_all_providers()
        for provider in all_providers:
            assert provider.id
            assert provider.provider
            assert provider.type in ["optical", "sar", "dem", "weather", "ancillary"]
            assert isinstance(provider.capabilities, dict)
            assert isinstance(provider.access, dict)
            assert isinstance(provider.applicability, dict)
            assert isinstance(provider.cost, dict)
            assert isinstance(provider.metadata, dict)

    def test_get_providers_by_type(self, populated_registry):
        """Test filtering providers by type."""
        optical_providers = populated_registry.get_providers_by_type("optical")
        assert len(optical_providers) > 0
        assert all(p.type == "optical" for p in optical_providers)

        sar_providers = populated_registry.get_providers_by_type("sar")
        assert len(sar_providers) > 0
        assert all(p.type == "sar" for p in sar_providers)

        weather_providers = populated_registry.get_providers_by_type("weather")
        assert len(weather_providers) > 0
        assert all(p.type == "weather" for p in weather_providers)

    def test_provider_cost_tiers(self, populated_registry):
        """Test that providers have valid cost tiers."""
        all_providers = populated_registry.get_all_providers()
        valid_tiers = {"open", "open_restricted", "commercial"}

        for provider in all_providers:
            tier = provider.cost.get("tier")
            assert tier in valid_tiers, f"Invalid tier '{tier}' for {provider.id}"

    def test_provider_preference_scores(self, populated_registry):
        """Test that providers have preference scores."""
        all_providers = populated_registry.get_all_providers()

        for provider in all_providers:
            score = provider.metadata.get("preference_score")
            if score is not None:
                assert 0 <= score <= 1, f"Invalid preference score {score} for {provider.id}"


class TestProviderApplicability:
    """Test provider applicability matching."""

    @pytest.fixture
    def registry(self):
        """Get populated registry."""
        return create_default_registry()

    def test_matches_event_class_exact(self):
        """Test exact event class matching."""
        provider = Provider(
            id="test",
            provider="test",
            type="optical",
            applicability={"event_classes": ["flood.coastal"]},
        )

        assert provider.matches_event_class("flood.coastal")
        assert not provider.matches_event_class("flood.riverine")

    def test_matches_event_class_wildcard(self):
        """Test wildcard event class matching."""
        provider = Provider(
            id="test",
            provider="test",
            type="optical",
            applicability={"event_classes": ["flood.*"]},
        )

        assert provider.matches_event_class("flood.coastal")
        assert provider.matches_event_class("flood.riverine")
        assert provider.matches_event_class("flood.urban")
        assert not provider.matches_event_class("wildfire.forest")

    def test_matches_multiple_patterns(self):
        """Test matching with multiple event class patterns."""
        provider = Provider(
            id="test",
            provider="test",
            type="optical",
            applicability={"event_classes": ["flood.*", "storm.*"]},
        )

        assert provider.matches_event_class("flood.coastal")
        assert provider.matches_event_class("storm.tropical_cyclone")
        assert not provider.matches_event_class("wildfire.forest")

    def test_get_applicable_providers_for_flood(self, registry):
        """Test getting providers applicable to flood events."""
        applicable = registry.get_applicable_providers("flood.coastal")
        assert len(applicable) > 0

        # Verify all returned providers match the event class
        for provider in applicable:
            assert provider.matches_event_class("flood.coastal")

    def test_get_applicable_providers_for_wildfire(self, registry):
        """Test getting providers applicable to wildfire events."""
        applicable = registry.get_applicable_providers("wildfire.forest")
        assert len(applicable) > 0

        for provider in applicable:
            assert provider.matches_event_class("wildfire.forest")

    def test_get_applicable_providers_with_data_type_filter(self, registry):
        """Test filtering applicable providers by data type."""
        # Get only SAR providers for flood events
        applicable = registry.get_applicable_providers("flood.coastal", data_types=["sar"])
        assert len(applicable) > 0
        assert all(p.type == "sar" for p in applicable)

    def test_preference_ordering(self, registry):
        """Test that providers are ordered by preference."""
        applicable = registry.get_applicable_providers("flood.*")

        if len(applicable) > 1:
            # Check that open providers come before commercial
            tiers = [p.cost.get("tier", "open") for p in applicable]

            # Find indices of open and commercial providers
            open_indices = [i for i, tier in enumerate(tiers) if tier == "open"]
            commercial_indices = [i for i, tier in enumerate(tiers) if tier == "commercial"]

            # If both exist, open should come before commercial
            if open_indices and commercial_indices:
                assert max(open_indices) < min(commercial_indices)


class TestSpecificProviders:
    """Test specific provider implementations."""

    @pytest.fixture
    def registry(self):
        """Get populated registry."""
        return create_default_registry()

    def test_sentinel2_provider(self, registry):
        """Test Sentinel-2 provider configuration."""
        s2 = registry.get_provider("sentinel2_l2a")
        assert s2 is not None
        assert s2.type == "optical"
        assert s2.provider == "copernicus"
        assert s2.cost.get("tier") == "open"
        assert "sentinel-2-l2a" in s2.metadata.get("stac_collection", "")

    def test_sentinel1_providers(self, registry):
        """Test Sentinel-1 SAR provider configurations."""
        s1_grd = registry.get_provider("sentinel1_grd")
        assert s1_grd is not None
        assert s1_grd.type == "sar"

        s1_slc = registry.get_provider("sentinel1_slc")
        assert s1_slc is not None
        assert s1_slc.type == "sar"

    def test_landsat_providers(self, registry):
        """Test Landsat provider configurations."""
        l8 = registry.get_provider("landsat8_c2l2")
        assert l8 is not None
        assert l8.type == "optical"
        assert l8.cost.get("tier") == "open"

        l9 = registry.get_provider("landsat9_c2l2")
        assert l9 is not None
        assert l9.type == "optical"

    def test_weather_providers(self, registry):
        """Test weather data provider configurations."""
        era5 = registry.get_provider("era5")
        assert era5 is not None
        assert era5.type == "weather"
        assert "precipitation" in era5.capabilities.get("variables", [])

        gfs = registry.get_provider("gfs")
        assert gfs is not None
        assert gfs.type == "weather"

    def test_dem_providers(self, registry):
        """Test DEM provider configurations."""
        cop_dem_30 = registry.get_provider("copernicus_dem_30")
        assert cop_dem_30 is not None
        assert cop_dem_30.type == "dem"

        srtm_30 = registry.get_provider("srtm_30")
        assert srtm_30 is not None
        assert srtm_30.type == "dem"

    def test_ancillary_providers(self, registry):
        """Test ancillary data provider configurations."""
        osm = registry.get_provider("osm")
        assert osm is not None
        assert osm.type == "ancillary"

        wsf = registry.get_provider("wsf_2019")
        assert wsf is not None
        assert wsf.type == "ancillary"


class TestProviderCapabilities:
    """Test provider capability metadata."""

    @pytest.fixture
    def registry(self):
        """Get populated registry."""
        return create_default_registry()

    def test_optical_providers_have_resolution(self, registry):
        """Test that optical providers specify resolution."""
        optical = registry.get_providers_by_type("optical")
        for provider in optical:
            assert "resolution_m" in provider.capabilities
            assert provider.capabilities["resolution_m"] > 0

    def test_sar_providers_have_polarizations(self, registry):
        """Test that SAR providers specify polarizations."""
        sar = registry.get_providers_by_type("sar")
        for provider in sar:
            assert "polarizations" in provider.capabilities
            assert len(provider.capabilities["polarizations"]) > 0

    def test_weather_providers_have_variables(self, registry):
        """Test that weather providers specify variables."""
        weather = registry.get_providers_by_type("weather")
        for provider in weather:
            assert "variables" in provider.capabilities
            assert len(provider.capabilities["variables"]) > 0

    def test_providers_have_temporal_coverage(self, registry):
        """Test that providers specify temporal coverage."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            if "temporal_coverage" in provider.capabilities:
                coverage = provider.capabilities["temporal_coverage"]
                assert "start" in coverage
                # end can be None for ongoing missions

    def test_providers_have_access_config(self, registry):
        """Test that providers have access configuration."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            assert "protocol" in provider.access
            assert provider.access["protocol"] in [
                "stac", "api", "wms", "wcs", "s3", "https", "http"
            ]


class TestProviderLoader:
    """Test provider loader functionality."""

    def test_load_all_providers(self):
        """Test loading all providers."""
        registry = ProviderRegistry()
        load_all_providers(registry)

        # Verify providers were loaded
        assert len(registry.get_all_providers()) > 0

    def test_create_default_registry(self):
        """Test creating default registry."""
        registry = create_default_registry()

        assert isinstance(registry, ProviderRegistry)
        assert len(registry.get_all_providers()) > 0

        # Verify we have providers of each type
        assert len(registry.get_providers_by_type("optical")) > 0
        assert len(registry.get_providers_by_type("sar")) > 0
        assert len(registry.get_providers_by_type("dem")) > 0
        assert len(registry.get_providers_by_type("weather")) > 0
        assert len(registry.get_providers_by_type("ancillary")) > 0

    def test_no_duplicate_provider_ids(self):
        """Test that there are no duplicate provider IDs."""
        registry = create_default_registry()
        all_providers = registry.get_all_providers()

        provider_ids = [p.id for p in all_providers]
        assert len(provider_ids) == len(set(provider_ids)), "Duplicate provider IDs found"


class TestProviderMetadata:
    """Test provider metadata completeness."""

    @pytest.fixture
    def registry(self):
        """Get populated registry."""
        return create_default_registry()

    def test_providers_have_names(self, registry):
        """Test that all providers have names."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            assert "name" in provider.metadata
            assert provider.metadata["name"]

    def test_providers_have_descriptions(self, registry):
        """Test that all providers have descriptions."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            assert "description" in provider.metadata
            assert provider.metadata["description"]

    def test_providers_have_licenses(self, registry):
        """Test that all providers specify licenses."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            assert "license" in provider.metadata
            assert provider.metadata["license"]

    def test_open_providers_are_documented(self, registry):
        """Test that open providers have documentation URLs."""
        all_providers = registry.get_all_providers()
        for provider in all_providers:
            if provider.cost.get("tier") == "open":
                # Open providers should have documentation
                assert "documentation_url" in provider.metadata
