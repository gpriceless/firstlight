"""
Comprehensive tests for the Algorithm Registry (Group E, Track 1).

Tests cover:
- Algorithm registration and retrieval
- Event type pattern matching with wildcards
- Data availability filtering
- Resource constraint filtering
- YAML loading and parsing
- Index management
- Statistics generation
- Edge cases and error handling
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from core.analysis.library.registry import (
    AlgorithmRegistry,
    AlgorithmMetadata,
    AlgorithmCategory,
    DataType,
    ResourceRequirements,
    ValidationMetrics,
    get_global_registry,
    load_default_algorithms
)


class TestAlgorithmMetadata:
    """Test AlgorithmMetadata dataclass functionality"""

    def test_metadata_creation(self):
        """Test basic metadata creation"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.id == "test.baseline.simple"
        assert metadata.name == "Test Algorithm"
        assert metadata.category == AlgorithmCategory.BASELINE
        assert metadata.version == "1.0.0"
        assert metadata.event_types == ["flood.*"]
        assert metadata.required_data_types == [DataType.SAR]

    def test_to_dict_conversion(self):
        """Test conversion to dictionary"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            optional_data_types=[DataType.DEM]
        )

        result = metadata.to_dict()

        assert result['id'] == "test.baseline.simple"
        assert result['category'] == "baseline"
        assert result['required_data_types'] == ["sar"]
        assert result['optional_data_types'] == ["dem"]

    def test_matches_event_type_exact(self):
        """Test exact event type matching"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.coastal"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.matches_event_type("flood.coastal")
        assert not metadata.matches_event_type("flood.riverine")
        assert not metadata.matches_event_type("wildfire.forest")

    def test_matches_event_type_wildcard(self):
        """Test wildcard event type matching"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.matches_event_type("flood.coastal")
        assert metadata.matches_event_type("flood.riverine")
        assert metadata.matches_event_type("flood.urban")
        assert not metadata.matches_event_type("flood")
        assert not metadata.matches_event_type("wildfire.forest")

    def test_matches_event_type_multiple_patterns(self):
        """Test matching with multiple event type patterns"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*", "storm.*"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.matches_event_type("flood.coastal")
        assert metadata.matches_event_type("storm.hurricane")
        assert not metadata.matches_event_type("wildfire.forest")

    def test_matches_event_type_hierarchical_wildcard(self):
        """Test hierarchical wildcard matching"""
        metadata = AlgorithmMetadata(
            id="test.baseline.simple",
            name="Test Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.coastal.*"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.matches_event_type("flood.coastal.storm_surge")
        assert metadata.matches_event_type("flood.coastal.king_tide")
        assert not metadata.matches_event_type("flood.coastal")
        assert not metadata.matches_event_type("flood.riverine")


class TestAlgorithmRegistry:
    """Test AlgorithmRegistry functionality"""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test"""
        return AlgorithmRegistry()

    @pytest.fixture
    def sample_algorithm(self):
        """Create a sample algorithm for testing"""
        return AlgorithmMetadata(
            id="flood.baseline.threshold_sar",
            name="SAR Threshold",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            optional_data_types=[DataType.DEM],
            resources=ResourceRequirements(
                memory_mb=2048,
                gpu_required=False
            )
        )

    def test_register_algorithm(self, registry, sample_algorithm):
        """Test registering an algorithm"""
        registry.register(sample_algorithm)

        assert len(registry.algorithms) == 1
        assert sample_algorithm.id in registry.algorithms
        assert registry.get(sample_algorithm.id) == sample_algorithm

    def test_register_duplicate_algorithm(self, registry, sample_algorithm):
        """Test registering the same algorithm twice"""
        registry.register(sample_algorithm)
        registry.register(sample_algorithm)

        # Should only have one instance
        assert len(registry.algorithms) == 1

    def test_get_nonexistent_algorithm(self, registry):
        """Test getting an algorithm that doesn't exist"""
        assert registry.get("nonexistent.id") is None

    def test_list_all(self, registry, sample_algorithm):
        """Test listing all algorithms"""
        algorithm2 = AlgorithmMetadata(
            id="wildfire.baseline.nbr",
            name="NBR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.OPTICAL]
        )

        registry.register(sample_algorithm)
        registry.register(algorithm2)

        all_algos = registry.list_all()
        assert len(all_algos) == 2
        assert sample_algorithm in all_algos
        assert algorithm2 in all_algos

    def test_list_by_category(self, registry, sample_algorithm):
        """Test listing algorithms by category"""
        experimental_algo = AlgorithmMetadata(
            id="flood.experimental.ml",
            name="ML Flood",
            category=AlgorithmCategory.EXPERIMENTAL,
            version="0.1.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        registry.register(sample_algorithm)
        registry.register(experimental_algo)

        baseline = registry.list_by_category(AlgorithmCategory.BASELINE)
        experimental = registry.list_by_category(AlgorithmCategory.EXPERIMENTAL)

        assert len(baseline) == 1
        assert sample_algorithm in baseline
        assert len(experimental) == 1
        assert experimental_algo in experimental

    def test_search_by_event_type(self, registry):
        """Test searching algorithms by event type"""
        flood_algo = AlgorithmMetadata(
            id="flood.baseline.sar",
            name="Flood SAR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        wildfire_algo = AlgorithmMetadata(
            id="wildfire.baseline.nbr",
            name="NBR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.OPTICAL]
        )

        registry.register(flood_algo)
        registry.register(wildfire_algo)

        flood_results = registry.search_by_event_type("flood.coastal")
        assert len(flood_results) == 1
        assert flood_algo in flood_results

        wildfire_results = registry.search_by_event_type("wildfire.forest")
        assert len(wildfire_results) == 1
        assert wildfire_algo in wildfire_results

        storm_results = registry.search_by_event_type("storm.hurricane")
        assert len(storm_results) == 0

    def test_search_by_data_availability(self, registry):
        """Test searching algorithms by available data types"""
        sar_algo = AlgorithmMetadata(
            id="flood.baseline.sar",
            name="SAR Only",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        optical_algo = AlgorithmMetadata(
            id="flood.baseline.optical",
            name="Optical Only",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.OPTICAL]
        )

        multi_algo = AlgorithmMetadata(
            id="flood.baseline.multi",
            name="Multi Sensor",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR, DataType.DEM]
        )

        registry.register(sar_algo)
        registry.register(optical_algo)
        registry.register(multi_algo)

        # Only SAR available
        results = registry.search_by_data_availability([DataType.SAR])
        assert len(results) == 1
        assert sar_algo in results

        # SAR and DEM available
        results = registry.search_by_data_availability([DataType.SAR, DataType.DEM])
        assert len(results) == 2
        assert sar_algo in results
        assert multi_algo in results

        # All data types available
        results = registry.search_by_data_availability([
            DataType.SAR, DataType.OPTICAL, DataType.DEM
        ])
        assert len(results) == 3

    def test_search_by_data_availability_with_event_type(self, registry):
        """Test combined data availability and event type search"""
        flood_sar = AlgorithmMetadata(
            id="flood.baseline.sar",
            name="Flood SAR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        wildfire_sar = AlgorithmMetadata(
            id="wildfire.baseline.sar",
            name="Wildfire SAR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.SAR]
        )

        registry.register(flood_sar)
        registry.register(wildfire_sar)

        results = registry.search_by_data_availability(
            [DataType.SAR],
            event_type="flood.coastal"
        )
        assert len(results) == 1
        assert flood_sar in results

    def test_search_by_requirements_memory(self, registry):
        """Test searching with memory constraints"""
        light_algo = AlgorithmMetadata(
            id="test.light",
            name="Light",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=1024)
        )

        heavy_algo = AlgorithmMetadata(
            id="test.heavy",
            name="Heavy",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=8192)
        )

        registry.register(light_algo)
        registry.register(heavy_algo)

        # 2GB memory available
        results = registry.search_by_requirements(max_memory_mb=2048)
        assert len(results) == 1
        assert light_algo in results

        # 10GB memory available
        results = registry.search_by_requirements(max_memory_mb=10240)
        assert len(results) == 2

    def test_search_by_requirements_gpu(self, registry):
        """Test searching with GPU constraints"""
        cpu_algo = AlgorithmMetadata(
            id="test.cpu",
            name="CPU",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(gpu_required=False)
        )

        gpu_algo = AlgorithmMetadata(
            id="test.gpu",
            name="GPU",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(gpu_required=True)
        )

        registry.register(cpu_algo)
        registry.register(gpu_algo)

        # No GPU available
        results = registry.search_by_requirements(gpu_available=False)
        assert len(results) == 1
        assert cpu_algo in results

        # GPU available
        results = registry.search_by_requirements(gpu_available=True)
        assert len(results) == 2

    def test_search_by_requirements_deterministic(self, registry):
        """Test searching for deterministic algorithms"""
        deterministic = AlgorithmMetadata(
            id="test.deterministic",
            name="Deterministic",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=True
        )

        stochastic = AlgorithmMetadata(
            id="test.stochastic",
            name="Stochastic",
            category=AlgorithmCategory.ADVANCED,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deterministic=False
        )

        registry.register(deterministic)
        registry.register(stochastic)

        results = registry.search_by_requirements(require_deterministic=True)
        assert len(results) == 1
        assert deterministic in results

    def test_search_by_requirements_deprecated(self, registry):
        """Test that deprecated algorithms are excluded"""
        active = AlgorithmMetadata(
            id="test.active",
            name="Active",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deprecated=False
        )

        deprecated = AlgorithmMetadata(
            id="test.deprecated",
            name="Deprecated",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deprecated=True
        )

        registry.register(active)
        registry.register(deprecated)

        results = registry.search_by_requirements(event_type="flood.*")
        assert len(results) == 1
        assert active in results

    def test_search_by_requirements_combined(self, registry):
        """Test searching with multiple combined constraints"""
        perfect_algo = AlgorithmMetadata(
            id="test.perfect",
            name="Perfect Match",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.coastal"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(
                memory_mb=1024,
                gpu_required=False
            ),
            deterministic=True,
            deprecated=False
        )

        wrong_event = AlgorithmMetadata(
            id="test.wrong_event",
            name="Wrong Event",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.SAR],
            resources=ResourceRequirements(memory_mb=1024),
            deterministic=True
        )

        needs_optical = AlgorithmMetadata(
            id="test.needs_optical",
            name="Needs Optical",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.OPTICAL],
            resources=ResourceRequirements(memory_mb=1024),
            deterministic=True
        )

        registry.register(perfect_algo)
        registry.register(wrong_event)
        registry.register(needs_optical)

        results = registry.search_by_requirements(
            event_type="flood.coastal",
            available_data_types=[DataType.SAR],
            max_memory_mb=2048,
            gpu_available=False,
            require_deterministic=True
        )

        assert len(results) == 1
        assert perfect_algo in results


class TestYAMLLoading:
    """Test loading algorithms from YAML files"""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test"""
        return AlgorithmRegistry()

    def test_load_from_yaml(self, registry):
        """Test loading algorithms from a YAML file"""
        yaml_content = """
algorithms:
  flood.baseline.test:
    name: "Test Algorithm"
    category: "baseline"
    version: "1.0.0"
    event_types:
      - "flood.*"
    required_data_types:
      - "sar"
    optional_data_types:
      - "dem"
    resources:
      memory_mb: 2048
      gpu_required: false
    deterministic: true
    description: "Test algorithm description"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            registry.load_from_yaml(temp_path)

            assert len(registry.algorithms) == 1
            algo = registry.get("flood.baseline.test")
            assert algo is not None
            assert algo.name == "Test Algorithm"
            assert algo.category == AlgorithmCategory.BASELINE
            assert algo.version == "1.0.0"
            assert DataType.SAR in algo.required_data_types
            assert DataType.DEM in algo.optional_data_types
        finally:
            temp_path.unlink()

    def test_load_with_validation_metrics(self, registry):
        """Test loading algorithms with validation metrics"""
        yaml_content = """
algorithms:
  flood.baseline.test:
    name: "Test Algorithm"
    category: "baseline"
    version: "1.0.0"
    event_types:
      - "flood.*"
    required_data_types:
      - "sar"
    validation:
      accuracy_min: 0.75
      accuracy_max: 0.92
      accuracy_median: 0.85
      precision: 0.82
      recall: 0.88
      f1_score: 0.85
      validated_regions:
        - "North America"
        - "Europe"
      validation_dataset_count: 47
      last_validated: "2024-08-15"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            registry.load_from_yaml(temp_path)

            algo = registry.get("flood.baseline.test")
            assert algo.validation is not None
            assert algo.validation.accuracy_median == 0.85
            assert algo.validation.precision == 0.82
            assert "North America" in algo.validation.validated_regions
        finally:
            temp_path.unlink()

    def test_load_multiple_algorithms(self, registry):
        """Test loading multiple algorithms from one file"""
        yaml_content = """
algorithms:
  flood.baseline.sar:
    name: "SAR Flood"
    category: "baseline"
    version: "1.0.0"
    event_types:
      - "flood.*"
    required_data_types:
      - "sar"

  wildfire.baseline.nbr:
    name: "NBR Wildfire"
    category: "baseline"
    version: "1.0.0"
    event_types:
      - "wildfire.*"
    required_data_types:
      - "optical"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            registry.load_from_yaml(temp_path)

            assert len(registry.algorithms) == 2
            assert registry.get("flood.baseline.sar") is not None
            assert registry.get("wildfire.baseline.nbr") is not None
        finally:
            temp_path.unlink()

    def test_load_from_directory(self, registry):
        """Test loading all YAML files from a directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create first YAML file
            file1 = tmpdir_path / "algorithms1.yaml"
            file1.write_text("""
algorithms:
  flood.baseline.test1:
    name: "Test 1"
    category: "baseline"
    version: "1.0.0"
    event_types: ["flood.*"]
    required_data_types: ["sar"]
""")

            # Create second YAML file
            file2 = tmpdir_path / "algorithms2.yaml"
            file2.write_text("""
algorithms:
  flood.baseline.test2:
    name: "Test 2"
    category: "baseline"
    version: "1.0.0"
    event_types: ["flood.*"]
    required_data_types: ["optical"]
""")

            registry.load_from_directory(tmpdir_path, recursive=False)

            assert len(registry.algorithms) == 2
            assert registry.get("flood.baseline.test1") is not None
            assert registry.get("flood.baseline.test2") is not None

    def test_load_from_directory_recursive(self, registry):
        """Test loading YAML files from subdirectories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            subdir = tmpdir_path / "subdir"
            subdir.mkdir()

            # Create file in subdirectory
            file1 = subdir / "algorithms.yaml"
            file1.write_text("""
algorithms:
  flood.baseline.test:
    name: "Test"
    category: "baseline"
    version: "1.0.0"
    event_types: ["flood.*"]
    required_data_types: ["sar"]
""")

            registry.load_from_directory(tmpdir_path, recursive=True)

            assert len(registry.algorithms) == 1
            assert registry.get("flood.baseline.test") is not None


class TestExportAndStatistics:
    """Test export and statistics functionality"""

    @pytest.fixture
    def registry(self):
        """Create a registry with sample algorithms"""
        reg = AlgorithmRegistry()

        reg.register(AlgorithmMetadata(
            id="flood.baseline.sar",
            name="SAR",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deprecated=False,
            deterministic=True,
            resources=ResourceRequirements(gpu_required=False)
        ))

        reg.register(AlgorithmMetadata(
            id="wildfire.advanced.ml",
            name="ML Wildfire",
            category=AlgorithmCategory.ADVANCED,
            version="2.0.0",
            event_types=["wildfire.*"],
            required_data_types=[DataType.OPTICAL],
            deprecated=False,
            deterministic=False,
            resources=ResourceRequirements(gpu_required=True)
        ))

        reg.register(AlgorithmMetadata(
            id="flood.baseline.old",
            name="Old Algorithm",
            category=AlgorithmCategory.BASELINE,
            version="0.5.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR],
            deprecated=True,
            deterministic=True,
            resources=ResourceRequirements(gpu_required=False)
        ))

        return reg

    def test_export_to_yaml(self, registry):
        """Test exporting algorithms to YAML"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = Path(f.name)

        try:
            registry.export_to_yaml(temp_path)

            # Load it back
            with open(temp_path, 'r') as f:
                data = yaml.safe_load(f)

            assert 'algorithms' in data
            assert len(data['algorithms']) == 3
            assert 'flood.baseline.sar' in data['algorithms']
            assert data['algorithms']['flood.baseline.sar']['name'] == 'SAR'
        finally:
            temp_path.unlink()

    def test_get_statistics(self, registry):
        """Test statistics generation"""
        stats = registry.get_statistics()

        assert stats['total_algorithms'] == 3
        assert stats['by_category']['baseline'] == 2
        assert stats['by_category']['advanced'] == 1
        assert stats['by_category']['experimental'] == 0
        assert stats['deprecated'] == 1
        assert stats['deterministic'] == 2
        assert stats['gpu_required'] == 1


class TestGlobalRegistry:
    """Test global registry functionality"""

    def test_get_global_registry(self):
        """Test getting the global registry singleton"""
        reg1 = get_global_registry()
        reg2 = get_global_registry()

        # Should be the same instance
        assert reg1 is reg2

    def test_load_default_algorithms(self):
        """Test loading default algorithms"""
        registry = load_default_algorithms()

        # Should load algorithms from the baseline_algorithms.yaml file
        # Check if at least some baseline algorithms are loaded
        assert len(registry.algorithms) > 0

        # Verify some known algorithms exist
        flood_algos = registry.search_by_event_type("flood.coastal")
        assert len(flood_algos) > 0

        wildfire_algos = registry.search_by_event_type("wildfire.forest")
        assert len(wildfire_algos) > 0


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test"""
        return AlgorithmRegistry()

    def test_empty_registry_searches(self, registry):
        """Test searches on empty registry"""
        assert len(registry.list_all()) == 0
        assert len(registry.search_by_event_type("flood.*")) == 0
        assert len(registry.search_by_data_availability([DataType.SAR])) == 0
        assert registry.get("nonexistent") is None

    def test_pattern_matching_edge_cases(self):
        """Test pattern matching with edge cases"""
        metadata = AlgorithmMetadata(
            id="test",
            name="Test",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[DataType.SAR]
        )

        # Should not match bare prefix
        assert not metadata.matches_event_type("flood")

        # Should not match partial prefix
        assert not metadata.matches_event_type("floo")

        # Should match with dot separator
        assert metadata.matches_event_type("flood.coastal")

    def test_multiple_wildcard_patterns(self):
        """Test algorithm with multiple wildcard patterns"""
        metadata = AlgorithmMetadata(
            id="test",
            name="Test",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*", "storm.*", "hurricane.*"],
            required_data_types=[DataType.SAR]
        )

        assert metadata.matches_event_type("flood.coastal")
        assert metadata.matches_event_type("storm.tropical")
        assert metadata.matches_event_type("hurricane.category5")
        assert not metadata.matches_event_type("wildfire.forest")

    def test_data_availability_empty_requirements(self, registry):
        """Test data availability with algorithm having no data requirements"""
        algo = AlgorithmMetadata(
            id="test",
            name="Test",
            category=AlgorithmCategory.BASELINE,
            version="1.0.0",
            event_types=["flood.*"],
            required_data_types=[]  # No requirements
        )

        registry.register(algo)

        # Should match any data availability
        results = registry.search_by_data_availability([])
        assert len(results) == 1

        results = registry.search_by_data_availability([DataType.SAR])
        assert len(results) == 1

    def test_invalid_yaml_file(self, registry):
        """Test handling of invalid YAML file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {")
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception):
                registry.load_from_yaml(temp_path)
        finally:
            temp_path.unlink()

    def test_yaml_missing_algorithms_key(self, registry):
        """Test YAML file without 'algorithms' key"""
        yaml_content = """
other_data:
  something: "value"
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = Path(f.name)

        try:
            registry.load_from_yaml(temp_path)
            # Should not raise error, just log warning
            assert len(registry.algorithms) == 0
        finally:
            temp_path.unlink()


class TestResourceRequirements:
    """Test ResourceRequirements dataclass"""

    def test_default_values(self):
        """Test default resource requirements"""
        req = ResourceRequirements()

        assert req.memory_mb is None
        assert req.gpu_required is False
        assert req.gpu_memory_mb is None
        assert req.max_runtime_minutes is None
        assert req.distributed is False

    def test_custom_values(self):
        """Test custom resource requirements"""
        req = ResourceRequirements(
            memory_mb=4096,
            gpu_required=True,
            gpu_memory_mb=8192,
            max_runtime_minutes=30,
            distributed=True
        )

        assert req.memory_mb == 4096
        assert req.gpu_required is True
        assert req.gpu_memory_mb == 8192
        assert req.max_runtime_minutes == 30
        assert req.distributed is True


class TestValidationMetrics:
    """Test ValidationMetrics dataclass"""

    def test_default_values(self):
        """Test default validation metrics"""
        metrics = ValidationMetrics()

        assert metrics.accuracy_min is None
        assert metrics.accuracy_max is None
        assert metrics.accuracy_median is None
        assert metrics.precision is None
        assert metrics.recall is None
        assert metrics.f1_score is None
        assert metrics.validated_regions == []
        assert metrics.validation_dataset_count == 0
        assert metrics.last_validated is None

    def test_custom_values(self):
        """Test custom validation metrics"""
        metrics = ValidationMetrics(
            accuracy_min=0.75,
            accuracy_max=0.92,
            accuracy_median=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            validated_regions=["North America", "Europe"],
            validation_dataset_count=47,
            last_validated="2024-08-15"
        )

        assert metrics.accuracy_median == 0.85
        assert metrics.f1_score == 0.85
        assert len(metrics.validated_regions) == 2
        assert metrics.validation_dataset_count == 47
