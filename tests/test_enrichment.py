"""
Tests for the Enrichment Module (Group G, Track 4).

Tests cover:
- Overview generation (overviews.py)
- Statistics calculation (statistics.py)
- Quality assessment (quality.py)

Focus on:
- Edge cases (empty arrays, NaN/Inf values, single elements)
- Division by zero guards
- Configuration validation
- Integration between components
"""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

# Import enrichment module components
from core.data.ingestion.enrichment import (
    BandStatistics,
    DimensionScore,
    HistogramType,
    OverviewConfig,
    OverviewFormat,
    OverviewGenerator,
    OverviewLevel,
    OverviewResampling,
    OverviewResult,
    QualityAssessor,
    QualityConfig,
    QualityDimension,
    QualityFlag,
    QualityIssue,
    QualityLevel,
    QualitySummary,
    RasterStatistics,
    StatisticsCalculator,
    StatisticsConfig,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_2d_array():
    """Create a simple 2D array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)


@pytest.fixture
def sample_3d_array():
    """Create a 3D array (bands, height, width) for testing."""
    np.random.seed(42)
    return np.random.rand(3, 100, 100).astype(np.float32) * 100


@pytest.fixture
def sample_array_with_nodata():
    """Create an array with nodata values."""
    data = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]], dtype=np.float32)
    return data


@pytest.fixture
def sample_array_with_inf():
    """Create an array with Inf values."""
    data = np.array([[1, 2, np.inf], [4, -np.inf, 6], [7, 8, 9]], dtype=np.float32)
    return data


@pytest.fixture
def empty_array():
    """Create an empty array."""
    return np.array([], dtype=np.float32).reshape(0, 0)


@pytest.fixture
def single_value_array():
    """Create an array with a single value."""
    return np.array([[5.0]], dtype=np.float32)


@pytest.fixture
def all_same_array():
    """Create an array with all same values."""
    return np.full((10, 10), 42.0, dtype=np.float32)


@pytest.fixture
def integer_categorical_array():
    """Create an integer array with categorical values."""
    return np.array([[0, 1, 1], [2, 0, 1], [2, 2, 0]], dtype=np.int32)


# =============================================================================
# OverviewConfig Tests
# =============================================================================


class TestOverviewConfig:
    """Tests for OverviewConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OverviewConfig()
        assert config.factors == [2, 4, 8, 16, 32]
        assert config.resampling == OverviewResampling.AVERAGE
        assert config.format == OverviewFormat.INTERNAL
        assert config.blocksize == 512
        assert config.compress is True
        assert config.min_size == 256
        assert config.force_power_of_two is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OverviewConfig(
            factors=[2, 4],
            resampling=OverviewResampling.NEAREST,
            blocksize=256,
        )
        assert config.factors == [2, 4]
        assert config.resampling == OverviewResampling.NEAREST
        assert config.blocksize == 256

    def test_invalid_factors_empty(self):
        """Test that empty factors list raises error."""
        with pytest.raises(ValueError, match="At least one overview factor"):
            OverviewConfig(factors=[])

    def test_invalid_factors_less_than_two(self):
        """Test that factors < 2 raise error."""
        with pytest.raises(ValueError, match="factors must be >= 2"):
            OverviewConfig(factors=[1, 2, 4])

    def test_invalid_factors_not_power_of_two(self):
        """Test that non-power-of-2 factors raise error when required."""
        with pytest.raises(ValueError, match="not a power of 2"):
            OverviewConfig(factors=[2, 3, 4], force_power_of_two=True)

    def test_non_power_of_two_allowed(self):
        """Test that non-power-of-2 factors work when not required."""
        config = OverviewConfig(factors=[2, 3, 6], force_power_of_two=False)
        assert config.factors == [2, 3, 6]

    def test_invalid_blocksize(self):
        """Test that invalid blocksize raises error."""
        with pytest.raises(ValueError, match="blocksize must be"):
            OverviewConfig(blocksize=100)

    def test_auto_factors(self):
        """Test automatic factor calculation."""
        config = OverviewConfig(min_size=256)
        factors = config.auto_factors(width=4096, height=4096)
        assert factors == [2, 4, 8, 16]

    def test_auto_factors_small_image(self):
        """Test auto factors for small image."""
        config = OverviewConfig(min_size=256)
        factors = config.auto_factors(width=512, height=512)
        assert factors == [2]

    def test_auto_factors_very_small_image(self):
        """Test auto factors for very small image."""
        config = OverviewConfig(min_size=256)
        factors = config.auto_factors(width=100, height=100)
        # Should return at least [2] even if image is small
        assert factors == [2]


class TestOverviewResampling:
    """Tests for OverviewResampling enum."""

    def test_for_data_type_continuous(self):
        """Test resampling selection for continuous data."""
        assert OverviewResampling.for_data_type("continuous") == OverviewResampling.AVERAGE

    def test_for_data_type_categorical(self):
        """Test resampling selection for categorical data."""
        assert OverviewResampling.for_data_type("categorical") == OverviewResampling.MODE

    def test_for_data_type_mask(self):
        """Test resampling selection for mask data."""
        assert OverviewResampling.for_data_type("mask") == OverviewResampling.NEAREST

    def test_for_data_type_imagery(self):
        """Test resampling selection for imagery."""
        assert OverviewResampling.for_data_type("imagery") == OverviewResampling.LANCZOS

    def test_for_data_type_unknown(self):
        """Test resampling selection for unknown type defaults to AVERAGE."""
        assert OverviewResampling.for_data_type("unknown") == OverviewResampling.AVERAGE

    def test_for_data_type_case_insensitive(self):
        """Test that data type matching is case insensitive."""
        assert OverviewResampling.for_data_type("CONTINUOUS") == OverviewResampling.AVERAGE


# =============================================================================
# OverviewGenerator Tests
# =============================================================================


class TestOverviewGenerator:
    """Tests for OverviewGenerator class."""

    def test_generate_from_array_2d(self, sample_2d_array):
        """Test generating overviews from 2D array."""
        generator = OverviewGenerator(OverviewConfig(factors=[2]))
        overviews = generator.generate_from_array(sample_2d_array, factors=[2])
        assert len(overviews) == 1
        assert overviews[0].shape[0] == 1  # bands
        assert overviews[0].shape[1] == 1  # height // 2
        assert overviews[0].shape[2] == 1  # width // 2

    def test_generate_from_array_3d(self, sample_3d_array):
        """Test generating overviews from 3D array."""
        generator = OverviewGenerator(OverviewConfig(factors=[2, 4]))
        overviews = generator.generate_from_array(sample_3d_array, factors=[2, 4])
        assert len(overviews) == 2
        # Factor 2
        assert overviews[0].shape == (3, 50, 50)
        # Factor 4
        assert overviews[1].shape == (3, 25, 25)

    def test_generate_from_array_with_nan(self, sample_array_with_nodata):
        """Test that NaN values are handled in overviews."""
        data = np.array([sample_array_with_nodata])  # Add band dimension
        generator = OverviewGenerator(
            OverviewConfig(factors=[2], resampling=OverviewResampling.AVERAGE)
        )
        overviews = generator.generate_from_array(data, factors=[2])
        # Should not crash and should use nanmean
        assert len(overviews) == 1

    def test_generate_from_array_nearest(self, sample_3d_array):
        """Test nearest neighbor resampling."""
        config = OverviewConfig(factors=[2], resampling=OverviewResampling.NEAREST)
        generator = OverviewGenerator(config)
        overviews = generator.generate_from_array(sample_3d_array, factors=[2])
        assert overviews[0].shape == (3, 50, 50)

    def test_generate_from_array_max(self, sample_3d_array):
        """Test max resampling."""
        config = OverviewConfig(factors=[2], resampling=OverviewResampling.MAX)
        generator = OverviewGenerator(config)
        overviews = generator.generate_from_array(sample_3d_array, factors=[2])
        assert overviews[0].shape == (3, 50, 50)

    def test_generate_from_array_min(self, sample_3d_array):
        """Test min resampling."""
        config = OverviewConfig(factors=[2], resampling=OverviewResampling.MIN)
        generator = OverviewGenerator(config)
        overviews = generator.generate_from_array(sample_3d_array, factors=[2])
        assert overviews[0].shape == (3, 50, 50)

    def test_generate_from_array_mode(self, integer_categorical_array):
        """Test mode resampling for categorical data."""
        data = np.array([integer_categorical_array])  # Add band dimension
        config = OverviewConfig(factors=[2], resampling=OverviewResampling.MODE)
        generator = OverviewGenerator(config)
        overviews = generator.generate_from_array(data.astype(np.float32), factors=[2])
        assert overviews[0].shape == (1, 1, 1)

    def test_generate_from_array_unsupported_method(self, sample_3d_array):
        """Test fallback for unsupported resampling method."""
        config = OverviewConfig(factors=[2], resampling=OverviewResampling.CUBIC)
        generator = OverviewGenerator(config)
        overviews = generator.generate_from_array(sample_3d_array, factors=[2])
        # Should fall back to nearest
        assert overviews[0].shape == (3, 50, 50)


class TestOverviewLevel:
    """Tests for OverviewLevel dataclass."""

    def test_dimensions_property(self):
        """Test dimensions property."""
        level = OverviewLevel(
            factor=2, width=100, height=50, size_bytes=10000, resampling="average"
        )
        assert level.dimensions == (100, 50)

    def test_reduction_property(self):
        """Test reduction property."""
        level = OverviewLevel(
            factor=4, width=256, height=128, size_bytes=65536, resampling="average"
        )
        assert level.reduction == "1:4 (256x128)"


class TestOverviewResult:
    """Tests for OverviewResult dataclass."""

    def test_level_count(self):
        """Test level_count property."""
        result = OverviewResult(
            input_path=Path("test.tif"),
            output_path=None,
            levels=[
                OverviewLevel(2, 500, 500, 250000, "average"),
                OverviewLevel(4, 250, 250, 62500, "average"),
            ],
            total_overview_bytes=312500,
            overhead_percent=10.0,
            resampling_method="average",
            format=OverviewFormat.INTERNAL,
            metadata={},
        )
        assert result.level_count == 2

    def test_factors_property(self):
        """Test factors property."""
        result = OverviewResult(
            input_path=Path("test.tif"),
            output_path=None,
            levels=[
                OverviewLevel(2, 500, 500, 250000, "average"),
                OverviewLevel(4, 250, 250, 62500, "average"),
            ],
            total_overview_bytes=312500,
            overhead_percent=10.0,
            resampling_method="average",
            format=OverviewFormat.INTERNAL,
            metadata={},
        )
        assert result.factors == [2, 4]

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = OverviewResult(
            input_path=Path("test.tif"),
            output_path=None,
            levels=[OverviewLevel(2, 500, 500, 250000, "average")],
            total_overview_bytes=250000,
            overhead_percent=10.0,
            resampling_method="average",
            format=OverviewFormat.INTERNAL,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["input_path"] == "test.tif"
        assert d["output_path"] is None
        assert len(d["levels"]) == 1
        assert d["format"] == "internal"


# =============================================================================
# StatisticsConfig Tests
# =============================================================================


class TestStatisticsConfig:
    """Tests for StatisticsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StatisticsConfig()
        assert config.compute_histogram is True
        assert config.histogram_bins == 256
        assert config.histogram_type == HistogramType.LINEAR
        assert config.percentiles == [1, 5, 10, 25, 50, 75, 90, 95, 99]
        assert config.sample_size is None
        assert config.approx_ok is True
        assert config.per_block is False

    def test_invalid_histogram_bins(self):
        """Test that invalid histogram bins raise error."""
        with pytest.raises(ValueError, match="histogram_bins must be >= 1"):
            StatisticsConfig(histogram_bins=0)

    def test_invalid_percentiles(self):
        """Test that invalid percentiles raise error."""
        with pytest.raises(ValueError, match="percentiles must be in"):
            StatisticsConfig(percentiles=[0, 50, 101])


# =============================================================================
# StatisticsCalculator Tests
# =============================================================================


class TestStatisticsCalculator:
    """Tests for StatisticsCalculator class."""

    def test_compute_from_array_2d(self, sample_2d_array):
        """Test computing statistics from 2D array."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_2d_array)
        assert len(stats) == 1
        assert stats[0].band_index == 1
        assert stats[0].min == 1.0
        assert stats[0].max == 9.0
        assert stats[0].mean == 5.0
        assert stats[0].valid_count == 9

    def test_compute_from_array_3d(self, sample_3d_array):
        """Test computing statistics from 3D array."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_3d_array)
        assert len(stats) == 3
        for i, band_stat in enumerate(stats):
            assert band_stat.band_index == i + 1
            assert band_stat.valid_count == 10000

    def test_compute_from_array_with_nodata(self, sample_array_with_nodata):
        """Test that NaN values are excluded from statistics."""
        calculator = StatisticsCalculator(StatisticsConfig(nodata_values=[]))
        stats = calculator.compute_from_array(sample_array_with_nodata)
        assert stats[0].valid_count == 7  # 2 NaN values excluded
        assert stats[0].nodata_count == 2
        assert np.isclose(stats[0].valid_percent, 77.78, atol=0.1)

    def test_compute_from_array_with_inf(self, sample_array_with_inf):
        """Test that Inf values are excluded from statistics."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_array_with_inf)
        assert stats[0].valid_count == 7  # 2 Inf values excluded
        assert stats[0].nodata_count == 2

    def test_compute_from_array_all_nodata(self):
        """Test handling of all-nodata array."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(data)
        assert stats[0].valid_count == 0
        assert np.isnan(stats[0].min)
        assert np.isnan(stats[0].mean)

    def test_compute_from_array_single_value(self, single_value_array):
        """Test statistics for single value array."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(single_value_array)
        assert stats[0].min == 5.0
        assert stats[0].max == 5.0
        assert stats[0].mean == 5.0
        assert stats[0].std == 0.0

    def test_compute_from_array_all_same(self, all_same_array):
        """Test statistics for array with all same values."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(all_same_array)
        assert stats[0].min == 42.0
        assert stats[0].max == 42.0
        assert stats[0].mean == 42.0
        assert stats[0].std == 0.0
        assert stats[0].cv == 0.0  # Coefficient of variation

    def test_compute_from_array_invalid_dims(self):
        """Test that 4D array raises error."""
        data = np.zeros((2, 3, 4, 5), dtype=np.float32)
        calculator = StatisticsCalculator()
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            calculator.compute_from_array(data)

    def test_histogram_linear(self, sample_3d_array):
        """Test linear histogram computation."""
        config = StatisticsConfig(
            compute_histogram=True, histogram_type=HistogramType.LINEAR
        )
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(sample_3d_array)
        assert stats[0].histogram is not None
        counts, edges = stats[0].histogram
        assert len(counts) == 256
        assert len(edges) == 257

    def test_histogram_logarithmic(self):
        """Test logarithmic histogram computation."""
        # Create data with positive values for log
        data = np.random.uniform(1, 100, (100, 100)).astype(np.float32)
        config = StatisticsConfig(
            compute_histogram=True, histogram_type=HistogramType.LOGARITHMIC
        )
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(data)
        assert stats[0].histogram is not None

    def test_histogram_logarithmic_with_zeros(self):
        """Test log histogram falls back for non-positive data."""
        data = np.random.uniform(-10, 100, (100, 100)).astype(np.float32)
        config = StatisticsConfig(
            compute_histogram=True, histogram_type=HistogramType.LOGARITHMIC
        )
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(data)
        # Should fall back to linear
        assert stats[0].histogram is not None

    def test_histogram_percentile(self, sample_3d_array):
        """Test percentile-based histogram."""
        config = StatisticsConfig(
            compute_histogram=True, histogram_type=HistogramType.PERCENTILE
        )
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(sample_3d_array)
        assert stats[0].histogram is not None

    def test_histogram_automatic(self, sample_3d_array):
        """Test automatic histogram bin selection."""
        config = StatisticsConfig(
            compute_histogram=True, histogram_type=HistogramType.AUTOMATIC
        )
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(sample_3d_array)
        assert stats[0].histogram is not None

    def test_percentiles(self, sample_3d_array):
        """Test percentile computation."""
        config = StatisticsConfig(percentiles=[5, 50, 95])
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(sample_3d_array)
        assert 5 in stats[0].percentiles
        assert 50 in stats[0].percentiles
        assert 95 in stats[0].percentiles

    def test_sampling(self, sample_3d_array):
        """Test sampling for large data."""
        config = StatisticsConfig(sample_size=1000)
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(sample_3d_array)
        # Should still compute statistics
        assert stats[0].mean is not None

    def test_categorical_detection(self, integer_categorical_array):
        """Test categorical data detection."""
        config = StatisticsConfig(compute_histogram=False)
        calculator = StatisticsCalculator(config)
        stats = calculator.compute_from_array(integer_categorical_array)
        assert stats[0].is_integer is True
        assert stats[0].unique_count == 3  # 0, 1, 2

    def test_get_stretch_params_minmax(self, sample_3d_array):
        """Test minmax stretch parameters."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_3d_array)
        stretch = calculator.get_stretch_params(stats[0], method="minmax")
        assert 1 in stretch
        min_val, max_val = stretch[1]
        assert np.isclose(min_val, stats[0].min)
        assert np.isclose(max_val, stats[0].max)

    def test_get_stretch_params_stddev(self, sample_3d_array):
        """Test stddev stretch parameters."""
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_3d_array)
        stretch = calculator.get_stretch_params(stats[0], method="stddev")
        assert 1 in stretch


class TestBandStatistics:
    """Tests for BandStatistics dataclass."""

    def test_range_property(self):
        """Test range property."""
        stats = BandStatistics(
            band_index=1,
            min=0.0,
            max=100.0,
            mean=50.0,
            std=25.0,
            median=50.0,
            variance=625.0,
            sum=5000.0,
            count=100,
            valid_count=100,
            nodata_count=0,
            valid_percent=100.0,
            percentiles={},
            histogram=None,
            dtype="float32",
            unique_count=None,
            is_integer=False,
            is_categorical=False,
        )
        assert stats.range == 100.0

    def test_cv_property(self):
        """Test coefficient of variation property."""
        stats = BandStatistics(
            band_index=1,
            min=0.0,
            max=100.0,
            mean=50.0,
            std=10.0,
            median=50.0,
            variance=100.0,
            sum=5000.0,
            count=100,
            valid_count=100,
            nodata_count=0,
            valid_percent=100.0,
            percentiles={},
            histogram=None,
            dtype="float32",
            unique_count=None,
            is_integer=False,
            is_categorical=False,
        )
        assert stats.cv == 0.2  # 10/50

    def test_cv_property_zero_mean(self):
        """Test CV when mean is zero."""
        stats = BandStatistics(
            band_index=1,
            min=-50.0,
            max=50.0,
            mean=0.0,
            std=25.0,
            median=0.0,
            variance=625.0,
            sum=0.0,
            count=100,
            valid_count=100,
            nodata_count=0,
            valid_percent=100.0,
            percentiles={},
            histogram=None,
            dtype="float32",
            unique_count=None,
            is_integer=False,
            is_categorical=False,
        )
        assert stats.cv == float("inf")

    def test_snr_property(self):
        """Test signal-to-noise ratio property."""
        stats = BandStatistics(
            band_index=1,
            min=0.0,
            max=100.0,
            mean=50.0,
            std=10.0,
            median=50.0,
            variance=100.0,
            sum=5000.0,
            count=100,
            valid_count=100,
            nodata_count=0,
            valid_percent=100.0,
            percentiles={},
            histogram=None,
            dtype="float32",
            unique_count=None,
            is_integer=False,
            is_categorical=False,
        )
        assert stats.snr == 5.0  # 50/10

    def test_snr_property_zero_std(self):
        """Test SNR when std is zero."""
        stats = BandStatistics(
            band_index=1,
            min=50.0,
            max=50.0,
            mean=50.0,
            std=0.0,
            median=50.0,
            variance=0.0,
            sum=5000.0,
            count=100,
            valid_count=100,
            nodata_count=0,
            valid_percent=100.0,
            percentiles={},
            histogram=None,
            dtype="float32",
            unique_count=None,
            is_integer=False,
            is_categorical=False,
        )
        assert stats.snr == float("inf")

    def test_to_dict_with_nan(self):
        """Test to_dict handles NaN values."""
        stats = BandStatistics(
            band_index=1,
            min=float("nan"),
            max=float("nan"),
            mean=float("nan"),
            std=float("nan"),
            median=float("nan"),
            variance=float("nan"),
            sum=0.0,
            count=100,
            valid_count=0,
            nodata_count=100,
            valid_percent=0.0,
            percentiles={50: float("nan")},
            histogram=None,
            dtype="float32",
            unique_count=0,
            is_integer=False,
            is_categorical=False,
        )
        d = stats.to_dict()
        assert d["min"] is None
        assert d["max"] is None
        assert d["mean"] is None

    def test_to_dict_with_histogram(self, sample_2d_array):
        """Test to_dict includes histogram data."""
        calculator = StatisticsCalculator(StatisticsConfig(compute_histogram=True))
        stats = calculator.compute_from_array(sample_2d_array)
        d = stats[0].to_dict()
        assert "histogram" in d
        assert "counts" in d["histogram"]
        assert "bin_edges" in d["histogram"]


class TestRasterStatistics:
    """Tests for RasterStatistics dataclass."""

    def test_get_band(self):
        """Test get_band method."""
        band_stats = [
            BandStatistics(
                band_index=i,
                min=0.0,
                max=100.0,
                mean=50.0,
                std=10.0,
                median=50.0,
                variance=100.0,
                sum=5000.0,
                count=100,
                valid_count=100,
                nodata_count=0,
                valid_percent=100.0,
                percentiles={},
                histogram=None,
                dtype="float32",
                unique_count=None,
                is_integer=False,
                is_categorical=False,
            )
            for i in range(1, 4)
        ]
        raster_stats = RasterStatistics(
            path=Path("test.tif"),
            band_stats=band_stats,
            width=100,
            height=100,
            band_count=3,
            crs="EPSG:4326",
            bounds=(-180, -90, 180, 90),
            global_min=0.0,
            global_max=100.0,
            total_pixels=30000,
            total_valid_pixels=30000,
            overall_valid_percent=100.0,
            metadata={},
        )
        band = raster_stats.get_band(2)
        assert band.band_index == 2

    def test_get_band_invalid_index(self):
        """Test get_band with invalid index."""
        raster_stats = RasterStatistics(
            path=Path("test.tif"),
            band_stats=[],
            width=100,
            height=100,
            band_count=0,
            crs=None,
            bounds=None,
            global_min=0.0,
            global_max=0.0,
            total_pixels=0,
            total_valid_pixels=0,
            overall_valid_percent=0.0,
            metadata={},
        )
        with pytest.raises(IndexError):
            raster_stats.get_band(1)

    def test_global_range(self):
        """Test global_range property."""
        raster_stats = RasterStatistics(
            path=Path("test.tif"),
            band_stats=[],
            width=100,
            height=100,
            band_count=0,
            crs=None,
            bounds=None,
            global_min=10.0,
            global_max=90.0,
            total_pixels=0,
            total_valid_pixels=0,
            overall_valid_percent=0.0,
            metadata={},
        )
        assert raster_stats.global_range == 80.0


# =============================================================================
# QualityConfig Tests
# =============================================================================


class TestQualityConfig:
    """Tests for QualityConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QualityConfig()
        assert config.min_valid_percent == 80.0
        assert config.max_cloud_cover == 30.0
        assert config.max_noise_threshold == 0.1
        assert config.check_geometric is True
        assert config.check_atmospheric is True

    def test_dimension_weights_normalized(self):
        """Test that dimension weights are normalized."""
        config = QualityConfig(
            dimension_weights={
                QualityDimension.COMPLETENESS: 1.0,
                QualityDimension.SPATIAL: 1.0,
            }
        )
        total = sum(config.dimension_weights.values())
        assert np.isclose(total, 1.0)


# =============================================================================
# QualityLevel Tests
# =============================================================================


class TestQualityLevel:
    """Tests for QualityLevel enum."""

    def test_from_score_excellent(self):
        """Test excellent level threshold."""
        assert QualityLevel.from_score(0.95) == QualityLevel.EXCELLENT
        assert QualityLevel.from_score(0.9) == QualityLevel.EXCELLENT

    def test_from_score_good(self):
        """Test good level threshold."""
        assert QualityLevel.from_score(0.85) == QualityLevel.GOOD
        assert QualityLevel.from_score(0.7) == QualityLevel.GOOD

    def test_from_score_acceptable(self):
        """Test acceptable level threshold."""
        assert QualityLevel.from_score(0.6) == QualityLevel.ACCEPTABLE
        assert QualityLevel.from_score(0.5) == QualityLevel.ACCEPTABLE

    def test_from_score_degraded(self):
        """Test degraded level threshold."""
        assert QualityLevel.from_score(0.4) == QualityLevel.DEGRADED
        assert QualityLevel.from_score(0.3) == QualityLevel.DEGRADED

    def test_from_score_poor(self):
        """Test poor level threshold."""
        assert QualityLevel.from_score(0.2) == QualityLevel.POOR
        assert QualityLevel.from_score(0.0) == QualityLevel.POOR


# =============================================================================
# QualityFlag Tests
# =============================================================================


class TestQualityFlag:
    """Tests for QualityFlag dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        flag = QualityFlag(
            issue=QualityIssue.CLOUD_COVER,
            severity=0.7,
            affected_area_percent=50.0,
            description="50% cloud cover",
            recommendation="Use SAR data",
            metadata={"cloud_percent": 50},
        )
        d = flag.to_dict()
        assert d["issue"] == "cloud_cover"
        assert d["severity"] == 0.7
        assert d["affected_area_percent"] == 50.0
        assert d["recommendation"] == "Use SAR data"


# =============================================================================
# DimensionScore Tests
# =============================================================================


class TestDimensionScore:
    """Tests for DimensionScore dataclass."""

    def test_weighted_score(self):
        """Test weighted_score property."""
        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.8,
            weight=0.25,
            flags=[],
            details={},
        )
        assert ds.weighted_score == 0.2

    def test_level_property(self):
        """Test level property."""
        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.8,
            weight=0.25,
            flags=[],
            details={},
        )
        assert ds.level == QualityLevel.GOOD

    def test_to_dict(self):
        """Test to_dict conversion."""
        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.8,
            weight=0.25,
            flags=[],
            details={"valid_percent": 80},
        )
        d = ds.to_dict()
        assert d["dimension"] == "completeness"
        assert d["score"] == 0.8
        assert d["weighted_score"] == 0.2


# =============================================================================
# QualityAssessor Tests
# =============================================================================


class TestQualityAssessor:
    """Tests for QualityAssessor class."""

    def test_assess_completeness_full_data(self):
        """Test completeness assessment with full valid data."""
        assessor = QualityAssessor()
        data = np.random.rand(3, 100, 100).astype(np.float32)
        score, flags, details = assessor._assess_completeness(data, None, None, None)
        assert score == 1.0
        assert len(flags) == 0
        assert details["valid_percent"] == 100.0

    def test_assess_completeness_with_nodata(self):
        """Test completeness assessment with nodata."""
        assessor = QualityAssessor(QualityConfig(min_valid_percent=80.0))
        data = np.random.rand(1, 100, 100).astype(np.float32)
        data[0, :50, :] = np.nan  # 50% nodata
        score, flags, details = assessor._assess_completeness(data, None, None, None)
        assert score == 0.5
        assert len(flags) >= 1
        assert any(f.issue == QualityIssue.MISSING_DATA for f in flags)

    def test_assess_spatial(self):
        """Test spatial quality assessment."""
        assessor = QualityAssessor()
        data = np.random.rand(3, 100, 100).astype(np.float32)
        score, flags, details = assessor._assess_spatial(data, 100, 100)
        assert 0.0 <= score <= 1.0

    def test_assess_spatial_with_edge_artifacts(self):
        """Test spatial assessment detects edge artifacts."""
        assessor = QualityAssessor()
        # Create data with strong edge artifacts (edge deviation > 3 std)
        np.random.seed(42)
        data = np.random.normal(50.0, 5.0, (1, 100, 100)).astype(np.float32)
        # Add extremely different edge values to exceed 3 std threshold
        data[0, :10, :] = 1000.0  # Top edge extremely higher
        score, flags, details = assessor._assess_spatial(data, 100, 100)
        # Should detect edge artifacts
        assert any(f.issue == QualityIssue.EDGE_ARTIFACTS for f in flags)

    def test_assess_temporal_with_timestamp(self):
        """Test temporal assessment with valid timestamp."""
        assessor = QualityAssessor()
        recent_time = datetime.now(timezone.utc) - timedelta(days=5)
        score, flags, details = assessor._assess_temporal(recent_time)
        assert score == 1.0
        assert len(flags) == 0

    def test_assess_temporal_stale_data(self):
        """Test temporal assessment with stale data."""
        assessor = QualityAssessor()
        old_time = datetime.now(timezone.utc) - timedelta(days=60)
        score, flags, details = assessor._assess_temporal(old_time)
        assert score < 1.0
        assert any(f.issue == QualityIssue.STALE_DATA for f in flags)

    def test_assess_temporal_no_timestamp(self):
        """Test temporal assessment without timestamp."""
        assessor = QualityAssessor()
        score, flags, details = assessor._assess_temporal(None)
        assert score == 0.7
        assert any(f.issue == QualityIssue.TIMESTAMP_MISSING for f in flags)

    def test_assess_radiometric(self):
        """Test radiometric quality assessment."""
        assessor = QualityAssessor()
        data = np.random.rand(3, 100, 100).astype(np.float32) * 100
        score, flags, details = assessor._assess_radiometric(data, "float32")
        assert 0.0 <= score <= 1.0

    def test_assess_radiometric_saturation(self):
        """Test radiometric assessment detects saturation."""
        assessor = QualityAssessor()
        data = np.random.randint(0, 255, (1, 100, 100), dtype=np.uint8)
        # Add 5% saturated pixels
        data[0, :5, :] = 255
        score, flags, details = assessor._assess_radiometric(data, "uint8")
        assert any(f.issue == QualityIssue.SATURATION for f in flags)

    def test_assess_geometric_with_crs(self):
        """Test geometric assessment with valid CRS."""
        assessor = QualityAssessor()
        # Create mock bounds object
        class MockBounds:
            left = -180
            bottom = -90
            right = 180
            top = 90
        score, flags, details = assessor._assess_geometric("EPSG:4326", MockBounds())
        assert score == 1.0
        assert len(flags) == 0

    def test_assess_geometric_no_crs(self):
        """Test geometric assessment without CRS."""
        assessor = QualityAssessor()
        score, flags, details = assessor._assess_geometric(None, None)
        assert score < 1.0
        assert any(f.issue == QualityIssue.PROJECTION_ERROR for f in flags)

    def test_assess_geometric_suspicious_bounds(self):
        """Test geometric assessment with suspicious bounds."""
        assessor = QualityAssessor()
        class MockBounds:
            left = 0
            bottom = 0
            right = 100
            top = 100
        score, flags, details = assessor._assess_geometric("EPSG:4326", MockBounds())
        assert any(f.issue == QualityIssue.GEOREFERENCING_ERROR for f in flags)

    def test_assess_atmospheric(self):
        """Test atmospheric quality assessment."""
        assessor = QualityAssessor()
        # 20% cloud cover
        cloud_mask = np.zeros((100, 100), dtype=bool)
        cloud_mask[:20, :] = True
        score, flags, details = assessor._assess_atmospheric(cloud_mask)
        assert score == 0.8  # 1 - 0.2

    def test_assess_atmospheric_high_clouds(self):
        """Test atmospheric assessment with high cloud cover."""
        assessor = QualityAssessor(QualityConfig(max_cloud_cover=30.0))
        # 50% cloud cover
        cloud_mask = np.zeros((100, 100), dtype=bool)
        cloud_mask[:50, :] = True
        score, flags, details = assessor._assess_atmospheric(cloud_mask)
        assert score == 0.5
        assert any(f.issue == QualityIssue.CLOUD_COVER for f in flags)

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        assessor = QualityAssessor()
        flags = [
            QualityFlag(
                issue=QualityIssue.CLOUD_COVER,
                severity=0.5,
                affected_area_percent=50.0,
                description="50% cloud cover",
                recommendation="Use SAR data",
            )
        ]
        dimension_scores = [
            DimensionScore(
                dimension=QualityDimension.ATMOSPHERIC,
                score=0.4,
                weight=0.1,
                flags=[],
                details={},
            )
        ]
        recommendations = assessor._generate_recommendations(flags, dimension_scores)
        assert "Use SAR data" in recommendations
        assert "radar-based" in recommendations[1].lower()


class TestQualitySummary:
    """Tests for QualitySummary dataclass."""

    def test_critical_issues(self):
        """Test critical_issues property."""
        summary = QualitySummary(
            path=Path("test.tif"),
            overall_score=0.5,
            overall_level=QualityLevel.ACCEPTABLE,
            confidence=0.8,
            dimension_scores=[],
            flags=[
                QualityFlag(
                    issue=QualityIssue.CLOUD_COVER,
                    severity=0.8,
                    affected_area_percent=80.0,
                    description="High cloud cover",
                ),
                QualityFlag(
                    issue=QualityIssue.STALE_DATA,
                    severity=0.3,
                    affected_area_percent=100.0,
                    description="Old data",
                ),
            ],
            usable=True,
            recommendations=[],
            assessed_at=datetime.now(timezone.utc),
            metadata={},
        )
        critical = summary.critical_issues
        assert len(critical) == 1
        assert critical[0].issue == QualityIssue.CLOUD_COVER

    def test_has_critical_issues(self):
        """Test has_critical_issues property."""
        summary_with_critical = QualitySummary(
            path=Path("test.tif"),
            overall_score=0.5,
            overall_level=QualityLevel.ACCEPTABLE,
            confidence=0.8,
            dimension_scores=[],
            flags=[
                QualityFlag(
                    issue=QualityIssue.CLOUD_COVER,
                    severity=0.8,
                    affected_area_percent=80.0,
                    description="High cloud cover",
                ),
            ],
            usable=True,
            recommendations=[],
            assessed_at=datetime.now(timezone.utc),
            metadata={},
        )
        assert summary_with_critical.has_critical_issues is True

        summary_no_critical = QualitySummary(
            path=Path("test.tif"),
            overall_score=0.8,
            overall_level=QualityLevel.GOOD,
            confidence=0.8,
            dimension_scores=[],
            flags=[],
            usable=True,
            recommendations=[],
            assessed_at=datetime.now(timezone.utc),
            metadata={},
        )
        assert summary_no_critical.has_critical_issues is False

    def test_get_dimension(self):
        """Test get_dimension method."""
        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.9,
            weight=0.25,
            flags=[],
            details={},
        )
        summary = QualitySummary(
            path=Path("test.tif"),
            overall_score=0.9,
            overall_level=QualityLevel.EXCELLENT,
            confidence=0.8,
            dimension_scores=[ds],
            flags=[],
            usable=True,
            recommendations=[],
            assessed_at=datetime.now(timezone.utc),
            metadata={},
        )
        result = summary.get_dimension(QualityDimension.COMPLETENESS)
        assert result is not None
        assert result.score == 0.9

        result_missing = summary.get_dimension(QualityDimension.ATMOSPHERIC)
        assert result_missing is None

    def test_to_dict(self):
        """Test to_dict conversion."""
        summary = QualitySummary(
            path=Path("test.tif"),
            overall_score=0.85,
            overall_level=QualityLevel.GOOD,
            confidence=0.9,
            dimension_scores=[],
            flags=[],
            usable=True,
            recommendations=["Consider this"],
            assessed_at=datetime.now(timezone.utc),
            metadata={"key": "value"},
        )
        d = summary.to_dict()
        assert d["path"] == "test.tif"
        assert d["overall_score"] == 0.85
        assert d["overall_level"] == "good"
        assert d["usable"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnrichmentIntegration:
    """Integration tests for enrichment module."""

    def test_overview_then_statistics(self, sample_3d_array):
        """Test generating overviews and then computing statistics."""
        # Generate overviews
        generator = OverviewGenerator(OverviewConfig(factors=[2, 4]))
        overviews = generator.generate_from_array(sample_3d_array, factors=[2, 4])

        # Compute statistics on each overview
        calculator = StatisticsCalculator()
        for i, overview in enumerate(overviews):
            stats = calculator.compute_from_array(overview)
            assert len(stats) == 3  # 3 bands
            for band_stat in stats:
                assert band_stat.valid_count > 0

    def test_statistics_informs_quality(self, sample_3d_array):
        """Test that statistics can inform quality assessment."""
        # Compute statistics
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(sample_3d_array)

        # Use statistics to inform quality check
        # For example, check if SNR is acceptable
        for band_stat in stats:
            snr = band_stat.snr
            # Random data should have reasonable SNR
            assert snr > 0

    def test_module_exports(self):
        """Test that all expected exports are available."""
        from core.data.ingestion.enrichment import (
            BandStatistics,
            DimensionScore,
            HistogramType,
            OverviewConfig,
            OverviewFormat,
            OverviewGenerator,
            OverviewLevel,
            OverviewResampling,
            OverviewResult,
            QualityAssessor,
            QualityConfig,
            QualityDimension,
            QualityFlag,
            QualityIssue,
            QualityLevel,
            QualitySummary,
            RasterStatistics,
            StatisticsCalculator,
            StatisticsConfig,
            assess_quality,
            compute_statistics,
            generate_overviews,
        )
        # All imports should succeed
        assert True


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_array(self):
        """Test with 1x1 array."""
        data = np.array([[5.0]], dtype=np.float32)
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(data)
        assert stats[0].min == 5.0
        assert stats[0].max == 5.0

    def test_very_large_values(self):
        """Test with very large values."""
        data = np.array([[1e38, 1e38]], dtype=np.float32)
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(data)
        assert np.isfinite(stats[0].mean)

    def test_all_nan_array(self):
        """Test with all NaN values."""
        data = np.full((10, 10), np.nan, dtype=np.float32)
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(data)
        assert stats[0].valid_count == 0
        assert np.isnan(stats[0].mean)

    def test_mixed_finite_and_inf(self):
        """Test with mixed finite and infinite values."""
        data = np.array([[1.0, np.inf, 3.0], [4.0, -np.inf, 6.0]], dtype=np.float32)
        calculator = StatisticsCalculator()
        stats = calculator.compute_from_array(data)
        assert stats[0].valid_count == 4
        assert stats[0].nodata_count == 2

    def test_overview_tiny_image(self):
        """Test overview generation for tiny image."""
        data = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        generator = OverviewGenerator(OverviewConfig(factors=[2]))
        overviews = generator.generate_from_array(data, factors=[2])
        assert overviews[0].shape == (1, 1, 1)

    def test_quality_zero_total_weight(self):
        """Test quality assessment with zero total weight."""
        assessor = QualityAssessor(
            QualityConfig(
                dimension_weights={
                    QualityDimension.COMPLETENESS: 0.0,
                    QualityDimension.SPATIAL: 0.0,
                }
            )
        )
        # Weights should be normalized
        assert sum(assessor.config.dimension_weights.values()) > 0

    def test_temporal_naive_datetime(self):
        """Test temporal assessment with naive datetime."""
        assessor = QualityAssessor()
        naive_time = datetime.now() - timedelta(days=5)  # No timezone
        score, flags, details = assessor._assess_temporal(naive_time)
        # Should handle gracefully
        assert 0.0 <= score <= 1.0
