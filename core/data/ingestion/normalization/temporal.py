"""
Temporal Alignment and Time Handling for Data Normalization.

Provides tools for temporal normalization of geospatial data including
time zone handling, temporal resampling, and time series alignment.

Key Capabilities:
- Time zone conversion and standardization to UTC
- Temporal resampling and aggregation
- Time series alignment across datasets
- Temporal interpolation and gap filling
- Date range operations and filtering
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TemporalResolution(Enum):
    """Standard temporal resolutions for aggregation."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class AggregationMethod(Enum):
    """Methods for temporal aggregation."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    FIRST = "first"
    LAST = "last"
    NEAREST = "nearest"


class InterpolationMethod(Enum):
    """Methods for temporal interpolation."""

    LINEAR = "linear"
    NEAREST = "nearest"
    PREVIOUS = "previous"
    NEXT = "next"
    CUBIC = "cubic"
    SPLINE = "spline"


@dataclass
class TimeRange:
    """
    A time range with start and end timestamps.

    Attributes:
        start: Start time (inclusive)
        end: End time (exclusive)
        timezone: Timezone (default UTC)
    """

    start: datetime
    end: datetime
    tz: timezone = timezone.utc

    def __post_init__(self):
        """Validate and normalize time range."""
        # Ensure timestamps have timezone
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=self.tz)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=self.tz)

        # Convert to specified timezone
        self.start = self.start.astimezone(self.tz)
        self.end = self.end.astimezone(self.tz)

        if self.start >= self.end:
            raise ValueError("Start must be before end")

    @property
    def duration(self) -> timedelta:
        """Duration of the time range."""
        return self.end - self.start

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration.total_seconds()

    @property
    def midpoint(self) -> datetime:
        """Midpoint of the time range."""
        return self.start + self.duration / 2

    def contains(self, dt: datetime) -> bool:
        """Check if a datetime is within this range."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.tz)
        return self.start <= dt < self.end

    def overlaps(self, other: "TimeRange") -> bool:
        """Check if this range overlaps another."""
        return self.start < other.end and other.start < self.end

    def intersection(self, other: "TimeRange") -> Optional["TimeRange"]:
        """Get intersection with another range, or None if no overlap."""
        if not self.overlaps(other):
            return None
        return TimeRange(
            start=max(self.start, other.start),
            end=min(self.end, other.end),
            tz=self.tz,
        )

    def union(self, other: "TimeRange") -> "TimeRange":
        """Get union with another range (may include gap)."""
        return TimeRange(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            tz=self.tz,
        )

    def split(self, n: int) -> List["TimeRange"]:
        """Split into n equal parts."""
        if n < 1:
            raise ValueError("n must be positive")
        step = self.duration / n
        ranges = []
        for i in range(n):
            ranges.append(
                TimeRange(
                    start=self.start + i * step,
                    end=self.start + (i + 1) * step,
                    tz=self.tz,
                )
            )
        return ranges

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_center(
        cls,
        center: datetime,
        duration: timedelta,
        tz: timezone = timezone.utc,
    ) -> "TimeRange":
        """Create a time range centered on a datetime."""
        half = duration / 2
        return cls(start=center - half, end=center + half, tz=tz)


@dataclass
class TemporalSample:
    """
    A single temporal sample with timestamp and data.

    Attributes:
        timestamp: Sample timestamp
        data: Sample data (scalar, array, or dict)
        quality: Quality indicator (0-1)
        source: Source identifier
        metadata: Additional metadata
    """

    timestamp: datetime
    data: Any
    quality: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure timestamp has timezone."""
        if self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "data": self.data if isinstance(self.data, (int, float, str)) else str(type(self.data)),
            "quality": self.quality,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class TemporalAlignmentConfig:
    """
    Configuration for temporal alignment.

    Attributes:
        reference_time: Reference time for alignment (or None for auto)
        resolution: Target temporal resolution
        aggregation: Aggregation method for downsampling
        interpolation: Interpolation method for upsampling
        fill_gaps: Whether to fill temporal gaps
        max_gap: Maximum gap to fill (None for unlimited)
        tolerance: Tolerance for time matching
    """

    reference_time: Optional[datetime] = None
    resolution: Optional[timedelta] = None
    aggregation: AggregationMethod = AggregationMethod.MEAN
    interpolation: InterpolationMethod = InterpolationMethod.LINEAR
    fill_gaps: bool = True
    max_gap: Optional[timedelta] = None
    tolerance: timedelta = timedelta(seconds=1)


@dataclass
class AlignmentResult:
    """Result from temporal alignment operation."""

    samples: List[TemporalSample]
    time_range: TimeRange
    resolution: timedelta
    num_original: int
    num_interpolated: int
    gaps_filled: int
    quality_stats: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_samples": len(self.samples),
            "time_range": self.time_range.to_dict(),
            "resolution_seconds": self.resolution.total_seconds(),
            "num_original": self.num_original,
            "num_interpolated": self.num_interpolated,
            "gaps_filled": self.gaps_filled,
            "quality_stats": self.quality_stats,
        }


class TimestampHandler:
    """
    Handles timestamp parsing, conversion, and standardization.

    Standardizes timestamps to UTC and handles various input formats.

    Example:
        handler = TimestampHandler()

        # Parse various formats
        dt = handler.parse("2024-03-15T10:30:00Z")
        dt = handler.parse("2024-03-15 10:30:00+05:30")
        dt = handler.parse(1710499800)  # Unix timestamp

        # Convert to UTC
        utc_dt = handler.to_utc(local_dt)
    """

    # Common timestamp formats
    FORMATS = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y%m%dT%H%M%S",
        "%Y%m%d",
    ]

    def parse(
        self,
        value: Union[str, int, float, datetime],
        default_tz: timezone = timezone.utc,
    ) -> datetime:
        """
        Parse a timestamp from various formats.

        Args:
            value: Timestamp value (string, unix timestamp, or datetime)
            default_tz: Default timezone if not specified

        Returns:
            Datetime in UTC

        Raises:
            ValueError: If timestamp cannot be parsed
        """
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=default_tz)
            return value.astimezone(timezone.utc)

        if isinstance(value, (int, float)):
            # Unix timestamp
            return datetime.fromtimestamp(value, tz=timezone.utc)

        if isinstance(value, str):
            # Try ISO format first (handles most cases)
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc)
            except ValueError:
                pass

            # Try other formats
            for fmt in self.FORMATS:
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=default_tz)
                    return dt.astimezone(timezone.utc)
                except ValueError:
                    continue

            raise ValueError(f"Cannot parse timestamp: {value}")

        raise ValueError(f"Unsupported timestamp type: {type(value)}")

    def to_utc(self, dt: datetime) -> datetime:
        """
        Convert datetime to UTC.

        Args:
            dt: Datetime (with or without timezone)

        Returns:
            Datetime in UTC
        """
        if dt.tzinfo is None:
            # Assume UTC for naive datetimes
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def format_iso(self, dt: datetime) -> str:
        """
        Format datetime as ISO 8601 string.

        Args:
            dt: Datetime to format

        Returns:
            ISO format string
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def format_filename(self, dt: datetime) -> str:
        """
        Format datetime for use in filenames.

        Args:
            dt: Datetime to format

        Returns:
            Filename-safe string (YYYYMMDD_HHMMSS)
        """
        utc_dt = self.to_utc(dt)
        return utc_dt.strftime("%Y%m%d_%H%M%S")


class TemporalResampler:
    """
    Resamples time series data to different temporal resolutions.

    Supports both upsampling (interpolation) and downsampling
    (aggregation) with configurable methods.

    Example:
        resampler = TemporalResampler()

        # Resample hourly data to daily
        daily = resampler.resample(
            samples=hourly_samples,
            target_resolution=timedelta(days=1),
            method=AggregationMethod.MEAN
        )
    """

    def __init__(self):
        """Initialize resampler."""
        self._aggregation_funcs: Dict[AggregationMethod, Callable] = {
            AggregationMethod.MEAN: np.nanmean,
            AggregationMethod.MEDIAN: np.nanmedian,
            AggregationMethod.MIN: np.nanmin,
            AggregationMethod.MAX: np.nanmax,
            AggregationMethod.SUM: np.nansum,
            AggregationMethod.COUNT: lambda x: np.sum(~np.isnan(x)),
            AggregationMethod.FIRST: lambda x: x[0] if len(x) > 0 else np.nan,
            AggregationMethod.LAST: lambda x: x[-1] if len(x) > 0 else np.nan,
        }

    def resample(
        self,
        samples: List[TemporalSample],
        target_resolution: timedelta,
        config: Optional[TemporalAlignmentConfig] = None,
    ) -> List[TemporalSample]:
        """
        Resample time series to target resolution.

        Args:
            samples: Input temporal samples
            target_resolution: Target resolution
            config: Alignment configuration

        Returns:
            Resampled samples
        """
        if not samples:
            return []

        config = config or TemporalAlignmentConfig()

        # Sort by timestamp
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)

        # Determine source resolution (median time difference)
        if len(sorted_samples) > 1:
            diffs = [
                (sorted_samples[i + 1].timestamp - sorted_samples[i].timestamp).total_seconds()
                for i in range(len(sorted_samples) - 1)
            ]
            source_resolution = timedelta(seconds=np.median(diffs))
        else:
            source_resolution = target_resolution

        # Determine if upsampling or downsampling
        if target_resolution < source_resolution:
            return self._upsample(sorted_samples, target_resolution, config)
        elif target_resolution > source_resolution:
            return self._downsample(sorted_samples, target_resolution, config)
        else:
            return sorted_samples

    def _downsample(
        self,
        samples: List[TemporalSample],
        target_resolution: timedelta,
        config: TemporalAlignmentConfig,
    ) -> List[TemporalSample]:
        """Aggregate samples to lower resolution."""
        if not samples:
            return []

        # Determine time bins
        start_time = self._floor_time(samples[0].timestamp, target_resolution)
        end_time = samples[-1].timestamp

        result = []
        current_bin_start = start_time

        while current_bin_start < end_time:
            current_bin_end = current_bin_start + target_resolution

            # Get samples in this bin
            bin_samples = [
                s for s in samples
                if current_bin_start <= s.timestamp < current_bin_end
            ]

            if bin_samples:
                # Aggregate data
                aggregated = self._aggregate_samples(bin_samples, config.aggregation)

                # Use bin center as timestamp
                bin_center = current_bin_start + target_resolution / 2

                result.append(
                    TemporalSample(
                        timestamp=bin_center,
                        data=aggregated,
                        quality=np.mean([s.quality for s in bin_samples]),
                        source="aggregated",
                        metadata={
                            "method": config.aggregation.value,
                            "count": len(bin_samples),
                        },
                    )
                )

            current_bin_start = current_bin_end

        return result

    def _upsample(
        self,
        samples: List[TemporalSample],
        target_resolution: timedelta,
        config: TemporalAlignmentConfig,
    ) -> List[TemporalSample]:
        """Interpolate samples to higher resolution."""
        if len(samples) < 2:
            return samples

        start_time = samples[0].timestamp
        end_time = samples[-1].timestamp

        result = []
        current_time = start_time

        while current_time <= end_time:
            # Find surrounding samples
            interpolated = self._interpolate_at(
                current_time, samples, config.interpolation
            )

            if interpolated is not None:
                result.append(
                    TemporalSample(
                        timestamp=current_time,
                        data=interpolated,
                        quality=0.9,  # Slightly lower quality for interpolated
                        source="interpolated",
                        metadata={"method": config.interpolation.value},
                    )
                )

            current_time += target_resolution

        return result

    def _aggregate_samples(
        self,
        samples: List[TemporalSample],
        method: AggregationMethod,
    ) -> Any:
        """Aggregate sample data using specified method."""
        if not samples:
            return None

        # Extract data values
        data_values = [s.data for s in samples]

        # Handle numpy arrays
        if isinstance(data_values[0], np.ndarray):
            stacked = np.stack(data_values)
            func = self._aggregation_funcs.get(method, np.nanmean)
            return func(stacked, axis=0)

        # Handle scalars
        try:
            values = np.array(data_values, dtype=float)
            func = self._aggregation_funcs.get(method, np.nanmean)
            return float(func(values))
        except (TypeError, ValueError):
            # For non-numeric, use first/last
            if method == AggregationMethod.LAST:
                return data_values[-1]
            return data_values[0]

    def _interpolate_at(
        self,
        target_time: datetime,
        samples: List[TemporalSample],
        method: InterpolationMethod,
    ) -> Any:
        """Interpolate value at a specific time."""
        # Find surrounding samples
        before = None
        after = None

        for i, s in enumerate(samples):
            if s.timestamp <= target_time:
                before = s
            if s.timestamp >= target_time and after is None:
                after = s
                break

        if before is None or after is None:
            return None

        if before.timestamp == target_time:
            return before.data

        if method == InterpolationMethod.NEAREST:
            before_dist = (target_time - before.timestamp).total_seconds()
            after_dist = (after.timestamp - target_time).total_seconds()
            return before.data if before_dist <= after_dist else after.data

        elif method == InterpolationMethod.PREVIOUS:
            return before.data

        elif method == InterpolationMethod.NEXT:
            return after.data

        elif method == InterpolationMethod.LINEAR:
            # Linear interpolation
            total_duration = (after.timestamp - before.timestamp).total_seconds()
            if total_duration == 0:
                return before.data

            fraction = (target_time - before.timestamp).total_seconds() / total_duration

            if isinstance(before.data, np.ndarray):
                return before.data + fraction * (after.data - before.data)
            elif isinstance(before.data, (int, float)):
                return before.data + fraction * (after.data - before.data)
            else:
                return before.data

        else:
            return before.data

    def _floor_time(self, dt: datetime, resolution: timedelta) -> datetime:
        """Floor datetime to resolution boundary."""
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        seconds = (dt - epoch).total_seconds()
        resolution_seconds = resolution.total_seconds()
        if resolution_seconds <= 0:
            raise ValueError("Resolution must be positive")
        floored_seconds = (seconds // resolution_seconds) * resolution_seconds
        return epoch + timedelta(seconds=floored_seconds)


class TemporalAligner:
    """
    Aligns multiple time series to a common temporal reference.

    Handles datasets with different temporal resolutions and
    timestamps, producing aligned output suitable for fusion.

    Example:
        aligner = TemporalAligner()

        # Align multiple datasets
        aligned = aligner.align_datasets(
            datasets={
                "sentinel2": sentinel_samples,
                "sentinel1": sar_samples,
                "weather": weather_samples,
            },
            reference_times=target_times
        )
    """

    def __init__(self):
        """Initialize aligner."""
        self.resampler = TemporalResampler()
        self.timestamp_handler = TimestampHandler()

    def align_to_times(
        self,
        samples: List[TemporalSample],
        target_times: List[datetime],
        config: Optional[TemporalAlignmentConfig] = None,
    ) -> List[TemporalSample]:
        """
        Align samples to specific target times.

        Args:
            samples: Input samples
            target_times: Target timestamps
            config: Alignment configuration

        Returns:
            Aligned samples (one per target time)
        """
        if not samples or not target_times:
            return []

        config = config or TemporalAlignmentConfig()
        sorted_samples = sorted(samples, key=lambda s: s.timestamp)
        sorted_targets = sorted(target_times)

        result = []
        for target in sorted_targets:
            # Find nearest sample within tolerance or interpolate
            matched = self._find_nearest(sorted_samples, target, config.tolerance)

            if matched:
                # Use exact match
                result.append(
                    TemporalSample(
                        timestamp=target,
                        data=matched.data,
                        quality=matched.quality,
                        source=matched.source,
                        metadata={**matched.metadata, "aligned": True},
                    )
                )
            elif config.fill_gaps:
                # Interpolate
                interpolated = self.resampler._interpolate_at(
                    target, sorted_samples, config.interpolation
                )
                if interpolated is not None:
                    result.append(
                        TemporalSample(
                            timestamp=target,
                            data=interpolated,
                            quality=0.8,
                            source="interpolated",
                            metadata={
                                "method": config.interpolation.value,
                                "aligned": True,
                            },
                        )
                    )

        return result

    def align_datasets(
        self,
        datasets: Dict[str, List[TemporalSample]],
        reference_times: Optional[List[datetime]] = None,
        config: Optional[TemporalAlignmentConfig] = None,
    ) -> Dict[str, List[TemporalSample]]:
        """
        Align multiple datasets to common time reference.

        Args:
            datasets: Dictionary of dataset name to samples
            reference_times: Target times (auto-generated if None)
            config: Alignment configuration

        Returns:
            Dictionary of aligned datasets
        """
        if not datasets:
            return {}

        config = config or TemporalAlignmentConfig()

        # Generate reference times if not provided
        if reference_times is None:
            reference_times = self._generate_reference_times(datasets, config)

        # Align each dataset
        aligned = {}
        for name, samples in datasets.items():
            aligned[name] = self.align_to_times(samples, reference_times, config)

        return aligned

    def find_overlapping_times(
        self,
        datasets: Dict[str, List[TemporalSample]],
        tolerance: timedelta = timedelta(hours=1),
    ) -> List[datetime]:
        """
        Find times where all datasets have data.

        Args:
            datasets: Dictionary of datasets
            tolerance: Time matching tolerance

        Returns:
            List of overlapping times
        """
        if not datasets:
            return []

        # Get all unique times from first dataset
        first_name = list(datasets.keys())[0]
        first_samples = datasets[first_name]
        candidate_times = [s.timestamp for s in first_samples]

        # Filter to times where all datasets have coverage
        overlapping = []
        for target in candidate_times:
            has_all = True
            for name, samples in datasets.items():
                if name == first_name:
                    continue
                # Check if any sample is within tolerance
                has_nearby = any(
                    abs((s.timestamp - target).total_seconds()) <= tolerance.total_seconds()
                    for s in samples
                )
                if not has_nearby:
                    has_all = False
                    break
            if has_all:
                overlapping.append(target)

        return overlapping

    def _find_nearest(
        self,
        samples: List[TemporalSample],
        target: datetime,
        tolerance: timedelta,
    ) -> Optional[TemporalSample]:
        """Find nearest sample within tolerance."""
        best_sample = None
        best_dist = float("inf")

        for sample in samples:
            dist = abs((sample.timestamp - target).total_seconds())
            if dist < best_dist and dist <= tolerance.total_seconds():
                best_dist = dist
                best_sample = sample

        return best_sample

    def _generate_reference_times(
        self,
        datasets: Dict[str, List[TemporalSample]],
        config: TemporalAlignmentConfig,
    ) -> List[datetime]:
        """Generate reference times from datasets."""
        # Find overall time range
        all_times = []
        for samples in datasets.values():
            all_times.extend(s.timestamp for s in samples)

        if not all_times:
            return []

        start = min(all_times)
        end = max(all_times)

        # Determine resolution
        if config.resolution:
            resolution = config.resolution
        else:
            # Use median resolution from all datasets
            all_diffs = []
            for samples in datasets.values():
                sorted_s = sorted(samples, key=lambda s: s.timestamp)
                for i in range(len(sorted_s) - 1):
                    diff = (sorted_s[i + 1].timestamp - sorted_s[i].timestamp).total_seconds()
                    all_diffs.append(diff)
            if all_diffs:
                resolution = timedelta(seconds=np.median(all_diffs))
            else:
                resolution = timedelta(hours=1)

        # Generate times
        times = []
        current = start
        while current <= end:
            times.append(current)
            current += resolution

        return times


def generate_time_range(
    start: Union[str, datetime],
    end: Union[str, datetime],
    resolution: Optional[timedelta] = None,
) -> Generator[datetime, None, None]:
    """
    Generate timestamps within a range.

    Args:
        start: Start time
        end: End time
        resolution: Step size (default 1 hour)

    Yields:
        Timestamps in range
    """
    handler = TimestampHandler()
    start_dt = handler.parse(start)
    end_dt = handler.parse(end)

    if resolution is None:
        resolution = timedelta(hours=1)

    current = start_dt
    while current <= end_dt:
        yield current
        current += resolution


def align_samples(
    samples: List[TemporalSample],
    target_resolution: timedelta,
    method: str = "mean",
) -> List[TemporalSample]:
    """
    Convenience function to align samples to resolution.

    Args:
        samples: Input samples
        target_resolution: Target resolution
        method: Aggregation method name

    Returns:
        Aligned samples
    """
    config = TemporalAlignmentConfig(
        resolution=target_resolution,
        aggregation=AggregationMethod(method.lower()),
    )
    resampler = TemporalResampler()
    return resampler.resample(samples, target_resolution, config)


def parse_timestamp(value: Union[str, int, float, datetime]) -> datetime:
    """
    Parse a timestamp from various formats.

    Args:
        value: Timestamp in any supported format

    Returns:
        Datetime in UTC
    """
    handler = TimestampHandler()
    return handler.parse(value)
