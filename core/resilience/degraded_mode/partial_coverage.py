"""
Partial Coverage Handler.

Handles incomplete spatial coverage situations by:
- Mosaicking multiple partial acquisitions
- Interpolating/extrapolating missing areas
- Marking uncertainty in gap regions
- Tracking coverage percentages

When full spatial coverage is unavailable, this module provides
mechanisms to produce the best possible output while clearly
documenting areas of uncertainty.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


class GapFillMethod(Enum):
    """Methods for filling coverage gaps."""
    NONE = "none"                    # Leave gaps as nodata
    NEAREST = "nearest"              # Nearest neighbor interpolation
    LINEAR = "linear"                # Linear interpolation
    CUBIC = "cubic"                  # Cubic interpolation
    IDW = "idw"                      # Inverse distance weighting
    KRIGING = "kriging"              # Kriging interpolation
    EXTRAPOLATE = "extrapolate"      # Extrapolate from edges


class CoverageQuality(Enum):
    """Quality assessment of coverage regions."""
    OBSERVED = "observed"            # Direct observation, high confidence
    INTERPOLATED = "interpolated"    # Interpolated from neighbors
    EXTRAPOLATED = "extrapolated"    # Extrapolated beyond observations
    GAP = "gap"                      # No data, unfilled


@dataclass
class CoverageGap:
    """
    Represents a gap in spatial coverage.

    Attributes:
        gap_id: Unique identifier for this gap
        bounds: Bounding box [min_x, min_y, max_x, max_y]
        area_km2: Area of the gap in square kilometers
        mask: Boolean mask of gap pixels (True=gap)
        fill_method: Method used to fill (or None if unfilled)
        fill_confidence: Confidence in gap fill (0-1)
        reason: Why this gap exists
        metadata: Additional gap information
    """
    gap_id: str
    bounds: Tuple[float, float, float, float]
    area_km2: float
    mask: np.ndarray
    fill_method: Optional[GapFillMethod] = None
    fill_confidence: float = 0.0
    reason: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without mask array)."""
        return {
            "gap_id": self.gap_id,
            "bounds": self.bounds,
            "area_km2": round(self.area_km2, 2),
            "fill_method": self.fill_method.value if self.fill_method else None,
            "fill_confidence": round(self.fill_confidence, 3),
            "reason": self.reason,
            "pixel_count": int(np.sum(self.mask)) if self.mask is not None else 0,
            "metadata": self.metadata,
        }


@dataclass
class CoverageRegion:
    """
    Represents a region of spatial coverage.

    Attributes:
        region_id: Unique identifier
        bounds: Bounding box [min_x, min_y, max_x, max_y]
        quality: Quality assessment of this region
        confidence: Confidence score (0-1)
        source: Source of the data (acquisition ID, interpolation, etc.)
        timestamp: Acquisition or generation timestamp
        mask: Boolean mask of this region's pixels
    """
    region_id: str
    bounds: Tuple[float, float, float, float]
    quality: CoverageQuality
    confidence: float
    source: str
    timestamp: datetime
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without mask array)."""
        return {
            "region_id": self.region_id,
            "bounds": self.bounds,
            "quality": self.quality.value,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PartialAcquisition:
    """
    Represents a partial data acquisition.

    Attributes:
        acquisition_id: Unique identifier
        data: Raster data array
        bounds: Geospatial bounds [min_x, min_y, max_x, max_y]
        valid_mask: Boolean mask of valid (non-nodata) pixels
        timestamp: Acquisition timestamp
        quality_score: Overall quality score (0-1)
        sensor: Sensor name
        metadata: Additional acquisition metadata
    """
    acquisition_id: str
    data: np.ndarray
    bounds: Tuple[float, float, float, float]
    valid_mask: np.ndarray
    timestamp: datetime
    quality_score: float
    sensor: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def coverage_percent(self) -> float:
        """Calculate coverage percentage."""
        if self.valid_mask is None or self.valid_mask.size == 0:
            return 0.0
        return float(np.mean(self.valid_mask) * 100)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get data shape."""
        return self.data.shape


@dataclass
class CoverageMosaicResult:
    """
    Result of mosaicking partial acquisitions.

    Attributes:
        data: Mosaicked raster data
        quality_map: Per-pixel quality assessment
        confidence_map: Per-pixel confidence scores
        coverage_percent: Overall coverage percentage
        gaps: List of identified coverage gaps
        regions: List of coverage regions
        sources_used: List of acquisition IDs used
        fill_method: Gap fill method used
        metadata: Additional result metadata
    """
    data: np.ndarray
    quality_map: np.ndarray
    confidence_map: np.ndarray
    coverage_percent: float
    gaps: List[CoverageGap]
    regions: List[CoverageRegion]
    sources_used: List[str]
    fill_method: GapFillMethod
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_gaps(self) -> bool:
        """Check if there are unfilled gaps."""
        return any(g.fill_method is None for g in self.gaps)

    @property
    def gap_count(self) -> int:
        """Get number of gaps."""
        return len(self.gaps)

    @property
    def total_gap_area_km2(self) -> float:
        """Get total gap area in km2."""
        return sum(g.area_km2 for g in self.gaps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "coverage_percent": round(self.coverage_percent, 2),
            "gap_count": self.gap_count,
            "total_gap_area_km2": round(self.total_gap_area_km2, 2),
            "has_gaps": self.has_gaps,
            "gaps": [g.to_dict() for g in self.gaps],
            "regions": [r.to_dict() for r in self.regions],
            "sources_used": self.sources_used,
            "fill_method": self.fill_method.value,
            "data_shape": list(self.data.shape),
            "metadata": self.metadata,
        }


@dataclass
class PartialCoverageConfig:
    """
    Configuration for partial coverage handling.

    Attributes:
        min_coverage_for_output: Minimum coverage % to produce output
        gap_fill_method: Default method for filling gaps
        max_gap_fill_distance: Maximum distance (pixels) for interpolation
        quality_weight_temporal: Weight for temporal proximity in mosaicking
        quality_weight_quality: Weight for quality scores in mosaicking
        extrapolation_max_distance: Maximum extrapolation distance (pixels)
        mark_interpolated_uncertainty: Mark interpolated areas in uncertainty map
        pixel_size_m: Pixel size in meters for area calculations
    """
    min_coverage_for_output: float = 30.0
    gap_fill_method: GapFillMethod = GapFillMethod.LINEAR
    max_gap_fill_distance: int = 50
    quality_weight_temporal: float = 0.4
    quality_weight_quality: float = 0.6
    extrapolation_max_distance: int = 20
    mark_interpolated_uncertainty: bool = True
    pixel_size_m: float = 10.0


class PartialCoverageHandler:
    """
    Handles partial spatial coverage situations.

    Provides mosaicking, interpolation, and gap analysis for
    situations where complete spatial coverage is unavailable.

    Example:
        handler = PartialCoverageHandler()

        # Mosaic multiple partial acquisitions
        result = handler.mosaic_partial_acquisitions([
            PartialAcquisition(...),
            PartialAcquisition(...),
        ])

        if result.has_gaps:
            print(f"Warning: {result.gap_count} gaps remain")
            for gap in result.gaps:
                print(f"  Gap at {gap.bounds}: {gap.area_km2} km2")
    """

    def __init__(self, config: Optional[PartialCoverageConfig] = None):
        """
        Initialize the partial coverage handler.

        Args:
            config: Configuration options
        """
        self.config = config or PartialCoverageConfig()

    def mosaic_partial_acquisitions(
        self,
        acquisitions: List[PartialAcquisition],
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
        target_shape: Optional[Tuple[int, int]] = None,
        fill_gaps: bool = True,
    ) -> CoverageMosaicResult:
        """
        Mosaic multiple partial acquisitions into a single coverage.

        Args:
            acquisitions: List of partial acquisitions to mosaic
            target_bounds: Target output bounds (uses union if None)
            target_shape: Target output shape (uses max if None)
            fill_gaps: Whether to fill gaps with interpolation

        Returns:
            CoverageMosaicResult with mosaicked data and metadata
        """
        if not acquisitions:
            raise ValueError("No acquisitions provided")

        # Sort by quality and timestamp (prefer higher quality, more recent)
        sorted_acqs = sorted(
            acquisitions,
            key=lambda a: (a.quality_score, -a.timestamp.timestamp()),
            reverse=True,
        )

        # Determine output shape
        if target_shape is None:
            target_shape = max(a.shape[:2] for a in acquisitions)

        # Initialize output arrays
        height, width = target_shape
        output_data = np.full((height, width), np.nan, dtype=np.float32)
        quality_map = np.full((height, width), CoverageQuality.GAP.value, dtype=object)
        confidence_map = np.zeros((height, width), dtype=np.float32)
        coverage_mask = np.zeros((height, width), dtype=bool)

        sources_used = []
        regions = []

        # Mosaic acquisitions (priority to higher quality)
        for acq in sorted_acqs:
            # Resample if needed
            acq_data, acq_mask = self._resample_to_shape(
                acq.data, acq.valid_mask, target_shape
            )

            # Only fill where we don't have data yet
            fill_mask = acq_mask & ~coverage_mask

            if not np.any(fill_mask):
                continue

            output_data[fill_mask] = acq_data[fill_mask]
            quality_map[fill_mask] = CoverageQuality.OBSERVED.value
            confidence_map[fill_mask] = acq.quality_score
            coverage_mask[fill_mask] = True

            sources_used.append(acq.acquisition_id)

            # Create region record
            region = CoverageRegion(
                region_id=f"region_{len(regions)}",
                bounds=acq.bounds,
                quality=CoverageQuality.OBSERVED,
                confidence=acq.quality_score,
                source=acq.acquisition_id,
                timestamp=acq.timestamp,
                mask=fill_mask.copy(),
            )
            regions.append(region)

        # Identify gaps
        gap_mask = ~coverage_mask
        gaps = self._identify_gaps(gap_mask, target_bounds)

        # Fill gaps if requested
        if fill_gaps and np.any(gap_mask):
            output_data, quality_map, confidence_map, gaps = self._fill_gaps(
                output_data, coverage_mask, quality_map, confidence_map, gaps
            )

        # Calculate coverage percentage
        coverage_percent = float(np.mean(~np.isnan(output_data)) * 100)

        return CoverageMosaicResult(
            data=output_data,
            quality_map=quality_map,
            confidence_map=confidence_map,
            coverage_percent=coverage_percent,
            gaps=gaps,
            regions=regions,
            sources_used=sources_used,
            fill_method=self.config.gap_fill_method if fill_gaps else GapFillMethod.NONE,
            metadata={
                "acquisition_count": len(acquisitions),
                "target_shape": target_shape,
                "fill_gaps": fill_gaps,
            },
        )

    def analyze_coverage(
        self,
        data: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze spatial coverage of a dataset.

        Args:
            data: Raster data array
            valid_mask: Boolean mask of valid pixels (inferred if None)
            bounds: Geospatial bounds for area calculations

        Returns:
            Dictionary with coverage analysis results
        """
        if valid_mask is None:
            valid_mask = ~np.isnan(data) & np.isfinite(data)

        total_pixels = valid_mask.size
        valid_pixels = np.sum(valid_mask)
        gap_pixels = total_pixels - valid_pixels

        coverage_percent = (valid_pixels / total_pixels * 100) if total_pixels > 0 else 0

        # Calculate areas
        pixel_area_km2 = (self.config.pixel_size_m ** 2) / 1e6
        total_area_km2 = total_pixels * pixel_area_km2
        covered_area_km2 = valid_pixels * pixel_area_km2
        gap_area_km2 = gap_pixels * pixel_area_km2

        # Identify gap regions
        gaps = self._identify_gaps(~valid_mask, bounds)

        # Calculate gap statistics
        gap_sizes = [g.area_km2 for g in gaps]

        return {
            "coverage_percent": round(coverage_percent, 2),
            "total_pixels": int(total_pixels),
            "valid_pixels": int(valid_pixels),
            "gap_pixels": int(gap_pixels),
            "total_area_km2": round(total_area_km2, 2),
            "covered_area_km2": round(covered_area_km2, 2),
            "gap_area_km2": round(gap_area_km2, 2),
            "gap_count": len(gaps),
            "mean_gap_size_km2": round(np.mean(gap_sizes), 2) if gap_sizes else 0,
            "max_gap_size_km2": round(max(gap_sizes), 2) if gap_sizes else 0,
            "gaps": [g.to_dict() for g in gaps],
        }

    def interpolate_gaps(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        method: Optional[GapFillMethod] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate values in gap regions.

        Args:
            data: Raster data with gaps
            valid_mask: Boolean mask of valid pixels
            method: Interpolation method (uses config default if None)

        Returns:
            Tuple of (filled_data, confidence_map)
        """
        method = method or self.config.gap_fill_method

        if method == GapFillMethod.NONE:
            return data.copy(), np.where(valid_mask, 1.0, 0.0).astype(np.float32)

        filled = data.copy()
        confidence = np.where(valid_mask, 1.0, 0.0).astype(np.float32)
        gap_mask = ~valid_mask

        if not np.any(gap_mask):
            return filled, confidence

        if method == GapFillMethod.NEAREST:
            filled, confidence = self._fill_nearest(filled, valid_mask, gap_mask)
        elif method == GapFillMethod.LINEAR:
            filled, confidence = self._fill_linear(filled, valid_mask, gap_mask)
        elif method == GapFillMethod.IDW:
            filled, confidence = self._fill_idw(filled, valid_mask, gap_mask)
        else:
            # Default to nearest for unsupported methods
            filled, confidence = self._fill_nearest(filled, valid_mask, gap_mask)
            logger.warning(f"Unsupported fill method {method}, using nearest")

        return filled, confidence

    def _resample_to_shape(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resample data and mask to target shape."""
        from scipy import ndimage

        if data.shape[:2] == target_shape:
            return data, mask

        # Calculate zoom factors
        zoom_y = target_shape[0] / data.shape[0]
        zoom_x = target_shape[1] / data.shape[1]

        # Resample data with bilinear interpolation
        resampled_data = ndimage.zoom(data, (zoom_y, zoom_x), order=1)

        # Resample mask with nearest neighbor
        resampled_mask = ndimage.zoom(mask.astype(float), (zoom_y, zoom_x), order=0) > 0.5

        return resampled_data, resampled_mask

    def _identify_gaps(
        self,
        gap_mask: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]],
    ) -> List[CoverageGap]:
        """Identify individual gap regions using connected components."""
        from scipy import ndimage

        if not np.any(gap_mask):
            return []

        # Label connected components
        labeled, num_features = ndimage.label(gap_mask)

        gaps = []
        pixel_area_km2 = (self.config.pixel_size_m ** 2) / 1e6

        for i in range(1, num_features + 1):
            component_mask = labeled == i
            pixel_count = np.sum(component_mask)

            # Calculate bounds (in pixel coordinates)
            rows, cols = np.where(component_mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()

            # Convert to geographic bounds if available
            if bounds is not None:
                min_x, min_y, max_x, max_y = bounds
                height, width = gap_mask.shape
                gap_bounds = (
                    min_x + min_col / width * (max_x - min_x),
                    min_y + min_row / height * (max_y - min_y),
                    min_x + max_col / width * (max_x - min_x),
                    min_y + max_row / height * (max_y - min_y),
                )
            else:
                gap_bounds = (float(min_col), float(min_row), float(max_col), float(max_row))

            gap = CoverageGap(
                gap_id=f"gap_{i}",
                bounds=gap_bounds,
                area_km2=pixel_count * pixel_area_km2,
                mask=component_mask,
                reason="missing_data",
                metadata={"pixel_count": int(pixel_count)},
            )
            gaps.append(gap)

        return gaps

    def _fill_gaps(
        self,
        data: np.ndarray,
        coverage_mask: np.ndarray,
        quality_map: np.ndarray,
        confidence_map: np.ndarray,
        gaps: List[CoverageGap],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[CoverageGap]]:
        """Fill gaps using configured method."""
        filled_data = data.copy()
        updated_quality = quality_map.copy()
        updated_confidence = confidence_map.copy()
        updated_gaps = []

        for gap in gaps:
            gap_mask = gap.mask

            # Check if gap is within fillable distance
            if not self._is_fillable(gap_mask, coverage_mask):
                updated_gaps.append(gap)
                continue

            # Fill using configured method
            filled_region, region_confidence = self._interpolate_region(
                filled_data, coverage_mask, gap_mask
            )

            # Update arrays
            filled_data[gap_mask] = filled_region[gap_mask]
            updated_quality[gap_mask] = CoverageQuality.INTERPOLATED.value
            updated_confidence[gap_mask] = region_confidence[gap_mask]

            # Update gap record
            filled_gap = CoverageGap(
                gap_id=gap.gap_id,
                bounds=gap.bounds,
                area_km2=gap.area_km2,
                mask=gap.mask,
                fill_method=self.config.gap_fill_method,
                fill_confidence=float(np.mean(region_confidence[gap_mask])),
                reason=gap.reason,
                metadata=gap.metadata,
            )
            updated_gaps.append(filled_gap)

        return filled_data, updated_quality, updated_confidence, updated_gaps

    def _is_fillable(self, gap_mask: np.ndarray, coverage_mask: np.ndarray) -> bool:
        """Check if a gap can be filled based on proximity to valid data."""
        from scipy import ndimage

        # Calculate distance from gap to nearest valid pixel
        if not np.any(coverage_mask):
            return False

        distance = ndimage.distance_transform_edt(~coverage_mask)
        max_distance = np.max(distance[gap_mask])

        return max_distance <= self.config.max_gap_fill_distance

    def _interpolate_region(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        gap_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate values within a gap region."""
        return self.interpolate_gaps(data, valid_mask, self.config.gap_fill_method)

    def _fill_nearest(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        gap_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill gaps with nearest neighbor values."""
        from scipy import ndimage

        filled = data.copy()
        confidence = np.where(valid_mask, 1.0, 0.0).astype(np.float32)

        # Get indices of nearest valid pixels
        _, indices = ndimage.distance_transform_edt(
            ~valid_mask, return_indices=True
        )

        # Fill gaps with nearest values
        filled[gap_mask] = data[indices[0][gap_mask], indices[1][gap_mask]]

        # Calculate confidence based on distance
        distances = ndimage.distance_transform_edt(~valid_mask)
        max_dist = self.config.max_gap_fill_distance
        confidence[gap_mask] = np.clip(1.0 - distances[gap_mask] / max_dist, 0.1, 0.8)

        return filled, confidence

    def _fill_linear(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        gap_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill gaps with linear interpolation."""
        from scipy import interpolate, ndimage

        filled = data.copy()
        confidence = np.where(valid_mask, 1.0, 0.0).astype(np.float32)

        # Get coordinates of valid and gap pixels
        valid_points = np.array(np.where(valid_mask)).T
        gap_points = np.array(np.where(gap_mask)).T

        if len(valid_points) < 4 or len(gap_points) == 0:
            # Fall back to nearest if not enough points
            return self._fill_nearest(data, valid_mask, gap_mask)

        try:
            # Create interpolator
            values = data[valid_mask]
            interp = interpolate.LinearNDInterpolator(valid_points, values)

            # Interpolate gap values
            gap_values = interp(gap_points)

            # Fill where interpolation succeeded
            valid_interp = ~np.isnan(gap_values)
            gap_rows = gap_points[valid_interp, 0]
            gap_cols = gap_points[valid_interp, 1]
            filled[gap_rows, gap_cols] = gap_values[valid_interp]

            # Calculate confidence
            distances = ndimage.distance_transform_edt(~valid_mask)
            max_dist = self.config.max_gap_fill_distance
            confidence[gap_mask] = np.clip(1.0 - distances[gap_mask] / max_dist, 0.1, 0.7)

        except Exception as e:
            logger.warning(f"Linear interpolation failed: {e}, using nearest")
            return self._fill_nearest(data, valid_mask, gap_mask)

        return filled, confidence

    def _fill_idw(
        self,
        data: np.ndarray,
        valid_mask: np.ndarray,
        gap_mask: np.ndarray,
        power: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill gaps using inverse distance weighting."""
        from scipy import ndimage

        filled = data.copy()
        confidence = np.where(valid_mask, 1.0, 0.0).astype(np.float32)

        # Get valid pixel coordinates and values
        valid_points = np.array(np.where(valid_mask)).T
        valid_values = data[valid_mask]

        if len(valid_points) == 0:
            return filled, confidence

        # Get gap pixel coordinates
        gap_rows, gap_cols = np.where(gap_mask)

        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, len(gap_rows), batch_size):
            batch_rows = gap_rows[i:i + batch_size]
            batch_cols = gap_cols[i:i + batch_size]

            for j, (row, col) in enumerate(zip(batch_rows, batch_cols)):
                # Calculate distances to all valid points
                distances = np.sqrt(
                    (valid_points[:, 0] - row) ** 2 +
                    (valid_points[:, 1] - col) ** 2
                )

                # Use only nearby points
                max_dist = self.config.max_gap_fill_distance
                nearby_mask = distances <= max_dist

                if not np.any(nearby_mask):
                    # No nearby points, skip
                    continue

                nearby_dist = distances[nearby_mask]
                nearby_values = valid_values[nearby_mask]

                # Handle exact matches
                if np.any(nearby_dist == 0):
                    filled[row, col] = nearby_values[nearby_dist == 0][0]
                    confidence[row, col] = 1.0
                else:
                    # IDW calculation
                    weights = 1.0 / (nearby_dist ** power)
                    filled[row, col] = np.sum(weights * nearby_values) / np.sum(weights)
                    confidence[row, col] = np.clip(1.0 - np.min(nearby_dist) / max_dist, 0.1, 0.6)

        return filled, confidence


def mosaic_partial_coverage(
    acquisitions: List[PartialAcquisition],
    fill_gaps: bool = True,
    config: Optional[PartialCoverageConfig] = None,
) -> CoverageMosaicResult:
    """
    Convenience function to mosaic partial acquisitions.

    Args:
        acquisitions: List of partial acquisitions
        fill_gaps: Whether to fill gaps with interpolation
        config: Configuration options

    Returns:
        CoverageMosaicResult with mosaicked data

    Example:
        result = mosaic_partial_coverage([
            PartialAcquisition(...),
            PartialAcquisition(...),
        ])
        print(f"Coverage: {result.coverage_percent:.1f}%")
    """
    handler = PartialCoverageHandler(config)
    return handler.mosaic_partial_acquisitions(acquisitions, fill_gaps=fill_gaps)
