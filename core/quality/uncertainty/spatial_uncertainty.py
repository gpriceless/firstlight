"""
Spatial Uncertainty Mapping for Quality Control.

Provides tools for generating and analyzing spatial maps of uncertainty,
enabling localized quality assessment and uncertainty visualization.

Key Concepts:
- Spatial uncertainty varies across the domain
- Local statistics capture neighborhood uncertainty
- Hotspot detection identifies high-uncertainty regions
- Spatial correlation affects uncertainty propagation

Features:
- Per-pixel uncertainty estimation
- Local uncertainty statistics (moving window)
- Uncertainty surface smoothing
- Hotspot and anomaly detection
- Spatial autocorrelation analysis
- Uncertainty zoning/regionalization

Example:
    from core.quality.uncertainty.spatial_uncertainty import (
        SpatialUncertaintyMapper,
        compute_local_uncertainty,
        detect_uncertainty_hotspots,
    )

    # Generate uncertainty map
    mapper = SpatialUncertaintyMapper()
    uncertainty_map = mapper.compute_uncertainty_surface(data, reference)

    # Find high-uncertainty regions
    hotspots = detect_uncertainty_hotspots(uncertainty_map)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class SpatialUncertaintyMethod(Enum):
    """Methods for computing spatial uncertainty."""
    LOCAL_VARIANCE = "local_variance"
    LOCAL_CV = "local_cv"
    RESIDUAL_MAP = "residual_map"
    ENSEMBLE_SPREAD = "ensemble_spread"
    BOOTSTRAP_LOCAL = "bootstrap_local"
    GEOSTATISTICAL = "geostatistical"


class SmoothingMethod(Enum):
    """Methods for smoothing uncertainty surfaces."""
    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    SAVITZKY_GOLAY = "savitzky_golay"
    ADAPTIVE = "adaptive"


class HotspotMethod(Enum):
    """Methods for hotspot detection."""
    THRESHOLD = "threshold"
    ZSCORE = "zscore"
    GETIS_ORD = "getis_ord"
    LOCAL_MORAN = "local_moran"


@dataclass
class SpatialUncertaintyConfig:
    """
    Configuration for spatial uncertainty mapping.

    Attributes:
        window_size: Size of local window (pixels)
        min_valid_fraction: Minimum fraction of valid pixels in window
        method: Method for computing uncertainty
        smoothing_method: Post-computation smoothing
        smoothing_sigma: Sigma for Gaussian smoothing
        threshold_percentile: Percentile for hotspot threshold
        min_cluster_size: Minimum pixels for hotspot cluster
        compute_autocorrelation: Whether to compute spatial autocorrelation
        lag_distance: Distance for autocorrelation (pixels)
    """
    window_size: int = 5
    min_valid_fraction: float = 0.3
    method: SpatialUncertaintyMethod = SpatialUncertaintyMethod.LOCAL_VARIANCE
    smoothing_method: SmoothingMethod = SmoothingMethod.GAUSSIAN
    smoothing_sigma: float = 1.0
    threshold_percentile: float = 95.0
    min_cluster_size: int = 10
    compute_autocorrelation: bool = False
    lag_distance: int = 1


@dataclass
class UncertaintySurface:
    """
    Spatial uncertainty surface with metadata.

    Attributes:
        uncertainty: 2D uncertainty array
        method: Method used for computation
        mean_uncertainty: Mean uncertainty value
        max_uncertainty: Maximum uncertainty value
        uncertainty_variance: Variance of uncertainty values
        autocorrelation: Spatial autocorrelation coefficient (optional)
        effective_resolution: Effective spatial resolution of uncertainty
        metadata: Additional metadata
    """
    uncertainty: np.ndarray
    method: str
    mean_uncertainty: float
    max_uncertainty: float
    uncertainty_variance: float
    autocorrelation: Optional[float] = None
    effective_resolution: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without large array)."""
        return {
            "shape": list(self.uncertainty.shape),
            "method": self.method,
            "mean_uncertainty": self.mean_uncertainty,
            "max_uncertainty": self.max_uncertainty,
            "uncertainty_variance": self.uncertainty_variance,
            "autocorrelation": self.autocorrelation,
            "effective_resolution": self.effective_resolution,
            "metadata": self.metadata,
        }


@dataclass
class UncertaintyHotspot:
    """
    A region of elevated uncertainty.

    Attributes:
        region_id: Unique identifier for the hotspot
        centroid: (row, col) centroid coordinates
        area_pixels: Number of pixels in the hotspot
        mean_uncertainty: Mean uncertainty in the hotspot
        max_uncertainty: Maximum uncertainty in the hotspot
        bounding_box: (min_row, min_col, max_row, max_col)
        severity: Severity level (1-5)
        mask: Boolean mask for the hotspot region
    """
    region_id: int
    centroid: Tuple[float, float]
    area_pixels: int
    mean_uncertainty: float
    max_uncertainty: float
    bounding_box: Tuple[int, int, int, int]
    severity: int
    mask: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without mask)."""
        return {
            "region_id": self.region_id,
            "centroid": list(self.centroid),
            "area_pixels": self.area_pixels,
            "mean_uncertainty": self.mean_uncertainty,
            "max_uncertainty": self.max_uncertainty,
            "bounding_box": list(self.bounding_box),
            "severity": self.severity,
        }


@dataclass
class HotspotAnalysis:
    """
    Results from hotspot detection.

    Attributes:
        hotspots: List of detected hotspots
        total_hotspot_area: Total area in hotspots
        hotspot_fraction: Fraction of domain in hotspots
        mean_hotspot_uncertainty: Mean uncertainty across all hotspots
        hotspot_label_map: Label map for all hotspots
        threshold_used: Threshold value used for detection
    """
    hotspots: List[UncertaintyHotspot]
    total_hotspot_area: int
    hotspot_fraction: float
    mean_hotspot_uncertainty: float
    hotspot_label_map: np.ndarray
    threshold_used: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "num_hotspots": len(self.hotspots),
            "total_hotspot_area": self.total_hotspot_area,
            "hotspot_fraction": self.hotspot_fraction,
            "mean_hotspot_uncertainty": self.mean_hotspot_uncertainty,
            "threshold_used": self.threshold_used,
            "hotspots": [h.to_dict() for h in self.hotspots],
        }


@dataclass
class LocalStatistics:
    """
    Local statistics computed in a window.

    Attributes:
        local_mean: Local mean array
        local_std: Local standard deviation array
        local_cv: Local coefficient of variation
        local_min: Local minimum
        local_max: Local maximum
        local_range: Local range (max - min)
        valid_fraction: Fraction of valid pixels per window
    """
    local_mean: np.ndarray
    local_std: np.ndarray
    local_cv: np.ndarray
    local_min: np.ndarray
    local_max: np.ndarray
    local_range: np.ndarray
    valid_fraction: np.ndarray


class SpatialUncertaintyMapper:
    """
    Computes spatial uncertainty maps from data.

    Provides methods for generating per-pixel uncertainty estimates
    using various local and global methods.
    """

    def __init__(self, config: Optional[SpatialUncertaintyConfig] = None):
        """
        Initialize spatial uncertainty mapper.

        Args:
            config: Mapping configuration
        """
        self.config = config or SpatialUncertaintyConfig()

    def compute_uncertainty_surface(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        ensemble: Optional[List[np.ndarray]] = None,
    ) -> UncertaintySurface:
        """
        Compute spatial uncertainty surface.

        Args:
            data: Input data array
            reference: Reference/truth data (for residual method)
            ensemble: Ensemble predictions (for ensemble spread method)

        Returns:
            UncertaintySurface with uncertainty values
        """
        method = self.config.method

        if method == SpatialUncertaintyMethod.RESIDUAL_MAP and reference is not None:
            uncertainty = self._compute_residual_uncertainty(data, reference)
        elif method == SpatialUncertaintyMethod.ENSEMBLE_SPREAD and ensemble is not None:
            uncertainty = self._compute_ensemble_uncertainty(ensemble)
        elif method == SpatialUncertaintyMethod.LOCAL_CV:
            uncertainty = self._compute_local_cv(data)
        else:
            # Default: local variance
            uncertainty = self._compute_local_variance(data)

        # Apply smoothing
        if self.config.smoothing_method != SmoothingMethod.NONE:
            uncertainty = self._smooth_surface(uncertainty)

        # Compute statistics
        valid_mask = np.isfinite(uncertainty)
        valid_uncertainty = uncertainty[valid_mask]

        mean_unc = float(np.mean(valid_uncertainty)) if len(valid_uncertainty) > 0 else np.nan
        max_unc = float(np.max(valid_uncertainty)) if len(valid_uncertainty) > 0 else np.nan
        var_unc = float(np.var(valid_uncertainty)) if len(valid_uncertainty) > 0 else np.nan

        # Compute autocorrelation if requested
        autocorr = None
        if self.config.compute_autocorrelation:
            autocorr = self._compute_autocorrelation(uncertainty)

        return UncertaintySurface(
            uncertainty=uncertainty,
            method=method.value,
            mean_uncertainty=mean_unc,
            max_uncertainty=max_unc,
            uncertainty_variance=var_unc,
            autocorrelation=autocorr,
            metadata={
                "window_size": self.config.window_size,
                "smoothing": self.config.smoothing_method.value,
            },
        )

    def _compute_local_variance(self, data: np.ndarray) -> np.ndarray:
        """Compute local variance in moving window."""
        window = self.config.window_size
        half = window // 2

        # Pad data for edge handling
        padded = np.pad(data.astype(np.float64), half, mode='reflect')

        # Use convolution for efficiency
        kernel = np.ones((window, window)) / (window * window)

        # Local mean
        from scipy import ndimage
        local_mean = ndimage.convolve(padded, kernel, mode='constant', cval=np.nan)

        # Local variance = E[X^2] - E[X]^2
        local_mean_sq = ndimage.convolve(padded ** 2, kernel, mode='constant', cval=np.nan)
        local_var = local_mean_sq - local_mean ** 2

        # Clip to valid region
        local_var = local_var[half:-half, half:-half] if half > 0 else local_var

        # Ensure non-negative (numerical precision)
        local_var = np.maximum(local_var, 0)

        # Standard deviation
        return np.sqrt(local_var)

    def _compute_local_cv(self, data: np.ndarray) -> np.ndarray:
        """Compute local coefficient of variation."""
        local_stats = self.compute_local_statistics(data)
        return local_stats.local_cv

    def _compute_residual_uncertainty(
        self,
        data: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Compute uncertainty from residuals."""
        residuals = np.abs(data.astype(np.float64) - reference.astype(np.float64))

        # Smooth residuals to get uncertainty surface
        window = self.config.window_size
        half = window // 2

        padded = np.pad(residuals, half, mode='reflect')
        kernel = np.ones((window, window)) / (window * window)

        from scipy import ndimage
        smoothed = ndimage.convolve(padded, kernel, mode='constant', cval=np.nan)

        return smoothed[half:-half, half:-half] if half > 0 else smoothed

    def _compute_ensemble_uncertainty(
        self,
        ensemble: List[np.ndarray],
    ) -> np.ndarray:
        """Compute uncertainty from ensemble spread."""
        if not ensemble:
            raise ValueError("Empty ensemble")

        stack = np.stack([e.astype(np.float64) for e in ensemble])
        return np.std(stack, axis=0, ddof=1)

    def _smooth_surface(self, uncertainty: np.ndarray) -> np.ndarray:
        """Apply smoothing to uncertainty surface."""
        from scipy import ndimage

        method = self.config.smoothing_method
        sigma = self.config.smoothing_sigma

        if method == SmoothingMethod.GAUSSIAN:
            return ndimage.gaussian_filter(uncertainty, sigma=sigma)
        elif method == SmoothingMethod.MEDIAN:
            size = max(3, int(2 * sigma + 1))
            return ndimage.median_filter(uncertainty, size=size)
        elif method == SmoothingMethod.ADAPTIVE:
            # Adaptive smoothing based on local gradient
            gradient = np.sqrt(ndimage.sobel(uncertainty, axis=0)**2 +
                             ndimage.sobel(uncertainty, axis=1)**2)
            adaptive_sigma = sigma / (1 + gradient / (np.nanmean(gradient) + 1e-10))
            # Can't do per-pixel sigma with standard filter, use approximation
            return ndimage.gaussian_filter(uncertainty, sigma=np.mean(adaptive_sigma))
        else:
            return uncertainty

    def _compute_autocorrelation(
        self,
        uncertainty: np.ndarray,
        lag: Optional[int] = None,
    ) -> float:
        """Compute spatial autocorrelation (Moran's I approximation)."""
        lag = lag or self.config.lag_distance

        valid_mask = np.isfinite(uncertainty)
        if np.sum(valid_mask) < 10:
            return np.nan

        # Simple lag-1 autocorrelation approximation
        data = uncertainty[valid_mask]
        mean_val = np.mean(data)
        variance = np.var(data)

        if variance < 1e-10:
            return 1.0  # Constant field

        # Horizontal lag correlation
        if uncertainty.shape[1] > lag:
            x1 = uncertainty[:, :-lag][valid_mask[:, :-lag]]
            x2 = uncertainty[:, lag:][valid_mask[:, lag:]]
            min_len = min(len(x1), len(x2))
            if min_len > 0:
                cov_h = np.mean((x1[:min_len] - mean_val) * (x2[:min_len] - mean_val))
            else:
                cov_h = 0
        else:
            cov_h = 0

        # Vertical lag correlation
        if uncertainty.shape[0] > lag:
            y1 = uncertainty[:-lag, :][valid_mask[:-lag, :]]
            y2 = uncertainty[lag:, :][valid_mask[lag:, :]]
            min_len = min(len(y1), len(y2))
            if min_len > 0:
                cov_v = np.mean((y1[:min_len] - mean_val) * (y2[:min_len] - mean_val))
            else:
                cov_v = 0
        else:
            cov_v = 0

        # Average correlation
        autocorr = (cov_h + cov_v) / (2 * variance + 1e-10)
        return float(np.clip(autocorr, -1, 1))

    def compute_local_statistics(self, data: np.ndarray) -> LocalStatistics:
        """
        Compute comprehensive local statistics.

        Args:
            data: Input data array

        Returns:
            LocalStatistics with all local metrics
        """
        from scipy import ndimage

        window = self.config.window_size
        half = window // 2

        # Pad data
        padded = np.pad(data.astype(np.float64), half, mode='reflect')
        kernel = np.ones((window, window)) / (window * window)

        # Valid pixel counting (for non-NaN handling)
        valid_mask = np.isfinite(padded).astype(np.float64)
        valid_count = ndimage.convolve(valid_mask, np.ones((window, window)), mode='constant', cval=0)
        valid_fraction = valid_count / (window * window)

        # Local mean (handling NaN)
        data_filled = np.where(np.isfinite(padded), padded, 0)
        local_sum = ndimage.convolve(data_filled, np.ones((window, window)), mode='constant', cval=0)
        local_mean = np.where(valid_count > 0, local_sum / valid_count, np.nan)

        # Local variance
        sq_diff = (data_filled - local_mean) ** 2
        sq_diff = np.where(np.isfinite(padded), sq_diff, 0)
        local_var_sum = ndimage.convolve(sq_diff, np.ones((window, window)), mode='constant', cval=0)
        local_var = np.where(valid_count > 1, local_var_sum / (valid_count - 1), np.nan)
        local_std = np.sqrt(np.maximum(local_var, 0))

        # Local CV
        local_cv = np.where(np.abs(local_mean) > 1e-10, local_std / np.abs(local_mean), np.nan)

        # Local min/max using generic_filter
        local_min = ndimage.minimum_filter(padded, size=window, mode='constant', cval=np.inf)
        local_max = ndimage.maximum_filter(padded, size=window, mode='constant', cval=-np.inf)
        local_range = local_max - local_min

        # Clip to valid region
        def clip(arr):
            return arr[half:-half, half:-half] if half > 0 else arr

        return LocalStatistics(
            local_mean=clip(local_mean),
            local_std=clip(local_std),
            local_cv=clip(local_cv),
            local_min=clip(local_min),
            local_max=clip(local_max),
            local_range=clip(local_range),
            valid_fraction=clip(valid_fraction),
        )


class HotspotDetector:
    """
    Detects regions of elevated uncertainty.

    Identifies spatial clusters where uncertainty exceeds
    thresholds or is statistically significant.
    """

    def __init__(self, config: Optional[SpatialUncertaintyConfig] = None):
        """
        Initialize hotspot detector.

        Args:
            config: Detection configuration
        """
        self.config = config or SpatialUncertaintyConfig()

    def detect_hotspots(
        self,
        uncertainty: np.ndarray,
        method: Optional[HotspotMethod] = None,
    ) -> HotspotAnalysis:
        """
        Detect uncertainty hotspots.

        Args:
            uncertainty: Uncertainty surface
            method: Detection method (default: threshold)

        Returns:
            HotspotAnalysis with detected hotspots
        """
        method = method or HotspotMethod.THRESHOLD

        if method == HotspotMethod.ZSCORE:
            hotspot_mask, threshold = self._detect_zscore(uncertainty)
        elif method == HotspotMethod.GETIS_ORD:
            hotspot_mask, threshold = self._detect_getis_ord(uncertainty)
        else:
            # Default: threshold
            hotspot_mask, threshold = self._detect_threshold(uncertainty)

        # Label connected components
        from scipy import ndimage
        labeled_map, num_features = ndimage.label(hotspot_mask)

        # Extract hotspot properties
        hotspots = []
        for region_id in range(1, num_features + 1):
            region_mask = labeled_map == region_id
            area = int(np.sum(region_mask))

            if area < self.config.min_cluster_size:
                continue

            # Get region properties
            rows, cols = np.where(region_mask)
            centroid = (float(np.mean(rows)), float(np.mean(cols)))
            bbox = (int(np.min(rows)), int(np.min(cols)),
                   int(np.max(rows)), int(np.max(cols)))

            region_uncertainty = uncertainty[region_mask]
            mean_unc = float(np.nanmean(region_uncertainty))
            max_unc = float(np.nanmax(region_uncertainty))

            # Severity based on percentile of mean uncertainty
            valid_unc = uncertainty[np.isfinite(uncertainty)]
            if len(valid_unc) > 0:
                severity_pct = np.searchsorted(np.sort(valid_unc), mean_unc) / len(valid_unc) * 100
                severity = min(5, max(1, int(severity_pct / 20) + 1))
            else:
                severity = 3

            hotspots.append(UncertaintyHotspot(
                region_id=region_id,
                centroid=centroid,
                area_pixels=area,
                mean_uncertainty=mean_unc,
                max_uncertainty=max_unc,
                bounding_box=bbox,
                severity=severity,
                mask=region_mask,
            ))

        # Compute summary statistics
        total_area = sum(h.area_pixels for h in hotspots)
        total_pixels = np.sum(np.isfinite(uncertainty))
        hotspot_fraction = total_area / total_pixels if total_pixels > 0 else 0.0

        if hotspots:
            mean_hotspot_unc = np.mean([h.mean_uncertainty for h in hotspots])
        else:
            mean_hotspot_unc = 0.0

        return HotspotAnalysis(
            hotspots=hotspots,
            total_hotspot_area=total_area,
            hotspot_fraction=hotspot_fraction,
            mean_hotspot_uncertainty=float(mean_hotspot_unc),
            hotspot_label_map=labeled_map,
            threshold_used=threshold,
        )

    def _detect_threshold(self, uncertainty: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect hotspots using percentile threshold."""
        valid = uncertainty[np.isfinite(uncertainty)]
        if len(valid) == 0:
            return np.zeros_like(uncertainty, dtype=bool), np.nan

        threshold = np.percentile(valid, self.config.threshold_percentile)
        mask = uncertainty > threshold

        return mask, float(threshold)

    def _detect_zscore(self, uncertainty: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect hotspots using z-score threshold."""
        valid_mask = np.isfinite(uncertainty)
        valid = uncertainty[valid_mask]

        if len(valid) == 0:
            return np.zeros_like(uncertainty, dtype=bool), np.nan

        mean_val = np.mean(valid)
        std_val = np.std(valid)

        if std_val < 1e-10:
            return np.zeros_like(uncertainty, dtype=bool), np.nan

        z_threshold = 2.0  # 2 standard deviations
        threshold = mean_val + z_threshold * std_val
        mask = np.zeros_like(uncertainty, dtype=bool)
        mask[valid_mask] = uncertainty[valid_mask] > threshold

        return mask, float(threshold)

    def _detect_getis_ord(self, uncertainty: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect hotspots using Getis-Ord Gi* statistic.

        This is a local indicator of spatial autocorrelation.
        """
        from scipy import ndimage

        window = self.config.window_size
        half = window // 2

        valid_mask = np.isfinite(uncertainty)
        valid = uncertainty[valid_mask]

        if len(valid) < 10:
            return np.zeros_like(uncertainty, dtype=bool), np.nan

        # Global statistics
        global_mean = np.mean(valid)
        global_std = np.std(valid)
        n = len(valid)

        if global_std < 1e-10:
            return np.zeros_like(uncertainty, dtype=bool), np.nan

        # Local sum in window
        data_filled = np.where(np.isfinite(uncertainty), uncertainty, 0)
        kernel = np.ones((window, window))
        local_sum = ndimage.convolve(data_filled, kernel, mode='constant', cval=0)
        local_count = ndimage.convolve(valid_mask.astype(float), kernel, mode='constant', cval=0)

        # Gi* statistic
        w_count = window * window
        expected = w_count * global_mean
        std_gi = global_std * np.sqrt((n * w_count - w_count**2) / (n - 1 + 1e-10))

        gi_star = np.where(std_gi > 1e-10,
                          (local_sum - expected) / std_gi,
                          0)

        # Significant hotspots at z > 1.96 (95% confidence)
        z_threshold = 1.96
        mask = gi_star > z_threshold

        # Threshold in original units
        threshold = global_mean + z_threshold * global_std

        return mask, float(threshold)


class SpatialUncertaintyAnalyzer:
    """
    Comprehensive spatial uncertainty analysis.

    Combines mapping and hotspot detection with additional
    analysis capabilities.
    """

    def __init__(self, config: Optional[SpatialUncertaintyConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or SpatialUncertaintyConfig()
        self.mapper = SpatialUncertaintyMapper(config)
        self.detector = HotspotDetector(config)

    def analyze(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        ensemble: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Perform complete spatial uncertainty analysis.

        Args:
            data: Input data
            reference: Reference data (optional)
            ensemble: Ensemble predictions (optional)

        Returns:
            Dictionary with analysis results
        """
        # Compute uncertainty surface
        surface = self.mapper.compute_uncertainty_surface(data, reference, ensemble)

        # Detect hotspots
        hotspots = self.detector.detect_hotspots(surface.uncertainty)

        # Compute zonal statistics
        zonal_stats = self._compute_zonal_statistics(surface.uncertainty)

        # Compute gradient analysis
        gradient_analysis = self._analyze_uncertainty_gradient(surface.uncertainty)

        return {
            "surface": surface,
            "hotspots": hotspots,
            "zonal_statistics": zonal_stats,
            "gradient_analysis": gradient_analysis,
            "summary": {
                "mean_uncertainty": surface.mean_uncertainty,
                "max_uncertainty": surface.max_uncertainty,
                "num_hotspots": len(hotspots.hotspots),
                "hotspot_fraction": hotspots.hotspot_fraction,
                "autocorrelation": surface.autocorrelation,
            },
        }

    def _compute_zonal_statistics(
        self,
        uncertainty: np.ndarray,
        num_zones: int = 5,
    ) -> Dict[str, Any]:
        """Compute statistics for uncertainty zones."""
        valid = uncertainty[np.isfinite(uncertainty)]
        if len(valid) == 0:
            return {"zones": [], "breaks": []}

        # Define zones by quantiles
        breaks = np.percentile(valid, np.linspace(0, 100, num_zones + 1))

        zones = []
        for i in range(num_zones):
            lower = breaks[i]
            upper = breaks[i + 1] if i < num_zones - 1 else np.inf
            zone_mask = (uncertainty >= lower) & (uncertainty < upper)
            zone_values = uncertainty[zone_mask]

            zones.append({
                "zone_id": i + 1,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "area_pixels": int(np.sum(zone_mask)),
                "mean_uncertainty": float(np.mean(zone_values)) if len(zone_values) > 0 else np.nan,
                "std_uncertainty": float(np.std(zone_values)) if len(zone_values) > 0 else np.nan,
            })

        return {
            "zones": zones,
            "breaks": breaks.tolist(),
        }

    def _analyze_uncertainty_gradient(
        self,
        uncertainty: np.ndarray,
    ) -> Dict[str, Any]:
        """Analyze spatial gradient of uncertainty."""
        from scipy import ndimage

        # Compute gradients
        grad_y = ndimage.sobel(uncertainty, axis=0)
        grad_x = ndimage.sobel(uncertainty, axis=1)
        gradient_mag = np.sqrt(grad_y**2 + grad_x**2)
        gradient_dir = np.arctan2(grad_y, grad_x)

        valid_mag = gradient_mag[np.isfinite(gradient_mag)]

        return {
            "mean_gradient": float(np.mean(valid_mag)) if len(valid_mag) > 0 else np.nan,
            "max_gradient": float(np.max(valid_mag)) if len(valid_mag) > 0 else np.nan,
            "gradient_std": float(np.std(valid_mag)) if len(valid_mag) > 0 else np.nan,
            "dominant_direction": float(np.nanmean(gradient_dir)) if len(valid_mag) > 0 else np.nan,
        }


# Convenience functions

def compute_local_uncertainty(
    data: np.ndarray,
    window_size: int = 5,
    method: str = "variance",
) -> np.ndarray:
    """
    Compute local uncertainty map.

    Args:
        data: Input data array
        window_size: Size of local window
        method: "variance", "cv", or "range"

    Returns:
        Uncertainty array
    """
    config = SpatialUncertaintyConfig(
        window_size=window_size,
        method=SpatialUncertaintyMethod.LOCAL_VARIANCE if method == "variance"
               else SpatialUncertaintyMethod.LOCAL_CV,
        smoothing_method=SmoothingMethod.NONE,
    )
    mapper = SpatialUncertaintyMapper(config)

    if method == "range":
        stats = mapper.compute_local_statistics(data)
        return stats.local_range
    else:
        surface = mapper.compute_uncertainty_surface(data)
        return surface.uncertainty


def compute_ensemble_uncertainty(
    ensemble: List[np.ndarray],
    smooth: bool = True,
) -> np.ndarray:
    """
    Compute uncertainty from ensemble spread.

    Args:
        ensemble: List of ensemble members
        smooth: Whether to smooth the result

    Returns:
        Uncertainty array
    """
    config = SpatialUncertaintyConfig(
        method=SpatialUncertaintyMethod.ENSEMBLE_SPREAD,
        smoothing_method=SmoothingMethod.GAUSSIAN if smooth else SmoothingMethod.NONE,
    )
    mapper = SpatialUncertaintyMapper(config)
    surface = mapper.compute_uncertainty_surface(None, None, ensemble)
    return surface.uncertainty


def detect_uncertainty_hotspots(
    uncertainty: np.ndarray,
    threshold_percentile: float = 95.0,
    min_cluster_size: int = 10,
) -> HotspotAnalysis:
    """
    Detect regions of elevated uncertainty.

    Args:
        uncertainty: Uncertainty array
        threshold_percentile: Percentile threshold for hotspots
        min_cluster_size: Minimum pixels for a hotspot

    Returns:
        HotspotAnalysis with detected hotspots
    """
    config = SpatialUncertaintyConfig(
        threshold_percentile=threshold_percentile,
        min_cluster_size=min_cluster_size,
    )
    detector = HotspotDetector(config)
    return detector.detect_hotspots(uncertainty)


def compute_spatial_autocorrelation(
    data: np.ndarray,
    lag: int = 1,
) -> float:
    """
    Compute spatial autocorrelation.

    Args:
        data: Data array
        lag: Lag distance in pixels

    Returns:
        Autocorrelation coefficient (-1 to 1)
    """
    config = SpatialUncertaintyConfig(
        compute_autocorrelation=True,
        lag_distance=lag,
    )
    mapper = SpatialUncertaintyMapper(config)
    return mapper._compute_autocorrelation(data, lag)
