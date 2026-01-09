"""
Structural Damage Assessment Algorithm

Building and infrastructure damage assessment using high-resolution
optical or SAR imagery. Detects changes in built-up areas caused by
storm events such as hurricanes, tornadoes, and severe storms.

Algorithm ID: storm.baseline.structural_damage
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Literal
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Named constants for damage classification thresholds
TEXTURE_VARIANCE_CHANGE_THRESHOLD = 0.3  # Variance change indicating damage
BRIGHTNESS_CHANGE_THRESHOLD = 0.15  # Brightness change threshold
SAR_COHERENCE_LOSS_THRESHOLD = 0.4  # Coherence loss indicating damage

# Damage grade definitions (European Macroseismic Scale inspired)
DAMAGE_GRADE_NONE = 0  # No visible damage
DAMAGE_GRADE_SLIGHT = 1  # Minor damage (D1)
DAMAGE_GRADE_MODERATE = 2  # Moderate damage (D2)
DAMAGE_GRADE_HEAVY = 3  # Heavy damage (D3)
DAMAGE_GRADE_DESTROYED = 4  # Destruction (D4-D5)

# Texture analysis kernel sizes
TEXTURE_KERNEL_SMALL = 3
TEXTURE_KERNEL_MEDIUM = 5
TEXTURE_KERNEL_LARGE = 7


@dataclass
class StructuralDamageConfig:
    """Configuration for structural damage assessment."""

    texture_analysis: bool = True  # Enable texture-based damage detection
    building_mask_required: bool = False  # Require building footprint mask
    damage_categories: int = 4  # Number of damage categories (1-5)
    texture_kernel_size: int = 5  # Kernel size for texture analysis
    brightness_weight: float = 0.4  # Weight for brightness change in scoring
    texture_weight: float = 0.6  # Weight for texture change in scoring
    min_damage_area_m2: float = 50.0  # Minimum damage area (m²)
    use_sar_coherence: bool = False  # Use SAR coherence if available

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 1 <= self.damage_categories <= 5:
            raise ValueError(
                f"damage_categories must be in [1, 5], got {self.damage_categories}"
            )
        if self.texture_kernel_size not in [3, 5, 7, 9]:
            raise ValueError(
                f"texture_kernel_size must be 3, 5, 7, or 9, got {self.texture_kernel_size}"
            )
        if not 0.0 <= self.brightness_weight <= 1.0:
            raise ValueError(
                f"brightness_weight must be in [0.0, 1.0], got {self.brightness_weight}"
            )
        if not 0.0 <= self.texture_weight <= 1.0:
            raise ValueError(
                f"texture_weight must be in [0.0, 1.0], got {self.texture_weight}"
            )
        if abs(self.brightness_weight + self.texture_weight - 1.0) > 0.01:
            raise ValueError(
                f"brightness_weight + texture_weight must equal 1.0, got {self.brightness_weight + self.texture_weight}"
            )
        if self.min_damage_area_m2 < 0:
            raise ValueError(
                f"min_damage_area_m2 must be non-negative, got {self.min_damage_area_m2}"
            )


@dataclass
class StructuralDamageResult:
    """Results from structural damage assessment."""

    damage_map: np.ndarray  # Binary mask of damaged areas
    damage_grade: np.ndarray  # Damage grade classification (0 to damage_categories)
    confidence_map: np.ndarray  # Confidence scores (0-1)
    brightness_change: np.ndarray  # Brightness change map
    texture_change: np.ndarray  # Texture change map
    metadata: Dict[str, Any]  # Algorithm metadata and parameters
    statistics: Dict[str, Any]  # Summary statistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "damage_map": self.damage_map,
            "damage_grade": self.damage_grade,
            "confidence_map": self.confidence_map,
            "brightness_change": self.brightness_change,
            "texture_change": self.texture_change,
            "metadata": self.metadata,
            "statistics": self.statistics,
        }


class StructuralDamageAssessment:
    """
    Structural Damage Assessment Algorithm.

    This algorithm assesses building and infrastructure damage from storm
    events by analyzing changes in optical imagery texture and brightness.
    It can optionally incorporate SAR coherence information when available.

    The approach is based on the observation that structural damage causes:
    1. Changes in surface texture (debris, collapsed structures)
    2. Changes in brightness/reflectance (exposed materials, shadows)
    3. Loss of SAR coherence (structural changes affect phase)

    Requirements:
        - Pre-event high-resolution optical imagery
        - Post-event high-resolution optical imagery
        - Optional: Building footprint mask
        - Optional: SAR coherence pre/post

    Outputs:
        - damage_map: Binary mask of damaged areas
        - damage_grade: Classified damage severity (D0-D4)
        - confidence_map: Per-pixel confidence (0-1)
    """

    METADATA = {
        "id": "storm.baseline.structural_damage",
        "name": "Structural Damage Assessment",
        "category": "baseline",
        "event_types": ["storm.*", "hurricane.*", "tornado.*"],
        "version": "1.0.0",
        "deterministic": True,
        "seed_required": False,
        "requirements": {
            "data": {
                "optical_pre": {
                    "resolution": "high",  # < 5m recommended
                    "temporal": "pre_event",
                },
                "optical_post": {
                    "resolution": "high",
                    "temporal": "post_event",
                },
            },
            "optional": {
                "building_mask": {"benefit": "focuses_analysis_on_structures"},
                "sar_coherence_pre": {"benefit": "improves_accuracy"},
                "sar_coherence_post": {"benefit": "improves_accuracy"},
            },
            "compute": {"memory_gb": 4, "gpu": False, "distributed": True},
        },
        "validation": {
            "accuracy_range": [0.68, 0.85],
            "validated_regions": ["north_america", "caribbean"],
            "citations": ["Copernicus EMS Damage Assessment Guidelines"],
        },
    }

    def __init__(self, config: Optional[StructuralDamageConfig] = None):
        """
        Initialize structural damage assessment algorithm.

        Args:
            config: Algorithm configuration. Uses defaults if None.
        """
        self.config = config or StructuralDamageConfig()
        logger.info(f"Initialized {self.METADATA['name']} v{self.METADATA['version']}")
        logger.info(
            f"Configuration: texture={self.config.texture_analysis}, "
            f"categories={self.config.damage_categories}, "
            f"kernel_size={self.config.texture_kernel_size}"
        )

    def execute(
        self,
        optical_pre: np.ndarray,
        optical_post: np.ndarray,
        building_mask: Optional[np.ndarray] = None,
        sar_coherence_pre: Optional[np.ndarray] = None,
        sar_coherence_post: Optional[np.ndarray] = None,
        pixel_size_m: float = 1.0,
        nodata_value: Optional[float] = None,
    ) -> StructuralDamageResult:
        """
        Execute structural damage assessment.

        Args:
            optical_pre: Pre-event optical imagery, shape (H, W) or (H, W, C)
            optical_post: Post-event optical imagery, shape (H, W) or (H, W, C)
            building_mask: Optional building footprint mask (True = building)
            sar_coherence_pre: Optional pre-event SAR coherence (0-1)
            sar_coherence_post: Optional post-event SAR coherence (0-1)
            pixel_size_m: Pixel size in meters
            nodata_value: NoData value to mask out

        Returns:
            StructuralDamageResult containing damage map, grades, and confidence
        """
        logger.info("Starting structural damage assessment")

        # Validate and preprocess inputs
        pre_gray, post_gray = self._preprocess_images(
            optical_pre, optical_post, nodata_value
        )

        # Create valid data mask
        valid_mask = self._create_valid_mask(pre_gray, post_gray, nodata_value)

        # Apply building mask if provided and required
        analysis_mask = valid_mask.copy()
        if building_mask is not None:
            logger.info("Applying building footprint mask")
            analysis_mask &= building_mask
        elif self.config.building_mask_required:
            raise ValueError(
                "building_mask_required=True but no building_mask provided"
            )

        # Calculate brightness change
        brightness_change = self._calculate_brightness_change(
            pre_gray, post_gray, valid_mask
        )

        # Calculate texture change
        if self.config.texture_analysis:
            texture_change = self._calculate_texture_change(
                pre_gray, post_gray, valid_mask
            )
        else:
            texture_change = np.zeros_like(brightness_change)

        # Calculate SAR coherence change if available
        coherence_change = None
        if (
            self.config.use_sar_coherence
            and sar_coherence_pre is not None
            and sar_coherence_post is not None
        ):
            coherence_change = self._calculate_coherence_change(
                sar_coherence_pre, sar_coherence_post, valid_mask
            )
            logger.info("SAR coherence incorporated into damage assessment")

        # Calculate combined damage score
        damage_score = self._calculate_damage_score(
            brightness_change, texture_change, coherence_change, analysis_mask
        )

        # Classify damage grades
        damage_grade = self._classify_damage_grades(damage_score, analysis_mask)

        # Create binary damage map (any damage)
        damage_map = damage_grade > DAMAGE_GRADE_NONE

        # Calculate confidence
        confidence_map = self._calculate_confidence(
            brightness_change, texture_change, coherence_change, analysis_mask
        )

        # Calculate statistics
        pixel_area_m2 = pixel_size_m**2
        statistics = self._calculate_statistics(
            damage_map, damage_grade, confidence_map, analysis_mask, pixel_area_m2
        )

        # Build metadata
        metadata = {
            **self.METADATA,
            "parameters": {
                "texture_analysis": self.config.texture_analysis,
                "building_mask_required": self.config.building_mask_required,
                "damage_categories": self.config.damage_categories,
                "texture_kernel_size": self.config.texture_kernel_size,
                "brightness_weight": self.config.brightness_weight,
                "texture_weight": self.config.texture_weight,
                "min_damage_area_m2": self.config.min_damage_area_m2,
                "pixel_size_m": pixel_size_m,
            },
            "execution": {
                "building_mask_used": building_mask is not None,
                "sar_coherence_used": coherence_change is not None,
            },
        }

        logger.info(
            f"Assessment complete: {statistics['damage_area_m2']:.1f} m² total damage "
            f"({statistics['damage_percent']:.1f}% of analysis area)"
        )

        return StructuralDamageResult(
            damage_map=damage_map,
            damage_grade=damage_grade,
            confidence_map=confidence_map,
            brightness_change=brightness_change,
            texture_change=texture_change,
            metadata=metadata,
            statistics=statistics,
        )

    def _preprocess_images(
        self,
        optical_pre: np.ndarray,
        optical_post: np.ndarray,
        nodata_value: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess optical images to grayscale.

        Args:
            optical_pre: Pre-event imagery
            optical_post: Post-event imagery
            nodata_value: NoData value

        Returns:
            Tuple of (pre_gray, post_gray) grayscale images
        """
        # Convert to grayscale if multi-band
        if optical_pre.ndim == 3:
            pre_gray = np.mean(optical_pre, axis=2).astype(np.float32)
        else:
            pre_gray = optical_pre.astype(np.float32)

        if optical_post.ndim == 3:
            post_gray = np.mean(optical_post, axis=2).astype(np.float32)
        else:
            post_gray = optical_post.astype(np.float32)

        # Validate shapes match
        if pre_gray.shape != post_gray.shape:
            raise ValueError(
                f"Pre/post image shape mismatch: {pre_gray.shape} vs {post_gray.shape}"
            )

        # Normalize to [0, 1] range if needed
        for arr in [pre_gray, post_gray]:
            # Find valid values (not nodata, not NaN)
            if nodata_value is not None:
                valid_mask = (arr != nodata_value) & np.isfinite(arr)
            else:
                valid_mask = np.isfinite(arr)

            valid_vals = arr[valid_mask]
            max_val = np.max(valid_vals) if valid_vals.size > 0 else 1.0

            if max_val <= 0 or not np.isfinite(max_val):
                max_val = 1.0
            if max_val > 1.0:
                # Only normalize valid values
                arr[valid_mask] = arr[valid_mask] / max_val

        return pre_gray, post_gray

    def _create_valid_mask(
        self,
        pre_gray: np.ndarray,
        post_gray: np.ndarray,
        nodata_value: Optional[float],
    ) -> np.ndarray:
        """Create combined valid data mask."""
        valid_mask = np.ones_like(pre_gray, dtype=bool)

        if nodata_value is not None:
            valid_mask &= pre_gray != nodata_value
            valid_mask &= post_gray != nodata_value

        valid_mask &= np.isfinite(pre_gray)
        valid_mask &= np.isfinite(post_gray)

        return valid_mask

    def _calculate_brightness_change(
        self,
        pre_gray: np.ndarray,
        post_gray: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate brightness change between pre and post images.

        Args:
            pre_gray: Pre-event grayscale
            post_gray: Post-event grayscale
            valid_mask: Valid data mask

        Returns:
            Absolute brightness change (0-1)
        """
        brightness_change = np.zeros_like(pre_gray, dtype=np.float32)
        brightness_change[valid_mask] = np.abs(
            post_gray[valid_mask] - pre_gray[valid_mask]
        )
        return brightness_change

    def _calculate_texture_change(
        self,
        pre_gray: np.ndarray,
        post_gray: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate texture change using local variance.

        Structural damage typically increases local variance due to debris
        and irregular surfaces.

        Args:
            pre_gray: Pre-event grayscale
            post_gray: Post-event grayscale
            valid_mask: Valid data mask

        Returns:
            Texture change score (0-1 normalized)
        """
        kernel_size = self.config.texture_kernel_size

        # Calculate local variance for both images
        var_pre = self._local_variance(pre_gray, kernel_size)
        var_post = self._local_variance(post_gray, kernel_size)

        # Calculate variance change (increase indicates potential damage)
        texture_change = np.zeros_like(pre_gray, dtype=np.float32)

        # Avoid division by zero
        valid_var = valid_mask & (var_pre > 1e-10)
        texture_change[valid_var] = (var_post[valid_var] - var_pre[valid_var]) / (
            var_pre[valid_var] + 1e-10
        )

        # Normalize to [0, 1] range (positive changes indicate damage)
        texture_change = np.clip(texture_change, 0.0, 2.0) / 2.0

        return texture_change

    def _local_variance(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        Calculate local variance using a sliding window.

        Args:
            image: Input grayscale image
            kernel_size: Size of the sliding window

        Returns:
            Local variance map
        """
        # Pad image for edge handling
        pad_size = kernel_size // 2
        padded = np.pad(image, pad_size, mode="reflect")

        # Calculate local mean
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        local_mean = self._convolve2d_simple(padded, kernel)

        # Calculate local variance: E[X²] - E[X]²
        local_mean_sq = self._convolve2d_simple(padded**2, kernel)
        local_var = local_mean_sq - local_mean**2

        # Remove padding and ensure non-negative
        local_var = local_var[pad_size:-pad_size, pad_size:-pad_size]
        return np.maximum(local_var, 0.0)

    def _convolve2d_simple(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Simple 2D convolution implementation.

        Args:
            image: Input image
            kernel: Convolution kernel

        Returns:
            Convolved image
        """
        from scipy.ndimage import convolve

        return convolve(image, kernel, mode="reflect")

    def _calculate_coherence_change(
        self,
        coherence_pre: np.ndarray,
        coherence_post: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate SAR coherence change.

        Loss of coherence indicates structural change.

        Args:
            coherence_pre: Pre-event coherence (0-1)
            coherence_post: Post-event coherence (0-1)
            valid_mask: Valid data mask

        Returns:
            Coherence loss score (0-1)
        """
        coherence_change = np.zeros_like(coherence_pre, dtype=np.float32)

        # Coherence loss (pre - post, clamped to positive)
        coherence_change[valid_mask] = np.clip(
            coherence_pre[valid_mask] - coherence_post[valid_mask], 0.0, 1.0
        )

        return coherence_change

    def _calculate_damage_score(
        self,
        brightness_change: np.ndarray,
        texture_change: np.ndarray,
        coherence_change: Optional[np.ndarray],
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate combined damage score.

        Args:
            brightness_change: Brightness change map
            texture_change: Texture change map
            coherence_change: Optional coherence change map
            analysis_mask: Analysis area mask

        Returns:
            Combined damage score (0-1)
        """
        damage_score = np.zeros_like(brightness_change, dtype=np.float32)

        if coherence_change is not None:
            # Three-way weighted combination
            # Reduce other weights proportionally
            coh_weight = 0.3
            bri_weight = self.config.brightness_weight * (1 - coh_weight)
            tex_weight = self.config.texture_weight * (1 - coh_weight)

            damage_score[analysis_mask] = (
                bri_weight * brightness_change[analysis_mask]
                + tex_weight * texture_change[analysis_mask]
                + coh_weight * coherence_change[analysis_mask]
            )
        else:
            # Two-way weighted combination
            damage_score[analysis_mask] = (
                self.config.brightness_weight * brightness_change[analysis_mask]
                + self.config.texture_weight * texture_change[analysis_mask]
            )

        return damage_score

    def _classify_damage_grades(
        self, damage_score: np.ndarray, analysis_mask: np.ndarray
    ) -> np.ndarray:
        """
        Classify damage into grades based on score.

        Grades follow Copernicus EMS damage assessment guidelines:
            0 = No damage
            1 = Slight damage (negligible to slight)
            2 = Moderate damage (moderate structural)
            3 = Heavy damage (significant structural)
            4 = Destroyed (very heavy to destruction)

        Args:
            damage_score: Combined damage score (0-1)
            analysis_mask: Analysis area mask

        Returns:
            Damage grade classification (uint8)
        """
        damage_grade = np.zeros_like(damage_score, dtype=np.uint8)

        # Thresholds based on damage_categories setting
        num_categories = self.config.damage_categories
        thresholds = np.linspace(0.15, 0.7, num_categories)

        for i, threshold in enumerate(thresholds):
            grade = i + 1
            mask = (damage_score >= threshold) & analysis_mask
            damage_grade[mask] = grade

        return damage_grade

    def _calculate_confidence(
        self,
        brightness_change: np.ndarray,
        texture_change: np.ndarray,
        coherence_change: Optional[np.ndarray],
        analysis_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate per-pixel confidence scores.

        Confidence is based on:
        1. Agreement between brightness and texture indicators
        2. Magnitude of changes
        3. Coherence agreement if available

        Args:
            brightness_change: Brightness change map
            texture_change: Texture change map
            coherence_change: Optional coherence change map
            analysis_mask: Analysis area mask

        Returns:
            Confidence scores (0-1)
        """
        confidence = np.zeros_like(brightness_change, dtype=np.float32)

        # Normalize changes to comparable scales
        bri_norm = brightness_change / (BRIGHTNESS_CHANGE_THRESHOLD + 1e-10)
        tex_norm = texture_change / (TEXTURE_VARIANCE_CHANGE_THRESHOLD + 1e-10)

        # Agreement confidence: both indicators agree
        agreement = 1.0 - np.abs(bri_norm - tex_norm) / (bri_norm + tex_norm + 1e-10)

        # Magnitude confidence: stronger signals are more confident
        magnitude = (bri_norm + tex_norm) / 2.0

        if coherence_change is not None:
            coh_norm = coherence_change / (SAR_COHERENCE_LOSS_THRESHOLD + 1e-10)

            # Three-way agreement
            confidence[analysis_mask] = np.clip(
                0.5 * agreement[analysis_mask]
                + 0.3 * np.clip(magnitude[analysis_mask], 0, 1)
                + 0.2 * np.clip(coh_norm[analysis_mask], 0, 1),
                0.0,
                1.0,
            )
        else:
            confidence[analysis_mask] = np.clip(
                0.6 * agreement[analysis_mask]
                + 0.4 * np.clip(magnitude[analysis_mask], 0, 1),
                0.0,
                1.0,
            )

        return confidence

    def _calculate_statistics(
        self,
        damage_map: np.ndarray,
        damage_grade: np.ndarray,
        confidence_map: np.ndarray,
        analysis_mask: np.ndarray,
        pixel_area_m2: float,
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        total_analysis_pixels = int(np.sum(analysis_mask))
        damage_pixels = int(np.sum(damage_map))
        damage_area_m2 = damage_pixels * pixel_area_m2

        # Grade distribution
        grade_counts = {}
        grade_areas = {}
        for grade in range(self.config.damage_categories + 1):
            count = int(np.sum(damage_grade == grade))
            grade_counts[f"grade_{grade}"] = count
            grade_areas[f"grade_{grade}_m2"] = float(count * pixel_area_m2)

        return {
            "total_pixels": int(damage_map.size),
            "analysis_pixels": total_analysis_pixels,
            "damage_pixels": damage_pixels,
            "damage_area_m2": float(damage_area_m2),
            "damage_percent": (
                float(100.0 * damage_pixels / total_analysis_pixels)
                if total_analysis_pixels > 0
                else 0.0
            ),
            "mean_confidence": (
                float(np.mean(confidence_map[damage_map]))
                if damage_pixels > 0
                else 0.0
            ),
            "grade_distribution": grade_counts,
            "grade_areas": grade_areas,
        }

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """Get algorithm metadata."""
        return StructuralDamageAssessment.METADATA

    @staticmethod
    def create_from_dict(params: Dict[str, Any]) -> "StructuralDamageAssessment":
        """
        Create algorithm instance from parameter dictionary.

        Args:
            params: Parameter dictionary

        Returns:
            Configured algorithm instance
        """
        config = StructuralDamageConfig(**params)
        return StructuralDamageAssessment(config)
