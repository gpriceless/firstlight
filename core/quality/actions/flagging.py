"""
Quality Flag System.

Provides a comprehensive quality flagging system for geospatial products,
allowing machine-readable and human-understandable quality indicators.

Key Concepts:
- Flags indicate specific quality conditions (not just pass/fail)
- Flags can be applied at product, region, or pixel level
- Standard flags align with quality.schema.json definitions
- Custom flags can be registered for domain-specific needs
- Flags support aggregation and summarization
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class FlagLevel(Enum):
    """Levels at which flags can be applied."""
    PRODUCT = "product"     # Entire product flagged
    REGION = "region"       # Specific geographic region
    PIXEL = "pixel"         # Per-pixel quality layer
    TEMPORAL = "temporal"   # Specific time period
    BAND = "band"           # Specific spectral band


class FlagSeverity(Enum):
    """Severity of quality flags."""
    CRITICAL = "critical"       # Major quality issue
    WARNING = "warning"         # Notable quality concern
    INFORMATIONAL = "informational"  # Quality note, not necessarily bad
    METADATA = "metadata"       # Quality metadata without issue


class StandardFlag(Enum):
    """
    Standard quality flags aligned with common.schema.json quality_flag.

    These flags provide consistent, cross-platform quality indicators.
    """
    # Confidence levels
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"
    MEDIUM_CONFIDENCE = "MEDIUM_CONFIDENCE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INSUFFICIENT_CONFIDENCE = "INSUFFICIENT_CONFIDENCE"

    # Degraded modes
    RESOLUTION_DEGRADED = "RESOLUTION_DEGRADED"
    SINGLE_SENSOR_MODE = "SINGLE_SENSOR_MODE"
    TEMPORALLY_INTERPOLATED = "TEMPORALLY_INTERPOLATED"
    HISTORICAL_PROXY = "HISTORICAL_PROXY"

    # Data issues
    MISSING_OBSERVABLE = "MISSING_OBSERVABLE"
    FORECAST_DISCREPANCY = "FORECAST_DISCREPANCY"
    SPATIAL_UNCERTAINTY = "SPATIAL_UNCERTAINTY"
    MAGNITUDE_CONFLICT = "MAGNITUDE_CONFLICT"
    CONSERVATIVE_ESTIMATE = "CONSERVATIVE_ESTIMATE"

    # Extended flags (beyond schema)
    CLOUD_AFFECTED = "CLOUD_AFFECTED"
    SHADOW_AFFECTED = "SHADOW_AFFECTED"
    EDGE_ARTIFACT = "EDGE_ARTIFACT"
    SENSOR_ARTIFACT = "SENSOR_ARTIFACT"
    ATMOSPHERIC_UNCERTAINTY = "ATMOSPHERIC_UNCERTAINTY"
    TERRAIN_SHADOW = "TERRAIN_SHADOW"
    WATER_MIXED_PIXEL = "WATER_MIXED_PIXEL"
    SATURATION = "SATURATION"
    NODATA = "NODATA"


@dataclass
class FlagDefinition:
    """
    Definition of a quality flag.

    Attributes:
        flag_id: Unique identifier
        name: Human-readable name
        description: Detailed description
        severity: Flag severity level
        applies_to: Levels where flag can be applied
        affects_confidence: If True, flag affects confidence scoring
        confidence_modifier: Multiplier to apply to confidence (if affects_confidence)
        category: Grouping category
    """
    flag_id: str
    name: str
    description: str
    severity: FlagSeverity
    applies_to: List[FlagLevel]
    affects_confidence: bool = True
    confidence_modifier: float = 1.0
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "flag_id": self.flag_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "applies_to": [level.value for level in self.applies_to],
            "affects_confidence": self.affects_confidence,
            "confidence_modifier": self.confidence_modifier,
            "category": self.category,
        }


@dataclass
class AppliedFlag:
    """
    A flag applied to a product or region.

    Attributes:
        flag_id: The flag identifier
        level: Level at which flag is applied
        spatial_extent: GeoJSON geometry (for region-level flags)
        pixel_mask: Boolean mask (for pixel-level flags)
        temporal_range: Time range (for temporal flags)
        band_ids: Affected bands (for band-level flags)
        reason: Why the flag was applied
        metric_value: Associated metric if applicable
        metadata: Additional flag metadata
        timestamp: When the flag was applied
    """
    flag_id: str
    level: FlagLevel
    spatial_extent: Optional[Dict[str, Any]] = None
    pixel_mask: Optional[np.ndarray] = None
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    band_ids: Optional[List[str]] = None
    reason: str = ""
    metric_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "flag_id": self.flag_id,
            "level": self.level.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.spatial_extent:
            result["spatial_extent"] = self.spatial_extent
        if self.pixel_mask is not None:
            result["pixel_mask_shape"] = self.pixel_mask.shape
            # Handle empty arrays - np.mean on empty array returns NaN
            if self.pixel_mask.size > 0:
                result["pixel_mask_coverage"] = float(np.mean(self.pixel_mask))
            else:
                result["pixel_mask_coverage"] = 0.0
        if self.temporal_range:
            result["temporal_range"] = {
                "start": self.temporal_range[0].isoformat(),
                "end": self.temporal_range[1].isoformat(),
            }
        if self.band_ids:
            result["band_ids"] = self.band_ids
        if self.metric_value is not None:
            result["metric_value"] = self.metric_value
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class FlagSummary:
    """
    Summary of all flags applied to a product.

    Attributes:
        product_id: Product being flagged
        event_id: Associated event
        total_flags: Total number of flags applied
        flags_by_severity: Count by severity level
        flags_by_level: Count by application level
        critical_flags: List of critical flag IDs
        overall_confidence_modifier: Combined confidence modifier
        standard_flag_list: List of standard flags (for schema)
        timestamp: When summary was generated
    """
    product_id: str
    event_id: str
    total_flags: int = 0
    flags_by_severity: Dict[str, int] = field(default_factory=dict)
    flags_by_level: Dict[str, int] = field(default_factory=dict)
    critical_flags: List[str] = field(default_factory=list)
    overall_confidence_modifier: float = 1.0
    standard_flag_list: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "product_id": self.product_id,
            "event_id": self.event_id,
            "total_flags": self.total_flags,
            "flags_by_severity": self.flags_by_severity,
            "flags_by_level": self.flags_by_level,
            "critical_flags": self.critical_flags,
            "overall_confidence_modifier": self.overall_confidence_modifier,
            "standard_flag_list": self.standard_flag_list,
            "timestamp": self.timestamp.isoformat(),
        }


class FlagRegistry:
    """
    Registry of available quality flags.

    Manages flag definitions and provides lookup functionality.
    """

    def __init__(self):
        self._flags: Dict[str, FlagDefinition] = {}
        self._register_standard_flags()

    def _register_standard_flags(self) -> None:
        """Register standard flags from schema."""
        standard_definitions = [
            # Confidence levels
            FlagDefinition(
                flag_id="HIGH_CONFIDENCE",
                name="High Confidence",
                description="Results have high confidence based on multiple consistent sources",
                severity=FlagSeverity.METADATA,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                affects_confidence=False,
                category="confidence",
            ),
            FlagDefinition(
                flag_id="MEDIUM_CONFIDENCE",
                name="Medium Confidence",
                description="Results have acceptable confidence, some sources missing or degraded",
                severity=FlagSeverity.INFORMATIONAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.9,
                category="confidence",
            ),
            FlagDefinition(
                flag_id="LOW_CONFIDENCE",
                name="Low Confidence",
                description="Results have low confidence, significant data gaps or disagreements",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.7,
                category="confidence",
            ),
            FlagDefinition(
                flag_id="INSUFFICIENT_CONFIDENCE",
                name="Insufficient Confidence",
                description="Results have insufficient confidence for operational use",
                severity=FlagSeverity.CRITICAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.4,
                category="confidence",
            ),
            # Degraded modes
            FlagDefinition(
                flag_id="RESOLUTION_DEGRADED",
                name="Resolution Degraded",
                description="Product resolution lower than optimal due to data availability",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.85,
                category="degraded",
            ),
            FlagDefinition(
                flag_id="SINGLE_SENSOR_MODE",
                name="Single Sensor Mode",
                description="Only one sensor used instead of multi-sensor fusion",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.8,
                category="degraded",
            ),
            FlagDefinition(
                flag_id="TEMPORALLY_INTERPOLATED",
                name="Temporally Interpolated",
                description="Values interpolated between observations",
                severity=FlagSeverity.INFORMATIONAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.9,
                category="degraded",
            ),
            FlagDefinition(
                flag_id="HISTORICAL_PROXY",
                name="Historical Proxy",
                description="Historical data used as proxy for unavailable current data",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.75,
                category="degraded",
            ),
            # Data issues
            FlagDefinition(
                flag_id="MISSING_OBSERVABLE",
                name="Missing Observable",
                description="One or more required observables could not be derived",
                severity=FlagSeverity.CRITICAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.6,
                category="data_issue",
            ),
            FlagDefinition(
                flag_id="FORECAST_DISCREPANCY",
                name="Forecast Discrepancy",
                description="Significant discrepancy between forecast and observations",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.85,
                category="data_issue",
            ),
            FlagDefinition(
                flag_id="SPATIAL_UNCERTAINTY",
                name="Spatial Uncertainty",
                description="High uncertainty in spatial extent or boundaries",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.85,
                category="uncertainty",
            ),
            FlagDefinition(
                flag_id="MAGNITUDE_CONFLICT",
                name="Magnitude Conflict",
                description="Sources disagree on magnitude/intensity of phenomenon",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.8,
                category="data_issue",
            ),
            FlagDefinition(
                flag_id="CONSERVATIVE_ESTIMATE",
                name="Conservative Estimate",
                description="Conservative estimate used due to data limitations",
                severity=FlagSeverity.INFORMATIONAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.95,
                category="data_issue",
            ),
            # Extended flags
            FlagDefinition(
                flag_id="CLOUD_AFFECTED",
                name="Cloud Affected",
                description="Region affected by cloud cover in optical imagery",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.7,
                category="atmospheric",
            ),
            FlagDefinition(
                flag_id="SHADOW_AFFECTED",
                name="Shadow Affected",
                description="Region affected by cloud or terrain shadow",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.8,
                category="atmospheric",
            ),
            FlagDefinition(
                flag_id="EDGE_ARTIFACT",
                name="Edge Artifact",
                description="Artifacts at tile or scene edges",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.75,
                category="artifact",
            ),
            FlagDefinition(
                flag_id="SENSOR_ARTIFACT",
                name="Sensor Artifact",
                description="Known sensor-specific artifacts (striping, banding, etc.)",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL, FlagLevel.BAND],
                confidence_modifier=0.8,
                category="artifact",
            ),
            FlagDefinition(
                flag_id="ATMOSPHERIC_UNCERTAINTY",
                name="Atmospheric Uncertainty",
                description="High uncertainty in atmospheric correction",
                severity=FlagSeverity.INFORMATIONAL,
                applies_to=[FlagLevel.PRODUCT, FlagLevel.REGION],
                confidence_modifier=0.9,
                category="atmospheric",
            ),
            FlagDefinition(
                flag_id="TERRAIN_SHADOW",
                name="Terrain Shadow",
                description="Region in terrain shadow",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL],
                confidence_modifier=0.75,
                category="terrain",
            ),
            FlagDefinition(
                flag_id="WATER_MIXED_PIXEL",
                name="Water Mixed Pixel",
                description="Pixel contains mixed water/land (subpixel water)",
                severity=FlagSeverity.INFORMATIONAL,
                applies_to=[FlagLevel.PIXEL],
                confidence_modifier=0.85,
                category="classification",
            ),
            FlagDefinition(
                flag_id="SATURATION",
                name="Saturation",
                description="Sensor saturation detected",
                severity=FlagSeverity.WARNING,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL, FlagLevel.BAND],
                confidence_modifier=0.6,
                category="sensor",
            ),
            FlagDefinition(
                flag_id="NODATA",
                name="No Data",
                description="No valid data available",
                severity=FlagSeverity.CRITICAL,
                applies_to=[FlagLevel.REGION, FlagLevel.PIXEL, FlagLevel.BAND],
                confidence_modifier=0.0,
                category="data_issue",
            ),
        ]

        for flag_def in standard_definitions:
            self.register_flag(flag_def)

    def register_flag(self, flag_def: FlagDefinition) -> None:
        """Register a flag definition."""
        self._flags[flag_def.flag_id] = flag_def
        logger.debug(f"Registered flag: {flag_def.flag_id}")

    def get_flag(self, flag_id: str) -> Optional[FlagDefinition]:
        """Get flag definition by ID."""
        return self._flags.get(flag_id)

    def list_flags(
        self,
        category: Optional[str] = None,
        severity: Optional[FlagSeverity] = None,
    ) -> List[FlagDefinition]:
        """List flags, optionally filtered by category or severity."""
        flags = list(self._flags.values())

        if category:
            flags = [f for f in flags if f.category == category]
        if severity:
            flags = [f for f in flags if f.severity == severity]

        return flags

    def get_categories(self) -> List[str]:
        """Get list of all flag categories."""
        return list(set(f.category for f in self._flags.values()))


class QualityFlagger:
    """
    Main quality flagging system.

    Manages flag application, aggregation, and summarization for products.
    """

    def __init__(self, registry: Optional[FlagRegistry] = None):
        """
        Initialize quality flagger.

        Args:
            registry: Flag registry (uses default if None)
        """
        self.registry = registry or FlagRegistry()
        self._applied_flags: Dict[str, List[AppliedFlag]] = {}  # product_id -> flags

    def apply_flag(
        self,
        product_id: str,
        flag_id: str,
        level: FlagLevel = FlagLevel.PRODUCT,
        reason: str = "",
        spatial_extent: Optional[Dict[str, Any]] = None,
        pixel_mask: Optional[np.ndarray] = None,
        temporal_range: Optional[Tuple[datetime, datetime]] = None,
        band_ids: Optional[List[str]] = None,
        metric_value: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AppliedFlag:
        """
        Apply a flag to a product.

        Args:
            product_id: Product to flag
            flag_id: Flag to apply
            level: Level at which to apply flag
            reason: Why the flag is being applied
            spatial_extent: GeoJSON geometry for region flags
            pixel_mask: Boolean mask for pixel flags
            temporal_range: Time range for temporal flags
            band_ids: Band identifiers for band flags
            metric_value: Associated metric value
            metadata: Additional metadata

        Returns:
            The applied flag

        Raises:
            ValueError: If flag_id is not registered or level not applicable
        """
        flag_def = self.registry.get_flag(flag_id)
        if not flag_def:
            raise ValueError(f"Unknown flag: {flag_id}")

        if level not in flag_def.applies_to:
            raise ValueError(
                f"Flag {flag_id} cannot be applied at level {level.value}. "
                f"Valid levels: {[l.value for l in flag_def.applies_to]}"
            )

        applied = AppliedFlag(
            flag_id=flag_id,
            level=level,
            spatial_extent=spatial_extent,
            pixel_mask=pixel_mask,
            temporal_range=temporal_range,
            band_ids=band_ids,
            reason=reason or flag_def.description,
            metric_value=metric_value,
            metadata=metadata or {},
        )

        if product_id not in self._applied_flags:
            self._applied_flags[product_id] = []
        self._applied_flags[product_id].append(applied)

        logger.info(f"Applied flag {flag_id} to product {product_id} at {level.value} level")
        return applied

    def apply_standard_flag(
        self,
        product_id: str,
        flag: StandardFlag,
        level: FlagLevel = FlagLevel.PRODUCT,
        reason: str = "",
        **kwargs,
    ) -> AppliedFlag:
        """
        Apply a standard flag (convenience method).

        Args:
            product_id: Product to flag
            flag: Standard flag enum value
            level: Level at which to apply flag
            reason: Why the flag is being applied
            **kwargs: Additional arguments passed to apply_flag

        Returns:
            The applied flag
        """
        return self.apply_flag(
            product_id=product_id,
            flag_id=flag.value,
            level=level,
            reason=reason,
            **kwargs,
        )

    def remove_flag(
        self,
        product_id: str,
        flag_id: str,
        level: Optional[FlagLevel] = None,
    ) -> int:
        """
        Remove flag(s) from a product.

        Args:
            product_id: Product to modify
            flag_id: Flag to remove
            level: If specified, only remove flags at this level

        Returns:
            Number of flags removed
        """
        if product_id not in self._applied_flags:
            return 0

        original_count = len(self._applied_flags[product_id])

        if level:
            self._applied_flags[product_id] = [
                f for f in self._applied_flags[product_id]
                if not (f.flag_id == flag_id and f.level == level)
            ]
        else:
            self._applied_flags[product_id] = [
                f for f in self._applied_flags[product_id]
                if f.flag_id != flag_id
            ]

        removed = original_count - len(self._applied_flags[product_id])
        if removed > 0:
            logger.info(f"Removed {removed} flag(s) {flag_id} from product {product_id}")
        return removed

    def get_flags(
        self,
        product_id: str,
        level: Optional[FlagLevel] = None,
        severity: Optional[FlagSeverity] = None,
    ) -> List[AppliedFlag]:
        """
        Get flags applied to a product.

        Args:
            product_id: Product to query
            level: Filter by level
            severity: Filter by severity

        Returns:
            List of applied flags
        """
        flags = self._applied_flags.get(product_id, [])

        if level:
            flags = [f for f in flags if f.level == level]

        if severity:
            flags = [
                f for f in flags
                if self.registry.get_flag(f.flag_id) and
                   self.registry.get_flag(f.flag_id).severity == severity
            ]

        return flags

    def has_flag(self, product_id: str, flag_id: str) -> bool:
        """Check if a product has a specific flag."""
        return any(f.flag_id == flag_id for f in self._applied_flags.get(product_id, []))

    def get_pixel_quality_mask(
        self,
        product_id: str,
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Generate combined pixel quality mask.

        Args:
            product_id: Product to query
            shape: Output shape (height, width)

        Returns:
            Float array (0-1) where 1 = highest quality
        """
        quality_mask = np.ones(shape, dtype=np.float32)

        for flag in self.get_flags(product_id, level=FlagLevel.PIXEL):
            if flag.pixel_mask is not None:
                flag_def = self.registry.get_flag(flag.flag_id)
                if flag_def and flag_def.affects_confidence:
                    # Apply confidence modifier where flag is set
                    mask_resized = flag.pixel_mask
                    if mask_resized.shape != shape:
                        # Simple resize - in production use proper resampling
                        from scipy.ndimage import zoom
                        # Guard against division by zero for empty mask dimensions
                        if mask_resized.shape[0] == 0 or mask_resized.shape[1] == 0:
                            continue  # Skip empty masks
                        zoom_factors = (shape[0] / mask_resized.shape[0],
                                        shape[1] / mask_resized.shape[1])
                        mask_resized = zoom(mask_resized.astype(float), zoom_factors) > 0.5

                    quality_mask[mask_resized] *= flag_def.confidence_modifier

        return quality_mask

    def summarize(
        self,
        product_id: str,
        event_id: str = "",
    ) -> FlagSummary:
        """
        Generate flag summary for a product.

        Args:
            product_id: Product to summarize
            event_id: Associated event ID

        Returns:
            Flag summary
        """
        flags = self.get_flags(product_id)

        # Count by severity
        by_severity: Dict[str, int] = {}
        for sev in FlagSeverity:
            count = len([
                f for f in flags
                if self.registry.get_flag(f.flag_id) and
                   self.registry.get_flag(f.flag_id).severity == sev
            ])
            if count > 0:
                by_severity[sev.value] = count

        # Count by level
        by_level: Dict[str, int] = {}
        for level in FlagLevel:
            count = len([f for f in flags if f.level == level])
            if count > 0:
                by_level[level.value] = count

        # Critical flags
        critical = [
            f.flag_id for f in flags
            if self.registry.get_flag(f.flag_id) and
               self.registry.get_flag(f.flag_id).severity == FlagSeverity.CRITICAL
        ]

        # Calculate overall confidence modifier
        confidence = 1.0
        seen_flags: Set[str] = set()
        for flag in flags:
            if flag.flag_id in seen_flags:
                continue
            seen_flags.add(flag.flag_id)
            flag_def = self.registry.get_flag(flag.flag_id)
            if flag_def and flag_def.affects_confidence:
                confidence *= flag_def.confidence_modifier

        # Standard flags for schema
        standard_flags = list(set(
            f.flag_id for f in flags
            if f.flag_id in [sf.value for sf in StandardFlag]
        ))

        return FlagSummary(
            product_id=product_id,
            event_id=event_id,
            total_flags=len(flags),
            flags_by_severity=by_severity,
            flags_by_level=by_level,
            critical_flags=list(set(critical)),
            overall_confidence_modifier=confidence,
            standard_flag_list=standard_flags,
        )

    def clear_flags(self, product_id: str) -> int:
        """
        Clear all flags from a product.

        Args:
            product_id: Product to clear

        Returns:
            Number of flags removed
        """
        count = len(self._applied_flags.get(product_id, []))
        if product_id in self._applied_flags:
            del self._applied_flags[product_id]
        return count


# Convenience functions

def create_confidence_flag(confidence: float) -> StandardFlag:
    """
    Get appropriate confidence flag for a confidence value.

    Args:
        confidence: Confidence score 0-1

    Returns:
        Appropriate standard flag
    """
    if confidence >= 0.8:
        return StandardFlag.HIGH_CONFIDENCE
    elif confidence >= 0.6:
        return StandardFlag.MEDIUM_CONFIDENCE
    elif confidence >= 0.4:
        return StandardFlag.LOW_CONFIDENCE
    else:
        return StandardFlag.INSUFFICIENT_CONFIDENCE


def flag_from_conditions(
    cloud_cover: Optional[float] = None,
    single_sensor: bool = False,
    historical_proxy: bool = False,
    interpolated: bool = False,
    degraded_resolution: bool = False,
) -> List[StandardFlag]:
    """
    Generate list of flags based on conditions.

    Args:
        cloud_cover: Cloud cover percentage (0-100)
        single_sensor: If only one sensor was used
        historical_proxy: If historical data was used as proxy
        interpolated: If values were temporally interpolated
        degraded_resolution: If resolution is degraded

    Returns:
        List of applicable standard flags
    """
    flags = []

    if cloud_cover is not None and cloud_cover > 30:
        flags.append(StandardFlag.CLOUD_AFFECTED)

    if single_sensor:
        flags.append(StandardFlag.SINGLE_SENSOR_MODE)

    if historical_proxy:
        flags.append(StandardFlag.HISTORICAL_PROXY)

    if interpolated:
        flags.append(StandardFlag.TEMPORALLY_INTERPOLATED)

    if degraded_resolution:
        flags.append(StandardFlag.RESOLUTION_DEGRADED)

    return flags
