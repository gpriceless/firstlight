"""
Configuration for Image Validation.

Provides dataclasses and utilities for configuring the image validation
pipeline, including thresholds, screenshot settings, and behavior options.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# The shipped thresholds, found from this file so they load wherever the
# process starts (config.py -> validation -> ingestion -> data -> core -> repo).
SHIPPED_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "ingestion.yaml"
USER_CONFIG_PATH = "~/.firstlight/config/ingestion.yaml"
SYSTEM_CONFIG_PATH = Path("/etc/firstlight/ingestion.yaml")


class ValidationConfigError(ValueError):
    """A validation config file is present but cannot be read or parsed."""


@dataclass
class OpticalThresholds:
    """
    Thresholds for optical imagery validation.

    Attributes:
        std_dev_min: Minimum standard deviation of an integer band with no declared
            scale, in digital numbers (below it the band is flagged as blank)
        std_dev_min_reflectance: The same floor in reflectance, for a band with a
            declared scale or a float band (0.0001 is one Sentinel-2 L2A step)
        non_zero_ratio_min: Minimum ratio of non-zero pixels (0.0 to 1.0)
        nodata_ratio_max: Maximum ratio of NoData pixels (0.0 to 1.0)
        value_range_uint16: Valid range for the mean of a UInt16 band with no declared scale
        value_range_float32: Valid range for the mean of a Float32 band, and for reflectance
        resolution_tolerance: Tolerance for resolution matching (as fraction, e.g., 0.1 = 10%)
    """

    std_dev_min: float = 1.0
    std_dev_min_reflectance: float = 0.0001
    non_zero_ratio_min: float = 0.05
    nodata_ratio_max: float = 0.95
    value_range_uint16: tuple = (0, 10000)
    value_range_float32: tuple = (0.0, 1.0)
    resolution_tolerance: float = 0.1


@dataclass
class SARThresholds:
    """
    Thresholds for SAR imagery validation.

    SAR imagery requires different thresholds due to speckle noise
    and different value ranges (dB scale).

    Attributes:
        std_dev_min_db: Minimum standard deviation in dB (higher than optical due to speckle)
        backscatter_range_db: Valid backscatter range in dB
        extreme_value_threshold_db: Threshold for extreme backscatter values
        extreme_value_ratio_max: Maximum ratio of pixels with extreme values
        nodata_ratio_max: Maximum ratio of NoData pixels
    """

    std_dev_min_db: float = 2.0
    backscatter_range_db: tuple = (-50, 20)
    extreme_value_threshold_db: tuple = (-30, 10)
    extreme_value_ratio_max: float = 0.5
    nodata_ratio_max: float = 0.95


@dataclass
class ScreenshotConfig:
    """
    Configuration for screenshot capture.

    Screenshots are optional and used for debugging and audit trails.

    Attributes:
        enabled: Whether screenshot capture is enabled
        output_dir: Directory for saving screenshots
        format: Image format (png, jpg)
        resolution: Screenshot resolution (width, height)
        bands_to_render: List of band types to render
        metadata_overlay: Whether to overlay metadata on screenshots
        on_failure_only: Only capture screenshots for failed validations
        retention: Screenshot retention policy ('temporary' or 'permanent')
    """

    enabled: bool = False
    output_dir: str = "~/.firstlight/screenshots"
    format: str = "png"
    resolution: tuple = (1200, 800)
    bands_to_render: List[str] = field(
        default_factory=lambda: ["rgb_composite", "nir", "swir1"]
    )
    metadata_overlay: bool = True
    on_failure_only: bool = False
    retention: str = "temporary"  # 'temporary' = delete after validation, 'permanent' = keep

    def get_output_path(self) -> Path:
        """Get expanded output directory path."""
        return Path(os.path.expanduser(self.output_dir))


@dataclass
class PerformanceConfig:
    """
    Performance settings for validation.

    Attributes:
        max_validation_time_seconds: Maximum time for validation before timeout
        sample_ratio: Ratio of pixels to sample for large images (1.0 = all, 0.3 = 30%)
        sample_threshold_pixels: Image size threshold above which sampling is used
        parallel_bands: Whether to validate bands in parallel
        max_memory_mb: Maximum memory to use for validation
    """

    max_validation_time_seconds: float = 30.0
    sample_ratio: float = 0.3  # 30% sampling for large images
    sample_threshold_pixels: int = 5000 * 5000  # 25M pixels
    parallel_bands: bool = True
    max_memory_mb: int = 500


@dataclass
class ActionConfig:
    """
    Configuration for validation actions/behaviors.

    Attributes:
        reject_on_blank_band: Reject image if any required band is blank
        reject_on_missing_required_band: Reject if required bands are missing
        reject_on_invalid_crs: Reject if CRS is invalid or missing
        warn_on_high_nodata: Warn (but continue) if NoData ratio is high
        warn_on_missing_optional_band: Warn if optional bands are missing
        cleanup_invalid_files: Delete invalid raster files
        max_load_retries: Maximum retries for file loading errors
    """

    reject_on_blank_band: bool = True
    reject_on_missing_required_band: bool = True
    reject_on_invalid_crs: bool = True
    warn_on_high_nodata: bool = True
    warn_on_missing_optional_band: bool = True
    cleanup_invalid_files: bool = False
    max_load_retries: int = 3


@dataclass
class AlertConfig:
    """
    Configuration for validation alerting.

    Attributes:
        enabled: Whether alerting is enabled
        hung_process_timeout_seconds: Alert if validation hangs for this long
        failure_rate_threshold: Alert if failure rate exceeds this (0.0 to 1.0)
        alert_callback: Optional callback function for alerts
    """

    enabled: bool = True
    hung_process_timeout_seconds: float = 60.0
    failure_rate_threshold: float = 0.1
    alert_callback: Optional[Any] = None


@dataclass
class ValidationConfig:
    """
    Complete configuration for image validation.

    This is the top-level configuration object that combines all
    validation settings.

    Attributes:
        enabled: Master switch for validation
        optical: Optical imagery thresholds
        sar: SAR imagery thresholds
        screenshots: Screenshot capture settings
        performance: Performance tuning settings
        actions: Behavior/action settings
        alerts: Alerting configuration
        required_optical_bands: Generic names of required optical bands
        optional_optical_bands: Generic names of optional optical bands
        required_sar_polarizations: Required SAR polarizations
        optional_sar_polarizations: Optional SAR polarizations
    """

    enabled: bool = True
    optical: OpticalThresholds = field(default_factory=OpticalThresholds)
    sar: SARThresholds = field(default_factory=SARThresholds)
    screenshots: ScreenshotConfig = field(default_factory=ScreenshotConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)

    # Band requirements (generic names)
    required_optical_bands: List[str] = field(
        default_factory=lambda: ["blue", "green", "red", "nir"]
    )
    optional_optical_bands: List[str] = field(
        default_factory=lambda: ["swir1", "swir2", "coastal", "red_edge"]
    )
    required_sar_polarizations: List[str] = field(default_factory=lambda: ["VV"])
    optional_sar_polarizations: List[str] = field(
        default_factory=lambda: ["VH", "HH", "HV"]
    )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ValidationConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ValidationConfig instance
        """
        # Extract nested configs
        optical_dict = config_dict.get("optical", {})
        sar_dict = config_dict.get("sar", {})
        screenshots_dict = config_dict.get("screenshots", {})
        performance_dict = config_dict.get("performance", {})
        actions_dict = config_dict.get("actions", {})
        alerts_dict = config_dict.get("alerts", {})

        # Handle threshold nested in optical/sar
        optical_thresholds = optical_dict.pop("thresholds", {})
        sar_thresholds = sar_dict.pop("thresholds", {})

        # Build config objects
        optical = OpticalThresholds(**optical_thresholds)
        sar = SARThresholds(**sar_thresholds)
        screenshots = ScreenshotConfig(**screenshots_dict)
        performance = PerformanceConfig(**performance_dict)
        actions = ActionConfig(**actions_dict)
        alerts = AlertConfig(**{k: v for k, v in alerts_dict.items() if k != "alert_callback"})

        return cls(
            enabled=config_dict.get("enabled", True),
            optical=optical,
            sar=sar,
            screenshots=screenshots,
            performance=performance,
            actions=actions,
            alerts=alerts,
            required_optical_bands=optical_dict.get(
                "required_bands", ["blue", "green", "red", "nir"]
            ),
            optional_optical_bands=optical_dict.get(
                "optional_bands", ["swir1", "swir2"]
            ),
            required_sar_polarizations=sar_dict.get(
                "required_polarizations", ["VV"]
            ),
            optional_sar_polarizations=sar_dict.get(
                "optional_polarizations", ["VH", "HH", "HV"]
            ),
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ValidationConfig":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            ValidationConfig instance
        """
        path = Path(yaml_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Extract validation section if present
        if "validation" in config_dict:
            config_dict = config_dict["validation"]

        return cls.from_dict(config_dict)

    @classmethod
    def from_environment(cls) -> "ValidationConfig":
        """
        Create configuration from environment variables.

        Environment variables override default values:
        - FIRSTLIGHT_VALIDATION_ENABLED
        - FIRSTLIGHT_VALIDATION_SCREENSHOT_DIR
        - FIRSTLIGHT_VALIDATION_SAMPLE_RATIO
        - FIRSTLIGHT_VALIDATION_MAX_TIME

        Returns:
            ValidationConfig instance
        """
        config = cls()

        # Check for environment variable overrides
        if os.environ.get("FIRSTLIGHT_VALIDATION_ENABLED"):
            config.enabled = os.environ.get("FIRSTLIGHT_VALIDATION_ENABLED", "true").lower() == "true"

        if os.environ.get("FIRSTLIGHT_VALIDATION_SCREENSHOT_DIR"):
            config.screenshots.output_dir = os.environ.get("FIRSTLIGHT_VALIDATION_SCREENSHOT_DIR")

        if os.environ.get("FIRSTLIGHT_VALIDATION_SAMPLE_RATIO"):
            try:
                config.performance.sample_ratio = float(
                    os.environ.get("FIRSTLIGHT_VALIDATION_SAMPLE_RATIO")
                )
            except ValueError:
                pass

        if os.environ.get("FIRSTLIGHT_VALIDATION_MAX_TIME"):
            try:
                config.performance.max_validation_time_seconds = float(
                    os.environ.get("FIRSTLIGHT_VALIDATION_MAX_TIME")
                )
            except ValueError:
                pass

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enabled": self.enabled,
            "optical": {
                "required_bands": self.required_optical_bands,
                "optional_bands": self.optional_optical_bands,
                "thresholds": {
                    "std_dev_min": self.optical.std_dev_min,
                    "std_dev_min_reflectance": self.optical.std_dev_min_reflectance,
                    "non_zero_ratio_min": self.optical.non_zero_ratio_min,
                    "nodata_ratio_max": self.optical.nodata_ratio_max,
                    "value_range_uint16": list(self.optical.value_range_uint16),
                    "value_range_float32": list(self.optical.value_range_float32),
                },
            },
            "sar": {
                "required_polarizations": self.required_sar_polarizations,
                "optional_polarizations": self.optional_sar_polarizations,
                "thresholds": {
                    "std_dev_min_db": self.sar.std_dev_min_db,
                    "backscatter_range_db": list(self.sar.backscatter_range_db),
                    "extreme_value_threshold_db": list(self.sar.extreme_value_threshold_db),
                    "extreme_value_ratio_max": self.sar.extreme_value_ratio_max,
                },
            },
            "screenshots": {
                "enabled": self.screenshots.enabled,
                "output_dir": self.screenshots.output_dir,
                "format": self.screenshots.format,
                "resolution": list(self.screenshots.resolution),
                "on_failure_only": self.screenshots.on_failure_only,
                "retention": self.screenshots.retention,
            },
            "performance": {
                "max_validation_time_seconds": self.performance.max_validation_time_seconds,
                "sample_ratio": self.performance.sample_ratio,
                "parallel_bands": self.performance.parallel_bands,
            },
            "actions": {
                "reject_on_blank_band": self.actions.reject_on_blank_band,
                "reject_on_missing_required_band": self.actions.reject_on_missing_required_band,
                "reject_on_invalid_crs": self.actions.reject_on_invalid_crs,
                "cleanup_invalid_files": self.actions.cleanup_invalid_files,
            },
        }


# Default configuration instance
DEFAULT_CONFIG = ValidationConfig()


def _read_validation_section(path: Path) -> Dict[str, Any]:
    """Read one config file and return its ``validation`` section."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    if not isinstance(config_dict, dict):
        raise ValueError(f"expected a mapping, got {type(config_dict).__name__}")
    section = config_dict.get("validation", config_dict)
    if not isinstance(section, dict):
        raise ValueError(f"'validation' must be a mapping, got {type(section).__name__}")
    return section


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``base`` with ``over`` laid on top, mapping by mapping."""
    merged = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    yaml_path: Optional[str] = None,
    use_environment: bool = True,
) -> ValidationConfig:
    """
    Load validation configuration.

    Files are layered. The shipped ``config/ingestion.yaml`` beside the
    package is the base, ``/etc/firstlight/ingestion.yaml`` goes on top of it,
    then ``~/.firstlight/config/ingestion.yaml``, then the explicit
    ``yaml_path``. A key one file leaves out keeps the value from the file
    below it, so a user file that sets one threshold changes only that
    threshold. Built-in defaults fill only keys no file sets. Environment
    overrides go on top. The files read are logged, highest first.

    Args:
        yaml_path: Optional explicit path to YAML config
        use_environment: Whether to apply environment variable overrides

    Returns:
        ValidationConfig instance

    Raises:
        FileNotFoundError: ``yaml_path`` was given and does not exist.
        ValidationConfigError: a config file exists but cannot be read or
            parsed. The run stops rather than fall back to defaults.
    """
    candidates = [
        Path(USER_CONFIG_PATH).expanduser(),
        SYSTEM_CONFIG_PATH,
        SHIPPED_CONFIG_PATH,
    ]
    if yaml_path:
        path = Path(yaml_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Validation config not found: {path}")
        candidates.insert(0, path)

    found = [path for path in candidates if path.exists()]
    merged: Dict[str, Any] = {}
    for path in reversed(found):
        try:
            merged = _merge(merged, _read_validation_section(path))
        except Exception as e:
            raise ValidationConfigError(
                f"Validation config {path} could not be loaded: {e}"
            ) from e

    if found:
        try:
            config = ValidationConfig.from_dict(merged)
        except Exception as e:
            raise ValidationConfigError(
                f"Validation config from {', '.join(str(p) for p in found)} "
                f"could not be loaded: {e}"
            ) from e
        logger.info(
            "Validation config loaded from "
            + " over ".join(str(p) for p in found)
        )
    else:
        logger.warning(
            "No validation config file found (looked in "
            f"{', '.join(str(p) for p in candidates)}); using built-in defaults"
        )
        config = ValidationConfig()

    # Apply environment overrides
    if use_environment:
        env_config = ValidationConfig.from_environment()
        if os.environ.get("FIRSTLIGHT_VALIDATION_ENABLED"):
            config.enabled = env_config.enabled
        if os.environ.get("FIRSTLIGHT_VALIDATION_SCREENSHOT_DIR"):
            config.screenshots.output_dir = env_config.screenshots.output_dir
        if os.environ.get("FIRSTLIGHT_VALIDATION_SAMPLE_RATIO"):
            config.performance.sample_ratio = env_config.performance.sample_ratio
        if os.environ.get("FIRSTLIGHT_VALIDATION_MAX_TIME"):
            config.performance.max_validation_time_seconds = env_config.performance.max_validation_time_seconds

    return config
