"""
Report Visual Pipeline for FirstLight (VIS-1.5).

Orchestrates visual product generation for reports by wiring together:
- VIS-1.1: Satellite Imagery Renderer
- VIS-1.2: Before/After Image Generation
- VIS-1.3: Detection Overlay Rendering

Provides unified interface for generating all visual products needed for reports,
with caching to avoid regeneration and optimized outputs for web/print.

Example:
    # Initialize pipeline
    pipeline = ReportVisualPipeline(output_dir=Path("./outputs/visuals"))

    # Generate all visuals from analysis result
    manifest = pipeline.generate_all_visuals(
        analysis_result=flood_result,
        sensor="sentinel2",
        event_date=datetime(2024, 1, 15)
    )

    # Access generated images
    print(f"Satellite image: {manifest.satellite_image}")
    print(f"Before/after: {manifest.side_by_side}")
    print(f"Detection overlay: {manifest.detection_overlay}")

    # Get web-optimized version
    web_image = pipeline.get_web_optimized(manifest.satellite_image)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib
import json

import numpy as np
from PIL import Image

from .renderer import ImageryRenderer, RendererConfig, RenderedImage
from .before_after import BeforeAfterGenerator, BeforeAfterResult, BeforeAfterConfig, OutputConfig
from .overlay import DetectionOverlay, OverlayConfig, OverlayResult
from .export import ImageExporter, ExportConfig
from .comparison import normalize_histograms


@dataclass
class PipelineConfig:
    """
    Configuration for visual product pipeline.

    Attributes:
        cache_ttl_hours: Hours until cached images expire (0 = no caching)
        web_max_width: Maximum width for web-optimized images in pixels
        web_jpeg_quality: JPEG quality for web images (1-100)
        print_dpi: DPI for print-optimized images
        before_after_gap: Gap width between before/after images in pixels
        overlay_alpha: Base transparency for detection overlays (0-1)
        generate_animated_gif: Whether to generate animated GIF for before/after
        gif_frame_duration_ms: Duration of each GIF frame in milliseconds
    """

    cache_ttl_hours: int = 24
    web_max_width: int = 1200
    web_jpeg_quality: int = 85
    print_dpi: int = 300
    before_after_gap: int = 20
    overlay_alpha: float = 0.6
    generate_animated_gif: bool = True
    gif_frame_duration_ms: int = 1000

    def __post_init__(self):
        """Validate configuration."""
        if self.cache_ttl_hours < 0:
            raise ValueError(
                f"cache_ttl_hours must be non-negative, got {self.cache_ttl_hours}"
            )

        if self.web_max_width <= 0:
            raise ValueError(
                f"web_max_width must be positive, got {self.web_max_width}"
            )

        if not 1 <= self.web_jpeg_quality <= 100:
            raise ValueError(
                f"web_jpeg_quality must be 1-100, got {self.web_jpeg_quality}"
            )

        if self.print_dpi <= 0:
            raise ValueError(f"print_dpi must be positive, got {self.print_dpi}")

        if not 0 <= self.overlay_alpha <= 1:
            raise ValueError(
                f"overlay_alpha must be 0-1, got {self.overlay_alpha}"
            )


@dataclass
class ImageManifest:
    """
    Manifest tracking all generated visual products.

    Attributes:
        satellite_image: Path to rendered satellite imagery
        before_image: Path to before image (if available)
        after_image: Path to after image (if available)
        detection_overlay: Path to detection overlay composite
        side_by_side: Path to before/after side-by-side comparison
        labeled_comparison: Path to labeled before/after comparison
        animated_gif: Path to animated GIF (optional)
        web_satellite_image: Path to web-optimized satellite image
        web_detection_overlay: Path to web-optimized detection overlay
        print_satellite_image: Path to print-optimized satellite image
        print_detection_overlay: Path to print-optimized detection overlay
        generated_at: Timestamp when images were generated
        cache_valid_until: Timestamp when cache expires
        metadata: Additional metadata about the visual products
    """

    satellite_image: Optional[Path] = None
    before_image: Optional[Path] = None
    after_image: Optional[Path] = None
    detection_overlay: Optional[Path] = None
    side_by_side: Optional[Path] = None
    labeled_comparison: Optional[Path] = None
    animated_gif: Optional[Path] = None
    web_satellite_image: Optional[Path] = None
    web_detection_overlay: Optional[Path] = None
    print_satellite_image: Optional[Path] = None
    print_detection_overlay: Optional[Path] = None
    generated_at: datetime = field(default_factory=datetime.now)
    cache_valid_until: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert manifest to dictionary for JSON serialization."""
        return {
            "satellite_image": str(self.satellite_image) if self.satellite_image else None,
            "before_image": str(self.before_image) if self.before_image else None,
            "after_image": str(self.after_image) if self.after_image else None,
            "detection_overlay": str(self.detection_overlay) if self.detection_overlay else None,
            "side_by_side": str(self.side_by_side) if self.side_by_side else None,
            "labeled_comparison": str(self.labeled_comparison) if self.labeled_comparison else None,
            "animated_gif": str(self.animated_gif) if self.animated_gif else None,
            "web_satellite_image": str(self.web_satellite_image) if self.web_satellite_image else None,
            "web_detection_overlay": str(self.web_detection_overlay) if self.web_detection_overlay else None,
            "print_satellite_image": str(self.print_satellite_image) if self.print_satellite_image else None,
            "print_detection_overlay": str(self.print_detection_overlay) if self.print_detection_overlay else None,
            "generated_at": self.generated_at.isoformat(),
            "cache_valid_until": self.cache_valid_until.isoformat() if self.cache_valid_until else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ImageManifest":
        """Create manifest from dictionary."""
        return cls(
            satellite_image=Path(data["satellite_image"]) if data.get("satellite_image") else None,
            before_image=Path(data["before_image"]) if data.get("before_image") else None,
            after_image=Path(data["after_image"]) if data.get("after_image") else None,
            detection_overlay=Path(data["detection_overlay"]) if data.get("detection_overlay") else None,
            side_by_side=Path(data["side_by_side"]) if data.get("side_by_side") else None,
            labeled_comparison=Path(data["labeled_comparison"]) if data.get("labeled_comparison") else None,
            animated_gif=Path(data["animated_gif"]) if data.get("animated_gif") else None,
            web_satellite_image=Path(data["web_satellite_image"]) if data.get("web_satellite_image") else None,
            web_detection_overlay=Path(data["web_detection_overlay"]) if data.get("web_detection_overlay") else None,
            print_satellite_image=Path(data["print_satellite_image"]) if data.get("print_satellite_image") else None,
            print_detection_overlay=Path(data["print_detection_overlay"]) if data.get("print_detection_overlay") else None,
            generated_at=datetime.fromisoformat(data["generated_at"]),
            cache_valid_until=datetime.fromisoformat(data["cache_valid_until"]) if data.get("cache_valid_until") else None,
            metadata=data.get("metadata", {}),
        )


class ReportVisualPipeline:
    """
    Orchestrator for generating visual products for reports.

    Wires together satellite imagery rendering, before/after generation,
    and detection overlays into a unified pipeline with caching support.

    Examples:
        # Basic usage
        pipeline = ReportVisualPipeline(output_dir=Path("./outputs"))
        manifest = pipeline.generate_all_visuals(
            bands=band_dict,
            detection_mask=flood_mask,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15)
        )

        # With before/after imagery
        manifest = pipeline.generate_all_visuals(
            bands=band_dict,
            detection_mask=flood_mask,
            sensor="sentinel2",
            event_date=datetime(2024, 1, 15),
            before_bands=before_band_dict,
            before_date=datetime(2024, 1, 1)
        )

        # Get web-optimized version
        web_path = pipeline.get_web_optimized(manifest.satellite_image)
    """

    def __init__(
        self,
        output_dir: Path,
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize visual product pipeline.

        Args:
            output_dir: Directory for generated images
            config: Pipeline configuration (uses defaults if None)
        """
        self.output_dir = Path(output_dir)
        self.config = config if config is not None else PipelineConfig()

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.output_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.web_dir = self.output_dir / "web"
        self.web_dir.mkdir(exist_ok=True)
        self.print_dir = self.output_dir / "print"
        self.print_dir.mkdir(exist_ok=True)

        # Initialize component generators
        self.imagery_renderer = ImageryRenderer()
        self.before_after_generator = BeforeAfterGenerator()
        self.overlay_renderer = DetectionOverlay()
        self.image_exporter = ImageExporter()

    def generate_satellite_imagery(
        self,
        bands: Dict[str, np.ndarray],
        sensor: str,
        composite_name: str = "true_color",
        nodata_value: Optional[float] = None
    ) -> Path:
        """
        Render satellite bands to viewable RGB image.

        Args:
            bands: Dictionary mapping band names to 2D arrays
            sensor: Sensor identifier ('sentinel2', 'landsat8', etc.)
            composite_name: Band composite to use (e.g., 'true_color')
            nodata_value: Optional nodata value to handle

        Returns:
            Path to saved satellite imagery PNG

        Example:
            >>> bands = {"B04": red, "B03": green, "B02": blue}
            >>> path = pipeline.generate_satellite_imagery(bands, "sentinel2")
        """
        # Generate cache key
        cache_key = self._compute_cache_key("satellite", bands, sensor, composite_name)

        # Check cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Render imagery
        config = RendererConfig(composite_name=composite_name, output_format="uint8")
        renderer = ImageryRenderer(config)
        result = renderer.render(bands, sensor=sensor, nodata_value=nodata_value)

        # Save to file
        output_path = self.output_dir / f"satellite_{cache_key}.png"
        self.image_exporter.export_png(result.rgb_array, output_path)

        # Save to cache
        self._save_to_cache(cache_key, output_path)

        return output_path

    def generate_before_after(
        self,
        before_bands: Dict[str, np.ndarray],
        after_bands: Dict[str, np.ndarray],
        before_date: datetime,
        after_date: datetime,
        sensor: str,
        composite_name: str = "true_color"
    ) -> Tuple[Path, Path, Path, Optional[Path]]:
        """
        Generate before/after comparison pair with side-by-side composite.

        Args:
            before_bands: Band dictionary for before image
            after_bands: Band dictionary for after image
            before_date: Acquisition date of before image
            after_date: Acquisition date of after image
            sensor: Sensor identifier
            composite_name: Band composite to use

        Returns:
            Tuple of (before_path, after_path, side_by_side_path, labeled_path, gif_path)

        Example:
            >>> before, after, composite, labeled, gif = pipeline.generate_before_after(
            ...     before_bands, after_bands,
            ...     datetime(2024, 1, 1), datetime(2024, 1, 15),
            ...     "sentinel2"
            ... )
        """
        # Generate cache key
        cache_key = self._compute_cache_key(
            "before_after", before_bands, after_bands, before_date, after_date
        )

        # Check cache for side-by-side (if it exists, all exist)
        side_by_side_path = self.output_dir / f"before_after_{cache_key}.png"
        if side_by_side_path.exists() and self._is_cache_valid(cache_key):
            before_path = self.output_dir / f"before_{cache_key}.png"
            after_path = self.output_dir / f"after_{cache_key}.png"
            labeled_path = self.output_dir / f"before_after_labeled_{cache_key}.png"
            gif_path = self.output_dir / f"before_after_{cache_key}.gif"
            return (
                before_path,
                after_path,
                side_by_side_path,
                labeled_path,
                gif_path if gif_path.exists() else None
            )

        # Render before and after images
        config = RendererConfig(composite_name=composite_name, output_format="uint8")
        renderer = ImageryRenderer(config)

        before_result = renderer.render(before_bands, sensor=sensor)
        after_result = renderer.render(after_bands, sensor=sensor)

        # Ensure images are same size (should be from same data source)
        if before_result.rgb_array.shape != after_result.rgb_array.shape:
            raise ValueError(
                f"Before and after images must have same shape. "
                f"Got before={before_result.rgb_array.shape}, "
                f"after={after_result.rgb_array.shape}"
            )

        # Normalize histograms for consistent appearance
        before_norm, after_norm = normalize_histograms(
            before_result.rgb_array, after_result.rgb_array
        )

        # Create BeforeAfterResult
        ba_result = BeforeAfterResult(
            before_image=before_norm,
            after_image=after_norm,
            before_date=before_date,
            after_date=after_date
        )

        # Generate comparison products
        output_config = OutputConfig(gap_width=self.config.before_after_gap)

        # Side-by-side
        side_by_side = self.before_after_generator.generate_side_by_side(
            ba_result, config=output_config
        )
        side_by_side_path = self.output_dir / f"before_after_{cache_key}.png"
        Image.fromarray(side_by_side).save(side_by_side_path)

        # Labeled comparison
        labeled = self.before_after_generator.generate_labeled_comparison(
            ba_result, config=output_config
        )
        labeled_path = self.output_dir / f"before_after_labeled_{cache_key}.png"
        Image.fromarray(labeled).save(labeled_path)

        # Individual images
        before_path = self.output_dir / f"before_{cache_key}.png"
        after_path = self.output_dir / f"after_{cache_key}.png"
        Image.fromarray(ba_result.before_image).save(before_path)
        Image.fromarray(ba_result.after_image).save(after_path)

        # Animated GIF (optional)
        gif_path = None
        if self.config.generate_animated_gif:
            gif_path = self.output_dir / f"before_after_{cache_key}.gif"
            self.before_after_generator.generate_animated_gif(
                ba_result,
                gif_path,
                frame_duration_ms=self.config.gif_frame_duration_ms
            )

        # Save to cache
        self._save_to_cache(cache_key, side_by_side_path)

        return before_path, after_path, side_by_side_path, labeled_path, gif_path

    def generate_detection_overlay(
        self,
        background: np.ndarray,
        detection_mask: np.ndarray,
        overlay_type: str,
        confidence: Optional[np.ndarray] = None,
        severity: Optional[np.ndarray] = None
    ) -> Path:
        """
        Overlay detection results on satellite imagery.

        Args:
            background: RGB background image (H, W, 3)
            detection_mask: Binary detection mask (H, W)
            overlay_type: Type of overlay ('flood', 'fire')
            confidence: Optional confidence scores (H, W)
            severity: Optional severity values (H, W)

        Returns:
            Path to saved overlay composite image

        Example:
            >>> overlay_path = pipeline.generate_detection_overlay(
            ...     satellite_rgb,
            ...     flood_mask,
            ...     "flood",
            ...     confidence=confidence_array
            ... )
        """
        # Generate cache key
        cache_key = self._compute_cache_key(
            "overlay", background, detection_mask, overlay_type
        )

        # Check cache
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Create overlay
        config = OverlayConfig(
            overlay_type=overlay_type,
            alpha_base=self.config.overlay_alpha,
            use_confidence_alpha=True
        )
        overlay = DetectionOverlay(config)
        result = overlay.render(
            background=background,
            detection_mask=detection_mask,
            confidence=confidence,
            severity=severity
        )

        # Save composite
        output_path = self.output_dir / f"overlay_{overlay_type}_{cache_key}.png"
        Image.fromarray(result.composite_image).save(output_path)

        # Save legend separately
        legend_path = self.output_dir / f"legend_{overlay_type}_{cache_key}.png"
        Image.fromarray(result.legend_image).save(legend_path)

        # Save to cache
        self._save_to_cache(cache_key, output_path)

        return output_path

    def generate_all_visuals(
        self,
        bands: Dict[str, np.ndarray],
        sensor: str,
        event_date: datetime,
        detection_mask: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
        overlay_type: str = "flood",
        before_bands: Optional[Dict[str, np.ndarray]] = None,
        before_date: Optional[datetime] = None,
        composite_name: str = "true_color"
    ) -> ImageManifest:
        """
        Generate all visual products for a complete report.

        This is the main orchestration method that generates:
        - Satellite imagery (base visualization)
        - Before/after comparison (if before data provided)
        - Detection overlay (if detection mask provided)
        - Web-optimized versions of all images
        - Print-optimized versions of key images

        Args:
            bands: Band dictionary for primary (after) imagery
            sensor: Sensor identifier
            event_date: Date of primary imagery
            detection_mask: Optional detection mask for overlay
            confidence: Optional confidence scores
            overlay_type: Type of overlay ('flood', 'fire')
            before_bands: Optional before imagery bands
            before_date: Optional before imagery date
            composite_name: Band composite to use

        Returns:
            ImageManifest with paths to all generated products

        Example:
            >>> manifest = pipeline.generate_all_visuals(
            ...     bands=after_bands,
            ...     sensor="sentinel2",
            ...     event_date=datetime(2024, 1, 15),
            ...     detection_mask=flood_mask,
            ...     before_bands=before_bands,
            ...     before_date=datetime(2024, 1, 1)
            ... )
            >>> print(f"Web image: {manifest.web_satellite_image}")
        """
        manifest = ImageManifest()

        # 1. Generate satellite imagery
        satellite_path = self.generate_satellite_imagery(
            bands=bands,
            sensor=sensor,
            composite_name=composite_name
        )
        manifest.satellite_image = satellite_path

        # 2. Generate before/after if before data provided
        if before_bands is not None and before_date is not None:
            before_path, after_path, side_by_side, labeled, gif = self.generate_before_after(
                before_bands=before_bands,
                after_bands=bands,
                before_date=before_date,
                after_date=event_date,
                sensor=sensor,
                composite_name=composite_name
            )
            manifest.before_image = before_path
            manifest.after_image = after_path
            manifest.side_by_side = side_by_side
            manifest.labeled_comparison = labeled
            manifest.animated_gif = gif

        # 3. Generate detection overlay if detection mask provided
        if detection_mask is not None:
            # Load satellite image as background
            background = np.array(Image.open(satellite_path))
            overlay_path = self.generate_detection_overlay(
                background=background,
                detection_mask=detection_mask,
                overlay_type=overlay_type,
                confidence=confidence
            )
            manifest.detection_overlay = overlay_path

        # 4. Generate web-optimized versions
        manifest.web_satellite_image = self.get_web_optimized(satellite_path)
        if manifest.detection_overlay:
            manifest.web_detection_overlay = self.get_web_optimized(manifest.detection_overlay)

        # 5. Generate print-optimized versions
        manifest.print_satellite_image = self.get_print_optimized(satellite_path)
        if manifest.detection_overlay:
            manifest.print_detection_overlay = self.get_print_optimized(manifest.detection_overlay)

        # 6. Set cache expiry
        if self.config.cache_ttl_hours > 0:
            manifest.cache_valid_until = datetime.now() + timedelta(
                hours=self.config.cache_ttl_hours
            )

        # 7. Add metadata
        manifest.metadata = {
            "sensor": sensor,
            "event_date": event_date.isoformat(),
            "composite_name": composite_name,
            "overlay_type": overlay_type if detection_mask is not None else None,
            "has_before_after": before_bands is not None,
            "has_detection_overlay": detection_mask is not None,
        }

        # 8. Save manifest to disk
        self._save_manifest(manifest)

        return manifest

    def get_web_optimized(self, image_path: Path) -> Path:
        """
        Return web-optimized version of image (compressed, resized).

        Creates a JPEG version optimized for web display:
        - Resizes to max width if needed
        - Compresses to reduce file size (target < 500KB)
        - Caches result for future requests

        Args:
            image_path: Path to source image

        Returns:
            Path to web-optimized image

        Example:
            >>> web_path = pipeline.get_web_optimized(satellite_path)
            >>> # Result is compressed JPEG < 500KB
        """
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        # Generate web path
        web_path = self.web_dir / f"{image_path.stem}_web.jpg"

        # Return cached if exists and valid
        if web_path.exists() and web_path.stat().st_mtime >= image_path.stat().st_mtime:
            return web_path

        # Load and resize image
        img = Image.open(image_path)

        # Convert RGBA to RGB if needed
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha as mask
            img = background

        # Resize if wider than max width
        if img.width > self.config.web_max_width:
            ratio = self.config.web_max_width / img.width
            new_height = int(img.height * ratio)
            img = img.resize((self.config.web_max_width, new_height), Image.LANCZOS)

        # Save as JPEG with compression
        img.save(web_path, "JPEG", quality=self.config.web_jpeg_quality, optimize=True)

        return web_path

    def get_print_optimized(self, image_path: Path, dpi: Optional[int] = None) -> Path:
        """
        Return high-resolution version for PDF embedding (300 DPI).

        Creates a PNG version optimized for printing:
        - Saves with specified DPI metadata
        - No compression loss
        - Suitable for PDF embedding

        Args:
            image_path: Path to source image
            dpi: DPI for print output (uses config default if None)

        Returns:
            Path to print-optimized image

        Example:
            >>> print_path = pipeline.get_print_optimized(satellite_path, dpi=300)
        """
        if not image_path or not image_path.exists():
            raise FileNotFoundError(f"Source image not found: {image_path}")

        dpi = dpi if dpi is not None else self.config.print_dpi

        # Generate print path
        print_path = self.print_dir / f"{image_path.stem}_print.png"

        # Return cached if exists and valid
        if print_path.exists() and print_path.stat().st_mtime >= image_path.stat().st_mtime:
            return print_path

        # Load image and save with DPI metadata
        img = Image.open(image_path)
        img.save(print_path, "PNG", dpi=(dpi, dpi))

        return print_path

    # === Private Methods ===

    def _compute_cache_key(self, prefix: str, *args) -> str:
        """Compute cache key from arguments using hash."""
        # Create a string representation of all arguments
        key_parts = [prefix]
        for arg in args:
            if isinstance(arg, np.ndarray):
                # Hash array contents
                key_parts.append(hashlib.sha256(arg.tobytes()).hexdigest()[:16])
            elif isinstance(arg, dict):
                # Hash dictionary keys and array contents
                dict_str = str(sorted(arg.keys()))
                key_parts.append(hashlib.sha256(dict_str.encode()).hexdigest()[:16])
            elif isinstance(arg, datetime):
                key_parts.append(arg.isoformat())
            else:
                key_parts.append(str(arg))

        # Create final hash
        combined = "_".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _load_from_cache(self, cache_key: str) -> Optional[Path]:
        """Load path from cache if valid."""
        if self.config.cache_ttl_hours == 0:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # Load cache entry
        with open(cache_file, "r") as f:
            entry = json.load(f)

        # Check if expired
        if entry.get("expires_at"):
            expires_at = datetime.fromisoformat(entry["expires_at"])
            if datetime.now() > expires_at:
                return None

        # Check if file still exists
        cached_path = Path(entry["path"])
        if not cached_path.exists():
            return None

        return cached_path

    def _save_to_cache(self, cache_key: str, path: Path):
        """Save path to cache with expiry."""
        if self.config.cache_ttl_hours == 0:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        expires_at = datetime.now() + timedelta(hours=self.config.cache_ttl_hours)

        entry = {
            "path": str(path),
            "cached_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(entry, f, indent=2)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        return self._load_from_cache(cache_key) is not None

    def _save_manifest(self, manifest: ImageManifest):
        """Save manifest to disk as JSON."""
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest.to_dict(), f, indent=2)

    def load_manifest(self) -> Optional[ImageManifest]:
        """
        Load manifest from disk if it exists.

        Returns:
            ImageManifest if found and valid, None otherwise
        """
        manifest_path = self.output_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        with open(manifest_path, "r") as f:
            data = json.load(f)

        manifest = ImageManifest.from_dict(data)

        # Check if cache is still valid
        if manifest.cache_valid_until and datetime.now() > manifest.cache_valid_until:
            return None

        return manifest
