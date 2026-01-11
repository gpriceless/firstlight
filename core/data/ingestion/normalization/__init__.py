"""
Normalization Tools for Ingestion Pipeline.

Provides tools for normalizing geospatial data across different:
- Coordinate reference systems (projection)
- Tile schemes and grids (tiling)
- Temporal resolutions (temporal)
- Spatial resolutions (resolution)

Each module follows a consistent pattern:
- Config dataclass for operation options
- Result dataclass with operation details
- Handler/Processor class with main methods
- Convenience functions for simple usage
"""

# Projection and CRS handling
from core.data.ingestion.normalization.projection import (
    CRSHandler,
    CRSInfo,
    CRSType,
    RasterReprojector,
    ReprojectionConfig,
    ReprojectionResult,
    ResamplingMethod as ProjectionResamplingMethod,
    VectorReprojector,
    get_crs_info,
    reproject_to_crs,
    suggest_target_crs,
)

# Tiling and tile schemes
from core.data.ingestion.normalization.tiling import (
    RasterTiler,
    Tile,
    TileBounds,
    TileGrid,
    TileGridConfig,
    TileGridInfo,
    TileIndex,
    TileOrigin,
    TileScheme,
    WebMercatorTiles,
    create_tile_grid,
    get_web_mercator_tiles,
)

# Temporal alignment
from core.data.ingestion.normalization.temporal import (
    AggregationMethod,
    AlignmentResult,
    InterpolationMethod,
    TemporalAligner,
    TemporalAlignmentConfig,
    TemporalResampler,
    TemporalResolution,
    TemporalSample,
    TimeRange,
    TimestampHandler,
    align_samples,
    generate_time_range,
    parse_timestamp,
)

# Resolution and resampling
from core.data.ingestion.normalization.resolution import (
    Resolution,
    ResamplingConfig,
    ResamplingMethod,
    ResamplingResult,
    ResolutionCalculator,
    ResolutionHarmonizer,
    ResolutionUnit,
    SpatialResampler,
    calculate_resolution,
    resample_to_resolution,
)

__all__ = [
    # Projection
    "CRSHandler",
    "CRSInfo",
    "CRSType",
    "RasterReprojector",
    "ReprojectionConfig",
    "ReprojectionResult",
    "ProjectionResamplingMethod",
    "VectorReprojector",
    "get_crs_info",
    "reproject_to_crs",
    "suggest_target_crs",
    # Tiling
    "RasterTiler",
    "Tile",
    "TileBounds",
    "TileGrid",
    "TileGridConfig",
    "TileGridInfo",
    "TileIndex",
    "TileOrigin",
    "TileScheme",
    "WebMercatorTiles",
    "create_tile_grid",
    "get_web_mercator_tiles",
    # Temporal
    "AggregationMethod",
    "AlignmentResult",
    "InterpolationMethod",
    "TemporalAligner",
    "TemporalAlignmentConfig",
    "TemporalResampler",
    "TemporalResolution",
    "TemporalSample",
    "TimeRange",
    "TimestampHandler",
    "align_samples",
    "generate_time_range",
    "parse_timestamp",
    # Resolution
    "Resolution",
    "ResamplingConfig",
    "ResamplingMethod",
    "ResamplingResult",
    "ResolutionCalculator",
    "ResolutionHarmonizer",
    "ResolutionUnit",
    "SpatialResampler",
    "calculate_resolution",
    "resample_to_resolution",
]
