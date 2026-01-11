"""
Enrichment Tools for Ingested Data.

Provides tools to enrich ingested data with additional metadata and
derived products, improving usability and enabling quality-based decisions.

Modules:
- overviews: Image pyramid generation for multi-resolution display
- statistics: Band statistics computation for analysis and visualization
- quality: Quality assessment and summary generation

Each enrichment tool follows a consistent pattern:
- Config dataclass for enrichment options
- Result dataclass with enrichment details
- Main class with enrich/compute methods
- Convenience function for simple usage
"""

from core.data.ingestion.enrichment.overviews import (
    OverviewConfig,
    OverviewFormat,
    OverviewGenerator,
    OverviewLevel,
    OverviewResampling,
    OverviewResult,
    generate_overviews,
)
from core.data.ingestion.enrichment.quality import (
    DimensionScore,
    QualityAssessor,
    QualityConfig,
    QualityDimension,
    QualityFlag,
    QualityIssue,
    QualityLevel,
    QualitySummary,
    assess_quality,
)
from core.data.ingestion.enrichment.statistics import (
    BandStatistics,
    HistogramType,
    RasterStatistics,
    StatisticsCalculator,
    StatisticsConfig,
    compute_statistics,
)

__all__ = [
    # Overviews
    "OverviewConfig",
    "OverviewFormat",
    "OverviewGenerator",
    "OverviewLevel",
    "OverviewResampling",
    "OverviewResult",
    "generate_overviews",
    # Statistics
    "BandStatistics",
    "HistogramType",
    "RasterStatistics",
    "StatisticsCalculator",
    "StatisticsConfig",
    "compute_statistics",
    # Quality
    "DimensionScore",
    "QualityAssessor",
    "QualityConfig",
    "QualityDimension",
    "QualityFlag",
    "QualityIssue",
    "QualityLevel",
    "QualitySummary",
    "assess_quality",
]
