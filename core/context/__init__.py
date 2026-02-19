"""
Context Data Lakehouse models and repository.

Provides Pydantic models for context records (datasets, buildings,
infrastructure, weather) and a ContextRepository for PostGIS-backed
storage with deduplication and spatial queries.
"""

from core.context.models import (
    BuildingRecord,
    ContextResult,
    ContextSummary,
    DatasetRecord,
    InfrastructureRecord,
    JobContextUsage,
    WeatherRecord,
)

__all__ = [
    "BuildingRecord",
    "ContextResult",
    "ContextSummary",
    "DatasetRecord",
    "InfrastructureRecord",
    "JobContextUsage",
    "WeatherRecord",
]
