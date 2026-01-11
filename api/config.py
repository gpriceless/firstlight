"""
API Configuration using Pydantic Settings.

Provides centralized configuration management with environment variable loading,
validation, and sensible defaults for development and production environments.
"""

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment modes."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        extra="ignore",
    )

    driver: str = Field(default="postgresql+asyncpg", description="Database driver")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="multiverse_dive", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Max overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    echo: bool = Field(default=False, description="Echo SQL statements")

    @property
    def url(self) -> str:
        """Construct database connection URL."""
        if self.password:
            return f"{self.driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        return f"{self.driver}://{self.user}@{self.host}:{self.port}/{self.name}"


class RedisSettings(BaseSettings):
    """Redis connection settings for caching and pub/sub."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ssl: bool = Field(default=False, description="Use SSL connection")
    pool_size: int = Field(default=10, description="Connection pool size")

    @property
    def url(self) -> str:
        """Construct Redis connection URL."""
        scheme = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{scheme}://{auth}{self.host}:{self.port}/{self.db}"


class StorageSettings(BaseSettings):
    """Object storage settings for products and artifacts."""

    model_config = SettingsConfigDict(
        env_prefix="STORAGE_",
        env_file=".env",
        extra="ignore",
    )

    backend: str = Field(default="local", description="Storage backend: local, s3, gcs")
    local_path: Path = Field(
        default=Path("./data/products"), description="Local storage path"
    )
    bucket: str = Field(default="multiverse-dive-products", description="S3/GCS bucket")
    region: str = Field(default="us-east-1", description="Cloud region")
    endpoint_url: Optional[str] = Field(
        default=None, description="Custom S3 endpoint URL"
    )
    access_key: Optional[str] = Field(default=None, description="Access key")
    secret_key: Optional[str] = Field(default=None, description="Secret key")


class AuthSettings(BaseSettings):
    """Authentication and authorization settings."""

    model_config = SettingsConfigDict(
        env_prefix="AUTH_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable authentication")
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        description="JWT secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30, description="Access token expiration in minutes"
    )
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    allowed_api_keys: List[str] = Field(
        default_factory=list, description="List of allowed API keys"
    )

    @field_validator("allowed_api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> List[str]:
        """Parse comma-separated API keys from environment variable."""
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v or []


class CORSSettings(BaseSettings):
    """Cross-Origin Resource Sharing settings."""

    model_config = SettingsConfigDict(
        env_prefix="CORS_",
        env_file=".env",
        extra="ignore",
    )

    enabled: bool = Field(default=True, description="Enable CORS")
    allow_origins: List[str] = Field(
        default_factory=lambda: ["*"], description="Allowed origins"
    )
    allow_methods: List[str] = Field(
        default_factory=lambda: ["*"], description="Allowed HTTP methods"
    )
    allow_headers: List[str] = Field(
        default_factory=lambda: ["*"], description="Allowed headers"
    )
    allow_credentials: bool = Field(default=False, description="Allow credentials")
    max_age: int = Field(default=600, description="Preflight cache max age")

    @field_validator("allow_origins", "allow_methods", "allow_headers", mode="before")
    @classmethod
    def parse_list(cls, v: Any) -> List[str]:
        """Parse comma-separated lists from environment variables."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v or []


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="MULTIVERSE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(
        default="Multiverse Dive", description="Application name"
    )
    app_version: str = Field(default="0.1.0", description="Application version")
    environment: Environment = Field(
        default=Environment.DEVELOPMENT, description="Environment mode"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")

    # API Server
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # Request handling
    request_timeout: int = Field(
        default=300, description="Request timeout in seconds"
    )
    max_request_size: int = Field(
        default=100 * 1024 * 1024, description="Max request size in bytes"
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(
        default=100, description="Max requests per window"
    )
    rate_limit_window: int = Field(
        default=60, description="Rate limit window in seconds"
    )

    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)

    # OpenSpec paths
    openspec_schemas_dir: Path = Field(
        default=Path("openspec/schemas"), description="Path to OpenSpec schemas"
    )
    openspec_definitions_dir: Path = Field(
        default=Path("openspec/definitions"), description="Path to OpenSpec definitions"
    )

    @model_validator(mode="after")
    def configure_environment_defaults(self) -> "Settings":
        """Set environment-specific defaults."""
        if self.environment == Environment.DEVELOPMENT:
            if not self.debug:
                object.__setattr__(self, "debug", True)
            if self.log_level == LogLevel.INFO:
                object.__setattr__(self, "log_level", LogLevel.DEBUG)
            if not self.reload:
                object.__setattr__(self, "reload", True)
        elif self.environment == Environment.PRODUCTION:
            if self.debug:
                object.__setattr__(self, "debug", False)
            if self.reload:
                object.__setattr__(self, "reload", False)
        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == Environment.TESTING

    def get_openapi_config(self) -> Dict[str, Any]:
        """Get OpenAPI configuration."""
        return {
            "title": self.app_name,
            "version": self.app_version,
            "description": (
                "Geospatial Event Intelligence Platform API. "
                "Transforms (area, time window, event type) into decision products."
            ),
            "contact": {
                "name": "Multiverse Dive Team",
                "url": "https://github.com/multiverse-dive",
            },
            "license_info": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.

    Returns:
        Settings instance with loaded configuration.
    """
    return Settings()


def get_settings_uncached() -> Settings:
    """
    Get fresh settings instance (useful for testing).

    Returns:
        New Settings instance with loaded configuration.
    """
    return Settings()
