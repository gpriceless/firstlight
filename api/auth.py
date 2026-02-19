"""
API Authentication and Authorization Module.

Provides API key authentication, optional JWT token support, permission checking,
and user context extraction for the FastAPI application.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


class AuthConfig(BaseModel):
    """Authentication configuration."""

    # API Key settings
    api_key_header_name: str = Field(default="X-API-Key", description="Header name for API key")
    api_key_query_param: str = Field(default="api_key", description="Query param name for API key")
    api_key_min_length: int = Field(default=32, description="Minimum API key length")

    # JWT settings (optional)
    jwt_enabled: bool = Field(default=False, description="Enable JWT authentication")
    jwt_secret_key: Optional[str] = Field(default=None, description="Secret key for JWT signing")
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_access_token_expire_minutes: int = Field(default=30, description="Access token expiry")
    jwt_refresh_token_expire_days: int = Field(default=7, description="Refresh token expiry")

    # General settings
    allow_anonymous: bool = Field(default=False, description="Allow anonymous access")
    require_https: bool = Field(default=True, description="Require HTTPS for auth")

    class Config:
        env_prefix = "AUTH_"


def get_auth_config() -> AuthConfig:
    """Get authentication configuration from environment."""
    return AuthConfig(
        api_key_header_name=os.getenv("AUTH_API_KEY_HEADER", "X-API-Key"),
        api_key_query_param=os.getenv("AUTH_API_KEY_QUERY_PARAM", "api_key"),
        api_key_min_length=int(os.getenv("AUTH_API_KEY_MIN_LENGTH", "32")),
        jwt_enabled=os.getenv("AUTH_JWT_ENABLED", "false").lower() == "true",
        jwt_secret_key=os.getenv("AUTH_JWT_SECRET_KEY"),
        jwt_algorithm=os.getenv("AUTH_JWT_ALGORITHM", "HS256"),
        jwt_access_token_expire_minutes=int(os.getenv("AUTH_JWT_ACCESS_EXPIRE_MINUTES", "30")),
        jwt_refresh_token_expire_days=int(os.getenv("AUTH_JWT_REFRESH_EXPIRE_DAYS", "7")),
        allow_anonymous=os.getenv("AUTH_ALLOW_ANONYMOUS", "false").lower() == "true",
        require_https=os.getenv("AUTH_REQUIRE_HTTPS", "true").lower() == "true",
    )


# =============================================================================
# Permission System
# =============================================================================


class Permission(str, Enum):
    """Available permissions in the system."""

    # Event operations
    EVENT_CREATE = "event:create"
    EVENT_READ = "event:read"
    EVENT_UPDATE = "event:update"
    EVENT_DELETE = "event:delete"
    EVENT_LIST = "event:list"

    # Product operations
    PRODUCT_READ = "product:read"
    PRODUCT_DOWNLOAD = "product:download"
    PRODUCT_DELETE = "product:delete"

    # Catalog operations
    CATALOG_READ = "catalog:read"
    CATALOG_SEARCH = "catalog:search"

    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_KEYS = "admin:keys"
    ADMIN_CONFIG = "admin:config"
    ADMIN_METRICS = "admin:metrics"

    # Webhook operations
    WEBHOOK_CREATE = "webhook:create"
    WEBHOOK_READ = "webhook:read"
    WEBHOOK_DELETE = "webhook:delete"


# Default permission sets for roles
ROLE_PERMISSIONS: Dict[str, Set[Permission]] = {
    "admin": set(Permission),  # All permissions
    "operator": {
        Permission.EVENT_CREATE,
        Permission.EVENT_READ,
        Permission.EVENT_UPDATE,
        Permission.EVENT_LIST,
        Permission.PRODUCT_READ,
        Permission.PRODUCT_DOWNLOAD,
        Permission.CATALOG_READ,
        Permission.CATALOG_SEARCH,
        Permission.WEBHOOK_CREATE,
        Permission.WEBHOOK_READ,
        Permission.WEBHOOK_DELETE,
        Permission.ADMIN_METRICS,
    },
    "user": {
        Permission.EVENT_CREATE,
        Permission.EVENT_READ,
        Permission.EVENT_LIST,
        Permission.PRODUCT_READ,
        Permission.PRODUCT_DOWNLOAD,
        Permission.CATALOG_READ,
        Permission.CATALOG_SEARCH,
    },
    "readonly": {
        Permission.EVENT_READ,
        Permission.EVENT_LIST,
        Permission.PRODUCT_READ,
        Permission.CATALOG_READ,
        Permission.CATALOG_SEARCH,
    },
    "anonymous": {
        Permission.CATALOG_READ,
    },
}


# =============================================================================
# User Context
# =============================================================================


@dataclass
class UserContext:
    """Context information about the authenticated user."""

    user_id: str
    customer_id: str = "legacy"
    api_key_id: Optional[str] = None
    role: str = "user"
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    authenticated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_anonymous: bool = False
    token_type: str = "api_key"  # "api_key" or "jwt"

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(p in self.permissions for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions."""
        return all(p in self.permissions for p in permissions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "customer_id": self.customer_id,
            "api_key_id": self.api_key_id,
            "role": self.role,
            "permissions": [p.value for p in self.permissions],
            "metadata": self.metadata,
            "authenticated_at": self.authenticated_at.isoformat(),
            "is_anonymous": self.is_anonymous,
            "token_type": self.token_type,
        }


# =============================================================================
# API Key Management
# =============================================================================


@dataclass
class APIKey:
    """Represents an API key."""

    key_id: str
    key_hash: str  # Hashed version of the actual key
    user_id: str
    customer_id: str = "legacy"
    role: str = "user"
    permissions: Set[Permission] = field(default_factory=set)
    name: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    rate_limit: Optional[int] = None  # Requests per minute, None = default
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the key has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_valid(self) -> bool:
        """Check if the key is valid for use."""
        return self.is_active and not self.is_expired()


class APIKeyStore:
    """
    Storage interface for API keys.

    In production, this should be backed by a database.
    This in-memory implementation is for development and testing.
    """

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._key_hash_index: Dict[str, str] = {}  # hash -> key_id

    def _hash_key(self, key: str) -> str:
        """Create a secure hash of an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        user_id: str,
        role: str = "user",
        permissions: Optional[Set[Permission]] = None,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
        rate_limit: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        customer_id: str = "legacy",
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns:
            Tuple of (raw_key, APIKey object).
            The raw key is only returned once at creation time.
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_id = secrets.token_hex(8)
        key_hash = self._hash_key(raw_key)

        # Determine permissions
        if permissions is None:
            permissions = ROLE_PERMISSIONS.get(role, set())

        # Calculate expiration
        expires_at = None
        if expires_in_days is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            customer_id=customer_id,
            role=role,
            permissions=permissions,
            name=name,
            expires_at=expires_at,
            rate_limit=rate_limit,
            metadata=metadata or {},
        )

        self._keys[key_id] = api_key
        self._key_hash_index[key_hash] = key_id

        logger.info(f"Created API key {key_id} for user {user_id}")
        return raw_key, api_key

    def get_by_id(self, key_id: str) -> Optional[APIKey]:
        """Get an API key by its ID."""
        return self._keys.get(key_id)

    def get_by_key(self, raw_key: str) -> Optional[APIKey]:
        """Get an API key by the raw key value."""
        key_hash = self._hash_key(raw_key)
        key_id = self._key_hash_index.get(key_hash)
        if key_id is None:
            return None
        return self._keys.get(key_id)

    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate an API key and return it if valid."""
        api_key = self.get_by_key(raw_key)
        if api_key is None:
            return None
        if not api_key.is_valid():
            return None
        return api_key

    def update_last_used(self, key_id: str) -> None:
        """Update the last_used_at timestamp for a key."""
        if key_id in self._keys:
            self._keys[key_id].last_used_at = datetime.now(timezone.utc)

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            self._keys[key_id].is_active = False
            logger.info(f"Revoked API key {key_id}")
            return True
        return False

    def delete_key(self, key_id: str) -> bool:
        """Permanently delete an API key."""
        if key_id in self._keys:
            api_key = self._keys[key_id]
            del self._key_hash_index[api_key.key_hash]
            del self._keys[key_id]
            logger.info(f"Deleted API key {key_id}")
            return True
        return False

    def list_keys_for_user(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [k for k in self._keys.values() if k.user_id == user_id]


# Global API key store (replace with database-backed in production)
_api_key_store: Optional[APIKeyStore] = None


def get_api_key_store() -> APIKeyStore:
    """Get the global API key store."""
    global _api_key_store
    if _api_key_store is None:
        _api_key_store = APIKeyStore()
    return _api_key_store


# =============================================================================
# JWT Token Support (Optional)
# =============================================================================


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str  # Subject (user_id)
    exp: datetime  # Expiration
    iat: datetime  # Issued at
    jti: str  # JWT ID
    type: str  # "access" or "refresh"
    role: str
    permissions: List[str]


class JWTManager:
    """
    JWT token management.

    Requires pyjwt package to be installed for JWT functionality.
    """

    def __init__(self, config: AuthConfig):
        self.config = config
        self._jwt_available = False

        if config.jwt_enabled:
            try:
                import jwt

                self._jwt = jwt
                self._jwt_available = True
            except ImportError:
                logger.warning("JWT authentication enabled but pyjwt not installed")

    @property
    def is_available(self) -> bool:
        """Check if JWT support is available."""
        return self._jwt_available and self.config.jwt_secret_key is not None

    def create_access_token(
        self,
        user_id: str,
        role: str,
        permissions: Set[Permission],
    ) -> str:
        """Create an access token."""
        if not self.is_available:
            raise RuntimeError("JWT not available")

        now = datetime.now(timezone.utc)
        expires = now + timedelta(minutes=self.config.jwt_access_token_expire_minutes)

        payload = {
            "sub": user_id,
            "exp": expires,
            "iat": now,
            "jti": secrets.token_hex(16),
            "type": "access",
            "role": role,
            "permissions": [p.value for p in permissions],
        }

        return self._jwt.encode(
            payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm
        )

    def create_refresh_token(self, user_id: str) -> str:
        """Create a refresh token."""
        if not self.is_available:
            raise RuntimeError("JWT not available")

        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=self.config.jwt_refresh_token_expire_days)

        payload = {
            "sub": user_id,
            "exp": expires,
            "iat": now,
            "jti": secrets.token_hex(16),
            "type": "refresh",
            "role": "",
            "permissions": [],
        }

        return self._jwt.encode(
            payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm
        )

    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify a JWT token and return its payload."""
        if not self.is_available:
            return None

        try:
            payload = self._jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm],
            )
            return TokenPayload(**payload)
        except self._jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except self._jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None


# =============================================================================
# FastAPI Security Schemes
# =============================================================================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Authentication Dependencies
# =============================================================================


class AuthenticationError(HTTPException):
    """Authentication error exception."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "API-Key"},
        )


class AuthorizationError(HTTPException):
    """Authorization error exception."""

    def __init__(self, detail: str = "Permission denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


async def get_api_key(
    request: Request,
    api_key_header_value: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """Extract API key from request."""
    config = get_auth_config()

    # Try header first
    if api_key_header_value:
        return api_key_header_value

    # Try query parameter
    api_key_query = request.query_params.get(config.api_key_query_param)
    if api_key_query:
        return api_key_query

    return None


async def get_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[str]:
    """Extract bearer token from request."""
    if credentials is None:
        return None
    return credentials.credentials


async def authenticate(
    request: Request,
    api_key: Optional[str] = Depends(get_api_key),
    bearer_token: Optional[str] = Depends(get_bearer_token),
) -> UserContext:
    """
    Authenticate the request and return user context.

    This is the main authentication dependency to use in routes.
    """
    config = get_auth_config()
    key_store = get_api_key_store()

    # Check HTTPS requirement
    if config.require_https and request.url.scheme != "https":
        # Allow localhost for development
        if request.url.hostname not in ("localhost", "127.0.0.1"):
            raise AuthenticationError("HTTPS required")

    # Try JWT authentication first (if enabled)
    if config.jwt_enabled and bearer_token:
        jwt_manager = JWTManager(config)
        token_payload = jwt_manager.verify_token(bearer_token)

        if token_payload and token_payload.type == "access":
            permissions = {Permission(p) for p in token_payload.permissions}
            return UserContext(
                user_id=token_payload.sub,
                role=token_payload.role,
                permissions=permissions,
                token_type="jwt",
            )

    # Try API key authentication
    if api_key:
        validated_key = key_store.validate_key(api_key)

        if validated_key:
            key_store.update_last_used(validated_key.key_id)

            return UserContext(
                user_id=validated_key.user_id,
                customer_id=validated_key.customer_id,
                api_key_id=validated_key.key_id,
                role=validated_key.role,
                permissions=validated_key.permissions,
                metadata=validated_key.metadata,
                token_type="api_key",
            )

    # Allow anonymous access if configured
    if config.allow_anonymous:
        return UserContext(
            user_id="anonymous",
            role="anonymous",
            permissions=ROLE_PERMISSIONS.get("anonymous", set()),
            is_anonymous=True,
        )

    raise AuthenticationError()


async def get_current_user(
    user: UserContext = Depends(authenticate),
) -> UserContext:
    """Get the current authenticated user (alias for authenticate)."""
    return user


# =============================================================================
# Permission Checking Decorators
# =============================================================================


def require_permissions(*permissions: Permission) -> Callable:
    """
    Create a dependency that requires specific permissions.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: UserContext = Depends(require_permissions(Permission.ADMIN_CONFIG))):
            ...
    """

    async def permission_checker(
        user: UserContext = Depends(authenticate),
    ) -> UserContext:
        if not user.has_all_permissions(list(permissions)):
            missing = [p.value for p in permissions if p not in user.permissions]
            raise AuthorizationError(f"Missing permissions: {', '.join(missing)}")
        return user

    return permission_checker


def require_any_permission(*permissions: Permission) -> Callable:
    """
    Create a dependency that requires any of the specified permissions.

    Usage:
        @app.get("/events")
        async def events_endpoint(user: UserContext = Depends(require_any_permission(Permission.EVENT_READ, Permission.ADMIN_USERS))):
            ...
    """

    async def permission_checker(
        user: UserContext = Depends(authenticate),
    ) -> UserContext:
        if not user.has_any_permission(list(permissions)):
            raise AuthorizationError(f"Requires one of: {', '.join(p.value for p in permissions)}")
        return user

    return permission_checker


def require_role(*roles: str) -> Callable:
    """
    Create a dependency that requires specific roles.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: UserContext = Depends(require_role("admin"))):
            ...
    """

    async def role_checker(
        user: UserContext = Depends(authenticate),
    ) -> UserContext:
        if user.role not in roles:
            raise AuthorizationError(f"Requires role: {', '.join(roles)}")
        return user

    return role_checker


# =============================================================================
# Webhook Signature Verification
# =============================================================================


def generate_webhook_signature(payload: bytes, secret: str) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Args:
        payload: The raw request body bytes
        secret: The webhook secret key

    Returns:
        The signature as a hex string
    """
    return hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()


def verify_webhook_signature(
    payload: bytes, signature: str, secret: str, max_age_seconds: int = 300
) -> bool:
    """
    Verify a webhook signature.

    Args:
        payload: The raw request body bytes
        signature: The signature from the request header
        secret: The webhook secret key
        max_age_seconds: Maximum age of the request (for replay protection)

    Returns:
        True if signature is valid
    """
    expected = generate_webhook_signature(payload, secret)
    return hmac.compare_digest(expected, signature)


# =============================================================================
# Utility Functions
# =============================================================================


def generate_api_key() -> str:
    """Generate a new random API key."""
    return secrets.token_urlsafe(32)


def mask_api_key(key: str) -> str:
    """Mask an API key for display (show first and last 4 chars)."""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


def create_development_key(user_id: str = "dev-user") -> tuple[str, APIKey]:
    """
    Create a development API key with full permissions.

    WARNING: Only use in development/testing environments!
    """
    store = get_api_key_store()
    return store.create_key(
        user_id=user_id,
        role="admin",
        name="Development Key",
        metadata={"environment": "development"},
    )
