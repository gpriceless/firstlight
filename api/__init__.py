"""
Multiverse Dive API Package.

This package provides the FastAPI-based REST API for the Multiverse Dive
geospatial event intelligence platform.

Modules:
    auth: API key authentication, JWT support, and permission checking
    rate_limit: Rate limiting middleware with Redis/memory backend
    cors: CORS configuration
    webhooks: Webhook management and event delivery
    notifications: Multi-channel notification dispatcher

Usage:
    from api.auth import authenticate, require_permissions, Permission
    from api.rate_limit import RateLimitMiddleware, rate_limit
    from api.cors import setup_cors
    from api.webhooks import WebhookManager, WebhookEventType
    from api.notifications import NotificationDispatcher
"""

# Version
__version__ = "0.1.0"

# Authentication and Authorization
from api.auth import (
    APIKey,
    APIKeyStore,
    AuthConfig,
    AuthenticationError,
    AuthorizationError,
    JWTManager,
    Permission,
    ROLE_PERMISSIONS,
    TokenPayload,
    UserContext,
    authenticate,
    create_development_key,
    generate_api_key,
    generate_webhook_signature,
    get_api_key_store,
    get_auth_config,
    get_current_user,
    mask_api_key,
    require_any_permission,
    require_permissions,
    require_role,
    verify_webhook_signature,
)

# Rate Limiting
from api.rate_limit import (
    EndpointRateLimit,
    MemoryRateLimitBackend,
    RateLimitBackend,
    RateLimitConfig,
    RateLimitMiddleware,
    RateLimitResult,
    RateLimitRule,
    RateLimiter,
    RedisRateLimitBackend,
    TimeWindow,
    check_rate_limit,
    configure_rate_limits,
    get_rate_limit_config,
    get_rate_limiter,
    rate_limit,
    reset_rate_limit,
)

# CORS
from api.cors import (
    CORSConfig,
    create_cors_middleware_config,
    get_cors_config,
    get_development_origins,
    get_production_origins,
    is_origin_allowed,
    setup_cors,
    validate_cors_config,
)

# Webhooks
from api.webhooks import (
    DeliveryAttempt,
    DeliveryTask,
    Webhook,
    WebhookConfig,
    WebhookEvent,
    WebhookEventType,
    WebhookManager,
    WebhookStore,
    EVENT_TYPE_GROUPS,
    generate_signature,
    get_webhook_config,
    get_webhook_manager,
    initialize_webhooks,
    shutdown_webhooks,
    verify_signature,
)

# Notifications
from api.notifications import (
    Notification,
    NotificationChannel,
    NotificationConfig,
    NotificationDispatcher,
    NotificationPriority,
    NotificationResult,
    WebSocketConnectionManager,
    get_notification_config,
    get_notification_dispatcher,
    initialize_notifications,
    shutdown_notifications,
)

__all__ = [
    # Version
    "__version__",
    # Auth
    "APIKey",
    "APIKeyStore",
    "AuthConfig",
    "AuthenticationError",
    "AuthorizationError",
    "JWTManager",
    "Permission",
    "ROLE_PERMISSIONS",
    "TokenPayload",
    "UserContext",
    "authenticate",
    "create_development_key",
    "generate_api_key",
    "generate_webhook_signature",
    "get_api_key_store",
    "get_auth_config",
    "get_current_user",
    "mask_api_key",
    "require_any_permission",
    "require_permissions",
    "require_role",
    "verify_webhook_signature",
    # Rate Limit
    "EndpointRateLimit",
    "MemoryRateLimitBackend",
    "RateLimitBackend",
    "RateLimitConfig",
    "RateLimitMiddleware",
    "RateLimitResult",
    "RateLimitRule",
    "RateLimiter",
    "RedisRateLimitBackend",
    "TimeWindow",
    "check_rate_limit",
    "configure_rate_limits",
    "get_rate_limit_config",
    "get_rate_limiter",
    "rate_limit",
    "reset_rate_limit",
    # CORS
    "CORSConfig",
    "create_cors_middleware_config",
    "get_cors_config",
    "get_development_origins",
    "get_production_origins",
    "is_origin_allowed",
    "setup_cors",
    "validate_cors_config",
    # Webhooks
    "DeliveryAttempt",
    "DeliveryTask",
    "Webhook",
    "WebhookConfig",
    "WebhookEvent",
    "WebhookEventType",
    "WebhookManager",
    "WebhookStore",
    "EVENT_TYPE_GROUPS",
    "generate_signature",
    "get_webhook_config",
    "get_webhook_manager",
    "initialize_webhooks",
    "shutdown_webhooks",
    "verify_signature",
    # Notifications
    "Notification",
    "NotificationChannel",
    "NotificationConfig",
    "NotificationDispatcher",
    "NotificationPriority",
    "NotificationResult",
    "WebSocketConnectionManager",
    "get_notification_config",
    "get_notification_dispatcher",
    "initialize_notifications",
    "shutdown_notifications",
]
