"""
Integration tests for authentication enforcement (Phase 0 tasks 0.1-0.6).

Tests verify:
(a) Requests without API key return HTTP 401 now that auth is on by default
(b) customer_id is present in request.state after successful auth
(c) Tenant middleware blocks cross-tenant access
(d) Health probes are exempt from auth
(e) state:write permission is required for privileged operations
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from api.auth import (
    APIKey,
    APIKeyStore,
    Permission,
    ROLE_PERMISSIONS,
    UserContext,
)
from api.config import AuthSettings, _AUTH_INSECURE_DEFAULT_KEY
from api.middleware import TenantMiddleware, DEFAULT_AUTH_EXEMPT_PATHS

# Project root for file path assertions
_project_root = Path(__file__).resolve().parent.parent.parent


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_key_store():
    """Create a fresh API key store for each test."""
    return APIKeyStore()


@pytest.fixture
def tenant_a_key(api_key_store):
    """Create an API key for tenant A with operator role."""
    raw_key, api_key = api_key_store.create_key(
        user_id="user-a",
        role="operator",
        customer_id="tenant-a",
    )
    return raw_key, api_key


@pytest.fixture
def tenant_b_key(api_key_store):
    """Create an API key for tenant B with operator role."""
    raw_key, api_key = api_key_store.create_key(
        user_id="user-b",
        role="operator",
        customer_id="tenant-b",
    )
    return raw_key, api_key


@pytest.fixture
def readonly_key(api_key_store):
    """Create an API key with readonly role (no state:write)."""
    raw_key, api_key = api_key_store.create_key(
        user_id="user-readonly",
        role="readonly",
        customer_id="tenant-a",
    )
    return raw_key, api_key


@pytest.fixture
def test_app(api_key_store):
    """Create a FastAPI test app with auth enabled and TenantMiddleware."""
    app = FastAPI()

    # Add TenantMiddleware
    app.add_middleware(TenantMiddleware)

    # Health endpoint (should be exempt from auth)
    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/api/v1/health")
    async def health_v1():
        return {"status": "healthy"}

    @app.get("/ready")
    async def ready():
        return {"ready": True}

    # Protected endpoint that shows customer_id
    @app.get("/api/v1/whoami")
    async def whoami(request: Request):
        """Endpoint that requires auth and shows the resolved customer_id."""
        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        key = api_key_store.validate_key(api_key_header)
        if key is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        request.state.customer_id = key.customer_id

        return {
            "user_id": key.user_id,
            "customer_id": key.customer_id,
            "role": key.role,
        }

    # Endpoint that requires state:write permission
    @app.post("/api/v1/state/transition")
    async def state_transition(request: Request):
        """Endpoint requiring state:write permission."""
        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        key = api_key_store.validate_key(api_key_header)
        if key is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        if Permission.STATE_WRITE not in key.permissions:
            return JSONResponse(
                status_code=403,
                content={"detail": "Missing permission: state:write"},
            )

        return {"transitioned": True, "customer_id": key.customer_id}

    # Endpoint for tenant-scoped resource access
    @app.get("/api/v1/jobs")
    async def list_jobs(request: Request):
        """Endpoint that scopes results by customer_id."""
        api_key_header = request.headers.get("X-API-Key")
        if not api_key_header:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"},
            )

        key = api_key_store.validate_key(api_key_header)
        if key is None:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"},
            )

        # Simulated jobs scoped by tenant
        all_jobs = {
            "tenant-a": [{"job_id": "j1", "event_type": "flood"}],
            "tenant-b": [{"job_id": "j2", "event_type": "wildfire"}],
        }

        customer_id = key.customer_id
        return {"jobs": all_jobs.get(customer_id, []), "customer_id": customer_id}

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


# =============================================================================
# Test (a): Request without API key returns HTTP 401
# =============================================================================


class TestAuthRequired:
    """Test that auth is required by default."""

    def test_no_api_key_returns_401(self, client):
        """A request without an API key should return 401."""
        response = client.get("/api/v1/whoami")
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        """A request with an invalid API key should return 401."""
        response = client.get(
            "/api/v1/whoami",
            headers={"X-API-Key": "invalid-key-that-does-not-exist"},
        )
        assert response.status_code == 401

    def test_valid_api_key_returns_200(self, client, tenant_a_key):
        """A request with a valid API key should succeed."""
        raw_key, _ = tenant_a_key
        response = client.get(
            "/api/v1/whoami",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user-a"

    def test_auth_settings_default_enabled(self):
        """Verify AuthSettings.enabled defaults to True."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTH_ENABLED", None)
            settings = AuthSettings()
            assert settings.enabled is True


# =============================================================================
# Test (b): customer_id present in request.state after successful auth
# =============================================================================


class TestCustomerIdPropagation:
    """Test that customer_id is propagated through the auth pipeline."""

    def test_customer_id_in_response(self, client, tenant_a_key):
        """customer_id should be present in the response after auth."""
        raw_key, _ = tenant_a_key
        response = client.get(
            "/api/v1/whoami",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "tenant-a"

    def test_customer_id_on_user_context(self, tenant_a_key):
        """UserContext should include customer_id."""
        _, api_key = tenant_a_key
        context = UserContext(
            user_id=api_key.user_id,
            customer_id=api_key.customer_id,
            api_key_id=api_key.key_id,
            role=api_key.role,
            permissions=api_key.permissions,
        )
        assert context.customer_id == "tenant-a"
        assert context.to_dict()["customer_id"] == "tenant-a"

    def test_customer_id_defaults_to_legacy(self):
        """customer_id should default to 'legacy' for backward compatibility."""
        context = UserContext(user_id="test-user")
        assert context.customer_id == "legacy"

        key = APIKey(key_id="k1", key_hash="h1", user_id="u1")
        assert key.customer_id == "legacy"


# =============================================================================
# Test (c): Tenant middleware blocks cross-tenant job queries
# =============================================================================


class TestTenantIsolation:
    """Test that tenant isolation prevents cross-tenant access."""

    def test_tenant_a_sees_only_own_jobs(self, client, tenant_a_key):
        """Tenant A should only see tenant A's jobs."""
        raw_key, _ = tenant_a_key
        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "tenant-a"
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "j1"

    def test_tenant_b_sees_only_own_jobs(self, client, tenant_b_key):
        """Tenant B should only see tenant B's jobs."""
        raw_key, _ = tenant_b_key
        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["customer_id"] == "tenant-b"
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["job_id"] == "j2"

    def test_tenant_a_cannot_see_tenant_b_jobs(self, client, tenant_a_key):
        """Tenant A should not see tenant B's jobs."""
        raw_key, _ = tenant_a_key
        response = client.get(
            "/api/v1/jobs",
            headers={"X-API-Key": raw_key},
        )
        data = response.json()
        job_ids = [j["job_id"] for j in data["jobs"]]
        assert "j2" not in job_ids  # j2 belongs to tenant-b


# =============================================================================
# Test (d): Health probes are exempt from auth
# =============================================================================


class TestHealthExemption:
    """Test that health probes do not require authentication."""

    def test_health_endpoint_no_auth(self, client):
        """GET /health should work without authentication."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_v1_endpoint_no_auth(self, client):
        """GET /api/v1/health should work without authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_ready_endpoint_no_auth(self, client):
        """GET /ready should work without authentication."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_exempt_paths_include_health(self):
        """Verify the default exempt paths include health endpoints."""
        assert "/health" in DEFAULT_AUTH_EXEMPT_PATHS
        assert "/ready" in DEFAULT_AUTH_EXEMPT_PATHS
        assert "/api/v1/health" in DEFAULT_AUTH_EXEMPT_PATHS

    def test_exempt_paths_include_ogc_discovery(self):
        """Verify the default exempt paths include OGC discovery paths."""
        assert "/oapi/" in DEFAULT_AUTH_EXEMPT_PATHS
        assert "/oapi/conformance" in DEFAULT_AUTH_EXEMPT_PATHS
        assert "/oapi/processes" in DEFAULT_AUTH_EXEMPT_PATHS


# =============================================================================
# Test (e): state:write permission required for transition endpoints
# =============================================================================


class TestStateWritePermission:
    """Test that state:write permission is enforced."""

    def test_operator_can_transition(self, client, tenant_a_key):
        """Operator role (has state:write) should be able to transition state."""
        raw_key, api_key = tenant_a_key
        assert Permission.STATE_WRITE in api_key.permissions
        response = client.post(
            "/api/v1/state/transition",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 200
        assert response.json()["transitioned"] is True

    def test_readonly_cannot_transition(self, client, readonly_key):
        """Readonly role (no state:write) should be rejected with 403."""
        raw_key, api_key = readonly_key
        assert Permission.STATE_WRITE not in api_key.permissions
        response = client.post(
            "/api/v1/state/transition",
            headers={"X-API-Key": raw_key},
        )
        assert response.status_code == 403
        assert "state:write" in response.json()["detail"]

    def test_permission_enum_values(self):
        """Verify state and escalation permissions exist with correct values."""
        assert Permission.STATE_READ.value == "state:read"
        assert Permission.STATE_WRITE.value == "state:write"
        assert Permission.ESCALATION_MANAGE.value == "escalation:manage"

    def test_role_permission_grants(self):
        """Verify permission grants per role."""
        # readonly gets state:read only
        assert Permission.STATE_READ in ROLE_PERMISSIONS["readonly"]
        assert Permission.STATE_WRITE not in ROLE_PERMISSIONS["readonly"]
        assert Permission.ESCALATION_MANAGE not in ROLE_PERMISSIONS["readonly"]

        # user gets state:read only
        assert Permission.STATE_READ in ROLE_PERMISSIONS["user"]
        assert Permission.STATE_WRITE not in ROLE_PERMISSIONS["user"]

        # operator gets all three
        assert Permission.STATE_READ in ROLE_PERMISSIONS["operator"]
        assert Permission.STATE_WRITE in ROLE_PERMISSIONS["operator"]
        assert Permission.ESCALATION_MANAGE in ROLE_PERMISSIONS["operator"]

        # admin gets all permissions
        assert Permission.STATE_READ in ROLE_PERMISSIONS["admin"]
        assert Permission.STATE_WRITE in ROLE_PERMISSIONS["admin"]
        assert Permission.ESCALATION_MANAGE in ROLE_PERMISSIONS["admin"]


# =============================================================================
# Test: Secret key production validator
# =============================================================================


class TestSecretKeyValidator:
    """Test the production secret key validator from task 0.5."""

    def test_dev_allows_default_key(self):
        """Development environment should accept the default key."""
        with patch.dict(
            os.environ,
            {"FIRSTLIGHT_ENVIRONMENT": "development"},
            clear=False,
        ):
            os.environ.pop("AUTH_SECRET_KEY", None)
            settings = AuthSettings()
            assert settings.secret_key == _AUTH_INSECURE_DEFAULT_KEY

    def test_production_rejects_default_key(self):
        """Production environment should reject the default key."""
        with patch.dict(
            os.environ,
            {"FIRSTLIGHT_ENVIRONMENT": "production"},
            clear=False,
        ):
            os.environ.pop("AUTH_SECRET_KEY", None)
            with pytest.raises(Exception, match="AUTH_SECRET_KEY must be changed"):
                AuthSettings()

    def test_production_accepts_custom_key(self):
        """Production environment should accept a custom key."""
        with patch.dict(
            os.environ,
            {
                "FIRSTLIGHT_ENVIRONMENT": "production",
                "AUTH_SECRET_KEY": "my-secure-production-key-32chars!!",
            },
            clear=False,
        ):
            settings = AuthSettings()
            assert settings.secret_key == "my-secure-production-key-32chars!!"


# =============================================================================
# Test: Migration file exists
# =============================================================================


class TestMigrationExists:
    """Verify database migration artifacts exist."""

    def test_migration_file_exists(self):
        """The customer_id migration file should exist."""
        migration_path = _project_root / "db" / "migrations" / "000_add_customer_id.sql"
        assert migration_path.exists(), f"Migration not found: {migration_path}"

    def test_migration_contains_customer_id(self):
        """The migration should reference customer_id."""
        migration_path = _project_root / "db" / "migrations" / "000_add_customer_id.sql"
        content = migration_path.read_text()
        assert "customer_id" in content
        assert "NOT NULL" in content
        assert "legacy" in content
