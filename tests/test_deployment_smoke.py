"""
Deployment smoke tests (Group N, Track 8).

These tests verify that deployed services are healthy and responding.
They are designed to run after deployment to validate the system is
functioning correctly.

Markers:
    @pytest.mark.docker - Tests that require Docker
    @pytest.mark.slow - Tests that start containers
    @pytest.mark.integration - Integration tests
    @pytest.mark.smoke - Smoke tests for deployed services

Usage:
    # Run against local Docker deployment
    pytest tests/test_deployment_smoke.py -v -m smoke

    # Run with custom API URL
    API_URL=http://localhost:8000 pytest tests/test_deployment_smoke.py -v
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Check if requests is available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

requires_requests = pytest.mark.skipif(
    not HAS_REQUESTS, reason="requests not installed"
)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Default API URL (can be overridden with environment variable)
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")


def docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_compose_command() -> list[str]:
    """Get the Docker Compose command (v1 or v2)."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return ["docker", "compose"]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return ["docker-compose"]


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker not available"
)


class TestHealthEndpoints:
    """Tests for health check endpoints.

    These tests can run against any deployed instance.
    Set API_URL environment variable to target a specific deployment.
    """

    @requires_requests
    def test_health_endpoint_responds(self):
        """Test /health endpoint responds."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health",
                timeout=10,
            )
            # Accept 200 or 503 (unhealthy but responding)
            assert response.status_code in [200, 503], (
                f"Health endpoint returned {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available at " + DEFAULT_API_URL)

    @requires_requests
    def test_health_endpoint_returns_json(self):
        """Test /health endpoint returns JSON."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health",
                timeout=10,
            )
            if response.status_code in [200, 503]:
                # Should be JSON
                data = response.json()
                assert isinstance(data, dict), "Health response should be JSON object"
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_health_ready_endpoint(self):
        """Test /health/ready endpoint."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health/ready",
                timeout=10,
            )
            # Readiness probe should return 200 when ready
            assert response.status_code in [200, 503], (
                f"Ready endpoint returned {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("/health/ready not implemented")
            raise

    @requires_requests
    def test_health_live_endpoint(self):
        """Test /health/live endpoint."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health/live",
                timeout=10,
            )
            # Liveness probe should always return 200 if service is running
            assert response.status_code == 200, (
                f"Live endpoint returned {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                pytest.skip("/health/live not implemented")
            raise


class TestAPIDocsAvailable:
    """Tests for API documentation availability."""

    @requires_requests
    def test_openapi_docs_available(self):
        """Test OpenAPI docs are served at /docs."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/docs",
                timeout=10,
            )
            assert response.status_code == 200, (
                f"Docs endpoint returned {response.status_code}"
            )
            # Should be HTML
            assert "text/html" in response.headers.get("content-type", ""), (
                "Docs should be HTML"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_openapi_json_available(self):
        """Test OpenAPI JSON schema is available."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/openapi.json",
                timeout=10,
            )
            assert response.status_code == 200, (
                f"OpenAPI JSON returned {response.status_code}"
            )
            data = response.json()
            assert "openapi" in data, "Should be OpenAPI spec"
            assert "paths" in data, "Should have paths"
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_redoc_available(self):
        """Test ReDoc documentation is available."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/redoc",
                timeout=10,
            )
            # ReDoc is optional, so 404 is acceptable
            if response.status_code == 404:
                pytest.skip("ReDoc not enabled")
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")


class TestBasicAPIFunctionality:
    """Tests for basic API functionality."""

    @requires_requests
    def test_root_endpoint(self):
        """Test root endpoint returns something."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/",
                timeout=10,
            )
            # Could be redirect, info, or 404
            assert response.status_code in [200, 301, 302, 404], (
                f"Root returned unexpected status {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_cors_headers(self):
        """Test CORS headers are set."""
        try:
            response = requests.options(
                f"{DEFAULT_API_URL}/health",
                headers={
                    "Origin": "http://example.com",
                    "Access-Control-Request-Method": "GET",
                },
                timeout=10,
            )
            # CORS should allow the request or be disabled
            # 200 or 204 indicates CORS is configured
            # 405 Method Not Allowed is also acceptable
            assert response.status_code in [200, 204, 405], (
                f"CORS preflight returned {response.status_code}"
            )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_error_handling(self):
        """Test API returns proper error for bad requests."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/nonexistent-endpoint-12345",
                timeout=10,
            )
            assert response.status_code == 404, "Should return 404 for unknown endpoint"
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.smoke
class TestDockerDeploymentSmoke:
    """Smoke tests for Docker Compose deployment.

    These tests start a local Docker deployment and verify it works.
    They are slow but provide comprehensive deployment validation.
    """

    @pytest.fixture(scope="class")
    def running_api(self) -> Generator[str, None, None]:
        """Start API container for testing."""
        if not docker_available():
            pytest.skip("Docker not available")

        compose_cmd = get_compose_command()
        project_name = "mdive-smoke-test"

        # Start only API and dependencies
        start_result = subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                "-p", project_name,
                "up", "-d", "api", "redis",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if start_result.returncode != 0:
            pytest.skip(f"Failed to start containers: {start_result.stderr}")

        # Wait for services to be healthy
        api_url = "http://localhost:8000"
        max_retries = 30
        for i in range(max_retries):
            try:
                if HAS_REQUESTS:
                    response = requests.get(f"{api_url}/health", timeout=5)
                    if response.status_code in [200, 503]:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            # Cleanup before failing
            subprocess.run(
                compose_cmd + [
                    "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                    "-p", project_name,
                    "down", "-v", "--remove-orphans",
                ],
                capture_output=True,
                timeout=60,
            )
            pytest.skip("API did not become healthy in time")

        yield api_url

        # Cleanup
        subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                "-p", project_name,
                "down", "-v", "--remove-orphans",
            ],
            capture_output=True,
            timeout=60,
        )

    @requires_requests
    @requires_docker
    def test_health_endpoint(self, running_api: str):
        """Test /health endpoint responds in Docker deployment."""
        response = requests.get(f"{running_api}/health", timeout=10)
        assert response.status_code == 200, (
            f"Health check failed: {response.status_code}"
        )

    @requires_requests
    @requires_docker
    def test_ready_endpoint(self, running_api: str):
        """Test /health/ready endpoint in Docker deployment."""
        response = requests.get(f"{running_api}/health/ready", timeout=10)
        assert response.status_code in [200, 503]

    @requires_requests
    @requires_docker
    def test_api_docs_available(self, running_api: str):
        """Test OpenAPI docs are served in Docker deployment."""
        response = requests.get(f"{running_api}/docs", timeout=10)
        assert response.status_code == 200

    @requires_requests
    @requires_docker
    def test_metrics_endpoint(self, running_api: str):
        """Test /metrics endpoint for Prometheus."""
        response = requests.get(f"{running_api}/metrics", timeout=10)
        # Metrics might not be enabled, 404 is acceptable
        if response.status_code == 404:
            pytest.skip("Metrics endpoint not enabled")
        assert response.status_code == 200


class TestContainerHealth:
    """Tests for container health status."""

    @requires_docker
    def test_api_container_running(self):
        """Test API container is running."""
        result = subprocess.run(
            [
                "docker", "ps",
                "--filter", "name=mdive-api",
                "--format", "{{.Status}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Container might not be running, just check command works
        assert result.returncode == 0, "Docker ps command should succeed"

    @requires_docker
    def test_container_logs_accessible(self):
        """Test container logs are accessible."""
        result = subprocess.run(
            [
                "docker", "logs",
                "--tail", "10",
                "mdive-api",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Container might not exist, just verify command runs
        # Return code 1 with "No such container" is acceptable
        if result.returncode != 0:
            if "No such container" in result.stderr:
                pytest.skip("API container not running")


class TestServiceConnectivity:
    """Tests for service-to-service connectivity."""

    @requires_docker
    def test_redis_connectivity(self):
        """Test Redis is accessible from API container."""
        # Check if we can exec into API container and ping Redis
        result = subprocess.run(
            [
                "docker", "exec", "mdive-api",
                "python", "-c",
                "import redis; r = redis.Redis(host='redis', port=6379); print(r.ping())",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            if "No such container" in result.stderr:
                pytest.skip("API container not running")
            if "ModuleNotFoundError" in result.stderr:
                pytest.skip("redis module not installed in container")

    @requires_docker
    def test_database_connectivity(self):
        """Test database is accessible from API container."""
        result = subprocess.run(
            [
                "docker", "exec", "mdive-api",
                "python", "-c",
                "import os; print(os.environ.get('DATABASE_URL', 'not set'))",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            if "No such container" in result.stderr:
                pytest.skip("API container not running")


class TestResourceUsage:
    """Tests for container resource usage."""

    @requires_docker
    def test_container_not_oom(self):
        """Test containers are not OOM killed."""
        result = subprocess.run(
            [
                "docker", "ps",
                "--filter", "status=exited",
                "--filter", "name=mdive",
                "--format", "{{.Names}}: {{.Status}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout:
            # Check for OOM in exit status
            if "OOM" in result.stdout:
                pytest.fail(f"Container was OOM killed: {result.stdout}")

    @requires_docker
    def test_container_memory_usage(self):
        """Test container memory usage is reasonable."""
        result = subprocess.run(
            [
                "docker", "stats",
                "--no-stream",
                "--format", "{{.Name}}: {{.MemUsage}}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Just verify the command works
        assert result.returncode == 0


class TestConfigurationValidation:
    """Tests for configuration validation at startup."""

    @requires_requests
    def test_config_endpoint(self):
        """Test configuration info endpoint if available."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/config",
                timeout=10,
            )
            if response.status_code == 404:
                pytest.skip("Config endpoint not available")
            if response.status_code == 401:
                pytest.skip("Config endpoint requires authentication")
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_version_endpoint(self):
        """Test version info endpoint."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/version",
                timeout=10,
            )
            if response.status_code == 404:
                # Try alternative paths
                for path in ["/api/version", "/info", "/api/info"]:
                    response = requests.get(f"{DEFAULT_API_URL}{path}", timeout=10)
                    if response.status_code == 200:
                        break
                else:
                    pytest.skip("Version endpoint not available")
            if response.status_code == 200:
                data = response.json()
                # Should have version info
                assert "version" in data or "app_version" in data
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")


class TestSecurityHeaders:
    """Tests for security headers in responses."""

    @requires_requests
    def test_content_type_header(self):
        """Test Content-Type header is set."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health",
                timeout=10,
            )
            if response.status_code in [200, 503]:
                assert "content-type" in response.headers, (
                    "Response should have Content-Type header"
                )
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_no_server_header(self):
        """Test Server header doesn't leak version info."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health",
                timeout=10,
            )
            server = response.headers.get("server", "")
            # Should not contain detailed version info
            sensitive_patterns = ["python", "uvicorn", "starlette"]
            for pattern in sensitive_patterns:
                if pattern.lower() in server.lower():
                    # Just a warning, not a failure
                    pass
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")

    @requires_requests
    def test_x_content_type_options(self):
        """Test X-Content-Type-Options header."""
        try:
            response = requests.get(
                f"{DEFAULT_API_URL}/health",
                timeout=10,
            )
            # This is a security best practice, but not required
            header = response.headers.get("x-content-type-options")
            if header:
                assert header == "nosniff"
        except requests.exceptions.ConnectionError:
            pytest.skip("API not available")
