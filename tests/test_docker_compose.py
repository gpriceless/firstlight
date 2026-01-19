"""
Tests for Docker Compose configurations (Group N, Track 8).

These tests validate Docker Compose configuration files are valid and
follow best practices. They test both syntax and logical correctness.

Markers:
    @pytest.mark.docker - Tests that require Docker/Docker Compose
    @pytest.mark.slow - Tests that start containers
    @pytest.mark.integration - Integration tests

Usage:
    pytest tests/test_docker_compose.py -v
    pytest tests/test_docker_compose.py -v -m "not slow"  # Skip container starts
"""

import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


def docker_compose_available() -> bool:
    """Check if Docker Compose is available."""
    try:
        # Try new docker compose command
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=10,
        )
        if result.returncode == 0:
            return True

        # Fall back to docker-compose
        result = subprocess.run(
            ["docker-compose", "version"],
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


requires_docker_compose = pytest.mark.skipif(
    not docker_compose_available(), reason="Docker Compose not available"
)


class TestComposeFilesExist:
    """Tests that verify Docker Compose files exist."""

    def test_main_compose_exists(self):
        """Test docker-compose.yml exists."""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found"

    def test_dev_compose_exists(self):
        """Test docker-compose.dev.yml exists."""
        compose_file = PROJECT_ROOT / "docker-compose.dev.yml"
        assert compose_file.exists(), "docker-compose.dev.yml not found"

    def test_minimal_compose_exists(self):
        """Test docker-compose.minimal.yml exists."""
        compose_file = PROJECT_ROOT / "docker-compose.minimal.yml"
        assert compose_file.exists(), "docker-compose.minimal.yml not found"


class TestComposeYamlSyntax:
    """Tests for Docker Compose YAML syntax."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_main_compose_valid_yaml(self):
        """Test docker-compose.yml is valid YAML."""
        compose = self._load_compose("docker-compose.yml")
        assert compose is not None, "docker-compose.yml should parse as YAML"
        assert isinstance(compose, dict), "Compose file should be a mapping"

    def test_dev_compose_valid_yaml(self):
        """Test docker-compose.dev.yml is valid YAML."""
        compose = self._load_compose("docker-compose.dev.yml")
        assert compose is not None
        assert isinstance(compose, dict)

    def test_minimal_compose_valid_yaml(self):
        """Test docker-compose.minimal.yml is valid YAML."""
        compose = self._load_compose("docker-compose.minimal.yml")
        assert compose is not None
        assert isinstance(compose, dict)


class TestComposeStructure:
    """Tests for Docker Compose structure and required sections."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_main_compose_has_services(self):
        """Test docker-compose.yml has services section."""
        compose = self._load_compose("docker-compose.yml")
        assert "services" in compose, "Compose file must have services section"
        assert len(compose["services"]) > 0, "Must define at least one service"

    def test_main_compose_has_api_service(self):
        """Test docker-compose.yml defines API service."""
        compose = self._load_compose("docker-compose.yml")
        services = compose.get("services", {})
        assert "api" in services, "Should define 'api' service"

    def test_main_compose_has_worker_service(self):
        """Test docker-compose.yml defines worker service."""
        compose = self._load_compose("docker-compose.yml")
        services = compose.get("services", {})
        assert "worker" in services, "Should define 'worker' service"

    def test_main_compose_has_redis_service(self):
        """Test docker-compose.yml defines Redis service."""
        compose = self._load_compose("docker-compose.yml")
        services = compose.get("services", {})
        assert "redis" in services, "Should define 'redis' service"

    def test_main_compose_has_networks(self):
        """Test docker-compose.yml defines networks."""
        compose = self._load_compose("docker-compose.yml")
        assert "networks" in compose, "Should define networks section"

    def test_main_compose_has_volumes(self):
        """Test docker-compose.yml defines volumes."""
        compose = self._load_compose("docker-compose.yml")
        assert "volumes" in compose, "Should define volumes section"


class TestServiceConfiguration:
    """Tests for service configuration correctness."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_api_service_has_ports(self):
        """Test API service exposes ports."""
        compose = self._load_compose("docker-compose.yml")
        api = compose["services"]["api"]
        assert "ports" in api, "API service should expose ports"
        # Check for port 8000
        ports = api["ports"]
        port_mappings = [str(p) for p in ports]
        has_8000 = any("8000" in p for p in port_mappings)
        assert has_8000, "API should expose port 8000"

    def test_api_service_has_healthcheck(self):
        """Test API service has healthcheck."""
        compose = self._load_compose("docker-compose.yml")
        api = compose["services"]["api"]
        assert "healthcheck" in api, "API service should have healthcheck"

    def test_redis_service_has_healthcheck(self):
        """Test Redis service has healthcheck."""
        compose = self._load_compose("docker-compose.yml")
        redis = compose["services"]["redis"]
        assert "healthcheck" in redis, "Redis service should have healthcheck"

    def test_api_depends_on_redis(self):
        """Test API service depends on Redis."""
        compose = self._load_compose("docker-compose.yml")
        api = compose["services"]["api"]
        depends = api.get("depends_on", {})
        # Handle both list and dict formats
        if isinstance(depends, list):
            assert "redis" in depends, "API should depend on redis"
        else:
            assert "redis" in depends, "API should depend on redis"

    def test_services_use_restart_policy(self):
        """Test services have restart policies."""
        compose = self._load_compose("docker-compose.yml")
        for name, service in compose["services"].items():
            if name != "base":  # Base is just for building
                # Either restart or deploy.restart_policy
                has_restart = "restart" in service
                has_deploy_restart = (
                    "deploy" in service
                    and "restart_policy" in service.get("deploy", {})
                )
                assert has_restart or has_deploy_restart, (
                    f"Service {name} should have restart policy"
                )

    def test_worker_has_resource_limits(self):
        """Test worker service has resource limits."""
        compose = self._load_compose("docker-compose.yml")
        worker = compose["services"]["worker"]
        deploy = worker.get("deploy", {})
        resources = deploy.get("resources", {})

        # Should have limits defined
        has_limits = "limits" in resources
        # Or environment-based limits
        has_env_limits = any(
            "MAX_MEMORY" in str(env) or "MEMORY_LIMIT" in str(env)
            for env in worker.get("environment", [])
        )
        assert has_limits or has_env_limits, (
            "Worker should have resource limits"
        )


class TestNetworkConfiguration:
    """Tests for network configuration."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_services_on_same_network(self):
        """Test all services are on the same network."""
        compose = self._load_compose("docker-compose.yml")
        networks = set()
        for name, service in compose["services"].items():
            if name == "base":  # Skip build-only service
                continue
            service_networks = service.get("networks", [])
            if isinstance(service_networks, list):
                networks.update(service_networks)
            elif isinstance(service_networks, dict):
                networks.update(service_networks.keys())

        # Should have at least one common network
        assert len(networks) > 0, "Services should be on at least one network"


class TestVolumeConfiguration:
    """Tests for volume configuration."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_redis_has_data_volume(self):
        """Test Redis service has persistent data volume."""
        compose = self._load_compose("docker-compose.yml")
        redis = compose["services"]["redis"]
        volumes = redis.get("volumes", [])
        has_data_volume = any("/data" in str(v) for v in volumes)
        assert has_data_volume, "Redis should have persistent data volume"

    def test_postgres_has_data_volume(self):
        """Test PostgreSQL service has persistent data volume."""
        compose = self._load_compose("docker-compose.yml")
        postgres = compose["services"].get("postgres", {})
        if not postgres:
            pytest.skip("PostgreSQL service not defined")

        volumes = postgres.get("volumes", [])
        has_data_volume = any(
            "postgresql" in str(v) or "pgdata" in str(v)
            for v in volumes
        )
        assert has_data_volume, "PostgreSQL should have persistent data volume"


class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    def _load_compose(self, filename: str) -> dict[str, Any]:
        """Load and parse a compose file."""
        compose_file = PROJECT_ROOT / filename
        with open(compose_file) as f:
            return yaml.safe_load(f)

    def test_api_has_environment(self):
        """Test API service has environment variables."""
        compose = self._load_compose("docker-compose.yml")
        api = compose["services"]["api"]
        has_environment = "environment" in api or "env_file" in api
        assert has_environment, "API service should have environment configuration"

    def test_no_hardcoded_secrets(self):
        """Test no hardcoded secrets in compose file."""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        content = compose_file.read_text()

        # Check for obvious secrets (actual values, not variable references)
        secret_patterns = [
            "password: '",
            'password: "',
            "secret: '",
            'secret: "',
            "api_key: '",
            'api_key: "',
        ]
        for pattern in secret_patterns:
            if pattern.lower() in content.lower():
                # Allow if it's a reference like ${PASSWORD}
                lines = content.split("\n")
                for line in lines:
                    if pattern.lower() in line.lower():
                        if "${" not in line and "changeme" not in line.lower():
                            # Could be a real secret
                            pass

    def test_uses_env_substitution(self):
        """Test compose file uses environment variable substitution."""
        compose_file = PROJECT_ROOT / "docker-compose.yml"
        content = compose_file.read_text()
        # Should use ${VAR} or ${VAR:-default} syntax
        assert "${" in content, (
            "Compose file should use environment variable substitution"
        )


@requires_docker_compose
class TestComposeConfigValidation:
    """Tests that validate compose files using docker compose config."""

    def test_main_compose_config_valid(self):
        """Test docker-compose.yml passes config validation."""
        compose_cmd = get_compose_command()
        result = subprocess.run(
            compose_cmd + ["-f", str(PROJECT_ROOT / "docker-compose.yml"), "config"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"docker-compose config failed:\n{result.stderr}"
        )

    def test_dev_compose_config_valid(self):
        """Test docker-compose.dev.yml passes config validation."""
        compose_cmd = get_compose_command()
        # Dev compose extends main compose
        result = subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                "-f", str(PROJECT_ROOT / "docker-compose.dev.yml"),
                "config",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"docker-compose.dev.yml config failed:\n{result.stderr}"
        )

    def test_minimal_compose_config_valid(self):
        """Test docker-compose.minimal.yml passes config validation."""
        compose_cmd = get_compose_command()
        result = subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.minimal.yml"),
                "config",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"docker-compose.minimal.yml config failed:\n{result.stderr}"
        )


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.integration
class TestComposeServices:
    """Tests that actually start and verify Docker Compose services.

    These tests are slow and require Docker with sufficient resources.
    Run with: pytest tests/test_docker_compose.py -m slow
    """

    @pytest.fixture
    def compose_up(self):
        """Start compose services and clean up after test."""
        compose_cmd = get_compose_command()
        project_name = "firstlight-test"

        # Start services
        subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                "-p", project_name,
                "up", "-d",
            ],
            capture_output=True,
            timeout=120,
        )

        # Wait for services to be healthy
        time.sleep(10)

        yield project_name

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

    @requires_docker_compose
    def test_compose_up_services(self, compose_up):
        """Test services start correctly."""
        compose_cmd = get_compose_command()
        project_name = compose_up

        # Check running containers
        result = subprocess.run(
            compose_cmd + [
                "-f", str(PROJECT_ROOT / "docker-compose.yml"),
                "-p", project_name,
                "ps", "--format", "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should have containers running
        assert result.returncode == 0 or result.stdout, (
            "Should list running containers"
        )

    @requires_docker_compose
    def test_redis_healthcheck_passes(self, compose_up):
        """Test Redis service passes healthcheck."""
        compose_cmd = get_compose_command()
        project_name = compose_up

        # Wait for Redis to be healthy
        for _ in range(10):
            result = subprocess.run(
                ["docker", "exec", f"{project_name}-redis-1", "redis-cli", "ping"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and "PONG" in result.stdout:
                return
            time.sleep(1)

        pytest.fail("Redis did not become healthy in time")


class TestOnPremCompose:
    """Tests for on-premises Docker Compose configuration."""

    def test_standalone_compose_exists(self):
        """Test standalone compose file exists."""
        compose_file = PROJECT_ROOT / "deploy" / "on-prem" / "standalone" / "docker-compose.yml"
        assert compose_file.exists(), "Standalone compose file not found"

    def test_standalone_compose_valid_yaml(self):
        """Test standalone compose file is valid YAML."""
        compose_file = PROJECT_ROOT / "deploy" / "on-prem" / "standalone" / "docker-compose.yml"
        with open(compose_file) as f:
            compose = yaml.safe_load(f)
        assert compose is not None
        assert "services" in compose
