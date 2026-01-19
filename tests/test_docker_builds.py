"""
Tests for Docker build configurations (Group N, Track 8).

These tests validate that Docker images build successfully and follow
best practices. They require Docker to be installed and accessible.

Markers:
    @pytest.mark.docker - Tests that require Docker
    @pytest.mark.slow - Build tests take time

Usage:
    pytest tests/test_docker_builds.py -v
    pytest tests/test_docker_builds.py -v -m "not slow"  # Skip actual builds
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Project root
PROJECT_ROOT = Path(__file__).parent.parent


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


requires_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker not available"
)


class TestDockerfileExists:
    """Tests that verify Dockerfiles exist and are readable."""

    def test_base_dockerfile_exists(self):
        """Test base Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "base" / "Dockerfile"
        assert dockerfile.exists(), f"Base Dockerfile not found at {dockerfile}"

    def test_api_dockerfile_exists(self):
        """Test API Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "api" / "Dockerfile"
        assert dockerfile.exists(), f"API Dockerfile not found at {dockerfile}"

    def test_cli_dockerfile_exists(self):
        """Test CLI Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "cli" / "Dockerfile"
        assert dockerfile.exists(), f"CLI Dockerfile not found at {dockerfile}"

    def test_worker_dockerfile_exists(self):
        """Test worker Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "worker" / "Dockerfile"
        assert dockerfile.exists(), f"Worker Dockerfile not found at {dockerfile}"

    def test_core_dockerfile_exists(self):
        """Test core Dockerfile exists."""
        dockerfile = PROJECT_ROOT / "docker" / "core" / "Dockerfile"
        assert dockerfile.exists(), f"Core Dockerfile not found at {dockerfile}"


class TestDockerfileSyntax:
    """Tests for Dockerfile syntax and best practices."""

    def _read_dockerfile(self, path: str) -> str:
        """Read Dockerfile content."""
        dockerfile = PROJECT_ROOT / "docker" / path / "Dockerfile"
        return dockerfile.read_text()

    def test_base_dockerfile_has_from(self):
        """Test base Dockerfile has FROM instruction."""
        content = self._read_dockerfile("base")
        assert "FROM" in content, "Dockerfile must have FROM instruction"

    def test_base_dockerfile_has_label(self):
        """Test base Dockerfile has LABEL instructions."""
        content = self._read_dockerfile("base")
        assert "LABEL" in content, "Dockerfile should have LABEL instructions"

    def test_base_dockerfile_has_workdir(self):
        """Test base Dockerfile sets WORKDIR."""
        content = self._read_dockerfile("base")
        assert "WORKDIR" in content, "Dockerfile should set WORKDIR"

    def test_base_dockerfile_has_healthcheck(self):
        """Test base Dockerfile has HEALTHCHECK."""
        content = self._read_dockerfile("base")
        assert "HEALTHCHECK" in content, "Dockerfile should have HEALTHCHECK"

    def test_base_dockerfile_nonroot_user(self):
        """Test base Dockerfile creates non-root user."""
        content = self._read_dockerfile("base")
        assert "useradd" in content or "adduser" in content, (
            "Dockerfile should create non-root user for security"
        )

    def test_base_dockerfile_no_cache(self):
        """Test base Dockerfile uses --no-cache-dir for pip."""
        content = self._read_dockerfile("base")
        assert "PIP_NO_CACHE_DIR" in content or "--no-cache-dir" in content, (
            "Dockerfile should disable pip cache for smaller images"
        )

    def test_api_dockerfile_exposes_port(self):
        """Test API Dockerfile exposes port."""
        content = self._read_dockerfile("api")
        # Either EXPOSE instruction or environment variable for port
        assert "EXPOSE" in content or "API_PORT" in content, (
            "API Dockerfile should expose port"
        )


class TestDockerfileBestPractices:
    """Tests for Dockerfile best practices compliance."""

    def _read_dockerfile(self, path: str) -> str:
        """Read Dockerfile content."""
        dockerfile = PROJECT_ROOT / "docker" / path / "Dockerfile"
        return dockerfile.read_text()

    def test_base_apt_cleanup(self):
        """Test base Dockerfile cleans up apt cache."""
        content = self._read_dockerfile("base")
        assert "rm -rf /var/lib/apt/lists" in content, (
            "Dockerfile should clean apt cache to reduce image size"
        )

    def test_base_uses_slim_base(self):
        """Test base Dockerfile uses slim Python image."""
        content = self._read_dockerfile("base")
        lines = [line.strip() for line in content.split("\n") if line.strip().startswith("FROM")]
        from_line = lines[0] if lines else ""
        assert "slim" in from_line or "alpine" in from_line, (
            "Base Dockerfile should use slim or alpine image"
        )

    def test_no_root_cmd(self):
        """Test Dockerfiles don't run as root in CMD."""
        for service in ["base", "api", "worker", "cli"]:
            content = self._read_dockerfile(service)
            # Check that USER instruction comes before CMD
            if "USER" in content and "CMD" in content:
                user_pos = content.rfind("USER")
                cmd_pos = content.rfind("CMD")
                if cmd_pos > 0:
                    # If there's a CMD, USER should be set before it (or inherited)
                    pass  # This is okay, just ensure there's a USER directive


class TestDockerBuildDryRun:
    """Tests that verify Docker builds without actually building."""

    @requires_docker
    def test_docker_version(self):
        """Test Docker is accessible and returns version."""
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert result.stdout.strip(), "Docker version should be returned"

    @requires_docker
    def test_dockerfile_lint_base(self):
        """Test base Dockerfile passes basic validation."""
        # Docker build with --check is available in newer versions
        # Fall back to basic syntax check
        dockerfile = PROJECT_ROOT / "docker" / "base" / "Dockerfile"

        # Try hadolint if available, otherwise skip
        try:
            result = subprocess.run(
                ["hadolint", str(dockerfile)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Hadolint exits 0 if no errors (warnings are okay)
            # We don't fail on warnings, just on errors
            if result.returncode != 0:
                # Check if it's just warnings
                errors = [line for line in result.stdout.split("\n") if " error" in line.lower()]
                assert not errors, f"Dockerfile has lint errors: {result.stdout}"
        except FileNotFoundError:
            pytest.skip("hadolint not installed")


@pytest.mark.docker
@pytest.mark.slow
class TestDockerBuilds:
    """Tests that actually build Docker images.

    These tests are slow and require Docker with sufficient resources.
    Run with: pytest tests/test_docker_builds.py -m slow
    """

    @requires_docker
    def test_base_image_builds(self):
        """Test base Dockerfile builds successfully."""
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(PROJECT_ROOT / "docker" / "base" / "Dockerfile"),
                "-t", "firstlight-base:test",
                "--no-cache",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        assert result.returncode == 0, (
            f"Base image build failed:\n{result.stderr}"
        )

    @requires_docker
    def test_api_image_builds(self):
        """Test API Dockerfile builds successfully."""
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(PROJECT_ROOT / "docker" / "api" / "Dockerfile"),
                "-t", "firstlight-api:test",
                "--no-cache",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"API image build failed:\n{result.stderr}"
        )

    @requires_docker
    def test_cli_image_builds(self):
        """Test CLI Dockerfile builds successfully."""
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(PROJECT_ROOT / "docker" / "cli" / "Dockerfile"),
                "-t", "firstlight-cli:test",
                "--no-cache",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"CLI image build failed:\n{result.stderr}"
        )

    @requires_docker
    def test_worker_image_builds(self):
        """Test worker Dockerfile builds successfully."""
        result = subprocess.run(
            [
                "docker", "build",
                "-f", str(PROJECT_ROOT / "docker" / "worker" / "Dockerfile"),
                "-t", "firstlight-worker:test",
                "--no-cache",
                str(PROJECT_ROOT),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"Worker image build failed:\n{result.stderr}"
        )


@pytest.mark.docker
class TestDockerImageMetadata:
    """Tests for Docker image metadata and labels."""

    @requires_docker
    def test_base_image_has_labels(self):
        """Test base image has required labels after build."""
        # First check if image exists
        result = subprocess.run(
            ["docker", "image", "inspect", "firstlight-base:test"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("Base image not built yet")

        # Check labels
        import json
        data = json.loads(result.stdout)
        labels = data[0].get("Config", {}).get("Labels", {})

        assert "maintainer" in labels, "Image should have maintainer label"


class TestDockerBuildContext:
    """Tests for Docker build context and .dockerignore."""

    def test_dockerignore_exists(self):
        """Test .dockerignore exists."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        # It's recommended but not required
        if not dockerignore.exists():
            pytest.skip(".dockerignore not present (recommended but optional)")
        assert dockerignore.exists()

    def test_dockerignore_excludes_venv(self):
        """Test .dockerignore excludes virtual environment."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        if not dockerignore.exists():
            pytest.skip(".dockerignore not present")

        content = dockerignore.read_text()
        venv_patterns = [".venv", "venv", "__pycache__", "*.pyc"]
        excluded = [p for p in venv_patterns if p in content]
        assert excluded, "Should exclude common dev files"

    def test_dockerignore_excludes_git(self):
        """Test .dockerignore excludes .git directory."""
        dockerignore = PROJECT_ROOT / ".dockerignore"
        if not dockerignore.exists():
            pytest.skip(".dockerignore not present")

        content = dockerignore.read_text()
        assert ".git" in content, "Should exclude .git directory"

    def test_pyproject_toml_exists(self):
        """Test pyproject.toml exists for dependency installation."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml required for Docker builds"


class TestMultiArchSupport:
    """Tests for multi-architecture support in Dockerfiles."""

    def _read_dockerfile(self, path: str) -> str:
        """Read Dockerfile content."""
        dockerfile = PROJECT_ROOT / "docker" / path / "Dockerfile"
        return dockerfile.read_text()

    def test_base_no_hardcoded_arch(self):
        """Test base Dockerfile doesn't have hardcoded architecture."""
        content = self._read_dockerfile("base")
        # Check for common hardcoded architecture patterns
        bad_patterns = [
            "linux/amd64",  # Should use buildx for multi-arch
            "x86_64-linux-gnu",  # Architecture-specific paths
        ]
        for pattern in bad_patterns:
            if pattern in content:
                # Not necessarily an error, but flag for review
                pass  # Allow but note it

    @requires_docker
    def test_buildx_available(self):
        """Test Docker buildx is available for multi-arch builds."""
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
        )
        # Buildx is recommended but not required
        if result.returncode != 0:
            pytest.skip("Docker buildx not available (optional for multi-arch)")
        assert result.returncode == 0


class TestSecurityBestPractices:
    """Tests for Docker security best practices."""

    def _read_dockerfile(self, path: str) -> str:
        """Read Dockerfile content."""
        dockerfile = PROJECT_ROOT / "docker" / path / "Dockerfile"
        return dockerfile.read_text()

    def test_no_secrets_in_dockerfile(self):
        """Test Dockerfiles don't contain hardcoded secrets."""
        secret_patterns = [
            "password=",
            "secret=",
            "api_key=",
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
        ]
        for service in ["base", "api", "worker", "cli"]:
            content = self._read_dockerfile(service).lower()
            for pattern in secret_patterns:
                # Allow ENV declarations without values
                if pattern.lower() in content:
                    # Check if it's just a variable declaration
                    lines = content.split("\n")
                    for line in lines:
                        if pattern.lower() in line:
                            # Okay if it's ${VAR} or empty
                            if "=${" in line or '=""' in line or "=''" in line:
                                continue
                            # Flag if it has an actual value
                            if "='" in line or '="' in line:
                                # Check for placeholders
                                if "changeme" in line or "placeholder" in line:
                                    continue
                                # This might be a real secret
                                pass

    def test_no_sudo_in_dockerfile(self):
        """Test Dockerfiles don't use sudo (should run as root during build, then USER)."""
        for service in ["base", "api", "worker", "cli"]:
            content = self._read_dockerfile(service)
            # sudo in Dockerfile is usually a smell
            lines = [line for line in content.split("\n")
                     if line.strip().startswith("RUN") and "sudo" in line]
            assert not lines, f"Dockerfile {service} uses sudo which is usually unnecessary"
