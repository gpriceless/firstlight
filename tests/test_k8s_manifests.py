"""
Tests for Kubernetes manifest validation (Group N, Track 8).

These tests validate that Kubernetes YAML manifests are syntactically
correct and follow best practices. They do not require a Kubernetes
cluster - they only validate the YAML structure.

Markers:
    @pytest.mark.slow - Tests that involve kubectl validation
    @pytest.mark.integration - Tests requiring kubectl

Usage:
    pytest tests/test_k8s_manifests.py -v
    pytest tests/test_k8s_manifests.py -v -m "not slow"
"""

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Kubernetes manifest directories
K8S_DIRS = [
    PROJECT_ROOT / "deploy" / "gcp" / "kubernetes",
    PROJECT_ROOT / "deploy" / "azure" / "aks",
]


def kubectl_available() -> bool:
    """Check if kubectl is available."""
    try:
        result = subprocess.run(
            ["kubectl", "version", "--client", "--output=yaml"],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


requires_kubectl = pytest.mark.skipif(
    not kubectl_available(), reason="kubectl not available"
)


def find_k8s_manifests() -> list[Path]:
    """Find all Kubernetes YAML manifests."""
    manifests = []
    for k8s_dir in K8S_DIRS:
        if k8s_dir.exists():
            manifests.extend(k8s_dir.glob("**/*.yaml"))
            manifests.extend(k8s_dir.glob("**/*.yml"))
    return manifests


class TestK8sManifestsExist:
    """Tests that verify Kubernetes manifests exist."""

    def test_gcp_deployment_exists(self):
        """Test GCP Kubernetes deployment exists."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        assert deployment.exists(), "GCP Kubernetes deployment not found"

    def test_azure_deployment_exists(self):
        """Test Azure AKS deployment exists."""
        deployment = PROJECT_ROOT / "deploy" / "azure" / "aks" / "deployment.yaml"
        assert deployment.exists(), "Azure AKS deployment not found"


class TestK8sYamlSyntax:
    """Tests for Kubernetes YAML syntax."""

    @pytest.mark.parametrize("manifest_path", find_k8s_manifests())
    def test_manifest_valid_yaml(self, manifest_path: Path):
        """Test manifest is valid YAML."""
        with open(manifest_path) as f:
            # Handle multi-document YAML
            docs = list(yaml.safe_load_all(f))
        assert docs, f"{manifest_path.name} should contain at least one document"

    def test_gcp_deployment_valid_yaml(self):
        """Test GCP deployment is valid YAML."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        with open(deployment) as f:
            manifest = yaml.safe_load(f)
        assert manifest is not None
        assert isinstance(manifest, dict)


class TestK8sResourceStructure:
    """Tests for Kubernetes resource structure."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def test_gcp_deployment_has_required_fields(self):
        """Test GCP deployment has required fields."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)

        assert "apiVersion" in manifest, "Deployment must have apiVersion"
        assert "kind" in manifest, "Deployment must have kind"
        assert "metadata" in manifest, "Deployment must have metadata"
        assert "spec" in manifest, "Deployment must have spec"

    def test_gcp_deployment_kind_correct(self):
        """Test GCP deployment has correct kind."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        assert manifest["kind"] == "Deployment", "Kind should be Deployment"

    def test_deployment_has_replicas(self):
        """Test deployment specifies replicas."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        spec = manifest.get("spec", {})
        assert "replicas" in spec, "Deployment should specify replicas"
        assert isinstance(spec["replicas"], int), "Replicas should be integer"
        assert spec["replicas"] >= 1, "Should have at least 1 replica"

    def test_deployment_has_selector(self):
        """Test deployment has selector."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        spec = manifest.get("spec", {})
        assert "selector" in spec, "Deployment should have selector"
        assert "matchLabels" in spec["selector"], "Selector should have matchLabels"

    def test_deployment_has_template(self):
        """Test deployment has pod template."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        spec = manifest.get("spec", {})
        assert "template" in spec, "Deployment should have template"
        template = spec["template"]
        assert "metadata" in template, "Template should have metadata"
        assert "spec" in template, "Template should have spec"


class TestK8sPodSpec:
    """Tests for Kubernetes Pod specifications."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _get_pod_spec(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Extract pod spec from deployment."""
        return manifest.get("spec", {}).get("template", {}).get("spec", {})

    def test_deployment_has_containers(self):
        """Test deployment defines containers."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        assert "containers" in pod_spec, "Pod spec should have containers"
        assert len(pod_spec["containers"]) > 0, "Should have at least one container"

    def test_container_has_image(self):
        """Test container specifies image."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        for container in pod_spec.get("containers", []):
            assert "image" in container, f"Container {container.get('name')} should have image"
            assert container["image"], "Image should not be empty"

    def test_container_has_resources(self):
        """Test container specifies resource limits."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        for container in pod_spec.get("containers", []):
            resources = container.get("resources", {})
            has_limits = "limits" in resources
            has_requests = "requests" in resources
            assert has_limits or has_requests, (
                f"Container {container.get('name')} should have resource limits/requests"
            )

    def test_container_has_liveness_probe(self):
        """Test container has liveness probe."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        for container in pod_spec.get("containers", []):
            assert "livenessProbe" in container, (
                f"Container {container.get('name')} should have livenessProbe"
            )

    def test_container_has_readiness_probe(self):
        """Test container has readiness probe."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        for container in pod_spec.get("containers", []):
            assert "readinessProbe" in container, (
                f"Container {container.get('name')} should have readinessProbe"
            )


class TestK8sSecurityBestPractices:
    """Tests for Kubernetes security best practices."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _get_pod_spec(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Extract pod spec from deployment."""
        return manifest.get("spec", {}).get("template", {}).get("spec", {})

    def test_uses_service_account(self):
        """Test deployment uses service account."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        assert "serviceAccountName" in pod_spec, (
            "Deployment should specify serviceAccountName"
        )

    def test_no_host_network(self):
        """Test deployment doesn't use host network."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        assert not pod_spec.get("hostNetwork", False), (
            "Should not use hostNetwork"
        )

    def test_no_host_pid(self):
        """Test deployment doesn't use host PID namespace."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        assert not pod_spec.get("hostPID", False), (
            "Should not use hostPID"
        )

    def test_secrets_from_secret_refs(self):
        """Test secrets are loaded from SecretKeyRef, not inline."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        for container in pod_spec.get("containers", []):
            for env in container.get("env", []):
                if "secret" in env.get("name", "").lower():
                    # Should use valueFrom.secretKeyRef
                    value_from = env.get("valueFrom", {})
                    assert "secretKeyRef" in value_from, (
                        f"Secret {env['name']} should use secretKeyRef"
                    )


class TestK8sMetadata:
    """Tests for Kubernetes metadata best practices."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def test_has_namespace(self):
        """Test deployment specifies namespace."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        metadata = manifest.get("metadata", {})

        assert "namespace" in metadata, "Deployment should specify namespace"

    def test_has_labels(self):
        """Test deployment has labels."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        metadata = manifest.get("metadata", {})

        assert "labels" in metadata, "Deployment should have labels"
        labels = metadata["labels"]
        assert "app" in labels, "Should have 'app' label"

    def test_pod_template_has_labels(self):
        """Test pod template has matching labels."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)

        spec = manifest.get("spec", {})
        selector_labels = spec.get("selector", {}).get("matchLabels", {})
        template_labels = (
            spec.get("template", {}).get("metadata", {}).get("labels", {})
        )

        # Selector labels should be subset of template labels
        for key, value in selector_labels.items():
            assert key in template_labels, f"Template missing selector label: {key}"
            assert template_labels[key] == value, (
                f"Template label {key} doesn't match selector"
            )


class TestK8sDeploymentStrategy:
    """Tests for deployment strategy configuration."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def test_has_update_strategy(self):
        """Test deployment has update strategy."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        spec = manifest.get("spec", {})

        assert "strategy" in spec, "Deployment should specify update strategy"

    def test_rolling_update_configured(self):
        """Test rolling update is configured properly."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        spec = manifest.get("spec", {})
        strategy = spec.get("strategy", {})

        if strategy.get("type") == "RollingUpdate":
            rolling = strategy.get("rollingUpdate", {})
            # Should have maxUnavailable and/or maxSurge
            has_max_unavailable = "maxUnavailable" in rolling
            has_max_surge = "maxSurge" in rolling
            assert has_max_unavailable or has_max_surge, (
                "RollingUpdate should configure maxUnavailable and/or maxSurge"
            )


class TestK8sAffinity:
    """Tests for pod affinity/anti-affinity configuration."""

    def _load_manifest(self, path: Path) -> dict[str, Any]:
        """Load a Kubernetes manifest."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _get_pod_spec(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Extract pod spec from deployment."""
        return manifest.get("spec", {}).get("template", {}).get("spec", {})

    def test_has_pod_anti_affinity(self):
        """Test deployment has pod anti-affinity for HA."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        manifest = self._load_manifest(deployment)
        pod_spec = self._get_pod_spec(manifest)

        affinity = pod_spec.get("affinity", {})
        pod_anti_affinity = affinity.get("podAntiAffinity", {})

        # Should have some anti-affinity rule
        has_preferred = "preferredDuringSchedulingIgnoredDuringExecution" in pod_anti_affinity
        has_required = "requiredDuringSchedulingIgnoredDuringExecution" in pod_anti_affinity

        assert has_preferred or has_required, (
            "Production deployments should have podAntiAffinity for HA"
        )


@requires_kubectl
class TestK8sKubectlValidation:
    """Tests that use kubectl for validation.

    These tests require kubectl to be installed but don't require
    a running cluster. They use --dry-run=client for validation.
    """

    @pytest.mark.parametrize("manifest_path", find_k8s_manifests())
    def test_manifest_dry_run(self, manifest_path: Path):
        """Test manifest passes kubectl dry-run validation."""
        result = subprocess.run(
            [
                "kubectl", "apply",
                "-f", str(manifest_path),
                "--dry-run=client",
                "--validate=true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"Manifest {manifest_path.name} failed validation:\n{result.stderr}"
        )

    def test_gcp_deployment_dry_run(self):
        """Test GCP deployment passes kubectl validation."""
        deployment = PROJECT_ROOT / "deploy" / "gcp" / "kubernetes" / "deployment.yaml"
        result = subprocess.run(
            [
                "kubectl", "apply",
                "-f", str(deployment),
                "--dry-run=client",
                "--validate=true",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"GCP deployment failed validation:\n{result.stderr}"
        )


class TestAzureAKSManifest:
    """Tests specific to Azure AKS deployment manifest."""

    def _load_manifest(self) -> dict[str, Any]:
        """Load Azure AKS manifest."""
        deployment = PROJECT_ROOT / "deploy" / "azure" / "aks" / "deployment.yaml"
        with open(deployment) as f:
            return yaml.safe_load(f)

    def test_azure_deployment_valid_structure(self):
        """Test Azure deployment has valid structure."""
        manifest = self._load_manifest()
        assert "apiVersion" in manifest
        assert "kind" in manifest
        assert "spec" in manifest

    def test_azure_deployment_has_containers(self):
        """Test Azure deployment has containers."""
        manifest = self._load_manifest()
        containers = (
            manifest.get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        assert len(containers) > 0, "Azure deployment should have containers"
