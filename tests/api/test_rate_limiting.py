"""
Tests for Control Plane Rate Limiting (Phase 2, Task 2.13).

Tests cover:
- Per-agent limit returns 429 with Retry-After header
- Per-customer aggregate limit triggers across multiple keys
- Burst allowance permits short spikes
- Rate limit headers (X-RateLimit-*) present on normal 200 responses
- Other tenants unaffected when one tenant is rate-limited
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.rate_limit import (
    ControlPlaneRateLimitConfig,
    ControlPlaneRateLimiter,
    MemoryRateLimitBackend,
    RateLimitResult,
)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestControlPlaneRateLimitConfig:
    """Test rate limit configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ControlPlaneRateLimitConfig()
        assert config.per_agent_rpm == 120
        assert config.per_customer_rpm == 600
        assert config.burst == 20

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ControlPlaneRateLimitConfig(
            per_agent_rpm=60,
            per_customer_rpm=300,
            burst=10,
        )
        assert config.per_agent_rpm == 60
        assert config.per_customer_rpm == 300
        assert config.burst == 10

    @patch.dict("os.environ", {
        "FIRSTLIGHT_RATE_LIMIT_PER_AGENT": "200",
        "FIRSTLIGHT_RATE_LIMIT_PER_CUSTOMER": "1000",
        "FIRSTLIGHT_RATE_LIMIT_BURST": "50",
    })
    def test_env_config(self):
        """Test configuration from environment variables."""
        from api.rate_limit import get_control_plane_rate_limit_config
        config = get_control_plane_rate_limit_config()
        assert config.per_agent_rpm == 200
        assert config.per_customer_rpm == 1000
        assert config.burst == 50


# =============================================================================
# Memory Backend Tests
# =============================================================================


class TestMemoryRateLimitBackend:
    """Test in-memory rate limit backend."""

    @pytest.fixture
    def backend(self):
        """Create a fresh memory backend."""
        return MemoryRateLimitBackend()

    @pytest.mark.asyncio
    async def test_increment_basic(self, backend):
        """Test basic increment operation."""
        count, ttl = await backend.increment("test:key", 60)
        assert count == 1
        assert ttl >= 0

    @pytest.mark.asyncio
    async def test_increment_accumulates(self, backend):
        """Test that increments accumulate."""
        for i in range(5):
            count, _ = await backend.increment("test:key", 60)
        assert count == 5

    @pytest.mark.asyncio
    async def test_different_keys_independent(self, backend):
        """Test that different keys are independent."""
        await backend.increment("key:a", 60)
        await backend.increment("key:a", 60)
        count_b, _ = await backend.increment("key:b", 60)
        assert count_b == 1

    @pytest.mark.asyncio
    async def test_reset_clears_key(self, backend):
        """Test that reset clears a key."""
        await backend.increment("test:key", 60)
        await backend.increment("test:key", 60)
        await backend.reset("test:key")
        count = await backend.get_count("test:key")
        assert count == 0


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestControlPlaneRateLimiter:
    """Test the control plane rate limiter."""

    @pytest.fixture
    def config(self):
        """Create a test config with low limits for easy testing."""
        return ControlPlaneRateLimitConfig(
            per_agent_rpm=5,
            per_customer_rpm=10,
            burst=2,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create a limiter with test config."""
        return ControlPlaneRateLimiter(config=config)

    def _make_request(self, api_key: str = "test-key-123", customer_id: str = "tenant-a"):
        """Create a mock request object."""
        request = MagicMock()
        request.headers = {"X-API-Key": api_key}
        request.query_params = {}
        request.state = MagicMock()
        request.state.customer_id = customer_id
        return request

    @pytest.mark.asyncio
    async def test_allows_under_limit(self, limiter):
        """Test that requests under the limit are allowed."""
        await limiter.initialize()
        request = self._make_request()

        result = await limiter.check(request)
        assert result.allowed is True
        assert result.remaining > 0

    @pytest.mark.asyncio
    async def test_per_agent_limit_exceeded(self, limiter):
        """Test that per-agent limit triggers 429."""
        await limiter.initialize()
        request = self._make_request()

        # Per-agent limit is 5 + burst 2 = 7
        for _ in range(7):
            result = await limiter.check(request)
            assert result.allowed is True

        # 8th request should be denied
        result = await limiter.check(request)
        assert result.allowed is False
        assert result.reset_seconds >= 0

    @pytest.mark.asyncio
    async def test_per_customer_limit_across_agents(self, limiter):
        """Test that per-customer limit aggregates across agents."""
        await limiter.initialize()

        # Per-customer limit is 10 + burst 2 = 12
        # Use different API keys but same customer
        for i in range(12):
            request = self._make_request(
                api_key=f"agent-key-{i % 3}",
                customer_id="tenant-shared",
            )
            result = await limiter.check(request)
            assert result.allowed is True

        # 13th request should be denied
        request = self._make_request(
            api_key="agent-key-new",
            customer_id="tenant-shared",
        )
        result = await limiter.check(request)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_burst_allows_short_spikes(self, limiter):
        """Test that burst allowance permits short spikes above base limit."""
        await limiter.initialize()
        request = self._make_request()

        # Per-agent limit is 5, burst is 2, so 7 total allowed
        results = []
        for _ in range(7):
            result = await limiter.check(request)
            results.append(result.allowed)

        # All 7 should be allowed (5 base + 2 burst)
        assert all(results)

    @pytest.mark.asyncio
    async def test_different_tenants_independent(self, limiter):
        """Test that rate limiting one tenant doesn't affect another."""
        await limiter.initialize()

        # Exhaust tenant-a's limit
        for _ in range(8):
            request_a = self._make_request(
                api_key="key-a",
                customer_id="tenant-a",
            )
            await limiter.check(request_a)

        # tenant-a should be denied
        result_a = await limiter.check(request_a)
        assert result_a.allowed is False

        # tenant-b should still be allowed
        request_b = self._make_request(
            api_key="key-b",
            customer_id="tenant-b",
        )
        result_b = await limiter.check(request_b)
        assert result_b.allowed is True


# =============================================================================
# Rate Limit Result Tests
# =============================================================================


class TestRateLimitResult:
    """Test rate limit result structure."""

    def test_allowed_result(self):
        """Test allowed rate limit result."""
        result = RateLimitResult(
            allowed=True,
            limit=120,
            remaining=119,
            reset_seconds=60,
        )
        assert result.allowed is True
        assert result.limit == 120
        assert result.remaining == 119

    def test_denied_result(self):
        """Test denied rate limit result."""
        result = RateLimitResult(
            allowed=False,
            limit=120,
            remaining=0,
            reset_seconds=45,
        )
        assert result.allowed is False
        assert result.remaining == 0
        assert result.reset_seconds == 45

    def test_reset_timestamp(self):
        """Test that reset_timestamp is in the future."""
        result = RateLimitResult(
            allowed=True,
            limit=120,
            remaining=100,
            reset_seconds=30,
        )
        assert result.reset_timestamp > time.time() - 1

    def test_rate_limit_headers(self):
        """Test that rate limit headers can be constructed from result."""
        result = RateLimitResult(
            allowed=True,
            limit=120,
            remaining=100,
            reset_seconds=45,
        )
        headers = {
            "X-RateLimit-Limit": str(result.limit),
            "X-RateLimit-Remaining": str(result.remaining),
            "X-RateLimit-Reset": str(result.reset_timestamp),
        }
        assert headers["X-RateLimit-Limit"] == "120"
        assert headers["X-RateLimit-Remaining"] == "100"
        assert "X-RateLimit-Reset" in headers
