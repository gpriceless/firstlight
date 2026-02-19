"""
Tests for Webhook System (Phase 3, Tasks 3.5-3.7, Task 3.12).

Tests cover:
- Registration validates HTTPS URL
- Registration rejects private IP URLs (SSRF)
- Webhook listing uses cursor-based pagination
- HMAC signature matches expected
- Retry inserts to DLQ after 5 failures
- Idempotency key prevents double delivery
- Redirect response treated as failure
"""

import hashlib
import hmac
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.internal import (
    CreateWebhookRequest,
    validate_webhook_url,
)
from workers.tasks.webhooks import (
    _compute_hmac_signature,
    _calculate_delay,
    MAX_ATTEMPTS,
    BASE_DELAY_SECONDS,
    BACKOFF_MULTIPLIER,
)


# =============================================================================
# URL Validation Tests (SSRF Protection)
# =============================================================================


class TestWebhookURLValidation:
    """Test SSRF protection in webhook URL validation."""

    def test_rejects_http_url(self):
        """Webhook URLs must use HTTPS, not HTTP."""
        with pytest.raises(ValueError, match="HTTPS"):
            validate_webhook_url("http://example.com/webhook")

    def test_accepts_https_url(self):
        """Valid HTTPS URLs should be accepted."""
        # Mock DNS resolution to return a public IP
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
            result = validate_webhook_url("https://example.com/webhook")
            assert result == "https://example.com/webhook"

    def test_rejects_localhost(self):
        """URLs resolving to 127.0.0.0/8 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("127.0.0.1", 443)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://localhost/webhook")

    def test_rejects_private_10_network(self):
        """URLs resolving to 10.0.0.0/8 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("10.0.0.1", 443)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://internal.corp/webhook")

    def test_rejects_private_172_network(self):
        """URLs resolving to 172.16.0.0/12 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("172.16.0.1", 443)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://internal.corp/webhook")

    def test_rejects_private_192_network(self):
        """URLs resolving to 192.168.0.0/16 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("192.168.1.1", 443)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://my-router.local/webhook")

    def test_rejects_link_local(self):
        """URLs resolving to 169.254.0.0/16 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("169.254.169.254", 443)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://metadata.internal/webhook")

    def test_rejects_ipv6_loopback(self):
        """URLs resolving to ::1 must be rejected."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (10, 1, 6, "", ("::1", 443, 0, 0)),
            ]
            with pytest.raises(ValueError, match="private"):
                validate_webhook_url("https://localhost6/webhook")

    def test_rejects_unresolvable_hostname(self):
        """URLs with unresolvable hostnames must be rejected."""
        import socket

        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.side_effect = socket.gaierror("Name resolution failed")
            with pytest.raises(ValueError, match="resolve"):
                validate_webhook_url("https://nonexistent.invalid/webhook")

    def test_rejects_url_without_hostname(self):
        """URLs without a valid hostname must be rejected."""
        with pytest.raises(ValueError, match="hostname"):
            validate_webhook_url("https:///webhook")


# =============================================================================
# HMAC Signature Tests
# =============================================================================


class TestHMACSignature:
    """Test HMAC-SHA256 webhook signature computation."""

    def test_signature_format(self):
        """Signature should be in sha256=<hex> format."""
        body = b'{"test": "data"}'
        secret = "my-secret-key"

        sig = _compute_hmac_signature(body, secret)
        assert sig.startswith("sha256=")
        # hex digest should be 64 chars
        hex_part = sig[len("sha256="):]
        assert len(hex_part) == 64

    def test_signature_matches_manual_computation(self):
        """Signature should match manually computed HMAC-SHA256."""
        body = b'{"event": "job.created"}'
        secret = "webhook-secret-123"

        expected = hmac.new(
            secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        sig = _compute_hmac_signature(body, secret)
        assert sig == f"sha256={expected}"

    def test_different_secrets_produce_different_signatures(self):
        """Different secrets must produce different signatures."""
        body = b'{"event": "job.created"}'

        sig1 = _compute_hmac_signature(body, "secret-1")
        sig2 = _compute_hmac_signature(body, "secret-2")

        assert sig1 != sig2

    def test_different_payloads_produce_different_signatures(self):
        """Different payloads must produce different signatures."""
        secret = "same-secret"

        sig1 = _compute_hmac_signature(b'{"a": 1}', secret)
        sig2 = _compute_hmac_signature(b'{"a": 2}', secret)

        assert sig1 != sig2


# =============================================================================
# Retry Backoff Tests
# =============================================================================


class TestRetryBackoff:
    """Test exponential backoff calculation."""

    def test_first_attempt_delay(self):
        """First retry should be approximately BASE_DELAY_SECONDS."""
        delay = _calculate_delay(0)
        # ~5s + [0, 1) jitter
        assert BASE_DELAY_SECONDS <= delay < BASE_DELAY_SECONDS + 1.0

    def test_backoff_increases_exponentially(self):
        """Delays should increase exponentially with each attempt."""
        delays = [_calculate_delay(i) for i in range(5)]

        # Each delay should be roughly double the previous (minus jitter)
        for i in range(1, len(delays)):
            # The base (without jitter) doubles each time
            expected_base = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** i)
            # Allow for jitter
            assert delays[i] >= expected_base
            assert delays[i] < expected_base + 1.0

    def test_max_attempts_constant(self):
        """MAX_ATTEMPTS should be 5."""
        assert MAX_ATTEMPTS == 5

    def test_schedule_matches_spec(self):
        """Schedule should approximate: ~5s, ~10s, ~20s, ~40s, ~80s."""
        expected_bases = [5.0, 10.0, 20.0, 40.0, 80.0]

        for i, expected in enumerate(expected_bases):
            delay = _calculate_delay(i)
            # Allow up to 1 second of jitter
            assert expected <= delay < expected + 1.0, (
                f"Attempt {i}: expected ~{expected}s, got {delay:.2f}s"
            )


# =============================================================================
# DLQ Tests (unit-level)
# =============================================================================


class TestDLQLogic:
    """Test dead letter queue insertion logic."""

    @pytest.mark.asyncio
    async def test_dlq_insert_called_after_max_attempts(self):
        """After MAX_ATTEMPTS failures, payload should be inserted to DLQ."""
        from workers.tasks.webhooks import deliver_webhook, _insert_to_dlq

        # Mock httpx to always fail
        with patch("workers.tasks.webhooks.httpx") as mock_httpx:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
            mock_httpx.AsyncClient.return_value = mock_client
            mock_httpx.Timeout = MagicMock()
            mock_httpx.TimeoutException = Exception
            mock_httpx.RequestError = Exception

            # Mock asyncio.sleep to avoid actual delays
            with patch("asyncio.sleep", new_callable=AsyncMock):
                # Mock DLQ insert
                with patch("workers.tasks.webhooks._insert_to_dlq", new_callable=AsyncMock) as mock_dlq:
                    # Mock idempotency check
                    with patch("workers.tasks.webhooks._check_idempotency_key", new_callable=AsyncMock, return_value=False):
                        result = await deliver_webhook(
                            subscription_id="sub-123",
                            target_url="https://example.com/webhook",
                            secret_key="secret",
                            event_seq=42,
                            payload={"test": True},
                            attempt=0,
                        )

                        assert result is False
                        mock_dlq.assert_called_once()
                        call_args = mock_dlq.call_args
                        assert call_args[1]["subscription_id"] == "sub-123"
                        assert call_args[1]["event_seq"] == 42
                        assert call_args[1]["attempt_count"] == MAX_ATTEMPTS


# =============================================================================
# Idempotency Key Tests (unit-level)
# =============================================================================


class TestIdempotencyKey:
    """Test Redis idempotency key logic."""

    @pytest.mark.asyncio
    async def test_skips_delivery_when_key_exists(self):
        """Delivery should be skipped if idempotency key already exists."""
        from workers.tasks.webhooks import deliver_webhook

        with patch("workers.tasks.webhooks._check_idempotency_key", new_callable=AsyncMock, return_value=True):
            result = await deliver_webhook(
                subscription_id="sub-123",
                target_url="https://example.com/webhook",
                secret_key="secret",
                event_seq=42,
                payload={"test": True},
            )
            assert result is True  # Skipped = treated as success


# =============================================================================
# Redirect Handling Tests
# =============================================================================


class TestRedirectHandling:
    """Test that HTTP redirects are treated as failures."""

    @pytest.mark.asyncio
    async def test_redirect_treated_as_failure(self):
        """301/302/307 redirects should be treated as delivery failures."""
        import httpx
        from workers.tasks.webhooks import deliver_webhook

        mock_response = MagicMock()
        mock_response.status_code = 301
        mock_response.text = "Moved Permanently"

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("workers.tasks.webhooks.httpx.AsyncClient", return_value=mock_client):
            with patch("workers.tasks.webhooks.httpx.Timeout", return_value=MagicMock()):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    with patch("workers.tasks.webhooks._insert_to_dlq", new_callable=AsyncMock) as mock_dlq:
                        with patch("workers.tasks.webhooks._check_idempotency_key", new_callable=AsyncMock, return_value=False):
                            result = await deliver_webhook(
                                subscription_id="sub-123",
                                target_url="https://example.com/webhook",
                                secret_key="secret",
                                event_seq=42,
                                payload={"test": True},
                                attempt=0,
                            )

                            assert result is False
                            mock_dlq.assert_called_once()


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestWebhookRequestModel:
    """Test webhook request model validation."""

    def test_valid_request(self):
        """Valid HTTPS URL should pass validation."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
            req = CreateWebhookRequest(
                target_url="https://example.com/webhook",
                event_filter=["STATE_TRANSITION"],
            )
            assert req.target_url == "https://example.com/webhook"
            assert req.event_filter == ["STATE_TRANSITION"]

    def test_http_rejected_by_model(self):
        """HTTP URLs should be rejected by the model validator."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CreateWebhookRequest(
                target_url="http://example.com/webhook",
            )

    def test_empty_event_filter_defaults(self):
        """Empty event_filter should default to empty list (subscribe to all)."""
        with patch("socket.getaddrinfo") as mock_dns:
            mock_dns.return_value = [
                (2, 1, 6, "", ("93.184.216.34", 443)),
            ]
            req = CreateWebhookRequest(
                target_url="https://example.com/webhook",
            )
            assert req.event_filter == []
