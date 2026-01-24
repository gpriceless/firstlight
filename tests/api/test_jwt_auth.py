"""
Tests for JWT authentication.

Tests JWT token validation, expiration handling, and invalid token rejection.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from fastapi import HTTPException

from api.dependencies import get_bearer_token
from api.jwt_handler import JWTConfig, JWTHandler
from api.models.errors import AuthenticationError


class TestJWTConfig:
    """Test JWT configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = JWTConfig()
        assert config.algorithm == "HS256"
        assert config.issuer == "firstlight-api"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            os.environ,
            {
                "JWT_SECRET": "test-secret",
                "JWT_ALGORITHM": "HS512",
                "JWT_ISSUER": "test-issuer",
            },
        ):
            config = JWTConfig()
            assert config.secret == "test-secret"
            assert config.algorithm == "HS512"
            assert config.issuer == "test-issuer"

    def test_is_configured(self):
        """Test configuration check."""
        with patch.dict(os.environ, {"JWT_SECRET": "test-secret"}):
            config = JWTConfig()
            assert config.is_configured()

        with patch.dict(os.environ, {"JWT_SECRET": ""}):
            config = JWTConfig()
            assert not config.is_configured()


class TestJWTHandler:
    """Test JWT handler functionality."""

    @pytest.fixture
    def jwt_config(self):
        """Create test JWT configuration."""
        config = JWTConfig()
        config.secret = "test-secret-key-for-testing"
        config.algorithm = "HS256"
        config.issuer = "firstlight-api"
        return config

    @pytest.fixture
    def jwt_handler(self, jwt_config):
        """Create JWT handler with test configuration."""
        return JWTHandler(jwt_config)

    @pytest.fixture
    def valid_token(self, jwt_config):
        """Create a valid JWT token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now + timedelta(hours=1),
            "iss": jwt_config.issuer,
            "iat": now,
        }
        return jwt.encode(payload, jwt_config.secret, algorithm=jwt_config.algorithm)

    @pytest.fixture
    def expired_token(self, jwt_config):
        """Create an expired JWT token."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now - timedelta(hours=1),  # Expired 1 hour ago
            "iss": jwt_config.issuer,
            "iat": now - timedelta(hours=2),
        }
        return jwt.encode(payload, jwt_config.secret, algorithm=jwt_config.algorithm)

    @pytest.fixture
    def invalid_signature_token(self, jwt_config):
        """Create a token with invalid signature."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now + timedelta(hours=1),
            "iss": jwt_config.issuer,
            "iat": now,
        }
        # Sign with different secret
        return jwt.encode(payload, "wrong-secret", algorithm=jwt_config.algorithm)

    @pytest.fixture
    def wrong_issuer_token(self, jwt_config):
        """Create a token with wrong issuer."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now + timedelta(hours=1),
            "iss": "wrong-issuer",
            "iat": now,
        }
        return jwt.encode(payload, jwt_config.secret, algorithm=jwt_config.algorithm)

    def test_decode_valid_token(self, jwt_handler, valid_token):
        """Test decoding a valid token."""
        payload = jwt_handler.decode_token(valid_token)
        assert payload is not None
        assert payload["sub"] == "user-123"
        assert "exp" in payload
        assert payload["iss"] == "firstlight-api"

    def test_decode_expired_token(self, jwt_handler, expired_token):
        """Test that expired tokens raise ExpiredSignatureError."""
        from jwt.exceptions import ExpiredSignatureError

        with pytest.raises(ExpiredSignatureError):
            jwt_handler.decode_token(expired_token)

    def test_decode_invalid_signature(self, jwt_handler, invalid_signature_token):
        """Test that tokens with invalid signature raise InvalidTokenError."""
        from jwt.exceptions import InvalidTokenError

        with pytest.raises(InvalidTokenError):
            jwt_handler.decode_token(invalid_signature_token)

    def test_decode_wrong_issuer(self, jwt_handler, wrong_issuer_token):
        """Test that tokens with wrong issuer raise InvalidTokenError."""
        from jwt.exceptions import InvalidTokenError

        with pytest.raises(InvalidTokenError, match="Invalid issuer"):
            jwt_handler.decode_token(wrong_issuer_token)

    def test_validate_token_valid(self, jwt_handler, valid_token):
        """Test validate_token with valid token."""
        payload = jwt_handler.validate_token(valid_token)
        assert payload is not None
        assert payload["sub"] == "user-123"

    def test_validate_token_expired(self, jwt_handler, expired_token):
        """Test validate_token with expired token returns None."""
        payload = jwt_handler.validate_token(expired_token)
        assert payload is None

    def test_validate_token_invalid(self, jwt_handler, invalid_signature_token):
        """Test validate_token with invalid signature returns None."""
        payload = jwt_handler.validate_token(invalid_signature_token)
        assert payload is None

    def test_get_user_id_valid(self, jwt_handler, valid_token):
        """Test extracting user ID from valid token."""
        user_id = jwt_handler.get_user_id(valid_token)
        assert user_id == "user-123"

    def test_get_user_id_invalid(self, jwt_handler, expired_token):
        """Test extracting user ID from invalid token returns None."""
        user_id = jwt_handler.get_user_id(expired_token)
        assert user_id is None

    def test_check_expiration_expired(self, jwt_handler, expired_token):
        """Test checking expiration on expired token."""
        assert jwt_handler.check_expiration(expired_token) is True

    def test_check_expiration_valid(self, jwt_handler, valid_token):
        """Test checking expiration on valid token."""
        assert jwt_handler.check_expiration(valid_token) is False

    def test_handler_without_secret(self):
        """Test handler behavior when JWT_SECRET not configured."""
        config = JWTConfig()
        config.secret = ""
        handler = JWTHandler(config)

        with pytest.raises(Exception):  # Should raise InvalidTokenError
            handler.decode_token("any-token")


class TestGetBearerToken:
    """Test the get_bearer_token dependency."""

    @pytest.fixture
    def mock_settings(self):
        """Create mock settings."""
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.auth.enabled = True
        return settings

    @pytest.fixture
    def mock_credentials(self):
        """Create mock bearer credentials."""
        from unittest.mock import MagicMock

        credentials = MagicMock()
        return credentials

    @pytest.mark.asyncio
    async def test_auth_disabled(self, mock_settings, mock_credentials):
        """Test that auth is bypassed when disabled."""
        mock_settings.auth.enabled = False
        result = await get_bearer_token(mock_credentials, mock_settings)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_credentials(self, mock_settings):
        """Test that missing credentials raises AuthenticationError."""
        with pytest.raises(AuthenticationError, match="Bearer token is required"):
            await get_bearer_token(None, mock_settings)

    @pytest.mark.asyncio
    async def test_invalid_token(self, mock_settings, mock_credentials):
        """Test that invalid token raises AuthenticationError."""
        mock_credentials.credentials = "invalid-token"

        with patch.dict(os.environ, {"JWT_SECRET": "test-secret"}):
            with pytest.raises(AuthenticationError, match="Invalid token"):
                await get_bearer_token(mock_credentials, mock_settings)

    @pytest.mark.asyncio
    async def test_expired_token(self, mock_settings, mock_credentials):
        """Test that expired token raises AuthenticationError with specific message."""
        # Create expired token
        secret = "test-secret-key"
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now - timedelta(hours=1),
            "iss": "firstlight-api",
        }
        expired_token = jwt.encode(payload, secret, algorithm="HS256")
        mock_credentials.credentials = expired_token

        with patch.dict(os.environ, {"JWT_SECRET": secret}):
            with pytest.raises(AuthenticationError, match="Token has expired"):
                await get_bearer_token(mock_credentials, mock_settings)

    @pytest.mark.asyncio
    async def test_valid_token(self, mock_settings, mock_credentials):
        """Test that valid token is accepted."""
        # Create valid token
        secret = "test-secret-key"
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "user-123",
            "exp": now + timedelta(hours=1),
            "iss": "firstlight-api",
        }
        valid_token = jwt.encode(payload, secret, algorithm="HS256")
        mock_credentials.credentials = valid_token

        with patch.dict(os.environ, {"JWT_SECRET": secret}):
            result = await get_bearer_token(mock_credentials, mock_settings)
            assert result == valid_token
