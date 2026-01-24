"""
JWT Token Handler.

Provides JWT decoding, validation, and verification for API authentication.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

logger = logging.getLogger(__name__)


class JWTConfig:
    """JWT configuration from environment variables."""

    def __init__(self):
        self.secret = os.getenv("JWT_SECRET", "")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.issuer = os.getenv("JWT_ISSUER", "firstlight-api")

    def is_configured(self) -> bool:
        """Check if JWT is properly configured."""
        return bool(self.secret)


class JWTHandler:
    """
    JWT token handler for authentication.

    Validates JWT tokens including signature, expiration, and issuer checks.
    """

    def __init__(self, config: Optional[JWTConfig] = None):
        """
        Initialize JWT handler.

        Args:
            config: JWT configuration. If None, loads from environment.
        """
        self.config = config or JWTConfig()
        if not self.config.is_configured():
            logger.warning(
                "JWT_SECRET not configured. JWT authentication will fail. "
                "Set JWT_SECRET environment variable."
            )

    def decode_token(self, token: str) -> Dict:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Decoded token payload as dictionary

        Raises:
            InvalidTokenError: If token signature is invalid
            ExpiredSignatureError: If token has expired
        """
        if not self.config.is_configured():
            raise InvalidTokenError("JWT authentication not configured")

        # Decode with signature verification
        payload = jwt.decode(
            token,
            self.config.secret,
            algorithms=[self.config.algorithm],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "require_exp": True,
            },
        )

        # Validate issuer if configured
        if self.config.issuer:
            token_issuer = payload.get("iss")
            if token_issuer != self.config.issuer:
                raise InvalidTokenError(
                    f"Invalid issuer: expected '{self.config.issuer}', got '{token_issuer}'"
                )

        return payload

    def validate_token(self, token: str) -> Optional[Dict]:
        """
        Validate a JWT token and return payload if valid.

        Args:
            token: JWT token string

        Returns:
            Token payload if valid, None if invalid
        """
        try:
            return self.decode_token(token)
        except ExpiredSignatureError:
            logger.debug("JWT token has expired")
            return None
        except InvalidTokenError as e:
            logger.debug(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error validating JWT: {e}")
            return None

    def get_user_id(self, token: str) -> Optional[str]:
        """
        Extract user ID from a valid token.

        Args:
            token: JWT token string

        Returns:
            User ID from 'sub' claim if token is valid, None otherwise
        """
        payload = self.validate_token(token)
        if payload:
            return payload.get("sub")
        return None

    def check_expiration(self, token: str) -> bool:
        """
        Check if a token is expired without full validation.

        Args:
            token: JWT token string

        Returns:
            True if token is expired, False otherwise
        """
        try:
            # Decode without verification to check expiration
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False},
            )
            exp = payload.get("exp")
            if exp is None:
                return True

            exp_datetime = datetime.fromtimestamp(exp, tz=timezone.utc)
            return datetime.now(timezone.utc) > exp_datetime
        except Exception:
            return True


# Global handler instance
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get the global JWT handler instance."""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler
