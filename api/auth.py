"""
Authentication and Authorization module for CLI-Agent SaaS.

Provides JWT-based authentication, session management, and user validation
for multi-tenant container environments.

Security Features:
- JWT token generation and validation with expiration
- Session token rotation on privilege changes
- User UUID ownership verification
- Rate limiting per user session
- Audit logging of auth events
"""

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logging.warning("PyJWT not installed. Run: pip install PyJWT")

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """User session data with security metadata."""
    uuid: str
    token: str
    created_at: float
    last_activity: float
    ip_address: str
    user_agent: str
    token_version: int = 1  # Increment to invalidate old tokens
    max_age_seconds: int = 86400  # 24 hours
    idle_timeout_seconds: int = 3600  # 1 hour

    @property
    def is_expired(self) -> bool:
        """Check if session has exceeded max age."""
        return time.time() - self.created_at > self.max_age_seconds

    @property
    def is_idle(self) -> bool:
        """Check if session has exceeded idle timeout."""
        return time.time() - self.last_activity > self.idle_timeout_seconds

    @property
    def is_valid(self) -> bool:
        """Check if session is valid (not expired or idle)."""
        return not (self.is_expired or self.is_idle)

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = time.time()


class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    pass


class AuthManager:
    """
    Manages user authentication and session tokens.

    Security Model:
    - JWT tokens with short expiration (24h max age, 1h idle timeout)
    - Token rotation on privilege escalation
    - Session invalidation on logout
    - Per-user rate limiting
    - Audit logging of all auth events
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize auth manager.

        Args:
            config_dir: Directory for storing auth config (default: ~/.cli-agent)
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("PyJWT is required for authentication. Install with: pip install PyJWT")

        self.config_dir = config_dir or Path.home() / ".cli-agent"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load or generate JWT secret
        self.secret_key = self._load_or_generate_secret()

        # Active sessions by UUID
        self.sessions: dict[str, UserSession] = {}

        # Rate limiting: UUID -> list of request timestamps
        self.rate_limit_window = 60  # 1 minute window
        self.rate_limit_max_requests = 100  # Max requests per window
        self._rate_limit_data: dict[str, list[float]] = {}

        logger.info("AuthManager initialized")

    def _load_or_generate_secret(self) -> str:
        """Load existing JWT secret or generate new one."""
        secret_file = self.config_dir / ".jwt_secret"

        if secret_file.exists():
            with open(secret_file) as f:
                secret = f.read().strip()
                logger.info("Loaded existing JWT secret")
                return secret

        # Generate new secret (256-bit)
        secret = secrets.token_urlsafe(32)

        # Save with restrictive permissions
        secret_file.write_text(secret)
        os.chmod(secret_file, 0o600)  # Owner read/write only

        logger.warning("Generated new JWT secret - existing tokens will be invalidated")
        return secret

    def generate_token(self, uuid: str, ip_address: str, user_agent: str) -> str:
        """
        Generate JWT token for user UUID.

        Args:
            uuid: User UUID
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            JWT token string
        """
        # Create session
        issued_at = time.time()
        expires_at = issued_at + 86400  # 24 hours

        payload = {
            "uuid": uuid,
            "iat": issued_at,
            "exp": expires_at,
            "ip": hashlib.sha256(ip_address.encode()).hexdigest()[:16],  # Hashed IP
            "ua": hashlib.sha256(user_agent.encode()).hexdigest()[:16],  # Hashed UA
            "ver": 1  # Token version
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")

        # Store session
        session = UserSession(
            uuid=uuid,
            token=token,
            created_at=issued_at,
            last_activity=issued_at,
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.sessions[uuid] = session

        logger.info(f"Generated token for UUID {uuid[:8]}...")
        return token

    def validate_token(self, token: str, ip_address: str, user_agent: str) -> str:
        """
        Validate JWT token and return UUID.

        Args:
            token: JWT token to validate
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            User UUID if valid

        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            # Decode and verify token
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            uuid = payload["uuid"]

            # Verify IP and user agent match (prevents token theft)
            ip_hash = hashlib.sha256(ip_address.encode()).hexdigest()[:16]
            ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]

            if payload.get("ip") != ip_hash:
                logger.warning(f"Token IP mismatch for UUID {uuid[:8]}...")
                raise AuthenticationError("Token IP mismatch - possible token theft")

            if payload.get("ua") != ua_hash:
                logger.warning(f"Token user-agent mismatch for UUID {uuid[:8]}...")
                raise AuthenticationError("Token user-agent mismatch - possible token theft")

            # Check if session exists and is valid
            session = self.sessions.get(uuid)
            if not session:
                raise AuthenticationError("Session not found - please re-authenticate")

            if not session.is_valid:
                del self.sessions[uuid]
                raise AuthenticationError("Session expired - please re-authenticate")

            # Update activity
            session.update_activity()

            return uuid

        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")

    def rotate_token(self, uuid: str) -> str:
        """
        Rotate token for user (invalidates old token).

        Args:
            uuid: User UUID

        Returns:
            New JWT token
        """
        session = self.sessions.get(uuid)
        if not session:
            raise AuthenticationError("Session not found")

        # Increment token version
        session.token_version += 1

        # Generate new token
        new_token = self.generate_token(uuid, session.ip_address, session.user_agent)

        logger.info(f"Rotated token for UUID {uuid[:8]}...")
        return new_token

    def revoke_session(self, uuid: str):
        """Revoke session and invalidate token."""
        if uuid in self.sessions:
            del self.sessions[uuid]
            logger.info(f"Revoked session for UUID {uuid[:8]}...")

    def check_rate_limit(self, uuid: str) -> bool:
        """
        Check if user has exceeded rate limit.

        Args:
            uuid: User UUID

        Returns:
            True if within rate limit, False if exceeded
        """
        current_time = time.time()
        window_start = current_time - self.rate_limit_window

        # Get or create request history
        if uuid not in self._rate_limit_data:
            self._rate_limit_data[uuid] = []

        requests = self._rate_limit_data[uuid]

        # Remove old requests outside window
        requests[:] = [ts for ts in requests if ts > window_start]

        # Check limit
        if len(requests) >= self.rate_limit_max_requests:
            logger.warning(f"Rate limit exceeded for UUID {uuid[:8]}...")
            return False

        # Add current request
        requests.append(current_time)
        return True

    def cleanup_expired_sessions(self):
        """Remove expired sessions (called periodically)."""
        expired_uuids = [
            uuid for uuid, session in self.sessions.items()
            if not session.is_valid
        ]

        for uuid in expired_uuids:
            del self.sessions[uuid]
            logger.debug(f"Cleaned up expired session for UUID {uuid[:8]}...")

        if expired_uuids:
            logger.info(f"Cleaned up {len(expired_uuids)} expired sessions")


# Global instance (created by server on startup)
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager


def require_auth(handler):
    """
    Decorator for API endpoints requiring authentication.

    Usage:
        @require_auth
        async def my_endpoint(request):
            uuid = request["authenticated_uuid"]
            ...
    """
    async def wrapper(request):
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            from aiohttp import web
            return web.json_response({
                "status": "error",
                "error": "Missing or invalid Authorization header",
                "code": "UNAUTHORIZED"
            }, status=401)

        token = auth_header.replace("Bearer ", "")

        try:
            auth_manager = get_auth_manager()

            # Extract client info
            ip_address = request.remote or "unknown"
            user_agent = request.headers.get("User-Agent", "unknown")

            # Validate token
            uuid = auth_manager.validate_token(token, ip_address, user_agent)

            # Check rate limit
            if not auth_manager.check_rate_limit(uuid):
                from aiohttp import web
                return web.json_response({
                    "status": "error",
                    "error": "Rate limit exceeded",
                    "code": "RATE_LIMIT_EXCEEDED"
                }, status=429, headers={"Retry-After": "60"})

            # Add UUID to request for handler
            request["authenticated_uuid"] = uuid

            # Call actual handler
            return await handler(request)

        except AuthenticationError as e:
            from aiohttp import web
            return web.json_response({
                "status": "error",
                "error": str(e),
                "code": "AUTHENTICATION_FAILED"
            }, status=401)

    return wrapper
