"""
Container Manager for CLI-Agent Multi-User Architecture

Manages per-UUID container lifecycle:
- Container creation/discovery by UUID
- Activity tracking and cleanup
- Storage-aware aggressive cleanup
- Per-container CPU quota management
- WebSocket and HTTP proxying to backend containers
"""

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from contextlib import contextmanager

import docker
import httpx
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool

# Configure logging with file persistence
from logging.handlers import RotatingFileHandler

LOG_DIR = Path("/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
console_handler.setFormatter(console_formatter)

# File handler with rotation
file_handler = RotatingFileHandler(
    LOG_DIR / "container_manager.log",
    maxBytes=50 * 1024 * 1024,  # 50MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)  # Log everything to file
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
file_handler.setFormatter(file_formatter)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_handler, file_handler]
)
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path("/data/container_manager.db")
STORAGE_THRESHOLD = 90  # Percent
DEFAULT_RETENTION_DAYS = 7
BASE_IMAGE = "cli-agent-backend:latest"
MAX_CONTAINERS_PER_USER = int(os.getenv("MAX_CONTAINERS_PER_USER", "3"))  # Configurable limit

# CORS Configuration
# Get allowed origins from environment variable or use default
ALLOWED_ORIGINS = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
# Strip whitespace from each origin
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS]

app = FastAPI(
    title="CLI-Agent Container Manager",
    description="Manages per-UUID container lifecycle with WebSocket proxying",
    version="1.0.0"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT", "PATCH"],
    allow_headers=["X-User-UUID", "Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
)

# UUID validation pattern (RFC 4122 UUID v4)
UUID_PATTERN = re.compile(
    r'^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$',
    re.IGNORECASE
)


def validate_uuid(uuid: str) -> str:
    """
    Validate UUID format to prevent injection attacks.

    Args:
        uuid: UUID string to validate

    Returns:
        str: The validated UUID

    Raises:
        HTTPException: If UUID is invalid or malformed
    """
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID cannot be empty")

    if len(uuid) > 36:
        raise HTTPException(status_code=400, detail="UUID too long")

    if not UUID_PATTERN.match(uuid):
        raise HTTPException(
            status_code=400,
            detail="Invalid UUID format. Expected UUID v4 format (xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx)"
        )

    return uuid


@dataclass
class ContainerInfo:
    """Container information"""
    uuid: str
    container_id: str
    container_name: str
    created_at: datetime
    last_activity: datetime
    retention_days: int = 7
    user_id: Optional[str] = None


class StorageMonitor:
    """Monitor disk storage usage"""

    def __init__(self, threshold_percent: int = 90):
        self.threshold = threshold_percent

    def get_usage_percent(self) -> float:
        """Get storage usage percentage"""
        stat = shutil.disk_usage('/')
        return (stat.used / stat.total) * 100

    def get_available_gb(self) -> float:
        """Get available storage in GB"""
        stat = shutil.disk_usage('/')
        return stat.free / (1024**3)

    def is_critical(self) -> bool:
        """Check if storage is critically low"""
        return self.get_usage_percent() > self.threshold


class ContainerManager:
    """Manages CLI-Agent container lifecycle"""

    def __init__(self):
        self.docker_client = docker.DockerClient(base_url='unix://var/run/docker.sock')
        self.storage_monitor = StorageMonitor(STORAGE_THRESHOLD)
        self.db_path = DB_PATH
        self.init_database()
        logger.info("Container Manager initialized")

    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections with WAL mode"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """Initialize SQLite database with WAL mode"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS containers (
                    uuid TEXT PRIMARY KEY,
                    container_id TEXT NOT NULL,
                    container_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    retention_days INTEGER DEFAULT 7,
                    user_id TEXT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS activity_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT,
                    FOREIGN KEY(uuid) REFERENCES containers(uuid)
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_activity ON containers(last_activity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_retention ON containers(retention_days)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_activity_uuid ON activity_log(uuid)")

            logger.info("Database initialized with WAL mode")

    def hash_uuid(self, uuid: str) -> str:
        """Generate short hash for UUID (12 chars for container names)"""
        return hashlib.sha256(uuid.encode()).hexdigest()[:12]

    def get_or_create_user_network(self, uuid: str) -> str:
        """
        Get or create an isolated network for a user.
        Each user gets their own network, preventing cross-user container communication.
        """
        uuid_short = self.hash_uuid(uuid)
        network_name = f"cli-agent-user-{uuid_short}"

        try:
            # Try to get existing network
            self.docker_client.networks.get(network_name)
            logger.debug(f"Using existing network: {network_name}")
            return network_name
        except docker.errors.NotFound:
            # Network doesn't exist, create it
            logger.info(f"Creating isolated network for UUID {uuid[:8]}...: {network_name}")

            # Create isolated network with labels for cleanup
            self.docker_client.networks.create(
                network_name,
                driver="bridge",
                internal=False,  # Allow external connections (for AI APIs)
                labels={
                    "cli-agent.managed": "true",
                    "cli-agent.user-uuid": uuid,
                    "cli-agent.network-type": "user-isolated"
                }
            )

            return network_name

    async def get_or_create_container(self, uuid: str) -> ContainerInfo:
        """Get existing container or create new one for UUID"""
        # Check if container exists
        existing = self.get_container(uuid)
        if existing:
            # Verify container is still running
            try:
                container = self.docker_client.containers.get(existing.container_id)
                if container.status == 'running':
                    # Update activity
                    self.update_activity(uuid)
                    return existing
                else:
                    logger.warning(f"Container {existing.container_name} not running, recreating")
                    await self.destroy_container(uuid)
            except docker.errors.NotFound:
                logger.warning(f"Container {existing.container_id} not found in Docker, cleaning DB")
                await self.destroy_container(uuid)

        # Check storage before creating
        if self.storage_monitor.is_critical():
            logger.warning(f"Storage critical: {self.storage_monitor.get_usage_percent():.1f}%")
            await self.aggressive_cleanup()

        # Check per-user container limit
        container_count = self.count_user_containers(uuid)
        if container_count >= MAX_CONTAINERS_PER_USER:
            logger.warning(f"Container limit reached for UUID {uuid[:8]}... ({container_count}/{MAX_CONTAINERS_PER_USER})")
            raise HTTPException(
                status_code=429,
                detail=f"Maximum containers per user reached ({MAX_CONTAINERS_PER_USER}). Please delete existing containers first."
            )

        # Create new container
        return await self.create_container(uuid)

    async def create_container(self, uuid: str) -> ContainerInfo:
        """Create new container for UUID with Docker labels"""
        uuid_short = self.hash_uuid(uuid)
        container_name = f"cli-agent-{uuid_short}"

        logger.info(f"Creating container for UUID {uuid[:8]}... (hash: {uuid_short})")

        # Labels for robust resource tracking
        labels = {
            "cli-agent.managed": "true",
            "cli-agent.user-uuid": uuid,
            "cli-agent.created": datetime.utcnow().isoformat()
        }

        try:
            # Create volumes with labels
            for suffix in ['data', 'logs', 'config']:
                vol_name = f"cli-agent-{uuid_short}-{suffix}"
                try:
                    self.docker_client.volumes.create(name=vol_name, labels=labels)
                    logger.debug(f"Created volume: {vol_name}")
                except docker.errors.APIError as e:
                    if "already exists" not in str(e):
                        raise

            # Calculate initial CPU quota
            cpu_quota = 100000  # 100% (full core by default)

            # Get or create isolated network for this user
            user_network = self.get_or_create_user_network(uuid)

            # Prepare environment with API keys from host
            env_vars = {
                "CLI_AGENT_MODE": "containerized",
                "CLI_AGENT_DATA_DIR": "/workspace/data",
                "CLI_AGENT_LOG_DIR": "/workspace/logs",
                "CLI_AGENT_CONFIG_DIR": "/workspace/config",
                "PYTHONUNBUFFERED": "1"
            }
            
            # Pass LLM API keys from host environment
            api_key_vars = [
                "GOOGLE_API_KEY", "GEMINI_API_KEY",
                "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "GROQ_API_KEY", "NVIDIA_API_KEY",
                "OLLAMA_HOST", "BRAVE_API_KEY"
            ]
            for key in api_key_vars:
                value = os.environ.get(key)
                if value:
                    env_vars[key] = value
                    logger.debug(f"Passing {key} to container")
            
            # Create container with resource limits
            container = self.docker_client.containers.run(
                image=BASE_IMAGE,
                name=container_name,
                ports={'7860/tcp': None},  # Docker assigns dynamic port
                environment=env_vars,
                volumes={
                    f"cli-agent-{uuid_short}-data": {"bind": "/workspace/data", "mode": "rw"},
                    f"cli-agent-{uuid_short}-logs": {"bind": "/workspace/logs", "mode": "rw"},
                    f"cli-agent-{uuid_short}-config": {"bind": "/workspace/config", "mode": "rw"}
                },
                network=user_network,  # Isolated per-user network
                detach=True,
                restart_policy={"Name": "unless-stopped"},

                # CPU/Memory limits
                cpu_quota=cpu_quota,
                cpu_period=100000,
                mem_limit="2g",

                # Process limits (prevent fork bombs)
                pids_limit=100,

                # Disk I/O limits (10 MB/s read, 10 MB/s write)
                # Note: Requires host device path, typically /dev/sda or /dev/vda
                device_write_bps=[{"Path": "/dev/sda", "Rate": 10485760}],  # 10 MB/s
                device_read_bps=[{"Path": "/dev/sda", "Rate": 10485760}],   # 10 MB/s

                # Security profiles
                security_opt=[
                    "no-new-privileges:true",      # Prevent privilege escalation
                    "apparmor=docker-default"      # Apply Docker's default AppArmor profile
                ],

                # Drop all Linux capabilities, add back only essential ones
                cap_drop=["ALL"],
                cap_add=[
                    "CHOWN",              # Change file ownership
                    "SETUID",             # Set user ID
                    "SETGID",             # Set group ID
                    "DAC_OVERRIDE",       # Bypass file permission checks (needed for some ops)
                    "NET_BIND_SERVICE"    # Bind to privileged ports (<1024)
                ],

                labels=labels
            )

            # Register in database
            container_info = ContainerInfo(
                uuid=uuid,
                container_id=container.id,
                container_name=container_name,
                created_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                retention_days=DEFAULT_RETENTION_DAYS
            )

            # Connect container to cli-agent-network for Container Manager proxy access
            # Container is now on TWO networks:
            # 1. user_network (isolated, prevents cross-user attacks)
            # 2. cli-agent-network (allows Container Manager to proxy WebSocket/HTTP)
            try:
                shared_network = self.docker_client.networks.get("cli-agent-network")
                shared_network.connect(container)
                logger.debug(f"Connected {container_name} to shared cli-agent-network for proxy")
            except Exception as e:
                logger.warning(f"Could not connect to cli-agent-network: {e}")
                # Not fatal - container still works, just proxy might not

            self.register_container(container_info)
            self.log_activity(uuid, "container_created")

            logger.info(f"Container created: {container_name} (ID: {container.id[:12]})")

            # Wait for container to become healthy (max 30 seconds)
            logger.info(f"Waiting for container {container_name} to become healthy...")
            await self._wait_for_container_healthy(container, timeout=30)

            return container_info

        except Exception as e:
            logger.error(f"Failed to create container for UUID {uuid[:8]}...: {e}")
            # Cleanup partial resources
            try:
                await self.cleanup_container_resources(uuid_short)
            except Exception as cleanup_err:
                logger.error(f"Cleanup after failed creation also failed: {cleanup_err}")
            raise HTTPException(status_code=500, detail=f"Container creation failed: {e}")

    async def _wait_for_container_healthy(self, container, timeout: int = 30):
        """Wait for container to become healthy"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container.reload()
                health = container.attrs.get('State', {}).get('Health', {})
                status = health.get('Status', 'none')
                
                if status == 'healthy':
                    logger.info(f"Container {container.name} is healthy")
                    return True
                elif status == 'unhealthy':
                    logger.warning(f"Container {container.name} is unhealthy")
                    return False
                    
                # For containers without healthcheck, check if running
                if container.status == 'running' and status == 'none':
                    # Wait a bit for services to start
                    await asyncio.sleep(3)
                    logger.info(f"Container {container.name} is running (no healthcheck)")
                    return True
                    
            except Exception as e:
                logger.debug(f"Health check error: {e}")
                
            await asyncio.sleep(1)
        
        logger.warning(f"Container {container.name} health check timeout after {timeout}s")
        return False

    async def cleanup_container_resources(self, uuid_short: str):
        """Cleanup container and volumes by hash (used in error recovery)"""
        try:
            container = self.docker_client.containers.get(f"cli-agent-{uuid_short}")
            container.stop(timeout=10)
            container.remove()
        except docker.errors.NotFound:
            pass

        for suffix in ['data', 'logs', 'config']:
            try:
                volume = self.docker_client.volumes.get(f"cli-agent-{uuid_short}-{suffix}")
                volume.remove()
            except docker.errors.NotFound:
                pass

    def get_container(self, uuid: str) -> Optional[ContainerInfo]:
        """Get container info for UUID"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT uuid, container_id, container_name, created_at, last_activity, retention_days, user_id FROM containers WHERE uuid = ?",
                (uuid,)
            )
            row = cursor.fetchone()

            if not row:
                return None

            return ContainerInfo(
                uuid=row[0],
                container_id=row[1],
                container_name=row[2],
                created_at=datetime.fromisoformat(row[3]),
                last_activity=datetime.fromisoformat(row[4]),
                retention_days=row[5],
                user_id=row[6]
            )

    def count_user_containers(self, uuid: str) -> int:
        """Count active containers for a given UUID"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM containers WHERE uuid = ?",
                (uuid,)
            )
            return cursor.fetchone()[0]

    def register_container(self, info: ContainerInfo):
        """Register container in database"""
        with self.get_db_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO containers
                   (uuid, container_id, container_name, created_at, last_activity, retention_days, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (info.uuid, info.container_id, info.container_name,
                 info.created_at.isoformat(), info.last_activity.isoformat(),
                 info.retention_days, info.user_id)
            )

    def update_activity(self, uuid: str):
        """Update last activity timestamp"""
        with self.get_db_connection() as conn:
            conn.execute(
                "UPDATE containers SET last_activity = ? WHERE uuid = ?",
                (datetime.utcnow().isoformat(), uuid)
            )
        self.log_activity(uuid, "activity_update")

    def log_activity(self, uuid: str, event_type: str):
        """Log activity event"""
        with self.get_db_connection() as conn:
            conn.execute(
                "INSERT INTO activity_log (uuid, timestamp, event_type) VALUES (?, ?, ?)",
                (uuid, datetime.utcnow().isoformat(), event_type)
            )

    def list_active_containers(self) -> List[ContainerInfo]:
        """List all active containers"""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT uuid, container_id, container_name, created_at, last_activity, retention_days, user_id FROM containers ORDER BY created_at DESC"
            )

            return [
                ContainerInfo(
                    uuid=row[0],
                    container_id=row[1],
                    container_name=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    last_activity=datetime.fromisoformat(row[4]),
                    retention_days=row[5],
                    user_id=row[6]
                )
                for row in cursor.fetchall()
            ]

    async def destroy_container(self, uuid: str):
        """Destroy container and volumes for UUID (idempotent with label-based cleanup)"""
        container_info = self.get_container(uuid)
        if not container_info:
            logger.warning(f"No container found for UUID {uuid[:8]}... in database")
            return

        logger.info(f"Destroying container for UUID {uuid[:8]}...")

        # Stop container (idempotent)
        try:
            container = self.docker_client.containers.get(container_info.container_id)
            container.stop(timeout=10)
            container.remove()
            logger.info(f"Container removed: {container_info.container_name}")
        except docker.errors.NotFound:
            logger.warning(f"Container already removed: {container_info.container_id}")
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")

        # Remove volumes using label filter (no name parsing)
        volumes = self.docker_client.volumes.list(
            filters={"label": f"cli-agent.user-uuid={uuid}"}
        )

        for volume in volumes:
            try:
                volume.remove()
                logger.info(f"Removed volume: {volume.name}")
            except docker.errors.NotFound:
                pass
            except docker.errors.APIError as e:
                if "volume is in use" in str(e):
                    logger.error(f"Volume {volume.name} still in use - will retry on next audit")
                else:
                    raise

        # Cleanup user's isolated network if no containers remain
        uuid_short = self.hash_uuid(uuid)
        network_name = f"cli-agent-user-{uuid_short}"
        try:
            network = self.docker_client.networks.get(network_name)

            # Check if any containers are still using this network
            remaining_containers = self.count_user_containers(uuid)
            if remaining_containers == 0:
                # No more containers for this user, safe to remove network
                network.remove()
                logger.info(f"Removed isolated network: {network_name}")
            else:
                logger.debug(f"Network {network_name} still has {remaining_containers} containers, keeping")
        except docker.errors.NotFound:
            logger.debug(f"Network {network_name} already removed or never existed")
        except Exception as e:
            logger.warning(f"Failed to cleanup network {network_name}: {e}")

        # Remove from database (idempotent)
        with self.get_db_connection() as conn:
            conn.execute("DELETE FROM containers WHERE uuid = ?", (uuid,))
            conn.execute("DELETE FROM activity_log WHERE uuid = ?", (uuid,))

        self.log_activity(uuid, "container_destroyed")
        logger.info(f"Container destroyed for UUID {uuid[:8]}...")

    async def cleanup_expired_containers(self):
        """Clean up containers that have exceeded retention period"""
        containers = self.list_active_containers()
        now = datetime.utcnow()

        for info in containers:
            cutoff = now - timedelta(days=info.retention_days)

            if info.last_activity < cutoff:
                logger.info(f"Cleaning up expired container for UUID {info.uuid[:8]}... (idle {info.retention_days}+ days)")
                await self.destroy_container(info.uuid)

    async def aggressive_cleanup(self):
        """Aggressively clean up oldest containers when storage is low"""
        logger.warning("Starting aggressive cleanup due to low storage")

        # Get containers sorted by creation time (oldest first)
        containers = sorted(
            self.list_active_containers(),
            key=lambda c: c.created_at
        )

        cleaned = 0
        for info in containers:
            await self.destroy_container(info.uuid)
            cleaned += 1

            # Check if we've freed enough space
            if self.storage_monitor.get_usage_percent() < 85:
                logger.info(f"Aggressive cleanup complete. Removed {cleaned} containers. Storage at {self.storage_monitor.get_usage_percent():.1f}%")
                break
        else:
            logger.warning(f"Aggressive cleanup removed all {cleaned} containers but storage still critical")

    async def audit_orphaned_volumes(self):
        """Periodic audit using label filters (efficient and robust)"""
        active_uuids = {c.uuid for c in self.list_active_containers()}

        # Get only volumes managed by this system (label filter)
        managed_volumes = self.docker_client.volumes.list(
            filters={"label": "cli-agent.managed=true"}
        )

        orphaned = 0
        for volume in managed_volumes:
            vol_uuid = volume.labels.get("cli-agent.user-uuid")
            if vol_uuid and vol_uuid not in active_uuids:
                try:
                    volume.remove()
                    logger.info(f"Removed orphaned volume {volume.name} for inactive UUID {vol_uuid[:8]}...")
                    orphaned += 1
                except Exception as e:
                    logger.error(f"Failed to remove orphaned volume {volume.name}: {e}")

        if orphaned > 0:
            logger.info(f"Orphaned volume audit complete: removed {orphaned} volumes")

    def _adjust_cpu_quotas_sync(self):
        """Synchronous CPU adjustment (runs in thread pool)"""
        for container_info in self.list_active_containers():
            try:
                container = self.docker_client.containers.get(container_info.container_id)
                stats = container.stats(stream=False)  # Blocking call (1-2s)

                # Calculate CPU usage percentage
                cpu_stats = stats.get('cpu_stats', {})
                precpu_stats = stats.get('precpu_stats', {})

                cpu_usage = cpu_stats.get('cpu_usage', {})
                precpu_usage = precpu_stats.get('cpu_usage', {})

                cpu_total = cpu_usage.get('total_usage', 0)
                precpu_total = precpu_usage.get('total_usage', 0)

                system_cpu = cpu_stats.get('system_cpu_usage', 0)
                precpu_system = precpu_stats.get('system_cpu_usage', 0)

                cpu_delta = cpu_total - precpu_total
                system_delta = system_cpu - precpu_system

                cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0

                # Adjust quota based on individual usage
                if cpu_percent > 90:
                    quota = 50000  # 50% of one core (throttle heavy user)
                    logger.warning(f"Throttling {container_info.uuid[:8]}...: CPU {cpu_percent:.1f}%")
                elif cpu_percent > 70:
                    quota = 75000  # 75%
                else:
                    quota = 100000  # 100% (full core)

                # Only update if quota changed significantly
                current_quota = container.attrs.get('HostConfig', {}).get('CpuQuota', 100000)
                if abs(current_quota - quota) > 10000:
                    container.update(cpu_quota=quota, cpu_period=100000)
                    logger.debug(f"Updated CPU quota for {container_info.uuid[:8]}...: {quota/1000}%")

            except Exception as e:
                logger.error(f"Failed to adjust CPU for {container_info.uuid[:8]}...: {e}")

    async def adjust_container_cpu_quotas(self):
        """Per-container CPU monitoring with fair quotas (non-blocking)"""
        # Run blocking Docker SDK calls in thread pool
        await run_in_threadpool(self._adjust_cpu_quotas_sync)


# Global manager instance
manager = ContainerManager()


# === API Endpoints ===

@app.get("/discover")
async def discover_container(request: Request):
    """
    Discover or create container for requesting UUID
    UUID passed in X-User-UUID header
    """
    uuid = request.headers.get("X-User-UUID")
    if not uuid:
        raise HTTPException(status_code=400, detail="Missing X-User-UUID header")

    # Validate UUID format
    uuid = validate_uuid(uuid)

    logger.info(f"Container discovery request for UUID: {uuid[:8]}...")

    try:
        container_info = await manager.get_or_create_container(uuid)

        return JSONResponse({
            "status": "success",
            "uuid": uuid,
            "container_name": container_info.container_name,
            "created_at": container_info.created_at.isoformat(),
            "last_activity": container_info.last_activity.isoformat()
        })
    except Exception as e:
        logger.error(f"Container discovery failed for {uuid[:8]}...: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/backend/{uuid}/ws")
async def websocket_proxy(websocket: WebSocket, uuid: str):
    """Bi-directional WebSocket proxy to backend container"""
    await websocket.accept()

    # Validate UUID format
    try:
        uuid = validate_uuid(uuid)
    except HTTPException as e:
        await websocket.close(code=1008, reason=e.detail)
        return

    logger.info(f"WebSocket connection established for UUID {uuid[:8]}...")

    container_info = manager.get_container(uuid)
    if not container_info:
        logger.error(f"Container not found for UUID {uuid[:8]}...")
        await websocket.close(code=1011, reason="Container not found")
        return

    # Update activity
    manager.update_activity(uuid)

    # Backend container's internal WebSocket URL
    backend_ws_url = f"ws://{container_info.container_name}:7860/ws"
    logger.debug(f"Connecting to backend WebSocket: {backend_ws_url}")

    try:
        # Connect to backend container's WebSocket
        async with websockets.connect(backend_ws_url, ping_interval=30, ping_timeout=10) as backend_ws:
            logger.info(f"Connected to backend WebSocket for UUID {uuid[:8]}...")

            async def forward_to_backend():
                """Forward messages from client to backend"""
                try:
                    while True:
                        data = await websocket.receive_text()
                        await backend_ws.send(data)
                        logger.debug(f"Forwarded message to backend (UUID {uuid[:8]}...)")
                except WebSocketDisconnect:
                    logger.info(f"Client disconnected (UUID {uuid[:8]}...)")
                except Exception as e:
                    logger.error(f"Error forwarding to backend (UUID {uuid[:8]}...): {e}")

            async def forward_to_client():
                """Forward messages from backend to client"""
                try:
                    while True:
                        data = await backend_ws.recv()
                        await websocket.send_text(data)
                        logger.debug(f"Forwarded message to client (UUID {uuid[:8]}...)")
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"Backend disconnected (UUID {uuid[:8]}...)")
                except Exception as e:
                    logger.error(f"Error forwarding to client (UUID {uuid[:8]}...): {e}")

            # Run both tasks concurrently
            await asyncio.gather(forward_to_client(), forward_to_backend())

    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        logger.info(f"WebSocket connection closed for UUID {uuid[:8]}...")
    except Exception as e:
        logger.error(f"WebSocket proxy error for UUID {uuid[:8]}...: {e}")
        await websocket.close(code=1011, reason="Proxy error")


@app.api_route("/api/backend/{uuid}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def http_proxy(request: Request, uuid: str, path: str):
    """HTTP proxy to backend container REST endpoints"""
    # Validate UUID format
    uuid = validate_uuid(uuid)

    container_info = manager.get_container(uuid)
    if not container_info:
        raise HTTPException(status_code=404, detail="Container not found")

    # Update activity
    manager.update_activity(uuid)

    # Backend container's internal HTTP URL
    backend_url = f"http://{container_info.container_name}:7860/{path}"

    # Forward query parameters
    if request.url.query:
        backend_url += f"?{request.url.query}"

    logger.debug(f"Proxying HTTP {request.method} to {backend_url}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Forward request
            response = await client.request(
                method=request.method,
                url=backend_url,
                headers=dict(request.headers),
                content=await request.body()
            )

            # Return response
            return StreamingResponse(
                response.aiter_bytes(),
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except Exception as e:
        logger.error(f"HTTP proxy error for UUID {uuid[:8]}...: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")


@app.post("/activity")
async def update_activity(request: Request):
    """Update last activity timestamp for UUID"""
    uuid = request.headers.get("X-User-UUID")
    if not uuid:
        raise HTTPException(status_code=400, detail="Missing X-User-UUID header")

    # Validate UUID format
    uuid = validate_uuid(uuid)

    manager.update_activity(uuid)
    return JSONResponse({"status": "success", "uuid": uuid})


@app.get("/containers")
async def list_containers():
    """List all active containers (admin endpoint)"""
    containers = manager.list_active_containers()
    now = datetime.utcnow()

    return JSONResponse({
        "status": "success",
        "count": len(containers),
        "containers": [
            {
                "uuid": c.uuid[:8] + "...",
                "container_name": c.container_name,
                "created_at": c.created_at.isoformat(),
                "last_activity": c.last_activity.isoformat(),
                "retention_days": c.retention_days,
                "age_days": (now - c.created_at).days,
                "idle_days": (now - c.last_activity).days
            }
            for c in containers
        ],
        "storage_usage": manager.storage_monitor.get_usage_percent(),
        "storage_available_gb": manager.storage_monitor.get_available_gb()
    })


@app.delete("/container/{uuid}")
async def delete_container(uuid: str):
    """Manually delete container for specific UUID (admin only)"""
    # Validate UUID format
    uuid = validate_uuid(uuid)

    await manager.destroy_container(uuid)
    return JSONResponse({"status": "success", "uuid": uuid})


@app.post("/container/{uuid}/wipe")
async def wipe_container(uuid: str):
    """Wipe and recreate container for UUID"""
    # Validate UUID format
    uuid = validate_uuid(uuid)

    logger.info(f"Wipe request for UUID {uuid[:8]}...")
    await manager.destroy_container(uuid)
    container_info = await manager.get_or_create_container(uuid)
    return JSONResponse({
        "status": "success",
        "uuid": uuid,
        "container_name": container_info.container_name
    })


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    containers = manager.list_active_containers()
    return JSONResponse({
        "status": "healthy",
        "containers": len(containers),
        "storage_usage": manager.storage_monitor.get_usage_percent(),
        "storage_available_gb": manager.storage_monitor.get_available_gb()
    })


# === Background Tasks ===

@app.on_event("startup")
async def startup():
    """Start background tasks"""
    logger.info("Starting background tasks...")
    asyncio.create_task(cleanup_loop())
    asyncio.create_task(cpu_adjustment_loop())
    asyncio.create_task(storage_monitoring_loop())
    logger.info("Container Manager started successfully")


async def cleanup_loop():
    """Background task for container cleanup"""
    while True:
        await asyncio.sleep(3600)  # Every hour

        try:
            logger.info("Running cleanup loop...")
            # Regular cleanup
            await manager.cleanup_expired_containers()
            # Orphaned volume audit
            await manager.audit_orphaned_volumes()
            logger.info("Cleanup loop complete")
        except Exception as e:
            logger.error(f"Cleanup loop error: {e}")


async def cpu_adjustment_loop():
    """Background task for CPU quota adjustment"""
    while True:
        await asyncio.sleep(900)  # Every 15 minutes

        try:
            logger.debug("Running CPU adjustment...")
            await manager.adjust_container_cpu_quotas()
        except Exception as e:
            logger.error(f"CPU adjustment loop error: {e}")


async def storage_monitoring_loop():
    """Background task for storage monitoring"""
    while True:
        await asyncio.sleep(600)  # Every 10 minutes

        try:
            if manager.storage_monitor.is_critical():
                logger.warning("Storage critical, triggering aggressive cleanup")
                await manager.aggressive_cleanup()
        except Exception as e:
            logger.error(f"Storage monitoring loop error: {e}")
